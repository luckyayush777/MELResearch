import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MIXED_DIR = ROOT / "Mixed Quantization setup"
if str(MIXED_DIR) not in sys.path:
    sys.path.insert(0, str(MIXED_DIR))

import evaluate_entity_text_tokens as evaluator  # noqa: E402


CACHE_MODES = {
    "fp32_ref": ("fp32", "fp32", "fp32", "fp32"),
    "text_tokens_int4": ("fp32", "fp32", "int4", "fp32"),
    "text_image_tokens_int4": ("fp32", "fp32", "int4", "int4"),
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantize_weight_symmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 32:
        return weight.detach().float()
    if bits == 16:
        return weight.detach().half().float()
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported weight quantization bits: {bits}")
    qmax = (2 ** (bits - 1)) - 1
    w = weight.detach().float()
    if w.ndim >= 2:
        reduce_dims = tuple(range(1, w.ndim))
        scale_shape = [w.shape[0]] + [1] * (w.ndim - 1)
        max_abs = w.abs().amax(dim=reduce_dims, keepdim=True).reshape(scale_shape)
    else:
        max_abs = w.abs().amax().reshape([1])
    scale = torch.clamp(max_abs / float(qmax), min=1e-12)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def should_quantize_linear(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "mimic_head":
        return not name.startswith("encoder.clip.")
    if scope == "matcher":
        return name.startswith("matcher.")
    raise ValueError(f"Unsupported linear quantization scope: {scope}")


def apply_linear_weight_quantization(model: nn.Module, mode: str, scope: str = "all"):
    if mode == "fp32":
        return {
            "weight_quantization_mode": "fp32",
            "quantized_linear_layers": 0,
            "quantized_linear_params": 0,
            "note": "No model weight quantization applied.",
        }
    if mode == "fp16":
        bits = 16
    elif mode.startswith("int"):
        bits = int(mode[3:])
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")

    quantized_layers = []
    quantized_params = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not should_quantize_linear(name, scope):
                continue
            module.weight.copy_(quantize_weight_symmetric(module.weight, bits).to(module.weight.device))
            quantized_layers.append(name)
            quantized_params += int(module.weight.numel())

    return {
        "weight_quantization_mode": mode,
        "weight_quantization_execution": "dequantized_fp32_execution",
        "linear_quantization_scope": scope,
        "quantized_linear_layers": len(quantized_layers),
        "quantized_linear_params": quantized_params,
        "quantized_layer_names": quantized_layers,
        "note": (
            "Smoke test quantizes all nn.Linear weights and stores the dequantized values back in the model. "
            "LayerNorm, embeddings, softmax, activations, and non-linear/custom ops are not quantized. "
            "This measures accuracy drift, not real low-bit kernel speed."
        ),
    }


def make_bnb_linear(module: nn.Linear, mode: str):
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        raise RuntimeError(f"bitsandbytes import failed: {exc}") from exc

    bias = module.bias is not None
    if mode == "bnb_int8":
        new_module = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=bias,
            has_fp16_weights=False,
        )
    elif mode == "bnb_nf4":
        new_module = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=bias,
            compute_dtype=torch.float16,
            quant_type="nf4",
        )
    else:
        raise ValueError(f"Unsupported bitsandbytes mode: {mode}")

    new_module = new_module.to(device=module.weight.device)
    with torch.no_grad():
        new_module.weight.data.copy_(module.weight.data)
        if bias:
            new_module.bias.data.copy_(module.bias.data)
    return new_module


def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_bnb_linear_quantization(model: nn.Module, mode: str, scope: str = "all"):
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and should_quantize_linear(name, scope)
    ]
    for name, module in targets:
        replace_module(model, name, make_bnb_linear(module, mode))
    return {
        "weight_quantization_mode": mode,
        "weight_quantization_execution": "bitsandbytes_cuda_linear",
        "linear_quantization_scope": scope,
        "quantized_linear_layers": len(targets),
        "quantized_linear_params": int(sum(module.weight.numel() for _, module in targets)),
        "quantized_layer_names": [name for name, _ in targets],
        "note": (
            "Real bitsandbytes CUDA Linear replacement. LayerNorm, embeddings, softmax, activations, "
            "and non-linear/custom ops are not replaced. Linear layers are executed through bitsandbytes low-bit modules."
        ),
    }


def row_cache_modes(cache_mode, tensor_modes):
    row = {"cache_mode": cache_mode}
    for name, mode in zip(evaluator.TENSOR_NAMES, tensor_modes):
        row[f"{name}_mode"] = mode
    row["entity_text_tokens_mode"] = row["text_tokens_mode"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test full-model Linear weight quantization for MIMIC.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--mimic-root", default=str(ROOT / "ExpSetup/external/MIMIC_reproduction"))
    parser.add_argument("--config", default=str(ROOT / "ExpSetup/external/MIMIC_reproduction/config/wikimel_noleak.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "ExpSetup/external/MIMIC_reproduction/runs/WikiMEL_noleak/version_4/checkpoints/final.ckpt"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-mode", choices=sorted(CACHE_MODES), default="text_image_tokens_int4")
    parser.add_argument("--weight-modes", nargs="+", default=["fp32", "int8"])
    parser.add_argument(
        "--linear-scope",
        choices=["all", "mimic_head", "matcher"],
        default="all",
        help="Which nn.Linear layers to quantize. mimic_head excludes encoder.clip.*; matcher only quantizes matcher.*.",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--matcher-tile-size", type=int, default=250)
    parser.add_argument("--entity-subtile-size", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=20)
    parser.add_argument("--entity-limit", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results/weight_quant_smoke"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    mimic_root = Path(args.mimic_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator.add_repo_paths(repo_root, mimic_root)
    from codes.utils.dataset import DataModuleForMIMIC

    cfg = evaluator.resolve_config_paths(OmegaConf.load(config_path), mimic_root)
    cfg.data.eval_chunk_size = args.chunk_size
    device = torch.device(args.device)
    env_meta = evaluator.environment_metadata(device)
    data_module = DataModuleForMIMIC(cfg)
    eval_loader = data_module.test_dataloader() if args.split == "test" else data_module.val_dataloader()
    tensor_modes = CACHE_MODES[args.cache_mode]

    rows = []
    details = {
        **env_meta,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "cache_mode": args.cache_mode,
        "tensor_modes": dict(zip(evaluator.TENSOR_NAMES, tensor_modes)),
        "weight_modes": args.weight_modes,
        "linear_scope": args.linear_scope,
        "chunk_size": args.chunk_size,
        "matcher_tile_size": args.matcher_tile_size,
        "entity_subtile_size": args.entity_subtile_size,
        "limit_queries": args.limit_queries,
        "entity_limit": args.entity_limit,
    }

    for weight_mode in args.weight_modes:
        label = f"{args.cache_mode}_weights_{weight_mode}_limit_{args.limit_queries}"
        if args.matcher_tile_size:
            label += f"_tglu_tile_{args.matcher_tile_size}"
        if args.entity_subtile_size:
            label += f"_entity_tile_{args.entity_subtile_size}"
        mode_dir = out_dir / label
        result_path = mode_dir / "result.json"
        if result_path.exists() and not args.force:
            row = evaluator.read_json(result_path)
            rows.append(row)
            print(f"Reusing {result_path}")
            continue

        try:
            model = evaluator.load_model(cfg, checkpoint_path, device)
            if weight_mode in {"bnb_int8", "bnb_nf4"}:
                quant_meta = apply_bnb_linear_quantization(model, weight_mode, args.linear_scope)
            else:
                quant_meta = apply_linear_weight_quantization(model, weight_mode, args.linear_scope)
            entity_cache, cache_meta = evaluator.build_or_load_entity_cache(
                model=model,
                dataloader=data_module.entity_dataloader(),
                device=device,
                label=label,
                tensor_modes=tensor_modes,
                mode_dir=mode_dir,
                limit_entities=args.entity_limit,
            )
            metrics = evaluator.evaluate_split_with_metrics(
                model=model,
                dataloader=eval_loader,
                entity_cache=entity_cache,
                device=device,
                label=label,
                mode_dir=mode_dir,
                chunk_size=args.chunk_size,
                limit_queries=args.limit_queries,
                matcher_tile_size=args.matcher_tile_size,
                entity_subtile_size=args.entity_subtile_size,
            )
            row = {
                **env_meta,
                **metrics,
                **row_cache_modes(args.cache_mode, tensor_modes),
                **quant_meta,
                "run_success": True,
                "failed_stage": "",
                "exception_type": "",
                "exception_message": "",
                "oom_boolean": False,
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "split": args.split,
                "num_entities": int(cache_meta.get("entities", 0)),
                "chunk_size": args.chunk_size,
                "eval_batch_size": int(getattr(cfg.data, "eval_batch_size", 0) or 0),
                **{(key if key.startswith("entity_cache_") else f"entity_cache_{key}"): value for key, value in cache_meta.items()},
            }
        except Exception as exc:
            row = {
                **env_meta,
                "mode": label,
                "cache_mode": args.cache_mode,
                "weight_quantization_mode": weight_mode,
                "run_success": False,
                "failed_stage": "smoke",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "oom_boolean": "out of memory" in str(exc).lower(),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "split": args.split,
                "chunk_size": args.chunk_size,
                "matcher_tile_size": int(args.matcher_tile_size or 0),
                "entity_subtile_size": int(args.entity_subtile_size or 0),
            }

        rows.append(row)
        write_json(result_path, row)
        details["results"] = rows
        write_json(out_dir / "weight_quant_smoke_results.json", details)
        write_csv(out_dir / "weight_quant_smoke_summary.csv", rows)
        print(json.dumps(row, indent=2))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(out_dir / "weight_quant_smoke_results.json", details)
    write_csv(out_dir / "weight_quant_smoke_summary.csv", rows)
    print(f"Saved smoke results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
