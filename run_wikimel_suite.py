import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full quantization benchmark suite on prepared WikiMEL artifacts."
    )
    parser.add_argument("--artifact-root", required=True, help="Root produced by prepare_wikimel_artifacts.py")
    parser.add_argument("--out-dir", default=os.path.join("benchmarks", "wikimel_v1"))
    parser.add_argument("--quant-modes", nargs="+", default=["fp32", "fp16", "int8"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0, help="Applied to encoder benchmark only.")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    emb = root / "embeddings"

    image_embeddings = emb / "image_embeddings.npy"
    paired_text_embeddings = emb / "text_embeddings.npy"
    kb_text_embeddings = emb / "kb_text_embeddings.npy"
    valid_metadata = emb / "valid_metadata.json"
    entity_kb = root / "entity_kb.json"

    required = [
        image_embeddings,
        paired_text_embeddings,
        kb_text_embeddings,
        valid_metadata,
        entity_kb,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_json = out_dir / "latest_results.json"
    quant_cmd = [
        sys.executable,
        "benchmark_quantization.py",
        "--image-embeddings",
        str(image_embeddings),
        "--text-embeddings",
        str(paired_text_embeddings),
        "--valid-metadata",
        str(valid_metadata),
        "--entity-kb",
        str(entity_kb),
        "--kb-text-embeddings",
        str(kb_text_embeddings),
        "--quant-modes",
        *args.quant_modes,
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--out-json",
        str(quant_json),
    ]
    run_command(quant_cmd)

    if not args.skip_encoder:
        enc_json = out_dir / "encoder_quantization_results.json"
        enc_plot_dir = out_dir / "encoder_quantization_plots"
        enc_cmd = [
            sys.executable,
            "benchmark_encoder_quantization.py",
            "--valid-metadata",
            str(valid_metadata),
            "--image-dir",
            ".",
            "--paired-text-embeddings",
            str(paired_text_embeddings),
            "--kb-text-embeddings",
            str(kb_text_embeddings),
            "--entity-kb",
            str(entity_kb),
            "--out-json",
            str(enc_json),
            "--plot-dir",
            str(enc_plot_dir),
        ]
        if args.max_queries > 0:
            enc_cmd.extend(["--max-queries", str(args.max_queries)])
        run_command(enc_cmd)

    if not args.skip_faiss:
        faiss_json = out_dir / "faiss_pq_results.json"
        faiss_plot_dir = out_dir / "faiss_pq_plots"
        faiss_cmd = [
            sys.executable,
            "benchmark_faiss_pq.py",
            "--image-embeddings",
            str(image_embeddings),
            "--valid-metadata",
            str(valid_metadata),
            "--kb-text-embeddings",
            str(kb_text_embeddings),
            "--entity-kb",
            str(entity_kb),
            "--out-json",
            str(faiss_json),
            "--plot-dir",
            str(faiss_plot_dir),
        ]
        run_command(faiss_cmd)

    print("\nBenchmark suite completed.")
    print(f"Results root: {out_dir}")


if __name__ == "__main__":
    main()
