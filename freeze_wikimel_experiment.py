"""Freeze evaluation and environment manifests after the data audit passes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    experiment_root = args.experiment_root.resolve()
    repository_root = Path(__file__).resolve().parent
    test = load_json(data_root / "WIKIMEL_test.json")
    kb = load_json(data_root / "kb_entity.json")
    audit = load_json(experiment_root / "audit_detailed.json")
    if not audit.get("acceptance_gate", {}).get("accepted"):
        raise SystemExit("Detailed audit has not passed; refusing to freeze evaluation manifest.")

    query_ids = [str(row.get("id")) for row in test]
    gold_ids = [str(row.get("answer")) for row in test]
    entity_ids = [str(entity.get("qid")) for entity in kb]
    evaluation = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str((data_root / "dataset_manifest.json").resolve()),
        "dataset_manifest_sha256": hashlib.sha256((data_root / "dataset_manifest.json").read_bytes()).hexdigest(),
        "query_count": len(query_ids),
        "entity_count": len(entity_ids),
        "test_query_ids": query_ids,
        "test_gold_entity_ids": gold_ids,
        "ordered_kb_entity_ids": entity_ids,
        "query_order_sha256": sha256_json(query_ids),
        "gold_order_sha256": sha256_json(gold_ids),
        "entity_id_ordering_sha256": sha256_json(entity_ids),
        "candidate_policy": "full_kb",
        "score_direction": "higher_is_better",
        "tie_handling": "rank = 1 + count(scores strictly greater than gold score)",
        "metrics": ["H@1", "H@3", "H@5", "H@10", "H@20", "MRR", "mean_rank", "median_rank", "p95_rank"],
        "evaluation_chunk_size": args.chunk_size,
        "model_selection": "best development MRR only",
        "test_policy": "evaluate once after configuration and checkpoint selection are frozen",
        "git_commit": command("git", "rev-parse", "HEAD", cwd=repository_root),
        "git_worktree_clean": not bool(command("git", "status", "--porcelain", cwd=repository_root)),
    }
    write_json(experiment_root / "evaluation_manifest.json", evaluation)

    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    packages = {}
    for package in ("numpy", "Pillow", "PyYAML", "pytorch-lightning", "transformers", "torchvision"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    environment = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_info,
        "packages": packages,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "git_commit": evaluation["git_commit"],
        "git_worktree_clean": evaluation["git_worktree_clean"],
    }
    write_json(experiment_root / "environment.json", environment)
    print(json.dumps({
        "evaluation_manifest": str(experiment_root / "evaluation_manifest.json"),
        "environment": str(experiment_root / "environment.json"),
        "queries": len(query_ids),
        "entities": len(entity_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
