import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_records(path: str) -> List[Dict]:
    lower = path.lower()
    with open(path, "r", encoding="utf-8") as f:
        if lower.endswith(".jsonl"):
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
    raise ValueError(f"Unsupported record format in {path}")


def resolve_image_path(raw_path: str, dataset_root: str, fallback_root: str) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    if dataset_root:
        full = Path(dataset_root) / p
        return str(full.resolve())
    full = Path(fallback_root) / p
    return str(full.resolve())


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_text(title: str, body: str, body_limit: int) -> str:
    title = normalize_text(title)
    body = normalize_text(body)[:body_limit]
    if title and body:
        return f"{title}. {body}"
    return title or body


def encode_images(
    image_paths: List[str],
    model,
    preprocess,
    device,
    batch_size: int,
    use_amp: bool,
) -> Tuple[np.ndarray, List[int], int]:
    import torch

    all_embeddings = []
    kept_indices = []
    skipped = 0

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        local_kept = []

        for i, image_path in enumerate(batch_paths):
            try:
                tensor = preprocess(Image.open(image_path).convert("RGB"))
                tensors.append(tensor)
                local_kept.append(start + i)
            except (UnidentifiedImageError, OSError, FileNotFoundError):
                skipped += 1

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        if use_amp:
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embeddings.append(feats.detach().cpu().float().numpy())
        kept_indices.extend(local_kept)

    if not all_embeddings:
        raise RuntimeError("No valid images could be encoded. Check image paths and formats.")

    return np.vstack(all_embeddings).astype(np.float32), kept_indices, skipped


def encode_texts(texts: List[str], model, tokenizer, device, batch_size: int, use_amp: bool) -> np.ndarray:
    import torch

    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch).to(device)

        if use_amp:
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)

        outputs.append(feats.detach().cpu().float().numpy())

    if not outputs:
        raise RuntimeError("No text embeddings were produced.")

    return np.vstack(outputs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WikiMEL-style artifacts compatible with this repository's benchmark scripts."
    )
    parser.add_argument("--queries", required=True, help="Path to query records (.json or .jsonl).")
    parser.add_argument("--entities", required=True, help="Path to entity records (.json or .jsonl).")
    parser.add_argument("--dataset-root", default="", help="Optional root for resolving relative image paths.")
    parser.add_argument("--out-dir", default=os.path.join("data", "wikimel_v1"))
    parser.add_argument("--max-queries", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--text-char-limit", type=int, default=300)

    parser.add_argument("--query-id-field", default="query_id")
    parser.add_argument("--query-entity-id-field", default="entity_id")
    parser.add_argument("--query-image-field", default="image_path")
    parser.add_argument("--query-title-field", default="title")
    parser.add_argument("--query-text-field", default="query_text")

    parser.add_argument("--entity-id-field", default="entity_id")
    parser.add_argument("--entity-title-field", default="title")
    parser.add_argument("--entity-description-field", default="description")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    query_rows = load_records(args.queries)
    entity_rows = load_records(args.entities)

    entity_map = {}
    entity_kb = []
    for row in entity_rows:
        entity_id = normalize_text(row.get(args.entity_id_field))
        if not entity_id:
            continue
        title = normalize_text(row.get(args.entity_title_field))
        description = normalize_text(row.get(args.entity_description_field))
        entity_map[entity_id] = {
            "entity_id": entity_id,
            "title": title,
            "description": description,
        }

    if not entity_map:
        raise RuntimeError("No valid entities loaded. Verify entity field names.")

    for entity_id in sorted(entity_map.keys()):
        entity_kb.append(entity_map[entity_id])

    # Build query set with strict validity checks.
    queries_parent = str(Path(args.queries).resolve().parent)
    prepared_queries = []
    dropped_missing_entity = 0
    dropped_missing_image = 0

    for row in query_rows:
        query_id = normalize_text(row.get(args.query_id_field))
        entity_id = normalize_text(row.get(args.query_entity_id_field))
        raw_image = normalize_text(row.get(args.query_image_field))

        if not entity_id or entity_id not in entity_map:
            dropped_missing_entity += 1
            continue
        if not raw_image:
            dropped_missing_image += 1
            continue

        image_path = resolve_image_path(raw_image, args.dataset_root, queries_parent)
        if not os.path.exists(image_path):
            dropped_missing_image += 1
            continue

        query_title = normalize_text(row.get(args.query_title_field))
        query_text = normalize_text(row.get(args.query_text_field))
        gold = entity_map[entity_id]

        # Prefer query-provided text when available; otherwise fall back to gold entity text.
        paired_text = build_text(
            title=query_title or gold["title"],
            body=query_text or gold["description"],
            body_limit=args.text_char_limit,
        )

        prepared_queries.append(
            {
                "query_id": query_id or f"q_{len(prepared_queries)}",
                "entity_id": entity_id,
                "image_path": image_path,
                "title": query_title or gold["title"],
                "lead_text": query_text,
                "paired_text": paired_text,
            }
        )

    if args.max_queries > 0:
        random.shuffle(prepared_queries)
        prepared_queries = prepared_queries[: args.max_queries]

    if not prepared_queries:
        raise RuntimeError("No valid queries found after filtering.")

    print(f"Loaded entities: {len(entity_kb)}")
    print(f"Valid queries: {len(prepared_queries)}")
    print(f"Dropped queries (missing entity): {dropped_missing_entity}")
    print(f"Dropped queries (missing image): {dropped_missing_image}")

    import torch
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(torch.cuda.is_available())

    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    query_image_paths = [q["image_path"] for q in prepared_queries]
    image_embeddings, kept_indices, skipped_unreadable = encode_images(
        image_paths=query_image_paths,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        use_amp=use_amp,
    )

    filtered_queries = [prepared_queries[i] for i in kept_indices]
    paired_texts = [q["paired_text"] for q in filtered_queries]

    text_embeddings = encode_texts(
        texts=paired_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        use_amp=use_amp,
    )

    kb_texts = [
        build_text(
            title=e.get("title", ""),
            body=e.get("description", ""),
            body_limit=args.text_char_limit,
        )
        for e in entity_kb
    ]
    kb_text_embeddings = encode_texts(
        texts=kb_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        use_amp=use_amp,
    )

    valid_metadata = []
    query_to_entity = []
    for q in filtered_queries:
        valid_metadata.append(
            {
                "query_id": q["query_id"],
                "title": q["title"],
                "image_file": q["image_path"],
                "lead_text": q.get("lead_text", ""),
                "entity_id": q["entity_id"],
            }
        )
        query_to_entity.append(
            {
                "query_id": q["query_id"],
                "entity_id": q["entity_id"],
            }
        )

    entity_kb_path = out_dir / "entity_kb.json"
    valid_meta_path = emb_dir / "valid_metadata.json"
    mapping_path = emb_dir / "query_to_entity.json"

    np.save(emb_dir / "image_embeddings.npy", image_embeddings)
    np.save(emb_dir / "text_embeddings.npy", text_embeddings)
    np.save(emb_dir / "kb_text_embeddings.npy", kb_text_embeddings)

    with entity_kb_path.open("w", encoding="utf-8") as f:
        json.dump(entity_kb, f, indent=2, ensure_ascii=False)

    with valid_meta_path.open("w", encoding="utf-8") as f:
        json.dump(valid_metadata, f, indent=2, ensure_ascii=False)

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(query_to_entity, f, indent=2, ensure_ascii=False)

    spec = {
        "artifacts": {
            "entity_kb": str(entity_kb_path),
            "valid_metadata": str(valid_meta_path),
            "image_embeddings": str(emb_dir / "image_embeddings.npy"),
            "paired_text_embeddings": str(emb_dir / "text_embeddings.npy"),
            "kb_text_embeddings": str(emb_dir / "kb_text_embeddings.npy"),
            "query_to_entity": str(mapping_path),
        },
        "dataset": {
            "entities": len(entity_kb),
            "queries_before_image_decode": len(prepared_queries),
            "queries_after_image_decode": len(filtered_queries),
            "dropped_missing_entity": dropped_missing_entity,
            "dropped_missing_image": dropped_missing_image,
            "dropped_unreadable_image": skipped_unreadable,
        },
        "model": {
            "name": args.model_name,
            "pretrained": args.pretrained,
            "device": str(device),
            "batch_size": args.batch_size,
            "mixed_precision": use_amp,
        },
        "input": {
            "queries": args.queries,
            "entities": args.entities,
            "dataset_root": args.dataset_root,
            "field_mapping": {
                "query_id_field": args.query_id_field,
                "query_entity_id_field": args.query_entity_id_field,
                "query_image_field": args.query_image_field,
                "query_title_field": args.query_title_field,
                "query_text_field": args.query_text_field,
                "entity_id_field": args.entity_id_field,
                "entity_title_field": args.entity_title_field,
                "entity_description_field": args.entity_description_field,
            },
        },
    }

    spec_path = out_dir / "benchmark_spec.json"
    with spec_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print("\nSaved WikiMEL-compatible artifacts:")
    print(f"  - {entity_kb_path}")
    print(f"  - {valid_meta_path}")
    print(f"  - {emb_dir / 'image_embeddings.npy'}")
    print(f"  - {emb_dir / 'text_embeddings.npy'}")
    print(f"  - {emb_dir / 'kb_text_embeddings.npy'}")
    print(f"  - {mapping_path}")
    print(f"  - {spec_path}")


if __name__ == "__main__":
    main()
