import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_key(row):
    return (row.get("mentions"), row.get("sentence"), row.get("imgPath"), row.get("answer"))


def text_key(row):
    return (row.get("mentions"), row.get("sentence"))


def image_key(row):
    return row.get("imgPath")


def pct(count: int, total: int) -> str:
    return f"{count / total:.2%}" if total else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MIMIC-formatted WikiMEL data for split and KB-image leakage.")
    parser.add_argument("--data-root", default="external/MIMIC_reproduction/data/WikiMEL")
    args = parser.parse_args()

    root = Path(args.data_root)
    splits = {
        "train": load_json(root / "WIKIMEL_train.json"),
        "dev": load_json(root / "WIKIMEL_dev.json"),
        "test": load_json(root / "WIKIMEL_test.json"),
    }
    kb = load_json(root / "kb_entity.json")
    qid2id = load_json(root / "qid2id.json")

    print(f"Data root: {root}")
    for name, rows in splits.items():
        print(
            f"{name}: rows={len(rows)} "
            f"unique_answers={len({row.get('answer') for row in rows})} "
            f"unique_images={len({row.get('imgPath') for row in rows})} "
            f"unique_texts={len({text_key(row) for row in rows})}"
        )

    print("\nSplit overlap:")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for label, key_fn in (("exact", row_key), ("mention_sentence", text_key), ("image", image_key)):
            a = {key_fn(row) for row in splits[left] if key_fn(row)}
            b = {key_fn(row) for row in splits[right] if key_fn(row)}
            overlap = len(a & b)
            print(f"{left}-{right} {label}: {overlap} overlap ({pct(overlap, min(len(a), len(b))) } of smaller)")

    print("\nGold entity image leakage:")
    for name, rows in splits.items():
        leaked = 0
        for row in rows:
            answer = row.get("answer")
            image = row.get("imgPath")
            if not answer or answer not in qid2id or not image:
                continue
            entity = kb[qid2id[answer]]
            if image in entity.get("image_list", []):
                leaked += 1
        print(f"{name}: {leaked}/{len(rows)} rows have mention image inside gold entity image_list ({pct(leaked, len(rows))})")


if __name__ == "__main__":
    main()
