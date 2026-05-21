import json

INPUT = "../datasets/Richpedia-MEL.json"
OUTPUT = "entities.jsonl"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)


entities = data["entities"]

out = open(OUTPUT, "w", encoding="utf-8")

for idx, e in enumerate(entities):
    title = e["title"].strip()
    desc = e.get("description", "").strip()

    if len(desc.split()) < 20:
        continue  # drop weak entities

    text = f"{title}. {desc}"

    record = {
        "entity_id": idx,
        "kb_id": e["id"],          # original ID
        "title": title,
        "text": text,
        "aliases": e.get("aliases", [])
    }

    out.write(json.dumps(record) + "\n")

out.close()
print("Wrote entities.jsonl")
