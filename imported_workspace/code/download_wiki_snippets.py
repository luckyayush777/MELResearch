import json
import time
import requests
from pathlib import Path

# -------- CONFIG --------
RICH_PEDIA_PATH = Path("../datasets/Richpedia-MEL.json")
OUTPUT_PATH = Path("../datasets/entity_snippets.json")
SLEEP_SEC = 0.5   # be polite to Wikipedia
LANG = "en"
# ------------------------

WIKI_API = f"https://{LANG}.wikipedia.org/api/rest_v1/page/summary/"

def fetch_summary(title):
    url = WIKI_API + title.replace(" ", "_")
    r = requests.get(url, headers={"User-Agent": "MEL-thesis-bot/1.0"})
    if r.status_code != 200:
        return None
    data = r.json()
    if "extract" not in data:
        return None
    return {
        "title": data.get("title", title),
        "extract": data.get("extract", "")
    }

def main():
    with open(RICH_PEDIA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # 1. collect unique entities
    entities = {}
    for v in data.values():
        qid = v["answer"]
        name = v["entities"]
        entities[qid] = name

    print(f"Found {len(entities)} unique entities")

    # 2. download snippets
    kb = {}
    for i, (qid, name) in enumerate(entities.items(), 1):
        print(f"[{i}/{len(entities)}] Fetching {name}")
        res = fetch_summary(name)
        if res and len(res["extract"].split()) >= 20:
            kb[qid] = {
                "qid": qid,
                "title": res["title"],
                "text": f'{res["title"]}. {res["extract"]}'
            }
        else:
            print(f"  -> skipped (no extract or too short)")

        time.sleep(SLEEP_SEC)

    # 3. save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(kb)} entity snippets to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
