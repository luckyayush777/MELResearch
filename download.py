import os
import requests
import time
import json
import traceback

SAVE_DIR = "data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
METADATA_PATH = os.path.join(SAVE_DIR, "metadata.json")

TARGET_COUNT = 10000
SLEEP_TIME = 0.5

HEADERS = {
    "User-Agent": "MEL-Quantization-Research/1.0 (ayushrocks456@gmail.com)"
}

os.makedirs(IMG_DIR, exist_ok=True)

# -------- LOAD EXISTING METADATA IF PRESENT --------
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        collected = json.load(f)
    print(f"Resuming. Already have {len(collected)} samples.")
else:
    collected = []

seen_titles = set(item["title"] for item in collected)
attempts = 0


def get_random_page():
    url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def download_image(url, save_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        return False
    return False


def save_metadata():
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)


while len(collected) < TARGET_COUNT:
    attempts += 1
    page = get_random_page()
    if page is None:
        continue

    title = page.get("title")
    if not title:
        continue

    if title in seen_titles:
        continue

    if page.get("type") == "disambiguation":
        continue

    if page.get("description") and "disambiguation" in page.get("description").lower():
        continue

    if title.startswith("List of"):
        continue

    image_info = page.get("originalimage")
    if not image_info:
        continue

    image_url = image_info.get("source")
    if not image_url:
        continue

    if not image_url.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filename = title.replace(" ", "_").replace("/", "_") + \
               os.path.splitext(image_url)[-1]
    save_path = os.path.join(IMG_DIR, filename)

    success = download_image(image_url, save_path)
    if not success:
        continue

    seen_titles.add(title)

    collected.append({
        "title": title,
        "image_file": filename,
        "lead_text": page.get("extract", "")
    })

    # 🔥 SAVE AFTER EVERY SUCCESS
    save_metadata()

    print(f"[SUCCESS] Collected {len(collected)}/{TARGET_COUNT}")
    time.sleep(SLEEP_TIME)

print(f"\nDone. Collected {len(collected)} pages.")