import os
import json

IMG_DIR = os.path.join("data", "images")
METADATA_PATH = os.path.join("data", "metadata.json")

# Load metadata and build set of referenced image filenames
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

referenced = set(item["image_file"] for item in metadata)
print(f"Metadata entries: {len(metadata)}")
print(f"Unique referenced images: {len(referenced)}")

# Find all files on disk
all_files = os.listdir(IMG_DIR)
print(f"Files on disk: {len(all_files)}")

# Identify orphans
orphans = [f for f in all_files if f not in referenced]
print(f"Orphan files (no metadata): {len(orphans)}")

if not orphans:
    print("Nothing to clean up.")
else:
    confirm = input(f"\nDelete {len(orphans)} orphan files? (y/n): ").strip().lower()
    if confirm == "y":
        deleted = 0
        for f in orphans:
            path = os.path.join(IMG_DIR, f)
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"  Failed to delete {f}: {e}")
        print(f"Deleted {deleted} orphan files.")
    else:
        print("Aborted. No files deleted.")
