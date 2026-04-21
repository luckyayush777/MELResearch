import os
import json
import random
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
import open_clip

# -- CONFIG --------------------------------------------------------------------
METADATA_PATH   = os.path.join("data", "metadata.json")
IMG_DIR         = os.path.join("data", "images")
OUT_DIR         = os.path.join("data", "embeddings")
SUBSET_PATH     = os.path.join("data", "subset_5k.json")

SAMPLE_SIZE     = 5000
BATCH_SIZE      = 256          # safe for 4090 with ViT-B/32
MODEL_NAME      = "ViT-B-32"
PRETRAINED      = "openai"     # OpenAI weights — the standard reference checkpoint
SEED            = 42
# ------------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

# -- 1. SNAPSHOT & SAMPLE ------------------------------------------------------
# Read metadata once and do not touch the file again (crawler may be writing it)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Metadata loaded: {len(metadata)} entries")

if os.path.exists(SUBSET_PATH):
    print(f"Loading existing subset from {SUBSET_PATH}")
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset = json.load(f)
else:
    # Filter to entries whose image actually exists on disk
    available = [item for item in metadata if os.path.exists(os.path.join(IMG_DIR, item["image_file"]))]
    print(f"Entries with image on disk: {len(available)}")
    if len(available) < SAMPLE_SIZE:
        raise RuntimeError(f"Only {len(available)} valid entries, need {SAMPLE_SIZE}. Run the crawler more.")
    subset = random.sample(available, SAMPLE_SIZE)
    with open(SUBSET_PATH, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"Saved subset to {SUBSET_PATH}")

print(f"Working with {len(subset)} samples")

# -- 2. LOAD MODEL ------------------------------------------------------------─
print(f"\nLoading {MODEL_NAME} ({PRETRAINED}) ...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(device).eval()
print("Model ready.")

# -- 3. HELPERS ----------------------------------------------------------------
def load_image(path):
    """Returns preprocessed tensor or None if image is unreadable."""
    try:
        img = Image.open(path).convert("RGB")
        return preprocess(img)
    except (UnidentifiedImageError, OSError):
        return None

def build_text(item):
    """title + first ~200 chars of lead_text, always fits in 77 CLIP tokens."""
    title = item.get("title", "")
    lead  = item.get("lead_text", "")[:200]
    return f"{title}. {lead}" if lead else title

# -- 4. IMAGE EMBEDDINGS ------------------------------------------------------─
print("\n-- Image embeddings --")
image_embeddings = []
valid_indices    = []   # track which subset entries succeeded

for batch_start in tqdm(range(0, len(subset), BATCH_SIZE)):
    batch_meta  = subset[batch_start : batch_start + BATCH_SIZE]
    tensors, idx = [], []

    for local_i, item in enumerate(batch_meta):
        t = load_image(os.path.join(IMG_DIR, item["image_file"]))
        if t is not None:
            tensors.append(t)
            idx.append(batch_start + local_i)
        else:
            print(f"  [WARN] Skipped unreadable image: {item['image_file']}")

    if not tensors:
        continue

    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise

    image_embeddings.append(feats.cpu().float().numpy())
    valid_indices.extend(idx)

image_embeddings = np.vstack(image_embeddings)   # (N, 512)
valid_indices    = np.array(valid_indices)
print(f"Image embeddings: {image_embeddings.shape}, valid samples: {len(valid_indices)}")

# -- 5. TEXT EMBEDDINGS --------------------------------------------------------
print("\n-- Text embeddings --")
valid_subset = [subset[i] for i in valid_indices]
text_embeddings = []

for batch_start in tqdm(range(0, len(valid_subset), BATCH_SIZE)):
    batch_meta = valid_subset[batch_start : batch_start + BATCH_SIZE]
    texts      = [build_text(item) for item in batch_meta]
    tokens     = tokenizer(texts).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    text_embeddings.append(feats.cpu().float().numpy())

text_embeddings = np.vstack(text_embeddings)   # (N, 512)
print(f"Text embeddings:  {text_embeddings.shape}")

# -- 6. SAVE ------------------------------------------------------------------─
print("\n-- Saving --")
np.save(os.path.join(OUT_DIR, "image_embeddings.npy"), image_embeddings)
np.save(os.path.join(OUT_DIR, "text_embeddings.npy"),  text_embeddings)
np.save(os.path.join(OUT_DIR, "valid_indices.npy"),    valid_indices)

# Also save the final valid subset metadata for easy lookup later
valid_meta_path = os.path.join(OUT_DIR, "valid_metadata.json")
with open(valid_meta_path, "w", encoding="utf-8") as f:
    json.dump(valid_subset, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUT_DIR}/")
print(f"  image_embeddings.npy  {image_embeddings.shape}")
print(f"  text_embeddings.npy   {text_embeddings.shape}")
print(f"  valid_indices.npy     {valid_indices.shape}")
print(f"  valid_metadata.json   {len(valid_subset)} entries")

# -- 7. QUICK SANITY CHECK ----------------------------------------------------─
print("\n-- Sanity check --")
sims = (image_embeddings[:10] * text_embeddings[:10]).sum(axis=1)
print(f"Diagonal cosine sims (image[i] · text[i]) for first 10 pairs:")
print(np.round(sims, 4))
print(f"Mean: {sims.mean():.4f}  (should be >> random baseline of ~0.0)")
print("\nDone.")
