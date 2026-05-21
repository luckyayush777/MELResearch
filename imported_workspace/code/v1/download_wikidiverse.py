#!/usr/bin/env python3
"""
WikiDiverse Dataset Downloader
==============================

Downloads all required files for the WikiDiverse multimodal entity linking dataset.

Source: https://github.com/wangxw5/wikiDiverse
Paper: ACL 2022 - WikiDiverse: A Multimodal Entity Linking Dataset with 
       Diversified Contextual Topics and Entity Types

Usage:
    python download_wikidiverse.py --output-dir ./data/wikidiverse
    python download_wikidiverse.py --output-dir ./data/wikidiverse --skip-images  # faster, no images
    python download_wikidiverse.py --output-dir ./data/wikidiverse --only-core    # minimal download
"""

import os
import sys
import argparse
import hashlib
import re
import json
import zipfile
from pathlib import Path
from typing import Optional
import subprocess

# Google Drive file IDs extracted from the repository
GDRIVE_FILES = {
    # Core dataset files
    "annotated_data": {
        "id": "1jsoa994_8tW9X19pb1cISKrMG8hTwItv",
        "filename": "wikidiverse_annotated.zip",
        "description": "Annotated data (train/valid/test splits)",
        "required": True,
    },
    "data_with_10_cands": {
        "id": "1ATTF_AzYAnUlM1N84S_dtFu-y867CELY",
        "filename": "wikidiverse_10cands.zip",
        "description": "Data with 10 retrieved candidates per mention",
        "required": True,
    },
    
    # Entity information
    "entity2desc_filtered": {
        "id": "1LKjcWrU6YdFfLX6iKi0cFKtyhf4t2bbe",
        "filename": "entity2desc_filtered.tsv",
        "description": "Entity descriptions (filtered)",
        "required": True,
    },
    "pem_data": {
        "id": "1Ss9cGb5c3nZtfzJvbFV_0-lEk1USBAAb",
        "filename": "pem_data.zip",
        "description": "P(e|m) prior probability data",
        "required": False,
    },
    
    # Images
    "wikinews_images": {
        "id": "1Xg7HxKbvhfKWrrHOYi2-59tE634ILTph",
        "filename": "wikinews_images.zip",
        "description": "Wikinews images (cleaned)",
        "required": False,  # Can skip for text-only experiments
    },
    "entity2img_urls": {
        "id": "1ukoThqll410GG3P0I7-29kg299OzYgOT",
        "filename": "entity2imgURLs.tsv",
        "description": "Entity to image URL mapping",
        "required": False,
    },
}

# Note: Original Wikipedia info is on Quark Drive (Chinese cloud), harder to download
QUARK_DRIVE_NOTE = """
Note: The original Wikipedia information is hosted on Quark Drive:
https://pan.quark.cn/s/d6a7b66efe21

This requires manual download. It contains:
- Full entity descriptions with EL annotations
- Entity images and captions
"""


def check_gdown():
    """Check if gdown is installed, install if not."""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        return True


def download_from_gdrive(file_id: str, output_path: Path, description: str = "") -> bool:
    """Download a file from Google Drive."""
    import gdown
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading: {description}")
    print(f"  -> {output_path}")
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        print(f"  Error: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file."""
    print(f"Extracting: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def convert_to_harness_format(data_dir: Path, output_dir: Path):
    """
    Convert WikiDiverse format to the harness expected format.
    
    WikiDiverse format (mention level with candidates):
    [sentence, img_url, mention, type, left_ctx, right_ctx, entity_url, candidates, topic, start, end]
    
    Harness format:
    mentions.json: [{id, mention, context, image_path, entity_id, candidates}]
    entities.json: [{id, name, description, image_path, wikidata_id, aliases}]
    """
    print("\nConverting to harness format...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the data files
    # They might be in subdirectories after extraction
    data_files = {}
    for split in ["train", "valid", "test"]:
        # Look for files with candidates (mention-level)
        patterns = [
            f"{split}_w_10cands.json",
            f"{split}.json",
            f"*/{split}_w_10cands.json",
            f"*/{split}.json",
        ]
        for pattern in patterns:
            matches = list(data_dir.glob(pattern))
            if matches:
                data_files[split] = matches[0]
                break
    
    if not data_files:
        print("  Warning: Could not find data files to convert")
        print(f"  Searched in: {data_dir}")
        return
    
    # Collect all entities and mentions
    all_mentions = []
    all_entities = {}
    mention_id = 0
    
    for split, filepath in data_files.items():
        print(f"  Processing {split}: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        split_mentions = []
        
        for item in data:
            # Parse WikiDiverse format
            if len(item) >= 11:
                # Mention-level format with candidates
                sentence, img_url, mention_text, mention_type, left_ctx, right_ctx, entity_url, candidates, topic, start, end = item[:11]
            elif len(item) >= 4:
                # Passage-level format
                sentence, img_url, topic, annotations = item[:4]
                for ann in annotations:
                    mention_text, mention_type, start, end, entity_url = ann
                    candidates = []
                    left_ctx = []
                    right_ctx = []
            else:
                continue
            
            # Extract entity ID from URL
            entity_id = entity_url.split("/wiki/")[-1] if entity_url else f"UNK_{mention_id}"
            
            # Create mention entry
            mention_entry = {
                "id": f"m_{mention_id}",
                "mention": mention_text,
                "context": sentence,
                "image_url": img_url,
                "entity_id": entity_id,
                "candidates": [c.split("/wiki/")[-1] if isinstance(c, str) and "/wiki/" in c else c for c in (candidates or [])],
                "topic": topic,
                "mention_type": mention_type,
                "start": start,
                "end": end,
                "split": split,
            }
            split_mentions.append(mention_entry)
            
            # Add entity
            if entity_id not in all_entities:
                all_entities[entity_id] = {
                    "id": entity_id,
                    "name": entity_id.replace("_", " "),
                    "url": entity_url,
                    "description": "",  # Will be filled from entity2desc if available
                    "image_path": None,
                    "wikidata_id": None,
                    "aliases": [],
                }
            
            mention_id += 1
        
        # Save split-specific mentions
        with open(output_dir / f"{split}_mentions.json", 'w', encoding='utf-8') as f:
            json.dump(split_mentions, f, indent=2, ensure_ascii=False)
        
        all_mentions.extend(split_mentions)
        print(f"    {len(split_mentions)} mentions")
    
    # Try to load entity descriptions
    desc_files = list(data_dir.glob("**/entity2desc*.tsv")) + list(data_dir.glob("**/entity2desc*.txt"))
    if desc_files:
        print(f"  Loading entity descriptions from {desc_files[0]}")
        try:
            with open(desc_files[0], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split("@@@@")
                    if len(parts) >= 2:
                        entity_name = parts[0].strip()
                        description = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Match to entities
                        entity_id = entity_name.replace(" ", "_")
                        if entity_id in all_entities:
                            all_entities[entity_id]["description"] = description
        except Exception as e:
            print(f"    Warning: Error loading descriptions: {e}")
    
    # Save entities
    with open(output_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(list(all_entities.values()), f, indent=2, ensure_ascii=False)
    
    print(f"  Total: {len(all_mentions)} mentions, {len(all_entities)} entities")
    print(f"  Output: {output_dir}")


def create_image_path_mapping(data_dir: Path, images_dir: Path):
    """
    Create mapping from image URLs to local paths.
    WikiDiverse uses MD5 hash of filename as prefix.
    """
    mapping = {}
    
    if not images_dir.exists():
        return mapping
    
    for img_path in images_dir.glob("*"):
        if img_path.is_file():
            mapping[img_path.name] = str(img_path)
    
    # Save mapping
    with open(data_dir / "image_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)
    
    return mapping


def download_wikidiverse(output_dir: str, 
                         skip_images: bool = False,
                         only_core: bool = False,
                         convert: bool = True):
    """Main download function."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloads_dir = output_path / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    check_gdown()
    
    print("=" * 60)
    print("WikiDiverse Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Skip images: {skip_images}")
    print(f"Only core files: {only_core}")
    print()
    
    # Determine which files to download
    files_to_download = {}
    for name, info in GDRIVE_FILES.items():
        if only_core and not info["required"]:
            continue
        if skip_images and "image" in name.lower():
            continue
        files_to_download[name] = info
    
    # Download files
    downloaded = []
    failed = []
    
    for name, info in files_to_download.items():
        file_path = downloads_dir / info["filename"]
        
        if file_path.exists():
            print(f"Already exists: {info['filename']}")
            downloaded.append(name)
            continue
        
        success = download_from_gdrive(
            info["id"], 
            file_path, 
            info["description"]
        )
        
        if success:
            downloaded.append(name)
        else:
            failed.append(name)
    
    print()
    
    # Extract zip files
    for name in downloaded:
        info = GDRIVE_FILES[name]
        if info["filename"].endswith(".zip"):
            zip_path = downloads_dir / info["filename"]
            if zip_path.exists():
                extract_zip(zip_path, output_path)
    
    # Convert to harness format
    if convert:
        convert_to_harness_format(output_path, output_path / "processed")
    
    # Create image mapping if images were downloaded
    if not skip_images:
        images_dir = output_path / "wikinewsImgs"
        if images_dir.exists():
            create_image_path_mapping(output_path, images_dir)
    
    # Summary
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Downloaded: {len(downloaded)} files")
    if failed:
        print(f"Failed: {len(failed)} files")
        for name in failed:
            print(f"  - {GDRIVE_FILES[name]['description']}")
    
    print()
    print("Directory structure:")
    print(f"  {output_path}/")
    print(f"    downloads/     - Raw downloaded files")
    print(f"    processed/     - Converted to harness format")
    print(f"      train_mentions.json")
    print(f"      valid_mentions.json")
    print(f"      test_mentions.json")
    print(f"      entities.json")
    
    if not skip_images:
        print(f"    wikinewsImgs/  - Wikinews images")
    
    print()
    print(QUARK_DRIVE_NOTE)
    
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download WikiDiverse dataset for multimodal entity linking"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/wikidiverse",
        help="Output directory for downloaded data"
    )
    
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip downloading images (faster, for text-only experiments)"
    )
    
    parser.add_argument(
        "--only-core",
        action="store_true",
        help="Download only core required files (annotated data + candidates)"
    )
    
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip converting to harness format"
    )
    
    args = parser.parse_args()
    
    success = download_wikidiverse(
        output_dir=args.output_dir,
        skip_images=args.skip_images,
        only_core=args.only_core,
        convert=not args.no_convert
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()