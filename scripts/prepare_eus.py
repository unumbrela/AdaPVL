"""
Prepare EUS dataset into Train_Folder/Val_Folder/Test_Folder structure.

Raw data layout (from Google Drive):
  data/EUS/C01-<timestamp>/C01/V1_0164.tif
  data/EUS/Annotations-<timestamp>/Annotations/C01/V1_0164.tif

Expected layout (for training):
  data/EUS/Train_Folder/img/C01_V1_0164.png
  data/EUS/Train_Folder/label/C01_V1_0164.png

The dataloader (cv2.imread) reads by file content, not extension,
so we can move .tif files and rename them to .png safely.

Usage:
  python scripts/prepare_eus.py           # move files (C-prefix only)
  python scripts/prepare_eus.py --all     # include H-prefix (incomplete)
  python scripts/prepare_eus.py --dry-run # report only
"""

import os
import glob
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

EUS_DIR = Path("data/EUS")
PROMPTS_DIR = EUS_DIR / "Prompts_Folder"


def find_raw_dirs():
    """Build mapping: collection_name -> (image_dir, annotation_dir)"""
    img_dirs = {}
    for d in sorted(glob.glob(str(EUS_DIR / "C*-*/C*/"))):
        name = Path(d).name
        img_dirs[name] = Path(d)

    ann_base = None
    for d in glob.glob(str(EUS_DIR / "Annotations-*/Annotations/")):
        ann_base = Path(d)
        break

    ann_dirs = {}
    if ann_base:
        for sub in sorted(ann_base.iterdir()):
            if sub.is_dir():
                ann_dirs[sub.name] = sub

    return img_dirs, ann_dirs


def build_file_index(img_dirs):
    """Build index: video_frame -> list of (collection_name) containing it."""
    index = defaultdict(list)
    for cname, cdir in img_dirs.items():
        for f in os.listdir(cdir):
            if f.endswith(".tif"):
                index[f.replace(".tif", "")].append(cname)
    return index


def resolve_c_prefix(prefix, video_frame, img_dirs, ann_dirs):
    """Resolve a C-prefix entry to source image and annotation paths."""
    img_dir = img_dirs.get(prefix)
    ann_dir = ann_dirs.get(prefix)
    if not img_dir:
        return None, None
    img_file = img_dir / f"{video_frame}.tif"
    ann_file = ann_dir / f"{video_frame}.tif" if ann_dir else None
    if not img_file.exists():
        return None, None
    if ann_file and not ann_file.exists():
        ann_file = None
    return img_file, ann_file


def move_file(src, dst, dry_run=False):
    """Move a .tif file to a .png-named destination (cv2 reads by content)."""
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return  # already processed
    shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser(description="Prepare EUS dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report statistics, don't move files")
    parser.add_argument("--all", action="store_true",
                        help="Include H-prefix entries (may be incomplete)")
    args = parser.parse_args()

    print("=== EUS Dataset Preparation ===\n")

    img_dirs, ann_dirs = find_raw_dirs()
    print(f"Image collections: {sorted(img_dirs.keys())}")
    print(f"Annotation collections: {sorted(ann_dirs.keys())}")
    file_index = build_file_index(img_dirs)
    print(f"Total unique frames: {len(file_index)}\n")

    splits = {
        "Train_Folder": PROMPTS_DIR / "Train_text.xlsx",
        "Val_Folder": PROMPTS_DIR / "Val_text.xlsx",
        "Test_Folder": PROMPTS_DIR / "Test_text_original.xlsx",
    }

    stats = defaultdict(lambda: {"ok": 0, "img_miss": 0, "ann_miss": 0,
                                  "h_skip": 0, "total": 0})
    missing_samples = []

    for split_name, excel_path in splits.items():
        df = pd.read_excel(excel_path)
        s = stats[split_name]
        s["total"] = len(df)

        img_out = EUS_DIR / split_name / "img"
        label_out = EUS_DIR / split_name / "label"

        for _, row in df.iterrows():
            filename = row["Image"]  # C01_V1_0164.png
            gt_filename = row["Ground Truth"]
            parts = filename.replace(".png", "").split("_", 1)
            prefix, video_frame = parts[0], parts[1]

            # Skip H-prefix unless --all
            if prefix.startswith("H") and not args.all:
                s["h_skip"] += 1
                continue

            # For C-prefix: direct resolution
            if prefix.startswith("C"):
                src_img, src_ann = resolve_c_prefix(
                    prefix, video_frame, img_dirs, ann_dirs)
            else:
                # H-prefix: try corresponding C dir, then search
                c_equiv = "C" + prefix[1:]
                src_img, src_ann = resolve_c_prefix(
                    c_equiv, video_frame, img_dirs, ann_dirs)
                if src_img is None:
                    # Search all C dirs
                    for cname in file_index.get(video_frame, []):
                        src_img, src_ann = resolve_c_prefix(
                            cname, video_frame, img_dirs, ann_dirs)
                        if src_img:
                            break

            if src_img is None:
                s["img_miss"] += 1
                if len(missing_samples) < 5:
                    missing_samples.append(filename)
                continue

            if src_ann is None:
                s["ann_miss"] += 1
                continue

            s["ok"] += 1
            move_file(src_img, img_out / filename, args.dry_run)
            move_file(src_ann, label_out / gt_filename, args.dry_run)

    # Print report
    print(f"{'Split':<15} {'Total':>7} {'OK':>7} {'H-skip':>7} {'Img miss':>9} {'Ann miss':>9}")
    print("-" * 60)
    grand = defaultdict(int)
    for split_name in splits:
        s = stats[split_name]
        print(f"{split_name:<15} {s['total']:>7} {s['ok']:>7} {s['h_skip']:>7} "
              f"{s['img_miss']:>9} {s['ann_miss']:>9}")
        for k in s:
            grand[k] += s[k]
    print("-" * 60)
    print(f"{'TOTAL':<15} {grand['total']:>7} {grand['ok']:>7} {grand['h_skip']:>7} "
          f"{grand['img_miss']:>9} {grand['ann_miss']:>9}")

    if missing_samples:
        print(f"\nSample missing: {missing_samples}")

    if args.dry_run:
        print("\n[DRY RUN] No files moved. Remove --dry-run to execute.")
    else:
        print(f"\nDone! Files moved to {EUS_DIR}/{{Train,Val,Test}}_Folder/")


if __name__ == "__main__":
    main()
