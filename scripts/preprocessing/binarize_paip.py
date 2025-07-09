"""
Batch-binarize PAIP2019 Kaggle segmentation masks into per-case
"seg_files_bin/0" folders, showing progress for all cases and per-case files.

Assumes your root input directory contains case subfolders like:
    F:/Datasets/PAIP2019/PAIP_2019/PAIP_2019/PAIP_2019/01_01_0083/
which inside have:
    seg_files/0/*.jpg

This script will write binary PNGs (pixel > 0 -> 255) to:
    01_01_0083/seg_files_bin/0/*.png
and show two progress bars: one for cases, one for files within each case.
"""

import os
import cv2
from tqdm import tqdm

# INPUT_ROOT = r"F:/Datasets/PAIP2019/PAIP_2019/PAIP_2019/PAIP_2019/"
INPUT_ROOT = r"/data/research/PAIP2019/PAIP_2019/PAIP_2019/PAIP_2019/"
THRESH     = 0  # Pixel > threshold becomes foreground

# Find all case names with a seg_files subfolder
cases = sorted(
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d, "seg_files"))
)

for case_name in tqdm(cases, desc="Cases", unit="case"):
    case_dir = os.path.join(INPUT_ROOT, case_name)
    seg_root = os.path.join(case_dir, "seg_files")

    # Gather all (in_path, out_path) for this case
    file_list = []
    for sub in sorted(os.listdir(seg_root)):
        in_subdir  = os.path.join(seg_root, sub)
        if not os.path.isdir(in_subdir):
            continue
        out_subdir = os.path.join(case_dir, "seg_files_bin", sub)
        os.makedirs(out_subdir, exist_ok=True)

        for fname in sorted(os.listdir(in_subdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            in_path  = os.path.join(in_subdir, fname)
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(out_subdir, out_name)
            file_list.append((in_path, out_path))

    # Process each mask with an inner progress bar
    for in_path, out_path in tqdm(file_list, desc=f"{case_name}", unit="file", leave=False):
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            tqdm.write(f"[WARNING] could not read {in_path}")
            continue

        # Binarize: any pixel > threshold -> 255
        _, bin_mask = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY)

        # Save as PNG with zero compression
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, bin_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])