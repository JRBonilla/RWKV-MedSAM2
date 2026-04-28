#!/usr/bin/env python3
# scripts/emit_qc_cache_from_pickled_loaders.py
"""
Emit QC cache files *purely* from previously pickled DataLoaders (no QC, no rebuild).
- Loads the three loaders using load_data_loaders(...) from train_sam2.py
- Walks their datasets to collect 3D mask paths per dataset
- Writes cache JSONs (ok=True) via utils.preprocessing.merge_qc_cache_results
- Never touches get_data_loaders / get_sequences / QC functions

Usage:
  python scripts/emit_qc_cache_from_pickled_loaders.py \
      --loaders /path/to/loaders.pkl \
      --out-dir /path/to/dripp_output_dir \
      [--downsample 2] [--min-voxels 64] [--min-slices 2] [--min-area 32] \
      [--skip-train] [--skip-test]
"""

import os
import sys
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rwkv_medsam2.train_sam2 import load_data_loaders
from rwkv_medsam2.utils.preprocessing import merge_qc_cache_results

def _qc_params(args):
    return {
        "min_voxels": int(args.min_voxels),
        "min_slices_any_axis": int(args.min_slices),
        "min_slice_area_px": int(args.min_area),
        "downsample": int(args.downsample),
    }

def _collect_3d_masks_from_dataset(ds):
    """
    SegmentationSequenceDataset holds a list of sequence dicts, usually on ds.data or ds.sequences:
      {'dataset', 'subdataset', 'tasks', 'sequence': [(img, mask), ...], 'dim': 2|3}
    We only want sequences with dim == 3 and collect their mask paths, grouped by 'dataset'.
    """
    by_ds = defaultdict(list)
    seqs = getattr(ds, "data", None) or getattr(ds, "sequences", None) or []
    for s in seqs:
        try:
            if int(s.get("dim", 0)) != 3:
                continue
            ds_name = s.get("dataset", "unknown")
            for (img_p, mask_p) in s.get("sequence", []):
                if mask_p:
                    by_ds[ds_name].append(mask_p)
        except Exception:
            # Be robust to any odd sequence entry
            continue
    return by_ds

def _dedup_keep_order(paths):
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loaders", required=True, help="Path to pickled loaders (created by save_data_loaders)")
    ap.add_argument("--out-dir", required=True, help="DRIPP output_dir (parent folder containing per-dataset subfolders)")
    ap.add_argument("--skip-train", action="store_true", help="Skip emitting train (train+val) cache")
    ap.add_argument("--skip-test", action="store_true", help="Skip emitting test cache")
    # Quality params MUST match what you want the cache to represent
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--min-voxels", type=int, default=64)
    ap.add_argument("--min-slices", type=int, default=2)
    ap.add_argument("--min-area",   type=int, default=32)
    args = ap.parse_args()

    # Load pickled loaders (no building/QC)
    train_loader, val_loader, test_loader = load_data_loaders(args.loaders)
    params = _qc_params(args)

    # -------- TRAIN + VAL → "train" cache --------
    if not args.skip_train and train_loader is not None and val_loader is not None:
        by_ds_train = _collect_3d_masks_from_dataset(train_loader.dataset)
        by_ds_val   = _collect_3d_masks_from_dataset(val_loader.dataset)
        for ds_name, masks in by_ds_val.items():
            by_ds_train[ds_name].extend(masks)

        for ds_name, masks in by_ds_train.items():
            masks = _dedup_keep_order(masks)
            if not masks:
                continue
            ds_dir = os.path.join(args.out_dir, ds_name)
            results = {p: (True, {"from": "pickled_loader"}) for p in masks}
            merge_qc_cache_results(
                ds_dir=ds_dir,
                split="train",
                quality_params=params,
                new_results=results,
            )
            print(f"[train] wrote {len(results):5d} entries for dataset '{ds_name}'")

    # -------- TEST → "test" cache --------
    if not args.skip_test and test_loader is not None:
        by_ds_test = _collect_3d_masks_from_dataset(test_loader.dataset)
        for ds_name, masks in by_ds_test.items():
            masks = _dedup_keep_order(masks)
            if not masks:
                continue
            ds_dir = os.path.join(args.out_dir, ds_name)
            results = {p: (True, {"from": "pickled_loader"}) for p in masks}
            merge_qc_cache_results(
                ds_dir=ds_dir,
                split="test",
                quality_params=params,
                new_results=results,
            )
            print(f"[test ] wrote {len(results):5d} entries for dataset '{ds_name}'")

    print("Done.")

if __name__ == "__main__":
    main()
