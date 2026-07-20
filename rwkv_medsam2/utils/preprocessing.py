# Dataset preprocessing, pairing, caching, and mask QC utilities.
#
# Builds train/validation/test sequence metadata, applies foreground quality
# checks, and caches formatted RWKV-MedSAM2 datasets.
import os
import math
import base64
import zlib
import warnings
import sys
import random
import torch
import numpy as np
import SimpleITK as sitk

from math import ceil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_MASK_QC_CACHE = {}

import json
import re
from typing import Dict, Tuple, Optional

import time
import hashlib
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List
from rwkv_medsam2.dataset import SegmentationSequenceDataset, SequenceTransform  # noqa: E402

DATASET_CACHE_VERSION = "v1.08"


def _pack_u32_zlib_b64(arr_u32: np.ndarray) -> str:
    """
    Compress a uint32 numpy array into a base64(zlib(bytes)) string.
    Keeps cache files small while allowing fast decode.

    Args:
        arr_u32 (np.ndarray): Unsigned integer array to encode.

    Returns:
        str: Base64-encoded compressed byte string.
    """
    arr_u32 = np.asarray(arr_u32, dtype=np.uint32, order="C")
    raw = arr_u32.tobytes()
    comp = zlib.compress(raw, level=6)
    return base64.b64encode(comp).decode("ascii")

def _unpack_u32_zlib_b64(s: str) -> np.ndarray:
    """
    Decode base64(zlib(bytes)) back into a uint32 numpy array.

    Args:
        s (str): Encoded string produced by ``_pack_u32_zlib_b64``.

    Returns:
        np.ndarray: Decoded uint32 array.
    """
    comp = base64.b64decode(s.encode("ascii"))
    raw = zlib.decompress(comp)
    return np.frombuffer(raw, dtype=np.uint32)

def _stats_cache_filename(ds_dir: str, split: str, downsample: int) -> str:
    """
    Build the JSONL mask-stat cache filename.

    Args:
        ds_dir (str): Dataset cache directory.
        split (str): Data split name.
        downsample (int): Downsample factor.

    Returns:
        str: Cache file path.
    """
    ds = int(max(1, downsample))
    return os.path.join(ds_dir, f"maskstats_{split}_ds{ds}.jsonl")

def load_maskstats_cache(
    ds_dir: str,
    split: str,
    downsample: int,
    wanted_paths: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Stream-load a compact per-mask stats cache (JSONL).

    If wanted_paths is provided, only returns records for those paths.
    Records whose file_key no longer matches are ignored.

    Args:
        ds_dir (str): Dataset cache directory.
        split (str): Data split name.
        downsample (int): Downsample factor.
        wanted_paths (set | None): Optional paths to load.

    Returns:
        dict[str, dict[str, Any]]: Fresh cached stats keyed by mask path.
    """
    path = _stats_cache_filename(ds_dir, split, downsample)
    if not os.path.isfile(path):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                p = rec.get("path", None)
                if not p:
                    continue
                if wanted_paths is not None and p not in wanted_paths:
                    continue

                cached_key = tuple(rec.get("file_key", (p, 0, 0)))
                current_key = _mask_qc_cache_key(p)
                if cached_key != current_key:
                    continue

                stats = rec.get("stats", None)
                if isinstance(stats, dict):
                    out[p] = stats

        return out
    except Exception:
        return {}

def append_maskstats_cache(
    ds_dir: str,
    split: str,
    downsample: int,
    records: Dict[str, Dict[str, Any]],
) -> None:
    """
    Append per-mask stats records to the JSONL cache.

    records maps: mask_path -> stats_dict

    Args:
        ds_dir (str): Dataset cache directory.
        split (str): Data split name.
        downsample (int): Downsample factor.
        records (dict[str, dict[str, Any]]): Stats records keyed by mask path.

    Returns:
        None.
    """
    if not records:
        return

    path = _stats_cache_filename(ds_dir, split, downsample)
    os.makedirs(ds_dir, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for p, stats in records.items():
            payload = {
                "path": p,
                "file_key": _mask_qc_cache_key(p),
                "stats": stats,
            }
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

def _normalize_modality(mod):
    """
    Normalize a modality label to lowercase text.

    Args:
        mod (Any): Raw modality value.

    Returns:
        str | None: Normalized modality label, or None.
    """
    if mod is None:
        return None
    return str(mod).strip().lower()

def _dripp_policy_satisfies_quality(policy, quality_params):
    """Return whether DRIPP already enforced the active RWKV 3D QC contract."""
    if not isinstance(policy, dict) or policy.get("applied") is not True:
        return False

    integer_keys = (
        "min_component_voxels_3d",
        "mask_qc_downsample_3d",
        "min_slice_area_px_3d",
        "min_qualified_slices_3d",
    )
    if any(
        isinstance(policy.get(key), bool)
        or not isinstance(policy.get(key), int)
        for key in integer_keys
    ):
        return False
    fraction = policy.get("min_slice_area_fraction_3d")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return False
    if policy.get("require_contiguous_qualified_slices_3d") is not True:
        return False

    return (
        policy["min_component_voxels_3d"]
        >= int(quality_params.get("min_voxels", 64))
        and policy["mask_qc_downsample_3d"]
        == int(quality_params.get("downsample", 2))
        and policy["min_slice_area_px_3d"]
        >= int(quality_params.get("min_slice_area_px", 32))
        and float(fraction)
        >= float(quality_params.get("min_frac_2d_area") or 0.0)
        and policy["min_qualified_slices_3d"]
        >= int(quality_params.get("min_slices_any_axis", 2))
    )

_MASK_FG2D_CACHE = {}

def _mask_has_fg_2d(mask_path: str) -> bool:
    """
    Fast check: does this 2D mask contain any foreground pixels (>0)?
    Uses a small cache keyed by (path, mtime, size) so we don't reread
    the same PNGs over and over.

    Args:
        mask_path (str): Path to a 2D mask image.

    Returns:
        bool: True when the mask contains foreground.
    """
    key = _mask_qc_cache_key(mask_path)  # (path, mtime, size)

    if key in _MASK_FG2D_CACHE:
        return _MASK_FG2D_CACHE[key]

    try:
        itk = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(itk)
        has_fg = bool((arr > 0).any())
    except Exception:
        # If we can't read it, treat as no-FG and let it be dropped
        has_fg = False

    _MASK_FG2D_CACHE[key] = has_fg
    return has_fg

def get_pairings(out_dir, datasets, split="train", tasks_map=None, quality_params=None):
    """
    Loads the groupings.json files from each folder in the given output directory
    for the specified split.

    Each groupings.json file is expected to contain a list of entries, where each
    entry is a dictionary with the following keys:
    - "proc_images": A list of paths to the preprocessed images.
    - "proc_masks": A list of paths to the preprocessed masks.

    The function pairs each mask with its corresponding image by searching for
    the index in the image path. The index is assumed to be in the format
    "(?:img|frame|slice)(\\d+)" and is used to match the mask with its
    corresponding image.

    If 'tasks_map' is provided, the per-pair 'tasks' list is refined so that it
    only includes tasks whose 'classes' entry contains that pair's mask class.

    Args:
        out_dir (str): The output directory containing the dataset folders.
        datasets (list): The list of dataset names to load groupings for.
        split (str, optional): The split to load groupings for. Default is "train".
        tasks_map (dict, optional): Global task map (e.g. from datasets_tasks.json)
            used to filter each pair's task list based on its class name.
        quality_params (dict, optional): QC configuration. If None, defaults are used.

    Returns:
        list: A list of dicts, one per (image, mask) pairing, each containing
              dataset, subdataset, tasks, class, pair, dim, and modality.
    """
    # Optional: refine tasks per-pair based on its class and the global tasks map
    def _filter_tasks_for_class(entry_tasks, cls_name):
        """Filter task ids to those compatible with a class."""
        if not tasks_map or not entry_tasks or cls_name is None:
            return entry_tasks

        filtered = []
        for t in entry_tasks:
            info = tasks_map.get(t, {})
            cls_list = info.get("classes", [])
            if cls_name in cls_list:
                filtered.append(t)

        # If no task explicitly lists this class, fall back to original tasks
        return filtered or entry_tasks

    # Find all image and mask pairs for each dataset and collect them
    _idx_pattern      = re.compile(r"_(?:img|frame|slice)(\d+)")
    _mask_idx_pattern = re.compile(r"_mask(\d+)")  # Prefer this for WSI tiles like PAIP2019
    all_pairs         = []
    total_kept_global = 0  # Global count of kept pairs

    # Quality control config
    use_gpu_qc = True
    qc_device  = "cuda" if torch.cuda.is_available() else "cpu"
    qc_batch   = 16

    # Allow caller to override QC knobs; fall back to defaults if not provided.
    if quality_params is None:
        quality_params = {
            "min_voxels": 64,
            # Require at least seq_len contiguous FG slices on highest-res axis
            "min_slices_any_axis": 8,
            "min_slice_area_px": 32,
            "downsample": 2,
            "min_frac_2d_area": 0.00153,
        }

    QC_CACHE_ROOT = os.environ.get("QC_CACHE_ROOT", "/data/DatasetIndexes/QC_Cache/")

    for ds in datasets:
        # Check if dataset grouping json exists
        ds_dir  = os.path.join(out_dir, ds)
        grp_file = os.path.join(ds_dir, f'{ds}_groups.json')
        if not os.path.isfile(grp_file):
            print(f"Could not find {ds}_groups.json in {ds_dir}")
            continue

        # Parse the groups file
        with open(grp_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        entries = []
        for sub in data.get("subdatasets", []):
            sub_name     = sub.get("name", "default")
            sub_modality = _normalize_modality(sub.get("modality", None))  # fallback if entry doesn't have it
            sub_tasks    = sub.get("tasks", [])
            sub_classes  = sub.get("classes", [])

            for entry in sub.get(split, []):
                if entry.get("preprocessing", {}).get("status") in {"rejected", "failed"}:
                    continue

                entry["subdataset_name"] = sub_name
                entry["tasks"]           = sub_tasks
                entry["mask_classes"]    = sub_classes

                # Prefer per-entry "subdataset_modality", fall back to sub-level "modality"
                entry["subdataset_modality"] = _normalize_modality(
                    entry.get("subdataset_modality", sub_modality)
                )

                entries.append(entry)
        print(f"Found {len(entries)} '{split}' entries in {grp_file}")

        # Initialize progress bar
        total_candidates = sum(len(e.get("proc_masks", [])) for e in entries)
        pbar = tqdm(total=total_candidates, desc=f"[{ds}] masks→pairs", unit="mask", leave=False, file=sys.stdout)

        # Iterate through the groupings and create image/mask pairs
        ds_kept    = 0  # Per-dataset kept count
        ds_skipped = 0  # Per-dataset skipped count

        for entry in entries:
            # Tolerate missing 'split' field (default to current split)
            if entry.get('split', split) != split:
                continue

            entry_modality = _normalize_modality(entry.get("subdataset_modality", "unknown"))

            imgs = entry.get('proc_images', [])
            msks = entry.get('proc_masks', [])

            # Create a map from index to image
            idx_map = {}
            for img_path in imgs:
                base = os.path.basename(img_path)
                m = _idx_pattern.search(base)
                if m:
                    idx_map[int(m.group(1))] = img_path

            # Build a fallback image-key map by normalized basename
            img_by_key = {}
            for ip in imgs:
                nm   = os.path.basename(ip).lower()
                stem = os.path.splitext(nm)[0]
                stem = re.sub(r'(?:_mask|-mask|\.mask|_gt|_label|-label|_anno|_annotation)', '', stem)
                stem = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
                img_by_key[stem] = ip

            # Pair each mask with its corresponding image and add to the list of pairs
            pairs = []
            for mask in msks:
                pbar.update(1)  # Update progress bar
                msk_path = mask['path']
                cls_name = mask['class']
                if cls_name is None:
                    continue

                if "image_index" in mask:
                    image_index = mask.get("image_index")
                    dimension = mask.get("dimension")
                    if (isinstance(image_index, bool)
                            or not isinstance(image_index, int)
                            or not 0 <= image_index < len(imgs)):
                        warnings.warn(
                            f"Skipping malformed explicit pairing for {msk_path}: "
                            f"image_index={image_index!r} does not reference proc_images",
                            RuntimeWarning,
                        )
                        continue
                    if dimension is not None and dimension not in (2, 3):
                        warnings.warn(
                            f"Skipping malformed explicit pairing for {msk_path}: "
                            f"dimension={dimension!r} is not 2 or 3",
                            RuntimeWarning,
                        )
                        continue
                    pairs.append({
                        "image": imgs[image_index], "mask": msk_path,
                        "class": cls_name, "_dimension": dimension,
                    })
                    continue

                base = os.path.basename(msk_path)

                # Prefer _mask#### first (PAIP2019 writes ..._img000_mask2272_...)
                m = _mask_idx_pattern.search(base)
                if m:
                    idx = int(m.group(1))
                    ip  = idx_map.get(idx)
                    if ip:
                        pairs.append({'image': ip, 'mask': msk_path, 'class': cls_name})
                        continue

                # Fallback: use _(img|frame|slice)####
                m = _idx_pattern.search(base)
                if m:
                    idx = int(m.group(1))
                    ip  = idx_map.get(idx)
                    if ip:
                        pairs.append({'image': ip, 'mask': msk_path, 'class': cls_name})
                        continue

                # Last resort: normalized key match
                nm   = base.lower()
                stem = os.path.splitext(nm)[0]
                stem = re.sub(r'(?:_mask|-mask|\.mask|_gt|_label|-label|_anno|_annotation)', '', stem)
                stem = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
                ip   = img_by_key.get(stem)
                if ip:
                    pairs.append({'image': ip, 'mask': msk_path, 'class': cls_name})

            entry_policy_3d = (
                entry.get("preprocessing", {}).get("quality_policy_3d") or {}
            )
            for pair in pairs:
                pair["_quality_policy_3d"] = entry_policy_3d

            # If no pairs found, skip
            if not pairs:
                continue

            # Sort the pairs to ensure temporal order
            def _sort_idx(pth):  # Inline sorter tolerant to either token
                """
                Extract a frame index from an image or mask path.

                Args:
                    pth (str): Image or mask path.

                Returns:
                    int: Parsed index, or 0 when absent.
                """
                b  = os.path.basename(pth).lower()
                mm = _mask_idx_pattern.search(b) or _idx_pattern.search(b)
                return int(mm.group(1)) if mm else 0

            pairs.sort(key=lambda x: _sort_idx(x['image']))

            # Build dims once
            dims = [
                p.pop("_dimension", None)
                or (2 if os.path.splitext(p["image"])[1].lower() == ".png" else 3)
                for p in pairs
            ]

            # Drop 2D pairs whose masks have *no* foreground
            keep_idx = []
            for i, pair in enumerate(pairs):
                if dims[i] == 2:
                    # Skip 2D masks with zero FG
                    if not _mask_has_fg_2d(pair['mask']):
                        ds_skipped += 1
                        continue
                keep_idx.append(i)

            if not keep_idx:
                # Everything in this entry was empty / bad – nothing to keep
                continue

            pairs = [pairs[i] for i in keep_idx]
            dims  = [dims[i]  for i in keep_idx]

            # Trust sufficient DRIPP policies; retain disk QC for legacy/weaker data.
            idx_3d = [i for i, d in enumerate(dims) if d == 3]
            qc_ok = {}
            fallback_idx_3d = []
            for pair_index in idx_3d:
                policy = pairs[pair_index].get("_quality_policy_3d")
                if _dripp_policy_satisfies_quality(policy, quality_params):
                    qc_ok[pair_index] = (
                        True, {"reason": "quality_enforced_by_dripp"}
                    )
                else:
                    fallback_idx_3d.append(pair_index)

            if fallback_idx_3d:
                mask_paths_3d = [pairs[i]["mask"] for i in fallback_idx_3d]

                # Centralized cache dir: /data/DatasetIndexes/QC_Cache/<dataset> (or QC_CACHE_ROOT override)
                ds_dir_default   = os.path.join(out_dir, ds)
                ds_dir_cache     = os.path.join(QC_CACHE_ROOT, ds)
                cache_dir_to_use = ds_dir_cache if os.path.isdir(ds_dir_cache) else ds_dir_default

                ds_factor = int(quality_params.get("downsample", 2))
                wanted = set(mask_paths_3d)

                stats_map = load_maskstats_cache(
                    ds_dir=cache_dir_to_use,
                    split=split,
                    downsample=ds_factor,
                    wanted_paths=wanted,
                )

                missing = [p for p in mask_paths_3d if p not in stats_map]
                if missing:
                    new_records = {}
                    for p in missing:
                        ok_s, stats = compute_mask_stats_3d(
                            p, downsample=ds_factor
                        )
                        if ok_s:
                            new_records[p] = stats
                            stats_map[p] = stats
                    append_maskstats_cache(
                        ds_dir=cache_dir_to_use,
                        split=split,
                        downsample=ds_factor,
                        records=new_records,
                    )

                for pair_index, mask_path in zip(
                    fallback_idx_3d, mask_paths_3d
                ):
                    stats = stats_map.get(mask_path)
                    if stats is None:
                        qc_ok[pair_index] = (
                            False, {"reason": "stats_missing"}
                        )
                        continue
                    qc_ok[pair_index] = evaluate_quality_from_stats(
                        stats, quality_params
                    )

            # Now build outputs while consulting qc_ok for 3D pairs
            if len(pairs) > 1:
                for i, pair in enumerate(pairs):
                    dim = dims[i]
                    if dim == 3:
                        ok, info = qc_ok.get(i, (False, {"reason": "qc_missing"}))
                        if not ok:
                            ds_skipped += 1
                            continue

                    cls_name       = pair['class']
                    tasks_for_pair = _filter_tasks_for_class(entry['tasks'], cls_name)

                    all_pairs.append({
                        'dataset':    ds,
                        'subdataset': entry['subdataset_name'],
                        'tasks':      tasks_for_pair,
                        'class':      cls_name,
                        'pair':       (pair['image'], pair['mask']),
                        'dim':        dim,
                        'modality':   entry_modality,
                    })
                    ds_kept += 1
                    total_kept_global += 1
            else:
                dim = dims[0]
                if dim == 3:
                    ok, info = qc_ok.get(0, (False, {"reason": "qc_missing"}))
                    if not ok:
                        ds_skipped += 1
                    else:
                        cls_name       = pairs[0]['class']
                        tasks_for_pair = _filter_tasks_for_class(entry['tasks'], cls_name)

                        all_pairs.append({
                            'dataset':      ds,
                            'subdataset':   entry['subdataset_name'],
                            'tasks':        tasks_for_pair,
                            'class':        cls_name,
                            'mask_classes': cls_name,  # keep for backward compatibility
                            'pair':         (pairs[0]['image'], pairs[0]['mask']),
                            'dim':          dim,
                            'modality':     entry_modality,
                        })
                        ds_kept += 1
                        total_kept_global += 1
                else:
                    cls_name       = pairs[0]['class']
                    tasks_for_pair = _filter_tasks_for_class(entry['tasks'], cls_name)

                    all_pairs.append({
                        'dataset':      ds,
                        'subdataset':   entry['subdataset_name'],
                        'tasks':        tasks_for_pair,
                        'class':        cls_name,
                        'mask_classes': cls_name,  # keep for backward compatibility
                        'pair':         (pairs[0]['image'], pairs[0]['mask']),
                        'dim':          dim,
                        'modality':     entry_modality,
                    })
                    ds_kept += 1
                    total_kept_global += 1

        # Close the progress bar and print a summary
        pbar.close()
        tqdm.write(f"[{ds}] kept {ds_kept}, skipped {ds_skipped}, total processed {ds_kept + ds_skipped} for '{split}'")

    print(f"Total kept pairs for '{split}': {total_kept_global}")  # Global summary
    return all_pairs

def get_sequences(
    out_dir,
    split="train",
    val_frac=0.1,
    seed=42,
    max_frames_per_sequence=8,
    tasks_map=None,
    quality_params=None,
):
    """
    Load and assemble 2D and 3D data into "sequences" for unified video-style training.

    A "sequence" is a list of frames, where each frame is an (image_path, mask_path)
    tuple. For 2D tasks, each sequence contains a single (image, mask) pair. For 3D
    volumes, each sequence represents one full volume as a single-element list; the
    dataset loader is responsible for turning that volume into an N-slice clip when
    truncation is enabled.

    Processing steps:
      1. Load raw (image, mask) pairs via 'get_pairings()', each dict tagged with:
         dataset, subdataset, tasks, class, pair, dim.
      2. If 'tasks_map' is provided, 'get_pairings()' uses it to refine the per-pair
         task list based on the mask class.
      3. Group all entries by (dataset, subdataset, tasks, modality).
      4. For 2D, each pair becomes its own single-frame sequence.
      5. Treat each 3D entry as a single-element sequence.
      6. If 'split=="train"', shuffle and split within each group into train/validation
         subsets using 'val_frac'.
      7. Return:
         - For 'split=="train"': a tuple '(train_seqs, val_seqs)'.
         - For 'split=="test"': a list 'test_seqs'.

    Args:
        out_dir (str): Path to the directory containing dataset subfolders.
        split (str): One of '"train"' or '"test"'.
        val_frac (float): Fraction reserved for validation within each group
                          (only used when 'split=="train"').
        seed (int): Random seed for reproducibility.
        max_frames_per_sequence (int): Maximum number of frames per training sample.
            This is used by the dataset loader (e.g., 3D clip length) and retained
            for API compatibility.
        tasks_map (dict, optional): Global task map (e.g. from datasets_tasks.json)
            forwarded to 'get_pairings()' to refine per-pair task lists.
        quality_params (dict, optional): QC configuration forwarded to 'get_pairings()'.

    Returns:
        - If 'split=="train"': '(train_seqs, val_seqs)', each a list of sequence dicts.
        - If 'split=="test"': 'test_seqs', a list of sequence dicts.

    Each sequence dict contains:
        - 'dataset' (str)
        - 'subdataset' (str)
        - 'tasks' (List[str])
        - 'sequence' (List[Tuple[str, str]]): ordered image/mask path tuples
        - 'dim' (int): 2 or 3
        - 'modality' (str)
    """
    # 0a) Find dataset subfolders
    try:
        entries = sorted(os.listdir(out_dir))
    except Exception:
        raise RuntimeError(f"Could not find output directory {out_dir}")
    datasets = [d for d in entries if os.path.isdir(os.path.join(out_dir, d))]

    # 1) Load all (image, mask) pairs
    pairs = get_pairings(
        out_dir, datasets, split, tasks_map=tasks_map, quality_params=quality_params
    )
    pairs2D = [p for p in pairs if p["dim"] == 2]
    pairs3D = [p for p in pairs if p["dim"] == 3]

    # 2) Group by (dataset, subdataset, tasks, modality)
    grouped2D = defaultdict(list)
    for p in pairs2D:
        key = (p["dataset"], p["subdataset"], tuple(p["tasks"]), p.get("modality", "unknown"))
        grouped2D[key].append(p["pair"])  # (img_path, mask_path)

    grouped3D = defaultdict(list)
    for p in pairs3D:
        key = (p["dataset"], p["subdataset"], tuple(p["tasks"]), p.get("modality", "unknown"))
        grouped3D[key].append(p["pair"])  # (vol_img_path, vol_mask_path)

    rng = random.Random(seed)

    # 3) Build train/val or test lists
    if split == "train":
        train_seqs, val_seqs = [], []

        # 3a) 2D: split at pair-level; each pair becomes a single-frame sequence
        for (ds, sub, tasks, modality), frames in grouped2D.items():
            frames = frames[:]  # copy
            rng.shuffle(frames)

            n_val = int(len(frames) * val_frac)
            val_f = frames[:n_val]
            train_f = frames[n_val:]

            for pair in train_f:
                train_seqs.append(
                    {
                        "dataset": ds,
                        "subdataset": sub,
                        "tasks": list(tasks),
                        "sequence": [pair],  # one frame
                        "dim": 2,
                        "modality": modality,
                    }
                )
            for pair in val_f:
                val_seqs.append(
                    {
                        "dataset": ds,
                        "subdataset": sub,
                        "tasks": list(tasks),
                        "sequence": [pair],  # one frame
                        "dim": 2,
                        "modality": modality,
                    }
                )

        # 3b) 3D: split volumes per sequence; each volume stays as a single-element sequence
        for (ds, sub, tasks, modality), vols in grouped3D.items():
            vols = vols[:]  # copy
            rng.shuffle(vols)

            n_val = int(len(vols) * val_frac)
            val_v = vols[:n_val]
            train_v = vols[n_val:]

            for v in train_v:
                train_seqs.append(
                    {
                        "dataset": ds,
                        "subdataset": sub,
                        "tasks": list(tasks),
                        "sequence": [v],
                        "dim": 3,
                        "modality": modality,
                    }
                )
            for v in val_v:
                val_seqs.append(
                    {
                        "dataset": ds,
                        "subdataset": sub,
                        "tasks": list(tasks),
                        "sequence": [v],
                        "dim": 3,
                        "modality": modality,
                    }
                )

        return train_seqs, val_seqs

    # Test split
    test_seqs = []

    # 2D: each pair becomes a single-frame sequence
    for (ds, sub, tasks, modality), frames in grouped2D.items():
        for pair in frames:
            test_seqs.append(
                {
                    "dataset": ds,
                    "subdataset": sub,
                    "tasks": list(tasks),
                    "sequence": [pair],  # one frame
                    "dim": 2,
                    "modality": modality,
                }
            )

    # 3D: each volume stays as a single-element sequence
    for (ds, sub, tasks, modality), vols in grouped3D.items():
        for v in vols:
            test_seqs.append(
                {
                    "dataset": ds,
                    "subdataset": sub,
                    "tasks": list(tasks),
                    "sequence": [v],
                    "dim": 3,
                    "modality": modality,
                }
            )

    return test_seqs

@dataclass
class DatasetSignature:
    """A minimal signature to verify cached datasets are compatible with current run settings."""
    version: str                       # bump when dataset class/transform changes
    out_dir: str
    split: str                         # "train" or "test"
    seq_len: int
    min_fg_frames_in_window: int
    truncate_val_test: bool
    val_frac: float
    seed: int
    aug_probs: Tuple[float, float, float]   # (base_prob, lr_prob, flip_prob)
    quality_params: Dict[str, Any]          # QC knobs used during pairing
    tasks_file_fingerprint: Optional[str]   # if you want to include DRIPP tasks.json fingerprint
    fg_min_pixels_frac: float
    dripp_manifests_fingerprint: Optional[str]

    def to_hash(self) -> str:
        """
        Hash the dataset signature into a stable cache key.

        Args:
            None.

        Returns:
            str: MD5 hash for the signature.
        """
        s = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(s).hexdigest()

def _fingerprint_file(path: Optional[str]) -> Optional[str]:
    """
    Compute the MD5 hash of a file at the given path.

    Args:
        path (Optional[str]): The path to the file to hash.

    Returns:
        Optional[str]: The MD5 hash of the file, or None if the file does not exist.
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_dripp_manifests(out_dir: str) -> Optional[str]:
    """Fingerprint manifest metadata without reading large JSON documents."""
    if not os.path.isdir(out_dir):
        return None
    records = []
    for dataset in sorted(os.listdir(out_dir)):
        path = os.path.abspath(os.path.join(out_dir, dataset, f"{dataset}_groups.json"))
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        records.append((path, stat.st_size, stat.st_mtime_ns))
    if not records:
        return None
    payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _cache_paths(cache_root: str, sig: DatasetSignature) -> Dict[str, str]:
    """
    Build cache paths for a formatted dataset signature.

    Args:
        cache_root (str): Root cache directory.
        sig (DatasetSignature): Dataset cache signature.

    Returns:
        dict[str, str]: Paths for root, metadata, and split dataset files.
    """
    h = sig.to_hash()
    root = os.path.join(cache_root, "formatted_datasets", h)
    os.makedirs(root, exist_ok=True)
    return {
        "root": root,
        "meta": os.path.join(root, "meta.json"),
        "train": os.path.join(root, "train_ds.pt"),
        "val": os.path.join(root, "val_ds.pt"),
        "test": os.path.join(root, "test_ds.pt"),
    }

def _save_meta(meta_path: str, sig: DatasetSignature, extra: Dict[str, Any]) -> None:
    """
    Save formatted dataset cache metadata.

    Args:
        meta_path (str): Metadata file path.
        sig (DatasetSignature): Dataset cache signature.
        extra (dict): Additional metadata to include.

    Returns:
        None.
    """
    payload = {"signature": asdict(sig), "extra": extra, "saved_at": time.time()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _load_cached_datasets(
    cache_root: str,
    sig: DatasetSignature,
    need_train_val: bool
) -> Optional[Tuple[Optional[SegmentationSequenceDataset],
                    Optional[SegmentationSequenceDataset],
                    Optional[SegmentationSequenceDataset]]]:
    """
    Load cached formatted datasets when all split files are present.

    Args:
        cache_root (str): Root cache directory.
        sig (DatasetSignature): Dataset cache signature.
        need_train_val (bool): Whether train and validation splits are required.

    Returns:
        tuple | None: Cached train, validation, and test datasets, or None on miss.
    """
    paths = _cache_paths(cache_root, sig)
    if not os.path.isfile(paths["meta"]):
        print("paths[meta] does not exist: ", paths["meta"])
        return None

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        sig_on_disk = DatasetSignature(**meta["signature"])
        if sig_on_disk.to_hash() != sig.to_hash():
            print("Cached datasets are incompatible with current run settings.")
            print("Proceeding to recompute datasets...")
            return None  # settings changed

        # IMPORTANT: weights_only=False because these are pickled class instances, not just tensors
        load = lambda p: torch.load(p, map_location="cpu", weights_only=False)

        train_ds = load(paths["train"]) if (need_train_val and os.path.isfile(paths["train"])) else None
        val_ds   = load(paths["val"])   if (need_train_val and os.path.isfile(paths["val"]))   else None
        test_ds  = load(paths["test"])  if os.path.isfile(paths["test"]) else None

        if need_train_val and (train_ds is None or val_ds is None):
            return None
        return (train_ds, val_ds, test_ds)
    except Exception:
        print("Failed to load cached datasets.")
        return None

def _progress_scan(out_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    Wrapper that calls the existing pairing routine but renders TWO progress bars:
      1) datasets
      2) pairings within the current dataset
    Returns the full list of pair dicts like get_pairings().

    Args:
        out_dir (str): DRIPP output directory.
        split (str): Split to scan.

    Returns:
        list[dict[str, Any]]: Pairing dictionaries from ``get_pairings``.
    """
    # Discover dataset folders once, then call get_pairings per dataset.
    entries = sorted(os.listdir(out_dir))
    datasets = [d for d in entries if os.path.isdir(os.path.join(out_dir, d))]

    all_pairs = []
    for ds in datasets:
        # get_pairings already supports a list of datasets; we pass [ds] and rely
        # on its internal per-mask tqdm. We temporarily suppress its prints by
        # letting tqdm handle the lines.
        pairs = get_pairings(out_dir, [ds], split=split)
        all_pairs.extend(pairs)
    return all_pairs

def build_and_cache_formatted_splits(
    *,
    config,
    cache_root: str,
) -> Tuple[SegmentationSequenceDataset, SegmentationSequenceDataset, SegmentationSequenceDataset]:
    """
    One-time materialization:
      - builds sequences (train/val from 'train', test from 'test'),
      - constructs SegmentationSequenceDataset objects with the SAME knobs,
      - saves them (.pt),
      - returns the three dataset objects.
    Subsequent runs can load from cache via `load_datasets(...)`.

    Args:
        config (OmegaConf.DictConfig): Dataset and training configuration.
        cache_root (str): Root directory for formatted dataset cache.

    Returns:
        tuple[SegmentationSequenceDataset, SegmentationSequenceDataset, SegmentationSequenceDataset]:
            Train, validation, and test datasets.
    """
    out_dir  = str(config.dripp.output_dir)
    seq_len  = int(getattr(config.sampler, "seq_len", 8))
    min_fg   = int(getattr(config.sampler, "min_fg_frames_in_window", 2))
    val_frac = float(getattr(config.training, "val_frac", 0.1))
    seed     = int(getattr(config.training, "seed", 42))
    # Train-time augs as in your current script
    base_prob = 0.15
    lr_prob   = 0.25
    flip_prob = 0.50

    # IMPORTANT: enforce at least `seq_len` foreground slices in one axis for 3D masks
    quality_params = {
        "min_voxels": 64,
        # Require at least seq_len contiguous FG slices on highest-res axis
        "min_slices_any_axis": seq_len,
        "min_slice_area_px": 32,
        "downsample": 2,
        "min_frac_2d_area": 0.00153,
    }
    tasks_fp = _fingerprint_file(getattr(getattr(config, "dripp", object()), "tasks_file", None))
    fg_min_frac = float(getattr(getattr(config, "sampler", {}), "fg_min_pixels_frac", 0.0002))
    manifests_fp = _fingerprint_dripp_manifests(out_dir)

    sig = DatasetSignature(
        version=DATASET_CACHE_VERSION,
        out_dir=out_dir,
        split="both",
        seq_len=seq_len,
        min_fg_frames_in_window=min_fg,
        truncate_val_test=False,
        val_frac=val_frac,
        seed=seed,
        aug_probs=(base_prob, lr_prob, flip_prob),
        quality_params=quality_params,
        fg_min_pixels_frac=fg_min_frac,
        dripp_manifests_fingerprint=manifests_fp,
        tasks_file_fingerprint=tasks_fp,
    )

    # Load DRIPP tasks map so we can refine tasks per mask class when building sequences
    tasks_map_path = getattr(getattr(config, "dripp", object()), "tasks_file", None)
    if tasks_map_path is not None:
        with open(tasks_map_path, "r") as f:
            tasks_map = json.load(f)
    else:
        tasks_map = None

    # Try cached first
    cached = _load_cached_datasets(cache_root, sig, need_train_val=True)
    if cached is not None and all(cached):
        return cached  # (train, val, test)

    # -------------------------
    # Build train/val sequences
    # -------------------------
    train_seqs, val_seqs = get_sequences(
        out_dir,
        split="train",
        val_frac=val_frac,
        seed=seed,
        max_frames_per_sequence=seq_len,
        tasks_map=tasks_map,
        quality_params=quality_params,
    )

    train_ds = SegmentationSequenceDataset(
        train_seqs,
        transform=SequenceTransform(base_prob=base_prob, lr_prob=lr_prob, flip_prob=flip_prob),
        max_frames_per_sequence=seq_len,
        min_fg_frames_in_window=min_fg,
        fg_min_pixels_frac=fg_min_frac,
        truncate=True,
    )
    val_ds = SegmentationSequenceDataset(
        val_seqs,
        transform=SequenceTransform(base_prob=0, lr_prob=0, flip_prob=0),
        max_frames_per_sequence=seq_len,
        min_fg_frames_in_window=min_fg,
        fg_min_pixels_frac=fg_min_frac,
        truncate=False,
    )

    # --------------
    # Build test set
    # --------------
    test_seqs = get_sequences(
        out_dir,
        split="test",
        max_frames_per_sequence=seq_len,
        tasks_map=tasks_map,
        quality_params=quality_params,
    )

    test_ds = SegmentationSequenceDataset(
        test_seqs,
        transform=SequenceTransform(base_prob=0, lr_prob=0, flip_prob=0),
        max_frames_per_sequence=seq_len,
        min_fg_frames_in_window=min_fg,
        fg_min_pixels_frac=fg_min_frac,
        truncate=False,
    )

    # Save
    paths = _cache_paths(cache_root, sig)
    torch.save(train_ds, paths["train"])
    torch.save(val_ds, paths["val"])
    torch.save(test_ds, paths["test"])
    _save_meta(paths["meta"], sig, extra={"note": "SegmentationSequenceDataset cached"})

    return train_ds, val_ds, test_ds

def load_datasets(*, config, cache_root):
    """
    Loads three datasets from cache or builds them from scratch if they don't exist.

    Args:
        config (OmegaConf): Configuration object containing experiment parameters.
        cache_root (str): Path to the cache directory.

    Returns:
        Tuple[SegmentationSequenceDataset, SegmentationSequenceDataset, SegmentationSequenceDataset]:
            Train, validation, and test datasets.
    """
    out_dir  = str(config.dripp.output_dir)
    seq_len  = int(getattr(config.sampler, "seq_len", 8))
    min_fg   = int(getattr(config.sampler, "min_fg_frames_in_window", 2))
    val_frac = float(getattr(config.training, "val_frac", 0.1))
    seed     = int(getattr(config.training, "seed", 42))
    base_prob, lr_prob, flip_prob = 0.15, 0.25, 0.50

    # Must match build_and_cache_formatted_splits so the signature/caching is consistent
    quality_params = {
        "min_voxels": 64,
        "min_slices_any_axis": seq_len,
        "min_slice_area_px": 32,
        "downsample": 2,
        "min_frac_2d_area": 0.00153,
    }
    tasks_fp = _fingerprint_file(getattr(getattr(config, "dripp", object()), "tasks_file", None))
    fg_min_frac = float(getattr(getattr(config, "sampler", {}), "fg_min_pixels_frac", 0.0002))
    manifests_fp = _fingerprint_dripp_manifests(out_dir)

    # Build signature
    # Bump the version when sequence formatting changes so cached datasets are not reused incorrectly.
    sig = DatasetSignature(
        version=DATASET_CACHE_VERSION,
        out_dir=out_dir,
        split="both",
        seq_len=seq_len,
        min_fg_frames_in_window=min_fg,
        truncate_val_test=False,
        val_frac=val_frac,
        seed=seed,
        aug_probs=(base_prob, lr_prob, flip_prob),
        quality_params=quality_params,
        tasks_file_fingerprint=tasks_fp,
        fg_min_pixels_frac=fg_min_frac,
        dripp_manifests_fingerprint=manifests_fp,
    )

    # Try cached first
    cached = _load_cached_datasets(cache_root, sig, need_train_val=True)
    if cached is not None and all(cached):
        train_ds, val_ds, test_ds = cached
    else:
        print("No cached datasets found. Building new datasets...")
        train_ds, val_ds, test_ds = build_and_cache_formatted_splits(config=config, cache_root=cache_root)

    # Rebuild datasets to keep constructor options aligned with the active configuration.
    def _rebuild_dataset(ds):
        """Recreate a dataset with active constructor settings."""
        return SegmentationSequenceDataset(
            ds.sequences,
            transform=ds.transform,
            max_frames_per_sequence=ds.max_frames_per_sequence,
            min_fg_frames_in_window=ds.min_fg_frames_in_window,
            fg_min_pixels_frac=fg_min_frac,
            truncate=ds.truncate,
        )

    train_ds = _rebuild_dataset(train_ds)
    val_ds   = _rebuild_dataset(val_ds)
    test_ds  = _rebuild_dataset(test_ds)

    prompt_cfg   = getattr(config, "prompt", None)
    max_per_seq  = 4
    mask_prob    = 0.5
    click_prob   = 0.0

    if prompt_cfg is not None:
        max_per_seq = int(getattr(prompt_cfg, "max_per_seq", max_per_seq))
        mask_prob   = float(getattr(prompt_cfg, "mask_prob", mask_prob))
        click_prob  = float(getattr(prompt_cfg, "click_prob", click_prob))

    bbox_prob  = max(0.0, 1.0 - mask_prob - click_prob)
    prompt_mix = {"mask": mask_prob, "click": click_prob, "bbox": bbox_prob}

    for ds in (train_ds, val_ds, test_ds):
        ds.max_prompt_frames = max_per_seq
        s = sum(prompt_mix.values()) + 1e-8
        ds.prompt_mix = {k: v / s for k, v in prompt_mix.items()}

    val_ds.prompt_mix = {"mask": 0.0, "click": 0.0, "bbox": 1.0}

    return train_ds, val_ds, test_ds

def _normalize_quality_params(params: dict) -> dict:
    """
    Return only the fields that actually affect pass/fail so the signature is stable.

    Args:
        params (dict): Raw quality parameter dictionary.

    Returns:
        dict: Normalized pass/fail quality parameters.
    """
    return {
        "min_voxels": int(params.get("min_voxels", 64)),
        "min_slices_any_axis": int(params.get("min_slices_any_axis", 2)),
        "min_slice_area_px": int(params.get("min_slice_area_px", 32)),
        "downsample": int(params.get("downsample", 2)),
        "min_frac_2d_area": float(params.get("min_frac_2d_area", 0.0)),
    }

def _qc_cache_filename(ds_dir: str, split: str, quality_params: dict) -> str:
    """
    Build a quality-control cache filename from effective QC parameters.

    Args:
        ds_dir (str): Dataset directory.
        split (str): Data split name.
        quality_params (dict): Mask QC parameter dictionary.

    Returns:
        str: QC cache file path.
    """
    qp = _normalize_quality_params(quality_params)
    return os.path.join(
        ds_dir,
        f"qc_{split}"
        f"_vox{qp['min_voxels']}"
        f"_sl{qp['min_slices_any_axis']}"
        f"_area{qp['min_slice_area_px']}"
        f"_frac{qp['min_frac_2d_area']}"
        f"_ds{qp['downsample']}.json"
    )

def load_qc_cache(ds_dir: str, split: str, quality_params: dict) -> Optional[Dict[str, Tuple[bool, dict]]]:
    """
    Load QC results if they exist AND match quality settings AND file states haven't changed.
    Returns a dict: mask_path -> (ok, info) or None if not usable.

    Args:
        ds_dir (str): Dataset directory.
        split (str): Data split name.
        quality_params (dict): Mask QC parameter dictionary.

    Returns:
        dict[str, tuple[bool, dict]] | None: Fresh QC results keyed by mask path.
    """
    path = _qc_cache_filename(ds_dir, split, quality_params)
    if not os.path.isfile(path):
        return None
    try:
        data = json.load(open(path, "r"))
    except Exception:
        return None

    expected = _normalize_quality_params(quality_params)
    if _normalize_quality_params(data.get("quality", {})) != expected:
        return None
    if data.get("split") != split:
        return None

    out = {}
    for rec in data.get("masks", []):
        p = rec.get("path")
        ok = bool(rec.get("ok", False))
        info = rec.get("info", {})
        # Validate staleness using cached key vs. current file stats
        cached = tuple(rec.get("file_key", (p, 0, 0)))
        current = _mask_qc_cache_key(p)
        if cached == current:
            out[p] = (ok, info)
        # else: file changed; force recompute for this path (don’t include)
    return out

def save_qc_cache(ds_dir: str, split: str, quality_params: dict, results: Dict[str, Tuple[bool, dict]]):
    """
    Overwrites cache file with the provided results (path -> (ok, info)).

    Args:
        ds_dir (str): Dataset directory.
        split (str): Data split name.
        quality_params (dict): Mask QC parameter dictionary.
        results (dict[str, tuple[bool, dict]]): QC results keyed by mask path.

    Returns:
        None.
    """
    path = _qc_cache_filename(ds_dir, split, quality_params)
    os.makedirs(ds_dir, exist_ok=True)
    payload = {
        "split": split,
        "quality": _normalize_quality_params(quality_params),
        "masks": [
            {
                "path": p,
                "ok": bool(ok),
                "info": info,
                "file_key": _mask_qc_cache_key(p),
            }
            for p, (ok, info) in results.items()
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)

def merge_qc_cache_results(ds_dir: str, split: str, quality_params: dict,
                           new_results: Dict[str, Tuple[bool, dict]]):
    """
    Merge new per-mask results into the existing cache (if any) and save.

    Args:
        ds_dir (str): Dataset directory.
        split (str): Data split name.
        quality_params (dict): Mask QC parameter dictionary.
        new_results (dict[str, tuple[bool, dict]]): Results to merge.

    Returns:
        None.
    """
    existing = load_qc_cache(ds_dir, split, quality_params) or {}
    # Replace or add entries from new_results
    existing.update(new_results)
    save_qc_cache(ds_dir, split, quality_params, existing)

def _mask_qc_cache_key(p):
    """
    Generates a cache key for a given file path based on its metadata.

    Args:
        p (str): The file path for which to generate the cache key.

    Returns:
        tuple: A tuple containing the file path, last modification time,
               and file size. If the file cannot be accessed, returns
               the file path, 0, and 0.
    """
    try:
        st = os.stat(p)
        return (p, int(st.st_mtime), int(st.st_size))
    except OSError:
        return (p, 0, 0)

def check_mask_quality(
    mask_path,
    min_voxels=128,
    min_slices_any_axis=2,       # now interpreted as: required contiguous window length on highest-res axis
    min_slice_area_px=64,
    *,
    downsample=2,                 # Downsample factor: 1 (exact, slower), 2 (fast), 4 (very fast)
    use_full_if_borderline=True,  # Refine borderline fails at full-res
    borderline_ratio=0.8,         # How close to thresholds counts as "borderline"
    min_frac_2d_area=None,
):
    """
    Checks the quality of a 3D mask.

    New semantics for `min_slices_any_axis`:
      - Require at least one contiguous window of length N = min_slices_any_axis
        along the highest-resolution axis, where each slice in that window has
        foreground area >= max(min_slice_area_px, min_frac_2d_area * slice_area).

    Args:
        mask_path (str): The file path to the mask image.
        min_voxels (int, optional): Minimum number of non-zero voxels required.
        min_slices_any_axis (int, optional): Contiguous FG window length on the
            highest-resolution axis. Defaults to 2 (but we set it to seq_len).
        min_slice_area_px (int, optional): Minimum per-slice FG area in pixels.
        downsample (int, optional): Factor to downsample for a fast first pass.
        use_full_if_borderline (bool): If True, re-check at full-res when close
            to thresholds.
        borderline_ratio (float): Borderline band as a fraction of thresholds.
        min_frac_2d_area (float, optional): Fraction of 2D slice area required
            for per-slice FG (e.g. 0.00153 = 0.153%).

    Returns:
        tuple: (ok: bool, info: dict)
    """
    # Check cache first
    key = _mask_qc_cache_key(mask_path)
    if key in _MASK_QC_CACHE:
        return _MASK_QC_CACHE[key]

    # Read mask
    try:
        msk_itk = sitk.ReadImage(mask_path)
        m = sitk.GetArrayFromImage(msk_itk)  # [Z, Y, X]
    except Exception as e:
        res = (False, {"reason": "read_error", "error": str(e)})
        _MASK_QC_CACHE[key] = res
        return res

    # Binary 3D mask
    m_bin = (np.asarray(m) > 0)

    # Only enforce checks for 3D
    if m_bin.ndim != 3:
        res = (True, {"reason": "not_3d"})
        _MASK_QC_CACHE[key] = res
        return res

    # Full-res dims
    Z_full, Y_full, X_full = m_bin.shape

    # Downsampled grid for fast checks
    ds = int(max(1, downsample))
    m_ds = m_bin[::ds, ::ds, ::ds] if ds > 1 else m_bin
    Z_ds, Y_ds, X_ds = m_ds.shape

    # Scale thresholds to downsampled grid
    # Voxels ~ ds^3, areas ~ ds^2
    min_vox_ds = max(1, ceil(min_voxels / (ds ** 3)))
    win_len    = int(max(1, min_slices_any_axis))

    # --- 1) Total FG voxels (quick fail) ---
    vox_ds = int(m_ds.sum())
    if vox_ds < min_vox_ds:
        if use_full_if_borderline and vox_ds >= int(borderline_ratio * min_vox_ds) and ds > 1:
            return check_mask_quality(
                mask_path,
                min_voxels,
                min_slices_any_axis,
                min_slice_area_px,
                downsample=1,
                use_full_if_borderline=False,
                borderline_ratio=borderline_ratio,
                min_frac_2d_area=min_frac_2d_area,
            )
        res = (False, {
            "voxels": vox_ds * (ds ** 3),
            "slices_by_axis": [0, 0, 0],
            "max_area": 0,
            "reason": "too_few_voxels",
        })
        _MASK_QC_CACHE[key] = res
        return res

    # For info only (not gating anymore)
    sZ = int(np.count_nonzero(m_ds.any(axis=(1, 2))))
    sY = int(np.count_nonzero(m_ds.any(axis=(0, 2))))
    sX = int(np.count_nonzero(m_ds.any(axis=(0, 1))))
    slices_by_axis_ds = [sZ, sY, sX]

    max_area_ds = int(max(
        m_ds.sum(axis=(1, 2)).max(initial=0),
        m_ds.sum(axis=(0, 2)).max(initial=0),
        m_ds.sum(axis=(0, 1)).max(initial=0),
    ))

    # --- 2) Choose highest-resolution axis (same heuristic as _load_3d) ---
    # In-plane pixel counts per viewing axis (full-res proxy)
    inplane_area = np.array([
        Y_full * X_full,  # axis 0: Z-slices are YxX
        Z_full * X_full,  # axis 1: Y-slices are ZxX
        Z_full * Y_full,  # axis 2: X-slices are ZxY
    ], dtype=np.int64)
    hi_axis = int(inplane_area.argmax())

    # Full-res slice area for that axis
    if hi_axis == 0:
        full_slice_area = Y_full * X_full
    elif hi_axis == 1:
        full_slice_area = Z_full * X_full
    else:
        full_slice_area = Z_full * Y_full

    # Per-slice area along highest-res axis on the DS grid
    if hi_axis == 0:
        per_slice_area_ds = m_ds.sum(axis=(1, 2))  # [Z_ds]
        L = per_slice_area_ds.shape[0]
    elif hi_axis == 1:
        per_slice_area_ds = m_ds.sum(axis=(0, 2))  # [Y_ds]
        L = per_slice_area_ds.shape[0]
    else:
        per_slice_area_ds = m_ds.sum(axis=(0, 1))  # [X_ds]
        L = per_slice_area_ds.shape[0]

    # If the axis is shorter than the window length, we cannot get N contiguous slices
    if L < win_len:
        if use_full_if_borderline and L >= int(borderline_ratio * win_len) and ds > 1:
            return check_mask_quality(
                mask_path,
                min_voxels,
                min_slices_any_axis,
                min_slice_area_px,
                downsample=1,
                use_full_if_borderline=False,
                borderline_ratio=borderline_ratio,
                min_frac_2d_area=min_frac_2d_area,
            )
        res = (False, {
            "voxels": vox_ds * (ds ** 3),
            "slices_by_axis": slices_by_axis_ds,
            "max_area": max_area_ds * (ds ** 2),
            "reason": "axis_too_short_for_contig_window",
        })
        _MASK_QC_CACHE[key] = res
        return res

    # --- 3) Per-slice FG area threshold on that axis (full-res then scaled to DS) ---
    if min_frac_2d_area is not None:
        thr_full_frac = ceil(min_frac_2d_area * full_slice_area)
        thr_full_px   = max(min_slice_area_px, thr_full_frac)
    else:
        thr_full_px = int(min_slice_area_px)

    thr_area_ds = max(1, ceil(thr_full_px / (ds ** 2)))  # convert to DS grid

    # Boolean "good" slices on highest-res axis
    good = (per_slice_area_ds >= thr_area_ds)

    if win_len <= 0:
        contig_ok = bool(good.any())
        max_win = int(good.sum())
    else:
        x = good.astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(x)])  # [L+1]
        win_sums = cs[win_len:] - cs[:-win_len]   # [L-win_len+1]
        max_win = int(win_sums.max(initial=0) if win_sums.size > 0 else 0)
        contig_ok = bool((win_sums == win_len).any()) if win_sums.size > 0 else False

    if not contig_ok:
        # Optional full-res retry for borderline cases (e.g. 7 of 8 slices)
        if (use_full_if_borderline and ds > 1 and
            max_win >= int(borderline_ratio * win_len)):
            return check_mask_quality(
                mask_path,
                min_voxels,
                min_slices_any_axis,
                min_slice_area_px,
                downsample=1,
                use_full_if_borderline=False,
                borderline_ratio=borderline_ratio,
                min_frac_2d_area=min_frac_2d_area,
            )

        res = (False, {
            "voxels": vox_ds * (ds ** 3),
            "slices_by_axis": slices_by_axis_ds,
            "max_area": max_area_ds * (ds ** 2),
            "reason": "no_contiguous_fg_window",
            "hi_axis": hi_axis,
            "max_contig_fg_slices": max_win,
        })
        _MASK_QC_CACHE[key] = res
        return res

    # Success
    res = (True, {
        "voxels": vox_ds * (ds ** 3),
        "slices_by_axis": slices_by_axis_ds,
        "max_area": max_area_ds * (ds ** 2),
        "hi_axis": hi_axis,
        "contig_window_len": win_len,
    })
    _MASK_QC_CACHE[key] = res
    return res

def load_mask_bool_3d(path):
    """
    Read a 3D mask from disk and convert it to a PyTorch bool tensor on the CPU.

    Args:
        path (str): File path to the mask image.

    Returns:
        tuple: (mask: bool tensor, info: dict). info contains:
               - "reason": either "read_error" or "not_3d"
               - "error": if "read_error", a string with the exception message
    """
    try:
        itk = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(itk)  # [Z,Y,X]
        if arr.ndim != 3:
            return None, {"reason": "not_3d"}
        return torch.from_numpy((arr > 0)), None  # CPU bool tensor
    except Exception as e:
        return None, {"reason": "read_error", "error": str(e)}

def check_masks_quality_gpu(
    mask_paths,
    min_voxels=128,
    min_slices_any_axis=2,   # interpreted as contiguous window length on highest-res axis
    min_slice_area_px=64,
    downsample=2,
    batch_size=16,
    device=None,
    parallel_reads=8,
    min_frac_2d_area=None,
):
    """
    Vectorized GPU check for a list of 3D mask paths.
    New semantics:
      - For each 3D mask, require that on the highest-resolution axis there exists
        a contiguous window of length N = min_slices_any_axis where each slice has
        foreground area >= max(min_slice_area_px, min_frac_2d_area * slice_area).

    Returns:
        list: List of (ok: bool, info: dict) aligned to mask_paths.

    Args:
        mask_paths (list[str]): 3D mask paths to evaluate.
        min_voxels (int): Minimum foreground voxel estimate.
        min_slices_any_axis (int): Required contiguous window length.
        min_slice_area_px (int): Minimum foreground area per qualifying slice.
        downsample (int): Stride used for QC evaluation.
        batch_size (int): Number of masks per GPU batch.
        device (str | None): Torch device string, or None to auto-select.
        parallel_reads (int): Number of parallel CPU image readers.
        min_frac_2d_area (float | None): Optional per-slice fractional area threshold.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = [None] * len(mask_paths)
    scale_vox  = max(1, downsample ** 3)
    scale_area = max(1, downsample ** 2)
    win_len    = int(max(1, min_slices_any_axis))
    ds = max(1, int(downsample))

    for start in range(0, len(mask_paths), batch_size):
        chunk_paths = mask_paths[start:start + batch_size]

        # Parallel CPU reads
        with ThreadPoolExecutor(max_workers=min(parallel_reads, len(chunk_paths))) as ex:
            loaded = list(ex.map(load_mask_bool_3d, chunk_paths))

        # Prepare per-chunk outputs, build tensors list for real 3D
        to_stack = []
        idx_map = []      # Map stacked index -> results index in full list
        orig_shapes = []  # full-res shapes [(Z,Y,X),...]
        ds_shapes = []    # downsampled shapes [(Z',Y',X'),...]

        for j, (t_cpu, info) in enumerate(loaded):
            out_idx = start + j
            if t_cpu is None:
                # Either not_3d (pass) or read_error (fail)
                if info.get("reason") == "not_3d":
                    results[out_idx] = (True, info)
                else:
                    results[out_idx] = (False, info)
                continue

            # Downsample by stride on CPU, then send to GPU
            t_ds = t_cpu[::ds, ::ds, ::ds].contiguous()
            to_stack.append(t_ds)
            idx_map.append(out_idx)
            orig_shapes.append(tuple(t_cpu.shape))
            ds_shapes.append(tuple(t_ds.shape))

        if not to_stack:
            continue

        # Pad to common shape, stack, send to GPU
        D = max(t.shape[0] for t in to_stack)
        H = max(t.shape[1] for t in to_stack)
        W = max(t.shape[2] for t in to_stack)
        padded = []
        for t in to_stack:
            pad = (0, W - t.shape[2], 0, H - t.shape[1], 0, D - t.shape[0])  # W, H, D
            padded.append(torch.nn.functional.pad(t, pad))  # Pad with False
        batch = torch.stack(padded, 0).to(device=device, dtype=torch.bool, non_blocking=True)  # [B,D,H,W]

        # Vectorized metrics on GPU
        vox_ds = batch.sum(dim=(1, 2, 3))  # [B]

        # Areas per slice on each axis
        areaZ = batch.sum(dim=(2, 3))  # [B, D]
        areaY = batch.sum(dim=(1, 3))  # [B, H]
        areaX = batch.sum(dim=(1, 2))  # [B, W]

        # For info only: slices with any FG & max area across any axis
        sZ = (batch.any(dim=(2, 3))).sum(dim=1)  # [B]
        sY = (batch.any(dim=(1, 3))).sum(dim=1)
        sX = (batch.any(dim=(1, 2))).sum(dim=1)

        max_area_ds = torch.stack([
            areaZ.max(dim=1).values,
            areaY.max(dim=1).values,
            areaX.max(dim=1).values
        ], dim=1).amax(dim=1)  # [B]

        # Scale voxel threshold
        thr_vox_ds = math.ceil(min_voxels / scale_vox)

        # Now decide per-sample in Python (B is small)
        vox_out  = (vox_ds * scale_vox).to(torch.int64).tolist()
        sZ_out   = sZ.to(torch.int64).tolist()
        sY_out   = sY.to(torch.int64).tolist()
        sX_out   = sX.to(torch.int64).tolist()
        area_out = (max_area_ds * scale_area).to(torch.int64).tolist()

        for bi, out_idx in enumerate(idx_map):
            # Base voxel check
            if vox_ds[bi].item() < thr_vox_ds:
                results[out_idx] = (
                    False,
                    {
                        "voxels": int(vox_out[bi]),
                        "slices_by_axis": [int(sZ_out[bi]), int(sY_out[bi]), int(sX_out[bi])],
                        "max_area": int(area_out[bi]),
                        "reason": "too_few_voxels",
                    },
                )
                continue

            Z_full, Y_full, X_full = orig_shapes[bi]
            Z_ds_,  Y_ds_,  X_ds_  = ds_shapes[bi]

            # Highest-resolution axis using same proxy as CPU
            inplane_area = np.array([
                Y_full * X_full,  # axis 0
                Z_full * X_full,  # axis 1
                Z_full * Y_full,  # axis 2
            ], dtype=np.int64)
            hi_axis = int(inplane_area.argmax())

            # Full-res slice area for that axis
            if hi_axis == 0:
                full_slice_area = Y_full * X_full
                per_slice_ds = areaZ[bi][:Z_ds_]  # only real slices, ignore pad
                L = Z_ds_
            elif hi_axis == 1:
                full_slice_area = Z_full * X_full
                per_slice_ds = areaY[bi][:Y_ds_]
                L = Y_ds_
            else:
                full_slice_area = Z_full * Y_full
                per_slice_ds = areaX[bi][:X_ds_]
                L = X_ds_

            # If axis is shorter than N, cannot satisfy window
            if L < win_len:
                results[out_idx] = (
                    False,
                    {
                        "voxels": int(vox_out[bi]),
                        "slices_by_axis": [int(sZ_out[bi]), int(sY_out[bi]), int(sX_out[bi])],
                        "max_area": int(area_out[bi]),
                        "reason": "axis_too_short_for_contig_window",
                        "hi_axis": hi_axis,
                    },
                )
                continue

            # Per-slice FG area threshold on that axis (full-res -> DS)
            if min_frac_2d_area is not None:
                thr_full_frac = math.ceil(min_frac_2d_area * full_slice_area)
                thr_full_px   = max(min_slice_area_px, thr_full_frac)
            else:
                thr_full_px = int(min_slice_area_px)

            thr_area_ds = max(1, math.ceil(thr_full_px / scale_area))

            # Contiguous FG window check on highest-res axis
            per_slice_arr = per_slice_ds.detach().cpu().numpy().astype(np.int64)
            good = per_slice_arr >= thr_area_ds

            if win_len <= 0:
                contig_ok = bool(good.any())
                max_win = int(good.sum())
            else:
                x = good.astype(np.int32)
                cs = np.concatenate([[0], np.cumsum(x)])
                win_sums = cs[win_len:] - cs[:-win_len]   # [L-win_len+1]
                max_win = int(win_sums.max(initial=0) if win_sums.size > 0 else 0)
                contig_ok = bool((win_sums == win_len).any()) if win_sums.size > 0 else False

            if not contig_ok:
                results[out_idx] = (
                    False,
                    {
                        "voxels": int(vox_out[bi]),
                        "slices_by_axis": [int(sZ_out[bi]), int(sY_out[bi]), int(sX_out[bi])],
                        "max_area": int(area_out[bi]),
                        "reason": "no_contiguous_fg_window",
                        "hi_axis": hi_axis,
                        "max_contig_fg_slices": max_win,
                    },
                )
            else:
                results[out_idx] = (
                    True,
                    {
                        "voxels": int(vox_out[bi]),
                        "slices_by_axis": [int(sZ_out[bi]), int(sY_out[bi]), int(sX_out[bi])],
                        "max_area": int(area_out[bi]),
                        "hi_axis": hi_axis,
                        "contig_window_len": win_len,
                    },
                )

        # Free GPU memory between chunks
        del batch, vox_ds, sZ, sY, sX, areaZ, areaY, areaX, max_area_ds
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results

def compute_mask_stats_3d(mask_path: str, downsample: int = 2) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute compact, reusable 3D mask stats.

    Stored stats include:
      - full_shape: [Z,Y,X]
      - ds: downsample factor used for per-slice arrays
      - voxels_est: estimated full-res FG voxel count from ds grid
      - slices_by_axis_ds: [sZ,sY,sX] counts of slices with any FG on ds grid
      - max_area_est: estimated max per-slice FG area (scaled to full-res)
      - hi_axis: axis chosen by in-plane resolution proxy
      - per_slice_area_b64: compressed per-slice FG areas along hi_axis on ds grid (uint32)
      - per_slice_len: length of that per-slice array on ds grid

    Args:
        mask_path (str): Path to a 3D mask image.
        downsample (int): Stride used for compact per-slice stats.

    Returns:
        tuple[bool, dict[str, Any]]: Success flag and stats or error info.
    """
    try:
        msk_itk = sitk.ReadImage(mask_path)
        m = sitk.GetArrayFromImage(msk_itk)  # [Z,Y,X]
    except Exception as e:
        return False, {"reason": "read_error", "error": str(e)}

    m_bin = (np.asarray(m) > 0)
    if m_bin.ndim != 3:
        return False, {"reason": "not_3d"}

    Z_full, Y_full, X_full = m_bin.shape

    ds = int(max(1, downsample))
    m_ds = m_bin[::ds, ::ds, ::ds] if ds > 1 else m_bin
    Z_ds, Y_ds, X_ds = m_ds.shape

    # Estimated full-res FG voxel count
    vox_ds = int(m_ds.sum())
    voxels_est = int(vox_ds * (ds ** 3))

    # Slices with any FG (ds grid) for reference/diagnostics
    sZ = int(np.count_nonzero(m_ds.any(axis=(1, 2))))
    sY = int(np.count_nonzero(m_ds.any(axis=(0, 2))))
    sX = int(np.count_nonzero(m_ds.any(axis=(0, 1))))

    # Max per-slice area across axes (ds grid), scaled to full-res estimate
    max_area_ds = int(max(
        m_ds.sum(axis=(1, 2)).max(initial=0),
        m_ds.sum(axis=(0, 2)).max(initial=0),
        m_ds.sum(axis=(0, 1)).max(initial=0),
    ))
    max_area_est = int(max_area_ds * (ds ** 2))

    # Choose highest in-plane resolution axis based on full-res proxy
    inplane_area = np.array([
        Y_full * X_full,  # axis 0: Z slices are YxX
        Z_full * X_full,  # axis 1: Y slices are ZxX
        Z_full * Y_full,  # axis 2: X slices are ZxY
    ], dtype=np.int64)
    hi_axis = int(inplane_area.argmax())

    # Per-slice FG area on the chosen axis (ds grid)
    if hi_axis == 0:
        per_slice_area_ds = m_ds.sum(axis=(1, 2)).astype(np.uint32)  # [Z_ds]
    elif hi_axis == 1:
        per_slice_area_ds = m_ds.sum(axis=(0, 2)).astype(np.uint32)  # [Y_ds]
    else:
        per_slice_area_ds = m_ds.sum(axis=(0, 1)).astype(np.uint32)  # [X_ds]

    stats = {
        "full_shape": [int(Z_full), int(Y_full), int(X_full)],
        "ds": int(ds),
        "voxels_est": int(voxels_est),
        "slices_by_axis_ds": [int(sZ), int(sY), int(sX)],
        "max_area_est": int(max_area_est),
        "hi_axis": int(hi_axis),
        "per_slice_len": int(per_slice_area_ds.shape[0]),
        "per_slice_area_b64": _pack_u32_zlib_b64(per_slice_area_ds),
    }
    return True, stats

def evaluate_quality_from_stats(stats: Dict[str, Any], quality_params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide pass/fail using cached stats and the current quality_params.

    Enforces:
      - min_voxels (estimated from ds grid)
      - existence of a contiguous window of length N=min_slices_any_axis along hi_axis
        where each slice has FG area >= max(min_slice_area_px, min_frac_2d_area * full_slice_area)

    Args:
        stats (dict[str, Any]): Cached mask stats from ``compute_mask_stats_3d``.
        quality_params (dict[str, Any]): Current mask QC thresholds.

    Returns:
        tuple[bool, dict[str, Any]]: Pass/fail flag and diagnostic info.
    """
    ds = int(stats.get("ds", 1))
    Z_full, Y_full, X_full = [int(x) for x in stats.get("full_shape", [0, 0, 0])]
    hi_axis = int(stats.get("hi_axis", 0))

    min_voxels = int(quality_params.get("min_voxels", 64))
    win_len = int(max(1, quality_params.get("min_slices_any_axis", 2)))
    min_slice_area_px = int(quality_params.get("min_slice_area_px", 32))
    min_frac_2d_area = quality_params.get("min_frac_2d_area", None)

    vox_est = int(stats.get("voxels_est", 0))
    if vox_est < min_voxels:
        return False, {"reason": "too_few_voxels", "voxels": vox_est}

    # Full-res slice area for the chosen hi_axis
    if hi_axis == 0:
        full_slice_area = int(Y_full * X_full)
    elif hi_axis == 1:
        full_slice_area = int(Z_full * X_full)
    else:
        full_slice_area = int(Z_full * Y_full)

    if min_frac_2d_area is not None:
        thr_full_frac = int(math.ceil(float(min_frac_2d_area) * full_slice_area))
        thr_full_px = max(min_slice_area_px, thr_full_frac)
    else:
        thr_full_px = min_slice_area_px

    # Convert full-res threshold to ds-grid threshold
    thr_area_ds = int(max(1, math.ceil(thr_full_px / (ds ** 2))))

    # Decode per-slice areas on ds grid
    per_slice_area_ds = _unpack_u32_zlib_b64(stats["per_slice_area_b64"])
    L = int(per_slice_area_ds.shape[0])

    if L < win_len:
        return False, {
            "reason": "axis_too_short_for_contig_window",
            "hi_axis": hi_axis,
            "contig_window_len": win_len,
            "axis_len": L,
        }

    good = (per_slice_area_ds >= thr_area_ds).astype(np.int32)

    # Max contiguous window sum via prefix sums
    cs = np.concatenate([[0], np.cumsum(good)])
    win_sums = cs[win_len:] - cs[:-win_len]
    max_win = int(win_sums.max(initial=0) if win_sums.size > 0 else 0)
    contig_ok = bool((win_sums == win_len).any()) if win_sums.size > 0 else False

    if not contig_ok:
        return False, {
            "reason": "no_contiguous_fg_window",
            "hi_axis": hi_axis,
            "contig_window_len": win_len,
            "max_contig_fg_slices": max_win,
            "thr_area_ds": thr_area_ds,
        }

    return True, {
        "hi_axis": hi_axis,
        "contig_window_len": win_len,
        "thr_area_ds": thr_area_ds,
        "voxels": vox_est,
        "max_area": int(stats.get("max_area_est", 0)),
        "slices_by_axis": stats.get("slices_by_axis_ds", [0, 0, 0]),
    }
