# fix_all_groupings_paths_v2.py
# Rebuild <dataset>_groups.json for each dataset by merging original index
# data with preprocessed entries from <dataset>/groupings.json.
#
# Matching is done by (raw_image_path, raw_mask_path), not by identifier.
# For each local preprocessed entry, we create a new index entry that
# copies the index metadata and overrides identifier, short_id, and
# proc_* fields with the local ones.
#
# Run from anywhere. Uses /data/Preprocessed as root.

import os
import json
import sys
import argparse
from copy import deepcopy
from tqdm import tqdm

PREPROC_ROOT = "/data/Preprocessed"
INDEX_DIR = "/data/DatasetIndexes/Groups"


def norm_path(p):
    if p is None:
        return None
    return os.path.normpath(p)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Loading JSON from {path}: {e}", file=sys.stderr)
        return None


def save_json(data, path):
    try:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Saving JSON to {path}: {e}", file=sys.stderr)


def build_local_lookup(local_groups, log_fn):
    """
    Build mapping:
      key = (raw_image_path, raw_mask_path) -> list of local entries
    """
    lookup = {}
    for idx, local_entry in enumerate(local_groups):
        imgs = local_entry.get("images", []) or []
        masks = local_entry.get("masks", []) or []

        if not imgs or not masks:
            log_fn(
                f"[LOCAL_SKIP] idx={idx} identifier='{local_entry.get('identifier', '')}' "
                f"has no images or masks"
            )
            continue

        img_path = norm_path(imgs[0].get("path"))
        mask_path = norm_path(masks[0].get("path"))
        key = (img_path, mask_path)

        if key not in lookup:
            lookup[key] = []
        lookup[key].append(local_entry)

        log_fn(
            f"[LOCAL] key=({img_path}, {mask_path}) "
            f"identifier='{local_entry.get('identifier', '')}', "
            f"short_id='{local_entry.get('short_id', '')}', "
            f"n_proc_images={len(local_entry.get('proc_images', []) or [])}, "
            f"n_proc_masks={len(local_entry.get('proc_masks', []) or [])}"
        )

    return lookup


def merge_for_dataset(dataset_name: str) -> bool:
    dataset_dir = os.path.join(PREPROC_ROOT, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    grouping_path = os.path.join(dataset_dir, "groupings.json")
    index_path = os.path.join(INDEX_DIR, f"{dataset_name}_groups.json")
    log_path = os.path.join(dataset_dir, "merging.log")

    with open(log_path, "w", encoding="utf-8") as log_f:

        def log_fn(msg: str):
            print(msg, file=log_f)

        log_fn("=" * 80)
        log_fn(f"Merging dataset: {dataset_name}")
        log_fn(f"  PREPROC_ROOT : {PREPROC_ROOT}")
        log_fn(f"  dataset_dir  : {dataset_dir}")
        log_fn(f"  index_path   : {index_path}")
        log_fn(f"  grouping_path: {grouping_path}")
        log_fn("=" * 80)

        if not os.path.isfile(index_path):
            msg = f"[SKIP] Index file missing for dataset '{dataset_name}': {index_path}"
            log_fn(msg)
            print(msg)
            return False

        index_data = load_json(index_path)
        if index_data is None:
            msg = f"[SKIP] Could not load index for '{dataset_name}'"
            log_fn(msg)
            print(msg)
            return False

        if not os.path.isfile(grouping_path):
            msg = (
                f"[INFO] No groupings.json for '{dataset_name}'. "
                f"Copying original index into {dataset_dir}/{dataset_name}_groups.json"
            )
            log_fn(msg)
            print(msg)
            out_path = os.path.join(dataset_dir, f"{dataset_name}_groups.json")
            log_fn(f"[SAVE] Writing index JSON to {out_path}")
            save_json(index_data, out_path)
            return True

        local_groups = load_json(grouping_path)
        if local_groups is None:
            msg = f"[SKIP] Could not load groupings.json for '{dataset_name}'"
            log_fn(msg)
            print(msg)
            return False

        log_fn("[STEP] Building local lookup by (image.path, mask.path)...")
        local_lookup = build_local_lookup(local_groups, log_fn)

        # We will track which local entries actually got used
        used_local_ids = set(id(le) for lst in local_lookup.values() for le in lst)

        total_index_entries = 0
        total_new_entries = 0
        total_matched_index_entries = 0

        # Build new subdatasets list with cloned entries
        new_subdatasets = []

        for subd_idx, subd in enumerate(index_data.get("subdatasets", [])):
            sub_name = subd.get("name", "default")
            log_fn("")
            log_fn("=" * 40)
            log_fn(f"[SUBDATASET] {sub_name} (index position {subd_idx})")
            log_fn("=" * 40)

            new_subd = deepcopy(subd)

            for split in ("train", "test"):
                entries = subd.get(split, []) or []
                new_entries = []

                log_fn("")
                log_fn(f"[SPLIT] subdataset={sub_name}, split={split}")
                log_fn(f"  original entry count: {len(entries)}")

                for ei, idx_entry in enumerate(entries):
                    total_index_entries += 1

                    imgs = idx_entry.get("images", []) or []
                    masks = idx_entry.get("masks", []) or []

                    if not imgs or not masks:
                        log_fn(
                            f"[INDEX_SKIP] subdataset={sub_name} split={split} "
                            f"idx={ei} identifier='{idx_entry.get('identifier', '')}' "
                            f"has no images or masks - keeping original entry unchanged"
                        )
                        new_entries.append(idx_entry)
                        continue

                    img_path = norm_path(imgs[0].get("path"))
                    mask_path = norm_path(masks[0].get("path"))
                    key = (img_path, mask_path)

                    local_matches = local_lookup.get(key, [])

                    log_fn(
                        f"[INDEX_ENTRY] idx={ei} identifier='{idx_entry.get('identifier', '')}' "
                        f"key=({img_path}, {mask_path}) local_matches={len(local_matches)}"
                    )

                    if not local_matches:
                        # No preprocessed version, keep original
                        new_entries.append(idx_entry)
                        log_fn("  [NO_LOCAL_MATCH] keeping original index entry")
                        continue

                    # There are one or more local entries for this key
                    total_matched_index_entries += 1

                    for li, local_entry in enumerate(local_matches):
                        # Mark this local entry as used
                        used_local_ids.discard(id(local_entry))

                        local_id = local_entry.get("identifier", "")
                        short_id = local_entry.get("short_id", None)
                        proc_images = local_entry.get("proc_images", []) or []
                        proc_masks = local_entry.get("proc_masks", []) or []
                        meta = local_entry.get("preprocessing_metadata", {}) or {}

                        clone = deepcopy(idx_entry)
                        old_identifier = clone.get("identifier", "")

                        clone["identifier"] = local_id
                        if short_id is not None:
                            clone["short_id"] = short_id

                        clone["proc_images"] = proc_images
                        clone["proc_masks"] = proc_masks
                        clone["preprocessing_metadata"] = meta

                        # Ensure split and additional at least exist
                        if "split" not in clone:
                            clone["split"] = split
                        if "additional" not in clone:
                            clone["additional"] = idx_entry.get("additional", {}) or {}

                        new_entries.append(clone)
                        total_new_entries += 1

                        log_fn(
                            f"  [CLONE_FROM_INDEX] local_match {li}: "
                            f"old_identifier='{old_identifier}' -> new_identifier='{local_id}', "
                            f"short_id='{short_id}', "
                            f"n_proc_images={len(proc_images)}, "
                            f"n_proc_masks={len(proc_masks)}"
                        )

                new_subd[split] = new_entries
                log_fn(
                    f"[SPLIT_SUMMARY] subdataset={sub_name} split={split} "
                    f"new_entry_count={len(new_entries)}"
                )

            new_subdatasets.append(new_subd)

        index_data["subdatasets"] = new_subdatasets

        # Any local entries never used by any index entry
        log_fn("")
        log_fn("[STEP] Local entries without any matching index entry:")
        unused_count = 0
        for key, local_list in local_lookup.items():
            img_path, mask_path = key
            for le in local_list:
                if id(le) in used_local_ids:
                    unused_count += 1
                    log_fn(
                        f"  [LOCAL_UNMATCHED] key=({img_path}, {mask_path}) "
                        f"identifier='{le.get('identifier', '')}', "
                        f"short_id='{le.get('short_id', '')}'"
                    )
        if unused_count == 0:
            log_fn("  None")

        log_fn("")
        log_fn("[SUMMARY] Merge results:")
        log_fn(f"  total index entries (original)       : {total_index_entries}")
        log_fn(f"  index entries with local match       : {total_matched_index_entries}")
        log_fn(f"  total new cloned entries from locals : {total_new_entries}")
        log_fn(f"  local entries with no index match    : {unused_count}")

        print(f"Dataset: {dataset_name}")
        print(f"  total index entries (original)       : {total_index_entries}")
        print(f"  index entries with local match       : {total_matched_index_entries}")
        print(f"  total new cloned entries from locals : {total_new_entries}")
        print(f"  local entries with no index match    : {unused_count}")

        # Save updated groups next to groupings.json
        out_path = os.path.join(dataset_dir, f"{dataset_name}_groups.json")
        log_fn("")
        log_fn(f"[SAVE] Writing merged index to {out_path}")
        save_json(index_data, out_path)

        log_fn("")
        log_fn("[DONE] Merge completed for this dataset.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild <dataset>_groups.json for all datasets using groupings.json, "
            "matching by image and mask paths and cloning index entries per "
            "preprocessed sample."
        )
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Single dataset folder to fix (name under /data/Preprocessed).",
        default=None,
    )
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            d
            for d in os.listdir(PREPROC_ROOT)
            if os.path.isdir(os.path.join(PREPROC_ROOT, d))
        )

    for ds in tqdm(datasets, desc="Datasets", leave=False):
        print(f"Merging dataset {ds}...")
        merge_for_dataset(ds)

    print("Done rebuilding group files.")
