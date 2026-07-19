import os
import sys
import json
import argparse
import pandas as pd
from dripp.preprocessor import Preprocessor
from dripp.helpers import parse_mask_classes

# Monkey-patch Preprocessor to intercept mask saving.
def fake_save_mask(self, mask, mode, idx, comp_idx, composite_id, out_dir, mask_tag=None):
    """
    Print mask classification instead of writing a mask file.

    Args:
        mask (Any): Mask image or component array.
        mode (Any): Preprocessing output mode.
        idx (Any): Slice index.
        comp_idx (Any): Component index.
        composite_id (Any): Composite group identifier.
        out_dir (Any): Output directory.
        mask_tag (Any): Resolved mask class tag.

    Returns:
        None.
    """
    # Print the classification of each component instead of writing to disk
    print(f"[CLASS] Group={composite_id} | component {idx}-{comp_idx} -> class: {mask_tag}")
    return None

def fake_save_image(self, img, mode, idx, composite_id, out_dir):
    """
    Skip image writes while testing mask classification.

    Args:
        img (Any): Image array.
        mode (Any): Preprocessing output mode.
        idx (Any): Slice index.
        composite_id (Any): Composite group identifier.
        out_dir (Any): Output directory.

    Returns:
        None.
    """
    # No-op for images
    return None

Preprocessor._save_mask = fake_save_mask
Preprocessor._save_image = fake_save_image

def flatten_groups(groups_data):
    """
    Flatten group JSON subdatasets into split-level group records.

    Args:
        groups_data (Any): Parsed groups JSON data.

    Returns:
        list[dict]: Flattened group records.
    """
    flat = []
    for sub in groups_data.get("subdatasets", []):
        for split in ("train", "test"):
            for grp in sub.get(split, []):
                grp["subdataset_name"] = sub.get("name", "default")
                # Carry over modality and pipeline.
                grp["modality"] = grp.get("subdataset_modality", sub.get("modality"))
                grp["pipeline"] = grp.get("subdataset_pipeline", sub.get("pipeline"))
                grp["split"] = split
                flat.append(grp)
    return flat

def main():
    """
    Run the mask-class tester command-line interface.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Test match_mask_class on processed mask components via Preprocessor"
    )
    parser.add_argument("-b", "--base_dir", default="/data/research",
                        help="Base directory for raw files (for relative paths)")
    parser.add_argument("-c", "--csv", default="/data/research/datasets.csv",
                        help="Path to datasets CSV")
    parser.add_argument("-i", "--index_dir", default="/data/DatasetIndexes/Groups",
                        help="Directory with [DatasetName]_groups.json files")
    parser.add_argument("-d", "--dataset", required=True,
                        help="Dataset name as in CSV and Groups JSON")
    parser.add_argument("-s", "--subdatasets", nargs="+",
                        help="Optional filter for subdataset names")
    args = parser.parse_args()

    # Load CSV and parse mask_classes
    df = pd.read_csv(args.csv)
    row = df[df['Dataset Name'] == args.dataset]
    if row.empty:
        print(f"[ERROR] Dataset '{args.dataset}' not found in CSV")
        return
    spec = row['Mask Classes'].iloc[0]
    mask_classes = parse_mask_classes(spec)

    # Load groups JSON
    groups_file = os.path.join(args.index_dir, f"{args.dataset}_groups.json")
    with open(groups_file, "r") as f:
        groups_data = json.load(f)
    flat = flatten_groups(groups_data)
    if args.subdatasets:
        flat = [g for g in flat if g["subdataset_name"] in args.subdatasets]

    # Instantiate Preprocessor (use defaults or adjust as needed)
    prep = Preprocessor(dataset_name=args.dataset)

    # Process each group without saving masks/images
    for grp in flat:
        sub = grp["subdataset_name"]
        pipeline = grp["pipeline"]
        modality = grp["modality"]
        comp_id = grp["identifier"]
        print(f"\n=== [{grp['split'].upper()}] Subdataset: {sub} | Group: {comp_id} ===")
        # Resolve file lists
        image_list = [rec["path"] for rec in grp.get("images", [])]
        mask_list  = [rec["path"] for rec in grp.get("masks", [])]
        # Normalize mask paths if relative
        mask_list = [
            m if os.path.isabs(m) else os.path.join(args.base_dir, m)
            for m in mask_list
        ]
        # Call preprocess (will invoke fake_save_mask)
        try:
            prep.preprocess_group(
                sub_name=sub,
                pipeline=pipeline,
                image_files=image_list,
                mask_files=mask_list,
                modality=modality,
                img_out_dir="/tmp/img_out",   # Dummy
                mask_out_dir="/tmp/msk_out",  # Dummy
                composite_id=comp_id,
                mask_classes=mask_classes
            )
        except FileNotFoundError as e:
            # Suppress only missing-mask_out_dir errors
            if 'mask_out_dir' in str(e) or '/tmp/msk_out' in str(e):
                sys.exit(0)
            else:
                raise

if __name__ == "__main__":
    main()
