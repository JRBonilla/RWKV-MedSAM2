import os
import argparse
import json
import logging
import datetime
import random
from tqdm import tqdm

from helpers import (
    normalize_path,
    collect_files,
    resolve_folder,
    resolve_mask_folders,
    load_dataset_metadata,
    extract_identifier_from_path,
    match_subdataset_regex,
    get_regex_configs,
    set_indexing_log,
    get_extension
)

# Define base directories
# BASE_UNPROC = "F:/Datasets/"
# INDEX_DIR   = "F:/DatasetIndexes"  # Directory where index JSON files and logs are stored
BASE_UNPROC = "/data/research/"
INDEX_DIR   = "/data/DatasetIndexes"  # Directory where index JSON files and logs are stored

# If a pipeline isn't specified in the header, infer it from file-extension:
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"} 
VOLUME_EXTS = {
    ".nii", ".nii.gz", ".nrrd", ".mhd", ".mha", ".img", ".hdr", ".dicom", ".dcm", ".npy"
}

# Configure logger: log to both console and a file named "indexing.log"
logger = logging.getLogger("DatasetIndexer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Stream (console) handler.
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    # File handler: logs saved to INDEX_DIR/indexing_<timestamp>.log
    os.makedirs(INDEX_DIR, exist_ok=True)
    # e.g. indexing_20250520_141305.log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(INDEX_DIR, f"indexing_{timestamp}.log")
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

def determine_pipeline(file_path):
    """
    Determine the pipeline to use for a given file by its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The pipeline to use ("Video", "2D", or "3D").
    """
    ext = get_extension(file_path)
    if ext in VIDEO_EXTS:
        return "Video"
    elif ext in VOLUME_EXTS:
        return "3D"
    else:
        return "2D"

def compute_group_key(rec, split, grouping_strategy, subdataset_configs, base_dir):
    """
    Compute the grouping key for a file record using either subdataset regex or a default extraction.

    Args:
        rec (dict): A dictionary representing the image record with at least a "path" key.
        split (str): The data split ("train" or "test").
        grouping_strategy (str): Strategy to use for grouping (e.g., "regex-file", "regex-folder", or "filename").
        subdataset_configs (list or None): List of subdataset configuration objects if available.
        base_dir (str): Base directory used for relative path calculations.

    Returns:
        tuple:
            - key (tuple or None): A tuple key uniquely identifying the group.
            - additional (dict or None): Additional data extracted for grouping.
            - config (object or None): The subdataset config used (if applicable).
    """
    if subdataset_configs:
        try:
            ident, additional, config = match_subdataset_regex(
                rec["path"], grouping_strategy, subdataset_configs, True, split,
                base_dir
            )
        except Exception as e:
            logger.error(f"Error in match_subdataset_regex for {rec['path']}: {e}")
            return None, None, None
        if ident is None:
            return None, None, None
        # Include config.name in the key so that groups from different subdatasets do not merge.
        key = (ident, split, config.name, frozenset(additional.items()) if additional else None)
        return key, additional, config
    else:
        try:
            ident, additional = extract_identifier_from_path(
                rec["path"], grouping_strategy, regex=None,
                base_dir=base_dir
            )
        except Exception as e:
            logger.error(f"Error in extract_identifier_from_path for {rec['path']}: {e}")
            return None, None, None
        if ident is None:
            return None, None, None
        key = (ident, split, frozenset(additional.items()) if additional else None)
        return key, additional, None

def build_groupings(index_data, dataset_dir, metadata):
    """
    Build grouping metadata and organize groups into nested subdatasets.

    The function splits image records into training and testing groups according to the provided
    metadata and subdataset configurations. It returns a dictionary with one key "subdatasets", which
    is a list of subdataset dictionaries. Each subdataset dictionary contains:
      - modality: the modality for the subdataset.
      - name: the subdataset name.
      - train: a list of training groups.
      - test: a list of test groups.

    Args:
        index_data (dict): Dictionary containing lists for 'images_train', 'images_test', and 'mask_files'.
        dataset_dir (str): The absolute path to the dataset directory.
        metadata (dict): Metadata for the dataset, including grouping settings.

    Returns:
        dict: A dictionary with key "subdatasets" containing grouped split data.
    """
    train_ratio = metadata.get("train_ratio", 0.8)
    grouping_strategy = metadata.get("grouping_strategy", "regex")
    grouping_regex = metadata.get("grouping_regex")
    subdataset_configs = get_regex_configs(grouping_regex, metadata) if grouping_regex else None

    # Use a dictionary to accumulate subdataset groups.
    subdatasets = {}  # Keys: subdataset name

    def ensure_subdataset(sname, modality):
        """
        Ensure that a subdataset with the given name exists in the subdatasets dictionary.
        If not, initialize it with empty train/test dictionaries.
        """
        if sname not in subdatasets:
            subdatasets[sname] = {
                "modality": modality,
                "name": sname,
                "train": {},
                "test": {}
            }

    # Process training images.
    logger.info("Processing training images for grouping...")
    for rec in tqdm(index_data.get("images_train", []), desc="Training images"):
        split = "train"
        key, additional, config = compute_group_key(rec, split, grouping_strategy, subdataset_configs, dataset_dir)
        if key is None:
            continue
        if subdataset_configs and config:
            sname = config.name or "default"
            modality = getattr(config, "modality", (metadata.get("modalities") or ["default"])[0])
            pipeline = config.pipeline or determine_pipeline(rec["path"])
        else:
            sname = "default"
            modality = (metadata.get("modalities") or ["default"])[0]
            pipeline = determine_pipeline(rec["path"])
        ensure_subdataset(sname, modality)
        group_dict = subdatasets[sname]["train"]
        if key not in group_dict:
            group_dict[key] = {
                "identifier": key[0],
                "split": split,
                "additional": additional,
                "images": [],
                "masks": [],
                "subdataset_name": sname,
                "subdataset_modality": modality,
                "subdataset_pipeline": pipeline
            }
        group_dict[key]["images"].append({"path": rec["path"], "id": rec["id"]})
    
    # Process test images.
    logger.info("Processing test images for grouping...")
    for rec in tqdm(index_data.get("images_test", []), desc="Test images"):
        split = "test"
        key, additional, config = compute_group_key(rec, split, grouping_strategy, subdataset_configs, dataset_dir)
        if key is None:
            continue
        if subdataset_configs and config:
            sname = config.name or "default"
            modality = getattr(config, "modality", (metadata.get("modalities") or ["default"])[0])
            pipeline = config.pipeline or determine_pipeline(rec["path"])
        else:
            sname = "default"
            modality = (metadata.get("modalities") or ["default"])[0]
            pipeline = determine_pipeline(rec["path"])
        ensure_subdataset(sname, modality)
        sub_test = subdatasets[sname]["test"]
        if key not in sub_test:
            sub_test[key] = {
                "identifier": key[0],
                "split": split,
                "additional": additional,
                "images": [],
                "masks": [],
                "subdataset_name": sname,
                "subdataset_modality": modality,
                "subdataset_pipeline": pipeline
            }
        sub_test[key]["images"].append({"path": rec["path"], "id": rec["id"]})
    
    # Process mask files.
    logger.info("Processing mask files for grouping...")
    if subdataset_configs:
        for rec in tqdm(index_data.get("mask_files", []), desc="Mask files (subdataset)"):
            # Try both test-mask and train-mask patterns
            m_ident = m_additional = config = None
            for split in ("train", "test"):
                try:
                    ident, additional, cfg = match_subdataset_regex(
                        rec["path"], grouping_strategy, subdataset_configs, False, split,
                        dataset_dir
                    )
                except Exception as e:
                    logger.error(f"Error processing mask {rec['path']}: {e}")
                    continue
                if ident is not None:
                    m_ident, m_additional, config = ident, additional, cfg
                    break
            if m_ident is None:
                continue

            sname = config.name or "default"
            modality = config.modality
            ensure_subdataset(sname, modality)

            # Pick the correct split
            group_dict = subdatasets[sname][split]
            mask_key = (m_ident, split, config.name, frozenset(m_additional.items()) if m_additional else None)
            if mask_key in group_dict:
                group_dict[mask_key]["masks"].append({"path": rec["path"], "id": rec.get("id")})
    else:
        for rec in tqdm(index_data.get("mask_files", []), desc="Mask files"):
            m_ident = m_additional = None
            for split in ("train", "test"):
                try:
                    ident, additional = extract_identifier_from_path(
                        rec["path"], grouping_strategy, regex=None,
                        base_dir=dataset_dir
                    )
                except Exception as e:
                    logger.error(f"Error processing mask {rec['path']}: {e}")
                    continue
                if ident is not None:
                    m_ident, m_additional = ident, additional
                    break
                
            if m_ident is None:
                continue
            
            sname = "default"
            modality = (metadata.get("modalities") or ["default"])[0]
            ensure_subdataset(sname, modality)

            # Determine the correct split by seeing if the mask is in train or test
            train_key = (m_ident, "train", frozenset(m_additional.items()) if m_additional else None)
            test_key  = (m_ident, "test",  frozenset(m_additional.items()) if m_additional else None)

            if train_key in subdatasets[sname]["train"]:
                split = "train"
                group_dict = subdatasets[sname]["train"]
                mask_key = train_key
            elif test_key in subdatasets[sname]["test"]:
                split = "test"
                group_dict = subdatasets[sname]["test"]
                mask_key = test_key
            else:
                continue

            group_dict[mask_key]["masks"].append({"path": rec["path"], "id": rec.get("id")})
    
    # Remove groups with no masks
    for sname, sub in subdatasets.items():
        for split in ("train","test"):
            orphan_keys = [k for k, grp in sub[split].items() if len(grp["masks"]) == 0]
            for k in orphan_keys:
                # Record and remove
                index_data.setdefault("unannotated_groups", []).append(f"{sname}.{split}:{k[0]}")
                logger.warning(f"Removing unannotated group {k} from {sname}.{split}")
                del sub[split][k]

    # If no test images were provided, randomly split each training group.
    for sname, sub in subdatasets.items():
        if len(index_data.get("images_test", [])) == 0:
            logger.info(f"No test images for subdataset {sname}. Randomly splitting training groups.")
            new_train = {}
            new_test = {}
            for key, group in sub["train"].items():
                if random.random() < train_ratio:
                    group["split"] = "train"
                    new_train[key] = group
                else:
                    group["split"] = "test"
                    new_test[key] = group
            sub["train"] = new_train
            sub["test"] = new_test
    
    # Convert each subdataset's group dictionaries to lists.
    out_subdatasets = []
    for sname, sub in subdatasets.items():
        train_list = list(sub["train"].values())
        test_list = list(sub["test"].values())
        out_subdatasets.append({
            "modality": sub["modality"],
            "name": sub["name"],
            "train": train_list,
            "test": test_list
        })
    
    return {"subdatasets": out_subdatasets}

def index_dataset(dataset_name, metadata):
    """
    Index a dataset by scanning for images and mask files, and grouping them by subdataset.

    The function collects image files from train and test folders and mask files from the mask folders.
    It then builds grouping metadata using build_groupings, writes out an index file and a separate groups-only file.

    Args:
        dataset_name (str): Name of the dataset.
        metadata (dict): Dataset metadata including folder names, file types, and grouping parameters.
    """
    set_indexing_log(INDEX_DIR, dataset_name)
    root_folder = metadata.get("root_directory") or ""
    dataset_dir = (normalize_path(os.path.join(BASE_UNPROC, dataset_name, root_folder))
                   if root_folder else normalize_path(os.path.join(BASE_UNPROC, dataset_name)))
    logger.info(f"Indexing dataset '{dataset_name}' from directory: {dataset_dir}")

    train_folders = metadata.get("train_folders") or []
    test_folders = metadata.get("test_folders") or []
    mask_folders = metadata.get("mask_folders") or []

    train_folder_paths = []
    for fld in train_folders:
        train_folder_paths.extend(resolve_folder(dataset_dir, fld))
    if not train_folder_paths:
        train_folder_paths = [dataset_dir]

    test_folder_paths = []
    for fld in test_folders:
        test_folder_paths.extend(resolve_folder(dataset_dir, fld))

    resolved_mask_folders = resolve_mask_folders(dataset_dir, mask_folders)
    if not resolved_mask_folders:
        resolved_mask_folders = [dataset_dir]

    allowed_image_types = metadata.get("image_file_types", [])
    allowed_mask_types = metadata.get("mask_file_types", [])
    mask_key = metadata.get("mask_key")

    index_data = {"dataset_name": dataset_name, "dataset_dir": dataset_dir, "metadata": metadata}
    if test_folder_paths:
        logger.info("Collecting training images from train folders...")
        images_train = collect_files(
            "",
            train_folder_paths,
            allowed_image_types,
            use_folder_identifier=False,
            exclude_folders=mask_folders,
            mask_key=mask_key
        )
        images_train = sorted(images_train, key=lambda x: x["path"])
        logger.info("Collecting test images from test folders...")
        images_test = collect_files(
            "",
            test_folder_paths,
            allowed_image_types,
            use_folder_identifier=False,
            exclude_folders=mask_folders,
            mask_key=mask_key
        )
        images_test = sorted(images_test, key=lambda x: x["path"])
        index_data["images_train"] = images_train
        index_data["images_test"] = images_test
    else:
        logger.info("No test folders found. Collecting all images from training folders for later random split...")
        all_images = collect_files(
            "",
            train_folder_paths,
            allowed_image_types,
            use_folder_identifier=False,
            exclude_folders=mask_folders,
            mask_key=mask_key
        )
        all_images = sorted(all_images, key=lambda x: x["path"])
        index_data["images_train"] = all_images
        index_data["images_test"] = []

    logger.info("Collecting mask files...")
    mask_files = collect_files(
        "",
        resolved_mask_folders,
        allowed_mask_types,
        is_mask=True,
        mask_key=mask_key
    )
    mask_files = sorted(mask_files, key=lambda x: x["path"])
    index_data["mask_files"] = mask_files

    groups = build_groupings(index_data, dataset_dir, metadata)
    index_data["groups"] = groups

    # Write the full index file.
    index_folder = os.path.join(INDEX_DIR, "Indexes")
    os.makedirs(index_folder, exist_ok=True)
    index_file_path = os.path.join(index_folder, f"{dataset_name}_index.json")
    with open(index_file_path, "w") as index_file:
        json.dump(index_data, index_file, indent=2)
    logger.info(f"Dataset index with grouped splits for '{dataset_name}' saved to {index_file_path}.")

    # Write out a separate groups-only file.
    groups_folder = os.path.join(INDEX_DIR, "Groups")
    os.makedirs(groups_folder, exist_ok=True)
    groups_file = os.path.join(groups_folder, f"{dataset_name}_groups.json")
    with open(groups_file, "w") as gf:
        json.dump(groups, gf, indent=2)
    logger.info(f"Groups for '{dataset_name}' saved to {groups_file}.")

    # Return for GUI
    return {
        "images_train":       index_data.get("images_train", []),
        "images_test":        index_data.get("images_test",  []),
        "mask_files":         index_data.get("mask_files",   []),
        "groups":             groups.get("subdatasets", []) if isinstance(groups, dict) else groups,
        "unannotated_groups": index_data.get("unannotated_groups", [])
    }


if __name__ == "__main__":
    # Command-line Arg Parsing
    parser = argparse.ArgumentParser(description="Index one or all datasets")
    parser.add_argument(
        '-d', '--dataset',
        help="Name of the specific dataset to index"
    )
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help="Index all datasets (mutually exclusive with --dataset)"
    )
    args = parser.parse_args()

    # Load metadata
    CSV_PATH = normalize_path(os.path.join(BASE_UNPROC, "datasets.csv"))
    datasets_metadata = load_dataset_metadata(CSV_PATH)

    # Run indexing
    if args.dataset:
        if args.dataset not in datasets_metadata:
            logger.error(f"Dataset '{args.dataset}' not found in {CSV_PATH}.")
            sys.exit(1)
        index_dataset(args.dataset, datasets_metadata[args.dataset])
    elif args.all:
        for name, meta in datasets_metadata.items():
            try:
                index_dataset(name, meta)
            except Exception as e:
                logger.error(f"Error indexing dataset '{name}': {e}")
    else:
        parser.error("You must specify either --dataset or --all")

    logger.info("Indexing complete.")
