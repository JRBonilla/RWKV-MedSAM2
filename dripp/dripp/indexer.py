import os
import argparse
import json
import logging
import datetime
import random
from tqdm import tqdm

from .helpers import (
    normalize_path,
    collect_files,
    resolve_folder,
    resolve_mask_folders,
    load_dataset_metadata,
    extract_identifier_from_path,
    match_subdataset_regex,
    get_regex_configs,
    set_indexing_log,
    get_extension,
    parse_mask_classes,
    parse_segmentation_tasks
)
from .config import BASE_UNPROC, INDEX_DIR, VIDEO_EXTS, VOLUME_EXTS, DEFAULT_LOG_LEVEL

# Accumulate tasks across all datasets
# Each entry has the following format:
# "task_name": {
#   "classes": set([...]),
#   "datasets": { dataset_name: set([subdataset_name, ...]), ...}
# }
GLOBAL_TASKS = {}

# Configure logger: log to both console and a file named "indexing.log"
logger = logging.getLogger("DatasetIndexer")
logger.setLevel(DEFAULT_LOG_LEVEL)
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
        # Filter out "id2" from additional
        grouping_additional = {k: v for k, v in additional.items() if k != "id2"} if additional else None
        # Include config.name in the key so that groups from different subdatasets do not merge.
        key = (ident, split, config.name, frozenset(grouping_additional.items()) if grouping_additional else None)
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
        # Filter out "id2" from additional
        grouping_additional = {k: v for k, v in additional.items() if k != "id2"} if additional else None
        key = (ident, split, frozenset(grouping_additional.items()) if grouping_additional else None)
        return key, additional, None

def build_groupings(index_data, dataset_dir, metadata):
    """
    Build grouping metadata and organize groups into nested subdatasets.

    The function splits image records into training and testing groups according
    to the provided metadata and subdataset configurations. It returns a dict
    with one key "subdatasets", which is a list of subdataset dicts. Each
    subdataset dict contains:
      - modality: the modality for the subdataset.
      - name: the subdataset name.
      - tasks: list of segmentation task names relevant to this subdataset.
      - classes: dict of mask class mappings for this subdataset.
      - train: a list of training groups.
      - test: a list of test groups.

    Args:
        index_data (dict): Contains 'images_train', 'images_test', and 'mask_files'.
        dataset_dir (str): Absolute path to the dataset directory.
        metadata (dict): Dataset metadata, including grouping strategy, regex,
                         train_ratio, segmentation_tasks, and mask_classes.

    Returns:
        dict: { "subdatasets": [ { modality, name, tasks, classes, train, test }, ... ] }
    """
    train_ratio         = metadata.get("train_ratio", 0.8)
    grouping_strategy   = metadata.get("grouping_strategy", "regex")
    grouping_regex      = metadata.get("grouping_regex")
    seg_tasks           = parse_segmentation_tasks(metadata.get('segmentation_tasks') or '')
    mask_classes        = parse_mask_classes(metadata.get('mask_classes') or '')
    subdataset_configs  = get_regex_configs(grouping_regex, metadata) if grouping_regex else None

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
        if subdataset_configs:
            # Match against each subdataset config
            for cfg in subdataset_configs:
                ident, additional, _ = match_subdataset_regex(
                    rec["path"],
                    grouping_strategy,
                    [cfg],              # only this config
                    True,               # is_image=True
                    split,
                    dataset_dir
                )
                if ident is None:
                    continue

                # This image belongs to cfg.name on the training split
                sname    = cfg.name or "default"
                modality = getattr(cfg, "modality", (metadata.get("modalities") or ["default"])[0])
                pipeline = cfg.pipeline or determine_pipeline(rec["path"])
                ensure_subdataset(sname, modality)
                group_dict = subdatasets[sname][split]
                group_add  = {k: v for k, v in additional.items() if k != "id2"} if additional else None

                key = (
                    ident,
                    split,
                    cfg.name,
                    frozenset(group_add.items()) if group_add else None
                )
                if key not in group_dict:
                    group_dict[key] = {
                        "identifier": ident,
                        "split": split,
                        "additional": additional,
                        "images": [],
                        "masks": [],
                        "subdataset_name": sname,
                        "subdataset_modality": modality,
                        "subdataset_pipeline": pipeline
                    }
                group_dict[key]["images"].append({"path": rec["path"], "id": rec["id"]})
                # do NOT break—allow other subdatasets to match
        else:
            # Default grouping if no subdataset configs
            key, additional, config = compute_group_key(rec, split, grouping_strategy, subdataset_configs, dataset_dir)
            if key is None:
                continue
            sname = "default"
            modality = (metadata.get("modalities") or ["default"])[0]
            pipeline = determine_pipeline(rec["path"])
            ensure_subdataset(sname, modality)
            group_dict = subdatasets[sname][split]
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
        if subdataset_configs:
            # Match against each subdataset config
            for cfg in subdataset_configs:
                ident, additional, _ = match_subdataset_regex(
                    rec["path"],
                    grouping_strategy,
                    [cfg],              # only this config
                    True,               # is_image=True
                    split,
                    dataset_dir
                )
                if ident is None:
                    continue

                # This image belongs to cfg.name on the test split
                sname    = cfg.name or "default"
                modality = getattr(cfg, "modality", (metadata.get("modalities") or ["default"])[0])
                pipeline = cfg.pipeline or determine_pipeline(rec["path"])
                ensure_subdataset(sname, modality)
                group_dict = subdatasets[sname][split]
                group_add  = {k: v for k, v in additional.items() if k != "id2"} if additional else None

                key = (
                    ident,
                    split,
                    cfg.name,
                    frozenset(group_add.items()) if group_add else None
                )
                if key not in group_dict:
                    group_dict[key] = {
                        "identifier": ident,
                        "split": split,
                        "additional": additional,
                        "images": [],
                        "masks": [],
                        "subdataset_name": sname,
                        "subdataset_modality": modality,
                        "subdataset_pipeline": pipeline
                    }
                group_dict[key]["images"].append({"path": rec["path"], "id": rec["id"]})
                # do NOT break—allow other subdatasets to match
        else:
            # Default grouping if no subdataset configs
            key, additional, config = compute_group_key(rec, split, grouping_strategy, subdataset_configs, dataset_dir)
            if key is None:
                continue
            sname = "default"
            modality = (metadata.get("modalities") or ["default"])[0]
            pipeline = determine_pipeline(rec["path"])
            ensure_subdataset(sname, modality)
            group_dict = subdatasets[sname][split]
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

    # Process mask files.
    logger.info("Processing mask files for grouping...")
    if subdataset_configs:
       for rec in tqdm(index_data.get("mask_files", []), desc="Mask files (subdataset)"):
           # Try matching every subdataset config, for both train & test
           for cfg in subdataset_configs:
               for split in ("train", "test"):
                   ident, additional, _ = match_subdataset_regex(
                       rec["path"],
                       grouping_strategy,
                       [cfg],              # test only this config
                       False,
                       split,
                       dataset_dir
                   )
                   if ident is None:
                       continue

                   # This mask belongs to cfg.name on this split
                   sname = cfg.name or "default"
                   ensure_subdataset(sname, cfg.modality)
                   group_dict = subdatasets[sname][split]
                   group_add  = {k: v for k, v in additional.items() if k != "id2"} if additional else None
                   mask_key = (
                       ident, split, cfg.name,
                       frozenset(group_add.items()) if group_add else None
                   )
                   if mask_key in group_dict:
                       group_dict[mask_key]["masks"].append({
                           "path": rec["path"],
                           "id": rec.get("id")
                       })
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

            # Remove id2 from additional
            m_group_add = {k: v for k, v in m_additional.items() if k != "id2"} if m_additional else None

            # Determine the correct split by seeing if the mask is in train or test
            train_key = (m_ident, "train", frozenset(m_group_add.items()) if m_group_add else None)
            test_key  = (m_ident, "test",  frozenset(m_group_add.items()) if m_group_add else None)

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
    # Enforce that within each subdataset, no image or mask is shared across groups
    for sname, sub in subdatasets.items():
        image_paths = {}
        mask_paths  = {}

        for split in ("train", "test"):
            for key, grp in sub[split].items():
                for img in grp["images"]:
                    p = img["path"]
                    if p in image_paths:
                        raise ValueError(
                            f"Duplicate image '{p}' in subdataset '{sname}' "
                            f"in groups {image_paths[p]} and {key}"
                        )
                    image_paths[p] = key
                for m in grp["masks"]:
                    p = m["path"]
                    if p in mask_paths:
                        raise ValueError(
                            f"Duplicate mask '{p}' in subdataset '{sname}' "
                            f"in groups {mask_paths[p]} and {key}"
                        )
                    mask_paths[p] = key

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
            "tasks": [
                t for t, cls_list in seg_tasks.items()
                if any(cls in mask_classes.get(sub["name"], {}) for cls in cls_list)
            ],
            "classes": mask_classes.get(sub["name"], {}),
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

    Returns:
        dict: A summary dict for GUI use, containing:
            - images_train (list): List of training image records.
            - images_test (list): List of test image records.
            - mask_files (list): List of mask file records.
            - groups (list): List of subdataset grouping dicts.
            - unannotated_groups (list): List of groups without any masks.
    """
    set_indexing_log(INDEX_DIR, dataset_name)
    root_folder = metadata.get("root_directory") or ""
    dataset_dir = (normalize_path(os.path.join(BASE_UNPROC, dataset_name, root_folder))
                   if root_folder else normalize_path(os.path.join(BASE_UNPROC, dataset_name)))
    logger.info(f"Indexing dataset '{dataset_name}' from directory: {dataset_dir}")

    # Parse the mask classes and segmentation tasks
    mask_classes = parse_mask_classes(metadata.get('mask_classes') or '')
    tasks        = parse_segmentation_tasks(metadata.get('segmentation_tasks') or '')

    # Ensure that mask classes and segmentation tasks are defined
    if not mask_classes:
        logger.error(f"No mask classes defined for dataset '{dataset_name}'.")
        return
    if not tasks:
        logger.error(f"No segmentation tasks defined for dataset '{dataset_name}'.")
        return

    # Register the segmentation tasks with the global task map
    for task_id, cls_list in tasks.items():
        entry = GLOBAL_TASKS.setdefault(task_id, {"classes": set(), "datasets": {}})
        # Collect all classes associated with this task
        entry["classes"].update(cls_list)
        # Find all subdatasets in this dataset that contain any of these classes
        for subname, cls_rules in mask_classes.items():
            if any(c in cls_rules for c in cls_list):
                ds_map = entry["datasets"].setdefault(dataset_name, set())
                ds_map.add(subname)

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
        # Index all
        for name, meta in datasets_metadata.items():
            try:
                index_dataset(name, meta)
            except Exception as e:
                logger.error(f"Error indexing dataset '{name}': {e}")

        # Save tasks
        out_tasks = {
            task: {
                "classes": sorted(list(info["classes"])),
                "datasets": {
                    ds: sorted(list(subs))
                    for ds, subs in info["datasets"].items()
                }
            }
            for task, info in GLOBAL_TASKS.items()
        }
        tasks_folder = os.path.join(INDEX_DIR, "Tasks")
        os.makedirs(tasks_folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(CSV_PATH))[0]
        tasks_file = os.path.join(tasks_folder, f"{base}_tasks.json")
        with open(tasks_file, "w") as tf:
            json.dump(out_tasks, tf, indent=2)
        logger.info(f"Tasks for all datasets saved to {tasks_file}.")
    else:
        parser.error("You must specify either --dataset or --all")

    logger.info("Indexing complete.")
