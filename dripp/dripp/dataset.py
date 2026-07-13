import os
import sys
import argparse
import json
import logging
import random
import pickle
import threading
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from collections import defaultdict

from . import config
from .config import (
    BASE_UNPROC,
    BASE_PROC,
    DEFAULT_LOG_LEVEL,
    GROUPS_DIR,
    CT_STATS_DIR,
    GPU_ENABLED,
    CSV_FILENAME
)
from .preprocessor import Preprocessor
from .ct_profiles import load_ct_profiles
from .helpers import *
from .output_structure import render_output_dirs

class SegmentationDataset:
    """
    A class to create a Segmentation Dataset based on pre-built groups JSON files.
    
    This class loads the lean groups JSON file that contains nested subdataset structure,
    flattens it into individual groups, and then copies the images and masks into the
    structured output directory.
    """
    def __init__(self, dataset_name, metadata):
        """
        Initialize the dataset with its name and metadata.

        Args:
            dataset_name (str): The name of the dataset.
            metadata (dict): The metadata describing the dataset (including grouping settings,
                             folder structure, file types, etc.).
        """
        self.dataset_name = dataset_name
        self.metadata = metadata
        root_dir = self.metadata.get("root_directory")
        if root_dir:
            self.dataset_dir = normalize_path(os.path.join(BASE_UNPROC, dataset_name, root_dir))
        else:
            self.dataset_dir = normalize_path(os.path.join(BASE_UNPROC, dataset_name))
        self.output_dir = normalize_path(os.path.join(BASE_PROC, dataset_name))
        modalities = self.metadata.get("modalities") or ["default"]
        for mod in modalities:
            os.makedirs(normalize_path(os.path.join(self.output_dir, mod)), exist_ok=True)
        os.makedirs(normalize_path(os.path.join(self.output_dir, ".preprocessing_logs")), exist_ok=True)
        self.processed_pairs = {"train": 0, "test": 0}
        self.mask_classes = parse_mask_classes(self.metadata.get("mask_classes"))

    def process(self, preprocessor: Preprocessor, max_groups=0, workers=1, modalities=None):
        """
        Process the dataset by reorganizing files according to the pre-built groups JSON file.

        Loads the groups file (which contains nested subdataset groupings), flattens the nested structure,
        and then calls _process_grouping_and_copy() to copy the image and mask files to the structured output.
        Groups are filtered based on the max_groups parameter separately for training and test splits.

        Args:
            preprocessor (Preprocessor): An instance for preprocessing (if needed).
            max_groups (int): Maximum number of groups to process per split (train/test).
        """
        self.preprocessor = preprocessor
        modality_filter = normalize_modality_filter(modalities)

        self.preprocessor.dataset_logger.info(f"Processing dataset '{self.dataset_name}'...")
        self.preprocessor.main_logger.info(f"\nProcessing dataset '{self.dataset_name}'...")

        grouping_regex = self.metadata.get("grouping_regex")
        subdataset_configs = get_regex_configs(grouping_regex, self.metadata) if grouping_regex else None

        # Load the groups JSON file instead of the full index file.
        groups_file = os.path.join(GROUPS_DIR, f"{self.dataset_name}_groups.json")
        try:
            with open(groups_file, "r") as f:
                groups_data = json.load(f)
            self.preprocessor.dataset_logger.info(f"Using groups file {groups_file} for dataset '{self.dataset_name}'.")

            # Expect the groups file to contain a nested structure under the key "subdatasets".
            subdatasets = groups_data.get("subdatasets", [])
            self.preprocessor.dataset_logger.info(f"Found {len(subdatasets)} subdataset(s).")
            
            # Flatten the nested subdataset structure: add subdataset information to each group.
            flat_groups = []
            for sub in subdatasets:
                sub_name = sub.get("name", "default")
                sub_modality = sub.get("modality", (self.metadata.get("modalities") or ["default"])[0])
                if not modality_is_selected(sub_modality, modality_filter):
                    continue
                for split in ["train", "test"]:
                    for group in sub.get(split, []):
                        group["subdataset_name"] = sub_name
                        group["subdataset_modality"] = sub_modality
                        flat_groups.append(group)

            # Sample up to max_groups per split per subdataset
            if max_groups > 0:
                groups_by_key = defaultdict(list)
                for g in flat_groups:
                    key = (g["subdataset_name"], g["split"])
                    groups_by_key[key].append(g)

                sampled = []
                for grp_list in groups_by_key.values():
                    if len(grp_list) > max_groups:
                        sampled.extend(random.sample(grp_list, max_groups))
                    else:
                        sampled.extend(grp_list)
                flat_groups = sampled

            # Build the identifiers dictionary for further processing.
            identifiers = {}
            for g in flat_groups:
                key = (g["identifier"], g["split"], g.get("subdataset_name"))
                identifiers[key] = g

            if modality_filter is not None:
                self.preprocessor.dataset_logger.info(
                    "Modality filter %s selected %d group(s)",
                    ", ".join(sorted(modality_filter)), len(identifiers),
                )

            started = time.perf_counter()
            grouping_metadata = self._process_grouping_and_copy(
                identifiers,
                subdataset_configs,
                self.preprocessor.dataset_logger,
                workers=workers,
            )
            self.preprocessor.dataset_logger.info(
                "Processed %d groups with %d worker(s) in %.2f seconds",
                len(identifiers), workers, time.perf_counter() - started,
            )
        except Exception as e:
            self.preprocessor.dataset_logger.error(f"Error loading groups file {groups_file}: {e}.")
            return

        total_groups = len(grouping_metadata)
        self.preprocessor.main_logger.info(f"Processing of dataset '{self.dataset_name}' complete. {total_groups} identifier group(s) processed. Outputs at: {self.output_dir}")
        # self.metadata["Num Images"] = sum(len(entry["images"]) for entry in grouping_metadata)
        # self.metadata["Num Masks"] = sum(len(entry["masks"]) for entry in grouping_metadata)
        self.metadata["Preprocessed?"] = "yes"

        grouping_meta_path = normalize_path(os.path.join(self.output_dir, "groupings.json"))
        with open(grouping_meta_path, "w") as f:
            json.dump(grouping_metadata, f, indent=2)

    def _process_grouping_and_copy(self, identifiers, subdataset_configs, logger, workers=1):
        """
        Process each group in the identifiers dictionary by copying and preprocessing the corresponding image and mask files.

        For each group, the function extracts and sorts file paths, builds output directories, and calls the preprocess_group method
        of the preprocessor with the pipeline as the first argument. It then collects all configured processed image and mask
        output files. If preprocess_group returned multiple modality volumes, this function splits the group into separate entries
        with distinct identifiers.

        Args:
            identifiers (dict): A dictionary with group identifiers as keys and group metadata as values.
            subdataset_configs (list): A list of subdataset configuration dictionaries.
            logger (logging.Logger): The logger to use for logging.

        Returns:
            list: A list of dictionaries, each containing the group identifier, original and processed image and mask file lists,
                  and preprocessing metadata.
        """
        if workers > 1 and len(identifiers) > 1:
            thread_state = threading.local()

            def process_one(item):
                if not hasattr(thread_state, "dataset"):
                    base = self.preprocessor
                    worker_preprocessor = Preprocessor(
                        target_size=base.target_size,
                        ct_profiles=base.ct_profiles,
                        min_mask_size=base.min_mask_size,
                        dataset_logger=logger,
                        dataset_name=base.dataset_name,
                        background_value=base.background_value,
                    )
                    worker_dataset = SegmentationDataset(self.dataset_name, self.metadata)
                    worker_dataset.preprocessor = worker_preprocessor
                    thread_state.dataset = worker_dataset

                key, entry = item
                started = time.perf_counter()
                result = thread_state.dataset._process_grouping_and_copy(
                    {key: entry}, subdataset_configs, logger, workers=1
                )
                logger.info("Processed group %s in %.2f seconds", key, time.perf_counter() - started)
                return result

            grouping_metadata = []
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dripp-group") as executor:
                for result in executor.map(process_one, identifiers.items()):
                    grouping_metadata.extend(result)
            return grouping_metadata

        grouping_metadata = []
        for key, entry in identifiers.items():
            if not entry["images"]:
                logger.warning(f"Group {key} has no images. Skipping...")
                continue

            # Extract and sort file paths
            image_paths = [
                item["path"] if isinstance(item, dict) else item
                for item in entry["images"]
            ]
            mask_paths = [
                item["path"] if isinstance(item, dict) else item
                for item in entry["masks"]
            ]
            image_paths.sort()
            mask_paths.sort()

            composite_id = get_composite_identifier(entry)

            # Build output directories
            sub_name = entry.get("subdataset_name") or entry.get("name")
            sub_modality = entry.get("subdataset_modality") or ((self.metadata.get("modalities") or ["default"])[0])
            sub_pipeline = entry.get("subdataset_pipeline")

            output_dirs = render_output_dirs(
                BASE_PROC,
                self.dataset_name,
                entry,
                sub_modality,
                sub_name,
                composite_id,
                config.OUTPUT_STRUCTURE["group_folder_template"],
                config.OUTPUT_STRUCTURE["images_folder"],
                config.OUTPUT_STRUCTURE["masks_folder"],
            )
            img_out_dir = output_dirs["img_out_dir"]
            mask_out_dir = output_dirs["mask_out_dir"]
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)

            # Call preprocess_group with pipeline as first argument
            try:
                group_options = entry.get("preprocessing_options")
                if group_options is None:
                    group_options = get_preprocessing_options(self.metadata, sub_name, include_defaults=False)
                preprocessing_metadata = self.preprocessor.preprocess_group(
                    sub_name,
                    sub_pipeline,
                    image_paths,
                    mask_paths,
                    sub_modality,
                    img_out_dir,
                    mask_out_dir,
                    composite_id,
                    self.mask_classes,
                    group_options
                )
            except Exception as e:
                logger.error(f"Error preprocessing group {key}: {e}", exc_info=True)
                continue

            # Gather all processed files from img_out_dir and mask_out_dir
            proc_imgs = sorted(
                os.path.join(img_out_dir, fn)
                for fn in os.listdir(img_out_dir)
                if fn.lower().endswith(tuple(config.OUTPUT_EXTS))
            )
            mask_files = sorted(
                os.path.join(mask_out_dir, fn)
                for fn in os.listdir(mask_out_dir)
                if fn.lower().endswith(tuple(config.OUTPUT_EXTS))
            )
            # Pair each mask path with its corresponding class
            proc_masks = [
                {"path": p, "class": extract_mask_class(p)}
                for p in mask_files
            ]

            # Drop this group if no masks were produced
            if not proc_masks:
                logger.warning(f"Group {key} produced no processed masks. Skipping group.")
                continue

            # If preprocess_group returned multiple modality NIfTIs, split into separate entries
            if (isinstance(preprocessing_metadata, dict)
                    and "image_niftis" in preprocessing_metadata
                    and len(preprocessing_metadata["image_niftis"]) > 1):
                for idx, img_nif in enumerate(preprocessing_metadata["image_niftis"]):
                    new_id = f"{composite_id}_modality{idx}"
                    mask_nifti_paths = preprocessing_metadata.get("mask_niftis", [])
                    proc_masks = [{"path": p, "class": extract_mask_class(p)} for p in mask_nifti_paths]
                    split_entry = {
                        "identifier":       new_id,
                        "short_id":         self.preprocessor._short_id(new_id),
                        "images":           entry["images"],
                        "masks":            entry["masks"],  # Original raw-mask paths
                        "proc_images":      [img_nif],       # Processed NIfTIs
                        "proc_masks":       proc_masks,
                        "preprocessing_metadata": {
                            "resize_shape": preprocessing_metadata.get("resize_shape"),
                            "volume_shape": preprocessing_metadata.get("volume_shape")
                        }
                    }
                    grouping_metadata.append(split_entry)
            elif group_options.get("split_processed_images_by_modality"):
                for idx, img in enumerate(proc_imgs):
                    new_id = f"{composite_id}_modality{idx}"
                    entry_dict = {
                        "identifier":       new_id,
                        "short_id":         self.preprocessor._short_id(new_id),
                        "images":           entry["images"],
                        "masks":            entry["masks"],  # Original raw-mask paths
                        "proc_images":      [img],       # Processed image
                        "proc_masks":       proc_masks,
                        "preprocessing_metadata": {}
                    }
                    grouping_metadata.append(entry_dict)
            else:
                entry_dict = {
                    "identifier":             composite_id,
                    "short_id":               self.preprocessor._short_id(composite_id),
                    "images":                 entry["images"],
                    "masks":                  entry["masks"],
                    "proc_images":            proc_imgs,
                    "proc_masks":             proc_masks,
                    "preprocessing_metadata": {}
                }

                # Always include resize_shape if present
                if isinstance(preprocessing_metadata, dict) and "resize_shape" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["resize_shape"] = preprocessing_metadata["resize_shape"]

                # If video metadata is present, include fps & num_frames
                if isinstance(preprocessing_metadata, dict) and "fps" in preprocessing_metadata and "num_frames" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["fps"] = preprocessing_metadata["fps"]
                    entry_dict["preprocessing_metadata"]["num_frames"] = preprocessing_metadata["num_frames"]

                # If 3D metadata is present, include volume_shape and NIfTI paths
                if isinstance(preprocessing_metadata, dict) and "volume_shape" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["volume_shape"] = preprocessing_metadata["volume_shape"]
                    entry_dict["preprocessing_metadata"]["image_niftis"] = preprocessing_metadata.get("image_niftis", [])
                    entry_dict["preprocessing_metadata"]["mask_niftis"] = preprocessing_metadata.get("mask_niftis", [])

                grouping_metadata.append(entry_dict)

        return grouping_metadata

class DatasetManager:
    """
    Loader class to manage processing of multiple Segmentation Datasets.
    
    Loads dataset metadata from a CSV file, instantiates SegmentationDataset objects,
    processes each dataset, and updates the CSV file with new metadata.
    """
    def __init__(self, csv_path):
        """
        Initialize the DatasetManager.

        Args:
            csv_path (str): Path to the CSV file containing dataset metadata.
            preprocessor (Preprocessor): Instance for preprocessing operations.
        """
        self.csv_path = csv_path
        self.metadata = load_dataset_metadata(csv_path)
        self.datasets = []
        for name, meta in self.metadata.items():
            self.datasets.append(SegmentationDataset(name, meta))

    def process_all(self, start_at=None, max_groups=0, workers=1, modalities=None):
        """
        Process all datasets by reorganizing files based on their groups JSON files.
        If start_at is given, skip everything up to (but not including) that dataset.

        Args:
            start_at (str, optional): Name of dataset to resume from.
            max_groups (int): Optional maximum number of groups per split (train/test) to process.
        """
        # Determine start index
        idx = 0
        if start_at:
            idx = next((i for i, ds in enumerate(self.datasets)
                        if ds.dataset_name == start_at), None)
            if idx is None:
                raise ValueError(f"Dataset '{start_at}' not found; cannot resume from it.")

        selected_modalities = normalize_modality_filter(modalities)
        datasets = self.datasets[idx:]
        if selected_modalities is not None:
            datasets = [
                dataset for dataset in datasets
                if any(
                    modality_is_selected(modality, selected_modalities)
                    for modality in dataset.metadata.get("modalities", [])
                )
            ]

        for dataset in tqdm(
            datasets,
            desc=(f"Processing all datasets from '{start_at}'" if start_at else "Processing all datasets"),
            total=len(datasets),
        ):
            self.process_dataset(
                dataset.dataset_name, max_groups=max_groups, workers=workers,
                modalities=selected_modalities,
            )

        self.update_csv()

    def process_dataset(self, dataset_name, max_groups=0, workers=1, modalities=None):
        """
        Process a single dataset by reorganizing files based on its groups JSON file.

        Args:
            dataset_name (str): Name of the dataset to process.
            max_groups (int, optional): Optional maximum number of groups per split (train/test) to process.

        Raises:
            ValueError: If the dataset name is not found in the metadata.
        """
        # Find the dataset object
        dataset = next((ds for ds in self.datasets if ds.dataset_name == dataset_name), None)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found in metadata")
        selected_modalities = normalize_modality_filter(modalities)
        if selected_modalities is not None and not any(
            modality_is_selected(modality, selected_modalities)
            for modality in dataset.metadata.get("modalities", [])
        ):
            logging.getLogger(f"SegmentationDataset.{dataset_name}").info(
                "Skipping dataset %s because it has no selected modalities: %s",
                dataset_name, ", ".join(sorted(selected_modalities)),
            )
            return False

        # Create and configure a per-dataset logger
        dataset_logger = logging.getLogger(f"SegmentationDataset.{dataset_name}")
        dataset_logger.setLevel(DEFAULT_LOG_LEVEL)
        log_dir = os.path.join(BASE_PROC, dataset_name, ".preprocessing_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{dataset_name}.log")
        if not dataset_logger.handlers:
            fh = RotatingFileHandler(
                log_path,
                maxBytes=400 * 1024 * 1024,    # Rotate after 400 MB
                backupCount=5                  # Keep 5 old log files
            )
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            dataset_logger.addHandler(fh)

        # Load validated subdataset-specific CT profiles if applicable
        meta = dataset.metadata
        processing_ct = any(m.lower() == "ct" for m in meta.get("modalities", [])) and (
            selected_modalities is None or "ct" in selected_modalities
        )
        if processing_ct:
            stats_path = os.path.join(CT_STATS_DIR, f"{dataset_name}_ct_stats.json")
            ct_profiles = load_ct_profiles(stats_path, expected_dataset=dataset_name)
            dataset_logger.info("Loaded %d validated CT profile(s) from %s", len(ct_profiles), stats_path)

            preprocessor = Preprocessor(
                target_size=(512, 512),
                dataset_logger=dataset_logger,
                ct_profiles=ct_profiles,
                dataset_name=dataset_name,
                background_value=meta.get("background_value", 0)
            )
        else:
            preprocessor = Preprocessor(
                target_size=(512, 512),
                dataset_logger=dataset_logger,
                dataset_name=dataset_name,
                background_value=meta.get("background_value", 0)
            )

        # Run the preprocessing for this one dataset
        dataset.process(
            preprocessor, max_groups=max_groups, workers=workers,
            modalities=selected_modalities,
        )
        # Preprocessors contain imported array modules and logger state that cannot be pickled.
        # The exported manager only needs the completed dataset metadata.
        dataset.preprocessor = None

        # Update the CSV metadata to reflect any changes
        self.update_csv()
        return True

    def update_csv(self):
        """
        Update the CSV file in the processed directory with the latest metadata after processing.
        """
        updated_data = []
        for dataset in self.datasets:
            meta = dataset.metadata.copy()
            meta["Dataset Name"] = dataset.dataset_name
            updated_data.append(meta)
        df = pd.DataFrame(updated_data)
        new_csv_path = normalize_path(os.path.join(BASE_PROC, CSV_FILENAME))
        df.to_csv(new_csv_path, index=False)

    def export(self, export_path):
        """
        Export the DatasetManager instance to a pickle file for later reuse.

        Args:
            export_path (str): Path to the export pickle file.
        """
        with open(normalize_path(export_path), 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset manager exported to {export_path}.")

    @classmethod
    def load_from_file(cls, export_path):
        """
        Load a DatasetManager instance from a pickle file.

        Args:
            export_path (str): Path to the pickle file.

        Returns:
            DatasetManager: The loaded DatasetManager instance.
        """
        with open(normalize_path(export_path), 'rb') as f:
            manager = pickle.load(f)
        print(f"Dataset manager loaded from {export_path}.")
        return manager

if __name__ == "__main__":
    # Command-line Arg Parsing
    parser = argparse.ArgumentParser(description="Process one or all segmentation datasets")
    parser.add_argument(
        '-preprocess',
        action='store_true',
        help="Run preprocessing on one or all datasets"
    )
    parser.add_argument(
        '-d', '--dataset',
        help="Name of the specific dataset to process"
    )
    parser.add_argument(
        '-a', '--all',
        nargs='?',            # Allow optional value
        const='',             # When provided without a value, args.all == ''
        metavar='START_AT',
        help="Process all datasets; optionally resume at START_AT"
    )
    parser.add_argument(
        '-m', '--max-groups',
        type=int,
        default=0,
        help="Maximum number of groups per split (train/test) to process"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for preprocessing."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=2,
        help="Number of concurrent CPU case workers (default: 2)."
    )
    parser.add_argument(
        "--output-dir",
        help="Override the configured preprocessing output directory."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --max-groups samples cases (default: 42)."
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=[],
        help="Only preprocess this modality (case-insensitive; repeatable or comma-separated)."
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    random.seed(args.seed)
    if args.output_dir:
        BASE_PROC = normalize_path(args.output_dir)
        config.BASE_PROC = BASE_PROC
    modality_filter = normalize_modality_filter(args.modality)

    # Override the config flag at runtime:
    config.GPU_ENABLED = args.gpu

    # Require the preprocess flag
    if not args.preprocess:
        parser.error("Missing required argument: -preprocess")

    csv_path = normalize_path(os.path.join(BASE_UNPROC, CSV_FILENAME))
    manager = DatasetManager(csv_path)

    # Run processing
    if args.dataset:
        if args.dataset not in manager.metadata:
            print(f"Dataset '{args.dataset}' not found in {csv_path}.")
            sys.exit(1)
        manager.process_dataset(
            args.dataset, max_groups=args.max_groups, workers=args.workers,
            modalities=modality_filter,
        )
    elif args.all is not None:
        # If args.all == '' then no start name provided, so resume from beginning
        start = args.all or None
        manager.process_all(
            start_at=start, max_groups=args.max_groups, workers=args.workers,
            modalities=modality_filter,
        )
    else:
        print("Please specify either --dataset/-d or --all/-a.")

    manager.export(normalize_path(os.path.join(BASE_PROC, "dataset_manager.pkl")))
