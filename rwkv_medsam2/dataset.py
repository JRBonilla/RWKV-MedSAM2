import os
import pickle
import pandas as pd
import json
import logging
from tqdm import tqdm
from preprocessor import Preprocessor
import random

from helpers import *

# Base directories
BASE_UNPROC = "F:/Datasets/"
BASE_PROC = "F:/Preprocessed/"
INDEX_DIR = "F:/DatasetIndexes"  # Directory where index and groups JSON files are stored

class UnifiedMedicalDataset:
    """
    A class to create a Unified Medical Dataset based on pre-built groups JSON files.
    
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

    def process(self, preprocessor: Preprocessor, force_structure=True, max_groups=0):
        """
        Process the dataset by reorganizing files according to the pre-built groups JSON file.

        Loads the groups file (which contains nested subdataset groupings), flattens the nested structure,
        and then calls _process_grouping_and_copy() to copy the image and mask files to the structured output.
        Groups are filtered based on the max_groups parameter separately for training and test splits.

        Args:
            preprocessor (Preprocessor): An instance for preprocessing (if needed).
            force_structure (bool): Whether to force a specific directory structure (not used in this example).
            max_groups (int): Maximum number of groups to process per split (train/test).
        """
        dataset_logger = logging.getLogger(f"UnifiedMedicalDataset.{self.dataset_name}")
        dataset_logger.setLevel(logging.INFO)
        dataset_log_path = normalize_path(os.path.join(self.output_dir, ".preprocessing_logs", f"{self.dataset_name}.log"))
        if not dataset_logger.handlers:
            fh = logging.FileHandler(dataset_log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            dataset_logger.addHandler(fh)

        dataset_logger.info(f"Processing dataset '{self.dataset_name}'...")
        print(f"\nProcessing dataset '{self.dataset_name}'...")

        grouping_regex = self.metadata.get("grouping_regex")
        subdataset_configs = get_regex_configs(grouping_regex, self.metadata) if grouping_regex else None

        # Load the groups JSON file instead of the full index file.
        groups_file = os.path.join(INDEX_DIR, f"{self.dataset_name}_groups.json")
        try:
            with open(groups_file, "r") as f:
                groups_data = json.load(f)
            dataset_logger.info(f"Using groups file {groups_file} for dataset '{self.dataset_name}'.")

            # Expect the groups file to contain a nested structure under the key "subdatasets".
            subdatasets = groups_data.get("subdatasets", [])
            dataset_logger.info("Found {} subdataset(s).".format(len(subdatasets)))
            
            # Flatten the nested subdataset structure: add subdataset information to each group.
            flat_groups = []
            for sub in subdatasets:
                sub_name = sub.get("name", "default")
                sub_modality = sub.get("modality", (self.metadata.get("modalities") or ["default"])[0])
                for split in ["train", "test"]:
                    for group in sub.get(split, []):
                        group["subdataset_name"] = sub_name
                        group["subdataset_modality"] = sub_modality
                        flat_groups.append(group)

            if max_groups > 0:
                # Apply max_groups separately per split.
                train_groups = [g for g in flat_groups if g["split"]=="train"]
                test_groups = [g for g in flat_groups if g["split"]=="test"]
                if len(train_groups) > max_groups:
                    train_groups = random.sample(train_groups, max_groups)
                if len(test_groups) > max_groups:
                    test_groups = random.sample(test_groups, max_groups)
                flat_groups = train_groups + test_groups

            # Build the identifiers dictionary for further processing.
            identifiers = {}
            for g in flat_groups:
                key = (g["identifier"], g["split"], g.get("subdataset_name"))
                identifiers[key] = g

            grouping_metadata = self._process_grouping_and_copy(identifiers, subdataset_configs, dataset_logger)
        except Exception as e:
            dataset_logger.error(f"Error loading groups file {groups_file}: {e}.")
            return

        total_groups = len(grouping_metadata)
        print(f"Processing of dataset '{self.dataset_name}' complete. {total_groups} identifier group(s) processed. Outputs at: {self.output_dir}")
        self.metadata["Num Images"] = sum(len(entry["images"]) for entry in grouping_metadata)
        self.metadata["Num Masks"] = sum(len(entry["masks"]) for entry in grouping_metadata)
        self.metadata["Preprocessed?"] = "yes"

        grouping_meta_path = normalize_path(os.path.join(self.output_dir, "groupings.json"))
        with open(grouping_meta_path, "w") as f:
            json.dump(grouping_metadata, f, indent=2)

    def _process_grouping_and_copy(self, identifiers, subdataset_configs, logger):
        """
        Process each group: sort image and mask file paths, build the appropriate output directory path,
        call the helper to copy the files, and return updated grouping metadata.

        Uses subdataset information to decide the output structure (i.e. modality and subdataset name).

        Args:
            identifiers (dict): Dictionary of groups keyed by an identifier tuple.
            subdataset_configs (list or None): List of subdataset configuration objects, if applicable.
            logger (logging.Logger): Logger instance for debug output.

        Returns:
            list: A list of processed group dictionaries containing file paths and subdataset info.
        """
        grouping_metadata = []
        for key, entry in identifiers.items():
            if not entry["images"]:
                logger.warning(f"Group {key} has no images. Skipping...")
                continue
            # Extract file paths.
            image_paths = [item["path"] if isinstance(item, dict) else item for item in entry["images"]]
            mask_paths = [item["path"] if isinstance(item, dict) else item for item in entry["masks"]]
            image_paths.sort()
            mask_paths.sort()
            composite_id = get_composite_identifier(entry)
            # Use subdataset information to build the output directory.
            sub_name = entry.get("subdataset_name") or entry.get("name")
            sub_modality = entry.get("subdataset_modality") or (self.metadata.get("modalities") or ["default"])[0]
            if sub_name:
                out_dir = os.path.join(self.output_dir, sub_modality, sub_name, entry["split"])
            else:
                out_dir = os.path.join(self.output_dir, sub_modality, entry["split"])
            id_subfolders = composite_id.split("_")
            out_dir = os.path.join(out_dir, *id_subfolders)
            out_dir = normalize_path(out_dir)
            img_out_dir = os.path.join(out_dir, "images")
            mask_out_dir = os.path.join(out_dir, "masks")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)
            copy_group_files(image_paths, mask_paths, img_out_dir, mask_out_dir, composite_id, logger=logger)
            grouping_metadata.append({
                "identifier": composite_id,
                "additional": entry["additional"],
                "subdataset_name": sub_name,
                "subdataset_modality": sub_modality,
                "split": entry["split"],
                "images": entry["images"],
                "masks": entry["masks"]
            })
        return grouping_metadata


class DatasetLoader:
    """
    Loader class to manage processing of multiple unified medical datasets.
    
    Loads dataset metadata from a CSV file, instantiates UnifiedMedicalDataset objects,
    processes each dataset, and updates the CSV file with new metadata.
    """
    def __init__(self, csv_path, preprocessor: Preprocessor):
        """
        Initialize the DatasetLoader.

        Args:
            csv_path (str): Path to the CSV file containing dataset metadata.
            preprocessor (Preprocessor): Instance for preprocessing operations.
        """
        self.csv_path = csv_path
        self.metadata = load_dataset_metadata(csv_path)
        self.datasets = []
        for name, meta in self.metadata.items():
            self.datasets.append(UnifiedMedicalDataset(name, meta))
        self.preprocessor = preprocessor

    def process_all(self, force_structure=True, max_groups=0):
        """
        Process all datasets by reorganizing files based on their groups JSON files.

        Args:
            force_structure (bool): (Optional) Whether to force a specific directory structure.
            max_groups (int): (Optional) Maximum number of groups per split (train/test) to process.
        """
        for dataset in tqdm(self.datasets, desc="Processing all datasets"):
            dataset.process(self.preprocessor, force_structure=force_structure, max_groups=max_groups)
        self.update_csv()

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
        new_csv_path = normalize_path(os.path.join(BASE_PROC, "datasets.csv"))
        df.to_csv(new_csv_path, index=False)

    def export(self, export_path):
        """
        Export the DatasetLoader instance to a pickle file for later reuse.

        Args:
            export_path (str): Path to the export pickle file.
        """
        with open(normalize_path(export_path), 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset loader exported to {export_path}.")

    @classmethod
    def load_from_file(cls, export_path):
        """
        Load a DatasetLoader instance from a pickle file.

        Args:
            export_path (str): Path to the pickle file.

        Returns:
            DatasetLoader: The loaded DatasetLoader instance.
        """
        with open(normalize_path(export_path), 'rb') as f:
            loader = pickle.load(f)
        print(f"Dataset loader loaded from {export_path}.")
        return loader


if __name__ == "__main__":
    csv_path = normalize_path(os.path.join(BASE_UNPROC, "datasets.csv"))
    preprocessor = Preprocessor(target_size=(1024, 1024))
    loader = DatasetLoader(csv_path, preprocessor)
    loader.process_all(force_structure=True, max_groups=2)
    export_path = normalize_path(os.path.join(BASE_PROC, "datasets.pkl"))
    loader.export(export_path)
