import os
import pickle
import pandas as pd
import numpy as np
from preprocessor import Preprocessor
import nibabel as nib
from PIL import Image
import random
import logging
from tqdm import tqdm

# Set up logging to record any errors
logging.basicConfig(
    filename="processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s"
)

#BASE_UNPROC = "/data/jrbonill/data/unprocessed/"
#BASE_PROC = "/data/jrbonill/data/preprocessed/"
BASE_UNPROC = "G:/Datasets/"
BASE_PROC = "G:/Preprocessed/"

def load_dataset_metadata(csv_path):
    """
    Load dataset metadata from a CSV file.
    Expected CSV columns:
      - "Dataset Name": The name of the dataset.
      - "Modality": The imaging modality (e.g., "ct", "mri").
      - "Image File Type": Allowed image file extensions (e.g., ".nii.gz,.dcm,.png").
      - "Mask File Type": Allowed mask file extensions.
      - "Root Folder": The root folder containing the dataset.
      - "Train Folders": (Optional) Comma-separated folder names designated as training.
      - "Test Folders": (Optional) Comma-separated folder names designated as testing.
      - "Mask Folders": (Optional) Comma-separated folder names containing mask files.
      - "Preprocessed?": Indicates if the dataset is already preprocessed ("yes" or "no").

    Args:
      csv_path: Path to the CSV file.

    Returns:
      A dictionary where the keys are dataset names and the values are dictionaries containing metadata for each dataset.
    """
    df = pd.read_csv(csv_path)
    metadata = {}
    for idx, row in df.iterrows():
        name = str(row["Dataset Name"]).strip()

        # Parse file type columns into lists, splitting on commas and stripping whitespace.
        image_types = (str(row["Image File Type"]).split(',') if pd.notna(row["Image File Type"]) else [])
        image_types = [ext.strip() for ext in image_types if ext.strip()]

        mask_types = (str(row["Mask File Type"]).split(',') if pd.notna(row["Mask File Type"]) else [])
        mask_types = [ext.strip() for ext in mask_types if ext.strip()]

        # Parse folder columns into lists, splitting on commas and stripping whitespace.
        root_folder = str(row["Root Folder"]).strip() if pd.notna(row["Root Folder"]) else ""
        
        train_folders = (str(row["Train Folders"]).split(',') if "Train Folders" in df.columns and pd.notna(row["Train Folders"]) else [])
        train_folders = [fld.strip() for fld in train_folders if fld.strip()]
        
        test_folders = (str(row["Test Folders"]).split(',') if "Test Folders" in df.columns and pd.notna(row["Test Folders"]) else [])
        test_folders = [fld.strip() for fld in test_folders if fld.strip()]

        mask_folders = (str(row["Mask Folders"]).split(',') if "Mask Folders" in df.columns and pd.notna(row["Mask Folders"]) else [])
        mask_folders = [fld.strip() for fld in mask_folders if fld.strip()]
        
        # Convert "Preprocessed?" column to boolean.
        preprocessed_str = str(row["Preprocessed?"]).strip().lower() if pd.notna(row["Preprocessed?"]) else None
        preprocessed_flag = preprocessed_str == "yes"

        # Parse mask suffix and image name columns.
        mask_key = (str(row["Mask Suffix"]).strip() if "Mask Suffix" in df.columns and pd.notna(row["Mask Suffix"]) else None)
        image_name = (str(row["Image Name"]).strip() if "Image Name" in df.columns and pd.notna(row["Image Name"]) else "")
        
        # Add dataset metadata to the dictionary.
        metadata[name] = {
            "modalities": [mod.strip().lower() for mod in str(row["Modality"]).split(',')] if pd.notna(row["Modality"]) else [],
            "image_file_types": image_types,
            "mask_file_types": mask_types,
            "train_folders": train_folders,
            "test_folders": test_folders,
            "mask_folders": mask_folders,
            "root_directory": root_folder,
            "mask_key": mask_key,
            "image_name": image_name,
            "preprocessed": preprocessed_flag
        }
    
    return metadata

def find_files_by_extension(root_folder, folder_list, allowed_extensions, is_mask=False, mask_key=None, exclude_folders=None, use_folder_identifier=False):
    """
    Find files in the specified folders (relative to root_folder) that have an allowed extension.
    
    If folder_list is provided, search only within those folders; otherwise, search recursively.
    
    For mask files (is_mask=True) with a mask_key provided, only files whose base name contains
    that suffix (anywhere) are returned, and that mask key is removed from the base name.
    For image files (is_mask=False) and if mask_key is provided, any file whose base name contains
    that suffix is skipped.
    
    Additionally, if exclude_folders is provided (a list of folder names), any file whose current directory's
    components include one of those names is skipped.
    
    If use_folder_identifier is True, the key used for pairing is the immediate parent folder's name.
    
    Args:
        root_folder (str): The root folder to start the search from.
        folder_list (list): A list of folder names to search within.
        allowed_extensions (list): A list of allowed file extensions.
        is_mask (bool): True if searching for mask files, False for image files.
        mask_key (str): The mask key that can appear anywhere in the filename.
        exclude_folders (list): A list of folder names to exclude from the search.
        use_folder_identifier (bool): True if the key should be the immediate parent folder's name.
    
    Returns:
        dict: A dictionary mapping file keys to their full paths.
    """
    found_files = {}
    if folder_list:
        search_folders = [os.path.join(root_folder, folder) for folder in folder_list if os.path.exists(os.path.join(root_folder, folder))]
    else:
        search_folders = [root_folder]
    if exclude_folders:
        exclude_folders = [x.lower() for x in exclude_folders]
    for folder in search_folders:
        for current_dir, _, filenames in os.walk(folder):
            if exclude_folders:
                path_components = [comp.lower() for comp in current_dir.split(os.sep)]
                if any(excluded in path_components for excluded in exclude_folders):
                    continue
            for fname in filenames:
                for ext in allowed_extensions:
                    ext_lower = ext.lower()
                    if ext_lower == ".nii.gz":
                        if not fname.lower().endswith(".nii.gz"):
                            continue
                        candidate = fname[:-len(".nii.gz")]
                    else:
                        if not fname.lower().endswith(ext_lower):
                            continue
                        candidate = os.path.splitext(fname)[0]
                    
                    if is_mask and mask_key:
                        if mask_key.lower() not in candidate.lower():
                            continue
                        # Remove all occurrences of mask_key from candidate.
                        base_name = candidate.lower().replace(mask_key.lower(), "")
                    elif (not is_mask) and mask_key:
                        if mask_key.lower() in candidate.lower():
                            continue
                        base_name = candidate
                    else:
                        base_name = candidate
                    if use_folder_identifier:
                        base_name = os.path.basename(current_dir)
                    if base_name not in found_files:
                        found_files[base_name] = os.path.join(current_dir, fname)
                    break  # Stop checking extensions once matched.
    return found_files

class UnifiedMedicalDataset:
    def __init__(self, dataset_name, metadata):
        """
        Represents a unified medical dataset.

        The unprocessed is assumed to reside at:
            /data/jrbonill/data/unprocessed/{dataset_name}/{root_folder}
        The preprocessed is will be written to:
            /data/jrbonill/data/preprocessed/{dataset_name}

        The CSV metadata may optionally designate train/test folders.
        If not provided, a random split (default 80% train, 20% test) is used.
        Image and mask files are paired using regex.

        Args:
            dataset_name (str): The name of the dataset.
            metadata (dict): A dictionary containing metadata for the dataset.
        """
        self.dataset_name = dataset_name
        self.metadata = metadata

        root_dir = self.metadata.get("root_folder")
        if root_dir:
            self.dataset_dir = os.path.join(BASE_UNPROC, dataset_name, root_dir)
        else:
            self.dataset_dir = os.path.join(BASE_UNPROC, dataset_name)
        self.output_dir = os.path.join(BASE_PROC, dataset_name)

        for split in ["train", "test"]:
            for category in ["images", "masks"]:
                os.makedirs(os.path.join(self.output_dir, split, category), exist_ok=True)

        # Dictionaries to hold preprocessed file paths
        self.preprocessed_images = {}
        self.preprocessed_masks = {}

    def process(self, preprocessor: Preprocessor, train_ratio=0.8, force_structure=True, max_pairs_per_split=2):
        """
        Process the dataset by pairing images and masks and assigning them to train/test splits.

        If train/test folders are provided in the metadata, those are used; otherwise, the entire dataset
        is searched recursively and a random split (default 80% train, 20% test) is performed.
        If only a training folder is specified, the images from that folder are randomly split.
        If an "Image Name" is provided in the CSV, the immediate parent folder name is used as the pairing key.
        
        Args:
            preprocessor (Preprocessor): An instance of the Preprocessor class.
            train_ratio (float, optional): The ratio of the dataset to assign to the train split. Defaults to 0.8.
            force_structure (bool, optional): Whether to force the structure of the preprocessed data. Defaults to True.
            max_pairs_per_split (int, optional): The maximum number of pairs to process. Defaults to 2.
        """
        # Set up a dedicated logger for this dataset:
        log_dir = os.path.join(self.output_dir, "preprocessing_logs")
        os.makedirs(log_dir, exist_ok=True)
        global logger
        logger = logging.getLogger(self.dataset_name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, f"{self.dataset_name}.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        if self.metadata.get("preprocessed", False) and not force_structure:
            logger.info(f"Dataset '{self.dataset_name}' already preprocessed. Skipping.")
            return
        logger.info(f"Processing dataset '{self.dataset_name}'...")
        
        if self.metadata.get("preprocessed", False) and not force_structure:
            print(f"Dataset '{self.dataset_name}' already preprocessed. Skipping.")
            return
        print(f"Processing dataset '{self.dataset_name}'...")

        # Process .h5 files and .xml/.txt annotations.
        self._process_h5_files(preprocessor)
        self._process_annotations(preprocessor)
        
        # Use lower-case keys from metadata.
        allowed_image_types = self.metadata.get("image_file_types", [])
        allowed_mask_types = self.metadata.get("mask_file_types", [])
        train_folders = self.metadata.get("train_folders")
        test_folders = self.metadata.get("test_folders")
        
        # If "Image Name" is provided, use the parent folder name as the identifier.
        use_folder = bool(self.metadata.get("image_name"))
        
        # Determine image files.
        if train_folders or test_folders:
            if train_folders and not test_folders:
                # Only training folders provided; perform random split on the images found.
                all_images = find_files_by_extension(self.dataset_dir, None, allowed_image_types,
                                                     exclude_folders=self.metadata.get("mask_folders"),
                                                     use_folder_identifier=use_folder)
                all_keys = list(all_images.keys())
                random.shuffle(all_keys)
                n_train = int(train_ratio * len(all_keys))
                train_ids = all_keys[:n_train]
                test_ids = all_keys[n_train:]
                train_images = {k: all_images[k] for k in train_ids}
                test_images = {k: all_images[k] for k in test_ids}
            else:
                exclude_list = self.metadata.get("mask_folders") if self.metadata.get("mask_folders") else None
                train_images = find_files_by_extension(self.dataset_dir, train_folders, allowed_image_types,
                                                       exclude_folders=exclude_list, use_folder_identifier=use_folder) if train_folders else {}
                test_images = find_files_by_extension(self.dataset_dir, test_folders, allowed_image_types,
                                                      exclude_folders=exclude_list, use_folder_identifier=use_folder) if test_folders else {}
        else:
            all_images = find_files_by_extension(self.dataset_dir, None, allowed_image_types, use_folder_identifier=use_folder)
            all_keys = list(all_images.keys())
            random.shuffle(all_keys)
            n_train = int(train_ratio * len(all_keys))
            train_ids = all_keys[:n_train]
            test_ids = all_keys[n_train:]
            train_images = {k: all_images[k] for k in train_ids}
            test_images = {k: all_images[k] for k in test_ids}
        
        # Determine mask files.
        if allowed_mask_types:
            mask_folders = self.metadata.get("mask_folders")
            if mask_folders:
                all_masks = find_files_by_extension(self.dataset_dir, mask_folders, allowed_mask_types)
            else:
                all_masks = find_files_by_extension(self.dataset_dir, None, allowed_mask_types)
        else:
            all_masks = {}
        train_masks = {k: all_masks[k] for k in train_images if k in all_masks}
        test_masks = {k: all_masks[k] for k in test_images if k in all_masks}
        
        train_count = self.process_image_mask_split(train_images, train_masks, "train", preprocessor, max_pairs_per_split)
        test_count = self.process_image_mask_split(test_images, test_masks, "test", preprocessor, max_pairs_per_split)
        total_pairs = train_count + test_count
        
        print(f"Processing of dataset '{self.dataset_name}' complete. {total_pairs} pair(s) processed. Outputs at: {self.output_dir}")

        # Update metadata with counts and flag.
        self.metadata["Num Images"] = len(train_images) + len(test_images)
        self.metadata["Num Masks"] = len(train_masks) + len(test_masks)
        self.metadata["Preprocessed?"] = "yes"

    def process_image_mask_split(self, image_dict, mask_dict, split_name, preprocessor, max_pairs=2):
        """
        Process image-mask pairs for a specific split and save preprocessed outputs.

        Args:
            image_dict (dict): A dictionary mapping image pair IDs to their full paths.
            mask_dict (dict): A dictionary mapping mask pair IDs to their full paths.
            split_name (str): The name of the split (train or test).
            preprocessor (Preprocessor): An instance of the Preprocessor class.
            max_pairs (int, optional): The maximum number of pairs to process. Defaults to 2.

        Returns:
        int: Number of pairs processed.
        """
        split_dir = os.path.join(self.output_dir, split_name)
        checkpoint_file = os.path.join(split_dir, "checkpoint.txt")
        next_index = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                try:
                    next_index = int(f.read().strip())
                except Exception:
                    next_index = 0

        # Sort the keys to ensure consistent processing order.
        keys = sorted(image_dict.keys())

        # Use existing output count to determine the next image file number.
        current_pair_number = len(os.listdir(os.path.join(split_dir, "images"))) + 1
        count = 0

        for idx in range(next_index, len(keys)):
            pair_id = keys[idx]
            processed_image = preprocessor.process_file(
                image_dict[pair_id],
                self.metadata.get("modalities")[0] if self.metadata.get("modalities") else None,
                is_mask=False
            )
            if processed_image is None:
                continue

            count += 1
            output_image_path = os.path.join(split_dir, "images", f"{current_pair_number}.png")
            try:
                if isinstance(processed_image, nib.Nifti1Image):
                    output_image_path = output_image_path.replace(".png", ".nii.gz")
                    nib.save(processed_image, output_image_path)
                else:
                    processed_image.save(output_image_path)
                logger.info(f"Saved {split_name} image to {output_image_path}")
            except Exception as e:
                logging.exception(f"Error saving {split_name} image to {output_image_path}: {e}")

            if mask_dict and pair_id in mask_dict:
                processed_masks = preprocessor.process_file(
                    mask_dict[pair_id],
                    self.metadata.get("modalities")[0] if self.metadata.get("modalities") else None,
                    is_mask=True
                )
                if processed_masks:
                    for i, mask_img in enumerate(processed_masks):
                        out_mask_path = os.path.join(split_dir, "masks", f"{current_pair_number}_{i}.png")
                        try:
                            mask_img.save(out_mask_path)
                            logger.info(f"Saved {split_name} mask to {out_mask_path}")
                        except Exception as e:
                            logging.exception(f"Error saving {split_name} mask to {out_mask_path}: {e}")

            current_pair_number += 1
            # Write checkpoint (next index to process)
            with open(checkpoint_file, "w") as f:
                f.write(str(idx + 1))

            if count >= max_pairs:
                break

        return count

    def _process_h5_files(self, preprocessor):
        """
        Process .h5 files in the dataset directory.
        Extract image and label data and save them (e.g., as NIfTI for images and PNG for masks).
        Update the converted files dictionaries and allowed file types in metadata.

        Args:
            preprocessor (Preprocessor): An instance of the Preprocessor class.
        """
        image_types_lower = [ext.lower() for ext in self.metadata.get("image_file_types", [])]
        if ".h5" not in image_types_lower:
            return
        h5_images = find_files_by_extension(self.dataset_dir, None, [".h5"])
        for key, file_path in tqdm(h5_images.items(), desc=f"[{self.dataset_name}] Processing .h5 files"):
            print(f"[{self.dataset_name}] Converting .h5 file: {file_path}")
            image_data, label_data = preprocessor.extract_h5_content(file_path)
            if image_data is not None:
                new_image_path = os.path.splitext(file_path)[0] + "_extracted.nii.gz"
                try:
                    img_nifti = nib.Nifti1Image(image_data, affine=np.eye(4))
                    nib.save(img_nifti, new_image_path)
                    self.converted_image_files[key] = new_image_path
                except Exception as e:
                    logging.exception(f"Error saving extracted image for {file_path}: {e}")
            if label_data is not None:
                new_mask_path = os.path.splitext(file_path)[0] + "_mask.png"
                try:
                    mask_img = Image.fromarray((label_data.astype('uint8') * 255))
                    mask_img.save(new_mask_path)
                    self.converted_mask_files[key] = new_mask_path
                except Exception as e:
                    logging.exception(f"Error saving extracted label for {file_path}: {e}")
        # Update allowed image types: remove .h5 and add .nii.gz.
        new_img_types = [ext for ext in self.metadata.get("image_file_types", []) if ext.lower() != ".h5"]
        if ".nii.gz" not in [t.lower() for t in new_img_types]:
            new_img_types.append(".nii.gz")
        self.metadata["image_file_types"] = new_img_types
        # Similarly update mask types if .h5 was used.
        new_mask_types = [ext for ext in self.metadata.get("mask_file_types", []) if ext.lower() != ".h5"]
        if ".png" not in [t.lower() for t in new_mask_types]:
            new_mask_types.append(".png")
        self.metadata["mask_file_types"] = new_mask_types

    def _process_annotations(self, preprocessor):
        """
        Process annotation files (.xml and .txt) in the dataset directory.
        Convert each annotation file to a binary mask (saved as PNG) and update the converted mask dictionary.
        Update allowed mask types in metadata.

        Args:
            preprocessor (Preprocessor): An instance of the Preprocessor class.
        """
        mask_types_lower = [ext.lower() for ext in self.metadata.get("mask_file_types", [])]
        annotation_exts = [ext for ext in mask_types_lower if ext in [".xml", ".txt"]]
        if not annotation_exts:
            return
        ann_masks = {}
        for ext in annotation_exts:
            ann_masks.update(find_files_by_extension(self.dataset_dir, None, [ext]))
        for key, file_path in tqdm(ann_masks.items(), desc=f"[{self.dataset_name}] Processing annotation files"):
            print(f"[{self.dataset_name}] Converting annotation ({os.path.splitext(file_path)[1].lower()}) to mask: {file_path}")
            new_mask_path = os.path.splitext(file_path)[0] + "_mask.png"
            mask_array = preprocessor.convert_annotation_to_mask(file_path, preprocessor.target_size)
            if mask_array is not None:
                try:
                    mask_img = Image.fromarray((mask_array * 255).astype("uint8"))
                    mask_img.save(new_mask_path)
                    self.converted_mask_files[key] = new_mask_path
                except Exception as e:
                    logging.exception(f"Error saving converted mask for {file_path}: {e}")
        new_mask_types = [ext for ext in self.metadata.get("mask_file_types", []) if ext.lower() not in [".xml", ".txt"]]
        if ".png" not in [t.lower() for t in new_mask_types]:
            new_mask_types.append(".png")
        self.metadata["mask_file_types"] = new_mask_types

class DatasetLoader:
    def __init__(self, csv_path, preprocessor: Preprocessor):
        """
        Load dataset metadata from a CSV file and create a list of UnifiedMedicalDataset objects.

        Args:
            csv_path (str): The path to the CSV file containing dataset metadata.
            preprocessor (Preprocessor): An instance of the Preprocessor class.
        """
        self.csv_path = csv_path
        self.metadata = load_dataset_metadata(csv_path)
        self.datasets = []
        for name, meta in self.metadata.items():
            self.datasets.append(UnifiedMedicalDataset(name, meta))
        self.preprocessor = preprocessor

    def process_all(self, force_structure=True, max_pairs_per_split=2):
        """
        Process all datasets in the list of UnifiedMedicalDataset objects.
        Saves the new CSV data to the base preprocessed directory.

        Args:
            force_structure (bool, optional): Whether to force the structure of the preprocessed data. Defaults to True.
        """
        for dataset in tqdm(self.datasets, desc="Processing all datasets"):
            dataset.process(self.preprocessor, force_structure=force_structure, max_pairs_per_split=max_pairs_per_split)
        self.update_csv()

    def update_csv(self):
        """
        Update the CSV file with the updated dataset metadata.
        """
        updated_data = []
        for dataset in self.datasets:
            meta = dataset.metadata.copy()
            meta["Dataset Name"] = dataset.dataset_name
            updated_data.append(meta)
        df = pd.DataFrame(updated_data)
        new_csv_path = os.path.join(BASE_PROC, "datasets.csv")
        df.to_csv(new_csv_path, index=False)

    def export(self, export_path):
        """
        Export the DatasetLoader object to a file.

        Args:
            export_path (str): The path to save the exported file.
        """
        with open(export_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset loader exported to {export_path}.")

    @classmethod
    def load_from_file(cls, export_path):
        """
        Load a DatasetLoader object from a file.

        Args:
            export_path (str): The path to the exported file.

        Returns:
            DatasetLoader: The loaded DatasetLoader object.
        """
        with open(export_path, 'rb') as f:
            loader = pickle.load(f)
        print(f"Dataset loader loaded from {export_path}.")
        return loader

if __name__ == "__main__":
    #csv_path = "/data/jrbonill/data/datasets.csv"
    csv_path = BASE_UNPROC + "/datasets.csv"
    preprocessor = Preprocessor(target_size=(1024, 1024))
    loader = DatasetLoader(csv_path, preprocessor)

    # For testing: process only a couple of pairs per split (deault 2 per split).
    loader.process_all(force_structure=False, max_pairs_per_split=1)
    export_path = BASE_PROC + "/datasets.pkl"
    loader.export(export_path)