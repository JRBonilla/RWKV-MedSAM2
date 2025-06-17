import os
import numpy as np
import cv2
from scipy.ndimage import label
import nibabel as nib
from PIL import Image
import pydicom
import logging
from helpers import get_extension
import hashlib
import scipy.io
import datetime
import SimpleITK as sitk
import gzip
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from logging.handlers import RotatingFileHandler
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# Configuration
# PREPROCESSING_LOG_DIR = "F:/PreprocessingLogs"
PREPROCESSING_LOG_DIR = "/data/PreprocessingLogs"

# Configure logger
main_logger = logging.getLogger("Preprocessor")
main_logger.setLevel(logging.INFO)
if not main_logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    main_logger.addHandler(ch)

    # File handler: logs save to "preprocessor_<date>.log"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(PREPROCESSING_LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(PREPROCESSING_LOG_DIR, f"preprocessor_{timestamp}.log")
    fh = RotatingFileHandler(log_file_path, maxBytes=400 * 1024 * 1024, backupCount=10)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    main_logger.addHandler(fh)

class Preprocessor:
    """
    Preprocessor for medical images and masks following the Medical-SAM2 pipeline:
        - CT: Windowing + global z-score normalization
        - MRI/X-ray/Ultrasound/etc.: Percentile clipping + per-image (or central region) z-score
        - Crop to non-zero region (flag if >=25% removed)
        - Resize in-plane to target size
        - Copy channels to 3 for grayscale
        - Split multi-class masks, connected components, and filter small regions
    """
    def __init__(self, target_size=(1024, 1024), ct_window=None, global_ct_stats=None, min_mask_size=100, dataset_logger=None, dataset_name=None, background_value=0):
        """
        Initialize the Preprocessor.

        Args:
            target_size (tuple, optional): (width, height) for in-plane resizing. Defaults to (1024, 1024).
            ct_window (tuple, optional): (window_width, window_level) for CT intensity windowing. Defaults to None.
            global_ct_stats (tuple, optional): (mean, std) for CT z-score across the dataset. Defaults to None.
            min_mask_size (int, optional): Minimum pixel area for keeping a mask component. Defaults to 100.
            dataset_logger (logging.Logger, optional): Logger for dataset-specific logging. Defaults to main logger.
            dataset_name (str, optional): Name of the dataset. Defaults to None.
            background_value (int, optional): Background value for normalization. Defaults to 0.
        """
        main_logger.info(f"Initializing Preprocessor: target_size={target_size}, ct_window={ct_window}, global_ct_stats={global_ct_stats}, min_mask_size={min_mask_size}")
        # Core configuration
        self.target_size = target_size
        self.ct_window = ct_window
        self.global_ct_mean, self.global_ct_std = global_ct_stats or (None, None)
        self.min_mask_size = min_mask_size
        self.dataset_name = dataset_name

        # File-type constants
        self.volume_exts = {".dcm", ".dicom", ".nii", ".nii.gz", ".nrrd", ".mhd", ".npy"}
        self.video_exts = {".avi", ".mp4", ".mov"}

        # Modality categories (lowercase)
        self.modalities_3d = {"ct", "mri"}
        self.modalities_2d = {
            "x-ray", "ultrasound", "retinal-scan", "colonoscopy",
            "fetoscopy", "histopathology", "dermoscopy", "mammography", "oct"
        }

        # Background value used for cropping and normalization decisions
        self.background_value = background_value

        self.main_logger = main_logger
        self.dataset_logger = dataset_logger or main_logger
        self.dataset_logger.info("Preprocessor initialized successfully.")

    def preprocess_group(self, sub_name, pipeline, image_files, mask_files, modality, img_out_dir, mask_out_dir, composite_id):
        """
        Dispatch a group of images and masks through the specified preprocessing pipeline.

        Based on the pipeline argument ('2D', '3D', or 'Video'), this method:
        1. Computes a unified bounding box over all input images.
        2. Calls the corresponding preprocessing routine.
        3. Adds an is_significant_crop flag if >=25 percent of the volume or area was removed.
        4. Logs removed masks (if any) and returns pipeline-specific metadata.

        Args:
            sub_name (str): Identifier for the subdataset.
            pipeline (str): One of '2D', '3D', or 'Video' specifying which workflow to use.
            image_files (list of str): Paths to input image files or series.
            mask_files (list of str): Paths to corresponding mask files or series.
            modality (str): Modality key for normalization, for example 'ct', 'mri', or 'ultrasound'.
            img_out_dir (str): Directory to save preprocessed images.
            mask_out_dir (str): Directory to save preprocessed masks.
            composite_id (str): Unique identifier used for naming output files.

        Returns:
            dict: Metadata including:
                - 'modality' (str): modality key used for normalization
                - 'resize_shape' (tuple): target in-plane size
                - 'is_significant_crop' (bool): True if >=25 percent was cropped
                plus pipeline-specific fields such as 'fps' for video or 'image_niftis' and 'mask_niftis' for 3D
        """
        main_logger.info(f"Preprocessing group: {len(image_files)} images, {len(mask_files)} masks, modality={modality}")
        
        # Normalize modality key for lookup
        self.dataset_logger.info(f"DEBUG: modality is {modality!r} of type {type(modality)}")
        key = modality.lower()
        modality_key = key if key in self.modalities_3d.union(self.modalities_2d) else "default"

        # Compute unified bounding box over all images
        group_bbox = self._compute_group_bbox(image_files, pipeline)
        main_logger.info(f"Group bounding box: {group_bbox}")
        
        # Preprocess
        try:
            if pipeline == "Video":
                main_logger.info(f"Using video pipeline for group: {len(image_files)} images, {len(mask_files)} masks, modality={modality}")
                metadata = self.preprocess_video(
                    sub_name, image_files, mask_files, modality_key, img_out_dir, mask_out_dir, composite_id, group_bbox
                )
            elif pipeline == "3D":
                main_logger.info(f"Using 3D pipeline for group: {len(image_files)} images, {len(mask_files)} masks, modality={modality}")
                metadata = self.preprocess_3d(
                    sub_name, image_files, mask_files, modality_key, img_out_dir, mask_out_dir, composite_id, group_bbox
                )
            elif pipeline == "2D":
                main_logger.info(f"Using 2D pipeline for group: {len(image_files)} images, {len(mask_files)} masks, modality={modality}")
                metadata = self.preprocess_2d(
                    sub_name, image_files, mask_files, modality_key, img_out_dir, mask_out_dir, composite_id, group_bbox
                )
            else:
                main_logger.warning(f"Pipeline either unrecognized or not implemented: {pipeline} with type {type(pipeline)}")
                main_logger.warning("Defaulting to 2D preprocessing.")
                metadata = self.preprocess_2d(
                    sub_name, image_files, mask_files, modality_key, img_out_dir, mask_out_dir, composite_id, group_bbox
                )
        except Exception as e:
            main_logger.error(f"Error in preprocess_group: {e}")
            return None

        # Add is_significant_crop and slices to metadata
        metadata["is_significant_crop"] = bool(self.is_significant_crop)

        # Log and return
        main_logger.info(f"Completed preprocessing group for modality {modality}")
        
        removed_masks = len(mask_files) - len(os.listdir(mask_out_dir))
        if removed_masks > 0:
            main_logger.info(f"Removed {removed_masks} masks due to no significant components found")
        return metadata
    
    def preprocess_2d(self, sub_name, image_files, mask_files, modality, img_out_dir, mask_out_dir, composite_id, group_bbox):
        """
        Preprocess a batch of 2D masks and images using a shared bounding box.

        1. Masks:
        - Load mask with load_mask
        - Crop to group_bbox
        - Resize to target_size
        - Split multiclass masks into binary masks
        - Split connected components and filter by min_mask_size
        - Record valid stems for PAIP2019 or e_optha
        - Save each component

        2. Images:
        - Load image with load_image
        - Skip images without valid masks for PAIP2019 or e_optha
        - Crop to group_bbox
        - Normalize intensity (percentile-clip + z-score, use central region if significant crop)
        - Resize to target_size
        - Ensure three channels
        - Save each slice or image

        Args:
            sub_name (str): Identifier for the subdataset.
            image_files (list of str): Paths to the 2D image files.
            mask_files (list of str): Paths to the mask files.
            modality (str): Lowercase modality key for normalization, e.g. 'ct', 'mri', 'ultrasound'.
            img_out_dir (str): Directory to save preprocessed images.
            mask_out_dir (str): Directory to save preprocessed masks.
            composite_id (str): Unique identifier for naming output files.
            group_bbox (tuple): (min_row, max_row, min_col, max_col) bounding box.

        Returns:
            dict: Metadata containing:
                - 'modality' (str): modality key used for normalization
                - 'resize_shape' (tuple): target in-plane size
        """
        main_logger.info(
            f"Starting 2D batch preprocessing: {len(image_files)} images, {len(mask_files)} masks, modality={modality}"
        )

        # For PAIP2019: collect mask stems that passed the size filter
        valid_mask_stems = set()

        # 1) Mask loop
        for i, msk_path in enumerate(tqdm(mask_files, desc=f"[{composite_id}] Processing 2D masks...")):
            self.dataset_logger.info(f"Loading mask {msk_path}")
            mask_arr, header = self.load_mask(msk_path)

            # Handle 3D mask volumes
            if isinstance(mask_arr, np.ndarray) and mask_arr.ndim == 3 and isinstance(header, sitk.Image):
                vol = mask_arr
                n_slices = vol.shape[0]
                for z in range(n_slices):
                    slice_mask = vol[z]
                    cropped_mask = self._crop(slice_mask, group_bbox)
                    resized_mask = self._resize(cropped_mask, is_mask=True)

                    comps = []
                    for binary in self._split_multiclass_mask(resized_mask):
                        for comp in self._split_connected_components(binary):
                            comps.append(comp)
                    filtered = self._filter_small_components(comps)

                    if not filtered:
                        self.dataset_logger.warning(
                            f"No significant components in slice {z} of mask {msk_path}"
                        )
                        continue

                    for j, comp in enumerate(filtered):
                        self._save_mask(comp, '2d', z, j, f"{composite_id}_slice{z}", mask_out_dir)
                        del comp
                    del filtered, comps, resized_mask, cropped_mask
                del vol

            # Truly 2D mask case
            else:
                cropped_mask = self._crop(mask_arr, group_bbox)
                resized_mask = self._resize(cropped_mask, is_mask=True)

                comps = []
                for binary in self._split_multiclass_mask(resized_mask):
                    for comp in self._split_connected_components(binary):
                        comps.append(comp)
                filtered = self._filter_small_components(comps)

                if not filtered:
                    self.dataset_logger.warning(f"No significant components found in mask {msk_path}")
                    continue

                # Record stems only for PAIP2019
                if self.dataset_name == 'PAIP2019':
                    valid_mask_stems.add(Path(msk_path).stem)
                elif self.dataset_name == 'e_optha':
                    if sub_name == 'EX':
                        valid_mask_stems.add(Path(msk_path).stem.split('_')[0])
                    elif sub_name == 'MA':
                        valid_mask_stems.add(Path(msk_path).stem)

                for j, comp in enumerate(filtered):
                    self._save_mask(comp, '2d', i, j, composite_id, mask_out_dir)
                    del comp
                del filtered, comps, resized_mask, cropped_mask, mask_arr

        # 2) Image loop
        for i, img_path in enumerate(tqdm(image_files, desc=f"[{composite_id}] Processing 2D images...")):
            # For PAIP2019, skip images without valid masks
            if self.dataset_name == 'PAIP2019' or self.dataset_name == 'e_optha':
                stem = Path(img_path).stem
                if stem not in valid_mask_stems:
                    continue

            self.dataset_logger.info(f"Loading image {img_path}")
            image_arr, header = self.load_image(img_path)

            # Handle 3D image volumes
            if isinstance(image_arr, np.ndarray) and image_arr.ndim == 3 and isinstance(header, sitk.Image):
                vol = image_arr
                n_slices = vol.shape[0]
                for z in range(n_slices):
                    slice_img = vol[z]
                    cropped_img = self._crop(slice_img, group_bbox)
                    norm_img = self._normalize_intensity(
                        cropped_img, modality, use_central_region=self.is_significant_crop
                    )
                    resized_img = self._resize(norm_img, is_mask=False)
                    final_img = self._ensure_three_channels(resized_img)

                    self._save_image(final_img, '2d', z, f"{composite_id}_slice{z}", img_out_dir)
                    del final_img, resized_img, norm_img, cropped_img
                del vol

            # Truly 2D image case
            else:
                cropped_img = self._crop(image_arr, group_bbox)
                norm_img = self._normalize_intensity(
                    cropped_img, modality, use_central_region=self.is_significant_crop
                )
                resized_img = self._resize(norm_img, is_mask=False)
                final_img = self._ensure_three_channels(resized_img)

                self._save_image(final_img, '2d', i, composite_id, img_out_dir)
                del final_img, resized_img, norm_img, cropped_img, image_arr

        metadata = {"modality": modality, "resize_shape": self.target_size}

        main_logger.info(
            f"Completed 2D batch preprocessing: {len(image_files)} images, {len(mask_files)} masks"
        )
        return metadata

    def preprocess_3d(self, sub_name, image_series, mask_series, modality, img_out_dir, mask_out_dir, composite_id, group_bbox):
        """
        Preprocess a batch of 3D volumes and their masks using a shared bounding box.

        1. Load image volume(s) and mask volume(s):
        - Support single-file volumes, DICOM series, multi-modality inputs, CHAOS dataset labels, and DICOM-SEG.
        2. Resample all mask volumes to the space of the first image modality.
        3. Crop each modality volume to group_bbox, normalize intensities per slice, resize to target_size, and save as NIfTI.
        4. Crop each resampled mask volume to group_bbox, resize slices, split multi-class or connected components, filter small regions, and save each mask as NIfTI.

        Args:
            sub_name (str): Identifier for the subdataset.
            image_series (str or list of str): Path(s) to the 3D image series.
            mask_series (str or list of str): Path(s) to the corresponding mask series.
            modality (str): Lowercase modality key for normalization, e.g. 'ct' or 'mri'.
            img_out_dir (str): Directory to save preprocessed image NIfTIs.
            mask_out_dir (str): Directory to save preprocessed mask NIfTIs.
            composite_id (str): Unique identifier used for naming output files.
            group_bbox (tuple): (z0, z1, y0, y1, x0, x1) bounding box coordinates.

        Returns:
            dict: Metadata including:
                - 'modality' (str): modality key used for normalization
                - 'resize_shape' (tuple): target in-plane size
                - 'volume_shape' (tuple): shape of the original volume
                - 'image_niftis' (list of str): paths to saved image NIfTI files
                - 'mask_niftis' (list of str): paths to saved mask NIfTI files
                - 'bbox' (tuple): bounding box used for cropping
        """
        self.main_logger.info(
            f"Starting 3D batch preprocessing: {len(image_series)} image slices, "
            f"{len(mask_series) if mask_series else 0} mask files, modality={modality}"
        )

        # Initialize lists for final outputs
        image_paths = []
        mask_paths = []

        # 1) Load image volume and metadata (SimpleITK image)
        img_vol, img_itk = self.load_image(image_series)

        # Prepare lists to hold one or more image ITK volumes and arrays
        image_itk_list = []
        image_array_list = []

        # Prepare a list to hold one or more mask ITK volumes
        mask_itk_list = []

        # Multi-modality case (img_vol is a list of numpy volumes, img_itk is a list of ITK images)
        if isinstance(img_vol, list) and all(isinstance(v, np.ndarray) for v in img_vol):
            # Load mask as a single SimpleITK volume, matching the first modality's geometry
            msk_vol, msk_itk = self.load_mask(mask_series)
            if msk_vol.ndim == 2:
                msk_vol = msk_vol[np.newaxis, ...]
                msk_itk = sitk.GetImageFromArray(msk_vol)
                # msk_itk.CopyInformation(img_itk[0])
            else:
                # msk_itk.CopyInformation(img_itk[0])
                pass
            mask_itk_list.append(msk_itk)

            # Build image_itk_list and image_array_list from each modality
            for m_idx, mod_numpy in enumerate(img_vol):
                mod_itk = sitk.GetImageFromArray(mod_numpy)
                mod_itk.CopyInformation(img_itk[m_idx])
                image_itk_list.append(mod_itk)
                image_array_list.append(mod_numpy)

        # CHAOS dataset: separate each label into its own binary volume
        elif isinstance(mask_series, (list, tuple)) and len(mask_series) > 1 and getattr(self, 'dataset_name', None) == 'CHAOS':
            mask_slices = []
            for z_idx, mpath in enumerate(mask_series):
                mask_arr, _ = self.load_mask(mpath)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[z_idx]
                if modality.lower() == 'ct':
                    label_map = (mask_arr > 0).astype(np.int32)
                else:
                    label_map = np.zeros_like(mask_arr, dtype=np.int32)
                    label_map[np.where(mask_arr == 63)]  = 1
                    label_map[np.where(mask_arr == 126)] = 2
                    label_map[np.where(mask_arr == 189)] = 3
                    label_map[np.where(mask_arr == 252)] = 4
                mask_slices.append(label_map)

            chaos_vol = np.stack(mask_slices, axis=0)
            unique_labels = np.unique(chaos_vol)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                binary_vol = (chaos_vol == lbl).astype(np.uint8)
                lbl_itk = sitk.GetImageFromArray(binary_vol)
                lbl_itk.CopyInformation(img_itk)
                mask_itk_list.append(lbl_itk)

            image_itk_list.append(img_itk)
            image_array_list.append(sitk.GetArrayFromImage(img_itk))

        # Generic DICOM-series masks or single 3D file
        else:
            exts = [get_extension(p).lower() for p in mask_series]
            if all(ext in {'.dcm', '.dicom'} for ext in exts):
                sop = pydicom.dcmread(mask_series[0], stop_before_pixels=True).SOPClassUID
                if sop == "1.2.840.10008.5.1.4.1.1.66.4":   # DICOM-SEG
                    # 1) Build a mapping from CT SOPInstanceUID to slice index using Z-position
                    ct_instances = []
                    for img_path in image_series:
                        ds_ct = pydicom.dcmread(img_path, stop_before_pixels=True)
                        # Use ImagePositionPatient[2] for true physical slice order
                        z_pos = float(ds_ct.ImagePositionPatient[2])
                        ct_instances.append((z_pos, ds_ct.SOPInstanceUID))
            
                    # sort by ascending Z
                    ct_instances.sort(key=lambda x: x[0])
                    ct_uids = [uid for _, uid in ct_instances]

                    # 2) Create a full-zero volume matching CT depth/shape
                    depth = len(ct_uids)
                    height = img_itk.GetHeight()
                    width  = img_itk.GetWidth()
                    full_vol = np.zeros((depth, height, width), dtype=np.uint8)

                    # 3) Insert each SEG slice into its proper z-location
                    for mp in mask_series:
                        ds_seg = pydicom.dcmread(mp, force=True)
                        ref_uid = ds_seg.PerFrameFunctionalGroupsSequence[0] \
                                         .DerivationImageSequence[0] \
                                         .SourceImageSequence[0] \
                                         .ReferencedSOPInstanceUID
                        z_idx = ct_uids.index(ref_uid)
                        seg_itk = sitk.ReadImage(mp) # read as 2D mask
                        seg_arr = sitk.GetArrayFromImage(seg_itk)
                        if seg_arr.ndim == 3 and seg_arr.shape[0] == 1:
                            seg_arr = seg_arr[0]
                        full_vol[z_idx] = seg_arr.astype(np.uint8)

                    # 4) Wrap back into ITK and copy CT metadata safely 
                    msk_itk = sitk.GetImageFromArray(full_vol)
                    msk_itk.CopyInformation(img_itk)
                else:
                    # Fallback for non-SEG DICOM masks
                    msk_vol, msk_itk = self.load_mask(mask_series)
                    if msk_vol.ndim == 2:
                        msk_vol = msk_vol[np.newaxis, ...]
                        msk_itk = sitk.GetImageFromArray(msk_vol)
                    msk_itk.CopyInformation(img_itk)
            else:
                msk_vol, msk_itk = self.load_mask(mask_series)
                if msk_vol.ndim == 2:
                    msk_vol = msk_vol[np.newaxis, ...]
                    msk_itk = sitk.GetImageFromArray(msk_vol)
                # msk_itk.CopyInformation(img_itk)
            mask_itk_list.append(msk_itk)

            image_itk_list.append(img_itk)
            image_array_list.append(sitk.GetArrayFromImage(img_itk))

        # 2) Resample each mask ITK to match the first image modality’s space
        resampled_mask_itk_list = []
        reference_itk = image_itk_list[0]
        for msk_itk in mask_itk_list:
            resampled = sitk.Resample(msk_itk, reference_itk)
            resampled_mask_itk_list.append(resampled)

        # 3) For each modality, crop + resize + save image volume
        z0, z1, y0, y1, x0, x1 = group_bbox
        orig_sp = reference_itk.GetSpacing()
        orig_org = reference_itk.GetOrigin()
        orig_dir = reference_itk.GetDirection()

        for m_idx, (mod_itk, mod_array) in enumerate(zip(image_itk_list, image_array_list)):
            img_crop = mod_array[z0:z1, y0:y1, x0:x1]

            processed_img_slices = []
            for idx in range(img_crop.shape[0]):
                slice_img = img_crop[idx]
                norm_img = self._normalize_intensity(
                    slice_img, modality,
                    use_central_region=self.is_significant_crop
                )
                resized_img = self._resize(norm_img, is_mask=False)
                processed_img_slices.append(resized_img)

            img_processed_vol = np.stack(processed_img_slices, axis=0)

            image_nifti = sitk.GetImageFromArray(img_processed_vol)
            new_spacing_img = (
                orig_sp[0] * (x1 - x0) / self.target_size[0],
                orig_sp[1] * (y1 - y0) / self.target_size[1],
                orig_sp[2] * (z1 - z0) / img_processed_vol.shape[0],
            )
            new_origin_img = (
                orig_org[0] + x0 * orig_sp[0],
                orig_org[1] + y0 * orig_sp[1],
                orig_org[2] + z0 * orig_sp[2],
            )
            image_nifti.SetSpacing(new_spacing_img)
            image_nifti.SetOrigin(new_origin_img)
            image_nifti.SetDirection(orig_dir)

            img_id = f"{composite_id}_modality{m_idx}"
            img_path = self._save_image(image_nifti, '3d', m_idx, img_id, img_out_dir)
            image_paths.append(img_path)

        # 4) Process each resampled mask: crop, resize, split components or labels, save
        for idx, msk_itk_res in enumerate(tqdm(resampled_mask_itk_list, desc=f"[{composite_id}] Processing 3D mask slices")):
            # (a) Convert to numpy and crop
            mask_array = sitk.GetArrayFromImage(msk_itk_res).astype(np.int32)
            mask_crop = mask_array[z0:z1, y0:y1, x0:x1]

            # (b) Resize each slice (nearest-neighbor for masks)
            processed_mask_slices = []
            for sidx in range(mask_crop.shape[0]):
                slice_mask = mask_crop[sidx]
                resized_mask = self._resize(slice_mask, is_mask=True)
                processed_mask_slices.append(resized_mask)

            # Keep it as int32 so that labels >1 survive resizing
            mask_processed_vol = np.stack(processed_mask_slices, axis=0).astype(np.int32)

            # (c) Build a SimpleITK image from the resized mask (either uint8 or int32 may be used;
            # casting to uint8 here is necessary for ConnectedComponent to work-will investigate later,
            # while mask_processed_vol remains int32 for label checking):
            mask_nifti_raw = sitk.GetImageFromArray(mask_processed_vol.astype(np.uint8))
            mask_nifti_raw.SetSpacing(new_spacing_img)
            mask_nifti_raw.SetOrigin(new_origin_img)
            mask_nifti_raw.SetDirection(orig_dir)

            # (d) Check which labels survived resizing
            unique_labels = np.unique(mask_processed_vol)

            # If it's NOT exactly {0,1}, treat every lbl>0 as its own volume, preserving the integer label:
            if not (len(unique_labels) == 2 and set(unique_labels) == {0, 1}):
                self.dataset_logger.info(f"Multi-class mask detected in mask index {idx}")
                self.dataset_logger.info(f"Splitting labels...")
                for lbl in unique_labels:
                    if lbl == 0:
                        continue
                    # Preserve the original label value (e.g. 2, 5, 63, etc):
                    label_vol = (mask_processed_vol == lbl).astype(np.int32) * int(lbl)
                    lbl_itk = sitk.GetImageFromArray(label_vol)
                    lbl_itk.CopyInformation(mask_nifti_raw)

                    mask_id = f"{composite_id}_mask{idx}_label{lbl}"
                    mask_path = self._save_mask(
                        lbl_itk, '3d', idx, lbl, mask_id, mask_out_dir
                    )
                    mask_paths.append(mask_path)

            else:
                # Exactly {0,1} -> pure binary (or empty).
                # (1) If it’s entirely background, skip.
                if np.array_equal(unique_labels, [0]):
                    self.dataset_logger.warning(f"No foreground found in mask index {idx}, skipping.")
                    continue

                self.dataset_logger.info(f"Binary mask detected in mask index {idx}")
                self.dataset_logger.info(f"Splitting connected components...")

                # (2) Otherwise do Connected Component on the binary mask
                binary_itk = sitk.Cast(
                    sitk.GetImageFromArray((mask_processed_vol != self.background_value).astype(np.uint8)),
                    sitk.sitkUInt8
                )
                binary_itk.CopyInformation(mask_nifti_raw)

                cc_filter = sitk.ConnectedComponentImageFilter()
                cc_filter.FullyConnectedOn()
                labeled_cc = cc_filter.Execute(binary_itk)

                relabel = sitk.RelabelComponentImageFilter()
                relabel.SortByObjectSizeOn()
                relabeled_cc = relabel.Execute(labeled_cc)

                comp_stats = sitk.LabelShapeStatisticsImageFilter()
                comp_stats.Execute(relabeled_cc)

                num_comps = relabel.GetNumberOfObjects()
                for comp_idx in range(1, num_comps + 1):
                    comp_binary = sitk.Equal(relabeled_cc, comp_idx)
                    voxel_count = comp_stats.GetNumberOfPixels(comp_idx)
                    if voxel_count < self.min_mask_size:
                        continue

                    comp_labeled = sitk.Cast(comp_binary, sitk.sitkInt32)
                    comp_labeled = sitk.Multiply(comp_labeled, comp_idx)
                    comp_labeled.CopyInformation(mask_nifti_raw)

                    mask_id = f"{composite_id}_mask{idx}_comp{comp_idx}"
                    mask_path = self._save_mask(
                        comp_labeled, '3d', idx, comp_idx, mask_id, mask_out_dir
                    )
                    mask_paths.append(mask_path)

        # Unified return with metadata
        return {
            "modality":     modality,
            "resize_shape": self.target_size,
            "volume_shape": image_array_list[0].shape if isinstance(image_array_list[0], np.ndarray) else None,
            "image_niftis": image_paths,
            "mask_niftis":  mask_paths,
            "bbox":         group_bbox,
        }

    def preprocess_video(self, sub_name, video_paths, mask_paths, modality, img_out_dir, mask_out_dir, composite_id, group_bbox):
        """
        Preprocess one or more video files and their corresponding mask videos (if any)
        as a batch, applying a shared crop bounding box, and save only frames with valid masks.

        Steps:
        1. Load frames (+ fps) from the first video in 'video_paths'.
        2. Load mask frames from the first mask in 'mask_paths' (optional).
        3. If mask size differs from video frames, resize masks to match.
        4. For each frame index:
            a) Process mask (if provided):
                i.  Crop using 'group_bbox'
                ii. Resize ('is_mask=True')
                iii. Split multi-class ('_split_multiclass_mask'),
                    split connected components ('_split_connected_components'),
                    filter small ('_filter_small_components')
                iv. If no components remain, skip this frame entirely
                v.  Save each mask component immediately
            b) If mask is valid (or no masks provided), process frame:
                i.  Crop using 'group_bbox'
                ii. Normalize intensities ('_normalize_intensity')
                iii.Resize to 'self.target_size'
                iv. Ensure 3 channels ('_ensure_three_channels')
                v.  Save frame immediately
        5. Clear intermediate tensors to free memory after each save.

        Args:
            sub_name     (str): Subject name.
            video_paths  (List[str]): Paths to video files (typically one).
            mask_paths   (List[str]): Paths to mask videos (optional).
            modality     (str): Normalized modality key.
            img_out_dir  (str): Output directory for frames.
            mask_out_dir (str): Output directory for masks.
            composite_id (str): Composite identifier for filename tagging.
            group_bbox   (tuple): (min_row, max_row, min_col, max_col).

        Returns:
            dict: Metadata, e.g. {fps, num_frames, modality, resize_shape}.
        """
        # 1) Load frames and fps
        frames, fps = self.load_video(video_paths[0])

        # 2) Load mask frames (if any)
        mask_frames = []
        if mask_paths:
            mask_frames, _ = self.load_mask(mask_paths[0])

        # 3) Match mask size to frame size
        frame_h, frame_w = frames[0].shape[:2]
        if mask_frames.shape[1:] != (frame_h, frame_w):
            mask_frames = [
                cv2.resize(m.astype(np.uint8), (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST)
                for m in mask_frames
            ]

        # 4) Process each frame
        num_frames = len(frames)
        for idx, frame in enumerate(tqdm(frames, desc=f'[{composite_id}] Processing frames')):
            keep_frame = True

            # 4a) Process mask and decide whether to keep frame
            if mask_paths:
                if idx < len(mask_frames):
                    cropped_m = self._crop(mask_frames[idx], group_bbox)
                    resized_m = self._resize(cropped_m, is_mask=True)

                    comps = []
                    for bin_mask in self._split_multiclass_mask(resized_m):
                        for comp in self._split_connected_components(bin_mask):
                            comps.append(comp)
                    filtered = self._filter_small_components(comps)

                    if not filtered:
                        self.dataset_logger.warning(
                            f"No significant mask components in frame {idx}, skipping"
                        )
                        keep_frame = False
                    else:
                        for comp_idx, comp in enumerate(filtered):
                            self._save_mask(comp, 'video', idx, comp_idx,
                                            composite_id, mask_out_dir)
                            del comp

                    del cropped_m, resized_m, comps, filtered
                else:
                    self.dataset_logger.warning(
                        f"No mask available for frame {idx}, skipping"
                    )
                    keep_frame = False

            # 4b) Skip frame if no valid mask
            if mask_paths and not keep_frame:
                continue

            # 5) Process and save image
            cropped = self._crop(frame, group_bbox)
            norm = self._normalize_intensity(cropped, modality)
            resized = self._resize(norm, is_mask=False)
            final = self._ensure_three_channels(resized)

            self._save_image(final, 'video', idx, composite_id, img_out_dir)

            del cropped, norm, resized, final

        metadata = {
            "fps": fps,
            "num_frames": num_frames,
            "modality": modality,
            "resize_shape": self.target_size
        }
        main_logger.info(
            f"Completed video preprocessing: {len(frames)} frames at {fps} fps, {len(mask_frames)} masks"
        )
        return metadata

    def load_image(self, image_paths):
        """
        Load image data from a file or series and return the image array(s) and header information.

        Supports:
        1. DICOM series (list of .dcm or .dicom files) using SimpleITK ImageSeriesReader.
        2. Numpy files (.npy): returns the array and a SimpleITK image created from the array.
        3. Single-file volumes (NIfTI, NRRD, MHD, single-slice DICOM): returns a numpy array and the SimpleITK image.
            - Multi-modality volumes are split into a list of 3D arrays and a list of corresponding SimpleITK images.
        4. Single-file images (PNG, JPEG, TIFF, optionally .gz): returns a numpy array and the PIL info dict.

        Args:
            image_paths (str or list of str): Path to an image file or a list of paths for a DICOM series.

        Returns:
            tuple:
                data (np.ndarray or list of np.ndarray): Image data array(s).
                header (SimpleITK.Image or list of SimpleITK.Image or dict): 
                    - For DICOM/volume inputs: a SimpleITK.Image or list of images.
                    - For multi-modality splits: a list of SimpleITK.Image objects.
                    - For single-file images: a dict of PIL image info.
        """
        # Normalize to list
        if isinstance(image_paths, (list, tuple)):
            paths = image_paths
        else:
            paths = [image_paths]
        self.dataset_logger.info(f"Loading image: {paths}")
        ext = get_extension(paths[0]).lower()

        try:
            # 1) DICOM series (multi-slice)
            if len(paths) > 1 and ext in ('.dcm', '.dicom'):
                # Sort by ImagePositionPatient Z (fall back to InstanceNumber)
                def _slice_key(fname):
                    ds = pydicom.dcmread(fname, stop_before_pixels=True)
                    if hasattr(ds, 'ImagePositionPatient'):
                        return float(ds.ImagePositionPatient[2])
                    return int(getattr(ds, 'InstanceNumber', 0))
                
                # Only sort if LIDC-IDRI
                if self.dataset_name == 'LIDC-IDRI':
                    paths = sorted(paths, key=_slice_key)
                else:
                    paths = image_paths

                # Read DICOM series
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(paths)
                img_itk = reader.Execute()
            # 2) Any numpy file
            elif ext == '.npy':
                data = np.load(paths[0])
                img_itk = sitk.GetImageFromArray(data)
                return data, img_itk
            # 3) Any single-file volume (NIfTI, NRRD, MHD, or single-slice DICOM)
            elif ext in self.volume_exts:
                img_itk = sitk.ReadImage(paths[0])
                data = sitk.GetArrayFromImage(img_itk)

                # If we have a multi-modality volume (e.g., shape (Z, Y, X, M)), split into M 3D volumes
                if isinstance(data, np.ndarray) and data.ndim == 4:
                    self.dataset_logger.info(f"Splitting 4D volume into 3D volumes")
                    vols = []
                    its = []
                    size = list(img_itk.GetSize()) # shape (Z, Y, X, M)
                    num_modalities = size[3]
                    for m in range(num_modalities):
                        # Extract a 3D SimpleITK volume for modality m
                        extract_size = [size[0], size[1], size[2], 0]
                        extract_index = [0, 0, 0, m]
                        itk3d = sitk.Extract(img_itk, extract_size, extract_index)
                        vol3d = sitk.GetArrayFromImage(itk3d)
                        vols.append(vol3d)
                        its.append(itk3d)
                    return vols, its

                # Otherwise, return the single volume
                return data, img_itk
            else:
                # 4) Any single-file image (PNG, JPEG, TIFF, etc.)
                opener = gzip.open if paths[0].lower().endswith('.gz') else open
                with opener(paths[0], 'rb') as f:
                    pil_img = Image.open(f)
                    data = np.array(pil_img)
                    header = pil_img.info
                return data, header

            # Convert SimpleITK volume to numpy
            data = sitk.GetArrayFromImage(img_itk)  # shape (Z, H, W) for volumes
            header = img_itk
            return data, header
        except Exception as e:
            self.dataset_logger.error(f"Failed to load image: {paths} ({e})")
            raise

    def load_mask(self, mask_path):
        """
        Load mask data from a file, series, or .mat file and return a mask array and header.

        Supports:
        1. list or tuple of paths: treated as a volume or DICOM series via load_image
        2. .npy files: loaded with numpy and converted to a SimpleITK image
        3. .mat files: loads 'predicted' or 'inst_map' key if present, otherwise the first key, as an int32 array
        4. single-file volumes (NIfTI, NRRD, MHD, single-slice DICOM): via load_image and converted to int32
        5. 2D color or RGBA images: collapse color codes to a label map or binary mask
        6. multi-channel arrays: any non-zero across channels yields a binary mask
        7. single-channel arrays: detects multiclass or binary mask

        Args:
            mask_path (str or list of str): Path or list of paths to the mask file(s).

        Returns:
            tuple:
                mask_array (np.ndarray): 2D or 3D mask array with dtype int32
                header (SimpleITK.Image or dict): Header or metadata returned by load_image, or empty dict for .mat
        """
        # Series of files -> delegate to load_image (handles DICOM series, NIfTI, etc.)
        if isinstance(mask_path, (list, tuple)):
            self.dataset_logger.info(f"Loading mask series: {mask_path}")
            raw, header = self.load_image(mask_path)
            return np.rint(raw).astype(np.int32), header

        self.dataset_logger.info(f"Loading mask: {mask_path!r}")
        ext = get_extension(mask_path).lower()

        if ext == '.npy':
            raw = np.load(mask_path)
            msk_itk = sitk.GetImageFromArray(raw.astype(np.int32))
            return np.rint(raw).astype(np.int32), msk_itk

        # .mat volume
        if ext == '.mat':
            mat = scipy.io.loadmat(mask_path)
            keys = [k for k in mat if not k.startswith('__')]
            key = 'predicted' if 'predicted' in keys else 'inst_map' if 'inst_map' in keys else keys[0]
            data = np.rint(mat[key]).astype(np.int32)
            self.dataset_logger.info(f"  -> .mat key='{key}', shape={data.shape}")
            return data, {}

        # All other formats via load_image
        raw, header = self.load_image(mask_path)
        raw = np.rint(raw).astype(np.int32)

        # True volumes: return as-is
        if ext in self.volume_exts:
            return raw, header

        # 2D color/RGBA
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            rgb = raw[..., :3].astype(np.uint8)
            if self._is_multiclass_mask(rgb):
                lbl = self._collapse_color_to_labels(rgb)
            else:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                lbl = (gray != self.background_value).astype(np.int32)
            self.dataset_logger.info(f"  -> color mask -> shape={lbl.shape}")
            return lbl, header

        # Multi-channel -> any non-zero
        if raw.ndim == 3:
            lbl = np.any(raw != self.background_value, axis=2).astype(np.int32)
            self.dataset_logger.info(f"  -> collapsed mask -> shape={lbl.shape}")
            return lbl, header

        # Single-channel -> binary or multiclass
        if self._is_multiclass_mask(raw):
            lbl = raw.astype(np.int32)
        else:
            lbl = (raw != self.background_value).astype(np.int32)
        self.dataset_logger.info(f"  -> single-channel mask -> shape={lbl.shape}")
        return lbl, header

    def load_video(self, video_path):
        """
        Load a video file into a list of frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tuple:
                frames (List[np.ndarray]): List of HxWx3 RGB frames.
                fps (float): Frames per second of the video.
        """
        self.dataset_logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        self.dataset_logger.info(f"Loaded {len(frames)} frames at {fps} fps")
        return frames, fps
    
    def _to_uint8(self, arr):
        """
        Scale [min,max] -> [0,255] and cast to uint8, with NaN/Inf -> 0.

        Args:
            arr (np.ndarray): Input array.

        Returns:
            np.ndarray: Scaled and cast array.
        """
        a, b = np.nanmin(arr), np.nanmax(arr)
        if b > a:
            scaled = (arr - a) / (b - a)
        else:
            scaled = arr * 0
        scaled = np.nan_to_num(scaled, nan=0, posinf=0, neginf=0)
        return (scaled * 255).astype(np.uint8)
    
    def _short_id(self, composite_id):
        """
        Generate a short identifier for a composite image using MD5.

        Args:
            composite_id (str): Composite identifier.

        Returns:
            str: Short identifier.
        """
        return hashlib.md5(composite_id.encode()).hexdigest()[:8]
    
    def _save_image(self, img, mode, idx, composite_id, out_dir):
        """
        Save a processed image (2D slice, video frame, or 3D volume) to disk.

        For 2D and video modes, writes PNGs; for 3D, writes a NIfTI file.

        Args:
            img (np.ndarray or sitk.Image): Image data to save.
            mode (str): '2d', 'video', or '3d'.
            idx (int): Index of the image within the group (unused for '3d').
            composite_id (str): Base filename (for '3d') or identifier used to generate filenames.
            out_dir (str): Output directory.
        Returns:
            str: Full path to the saved file.
        """
        if mode == '3d':
            # composite_id includes desired suffix (e.g., 'patient_image' or 'patient_modality0')
            filename = f"{composite_id}.nii.gz"
            path = os.path.join(out_dir, filename)
            if isinstance(img, sitk.Image):
                sitk.WriteImage(img, path)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(img), path)
            self.dataset_logger.info(f"Saved 3D NIfTI: {filename} to {out_dir}")
            return path

        # 2D or video: save as PNG
        sid = self._short_id(composite_id)
        token, pad = {
            '2d':    ('img',   3),
            'video': ('frame', 4),
        }[mode]
        filename = f"{sid}_{token}_{idx:0{pad}d}.png"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, self._to_uint8(img))
        self.dataset_logger.info(f"Saved {filename} to {out_dir}")
        return path

    def _save_mask(self, mask, mode, idx, comp_idx, composite_id, out_dir):
        """
        Save a processed mask (2D component, video frame, or 3D volume) to disk.

        For 2D and video modes, writes PNGs for each connected component; for 3D, writes a NIfTI file.

        Args:
            mask (np.ndarray or sitk.Image): Mask data to save.
            mode (str): '2d', 'video', or '3d'.
            idx (int): Index of the image or frame (unused for '3d').
            comp_idx (int): Index of the mask component (unused for '3d').
            composite_id (str): Base filename (for '3d') or identifier used to generate filenames.
            out_dir (str): Output directory.
        Returns:
            str: Full path to the saved file (for '3d') or last saved PNG path (for 2D/video).
        """
        if mode == '3d':
            # composite_id includes desired suffix (e.g., 'patient_mask')
            filename = f"{composite_id}.nii.gz"
            path = os.path.join(out_dir, filename)
            if isinstance(mask, sitk.Image):
                sitk.WriteImage(mask, path)
            else:
                sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.int32)), path)
            self.dataset_logger.info(f"Saved 3D Mask NIfTI: {filename} to {out_dir}")
            return path

        # 2D or video: save each mask component as PNG
        sid = self._short_id(composite_id)
        token, pad = {
            '2d':    ('img',   3),
            'video': ('frame', 4),
        }[mode]
        filename = f"{sid}_{token}_{idx:0{pad}d}_mask_{comp_idx}.png"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, self._to_uint8(mask))
        self.dataset_logger.info(f"Saved {filename} to {out_dir}")
        return path

    def _compute_group_bbox(self, image_paths, pipeline):
        """
        Compute a unified bounding box for a batch of 2D images or 3D volumes.

        If pipeline == '3D':
        - Load volume(s) via load_image, including multi-modality splits
        - Collapse foreground across modalities to find occupied voxels
        - Compute (z0, z1, y0, y1, x0, x1) bounding indices
        - Set self.is_significant_crop to True if >=25 percent of voxels are removed

        Otherwise (2D):
        - For each image or first frame of video, compute a 2D foreground mask
        - Map tile coordinates to global coordinates for tiled datasets (e.g. PAIP2019)
        - Aggregate extents to compute (y0, y1, x0, x1)
        - Set self.is_significant_crop to True if >=25 percent of area is removed

        Args:
            image_paths (list of str): Paths to image files or series.
            pipeline (str): '3D' to perform 3D bounding box, otherwise 2D.

        Returns:
            tuple:
                For 3D: (z0, z1, y0, y1, x0, x1)
                For 2D: (y0, y1, x0, x1)
        """
        self.dataset_logger.info(f"Computing group bounding box for {len(image_paths)} inputs")
        ext0 = get_extension(image_paths[0]).lower()

        # 1) 3D volume case (including multi-modality split): NIfTI/NRRD/MHD or multi-file DICOM
        if pipeline == "3D":
            # Load_image may return (np.ndarray, itk) or (List[np.ndarray], List[itk])
            vol_data, vol_header = self.load_image(image_paths)

            # If load_image returned a list of 3D volumes (i.e., original was multi-modality),
            # collapse foreground across all modalities.
            if isinstance(vol_data, list):
                # Stack per-modality volumes and compute foreground mask
                stacked = np.stack([vol != self.background_value for vol in vol_data], axis=0)  # (M, Z, Y, X)
                fg_all = np.any(stacked, axis=0)  # (Z, Y, X)
                vol_shape = fg_all.shape  # (Z, Y, X)
                if not np.any(fg_all):
                    Z, Y, X = vol_shape
                    self.dataset_logger.warning("No foreground found in multi-modality volume; using full extent")
                    return (0, Z, 0, Y, 0, X)
                zs, ys, xs = np.where(fg_all)

            else:
                # vol_data is a single 3D or 2D array
                vol = vol_data
                # If a 2D slice was loaded as 2D, treat it as single-slice volume
                if vol.ndim == 2:
                    vol = vol[np.newaxis, ...]
                # Now vol.ndim is either 3 or (rarely) 4 if load_image didn't split it
                if vol.ndim == 4:
                    # Collapse across modality axis
                    fg_all = np.any(vol != self.background_value, axis=0)
                    vol_shape = fg_all.shape
                    if not np.any(fg_all):
                        Z, Y, X = vol_shape
                        self.dataset_logger.warning("No foreground found in 4D volume; using full extent")
                        return (0, Z, 0, Y, 0, X)
                    zs, ys, xs = np.where(fg_all)

                else:
                    # vol.ndim == 3 here
                    vol_shape = vol.shape
                    fg = (vol != self.background_value)
                    if not np.any(fg):
                        Z, Y, X = vol_shape
                        self.dataset_logger.warning("No foreground found in volume; using full extent")
                        return (0, Z, 0, Y, 0, X)
                    zs, ys, xs = np.where(fg)

            # Compute the bounding indices
            z0, z1 = int(zs.min()), int(zs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            # Calculate crop significance
            orig_vol = np.prod(vol_shape)
            crop_vol = (z1 - z0) * (y1 - y0) * (x1 - x0)
            self.is_significant_crop = ((orig_vol - crop_vol) / orig_vol) >= 0.25

            bbox3d = (z0, z1, y0, y1, x0, x1)
            self.dataset_logger.info(f"3D group bbox: {bbox3d}")
            return bbox3d

        # 2) 2D case: images or videos -> compute 2D bbox tile-by-tile
        global_min_y = float(+1e9)
        global_min_x = float(+1e9)
        global_max_y = float(-1e9)
        global_max_x = float(-1e9)
        any_fg_found = False

        for path in tqdm(image_paths, desc=f"Computing 2D bbox"):
            ext = get_extension(path).lower()
            # If it's a video, grab only the first frame
            if ext in self.video_exts:
                frames, _ = self.load_video(path)
                data_list = [frames[0]] if frames else []
            else:
                img, _ = self.load_image(path)
                data_list = [img]

            for data in data_list:
                # Compute a 2D foreground mask for this tile/frame
                if data.ndim == 3:
                    H, W, C = data.shape
                    # If color image (HxWx3 or HxWx4), collapse on the channel axis
                    if C in (3, 4) and max(data.shape[:2]) != C:
                        fg_mask = np.any(data != self.background_value, axis=2)
                    else:
                        # If it’s actually a small 3D volume ZxHxW, collapse over Z
                        fg_mask = np.any(data != self.background_value, axis=0)
                else:
                    # data is already 2D (HxW)
                    fg_mask = (data != self.background_value)

                ys_local, xs_local = np.nonzero(fg_mask)
                if ys_local.size == 0:
                    continue

                any_fg_found = True

                # Parse the tile’s row/col indices from its filename, e.g. "0_10.jpeg"
                basename = os.path.basename(path)
                name_no_ext = os.path.splitext(basename)[0]
                if getattr(self, "dataset_name", "").lower() == "paip2019":
                    try:
                        row_str, col_str = name_no_ext.split("_")
                        row_idx, col_idx = int(row_str), int(col_str)
                    except ValueError:
                        row_idx, col_idx = 0, 0
                else:
                    row_idx, col_idx = 0, 0
                
                # Convert local coords to slide-global coords (tile size 512)
                ys_global = ys_local + (row_idx * 512)
                xs_global = xs_local + (col_idx * 512)

                current_min_y = int(ys_global.min())
                current_min_x = int(xs_global.min())
                current_max_y = int(ys_global.max())
                current_max_x = int(xs_global.max())

                global_min_y = min(global_min_y, current_min_y)
                global_min_x = min(global_min_x, current_min_x)
                global_max_y = max(global_max_y, current_max_y)
                global_max_x = max(global_max_x, current_max_x)

        # If never found any foreground, fall back to full extent of first image
        if not any_fg_found:
            first_path = image_paths[0]
            ext = get_extension(first_path).lower()
            if ext in self.video_exts:
                frames, _ = self.load_video(first_path)
                data0 = frames[0] if frames else None
            else:
                data0, _ = self.load_image(first_path)

            if data0 is not None:
                if data0.ndim == 3 and data0.shape[2] in (3, 4):
                    H0, W0 = data0.shape[:2]
                elif data0.ndim == 2:
                    H0, W0 = data0.shape
                else:
                    H0, W0 = data0.shape[-2:]
            else:
                H0, W0 = self.target_size

            self.dataset_logger.warning("No foreground found across group; using full image")
            return (0, H0, 0, W0)

        y0, y1 = int(global_min_y), int(global_max_y) + 1
        x0, x1 = int(global_min_x), int(global_max_x) + 1

        # Estimate full-slide area for significant-crop flag
        full_H = y1
        full_W = x1
        orig_area = full_H * full_W
        crop_area = (y1 - y0) * (x1 - x0)
        self.is_significant_crop = ((orig_area - crop_area) / orig_area) >= 0.25

        bbox2d = (y0, y1, x0, x1)
        self.dataset_logger.info(f"2D group bbox: {bbox2d}")
        return bbox2d

    def _crop(self, arr, bbox):
        """
        Crop a 2D image, volume slice, or mask array to the specified bounding box.

        Args:
            arr (np.ndarray): Array of shape (H, W) or (H, W, C).
            bbox (tuple): (min_row, max_row, min_col, max_col) to crop.

        Returns:
            np.ndarray: Cropped array of the same dimensionality as input.
        """
        self.dataset_logger.info(f"Cropping array with bbox={bbox}")
        r0, r1, c0, c1 = bbox
        return arr[r0:r1, c0:c1]

    def _normalize_intensity(self, image, modality, use_central_region=False):
        """
        Normalize image intensities according to modality-specific rules.

        CT:
            - Apply windowing (if ct_window set)
            - Identify foreground voxels (> background_value)
            - Clip to 0.5-99.5 percentile of foreground
            - Z-score using global CT mean/std (fallback to per-image if None)

        MRI/X-ray/Ultrasound/etc.:
            - Clip to 0.5-99.5 percentile of full image
            - Compute mean/std over central non-zero region if significant_crop, else full image
            - Z-score

        Default/unknown:
            - Clip 0.5-99.5 percentile
            - Z-score over full image

        Args:
            image (np.ndarray): Input image array (any dtype, 2D).
            modality (str): Lowercase modality key ('ct', 'mri', etc.).
            use_central_region (bool): If True, normalize over central region.

        Returns:
            np.ndarray: Normalized image (float32).
        """
        self.dataset_logger.info(f"Normalizing intensity image with modality={modality}")
        img = image.astype(np.float32)

        if modality == 'ct':
            # CT-specific path
            # Windowing, if provided
            if self.ct_window is not None:
                window_width, window_level = self.ct_window
                lower = window_level - window_width // 2.0
                upper = window_level + window_width // 2.0
                img = np.clip(img, lower, upper)

            # Foreground mask for clipping
            fg_mask = img > self.background_value

            # Percentile-based clipping on foreground
            if np.any(fg_mask):
                lower = np.percentile(img[fg_mask], 0.5)
                upper = np.percentile(img[fg_mask], 99.5)
                img = np.clip(img, lower, upper)

            # Global z-score normalization (fallback to per-image if None)
            if self.global_ct_mean is not None and self.global_ct_std is not None:
                mean = self.global_ct_mean
                std = self.global_ct_std
            else:
                if np.any(fg_mask):
                    mean = img[fg_mask].mean()
                    std = img[fg_mask].std()
                else:
                    mean = img.mean()
                    std = img.std()

            result = self._calculate_zscore(img, mean, std)
        elif modality in self.modalities_2d or modality == "mri":
            # MRI, X-ray, ultrasound, etc.
            # 1) Make a foreground mask
            fg_mask = img > self.background_value

            # 2) Percentile-based clipping on foreground only
            if np.any(fg_mask):
                lower = np.percentile(img[fg_mask], 0.5)
                upper = np.percentile(img[fg_mask], 99.5)
                img = np.clip(img, lower, upper)

            # 3) Decide which pixels to use for mean/std
            if use_central_region:
                # Only consider non-zero after the crop for mean/std
                region_mask = img > self.background_value
                if np.any(region_mask):
                    vals = img[region_mask]
                else:
                    vals = img.flatten()
            else:
                # Use all foreground pixels for mean/std
                if np.any(fg_mask):
                    vals = img[fg_mask]
                else:
                    vals = img.flatten()

            # 4) Compute mean and std
            mean = vals.mean()
            std = vals.std()

            result = self._calculate_zscore(img, mean, std)
        else:
            # Unknown modality
            # Global percentile clipping
            lower = np.percentile(img, 0.5)
            upper = np.percentile(img, 99.5)
            img = np.clip(img, lower, upper)

            # Compute mean and std
            mean = img.mean()
            std = img.std()

            result = self._calculate_zscore(img, mean, std)

        self.dataset_logger.info(f"Normalized image with modality={modality}, mean={mean:.3f}, std={std:.3f}")
        return result.astype(np.float32)

    def _calculate_zscore(self, img, mean, std):
        """
        Zero-centers or z-scores img, then replaces any NaN/Inf with 0.

        Args:
            img (np.ndarray): Input image array (any dtype, 2D).
            mean (float): Mean value.
            std (float): Standard deviation.

        Returns:
            np.ndarray: Normalized image (float32).
        """
        if std <= 0:
            result = img - mean
        else:
            result = (img - mean) / std

        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def _resize(self, data, is_mask=False):
        """
        Resize a 2D or 3D array (image or mask) to the target size.

        For image data (is_mask=False): use bicubic interpolation to preserve intensity continuity.
        For mask data (is_mask=True): use nearest-neighbor interpolation to preserve label values.

        Parameters:
            data (np.ndarray): Input array of shape (H, W) or (H, W, C).
            is_mask (bool): Flag indicating whether this is a mask (use nearest-neighbor).

        Returns:
            np.ndarray: Resized array of shape (target_height, target_width) or
                        (target_height, target_width, C) matching input channels.
        """
        self.dataset_logger.info(f"Resizing to {self.target_size}")
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
        target_h, target_w = self.target_size

        # If 2D or single-channel, this also works:
        return cv2.resize(data, (target_w, target_h), interpolation=interp)

    def _ensure_three_channels(self, img):
        """
        Ensure an image has three channels by replicating or trimming as needed.

        Parameters:
            img (np.ndarray): Input image array, either
                              2D (H, W) or 3D (H, W, C).

        Returns:
            np.ndarray: 3D image array of shape (H, W, 3).
        """
        if img.ndim == 2:
            # Single-channel: replicate to RGB
            return np.stack([img, img, img], axis=2)
        elif img.ndim == 3:
            h, w, c = img.shape
            if c == 3:
                # Already three channels
                return img
            elif c == 1:
                # One channel: replicate that channel
                return np.concatenate([img, img, img], axis=2)
            else:
                # More than 3 channels: take first three
                return img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    def _split_multiclass_mask(self, mask):
        """
        Split a multi-class segmentation mask into binary masks for each class.

        Args:
            mask (np.ndarray): Integer mask array where 0 = background and
                               positive integers represent distinct classes.

        Returns:
            List[np.ndarray]: List of binary mask arrays, one per class label > 0.
        """
        self.dataset_logger.info("Splitting multi-class mask")

        labels = np.unique(mask)
        fg_labels = [lbl for lbl in labels if lbl != self.background_value]

        binaries = [(mask == lbl).astype(np.uint8) for lbl in fg_labels]
        return binaries if binaries else [np.zeros_like(mask, dtype=np.uint8)]

    def _split_connected_components(self, binary_mask):
        """
        Split a binary mask into its connected components.

        Args:
            binary_mask (np.ndarray): 2D array of 0/1 values.

        Returns:
            List[np.ndarray]: List of binary masks, one per connected component.
        """
        self.dataset_logger.info("Splitting connected components")

        # Ensure binary
        mask = (binary_mask > 0).astype(np.uint8)

        # Label connected components
        num_labels, labels_im = cv2.connectedComponents(mask, connectivity=8)

        # Create masks for each component
        components = []
        for lbl in range(1, num_labels):
            comp_mask = (labels_im == lbl).astype(np.uint8)
            components.append(comp_mask)

        self.dataset_logger.info(f"Split into {len(components)} connected components")
        return components

    def _filter_small_components(self, mask_list):
        """
        Remove connected-component masks smaller than the minimum area threshold.

        Args:
            mask_list (List[np.ndarray]): List of binary mask arrays.

        Returns:
            List[np.ndarray]: Filtered list where each mask has area >= min_mask_size.
        """
        self.dataset_logger.info("Filtering small components")

        filtered = []
        for m in mask_list:
            # Count non-zero pixels
            area = int(m.sum())
            if area >= self.min_mask_size:
                filtered.append(m)

        self.dataset_logger.info(f"Filtered to {len(filtered)} components from {len(mask_list)} original components.")
        return filtered

    def _is_multiclass_mask(self, mask_array):
        """
        Determine if a mask array is multiclass (i.e. has more than 2 distinct labels)

        Args:
            mask_array (np.ndarray): Input mask array (uint8)

        Returns:
            bool: True if the mask is multiclass, False otherwise
        """
        # Check shape
        if mask_array.ndim == 3:
            # RGB(A) mask: check unique colours
            pixels = mask_array.reshape(-1, 3)
            unique = np.unique(pixels, axis=0)
            # Discard pure black background
            non_black = unique[~np.all(unique == [0, 0, 0], axis=1)]
            return len(non_black) > 1
        else:
            # Single-channel: any value beyond binary?
            return np.any(mask_array > 1)
        
    def _collapse_color_to_labels(self, mask_array, background_color=(0, 0, 0)):
        """
        Convert a color-coded mask array into a single-channel label map.

        This function collapses a mask array with color-coded labels into a 
        single-channel integer array, where each unique color in the mask is 
        mapped to a distinct integer label.

        Args:
            mask_array (np.ndarray): A 3D array of shape (H, W, 3) representing 
                                    the color-coded mask.
            background_color (tuple, optional): A tuple representing the RGB 
                                                color to be treated as the 
                                                background. Defaults to (0, 0, 0).

        Returns:
            np.ndarray: A 2D array of shape (H, W) where each pixel value 
                        corresponds to a label, with 0 representing the background 
                        and positive integers representing other classes.
        """
        h, w, _ = mask_array.shape
        flat = mask_array.reshape(-1, 3)
        unique = np.unique(flat, axis=0)

        # Build palette: background->0, others->1..N
        bg = tuple(background_color)
        palette = { bg: 0 }
        classes = [tuple(c) for c in unique if tuple(c) != bg]
        for idx, colour in enumerate(classes, start=1):
            palette[colour] = idx

        # Fill label map
        label_map = np.zeros((h, w), dtype=np.uint8)
        for colour, lbl in palette.items():
            mask = np.all(mask_array == colour, axis=-1)
            label_map[mask] = lbl

        return label_map
