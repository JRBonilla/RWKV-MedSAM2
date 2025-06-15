import os
import shutil
import pandas as pd
import pydicom
from tqdm import tqdm
import copy

# Base directories:
# images_base       = r"F:\Datasets\LIDC-IDRI\manifest-1600709154662"
# annotations_base  = r"F:\Datasets\LIDC-IDRI\DICOM-LIDC-IDRI-Nodules\manifest-1585232716547"
# output_base       = r"F:\Datasets\LIDC-IDRI\Merged-LIDC-IDRI"
images_base       = r"manifest-1600709154662"
annotations_base  = r"DICOM-LIDC-IDRI-Nodules/manifest-1585232716547"
output_base       = r"Merged-LIDC-IDRI"

def get_segmentation_dicom_file(seg_folder):
    """
    Returns the full path to the first file ending with '.dcm' in the folder.
    """
    if os.path.isdir(seg_folder):
        for fname in os.listdir(seg_folder):
            if fname.lower().endswith(".dcm"):
                return os.path.join(seg_folder, fname)
    print(f"No DICOM file found in folder: {seg_folder}")
    return None

def extract_frames_info(seg_dicom_path):
    """
    Read a segmentation DICOM file and for each frame, extract:
      - the referenced CT SOP Instance UID (from PerFrameFunctionalGroupsSequence)
      - the pixel data for that frame
      - the frame's functional group (to later store in the single-frame file)
    
    Returns a list of tuples: (ref_uid, frame_pixel_data, frame_functional_group)
    """
    try:
        ds = pydicom.dcmread(seg_dicom_path, force=True)
    except Exception as e:
        print(f"Error reading segmentation DICOM {seg_dicom_path}: {e}")
        return []
    
    try:
        pixel_array = ds.pixel_array
    except Exception as e:
        print(f"Error extracting pixel_array from {seg_dicom_path}: {e}")
        return []
    
    frames_info = []
    num_frames = pixel_array.shape[0] if pixel_array.ndim == 3 else 1

    if "PerFrameFunctionalGroupsSequence" not in ds:
        print(f"No PerFrameFunctionalGroupsSequence in {seg_dicom_path}")
        return []
    
    for i in range(num_frames):
        try:
            frame_group = ds.PerFrameFunctionalGroupsSequence[i]
            ref_uid = frame_group.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            frame_pixel_data = pixel_array[i] if num_frames > 1 else pixel_array
            frames_info.append((ref_uid, frame_pixel_data, frame_group))
        except Exception as e:
            print(f"Error processing frame {i} in {seg_dicom_path}: {e}")
    return frames_info

def create_single_frame_segmentation(ds, frame_pixel_data, frame_functional_group):
    """
    Create a new DICOM dataset from ds that contains only one frame.
    We deep-copy the dataset, update PixelData to contain only the selected frame's pixel data,
    update Rows/Columns, set NumberOfFrames to 1, and update the PerFrameFunctionalGroupsSequence.
    Also update pixel-related attributes such as BitsAllocated.
    """
    new_ds = copy.deepcopy(ds)
    new_ds.PixelData = frame_pixel_data.tobytes()
    new_ds.Rows, new_ds.Columns = frame_pixel_data.shape
    new_ds.NumberOfFrames = 1
    new_ds.PerFrameFunctionalGroupsSequence = [frame_functional_group]
    new_ds.SOPInstanceUID = pydicom.uid.generate_uid()
    new_ds.BitsAllocated = frame_pixel_data.dtype.itemsize * 8
    new_ds.BitsStored    = new_ds.BitsAllocated
    new_ds.HighBit       = new_ds.BitsStored - 1
    return new_ds

if __name__ == "__main__":
    # Paths for the two metadata CSVs
    seg_csv_path = os.path.join(annotations_base, "metadata.csv")
    ct_csv_path  = os.path.join(images_base,      "metadata.csv")
    
    # Read segmentation CSV and filter for SEG modality
    seg_df = pd.read_csv(seg_csv_path)
    seg_df = seg_df[seg_df["Modality"] == "SEG"]
    
    # Read CT CSV and filter for CT modality
    ct_df = pd.read_csv(ct_csv_path)
    ct_df = ct_df[ct_df["Modality"] == "CT"]
    
    # To avoid copying the same CT file more than once
    ct_files_copied = {}
    
    # Process each segmentation entry
    for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), desc="Processing segmentations"):
        # Get segmentation DICOM file path
        subject_id      = row["Subject ID"]
        study_uid       = row["Study UID"]
        raw_loc         = row["File Location"].lstrip(".\\/") # Normalize Windows-style File Location
        raw_loc         = raw_loc.replace("\\", os.sep) # Convert backslashes to OS separator before normalization
        folder_location = os.path.normpath(raw_loc)
        seg_folder      = os.path.join(annotations_base, folder_location)

        # Get segmentation DICOM file
        seg_dicom_file = get_segmentation_dicom_file(seg_folder)
        if seg_dicom_file is None:
            continue
        
        # Copy CT files
        for _, ct_row in ct_df[ct_df["Subject ID"] == subject_id].iterrows():
            raw_ct = ct_row["File Location"].lstrip(".\\/")
            raw_ct = raw_ct.replace("\\", os.sep)
            ct_folder_loc = os.path.normpath(raw_ct)
            ct_folder     = os.path.join(images_base, ct_folder_loc)
            if not os.path.isdir(ct_folder):
                print(f"No CT folder found: {ct_folder}")
                continue
            for root, _, files in os.walk(ct_folder):
                for f in files:
                    if not f.lower().endswith(".dcm"): continue
                    src = os.path.join(root, f)
                    try:
                        ds_ct = pydicom.dcmread(src, stop_before_pixels=True, force=True)
                        uid   = ds_ct.SOPInstanceUID
                        if uid in ct_files_copied: continue
                        dst = os.path.join(images_dest, f"{uid}.dcm")
                        shutil.copy2(src, dst)
                        ct_files_copied[uid] = True
                    except Exception as e:
                        print(f"Error copying CT file {src}: {e}")

        # Read the SEG dataset once for mask-splitting
        try:
            seg_ds = pydicom.dcmread(seg_dicom_file, force=True)
        except Exception as e:
            print(f"Error reading segmentation file {seg_dicom_file}: {e}")
            continue
        
        # Extract per-frame info and write masks (and leave images alone)
        frames_info = extract_frames_info(seg_dicom_file)
        if not frames_info:
            continue
        
        uid_counter = {}
        for ref_uid, frame_pixel_data, frame_functional_group in frames_info:
            uid_counter[ref_uid] = uid_counter.get(ref_uid, 0) + 1
            mask_fname = f"{ref_uid}_Mask_{uid_counter[ref_uid]:03d}.dcm"
            mask_out   = os.path.join(masks_dest, mask_fname)
            try:
                single_ds = create_single_frame_segmentation(
                    seg_ds, frame_pixel_data, frame_functional_group
                )
                single_ds.save_as(mask_out)
            except Exception as e:
                print(f"Error saving mask {mask_out}: {e}")
    
    print("Finished merging segmentation and CT scan files.")
