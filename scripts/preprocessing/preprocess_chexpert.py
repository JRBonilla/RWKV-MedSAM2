import cv2
import os
import json
import numpy as np
from tqdm import tqdm

def pad_to_square_centered(img):
    """
    Pads the given grayscale image to a square by centering it.
    Returns the padded image along with padding details.
    """
    h, w = img.shape[:2]
    longest_edge = max(h, w)
    result = np.zeros((longest_edge, longest_edge), dtype=img.dtype)
    pad_h = (longest_edge - h) // 2
    pad_w = (longest_edge - w) // 2
    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    pad_left = pad_w
    pad_top = pad_h
    pad_right = longest_edge - w - pad_w
    pad_bottom = longest_edge - h - pad_h

    return result, pad_left, pad_top, pad_right, pad_bottom

def pad_and_resize(image_path, target_size=1024):
    """
    Loads an image in grayscale, pads it to a square (centered),
    then resizes it to target_size x target_size.
    
    Args:
        image_path (str): Path to the input image.
        target_size (int): The desired output size (default=1024).
    
    Returns:
        resized_img (numpy.ndarray): The preprocessed image (grayscale).
        metadata (dict): Information to reverse the preprocessing.
    """
    # Read the image in grayscale (to match CheXmask's approach)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_height, original_width = img.shape[:2]
    padded_img, pad_left, pad_top, pad_right, pad_bottom = pad_to_square_centered(img)
    resized_img = cv2.resize(padded_img, (target_size, target_size))
    
    metadata = {
        "original_size": [original_height, original_width],
        "padding": {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right},
        "padded_size": [padded_img.shape[0], padded_img.shape[1]],
        "target_size": target_size
    }
    return resized_img, metadata

def collect_image_files(input_dir):
    """
    Recursively collects all image file paths (with their relative paths) from input_dir.
    
    Returns:
        List of tuples: (absolute_file_path, relative_file_path)
    """
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, input_dir)
                image_files.append((abs_path, rel_path))
    return image_files

def process_patients_from_batch(batch_input_dir, base_output_dir, target_size=1024):
    """
    Processes all patient folders (immediate subdirectories) from batch_input_dir.
    The output is saved directly under base_output_dir/<patient>/... preserving the patient folder's internal structure.
    A single progress bar tracks the number of patients processed.
    
    Returns:
        A dictionary mapping patient IDs to their metadata (per relative image path).
    """
    # List immediate patient directories
    patient_dirs = [d for d in os.listdir(batch_input_dir) if os.path.isdir(os.path.join(batch_input_dir, d))]
    metadata_dict = {}
    
    for patient in tqdm(patient_dirs, desc=f"Processing patients in {os.path.basename(batch_input_dir)}"):
        patient_input_dir = os.path.join(batch_input_dir, patient)
        # Output for each patient is directly in the base output directory.
        patient_output_dir = os.path.join(base_output_dir, patient)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Collect all images for this patient (recursively)
        image_list = collect_image_files(patient_input_dir)
        patient_meta = {}
        for abs_path, rel_path in image_list:
            out_file_path = os.path.join(patient_output_dir, rel_path)
            out_dir = os.path.dirname(out_file_path)
            os.makedirs(out_dir, exist_ok=True)
            try:
                processed_img, metadata = pad_and_resize(abs_path, target_size)
            except Exception as e:
                print(f"Error processing {abs_path}: {e}")
                continue
            cv2.imwrite(out_file_path, processed_img)
            patient_meta[rel_path] = metadata
        metadata_dict[patient] = patient_meta
    return metadata_dict

def main():
    # Training and testing input directories
    train_directories = [
        "G:/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 2 (train 1)",
        "G:/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 3 (train 2)",
        "G:/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 4 (train 3)"
    ]
    test_directory = "G:/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid"
    
    base_output_train_dir = "G:/CheXpert/preprocessed_chexpert/train/"
    base_output_test_dir = "G:/CheXpert/preprocessed_chexpert/test/"
    os.makedirs(base_output_train_dir, exist_ok=True)
    os.makedirs(base_output_test_dir, exist_ok=True)
    
    # Process training batches: each batch's patient folders are saved directly under base_output_train_dir.
    train_metadata = {}
    for batch_dir in train_directories:
        print(f"\nProcessing training batch: {batch_dir}")
        batch_meta = process_patients_from_batch(batch_dir, base_output_train_dir, target_size=1024)
        train_metadata.update(batch_meta)
    
    # Save training metadata
    train_meta_file = os.path.join(base_output_train_dir, "metadata.json")
    with open(train_meta_file, "w") as f:
        json.dump(train_metadata, f, indent=4)
    
    # Process test directory similarly.
    print(f"\nProcessing test directory: {test_directory}")
    test_metadata = process_patients_from_batch(test_directory, base_output_test_dir, target_size=1024)
    test_meta_file = os.path.join(base_output_test_dir, "metadata.json")
    with open(test_meta_file, "w") as f:
        json.dump(test_metadata, f, indent=4)

if __name__ == "__main__":
    main()
