import os
import h5py
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Path to the ThyroidUltra dataset
HDF5_PATH = "F:/Datasets/ThyroidUltra/thyroidultrasoundcineclip/dataset.hdf5"
OUTPUT_DIR = "F:/Datasets/ThyroidUltra/thyroidultrasoundcineclip/extracted_ThyroidUltra/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to overlay a mask onto an image for debugging (optional)
def overlay_mask(image, mask, save_path=None):
    """Overlay a segmentation mask onto an image to verify alignment."""
    overlay = cv2.merge([image, image, image])  # Convert grayscale to RGB
    overlay[mask > 0] = [255, 0, 0]  # Make segmentation areas red

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Image with Mask Overlay")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.show()

# Function to extract a small sample of images and masks
def extract_sample(num_samples=5):
    """Extracts a small sample of images and masks for testing."""
    sample_output_dir = os.path.join(OUTPUT_DIR, "sample")
    img_dir = os.path.join(sample_output_dir, "images")
    mask_dir = os.path.join(sample_output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    with h5py.File(HDF5_PATH, "r") as f:
        images = f["image"][:num_samples]
        masks = f["mask"][:num_samples]
        annot_ids = f["annot_id"][:num_samples]
        frame_nums = f["frame_num"][:num_samples]

        print(f"🔍 Extracting a sample of {num_samples} images/masks for testing...")

        for i in tqdm(range(num_samples), desc="Extracting Sample"):
            # Save ultrasound image
            img = Image.fromarray(images[i].astype(np.uint8))
            img.save(os.path.join(img_dir, f"image_{i}.png"))

            # Check the unique values in the mask
            unique_values = np.unique(masks[i])
            print(f"🔍 Mask {i} unique values: {unique_values}")

            # Normalize the mask properly
            mask_data = masks[i]
            if mask_data.max() > 1:  # If values are above 1, assume raw pixel values
                mask_data = (mask_data / mask_data.max()) * 255  # Normalize to 0-255
            elif mask_data.max() == 1:  # If it's binary (0 and 1)
                mask_data = mask_data * 255  # Scale up to 0-255
            else:
                print(f"⚠️ Warning: Mask {i} contains only zeros. It may be empty.")

            # Convert mask to 8-bit and save
            mask = Image.fromarray(mask_data.astype(np.uint8))
            mask.save(os.path.join(mask_dir, f"mask_{i}.png"))

        # Save metadata for the extracted sample
        metadata_sample_df = pd.DataFrame({"annot_id": annot_ids.flatten(), "frame_num": frame_nums.flatten()})
        metadata_sample_df.to_csv(os.path.join(sample_output_dir, "metadata_sample.csv"), index=False)

    print(f"✅ Sample extraction complete! Check '{sample_output_dir}'.")

# Function to extract the full dataset in chunks
def extract_full_dataset(chunk_size=500):
    """Extracts the full dataset in manageable chunks to prevent memory overload."""
    full_output_dir = os.path.join(OUTPUT_DIR, "full")
    img_dir = os.path.join(full_output_dir, "images")
    mask_dir = os.path.join(full_output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    metadata_list = []

    with h5py.File(HDF5_PATH, "r") as f:
        total_samples = f["image"].shape[0]
        print(f"📢 Total samples: {total_samples}. Extracting in chunks of {chunk_size}...")

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            print(f"🔄 Processing chunk {start}-{end}...")

            images = f["image"][start:end]
            masks = f["mask"][start:end]
            annot_ids = f["annot_id"][start:end]
            frame_nums = f["frame_num"][start:end]

            for i in tqdm(range(images.shape[0]), desc=f"Extracting {start}-{end}"):
                img_index = start + i

                # Save ultrasound image
                img = Image.fromarray(images[i].astype(np.uint8))
                img.save(os.path.join(img_dir, f"image_{img_index}.png"))

                # Normalize the mask properly
                mask_data = masks[i]
                if mask_data.max() > 1:  # If values are above 1, assume raw pixel values
                    mask_data = (mask_data / mask_data.max()) * 255  # Normalize to 0-255
                elif mask_data.max() == 1:  # If it's binary (0 and 1)
                    mask_data = mask_data * 255  # Scale up to 0-255
                else:
                    print(f"⚠️ Warning: Mask {img_index} contains only zeros. It may be empty.")

                # Convert mask to 8-bit and save
                mask = Image.fromarray(mask_data.astype(np.uint8))
                mask.save(os.path.join(mask_dir, f"mask_{img_index}.png"))

                metadata_list.append({"annot_id": annot_ids[i].flatten()[0], "frame_num": frame_nums[i].flatten()[0]})

            print(f"✅ Chunk {start}-{end} completed.")

    # Save full metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(os.path.join(full_output_dir, "metadata_full.csv"), index=False)

    print(f"✅ Full dataset extraction complete! Check '{full_output_dir}'.")

# Run sample extraction first
# extract_sample()

# Run full dataset extraction
extract_full_dataset(chunk_size=500)
