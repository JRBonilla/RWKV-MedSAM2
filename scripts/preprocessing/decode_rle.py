import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def decode_rle(rle, height, width):
    """
    Decode a run-length encoded (RLE) string into a binary mask.
    If `rle` is empty or NaN, returns an all-zero mask.
    
    Args:
        rle (str): The run-length encoded string.
        height (int): The height of the mask.
        width (int): The width of the mask.
    
    Returns:
        np.ndarray: The decoded binary mask.
    """
    if pd.isna(rle) or rle.strip() == "":
        return np.zeros((height, width), dtype=np.uint8)
    
    s = rle.strip().split()
    # Convert values to integers and adjust for 0-indexing.
    starts = np.array(s[0::2], dtype=int) - 1
    lengths = np.array(s[1::2], dtype=int)
    
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask_flat[start:start+length] = 1
    
    return mask_flat.reshape((height, width))

def main():
    csv_path = "./CheXpert.csv"
    
    print("Loading CSV file, please wait...")
    df = pd.read_csv(csv_path)
    print("CSV file loaded successfully.")
    
    # Define base output directories for training and test.
    base_output_train = "./preprocessed_chexpert/train"
    base_output_test = "./preprocessed_chexpert/test"
    os.makedirs(base_output_train, exist_ok=True)
    os.makedirs(base_output_test, exist_ok=True)
    
    # Use one global progress bar for all rows.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV rows"):
        image_path_csv = row["Path"]  # e.g., "train/patient00001/study1/view1_frontal.jpg" or "valid/patientXXXXX/studyX/view1_frontal.jpg"
        height = int(row["Height"])
        width = int(row["Width"])
        
        # Decode RLE masks.
        left_mask = decode_rle(row["Left Lung"], height, width)
        right_mask = decode_rle(row["Right Lung"], height, width)
        heart_mask = decode_rle(row["Heart"], height, width)
        
        # Convert masks to 8-bit grayscale images.
        left_mask_img = Image.fromarray((left_mask * 255).astype(np.uint8))
        right_mask_img = Image.fromarray((right_mask * 255).astype(np.uint8))
        heart_mask_img = Image.fromarray((heart_mask * 255).astype(np.uint8))
        
        # Determine the output folder based on the CSV path.
        parts = row["Path"].split('/')
        # Check if the first component is train/valid.
        if parts[0].lower() in ["train", "valid"]:
            base_out = base_output_train if parts[0].lower() == "train" else base_output_test
            # Remove the first component.
            relative_dir = os.path.join(*parts[1:-1]) if len(parts) > 1 else ""
        else:
            base_out = base_output_train
            relative_dir = os.path.dirname(row["Path"])
        
        # Final output directory: base_out joined with relative_dir.
        # Now, add an additional "masks" subfolder.
        output_folder = os.path.join(base_out, relative_dir, "masks")
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the masks using fixed names.
        left_mask_img.save(os.path.join(output_folder, "left_lung.png"))
        right_mask_img.save(os.path.join(output_folder, "right_lung.png"))
        heart_mask_img.save(os.path.join(output_folder, "heart.png"))
    
    print("Mask processing complete.")

if __name__ == "__main__":
    main()
