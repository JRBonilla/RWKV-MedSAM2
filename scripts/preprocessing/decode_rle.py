import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def decode_rle(rle, height, width):
    """
    Decode a run-length encoded (RLE) string into a binary mask.
    If `rle` is empty or NaN, returns an all-zero mask.
    """
    if pd.isna(rle) or rle.strip() == "":
        return np.zeros((height, width), dtype=np.uint8)
    
    s = rle.strip().split()
    starts = np.array(s[0::2], dtype=int) - 1
    lengths = np.array(s[1::2], dtype=int)
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask_flat[start:start+length] = 1
    
    return mask_flat.reshape((height, width))

def main():
    # csv_path = "F:/Datasets/CheXpert/chexpertchestxrays-u20210408/CheXpert.csv"
    csv_path = "/data/jrbonill/data/research/CheXpert/chexpertchestxrays-u20210408/CheXpert.csv"
    print("Loading CSV file, please wait...")
    df = pd.read_csv(csv_path)
    print("CSV file loaded successfully.")

    # base_output_train = "F:/Datasets/CheXpert/preprocessed_chexpert/train"
    # base_output_test = "F:/Datasets/CheXpert/preprocessed_chexpert/test"
    base_output_train = "/data/jrbonill/data/research/CheXpert/preprocessed_chexpert/train"
    base_output_test  = "/data/jrbonill/data/research/CheXpert/preprocessed_chexpert/test"
    os.makedirs(base_output_train, exist_ok=True)
    os.makedirs(base_output_test, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV rows"):
        image_path = row["Path"]
        height = int(row["Height"])
        width = int(row["Width"])

        left_mask = decode_rle(row["Left Lung"], height, width)
        right_mask = decode_rle(row["Right Lung"], height, width)
        heart_mask = decode_rle(row["Heart"], height, width)

        left_mask_img = Image.fromarray((left_mask * 255).astype(np.uint8))
        right_mask_img = Image.fromarray((right_mask * 255).astype(np.uint8))
        heart_mask_img = Image.fromarray((heart_mask * 255).astype(np.uint8))

        parts = image_path.split('/')
        filename = parts[-1]  # e.g., view1_frontal.jpg
        filename_base = os.path.splitext(filename)[0]  # remove .jpg extension

        if parts[0].lower() in ["train"]:
            base_out = base_output_train
            relative_dir = os.path.join(*parts[1:-1])  # e.g., patientXXX/studyY
        elif parts[0].lower() in ["valid", "test"]:
            base_out = base_output_test
            relative_dir = os.path.join(*parts[1:-1])
        else:
            base_out = base_output_train
            relative_dir = os.path.dirname(image_path)

        output_folder = os.path.join(base_out, relative_dir, "masks")
        os.makedirs(output_folder, exist_ok=True)

        # Save masks with modified filenames
        left_mask_img.save(os.path.join(output_folder, f"{filename_base}_left_lung.png"))
        right_mask_img.save(os.path.join(output_folder, f"{filename_base}_right_lung.png"))
        heart_mask_img.save(os.path.join(output_folder, f"{filename_base}_heart.png"))

    print("Mask processing complete.")

if __name__ == "__main__":
    main()
