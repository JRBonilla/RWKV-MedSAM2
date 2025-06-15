"""
Batch-binarize all grayscale mandible mask PNGs from the Pendal x-ray dataset using fixed paths.

This script processes masks stored in the 'Segmentation1' and 'Segmentation2'
folders under the 'PENDAL_ROOT' directory. It handles grayscale cutouts that
use a constant background shade (e.g., #202020) by inferring the background pixel
value from the image histogram and thresholding all pixels above it.
Results are saved to 'OUTPUT_DIR', preserving subfolder structure.
"""
import os
from PIL import Image

# PENDAL_ROOT = 'F:/Datasets/Pendal/Panoramic Dental X-rays With Segmented Mandibles/DentalPanoramicXrays'
PENDAL_ROOT = '/data/research/Pendal/Panoramic Dental X-rays With Segmented Mandibles/DentalPanoramicXrays'
OUTPUT_DIR = PENDAL_ROOT + "/Masks"

def binarize_pendal_masks(root_dir, output_dir):
    for subfolder in ["Segmentation1", "Segmentation2"]:
        in_folder = os.path.join(root_dir, subfolder)
        out_folder = os.path.join(output_dir, subfolder)
        if not os.path.isdir(in_folder):
            print(f"[Warning] Pendal subfolder not found, skipping: {in_folder}")
            continue
        os.makedirs(out_folder, exist_ok=True)
        for fname in os.listdir(in_folder):
            if not fname.lower().endswith('.png'):
                continue
            src = os.path.join(in_folder, fname)
            dst = os.path.join(out_folder, fname)
            try:
                img = Image.open(src)
                gray = img.convert('L')
                hist = gray.histogram()
                bg_val = max(range(len(hist)), key=lambda i: hist[i])
                mask = gray.point(lambda p: 255 if p > bg_val else 0)
                mask.save(dst)
                img.close()
                print(f"Binarized Pendal mask: {src} -> {dst}")
            except Exception as e:
                print(f"[Error] Could not binarize {src}: {e}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting binarization from {PENDAL_ROOT} to {OUTPUT_DIR}")
    binarize_pendal_masks(PENDAL_ROOT, OUTPUT_DIR)
    print("Binarization complete.")
