#!/usr/bin/env python3
"""
Script to clean up JPEG masks by binarizing, filling holes,
removing small artifacts, and saving as zero-compression PNGs.

Usage:
    python clean_binary_masks.py \
      -i path/to/input_masks \
      -o path/to/output_masks \
      -k 3    # optional closing kernel size
      -t 127  # optional binarization threshold
"""
import os
import cv2
import numpy as np
import argparse

def fill_holes(binary_mask):
    """
    Fill internal holes in a binary mask using flood fill.
    
    Args:
        binary_mask (np.ndarray): 2D array of 0/1 values.
    Returns:
        np.ndarray: Mask with holes filled.
    """
    inv = cv2.bitwise_not(binary_mask)
    h, w = binary_mask.shape
    flood = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 255)
    holes = cv2.bitwise_not(inv)
    return cv2.bitwise_or(binary_mask, holes)

def process_mask(in_path, out_path, thresh, kernel_size):
    """
    Process a single mask file.
    
    Args:
        in_path (str): Path to input JPEG mask.
        out_path (str): Path to save cleaned PNG mask.
        thresh (int): Grayscale threshold for binarization.
        kernel_size (int): Kernel size for morphological closing.
    """
    # Read as grayscale
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Could not read: {in_path}")
        return

    # Binarize (any pixel > thresh becomes 255)
    _, bin_mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # Fill holes
    filled = fill_holes(bin_mask)

    # Remove small artifacts via morphological closing
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    clean = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save as PNG with zero compression
    cv2.imwrite(out_path, clean, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved: {os.path.basename(out_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up Kvasir-SEG masks and save as PNGs"
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Directory containing input JPEG masks"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory to save cleaned PNG masks"
    )
    parser.add_argument(
        "-t", "--threshold", type=int, default=127,
        help="Grayscale threshold for binarization"
    )
    parser.add_argument(
        "-k", "--kernel", type=int, default=3,
        help="Kernel size for morphological closing"
    )
    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue
        base, _ = os.path.splitext(fname)
        inp  = os.path.join(args.input_dir, fname)
        outp = os.path.join(args.output_dir, f"{base}.png")
        process_mask(inp, outp, args.threshold, args.kernel)
