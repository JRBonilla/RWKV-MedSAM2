"""
Script to extract pure mask regions by thresholding pixels close to specified colors,
filling any internal holes so each region is completely solid,
closing small gaps to remove isolated black dots,
and saving the results as PNGs with zero compression.

Processes all JPEGs in an input directory and writes fixed masks to an output directory,
preserving base filenames but using the .png extension.

Usage:
    python clean_rgb_masks.py \
        -i path/to/input_masks \
        -o path/to/output_masks \
        -t 10  # optional tolerance (Euclidean distance)
"""
import os
import cv2
import numpy as np
import argparse

def fill_holes(color_mask):
    """
    Fill internal holes in a color mask using flood fill.
    
    Args:
        color_mask (np.ndarray): 2D array representing a color mask.
    Returns:
        np.ndarray: Mask with holes filled.
    """
    inv = cv2.bitwise_not(binary_mask)
    h, w = binary_mask.shape
    flood = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 255)
    holes = cv2.bitwise_not(inv)
    return cv2.bitwise_or(binary_mask, holes)

def process_image(input_path: str, output_path: str,
                  target_colors: list[np.ndarray], tol: float) -> None:
    # Read mask image and convert BGR -> RGB
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"[WARN] Could not read image: {input_path}")
        return
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)

    # Prepare blank output canvas
    new_img = np.zeros_like(img, dtype=np.uint8)

    # Threshold pixels close to each target color
    for color in target_colors:
        dist = np.linalg.norm(img - color.astype(np.int16), axis=2)
        mask = dist <= tol
        new_img[mask] = color

    # Fill any internal holes per color
    for color in target_colors:
        color_mask = (np.all(new_img == color, axis=2).astype(np.uint8) * 255)
        filled = fill_holes(color_mask)
        new_img[filled == 255] = color

    # Remove small black dots via morphological closing per color
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for color in target_colors:
        mask = (np.all(new_img == color, axis=2).astype(np.uint8) * 255)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        new_img[closed == 255] = color

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save final mask as PNG with zero compression (convert RGB -> BGR)
    out_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved: {os.path.basename(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Threshold mask pixels to nearest target color, fill holes, close gaps, and save PNGs"
    )
    parser.add_argument("-i", "--input_dir", default=".",
                        help="Folder of input JPEG masks")
    parser.add_argument("-o", "--output_dir", default="../train_gt_fixed/",
                        help="Folder to save processed PNG masks")
    parser.add_argument("-t", "--tolerance", type=float, default=10.0,
                        help="Max Euclidean distance to include edge pixels")
    args = parser.parse_args()

    # Define target colors in RGB (#fe0000 and #00ff01)
    target_colors = [np.array([254, 0, 0], dtype=np.uint8),
                     np.array([  0,255,  1], dtype=np.uint8)]

    # Process each JPEG file in input directory, save as PNG
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        base, _ = os.path.splitext(fname)
        inp = os.path.join(args.input_dir, fname)
        out = os.path.join(args.output_dir, f"{base}.png")
        process_image(inp, out, target_colors, args.tolerance)
