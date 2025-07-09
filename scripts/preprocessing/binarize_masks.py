"""
Script to binarize all images in a directory using OpenCV and save as PNGs with
zero compression.

Processes all JPEGs in an input directory and writes fixed masks to an output
directory, preserving base filenames but using the .png extension.

Usage:
    python binarize_masks.py \
        -i path/to/input_masks \
        -o path/to/output_masks \
        -t 10  # optional tolerance (Euclidean distance)
"""
import argparse
from pathlib import Path
import cv2

def binarize_image_cv(img_path, threshold):
    """
    Read an image with OpenCV, convert to grayscale if needed, threshold it,
    and return the binary image.

    Args:
        img_path (Path): Path to image file.
        threshold (int): Pixel-value threshold (0-255).

    Returns:
        np.ndarray: Binary image.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Convert to gray if it's not already single-channel
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply threshold: pixels > threshold -> 255, else 0
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def process_directory(input_dir, output_dir, threshold):
    """
    Walk input_dir, binarize each image, and save to output_dir as PNG with
    zero compression.

    Args:
        input_dir (Path): Directory containing source images.
        output_dir (Path): Directory to save binarized PNGs.
        threshold (int): Pixel-value threshold (0-255).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in input_dir.iterdir():
        if not in_path.is_file():
            continue

        binary = binarize_image_cv(in_path, threshold)
        if binary is None:
            # skip non-images or unreadable files
            continue

        out_path = (output_dir / in_path.name).with_suffix(".png")
        # PNG compression level: 0 = no compression
        cv2.imwrite(str(out_path), binary, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved binary image: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Binarize all images in a directory using OpenCV and save as PNGs."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Directory containing source images"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Directory to save binarized PNGs"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=0,
        help="Pixel-value threshold (0-255). Pixels above -> 255; default 0"
    )
    args = parser.parse_args()

    process_directory(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()
