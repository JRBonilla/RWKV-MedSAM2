import os
import shutil
import glob
import SimpleITK as sitk

# CONSTANTS
# Base directory containing the original QUBIQ2021 subdatasets
# BASE_DIR = r"F:\Datasets\QUBIQ2021\training_data_v3_QC"
BASE_DIR = r"/data/research/QUBIQ2021/training_data_v3_QC"
# Subdatasets to process
SUBDATASETS = [
    "pancreas",
    "pancreatic-lesion",
]

def reorient_image(image):
    """
    Move the original slice axis into the third dimension,
    swap sagittal and coronal planes,
    flip the posterior–anterior axis internally,
    and preserve the original world-space metadata.
    """
    permuted = sitk.PermuteAxes(image, [2, 1, 0])
    flipped = sitk.Flip(permuted, flipAxes=[False, True, False], flipAboutOrigin=False)
    flipped.SetOrigin(image.GetOrigin())
    flipped.SetSpacing(image.GetSpacing())
    flipped.SetDirection(image.GetDirection())
    return flipped

def process_file(fpath, output_dir, input_dir):
    """Reorient a single NIfTI file and save to output_dir preserving subfolder structure."""
    rel_path = os.path.relpath(fpath, start=input_dir)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = sitk.ReadImage(fpath)
    img_fixed = reorient_image(img)
    sitk.WriteImage(img_fixed, out_path)
    print(f"Reoriented and saved: {out_path}")

def process_directory(input_dir, output_dir):
    """Recursively process all .nii.gz files in input_dir, writing to output_dir."""
    pattern = os.path.join(input_dir, '**', '*.nii.gz')
    for fpath in glob.glob(pattern, recursive=True):
        process_file(fpath, output_dir, input_dir)

def main():
    """
    Copy each specified subdataset to a new folder with '-fixed' suffix,
    then reorient all NIfTI files inside that copy.
    """
    for sub in SUBDATASETS:
        src_dir = os.path.join(BASE_DIR, sub)
        dst_dir = os.path.join(BASE_DIR, f"{sub}-fixed")

        # Remove old fixed folder if exists
        if os.path.isdir(dst_dir):
            print(f"Removing existing directory: {dst_dir}")
            shutil.rmtree(dst_dir)

        # Copy original folder
        print(f"Copying {src_dir} → {dst_dir}")
        shutil.copytree(src_dir, dst_dir)

        # Reorient all NIfTI files in the copied folder
        print(f"Processing reorientation for: {dst_dir}")
        process_directory(dst_dir, dst_dir)

if __name__ == '__main__':
    main()
