"""
Script to convert MSCMR slices to NIfTI volumes.

Usage:
python scripts/preprocessing/mscmr_nifti.py
"""
import os
import re
import h5py
import numpy as np
import nibabel as nib
import SimpleITK as sitk

# Input and output directories
# input_dir     = "F:/Datasets/LGE-CMR/MSCMRSeg/MSCMR_preprocessed/MSCMR_training_slices"
# output_dir    = "F:/Datasets/LGE-CMR/MSCMRSeg/MSCMR_preprocessed/MSCMR_training_niftis"
input_dir     = "/data/research/LGE-CMR/MSCMRSeg/MSCMR_preprocessed/MSCMR_training_slices"
output_dir    = "/data/research/LGE-CMR/MSCMRSeg/MSCMR_preprocessed/MSCMR_training_niftis"
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to capture subject ID and slice index
pattern = re.compile(r"(subject\d+)_DE_slice_(\d+)\.h5")

# Gather files by subject
subjects = {}
for fname in os.listdir(input_dir):
    m = pattern.match(fname)
    if not m:
        continue
    subj      = m.group(1)           # e.g. "subject12"
    slice_idx = int(m.group(2))      # e.g. 5
    subjects.setdefault(subj, []).append((slice_idx, fname))

# Process each subject
for subj, slices in subjects.items():
    # Sort slices by slice index
    slices.sort(key=lambda x: x[0])
    
    # Stack data
    imgs = []
    labs = []
    for idx, fname in slices:
        path = os.path.join(input_dir, fname)
        with h5py.File(path, 'r') as f:
            imgs.append(f['image'][()])
            labs.append(f['label'][()])
    img_vol  = np.stack(imgs, axis=0)
    mask_vol = np.stack(labs, axis=0)
    
    # Make subject-specific folders
    subj_dir      = os.path.join(output_dir, subj)
    subj_img_dir  = os.path.join(subj_dir, "image")
    subj_mask_dir = os.path.join(subj_dir, "mask")
    os.makedirs(subj_img_dir,  exist_ok=True)
    os.makedirs(subj_mask_dir, exist_ok=True)
    
    # Convert to SITK images
    sitk_img  = sitk.GetImageFromArray(img_vol)
    sitk_mask = sitk.GetImageFromArray(mask_vol)
    
    # Write out into subject's subfolders
    out_img  = os.path.join(subj_img_dir,  f"{subj}_DE_image.nii.gz")
    out_mask = os.path.join(subj_mask_dir, f"{subj}_DE_label.nii.gz")
    sitk.WriteImage(sitk_img,  out_img)
    sitk.WriteImage(sitk_mask, out_mask)
    
    print(f"[{subj}] wrote image -> {out_img}")
    print(f"         wrote mask  -> {out_mask}")

print("All done.")