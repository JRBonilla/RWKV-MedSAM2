#!/usr/bin/env python3
import os
import re
import pandas as pd
import pydicom
from PIL import Image, ImageDraw, ImageChops
import shutil
from collections import defaultdict

# ================= USER CONFIGURATION =================
CSV_FILE = "scd_patientdata_updated.csv"    # CSV must have columns: PatientID, OriginalID, SAX_Series, CINESAX_Folder
ANNOTATIONS_ROOT = "./SCD_ManualContours"   # Root folder for manual contour annotations
DICOM_ROOT = "./"                           # Root directory where patient DICOM folders are found
OUTPUT_ROOT = "Merged_Cardiac_MRI"          # Output folder for merged Images and Masks
# ======================================================

# Regex to parse contour filenames including i, o, p1, p2 types
contour_pattern = re.compile(r'^IM-\d{4}-(?P<img>\d{4})-(?P<type>i|o|p1|p2)contour-manual\.txt$', re.IGNORECASE)

def create_mask(dicom_path, annotation_path):
    ds = pydicom.dcmread(dicom_path)
    height, width = ds.pixel_array.shape

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    points = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = round(float(parts[0])); y = round(float(parts[1]))
                    points.append((x, y))
                except ValueError:
                    continue
    if points:
        draw.polygon(points, outline=255, fill=255)
    return mask_img


def ds_local_size(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    h, w = ds.pixel_array.shape
    return (w, h)


def process_annotations(csv_file, annotations_root, dicom_root, output_root):
    df = pd.read_csv(csv_file, dtype=str)
    mapping = {row["OriginalID"].strip(): (
                    row["PatientID"].strip(),
                    row["SAX_Series"].strip(),
                    row["CINESAX_Folder"].strip()
                ) for _, row in df.iterrows()}

    for orig_folder in os.listdir(annotations_root):
        if orig_folder not in mapping:
            continue
        patient_id, sax_series, cine_folder = mapping[orig_folder]
        expert_dir = os.path.join(annotations_root, orig_folder, "contours-manual", "IRCCI-expert")
        if not os.path.isdir(expert_dir):
            continue

        ann_files = [f for f in os.listdir(expert_dir) if f.endswith("-manual.txt")]
        groups = defaultdict(dict)  # img_num -> { 'i','o','p1','p2' }
        for filename in ann_files:
            m = contour_pattern.match(filename)
            if not m:
                continue
            img_num = m.group('img')
            ctype = m.group('type')
            groups[img_num][ctype] = filename

        patient_out = os.path.join(output_root, patient_id)
        img_out = os.path.join(patient_out, "Images")
        mask_out = os.path.join(patient_out, "Masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for img_num, ctrs in groups.items():
            dicom_name = f"IM-{sax_series}-{img_num}.dcm"
            dicom_path = os.path.join(dicom_root, patient_id, cine_folder, dicom_name)
            if not os.path.exists(dicom_path):
                print(f"Missing DICOM {dicom_path}, skipping {sax_series}-{img_num}")
                continue

            dst_dcm = os.path.join(img_out, dicom_name)
            if not os.path.exists(dst_dcm):
                shutil.copy2(dicom_path, dst_dcm)

            # Paths for each contour type
            endo_path = ctrs.get('i') and os.path.join(expert_dir, ctrs['i'])
            epi_path  = ctrs.get('o') and os.path.join(expert_dir, ctrs['o'])
            p1_path   = ctrs.get('p1') and os.path.join(expert_dir, ctrs['p1'])
            p2_path   = ctrs.get('p2') and os.path.join(expert_dir, ctrs['p2'])

            # Generate masks
            endo = create_mask(dicom_path, endo_path) if endo_path else Image.new("L", ds_local_size(dicom_path), 0)
            epi  = create_mask(dicom_path, epi_path)  if epi_path  else Image.new("L", ds_local_size(dicom_path), 0)
            p1m  = create_mask(dicom_path, p1_path)   if p1_path   else Image.new("L", ds_local_size(dicom_path), 0)
            p2m  = create_mask(dicom_path, p2_path)   if p2_path   else Image.new("L", ds_local_size(dicom_path), 0)

            blood_pool = ImageChops.subtract(endo, p1m)
            blood_pool = ImageChops.subtract(blood_pool, p2m)
            myocardium = ImageChops.subtract(epi, endo)

            base = f"IM-{sax_series}-{img_num}"
            endo.save( os.path.join(mask_out, f"{base}-endocardium.png") )
            epi.save(  os.path.join(mask_out, f"{base}-epicardium.png") )
            blood_pool.save(os.path.join(mask_out, f"{base}-bloodpool.png"))
            myocardium.save(os.path.join(mask_out, f"{base}-myocardium.png"))

            print(f"Saved masks for {sax_series}-{img_num}: endocardium, epicardium, papillary1, papillary2, bloodpool, myocardium")

if __name__ == "__main__":
    process_annotations(CSV_FILE, ANNOTATIONS_ROOT, DICOM_ROOT, OUTPUT_ROOT)
