#!/usr/bin/env python3
import os
import re
import pandas as pd

# ----------------- USER CONFIGURATION -----------------
CSV_FILE = "scd_patientdata.csv"         # Input CSV filename.
DICOM_ROOT = "./"                        # Root directory where patient folders reside.
OUTPUT_CSV = "scd_patientdata_updated.csv"  # Updated CSV output filename.
# ------------------------------------------------------

# Regex to match an OriginalID of the forms:
# SC-HF-I-X, SC-HF-NI-X, SC-HYP-X, SC-N-X (where X can be one or two digits)
orig_id_pattern = re.compile(r'^(SC-(?:HF-I|HF-NI|HYP|N)-)(\d+)$', re.IGNORECASE)
# Regex to match a CINESAX folder, e.g., CINESAX_1, CINESAX_123, etc.
cinesax_folder_pattern = re.compile(r'^CINESAX_\d+$')
# Regex to match a DICOM filename of the form IM-[Series Number]-[Other Number].dcm
dicom_filename_pattern = re.compile(r'^IM-(\d{4})-(\d{4})\.dcm$', re.IGNORECASE)

def pad_original_id(original_id):
    """
    Ensure that the trailing number in OriginalID is formatted as two digits.
    e.g. "SC-HF-I-1" becomes "SC-HF-I-01", while "SC-HF-I-40" remains unchanged.
    """
    m = orig_id_pattern.match(original_id)
    if m:
        prefix, num = m.group(1), m.group(2)
        padded_num = f"{int(num):02d}"
        return prefix + padded_num
    return original_id

def choose_cinesax_folder(patient_id, dicom_root):
    """
    For a given patient, look into their folder and return the name (not full path)
    of the CINESAX folder that contains the most DICOM files.
    If no CINESAX folders are found, return None.
    """
    patient_folder = os.path.join(dicom_root, patient_id)
    if not os.path.isdir(patient_folder):
        print(f"Patient folder not found: {patient_folder}")
        return None
    candidate_folders = []
    for entry in os.listdir(patient_folder):
        entry_path = os.path.join(patient_folder, entry)
        if os.path.isdir(entry_path) and cinesax_folder_pattern.match(entry):
            # Count number of .dcm files inside this folder.
            file_count = sum(1 for f in os.listdir(entry_path) if f.lower().endswith(".dcm"))
            candidate_folders.append((entry, file_count))
    if not candidate_folders:
        return None
    # Return the folder with the maximum count.
    candidate_folders.sort(key=lambda x: x[1], reverse=True)
    return candidate_folders[0][0]

def extract_sax_series_from_folder(patient_id, cinesax_folder, dicom_root):
    """
    From the specified CINESAX folder for a patient, look for a DICOM file whose
    name matches "IM-[Series Number]-[Other Number].dcm" and extract the series number.
    Returns the series number (as a string) or None if not found.
    """
    folder_path = os.path.join(dicom_root, patient_id, cinesax_folder)
    if not os.path.isdir(folder_path):
        return None
    for file in os.listdir(folder_path):
        m = dicom_filename_pattern.match(file)
        if m:
            series_number = m.group(1)
            return series_number
    return None

def update_csv(csv_file, dicom_root, output_csv):
    # Load the CSV and force strings to preserve any formatting.
    df = pd.read_csv(csv_file, dtype=str)
    # Ensure necessary columns exist.
    if not {"PatientID", "OriginalID"}.issubset(df.columns):
        print("CSV must contain 'PatientID' and 'OriginalID' columns.")
        return

    new_original_ids = []
    sax_series_list = []
    cinesax_folder_list = []

    for idx, row in df.iterrows():
        orig_id = str(row["OriginalID"]).strip()
        patient_id = str(row["PatientID"]).strip()
        
        # Fix the OriginalID: pad the trailing number to two digits.
        fixed_orig_id = pad_original_id(orig_id)
        new_original_ids.append(fixed_orig_id)
        
        # Find the best CINESAX folder for the patient.
        chosen_folder = choose_cinesax_folder(patient_id, dicom_root)
        if chosen_folder is None:
            print(f"No CINESAX folder found for patient {patient_id}")
            sax_series_list.append("")
            cinesax_folder_list.append("")
            continue
        
        cinesax_folder_list.append(chosen_folder)
        # Extract the SAX_Series from one DICOM file in that folder.
        sax_series = extract_sax_series_from_folder(patient_id, chosen_folder, dicom_root)
        if sax_series is None:
            print(f"No DICOM file matching pattern found in {patient_id}/{chosen_folder}")
            sax_series_list.append("")
        else:
            sax_series_list.append(sax_series)

    # Update the DataFrame.
    df["OriginalID"] = new_original_ids
    df["SAX_Series"] = sax_series_list
    df["CINESAX_Folder"] = cinesax_folder_list

    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

if __name__ == "__main__":
    update_csv(CSV_FILE, DICOM_ROOT, OUTPUT_CSV)
