import SimpleITK as sitk
import pydicom
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

def load_and_fix(vol):
    """
    Adjusts the orientation of a given volume to the LIP orientation.

    This function checks the direction cosines of the input volume and determines 
    if the orientation deviates from the desired LIP (Left, Inferior, Posterior) 
    orientation. If it's not LIP, it computes the necessary axis flips to correct 
    the orientation.

    Args:
    vol (sitk.Image): The input volume to be checked and possibly reoriented.

    Returns:
    sitk.Image: The volume with corrected orientation, if needed, otherwise the 
                original volume.
    """
    direction = vol.GetDirection()
    orienter = sitk.DICOMOrientImageFilter()
    code = orienter.GetOrientationFromDirectionCosines(direction)

    flipX = flipY = flipZ = False
    if code != "LIP":
        print(f"Invalid orientation: {code} -> flipping to LIP")
        if code == 'IRP':       # I->L  (flip Y)
            flipY = True
        elif code == 'PIR':     # P->L  (flip Z)
            flipZ = True
        elif code == 'PRS':     # P->L, R->I
            flipY = True
            flipZ = True
        elif code == 'LPS':     # S->I
            flipZ = True
        elif code == 'IAR':     # I->L, A->P
            flipY = True
            flipZ = True

    fixed = sitk.Flip(
        sitk.PermuteAxes(vol, [0, 1, 2]),
        flipAxes=[flipX, flipY, flipZ],
        flipAboutOrigin=False
    )

    # Reset to identity so the data & header agree
    fixed.SetDirection((1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0))
    return fixed

def fix_orientation_and_export(study_dir: Path, out_dir: Path):
    """
    - Reads the DICOM series in true Z‐order
    - Applies manual fix to image & mask volumes
    - Writes out <id>_image.nii.gz and <id>_mask.nii.gz
    """
    # 1) Gather & sort DICOMs by physical Z
    image_dir = study_dir / "Images"
    dcm_files = sorted(
        image_dir.glob("*.dcm"),
        key=lambda f: float(
            pydicom.dcmread(str(f), stop_before_pixels=True)
                     .ImagePositionPatient[2]
        )
    )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(f) for f in dcm_files])
    img = reader.Execute()

    # 2) Fix orientation on image
    img_fixed = load_and_fix(img)
    img_fixed.SetOrigin(img.GetOrigin())
    img_fixed.SetSpacing(img.GetSpacing())

    # Ensure output folder exists
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / f"{study_dir.name}_image.nii.gz"
    sitk.WriteImage(img_fixed, str(img_out))
    print(f"Wrote image -> {img_out}")

    # 3) Build a full‐length mask volume in same DICOM order
    mask_dir = study_dir / "Masks"
    classes = {
        "bloodpool":   1,
        "myocardium":  2,
        "epicardium":  3,
        "endocardium": 4,
    }
    # Group all PNGs by their base key "IM-XXXX-YYYY"
    grouped = defaultdict(list)
    for png in mask_dir.glob("*.png"):
        base = png.stem.rsplit("-", 1)[0]
        grouped[base].append(png)

    # Get H,W from the first slice
    ref_ds = pydicom.dcmread(str(dcm_files[0]))
    H, W = ref_ds.pixel_array.shape

    mask_slices = []
    for dcm_path in dcm_files:
        base = dcm_path.stem
        lbl = np.zeros((H, W), dtype=np.uint8)
        for png in grouped.get(base, []):
            arr = np.array(Image.open(png)) > 0
            cls = png.stem.rsplit("-", 1)[1]
            lbl[arr] = classes.get(cls, 0)
        mask_slices.append(lbl)

    mask_vol = sitk.GetImageFromArray(np.stack(mask_slices, axis=0))
    # Copy original DICOM metadata before fixing
    mask_vol.CopyInformation(img)

    # 4) Fix orientation on mask
    mask_fixed = load_and_fix(mask_vol)
    mask_fixed.SetOrigin(img_fixed.GetOrigin())
    mask_fixed.SetSpacing(img_fixed.GetSpacing())

    mask_out = out_dir / f"{study_dir.name}_mask.nii.gz"
    sitk.WriteImage(mask_fixed, str(mask_out))
    print(f"Wrote mask -> {mask_out}")

if __name__ == "__main__":
    # ROOT   = Path(r"F:\Datasets\Cardiac MRI\Merged_Cardiac_MRI")
    # OUTPUT = Path(r"F:\Datasets\Cardiac MRI\Fixed_Niftis")
    ROOT   = Path(r"Merged_Cardiac_MRI")
    OUTPUT = Path(r"Fixed_Niftis")

    for study in ROOT.iterdir():
        if (study/"Images").is_dir() and (study/"Masks").is_dir():
            print(f"\nProcessing {study.name}…")
            fix_orientation_and_export(study, OUTPUT/study.name)
