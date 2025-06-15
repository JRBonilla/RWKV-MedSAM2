import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw

# CONFIGURATION
# INPUT_DIR   = Path(r"F:\Datasets\DDTI\DDTI")
INPUT_DIR   = Path(r"DDTI")
OUTPUT_ROOT = Path("extracted_ddti")
OUTPUT_ROOT.mkdir(exist_ok=True)

# Incomplete annotations that can be salvaged by manually editing the XML file
SALVAGE = {
    "197.xml": 'pe": "freehand"}]',     # completes '"regionType": "freehand"}]'
    "205.xml": 'ype": "freehand"}]'     # completes '"regionType": "freehand"}]'
}

# Keeps track of skipped files
skip_details = []

for xml_path in INPUT_DIR.glob("*.xml"):
    base = xml_path.stem  # e.g. "1", "100", etc.

    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        skip_details.append((xml_path.name, f"XML parse error: {e}"))
        continue

    # Prepare output directories
    case_out   = OUTPUT_ROOT / base
    images_out = case_out / "images"
    masks_out  = case_out / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out .mkdir(parents=True, exist_ok=True)

    # Find all <mark> blocks
    marks = root.findall('.//mark')
    if not marks:
        skip_details.append((xml_path.name, "no <mark> elements"))
        continue

    for mark in marks:
        # Pull out the image index
        img_idx = mark.findtext('image')
        if not img_idx:
            skip_details.append((xml_path.name, "missing <image> tag in a <mark>"))
            continue
        img_idx = img_idx.strip()

        # Pull out the raw JSON from <svg>...</svg>
        svg_elem = mark.find('svg')
        if svg_elem is None or not (svg_elem.text and svg_elem.text.strip()):
            skip_details.append((xml_path.name, f"empty or missing <svg> for image {img_idx}"))
            continue
        svg_text = svg_elem.text.strip()

        # Salvage incomplete annotations
        if (xml_path.name in SALVAGE) and (not svg_text.endswith(']')):
            svg_text += SALVAGE[xml_path.name]

        # Parse region list
        try:
            regions = json.loads(svg_text)
        except json.JSONDecodeError as e:
            skip_details.append((xml_path.name,
                f"JSON parse error in <svg> for image {img_idx}: {e.msg}"))
            continue

        # Copy only the referenced image
        src_img = None
        for ext in ('jpg', 'png'):
            candidate = INPUT_DIR / f"{base}_{img_idx}.{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            skip_details.append((xml_path.name, f"source image {base}_{img_idx}.* not found"))
            continue

        shutil.copy(src_img, images_out / src_img.name)

        # Generate one mask per region
        with Image.open(images_out / src_img.name) as ref:
            for region_idx, region in enumerate(regions, start=1):
                pts = [(pt["x"], pt["y"]) for pt in region.get("points", [])]
                if not pts:
                    continue

                mask = Image.new("L", ref.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(pts, outline=1, fill=1)
                # convert to 0/255
                mask = mask.point(lambda p: 255 if p else 0)

                out_name = f"{base}_{img_idx}_{region_idx}_mask.png"
                mask.save(masks_out / out_name)
                print(f"Saved mask: {base}/masks/{out_name}")

# Report skipped files and exit
if skip_details:
    print("\nSkipped the following XMLs:")
    for fname, reason in skip_details:
        print(f" - {fname}: {reason}")
else:
    print("\nAll XML files processed successfully.")
