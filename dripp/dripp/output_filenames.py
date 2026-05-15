"""Output filename rendering for DRIPP preprocessing."""

import re


DEFAULT_FILENAME_SEPARATOR = "_"
DEFAULT_IMAGE_FILENAME_SEGMENTS = ["short_id", "image_number", "source_tag"]
DEFAULT_MASK_FILENAME_SEGMENTS = [
    "short_id",
    "image_number",
    "source_tag",
    "mask_number",
    "class_name",
    "label_value",
    "component_number",
]

FILENAME_SEGMENT_LABELS = {
    "short_id": "Short ID",
    "image_number": "Image Number",
    "source_tag": "Source Tag",
    "mask_number": "Mask Number",
    "class_name": "Class",
    "label_value": "Label",
    "component_number": "Component",
}

SUPPORTED_FILENAME_SEGMENTS = set(FILENAME_SEGMENT_LABELS)
IMAGE_FILENAME_SEGMENTS = {"short_id", "image_number", "source_tag"}
MASK_FILENAME_SEGMENTS = set(DEFAULT_MASK_FILENAME_SEGMENTS)


class OutputFilenameError(ValueError):
    """Raised when an output filename setting cannot be rendered safely."""


def validate_filename_separator(value):
    """
    Validate and normalize a filename separator.
    """
    separator = str(value if value is not None else "").strip()
    if separator == "":
        return ""
    if not re.fullmatch(r"[-_.]+", separator):
        raise OutputFilenameError("Filename Separator may only contain '-', '_', or '.'.")
    return separator


def parse_filename_segments(value, allowed_segments, label):
    """
    Parse a comma-separated segment list.
    """
    if isinstance(value, (list, tuple)):
        segments = [str(item).strip() for item in value if str(item).strip()]
    else:
        segments = [part.strip() for part in str(value or "").replace("\n", ",").split(",") if part.strip()]
    if not segments:
        raise OutputFilenameError(f"{label} must include at least one filename tag.")
    if len(set(segments)) != len(segments):
        raise OutputFilenameError(f"{label} cannot contain duplicate filename tags.")
    unknown = [segment for segment in segments if segment not in allowed_segments]
    if unknown:
        allowed = ", ".join(sorted(allowed_segments))
        raise OutputFilenameError(f"{label} contains unsupported tags: {', '.join(unknown)}. Allowed: {allowed}.")
    return segments


def validate_output_filenames(image_segments, mask_segments, separator):
    """
    Validate the configured filename settings.
    """
    image = parse_filename_segments(image_segments, IMAGE_FILENAME_SEGMENTS, "Image Filename Tags")
    mask = parse_filename_segments(mask_segments, MASK_FILENAME_SEGMENTS, "Mask Filename Tags")
    sep = validate_filename_separator(separator)
    if "short_id" not in image:
        raise OutputFilenameError("Image Filename Tags must include Short ID.")
    if "short_id" not in mask:
        raise OutputFilenameError("Mask Filename Tags must include Short ID.")
    return {
        "image_segments": image,
        "mask_segments": mask,
        "separator": sep,
    }


def _number(prefix, value, pad):
    return f"{prefix}{int(value):0{pad}d}"


def _image_number(mode, idx):
    token, pad = {
        "2d": ("img", 3),
        "3d": ("img", 3),
        "video": ("frame", 4),
    }[mode]
    return _number(token, idx, pad)


def _mask_number(mode, idx):
    pad = 4 if mode == "video" else 3
    return _number("mask", idx, pad)


def _component_number(mode, idx):
    pad = 4 if mode == "video" else 3
    return _number("comp", idx, pad)


def _source_tag(tag):
    return f"~{tag}~" if tag else ""


def _render_segments(segments, context, separator):
    rendered = []
    for segment in segments:
        value = context.get(segment, "")
        if value not in (None, ""):
            rendered.append(str(value))
    if not rendered:
        raise OutputFilenameError("Rendered filename stem cannot be empty.")
    stem = separator.join(rendered)
    if any(char in stem for char in '/\\:*?"<>|'):
        raise OutputFilenameError(f"Rendered filename contains invalid filesystem characters: {stem}")
    return stem


def render_image_filename(
    mode,
    idx,
    short_id,
    extension,
    image_tag=None,
    image_segments=None,
    separator=DEFAULT_FILENAME_SEPARATOR,
):
    """
    Render an image filename with extension.
    """
    segments = parse_filename_segments(
        image_segments or DEFAULT_IMAGE_FILENAME_SEGMENTS,
        IMAGE_FILENAME_SEGMENTS,
        "Image Filename Tags",
    )
    sep = validate_filename_separator(separator)
    context = {
        "short_id": short_id,
        "image_number": _image_number(mode, idx),
        "source_tag": _source_tag(image_tag),
    }
    return f"{_render_segments(segments, context, sep)}{extension}"


def render_mask_filename(
    mode,
    img_idx,
    mask_idx,
    comp_idx,
    short_id,
    extension,
    class_tag,
    mask_tag=None,
    label_value=None,
    mask_segments=None,
    separator=DEFAULT_FILENAME_SEPARATOR,
):
    """
    Render a mask filename with extension.
    """
    segments = parse_filename_segments(
        mask_segments or DEFAULT_MASK_FILENAME_SEGMENTS,
        MASK_FILENAME_SEGMENTS,
        "Mask Filename Tags",
    )
    sep = validate_filename_separator(separator)
    context = {
        "short_id": short_id,
        "image_number": _image_number(mode, img_idx),
        "source_tag": _source_tag(mask_tag),
        "mask_number": _mask_number(mode, mask_idx),
        "class_name": f"%{class_tag}%",
        "label_value": _number("label", label_value, 3) if label_value is not None else "",
        "component_number": _component_number(mode, comp_idx),
    }
    return f"{_render_segments(segments, context, sep)}{extension}"


def render_filename_preview(image_segments, mask_segments, separator):
    """
    Render example image and mask filenames for Settings preview text.
    """
    return (
        render_image_filename("2d", 3, "a1b2c3d4", ".png", "modality0", image_segments, separator),
        render_mask_filename("2d", 3, 2, 5, "a1b2c3d4", ".png", "tumor", "lesionA", None, mask_segments, separator),
        render_mask_filename("3d", 3, 2, 5, "a1b2c3d4", ".nii.gz", "tumor", None, 7, mask_segments, separator),
    )
