"""
Output filename rendering for DRIPP preprocessing.
"""

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
    """
    Raised when an output filename setting cannot be rendered safely.
    """


def validate_filename_separator(value):
    """
    Validate and normalize a filename separator.

    Args:
        value (str): Setting or cell value to normalize, validate, or format.

    Returns:
        str: Stripped separator containing only hyphens, underscores, or periods,
            possibly empty.
    """
    separator = str(value if value is not None else "").strip()
    if separator == "":
        return ""
    if not re.fullmatch(r"[-_.]+", separator):
        raise OutputFilenameError(
            "Filename Separator may only contain '-', '_', or '.'."
        )

    return separator


def parse_filename_segments(value, allowed_segments, label):
    """
    Parse a comma-separated segment list.

    Args:
        value (str | Sequence[str]): Comma-separated tags or an ordered sequence of
            filename tags.
        allowed_segments (set[str]): Filename tag names permitted in the selected
            builder or output mode.
        label (str): Display label used in the editor or validation error messages.

    Returns:
        list[str]: Unique supported tag names in their configured order.
    """
    if isinstance(value, (list, tuple)):
        segments = [str(item).strip() for item in value if str(item).strip()]
    else:
        segments = [
            part.strip()
            for part in str(value or "").replace("\n", ",").split(",")
            if part.strip()
        ]
    if not segments:
        raise OutputFilenameError(f"{label} must include at least one filename tag.")
    if len(set(segments)) != len(segments):
        raise OutputFilenameError(f"{label} cannot contain duplicate filename tags.")

    unknown = [segment for segment in segments if segment not in allowed_segments]
    if unknown:
        allowed = ", ".join(sorted(allowed_segments))
        raise OutputFilenameError(
            f"{label} contains unsupported tags: {', '.join(unknown)}. Allowed: {allowed}."
        )

    return segments


def validate_output_filenames(image_segments, mask_segments, separator):
    """
    Validate the configured filename settings.

    Args:
        image_segments (str | list[str] | None): Ordered image filename tags, or None to
            use the default tag order.
        mask_segments (str | list[str] | None): Ordered mask filename tags, or None to
            use the default tag order.
        separator (str): Punctuation inserted between rendered filename tags.

    Returns:
        dict: Normalized image tags, mask tags, and separator.
    """
    image = parse_filename_segments(
        image_segments, IMAGE_FILENAME_SEGMENTS, "Image Filename Tags"
    )
    mask = parse_filename_segments(
        mask_segments, MASK_FILENAME_SEGMENTS, "Mask Filename Tags"
    )
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
    """
    Format an integer with a prefix and zero padding.

    Args:
        prefix (str): Text preceding the zero-padded numeric filename token.
        value (int): Integer to format with the prefix and zero padding.
        pad (int): Minimum width of the zero-padded integer portion.

    Returns:
        str: Prefixed, zero-padded numeric filename token.
    """
    return f"{prefix}{int(value):0{pad}d}"


def _image_number(mode, idx):
    """
    Format an image or video-frame number for the selected output mode.

    Args:
        mode (str): Preprocessing output mode, either 2d, 3d, or video.
        idx (int): Image or frame index to display or include in a filename.

    Returns:
        str: Image token padded to three digits or frame token padded to four digits.
    """
    token, pad = {
        "2d": ("img", 3),
        "3d": ("img", 3),
        "video": ("frame", 4),
    }[mode]

    return _number(token, idx, pad)


def _mask_number(mode, idx):
    """
    Format a mask number for the selected output mode.

    Args:
        mode (str): Preprocessing output mode, either 2d, 3d, or video.
        idx (int): Image or frame index to display or include in a filename.

    Returns:
        str: Mask token with mode-specific zero padding.
    """
    pad = 4 if mode == "video" else 3

    return _number("mask", idx, pad)


def _component_number(mode, idx):
    """
    Format a connected-component number for the selected output mode.

    Args:
        mode (str): Preprocessing output mode, either 2d, 3d, or video.
        idx (int): Image or frame index to display or include in a filename.

    Returns:
        str: Component token with mode-specific zero padding.
    """
    pad = 4 if mode == "video" else 3

    return _number("comp", idx, pad)


def _source_tag(tag):
    """
    Wrap a nonempty source tag in filename delimiters.

    Args:
        tag (str | None): Optional source tag to embed between filename delimiters.

    Returns:
        str: Tilde-delimited source tag, or an empty string when absent.
    """
    return f"~{tag}~" if tag else ""


def _render_segments(segments, context, separator):
    """
    Render supported tags and reject invalid filesystem characters.

    Args:
        segments (list[str]): Ordered filename tag names or relative path segments.
        context (dict[str, str]): Resolved placeholder values used to render the
            template or filename.
        separator (str): Punctuation inserted between rendered filename tags.

    Returns:
        str: Nonempty filename stem joined with the configured separator.
    """
    rendered = []
    for segment in segments:
        value = context.get(segment, "")
        if value not in (None, ""):
            rendered.append(str(value))
    if not rendered:
        raise OutputFilenameError("Rendered filename stem cannot be empty.")

    stem = separator.join(rendered)
    if any(char in stem for char in '/\\:*?"<>|'):
        raise OutputFilenameError(
            f"Rendered filename contains invalid filesystem characters: {stem}"
        )

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

    Args:
        mode (str): Preprocessing output mode, either 2d, 3d, or video.
        idx (int): Image or frame index to display or include in a filename.
        short_id (str): Stable compact identifier for the source group.
        extension (str): File extension, including its leading dot.
        image_tag (str | None): Optional image source or modality tag included in the
            filename.
        image_segments (str | list[str] | None): Ordered image filename tags, or None to
            use the default tag order.
        separator (str): Punctuation inserted between rendered filename tags.

    Returns:
        str: Image filename built from the ordered tag values and extension.
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

    Args:
        mode (str): Preprocessing output mode, either 2d, 3d, or video.
        img_idx (int): Index of the processed image to which the mask belongs.
        mask_idx (int): Mask index within the current image.
        comp_idx (int): Connected-component index within the current mask.
        short_id (str): Stable compact identifier for the source group.
        extension (str): File extension, including its leading dot.
        class_tag (str): Target class label recorded for the mask.
        mask_tag (str | None): Optional mask source tag included in the filename.
        label_value (int | None): Optional integer class label included in the
            volume-mask filename.
        mask_segments (str | list[str] | None): Ordered mask filename tags, or None to
            use the default tag order.
        separator (str): Punctuation inserted between rendered filename tags.

    Returns:
        str: Mask filename identifying the image, mask, class, label, or component as
            configured.
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
        "label_value": (
            _number("label", label_value, 3) if label_value is not None else ""
        ),
        "component_number": _component_number(mode, comp_idx),
    }

    return f"{_render_segments(segments, context, sep)}{extension}"


def render_filename_preview(image_segments, mask_segments, separator):
    """
    Render example image and mask filenames for Settings preview text.

    Args:
        image_segments (str | list[str] | None): Ordered image filename tags, or None to
            use the default tag order.
        mask_segments (str | list[str] | None): Ordered mask filename tags, or None to
            use the default tag order.
        separator (str): Punctuation inserted between rendered filename tags.

    Returns:
        tuple[str, str, str]: Example image, raster-mask, and volume-mask filenames.
    """
    return (
        render_image_filename(
            "2d", 3, "a1b2c3d4", ".png", "modality0", image_segments, separator
        ),
        render_mask_filename(
            "2d",
            3,
            2,
            5,
            "a1b2c3d4",
            ".png",
            "tumor",
            "lesionA",
            None,
            mask_segments,
            separator,
        ),
        render_mask_filename(
            "3d",
            3,
            2,
            5,
            "a1b2c3d4",
            ".nii.gz",
            "tumor",
            None,
            7,
            mask_segments,
            separator,
        ),
    )
