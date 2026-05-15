"""Output folder structure rendering for DRIPP preprocessing."""

import os
import re
import string


DEFAULT_GROUP_FOLDER_TEMPLATE = "{dataset}/{modality}/{subdataset}/{split}/{id_parts}"
DEFAULT_IMAGES_FOLDER = "images"
DEFAULT_MASKS_FOLDER = "masks"

SUPPORTED_OUTPUT_STRUCTURE_TOKENS = {
    "dataset",
    "modality",
    "subdataset",
    "split",
    "id",
    "id_parts",
}


class OutputStructureError(ValueError):
    """Raised when an output structure setting cannot be rendered safely."""


def _path_parts(value):
    return [part for part in re.split(r"[\\/]+", str(value).strip()) if part]


def _normalize_relative_path(parts):
    return os.path.normpath(os.path.join(*parts)).replace("\\", "/") if parts else ""


def _validate_relative_parts(parts, label):
    if not parts:
        raise OutputStructureError(f"{label} cannot be empty.")
    for part in parts:
        if part in {".", ".."}:
            raise OutputStructureError(f"{label} cannot contain '.' or '..' path segments.")
        if os.path.isabs(part):
            raise OutputStructureError(f"{label} must be relative.")


def validate_leaf_folder(value, label):
    """
    Validate and normalize an image/mask leaf folder name.

    Args:
        value (str): Folder setting value.
        label (str): User-facing field name for errors.

    Returns:
        str: Normalized folder name.
    """
    raw = str(value or "").strip()
    if not raw:
        raise OutputStructureError(f"{label} cannot be empty.")
    if "{" in raw or "}" in raw:
        raise OutputStructureError(f"{label} cannot contain template tokens.")
    if os.path.isabs(raw):
        raise OutputStructureError(f"{label} must be relative.")

    parts = _path_parts(raw)
    _validate_relative_parts(parts, label)
    if len(parts) != 1:
        raise OutputStructureError(f"{label} must be a single folder name.")
    return parts[0]


def validate_group_folder_template(template):
    """
    Validate and normalize the configurable group folder template.

    Args:
        template (str): Template containing supported ``{token}`` fields.

    Returns:
        str: Normalized template using forward slashes.
    """
    raw = str(template or "").strip()
    if not raw:
        raise OutputStructureError("Group Folder Template cannot be empty.")
    if os.path.isabs(raw):
        raise OutputStructureError("Group Folder Template must be relative.")
    if re.search(r"(^|[\\/])($|[\\/])", raw):
        raise OutputStructureError("Group Folder Template cannot contain empty path segments.")

    formatter = string.Formatter()
    try:
        parsed = list(formatter.parse(raw))
    except ValueError as exc:
        raise OutputStructureError(f"Invalid Group Folder Template: {exc}") from exc

    for _literal, field_name, _format_spec, _conversion in parsed:
        if field_name is None:
            continue
        if _format_spec or _conversion:
            raise OutputStructureError("Group Folder Template tokens cannot use format specifiers.")
        if field_name not in SUPPORTED_OUTPUT_STRUCTURE_TOKENS:
            allowed = ", ".join(f"{{{token}}}" for token in sorted(SUPPORTED_OUTPUT_STRUCTURE_TOKENS))
            raise OutputStructureError(
                f"Unknown token '{{{field_name}}}' in Group Folder Template. Supported tokens: {allowed}."
            )

    parts = _path_parts(raw)
    _validate_relative_parts(parts, "Group Folder Template")
    return "/".join(parts)


def _render_template_segment(segment, context):
    try:
        rendered = segment.format(**context)
    except KeyError as exc:
        raise OutputStructureError(f"Unknown token '{{{exc.args[0]}}}' in Group Folder Template.") from exc
    if not rendered.strip():
        return []
    return _path_parts(rendered)


def _id_parts(composite_id, split):
    parts = str(composite_id).split("_")
    if parts and parts[0] == split:
        parts = parts[1:]
    return parts or [str(composite_id)]


def render_output_dirs(
    base_proc,
    dataset_name,
    entry,
    modality,
    subdataset_name,
    composite_id,
    group_folder_template=DEFAULT_GROUP_FOLDER_TEMPLATE,
    images_folder=DEFAULT_IMAGES_FOLDER,
    masks_folder=DEFAULT_MASKS_FOLDER,
):
    """
    Render image and mask output directories for a grouped preprocessing entry.

    Args:
        base_proc (str): Root processed-output directory.
        dataset_name (str): Dataset name.
        entry (dict): Group entry with split and identifier fields.
        modality (str): Resolved modality for this group.
        subdataset_name (str | None): Resolved subdataset name.
        composite_id (str): Composite identifier used for output filenames.
        group_folder_template (str): Relative group folder template.
        images_folder (str): Image output leaf folder.
        masks_folder (str): Mask output leaf folder.

    Returns:
        dict: ``group_dir``, ``img_out_dir``, and ``mask_out_dir`` paths.
    """
    template = validate_group_folder_template(group_folder_template)
    image_leaf = validate_leaf_folder(images_folder, "Images Folder")
    mask_leaf = validate_leaf_folder(masks_folder, "Masks Folder")

    split = str(entry["split"])
    identifier = str(entry.get("identifier") or composite_id)
    context = {
        "dataset": str(dataset_name),
        "modality": str(modality or "default"),
        "subdataset": str(subdataset_name or ""),
        "split": split,
        "id": identifier,
        "id_parts": "/".join(_id_parts(composite_id, split)),
    }

    rendered_parts = []
    for segment in template.split("/"):
        rendered_parts.extend(_render_template_segment(segment, context))
    _validate_relative_parts(rendered_parts, "Rendered Group Folder Template")

    group_rel = _normalize_relative_path(rendered_parts)
    group_dir = os.path.normpath(os.path.join(base_proc, group_rel)).replace("\\", "/")
    return {
        "group_dir": group_dir,
        "img_out_dir": os.path.normpath(os.path.join(group_dir, image_leaf)).replace("\\", "/"),
        "mask_out_dir": os.path.normpath(os.path.join(group_dir, mask_leaf)).replace("\\", "/"),
    }


def render_preview(
    group_folder_template,
    images_folder,
    masks_folder,
    dataset_name="ExampleCT",
    modality="ct",
    subdataset_name="default",
    split="train",
    identifier="case001",
):
    """
    Render example image and mask paths for settings preview text.
    """
    dirs = render_output_dirs(
        "",
        dataset_name,
        {"split": split, "identifier": identifier},
        modality,
        subdataset_name,
        f"{split}_{identifier}",
        group_folder_template,
        images_folder,
        masks_folder,
    )
    return dirs["img_out_dir"].lstrip("/"), dirs["mask_out_dir"].lstrip("/")
