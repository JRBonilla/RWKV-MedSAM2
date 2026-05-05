"""
Central configuration for the dataset indexing and preprocessing pipeline.

User-facing defaults live in defaults.ini. This module loads the packaged
defaults, optional user overrides, and exports typed constants for existing
DRIPP code.
"""
import configparser
import logging
import os
import re
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "defaults.ini"
LOCAL_CONFIG_FILENAME = "dripp.ini"
ENV_CONFIG_VAR = "DRIPP_CONFIG"

SUPPORTED_IMAGE_OUTPUT_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}
SUPPORTED_VOLUME_OUTPUT_EXTS = {
    ".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"
}


CONFIG_FILES_LOADED = []


def _read_config():
    parser = configparser.ConfigParser()
    read_files = parser.read(DEFAULT_CONFIG_PATH)
    if not read_files:
        raise FileNotFoundError(f"Missing DRIPP default config: {DEFAULT_CONFIG_PATH}")
    CONFIG_FILES_LOADED.extend(read_files)

    env_path = os.environ.get(ENV_CONFIG_VAR)
    if env_path:
        read_files = parser.read(env_path)
        if not read_files:
            raise FileNotFoundError(f"{ENV_CONFIG_VAR} points to a missing or unreadable file: {env_path}")
        CONFIG_FILES_LOADED.extend(read_files)

    local_path = Path.cwd() / LOCAL_CONFIG_FILENAME
    if local_path.exists():
        read_files = parser.read(local_path)
        if not read_files:
            raise FileNotFoundError(f"Could not read local DRIPP config: {local_path}")
        CONFIG_FILES_LOADED.extend(read_files)

    return parser


def _split_csv(value):
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _normalize_ext(value):
    value = value.strip().lower()
    if not value:
        raise ValueError("File extension values cannot be empty.")
    return value if value.startswith(".") else f".{value}"


def _ext_set(parser, section, option):
    return {_normalize_ext(item) for item in _split_csv(parser.get(section, option))}


def _parse_target_size(value):
    parts = _split_csv(value)
    if len(parts) != 2:
        raise ValueError("preprocessing.target_size must contain exactly two integers, e.g. 512,512")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("preprocessing.target_size must contain integers, e.g. 512,512") from exc


def _parse_log_level(value):
    value = value.strip()
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported logging.default_log_level: {value!r}")
    return level


def _output_ext(parser, option, supported):
    ext = _normalize_ext(parser.get("output_formats", option))
    if ext not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported output format for output_formats.{option}: {ext}. Allowed: {allowed}")
    return ext


def _fmt_exts(values):
    return ", ".join(sorted(values))


_CONFIG = _read_config()

# ------------------------------------------------------------------------------
# Base directories for indexing and processed outputs
# ------------------------------------------------------------------------------
PREPROCESSING_LOG_DIR = _CONFIG.get("paths", "preprocessing_log_dir")
BASE_UNPROC = _CONFIG.get("paths", "base_unproc")
BASE_PROC = _CONFIG.get("paths", "base_proc")
GROUPS_DIR = _CONFIG.get("paths", "groups_dir")
INDEX_DIR = _CONFIG.get("paths", "index_dir")
CSV_FILENAME = _CONFIG.get("paths", "csv_filename")

# -------------------------------------------------------------------------------
# Default preprocessing settings
# -------------------------------------------------------------------------------
DEFAULT_TARGET_SIZE = _parse_target_size(_CONFIG.get("preprocessing", "target_size"))
DEFAULT_NORMALIZATION = {
    "percentile_min": _CONFIG.getfloat("preprocessing", "normalization_percentile_min"),
    "percentile_max": _CONFIG.getfloat("preprocessing", "normalization_percentile_max"),
    "do_zscore": _CONFIG.getboolean("preprocessing", "normalization_do_zscore"),
}
MIN_COMPONENT_SIZE = _CONFIG.getint("preprocessing", "min_component_size")

# -------------------------------------------------------------------------------
# File extensions
# -------------------------------------------------------------------------------
VIDEO_EXTS = _ext_set(_CONFIG, "input_extensions", "video")
VOLUME_EXTS = _ext_set(_CONFIG, "input_extensions", "volume")

OUTPUT_FORMATS = {
    "2d_image": _output_ext(_CONFIG, "2d_image_format", SUPPORTED_IMAGE_OUTPUT_EXTS),
    "2d_mask": _output_ext(_CONFIG, "2d_mask_format", SUPPORTED_IMAGE_OUTPUT_EXTS),
    "3d_image": _output_ext(_CONFIG, "3d_image_format", SUPPORTED_VOLUME_OUTPUT_EXTS),
    "3d_mask": _output_ext(_CONFIG, "3d_mask_format", SUPPORTED_VOLUME_OUTPUT_EXTS),
    "video_frame": _output_ext(_CONFIG, "video_frame_format", SUPPORTED_IMAGE_OUTPUT_EXTS),
    "video_mask": _output_ext(_CONFIG, "video_mask_format", SUPPORTED_IMAGE_OUTPUT_EXTS),
}
OUTPUT_EXTS = set(OUTPUT_FORMATS.values())
IMAGE_OUTPUT_EXTS = {ext for ext in OUTPUT_EXTS if ext in SUPPORTED_IMAGE_OUTPUT_EXTS}
VOLUME_OUTPUT_EXTS = {ext for ext in OUTPUT_EXTS if ext in SUPPORTED_VOLUME_OUTPUT_EXTS}

# -------------------------------------------------------------------------------
# Regexes
# -------------------------------------------------------------------------------
_CLASS_RE = re.compile(r'%([^%]+)%')
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
_DIGIT_RE = re.compile(r"^\d+$")

# -------------------------------------------------------------------------------
# CLI Defaults
# -------------------------------------------------------------------------------
GPU_ENABLED = _CONFIG.getboolean("runtime", "gpu_enabled")

# -------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------
DEFAULT_LOG_LEVEL = _parse_log_level(_CONFIG.get("logging", "default_log_level"))


def get_config_summary():
    """
    Return the active DRIPP configuration as a simple dictionary.
    """
    return {
        "config_files_loaded": list(CONFIG_FILES_LOADED),
        "paths": {
            "preprocessing_log_dir": PREPROCESSING_LOG_DIR,
            "base_unproc": BASE_UNPROC,
            "base_proc": BASE_PROC,
            "groups_dir": GROUPS_DIR,
            "index_dir": INDEX_DIR,
            "csv_filename": CSV_FILENAME,
        },
        "preprocessing": {
            "target_size": DEFAULT_TARGET_SIZE,
            "normalization": DEFAULT_NORMALIZATION,
            "min_component_size": MIN_COMPONENT_SIZE,
        },
        "input_extensions": {
            "video": sorted(VIDEO_EXTS),
            "volume": sorted(VOLUME_EXTS),
        },
        "output_formats": dict(OUTPUT_FORMATS),
        "runtime": {
            "gpu_enabled": GPU_ENABLED,
        },
        "logging": {
            "default_log_level": logging.getLevelName(DEFAULT_LOG_LEVEL),
        },
    }


def _default_config_template():
    return f"""# DRIPP local configuration
# Save this file as dripp.ini in your project directory, or point DRIPP_CONFIG
# at it. DRIPP loads packaged defaults first, then DRIPP_CONFIG, then local
# dripp.ini.

[paths]
# Root folder that contains datasets.csv and one folder per dataset.
preprocessing_log_dir = {PREPROCESSING_LOG_DIR}
base_unproc = {BASE_UNPROC}

# Preprocessed outputs and dataset_manager.pkl are written under this folder.
base_proc = {BASE_PROC}

# Indexes, grouping JSON, CT stats, logs, and task summaries live here.
groups_dir = {GROUPS_DIR}
index_dir = {INDEX_DIR}
csv_filename = {CSV_FILENAME}

[preprocessing]
# Two integers: height, width.
target_size = {DEFAULT_TARGET_SIZE[0]}, {DEFAULT_TARGET_SIZE[1]}
normalization_percentile_min = {DEFAULT_NORMALIZATION["percentile_min"]}
normalization_percentile_max = {DEFAULT_NORMALIZATION["percentile_max"]}
normalization_do_zscore = {str(DEFAULT_NORMALIZATION["do_zscore"]).lower()}
min_component_size = {MIN_COMPONENT_SIZE}

[input_extensions]
# Comma-separated extensions used when indexing raw datasets.
video = {_fmt_exts(VIDEO_EXTS)}
volume = {_fmt_exts(VOLUME_EXTS)}

[output_formats]
# 2D/video formats: {_fmt_exts(SUPPORTED_IMAGE_OUTPUT_EXTS)}
# 3D formats: {_fmt_exts(SUPPORTED_VOLUME_OUTPUT_EXTS)}
2d_image_format = {OUTPUT_FORMATS["2d_image"]}
2d_mask_format = {OUTPUT_FORMATS["2d_mask"]}
3d_image_format = {OUTPUT_FORMATS["3d_image"]}
3d_mask_format = {OUTPUT_FORMATS["3d_mask"]}
video_frame_format = {OUTPUT_FORMATS["video_frame"]}
video_mask_format = {OUTPUT_FORMATS["video_mask"]}

[runtime]
gpu_enabled = {str(GPU_ENABLED).lower()}

[logging]
# Use a standard Python logging level, such as DEBUG, INFO, WARNING, or ERROR.
default_log_level = {logging.getLevelName(DEFAULT_LOG_LEVEL)}
"""


def write_default_config(path, overwrite=False):
    """
    Write a commented DRIPP config template to path.

    Args:
        path (str or os.PathLike): Destination path.
        overwrite (bool): Whether to replace an existing file.

    Returns:
        str: The written config path.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_default_config_template(), encoding="utf-8")
    return str(target)
