"""
Central configuration for the dataset indexing and preprocessing pipeline.
All shared constants and default CLI flags live here.
"""
import logging
import re

#------------------------------------------------------------------------------
# Base directories for indexing and processed outputs
#------------------------------------------------------------------------------
# PREPROCESSING_LOG_DIR = "/data/PreprocessingLogs"
# BASE_UNPROC           = "/data/research/"
# BASE_PROC             = "/data/Preprocessed/"
# GROUPS_DIR            = "/data/DatasetIndexes/Groups"
# INDEX_DIR             = "/data/DatasetIndexes"

PREPROCESSING_LOG_DIR = "F:/PreprocessingLogs"
BASE_UNPROC           = "F:/Datasets/"
BASE_PROC             = "F:/Preprocessed/"
GROUPS_DIR            = "F:/DatasetIndexes/Groups"
INDEX_DIR             = "F:/DatasetIndexes"
CSV_FILENAME          = "datasets.csv"

#-------------------------------------------------------------------------------
# Default preprocessing settings
#-------------------------------------------------------------------------------
# Target image size (H, W)
DEFAULT_TARGET_SIZE = (512, 512)
DEFAULT_NORMALIZATION = {
    "percentile_min": 1,
    "percentile_max": 99,
    "do_zscore": True
}
# Minimum mask component size (in pixels)
MIN_COMPONENT_SIZE = 100

#-------------------------------------------------------------------------------
# File extensions
#-------------------------------------------------------------------------------
VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
} 
VOLUME_EXTS = {
    ".nii", ".nii.gz", ".nrrd", ".mhd", ".mha", ".img", ".hdr", ".dicom", ".dcm", ".npy"
}

#-------------------------------------------------------------------------------
# Regexes
#-------------------------------------------------------------------------------
_CLASS_RE = re.compile(r'%([^%]+)%')
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
_DIGIT_RE = re.compile(r"^\d+$")

#-------------------------------------------------------------------------------
# CLI Defaults
#-------------------------------------------------------------------------------
GPU_ENABLED = False

#-------------------------------------------------------------------------------
# Logging
#-------------------------------------------------------------------------------
DEFAULT_LOG_LEVEL = logging.INFO