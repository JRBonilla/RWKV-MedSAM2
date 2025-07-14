"""
DRIPP: Dataset Regex Indexing & Preprocessing Pipeline
"""

from .dataset       import SegmentationDataset, DatasetManager
from .preprocessor  import Preprocessor
from .helpers       import *
from .config        import *
from .indexer       import (
    index_dataset,
    determine_pipeline,
    compute_group_key,
    build_groupings,
)

__version__ = "0.1.0"