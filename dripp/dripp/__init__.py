"""
DRIPP: Dataset Regex Indexing & Preprocessing Pipeline.

The package keeps heavyweight modules lazy so command-line help and config
commands work before dataset paths have been created.
"""

from .config import *  # noqa: F401,F403

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "SegmentationDataset": ("dripp.dataset", "SegmentationDataset"),
    "DatasetManager": ("dripp.dataset", "DatasetManager"),
    "Preprocessor": ("dripp.preprocessor", "Preprocessor"),
    "index_dataset": ("dripp.indexer", "index_dataset"),
    "determine_pipeline": ("dripp.indexer", "determine_pipeline"),
    "compute_group_key": ("dripp.indexer", "compute_group_key"),
    "build_groupings": ("dripp.indexer", "build_groupings"),
    "set_indexing_log": ("dripp.helpers", "set_indexing_log"),
    "SubdatasetConfig": ("dripp.helpers", "SubdatasetConfig"),
    "normalize_path": ("dripp.helpers", "normalize_path"),
    "load_dataset_metadata": ("dripp.helpers", "load_dataset_metadata"),
    "resolve_folder": ("dripp.helpers", "resolve_folder"),
    "resolve_mask_folders": ("dripp.helpers", "resolve_mask_folders"),
    "get_extension": ("dripp.helpers", "get_extension"),
    "get_base_filename": ("dripp.helpers", "get_base_filename"),
    "get_composite_identifier": ("dripp.helpers", "get_composite_identifier"),
    "collect_files": ("dripp.helpers", "collect_files"),
    "extract_identifier_from_path": ("dripp.helpers", "extract_identifier_from_path"),
    "get_regex_configs": ("dripp.helpers", "get_regex_configs"),
    "match_subdataset_regex": ("dripp.helpers", "match_subdataset_regex"),
    "parse_mask_classes": ("dripp.helpers", "parse_mask_classes"),
    "parse_segmentation_tasks": ("dripp.helpers", "parse_segmentation_tasks"),
    "match_mask_class": ("dripp.helpers", "match_mask_class"),
    "match_tasks_for_class": ("dripp.helpers", "match_tasks_for_class"),
    "extract_mask_class": ("dripp.helpers", "extract_mask_class"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        from importlib import import_module

        attr = getattr(import_module(module_name), attr_name)
        globals()[name] = attr
        return attr

    raise AttributeError(name)


__all__ = sorted(
    name for name in globals()
    if name.isupper() or name in {"__version__"}
) + sorted(_LAZY_EXPORTS)
