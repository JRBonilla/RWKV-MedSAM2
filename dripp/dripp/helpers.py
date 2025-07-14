import os
import pandas as pd
import re
import shutil
import logging
import numpy as np
from .config import GPU_ENABLED
if GPU_ENABLED:
    import cupy as xp
else:
    xp = np
from typing import Dict, List

from .config import DEFAULT_LOG_LEVEL, _CLASS_RE, _RANGE_RE, _DIGIT_RE

# Global logger: logs to "processing_errors.log"
logger = logging.getLogger("SegmentationDatasetLogger")
logger.setLevel(DEFAULT_LOG_LEVEL)
if not logger.handlers:
    global_handler = logging.FileHandler("processing_errors.log")
    global_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    global_handler.setFormatter(global_formatter)
    logger.addHandler(global_handler)

def set_indexing_log(index_dir, dataset_name):
    """
    Reconfigure the SegmentationDataset logger so that all regex matching code
    now goes into:
        {index_dir}/Logs/{dataset_name}_indexing.log

    Args:
        index_dir (str): The directory where the index JSON files and logs are stored.
        dataset_name (str): The name of the dataset being processed.
    """
    # Remove any existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Ensure the logs folder exists
    logs_dir = os.path.join(index_dir, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Add the new handler
    handler = logging.FileHandler(os.path.join(logs_dir, f"{dataset_name}_indexing.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)

class SubdatasetConfig:
    """
    A configuration class that holds regex patterns and folder naming information
    for a subdataset.
    """
    def __init__(self, modality, name, pipeline, train_image, test_image, train_mask, test_mask):
        """
        Args:
            modality (str): modality (lowercase, e.g. "ct", "mri")
            name (str or None): subdataset name
            pipeline (str or None): one of "2D", "3D", "Video", or None for default
            train_image (str): regex for training‐image filenames
            test_image (str): regex for test‐image filenames
            train_mask (str): regex for training‐mask filenames
            test_mask (str): regex for test‐mask filenames
        """
        self.modality = modality
        self.name = name
        self.pipeline = pipeline  # NEW: store pipeline type
        self.train_image = train_image
        self.test_image = test_image
        self.train_mask = train_mask
        self.test_mask = test_mask

    def __repr__(self):
        return (
            f"SubdatasetConfig(modality='{self.modality}', name='{self.name}', pipeline='{self.pipeline}', "
            f"train_image='{self.train_image}', test_image='{self.test_image}', "
            f"train_mask='{self.train_mask}', test_mask='{self.test_mask}')"
        )

def normalize_path(path):
    """
    Normalize a file path to use forward slashes.
    
    Args:
        path (str): The file path to normalize.
    
    Returns:
        str: The normalized file path.
    """
    return os.path.normpath(path).replace("\\", "/")

def load_dataset_metadata(csv_path):
    """
    Load dataset metadata from a CSV file and return a dictionary.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        dict: A dictionary containing metadata for each dataset.
    """
    df = pd.read_csv(csv_path)
    metadata = {}
    for idx, row in df.iterrows():
        name = str(row["Dataset Name"]).strip()
        image_types = ([ext.strip() for ext in str(row["Image File Type"]).split(',')]
                       if pd.notna(row["Image File Type"]) else [])
        mask_types = ([ext.strip() for ext in str(row["Mask File Type"]).split(',')]
                      if pd.notna(row["Mask File Type"]) else [])
        root_folder = str(row["Root Folder"]).strip() if pd.notna(row["Root Folder"]) else ""
        train_folders = ([fld.strip() for fld in str(row["Train Folders"]).split(',')]
                         if "Train Folders" in df.columns and pd.notna(row["Train Folders"]) else [])
        test_folders = ([fld.strip() for fld in str(row["Test Folders"]).split(',')]
                        if "Test Folders" in df.columns and pd.notna(row["Test Folders"]) else [])
        mask_folders = ([fld.strip() for fld in str(row["Mask Folders"]).split(',')]
                        if "Mask Folders" in df.columns and pd.notna(row["Mask Folders"]) else [])
        preprocessed_str = str(row["Preprocessed?"]).strip().lower() if pd.notna(row["Preprocessed?"]) else None
        preprocessed_flag = preprocessed_str == "yes"
        mask_key = (str(row["Mask Key"]).strip() if "Mask Key" in df.columns and pd.notna(row["Mask Key"]) else None)
        grouping_strategy = (str(row["Grouping Strategy"]).strip().lower() 
                             if "Grouping Strategy" in df.columns and pd.notna(row["Grouping Strategy"])
                             else "regex")
        grouping_regex = (str(row["Grouping Regex"]).strip() 
                          if "Grouping Regex" in df.columns and pd.notna(row["Grouping Regex"])
                          else None)
        mask_classes = (str(row["Mask Classes"]).strip()
                        if "Mask Classes" in df.columns and pd.notna(row["Mask Classes"])
                        else None)
        segmentation_tasks = (str(row["Segmentation Tasks"]).strip()
                              if "Segmentation Tasks" in df.columns and pd.notna(row["Segmentation Tasks"])
                              else None)
        background_value = int(row["Background Value"]) if "Background Value" in df.columns and pd.notna(row["Background Value"]) else 0

        metadata[name] = {
            "modalities": [mod.strip() for mod in str(row["Modality"]).split(',')] if pd.notna(row["Modality"]) else [],
            "image_file_types": image_types,
            "mask_file_types": mask_types,
            "train_folders": train_folders,
            "test_folders": test_folders,
            "mask_folders": mask_folders,
            "root_directory": root_folder,
            "mask_key": mask_key,
            "grouping_strategy": grouping_strategy,
            "grouping_regex": grouping_regex,
            "background_value": background_value,
            "preprocessed": preprocessed_flag,
            "mask_classes": mask_classes,
            "segmentation_tasks": segmentation_tasks
        }
        for dataset, meta in metadata.items():
            for key, value in meta.items():
                if isinstance(value, str) and value == "":
                    meta[key] = None
    return metadata

def resolve_folder(root_folder, folder_item):
    """
    Resolve a folder path relative to a root folder. Supports wildcard searches.
    
    Args:
        root_folder (str): The root folder path.
        folder_item (str): The folder item, possibly with a wildcard.
    
    Returns:
        list: A list of resolved folder paths.
    """
    root_folder = normalize_path(root_folder.strip().strip('"').strip("'"))
    folder_item = folder_item.strip().strip('"').strip("'")
    if folder_item.startswith('*'):
        pattern = folder_item[1:]
        matching = []
        for current_dir, dirs, files in os.walk(root_folder):
            if pattern.lower() in current_dir.lower():
                matching.append(normalize_path(current_dir))
        logger.info(f"Resolved folder {folder_item} to {matching}")
        return matching
    else:
        full_path = normalize_path(os.path.join(root_folder, folder_item))
        return [full_path] if os.path.exists(full_path) else []

def resolve_mask_folders(root_folder, mask_folder_list):
    """
    Resolve mask folder paths given a list of folder names or patterns.
    Avoid duplicate resolutions.
    
    Args:
        root_folder (str): The root folder path.
        mask_folder_list (list): List of mask folder names or patterns.
    
    Returns:
        list: A sorted list of unique resolved mask folder paths.
    """
    result = set()
    root_folder = normalize_path(root_folder.strip().strip('"').strip("'"))
    for fld in mask_folder_list:
        fld = fld.strip().strip('"').strip("'")
        if fld.startswith('*'):
            pattern = fld[1:]
            for current_dir, dirs, files in os.walk(root_folder):
                if pattern.lower() in os.path.basename(current_dir).lower():
                    result.add(normalize_path(current_dir))
        else:
            for current_dir, dirs, files in os.walk(root_folder):
                if os.path.basename(current_dir).lower() == fld.lower():
                    result.add(normalize_path(current_dir))
    result = sorted(result)
    logger.info(f"Resolved mask folders {mask_folder_list} to {result}")
    return result

def get_extension(filename):
    """
    Return the file extension for a given filename.
    
    Args:
        filename (str): The file name.
    
    Returns:
        str: The file extension.
    """
    lower = filename.lower()
    if lower.endswith('.nii.gz'):
        return '.nii.gz'
    elif lower.endswith('.ppm.gz'):
        return '.ppm.gz'
    else:
        return os.path.splitext(filename)[1]

def get_base_filename(filename):
    """
    Return the base name of the file without the extension.
    
    Args:
        filename (str): The file name.
    
    Returns:
        str: The base filename.
    """
    lower = filename.lower()
    if lower.endswith('.nii.gz'):
        return filename[:-len('.nii.gz')]
    elif lower.endswith('.ppm.gz'):
        return filename[:-len('.ppm.gz')]
    else:
        return os.path.splitext(filename)[0]

def get_composite_identifier(entry):
    """
    Build a composite identifier string from the main identifier and additional identifiers.
    Includes a special marker for 'id2' if present so it can be parsed easily later.
    
    Args:
        entry (dict): The entry containing 'split', 'identifier', and optional 'additional' keys.
    
    Returns:
        str: The composite identifier.
    """
    parts = [entry["split"], entry["identifier"]]
    # Extract additional identifiers (if any)
    additional = entry.get("additional", {}) or {}
    # If 'id2' exists, append it with special delimiters for easy parsing in preprocess
    if "id2" in additional:
        parts.append(f"__ID2__{additional['id2']}__")
    # Append any other additional identifiers (excluding id2) in sorted order
    for key in sorted(additional):
        if key == "id2":
            continue
        parts.append(additional[key])
    return "_".join(parts)


def collect_files(root_folder, folder_list, allowed_extensions, is_mask=False, mask_key=None, exclude_folders=None, use_folder_identifier=False):
    """
    Collect files from a list of folders with allowed extensions, applying filters for masks and images.
    Only unique file paths are added to the results.
    
    Args:
        root_folder (str): The root directory.
        folder_list (list): List of folder paths to search.
        allowed_extensions (list): List of allowed file extensions.
        is_mask (bool): Whether to collect mask files.
        mask_key (str): Keyword to filter mask files.
        exclude_folders (list): Folders to exclude from search.
        use_folder_identifier (bool): Whether to use the folder name as identifier.
    
    Returns:
        list: A list of dictionaries with keys "id" and "path" for each file.
    """
    files = []
    seen_paths = set()

    root_folder = normalize_path(root_folder)
    search_folders = folder_list if folder_list else [root_folder]

    if exclude_folders:
        exclude_folders = [x.lower() for x in exclude_folders]

    for folder in search_folders:
        folder = normalize_path(folder)
        if not os.path.isdir(folder):
            continue

        for current_dir, dirs, filenames in os.walk(folder):
            dirs.sort()
            filenames.sort()

            if exclude_folders:
                path_components = [comp.lower() for comp in current_dir.split(os.sep)]
                if any(excl in path_components for excl in exclude_folders):
                    continue

            for fname in filenames:
                for ext in allowed_extensions:
                    ext_lower = ext.lower()
                    if ext_lower == ".nii.gz":
                        if not fname.lower().endswith(".nii.gz"):
                            continue
                        base = fname[:-len(".nii.gz")]
                    else:
                        if not fname.lower().endswith(ext_lower):
                            continue
                        base = os.path.splitext(fname)[0]

                    if mask_key:
                        has_key = re.search(re.escape(mask_key), base, re.IGNORECASE)
                        if is_mask and not has_key:
                            continue
                        if (not is_mask) and has_key:
                            continue

                    identifier = base
                    if use_folder_identifier:
                        identifier = os.path.basename(current_dir)

                    full_path = normalize_path(os.path.join(current_dir, fname))
                    if full_path not in seen_paths:
                        seen_paths.add(full_path)
                        files.append({"id": identifier, "path": full_path})
                        file_type = "mask" if is_mask else "image"
                        logger.info(f"Found {file_type}: {full_path} with identifier: {identifier}")
                    break
    return files

def extract_identifier_from_path(path, grouping_strategy, regex=None, base_dir=None):
    """
    Extract the identifier and additional identifiers from a file path using the specified grouping strategy and regex.
    
    Args:
        path (str): The file path.
        grouping_strategy (str): The grouping strategy (e.g., "regex-file", "regex-folder", "filename").
        regex (str): The regex pattern to use.
        base_dir (str): The base directory for calculating relative paths.
    
    Returns:
        tuple: (identifier (str or None), additional identifiers (dict))
    """
    # Get the relative path of the file
    path = normalize_path(path)
    relative_path = path
    if base_dir is not None and grouping_strategy in ["regex", "regex-file", "regex-folder"]:
        relative_path = os.path.relpath(path, base_dir)
        relative_path = normalize_path(relative_path)
    ident = None
    additional = {}
    if grouping_strategy == "regex-file":
        # Use the file name to extract the identifier using the regex
        target = os.path.basename(path)
        if regex:
            try:
                match = re.fullmatch(regex, target)
                if match:
                    gd = match.groupdict()
                    ident = gd.get("id")
                    additional = {k: v for k, v in gd.items() if k != "id" and v}
                else:
                    logger.warning(f"Regex pattern '{regex}' did not match target '{target}' in regex-file strategy for path '{path}'.")
                    return None, {}
            except re.error as re_err:
                logger.error(f"Regex error in regex-file: {regex} on {target}: {re_err}")
                return None, {}
        else:
            ident = target
    elif grouping_strategy == "regex-folder":
        # Use the relative path to extract the identifier using the regex
        target = relative_path
        if regex:
            try:
                match = re.fullmatch(regex, target)
                if match:
                    gd = match.groupdict()
                    ident = gd.get("id")
                    if ident is None:
                        ident = os.path.basename(os.path.dirname(target))
                    additional = {k: v for k, v in gd.items() if k != "id" and v}
                else:
                    logger.warning(f"Regex pattern '{regex}' did not match target '{target}' in regex-folder strategy for relative path '{relative_path}'.")
                    return None, {}
            except re.error as re_err:
                logger.error(f"Regex error in regex-folder: {regex} on {target}: {re_err}")
                return None, {}
        else:
            ident = os.path.basename(os.path.dirname(target))
    elif grouping_strategy == "filename":
        # Use the base name as identifier
        base_name = os.path.basename(path)
        ident = get_base_filename(base_name)
        additional = {}
    else:
        # Fallback to default: try to match the entire relative path against the regex if provided
        if regex:
            try:
                match = re.fullmatch(regex, relative_path)
                if match:
                    gd = match.groupdict()
                    ident = gd.get("id")
                    additional = {k: v for k, v in gd.items() if k != "id" and v}
                else:
                    logger.warning(f"Regex pattern '{regex}' did not match target '{relative_path}' in default strategy for path '{path}'.")
                    return None, {}
            except re.error as re_err:
                logger.error(f"Regex error in default: {regex} on {relative_path}: {re_err}")
                return None, {}
        else:
            ident = os.path.splitext(os.path.basename(path))[0]
    return ident, additional

def get_regex_configs(grouping_regex, metadata):
    """
    Parse the grouping regex string and return a list of SubdatasetConfig objects.

    There are two formats:
      - Format 1: When grouping_regex does NOT start with '['.
      - Format 2: When grouping_regex starts with '[',
                  i.e. using subdataset declarations.
    
    In Format 2, each configuration is expected to have the following structure:
      [Modality: Name: Pipeline] {images: <train_image>[, <test_image>]; masks: <train_mask>[, <test_mask>]}
    
    This implementation uses a manual parser to properly handle inner commas (e.g. in numeric ranges)
    and nested parentheses within the regex parts.
    
    Args:
      grouping_regex (str): The grouping regex string from the CSV.
      metadata (dict): The metadata dictionary for the dataset.
    
    Returns:
      list: A list of SubdatasetConfig objects.
    """
    if not grouping_regex:
        return None
    grouping_regex = grouping_regex.strip()
    configs = []
    
    # Format 2: Using subdataset declarations.
    if grouping_regex.startswith('['):
        pos = 0
        n = len(grouping_regex)
        while pos < n:
            # Find the header: look for '[' and then the corresponding ']'
            start_header = grouping_regex.find('[', pos)
            if start_header == -1:
                break
            end_header = grouping_regex.find(']', start_header)
            if end_header == -1:
                break

            # Extract the text between '[' and ']', e.g. "CT: Lungs: 2D"
            header = grouping_regex[start_header+1:end_header].strip()
            
            # Split into up to three parts: modality, name, (optional) pipeline
            header_parts = [p.strip() for p in header.split(':')]
            modality = header_parts[0]
            name = None
            pipeline = None

            if len(header_parts) >= 2:
                name = header_parts[1]
            if len(header_parts) >= 3:
                pipeline = header_parts[2]
                # Optional: Validate that pipeline is one of the allowed values
                if pipeline not in ("2D", "3D", "Video"):
                    raise ValueError(
                        f"Invalid pipeline '{pipeline}' in header '[{header}]'. "
                        f"Expected one of '2D', '3D', 'Video'."
                    )

            # Now find the body between '{' and the matching '}' after end_header
            start_body = grouping_regex.find('{', end_header)
            if start_body == -1:
                break
            # Use a counter to find the matching closing brace.
            counter = 1
            i = start_body + 1
            while i < n and counter > 0:
                if grouping_regex[i] == '{':
                    counter += 1
                elif grouping_regex[i] == '}':
                    counter -= 1
                i += 1
            if counter != 0:
                # Unmatched braces: abort parsing
                break
            body = grouping_regex[start_body+1 : i-1].strip()

            # Body format (example):
            #   images: <train_image>[, <test_image>]; masks: <train_mask>[, <test_mask>]
            if ';' in body:
                images_part, masks_part = body.split(';', 1)
            else:
                images_part = body
                masks_part = ''
            images_part = images_part.strip()
            masks_part = masks_part.strip()

            if images_part.lower().startswith("images:"):
                images_part = images_part[len("images:"):].strip()
            if masks_part.lower().startswith("masks:"):
                masks_part = masks_part[len("masks:"):].strip()

            # Helper: split on the first comma not inside parentheses
            def split_outside_parens(s):
                paren = 0
                for idx, ch in enumerate(s):
                    if ch == '(':
                        paren += 1
                    elif ch == ')':
                        paren = max(paren - 1, 0)
                    elif ch == ',' and paren == 0:
                        return s[:idx].strip(), s[idx+1:].strip()
                return s.strip(), None

            train_image, test_image = split_outside_parens(images_part)
            if not test_image:
                test_image = train_image

            train_mask, test_mask = split_outside_parens(masks_part)
            if not test_mask:
                test_mask = train_mask

            # Import SubdatasetConfig (which now expects modality, name, pipeline, train_image, ...)
            configs.append(
                SubdatasetConfig(
                    modality.strip().lower(),
                    name if name else None,
                    pipeline,          # ← new argument
                    train_image,
                    test_image,
                    train_mask,
                    test_mask
                )
            )

            # Advance past this block: move ‘pos’ to just after the ‘}’
            pos = i
            # Skip trailing commas or whitespace between blocks
            while pos < n and grouping_regex[pos] in [',', ' ']:
                pos += 1

        return configs

    # Format 1: A single set of patterns (no [ ] header)
    else:
        parts = grouping_regex.split(";")
        train_image = None
        test_image = None    
        train_mask = None
        test_mask = None
        for part in parts:
            part = part.strip()
            if part.lower().startswith("images:"):
                patterns = part[len("images:"):].split(",")
                train_image = patterns[0].strip() if patterns[0].strip() else None
                test_image = patterns[1].strip() if len(patterns) > 1 and patterns[1].strip() else train_image
            elif part.lower().startswith("masks:"):
                patterns = part[len("masks:"):].split(",")
                train_mask = patterns[0].strip() if patterns[0].strip() else None
                test_mask = patterns[1].strip() if len(patterns) > 1 and patterns[1].strip() else train_mask

        # If there's no header, default to the first modality listed in metadata
        modality = (metadata.get("modalities") or ["default"])[0]
        # In Format 1, we never provided a pipeline, so pass pipeline=None
        return [
            SubdatasetConfig(
                modality.lower(),
                None,
                None,           # pipeline = None (no header)
                train_image,
                test_image,
                train_mask,
                test_mask
            )
        ]

def match_subdataset_regex(path, grouping_strategy, subdataset_configs, is_image, split, base_dir):
    """
    Iterate over a list of SubdatasetConfig objects and attempt to extract an identifier from the given file path.

    For image files, the function uses the train_image or test_image regex from each configuration (based on the split value).
    For mask files, it uses the test_mask regex if split == "test", and otherwise uses train_mask.
    
    Args:
        path (str): The file path to match.
        grouping_strategy (str): The grouping strategy (e.g., "regex-file", "regex-folder", "filename").
        subdataset_configs (list): A list of SubdatasetConfig objects.
        is_image (bool): True if processing an image; False if processing a mask.
        split (str): 'train' or 'test'. For images, it determines which image regex to use. For masks, 'test' will use the test_mask.
        base_dir (str): The base directory for calculating relative paths.
    
    Returns:
        tuple: A tuple (identifier, additional, config) if a match is found, or (None, None, None) otherwise.
    """
    for config in subdataset_configs:
        if is_image:
            regex = config.train_image if split == "train" else config.test_image
        else:
            regex = config.train_mask if split == "train" else config.test_mask
        ident, additional = extract_identifier_from_path(
            path,
            grouping_strategy,
            regex=regex,
            base_dir=base_dir
        )
        if ident is not None:
            return ident, additional, config
    return None, None, None

def parse_mask_classes(spec):
    """
    Parse a class-mapping specification string into a nested dict.

    The spec can be in one of two forms:
      - "[Subdataset] {a=1|b=2}, [Subdataset2] {c=3|d=4}"
        returns {'Subdataset': {'a': '1', 'b': '2'},
                 'Subdataset2': {'c': '3', 'd': '4'}}
      - "a=1|b=2"
        returns {'default': {'a': '1', 'b': '2'}}

    Args:
        spec (str): The specification string to parse. May be empty.

    Returns:
        dict or None:  
          - A dict mapping each subdataset name to a {class_name: rule_str} dict.  
          - Returns None if `spec` is empty or falsy.
    """
    out = {}
    if not spec:
        return None

    spec = spec.strip()
    if spec.startswith('['):
        # Split [Subdataset] {a=1|b=2}, [Subdataset2] {...}
        for sub, body in re.findall(r'\[([^\]]+)\]\s*\{([^}]+)\}', spec):
            m = {}
            for pair in body.split('|'):
                key, value = pair.split('=', 1)
                m[key.strip()] = value.strip()
            out[sub.strip()] = m
    else:
        m = {}
        for pair in spec.split('|'):
            key, value = pair.split('=', 1)
            m[key.strip()] = value.strip()
        out['default'] = m
    return out

def parse_segmentation_tasks(spec):
    """
    Parse a segmentation-task specification string into a dict of tasks to classes.

    The spec has the form:
        "task1=class1,class2|task2=class3,class4"
    and is split on '|' into tasks, and on ',' into class lists.

    Args:
        spec (str): Specification string; tasks separated by '|', with each task
            in the form 'task_name=class1,class2,...'. May be empty or falsy.

    Returns:
        dict[str, list[str]] or None:
            - A dict mapping each task_name to a list of class names.
            - None if `spec` is empty or falsy.
    """
    out = {}
    if not spec:
        return None
    for chunk in spec.split('|'):
        task, classes = chunk.split('=', 1)
        out[task.strip()] = [c.strip() for c in classes.split(',') if c.strip()]
    return out

def match_mask_class(mask_path, mask_arr, mask_classes, subdataset_name, palette=None, logger=None, background_value=0):
    """
    Attempt to match a mask to a class based on rules:
        - filename keywords
        - hexadecimal palette colors
        - numeric label ranges or exact values
        - wildcard '*'

    Args:
        mask_path (str): The path to the mask file.
        mask_arr (xp.ndarray or None): The array containing the mask data, or None.
        mask_classes (dict): Mapping subdataset_name -> {class_name: rule_str, ...}.
        subdataset_name (str): The name of the subdataset to match against.
        palette (dict, optional): Mapping from original color tuples (R,G,B) to label ints.
        logger (logging.Logger, optional): Logger for debug/info messages.
        background_value (int, optional): Background label value. Defaults to 0.

    Returns:
        str or None: The matched class name, or None if no match is found.
    """
    rules = mask_classes.get(subdataset_name, mask_classes.get('default', {}))

    # 1) Filename keyword match (longest rules first)
    for cls_name, rule in sorted(rules.items(), key=lambda kv: len(kv[1]), reverse=True):
        if rule == "*":
            logger.info(f"Wildcard matching: {mask_path}.")
            return cls_name
        # skip hex or numeric rules here
        if rule.startswith("#") or _RANGE_RE.match(rule) or _DIGIT_RE.match(rule):
            continue
        if rule in mask_path:
            logger.info(f"Keyword matching: {mask_path}. Rule: {rule}")
            return cls_name

    # If no array provided, we're done
    if mask_arr is None:
        logger.info(f"No match for {mask_path} (no array to inspect).")
        return None

    arr = mask_arr

    # 2) Palette lookup (if provided)
    if palette is not None:
        logger.info(f"Palette matching: {mask_path}")
        # Invert palette: label -> (R,G,B)
        inv_palette = {lbl: col for col, lbl in palette.items()}
        # Find unique non-zero labels via bincount
        flat = arr.ravel()
        counts = xp.bincount(flat, minlength=background_value+1 if background_value > 0 else None)
        unique_lbls = set(xp.nonzero(counts)[0])
        unique_lbls.discard(background_value)

        for lbl in unique_lbls:
            col = inv_palette.get(lbl)
            if col is None:
                continue
            r, g, b = col
            hex_val = f"#{r:02x}{g:02x}{b:02x}"
            for cls_name, rule in rules.items():
                if rule.startswith("#") and rule.lower() == hex_val:
                    logger.info(f"Matched {mask_path} to {cls_name}. Rule: {rule}, Val: {hex_val}")
                    return cls_name

    # 3) Numeric matching: ranges and exact digits
    if xp.issubdtype(arr.dtype, xp.integer):
        # Get uniques via bincount
        flat = arr.ravel()
        counts = xp.bincount(flat)
        uniques = xp.nonzero(counts)[0]
        logger.info(f"Numeric matching: {mask_path}. Uniques: {uniques}")

        for cls_name, rule in rules.items():
            m = _RANGE_RE.match(rule)
            if m:
                lo, hi = map(int, m.groups())
                if any(lo <= u <= hi for u in uniques):
                    logger.info(f"Matched {mask_path} to {cls_name}. Rule: {rule}, range {lo}-{hi}")
                    return cls_name

            elif _DIGIT_RE.match(rule):
                d = int(rule)
                if d in uniques:
                    logger.info(f"Matched {mask_path} to {cls_name}. Rule: {rule}, contains {d}")
                    return cls_name

    # Nothing matched
    logger.info(f"No match for {mask_path}")
    return None

def match_tasks_for_class(cls_name, tasks_map):
    """
    Reverse lookup which segmentation task(s) include a given mask class.

    Args:
        cls_name (str): The name of the mask class
        tasks_map (dict): A dictionary of {task_name: [class_name, ...], ...}

    Returns:
        list: A list of task names that include the given class. If none match,
        returns an empty list.
    """
    return [task_id for task_id, class_list in tasks_map.items()
            if cls_name in class_list]

def extract_mask_class(path):
    """
    Extract the class name from a mask filename.

    Args:
        path (str): The path to the mask file

    Returns:
        str or None: The class name if found, or None
    """
    fn = os.path.basename(path)
    m  = _CLASS_RE.search(fn)
    return m.group(1) if m else None