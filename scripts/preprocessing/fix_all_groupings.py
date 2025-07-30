# fix_all_groupings.py
# This script is used to merge the preprocessed entries from 'groupings.json' into the indexed groups file.
# It simply exists to fix a mistake I made when writing the preprocessing script.
# To be run from within /data/Preprocessed otherwise it will not work as it assumes DatasetIndexes is one level up.
import os
import re
import json
import sys
import argparse
from tqdm import tqdm

# Directory containing the index files
INDEX_DIR = os.path.join('..', 'DatasetIndexes', 'Groups')

def load_json(path):
    """
    Load a JSON file from the given path, returning None if any error occurs.

    Args:
        path (str): The file path to load the JSON from.

    Returns:
        dict or None: The loaded JSON data, or None if there was an error.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}", file=sys.stderr)
        return None
        
def save_json(data, path):
    """
    Save the given data to the specified file path as a JSON object.

    Args:
        data (dict): The data to be saved as a JSON object.
        path (str): The file path to save the JSON data to.

    Returns:
        None

    Raises:
        Exception: If there is an error saving the JSON data to the specified path.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}", file=sys.stderr)

def merge_for_dataset(subfolder):
    '''
    Merge preprocessed entries into the indexed groups file for a given dataset subfolder.

    This function loads the local 'groupings.json' in the specified subfolder, locates the corresponding
    index file under '../DatasetIndexes/Groups/<subfolder>_groups.json', and merges each local entry's
    'proc_images', 'proc_masks', and 'preprocessing_metadata' into the matching entry in the index data.

    Entries are matched by a composite key constructed as:
        <split>_<identifier>_<additional values...>
    where:
      - 'split' is the data split ('train' or 'test').
      - 'identifier' is the base identifier string.
      - 'additional' is a dictionary of extra identifiers; its values are appended in sorted key order.
    If 'additional' is empty, the composite key is simply '<split>_<identifier>'.

    Args:
        subfolder (str): Name of the subfolder containing 'groupings.json' to process.

    Returns:
        bool: True if merge and save succeeded; False if any critical error occurred
              (missing files or JSON load failure).
    '''
    # Load local groupings file
    grouping_path = os.path.join(subfolder, 'groupings.json')
    if not os.path.isfile(grouping_path):
        print(f"Local groupings file missing: {grouping_path}", file=sys.stderr)
        return False
    
    local_groups = load_json(grouping_path)
    if local_groups is None:
        print(f"Local groupings file empty: {grouping_path}", file=sys.stderr)
        return False

    # Load index file
    index_filename = f"{subfolder}_groups.json"
    index_path = os.path.join(INDEX_DIR, index_filename)
    if not os.path.isfile(index_path):
        print(f"Index file missing: {index_path}", file=sys.stderr)
        return False
        
    index_data = load_json(index_path)
    if index_data is None:
        print(f"Index file empty: {index_path}", file=sys.stderr)
        return False

    # Merge each local entry into the corresponding index entry
    for entry in tqdm(local_groups, desc=f"Entries in {subfolder}", leave=False):
        # Get identifier from local entry and clean
        raw_id = entry.get('identifier', '')
        cleaned_id = re.sub(r'_modality\d+', '', raw_id) # Remove 'modality' followed by digits
        cleaned_id = re.sub(r'_+', '_', cleaned_id)      # Replace multiple underscores with a single one
        cleaned_id = re.sub(r'_ID2_', '_', cleaned_id)   # Remove '__ID2___' but keep what follows
        cleaned_id = cleaned_id.strip('_')               # Remove leading/trailing underscores
        
        merged = False
        for subd in index_data.get('subdatasets', []):
            for split in ('train', 'test'):
                for idx_entry in subd.get(split, []):
                    # Build composite key for the index entry
                    idx_split = idx_entry.get('split', '')
                    idx_id = idx_entry.get('identifier', '')
                    additional = idx_entry.get('additional', {}) or {}
                    parts = [idx_split, idx_id]
                    # Append additional values in sorted key order
                    for key in sorted(additional.keys()):
                        parts.append(str(additional[key]))
                    composite_key = "_".join(parts)

                    if composite_key == cleaned_id:
                        idx_entry['proc_images'] = entry.get('proc_images', [])
                        idx_entry['proc_masks'] = entry.get('proc_masks', [])
                        idx_entry['preprocessing_metadata'] = entry.get('preprocessing_metadata', {})
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break

        if not merged:
            print(f"Warning: no matching index entry for identifier '{cleaned_id}' in dataset '{subfolder}'", file=sys.stderr)

    # Save updated groups file next to the local grouping file
    out_path = os.path.join(subfolder, index_filename)
    save_json(index_data, out_path)
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge groupings into index groups for datasets.')
    parser.add_argument('--start', '-s', help="Subfolder name to start processing from", default=None)
    args = parser.parse_args()

    # Find all subfolders
    subfolders = sorted([d for d in os.listdir('.') if os.path.isdir(d)])
    
    # Start processing from the specified subfolder
    if args.start:
        if args.start not in subfolders:
            print(f"Error: start subfolder '{args.start}' not found.", file=sys.stderr)
            sys.exit(1)
        start_idx = subfolders.index(args.start)
        subfolders = subfolders[start_idx:]

    # Process each dataset
    for subfolder in tqdm(subfolders, desc="Datasets", leave=False):
        merge_for_dataset(subfolder)

    print("Done processing all datasets.")