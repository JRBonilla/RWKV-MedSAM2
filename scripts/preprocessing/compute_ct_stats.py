import os
import json
import numpy as np
import SimpleITK as sitk
import logging
from tqdm import tqdm
import random

from helpers import load_dataset_metadata, normalize_path

# Configuration constants
# GROUPS_DIR    = 'F:/DatasetIndexes/Groups'   # directory with *_groups.json files
# METADATA_CSV  = 'F:/Datasets/datasets.csv'   # dataset metadata CSV (or None)
# DATA_ROOT     = 'F:/Datasets'                # base directory for datasets if no metadata
# OUTPUT_DIR    = 'F:/DatasetIndexes/CTStats'  # where to save stats JSONs
# PERCENTILES   = (0.5, 99.5)                  # clipping percentiles

# Configuration constants
GROUPS_DIR    = '/data/DatasetIndexes/Groups'   # directory with *_groups.json files
METADATA_CSV  = '/data/research/datasets.csv'   # dataset metadata CSV (or None)
DATA_ROOT     = '/data/research'                # base directory for datasets if no metadata
OUTPUT_DIR    = '/data/DatasetIndexes/CTStats'  # where to save stats JSONs
PERCENTILES   = (0.5, 99.5)                  # clipping percentiles

# Sampling configuration
SAMPLE_FRAC = 1.0   # Fraction of CT volumes to sample (0 < frac <= 1)
SAMPLE_MAX  = None  # Maximum number of CT volumes to sample (None for no limit)

# Dataset filter
ONLY_DATASET = None

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

class StatsAccumulator:
    """Accumulate running mean and variance using a vectorized Welford algorithm."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, data):
        chunk = np.asarray(data, dtype=np.float64)
        n = chunk.size
        if n == 0:
            return
        chunk_mean = float(chunk.mean())
        chunk_var = float(chunk.var())
        new_count = self.count + n
        delta = chunk_mean - self.mean
        self.M2 += chunk_var * n + (delta ** 2) * self.count * n / new_count
        self.mean = (self.mean * self.count + chunk_mean * n) / new_count
        self.count = new_count

    def finalize(self):
        variance = self.M2 / (self.count - 1) if self.count > 1 else 0.0
        return self.mean, float(np.sqrt(variance))


def load_groups(dataset_name):
    """Load the groups JSON for a dataset and return its subdatasets list."""
    groups_file = os.path.join(GROUPS_DIR, f"{dataset_name}_groups.json")
    with open(groups_file, 'r') as f:
        data = json.load(f)
    return data.get('subdatasets', [])


def main():
    logger.info('Starting CT stats computation')

    # Load metadata if available
    metadata = {}
    if METADATA_CSV and os.path.isfile(METADATA_CSV):
        metadata = load_dataset_metadata(METADATA_CSV)
        logger.info('Loaded metadata for %d datasets from %s', len(metadata), METADATA_CSV)
    else:
        logger.info('Metadata CSV not found; proceeding without metadata')

    ensure_output_dir(OUTPUT_DIR)
    logger.info('Output directory: %s', OUTPUT_DIR)

    # Iterate over group files to find CT datasets
    for fname in os.listdir(GROUPS_DIR):
        if not fname.endswith('_groups.json'):
            continue
        dataset_name = fname.rsplit('_groups.json', 1)[0]
        # Skip if filtering to one dataset
        if ONLY_DATASET and dataset_name != ONLY_DATASET:
            logger.debug("Skipping dataset %s (ONLY_DATASET=%s)", dataset_name, ONLY_DATASET)
            continue

        ds_meta = metadata.get(dataset_name, {})
        # Determine base directory for dataset
        root_dir = ds_meta.get('root_directory')
        base = (os.path.join(DATA_ROOT, dataset_name, root_dir)
                if root_dir else os.path.join(DATA_ROOT, dataset_name))
        ds_base = normalize_path(base)
        logger.info("Dataset '%s' base directory: %s", dataset_name, ds_base)
        if not os.path.isdir(ds_base):
            logger.warning("Directory does not exist, skipping: %s", ds_base)
            continue

        # Load subdatasets and collect CT image paths
        subdatasets = load_groups(dataset_name)
        ct_paths = []
        for sub in subdatasets:
            if sub.get('modality','').lower() != 'ct':
                continue
            for split in ('train', 'test'):
                for group in sub.get(split, []):
                    for img_rec in group.get('images', []):
                        path = img_rec.get('path')
                        if path:
                            ct_paths.append(path)
        ct_paths = sorted(set(normalize_path(p) for p in ct_paths))
        orig_count = len(ct_paths)
        logger.info("Found %d CT image records for dataset %s", orig_count, dataset_name)
        if orig_count == 0:
            logger.warning("No CT image records for %s, skipping.", dataset_name)
            continue

        # Apply sampling of CT volumes
        k = int(orig_count * SAMPLE_FRAC)
        k = min(orig_count, k)
        if SAMPLE_MAX is not None:
            k = min(k, SAMPLE_MAX)
        if k < orig_count:
            ct_paths = random.sample(ct_paths, k)
        logger.info("Processing %d of %d volumes for %s", len(ct_paths), orig_count, dataset_name)

        acc = StatsAccumulator()
        p_low, p_high = PERCENTILES
        logger.info('Computing stats for %s over %d images', dataset_name, len(ct_paths))
        for img_path in tqdm(ct_paths, desc=dataset_name, unit='img'):
            # Load image: support .npy or medical image formats
            try:
                if img_path.lower().endswith('.npy'):
                    img = np.load(img_path)
                else:
                    img_itk = sitk.ReadImage(img_path)
                    img = sitk.GetArrayFromImage(img_itk)
            except Exception as e:
                logger.error("Failed to load %s: %s", img_path, e)
                continue

            mask = img != 0
            if not mask.any():
                continue
            values = img[mask].astype(np.float64)
            lower, upper = np.percentile(values, [p_low, p_high])
            clipped = np.clip(values, lower, upper)
            acc.update(clipped)

        if acc.count == 0:
            logger.warning("No valid CT voxels found for %s", dataset_name)
            continue

        mean, std = acc.finalize()
        logger.info("Stats for %s: mean=%.4f, std=%.4f over %d voxels",
                    dataset_name, mean, std, acc.count)
        stats = {'dataset_name': dataset_name, 'mean': mean,
                 'std': std, 'count': acc.count}

        out_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_ct_stats.json")
        with open(out_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info("Saved CT stats for %s to %s", dataset_name, out_file)

    logger.info('CT stats computation complete')

if __name__ == '__main__':
    main()
