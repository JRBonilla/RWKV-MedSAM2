# train_sam2.py
# Main training script for RWKV‑MedSAM2:
#   - load_config                   - Load YAML config and set random seeds
#   - setup_logger                  - Configure console + timed rotating file logging
#   - get_pairings                  - Load raw (image, mask) pairs
#   - get_sequence                  - Assemble (image, mask) pairs into sequences
#   - get_data_loaders              - Build SegmentationSequenceDataset + BalancedTaskSampler
#   - build_student/teacher         - Instantiate SAM2VideoPredictor models
#   - setup_optimizer_and_scheduler - Create AdamW optimizers and warm-up + cosine scheduler
#   - train_epoch/validate_epoch    - Unified 2D/3D training and validation loops
#   - CheckpointManager             - Save/load latest and best model checkpoints
import os
import re
import json
import math
import random
import logging
import argparse
import pickle

from collections import defaultdict

import cupy as cp  # type: ignore
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from logging.handlers import TimedRotatingFileHandler

import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from torchvision.transforms import functional as TF

from ext.sam2.build_sam import build_sam2_video_predictor
from ext.sam2.sam2_video_predictor import SAM2VideoPredictor

from dripp.helpers import normalize_path, get_extension

from .functions.func_2d import train_step_2d, validate_step_2d
from .functions.func_3d import train_step_3d, validate_step_3d

from .dataset import SegmentationSequenceDataset, BalancedTaskSampler, SequenceTransform
from .utils.vis import visualize_sequence

def load_config(config_path):
    """
    Loads a configuration file from a given path and resolves any variable interpolation.
    If the config contains a 'seed' field under 'training', sets the Python random seeds for reproducibility.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        OmegaConf.DictConfig: The resolved configuration.
    """
    config = OmegaConf.load(config_path)
    # Resolve any variable interpolation
    OmegaConf.resolve(config)

    # Set Python random seeds for reproducibility
    seed = getattr(config.training, "seed", 42)
    if seed is not None:
        random.seed(seed)
        cp.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return config

def setup_logger(log_cfg):
    """
    Sets up a logger for this module that logs to both the console and a timed rotating log file.

    Args:
        log_cfg (OmegaConf.DictConfig): A configuration containing the following fields:
            - output_dir (str):   The directory where the logs will be saved.
            - log_filename (str): The filename of the log, which will be prepended with the current date and appended with '.log'.
            - level (str):        The logging level to use for console logging. Should be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
            - file_level (str):   The logging level to use for file logging. Should be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
            - interval (int):     The interval at which the log file will be rotated, in minutes.
            - backups (int):      The number of log files to keep before deleting the oldest one.

    Returns:
        logger (logging.Logger): The configured logger.
    """
    # Ensure the output directory exists
    os.makedirs(log_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(log_cfg.output_dir, log_cfg.filename)

    # Set up the logger
    logger = logging.getLogger("RWKV-MedSAM2")
    logger.setLevel(getattr(logging, log_cfg.level.upper()))

    # Stream (console) handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_cfg.level.upper()))

    # File handler
    fh = TimedRotatingFileHandler(
        filename=log_path,
        when="M",
        interval=int(log_cfg.interval),
        backupCount=int(log_cfg.backups),
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, log_cfg.file_level.upper()))

    # Formatter
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)

    # Set formatters
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Print info
    logger.info(f"Logging to console at {log_cfg.level} and to {log_path} at {log_cfg.file_level}")

    return logger

class CheckpointManager:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.latest_path = os.path.join(output_dir, "latest.pth")
        self.best_path   = os.path.join(output_dir, "best.pth")
        self.best_metric = -float("inf")

    def save_latest(self, model, mask_decoder_opt, memory_opt, scheduler, epoch):
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "opt1":      mask_decoder_opt.state_dict(),
            "opt2":      memory_opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, self.latest_path)

    def save_best(self, model, mask_decoder_opt, memory_opt, scheduler, epoch, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "opt1":      mask_decoder_opt.state_dict(),
                "opt2":      memory_opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metric":    metric,
            }, self.best_path)

    def load(self, checkpoint_path, model, mask_decoder_opt=None, memory_opt=None, scheduler=None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return (
            ckpt["epoch"],
            ckpt["opt1"],
            ckpt["opt2"],
            ckpt["scheduler"]
        )

def get_pairings(out_dir, datasets, split="train"):
    """
    Loads the groupings.json files from each folder in the given output directory
    for the specified split.

    Each groupings.json file is expected to contain a list of entries, where each
    entry is a dictionary with the following keys:
    - "proc_images": A list of paths to the preprocessed images.
    - "proc_masks": A list of paths to the preprocessed masks.

    The function pairs each mask with its corresponding image by searching for
    the index in the image path. The index is assumed to be in the format
    "(?:img|frame|slice)(\d+)" and is used to match the mask with its
    corresponding image.

    Args:
        out_dir (str): The output directory containing the dataset folders.
        datasets (list): The list of dataset names to load groupings for.
        split (str, optional): The split to load groupings for. Default is "train".

    Returns:
        list: A list of lists, where each inner list contains pairs of image and
              mask paths.
    """
    # Find all image and mask pairs for each dataset and collect them
    _idx_pattern = re.compile(r"_(?:img|frame|slice)(\d+)")
    all_pairs = []
    for ds in datasets:
        # Check if dataset grouping json exists
        ds_dir = os.path.join(out_dir, ds)
        grp_file = os.path.join(ds_dir, f'{ds}_groups.json')
        if not os.path.isfile(grp_file):
            print(f"Could not find {ds}_groups.json in {ds_dir}")
            continue

        # Parse the groups file
        data = json.load(open(grp_file, 'r'))
        entries  = []
        for sub in data.get("subdatasets", []):
            for entry in sub.get(split, []):
                entry["subdataset_name"] = sub.get("name", "default")
                entry["tasks"]           = sub.get("tasks", [])
                entry["mask_classes"]    = sub.get("classes", [])
                entries.append(entry)
        print(f"Found {len(entries)} '{split}' entries in {grp_file}")

        # Iterate through the groupings and create image/mask pairs
        for entry in entries:
            # Skip entries that don't match the split
            if entry.get('split') != split:
                continue

            imgs = entry.get('proc_images', [])
            msks = entry.get('proc_masks', [])

            # Create a map from index to image
            idx_map = {}
            for img_path in imgs:
                m = _idx_pattern.search(os.path.basename(img_path))
                if m:
                    idx_map[int(m.group(1))] = img_path
            
            # Pair each mask with its corresponding image and add to the list of pairs
            # Each list of pairs corresponds to a single grouping
            pairs = []
            for mask in msks:
                msk_path = mask['path']
                cls_name = mask['class']
                m = _idx_pattern.search(os.path.basename(msk_path))
                if not m or cls_name is None: # Skip if no index or class
                    continue
                img_path = idx_map.get(int(m.group(1)))
                if img_path:
                    pairs.append({
                        'image': img_path,
                        'mask':  msk_path,
                        'class': cls_name
                    })

            # If no pairs found, skip
            if not pairs:
                continue

            # Add the pairs to the raw list
            # Sort the pairs to ensure temporal order
            pairs.sort(key=lambda x: int(_idx_pattern.search(os.path.basename(x['image'])).group(1)))
            # For multiple pairs, split the pairs into separate entries in the raw list
            if len(pairs) > 1:
                for pair in pairs:
                    all_pairs.append({
                        'dataset':      ds,
                        'subdataset':   entry['subdataset_name'],
                        'tasks':        entry['tasks'],
                        'class':        pair['class'],
                        'pair':         (pair['image'], pair['mask']),
                        'dim':          2 if get_extension(pair['image']) == '.png' else 3
                    })
            else:
                # For single pair, add to the raw list of pairs
                all_pairs.append({
                    'dataset':      ds,
                    'subdataset':   entry['subdataset_name'],
                    'tasks':        entry['tasks'],
                    'mask_classes': pairs[0]['class'],
                    'pair':         (pairs[0]['image'], pairs[0]['mask']),
                    'dim':          2 if get_extension(pairs[0]['image']) == '.png' else 3
                })
    print(f"Found {len(all_pairs)} pairs for {ds} '{split}' split")
    return all_pairs

def get_sequences(out_dir, split="train", val_frac=0.1, seed=42, max_frames_per_sequence=8):
    """
    Load and assemble 2D and 3D data into "sequences" for unified video-style training.

    A "sequence" is a list of frames, where each frame is an (image_path, mask_path)
    tuple. For 2D tasks, each sequence contains up to 'max_frames_per_sequence' frames—
    chunked (non-overlapping) from each dataset/subdataset/task group in temporal order.
    The final chunk may be shorter than 'max_frames_per_sequence'. For 3D volumes,
    each sequence represents one full volume as a single-element list.

    Processing steps:
      1. Load raw (image, mask) pairs via 'get_pairings()', each dict tagged with:
         dataset, subdataset, tasks, class, pair, dim.
      2. Group all 'dim==2' entries by (dataset, subdataset, tasks) into long frame lists.
      3. Chunk each 2D list into windows of length 'max_frames_per_sequence'.
      4. Treat each 'dim==3' entry as a single-element sequence.
      5. If 'split=='train', shuffle and split each set of chunks or volumes
         into train/validation subsets using 'val_frac'.
      6. Return:
         - For 'split=='train': a tuple '(train_seqs, val_seqs)'.
         - For 'split=='test': a list 'test_seqs'.

    Args:
        out_dir (str): Path to the directory containing dataset subfolders.
        split (str): One of '"train"' or '"test"'.
        val_frac (float): Fraction of each sequence reserved for validation
                          (only used when 'split=='train'').
        seed (int): Random seed for reproducibility.
        max_frames_per_sequence (int): Maximum number of frames per 2D sequence.

    Returns:
        - If 'split=='train'': '(train_seqs, val_seqs)', each a list of sequence dicts.
        - If 'split=='test'': 'test_seqs', a list of sequence dicts.

    Each sequence dict contains:
        - 'dataset' (str)
        - 'subdataset' (str)
        - 'tasks' (List[str])
        - 'sequence' (List[Tuple[str, str]]): ordered image/mask path tuples
        - 'dim' (int): 2 or 3
    """
    # 0a) Find dataset subfolders
    try:
        entries = sorted(os.listdir(out_dir))
    except Exception:
        raise RuntimeError(f"Could not find output directory {out_dir}")
    datasets = [d for d in entries if os.path.isdir(os.path.join(out_dir, d))]

    # 0b) Regex + helper to extract frame index
    idx_pattern = re.compile(r"_(?:img|frame|slice)(\d+)")
    def extract_idx(path):
        m = idx_pattern.search(os.path.basename(path))
        return int(m.group(1)) if m else 0

    # 1) Load all (image, mask) pairs
    pairs    = get_pairings(out_dir, datasets, split)
    pairs2D  = [p for p in pairs if p['dim'] == 2]
    pairs3D  = [p for p in pairs if p['dim'] == 3]

    # 2) Group into true sequences
    grouped2D = defaultdict(list)
    for p in pairs2D:
        key = (p['dataset'], p['subdataset'], tuple(p['tasks']))
        grouped2D[key].append(p['pair'])

    grouped3D = defaultdict(list)
    for p in pairs3D:
        key = (p['dataset'], p['subdataset'], tuple(p['tasks']))
        grouped3D[key].append(p['pair'])

    # 3) Build train/val or test lists
    if split == 'train':
        train_seqs, val_seqs = [], []

        # 3a) 2D: shuffle & chunk frames into many K‐length sequences
        K = max_frames_per_sequence
        for (ds, sub, tasks), frames in grouped2D.items():
            random.seed(seed)
            random.shuffle(frames)
            # Split into train / val at frame‐level
            n_val = int(len(frames) * val_frac)
            train_f, val_f = frames[n_val:], frames[:n_val]
            # Restore temporal order within each split
            train_f.sort(key=lambda pair: extract_idx(pair[0]))
            val_f.sort(  key=lambda pair: extract_idx(pair[0]))

            # Chunk each into sliding / non‐overlapping windows of length K
            for i in range(0, len(train_f), K):
                chunk = train_f[i : i + K] # Train
                if chunk:
                    train_seqs.append({
                        'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                        'sequence': chunk, 'dim': 2
                    })
            for i in range(0, len(val_f), K): # Val
                chunk = val_f[i : i + K]
                if chunk:
                    val_seqs.append({
                        'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                        'sequence': chunk, 'dim': 2
                    })

        # 3b) 3D: shuffle & split volumes per sequence
        for (ds, sub, tasks), vols in grouped3D.items():
            random.seed(seed)
            random.shuffle(vols)
            n_val   = int(len(vols) * val_frac)
            train_v = vols[n_val:]
            val_v   = vols[:n_val]

            for v in train_v:
                train_seqs.append({'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                                   'sequence': [v], 'dim': 3})
            for v in val_v:
                val_seqs.  append({'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                                   'sequence': [v], 'dim': 3})

        return train_seqs, val_seqs

    # Test split
    test_seqs = []
    for (ds, sub, tasks), frames in grouped2D.items():
        frames.sort(key=lambda pair: extract_idx(pair[0]))
        test_seqs.append({'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                          'sequence': frames, 'dim': 2})

    for (ds, sub, tasks), vols in grouped3D.items():
        for v in vols:
            test_seqs.append({'dataset': ds, 'subdataset': sub, 'tasks': list(tasks),
                              'sequence': [v], 'dim': 3})

    return test_seqs

def get_data_loaders(config):
    """
    Given a config object, this function will create a set of train, val, and test
    data loaders using the DRIPP pairings. The function works as follows:

    1) Load all DRIPP sequences for the train and test splits.
    2) Load the DRIPP tasks map.
    3) Build a dataset + loader for each group.

    Args:
        config: A config object with the following attributes:
            - dripp.output_dir: The output directory for the DRIPP pairings.
            - training.seed: The seed to use for shuffling the train and val sets.
            - training.val_frac: The fraction of the train set to use for validation.
            - training.batch_size: The batch size to use for all data loaders.
            - training.num_workers: The number of workers to use for all data loaders.
    Returns:
        tuple: A tuple of (train_loaders, val_loaders, test_loaders), where each
               loader is a dict mapping (dataset, subdataset) to a DataLoader instance.
    """
    # 1) Load all sequences
    train_seqs, val_seqs = get_sequences(
        config.dripp.output_dir,
        split='train',
        val_frac=config.training.val_frac,
        seed=config.training.seed
    )
    test_seqs = get_sequences(
        config.dripp.output_dir,
        split='test'
    )

    # 2) Load the tasks map
    tasks_map = json.load(open(config.dripp.tasks_file, 'r'))

    # 3) Build a dataset + loader for each group
    # Train
    train_ds      = SegmentationSequenceDataset(train_seqs, transform=SequenceTransform())
    train_sampler = BalancedTaskSampler(train_seqs, tasks_map, config.training.seed)
    train_loader  = DataLoader(train_ds,
                               batch_size=config.training.batch_size,
                               sampler=train_sampler,
                               num_workers=config.training.num_workers,
                               pin_memory=True)
    
    # Val
    val_loader = DataLoader(
        SegmentationSequenceDataset(val_seqs, transform=SequenceTransform()),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Test
    test_loader = DataLoader(
        SegmentationSequenceDataset(test_seqs, transform=SequenceTransform()),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    print("Data loaders created.")
    return train_loader, val_loader, test_loader

def save_data_loaders(save_path, train_loaders, val_loaders, test_loaders):
    """
    Save train, validation, and test DataLoader collections to disk via pickle.

    Args:
        save_path (str): Path to the output .pkl file.
        train_loaders (dict): Mapping (dataset, subdataset) -> train DataLoaders.
        val_loaders   (dict): Mapping (dataset, subdataset) -> validation DataLoaders.
        test_loaders  (dict): Mapping (dataset, subdataset) -> test DataLoaders.
    """
    # Ensure target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Serialize all three loader dicts in one file
    with open(save_path, "wb") as f:
        pickle.dump({
            "train_loaders": train_loaders,
            "val_loaders":   val_loaders,
            "test_loaders":  test_loaders,
        }, f)


def load_data_loaders(load_path):
    """
    Load train, validation, and test DataLoader collections from a pickle file.

    Args:
        load_path (str): Path to the .pkl file created by save_data_loaders.

    Returns:
        tuple:
            - train_loaders (dict): Mapping (dataset, subdataset) -> train DataLoaders.
            - val_loaders   (dict): Mapping (dataset, subdataset) -> validation DataLoaders.
            - test_loaders  (dict): Mapping (dataset, subdataset) -> test DataLoaders.
    """
    with open(load_path, "rb") as f:
        data = pickle.load(f)

    return (
        data["train_loaders"],
        data["val_loaders"],
        data["test_loaders"],
    )


def build_student_predictor(config):
    """
    Build the student video predictor.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        video_predictor (SAM2VideoPredictor): The student video predictor.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config._config_path,
        ckpt_path=None,
        device=config.training.device,
        mode='train',
        apply_postprocessing=True,
    )
    video_predictor.train()
    return video_predictor

def build_teacher_predictor(config):
    """
    Builds the teacher video predictor for SAM2 training.
    The teacher model is used for distillation.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        video_predictor (SAM2VideoPredictor): The teacher video predictor.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config.teacher._config_path,
        ckpt_path=config.teacher.ckpt_path,
        device=config.training.device,
        mode='eval',
        apply_postprocessing=True,
    )
    video_predictor.eval()
    return video_predictor

def setup_optimizer_and_scheduler(model, config, data_dimension):
    """
    Create two AdamW optimizers and composite LR scheduler:
        - Linear warm-up for the first 'warmup_epochs'
        - Cosine decay for the remaining epochs
    
    Expects the following parameters in config.training:
        - lr            : Full-net learning rate for 2D optimizer
        - mask_lr       : Mask decoder learning rate
        - mem_lr        : Memory encoder learning rate
        - weight_decay  : Weight decay for both optimizers
        - epochs        : Total number of epochs
        - warmup_epochs : Number of warmup epochs

    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): The configuration dictionary.
        data_dimension (int): The data dimension (2D or 3D).
    Returns:
        dict: A dictionary containing the following keys:
            - optimizers (list): List of optimizers. If data_dimension == 2,
                this is a list containing a single optimizer. If data_dimension == 3,
                this is a list containing two optimizers.
            - scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.
    """
    # 0) Build learning rate scheduler
    total_epochs  = int(config.training.epochs)
    warmup_epochs = int(config.training.warmup_epochs)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # 1) Split out parameter groups
    if data_dimension == 2:
        # 2D: Full-net optimizer (image_encoder + prompt_encoder + mask_decoder)
        all_2d_layers = (
            list(model.image_encoder.parameters()) +
            list(model.sam_prompt_encoder.parameters()) +
            list(model.sam_mask_decoder.parameters())
        )
        optimizer2d = AdamW(
            all_2d_layers,
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay)
        )
        scheduler2d = LambdaLR(optimizer2d, lr_lambda)
        return {
            'optimizers': [optimizer2d],
            'scheduler':  scheduler2d
        }
    
    # 3D: Mask-decoder + Memory-encoder + Memory-attention + Mask-downsampler
    sam_layers = list(model.sam_mask_decoder.parameters())
    mem_layers = (
        list(model.obj_ptr_proj.parameters()) +
        list(model.memory_encoder.parameters()) +
        list(model.memory_attention.parameters()) +
        list(model.mask_downsample.parameters())
    )

    # 2) Build optimizers
    mask_decoder_opt = AdamW(
        sam_layers,
        lr=float(config.training.mask_lr),
        weight_decay=float(config.training.weight_decay)
    )
    memory_opt = AdamW(
        mem_layers,
        lr=float(config.training.mem_lr),
        weight_decay=float(config.training.weight_decay)
    )

    # 3) Optionally include prompt_encoder in mask_decoder_opt if you want to train it
    # sam_layers += list(model.prompt_encoder.parameters())

    # 4) Scheduler
    total_epochs  = int(config.training.epochs)
    warmup_epochs = int(config.training.warmup_epochs)

    scheduler3d = LambdaLR(mask_decoder_opt, lr_lambda)

    return {
        'optimizers': [mask_decoder_opt, memory_opt],
        'scheduler':  scheduler3d
    }

def train_epoch(student, teacher, train_loader, optimizer2d, optimizers3d, sched2d, sched3d, config, scaler, epoch, logger):
    """
    Perform one epoch of training over the dataset.

    This function iterates over the training data and performs either a 2D or 3D training step
    depending on the data dimensionality. It also visualizes predictions for the first batch
    and logs the average loss for the epoch.

    Args:
        student (SAM2VideoPredictor): The student model to be trained.
        teacher (SAM2VideoPredictor): The teacher model for distillation.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer2d (torch.optim.Optimizer): Optimizer for 2D training.
        optimizers3d (tuple): Tuple containing optimizers for 3D training.
        sched2d (torch.optim.lr_scheduler._LRScheduler): Scheduler for 2D optimizer.
        sched3d (torch.optim.lr_scheduler._LRScheduler): Scheduler for 3D optimizers.
        config (dict): Configuration parameters.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        epoch (int): The current epoch number.
        logger (logging.Logger): Logger for logging training information.

    Returns:
        float: The average loss over all batches.
    """
    # 0) Set model in training mode and initialize variables
    student.train()
    total_loss = 0.0
    mem_bank = []
    first_vis = False

    # 1) Training loop
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch")
    for batch in pbar:
        # Show dataset/subdataset metadata in progress bar
        # If batch_size > 1, take the first element
        ds_name  = batch.get('dataset', 'NA')
        sub_name = batch.get('subdataset', 'NA')
        if isinstance(ds_name,  (list, tuple)): ds_name  = ds_name[0]
        if isinstance(sub_name, (list, tuple)): sub_name = sub_name[0]
        pbar.set_description(f"Train Epoch {epoch} - [{ds_name}/{sub_name}]")

        T = batch['image'].shape[1]
        if T == 1:
            # 2D step
            loss = train_step_2d(student, teacher, optimizer2d, batch, config, mem_bank, scaler)
            if not first_vis:
                # Visualize 2D
                # Get per‐frame logits
                _, logits_seq = validate_step_2d(student, batch, config, return_logits=True)
                visualize_sequence(batch['image'][0], batch['mask'][0].squeeze(1), logits_seq, threshold=0.5, fps=2,)
                first_vis = True
        else:
            # 3D step
            mask_decoder_opt, memory_opt = optimizers3d
            loss, pl, npl = train_step_3d(student, teacher, mask_decoder_opt, memory_opt, batch, config, scaler)
            if not first_vis:
                # Visualize 3D
                # Get full‐sequence logits [T,H,W]
                _, logits3_seq = validate_step_3d(student, batch, config, return_logits=True)
                visualize_sequence(batch['image'][0], batch['mask'][0].squeeze(1), logits3_seq, threshold=0.5, fps=2,)
                first_vis = True
        # Update total loss
        total_loss += loss
        pbar.set_postfix(loss=f"{loss:.4f}")
    # Close progress bar and compute average loss
    pbar.close()
    avg_loss = total_loss / len(train_loader)
    logger.info(f"[Epoch {epoch}] Train avg loss: {avg_loss:.4f}")

    # Step both schedulers
    sched2d.step()
    sched3d.step()

    return avg_loss

def validate_epoch(student, val_loader, config):
    """
    Run one full validation epoch, calling validate_step_2d or validate_step_3d
    on each batch and averaging the per-batch metrics.

    Args:
        student (SAM2VideoPredictor): the student model in eval mode
        val_loader (DataLoader): validation loader exposing .dataset.data_dimension
        config (DictConfig): configuration with .training.device, .training.out_size, .prompt.max_per_seq

    Returns:
        dict: average metrics {'iou', 'dice', 'hd95'} over all batches
    """
    # 0) Set model in eval and initialize variables
    student.eval()
    totals = {'iou': 0.0, 'dice': 0.0, 'hd95': 0.0}
    n = 0

    # 1) Validation loop
    with torch.no_grad():
        for batch in val_loader:
            T = batch['image'].shape[1]
            if T == 1:
                m = validate_step_2d(student, batch, config)
            else:
                m = validate_step_3d(student, batch, config)
            for k in totals:
                totals[k] += m[k]
            n += 1

    # 2) Compute metrics
    return {k: totals[k] / n for k in totals}


def main(config_path, resume, multi_gpu, amp):
    """
    Train a SAM2 model from scratch or from a checkpoint.

    This function will:

    1.  Load the configuration from the given YAML file.
    2.  Set up the data loaders for the training and validation datasets.
    3.  Build the student and teacher models.
    4.  Set up the optimizers, scheduler, and AMP (mixed precision) scaler.
    5.  If multi-GPU is enabled and there is more than one GPU, use DataParallel.
    6.  Load from a checkpoint if the resume argument is given.
    7.  Run the training loop for the specified number of epochs.
    8.  Run the validation loop at the end of each epoch depending on the validation frequency.
    9.  Save the latest and best checkpoints after validation.
    10. Log training metrics and validation metrics.

    Args:
        config_path (str): Path to the YAML configuration file.
        resume (str): Path to the checkpoint to resume from, or None to start from scratch.
        multi_gpu (bool): Whether to use DataParallel on multiple GPUs.
        amp (bool): Whether to use mixed precision training.
    """
    # 1) Config + logger
    config = load_config(config_path)
    logger = setup_logger(config.logging)

    # 2) Data loaders
    train_loader, val_loader, _ = get_data_loaders(config)

    # 3) Build Models
    student = build_student_predictor(config).to(config.training.device)
    teacher = build_teacher_predictor(config).to(config.training.device)

    # 4) Setup optimizers + schedulers
    opt2 = setup_optimizer_and_scheduler(student, config, data_dimension=2)
    optimizer2d, sched2d = opt2['optimizers'][0], opt2['scheduler']
    opt3 = setup_optimizer_and_scheduler(student, config, data_dimension=3)
    sched3d = opt3['scheduler']
    optimizers3d = opt3['optimizers']  # (mask_decoder_opt, memory_opt)

    # 5) AMP & multi-GPU
    scaler = torch.amp.GradScaler(enabled=amp)
    if multi_gpu and torch.cuda.device_count() > 1:
        student = torch.nn.DataParallel(student)
        teacher = torch.nn.DataParallel(teacher)

    # 6) Resume from checkpoint if needed
    ckpt_mgr = CheckpointManager(config.training.output_dir)
    start_ep = 0
    if resume:
        start_ep, s1, s2, sch = ckpt_mgr.load(resume, student)
        optimizer2d.load_state_dict(s1)
        optimizers3d[0].load_state_dict(s1)
        optimizers3d[1].load_state_dict(s2)
        sched2d.load_state_dict(sch)
        sched3d.load_state_dict(sch)
        logger.info(f"Resumed from epoch {start_ep}")

    # 7) Epoch loop
    for epoch in range(start_ep, int(config.training.epochs)):
        train_epoch(
            student, teacher, train_loader,
            optimizer2d, optimizers3d,
            sched2d, sched3d,
            config, scaler, epoch, logger
        )

        # 8) Validation
        if epoch % config.training.val_freq == 0 or epoch == config.training.epochs - 1:
            val_metrics = validate_epoch(student, val_loader, config)

            # 9) Save checkpoints
            ckpt_mgr.save_latest(student, optimizer2d, optimizers3d[0], sched2d, epoch)
            if val_metrics['iou'] > getattr(ckpt_mgr, 'best_metric', -1):
                ckpt_mgr.save_best(student, optimizer2d, optimizers3d[0], sched2d, epoch, metric=val_metrics['iou'])

            # 10) Log metrics
            logger.info(f"[Epoch {epoch}] Val metrics: {val_metrics}")

    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=args.resume,
        multi_gpu=args.multi_gpu,
        amp=args.amp
    )
