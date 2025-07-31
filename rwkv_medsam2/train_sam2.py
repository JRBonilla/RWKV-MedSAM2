# train_sam2.py
# Main training script for RWKV‑MedSAM2:
#   - load_config                   - Load YAML config and set random seeds
#   - setup_logger                  - Configure console + timed rotating file logging
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
from .utils.vis import visualize_predictions_2d, visualize_nifti_predictions

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

def get_pairings(out_dir, split="train"):    
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
        out_dir (str): The output directory containing the groupings.json files.
        split (str, optional): The split to load groupings for. Default is "train".

    Returns:
        list: A list of lists, where each inner list contains pairs of image and
              mask paths.
    """
    try:
        entries = sorted(os.listdir(out_dir))
    except:
        raise RuntimeError(f"Could not find output directory {out_dir}")
    # Filter out non-directories
    datasets = [dataset for dataset in entries if os.path.isdir(os.path.join(out_dir, dataset))]
    print(f"Found {len(datasets)} datasets in {out_dir}")

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
        raw_data = json.load(open(grp_file, 'r'))
        entries  = []
        for sub in raw_data.get("subdatasets", []):
            for entry in sub.get(split, []):
                entry["subdataset_name"] = sub.get("name", "default")
                entry["tasks"]           = sub.get("tasks", [])
                entry["mask_classes"]    = sub.get("classes", [])
                entries.append(entry)
        print(f"Found {len(entries)} '{split}' entries in {grp_file}")

        # Pair each mask with its corresponding image
        for entry in entries:
            # Skip entries that don't match the split
            if entry.get('split') != split:
                continue

            imgs = entry.get('proc_images', [])
            msks = entry.get('proc_masks', [])
            idx_map = {}
            for img_path in imgs:
                m = _idx_pattern.search(os.path.basename(img_path))
                if m:
                    idx_map[int(m.group(1))] = img_path
            pairs = []
            for mpath in msks:
                m = _idx_pattern.search(os.path.basename(mpath['path']))
                if not m:
                    continue
                img_p = idx_map.get(int(m.group(1)))
                if img_p:
                    pairs.append((img_p, mpath['path']))
            if pairs:
                # Sort the pairs to ensure temporal order
                pairs.sort(key=lambda x: int(_idx_pattern.search(os.path.basename(x[0])).group(1)))
                if len(pairs) > 1 and get_extension(pairs[0][0]) == '.png':
                    for img_p, m_p in pairs:
                        all_pairs.append({
                            'dataset':      ds,
                            'subdataset':   entry['subdataset_name'],
                            'tasks':        entry['tasks'],
                            'mask_classes': entry['mask_classes'],
                            'pairs':        [(img_p, m_p)]
                        })
                else:
                    all_pairs.append({
                        'dataset':      ds,
                        'subdataset':   entry['subdataset_name'],
                        'tasks':        entry['tasks'],
                        'mask_classes': entry['mask_classes'],
                        'pairs':        pairs
                    })
    print(f"Found {len(all_pairs)} pairs for {ds} '{split}' split")
    return all_pairs

def get_data_loaders(config):
    """
    Given a config object, this function will create a set of train, val, and test
    data loaders using the DRIPP pairings. The function works as follows:

    1) Load all DRIPP pairings for the train and test splits.
    2) Group each set of pairings by (dataset, subdataset).
    3) Split each train group into train/val using the val_frac parameter.
    4) Load the DRIPP tasks map.
    5) Build a dataset + loader for each group.

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
    train_pairings = get_pairings(config.dripp.output_dir, split='train')
    # 1) Load all DRIPP pairings
    test_pairs     = get_pairings(config.dripp.output_dir, split='test')

    # 2) Group by (dataset, subdataset).
    train_groups = defaultdict(list)
    for pair in train_pairings:
        key = (pair['dataset'], pair.get('subdataset', ''))
        train_groups[key].append(pair)

    test_groups = defaultdict(list)
    for pair in test_pairs:
        key = (pair['dataset'], pair.get('subdataset', ''))
        test_groups[key].append(pair)

    # 3) Split each train group into train/val
    val_groups = {}
    for key, pairs in train_groups.items():
        random.seed(config.training.seed)
        random.shuffle(pairs)
        n_val = int(len(pairs) * config.training.val_frac)
        val_groups[key]   = pairs[:n_val]
        train_groups[key] = pairs[n_val:]

    # Remove empty groups
    train_groups = {k: v for k, v in train_groups.items() if v}
    val_groups   = {k: v for k, v in val_groups.items()   if v}
    test_groups  = {k: v for k, v in test_groups.items()  if v}

    # 4) Load DRIPP tasks map
    tasks_map = json.load(open(config.dripp.tasks_file, 'r'))

    # 5) Build dataset + loader for each group
    train_loaders = {}
    val_loaders   = {}
    test_loaders  = {}

    for key, pairs in train_groups.items():
        ds = SegmentationSequenceDataset(pairs, transform=SequenceTransform())
        sampler = BalancedTaskSampler(pairs, tasks_map, config.training.seed)
        loader = DataLoader(ds, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.training.num_workers, pin_memory=True)
        train_loaders[key] = loader

    for key, pairs in val_groups.items():
        ds = SegmentationSequenceDataset(pairs, transform=SequenceTransform())
        loader = DataLoader(ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=True)
        val_loaders[key] = loader

    for key, pairs in test_groups.items():
        ds = SegmentationSequenceDataset(pairs, transform=SequenceTransform())
        loader = DataLoader(ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=True)
        test_loaders[key] = loader

    print(f"Created {len(train_loaders)} train loaders, {len(val_loaders)} val loaders, and {len(test_loaders)} test loaders.")
    return train_loaders, val_loaders, test_loaders

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
        list(model.mask_downsampler.parameters())
    )

    # 2) Build optimizers
    mask_decoder_opt= AdamW(
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

def train_epoch(student, teacher, train_loader, optimizers, scheduler, config, scaler, epoch, logger):
    """
    Single epoch over all batches, selecting 2D vs 3D training.
    """
    student.train()
    total_batch_loss      = 0.0
    total_prompt_loss     = 0.0
    total_non_prompt_loss = 0.0
    memory_bank = []  # for 2D only

    # detect 2D vs 3D
    dim = train_loader.dataset.data_dimension  # 2 or 3
    progress = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")

    for batch in progress:
        if dim == 2:
            optimizer2d = optimizers[0]
            batch_loss = train_step_2d(student, teacher, optimizer2d, batch, config, memory_bank, scaler)
        else:
            mask_decoder_opt, memory_opt = optimizers
            batch_loss, prompt_loss, non_prompt_loss = train_step_3d(student, teacher, mask_decoder_opt, memory_opt, batch, config, scaler)
            total_prompt_loss += prompt_loss
            total_non_prompt_loss += non_prompt_loss
        total_batch_loss += batch_loss
        progress.set_postfix(batch_loss=f"{batch_loss:.4f}")

    n = len(train_loader)
    avg_batch_loss = total_batch_loss / n
    
    if dim == 2:
        logger.info(f"[Epoch {epoch}] Average batch loss: {avg_batch_loss:.4f}")
    else:
        avg_prompt_loss     = total_prompt_loss / n
        avg_non_prompt_loss = total_non_prompt_loss / n
        logger.info(f"[Epoch {epoch}] Average loss: {avg_batch_loss:.4f}, Prompt loss: {avg_prompt_loss:.4f}, Non-prompt loss: {avg_non_prompt_loss:.4f}")

    # step the LR scheduler once per epoch
    if scheduler is not None:
        scheduler.step()

    if dim == 2:
        return {
            "avg_batch_loss": avg_batch_loss
        }
    else:
        return {
            "avg_batch_loss": avg_batch_loss,
            "avg_prompt_loss": avg_prompt_loss,
            "avg_non_prompt_loss": avg_non_prompt_loss
        }

def validate_epoch(student, val_loader, config):
    """
    Run one full validation epoch, calling validate_step_2d or validate_step_3d
    on each batch and averaging the per-batch metrics.

    Visualizes a single sample from the first batch for both 2D and 3D.

    Args:
        student (SAM2VideoPredictor): the student model in eval mode
        val_loader (DataLoader): validation loader exposing .dataset.data_dimension
        config (DictConfig): configuration with .training.device, .training.out_size, .prompt.max_per_seq

    Returns:
        dict: average metrics {'iou', 'dice', 'hd95'} over all batches
    """
    # Initialize model in eval
    student.eval()

    # Pick 2D vs 3D
    dim = val_loader.dataset.data_dimension  # 2 or 3

    # Initialize metrics
    total = {'iou': 0.0, 'dice': 0.0, 'hd95': 0.0}
    n_batches = 0

    # Run validation
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Select validation step based on data dimension
            if dim == 2:
                # Only visualize first batch
                if batch_idx == 0:
                    metrics, logits = validate_step_2d(student, batch, config, return_logits=True)
                    # Visualize single sample
                    img = batch['image'][0, 0]
                    msk = batch['mask'][0, 0]
                    pred = logits[0]
                    visualize_predictions_2d(img, msk, pred)
                else:
                    metrics = validate_step_2d(student, batch, config)
            else:
                # Only visualize first batch
                if batch_idx == 0:
                    metrics, logits = validate_step_3d(student, batch, config, return_logits=True)
                    # Image mask paths from dataset
                    img_path  = batch['image'][0]
                    msk_paths = batch['mask'][0]
                    visualize_nifti_predictions(img_path, msk_paths, logits)
                else:
                    metrics = validate_step_3d(student, batch, config)

            # Accumulate metrics
            total['iou']  += metrics['iou']
            total['dice'] += metrics['dice']
            total['hd95'] += metrics['hd95']
            n_batches     += 1

    # Average metrics over all batches
    avg = {k: total[k] / n_batches for k in total}
    return avg

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
    # 1) Load config + logging
    config  = load_config(config_path)
    log_cfg = config.logging
    logger  = setup_logger(log_cfg)

    # 2) Data loaders
    train_loaders, val_loaders, test_loaders = get_data_loaders(config)

    # 3) Models
    student = build_student_predictor(config)
    teacher = build_teacher_predictor(config)

    # 4) optimizer + scheduler + scaler
    # 2D optimizer + scheduler
    opt2d_cfg = setup_optimizer_and_scheduler(student, config, data_dimension=2)
    optimizer2d, sched2d = opt2d_cfg['optimizers'][0], opt2d_cfg['scheduler']

    # 3D optimizer + scheduler
    opt3d_cfg = setup_optimizer_and_scheduler(student, config, data_dimension=3)
    mask_decoder_opt, memory_opt = opt3d_cfg['optimizers']
    sched3d = opt3d_cfg['scheduler']

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 5) Multi-GPU setup
    if multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        student = torch.nn.DataParallel(student)
        teacher = torch.nn.DataParallel(teacher)

    # 6) Load from checkpoint if resuming
    ckpt_mgr = CheckpointManager(config.training.output_dir)
    start_ep = 0
    if resume:
        start_ep, opt1_state, opt2_state, sched_state = ckpt_mgr.load(resume, student)
        # load checkpoint weights into both 2D and 3D optimizers/schedulers
        optimizer2d.load_state_dict(opt1_state)
        mask_decoder_opt.load_state_dict(opt1_state)
        memory_opt.load_state_dict(opt2_state)
        sched2d.load_state_dict(sched_state)
        sched3d.load_state_dict(sched_state)
        logger.info(f"Resumed from epoch {start_ep}")

    # 7) Training and validation loops over each sub-dataset
    train_loss, val_metrics = None, None
    for epoch in range(start_ep, int(config.training.epochs)):
        # Train: Iterate through all 2D and 3D loaders
        for key, loader in train_loaders.items():
            dim = loader.dataset.data_dimension
            if dim == 2:
                train_loss[key] = train_epoch(student, teacher, loader, [optimizer2d], sched2d, config, scaler, epoch, logger)
            else:
                train_loss[key] = train_epoch(student, teacher, loader, [mask_decoder_opt, memory_opt], sched3d, config, scaler, epoch, logger)

        # Step both schedulers once per epoch
        sched2d.step()
        sched3d.step()

        # Validation: Evaluate each loader at the configured frequency
        if epoch % config.training.val_freq == 0 or epoch == int(config.training.epochs) - 1:
            for key, loader in val_loaders.items():
                dim = loader.dataset.data_dimension
                val_metrics[key] = validate_epoch(student, loader, config)

                # Save per-modality checkpoints
                if dim == 2:
                    ckpt_mgr.save_latest(student, optimizer2d, optimizer2d, sched2d, epoch)
                    ckpt_mgr.save_best(  student, optimizer2d, optimizer2d, sched2d, epoch, val_metrics[key]['iou'])
                else:
                    ckpt_mgr.save_latest(student, mask_decoder_opt, memory_opt, sched3d, epoch)
                    ckpt_mgr.save_best(  student, mask_decoder_opt, memory_opt, sched3d, epoch, val_metrics[key]['iou'])

    logger.info("Training complete.")

if __name__ == "__main__":
    import argparse

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
