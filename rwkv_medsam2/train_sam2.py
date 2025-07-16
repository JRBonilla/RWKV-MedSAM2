import os
import json
import argparse
import random
import re
import logging
import datetime

import yaml
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms.functional as F
import SimpleITK as sitk

from ext.sam2.build_sam import build_sam2
from ext.sam2.sam2_image_predictor import SAM2ImagePredictor

TRAIN_LOG_DIR = "/data/TrainingLogs"

# Configure logger: log to both console and a file named "train_<timestamp>.log"
logger = logging.getLogger("TrainSAM2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # 1) Stream (console) handler.
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    # 2) File handler: logs saved to LOG_DIR/train_<timestamp>.log
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(INDEX_DIR, f"train_{timestamp}.log")
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file_path}")

class SelfSortingMemoryBank:
    """
    A memory bank that stores embeddings sorted by confidence.
    """
    def __init__(self, capacity=16):
        """
        Initializes the memory bank with a given capacity.

        Args:
            capacity (int, optional): The maximum number of embeddings
                that the memory bank can store. Defaults to 16.
        """
        self.capacity = capacity
        self.embeddings = []
    
    def update(self, new_embed, confidence):
        """
        Add a new embedding to the memory bank with its confidence.

        The memory bank is sorted by confidence in descending order, and
        only keeps the most confident 'capacity' embeddings. If the memory
        bank is full, the least confident embedding is discarded.

        Args:
            new_embed (torch.Tensor): The new embedding to add.
            confidence (float): The confidence of the new embedding.
        """
        self.embeddings.append((new_embed, confidence))
        self.embeddings.sort(key=lambda x: x[1], reverse=True)
        self.embeddings = self.embeddings[:self.capacity]

    def get_memory(self):
        """
        Returns the memory as a tensor of shape (capacity, embedding_dim),
        or None if the memory bank is empty.
        """
        if not self.embeddings:
            return None
        return torch.stack([e for e, _ in self.embeddings])

class BalancedTaskSampler(Sampler):
    """
    A sampler that samples sequences from the groupings, balancing the
    distribution of tasks.
    """
    def __init__(self, groupings, tasks_map, seq_len=8, max_random=2):
        """
        Initializes the sampler with the given groupings and tasks map.

        Args:
            groupings (list): A list of dictionaries containing information about
                the groupings, such as the dataset and subdataset.
            tasks_map (dict): A dictionary with task IDs as keys and
                dictionaries containing information about the task as values.
                The dictionaries should have the following keys:
                - 'datasets' (dict): A dictionary with dataset IDs as keys and
                    subdataset IDs as values.
                - 'classes' (set): A set of class IDs.
            seq_len (int, optional): The sequence length to sample. Defaults to 8.
            max_random (int, optional): The maximum number of frames to randomly
                drop from the sequence. Defaults to 2.
        """
        self.groupings = groupings
        self.tasks_map = tasks_map
        self.seq_len = seq_len
        self.max_random = max_random
        self.by_task = {}

        # Precompute valid indices per task
        for task_id, info in self.tasks_map.items():
            valid = []
            for idx, grp in enumerate(groupings):
                ds = grp['dataset']
                sd = grp.get('subdataset')
                # Check dataset/subdataset eligibility
                if ds in info['datasets'] and sd in info['datasets'][ds]:
                    # Check class intersection
                    grp_classes = set(grp.get('mask_classes', {}).keys())
                    if grp_classes & info['classes']:
                        valid.append(idx)
            if valid:
                self.by_task[task_id] = valid
        self.tasks = list(self.by_task.keys())

    def __iter__(self):
        """
        Randomly sample a sequence from the groupings.

        The sampling process works as follows:

        1. Pick a random task from the tasks available in the groupings.
        2. Pick a random grouping for that task.
        3. Sample a contiguous window of length seq_len from the sequence.
        4. Drop a random number of frames from the window (up to max_random).

        The sequence is then yielded.
        """
        while True:
            # 1) Pick a random task
            task = random.choice(self.tasks)
            indices = self.by_task[task]
            #2) Pick a random grouping for that task
            grp_idx = random.choice(indices)
            sequence = self.groupings[grp_idx]['pairs']
            n = len(sequence)
            if n < self.seq_len:
                continue
            # 3) Sample contiguous window
            start = torch.randint(0, n - self.seq_len + 1, (1,)).item()
            subseq = sequence[start:start + self.seq_len]
            # 4) Drop random frames
            drop = random.randint(0, self.max_random + 1, (1,)).item()
            drop_idx = torch.randperm(self.seq_len)[:drop]
            yield [p for i, p in enumerate(subseq) if i not in drop_idx]
    
    def __len__(self):
        """
        Get the total number of possible sequences.

        Returns the sum of the lengths of the valid indices for each task.
        """
        return sum(len(idxs) for idxs in self.by_task.values())

class GroupingDataset(Dataset):
    def __init__(self, groupings, transform=None):
        self.groupings = groupings
        self.transform = transform

    def __len__(self):
        return len(self.groupings)

    def __getitem__(self, idx):
        group = self.groupings[idx]
        imgs, masks = [], []
        for img_path, mask_path in group:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = torch.from_numpy(arr).float()
            if img.dim() == 2:
                img = img.unsqueeze(0)
            m_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask  = torch.from_numpy(m_arr).long()
            if self.transform:
                img, mask = self.transform(img, mask)
            imgs.append(img)
            masks.append(mask)
        return torch.stack(imgs, dim=1), torch.stack(masks, dim=1)
    
def get_transforms():
    """
    Returns a callable that randomly crops and resizes a given image and mask to
    the target size. The cropping is done by sampling a random top-left corner
    and a random width and height from a uniform distribution, such that the
    cropped region is within the image and has an aspect ratio between 0.9 and
    1.1. The cropped region is then resized to the target size using bilinear
    interpolation.

    Args:
        image (torch.Tensor): The image to be cropped and resized.
        mask (torch.Tensor): The corresponding mask to be cropped and resized.

    Returns:
        tuple: A tuple containing the cropped and resized image and mask.
    """
    def random_resized_crop(image, mask):
        target_size = (512, 512)
        top, left, height, width = F.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        cropped_image = F.resized_crop(image, top, left, height, width, target_size)
        cropped_mask  = F.resized_crop(mask, top, left, height, width, target_size)
        return cropped_image, cropped_mask
    return random_resized_crop

def get_groupings(out_dir):    
    """
    Loads the groupings.json files from each folder in the given output directory.

    Each groupings.json file is expected to contain a list of entries, where each
    entry is a dictionary with the following keys:
    - "proc_images": A list of paths to the preprocessed images.
    - "proc_masks": A list of paths to the preprocessed masks.

    The function pairs each mask with its corresponding image by searching for
    the index in the image path. The index is assumed to be in the format
    "(?:img|frame|slice)_(\d+)" and is used to match the mask with its
    corresponding image.

    Args:
        out_dir (str): The output directory containing the groupings.json files.

    Returns:
        list: A list of lists, where each inner list contains pairs of image and
              mask paths.
    """
    _idx_pattern = re.compile(r"_(?:img|frame|slice)_(\d+)")
    groups = []
    for ds in sorted(os.listdir(out_dir)):
        ds_dir = os.path.join(out_dir, ds)
        grp_file = os.path.join(ds_dir, 'groupings.json')
        if not os.path.isfile(grp_file):
            continue
        with open(grp_file) as f:
            entries = json.load(f)
        for entry in entries:
            imgs = entry.get('proc_images', [])
            msks = entry.get('proc_masks', [])
            idx_map = {}
            for img_path in imgs:
                m = _idx_pattern.search(os.path.basename(img_path))
                if m:
                    idx_map[int(m.group(1))] = img_path
            pairs = []
            for mpath in msks:
                m = _idx_pattern.search(os.path.basename(mpath))
                if not m:
                    continue
                img_p = idx_map.get(int(m.group(1)))
                if img_p:
                    pairs.append((img_p, mpath))
            if pairs:
                groups.append({
                    'dataset': ds,
                    'subdataset': entry.get('subdataset_name'),
                    'tasks': entry.get('tasks', []),
                    'mask_classes': entry.get('mask_classes', {}),
                    'pairs': pairs
                })
    return groups

def generate_prompt(mask_tensor, mask_prob=0.5, click_prob=0.25):
    """
    Generate a prompt for training SAM2 from a given mask tensor.

    The prompt can be one of three types: a full mask, a positive click, or a bounding box.
    The type of prompt is determined by random sampling from uniform distributions.

    Args:
        mask_tensor (torch.Tensor): The input mask tensor.
        mask_prob (float, optional): The probability of generating a full mask prompt.
        click_prob (float, optional): The probability of generating a positive click prompt.

    Returns:
        dict: A dictionary containing the prompt, which can be one of the following:
            {'mask': torch.Tensor}: A full mask prompt.
            {'points': [(int, int)], 'labels': [int]}: A positive click prompt.
            {'bbox': (int, int, int, int)}: A bounding box prompt.
    """
    r = random.random()
    # Full mask prompt
    if r < mask_prob:
        return {'mask': mask_tensor}
    # Positive click prompt
    elif r < mask_prob + click_prob:
        # Find a random foreground pixel
        fg = (mask_tensor > 0).nonzero()
        if fg.numel() == 0:
            return {'mask': mask_tensor}
        idx = random.randint(0, fg.size(0) - 1)
        y, x = fg[idx].tolist()
        return {'points': [(x, y)], 'labels': [1]}
    # Bounding box prompt
    else:
        # Compute bounding box coords from mask
        ys, xs = (mask_tensor > 0).nonzero().unbind(1)
        if ys.numel() == 0:
            return {'mask': mask_tensor}
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        return {'bbox': (x1, y1, x2, y2)}

def train(config_path, checkpoint_dir, resume_from):
    """
    Train SAM2 with the given config file.

    This function takes as input a YAML config file, which must contain the following
    fields:

    - 'dripp': A dictionary containing the output directory and tasks file of the
      DRIPP pipeline.
    - 'sampler': A dictionary containing the sequence length and maximum number of
      frames to randomly sample from each sequence.
    - 'training': A dictionary containing the device, batch size, and number of
      workers for the data loader.
    - 'memory_bank': A dictionary containing the capacity of the memory bank.
    - 'prompt': An optional dictionary containing the maximum number of prompts per
      sequence, the probability of generating a full mask prompt, and the probability
      of generating a positive click prompt.
    - 'model': A dictionary containing the backbone architecture and SAM2
      hyperparameters.

    If 'resume_from' is provided and the file exists, training will resume from there.

    Args:
        config_path (str): The path to the YAML config file.
        checkpoint_dir (str): Directory where checkpoints will be saved.
        resume_from (str or None): Path to a checkpoint `.pth` to resume from.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(config_path) as cf:
        cfg = yaml.safe_load(cf)

    # DRIPP outputs
    out_dir = cfg['dripp']['output_dir']
    tasks_map = json.load(open(cfg['dripp']['tasks_file']))
    
    # Load groupings and transforms
    groupings = get_groupings(out_dir)
    transform = get_transforms()
    
    # Sampler and dataset
    sampler = BalancedTaskSampler(
        groupings,
        tasks_map,
        seq_len=cfg['sampler']['seq_len'],
        max_random=cfg['sampler']['max_random']
    )
    dataset = GroupingDataset(groupings, transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=sampler,
        num_workers=cfg['training']['num_workers']
    )
    
    # Initialize student model via build_sam2
    device = cfg['training']['device']
    model = build_sam2(
        config_file=config_path,
        ckpt_path=cfg['model'].get('ckpt_path', None),
        device=device,
        mode='train',
        apply_postprocessing=False
    )

    # Initialize teacher
    teacher = SAM2ImagePredictor.from_pretrained(
        cfg['model']['sam2']['model_name']
    )
    teacher.model.to(device).eval()

    # Optimizer, criterion, memory bank
    mem_bank = SelfSortingMemoryBank(capacity=cfg['memory_bank']['capacity'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    # Resume checkpoint if provided
    if resume_from is not None and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch']
        # Restore memory bank contents
        mem_bank.embeddings = ckpt.get('memory_bank', [])
        logger.info(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

    model.train()
    logger.info(f"Beginning training for {cfg['training']['epochs']} epochs (resuming at epoch {start_epoch+1})")
    for epoch in range(start_epoch, cfg['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{cfg['training']['epochs']} start")
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss_total = 0
            seq_len = imgs.size(1)

            # Prompt settings
            num_prompts = random.randint(1, cfg.get('prompt', {}).get('max_per_seq', 2))
            prompt_frames = set(random.sample(range(seq_len), num_prompts))
            
            for t in range(seq_len):
                img_t, msk_t = imgs[:, t], masks[:, t]
                mem = mem_bank.get_memory()
                # Generate prompt for selected frames
                prompt = generate_prompt(
                    msk_t,
                    cfg.get('prompt', {}).get('mask_prob', 0.5),
                    cfg.get('prompt', {}).get('click_prob', 0.5)
                ) if t in prompt_frames else None

                # Inject teacher embeddings
                # 1) Convert torch [B, 3, H, W] -> [H, W, 3] for first batch item
                np_img = img_t[0].permute(1, 2, 0).cpu().numpy()
                # 2) Compute teacher._features
                teacher.set_image(np_img)
                # 3) Monkey-patch student to use teacher._features
                model._features     = teacher._features
                model._is_image_set = True
                model._orig_hw      = teacher._orig_hw
                model._is_batch     = False

                # Forward with prompt and memory
                pred, embed, conf = model(img_t, prompt=prompt, memory=mem)
                loss_total += criterion(pred, msk_t)
                mem_bank.update(embed.detach(), conf.detach())
            loss_total.backward()
            optimizer.step()
            epoch_loss += loss_total.item()

        # End of epoch: save checkpoint and print loss
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'memory_bank': mem_bank.embeddings,
        }, ckpt_path)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} loss: {epoch_loss:.4f}. checkpoint saved to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAM2 model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory where to save and load checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a checkpoint .pth to resume from')
    args = parser.parse_args()

    train(args.config, args.checkpoint_dir, args.resume_from)
    logger.info("Training complete.")