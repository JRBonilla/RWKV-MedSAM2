import os
import re
import json
import math
import random
import logging

import cupy as cp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf, DictConfig

import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from torchvision.transforms import functional as TF

from ext.sam2.predictor import build_sam2_video_predictor, SAM2VideoPredictor

from dripp.helpers import normalize_path, get_extension

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

def compute_iou(pred_mask, gt_mask):
    """
    Returns the Intersection over Union (IoU) between two binary masks.
    The IoU is defined as:
        IoU = |pred_mask & gt_mask| / |pred_mask | gt_mask|
    
    Args:
        pred_mask (torch.Tensor): A binary mask of shape (H, W).
        gt_mask (torch.Tensor): A binary mask of shape (H, W).
    
    Returns:
        float: The IoU between the two masks.
    """
    p = cp.asarray(pred_mask.detach().cpu().numpy())
    g = cp.asarray(gt_mask.detach().cpu().numpy())
    intersection = cp.logical_and(p, g).sum()
    union = cp.logical_or(p, g).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def compute_dice(pred_mask, gt_mask):
    """
    Returns the Dice Similarity Coefficient (DSC) between two binary masks.
    The DSC is defined as:
        DSC = 2 * |pred_mask & gt_mask| / (|pred_mask| + |gt_mask|)
    
    Args:
        pred_mask (torch.Tensor): A binary mask of shape (H, W).
        gt_mask (torch.Tensor): A binary mask of shape (H, W).
    
    Returns:
        float: The DSC between the two masks.
    """
    p = cp.asarray(pred_mask.detach().cpu().numpy())
    g = cp.asarray(gt_mask.detach().cpu().numpy())
    intersection = cp.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0

def compute_hd95(pred_mask, gt_mask):
    """
    Returns the Hausdorff Distance 95 (HD95) between two binary masks.
    The HD95 is the 95th percentile of all boundary-to-boundary distances
    between the two masks. HD95 is defined as:
        hd95 = max(directed_hausdorff(b_p, b_g)[0], directed_hausdorff(b_g, b_p)[0])
        where:
            b_p = boundary_pts(pred_mask)
            b_g = boundary_pts(gt_mask)
    
    Args:
        pred_mask (torch.Tensor): A binary mask of shape (H, W).
        gt_mask (torch.Tensor): A binary mask of shape (H, W).
    
    Returns:
        float: The HD95 between the two masks.
    """
    # Extract boundary points
    def boundary_pts(mask):
        arr = mask.cpu().numpy().astype('uint8')
        sitk_img = sitk.GetImageFromArray(arr)
        contour = sitk.LabelContour(sitk_img)
        return cp.argwhere(sitk.GetArrayFromImage(contour) > 0)
    b_p = cp.asnumpy(boundary_pts(pred_mask))
    b_g = cp.asnumpy(boundary_pts(gt_mask))
    if len(b_p) == 0 or len(b_g) == 0:
        return 0
    # Directed distances botw ways
    d1 = directed_hausdorff(b_p, b_g)[0]
    d2 = directed_hausdorff(b_g, b_p)[0]
    return max(d1, d2)

class SegmentationSequenceDataset(Dataset):
    def __init__(self, pairings, transform):
        """
        Initializes a SegmentationSequenceDataset.
        
        Args:
            pairings (list[dict]): A list of dictionaries, each containing 'pairs' which is a list of tuples,
                where each tuple contains a path to an image and a path to its corresponding mask.
            transform (callable): An optional transformation to be applied to every image and mask pair.
        """
        self.pairings = pairings
        self.transform = transform

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, idx):
        entry = self.pairings[idx]
        pairs = entry['pairs']

        # Reset transformations
        if self.transform is not None:
            self.transform.reset()

        # Normalize and detect extension
        img0 = normalize_path(pairs[0][0])
        ext = get_extension(img0)

        # 3D if exactly one pair and NIfTI
        if len(pairs) == 1 and ext in {'.nii', '.nii.gz'}:
            return self._load_3d(img0, normalize_path(pairs[0][1]))
        else:
            return self._load_2d_sequence([(normalize_path(i), normalize_path(m)) for i, m in pairs])

    def _load_2d_sequence(self, pairs):
        imgs, masks = [], []
        for imp_p, msk_p in pairs:
            i_arr = sitk.GetArrayFromImage(sitk.ReadImage(imp_p))
            m_arr = sitk.GetArrayFromImage(sitk.ReadImage(msk_p))
            im_t, ms_t = self._to_tensor(i_arr, m_arr)
            if self.transform:
                im_t, ms_t = self.transform(im_t, ms_t)
            imgs.append(im_t)
            masks.append(ms_t)
        return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)

    def _load_3d(self, img_path, mask_path):
        """
        Load a 3D image and its mask as a sequence of 2D images. This function slices the 3D volumes
        along each axis (D, H, W) and loads each slice as a 2D image. The slices are then transformed
        (if a transformation is provided) and returned as a sequence of 2D images.

        Args:
            img_path (str): Path to the 3D image.
            mask_path (str): Path to the 3D mask.

        Returns:
            tuple: (seq_imgs, seq_masks) where seq_imgs is a sequence of 2D images and seq_masks is
                the corresponding sequence of 2D masks.
        """
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)) # [D, H, W]
        msk_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        seq_imgs, seq_masks = [], []
        # For each axis, gather slices forward and reverse
        for axis in (0, 1, 2):
            length = msk_vol.shape[axis]
            for i in range(length):
                slice_img = self._slice(img_vol, axis, i)
                slice_msk = self._slice(msk_vol, axis, i)
                im_t, ms_t = self._to_tensor(slice_img, slice_msk)
                if self.transform:
                    im_t, ms_t = self.transform(im_t, ms_t)
                seq_imgs.append(im_t)
                seq_masks.append(ms_t)
            for i in reversed(range(length)):
                slice_img = self._slice(img_vol, axis, i)
                slice_msk = self._slice(msk_vol, axis, i)
                im_t, ms_t = self._to_tensor(slice_img, slice_msk)
                if self.transform:
                    im_t, ms_t = self.transform(im_t, ms_t)
                seq_imgs.append(im_t)
                seq_masks.append(ms_t)
        return torch.stack(seq_imgs, dim=0), torch.stack(seq_masks, dim=0)

    def _slice(self, vol, axis, idx):
        """
        Helper function to slice a 3D volume along a given axis.
        
        Args:
            vol (numpy.ndarray): The 3D volume to slice.
            axis (int): The axis along which to slice the volume.
            idx (int): The index of the slice to extract.
        
        Returns:
            numpy.ndarray: The sliced 2D image.
        """
        if axis == 0:
            return vol[idx, :, :]
        elif axis == 1:
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    def _to_tensor(self, img_arr, msk_arr):
        """
        Convert NumPy arrays to PyTorch tensors.
        
        Args:
            img_arr (numpy.ndarray): The image array.
            msk_arr (numpy.ndarray): The mask array.
        
        Returns:
            tuple: A tuple containing the image tensor and the mask tensor.
        """
        img = torch.from_numpy(img_arr).float()
        if img.ndim == 2:               # Single channel
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)  # HxWxC -> CxHxW
        mask = torch.from_numpy(msk_arr).long()
        return img, mask

class BalancedTaskSampler(Sampler):
    def __init__(self, pairings, tasks_map, seed):
        """
        Initializes the sampler with the given pairings and tasks map.

        Args:
            pairings (list[dict]): A list of dictionaries, each containing information about the
                dataset/subdataset.
            tasks_map (dict): A dictionary with task IDs as keys and dictionaries containing
                information about the task as values.
            seed (int, optional): An optional seed for the random number generator. If not provided,
                the sampler will use the current system time as the seed.
        """
        self.pairings = pairings
        self.tasks_map = tasks_map
        self.num_samples = len(self.pairings)
        self.by_task = {} # task_id -> list of sample indices

        # Precompute valid indices per task
        for task_id, info in self.tasks_map.items():
            valid_idxs = []
            for idx, pair in enumerate(self.pairings):
                ds, sd = pair['dataset'], pair.get('subdataset')
                if ds in info['datasets'] and sd in info['datasets'][ds]:
                    if set(pair.get('mask_classes', {}).keys()) & info['classes']:
                        valid_idxs.append(idx)
            if valid_idxs:
                self.by_task[task_id] = valid_idxs

        self.tasks = list(self.by_task.keys())
        if seed is not None:
            random.seed(seed)

    def __iter__(self):
        """
        Yields a sequence of indices into the pairings, where each index is sampled
        uniformly at random from the valid indices for a randomly chosen task.
        """
        for _ in range(self.num_samples):
            task = random.choice(self.tasks)
            yield random.choice(self.by_task[task])

    def __len__(self):
        """
        Returns the total number of possible samples.
        Returns the number of indices in the pairings.
        """
        return self.num_samples


class SequenceTransform:
    """
    Apply the same random augmentations to every frame in a sequence.
    """
    def __init__(self, base_prob=0.15, lr_prob=0.25, flip_prob=0.5):
        """
        Initializes the SequenceTransform with specified probabilities for
        various augmentations.

        Args:
            base_prob (float, optional): Base probability for most augmentations
                such as rotation, scaling, noise, etc. Defaults to 0.15.
            lr_prob (float, optional): Probability for low-resolution simulation.
                Defaults to 0.25.
            flip_prob (float, optional): Probability for spatial flipping
                (horizontal and/or vertical). Defaults to 0.5.
        """
        self.base_prob = base_prob
        self.lr_prob = lr_prob
        self.flip_prob = flip_prob

    def reset(self):
        """
        Sample augmentation flags and parameters once per sequence.
        All augmentations other than low-resolution simulation and spatial
        flipping have a probability of base_prob. Low-resolution simulation
        has a probability of lr_prob and spatial flipping has a probability
        of flip_prob.

        Augmentations are sampled from the following distributions:
            1. Rotation                  (base_prob, angle in [-25, 25])
            2. Scaling                   (base_prob, scale in [0.7, 1.4])
            3. Gaussian noise            (base_prob, variance in [0, 0.1])
            4. Gaussian blur             (base_prob, kernel in [0.5, 1.5])
            5. Intensity adjustment      (base_prob, factor in [0.65, 1.2] or invert)
            6. Contrast adjustment       (base_prob, factor in [0.65, 1.2])
            7. Low-resolution simulation (lr_prob,   downsampling factor in [1, 2])
            8. Gamma correction          (base_prob, gamma in [0.7, 1.5])
            9. Spatial flip              (flip_prob, horizontal and/or vertical)
        """
        p = self.base_prob

        # 1) Rotation
        self.do_rotate = random.random() < p
        self.angle = random.uniform(-25, 25) if self.do_rotate else 0

        # 2) Scaling
        self.do_scale = random.random() < p
        self.scale = random.uniform(0.7, 1.4) if self.do_scale else 1.0

        # 3) Noise
        self.do_noise = random.random() < p
        self.noise_std = random.uniform(0, 0.1)**0.5 if self.do_noise else 0

        # 4) Blur
        self.do_blur = random.random() < p
        self.blur_sigma = random.uniform(0.5, 1.5) if self.do_blur else 0

        # 5) Intensity
        self.do_intensity = random.random() < p
        if self.do_intensity:
            if random.random() < 0.5:
                self.int_factor = random.uniform(0.65, 1.2)
                self.int_invert = False
            else:
                self.int_factor = 1.0
                self.int_invert = True

        # 6) Contrast
        self.do_contrast = random.random() < p
        self.contrast_factor = random.uniform(0.65, 1.2) if self.do_contrast else 1.0

        # 7) Low-resolution simulation
        self.do_lowres = random.random() < self.lr_prob
        self.down = random.uniform(1.0, 2.0) if self.do_lowres else 1.0

        # 8) Gamma correction
        self.do_gamma = random.random() < p
        self.gamma = random.uniform(0.7, 1.5) if self.do_gamma else 1.0

        # 9) Spatial flip
        self.hflip = random.random() < self.flip_prob
        self.vflip = random.random() < self.flip_prob

    def __call__(self, image, mask):
        """
        Apply sampled transforms to image and mask based on flags.

        Args:
            image (torch.Tensor): Image to transform.
            mask (torch.Tensor): Mask to transform.

        Returns:
            torch.Tensor: Transformed image.
            torch.Tensor: Transformed mask.
        """
        _, h, w = image.shape
        if self.do_rotate:
            image = TF.rotate(image, self.angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  self.angle, interpolation=TF.InterpolationMode.NEAREST)
        if self.do_scale:
            image = TF.affine(image, angle=0, translate=(0,0), scale=self.scale, shear=[0,0], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.affine(mask,  angle=0, translate=(0,0), scale=self.scale, shear=[0,0], interpolation=TF.InterpolationMode.NEAREST)
        if self.do_noise:
            image = image + torch.randn_like(image) * self.noise_std
        if self.do_blur:
            k = max(3, int(2 * round(self.blur_sigma * 2) + 1))
            image = TF.gaussian_blur(image, kernel_size=k, sigma=[self.blur_sigma, self.blur_sigma])
        if self.do_intensity:
            image = (1.0 - image) if self.int_invert else (image * self.int_factor)
            image = image.clamp(0,1)
        if self.do_contrast:
            image = TF.adjust_contrast(image, self.contrast_factor).clamp(0,1)
        if self.do_lowres:
            small_h, small_w = max(1,int(h/self.down)), max(1,int(w/self.down))
            image = TF.resize(image, [small_h, small_w], interpolation=TF.InterpolationMode.NEAREST)
            image = TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask.unsqueeze(0).float(), [small_h, small_w], interpolation=TF.InterpolationMode.NEAREST)
            mask  = TF.resize(mask, [h, w], interpolation=TF.InterpolationMode.NEAREST).long().squeeze(0)
        if self.do_gamma:
            image = image.clamp(0,1).pow(self.gamma)
        if self.hflip:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.vflip:
            image, mask = TF.vflip(image), TF.vflip(mask)
        return image, mask

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
    _idx_pattern = re.compile(r"_(?:img|frame|slice)(\d+)")
    all_pairs = []
    for ds in sorted(os.listdir(out_dir)):
        ds_dir = os.path.join(out_dir, ds)
        grp_file = os.path.join(ds_dir, 'groupings.json')
        if not os.path.isfile(grp_file):
            continue
        with open(grp_file) as f:
            entries = json.load(f)
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
                m = _idx_pattern.search(os.path.basename(mpath))
                if not m:
                    continue
                img_p = idx_map.get(int(m.group(1)))
                if img_p:
                    pairs.append((img_p, mpath))
            if pairs:
                # Sort the pairs to ensure temporal order
                pairs.sort(key=lambda x: int(_idx_pattern.search(os.path.basename(x[0])).group(1)))
                if len(pairs) > 1 and get_extension(pairs[0][0]) == '.png':
                    for img_p, m_p in pairs:
                        all_pairs.append({
                            'dataset': ds,
                            'subdataset': entry.get('subdataset_name'),
                            'tasks': entry.get('tasks', []),
                            'mask_classes': entry.get('mask_classes', {}),
                            'pairs': [(img_p, m_p)] # Single frame 2D
                        })
                else:
                    all_pairs.append({
                        'dataset': ds,
                        'subdataset': entry.get('subdataset_name'),
                        'tasks': entry.get('tasks', []),
                        'mask_classes': entry.get('mask_classes', {}),
                        'pairs': pairs
                    })
    return all_pairs

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

def get_data_loaders(config):
    """
    Create train and validation data loaders.

    1) Load all DRIPP pairings.
    2) Split into train/val by config.training.val_frac.
    3) Instantiate map-style datasets (with augment only on train).
    4) Build a BalancedTaskSampler over the train set.
    5) Return train_loader (with sampler) and val_loader (no sampler, no shuffle).

    Args:
        config (dict): The configuration dictionary.

    Returns:
        train_loader (torch.utils.data.DataLoader): The train data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
    """
    # 1) Load all DRIPP pairings
    all_pairings = get_pairings(config.dripp.output_dir, split='train')

    # 2) Split into train/val by config.training.val_frac
    random.seed(config.training.seed)
    random.shuffle(all_pairings)
    n_val       = int(len(all_pairings) * config.training.val_frac)
    val_pairs   = all_pairings[:n_val]
    train_pairs = all_pairings[n_val:]

    # 3) Instantiate map-style datasets (with augment only on train)
    train_ds = SegmentationSequenceDataset(pairings=train_pairs, transform=SequenceTransform())
    val_ds   = SegmentationSequenceDataset(pairings=val_pairs, transform=None)

    # 4) Build a BalancedTaskSampler over the train set
    tasks_map = json.load(open(config.dripp.tasks_file))
    train_sampler = BalancedTaskSampler(
        pairings=train_pairs,
        tasks_map=tasks_map,
        seed=config.training.seed
    )

    # 5) Return train_loader (with sampler) and val_loader (no sampler, no shuffle)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def build_student_model(config):
    """
    Build the student video predictor.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        video_predictor (SAM2VideoPredictor): The student video predictor.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config._config_path,
        ckpt_path=config.model.get('ckpt_path', None),
        device=config.training.device,
        mode='train',
        apply_postprocessing=False,
    )
    video_predictor.train()
    return video_predictor

def build_teacher_predictors(config):
    """
    Builds the teacher video predictor for SAM2 training. The teacher model is 
    used for distillation.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        video_predictor (SAM2VideoPredictor): The teacher video predictor.
    """
    video_predictor = build_sam2_video_predictor(
        config_file=config._config_path,
        ckpt_path=config.model.teacher.ckpt_path,
        device=config.training.device,
        mode='eval',
        apply_postprocessing=False,
    )
    video_predictor.model.eval()
    return video_predictor

def setup_optimizer_and_scheduler(model, config):
    """
    Create an AdamW optimizer and composite LR scheduler:
        - Linear warm-up for the first 'warmup_epochs'
        - Cosine decay for the remaining epochs

    Uses:
        config.training.lr            : base learning rate
        config.training.epochs        : total number of epochs
        config.training.warmup_epochs : number of warmup epochs (can be 0)

    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): The configuration dictionary.
    Returns:
        optimizer (torch.optim.AdamW): The optimizer.
        scheduler (torch.optim.lr_scheduler.CosineAnnealingLR): The scheduler.
    """
    # 1) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.lr))

    # 2) Scheduler
    total_epochs = int(config.training.epochs)
    warmup_epochs = int(getattr(config.training, 'warmup_epochs', 0))

    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Ramp linearly from 0 to 1 over warmup
                return float(epoch + 1) / float(warmup_epochs)
            else:
                # Cosine decay from 1 to 0 over remaining epochs
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        def lr_lambda(epoch):
            # Pure cosine decay from 1 to 0 over all epochs
            progress = float(epoch) / float(max(1, total_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

class CheckpointManager:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.latest_path = os.path.join(output_dir, "latest.pth")
        self.best_path   = os.path.join(output_dir, "best.pth")
        self.best_metric = -float("inf")

    def save_latest(self, model, optimizer, scheduler, epoch):
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, self.latest_path)

    def save_best(self, model, optimizer, scheduler, epoch, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metric":    metric,
            }, self.best_path)

    def load(self, checkpoint_path, model, optimizer=None, scheduler=None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt.get("epoch", 0)

def compute_distillation_loss(s_lowres, t_lowres, s_embeds, t_embeds, embed_weight=1.0, dist_weight=1.0):
    """
    Compute the distillation loss between the student and teacher:
        1) MSE between student and teacher low-res mask logits
        2) MSE between student and teacher image embeddings

    Args:
        s_lowres (torch.Tensor): [T,1,h,w] student low-res mask logits.
        t_lowres (torch.Tensor): [T,1,h,w] teacher low-res mask logits.
        s_embeds (torch.Tensor): [T,D] student image embeddings.
        t_embeds (torch.Tensor): [T,D] teacher image embeddings.
        embed_weight (float): Weight for the image embedding-MSE term.
        dist_weight (float): Weight for the whole distillation loss.
    Returns:
        dist_loss (torch.Tensor): The distillation loss.
    """
    loss_lowres = F.mse_loss(s_lowres, t_lowres)
    loss_embeds = F.mse_loss(s_embeds, t_embeds)
    return dist_weight * (loss_lowres + embed_weight * loss_embeds)

def _run_sequence(predictor, seq_imgs, seq_prompts, config, is_student=False):
    """
    Process a sequence of images through the SAM2 video predictor.

    This function runs a sequence of images through the SAM2 video predictor,
    handling both teacher and student models. It computes the necessary
    features, memory attentions, and prompt encodings to generate mask logits,
    low-resolution logits, and image embeddings.

    If the model is a student model, it also updates the memory bank.
    Otherwise, it simply processes the sequence using the teacher model.

    Args:
        predictor (SAM2VideoPredictor): The SAM2 video predictor instance.
        seq_imgs (torch.Tensor): The input sequence of images with shape [T, C, H, W].
        seq_prompts (list): A list of prompts for each frame in the sequence.
        config (dict): Configuration parameters for the processing.
        is_student (bool, optional): Flag indicating if the model is a student model.

    Returns:
        tuple: A tuple containing:
            - logits (torch.Tensor): Mask logits for each frame.
            - lowres (torch.Tensor): Low-resolution logits for each frame.
            - embeds (torch.Tensor): Image embeddings for each frame.
    """

    device = config.training.device
    T, C, H, W = seq_imgs.shape
    K      = config.memory_bank.capacity
    cthresh= config.memory_bank.c_thresh

    if not is_student:
        state = predictor.train_init_state(seq_imgs.unsqueeze(0))  # [1, T, C, H, W]
    else:
        memory_bank = []  # list of (feat, pos, iou, embed)

    logits, lowres, embeds = [], [], []

    for t in range(T):
        frame = seq_imgs[t].unsqueeze(0).to(device)  # [1,C,H,W]
        # 1) Backbone features + positional encodings
        vis_feats, vis_pos = predictor._prepare_backbone_features(frame)

        # 2) Memory attention
        if not is_student:
            mem_feats, mem_pos = state.get_memory()
            if mem_feats is not None:
                vis_feats[-1] = predictor.memory_attention(
                    curr=[vis_feats[-1]],
                    curr_pos=[vis_pos[-1]],
                    memory=mem_feats,
                    memory_pos=mem_pos,
                    num_obj_ptr_tokens=0
                )
        else:
            if memory_bank:
                # Stack bank features, positions, embeddings
                feats = torch.stack([m[0].flatten(2).permute(2,0,1) for m in memory_bank])
                poss  = torch.stack([m[1].flatten(2).permute(2,0,1) for m in memory_bank])
                embs  = F.normalize(torch.stack([m[3] for m in memory_bank]), p=2, dim=1)

                curr_flat = vis_feats[-1].permute(1,0,2).reshape(1, -1)
                curr_norm = F.normalize(curr_flat, p=2, dim=1)
                sims      = (embs @ curr_norm.t()).squeeze(1)           # [K]
                weights   = F.softmax(sims, dim=0)
                idx       = torch.multinomial(weights, num_samples=1).item()

                sel_feat = feats[idx].squeeze(3).permute(1,0,2)
                sel_pos  = poss[idx].squeeze(3).permute(1,0,2)
                mem_feats= sel_feat.reshape(1, sel_feat.size(1), sel_feat.size(2))
                mem_pos  = sel_pos.reshape(1, sel_pos.size(1), sel_pos.size(2))

                vis_feats[-1] = predictor.memory_attention(
                    curr=[vis_feats[-1]],
                    curr_pos=[vis_pos[-1]],
                    memory=mem_feats,
                    memory_pos=mem_pos,
                    num_obj_ptr_tokens=0
                )

        # 3) Prompt encoding + mask decoding
        prompt = seq_prompts[t]
        sparse_emb, sparse_iou = predictor.sam_prompt_encoder(
            prompts=[prompt], orig_size=(H, W)
        )
        mask_logit, iou_logit, lowres_logit = predictor.sam_mask_decoder(
            vis_feats, sparse_emb, image_pos=vis_pos
        )

        # 4) Encode new memory
        # Use high-res masks just decoded to match Medical-SAM2
        highres = F.interpolate(lowres_logit, size=(H, W), mode="bilinear", align_corners=False)
        new_feats, new_pos = predictor._encode_new_memory(
            current_vision_feats=vis_feats,
            feat_sizes=[f.shape[-2:] for f in vis_feats],
            pred_masks_high_res=highres,
            is_mask_from_pts=False
        )
        new_feats = new_feats.to(torch.bfloat16)
        new_pos   = new_pos[0].to(torch.bfloat16)
        new_iou   = iou_logit[0,0].item()
        new_emb   = predictor.get_image_embedding().reshape(-1).detach()

        if not is_student:
            state.add_memory(new_feats, new_pos)
        else:
            # Self-sorting bank update
            for i in range(new_feats.size(0)):
                f, p = new_feats[i].unsqueeze(0), new_pos[i].unsqueeze(0)
                if len(memory_bank) < K:
                    memory_bank.append((f, p, new_iou, new_emb))
                else:
                    flat = torch.stack([m[0].reshape(-1) for m in memory_bank])
                    norm = F.normalize(flat, p=2, dim=1)
                    simm = norm @ norm.t()
                    simm.fill_diagonal_(-float('inf'))

                    nnorm = F.normalize(f.reshape(-1), p=2, dim=0).unsqueeze(1)
                    sims  = (norm @ nnorm).squeeze()

                    i_min = torch.argmin(sims)
                    i_rep = torch.argmax(simm[i_min])
                    if sims[i_min] < simm[i_min, i_rep] and new_iou >= memory_bank[i_rep][2] - 0.1:
                        memory_bank.pop(i_rep)
                        memory_bank.append((f, p, new_iou, new_emb))

        # 5) Collect outputs
        logits.append(mask_logit.squeeze(0))
        lowres.append(lowres_logit.squeeze(0))
        embeds.append(predictor.get_image_embedding().squeeze(0))

    # 6) Stack outputs
    if not is_student:
        predictor.reset_state(state)
    logits = torch.stack(logits, dim=0).unsqueeze(1)
    lowres = torch.stack(lowres, dim=0).unsqueeze(1)
    embeds = torch.stack(embeds, dim=0)
    return logits, lowres, embeds


def train_epoch(student_predictor, teacher_predictor, data_loader, optimizer, scheduler, scaler, config, epoch, logger):
    """
    Perform a single training epoch for the student predictor model.

    This function processes batches of image sequences and their corresponding
    masks, applying random sampling and frame dropping to generate varying
    prompts for training. Both student and teacher models are used to compute
    predictions, with losses calculated for both cross-entropy and
    distillation. The optimizer and learning rate scheduler are updated
    accordingly.

    Args:
        student_predictor (SAM2VideoPredictor):    The student predictor model to be trained.
        teacher_predictor (SAM2VideoPredictor):    The teacher predictor model used for generating target outputs for distillation.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of image sequences and masks.
        optimizer (torch.optim.Optimizer):         Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler):      Learning rate scheduler for adjusting the learning rate.
        scaler (torch.cuda.amp.GradScaler):        Gradient scaler for mixed precision training.
        config (dict):                             Configuration object containing training parameters.
        epoch (int):                               The current epoch number.
        logger (Logger):                           Logger for recording training progress.

    Returns:
        None
    """

    device      = config.training.device
    seq_len     = config.sampler.seq_len
    max_random  = config.sampler.max_random
    max_prompts = config.prompt.max_per_seq
    mask_prob   = config.prompt.mask_prob
    click_prob  = config.prompt.click_prob
    dist_weight = config.training.get('dist_weight', 1.0)

    # Set model to training mode
    student_predictor.model.train()
    total_loss = total_ce = total_dist = 0.0
    total_frames = 0

    # Process batches
    for imgs, masks in data_loader:
        B, T, C, H, W = imgs.shape
        imgs, masks   = imgs.to(device), masks.to(device)

        # 1) Sample window of length <= seq_len
        if T > seq_len:
            start = torch.randint(0, T - seq_len + 1, (1,)).item()
            imgs  = imgs[:, start:start + seq_len]
            masks = masks[:, start:start + seq_len]
            T = seq_len

        # 2) Drop random frames
        if max_random > 0:
            drop_n = random.randint(0, max_random)
            if drop_n > 0:
                drop_idx = set(random.sample(range(T), drop_n))
                keep     = [i for i in range(T) if i not in drop_idx]
                imgs     = imgs[:, keep]
                masks    = masks[:, keep]
                T = imgs.size(1)

        # 3) Generate prompts per sample
        # prompts[b][t] is either None or a Dict
        prompts = [[None] * T for _ in range(B)]
        # Process each sequence in the batch separately
        for b in range(B):
            # Always full-mask on frame 0
            prompts[b][0] = {'mask': masks[b, 0]}
            # Up to max_prompts on other frames
            remaining = list(range(1, T))
            num = random.randint(1, min(len(remaining), max_prompts))
            for t in random.sample(remaining, num):
                prompts[b][t] = generate_prompt(masks[b, t], mask_prob=mask_prob, click_prob=click_prob)

        optimizer.zero_grad()
        for b in range(B):
            seq_imgs    = imgs[b]    # [T, C, H, W]
            seq_masks   = masks[b]   # [T, H, W]
            seq_prompts = prompts[b] # [T]

            # Teacher forward
            t_logits, t_lowres, t_embeds = _run_sequence(
                teacher_predictor, seq_imgs, seq_prompts, config, is_student=False
            )
            # Student forward
            s_logits, s_lowres, s_embeds = _run_sequence(
                student_predictor, seq_imgs, seq_prompts, config, is_student=True
            )

            # Compute losses
            ce_loss = F.cross_entropy(
                s_logits.view(-1, H, W),
                seq_masks.view(-1, H, W),
            )
            dist_loss = compute_distillation_loss(
                s_lowres, t_lowres, s_embeds, t_embeds,
                embed_weight=config.training.get("embed_weight", 1.0),
                dist_weight=dist_weight
            )

            # Backward
            loss = ce_loss + dist_loss
            scaler.scale(loss).backward()

            # Update metrics
            total_ce   += ce_loss.item() * T
            total_dist += dist_loss.item() * T
            total_loss += loss.item()
            total_frames += T

        # Update optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Logging
        logger.info(
            f"Epoch {epoch+1} train loss: {total_loss/total_frames:.4f}, "
            f"CE: {total_ce/total_frames:.4f}, Dist: {total_dist/total_frames:.4f}"
        )

    scheduler.step()

def validate_epoch(model, video_predictor, data_loader, config, epoch, logger):
    """
    Evaluate the performance of the model on the validation dataset for one epoch.

    This function evaluates the model by computing the mean Intersection over Union (mIoU)
    across all sequences in the validation dataset. It uses a full-mask prompt for the first 
    frame and no prompts for subsequent frames. The mIoU is logged for the current epoch.

    Args:
        model (nn.Module): The model to be evaluated.
        video_predictor (SAM2VideoPredictor): The video predictor used for generating predictions.
        memory_bank (Any): A storage mechanism for temporal information (not used in this function).
        data_loader (DataLoader): DataLoader providing validation image sequences and masks.
        config (dict): Configuration object containing training parameters.
        epoch (int): The current epoch number.
        logger (Logger): Logger for recording validation progress.

    Returns:
        float: The mean Intersection over Union (mIoU) for the validation dataset.
    """
    model.eval()
    device = config.training.device
    all_ious = []

    with torch.no_grad():
        for imgs, masks in data_loader:
            B, T, C, H, W = imgs.shape
            imgs, masks = imgs.to(device), masks.to(device)

            for b in range(B):
                seq_imgs  = imgs[b]         # [T,C,H,W]
                seq_masks = masks[b]        # [T,H,W]
                # Only full-mask prompt on frame0
                prompts   = [{'mask': seq_masks[0]}] + [None]*(T-1)
                preds, _, _ = _run_sequence(
                    video_predictor, seq_imgs, prompts, config, is_student=False
                )
                # Binarize
                preds = (preds > 0).long().squeeze(1)  # [T,H,W]
                for t in range(T):
                    all_ious.append(compute_iou(preds[t], seq_masks[t]))

    mean_iou = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0
    logger.info(f"Epoch {epoch+1} validation mIoU: {mean_iou:.4f}")
    model.train()
    return mean_iou

def test_epoch(video_predictor, data_loader, config, epoch, logger):
    """
    Run one test epoch, computing IoU, Dice, and HD95 per frame.

    Args:
        video_predictor (SAM2VideoPredictor): model wrapper for inference
        data_loader (DataLoader): provides (imgs, masks) batches
        config (DictConfig): config with .training.device, etc.
        epoch (int): current epoch (for logging)
        logger (Logger): for logging metrics

    Returns:
        dict: {'iou': mean_iou, 'dice': mean_dice, 'hd95': mean_hd95}
    """
    video_predictor.model.eval()
    device = config.training.device

    all_ious, all_dices, all_hd95s  = [], [], []

    with torch.no_grad():
        for imgs, masks in data_loader:
            B, T, C, H, W = imgs.shape
            imgs, masks = imgs.to(device), masks.to(device)

            for b in range(B):
                seq_imgs  = imgs[b]       # [T,C,H,W]
                seq_masks = masks[b]      # [T,H,W]
                # Full-mask prompt on frame 0
                prompts   = [{'mask': seq_masks[0]}] + [None]*(T-1)

                # Run inference
                preds, _, _ = _run_sequence(video_predictor, seq_imgs, prompts, config, is_student=False)
                # binarize logits
                preds = (preds > 0).long().squeeze(1)  # [T,H,W]

                for t in range(T):
                    p = preds[t]
                    g = seq_masks[t]

                    all_ious.append( compute_iou(p, g) )    # :contentReference[oaicite:0]{index=0}
                    all_dices.append(compute_dice(p, g))   # :contentReference[oaicite:1]{index=1}
                    all_hd95s.append(compute_hd95(p, g))   # :contentReference[oaicite:2]{index=2}

    # Calculate mean metrics for epoch
    mean_iou  = float(sum(all_ious)  / len(all_ious))  if all_ious  else 0.0
    mean_dice = float(sum(all_dices) / len(all_dices)) if all_dices else 0.0
    mean_hd95 = float(sum(all_hd95s)/ len(all_hd95s)) if all_hd95s else 0.0

    logger.info(
        f"Epoch {epoch+1} TEST - IoU: {mean_iou:.4f}, "
        f"Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}"
    )

    video_predictor.model.train()
    return {'iou': mean_iou, 'dice': mean_dice, 'hd95': mean_hd95}
    
def main(config_path, resume, multi_gpu, amp):
    """
    Train a SAM2 model from scratch or from a checkpoint.

    This function will:

    1. Load the configuration from the given YAML file.
    2. Set up the data loaders for the training and validation datasets.
    3. Build the student and teacher models.
    4. Set up the optimizer, scheduler, and AMP (mixed precision) scaler.
    5. If multi-GPU is enabled and there is more than one GPU, use DataParallel.
    6. Load from a checkpoint if the resume argument is given.
    7. Run the training loop for the specified number of epochs.

    Args:
        config_path (str): Path to the YAML configuration file.
        resume (str): Path to the checkpoint to resume from, or None to start from scratch.
        multi_gpu (bool): Whether to use DataParallel on multiple GPUs.
        amp (bool): Whether to use mixed precision training.
    """
    # 1) load config + logging
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SAM2")

    # 2) data loaders
    train_loader, val_loader = get_data_loaders(config)

    # 3) models
    student = build_student_model(config)
    teacher = build_teacher_predictors(config)

    # 4) optimizer + scheduler + scaler
    optimizer, scheduler = setup_optimizer_and_scheduler(student.model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 5) possible multi-GPU
    if multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        student.model = torch.nn.DataParallel(student.model)
        teacher.model = torch.nn.DataParallel(teacher.model)

    # 6) checkpoints
    ckpt_mgr = CheckpointManager(config.training.output_dir)
    start_ep = 0
    if resume:
        start_ep = ckpt_mgr.load(resume, student.model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_ep}")

    # 7) training loop
    for epoch in range(start_ep, int(config.training.epochs)):
        train_epoch(student, teacher, train_loader, optimizer, scheduler, scaler, config, epoch, logger)
        val_iou = validate_epoch(student.model, student, val_loader, config, epoch, logger)
        test_epoch(student, val_loader, config, epoch, logger)

        ckpt_mgr.save_latest(student.model, optimizer, scheduler, epoch)
        ckpt_mgr.save_best(  student.model, optimizer, scheduler, epoch, val_iou)

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
