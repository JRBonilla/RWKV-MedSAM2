# dataset.py
# Defines core data handling utilities for RWKV-MedSAM2:
#  - SegmentationSequenceDataset - Loads and preprocesses 2D/3D image-mask sequences
#  - BalancedTaskSampler         - Samples examples to balance tasks during training
#  - SequenceTransform           - Applies consistent, per-sequence augmentations
#  - generate_prompt             - Creates click or bbox prompts from binary masks
import random

import torch
from torch.utils.data import Dataset, Sampler

import SimpleITK as sitk
import torchvision.transforms.functional as TF

from dripp.helpers import normalize_path, get_extension

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
        self.data_dimension = self.get_data_dimension()
        self.prompt_type = 'bbox' if self.data_dimension == 3 else 'click'

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
        if self.data_dimension == 3:
            imgs, masks = self._load_3d(img0, normalize_path(pairs[0][1]))
        else:
            imgs, masks = self._load_2d_sequence([(normalize_path(i), normalize_path(m)) for i, m in pairs])

        # Generate prompts per slice
        pt_list, label_list, bbox_list = [], [], []
        T = masks.shape[0]
        for t in range(T):
            prompt = generate_prompt(masks[t], prompt_type=self.prompt_type)
            if 'points' in prompt:
                pt_list.append(prompt['points'])    # Tensor[n,2]
                label_list.append(prompt['labels']) # Tensor[n]
                bbox_list.append(None)
            else:
                pt_list.append(None)
                label_list.append(None)
                bbox_list.append(prompt['bbox'])    # Tensor[4]

        return {
            'image':   imgs,               # Tensor[T,C,H,W]
            'mask':    masks.unsqueeze(1), # Tensor[T,1,H,W]
            'pt_list': pt_list,            # List[T] of Tensor[n,2] or None
            'p_label': label_list,         # List[T] of Tensor[n] or None
            'bbox':    bbox_list           # List[T] of Tensor[4] or None
        }

    def get_data_dimension(self):
        img0 = normalize_path(self.pairings[0]['pairs'][0][0])
        ext = get_extension(img0)
        return 3 if len(self.pairings[0]['pairs']) == 1 and ext in {'.nii', '.nii.gz'} else 2

    def _load_2d_sequence(self, pairs):
        imgs, masks = [], []
        for imp_p, msk_p in pairs:
            i_arr = sitk.GetArrayFromImage(sitk.ReadImage(imp_p))
            m_arr = sitk.GetArrayFromImage(sitk.ReadImage(msk_p))
            im_t, ms_t = self._to_tensor(i_arr, m_arr)
            print(f"Image shape: {im_t.shape}, Mask shape: {ms_t.shape}")
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
            task_classes = set(info['classes'])
            valid_idxs = []
            for idx, pair in enumerate(self.pairings):
                ds, sd = pair['dataset'], pair.get('subdataset')
                if ds in info['datasets'] and sd in info['datasets'][ds]:
                    if set(pair.get('mask_classes', {}).keys()) & task_classes:
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

def generate_prompt(mask_tensor, prompt_type='click'):
    """
    Generate a prompt from a given binary mask tensor.

    Prompt types:
        - 'click': Sample one random foreground pixel and return it as a single
        point with label 1. If the mask is empty, return an empty tensor with
        shape (0, 2) and an empty tensor with shape (0,).
        - 'bbox': Return the bounding box of the foreground region. If the mask
        is empty, return a bounding box covering the full image.

    Args:
        mask_tensor (torch.Tensor): Binary mask tensor with shape (H, W).

    Returns:
        dict: A dictionary containing the prompt. The structure of the
        dictionary depends on the prompt type.

    Raises:
        ValueError: If the prompt type is unknown.
    """
    # Ensure binary mask
    fg = (mask_tensor > 0).nonzero(as_tuple=False)

    # Generate prompt based on prompt type
    if prompt_type == 'click':
        if fg.numel() == 0:
            # Fallback: no clicks
            return {
                'points': torch.empty((0, 2), dtype=torch.int64),
                'labels': torch.empty((0,), dtype=torch.int64)
            }
        # Sample one random foreground pixel
        idx = random.randint(0, fg.size(0) - 1)
        y, x = fg[idx].tolist()
        points = torch.tensor([[x, y]], dtype=torch.int64)
        labels = torch.tensor([1], dtype=torch.int64)
        return { 'points': points, 'labels': labels }
    elif prompt_type == 'bbox':
        if fg.numel() == 0:
            # Fallback: full coverage
            H, W = mask_tensor.shape
            return {'bbox': torch.tensor([0, 0, W-1, H-1], dtype=torch.int64)}
        ys, xs = fg.unbind(1)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        return {'bbox': torch.tensor([x1, y1, x2, y2], dtype=torch.int64)}
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')