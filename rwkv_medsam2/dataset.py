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
    def __init__(self, sequences, transform, max_frames_per_sequence=8):
        """
        Initializes a SegmentationSequenceDataset.

        Args:
            sequences (list[dict]): List of dictionaries containing sequence metadata.
                Each dictionary should contain the following keys:
                    - sequence (list[tuple]): List of tuples containing image and mask paths.
                    - dataset (str): Name of the dataset.
                    - subdataset (str, optional): Name of the subdataset, if applicable.
                    - dim (int): Dimension of the image (2 or 3).
            transform (callable): An optional transformation to be applied to every image and mask pair.
            max_frames_per_sequence (int, optional): Maximum number of frames per sequence. Default is 8.
        """
        self.sequences = sequences
        self.transform = transform
        self.entry_dims = [entry['dim'] for entry in sequences]
        self.prompt_types = ['bbox' if dim == 3 else 'click' for dim in self.entry_dims]
        self.max_frames_per_sequence = max_frames_per_sequence

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Gets a sequence of 2D/3D images and masks as tensors.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the following:
                - image (Tensor[T,C,H,W]): The sequence of images.
                - mask (Tensor[T,1,H,W]): The sequence of masks.
                - pt_list (List[T] of Tensor[n,2]): List of point prompts per frame.
                - p_label (List[T] of Tensor[n]): List of point labels per frame.
                - bbox (List[T] of Tensor[4]): List of bounding box prompts per frame.
                - dataset (str): Name of the dataset.
                - subdataset (str): Name of the subdataset, if applicable.
        """
        # Get entry and associated data
        entry       = self.sequences[idx]
        ds_name     = entry['dataset']
        sub_name    = entry.get('subdataset', '')
        seq         = entry['sequence']             # List[(img_path, mask_path)]
        data_dim    = self.entry_dims[idx]          # entry['dim']
        prompt_type = self.prompt_types[idx]

        # Reset transformations
        if self.transform is not None:
            self.transform.reset()

        # Normalize and detect extension
        img0 = normalize_path(seq[0][0])
        ext = get_extension(img0)

        if data_dim == 3:
            imgs, masks = self._load_3d(normalize_path(seq[0][0]), normalize_path(seq[0][1]))
        else:
            # Load entire 2D "pseudo-video"
            cleaned = [(normalize_path(i), normalize_path(m)) for i, m in seq]
            imgs, masks = self._load_2d_sequence(cleaned)

        # Limit sequence length if too long
        T_full = imgs.shape[0]
        if T_full > self.max_frames_per_sequence:
            # Randomly choose max_frames_per_sequence frames
            idxs = random.sample(range(T_full), self.max_frames_per_sequence)
            idxs.sort()
            imgs  = imgs[idxs]
            masks = masks[idxs]

        # Generate prompts per slice (return empty tensors if no prompt)
        pt_list, label_list, bbox_list = [], [], []
        T = masks.shape[0]
        for t in range(T):
            mask_slice = masks[t]
            # If has an extra channel dimension, squeeze it out
            if mask_slice.ndim == 3 and mask_slice.size(0) == 1:
              mask_slice = mask_slice.squeeze(0) # now [H,W]
            prompt = generate_prompt(mask_slice, prompt_type=prompt_type)
            if 'points' in prompt:
                pt_list.append(prompt['points'])     # Tensor[n,2]
                label_list.append(prompt['labels'])  # Tensor[n]
                # Empty bbox tensor so default_collate can stack
                bbox_list.append(torch.empty((0, 4), dtype=torch.int64))
            else:
                # Empty point tensor so default_collate can stack
                pt_list.append(torch.empty((0, 2), dtype=torch.int64))
                label_list.append(torch.empty((0,),   dtype=torch.int64))
                bbox_list.append(prompt['bbox'])       # Tensor[4]

        # Only unsqueeze if needed
        mask_seq = masks
        if mask_seq.ndim == 3:              # [T,H,W]
            mask_seq = mask_seq.unsqueeze(1)

        return {
            'image':      imgs,                 # Tensor[T,C,H,W]
            'mask':       mask_seq,             # Tensor[T,1,H,W]
            'pt_list':    pt_list,              # List[T] of Tensor[n,2]
            'p_label':    label_list,           # List[T] of Tensor[n]
            'bbox':       bbox_list,            # List[T] of Tensor[4]
            'dataset':    ds_name,              # Dataset name
            'subdataset': sub_name,             # Subdataset name
            'seq_idx':    idx                   # Index of the sequence (for debugging)
        }

    def _load_2d_sequence(self, seq):
        """
        Loads a sequence of 2D images and their corresponding masks as a sequence of tensors.

        Args:
            seq (list[tuple]): A list of tuples, where each tuple contains a path to an image and
                a path to its corresponding mask.

        Returns:
            tuple: A tuple containing two tensors. The first tensor is a sequence of images (Tensor[T,C,H,W]),
                and the second tensor is a sequence of masks (Tensor[T,1,H,W]).
        """
        imgs, masks = [], []
        for imp_p, msk_p in seq:
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
        _, H0, W0 = self._to_tensor(img_vol[0], msk_vol[0])[0].shape

        seq_imgs, seq_masks = [], []
        # For each axis, gather slices forward and reverse
        for axis in (0,1,2):
            length = msk_vol.shape[axis]
            for i in range(length):
                slice_img = self._slice(img_vol, axis, i)
                slice_msk = self._slice(msk_vol, axis, i)
                im_t, ms_t = self._to_tensor(slice_img, slice_msk)
                if self.transform:
                    im_t, ms_t = self.transform(im_t, ms_t)
                    # Resize to (H0, W0)
                    if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                        im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                        ms_t = TF.resize(ms_t.unsqueeze(0).float(), [H0, W0],
                                         interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
                seq_imgs.append(im_t)
                seq_masks.append(ms_t)
            for i in reversed(range(length)):
                slice_img = self._slice(img_vol, axis, i)
                slice_msk = self._slice(msk_vol, axis, i)
                im_t, ms_t = self._to_tensor(slice_img, slice_msk)
                if self.transform:
                    im_t, ms_t = self.transform(im_t, ms_t)
                    # Resize to (H0, W0)
                    if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                        im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                        ms_t = TF.resize(ms_t.unsqueeze(0).float(), [H0, W0],
                                         interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
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
    def __init__(self, sequences, tasks_map, seed=None):
        """
        Sampler that balances both tasks and modalities (2D vs 3D).

        Args:
            sequences (list[dict]): List of sequence dicts, each with keys
                'dataset', 'subdataset', 'tasks', 'dim', etc.
            tasks_map (dict): Mapping from task IDs to configs, each containing:
                - 'classes': List of classes for that task
                - 'datasets': { dataset_name: [subdataset_names...] }
            seed (int, optional): Random seed for reproducibility.
        """
        self.sequences = sequences
        self.tasks_map = tasks_map
        self.num_samples = len(sequences)

        # Build buckets keyed by (task_id, modality)
        # modality = dim (2 or 3)
        self.buckets = {}  # (task_id, dim) -> list of indices
        for idx, seq in enumerate(sequences):
            ds = seq['dataset']
            sd = seq.get('subdataset')
            dim = seq.get('dim')
            for task_id in seq.get('tasks', []):
                info = tasks_map.get(task_id, {})
                # check dataset/subdataset eligibility
                if ds in info.get('datasets', {}) and sd in info['datasets'].get(ds, []):
                    key = (task_id, dim)
                    self.buckets.setdefault(key, []).append(idx)

        # Keep only non-empty buckets
        self.keys = [k for k, v in self.buckets.items() if v]
        if seed is not None:
            random.seed(seed)

    def __iter__(self):
        """
        Yields indices by:
          1. Sampling a random (task, modality) bucket
          2. Sampling a random sequence index within that bucket
        """
        for _ in range(self.num_samples):
            key = random.choice(self.keys)          # (task_id, dim)
            yield random.choice(self.buckets[key])

    def __len__(self):
        """Total number of samples per epoch."""
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
        # Ensure mask has a channel dim so torchvision transforms work
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [1,H,W]

        # Keep originals around in case we need to roll back
        orig_image, orig_mask = image.clone(), mask.clone()
        
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

        # Safety check: If this transform zeroed out the mask, revert
        if (mask > 0).sum() == 0:
            return orig_image, orig_mask
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