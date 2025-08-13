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
import numpy as np
import torchvision.transforms.functional as TF

from dripp.helpers import normalize_path, get_extension

class SegmentationSequenceDataset(Dataset):
    def __init__(
            self,
            sequences,
            transform,
            truncate=True,
            max_frames_per_sequence=8,
            min_fg_frames_in_window=2,
            prompt_mix=None,
            max_prompt_frames=2,
            always_prompt_first=True,
            enable_negative_clicks=True,
            num_pos_clicks=1,
            num_neg_clicks=1,
            neg_margin_frac=0.1,
            reverse_prob=0.5
    ):
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
            truncate (bool, optional): Whether to truncate sequences to max_frames_per_sequence. Default is True.
            max_frames_per_sequence (int, optional): Maximum number of frames per sequence. Default is 8.
            min_fg_frames_in_window (int, optional): Minimum number of foreground frames per window. Default is 2.
            prompt_mix (list[float], optional): List of probabilities for each prompt type. Default is None.
            max_prompt_frames (int, optional): 
            always_prompt_first (bool, optional): Whether to always prompt the first frame. Default is True.
            enable_negative_clicks (bool, optional): Whether to enable negative clicks. Default is True.
            num_pos_clicks (int, optional): Number of positive clicks per frame. Default is 1.
            num_neg_clicks (int, optional): Number of negative clicks per frame. Default is 1.
            neg_margin_frac (float, optional): Fraction of negative click margin. Default is 0.1.
            reverse_prob (float, optional): Probability of sampling reverse axis. Default is 0.5.
        """
        self.sequences = sequences
        self.transform = transform
        self.entry_dims = [entry['dim'] for entry in sequences]
        self.truncate = truncate
        
        self.max_frames_per_sequence = int(max_frames_per_sequence)
        self.min_fg_frames_in_window = int(min_fg_frames_in_window)

        self.prompt_mix = prompt_mix or {"mask": 0.5, "click": 0.25, "bbox": 0.25}

        # Normalize prompt mix
        s = sum(self.prompt_mix.values())
        self.prompt_mix = {k: v/(s + 1e-8) for k,v in self.prompt_mix.items()}

        # Prompt config
        self.max_prompt_frames      = max_prompt_frames
        self.always_prompt_first    = always_prompt_first
        self.enable_negative_clicks = enable_negative_clicks
        self.num_pos_clicks         = num_pos_clicks
        self.num_neg_clicks         = num_neg_clicks
        self.neg_margin_frac        = neg_margin_frac

        # Probability of sampling reverse
        self.reverse_prob = reverse_prob

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

        # Reset transformations
        if self.transform is not None:
            self.transform.reset()

        # Load images, masks, and axis lengths if 3D.
        if data_dim == 3:
            imgs, masks, axis_lengths = self._load_3d(normalize_path(seq[0][0]), normalize_path(seq[0][1]))
        else:
            # Load entire 2D "pseudo-video"
            cleaned = [(normalize_path(i), normalize_path(m)) for i, m in seq]
            imgs, masks = self._load_2d_sequence(cleaned)
            axis_lengths = None

        # Truncate if necessary — only use the chosen axis segment
        if axis_lengths is not None and self.truncate:
            N = int(self.max_frames_per_sequence)   # Window size
            L = imgs.shape[0] // 2                  # Length for each direction
            if L < N:
                raise ValueError(f"Axis too short: need {N}, have {L}")

            # Randomly sample direction
            use_reverse = (random.random() < self.reverse_prob)
            s0 = L if use_reverse else 0

            # Check if there is any foreground
            seg_masks = masks[s0:s0 + L]
            fg = seg_masks.reshape(L, -1).any(dim=1)

            # If no foreground, try other direction
            if int(fg.sum().item()) == 0:
                # Try reverse direction once
                s0 = 0 if use_reverse else L
                seg_masks = masks[s0:s0 + L]
                fg = seg_masks.reshape(L, -1).any(dim=1)

            if int(fg.sum().item()) == 0:
                raise ValueError(f'No FG in either direction for idx={idx}')

            # Randomly sample window
            min_fg = int(self.min_fg_frames_in_window)
            cs = torch.cat([torch.tensor([0], device=fg.device), torch.cumsum(fg.int(), dim=0)])
            win_sums = cs[N:] - cs[:-N]
            valid = torch.nonzero(win_sums >= min_fg).squeeze(1)
            if valid.numel() == 0:
                valid = torch.nonzero(win_sums > 0).squeeze(1)
            i0  = 0 if valid.numel() == 0 else int(valid[torch.randint(0, valid.numel(), (1,))].item())
            sel = list(range(s0 + i0, s0 + i0 + N))
            imgs, masks = imgs[sel], masks[sel]

        # Generate prompts per slice (return empty tensors if no prompt)
        pt_list, label_list, bbox_list, m_prompt_list = [], [], [], []
        
        # Get mask shape
        T = masks.shape[0]
        H = masks.shape[-2]
        W = masks.shape[-1]

        # Select which frames get prompts
        prompt_frames = set()
        if self.always_prompt_first and T > 0:
            prompt_frames.add(0)
        remaining_slots = max(0, self.max_prompt_frames - len(prompt_frames))
        if remaining_slots > 0 and T > 1:
            pool = list(range(1, T))
            pick = random.sample(pool, k=min(remaining_slots, len(pool)))
            prompt_frames.update(pick)

        # Prompt mix
        keys  = list(self.prompt_mix.keys())
        probs = [self.prompt_mix[k] for k in keys]

        # Select exactly one prompt type per frame
        def sample_types_for_frame(): return {random.choices(keys, weights=probs, k=1)[0]}
        
        # Generate prompts
        for t in range(T):
            mask_slice = masks[t]
            # If mask has an extra channel dimension, squeeze it out
            if mask_slice.ndim == 3 and mask_slice.size(0) == 1:
                mask_slice = mask_slice.squeeze(0) # now [H,W]
            
            if t in prompt_frames:
                want_types = sample_types_for_frame()

                # Accumulators for this frame (start empty)
                pts_t   = torch.empty((0, 2),   dtype=torch.int64)
                lbls_t  = torch.empty((0,),     dtype=torch.int64)
                bbox_t  = torch.empty((0, 4),   dtype=torch.int64)
                mask_t  = torch.empty((0, H, W),dtype=torch.uint8)

                # 1) Clicks (pos + optional negatives)
                if "click" in want_types:
                    p_click = generate_prompt(
                        mask_slice,
                        prompt_type="click",
                        num_pos=self.num_pos_clicks,
                        num_neg=(self.num_neg_clicks if self.enable_negative_clicks else 0),
                        neg_margin_frac=self.neg_margin_frac
                    )
                    if 'points' in p_click:
                        pts_t  = p_click['points']
                        lbls_t = p_click['labels']

                # 2) Bounding box
                if "bbox" in want_types:
                    p_box = generate_prompt(mask_slice, prompt_type="bbox")
                    if 'bbox' in p_box:
                        bbox_t = p_box['bbox'].to(torch.int64).view(1,4)

                # 3) Mask
                if "mask" in want_types:
                    p_mask = generate_prompt(mask_slice, prompt_type="mask")
                    if 'mask' in p_mask:
                        m_bin = p_mask['mask'].to(torch.uint8)        # [H,W]
                        mask_t = m_bin.unsqueeze(0)                   # [1,H,W]

                # Push per-type (leave empty tensors if that type wasn't selected / invalid)
                pt_list.append(pts_t)
                label_list.append(lbls_t)
                bbox_list.append(bbox_t if bbox_t.numel() > 0 else torch.empty((0,4), dtype=torch.int64))
                m_prompt_list.append(mask_t if mask_t.numel() > 0 else torch.empty((0,H,W), dtype=torch.uint8))
            else:
                pt_list.append(torch.empty((0, 2),   dtype=torch.int64))
                label_list.append(torch.empty((0,),  dtype=torch.int64))
                bbox_list.append(torch.empty((0, 4), dtype=torch.int64))
                m_prompt_list.append(torch.empty((0, H, W), dtype=torch.uint8))

        # Only unsqueeze if needed
        mask_seq = masks
        if mask_seq.ndim == 3:              # [T,H,W]
            mask_seq = mask_seq.unsqueeze(1)

        # Convert only if axis_lengths is not None and we are not truncating. Otherwise, return None.
        axis_lengths_i = tuple(map(int, axis_lengths)) if axis_lengths and not self.truncate else None

        # Return
        output = {
            'image':        imgs,               # Tensor[T,C,H,W]
            'mask':         mask_seq,           # Tensor[T,1,H,W]
            'pt_list':      pt_list,            # List[T] of Tensor[n,2]
            'p_label':      label_list,         # List[T] of Tensor[n]
            'bbox':         bbox_list,          # List[T] of Tensor[4]
            'm_prompt':     m_prompt_list,      # List[T] of Tensor[1,H,W]
            'dataset':      ds_name,            # Dataset name
            'subdataset':   sub_name,           # Subdataset name
            'seq_idx':      idx,                # Index of the sequence (for debugging)
            'dim':          data_dim,           # 2 or 3
        }

        # Keep axis_lengths only when we did NOT truncate and lengths match the 6-orientation build
        if (axis_lengths is not None) and (not self.truncate) and (imgs.shape[0] == 2 * sum(axis_lengths)):
            output['axis_lengths'] = axis_lengths_i
        return output

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
            tuple: (seq_imgs, seq_masks, axis_lengths)
                seq_imgs (torch.Tensor): A sequence of 2D images (Tensor[T,C,H,W]).
                seq_masks (torch.Tensor): A sequence of 2D masks (Tensor[T,1,H,W]).
                axis_lengths (tuple): A tuple containing the lengths of the 3D volume along each axis (D, H, W).
        """
        itk_img = sitk.ReadImage(img_path)          # Read 3D image
        itk_msk = sitk.ReadImage(mask_path)         # Read 3D mask
        img_vol = sitk.GetArrayFromImage(itk_img)   # [D, H, W]
        msk_vol = sitk.GetArrayFromImage(itk_msk)
        _, H0, W0 = self._to_tensor(img_vol[0], msk_vol[0])[0].shape
        
        # Select the axis with the most foreground
        msk_np = np.array(msk_vol)
        min_fg = self.min_fg_frames_in_window
        chosen_axis = self._choose_axis(msk_np, min_fg) or self._choose_axis(msk_np, 1)

        # Build forward and reverse indices
        if not self.truncate:
            # Trim to the first and last slices with foreground for validation and testing
            has_fg = self._get_fg_slices(msk_np, chosen_axis).astype(bool)
            if has_fg.any():
                fg = np.flatnonzero(has_fg)
                start_k, end_k = fg[0], fg[-1] + 1
            else:
                start_k, end_k = 0, msk_np.shape[chosen_axis]
            idxs_fwd = list(range(start_k, end_k))
            idxs_rev = list(reversed(idxs_fwd))
        else:
            # Otherwise, build full sequence
            idxs_fwd = list(range(msk_np.shape[chosen_axis]))
            idxs_rev = list(reversed(idxs_fwd))

        # Build forward + reverse sequence of 2D images and masks on the chosen axis
        seq_imgs, seq_masks = [], []
        for i in idxs_fwd:
            slice_img = self._slice(img_vol, chosen_axis, i)
            slice_msk = self._slice(msk_vol, chosen_axis, i)
            im_t, ms_t = self._to_tensor(slice_img, slice_msk)
            if self.transform:
                im_t, ms_t = self.transform(im_t, ms_t)
                # Resize to (H0, W0)
                if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                    im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                    ms_t = TF.resize(ms_t.unsqueeze(0).float(), [H0, W0], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
            seq_imgs.append(im_t)
            seq_masks.append(ms_t)

        for i in idxs_rev:
            slice_img = self._slice(img_vol, chosen_axis, i)
            slice_msk = self._slice(msk_vol, chosen_axis, i)
            im_t, ms_t = self._to_tensor(slice_img, slice_msk)
            if self.transform:
                im_t, ms_t = self.transform(im_t, ms_t)
                # Resize to (H0, W0)
                if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                    im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                    ms_t = TF.resize(ms_t.unsqueeze(0).float(), [H0, W0], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
            seq_imgs.append(im_t)
            seq_masks.append(ms_t)

        D, H, W = img_vol.shape
        return torch.stack(seq_imgs, dim=0), torch.stack(seq_masks, dim=0), (D, H, W)
    
    def _get_fg_slices(self, msk_np, axis):
        """
        Given a 3D mask and an axis, return a boolean array indicating which slices
        have any foreground pixels.

        Args:
            msk_np (numpy.ndarray): The 3D mask.
            axis (int): The axis along which to take slices.

        Returns:
            numpy.ndarray: A boolean array indicating which slices have any foreground pixels.
        """
        if axis == 0: return np.any(msk_np, axis=(1,2)) # Axial Z
        if axis == 1: return np.any(msk_np, axis=(0,2)) # Coronal Y
        return np.any(msk_np, axis=(0,1))               # Sagittal X

    def _choose_axis(self, msk_np, min_fg=1):
        """
        Given a 3D mask, choose a viewing axis based on a score computed from:

        - in-plane resolution proxy (higher is better)
        - fraction of slices with any foreground (higher is better)
        - mean foreground area (higher is better)

        The score is computed as the product of these three terms, each raised to a power determined
        by the corresponding hyperparameter. The axis with the highest score is chosen.

        If no axis has any foreground, returns None.

        Hyperparameters:

        - max_frames_per_sequence (int, default=8): maximum number of slices to take in a sequence
        - axis_min_prob (float, default=0.20): minimum probability assigned to each axis
        - axis_res_alpha (float, default=1.0): exponent for in-plane resolution proxy
        - axis_fgfrac_beta (float, default=1.0): exponent for fraction of slices with any foreground
        - axis_area_gamma (float, default=0.5): exponent for mean foreground area

        Args:
            msk_np (numpy.ndarray): The 3D mask.
            min_fg (int, optional): minimum number of foreground slices required in a window of size
                max_frames_per_sequence. Defaults to 1.

        Returns:
            int or None: The chosen axis, or None if no axis has any foreground.
        """
        assert msk_np.ndim == 3, "msk_np must be [D,H,W]"
        D, H, W = msk_np.shape

        # Hyperparameters
        N = self.max_frames_per_sequence
        p_min = float(getattr(self, "axis_min_prob",    0.2))       # Per-axis floor among candidates
        alpha = float(getattr(self, "axis_res_alpha",   1.0))       # Weight for in-plane resolution proxy
        beta  = float(getattr(self, "axis_fgfrac_beta", 1.0))       # Weight for fraction of FG slices
        gamma = float(getattr(self, "axis_area_gamma",  0.5))       # Weight for mean FG area
        eps = 1e-8

        # In-plane pixel counts per viewing axis: higher means finer in-plane grid
        inplane_area = np.array([H * W, D * W, D * H], dtype=np.float64)  # axes 0,1,2

        candidates = []
        stats = []  # Stores per-axis measurements

        for axis in (0, 1, 2):
            other = tuple(i for i in (0, 1, 2) if i != axis)
            per_slice_area = msk_np.sum(axis=other)                 # [L], number of FG pixels per slice
            pos = per_slice_area > 0                                # [L], slice has any FG
            L = per_slice_area.shape[0]
            if L == 0:
                continue

            # Window constraint: ensure there exists a window of size N with at least min_fg FG slices
            N_eff = min(N, L)
            ok = False
            if min_fg <= 0:
                ok = pos.any()
            else:
                x = pos.astype(np.int32)
                if N_eff <= L:
                    cs = np.concatenate([[0], np.cumsum(x)])
                    win = cs[N_eff:] - cs[:-N_eff]                  # Rolling sum in windows of size N_eff
                    ok = (win >= min_fg).any()
                else:
                    ok = x.sum() >= min_fg

            if not ok:
                continue

            fg_frac = pos.mean()                                    # Fraction of slices with any FG
            if pos.any():
                mean_area_norm = (per_slice_area[pos] / inplane_area[axis]).mean()
            else:
                mean_area_norm = 0.0

            candidates.append(axis)
            stats.append({
                "axis": axis,
                "res_proxy": inplane_area[axis],                    # Proxy for in-plane resolution
                "fg_frac": float(fg_frac),
                "mean_area": float(mean_area_norm),
            })

        if not candidates:
            return None

        # Min-max normalize each feature across candidates
        def norm(v):
            v = np.asarray(v, dtype=np.float64)
            vmin, vmax = v.min(), v.max()
            return np.ones_like(v) if abs(vmax - vmin) < eps else (v - vmin) / (vmax - vmin)

        res_n   = norm([s["res_proxy"] for s in stats])
        frac_n  = norm([s["fg_frac"] for s in stats])
        area_n  = norm([s["mean_area"] for s in stats])

        # Quality score and probabilities
        quality = (res_n ** alpha) * ((eps + frac_n) ** beta) * ((eps + area_n) ** gamma)

        # Apply per-axis probability floor among the current candidates
        k = len(candidates)

        # Base distribution from quality
        s = float(quality.sum())
        if not np.isfinite(s) or s <= 0:
            base = np.ones(k, dtype=np.float64) / k                 # Uniform if all qualities are 0
        else:
            base = quality / s                                      # Sums to 1 exactly

        # Ensure feasibility: p_min cannot exceed 1/k
        p_min_eff = min(p_min, 1.0 / k - 1e-6)
        probs = base * (1.0 - p_min_eff * k) + p_min_eff

        # Sample an axis according to probs
        idx = np.random.choice(np.arange(k), p=probs)
        return candidates[idx]


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
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
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
        had_fg = (mask > 0).any()  # True if there is at least one foreground pixel
        
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
        lost_fg = not (mask > 0).any()
        if had_fg and lost_fg:
            return orig_image, orig_mask
        return image, mask

def generate_prompt(mask_2d, prompt_type='click', num_pos=1, num_neg=0, neg_margin_frac=0.1):
    """
    Create a prompt from a binary mask.

    Args:
        mask_2d (Tensor[H,W] or np.ndarray): binary mask
        prompt_type (str): "click", "bbox", or "mask"
        num_pos (int): number of positive clicks (for "click")
        num_neg (int): number of negative clicks (for "click")
        neg_margin_frac (float): margin to expand bbox when sampling negatives

    Returns:
        dict with one of:
          - {"points": Tensor[n,2], "labels": Tensor[n]}   # n = num_pos + num_neg
          - {"bbox":   Tensor[4]}                          # [x0,y0,x1,y1]
          - {"mask":   Tensor[H,W]}                        # binary mask
        If mask is empty and prompt cannot be formed, returns {}.
    """
    if isinstance(mask_2d, torch.Tensor):
        m = mask_2d.detach().cpu()
    else:
        m = torch.from_numpy(mask_2d)

    m = (m > 0).to(torch.uint8)
    H, W = int(m.shape[-2]), int(m.shape[-1])

    ys, xs = torch.where(m > 0)
    if prompt_type == "mask":
        if ys.numel() == 0:
            return {}
        return {"mask": m}
    
    if prompt_type == "bbox":
        if ys.numel() == 0:
            return {}
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        return {"bbox": torch.tensor([x0, y0, x1, y1], dtype=torch.int64)}
    
    if prompt_type == "click":
        points, labels = [], []

        # Positives
        if ys.numel() > 0 and num_pos > 0:
            idx = torch.randperm(ys.numel())[:num_pos]
            pos_xy = torch.stack([xs[idx], ys[idx]], dim=1).to(torch.int64)
            points.append(pos_xy)
            labels.append(torch.ones(pos_xy.size(0), dtype=torch.int64))
        elif num_pos > 0:
            # No positives found, return empty prompt
            return {}
        
        # Negatives near bbox
        if num_neg > 0:
            if ys.numel() > 0:
                # Bounding box around positives
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())

                # Expand bounding box by margin
                dy = max(1, int((y1 - y0 + 1) * neg_margin_frac))
                dx = max(1, int((x1 - x0 + 1) * neg_margin_frac))
                y0n = max(0, y0 - dy)
                y1n = min(H - 1, y1 + dy)
                x0n = max(0, x0 - dx)
                x1n = min(W - 1, x1 + dx)

                # Select candidate negatives inside expanded window but outside mask
                YY, XX = torch.meshgrid(torch.arange(y0n, y1n + 1), torch.arange(x0n, x1n + 1), indexing='ij')
                cand = (m[YY, XX] == 0)
                cy, cx = YY[cand], XX[cand]
                if cy.numel() == 0:
                    cy, cx = torch.where(m == 0) # Fallback anywhere off-mask
            else:
                cy, cx = torch.where(m == 0)

            if cy.numel() > 0:
                idx = torch.randperm(cy.numel())[:num_neg]
                neg_xy = torch.stack([cx[idx], cy[idx]], dim=1).to(torch.int64)
                points.append(neg_xy)
                labels.append(torch.zeros(neg_xy.size(0), dtype=torch.int64))

        if points:
            pts  = torch.cat(points, dim=0)
            lbls = torch.cat(labels, dim=0)
            return {"points": pts, "labels": lbls}

        # No positives or negatives found, return empty prompt
        return {}