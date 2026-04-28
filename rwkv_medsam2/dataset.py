# Dataset and sampling utilities for RWKV-MedSAM2.
#
# Provides sequence loading for 2D images and 3D volumes, prompt generation,
# per-sequence augmentation, and balanced task/dimension batch sampling.
import random
import math
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, BatchSampler

import SimpleITK as sitk
import numpy as np
import torchvision.transforms.functional as TF

from dripp.helpers import normalize_path, get_extension

class SegmentationSequenceDataset(Dataset):
    """
    Load preformatted 2D and 3D segmentation sequences for SAM2-style training.

    Args:
        Dataset (type): PyTorch dataset base class.

    Returns:
        None.
    """

    def __init__(
            self,
            sequences,
            transform,
            truncate=True,
            max_frames_per_sequence=8,
            min_fg_frames_in_window=2,
            prompt_mix=None,
            max_prompt_frames=4,
            always_prompt_first=True,
            enable_negative_clicks=False,
            num_pos_clicks=1,
            num_neg_clicks=0,
            neg_margin_frac=0.1,
            reverse_prob=0.5,
            fg_min_pixels_frac=0.0002,
            window_select="weighted",
            window_area_power=1.0,
            prompt_bias_area=True,
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
            fg_min_pixels_frac (float, optional): Fraction of minimum foreground pixels. Default is 0.0002.
            window_select (str, optional): Window selection strategy ("weighted", "argmax", "uniform"). Default is "weighted".
            window_area_power (float, optional): Raise window area to this power before softmax. Default is 1.0.
            prompt_bias_area (bool, optional): Whether to bias extra prompt towards areas with more foreground. Default is True.

        Returns:
            None.
        """
        self.sequences = sequences
        self.transform = transform
        self.entry_dims = [entry['dim'] for entry in sequences]
        self.truncate = truncate

        self.max_frames_per_sequence = int(max_frames_per_sequence)
        self.min_fg_frames_in_window = int(min_fg_frames_in_window)

        self.prompt_mix = prompt_mix or {"mask": 0.5, "click": 0.0, "bbox": 0.25}

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

        # Minimum foreground pixels
        self.fg_min_pixels_frac = fg_min_pixels_frac

        # Window selection
        self.window_select = window_select
        self.window_area_power = window_area_power
        self.prompt_bias_area = prompt_bias_area

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Args:
            None.

        Returns:
            int: Length value.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Gets a sequence of 2D/3D images and masks as tensors.

        In addition to the training prompts (`pt_list`, `p_label`, `bbox`, `m_prompt`),
        this also prepares a validation-only prompt cache (`val_prompt_cache`) that uses
        the same probability-driven prompt-frame selection policy as validation.

        Args:
            idx (int | Tuple[int, str]): Index of the sequence to retrieve. If a tuple is
                provided, it is expected to be (index, task_id) coming from a balanced sampler.

        Returns:
            dict[str, Any]
        """
        # Unwrap batch-sampler-provided (index, task_id) if present
        chosen_task = None
        if isinstance(idx, (tuple, list)) and len(idx) == 2:
            idx, chosen_task = idx

        # Get entry and associated data
        entry       = self.sequences[idx]
        ds_name     = entry['dataset']
        sub_name    = entry.get('subdataset', '')
        seq         = entry['sequence']
        data_dim    = self.entry_dims[idx]
        modality    = entry.get('modality', entry.get('subdataset_modality', 'unknown'))

        # Get task info from entry (will be overridden by chosen_task if provided)
        task_id     = entry.get('task_id', entry.get('task', None))
        task_label  = entry.get('task_label', entry.get('task_name', None))

        # Reset transformations
        if self.transform is not None:
            self.transform.reset()

        # Load images, masks, and axis lengths if 3D
        if data_dim == 3:
            imgs, masks, axis_lengths = self._load_3d(
                normalize_path(seq[0][0]), normalize_path(seq[0][1])
            )
        else:
            cleaned = [(normalize_path(i), normalize_path(m)) for i, m in seq]
            imgs, masks = self._load_2d_sequence(cleaned)
            axis_lengths = None

        # Only 3D volumes are windowed. 2D samples stay as length-1 sequences.
        if data_dim == 3:
            imgs, masks = self.select_window(imgs, masks, axis_lengths, idx)

        # Training prompts: keep existing mixed-prompt behavior for train-time consumption
        pt_list, label_list, bbox_list, m_prompt_list = self.generate_prompts(
            masks,
            data_dim,
            strategy="train",
        )

        # Validation prompts: probability-driven frame selection, materialized for all modes
        val_prompt_cache = self._generate_validation_prompt_cache(masks, data_dim=data_dim)

        # Only unsqueeze if needed
        mask_seq = masks
        if mask_seq.ndim == 3:  # [T,H,W]
            mask_seq = mask_seq.unsqueeze(1)

        # Convert only if axis_lengths is not None and we are not truncating
        axis_lengths_i = tuple(map(int, axis_lengths)) if axis_lengths and not self.truncate else None

        # Build output
        output = {
            'image':            imgs,
            'mask':             mask_seq,
            'pt_list':          pt_list,
            'p_label':          label_list,
            'bbox':             bbox_list,
            'm_prompt':         m_prompt_list,
            'val_prompt_cache': val_prompt_cache,
            'dataset':          ds_name,
            'subdataset':       sub_name,
            'seq_idx':          idx,
            'dim':              data_dim,
            'modality':         modality,
        }

        # Keep axis_lengths only when we did not truncate and lengths match the 6-orientation build
        if (axis_lengths is not None) and (not self.truncate) and (imgs.shape[0] == 2 * sum(axis_lengths)):
            output['axis_lengths'] = axis_lengths_i

        # Prefer the sampler's chosen task if provided; otherwise fall back to entry values
        if chosen_task is not None:
            output['task_id'] = str(chosen_task)
            output['task_label'] = str(chosen_task)
        else:
            if task_id is not None:
                output['task_id'] = int(task_id) if isinstance(task_id, int) else task_id
            if task_label is not None:
                output['task_label'] = str(task_label)

        return output

    def select_window(self, imgs, masks, axis_lengths, idx):
        """
        Select a window of N frames from the given sequence of length L.
        The window is chosen so that it contains at least min_fg_frames_in_window
        frames with foreground (FG) in the mask. If no window can be found with
        the desired amount of FG, the function falls back to windows with any
        non-zero area of FG, and as a last resort, any non-empty window.

        Args:
            imgs (Tensor[L,C,H,W]): Image sequence.
            masks (Tensor[L,1,H,W]): Mask sequence.
            axis_lengths (tuple): Lengths of the 3D volume along each axis.
            idx (int): Index of the sequence (for debugging).

        Returns:
            tuple: (imgs, masks)
                imgs (Tensor[N,C,H,W]): Selected window of images.
                masks (Tensor[N,1,H,W]): Selected window of masks.
        """
        if (axis_lengths is None) or (not self.truncate):
            return imgs, masks

        N = int(self.max_frames_per_sequence)       # window size
        L = imgs.shape[0] // 2                      # length for each direction
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
            s0 = 0 if use_reverse else L
            seg_masks = masks[s0:s0 + L]
            fg = seg_masks.reshape(L, -1).any(dim=1)

        if int(fg.sum().item()) == 0:
            raise ValueError(f'No FG in either direction for idx={idx}')

        desired_min_fg = int(self.min_fg_frames_in_window)
        H_, W_ = int(seg_masks.shape[-2]), int(seg_masks.shape[-1])

        # Per-slice FG area and robust FG indicator (avoid 1–2 px slivers)
        area = seg_masks.reshape(L, -1).sum(dim=1).to(torch.float32)         # [L]
        fg_min_frac = float(getattr(self, "fg_min_pixels_frac", 0.0002))      # ~0.2% of pixels
        min_pix     = max(2, int(fg_min_frac * H_ * W_))
        fg_big      = (area >= min_pix)                                      # "real" FG
        fg_any      = (area > 0)                                             # fallback
        fg_for_count = fg_big if int(fg_big.sum()) > 0 else fg_any

        # Rolling counts of FG slices per N-window
        cs_cnt   = torch.cat(
            [torch.tensor([0], device=fg_for_count.device),
             torch.cumsum(fg_for_count.int(), dim=0)]
        )
        win_sums = cs_cnt[N:] - cs_cnt[:-N]  # [L-N+1]

        # --- New: Prefer windows where *all* N slices have FG ---
        full_fg_valid = torch.nonzero(win_sums == N).squeeze(1)  # starts of windows with FG on every slice

        if full_fg_valid.numel() > 0:
            # Ideal case: at least one all-FG window exists
            valid = full_fg_valid
        else:
            # Fallback: original behavior using min_fg_frames_in_window
            max_fg_in_window = int(win_sums.max().item()) if win_sums.numel() > 0 else 0
            eff_min_fg = max(1, min(desired_min_fg, max_fg_in_window))
            valid = torch.nonzero(win_sums >= eff_min_fg).squeeze(1)

            # Absolute last resort: any non-empty window by area
            if valid.numel() == 0:
                cs_any = torch.cat(
                    [torch.tensor([0], device=area.device),
                     torch.cumsum((area > 0).int(), dim=0)]
                )
                valid = torch.nonzero((cs_any[N:] - cs_any[:-N]) > 0).squeeze(1)

        # Choose window start i0 using area-based policy
        if valid.numel() == 0:
            i0 = 0
        else:
            # Rolling total FG area per window
            cs_area  = torch.cat([torch.tensor([0.0], device=area.device), torch.cumsum(area, dim=0)])
            win_area = cs_area[N:] - cs_area[:-N]                              # [L-N+1]
            policy = str(getattr(self, "window_select", "weighted"))          # "weighted"|"argmax"|"uniform"
            if policy == "argmax":
                j = torch.argmax(win_area[valid]).item()
                i0 = int(valid[j].item())
            elif policy == "weighted":
                temp = float(getattr(self, "window_temp", 0.25))
                pwr  = float(getattr(self, "window_area_power", 1.0))
                scores = win_area[valid].clamp_min(0)
                if pwr != 1.0:
                    scores = scores.pow(pwr)
                m = float(scores.max().item()) if scores.numel() > 0 else 0.0
                logits = scores / max(m, 1e-8) / max(temp, 1e-6)
                probs  = torch.softmax(logits, dim=0)
                j = int(torch.multinomial(probs, 1).item())
                i0 = int(valid[j].item())
            else:
                j = torch.randint(0, valid.numel(), (1,)).item()
                i0 = int(valid[j].item())

        sel = list(range(s0 + i0, s0 + i0 + N))
        return imgs[sel], masks[sel]

    def generate_prompts(self, masks, data_dim, strategy="train"):
        """
        Generates per-frame prompts.

        strategy="train":
            Preserve the existing training behavior:
            - choose prompt frames with self.select_prompt_frames(...)
            - sample exactly one prompt type per chosen frame from self.prompt_mix

        strategy="prob_eval":
            Use the same probability-driven prompt-frame selection policy validation used:
            - eligible = GT-nonempty frames
            - each eligible frame is selected independently with prompt_prob
            - if none selected, force the earliest eligible frame
            - for each selected frame, materialize prompts for all three modes
            (point, bbox, mask) so validation can choose among them later

        Returns:
            For strategy="train":
                (pt_list, label_list, bbox_list, m_prompt_list)

            For strategy="prob_eval":
                {
                    "chosen_frames": List[int],
                    "forced_first": bool,
                    "point": {"pt_list": ..., "p_label": ..., "bbox": ..., "m_prompt": ...},
                    "box":   {"pt_list": ..., "p_label": ..., "bbox": ..., "m_prompt": ...},
                    "mask":  {"pt_list": ..., "p_label": ..., "bbox": ..., "m_prompt": ...},
                }

        Args:
            masks (torch.Tensor): Sequence masks with shape ``[T, H, W]`` or ``[T, 1, H, W]``.
            data_dim (int): Source data dimension, either 2 or 3.
            strategy (str): Prompt strategy, either ``"train"`` or ``"prob_eval"``.
        """
        T = int(masks.shape[0])
        H = int(masks.shape[-2])
        W = int(masks.shape[-1])

        def _empty_lists():
            """Return empty prompt containers for each frame."""
            pts = [torch.empty((0, 2), dtype=torch.int64) for _ in range(T)]
            lbs = [torch.empty((0,), dtype=torch.int64) for _ in range(T)]
            box = [torch.empty((0, 4), dtype=torch.int64) for _ in range(T)]
            msk = [torch.empty((0, H, W), dtype=torch.uint8) for _ in range(T)]
            return pts, lbs, box, msk

        def _mask_at(t):
            """Return a 2D mask slice."""
            m = masks[t]
            if m.ndim == 3 and m.size(0) == 1:
                m = m.squeeze(0)
            return m

        if strategy == "train":
            prompt_frames = set(self.select_prompt_frames(masks, data_dim=data_dim))

            keys = list(self.prompt_mix.keys())
            probs = [self.prompt_mix[k] for k in keys]

            def sample_type_for_frame():
                """
                Sample one prompt type from the configured prompt mix.

                Args:
                    None.

                Returns:
                    str: Prompt type key.
                """
                return random.choices(keys, weights=probs, k=1)[0]

            pt_list, label_list, bbox_list, m_prompt_list = [], [], [], []

            for t in range(T):
                mask_slice = _mask_at(t)

                if t in prompt_frames:
                    want_type = sample_type_for_frame()

                    pts_t   = torch.empty((0, 2), dtype=torch.int64)
                    lbls_t  = torch.empty((0,), dtype=torch.int64)
                    bbox_t  = torch.empty((0, 4), dtype=torch.int64)
                    mask_t  = torch.empty((0, H, W), dtype=torch.uint8)

                    if want_type == "click":
                        p_click = generate_prompt(
                            mask_slice,
                            prompt_type="click",
                            num_pos=self.num_pos_clicks,
                            num_neg=(self.num_neg_clicks if self.enable_negative_clicks else 0),
                            neg_margin_frac=self.neg_margin_frac,
                        )
                        if 'points' in p_click:
                            pts_t = p_click['points'].to(torch.int64)
                            lbls_t = p_click['labels'].to(torch.int64)

                    elif want_type == "bbox":
                        p_box = generate_prompt(mask_slice, prompt_type="bbox")
                        if 'bbox' in p_box:
                            bbox_t = p_box['bbox'].to(torch.int64).view(1, 4)

                    elif want_type == "mask":
                        p_mask = generate_prompt(mask_slice, prompt_type="mask")
                        if 'mask' in p_mask:
                            mask_t = p_mask['mask'].to(torch.uint8).unsqueeze(0)

                    pt_list.append(pts_t)
                    label_list.append(lbls_t)
                    bbox_list.append(bbox_t)
                    m_prompt_list.append(mask_t)
                else:
                    pt_list.append(torch.empty((0, 2), dtype=torch.int64))
                    label_list.append(torch.empty((0,), dtype=torch.int64))
                    bbox_list.append(torch.empty((0, 4), dtype=torch.int64))
                    m_prompt_list.append(torch.empty((0, H, W), dtype=torch.uint8))

            return pt_list, label_list, bbox_list, m_prompt_list

        if strategy != "prob_eval":
            raise ValueError(f"Unknown prompt generation strategy: {strategy}")

        # Probability-driven validation prompt cache
        import numpy as np

        flat = masks.reshape(T, -1)
        gt_nonempty = flat.any(dim=1)
        eligible = [int(i) for i in torch.nonzero(gt_nonempty, as_tuple=False).squeeze(1).tolist()]

        point_pts, point_lbls, point_box, point_msk = _empty_lists()
        box_pts, box_lbls, box_box, box_msk = _empty_lists()
        mask_pts, mask_lbls, mask_box, mask_msk = _empty_lists()

        if len(eligible) == 0:
            return {
                "chosen_frames": [],
                "forced_first": False,
                "point": {
                    "pt_list": point_pts,
                    "p_label": point_lbls,
                    "bbox": point_box,
                    "m_prompt": point_msk,
                },
                "box": {
                    "pt_list": box_pts,
                    "p_label": box_lbls,
                    "bbox": box_box,
                    "m_prompt": box_msk,
                },
                "mask": {
                    "pt_list": mask_pts,
                    "p_label": mask_lbls,
                    "bbox": mask_box,
                    "m_prompt": mask_msk,
                },
            }

        prompt_prob = float(
            getattr(
                self,
                "val_prompt_prob_3d" if int(data_dim) == 3 else "val_prompt_prob_2d",
                0.25 if int(data_dim) == 3 else 0.30,
            )
        )

        base_seed = int(getattr(self, "seed", 42))
        rng = np.random.RandomState(base_seed)
        chosen = [t for t in eligible if rng.rand() < prompt_prob]
        forced_first = False
        if len(chosen) == 0:
            chosen = [eligible[0]]
            forced_first = True

        chosen = sorted(set(int(t) for t in chosen))

        for t in chosen:
            mask_slice = _mask_at(t)

            p_click = generate_prompt(
                mask_slice,
                prompt_type="click",
                num_pos=self.num_pos_clicks,
                num_neg=(self.num_neg_clicks if self.enable_negative_clicks else 0),
                neg_margin_frac=self.neg_margin_frac,
            )
            if 'points' in p_click:
                point_pts[t] = p_click['points'].to(torch.int64)
                point_lbls[t] = p_click['labels'].to(torch.int64)

            p_box = generate_prompt(mask_slice, prompt_type="bbox")
            if 'bbox' in p_box:
                box_box[t] = p_box['bbox'].to(torch.int64).view(1, 4)

            p_mask = generate_prompt(mask_slice, prompt_type="mask")
            if 'mask' in p_mask:
                mask_msk[t] = p_mask['mask'].to(torch.uint8).unsqueeze(0)

        return {
            "chosen_frames": chosen,
            "forced_first": forced_first,
            "point": {
                "pt_list": point_pts,
                "p_label": point_lbls,
                "bbox": point_box,
                "m_prompt": point_msk,
            },
            "box": {
                "pt_list": box_pts,
                "p_label": box_lbls,
                "bbox": box_box,
                "m_prompt": box_msk,
            },
            "mask": {
                "pt_list": mask_pts,
                "p_label": mask_lbls,
                "bbox": box_box.__class__([torch.empty((0, 4), dtype=torch.int64) for _ in range(T)]),
                "m_prompt": mask_msk,
            },
        }

    def _generate_validation_prompt_cache(self, masks, data_dim):
        """
        Build validation prompts for all supported prompt modes.

        Args:
            masks (torch.Tensor): Sequence masks.
            data_dim (int): Source data dimension, either 2 or 3.

        Returns:
            dict: Prompt cache keyed by chosen frames and prompt mode.
        """
        cache = self.generate_prompts(masks, data_dim=data_dim, strategy="prob_eval")

        # The generate_prompts(prob_eval) helper intentionally builds concrete prompts
        # for each mode. For mask mode, make sure bbox stays empty and vice versa.
        T = int(masks.shape[0])
        H = int(masks.shape[-2])
        W = int(masks.shape[-1])

        if "mask" in cache:
            cache["mask"]["bbox"] = [torch.empty((0, 4), dtype=torch.int64) for _ in range(T)]
            cache["mask"]["pt_list"] = [torch.empty((0, 2), dtype=torch.int64) for _ in range(T)]
            cache["mask"]["p_label"] = [torch.empty((0,), dtype=torch.int64) for _ in range(T)]

        if "box" in cache:
            cache["box"]["pt_list"] = [torch.empty((0, 2), dtype=torch.int64) for _ in range(T)]
            cache["box"]["p_label"] = [torch.empty((0,), dtype=torch.int64) for _ in range(T)]
            cache["box"]["m_prompt"] = [torch.empty((0, H, W), dtype=torch.uint8) for _ in range(T)]

        if "point" in cache:
            cache["point"]["bbox"] = [torch.empty((0, 4), dtype=torch.int64) for _ in range(T)]
            cache["point"]["m_prompt"] = [torch.empty((0, H, W), dtype=torch.uint8) for _ in range(T)]

        return cache

    def select_prompt_frames(self, masks, data_dim=2):
        """
        Given a sequence of masks, select frames to be used as prompts.

        1. Seed earliest FG (or 0)
        2. Fill remaining by area-biased sampling
        3. 3D-only: Include frame 0 if it has FG and capacity remains
        4. 3D-only: Add one more: the promptable frame with largest FG among the *others*
        5. 3D-only: If still empty but we have any FG, force the max-FG frame

        Truncates to max_prompt_frames, keeping earliest indices for determinism.

        Args:
            masks (Tensor): [T,H,W] sequence of masks
            data_dim (int): Dimensionality of the data (2 or 3)

        Returns:
            List[int]: Selected frame indices
        """
        T = int(masks.shape[0])
        if T <= 0:
            return []

        # Flatten mask to per-frame FG stats
        flat = masks.reshape(T,-1)
        fg_per_frame = flat.any(dim=1)                      # [T] bool
        area_per_frame = flat.sum(dim=1).to(torch.float32)  # [T] float

        prompt_frames = set()

        # 1) Seed earliest FG (or 0)
        if getattr(self, 'always_prompt_first', True) and T > 0:
            nz   = torch.nonzero(fg_per_frame, as_tuple=False).squeeze(1)
            seed = int(nz[0].item()) if nz.numel() > 0 else 0
            prompt_frames.add(seed)

        # 2) Fill remaining by area-biased sampling
        remaining = max(0, int(getattr(self, "max_prompt_frames", 2)) - len(prompt_frames))
        if remaining > 0 and T > 1:
            bias_area = bool(getattr(self, "prompt_bias_area", True))
            pool = [i for i in range(T) if i not in prompt_frames]
            if len(pool) > 0:
                if bias_area:
                    idx = torch.tensor(pool, dtype=torch.long, device=area_per_frame.device)
                    scores = area_per_frame[idx].clamp_min(0)
                    if float(scores.sum().item()) > 0.0:
                        # mild temperature to keep variety
                        logits = scores / max(float(scores.max().item()), 1e-8) / 0.5
                        probs  = torch.softmax(logits, dim=0).cpu().numpy()
                        import numpy as _np
                        k = min(remaining, len(pool))
                        picked = _np.random.choice(pool, size=k, replace=False, p=probs)
                        prompt_frames.update(int(i) for i in picked.tolist())
                    else:
                        import random as _rnd
                        pick = _rnd.sample(pool, k=min(remaining, len(pool)))
                        prompt_frames.update(pick)
                else:
                    import random as _rnd
                    pick = _rnd.sample(pool, k=min(remaining, len(pool)))
                    prompt_frames.update(pick)

        # === 3D-only heuristics ===
        if int(data_dim) == 3:
            # a) Include frame 0 if it has FG and capacity remains
            promptable = [i for i in range(T) if bool(fg_per_frame[i].item())]
            if 0 in promptable and 0 not in prompt_frames and len(prompt_frames) < int(getattr(self, "max_prompt_frames", 2)):
                prompt_frames.add(0)

            # b) Add one more: the promptable frame with largest FG among the *others*
            if len(prompt_frames) < int(getattr(self, "max_prompt_frames", 2)):
                others = [i for i in promptable if i not in prompt_frames]
                if others:
                    # pick the one with max area
                    with torch.no_grad():
                        areas = area_per_frame[torch.tensor(others, dtype=torch.long)]
                        best_j = int(torch.argmax(areas).item())
                    prompt_frames.add(int(others[best_j]))

            # c) If still empty but we have any FG, force the max-FG frame
            if len(prompt_frames) == 0 and len(promptable) > 0:
                with torch.no_grad():
                    areas = area_per_frame[torch.tensor(promptable, dtype=torch.long)]
                best_j = int(torch.argmax(areas).item())
                prompt_frames.add(int(promptable[best_j]))

        # Truncate / top up to exactly max_prompt_frames (or T if shorter)
        max_k_cfg = int(getattr(self, "max_prompt_frames", 2))
        max_k = min(max_k_cfg, T)

        # If we somehow asked for 0, just return what we have
        if max_k <= 0:
            return sorted(prompt_frames)

        # If too many, keep earliest for determinism
        if len(prompt_frames) > max_k:
            prompt_frames = set(sorted(prompt_frames)[:max_k])

        # If too few, TOP UP by preferring FG frames with larger area,
        # then background frames if needed.
        elif len(prompt_frames) < max_k:
            missing = max_k - len(prompt_frames)
            pool = [i for i in range(T) if i not in prompt_frames]

            if pool:
                # Prefer FG frames in the pool if available
                fg_pool = [i for i in pool if bool(fg_per_frame[i].item())]
                if fg_pool:
                    pool = fg_pool

                bias_area = bool(getattr(self, "prompt_bias_area", True))
                if bias_area:
                    idx = torch.tensor(pool, dtype=torch.long, device=area_per_frame.device)
                    scores = area_per_frame[idx].clamp_min(0)

                    if float(scores.sum().item()) > 0.0:
                        # mild temperature to keep variety
                        logits = scores / max(float(scores.max().item()), 1e-8) / 0.5
                        probs  = torch.softmax(logits, dim=0).cpu().numpy()
                        import numpy as _np
                        k = min(missing, len(pool))
                        extra = _np.random.choice(pool, size=k, replace=False, p=probs)
                        prompt_frames.update(int(i) for i in extra.tolist())
                    else:
                        import random as _rnd
                        extra = _rnd.sample(pool, k=min(missing, len(pool)))
                        prompt_frames.update(extra)
                else:
                    import random as _rnd
                    extra = _rnd.sample(pool, k=min(missing, len(pool)))
                    prompt_frames.update(extra)

        return sorted(prompt_frames)

    def _load_2d_sequence(self, seq):
        """
        Load a single 2D image-mask pair as a length-1 sequence.

        The active 2D pipeline uses one image per sample. Multi-frame 2D sequences
        are intentionally not supported here.

        Args:
            seq (list[tuple[str, str]]): Single image/mask path pair.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image and mask tensors with a sequence dimension.
        """
        if len(seq) != 1:
            raise ValueError(f"2D samples must contain exactly one image-mask pair, got {len(seq)}")

        imp_p, msk_p = seq[0]
        i_arr = sitk.GetArrayFromImage(sitk.ReadImage(imp_p))
        m_arr = sitk.GetArrayFromImage(sitk.ReadImage(msk_p))
        im_t, ms_t = self._to_tensor(i_arr, m_arr)
        if self.transform:
            im_t, ms_t = self.transform(im_t, ms_t)
        return im_t.unsqueeze(0), ms_t.unsqueeze(0)

    def _load_3d(self, img_path, mask_path):
        """
        Load a 3D image and its mask as a sequence of 2D images.
        When truncate=True, select ONLY the highest in-plane resolution plane (usually axial),
        and pick a window centered so the largest-FG slice is in the middle (with a fallback
        to the closest valid start). Only the selected N slices are loaded/transformed.
        When truncate=False, keep old behavior (full forward+reverse) but on that fixed plane.

        Args:
            img_path (str): Path to the 3D image volume.
            mask_path (str): Path to the 3D mask volume.

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]: Image sequence,
                mask sequence, and axis lengths.
        """
        # Read volumes
        itk_img = sitk.ReadImage(img_path)
        itk_msk = sitk.ReadImage(mask_path)
        img_vol = sitk.GetArrayFromImage(itk_img)   # [D,H,W]
        msk_vol = sitk.GetArrayFromImage(itk_msk)   # [D,H,W]

        # Reference output size (old behavior used the first axial slice)
        _, H0, W0 = self._to_tensor(img_vol[0], msk_vol[0])[0].shape

        # Axis selection: ONLY use the highest in-plane resolution plane (usually axial)
        msk_np = np.array(msk_vol, copy=False)
        min_fg = int(self.min_fg_frames_in_window)
        D, H, W = img_vol.shape
        inplane_area = np.array([H * W, D * W, D * H], dtype=np.int64)  # axes 0,1,2
        chosen_axis = int(inplane_area.argmax())

        # ---------- Fast path when truncating: select only the needed N slices ----------
        if self.truncate:
            N = int(self.max_frames_per_sequence)

            # per-slice area for the chosen axis (vectorized)
            other = tuple(i for i in (0, 1, 2) if i != chosen_axis)
            per_slice_area = msk_np.sum(axis=other).astype(np.int64)  # [L]
            L = int(per_slice_area.shape[0])
            if L < N:
                raise ValueError(f"Axis too short: need {N}, have {L}")

            # if no FG on this plane at all, bail
            if per_slice_area.sum() <= 0:
                raise ValueError("No FG on highest-resolution plane")

            # robust FG gating (avoid tiny slivers) to match select_window
            if chosen_axis == 0:
                H_, W_ = H, W
            elif chosen_axis == 1:
                H_, W_ = D, W
            else:
                H_, W_ = D, H

            fg_min_frac = float(getattr(self, "fg_min_pixels_frac", 0.0002))
            min_pix     = max(2, int(fg_min_frac * H_ * W_))
            fg_big      = (per_slice_area >= min_pix)
            fg_any      = (per_slice_area > 0)
            fg_for_count = fg_big if fg_big.any() else fg_any

            # Rolling FG count over windows of size N
            cs_cnt   = np.concatenate([[0], np.cumsum(fg_for_count.astype(np.int32))])
            win_sums = cs_cnt[N:] - cs_cnt[:-N]  # [L-N+1]

            # rolling area per window (for fallback tiebreaks)
            cs_area = np.concatenate([[0], np.cumsum(per_slice_area.astype(np.float64))])
            win_area = cs_area[N:] - cs_area[:-N]  # [L-N+1]

            # Index of largest-FG slice
            k_star    = int(per_slice_area.argmax())
            i0_center = max(0, min(L - N, k_star - (N // 2)))

            # --- New: prefer windows where *all* N slices have FG ---
            full_fg_valid = np.flatnonzero(win_sums == N)

            if full_fg_valid.size > 0:
                # Ideal case: at least one all-FG window exists
                valid = full_fg_valid

                # If the centered window is one of them, use it
                if (0 <= i0_center < win_sums.shape[0]) and (win_sums[i0_center] == N):
                    i0 = i0_center
                else:
                    # Otherwise pick the valid window closest to center, breaking ties by area
                    dists = np.abs(valid - i0_center)
                    best  = np.lexsort((-win_area[valid], dists))  # sort by dist asc, area desc
                    i0    = int(valid[best[0]])

            else:
                # Fallback: original behavior using min_fg_frames_in_window
                min_fg = int(self.min_fg_frames_in_window)
                max_fg_in_window = int(win_sums.max()) if win_sums.size > 0 else 0
                eff_min_fg = max(1, min(min_fg, max_fg_in_window))

                valid = np.flatnonzero(win_sums >= eff_min_fg)

                # last resort: any window with any FG present
                if valid.size == 0:
                    cs_any = np.concatenate([[0], np.cumsum((per_slice_area > 0).astype(np.int32))])
                    valid  = np.flatnonzero((cs_any[N:] - cs_any[:-N]) > 0)

                if (0 <= i0_center < win_sums.shape[0]) and (win_sums[i0_center] >= eff_min_fg):
                    i0 = i0_center
                elif valid.size > 0:
                    # choose the valid start closest to the centered start,
                    # breaking ties by larger window area
                    dists = np.abs(valid - i0_center)
                    best  = np.lexsort((-win_area[valid], dists))  # dist asc, area desc
                    i0    = int(valid[best[0]])
                else:
                    i0 = 0  # absolute last resort

            # forward time order
            chosen_idx = list(range(i0, i0 + N))

            # load + transform ONLY those N slices, resizing to (H0, W0) as before
            seq_imgs, seq_masks = [], []
            for k in chosen_idx:
                slice_img = self._slice(img_vol, chosen_axis, k)
                slice_msk = self._slice(msk_vol, chosen_axis, k)
                im_t, ms_t = self._to_tensor(slice_img, slice_msk)
                if self.transform:
                    im_t, ms_t = self.transform(im_t, ms_t)
                # enforce (H0, W0) to keep shapes consistent with old pipeline
                if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                    im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                    ms_t = TF.resize(
                        ms_t.unsqueeze(0).float(), [H0, W0],
                        interpolation=TF.InterpolationMode.NEAREST
                    ).squeeze(0).to(torch.uint8)
                seq_imgs.append(im_t)
                seq_masks.append(ms_t)

            # axis_lengths=None -> skip select_window for these (already windowed)
            return torch.stack(seq_imgs, dim=0), torch.stack(seq_masks, dim=0), None

        # ---------- Validation / test path (not truncating): keep old behavior ----------
        # Build forward+reverse full sequences and transform everything (as before) on chosen_axis.
        if not self.truncate:
            # Trim to first..last FG slice on chosen axis
            if chosen_axis == 0:
                has_fg = np.any(msk_np, axis=(1, 2))
            elif chosen_axis == 1:
                has_fg = np.any(msk_np, axis=(0, 2))
            else:
                has_fg = np.any(msk_np, axis=(0, 1))
            has_fg = has_fg.astype(bool)

            if has_fg.any():
                fg = np.flatnonzero(has_fg)
                start_k, end_k = int(fg[0]), int(fg[-1] + 1)
            else:
                start_k, end_k = 0, msk_np.shape[chosen_axis]
            idxs_fwd = list(range(start_k, end_k))
            idxs_rev = list(reversed(idxs_fwd))
        else:
            # (won't reach here because truncate case returned above)
            idxs_fwd = list(range(msk_np.shape[chosen_axis]))
            idxs_rev = list(reversed(idxs_fwd))

        seq_imgs, seq_masks = [], []
        for i in idxs_fwd + idxs_rev:
            slice_img = self._slice(img_vol, chosen_axis, i)
            slice_msk = self._slice(msk_vol, chosen_axis, i)
            im_t, ms_t = self._to_tensor(slice_img, slice_msk)
            if self.transform:
                im_t, ms_t = self.transform(im_t, ms_t)
                if (im_t.shape[1], im_t.shape[2]) != (H0, W0):
                    im_t = TF.resize(im_t, [H0, W0], interpolation=TF.InterpolationMode.BILINEAR)
                    ms_t = TF.resize(
                        ms_t.unsqueeze(0).float(), [H0, W0],
                        interpolation=TF.InterpolationMode.NEAREST
                    ).squeeze(0).long()
            seq_imgs.append(im_t)
            seq_masks.append(ms_t)

        return torch.stack(seq_imgs, dim=0), torch.stack(seq_masks, dim=0), (D, H, W)

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
        # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if img.ndim == 2:               # Single channel
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)  # HxWxC -> CxHxW
        # Convert mask to PyTorch long tensor after binarization
        mask = torch.from_numpy((msk_arr > 0).astype(np.uint8))
        return img, mask

class BalancedTaskBatchSampler(BatchSampler):
    """
    Build homogeneous batches balanced by dimension and task membership.

    Args:
        BatchSampler (type): PyTorch batch sampler base class.

    Returns:
        None.
    """

    def __init__(
        self,
        sequences,
        tasks_map,
        batch_size,
        drop_last=False,
        seed=None,
        num_samples=None,
        min_per_bucket=1,
        target_dim_ratio=None,
        undersample_bias_power=1.5,
    ):
        """
        Initialize sampler memberships, quotas, and random state.

        Args:
            sequences (list[dict]): Dataset sequence metadata.
            tasks_map (dict): Task catalog keyed by task id.
            batch_size (int): Number of samples per batch.
            drop_last (bool): Whether to drop incomplete batches.
            seed (int | None): Base random seed.
            num_samples (int | None): Target sample count per epoch.
            min_per_bucket (int): Minimum allocation per eligible bucket.
            target_dim_ratio (dict | None): Optional target ratio for 2D/3D items.
            undersample_bias_power (float): Strength of rare-task protection.

        Returns:
            None.
        """
        self.sequences = sequences
        self.tasks_map = tasks_map
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed) if seed is not None else None
        self.rng = random.Random(self.seed)
        self.min_per_bucket = int(max(0, min_per_bucket))
        self.num_samples = int(num_samples) if num_samples is not None else len(sequences)
        self.target_dim_ratio = target_dim_ratio
        self.undersample_bias_power = float(max(0.0, undersample_bias_power))

        if self.target_dim_ratio is not None:
            r2 = float(self.target_dim_ratio.get(2, 0.0))
            r3 = float(self.target_dim_ratio.get(3, 0.0))
            s = r2 + r3
            if s <= 0:
                raise ValueError("target_dim_ratio must assign positive mass to dim 2 and/or 3")
            self.target_dim_ratio = {2: r2 / s, 3: r3 / s}

        # memberships[idx] = [(dim, task_id), ...]
        self.memberships = []
        self.valid_indices = []
        raw_bucket_sizes = Counter()

        for idx, seq in enumerate(sequences):
            ds = seq["dataset"]
            sd = seq.get("subdataset")
            dim = int(seq.get("dim", -1))

            keys = []
            for task_id in seq.get("tasks", []):
                info = tasks_map.get(task_id, {})
                ds_map = info.get("datasets", {})
                if ds in ds_map and sd in ds_map.get(ds, []) and dim in (2, 3):
                    key = (dim, str(task_id))
                    keys.append(key)
                    raw_bucket_sizes[key] += 1

            keys = sorted(set(keys))
            self.memberships.append(keys)
            if keys:
                self.valid_indices.append(idx)

        self.raw_bucket_sizes = dict(raw_bucket_sizes)

        # Representation counts by (dim, task). This is what drives biased
        # undersampling within the overrepresented dimension.
        dim_task_counts = Counter()
        dim_counts = Counter()
        task_counts = Counter()

        for (dim, task_id), n in self.raw_bucket_sizes.items():
            dim_task_counts[(int(dim), str(task_id))] += int(n)
            dim_counts[int(dim)] += int(n)
            task_counts[str(task_id)] += int(n)

        self.dim_task_dataset_counts = dict(dim_task_counts)
        self.dim_dataset_counts = dict(dim_counts)
        self.task_dataset_counts = dict(task_counts)

        max_unique = len(self.valid_indices)
        target_items = min(self.num_samples, max_unique)
        self.effective_target = (
            (target_items // self.batch_size) * self.batch_size
            if self.drop_last
            else target_items
        )

        self.post_dim_effective_target = self.effective_target

    def __len__(self):
        """
        Return the number of batches in the current epoch schedule.

        Args:
            None.

        Returns:
            int: Number of batches.
        """
        target = getattr(self, "post_dim_effective_target", self.effective_target)
        if target <= 0:
            return 0
        if self.drop_last:
            return target // self.batch_size
        return (target + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        """
        Reset the sampler RNG for an epoch.

        Args:
            epoch (int): Epoch index.

        Returns:
            None.
        """
        base = 0 if self.seed is None else self.seed
        self.rng = random.Random(base + int(epoch))

    def _allocate_largest_remainder(self, capacities, target_units, floor_units):
        """
        Allocate integer quotas across buckets with largest-remainder rounding.

        Args:
            capacities (dict): Maximum units available per bucket.
            target_units (int): Total units to allocate.
            floor_units (int): Preferred minimum allocation per bucket.

        Returns:
            dict: Bucket-to-quota mapping.
        """
        if target_units <= 0 or not capacities:
            return {k: 0 for k in capacities}

        quotas = {k: min(floor_units, cap) for k, cap in capacities.items()}
        used = sum(quotas.values())

        if used > target_units:
            over = used - target_units
            for k, _ in sorted(quotas.items(), key=lambda kv: kv[1], reverse=True):
                if over == 0:
                    break
                take = min(over, quotas[k])
                quotas[k] -= take
                over -= take
            return quotas

        remaining = target_units - used
        if remaining == 0:
            return quotas

        residual_cap = {k: max(0, capacities[k] - quotas[k]) for k in capacities}
        cap_sum = sum(residual_cap.values())
        if cap_sum == 0:
            return quotas

        raw = {k: remaining * (residual_cap[k] / cap_sum) for k in capacities}
        ints = {k: min(residual_cap[k], int(raw[k])) for k in capacities}

        for k, v in ints.items():
            quotas[k] += v

        leftover = target_units - sum(quotas.values())
        if leftover > 0:
            order = sorted(
                capacities,
                key=lambda k: (raw[k] - int(raw[k]), residual_cap[k], self.rng.random()),
                reverse=True,
            )
            for k in order:
                if leftover == 0:
                    break
                if quotas[k] < capacities[k]:
                    quotas[k] += 1
                    leftover -= 1

        return quotas

    def _assign_primary_buckets(self):
        """
        Assign each index to exactly one eligible bucket for this epoch.

        Args:
            None.

        Returns:
            dict: Bucket-to-index-list mapping.
        """
        assigned = defaultdict(list)
        assigned_counts = Counter()

        order = self.valid_indices[:]
        self.rng.shuffle(order)
        order.sort(key=lambda idx: (len(self.memberships[idx]), self.rng.random()))

        for idx in order:
            choices = self.memberships[idx]

            def score(key):
                """
                Score a candidate bucket for primary assignment.

                Args:
                    key (tuple): Candidate ``(dim, task_id)`` bucket.

                Returns:
                    tuple: Sortable score where lower is preferred.
                """
                raw = max(1, self.raw_bucket_sizes.get(key, 1))
                return (
                    assigned_counts[key] / raw,
                    assigned_counts[key],
                    raw,
                    self.rng.random(),
                )

            key = min(choices, key=score)
            assigned[key].append(idx)
            assigned_counts[key] += 1

        for key in assigned:
            self.rng.shuffle(assigned[key])

        return dict(assigned)

    def _weighted_sample_without_replacement(self, items, weights, k):
        """
        Sample items without replacement using Efraimidis-Spirakis weights.

        Args:
            items (list): Candidate items.
            weights (list[float]): Sampling weights aligned to items.
            k (int): Number of items to keep.

        Returns:
            list: Selected items.
        """
        n = len(items)
        if k <= 0 or n == 0:
            return []
        if k >= n:
            return list(items)

        keys = []
        for item, w in zip(items, weights):
            w = max(float(w), 1e-12)
            u = self.rng.random()
            while u <= 0.0:
                u = self.rng.random()
            key = u ** (1.0 / w)
            keys.append((key, item))

        keys.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in keys[:k]]

    def _dim_task_keep_weight(self, dim, task_id):
        """
        Compute the keep weight for a sample in a dimension/task bucket.

        Args:
            dim (int): Data dimension, usually 2 or 3.
            task_id (str): Task identifier.

        Returns:
            float: Relative keep weight.
        """
        dim = int(dim)
        task_id = str(task_id)

        count = self.dim_task_dataset_counts.get((dim, task_id), 0)
        if count <= 0:
            return 1.0

        return 1.0 / (float(count) ** self.undersample_bias_power)

    def _apply_dim_budget(self, pools):
        """
        Enforce target_dim_ratio by undersampling within dimensions.

        Args:
            pools (dict): Bucket-to-index-list mapping.

        Returns:
            dict: Budgeted bucket pools.
        """
        if self.target_dim_ratio is None:
            self.post_dim_effective_target = self.effective_target
            return pools

        by_dim = {2: [], 3: []}
        for (dim, task), idxs in pools.items():
            if dim in by_dim:
                by_dim[dim].extend([(dim, task, idx) for idx in idxs])

        avail = {2: len(by_dim[2]), 3: len(by_dim[3])}

        # If one dim is absent, ratio balancing is not meaningful.
        if avail[2] == 0 or avail[3] == 0:
            self.post_dim_effective_target = self.effective_target
            return pools

        total_target = int(self.effective_target)
        r2 = float(self.target_dim_ratio.get(2, 0.0))
        r3 = float(self.target_dim_ratio.get(3, 0.0))

        if r2 > 0 and r3 > 0:
            max_total_from_2 = avail[2] / r2
            max_total_from_3 = avail[3] / r3
            feasible_total = int(min(total_target, max_total_from_2, max_total_from_3))
            target2 = min(avail[2], int(round(feasible_total * r2)))
            target3 = min(avail[3], feasible_total - target2)
        else:
            target2 = min(avail[2], int(round(total_target * r2)))
            target3 = min(avail[3], total_target - target2)

        if self.drop_last:
            target2 = (target2 // self.batch_size) * self.batch_size
            target3 = (target3 // self.batch_size) * self.batch_size

        self.post_dim_effective_target = int(target2 + target3)

        keep = set()

        # Keep a biased-random subset within each dimension.
        # The dimension that is not over budget will typically keep everything.
        for dim, target_dim in ((2, target2), (3, target3)):
            items = by_dim[dim]
            if not items or target_dim <= 0:
                continue

            if target_dim >= len(items):
                keep.update(items)
                continue

            weights = [
                self._dim_task_keep_weight(dim, task)
                for dim, task, idx in items
            ]
            kept_items = self._weighted_sample_without_replacement(items, weights, target_dim)
            keep.update(kept_items)

        new_pools = defaultdict(list)
        for (dim, task), idxs in pools.items():
            for idx in idxs:
                if (dim, task, idx) in keep:
                    new_pools[(dim, task)].append(idx)

        for key in new_pools:
            self.rng.shuffle(new_pools[key])

        return dict(new_pools)

    def _make_batches(self, pools):
        """
        Convert sampled bucket pools into concrete batches.

        Args:
            pools (dict): Bucket-to-index-list mapping.

        Returns:
            tuple[dict, int]: Batches grouped by bucket and total batch count.
        """
        target = getattr(self, "post_dim_effective_target", self.effective_target)

        if target <= 0:
            return {}, 0

        if self.drop_last:
            capacities = {
                k: len(v) // self.batch_size
                for k, v in pools.items()
                if len(v) >= self.batch_size
            }
            target_batches = target // self.batch_size
            floor_batches = math.ceil(self.min_per_bucket / self.batch_size) if self.min_per_bucket > 0 else 0
            batch_quotas = self._allocate_largest_remainder(
                capacities, target_batches, floor_batches
            )

            out = {}
            total_batches = 0
            for key, n_batches in batch_quotas.items():
                if n_batches <= 0:
                    continue
                idxs = pools[key][: n_batches * self.batch_size]
                batches = [
                    [(idx, key[1]) for idx in idxs[i:i + self.batch_size]]
                    for i in range(0, len(idxs), self.batch_size)
                ]
                if batches:
                    out[key] = batches
                    total_batches += len(batches)

            return out, total_batches

        capacities = {k: len(v) for k, v in pools.items() if v}
        item_quotas = self._allocate_largest_remainder(
            capacities, target, self.min_per_bucket
        )

        out = {}
        total_batches = 0
        for key, n_items in item_quotas.items():
            if n_items <= 0:
                continue
            idxs = pools[key][:n_items]
            batches = [
                [(idx, key[1]) for idx in idxs[i:i + self.batch_size]]
                for i in range(0, len(idxs), self.batch_size)
            ]
            if batches:
                out[key] = batches
                total_batches += len(batches)

        return out, total_batches

    def _build_schedule(self, batches_by_bucket, total_batches):
        """
        Spread bucket batches across an epoch schedule.

        Args:
            batches_by_bucket (dict): Bucket-to-batch-list mapping.
            total_batches (int): Total number of batches to schedule.

        Returns:
            list: Ordered bucket keys to emit.
        """
        descriptors = []
        for key, batches in batches_by_bucket.items():
            n = len(batches)
            for j in range(n):
                ideal_pos = (j + 0.5) * total_batches / max(1, n)
                descriptors.append((ideal_pos, self.rng.random(), key))

        descriptors.sort(key=lambda x: (x[0], x[1]))
        return [key for _, _, key in descriptors]

    def __iter__(self):
        """
        Yield batches of ``(index, task_id)`` tuples.

        Args:
            None.

        Returns:
            Iterator[list[tuple[int, str]]]: Batch iterator.
        """
        if self.effective_target <= 0 or not self.valid_indices:
            return
            yield  # pragma: no cover

        # 1) Assign each sample to one bucket for this epoch
        pools = self._assign_primary_buckets()

        # 2) Enforce requested 2D/3D ratio
        pools = self._apply_dim_budget(pools)

        # 3) Allocate quotas and chunk into batches
        batches_by_bucket, total_batches = self._make_batches(pools)
        if total_batches == 0:
            return
            yield  # pragma: no cover

        # 4) Spread bucket batches across the epoch
        schedule = self._build_schedule(batches_by_bucket, total_batches)

        cursors = {k: 0 for k in batches_by_bucket}
        emitted = 0
        target = getattr(self, "post_dim_effective_target", self.effective_target)

        for key in schedule:
            i = cursors[key]
            if i >= len(batches_by_bucket[key]):
                continue

            batch = batches_by_bucket[key][i]
            cursors[key] += 1

            if not batch:
                continue

            emitted += len(batch)
            yield batch

            if emitted >= target:
                break

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

        Returns:
            None.
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

        Args:
            None.

        Returns:
            None.
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
            if self.int_invert:
                cmin = image.amin(dim=(1,2), keepdim=True)
                cmax = image.amax(dim=(1,2), keepdim=True)
                image = (cmin + cmax) - image                       # Invert within native window
            else:
                image = image * self.int_factor                     # Linear intensity scale
        if self.do_contrast:
            center = image.mean(dim=(1,2), keepdim=True)
            image = (image - center) * self.contrast_factor + center  # native contrast
        if self.do_lowres:
            small_h, small_w = max(1,int(h/self.down)), max(1,int(w/self.down))
            image = TF.resize(image, [small_h, small_w], interpolation=TF.InterpolationMode.NEAREST)
            image = TF.resize(image, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask.unsqueeze(0).float(), [small_h, small_w], interpolation=TF.InterpolationMode.NEAREST)
            mask  = TF.resize(mask, [h, w], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).to(torch.uint8)
        if self.do_gamma:
            # Window per channel, gamma in unit window, then restore native range
            cmin = image.amin(dim=(1,2), keepdim=True)
            cmax = image.amax(dim=(1,2), keepdim=True)
            span = (cmax - cmin).clamp_min(1e-6)
            u = (image - cmin) / span
            image = u.pow(self.gamma) * span + cmin
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

    # Collapse any leading channel/time dims so always have 2D (H,W)
    if m.ndim > 2:
        # Reduce all leading dims by OR -> foreground if present in any leading slice
        lead = tuple(range(m.ndim - 2))
        m = m.any(dim=lead)

    H, W = int(m.shape[-2]), int(m.shape[-1])

    # Find all foreground pixels
    ys, xs = torch.nonzero(m, as_tuple=True)

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
