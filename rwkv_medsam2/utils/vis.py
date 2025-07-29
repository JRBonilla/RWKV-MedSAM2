# utils/vis.py
# Helpers for visualizing predictions and ground truth masks.
# Mostly lifted from DRIPP debugging code.
# - visualize_predictions_2d    - Visualize a 2D image, its ground truth mask, and its predicted mask.
# - visualize_nifti_predictions - Visualize a 3D image, its ground truth mask, and its predicted mask.
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Union, List

def visualize_predictions_2d(image, mask, pred_logits, threshold=0.5, figsize=(12, 4)):
    """
    Visualize a 2D image, its ground truth mask, and its predicted mask.

    Args:
        image (torch.tensor or numpy.ndarray): The input image.
        mask (torch.tensor or numpy.ndarray): The ground truth mask.
        pred_logits (torch.tensor or numpy.ndarray): The predicted logits.
        threshold (float): The threshold for binarizing the predicted mask.
        figsize (tuple[int]): The figure size in inches.
    """
    # Convert to numpy if needed
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
        mask = mask.cpu().numpy()
        pred_logits = pred_logits.cpu().numpy()

    img_np  = np.transpose(image, (1, 2, 0)) if image.ndim == 3 else image
    mask_np = mask.squeeze(0)
    pred_np = (1 / (1 + np.exp(-pred_logits))).squeeze(0) >= threshold

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title(f'Predicted >= {threshold}')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_nifti_predictions(img_vol, gt_mask_vol, pred_logits, axes=('Axial', 'Coronal', 'Sagittal'), alpha=0.5, threshold=0.5, figsize=(12, 4)):
    """
    Visualize 3D predictions on a NiFTI volume. The function overlays the
    ground truth mask and predicted mask on top of the input image, and
    displays the results in three views (axial, coronal, sagittal).

    Args:
        img_vol (torch.Tensor or numpy.ndarray): Input image volume. Shape either [Z,C,H,W] or [Z,H,W].
        gt_mask_vol (torch.Tensor or numpy.ndarray): Ground truth mask volume. Shape either [Z,1,H,W] or [Z,H,W].
        pred_logits (torch.Tensor or numpy.ndarray): Predicted logits of shape  [Z,1,H,W] or [Z,H,W].
        axes (tuple[str]): Tuple of view names to display. Default is ('Axial', 'Coronal', 'Sagittal').
        alpha (float): Transparency of the overlay. Default is 0.5.
        threshold (float): Threshold for binarizing the predicted mask. Default is 0.5.
        figsize (tuple[int]): Figure size in inches. Default is (12, 4).
    """
    # Ensure shape
    assert img_vol.ndim in (3,4), "Expected img_vol of shape [Z,H,W] or [Z,C,H,W]"

    # Convert to numpy
    if isinstance(img_vol, torch.Tensor):
        img_vol     = img_vol.cpu().numpy()
        gt_mask_vol = gt_mask_vol.cpu().numpy()
        pred_logits = pred_logits.cpu().numpy()

    # Ensure shape [Z,H,W]
    if img_vol.ndim == 4:
        # [Z,C,H,W] -> take first channel
        img_vol = img_vol[:, 0]
    if gt_mask_vol.ndim == 4:
        gt_mask_vol = gt_mask_vol[:, 0]
    pred_vol = (1 / (1 + np.exp(-pred_logits)))
    if pred_vol.ndim == 4:
        pred_vol = pred_vol[:, 0]

    # Binarize
    pred_mask_vol = (pred_vol >= threshold).astype(np.int32)

    # Overlay helper
    def _overlay(slice_img, slice_mask, color, alpha):
        mn, mx = slice_img.min(), slice_img.max()
        norm = ((slice_img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(slice_img, np.uint8)
        rgb = np.stack([norm]*3, -1).astype(float)
        msk = slice_mask.astype(bool)
        for c in range(3):
            rgb[..., c] = np.where(
                msk,
                (1 - alpha) * rgb[..., c] + alpha * color[c],
                rgb[..., c]
            )
        return rgb.astype(np.uint8)

    # Get center slices
    Z, H, W = img_vol.shape
    idx = {'Axial':   Z//2,
           'Coronal': H//2,
           'Sagittal':W//2}

    # Plot
    fig, axs = plt.subplots(1, len(axes), figsize=figsize)
    for ax, name in zip(axs, axes):
        if name == 'Axial':
            im_s = img_vol[idx[name], :, :]
            gt_s = gt_mask_vol[idx[name], :, :]
            pr_s = pred_mask_vol[idx[name], :, :]
        elif name == 'Coronal':
            im_s = img_vol[:, idx[name], :]
            gt_s = gt_mask_vol[:, idx[name], :]
            pr_s = pred_mask_vol[:, idx[name], :]
        else:
            im_s = img_vol[:, :, idx[name]]
            gt_s = gt_mask_vol[:, :, idx[name]]
            pr_s = pred_mask_vol[:, :, idx[name]]

        raw_rgb = np.stack([((im_s - im_s.min())/(im_s.max()-im_s.min())*255).astype(np.uint8)]*3, -1)
        gt_rgb  = _overlay(im_s, gt_s, color=(255, 0, 0), alpha=alpha)
        pr_rgb  = _overlay(im_s, pr_s, color=(0, 255, 0), alpha=alpha)

        # Concatenate images
        comp = np.concatenate([raw_rgb, gt_rgb, pr_rgb], axis=1)
        ax.imshow(comp)
        ax.set_title(f"{name} (raw -> gt -> pred)")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
