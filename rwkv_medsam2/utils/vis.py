# utils/vis.py
# Helpers for visualizing predictions and ground truth masks.
# Mostly lifted from DRIPP debugging code.
# - visualize_predictions_2d    - Visualize a 2D image, its ground truth mask, and its predicted mask.
# - visualize_nifti_predictions - Visualize a 3D image, its ground truth mask, and its predicted mask.
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import numpy as np
import torch
from typing import Union, List
from pathlib import Path
import SimpleITK as sitk

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

    # Collapse any leading singleton dims, then handle channels
    img_np = np.squeeze(image)              # now HxW or CxHxW
    if img_np.ndim == 3:                    # CxHxW -> HxWxC
        img_np = np.transpose(img_np, (1, 2, 0))
    # Normalize to [0,1] for safe imshow
    img_np = img_np.astype(np.float32)
    img_np -= img_np.min()
    img_np /= (img_np.max() + 1e-8)

    mask_np = np.squeeze(mask)              # Drop any 1-sized dims -> HxW
    
    pred_arr = pred_logits
    pred_arr = np.squeeze(pred_arr)         # Drop any channel dim -> HxW
    pred_np = (1 / (1 + np.exp(-pred_arr))) >= threshold

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
    # 1) Load from filepaths if needed
    if isinstance(img_vol, (str, Path)):
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(img_vol)))  # [Z, H, W]
    if isinstance(gt_mask_vol, (str, Path)):
        gt_mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_mask_vol)))
    if isinstance(gt_mask_vol, list):
        gt_mask_vol = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in gt_mask_vol], axis=0)

    # 2) Convert Tensors to numpy
    if isinstance(img_vol, torch.Tensor):
        img_vol = img_vol.cpu().numpy()
    if isinstance(gt_mask_vol, torch.Tensor):
        gt_mask_vol = gt_mask_vol.cpu().numpy()
    if isinstance(pred_logits, torch.Tensor):
        pred_logits = pred_logits.cpu().numpy()

    # 3) Flatten any extra dims so volumes are [Z, H, W]
    if img_vol.ndim > 3:
        img_vol = img_vol.reshape(-1, img_vol.shape[-2], img_vol.shape[-1])
    if gt_mask_vol.ndim > 3:
        gt_mask_vol = gt_mask_vol.reshape(-1, gt_mask_vol.shape[-2], gt_mask_vol.shape[-1])

    # 4) Sigmoid + flatten preds similarly
    pred_vol = 1 / (1 + np.exp(-pred_logits))
    if pred_vol.ndim > 3:
        pred_vol = pred_vol.reshape(-1, pred_vol.shape[-2], pred_vol.shape[-1])

    # 5) Binarize predictions
    pred_mask_vol = (pred_vol >= threshold).astype(np.int32)

    # 6) Compute center indices based on mask shape
    Zm, Hm, Wm = gt_mask_vol.shape
    idx = {
        'Axial':    Zm // 2,
        'Coronal':  Hm // 2,
        'Sagittal': Wm // 2
    }

    # 7) Overlay helper
    def _overlay(slice_img, slice_mask, color, alpha):
        mn, mx = slice_img.min(), slice_img.max()
        norm = ((slice_img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(slice_img, np.uint8)
        rgb = np.stack([norm]*3, -1).astype(float)
        msk = slice_mask.astype(bool)
        for c in range(3):
            rgb[..., c] = np.where(msk, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
        return rgb.astype(np.uint8)

    # 8) Plot
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

        comp = np.concatenate([raw_rgb, gt_rgb, pr_rgb], axis=1)
        ax.imshow(comp)
        ax.set_title(f"{name} (raw -> gt -> pred)")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_sequence(image_seq, mask_seq, pred_logits_seq, threshold=0.5, fps=2, figsize=(12,4)):
    """
    Animate a sequence of 2D frames side-by-side:
      [Input image | GT mask | Predicted mask]

    Plays forward then backward in a loop (ping-pong style).

    Args:
        image_seq       (torch.Tensor or np.ndarray): [T,C,H,W] or [T,H,W]
        mask_seq        (torch.Tensor or np.ndarray): [T,1,H,W] or [T,H,W]
        pred_logits_seq (torch.Tensor or np.ndarray): [T,1,H,W] or [T,H,W]
        threshold       (float): sigmoid cutoff for binarizing pred.
        fps             (int): frames per second.
        figsize         (tuple): size of the matplotlib figure.
    Returns:
        ani (FuncAnimation): the animation object (can be saved or displayed).
    """
    # 1) to numpy & drop singleton channels → [T,H,W]
    if hasattr(image_seq, 'cpu'):
        image_seq       = image_seq.cpu().numpy()
        mask_seq        = mask_seq.cpu().numpy()
        pred_logits_seq = pred_logits_seq.cpu().numpy()
    if mask_seq.ndim == 4 and mask_seq.shape[1] == 1:
        mask_seq = mask_seq[:,0]
    if pred_logits_seq.ndim == 4 and pred_logits_seq.shape[1] == 1:
        pred_logits_seq = pred_logits_seq[:,0]

    T = image_seq.shape[0]
    # 2) build normalized frames
    frames = []
    for t in range(T):
        img = np.squeeze(image_seq[t])
        # Only transpose if channel-first and not HxWxC
        if img.ndim == 3 and img.shape[0] in (1,3,4):
            img = img.transpose(1,2,0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        gt    = np.squeeze(mask_seq[t])
        pred  = (1/(1+np.exp(-np.squeeze(pred_logits_seq[t])))) >= threshold
        frames.append((img, gt, pred))

    # 3) set up figure & initial images
    fig, (ax0,ax1,ax2) = plt.subplots(1,3, figsize=figsize)
    for ax in (ax0,ax1,ax2): ax.axis('off')
    ax0.set_title('Input'); ax1.set_title('GT'); ax2.set_title('Pred')

    img0 = frames[0][0] # Use gray for 2D and RGB for anything else
    if img0.ndim == 2:
        im0 = ax0.imshow(img0, cmap='gray', animated=True)
    else:
        im0 = ax0.imshow(img0, animated=True)
    im1 = ax1.imshow(frames[0][1], cmap='gray', animated=True)
    im2 = ax2.imshow(frames[0][2], cmap='gray', animated=True)

    # 4) ping-pong frame indices & interval
    seq = list(range(T)) + list(range(T-2, 0, -1))
    interval = 1000 / fps  # ms per frame

    def _update(i):
        img, gt, pr = frames[i]
        im0.set_array(img)
        im1.set_array(gt)
        im2.set_array(pr)
        return im0, im1, im2

    ani = animation.FuncAnimation(
        fig, _update, frames=seq,
        interval=interval, blit=True, repeat=True
    )

    plt.close(fig)                     # prevent double‐display
    display(HTML(ani.to_jshtml()))