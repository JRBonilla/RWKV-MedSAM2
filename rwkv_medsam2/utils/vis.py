# utils/vis.py
# Helpers for visualizing predictions and ground truth masks.
# Mostly lifted from DRIPP debugging code.
# - visualize_sequence          - Visualize a sequence of 2D frames side-by-side. Plays forward then
#                                 backward in a loop (ping-pong style).
# - montage_overlays            - Create a montage of overlays for a 3D volume.
# - view_triplanar_interactive  - Interactive visualization of triplanar views of a 3D volume.
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
from IPython.display import display, HTML
import numpy as np
import torch
from typing import Union, List
from pathlib import Path
from scipy.ndimage import binary_erosion
import SimpleITK as sitk

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

        # Binarize GT
        gt_raw   = np.squeeze(mask_seq[t])
        gt       = (gt_raw > 0).astype(np.uint8)
        logits_t = torch.as_tensor(np.squeeze(pred_logits_seq[t]), dtype=torch.float32)
        prob     = torch.sigmoid(logits_t).cpu().numpy()
        pred     = (prob >= threshold).astype(np.uint8)
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
    im1 = ax1.imshow(frames[0][1], cmap='gray', vmin=0, vmax=1, animated=True)
    im2 = ax2.imshow(frames[0][2], cmap='gray', vmin=0, vmax=1, animated=True)

    # 4) ping-pong frame indices & interval
    forward  = list(range(T))
    backward = list(range(T-2, 0, -1))
    full_seq = forward + backward
    pos = full_seq.index(0)
    seq = full_seq[pos:] + full_seq[:pos]
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

def _to_np_vol(x):
    """Return volume as [Z,H,W] float32 numpy array."""
    if isinstance(x, torch.Tensor): x = x.detach().cpu.numpy()
    elif isinstance(x, str): x = sitk.GetArrayFromImage(sitk.ReadImage(x))
    else: x = np.array(x)
    while x.ndim > 3: x = np.squeeze(x, axis=0)
    if x.ndim != 3: raise ValueError(f"Expected 3D volume [Z,H,W], got {x.shape}")
    return x.astype(np.float32)

def _norm01(x):
    """Helper to normalize volume to [0,1]."""
    x = x.astype(np.float32)
    if np.isnan(x).any() or np.isinf(x).any(): x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn: x = (x - mn) / (mx - mn)
    else: x = np.zeros_like(x)
    return x

def _contour(mask2d):
    """Return the 2D mask with a single pixel contour."""
    m = mask2d.astype(bool)
    if m.size == 0: return m
    er = binary_erosion(m, iterations=1)
    return m & ~er

def save_prob_to_nifti(prob_vol, out_path, reference_img_path=None):
    """
    Save probability volume [Z,H,W] to NIfTI.

    Args:
        prob_vol     (torch.Tensor or np.ndarray): [Z,H,W] or [H,W]
        out_path     (str): path to save
        reference_img_path (str): path to reference image
    """
    prob = _to_np_vol(prob_vol)
    img  = sitk.GetImageFromArray(prob.astype(np.float32))
    if reference_img_path:
        ref = sitk.ReadImage(reference_img_path)
        img.CopyInformation(ref)
    sitk.WriteImage(img, out_path)

def montage_overlays(img_vol, prob_vol, gt_vol=None, plane='Axial', step=10, ncols=6, prob_alpha=0.5, threshold=0.5, figsize=(12, 4)):
    """
    Create a montage of overlays for a 3D volume.

    Args:
        img_vol (torch.Tensor or np.ndarray): [Z,H,W] or [H,W]
        prob_vol (torch.Tensor or np.ndarray): [Z,1,H,W] or [Z,H,W] or [H,W]
        gt_vol (torch.Tensor or np.ndarray, optional): [Z,1,H,W] or [Z,H,W] or [H,W]
        plane (str, optional): 'Axial', 'Coronal', or 'Sagittal'. Defaults to 'Axial'.
        step (int, optional): Step size for selecting frames. Defaults to 10.
        ncols (int, optional): Number of columns for the montage. Defaults to 6.
        prob_alpha (float, optional): Transparency of the overlay. Defaults to 0.5.
        threshold (float, optional): Threshold for converting probs to 0/1 mask. Defaults to 0.5.
        figsize (tuple[int], optional): Figure size in inches. Defaults to (12, 4).

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Prepare data
    img = _norm01(_to_np_vol(img_vol))              # Normalize image to [0,1]
    prob = _to_np_vol(prob_vol)                     # Aggregated probs expected to be in [0,1]
    if prob.max() > 1.0 or prob.min() < 0.0:
        prob = _norm01(prob)                        # Normalize probs to [0,1] if needed
    gt = _to_np_vol(gt_vol) if gt_vol else None

    # Select plane
    Z, H, W = img.shape
    if plane.lower() == 'axial':
        length = Z; getter = lambda k: (img[k,:,:], prob[k,:,:], gt[k,:,:] if gt else None)
    elif plane.lower() == 'coronal':
        length = H; getter = lambda k: (img[:,k,:], prob[:,k,:], gt[:,k,:] if gt else None)
    elif plane.lower() == 'sagittal':
        length = W; getter = lambda k: (img[:,:,k], prob[:,:,k], gt[:,:,k] if gt else None)
    else:
        raise ValueError(f"Expected plane in {'Axial', 'Coronal', 'Sagittal'}, got {plane}")

    # Create montage
    idxs = list(range(0, length, max(1, step)))
    n = len(idxs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    # Plot
    for i, k in enumerate(idxs):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        im, pr, gt2 = getter(k)
        ax.imshow(im, cmap='gray', interpolation='nearest')
        ax.imshow(pr, cmap='magma', alpha=prob_alpha, vmin=0, vmax=1, interpolation='nearest')
        if gt2 is not None:
            edge = _contour(gt2 > 0.5 if gt2.dtype != bool else gt2)
            ax.imshow(np.where(edge, 1.0, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=0.9, interpolation='nearest')
        ax.set_title(f"{plane} k={k}")
        ax.axis('off')

    # Hide any empty cells
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    # Save
    plt.tight_layout()
    return fig

def view_triplanar_interactive(img_vol, prob_vol, gt_vol=None, prob_alpha=0.5, figsize=(12, 4)):
    """
    Interactive visualization of triplanar views of a 3D volume with overlays.

    This function displays axial, coronal, and sagittal views of a 3D image
    volume with corresponding probability and ground truth overlays. It
    allows interactive exploration using sliders to navigate through slices.

    Args:
        img_vol (torch.Tensor or numpy.ndarray): 3D input image volume.
        prob_vol (torch.Tensor or numpy.ndarray): 3D probability volume.
        gt_vol (torch.Tensor or numpy.ndarray, optional): 3D ground truth mask volume.
        prob_alpha (float): Transparency level for probability overlay. Default is 0.5.
        figsize (tuple[int, int]): Size of the figure in inches. Default is (12, 4).

    Returns:
        fig (matplotlib.figure.Figure): The created matplotlib figure with plots.
    """
    # Prepare data
    img = _norm01(_to_np_vol(img_vol))              # Normalize image to [0,1]
    prob = _to_np_vol(prob_vol)                     # Aggregated probs expected to be in [0,1]
    if prob.max() > 1.0 or prob.min() < 0.0:
        prob = _norm01(prob)                        # Normalize probs to [0,1] if needed
    gt = _to_np_vol(gt_vol) if gt_vol else None
    Z, H, W = img.shape

    def slice_ax(k): return img[k, :, :], prob[k, :, :], gt[k, :, :] if gt else None
    def slice_co(k): return img[:, k, :], prob[:, k, :], gt[:, k, :] if gt else None
    def slice_sa(k): return img[:, :, k], prob[:, :, k], gt[:, :, k] if gt else None

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 3, height_ratios=[20, 1])
    axA = fig.add_subplot(gs[0, 0]); axC = fig.add_subplot(gs[0, 1]); axS = fig.add_subplot(gs[0, 2])

    kA, kC, kS = Z // 2, H // 2, W // 2
    def draw(ax, data, title):
        im, pr, gt2 = data
        ax.clear()
        ax.imshow(im, cmap='gray', interpolation='nearest')
        ax.imshow(pr, cmap='magma', alpha=prob_alpha, vmin=0, vmax=1, interpolation='nearest')
        if gt2 is not None:
            edge = _contour(gt2 > 0.5 if gt2.dtype != bool else gt2)
            ax.imshow(np.where(edge, 1.0, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=0.9, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')

    draw(axA, slice_ax(kA), f"Z={kA}")
    draw(axC, slice_co(kC), f"H={kC}")
    draw(axS, slice_sa(kS), f"W={kS}")

    # Sliders
    axA_s = fig.add_subplot(gs[1, 0]); axC_s = fig.add_subplot(gs[1, 1]); axS_s = fig.add_subplot(gs[1, 2])
    sA = Slider(axA_s, 'Axial',    0, Z - 1, valinit=kA, valfmt="%0.0f")
    sC = Slider(axC_s, 'Coronal',  0, H - 1, valinit=kC, valfmt="%0.0f")
    sS = Slider(axS_s, 'Sagittal', 0, W - 1, valinit=kS, valfmt="%0.0f")

    def on_change(_):
        draw(axA, slice_ax(int(sA.val)), f"Axial z={int(sA.val)}")
        draw(axC, slice_co(int(sC.val)), f"Coronal h={int(sC.val)}")
        draw(axS, slice_sa(int(sS.val)), f"Sagittal w={int(sS.val)}")
        fig.canvas.draw_idle()

    for s in (sA, sC, sS):
        s.on_changed(on_change)

    plt.tight_layout()
    return fig