# Visualization helpers for RWKV-MedSAM2 outputs.
#
# Provides 2D sequence animations, 3D overlay montages, triplanar viewers,
# GIF export, TensorBoard video conversion, and NIfTI probability export.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.widgets import Slider
from IPython.display import display, HTML
import numpy as np
import torch
from scipy.ndimage import binary_erosion
import SimpleITK as sitk
from PIL import Image

def visualize_sequence(image_seq, mask_seq, pred_logits_seq, threshold=0.5, fps=2, figsize=(16,4), ping_pong=False):
    """
    Animate a sequence of 2D frames side-by-side:
      [Input image | GT mask | Predicted mask | Image + GT/Pred overlay]

    Plays forward then backward in a loop (optional ping-pong style).

    Args:
        image_seq       (torch.Tensor or np.ndarray): [T,C,H,W] or [T,H,W]
        mask_seq        (torch.Tensor or np.ndarray): [T,1,H,W] or [T,H,W]
        pred_logits_seq (torch.Tensor or np.ndarray): [T,1,H,W] or [T,H,W]
        threshold       (float): sigmoid cutoff for binarizing pred.
        fps             (int): frames per second.
        figsize         (tuple): size of the matplotlib figure.
        ping_pong       (bool): play forward then backward in a loop (ping-pong style).
    Returns:
        ani (FuncAnimation): the animation object (can be saved or displayed).
    """
    # 1) To numpy & drop singleton channels -> [T,H,W]
    if hasattr(image_seq, 'cpu'):
        image_seq       = image_seq.cpu().numpy()
        mask_seq        = mask_seq.cpu().numpy()
        pred_logits_seq = pred_logits_seq.cpu().numpy()
    if mask_seq.ndim == 4 and mask_seq.shape[1] == 1:
        mask_seq = mask_seq[:,0]
    if pred_logits_seq.ndim == 4 and pred_logits_seq.shape[1] == 1:
        pred_logits_seq = pred_logits_seq[:,0]

    T = image_seq.shape[0]

    # 2)Build normalized frames
    frames = []
    for t in range(T):
        img = np.squeeze(image_seq[t])
        # Only transpose if channel-first and not HxWxC
        if img.ndim == 3 and img.shape[0] in (1,3,4):
            img = img.transpose(1,2,0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Binarize GT
        gt   = (np.squeeze(mask_seq[t]) > 0).astype(np.uint8)
        prob = torch.sigmoid(torch.as_tensor(np.squeeze(pred_logits_seq[t]), dtype=torch.float32)).numpy()
        pred = (prob >= threshold).astype(np.uint8)
        frames.append((img, gt, pred))

    # 3) Set up figure & initial images
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4, figsize=figsize)
    for ax in (ax0,ax1,ax2,ax3): ax.axis('off')
    ax0.set_title('Input'); ax1.set_title('GT'); ax2.set_title('Pred'); ax3.set_title('Overlay')

    img0, gt0, pr0 = frames[0]
    im0 = ax0.imshow(img0, cmap=('gray' if img0.ndim==2 else None), animated=True)
    im1 = ax1.imshow(gt0,  cmap='gray', vmin=0, vmax=1, animated=True)
    im2 = ax2.imshow(pr0,  cmap='gray', vmin=0, vmax=1, animated=True)
    im3 = ax3.imshow(_make_gt_pred_overlay(img0, gt0, pr0), animated=True)

    # 4) ping-pong frame indices & interval
    if ping_pong and T > 1:
        idx_seq = list(range(T)) + list(range(T-2, 0, -1))
    else:
        idx_seq = list(range(T))

    interval = 1000 / max(1,fps)

    def _update(i):
        """Update animation artists for one frame."""
        img, gt, pr = frames[i]
        im0.set_array(img); im1.set_array(gt); im2.set_array(pr); im3.set_array(_make_gt_pred_overlay(img, gt, pr))
        return im0, im1, im2, im3

    ani = animation.FuncAnimation(fig, _update, frames=idx_seq, interval=interval, blit=True, repeat=True)
    plt.close(fig)
    display(HTML(ani.to_jshtml()))

def _to_np_vol(x):
    """
    Return volume as [Z,H,W] float32 numpy array.

    Args:
        x (torch.Tensor | str | np.ndarray): Volume tensor, image path, or array.

    Returns:
        np.ndarray: Float32 volume with shape ``[Z, H, W]``.
    """
    if isinstance(x, torch.Tensor): x = x.detach().cpu.numpy()
    elif isinstance(x, str): x = sitk.GetArrayFromImage(sitk.ReadImage(x))
    else: x = np.array(x)
    while x.ndim > 3: x = np.squeeze(x, axis=0)
    if x.ndim != 3: raise ValueError(f"Expected 3D volume [Z,H,W], got {x.shape}")
    return x.astype(np.float32)

def _norm01(x):
    """
    Helper to normalize volume to [0,1].

    Args:
        x (np.ndarray): Input image or volume array.

    Returns:
        np.ndarray: Array normalized to ``[0, 1]``.
    """
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

def _contour(mask2d):
    """
    Return the 2D mask with a single pixel contour.

    Args:
        mask2d (np.ndarray): 2D binary mask.

    Returns:
        np.ndarray: Boolean contour mask.
    """
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

    Returns:
        Any: Saved path or None, depending on the helper.
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
    gt = _to_np_vol(gt_vol) if gt_vol is not None else None

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
    gt = _to_np_vol(gt_vol) if gt_vol is not None else None
    Z, H, W = img.shape

    def slice_ax(k):
        """Return an axial slice triple."""
        return img[k, :, :], prob[k, :, :], gt[k, :, :] if gt else None

    def slice_co(k):
        """Return a coronal slice triple."""
        return img[:, k, :], prob[:, k, :], gt[:, k, :] if gt else None

    def slice_sa(k):
        """Return a sagittal slice triple."""
        return img[:, :, k], prob[:, :, k], gt[:, :, k] if gt else None

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 3, height_ratios=[20, 1])
    axA = fig.add_subplot(gs[0, 0]); axC = fig.add_subplot(gs[0, 1]); axS = fig.add_subplot(gs[0, 2])

    kA, kC, kS = Z // 2, H // 2, W // 2
    def draw(ax, data, title):
        """Draw one triplanar overlay view."""
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
        """Redraw views after slider movement."""
        draw(axA, slice_ax(int(sA.val)), f"Axial z={int(sA.val)}")
        draw(axC, slice_co(int(sC.val)), f"Coronal h={int(sC.val)}")
        draw(axS, slice_sa(int(sS.val)), f"Sagittal w={int(sS.val)}")
        fig.canvas.draw_idle()

    for s in (sA, sC, sS):
        s.on_changed(on_change)

    plt.tight_layout()
    return fig

def _prob_to_rgb(prob_01: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """
    Map a [H,W] float array in [0,1] to an RGB uint8 image using a matplotlib colormap.

    Args:
        prob_01 (np.ndarray): Probability image in ``[0, 1]``.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        np.ndarray: RGB uint8 image with shape ``[H, W, 3]``.
    """
    p = np.nan_to_num(prob_01.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(p)                    # [H,W,4], float in [0,1]
    rgb  = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb

def _make_gt_pred_overlay(image_rgb, gt_mask, pred_mask, alpha=0.45):
    """
    Overlay GT and prediction masks on an RGB display image.

    Colors:
      - GT only: green
      - Prediction only: red
      - Overlap: yellow
    """
    base = np.asarray(image_rgb, dtype=np.float32)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    if base.max() > 1.0:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)

    gt = np.asarray(gt_mask) > 0
    pred = np.asarray(pred_mask) > 0
    if gt.shape != base.shape[:2]:
        gt = _to_hw(gt, reduce="max").astype(bool)
    if pred.shape != base.shape[:2]:
        pred = _to_hw(pred, reduce="max").astype(bool)

    color = np.zeros_like(base, dtype=np.float32)
    color[gt] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    color[pred] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    color[gt & pred] = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    mask = gt | pred
    out = base.copy()
    out[mask] = (1.0 - float(alpha)) * base[mask] + float(alpha) * color[mask]
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

def _to_hw(arr: np.ndarray, reduce="max") -> np.ndarray:
    """
    Robustly convert an array to shape [H,W].
    - Squeezes all singleton axes.
    - If still >2D, reduces all leading channels/axes down to [H,W] using max/mean.

    Args:
        arr (np.ndarray): Input array with at least two spatial dimensions.
        reduce (str): Reduction mode for extra axes, either ``"max"`` or ``"mean"``.

    Returns:
        np.ndarray: 2D array with shape ``[H, W]``.
    """
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # assume [C,H,W] or [H,W,C]; pick channel-first if it matches
        if a.shape[0] not in (a.shape[-1],):
            return a.max(axis=0) if reduce == "max" else a.mean(axis=0)
        return a.max(axis=-1) if reduce == "max" else a.mean(axis=-1)
    # Generic fallback: collapse to last two dims
    lead = tuple(range(0, a.ndim - 2))
    return (a.max(axis=lead) if reduce == "max" else a.mean(axis=lead))

def make_vis_frames(image_seq, mask_seq, pred_logits_seq, threshold=0.5):
    """
    Generate side-by-side RGB frames: [Input | GT | Pred-heatmap | Overlay].

    Args:
        image_seq (torch.Tensor | np.ndarray): Image sequence.
        mask_seq (torch.Tensor | np.ndarray): Ground-truth mask sequence.
        pred_logits_seq (torch.Tensor | np.ndarray): Predicted logit sequence.
        threshold (float): Retained for compatibility; heatmaps use probabilities.

    Returns:
        list[np.ndarray]: RGB visualization frames.
    """
    seq2np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    def seq2np(x):
        """Convert tensors or arrays to CPU numpy arrays."""
        # Ensure numpy gets CPU float32 - avoids unsupported bfloat16/float16
        if torch.is_tensor(x):
            return x.detach().to(dtype=torch.float32, device="cpu").numpy()
        return np.asarray(x)
    img = seq2np(image_seq)         # [T,C,H,W] or [T,H,W]
    msk = seq2np(mask_seq)          # [T,*,H,W] (any extras ok)
    log = seq2np(pred_logits_seq)   # [T,*,H,W] (any extras ok)

    if img.ndim == 3: img = img[:, None, ...]          # [T,1,H,W]
    # Ensure mask/logits are [T,H,W] regardless of weird shapes
    if msk.ndim >= 3:
        msk = np.stack([_to_hw(msk[t], reduce="max") for t in range(msk.shape[0])], axis=0)
    if log.ndim >= 3:
        log = np.stack([_to_hw(log[t], reduce="max") for t in range(log.shape[0])], axis=0)

    # --- Align sequence lengths: some batches may produce fewer logits than images ---
    T_img = img.shape[0]
    T_msk = msk.shape[0] if msk.ndim >= 3 else 0
    T_log = log.shape[0] if log.ndim >= 3 else 0
    T = int(min(T_img, T_msk, T_log))
    if T <= 0:
        raise ValueError(f"[make_vis_frames] Non-positive common length: T_img={T_img}, T_msk={T_msk}, T_log={T_log}")

    # Truncate to common prefix
    img = img[:T]
    msk = msk[:T]
    log = log[:T]

    _, _, H, W = img.shape
    frames = []
    for t in range(T):
        im_rgb = _disp_norm_tensor(img[t])                        # [H,W,3] in [0,1]
        gt = ( _to_hw(msk[t], reduce="max") > 0 ).astype(np.uint8) # [H,W]
        lt = _to_hw(log[t], reduce="max").astype(np.float32)       # [H,W]
        logits = np.clip(lt, -20.0, 20.0)                                 # stability
        prob = 1.0 / (1.0 + np.exp(-logits))                      # [H,W] in [0,1]
        pred = (prob >= float(threshold)).astype(np.uint8)         # [H,W]
        heat_rgb = _prob_to_rgb(prob, cmap_name="magma")          # [H,W,3] uint8

        gt_rgb = (np.stack([gt, gt, gt], axis=-1) * 255).astype(np.uint8)
        overlay_rgb = _make_gt_pred_overlay(im_rgb, gt, pred)

        panel = np.concatenate([(im_rgb * 255).astype(np.uint8), gt_rgb, heat_rgb, overlay_rgb], axis=1)
        frames.append(panel)

    return frames

def save_vis_gif(image_seq, mask_seq, pred_logits_seq, out_path, fps=2, threshold=0.5, ping_pong=False):
    """
    Save a visualization of a sequence of images, masks, and predicted logits as a GIF.

    Args:
        image_seq (torch.Tensor or np.ndarray): A 2D or 3D array of images. Each image is of shape (H, W) or (C, H, W).
        mask_seq (torch.Tensor or np.ndarray): A 2D or 3D array of masks. Each mask is of shape (H, W) or (1, H, W).
        pred_logits_seq (torch.Tensor or np.ndarray): A 2D or 3D array of predicted probabilities. Each prediction is of shape (H, W) or (1, H, W).
        out_path (str): The path to save the GIF.
        fps (int): The frames per second of the GIF. Defaults to 2.
        threshold (float): A value between 0 and 1 for binarizing the predicted probabilities. Defaults to 0.5.
        ping_pong (bool): Whether to play the GIF forward and backward in a loop. Defaults to False.

    Returns:
        tuple: A tuple containing the path to the saved GIF and the list of frames.
    """
    frames = make_vis_frames(image_seq, mask_seq, pred_logits_seq, threshold=threshold)
    if ping_pong and len(frames) > 1:
        frames = frames + frames[-2:0:-1]
    imgs = [Image.fromarray(f) for f in frames]
    duration_ms = int(round(1000 / max(1, fps)))
    imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
    return out_path, frames

def frames_to_tb_video(frames):
    """
    Convert a list of RGB frames to a tensor for visualization in TensorBoard.

    Args:
        frames (list): A list of RGB frames. Each frame is an array of shape (H, W, 3).

    Returns:
        torch.Tensor: A tensor of shape (1, T, 3, H, W) representing the video.
    """
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # [T,H,W,3]
    arr = arr.transpose(0,3,1,2)                                # [T,3,H,W]
    return torch.from_numpy(arr).unsqueeze(0)

def _disp_norm_tensor(img_chw: np.ndarray, mode: str = "p1p99", fixed_window=(-3.0, 3.0)) -> np.ndarray:
    """
    Normalize an image array for visualization.

    Args:
        img_chw (np.ndarray): input image array of shape (C,H,W) or (H,W).
        mode (str): normalization mode:
            "fixed": use the fixed_window parameter.
            "p1p99": use the 1st and 99th percentile of the image per channel.
        fixed_window (tuple): (min,max) of the fixed normalization window.

    Returns:
        np.ndarray: normalized image array of shape (3,H,W) with values in [0,1].
    """
    x = img_chw.astype(np.float32)
    if x.ndim == 2:                      # H,W -> 1,H,W
        x = x[None, ...]
    if x.shape[0] not in (1, 3):         # C not 1/3 -> assume HWC and transpose
        x = x.transpose(2, 0, 1)

    def norm01(ch, lo, hi):
        """Normalize one channel with a fixed display window."""
        return np.clip((ch - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    if mode == "fixed":
        lo, hi = fixed_window
        if x.shape[0] == 3:
            r = norm01(x[0], lo, hi); g = norm01(x[1], lo, hi); b = norm01(x[2], lo, hi)
            return np.stack([r, g, b], axis=-1)
        g = norm01(x[0], lo, hi)
        return np.stack([g, g, g], axis=-1)

    # mode == "p1p99": per-channel robust window
    if x.shape[0] == 3:
        out = []
        for c in range(3):
            lo, hi = np.percentile(x[c], (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = x[c].min(), x[c].max() + 1e-6
            out.append(norm01(x[c], lo, hi))
        return np.stack(out, axis=-1)
    lo, hi = np.percentile(x[0], (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = x[0].min(), x[0].max() + 1e-6
    g = norm01(x[0], lo, hi)
    return np.stack([g, g, g], axis=-1)
