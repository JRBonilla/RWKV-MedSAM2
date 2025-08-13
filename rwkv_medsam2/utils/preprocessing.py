# rwkv_medsam2/utils/preprocessing.py
# Preprocessing utils for RWKV-MedSAM2
# _mask_qc_cache_key(p)                 - Generate cache key for mask quality check
# check_mask_quality(mask_path)         - Check mask quality based on specified criteria
# load_mask_bool_3d(mask_path)          - Load a 3D mask as a boolean numpy array
# check_masks_quality_gpu(mask_paths)   - Check quality of multiple masks in parallel
import os
import math
import torch
import numpy as np
import SimpleITK as sitk

from math import ceil
from concurrent.futures import ThreadPoolExecutor

_MASK_QC_CACHE = {}

def _mask_qc_cache_key(p):
    """
    Generates a cache key for a given file path based on its metadata.

    Args:
        p (str): The file path for which to generate the cache key.

    Returns:
        tuple: A tuple containing the file path, last modification time, 
               and file size. If the file cannot be accessed, returns 
               the file path, 0, and 0.
    """
    try:
        st = os.stat(p)
        return (p, int(st.st_mtime), int(st.st_size))
    except OSError:
        return (p, 0, 0)

def check_mask_quality(
    mask_path,
    min_voxels=64,
    min_slices_any_axis=2,
    min_slice_area_px=32,
    *,
    downsample=2,                 # Downsample factor: 1 (exact, slower), 2 (fast), 4 (very fast)
    use_full_if_borderline=True,  # Refine borderline fails at full-res
    borderline_ratio=0.8          # How close to thresholds counts as "borderline"
):
    """
    Checks the quality of a 3D mask based on specified criteria.

    Args:
        mask_path (str): The file path to the mask image.
        min_voxels (int, optional): Minimum number of non-zero voxels required. Defaults to 64.
        min_slices_any_axis (int, optional): Minimum number of slices with non-zero voxels in any axis. Defaults to 2.
        min_slice_area_px (int, optional): Minimum area of non-zero voxels in any slice. Defaults to 32.
        downsample (int, optional): Factor to downsample each axis for a fast first pass. Defaults to 2.
        use_full_if_borderline (bool): If True, re-check at full res when close to thresholds.
        borderline_ratio (float): Borderline band as a fraction of thresholds (e.g., 0.8 = within 20%).

    Returns:
        tuple: (ok: bool, info: dict). info contains:
               - "voxels": total FG voxels (approx if downsample-only)
               - "slices_by_axis": [Z, Y, X] counts (approx if downsample-only)
               - "max_area": max per-slice area across any axis (approx if downsample-only)
               - "reason": optional reason like "read_error" or "not_3d"
    """
    # Check cache first
    key = _mask_qc_cache_key(mask_path)
    if key in _MASK_QC_CACHE:
        return _MASK_QC_CACHE[key]

    # Read mask
    try:
        msk_itk = sitk.ReadImage(mask_path)
        m = sitk.GetArrayFromImage(msk_itk)  # [Z, Y, X]
    except Exception as e:
        res = (False, {"reason": "read_error", "error": str(e)})
        _MASK_QC_CACHE[key] = res
        return res

    # Convert to binary
    m_bin = (np.asarray(m) > 0)

    # Only enforce checks for 3D
    if m_bin.ndim != 3:
        res = (True, {"reason": "not_3d"})
        _MASK_QC_CACHE[key] = res
        return res

    # Fast path: downsample first (integer stride, no copies) to reduce PCIe traffic
    ds = int(max(1, downsample))
    m_ds = m_bin[::ds, ::ds, ::ds] if ds > 1 else m_bin

    # Scale thresholds to downsampled grid
    # Voxels scale ~ ds^3; per-slice area scales ~ ds^2
    min_vox_ds   = max(1, ceil(min_voxels / (ds**3)))
    min_area_ds  = max(1, ceil(min_slice_area_px / (ds**2)))
    min_slices   = int(min_slices_any_axis)

    # Quick counts on the downsampled grid
    vox_ds = int(m_ds.sum())
    if vox_ds < min_vox_ds:
        # Hard fail unless borderline and we choose to double-check
        if use_full_if_borderline and vox_ds >= int(borderline_ratio * min_vox_ds):
            return check_mask_quality(mask_path, min_voxels, min_slices_any_axis, min_slice_area_px,
                                       downsample=1, use_full_if_borderline=False)
        res = (False, {"voxels": vox_ds * (ds**3), "slices_by_axis": [0,0,0], "max_area": 0})
        _MASK_QC_CACHE[key] = res
        return res

    # Slices with any FG along each axis
    sZ = int(np.count_nonzero(m_ds.any(axis=(1, 2))))
    sY = int(np.count_nonzero(m_ds.any(axis=(0, 2))))
    sX = int(np.count_nonzero(m_ds.any(axis=(0, 1))))
    slices_by_axis_ds = [sZ, sY, sX]
    if max(slices_by_axis_ds) < min_slices:
        if use_full_if_borderline and max(slices_by_axis_ds) >= int(borderline_ratio * min_slices):
            return check_mask_quality(mask_path, min_voxels, min_slices_any_axis, min_slice_area_px,
                                       downsample=1, use_full_if_borderline=False)
        res = (False, {"voxels": vox_ds * (ds**3), "slices_by_axis": slices_by_axis_ds, "max_area": 0})
        _MASK_QC_CACHE[key] = res
        return res

    # Max per-slice area across any axis (on DS grid)
    max_area_ds = int(max(
        m_ds.sum(axis=(1, 2)).max(initial=0),
        m_ds.sum(axis=(0, 2)).max(initial=0),
        m_ds.sum(axis=(0, 1)).max(initial=0),
    ))
    if max_area_ds < min_area_ds:
        if use_full_if_borderline and max_area_ds >= int(borderline_ratio * min_area_ds):
            return check_mask_quality(mask_path, min_voxels, min_slices_any_axis, min_slice_area_px,
                                       downsample=1, use_full_if_borderline=False)
        res = (False, {
            "voxels": vox_ds * (ds**3),
            "slices_by_axis": slices_by_axis_ds,
            "max_area": max_area_ds * (ds**2),
        })
        _MASK_QC_CACHE[key] = res
        return res

    # Success on the DS grid; report approx counts scaled back up
    res = (True, {
        "voxels": vox_ds * (ds**3),
        "slices_by_axis": slices_by_axis_ds,
        "max_area": max_area_ds * (ds**2),
    })
    _MASK_QC_CACHE[key] = res
    return res

def load_mask_bool_3d(path):
    """
    Read a 3D mask from disk and convert it to a PyTorch bool tensor on the CPU.

    Args:
        path (str): File path to the mask image.

    Returns:
        tuple: (mask: bool tensor, info: dict). info contains:
               - "reason": either "read_error" or "not_3d"
               - "error": if "read_error", a string with the exception message
    """
    try:
        itk = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(itk)  # [Z,Y,X]
        if arr.ndim != 3:
            return None, {"reason": "not_3d"}
        return torch.from_numpy((arr > 0)), None  # CPU bool tensor
    except Exception as e:
        return None, {"reason": "read_error", "error": str(e)}

def check_masks_quality_gpu(
    mask_paths,
    min_voxels=64,
    min_slices_any_axis=2,
    min_slice_area_px=32,
    downsample=2,
    batch_size=16,
    device=None,
    parallel_reads=8
):
    """
    Vectorized GPU check for a list of 3D mask paths.
    Returns list of (ok: bool, info: dict) aligned to mask_paths.
    Uses downsampled stride slicing on CPU before transfer to reduce PCIe traffic.
    Pads volumes in a batch to the max depth so we can stack.
    
    Args:
        mask_paths (list): List of file paths to 3D mask images.
        min_voxels (int, optional): Minimum number of voxels in the mask. Defaults to 64.
        min_slices_any_axis (int, optional): Minimum number of slices with any foreground along any axis. Defaults to 2.
        min_slice_area_px (int, optional): Minimum area (in pixels) of any slice. Defaults to 32.
        downsample (int, optional): Downsample factor for slices. Defaults to 2.
        batch_size (int, optional): Number of masks to check at once. Defaults to 16.
        device (str, optional): Device to run on. Defaults to None.
        parallel_reads (int, optional): Number of threads to use for parallel CPU reads. Defaults to 8.

    Returns:
        list: List of (ok: bool, info: dict) aligned to mask_paths.
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = [None] * len(mask_paths)
    scale_vox  = max(1, downsample ** 3)
    scale_area = max(1, downsample ** 2)

    for start in range(0, len(mask_paths), batch_size):
        chunk_paths = mask_paths[start:start + batch_size]

        # Parallel CPU reads
        with ThreadPoolExecutor(max_workers=min(parallel_reads, len(chunk_paths))) as ex:
            loaded = list(ex.map(load_mask_bool_3d, chunk_paths))

        # Prepare per-chunk outputs, build tensors list for real 3D
        to_stack = []
        idx_map = []  # Map stacked index -> results index in chunk
        for j, (t_cpu, info) in enumerate(loaded):
            out_idx = start + j
            if t_cpu is None:
                # Either not_3d (pass) or read_error (fail)
                if info.get("reason") == "not_3d":
                    results[out_idx] = (True, info)
                else:
                    results[out_idx] = (False, info)
                continue
            # Downsample by stride on CPU, then we will send to GPU
            ds = max(1, int(downsample))
            t_ds = t_cpu[::ds, ::ds, ::ds].contiguous()
            to_stack.append(t_ds)
            idx_map.append(out_idx)

        if not to_stack:
            continue

        # Pad to common shape, stack, send to GPU
        D = max(t.shape[0] for t in to_stack)
        H = max(t.shape[1] for t in to_stack)
        W = max(t.shape[2] for t in to_stack)
        padded = []
        for t in to_stack:
            pad = (0, W - t.shape[2], 0, H - t.shape[1], 0, D - t.shape[0])  # W, H, D
            padded.append(torch.nn.functional.pad(t, pad))  # Pad with False
        batch = torch.stack(padded, 0).to(device=device, dtype=torch.bool, non_blocking=True)

        # Vectorized metrics on GPU
        vox_ds = batch.sum(dim=(1, 2, 3))  # [B]
        sZ = (batch.any(dim=(2, 3))).sum(dim=1)  # [B]
        sY = (batch.any(dim=(1, 3))).sum(dim=1)
        sX = (batch.any(dim=(1, 2))).sum(dim=1)

        areaZ = batch.sum(dim=(2, 3))  # [B,D]
        areaY = batch.sum(dim=(1, 3))  # [B,H]
        areaX = batch.sum(dim=(1, 2))  # [B,W]
        max_area_ds = torch.stack([
            areaZ.max(dim=1).values,
            areaY.max(dim=1).values,
            areaX.max(dim=1).values
        ], dim=1).amax(dim=1)  # [B]

        # Scale thresholds to DS grid
        thr_vox_ds  = math.ceil(min_voxels / scale_vox)
        thr_area_ds = math.ceil(min_slice_area_px / scale_area)

        ok = (
            (vox_ds >= thr_vox_ds) &
            (torch.stack([sZ, sY, sX], dim=1).amax(dim=1) >= int(min_slices_any_axis)) &
            (max_area_ds >= thr_area_ds)
        )

        # Write results back aligned to the original list
        vox_out  = (vox_ds * scale_vox).to(torch.int64).tolist()
        sZ_out   = sZ.to(torch.int64).tolist()
        sY_out   = sY.to(torch.int64).tolist()
        sX_out   = sX.to(torch.int64).tolist()
        area_out = (max_area_ds * scale_area).to(torch.int64).tolist()

        for bi, out_idx in enumerate(idx_map):
            results[out_idx] = (
                bool(ok[bi].item()),
                {
                    "voxels": int(vox_out[bi]),
                    "slices_by_axis": [int(sZ_out[bi]), int(sY_out[bi]), int(sX_out[bi])],
                    "max_area": int(area_out[bi]),
                }
            )

        # Free GPU memory between chunks
        del batch, vox_ds, sZ, sY, sX, areaZ, areaY, areaX, max_area_ds, ok
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    return results