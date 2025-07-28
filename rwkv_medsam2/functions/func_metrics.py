# functions/func_metrics.py
# Defines evaluation metrics for RWKV‑MedSAM2:
#   - compute_iou  - Intersection over Union between two binary masks
#   - compute_dice - Dice Similarity Coefficient between two binary masks
#   - compute_hd95 - 95th percentile Hausdorff Distance between mask boundaries
import cupy as cp
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff

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