# Metric, loss, and aggregation helpers for RWKV-MedSAM2.
#
# Includes task/modality aggregators, segmentation losses, Dice/IoU, HD95,
# calibration metrics, AUC helpers, and prompt geometry utilities.
import torch
import json
import cupy as cp # type: ignore
import numpy as np
from cupyx.scipy.ndimage import distance_transform_edt, binary_erosion # type: ignore
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, List, Dict, Optional

class TaskAggregator:
    """
    Keeps weighted running averages of metrics per task.
    Slice-wise metrics use slice weight; volume metrics use volume weight.

    Also tracks how many 2D vs 3D items contributed per task so we can report
    2d% / 3d% in the summary table.

    Accepts either a flat id->label map, a full catalog (with classes/datasets),
    or a file path to a JSON containing either structure.
    """
    def __init__(self, tasks_map: Optional[Dict[str, Any] | str] = None):
        """
        Initialize task display labels and running aggregate stores.

        Args:
            tasks_map (dict | str | None): Task label mapping or JSON file path.

        Returns:
            None.
        """
        # Normalize incoming map/path into a flat id -> label map
        raw: Dict[str, Any] = {}
        if isinstance(tasks_map, str):
            try:
                with open(tasks_map, "r") as f:
                    raw = json.load(f)
            except Exception:
                raw = {}
        elif isinstance(tasks_map, dict):
            raw = tasks_map

        self._display_map: Dict[str, str] = self._to_display_map(raw)

        # Running metric sums and counts
        self.sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cnt: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Per-task 2D/3D item counts (for percentage reporting)
        self.dim_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"2d": 0, "3d": 0})

        # Keys averaged with volume weight
        self._vol_keys = {
            'iou3d@0.5', 'dice3d@0.5',
            'iou3d@best', 'dice3d@best',
            'iou3d@best_vol', 'dice3d@best_vol'
        }

    @staticmethod
    def _to_display_map(raw: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert an incoming tasks map to a flat id->label map.

        - If values are strings, assume it's already flat.
        - If values are dicts (catalog), prefer an explicit 'label' field,
          otherwise prettify the key (underscores -> spaces).

        Args:
            raw (dict[str, Any]): Raw task or modality mapping.

        Returns:
            dict[str, str]: Display-label mapping keyed by id.
        """
        if not raw:
            return {}
        first_val = next(iter(raw.values()))
        if isinstance(first_val, str):
            return {str(k): str(v) for k, v in raw.items()}
        disp: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and isinstance(v.get("label"), str):
                label = v["label"]
            else:
                label = str(k).replace("_", " ")
            disp[str(k)] = label
        return disp

    def label_for(self, task_id_or_label: Any) -> str:
        """
        Return a user-facing label string for a task id or label-like value.

        Args:
            task_id_or_label (Any): Task id or label-like value.

        Returns:
            str: Display label.
        """
        if task_id_or_label is None:
            return "unknown"
        if isinstance(task_id_or_label, dict):
            return "unknown"
        tid = str(task_id_or_label)
        return self._display_map.get(tid, tid)

    def update(
        self,
        task_label: str,
        metrics: Dict[str, float],
        slice_weight: int = 1,
        vol_weight: int = 1,
        *,
        dim: Optional[int] = None,
        n_items: Optional[int] = None,
    ):
        """
        Update aggregates for a task.

        Args:
            task_label: human-friendly label (bucket id is fine)
            metrics: dict of metric_name -> value
            slice_weight: weight for slice-based metrics (usually #valid frames)
            vol_weight: weight for volume-based metrics (usually #volumes)
            dim: 2 or 3 for this batch (used only for 2d%/3d% reporting)
            n_items: how many items contributed in this update call
                     (e.g., batch size). If not given, counts as 1.

        Returns:
            None.
        """
        # --- Update metrics ---
        for k, v in metrics.items():
            if k in ('n', 'best_thr', 'best_thr3d', 'n_vol'):
                continue
            if not isinstance(v, (int, float)):
                continue
            w = vol_weight if (k in self._vol_keys) else slice_weight
            self.sum[task_label][k] += float(v) * float(w)
            self.cnt[task_label][k] += float(w)

        # --- Update 2D/3D composition counts (for table only) ---
        if dim in (2, 3):
            c = int(n_items) if (n_items is not None and int(n_items) > 0) else 1
            if dim == 2:
                self.dim_counts[task_label]["2d"] += c
            else:
                self.dim_counts[task_label]["3d"] += c

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Return averaged metrics for each task.

        Args:
            None.

        Returns:
            dict[str, dict[str, float]]: Per-task metric averages.
        """
        out: Dict[str, Dict[str, float]] = {}
        for task, kv in self.sum.items():
            out[task] = {k: (kv[k] / max(1.0, self.cnt[task][k])) for k in kv.keys()}
        return out

    def log_to_tensorboard(self, writer, epoch: int, split: str = "val"):
        """
        Log per-task metrics to TensorBoard.

        Args:
            writer (SummaryWriter): TensorBoard writer.
            epoch (int): Epoch index.
            split (str): Metric namespace prefix.

        Returns:
            None.
        """
        per_task = self.as_dict()
        for task, kv in per_task.items():
            for k, v in kv.items():
                writer.add_scalar(f"{split}/task/{task}/{k}", float(v), epoch)

    def format_text_table(
        self,
        top_keys: Optional[List[str]] = None,
        decimals: int = 4,
        fenced: bool = True,
        fence_lang: str = "text",
        show_dim_pct: bool = True,
    ) -> str:
        """
        Render a fixed-width aligned table. If show_dim_pct=True, includes 2d% / 3d%.

        Args:
            top_keys (list[str] | None): Metric columns to include.
            decimals (int): Number of decimal places.
            fenced (bool): Whether to wrap the table in a Markdown code fence.
            fence_lang (str): Code-fence language tag.
            show_dim_pct (bool): Whether to include 2D/3D composition columns.

        Returns:
            str: Formatted table.
        """
        per_task = self.as_dict()
        if not per_task:
            return "No per-task metrics."

        if top_keys is None:
            top_keys = ['dice_best', 'iou_best', 'dice@0.5', 'iou@0.5', 'pr_auc', 'roc_auc']

        # Build header
        header = ['task']
        if show_dim_pct:
            header += ['2d%', '3d%']
        header += top_keys

        # Build rows (strings)
        rows: List[List[str]] = []
        for task in sorted(set(per_task.keys()) | set(self.dim_counts.keys())):
            row = [str(task)]

            if show_dim_pct:
                dc = self.dim_counts.get(task, {"2d": 0, "3d": 0})
                n2d, n3d = int(dc.get("2d", 0)), int(dc.get("3d", 0))
                tot = n2d + n3d
                if tot > 0:
                    p2d = 100.0 * n2d / tot
                    p3d = 100.0 * n3d / tot
                    row += [f"{p2d:5.1f}%", f"{p3d:5.1f}%"]
                else:
                    row += ["   --", "   --"]

            for k in top_keys:
                v = per_task.get(task, {}).get(k, 0.0)
                row.append(f"{float(v):.{decimals}f}")
            rows.append(row)

        # Compute column widths
        col_widths = [len(h) for h in header]
        for row in rows:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(cell))

        # Format one row (left align col 0, right align rest)
        def fmt_row(cells: List[str]) -> str:
            """Format one aligned table row."""
            parts = []
            for j, cell in enumerate(cells):
                w = col_widths[j]
                parts.append(cell.ljust(w) if j == 0 else cell.rjust(w))
            return "  ".join(parts)

        # Build table text
        lines = [fmt_row(header)]
        lines.append(fmt_row(["-" * w for w in col_widths]))
        for row in rows:
            lines.append(fmt_row(row))
        table = "\n".join(lines)

        # Wrap in a code fence so TensorBoard uses monospace
        if fenced:
            return f"```{fence_lang}\n{table}\n```"
        return table

class ModalityAggregator(TaskAggregator):
    """Keeps weighted running averages of metrics per imaging modality.

    This behaves like TaskAggregator but uses modality labels
    (for example "ct", "mri", "us") as the keys. It also tracks how many
    2D vs 3D items contributed per modality, so we can report 2d% / 3d%
    in the summary table.
    """

    def __init__(self, modalities_map: Optional[Dict[str, Any] | str] = None):
        """
        Initialize modality display labels and aggregate stores.

        Args:
            modalities_map (dict | str | None): Modality label mapping or JSON file path.

        Returns:
            None.
        """
        # Reuse TaskAggregator initialisation logic (display map etc.).
        super().__init__(tasks_map=modalities_map)

    def log_to_tensorboard(self, writer, epoch: int, split: str = "val"):
        """
        Log per-modality metrics to TensorBoard.

        Scalars are written under keys of the form:
            {split}/modality/{modality}/{metric}
        e.g. val/modality/ct/dice@0.5

        Args:
            writer (SummaryWriter): TensorBoard writer.
            epoch (int): Epoch index.
            split (str): Metric namespace prefix.

        Returns:
            None.
        """
        per_mod = self.as_dict()
        for mod, kv in per_mod.items():
            for k, v in kv.items():
                writer.add_scalar(f"{split}/modality/{mod}/{k}", float(v), epoch)

    def format_text_table(
        self,
        top_keys: Optional[List[str]] = None,
        decimals: int = 4,
        fenced: bool = True,
        fence_lang: str = "text",
        show_dim_pct: bool = True,
    ) -> str:
        """
        Render a fixed-width per-modality table.

        If show_dim_pct is True, includes 2d% / 3d% columns that show
        how many 2D and 3D items contributed (as a percentage) to each
        modality's metrics, analogous to TaskAggregator.format_text_table.

        Args:
            top_keys (list[str] | None): Metric columns to include.
            decimals (int): Number of decimal places.
            fenced (bool): Whether to wrap the table in a Markdown code fence.
            fence_lang (str): Code-fence language tag.
            show_dim_pct (bool): Whether to include 2D/3D composition columns.

        Returns:
            str: Formatted table.
        """
        per_mod = self.as_dict()
        if not per_mod:
            return "No per-modality metrics."

        if top_keys is None:
            top_keys = ['dice_best', 'iou_best', 'dice@0.5', 'iou@0.5', 'pr_auc', 'roc_auc']

        # Header row
        header = ['modality']
        if show_dim_pct:
            header += ['2d%', '3d%']
        header += top_keys

        # Build rows
        rows: List[List[str]] = []
        for mod in sorted(set(per_mod.keys()) | set(self.dim_counts.keys())):
            row = [str(mod)]
            if show_dim_pct:
                dc = self.dim_counts.get(mod, {"2d": 0, "3d": 0})
                n2d, n3d = int(dc.get("2d", 0)), int(dc.get("3d", 0))
                tot = n2d + n3d
                if tot > 0:
                    p2d = 100.0 * n2d / tot
                    p3d = 100.0 * n3d / tot
                    row += [f"{p2d:5.1f}%", f"{p3d:5.1f}%"]
                else:
                    row += ["  0.0%", "  0.0%"]

            stats = per_mod.get(mod, {})
            for k in top_keys:
                v = stats.get(k, float("nan"))
                if v == v:  # not NaN
                    row.append(f"{float(v):.{decimals}f}")
                else:
                    row.append("nan")
            rows.append(row)

        # Column widths
        col_widths = [
            max(len(str(header[c])), max(len(r[c]) for r in rows))
            for c in range(len(header))
        ]

        def fmt_row(cells: List[str]) -> str:
            """Format one aligned table row."""
            parts = []
            for j, cell in enumerate(cells):
                w = col_widths[j]
                parts.append(cell.ljust(w) if j == 0 else cell.rjust(w))
            return "  ".join(parts)

        lines = [fmt_row(header)]
        lines.append(fmt_row(["-" * w for w in col_widths]))
        for row in rows:
            lines.append(fmt_row(row))
        table = "\n".join(lines)

        if fenced:
            return f"```{fence_lang}\n{table}\n```"
        return table

def safe_float(x):
    """
    Convert scalar-like values to Python floats when possible.

    Args:
        x (Any): Value to convert.

    Returns:
        float | None: Converted float, or None for unsupported values.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().float().item())
        return None
    try:
        return float(x)
    except Exception:
        return None

@torch.jit.script
def tversky_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3, beta:  float = 0.7, smooth: float = 1.0) -> torch.Tensor:
    """
    Asymmetric Tversky loss (Dice generalization) computed from logits.
    alpha weights FP, beta weights FN. For tiny FG, beta > alpha helps recall.

    Args:
        logits (torch.Tensor): Predicted logits.
        targets (torch.Tensor): Binary target masks.
        alpha (float): False-positive weight.
        beta (float): False-negative weight.
        smooth (float): Smoothing constant.

    Returns:
        torch.Tensor: Scalar loss.
    """
    probs = torch.sigmoid(logits)
    p = probs.reshape(probs.size(0), -1)
    g = targets.to(probs.dtype).reshape(targets.size(0), -1)

    tp = (p * g).sum(dim=1)
    fp = (p * (1.0 - g)).sum(dim=1)
    fn = ((1.0 - p) * g).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky).mean()

@torch.jit.script
def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the soft Dice loss between the output probabilities of a binary
    classification model and the corresponding ground truth labels. The loss is
    computed as 1 - (2 * intersection + eps) / (sum of probabilities + eps)

    Args:
        logits (torch.Tensor): The output logits of a binary classification model
        targets (torch.Tensor): The ground truth labels
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-6

    Returns:
        torch.Tensor: The soft Dice loss
    """
    probs = torch.sigmoid(logits)
    # Flatten over all non-batch dims to avoid dynamic dim tuples in TorchScript
    probs_f   = probs.reshape(probs.size(0), -1)
    targets_f = targets.to(probs.dtype).reshape(targets.size(0), -1)
    inter = (probs_f * targets_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + targets_f.sum(dim=1) + eps
    dice  = (2.0 * inter + eps) / denom
    return (1.0 - dice).mean()

@torch.jit.script
def binary_focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = -1.0) -> torch.Tensor:
    """
    Computes the binary focal loss between the output probabilities of a binary
    classification model and the corresponding ground truth labels. The loss is
    computed as (1 - p_t)^gamma * bce, where p_t is the probability of the true
    class, and bce is the binary cross entropy loss. The alpha parameter allows
    weighting the positive and negative classes differently.

    Args:
        logits (torch.Tensor): The output logits of a binary classification model
        targets (torch.Tensor): The ground truth labels
        gamma (float, optional): The power to which the probability of the true
            class is raised. Defaults to 2.0
        alpha (float, optional): The weighting factor for the positive class if
            alpha >= 0.0. Defaults to -1.0, which disables weighting.

    Returns:
        torch.Tensor: The binary focal loss
    """
    # Base BCE per-element
    t    = targets.to(logits.dtype)
    bce  = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
    p    = torch.sigmoid(logits)
    p_t  = torch.where(t > 0, p, 1.0 - p)
    loss = (1.0 - p_t).pow(gamma) * bce
    # Optional alpha balancing without creating new tensors in TorchScript
    if alpha >= 0.0:
        pos = loss * alpha
        neg = loss * (1.0 - alpha)
        loss = torch.where(t > 0, pos, neg)
    return loss.mean()

def compute_hd95(pred_mask: torch.Tensor, gt_mask: torch.Tensor, spacing=None) -> float:
    """
    Compute the 95th percentile of the Hausdorff Distance between two binary masks.

    The Hausdorff Distance is the maximum distance between two sets of points, where
    the distance between two points is the Euclidean distance between them.

    If the masks are empty, returns 0.0. If one mask is empty and the other is not,
    returns float('nan').

    Args:
        pred_mask (torch.Tensor): A binary mask of shape (H, W).
        gt_mask (torch.Tensor): A binary mask of shape (H, W).
        spacing (tuple of float, optional): The spacing between points in the mask.
            Defaults to (1.0,) * ndim.

    Returns:
        float: The 95th percentile of the Hausdorff Distance between the two masks.
    """
    assert pred_mask.shape == gt_mask.shape, "pred_mask and gt_mask must have the same shape"
    x, y = pred_mask, gt_mask
    ndim = x.ndim
    if spacing is None:
        spacing = (1.0,) * ndim
    spacing = tuple(float(s) for s in spacing)

    # Check for empty masks
    p_any = bool((x != 0).any().item())
    g_any = bool((y != 0).any().item())
    if not p_any and not g_any:
        return 0.0
    if p_any ^ g_any:
        return float('nan')

    # Helper to convert torch.Tensor to cupy.ndarray
    def _to_cupy_bool(t: torch.Tensor) -> "cp.ndarray":
        """Convert a tensor mask to a CuPy boolean array."""
        t = (t != 0).to(torch.uint8).contiguous()
        if t.is_cuda:
            return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t)).astype(cp.bool_)
        return cp.asarray(t.cpu().numpy(), dtype=cp.bool_)

    # Convert to cupy
    pm = _to_cupy_bool(x)
    gm = _to_cupy_bool(y)

    # Return 0 if both masks are empty. Return nan if only one is empty
    if int(pm.sum()) == 0 and int(gm.sum()) == 0:
        return 0.0
    if int(pm.sum()) == 0 or int(gm.sum()) == 0:
        return float('nan')

    # Compute surface (zeros at surface)
    se = cp.ones((3,) * pm.ndim, dtype=cp.bool_)
    sp = cp.logical_xor(pm, binary_erosion(pm, structure=se, border_value=0))
    sg = cp.logical_xor(gm, binary_erosion(gm, structure=se, border_value=0))

    # Compute distance transform to surface of ground truth and prediction
    dt_to_g = distance_transform_edt(cp.logical_not(sg), sampling=spacing)
    dt_to_p = distance_transform_edt(cp.logical_not(sp), sampling=spacing)

    # Distances from pred-surface points to GT surface, and from GT-surface points to pred
    d_to_g = dt_to_g[sp]
    d_to_p = dt_to_p[sg]

    # Handle degenerate surfaces; take 95th pct over available (bi-directional) distances
    if d_to_g.size == 0 and d_to_p.size == 0:
        return 0.0
    if d_to_g.size == 0:
        d95 = cp.percentile(d_to_p.astype(cp.float32), 95)
    elif d_to_p.size == 0:
        d95 = cp.percentile(d_to_g.astype(cp.float32), 95)
    else:
        d95 = cp.percentile(cp.concatenate([d_to_g, d_to_p]).astype(cp.float32), 95)

    return float(cp.asnumpy(d95))

def dice_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    """
    Compute Dice and IoU for flattened binary masks.

    Args:
        pred (torch.Tensor): Predicted binary mask.
        gt (torch.Tensor): Ground-truth binary mask.
        eps (float): Numerical stability term.

    Returns:
        tuple[float, float]: Dice score and IoU score.
    """
    pred_f = pred.reshape(-1).float()
    gt_f = gt.reshape(-1).float()
    inter = (pred_f * gt_f).sum()
    ps = pred_f.sum()
    gs = gt_f.sum()
    dice = (2 * inter + eps) / (ps + gs + eps)
    union = ps + gs - inter
    iou = (inter + eps) / (union + eps)
    return float(dice.item()), float(iou.item())

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute expected calibration error for probabilities and labels.

    Args:
        probs (np.ndarray): Predicted probabilities.
        labels (np.ndarray): Binary labels.
        n_bins (int): Number of calibration bins.

    Returns:
        float: Expected calibration error.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(probs[m]))
        acc = float(np.mean(labels[m]))
        w = float(np.mean(m))
        e += w * abs(acc - conf)
    return float(e)

def try_auc(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute ROC-AUC and PR-AUC when labels contain both classes.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Predicted scores or probabilities.

    Returns:
        tuple[float, float]: ROC-AUC and average precision, or NaNs on failure.
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        if len(np.unique(y_true)) <= 1:
            return float("nan"), float("nan")
        roc = float(roc_auc_score(y_true, y_score))
        pr = float(average_precision_score(y_true, y_score))
        return roc, pr
    except Exception:
        return float("nan"), float("nan")

def fg_bin(fg_frac: float) -> str:
    """
    Map a foreground fraction to a reporting bin label.

    Args:
        fg_frac (float): Foreground area fraction.

    Returns:
        str: Bin label.
    """
    if fg_frac < 0.0005:
        return "fg<0.05%"
    if fg_frac < 0.002:
        return "fg0.05-0.2%"
    if fg_frac < 0.01:
        return "fg0.2-1%"
    return "fg>1%"

def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable sigmoid to numpy logits.

    Args:
        logits (np.ndarray): Input logits.

    Returns:
        np.ndarray: Sigmoid probabilities as float32.
    """
    out = np.empty_like(logits, dtype=np.float32)

    pos = logits >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_x = np.exp(logits[neg])
    out[neg] = exp_x / (1.0 + exp_x)

    return out

def bbox_from_mask(mask_hw: torch.Tensor, pad: int = 5):
    """
    Build an XYXY bounding box from a 2D binary mask.

    Args:
        mask_hw (torch.Tensor): Boolean or binary mask of shape ``[H, W]``.
        pad (int): Pixel padding around the foreground extent.

    Returns:
        torch.Tensor | None: Bounding box tensor, or None for empty masks.
    """
    ys, xs = torch.nonzero(mask_hw, as_tuple=True)
    if ys.numel() == 0:
        return None
    y0 = max(int(ys.min().item()) - pad, 0)
    y1 = min(int(ys.max().item()) + pad, int(mask_hw.shape[0]) - 1)
    x0 = max(int(xs.min().item()) - pad, 0)
    x1 = min(int(xs.max().item()) + pad, int(mask_hw.shape[1]) - 1)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32, device=mask_hw.device)
