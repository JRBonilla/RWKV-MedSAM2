"""
Train and validate volumes as slice sequences with SAM2 propagation and optional distillation.
"""

import math
import torch
import torch.nn.functional as F
from ..distributed import synchronize_gradients

from .func_metrics import (
    tversky_loss_from_logits,
    binary_focal_loss_with_logits,
    safe_float,
)


def train_step_3d(
    student,
    teacher,
    optimizer,
    batch,
    config,
    scaler,
    *,
    zero_grad=True,
    synchronize=True,
    clip_gradients=True,
    optimizer_step=True,
):
    """
    Run one 3D-as-sequence training step with dataset prompts.

    Args:
        student (SAM2VideoPredictor): Student predictor to optimize.
        teacher (SAM2VideoPredictor | None): Optional teacher predictor.
        optimizer (torch.optim.Optimizer): Optimizer for the student.
        batch (dict): Collated 3D sequence batch.
        config (OmegaConf.DictConfig): Training configuration.
        scaler (object | None): AMP scaler placeholder; bf16 autocast is used directly.
        zero_grad (bool): Whether to clear existing gradients before processing the batch.
        synchronize (bool): Whether to average gradients across distributed ranks after
            backward.
        clip_gradients (bool): Whether to clip gradients before the optimizer update.
        optimizer_step (bool): Whether to update parameters after accumulating gradients.

    Returns:
        dict: Step status, aggregate losses, and skip diagnostics on failure.
    """
    device = config.training.device
    out_size = int(config.training.out_size)

    # Loss weights
    bce_w = float(getattr(config.training, "bce_weight", 0.5))
    dice_w = float(getattr(config.training, "dice_weight", 1.0))
    focal_w = float(getattr(config.training, "focal_weight", 0.0))
    f_gamma = float(getattr(config.training, "focal_gamma", 2.0))
    f_alpha = float(getattr(config.training, "focal_alpha", -1.0))

    # Distillation heavy
    lam_base = float(getattr(config.training, "lambda_distill_3d", 0.0))
    heavy_mult = float(getattr(config.training, "distill_heavy_mult", 1.0))
    lam = lam_base * heavy_mult
    use_teacher = teacher is not None and lam > 0.0
    tau = float(getattr(config.training, "distill_temperature", 1.0))

    # Clip controls
    posw_max = float(getattr(config.training, "pos_weight_max", 8.0))
    logit_clip = float(getattr(config.training, "logit_clip", 12.0))
    frame_clip = float(getattr(config.training, "frame_loss_clip", 5.0))

    # Nonprompt scaling: fixed, single config value (no override)
    nonprompt_sf = float(getattr(config.training, "nonprompt_scale", 1.0))

    imgs_seq = batch["image"]  # [B,T,C,H,W]
    masks_seq = batch["mask"]  # [B,T,1,H,W]
    B, T, C, H, W = imgs_seq.shape

    raw_pt = batch["pt_list"]
    raw_lb = batch["p_label"]
    raw_box = batch["bbox"]
    raw_mpr = batch.get("m_prompt", None)

    student.train()

    if use_teacher:
        teacher.eval()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    def _zero_like_param():
        """
        Return a graph-connected zero scalar.

        Args:
            None.

        Returns:
            torch.Tensor: Graph-connected zero scalar.
        """
        try:
            return next(student.parameters()).sum() * 0.0
        except StopIteration:
            return imgs_seq.sum() * 0.0

    def _has_any_prompt(i, b):
        """
        Return whether a frame has any prompt.

        Args:
            i (int): Sequence frame index.
            b (int): Batch sample index.

        Returns:
            bool: Whether point, box, or mask input is present.
        """
        pts = raw_pt[i][b]
        bx = raw_box[i][b]
        mp = raw_mpr[i][b] if raw_mpr is not None else None
        has_pts = pts is not None and hasattr(pts, "numel") and pts.numel() > 0
        has_box = bx is not None and hasattr(bx, "numel") and bx.numel() == 4
        has_msk = mp is not None and hasattr(mp, "numel") and mp.numel() > 0

        return has_pts or has_box or has_msk

    seq_losses = []
    seq_prompt_bucket = []
    seq_np_bucket = []
    seq_np_base = []

    for b in range(B):
        imgs = imgs_seq[b].to(device=device, non_blocking=True)  # [T,C,H,W]
        gts = masks_seq[b].to(device=device, non_blocking=True)  # [T,1,H,W]

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        # Init states
        train_state = student.init_state_from_tensor(
            imgs_tensor=imgs.contiguous(), mode="train"
        )
        train_state["storage_device"] = torch.device(device)

        teacher_state = None

        if use_teacher:
            with torch.no_grad():
                teacher_state = teacher.init_state_from_tensor(
                    imgs_tensor=imgs.contiguous(), mode="eval"
                )
                teacher_state["storage_device"] = torch.device(device)

        # Base prompt frames from dataset ONLY
        prompt_idxs = [i for i in range(T) if _has_any_prompt(i, b)]
        prompt_idxs = sorted(set(prompt_idxs))

        if not prompt_idxs:
            z = _zero_like_param()
            seq_losses.append(z)
            seq_prompt_bucket.append(z)
            seq_np_bucket.append(None)
            seq_np_base.append(None)
            continue

        # ---- Inject prompts (dataset prompts only) ----
        for t in prompt_idxs:
            pts = raw_pt[t][b]
            lbs = raw_lb[t][b]
            bx = raw_box[t][b]
            mp = raw_mpr[t][b] if raw_mpr is not None else None

            has_pts = pts is not None and hasattr(pts, "numel") and pts.numel() > 0
            has_box = bx is not None and hasattr(bx, "numel") and bx.numel() == 4
            has_msk = mp is not None and hasattr(mp, "numel") and mp.numel() > 0

            if has_pts:
                p = pts.to(device).float()
                l = lbs.to(device).long()
                student.train_add_new_points_or_box(
                    train_state, t, 0, points=p, labels=l, clear_old_points=False
                )

                if teacher_state is not None:
                    with torch.no_grad():
                        teacher.add_new_points_or_box(
                            teacher_state,
                            t,
                            0,
                            points=p,
                            labels=l,
                            clear_old_points=False,
                        )

            elif has_box:
                bb = bx.to(device).to(torch.int64).reshape(4)
                ep = torch.empty((1, 0, 2), device=device, dtype=torch.float32)
                el = torch.empty((1, 0), device=device, dtype=torch.int64)
                student.train_add_new_points_or_box(
                    train_state,
                    t,
                    0,
                    points=ep,
                    labels=el,
                    box=bb,
                    clear_old_points=True,
                )

                if teacher_state is not None:
                    with torch.no_grad():
                        teacher.add_new_points_or_box(
                            teacher_state,
                            t,
                            0,
                            points=ep,
                            labels=el,
                            box=bb,
                            clear_old_points=True,
                        )

            elif has_msk:
                mm = mp.to(device).float()
                mm = (
                    F.interpolate(mm.unsqueeze(0), size=(H, W), mode="nearest")
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )
                student.train_add_new_mask(train_state, t, 0, mm)

                if teacher_state is not None:
                    with torch.no_grad():
                        teacher.add_new_mask(teacher_state, t, 0, mm)

        # ---- Propagate ----
        s_preds = {}
        start = min(prompt_idxs)

        for ti, _, mlog in student.train_propagate_in_video(
            train_state, start_frame_idx=start
        ):
            s_preds[ti] = mlog[0]

        t_preds = {}
        if teacher_state is not None:
            with torch.no_grad():
                for ti, _, mlog in teacher.propagate_in_video(
                    teacher_state, start_frame_idx=start
                ):
                    t_preds[ti] = mlog[0]

        frames = sorted(s_preds.keys())
        if len(frames) == 0:
            z = _zero_like_param()
            seq_losses.append(z)
            seq_prompt_bucket.append(z)
            seq_np_bucket.append(None)
            seq_np_base.append(None)
            continue

        prompt_set = set(prompt_idxs)
        p_losses = []
        np_losses = []

        base_zero = _zero_like_param()

        for ti in frames:
            slog = s_preds[ti]
            if slog.dim() == 2:
                slog = slog.unsqueeze(0).unsqueeze(0)
            elif slog.dim() == 3:
                slog = slog.unsqueeze(0)
            elif slog.dim() == 4:
                pass
            else:
                continue

            slog = slog.clamp(-logit_clip, logit_clip)

            gt = gts[ti].float().unsqueeze(0)

            stud_up = F.interpolate(
                slog, size=(out_size, out_size), mode="bilinear", align_corners=False
            )
            gt_up = F.interpolate(gt, size=(out_size, out_size), mode="nearest")

            with torch.amp.autocast("cuda", enabled=False):
                sp = stud_up.float()
                gm = gt_up.float()

                pos = gm.flatten(1).sum(1)
                tot = torch.full_like(pos, gm[0].numel(), dtype=gm.dtype)
                neg = tot - pos
                dyn_pw = ((neg + 1.0) / (pos + 1.0)).clamp(1.0, posw_max)

                bce_raw = F.binary_cross_entropy_with_logits(sp, gm, reduction="none")
                pixel_w = 1.0 + (dyn_pw.view(1, 1, 1, 1) - 1.0) * gm
                bce = (bce_raw * pixel_w).mean()

                tv = tversky_loss_from_logits(sp, gm, alpha=0.15, beta=0.85, smooth=1.0)
                foc = binary_focal_loss_with_logits(
                    sp, gm, gamma=f_gamma, alpha=float(f_alpha)
                )

                seg_loss = (bce_w * bce) + (dice_w * tv) + (focal_w * foc)
                seg_loss = seg_loss.clamp_max(frame_clip)

            dis_loss = base_zero
            if teacher_state is not None and (ti in t_preds):
                tlog = t_preds[ti]
                if tlog.dim() == 2:
                    tlog = tlog.unsqueeze(0).unsqueeze(0)
                elif tlog.dim() == 3:
                    tlog = tlog.unsqueeze(0)

                tlog = tlog.clamp(-logit_clip, logit_clip)
                t_up = F.interpolate(
                    tlog,
                    size=(out_size, out_size),
                    mode="bilinear",
                    align_corners=False,
                )

                with torch.amp.autocast("cuda", enabled=False):
                    tp = torch.sigmoid(t_up.float() / tau).clamp(1e-4, 1.0 - 1e-4)
                    dis_loss = F.binary_cross_entropy_with_logits(
                        stud_up.float(), tp, reduction="mean"
                    )
                    dis_loss = dis_loss.clamp_max(frame_clip)

            total_frame = seg_loss + lam * dis_loss

            if ti in prompt_set:
                p_losses.append(total_frame)
            else:
                np_losses.append(total_frame)

        if len(p_losses) == 0 and len(np_losses) == 0:
            z = _zero_like_param()
            seq_losses.append(z)
            seq_prompt_bucket.append(z)
            seq_np_bucket.append(None)
            seq_np_base.append(None)
            continue

        prompt_bucket = torch.stack(p_losses).mean() if len(p_losses) > 0 else base_zero
        np_bucket_base = torch.stack(np_losses).mean() if len(np_losses) > 0 else None
        np_bucket_scaled = (
            (np_bucket_base * nonprompt_sf) if np_bucket_base is not None else None
        )

        total = base_zero

        if np_bucket_scaled is None:
            total = prompt_bucket
        elif len(p_losses) == 0:
            total = np_bucket_scaled
        else:
            total = 0.5 * prompt_bucket + 0.5 * np_bucket_scaled

        seq_losses.append(total)
        seq_prompt_bucket.append(prompt_bucket)
        seq_np_bucket.append(np_bucket_scaled)
        seq_np_base.append(np_bucket_base)

    loss = torch.stack([x for x in seq_losses]).mean()

    prompt_avg_t = (
        torch.stack([x for x in seq_prompt_bucket]).mean()
        if len(seq_prompt_bucket)
        else None
    )
    np_avg_t = (
        torch.stack([x for x in [x for x in seq_np_bucket if x is not None]]).mean()
        if any(x is not None for x in seq_np_bucket)
        else None
    )
    np_base_avg_t = (
        torch.stack([x for x in [x for x in seq_np_base if x is not None]]).mean()
        if any(x is not None for x in seq_np_base)
        else None
    )

    bad = []

    if not torch.isfinite(loss.detach()).all():
        bad.append(("loss", safe_float(loss)))
    if prompt_avg_t is not None and not torch.isfinite(prompt_avg_t.detach()).all():
        bad.append(("prompt_loss", safe_float(prompt_avg_t)))
    if np_avg_t is not None and not torch.isfinite(np_avg_t.detach()).all():
        bad.append(("non_prompt_loss", safe_float(np_avg_t)))
    if np_base_avg_t is not None and not torch.isfinite(np_base_avg_t.detach()).all():
        bad.append(("non_prompt_loss_base", safe_float(np_base_avg_t)))

    if bad:
        optimizer.zero_grad(set_to_none=True)
        return {
            "ok": False,
            "skip_reason": "nonfinite_3d_loss",
            "loss": safe_float(loss),
            "prompt_loss": safe_float(prompt_avg_t),
            "non_prompt_loss": safe_float(np_avg_t),
            "non_prompt_loss_base": safe_float(np_base_avg_t),
            "bad_losses": bad,
        }

    loss.backward()
    if synchronize:
        synchronize_gradients(student)

    try:
        if clip_gradients:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), 1.0, error_if_nonfinite=True
            )
        else:
            grad_norm = torch.zeros((), device=device)
    except RuntimeError as e:
        bad = []
        max_items = 50
        for name, p in student.named_parameters():
            if p.grad is None:
                continue

            g = p.grad.detach()
            if g.numel() == 0:
                continue
            if not torch.isfinite(g).all():
                finite = g[torch.isfinite(g)]
                bad.append(
                    {
                        "name": name,
                        "shape": tuple(g.shape),
                        "grad_absmax_finite": (
                            float(finite.abs().max().item()) if finite.numel() else None
                        ),
                        "grad_min_finite": (
                            float(finite.min().item()) if finite.numel() else None
                        ),
                        "grad_max_finite": (
                            float(finite.max().item()) if finite.numel() else None
                        ),
                    }
                )
                if len(bad) >= max_items:
                    break

        optimizer.zero_grad(set_to_none=True)

        return {
            "ok": False,
            "skip_reason": "nonfinite_3d_grad_norm",
            "loss": safe_float(loss),
            "prompt_loss": safe_float(prompt_avg_t),
            "non_prompt_loss": safe_float(np_avg_t),
            "non_prompt_loss_base": safe_float(np_base_avg_t),
            "exception": repr(e),
            "bad_grad_params": bad,
        }

    if optimizer_step:
        optimizer.step()

    def _avg_or_none(xs):
        """
        Average non-None tensors as a float.

        Args:
            xs (list[torch.Tensor]): Scalar loss tensors to average.

        Returns:
            float | None: Mean scalar value, or None for an empty list.
        """
        xs = [x for x in xs if x is not None]

        return float(torch.stack(xs).mean().detach().item()) if xs else None

    prompt_avg = (
        float(torch.stack([x for x in seq_prompt_bucket]).mean().detach().item())
        if len(seq_prompt_bucket)
        else 0.0
    )
    np_avg = _avg_or_none(seq_np_bucket)
    np_base_avg = _avg_or_none(seq_np_base)

    return {
        "ok": True,
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm.detach().item()),
        "prompt_loss": float(prompt_avg),
        "non_prompt_loss": np_avg,
        "non_prompt_loss_base": np_base_avg,
    }


_FRAME_MAJOR_BATCH_KEYS = frozenset({"pt_list", "p_label", "bbox", "m_prompt"})


def split_3d_batch_by_volume(batch: dict, sample: int) -> dict:
    """
    Select one volume while preserving the frame-major prompt layout.

    Args:
        batch (dict): Collated images, masks, prompts, and sample metadata.
        sample (int): Batch index of the volume to retain.

    Returns:
        dict: Single-volume batch with aligned tensor and prompt fields.
    """
    batch_size = int(batch["image"].shape[0])
    if not 0 <= int(sample) < batch_size:
        raise IndexError(f"sample index {sample} is outside batch size {batch_size}")

    result = {}
    for key, value in batch.items():
        if key in _FRAME_MAJOR_BATCH_KEYS and value is not None:
            result[key] = [[frame[sample]] for frame in value]
        elif (
            torch.is_tensor(value)
            and value.ndim > 0
            and int(value.shape[0]) == batch_size
        ):
            result[key] = value[sample : sample + 1]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = [value[sample]]
        else:
            result[key] = value

    return result


def _average_output(outputs: list[dict], key: str):
    """
    Average the defined values of one output field across volumes.

    Args:
        outputs (list[dict]): Per-volume training outputs whose field values are averaged.
        key (str): Output field whose defined values are averaged.

    Returns:
        float | None: Mean output value, or None when the field has no defined values.
    """
    values = [output[key] for output in outputs if output.get(key) is not None]

    return float(sum(values) / len(values)) if values else None


def _bad_gradient_parameters(student, max_items: int = 50) -> list[dict]:
    """
    Collect parameters with nonfinite gradients for failure diagnostics.

    Args:
        student (SAM2VideoPredictor): Student predictor being optimized.
        max_items (int): Maximum number of problematic parameters included in diagnostics.

    Returns:
        list[dict]: Parameter names and gradient diagnostics, limited to max_items entries.
    """
    bad = []
    for name, parameter in student.named_parameters():
        if parameter.grad is None or parameter.grad.numel() == 0:
            continue

        gradient = parameter.grad.detach()
        if torch.isfinite(gradient).all():
            continue

        finite = gradient[torch.isfinite(gradient)]
        bad.append(
            {
                "name": name,
                "shape": tuple(gradient.shape),
                "grad_absmax_finite": (
                    float(finite.abs().max().item()) if finite.numel() else None
                ),
                "grad_min_finite": (
                    float(finite.min().item()) if finite.numel() else None
                ),
                "grad_max_finite": (
                    float(finite.max().item()) if finite.numel() else None
                ),
            }
        )
        if len(bad) >= max_items:
            break

    return bad


def train_step_3d_per_volume(
    student, teacher, optimizer, batch, config, scaler
) -> dict:
    """
    Run one logical batch while releasing each volume graph after backward.

    Loss averaging, distributed synchronization, gradient clipping, and the
    optimizer step match :func:`train_step_3d`; only the execution schedule
    changes to reduce peak activation memory.

    Args:
        student (SAM2VideoPredictor): Student predictor being optimized.
        teacher (SAM2VideoPredictor | None): Optional teacher predictor used for
            distillation.
        optimizer (torch.optim.Optimizer): Optimizer for the trainable parameters.
        batch (dict): Collated images, masks, prompts, and sample metadata.
        config (OmegaConf.DictConfig): Model, dataset, and training configuration.
        scaler (object | None): Optional mixed-precision gradient scaler.

    Returns:
        dict: Aggregate losses, gradient norm, and success or failure diagnostics.
    """
    batch_size = int(batch["image"].shape[0])
    if batch_size < 1:
        raise ValueError("3D training batches must contain at least one volume")

    optimizer.zero_grad(set_to_none=True)
    outputs = []
    for sample in range(batch_size):
        small_batch = split_3d_batch_by_volume(batch, sample)
        output = train_step_3d(
            student,
            teacher,
            optimizer,
            small_batch,
            config,
            scaler,
            zero_grad=False,
            synchronize=False,
            clip_gradients=False,
            optimizer_step=False,
        )
        del small_batch

        if not output.get("ok"):
            optimizer.zero_grad(set_to_none=True)
            return output

        outputs.append(output)

    for parameter in student.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(batch_size)

    synchronize_gradients(student)

    loss_values = {
        "loss": _average_output(outputs, "loss"),
        "prompt_loss": _average_output(outputs, "prompt_loss"),
        "non_prompt_loss": _average_output(outputs, "non_prompt_loss"),
        "non_prompt_loss_base": _average_output(outputs, "non_prompt_loss_base"),
    }
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=True
        )
    except RuntimeError as error:
        bad = _bad_gradient_parameters(student)
        optimizer.zero_grad(set_to_none=True)

        return {
            "ok": False,
            "skip_reason": "nonfinite_3d_grad_norm",
            **loss_values,
            "exception": repr(error),
            "bad_grad_params": bad,
        }

    optimizer.step()
    return {
        "ok": True,
        **loss_values,
        "grad_norm": float(grad_norm.detach().item()),
    }


@torch.inference_mode()
def validate_step_3d(
    predictor,
    video_tchw: torch.Tensor,
    gt_thw: torch.Tensor,
    *,
    prompt_cache: dict,
    prompt_mode: str,
    normalize_coords: bool,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    output_device: str | torch.device | None = None,
):
    """
    Validate a single 3D volume represented as a sequence of slices.

    Args:
        predictor (SAM2VideoPredictor): Predictor in evaluation mode.
        video_tchw (torch.Tensor): Sequence tensor with shape ``[T, C, H, W]``.
        gt_thw (torch.Tensor): Ground-truth masks with shape ``[T, H, W]``.
        prompt_cache (dict): Dataset-produced prompt cache.
        prompt_mode (str): Prompt mode, one of point, box, or mask.
        normalize_coords (bool): Whether SAM2 should normalize point/box coordinates.
        offload_video_to_cpu (bool): Keep normalized input frames on CPU and transfer
            them to the image encoder on demand.
        offload_state_to_cpu (bool): Store compact mask-memory outputs on CPU between
            propagation steps.
        output_device (str | torch.device | None): Device on which to accumulate the
            returned full-volume
            logits. ``None`` keeps the historical behavior and uses the model
            device; ``"cpu"`` avoids retaining one full-resolution GPU tensor
            per frame.

    Returns:
        dict | None: Logits and prompt metadata, or None if no prompt is available.
    """
    device = predictor.device
    use_amp = device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else torch.no_grad()
    )

    if prompt_cache is None:
        return None

    mode_key = str(prompt_mode).strip().lower()
    if mode_key in ("click", "clicks", "point", "points"):
        mode_key = "point"
    elif mode_key in ("box", "bbox", "boxes"):
        mode_key = "box"
    elif mode_key in ("mask", "masks"):
        mode_key = "mask"
    else:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode}")

    mode_payload = prompt_cache.get(mode_key, None)
    if mode_payload is None:
        return None

    chosen = [int(x) for x in prompt_cache.get("chosen_frames", [])]
    forced_first = bool(prompt_cache.get("forced_first", False))
    T = int(video_tchw.shape[0])

    state = predictor.init_state_from_tensor(
        imgs_tensor=video_tchw,
        offload_video_to_cpu=bool(offload_video_to_cpu),
        offload_state_to_cpu=bool(offload_state_to_cpu),
        mode="eval",
    )

    pk_counts = {"point": 0, "box": 0, "mask": 0}
    prompts_added = 0

    pt_list = mode_payload.get("pt_list", [])
    p_label = mode_payload.get("p_label", [])
    bbox = mode_payload.get("bbox", [])
    m_prompt = mode_payload.get("m_prompt", [])

    for t in chosen:
        if t < 0 or t >= T:
            continue

        if mode_key == "point":
            pts = pt_list[t] if t < len(pt_list) else None
            lbs = p_label[t] if t < len(p_label) else None
            if pts is None or lbs is None or pts.numel() == 0 or lbs.numel() == 0:
                continue

            pts_f = pts.to(device=device, dtype=torch.float32).unsqueeze(0)
            lbs_i = lbs.to(device=device, dtype=torch.int32).unsqueeze(0)

            with amp_ctx:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(t),
                    obj_id=1,
                    points=pts_f,
                    labels=lbs_i,
                    box=None,
                    clear_old_points=False,
                    normalize_coords=bool(normalize_coords),
                )

            pk_counts["point"] += int(pts.shape[0])
            prompts_added += 1

        elif mode_key == "box":
            bb = bbox[t] if t < len(bbox) else None
            if bb is None or (hasattr(bb, "numel") and bb.numel() == 0):
                continue

            if bb.ndim == 2:
                bb = bb[0]

            bb = bb.to(device=device, dtype=torch.float32)

            empty_points = torch.empty((1, 0, 2), dtype=torch.float32, device=device)
            empty_labels = torch.empty((1, 0), dtype=torch.int32, device=device)

            with amp_ctx:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(t),
                    obj_id=1,
                    points=empty_points,
                    labels=empty_labels,
                    box=bb,
                    clear_old_points=True,
                    normalize_coords=bool(normalize_coords),
                )

            pk_counts["box"] += 1
            prompts_added += 1

        else:  # mask
            mm = m_prompt[t] if t < len(m_prompt) else None
            if mm is None or (hasattr(mm, "numel") and mm.numel() == 0):
                continue

            if mm.ndim == 3 and mm.shape[0] == 1:
                mm = mm[0]

            mm = (mm > 0).to(device=device, dtype=torch.bool)

            with amp_ctx:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(t),
                    obj_id=1,
                    mask=mm,
                )

            pk_counts["mask"] += 1
            prompts_added += 1

    if prompts_added == 0:
        return None

    _, _, H, W = video_tchw.shape
    logits_device = device if output_device is None else torch.device(output_device)
    logits = torch.zeros((T, H, W), device=logits_device, dtype=torch.float32)

    with amp_ctx:
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False
        ):
            tt = int(frame_idx)
            frame_logits = video_res_masks[0, 0].float()
            if frame_logits.device != logits_device:
                # The destination accumulator is ordinary CPU memory. A
                # non-blocking device-to-host copy could be read by the next
                # CPU assignment before the transfer completes.
                frame_logits = frame_logits.to(logits_device, non_blocking=False)

            logits[tt] = frame_logits

    if device.type == "cuda" and logits_device.type == "cpu":
        # Complete any non-blocking device-to-host copies before returning the
        # output or releasing the inference state.
        torch.cuda.synchronize(device)

    return {
        "logits": logits,
        "chosen_frames": chosen,
        "forced_first": forced_first,
        "pk_counts": pk_counts,
    }
