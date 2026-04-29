# Per-batch 2D training and validation steps for RWKV-MedSAM2.
#
# Includes input normalization, mixed prompt encoding, student/teacher
# distillation training, and prompt-cache validation.

import math
import torch
import torch.nn.functional as F

from .func_metrics import (
    tversky_loss_from_logits,
    binary_focal_loss_with_logits,
    safe_float,
)

def normalize_sam2_input(x: torch.Tensor, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Normalize raw 2D inputs to the ImageNet scale expected by SAM2.

    Args:
        x (torch.Tensor): Input tensor of shape ``[B, C, H, W]``.
        img_mean (tuple[float, float, float]): Channel means.
        img_std (tuple[float, float, float]): Channel standard deviations.

    Returns:
        torch.Tensor: Normalized three-channel tensor.
    """
    x = x.to(torch.float32)

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    vmin = float(x.amin().item())
    vmax = float(x.amax().item())

    def _robust_01(t: torch.Tensor, qlo=0.01, qhi=0.99):
        """Robustly rescale a tensor to [0, 1]."""
        flat = t.reshape(-1)
        qs = torch.quantile(flat, torch.tensor([qlo, qhi], device=flat.device, dtype=flat.dtype))
        lo, hi = qs[0], qs[1]
        if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or float((hi - lo).item()) <= 1e-6:
            return torch.zeros_like(t)
        return ((t - lo) / (hi - lo)).clamp_(0.0, 1.0)

    if (-6.5 <= vmin <= 0.0) and (0.0 <= vmax <= 6.5):
        x = _robust_01(x) * 255.0
    elif 0.0 <= vmin and vmax <= 1.5:
        x = x * 255.0
    else:
        x = x.clamp_(0.0, 255.0)

    mean = torch.tensor(img_mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(img_std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)

    x = x / 255.0
    x = (x - mean) / std
    return x

def train_step_2d(student, teacher, optimizer, batch, config, scaler):
    """
    Run one 2D student training step with optional teacher distillation.

    Args:
        student (SAM2VideoPredictor): Student predictor to optimize.
        teacher (SAM2VideoPredictor | None): Optional teacher predictor.
        optimizer (torch.optim.Optimizer): Optimizer for the student.
        batch (dict): Collated single-frame 2D batch.
        config (OmegaConf.DictConfig): Training configuration.
        scaler (object | None): AMP scaler placeholder; bf16 autocast is used directly.

    Returns:
        dict: Step status and loss values, including skip diagnostics on failure.
    """
    device     = config.training.device
    feat_sizes = config.decoder.feat_sizes
    out_size   = int(config.training.out_size)

    # Loss weights
    bce_w    = float(getattr(config.training, "bce_weight",   1.0))
    dice_w   = float(getattr(config.training, "dice_weight",  1.0))
    focal_w  = float(getattr(config.training, "focal_weight", 0.0))
    f_gamma  = float(getattr(config.training, "focal_gamma",  2.0))
    f_alpha  = float(getattr(config.training, "focal_alpha", -1.0))

    # Distillation
    lam_base   = float(getattr(config.training, "lambda_distill_2d", 0.0))
    heavy_mult = float(getattr(config.training, "distill_heavy_mult", 1.0))
    lam        = lam_base * heavy_mult
    use_teacher = teacher is not None and lam > 0.0
    tau        = float(getattr(config.training, "distill_temperature", 1.0))
    logit_clip = float(getattr(config.training, "logit_clip", 12.0))

    imgs_all = batch["image"].to(device=device, non_blocking=True)  # [B,1,C,H,W]
    msks_all = batch["mask"].to(device=device, non_blocking=True)   # [B,1,1,H,W]
    B, T, C, H, W = imgs_all.shape
    if T != 1:
        raise ValueError(f"2D training expects single-frame sequences, got T={T}")

    raw_pts = batch["pt_list"]
    raw_lbl = batch["p_label"]
    raw_box = batch["bbox"]
    raw_mpr = batch.get("m_prompt", None)

    student.train()
    optimizer.zero_grad(set_to_none=True)

    pe_student = student.sam_prompt_encoder.get_dense_pe()
    pe_teacher = None
    if use_teacher:
        teacher.eval()
        pe_teacher = teacher.sam_prompt_encoder.get_dense_pe()

    imgs_t = imgs_all[:, 0]
    msks_t = msks_all[:, 0]

    imgs_t = normalize_sam2_input(imgs_t)

    pts_all = raw_pts[0]
    lbs_all = raw_lbl[0]
    box_all = raw_box[0]
    msk_all = (raw_mpr[0] if raw_mpr is not None else [None] * B)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        student_sparse, student_dense, _ = _encode_mixed_prompts(
            student.sam_prompt_encoder, pts_all, lbs_all, box_all, msk_all, imgs_t, device
        )

        bb_out = student.forward_image(imgs_t)
        _, vision_feats, _vision_pos, _ = student._prepare_backbone_features(bb_out)

        feats = [
            feat.permute(1, 2, 0).reshape(B, -1, *size)
            for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        hires_feats = feats[:-1]

        student_dense = F.interpolate(
            student_dense, size=image_embed.shape[-2:], mode="bilinear", align_corners=False
        )
        image_pe = pe_student
        if image_pe.shape[-2:] != image_embed.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=image_embed.shape[-2:], mode="bilinear", align_corners=False)
        image_pe = image_pe.to(image_embed.dtype)

        student_logits, _student_iou, _, _student_obj_score_logits = student.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=student_sparse,
            dense_prompt_embeddings=student_dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hires_feats,
        )

        student_logits = student_logits.clamp(-logit_clip, logit_clip)
        stud_up = F.interpolate(student_logits, size=(out_size, out_size), mode="bilinear", align_corners=False)
        gt_up   = F.interpolate(msks_t.float(),    size=(out_size, out_size), mode="nearest")

    with torch.amp.autocast("cuda", enabled=False):
        gm = gt_up.float()
        sp = stud_up.float()

        pos = gm.flatten(1).sum(1)
        tot = torch.full_like(pos, gm[0].numel(), dtype=gm.dtype)
        neg = tot - pos
        dyn_pw = ((neg + 1.0) / (pos + 1.0)).clamp(1.0, float(getattr(config.training, "pos_weight_max", 8.0)))
        dyn_pw_pix = dyn_pw.view(B, 1, 1, 1)

        bce_raw = F.binary_cross_entropy_with_logits(sp, gm, reduction="none")
        pixel_w = 1.0 + (dyn_pw_pix - 1.0) * gm
        bce = (bce_raw * pixel_w).mean()

        tv  = tversky_loss_from_logits(sp, gm, alpha=0.15, beta=0.85, smooth=1.0)
        foc = binary_focal_loss_with_logits(sp, gm, gamma=f_gamma, alpha=f_alpha)

        seg_loss = (bce_w * bce) + (dice_w * tv) + (focal_w * foc)

    if use_teacher:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_bb = teacher.forward_image(imgs_t)
                _, t_feats, _, t_sizes = teacher._prepare_backbone_features(t_bb)

                t_list = [
                    feat.permute(1, 2, 0).reshape(B, -1, *sz)
                    for feat, sz in zip(t_feats[::-1], t_sizes[::-1])
                ][::-1]
                t_embed = t_list[-1]

                t_sparse, t_dense, _ = _encode_mixed_prompts(
                    teacher.sam_prompt_encoder, pts_all, lbs_all, box_all, msk_all, imgs_t, device
                )
                t_dense = F.interpolate(t_dense, size=t_embed.shape[-2:], mode="bilinear", align_corners=False)

                t_pe = pe_teacher if pe_teacher is not None else teacher.sam_prompt_encoder.get_dense_pe()
                if t_pe.shape[-2:] != t_embed.shape[-2:]:
                    t_pe = F.interpolate(t_pe, size=t_embed.shape[-2:], mode="bilinear", align_corners=False)
                t_pe = t_pe.to(t_embed.dtype)

                t_logits, *_ = teacher.sam_mask_decoder(
                    image_embeddings=t_embed,
                    image_pe=t_pe,
                    sparse_prompt_embeddings=t_sparse,
                    dense_prompt_embeddings=t_dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=t_list[:-1],
                )
                t_logits = t_logits.clamp(-logit_clip, logit_clip)
                t_up = F.interpolate(t_logits, size=(out_size, out_size), mode="bilinear", align_corners=False)

        with torch.amp.autocast("cuda", enabled=False):
            tp = torch.sigmoid(t_up.float() / tau).clamp(1e-4, 1.0 - 1e-4)
            dis = F.binary_cross_entropy_with_logits(stud_up.float(), tp, reduction="mean")

        loss = seg_loss + lam * dis
    else:
        loss = seg_loss
    loss_value = safe_float(loss)

    if not torch.isfinite(loss.detach()).all():
        optimizer.zero_grad(set_to_none=True)
        return {
            "ok": False,
            "skip_reason": "nonfinite_2d_loss",
            "loss": loss_value,
        }

    loss.backward()

    try:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
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
                bad.append({
                    "name": name,
                    "shape": tuple(g.shape),
                    "grad_absmax_finite": float(finite.abs().max().item()) if finite.numel() else None,
                    "grad_min_finite": float(finite.min().item()) if finite.numel() else None,
                    "grad_max_finite": float(finite.max().item()) if finite.numel() else None,
                })
                if len(bad) >= max_items:
                    break

        optimizer.zero_grad(set_to_none=True)

        return {
            "ok": False,
            "skip_reason": "nonfinite_2d_grad_norm",
            "loss": loss_value,
            "exception": repr(e),
            "bad_grad_params": bad,
        }

    optimizer.step()

    return {
        "ok": True,
        "loss": float(loss.detach().item()),
    }

def _encode_mixed_prompts(prompter, pts_all, lbs_all, box_all, msk_all, imgs, device):
    """
    Encode mixed prompts per-subset (points/boxes/masks/none) and stitch to batch order.
    Returns (sparse_full, dense_full, image_pe)

    Args:
        prompter (PromptEncoder): Prompt encoder for encoding prompts.
        pts_all (list): List of point prompts for each image in the batch.
        lbs_all (list): List of label prompts for each image in the batch.
        box_all (list): List of box prompts for each image in the batch.
        msk_all (list): List of mask prompts for each image in the batch.
        imgs (torch.Tensor): Batch of images.
        device (torch.device): Device to encode prompts on.

    Returns:
        tuple: Tuple containing:
            sparse_full (torch.Tensor): Encoded sparse prompts for the batch.
            dense_full (torch.Tensor): Encoded dense prompts for the batch.
            image_pe (torch.Tensor): Image positional encoding for the batch.
    """
    batch_size = imgs.shape[0]

    # Use the prompt encoder's PE size as the canonical dense spatial size
    pe = prompter.get_dense_pe()
    Hd, Wd = int(pe.shape[-2]), int(pe.shape[-1])

    # Partition indices
    idx_pts, idx_box, idx_mask, idx_none = [], [], [], []
    for i, (p, b, m) in enumerate(zip(pts_all, box_all, msk_all)):
        has_p = (p is not None and hasattr(p, "numel") and p.numel() > 0)
        has_b = (b is not None and hasattr(b, "numel") and b.numel() == 4)
        has_m = (m is not None and hasattr(m, "numel") and m.numel() > 0)
        if has_p:   idx_pts.append(i)
        elif has_b: idx_box.append(i)
        elif has_m: idx_mask.append(i)
        else:       idx_none.append(i)

    # Gather helpers
    def _gather_points(idxs):
        """Gather and pad point prompts."""
        if not idxs: return None
        # Variable num clicks -> pad to Nmax with label -1
        Ns = [int(pts_all[i].shape[0]) for i in idxs]
        Nmax = max(Ns)  # >=1 because has_p checked above
        pts_pad, lbs_pad = [], []
        for i, n in zip(idxs, Ns):
            p = pts_all[i].to(device).to(torch.float32)
            if p.ndim == 1:
                p = p.reshape(-1, 2)
            l = lbs_all[i].to(device).to(torch.long).reshape(-1)
            if n < Nmax:
                pp = torch.zeros((Nmax, 2), device=device, dtype=torch.float32)
                ll = torch.full((Nmax,), -1, device=device, dtype=torch.long)
                if n > 0:
                    pp[:n] = p[:n]
                    ll[:n] = l[:n]
                pts_pad.append(pp)
                lbs_pad.append(ll)
            else:
                pts_pad.append(p[:Nmax])
                lbs_pad.append(l[:Nmax])
        pts = torch.stack(pts_pad, dim=0)  # [k, Nmax, 2]
        lbs = torch.stack(lbs_pad, dim=0)  # [k, Nmax]
        return (pts, lbs)

    def _gather_boxes(idxs):
        """Gather box prompts."""
        if not idxs: return None
        return torch.stack([box_all[i].to(device).to(torch.int64).reshape(4) for i in idxs], dim=0)  # [k,4]

    def _gather_masks(idxs):
        """Gather mask prompts."""
        if not idxs: return None
        return torch.stack([msk_all[i].to(device).to(torch.float32) for i in idxs], dim=0)       # [k,1,H,W]

    dense_chunks  = [None, None, None, None]   # in order: pts, box, mask, none
    sparse_chunks = [None, None, None, None]

    # Points
    if idx_pts:
        se, de = prompter(points=_gather_points(idx_pts), boxes=None, masks=None)
        if de.shape[-2:] != (Hd, Wd):
            de = F.interpolate(de, size=(Hd, Wd), mode='bilinear', align_corners=False)
        sparse_chunks[0] = se
        dense_chunks[0]  = de

    # Boxes
    if idx_box:
        se, de = prompter(points=None, boxes=_gather_boxes(idx_box), masks=None)
        if de.shape[-2:] != (Hd, Wd):
            de = F.interpolate(de, size=(Hd, Wd), mode='bilinear', align_corners=False)
        sparse_chunks[1] = se
        dense_chunks[1]  = de

    # Masks
    if idx_mask:
        se, de = prompter(points=None, boxes=None, masks=_gather_masks(idx_mask))
        if de.shape[-2:] != (Hd, Wd):
            de = F.interpolate(de, size=(Hd, Wd), mode='bilinear', align_corners=False)
        sparse_chunks[2] = se
        dense_chunks[2]  = de

    # None -> neutral zero mask to keep batch dims aligned
    if idx_none:
        _, _, H_img, W_img = imgs.shape
        zero_masks = torch.zeros((len(idx_none), 1, H_img, W_img), dtype=torch.float32, device=device)
        se, de = prompter(points=None, boxes=None, masks=zero_masks)
        if de.shape[-2:] != (Hd, Wd):
            de = F.interpolate(de, size=(Hd, Wd), mode='bilinear', align_corners=False)
        sparse_chunks[3] = se
        dense_chunks[3]  = de

    # Find a dense chunk to read channel/dtype (spatial is fixed to Hd/Wd above)
    any_dense = next(d for d in dense_chunks if d is not None)
    C_d = int(any_dense.shape[1])  # channels

    # Stitch dense -> [B*T, C_d, Hd, Wd]
    dense_full = torch.zeros((batch_size, C_d, Hd, Wd), dtype=any_dense.dtype, device=any_dense.device)
    if idx_pts:  dense_full[torch.tensor(idx_pts,  device=device)] = dense_chunks[0]
    if idx_box:  dense_full[torch.tensor(idx_box,  device=device)] = dense_chunks[1]
    if idx_mask: dense_full[torch.tensor(idx_mask, device=device)] = dense_chunks[2]
    if idx_none: dense_full[torch.tensor(idx_none, device=device)] = dense_chunks[3]

    # Stitch sparse: pad to Nmax across subsets that produced sparse tokens
    Ns, subset_maps = [], []
    for idxs, se_chunk in zip([idx_pts, idx_box, idx_mask, idx_none], sparse_chunks):
        if idxs and (se_chunk is not None):
            subset_maps.append((idxs, se_chunk))
            Ns.append(se_chunk.shape[1])  # N tokens

    if Ns:
        Nmax = max(Ns)
        C_s  = subset_maps[0][1].shape[-1]
        sparse_full = torch.zeros((batch_size, Nmax, C_s), dtype=subset_maps[0][1].dtype, device=device)
        for idxs, se_chunk in subset_maps:
            k, N, _ = se_chunk.shape
            sparse_full[torch.tensor(idxs, device=device), :N, :] = se_chunk
    else:
        sparse_full = None

    image_pe = None  # Caller pulls prompter.get_dense_pe()
    return sparse_full, dense_full, image_pe

@torch.inference_mode()
def validate_step_2d(
    predictor,
    video_tchw: torch.Tensor,
    gt_thw: torch.Tensor,
    *,
    prompt_cache: dict,
    prompt_mode: str,
    normalize_coords: bool,
):
    """
    Validate a single 2D sequence using cached prompts.

    Args:
        predictor (SAM2VideoPredictor): Predictor in evaluation mode.
        video_tchw (torch.Tensor): Sequence tensor with shape ``[T, C, H, W]``.
        gt_thw (torch.Tensor): Ground-truth masks with shape ``[T, H, W]``.
        prompt_cache (dict): Dataset-produced prompt cache.
        prompt_mode (str): Prompt mode, one of point, box, or mask.
        normalize_coords (bool): Whether SAM2 should normalize point/box coordinates.

    Returns:
        dict | None: Logits and prompt metadata, or None if no prompt is available.
    """
    device = predictor.device
    use_amp = (device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()

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
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
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
    logits = torch.zeros((T, H, W), device=device, dtype=torch.float32)

    with amp_ctx:
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False
        ):
            tt = int(frame_idx)
            logits[tt] = video_res_masks[0, 0].float()

    return {
        "logits": logits,
        "chosen_frames": chosen,
        "forced_first": forced_first,
        "pk_counts": pk_counts,
    }
