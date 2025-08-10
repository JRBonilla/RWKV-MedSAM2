# functions/func_3d.py
# Implements the per-batch 3D training and validation steps for
# student-teacher distillation in the RWKV-MedSAM2 model.

import torch
import torch.nn.functional as F
import random

from .func_metrics import compute_iou, compute_dice, compute_hd95

def train_step_3d(student, teacher, mask_decoder_opt, memory_opt, batch, config, scaler):
    """
    Single 3D (video) train step with teacher-student distillation
    using SAM2VideoPredictor's built-in stateful memory bank API.

    Args:
        student (SAM2VideoPredictor): The student model.
        teacher (SAM2VideoPredictor): The teacher model.
        mask_decoder_opt (torch.optim.Optimizer): The optimizer for the mask decoder.
        memory_opt (torch.optim.Optimizer): The optimizer for the memory encoder.
        batch (dict): A dictionary containing input tensors and labels, e.g.:
            'image': Tensor[T, 3, H, W],
            'mask':  Tensor[T, 1, H, W],
            'pt_list': list[T] of point-coordinate Tensors or None,
            'p_label': list[T] of label Tensors or None,
            'bbox': list[T] of bounding-box Tensors or None
        config (DictConfig): The configuration dictionary.
        scaler (torch.cuda.amp.GradScaler): The scaler.

    Returns:
        tuple:
            batch_loss (float): Combined loss over all frames.
            prompt_loss (float): Loss for frames with prompts.
            non_prompt_loss (Optional[float]): Loss for frames without prompts, or None.
    """
    # 0) Device and config
    device      = config.training.device
    out_size    = config.training.out_size
    pos_weight  = config.training.pos_weight
    alpha       = config.training.alpha
    max_prompts = config.prompt.max_per_seq
    image_size  = config.model.image_size

    # 1) Prepare video tensor and mask sequence
    imgs_seq = batch['image'].to(device) # [B,T,C,H,W]
    mask_seq = batch['mask'].to(device)  # [B,T,1,H,W]
    # Remove batch dimension -> [T,C,H,W] and [T,1,H,W]
    imgs     = imgs_seq.squeeze(0) # [T,C,H,W]
    mask_seq = mask_seq.squeeze(0) # [T,1,H,W]

    # 2) Initialize state for student and teacher
    if torch.cuda.get_device_properties(device).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    student.train()
    if mask_decoder_opt: mask_decoder_opt.zero_grad()
    if memory_opt: memory_opt.zero_grad()
    train_state = student.init_state_from_tensor(imgs_tensor=imgs, mode="train")
    train_state['storage_device'] = torch.device(device) # Ensure storage is on GPU

    teacher.eval()
    with torch.no_grad():
        teacher_state = teacher.init_state_from_tensor(imgs_tensor=imgs, mode="eval")
        teacher_state['storage_device'] = torch.device(device)

    # 3) Inject prompts into student and teacher
    # Get number of frames
    T = imgs.shape[0]

    # Helper function to check if a frame has any prompts
    def _has_any_prompt(i):
        pts = batch['pt_list'][i]
        bx  = batch['bbox'][i]
        mp  = batch.get('m_prompt', None)
        has_pts = pts is not None and hasattr(pts, 'numel') and pts.numel() > 0
        has_box = bx  is not None and hasattr(bx,  'numel') and bx.numel() == 4
        has_msk = (mp is not None) and ((mp[i].numel() > 0) if isinstance(mp, list) else mp.numel() > 0)
        return has_pts or has_box or has_msk

    # List of promptable frames
    promptable = [i for i in range(T) if _has_any_prompt(i)]

    prompt_idxs = []
    # Include first if promptable; otherwise can just force it below
    if 0 in promptable:
        prompt_idxs.append(0)

    # Add one more random promptable (distinct from first) if available
    others = [i for i in promptable if i != 0]
    if others and len(prompt_idxs) < config.prompt.max_per_seq:
        prompt_idxs.append(random.choice(others))

    # If no real prompts, force a GT-mask prompt on frame with most foreground
    forced_prompt = None
    if not prompt_idxs:
        fg_counts = mask_seq.reshape(T, -1).sum(dim=1)
        forced_prompt = int(torch.argmax(fg_counts).item())
        prompt_idxs = [forced_prompt]

    # Inject prompts (or forced GT prompt)
    for frame_idx in prompt_idxs:
        # Force GT prompt if available
        if forced_prompt is not None and frame_idx == forced_prompt and not _has_any_prompt(frame_idx):
            gt2d = mask_seq[frame_idx].detach().to(device).squeeze()
            if gt2d.dim() != 2:
                H, W = imgs.shape[-2], imgs.shape[-1]
                gt2d = gt2d.reshape(H, W)
            gt2d = gt2d.bool()
            student.train_add_new_mask(train_state, frame_idx, 0, gt2d)
            with torch.no_grad():
                teacher.add_new_mask(teacher_state, frame_idx, 0, gt2d)
            continue

        # Add prompt
        points = batch['pt_list'][frame_idx]
        labels = batch['p_label'][frame_idx]
        bbox   = batch['bbox'][frame_idx]
        mpr    = batch.get('m_prompt', None)
        mpr    = (mpr[frame_idx] if mpr is not None else None)

        # Check if this frame has any prompts
        has_points = points is not None and hasattr(points, 'numel') and points.numel() > 0
        has_bbox   = bbox   is not None and hasattr(bbox,   'numel') and bbox.numel() == 4
        has_mask   = mpr    is not None and hasattr(mpr,    'numel') and mpr.numel() > 0

        if has_points:
            student.train_add_new_points_or_box(train_state, frame_idx, 0,
                                                points=points.to(device).float(),
                                                labels=labels.to(device).long(),
                                                clear_old_points=False)
            with torch.no_grad():
                teacher.add_new_points_or_box(teacher_state, frame_idx, 0,
                                            points=points.to(device).float(),
                                            labels=labels.to(device).long(),
                                            clear_old_points=False)
        elif has_bbox:
            bbox_t = bbox.to(device).to(torch.int64).reshape(4)
            empty_pts = torch.empty((1, 0, 2), device=device, dtype=torch.float32)
            empty_lbl = torch.empty((1, 0),    device=device, dtype=torch.int64)
            student.train_add_new_points_or_box(train_state, frame_idx, 0,
                                                points=empty_pts, labels=empty_lbl,
                                                box=bbox_t, clear_old_points=True)
            with torch.no_grad():
                teacher.add_new_points_or_box(teacher_state, frame_idx, 0,
                                              points=empty_pts, labels=empty_lbl,
                                              box=bbox_t, clear_old_points=True)
        elif has_mask:
            mask_2d = mpr.detach()
            if mask_2d.dim() == 4: mask_2d = mask_2d.squeeze(0).squeeze(0)
            elif mask_2d.dim() == 3: mask_2d = mask_2d.squeeze(0)
            mask_t = mask_2d.to(device).to(torch.bool)
            student.train_add_new_mask(train_state, frame_idx, 0, mask_t)
            with torch.no_grad():
                teacher.add_new_mask(teacher_state, frame_idx, 0, mask_t)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # 5) Propagate through video
        student_preds = {}
        for frame_idx, obj_ids, mask_logits in student.train_propagate_in_video(train_state, start_frame_idx=0):
            # Mask logits: [num_objs, Hf, Wf]
            student_preds[frame_idx] = mask_logits[0].unsqueeze(0)

        teacher_preds = {}
        with torch.no_grad():
            for frame_idx, obj_ids, mask_logits in teacher.propagate_in_video(teacher_state, start_frame_idx=0):
                # Mask logits: [num_objs, Hf, Wf]
                teacher_preds[frame_idx] = mask_logits[0].unsqueeze(0)

        # 6) Compute loss
        p_seg_losses, p_dis_losses   = [], [] # Losses for frames with prompts
        np_seg_losses, np_dis_losses = [], [] # Losses for frames without prompts
        for frame_idx in range(T):
            student_logit = student_preds[frame_idx]            # [1,Hf,Wf]
            teacher_logit = teacher_preds[frame_idx]            # [1,Hf,Wf]

            # Resize to out_size
            student_pred = F.interpolate(student_logit, size=(out_size, out_size), mode='bilinear', align_corners=False) # [1,1,out_size,out_size]
            teacher_pred = F.interpolate(teacher_logit, size=(out_size, out_size), mode='bilinear', align_corners=False) # [1,1,out_size,out_size]

            gt_mask = mask_seq[frame_idx].float().unsqueeze(0)  # [1,1,H,W]

            with torch.no_grad():
                teacher_target = torch.sigmoid(teacher_pred)    # [!,1,H,W] in [0,1]

            frame_seg_loss = F.binary_cross_entropy_with_logits(student_pred, gt_mask, pos_weight=torch.tensor(pos_weight, device=device))
            frame_dis_loss = F.binary_cross_entropy_with_logits(student_pred, teacher_target, reduction='mean')

            if frame_idx in prompt_idxs:
                p_seg_losses.append(frame_seg_loss)
                p_dis_losses.append(frame_dis_loss)
            else:
                np_seg_losses.append(frame_seg_loss)
                np_dis_losses.append(frame_dis_loss)

        # 7) Combine and normalize each loss bucket
        avg_p_seg_loss = torch.stack(p_seg_losses).mean()
        avg_p_dis_loss = torch.stack(p_dis_losses).mean()
        prompt_loss    = alpha * avg_p_seg_loss + (1-alpha) * avg_p_dis_loss
        if len(np_seg_losses) > 0:
            avg_np_seg_loss = torch.stack(np_seg_losses).mean()
            avg_np_dis_loss = torch.stack(np_dis_losses).mean()
            non_prompt_loss = alpha * avg_np_seg_loss + (1-alpha) * avg_np_dis_loss
        else:
            non_prompt_loss = None

    # 8) Compute overall combined batch_loss (all frames)
    all_seg_losses = p_seg_losses + np_seg_losses
    all_dis_losses = p_dis_losses + np_dis_losses
    avg_b_seg_loss = torch.stack(all_seg_losses).mean()
    avg_b_dis_loss = torch.stack(all_dis_losses).mean()
    batch_loss     = alpha * avg_b_seg_loss + (1-alpha) * avg_b_dis_loss

    # 9) Backpropagation and optimizer updates
    # a) Non-prompt -> memory modules
    do_np = (non_prompt_loss is not None) and getattr(non_prompt_loss, "requires_grad", False)
    do_p  = getattr(prompt_loss, "requires_grad", False)

    # Scale loss according to available gradients
    if do_np and do_p:
        # Two passes over the same graph
        scaler.scale(non_prompt_loss).backward(retain_graph=True)
        scaler.scale(prompt_loss).backward()
    elif do_np and not do_p:
        # Only non-prompt carries grad in this batch
        scaler.scale(non_prompt_loss).backward()
    elif not do_np and do_p:
        # Only prompt carries grad in this batch
        scaler.scale(prompt_loss).backward()
    else:
        # Fallback: neither per-bucket loss has grad (edge case) — use combined loss
        scaler.scale(batch_loss).backward()

    # b) Update optimizers
    if memory_opt is not None and do_np:
        scaler.step(memory_opt)
    if mask_decoder_opt is not None and (do_p or not do_np):
        scaler.step(mask_decoder_opt)
    scaler.update()

    # Return all losses
    return batch_loss.item(), prompt_loss.item(), non_prompt_loss.item() if non_prompt_loss is not None else None

def validate_step_3d(student, batch, config, return_logits=False):
    """
    Single validation step for 3D (video) data.

    Args:
        student (SAM2VideoPredictor): The student model.
        batch (dict): A batch dict containing:
            'image':   Tensor[1,T,C,H,W]
            'mask':    Tensor[1,T,1,H',W']
            'pt_list': list[T] of Tensor[n,2] or None
            'p_label': list[T] of Tensor[n] or None
            'bbox':    list[T] of Tensor[4] or None
        config (DictConfig): Config with .training.device, .training.out_size, .prompt.max_per_seq
        return_logits (bool): Whether to return the predicted logits.

    Returns:
        if return_logits:
            dict: Average metrics {'iou', 'dice', 'hd95'} over all frames.
            torch.Tensor: Predicted logits for each frame.
        else:
            dict: Average metrics {'iou', 'dice', 'hd95'} over all frames.
    """
    device      = config.training.device
    out_size    = config.training.out_size
    max_prompts = config.prompt.max_per_seq

    # 1) Prepare video frames [T,C,H,W]
    imgs_seq = batch['image'].to(device) # [1,T,C,H,W]
    mask_seq = batch['mask'].to(device)  # [1,T,1,H,W]
    imgs     = imgs_seq.squeeze(0)       # [T,C,H,W]
    mask_seq = mask_seq.squeeze(0)       # [T,1,H,W]
    T        = imgs.shape[0]             # T = num_frames

    # 2) Initialize train state and inject prompts
    student.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            eval_state = student.init_state_from_tensor(imgs_tensor=imgs, mode="eval")
            eval_state['storage_device'] = torch.device(device)

            # Inject prompts once on the first frame (Q1), then propagate to all t>0
            q1 = 0
            points = batch['pt_list'][q1]
            labels = batch['p_label'][q1]
            bbox   = batch['bbox'][q1]
            mpr    = batch.get('m_prompt', None)
            cand   = (mpr[q1] if isinstance(mpr, list) else mpr) if mpr is not None else None
            mask   = cand if (torch.is_tensor(cand) and cand.numel() > 0) else mask_seq[q1]

            if points is not None and hasattr(points, "numel") and points.numel() > 0:
                student.add_new_points_or_box(
                    inference_state=eval_state,
                    frame_idx=q1, obj_id=0,
                    points=points.to(device).float(),
                    labels=labels.to(device).long(),
                    clear_old_points=True
                )
            elif bbox is not None and hasattr(bbox, "numel") and bbox.numel() == 4:
                empty_pts = torch.empty((1, 0, 2), device=device, dtype=torch.float32)
                empty_lbl = torch.empty((1, 0),    device=device, dtype=torch.int64)
                student.add_new_points_or_box(
                    inference_state=eval_state,
                    frame_idx=q1, obj_id=0,
                    points=empty_pts, labels=empty_lbl,
                    box=bbox.to(device).to(torch.int64).reshape(4),
                    clear_old_points=True
                )
            else:
                # Fallback: mask prompt on Q1  -> must be [H, W] bool on CUDA
                mask_2d = mask.to(device).squeeze()
                # If no mask prompt, use mask on Q1
                if mask_2d.numel() == 0:
                    mask_2d = mask_seq[q1].to(device).squeeze()
                # Resize if needed
                if mask_2d.dim() != 2:
                    H, W = imgs.shape[-2], imgs.shape[-1]
                    mask_2d = mask_2d.reshape(H, W)
                student.add_new_mask(eval_state, q1, 0, mask_2d.bool())

            # 3) Propagate using the first frame as the start
            preds = {}
            for frame_idx, obj_ids, mask_logits in student.propagate_in_video(eval_state, start_frame_idx=q1):
                # mask_logits: [num_objs, Hf, Wf]
                preds[frame_idx] = mask_logits[0].squeeze(0)

    # 4) Assemble raw logits volume and upsample each frame for visualization
    raw_logits_vol = torch.stack([preds[t] for t in sorted(preds.keys())], dim=0) # [T, Hf, Wf]
    logits_up = F.interpolate(
        raw_logits_vol.unsqueeze(1), # [T,Hf,Wf]
        size=(out_size, out_size),
        mode='bilinear', align_corners=False
    ) # [T,1,H,W]

    # 4b) Optional: Aggregate six orientations into a single 3D probability volume only if all dims > 0
    aggregated_prob = None
    axis_lengths = batch.get('axis_lengths', None)
    if axis_lengths is not None and logits_up.shape[0] == 2 * sum(axis_lengths):
        aggregated_prob = _aggregate_six_orientations(logits_up, axis_lengths) # [D, H, W]

    # 5) Compute per-frame metrics
    ious, dices, hd95s = [], [], []
    for t in range(T):
        # Compute predicted mask
        pred_logit = preds[t].unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
        pred_prob  = torch.sigmoid(F.interpolate(pred_logit, size=(out_size, out_size), mode='bilinear', align_corners=False))
        pred_mask = (pred_prob > 0.5).long().squeeze(0).squeeze(0)  # [H,W]

        # Compute ground truth mask
        gt_mask = mask_seq[t].unsqueeze(0) # [1,1,H,W]
        gt_up = F.interpolate(gt_mask.float(), size=(out_size, out_size), mode='nearest').long().squeeze(0).squeeze(0)

        # Compute metrics
        ious.append( compute_iou( pred_mask, gt_up))
        dices.append(compute_dice(pred_mask, gt_up))
        hd95s.append(compute_hd95(pred_mask, gt_up))

    # 6) Compute average metrics
    metrics = {
        'iou':  sum(ious)  / T,
        'dice': sum(dices) / T,
        'hd95': sum(hd95s) / T,
    }

    if return_logits:
        out = {'metrics': metrics, 'per_frame_logits': logits_up.squeeze(1).cpu()} #[T,H,W]
        if aggregated_prob is not None:
            out['aggregated_prob'] = aggregated_prob.detach().cpu() # [D,H,W] sigmoid probs
        return out
    return metrics

def _aggregate_six_orientations(pred_seq_logits, axis_lengths):
    """
    Aggregate a sequence of 2D prediction logits, each with size (Hs, Ws), into a 3D volume
    of size (D, H, W) by averaging 6 orientations.

    Args:
        pred_seq_logits (Tensor): [T, 1, Hs, Ws] sequence of 2D prediction logits.
        axis_lengths (Tuple[int, int, int]): (D, H, W) size of the output 3D volume.

    Returns:
        vol_prob (Tensor): [D, H, W] 3D volume of aggregated probabilities.
    """
    D, H, W = map(int, (axis_lengths.detach().cpu().reshape(-1).tolist() if torch.is_tensor(axis_lengths) else axis_lengths))
    T, _, Hs, Ws = pred_seq_logits.shape
    assert T == 2*(D + H + W), "Sequence length does not match 6-orientation layout"

    probs     = torch.sigmoid(pred_seq_logits.squeeze(1)) # [T, Hs, Ws]
    device    = probs.device
    vol_sum   = torch.zeros((D,H,W), device=device, dtype=probs.dtype)
    vol_count = torch.zeros((D,H,W), device=device, dtype=probs.dtype)

    idx = 0

    # Axial forward: z planes of shape (H, W)
    for z in range(D):
        p = F.interpolate(probs[idx+z].unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[z, :, :] += p
        vol_count[z, :, :] += 1
    idx += D

    # Axial reverse
    for zi in range(D):
        z = D - 1 - zi
        p = F.interpolate(probs[idx+zi].unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[z, :, :] += p
        vol_count[z, :, :] += 1
    idx += D

    # Coronal forward: y planes of shape (z, x) -> size (D, W)
    for y in range(H):
        p = F.interpolate(probs[idx+y].unsqueeze(0).unsqueeze(0), size=(D, W), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[:, y, :] += p
        vol_count[:, y, :] += 1
    idx += H

    # Coronal reverse
    for yi in range(H):
        y = H - 1 - yi
        p = F.interpolate(probs[idx+yi].unsqueeze(0).unsqueeze(0), size=(D, W), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[:, y, :] += p
        vol_count[:, y, :] += 1
    idx += H

    # Sagittal forward: x planes of shape (z, y) -> size (D, H)
    for x in range(W):
        p = F.interpolate(probs[idx+x].unsqueeze(0).unsqueeze(0), size=(D, H), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[:, :, x] += p
        vol_count[:, :, x] += 1
    idx += W

    # Sagittal reverse
    for xi in range(W):
        x = W - 1 - xi
        p = F.interpolate(probs[idx+xi].unsqueeze(0).unsqueeze(0), size=(D, H), mode='bilinear', align_corners=False).squeeze(0).squeeze()
        vol_sum[:, :, x] += p
        vol_count[:, :, x] += 1

    vol_prob = vol_sum / (vol_count.clamp_min(1))
    return vol_prob