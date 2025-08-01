# functions/func_3d.py
# Implements the per-batch 3D training and validation steps for
# student-teacher distillation in the RWKV-MedSAM2 model.

import torch
import torch.nn.functional as F

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
        batch (tuple): A tuple containing the input data and labels.
        config (DictConfig): The configuration dictionary.
        scaler (torch.cuda.amp.GradScaler): The scaler.

    Returns:
        loss (float): The loss value.
    """
    # 0) Device and config
    device      = config.training.device
    out_size    = config.training.out_size
    pos_weight  = config.training.pos_weight
    alpha       = config.training.alpha
    max_prompts = config.prompt.max_per_seq
    image_size  = config.model.image_size

    # 1) Prepare video tensor ([T,C,H,W])
    imgs_seq = batch['image'].to(device)
    # Assume batch size is 1 for 3D training
    imgs = imgs_seq.squeeze(0) # [T,C,H,W]

    # 2) Initialize state for student and teacher
    if torch.cuda.get_device_properties(device).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    student.train()
    if mask_decoder_opt: mask_decoder_opt.zero_grad()
    if memory_opt: memory_opt.zero_grad()
    train_state = student.train_init_state(imgs)

    teacher.eval()
    with torch.no_grad():
        teacher_state = teacher.train_init_state(imgs)

    # 3) Select prompt frames evenly
    T = imgs.shape[0] # T = num_frames
    prompt_idxs = torch.linspace(0, T-1, max_prompts).long().tolist()

    # 4) Inject prompts
    for frame_idx in prompt_idxs:
        points = batch['pt_list'][frame_idx]
        labels = batch['p_label'][frame_idx]
        mask   = batch['mask'][frame_idx] if 'mask' in batch else None
        if points is not None:
            points_t = points.to(device).float()
            labels_t = labels.to(device).long()
            student.train_add_new_points(
                inference_state=train_state,
                frame_idx=frame_idx,
                obj_id=0,
                points=points_t,
                labels=labels_t,
                clear_old_points=False
            )
            with torch.no_grad():
                teacher.train_add_new_points(
                    inference_state=teacher_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    points=points_t,
                    labels=labels_t,
                    clear_old_points=False
                )
        elif mask is None:
            bbox = batch['bbox'][frame_idx]
            bbox_t = bbox.to(device)
            student.train_add_new_bbox(
                inference_state=train_state,
                frame_idx=frame_idx,
                obj_id=0,
                bbox=bbox_t,
                clear_old_points=False
            )
            with torch.no_grad():
                teacher.train_add_new_bbox(
                    inference_state=teacher_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    bbox=bbox_t,
                    clear_old_points=False
                )
        else:
            # Mask prompt
            mask_t = mask.to(device).float().unsqueeze(0)
            student.train_add_new_mask(train_state, frame_idx, 0, mask_t)
            with torch.no_grad():
                teacher.train_add_new_mask(teacher_state, frame_idx, 0, mask_t)

    with torch.amp.autocast('cuda'):
        # 5) Propagate through video
        student_preds = {}
        for frame_idx, obj_ids, mask_logits in student.train_propagate_in_video(train_state, start_frame_idx=0):
            # Mask logits: [num_objs, Hf, Wf]
            student_preds[frame_idx] = mask_logits[0].unsqueeze(0)

        teacher_preds = {}
        with torch.no_grad():
            for frame_idx, obj_ids, mask_logits in teacher.train_propagate_in_video(teacher_state, start_frame_idx=0):
                # Mask logits: [num_objs, Hf, Wf]
                teacher_preds[frame_idx] = mask_logits[0].unsqueeze(0)

        # 6) Compute loss
        p_seg_losses, p_dis_losses   = [], [] # Losses for frames with prompts
        np_seg_losses, np_dis_losses = [], [] # Losses for frames without prompts
        mask_seq = batch['mask'].to(device).squeeze(0) # [T,1,H',W']
        for frame_idx in range(T):
            student_logit = student_preds[frame_idx]   # [1,Hf,Wf]
            teacher_logit = teacher_preds[frame_idx]   # [1,Hf,Wf]

            # Resize to out_size
            student_pred = F.interpolate(student_logit.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=False) # [1,1,out_size,out_size]
            teacher_pred = F.interpolate(teacher_logit.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=False) # [1,1,out_size,out_size]

            gt_mask = mask_seq[frame_idx].unsqueeze(0) # [1,1,out_size,out_size]

            frame_seg_loss = F.binary_cross_entropy_with_logits(student_pred, gt_mask, pos_weight=torch.tensor(pos_weight, device=device))
            frame_dis_loss = F.kl_div(F.log_softmax(student_pred, dim=1), F.softmax(teacher_pred, dim=1), reduction='batchmean')

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
    if len(prompt_idxs) > 1:
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

    # 7) Backpropagation and optimizer updates
    # a) Non-prompt -> memory modules
    if non_prompt_loss is not None:
        scaler.scale(non_prompt_loss).backward(retain_graph=True)
        scaler.step(memory_opt)
        scaler.update()
        memory_opt.zero_grad()

    # b) Prompt -> mask decoder (and backbone) + distillation
    scaler.scale(prompt_loss).backward()
    scaler.step(mask_decoder_opt)
    scaler.update()
    mask_decoder_opt.zero_grad()

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
    imgs     = imgs_seq.squeeze(0)       # [T,C,H,W]
    T        = imgs.shape[0]             # T = num_frames

    # 2) Initialize train state and inject prompts
    student.eval()
    with torch.no_grad():
        train_state = student.train_init_state(imgs)

        # Pick prompt frames evenly
        prompt_idxs = torch.linspace(0, T-1, steps=max_prompts, dtype=torch.int64).tolist()

        for frame_idx in prompt_idxs:
            points = batch['pt_list'][frame_idx]
            labels = batch['p_label'][frame_idx]
            bbox   = batch['bbox'][frame_idx]

            if points is not None:
                points_t = points.to(device).float()
                labels_t = labels.to(device).long()
                student.train_add_new_points(
                    inference_state=train_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    points=points_t,
                    labels=labels_t,
                    clear_old_points=False
                )
            elif bbox is not None:
                bbox_t = bbox.to(device)
                student.train_add_new_bbox(
                    inference_state=train_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    bbox=bbox_t,
                    clear_old_points=False
                )
            else:
                # Fallback to mask prompt if no point or bbox
                mask_t = batch['mask'][0, frame_idx].to(device).float().unsqueeze(0)
                student.train_add_new_mask(train_state, frame_idx, 0, mask_t)

        # 3) Propagate through video
        preds = {}
        for frame_idx, obj_ids, mask_logits in student.train_propagate_in_video(train_state, start_frame_idx=0):
            # mask_logits: [num_objs, Hf, Wf]
            preds[int(frame_idx)] = mask_logits[0]

    # 4) Resize and threshold + compute metrics
    mask_seq = batch['mask'][0].to(device).squeeze(1)  # [T, H',W']
    ious, dices, hd95s = [], [], []

    # Assemble raw logits volume and upsample each frame for visualization
    raw_logits_vol = torch.stack([preds[t] for t in sorted(preds.keys())], dim=0) # [T, Hf, Wf]
    logits_up = F.interpolate(
        raw_logits_vol.unsqueeze(1), # [T,1,Hf,Wf]
        size=(out_size, out_size),
        mode='bilinear', align_corners=False
    ).squeeze(1) # [T,H,W]

    for t in range(T):
        logit = preds[t].unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
        prob  = torch.sigmoid(F.interpolate(
            logit, size=(out_size, out_size),
            mode='bilinear', align_corners=False
        ))
        pred_mask = (prob > 0.5).long().squeeze(0).squeeze(0)  # [H,W]

        gt_mask = F.interpolate(
            mask_seq[t].unsqueeze(0).unsqueeze(0).float(),
            size=(out_size, out_size),
            mode='nearest'
        ).long().squeeze(0).squeeze(0)

        # Compute metrics
        ious.append(compute_iou(pred_mask, gt_mask) )
        dices.append(compute_dice(pred_mask, gt_mask) )
        hd95s.append(compute_hd95(pred_mask, gt_mask) )

    # 5) Compute average metrics
    metrics = {
        'iou':  sum(ious)  / T,
        'dice': sum(dices)/ T,
        'hd95': sum(hd95s)/ T,
    }

    if return_logits:
        return metrics, logits_up.cpu() # [T,H,W]
    else:
        return metrics