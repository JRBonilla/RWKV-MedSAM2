# functions/func_2d.py
# Implements the per-batch 2D training and validation steps for
# student-teacher distillation in the RWKV-MedSAM2 model.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .func_metrics import compute_iou, compute_dice, compute_hd95

def attach_highres_adapters(model, config, device=None, dtype=torch.float32):
    """
    Helper function to attach 2D adapters to the SAM mask decoder.
    1) Runs a dummy forward through model.image_encoder + _prepare_backbone_features  
    2) Inspects the spatial feature shapes that will be fed as high_res_features  
    3) Creates adapter_s0 and adapter_s1 (1x1 convs) to map them into
       the mask-decoder's expected channel dims  
    4) Attaches those convs onto model.sam_mask_decoder

    Args:
        model (SAM2VideoPredictor): The SAM2VideoPredictor model.
        config (DictConfig): The configuration dictionary.
        device (torch.device, optional): The device to run the model on.
        dtype (torch.dtype, optional): The data type for the adapters.
    """
    device = device or next(model.parameters()).device
    # 1) Dummy image
    H = config.model.image_size
    dummy = torch.zeros(1, 3, H, H, device=device)
    # 2) Forward to get vision_feats
    with torch.no_grad():
        backbone_out = model.image_encoder(dummy)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    # 3) Mimic review logic to get (B=1,C,H',W') tensors
    feat_sizes = config.decoder.feat_sizes           # e.g. [[128,128],[64,64],[32,32]]
    raw_feats = [
        feat.permute(1,2,0).view(1, -1, *size)
        for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
    ]
    hires_feats = raw_feats[:-1]                     # [feat_hr, feat_mr]
    in_ch_s0 = hires_feats[0].shape[1]
    in_ch_s1 = hires_feats[1].shape[1]
    # 4) Read out the target dims from the decoder
    dc1, ln1, act1, dc2, act2 = model.sam_mask_decoder.output_upscaling
    target_ch_s1 = dc1.out_channels                  # e.g. hidden_dim//4
    target_ch_s0 = dc2.out_channels                  # e.g. hidden_dim//8
    # 5) Create & attach adapters once
    dec = model.sam_mask_decoder
    dec.adapter_s0 = nn.Conv2d(in_ch_s0, target_ch_s0, kernel_size=1)\
                         .to(device=device, dtype=dtype)
    dec.adapter_s1 = nn.Conv2d(in_ch_s1, target_ch_s1, kernel_size=1)\
                         .to(device=device, dtype=dtype)

def train_step_2d(student, teacher, optimizer, batch, config, memory_bank, scaler):
    """
    Perform a single training step for a 2D model using student-teacher distillation.

    This function processes a batch of 2D images through a student model and a 
    teacher model to compute segmentation and distillation losses. It updates 
    the model parameters using these losses and manages a memory bank for 
    storing encoded features.

    Args:
        student (SAM2VideoPredictor): The student model for training.
        teacher (SAM2VideoPredictor): The teacher model for distillation.
        optimizer (torch.optim.Optimizer): The optimizer for updating the student model.
        batch (dict): A batch of input data containing images, masks, and prompts.
        config (DictConfig): Configuration parameters for the training process.
        memory_bank (list): A list to store memory features and positions for attention.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.

    Returns:
        float: The computed loss value for the training step.
    """
    # Device and config setup
    device     = config.training.device
    feat_sizes = config.decoder.feat_sizes  # [H, W]
    out_size   = config.training.out_size
    pos_weight = config.training.pos_weight
    alpha      = config.training.alpha
    capacity   = config.memory_bank.capacity
    c_thresh   = config.memory_bank.c_thresh

    # 1) Unpack first slice
    imgs  = batch['image'][:,0].to(device) # [B,C,H,W]
    masks = batch['mask'][:,0].to(device)  # [B,1,H,W]
    batch_size = imgs.size(0)              # B = batch_size

    # 2) Build sparse prompts for slice 0
    raw_points = batch['pt_list'][0]
    raw_labels = batch['p_label'][0]
    points, labels = [], []
    for i in range(batch_size):
        if raw_points[i] is not None:
            points.append(raw_points[i].to(device).unsqueeze(0).float()) # [1,N,2]
            labels.append(raw_labels[i].to(device).unsqueeze(0).long())  # [1,N]
    if points:
        sparse_points = torch.cat(points, dim=0)
        sparse_labels = torch.cat(labels, dim=0)
    else:
        sparse_points = sparse_labels = None

    # 3) Encode prompts
    sparse_embs, dense_embs = student.sam_prompt_encoder(
        points=(sparse_points, sparse_labels) if sparse_points is not None else None,
        boxes=None, masks=None
    )
    image_pe = student.sam_prompt_encoder.get_dense_pe()

    # 4) Mixed precision + TF32
    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    student.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # 5) Forward + memory attention
        backbone_out = student.forward_image(imgs)
        _, vision_feats, vision_pos, _ = student._prepare_backbone_features(backbone_out)

        if memory_bank:
            # Build [tokens, B, C] memory + pos
            mem_feats = torch.cat([m[0] for m in memory_bank], dim=0)
            mem_pos   = torch.cat([m[1] for m in memory_bank], dim=0)
            # Sample one entry per batch by similarity
            curr  = vision_feats[-1].permute(1,0,2).reshape(batch_size, -1)
            banks = torch.stack([e[3] for e in memory_bank], dim=0)
            banks = F.normalize(banks, p=2, dim=1)
            sims  = (banks @ F.normalize(curr, p=2, dim=1).t()).t()
            idx   = torch.multinomial(F.softmax(sims, dim=1), num_samples=1).squeeze(1)
            sel_f = mem_feats[:, idx, :].permute(1,2,0)
            sel_p = mem_pos[:, idx, :].permute(1,2,0)
            vision_feats[-1], vision_pos[-1] = student.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos[-1]],
                memory=sel_f,
                memory_pos=sel_p,
                num_obj_ptr_tokens=0
            )
        else:
            # If no memory yet, add a zero key/value so memory_attention always has something to attend to
            zeros_f = torch.zeros_like(vision_feats[-1])
            zeros_p = torch.zeros_like(vision_pos[-1])
            vision_feats[-1] = vision_feats[-1] + zeros_f
            vision_pos[-1]   = vision_pos[-1] + zeros_p

        # 6) Decode mask
        feats = [
            feat.permute(1,2,0).view(batch_size, -1, *size)
            for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        hires_feats = feats[:-1] # [feat_hr, feat_mr]

        # Resize dense prompt embeddings
        dense_embs = F.interpolate(dense_embs, size=image_embed.shape[-2:], mode='bilinear', align_corners=False)

        student_logits, student_iou, _, student_object_score_logits = student.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embs,
            dense_prompt_embeddings=dense_embs,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hires_feats
        )
        student_pred = F.interpolate(student_logits, size=(out_size, out_size), mode='bilinear', align_corners=False)

        # 7) Teacher distillation
        with torch.no_grad():
            teacher_backbone       = teacher.forward_image(imgs)
            _, teacher_feats, _, _ = teacher._prepare_backbone_features(teacher_backbone)
            teacher_embed          = teacher_feats[-1].permute(1,2,0).view(batch_size, -1, *feat_sizes[-1])
            teacher_hires_feats    = [f.permute(1,2,0).reshape(batch_size, -1, *size) for f, size in zip(teacher_feats[::-1][1:], feat_sizes[:-1])]

            teacher_sparse_embs, teacher_dense_embs = teacher.sam_prompt_encoder(
                points=(sparse_points, sparse_labels) if sparse_points is not None else None,
                boxes=None, masks=None
            )
            teacher_dense_prompts = F.interpolate(teacher_dense_prompts, size=teacher_embed.shape[-2:], mode='bilinear', align_corners=False)
            teacher_image_pe      = teacher.sam_prompt_encoder.get_dense_pe()
            teacher_logits, _, *_ = teacher.sam_mask_decoder(
                image_embeddings=teacher_embed,
                image_pe=teacher_image_pe,
                sparse_prompt_embeddings=teacher_sparse_embs,
                dense_prompt_embeddings=teacher_dense_embs,
                multimask_output=False,
                repeat_image=False,
                high_res_features=teacher_hires_feats
            )
            teacher_pred = F.interpolate(teacher_logits, size=(out_size, out_size), mode='bilinear', align_corners=False)

        # 8) Compute losses
        # Segmentation loss
        mask_target = masks.to(student_pred.dtype)
        pw = torch.tensor(pos_weight, device=device, dtype=student_pred.dtype)
        seg_loss = F.binary_cross_entropy_with_logits(student_pred, mask_target, pos_weight=pw)

        dis_loss = F.kl_div(F.log_softmax(student_pred, dim=1), F.softmax(teacher_pred, dim=1), reduction='batchmean')
        loss = alpha * seg_loss + (1-alpha) * dis_loss

    # 9) Memory encode and update
    new_feats, new_pos = student._encode_new_memory(
        current_vision_feats=vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=F.interpolate(student_logits, size=(config.model.image_size, config.model.image_size), mode='bilinear', align_corners=False).to(torch.float32),
        object_score_logits=student_object_score_logits,
        is_mask_from_pts=(sparse_points is not None)
    )
    new_feats = new_feats.to(torch.bfloat16).to(device)
    new_pos   = new_pos[0].to(torch.bfloat16).to(device)

    # Update memory
    for i in range(batch_size):
        entry = [
            new_feats[i:i+1].detach(),          # Features
            new_pos[i:i+1].detach(),            # Position embeddings
            student_iou[i,0].item(),            # IoU
            image_embed[i].reshape(-1).detach() # Image embedding
        ]
        # Only add automatically if memory bank is not full
        # Otherwise, replace the least similar entry
        if len(memory_bank) < capacity:
            memory_bank.append(entry)
        else:
            flats = torch.stack([e[0].reshape(-1) for e in memory_bank]) # Flatten entries in memory bank
            flats_norm = F.normalize(flats, p=2, dim=1)                  # Normalize
            new_flat   = entry[0].reshape(-1)                            # Flatten new entry
            sims_vec   = flats_norm @ F.normalize(new_flat, p=2, dim=0)  # Compute similarity of new entry with all others
            min_idx    = torch.argmin(sims_vec)                          # Get index of least similar
            sim_mat    = flats_norm @ flats_norm.t()                     # Compute similarity matrix
            sim_mat.fill_diagonal_(-float('inf'))                        # Set diagonal to -inf so we don't replace with self

            # Only add new entry if all of the following are true:
            # - IoU is greater than least similar IoU
            # - Similarity is greater than least similar
            # - IoU is greater than threshold
            if entry[2] >= memory_bank[min_idx][2] - 0.1 and sims_vec[min_idx] < sim_mat[min_idx].max():
                if entry[2] >= c_thresh:
                    memory_bank[min_idx] = entry # Replace least similar entry

    # 10) Backward and step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def validate_step_2d(student, batch, config, return_logits=False):
    """
    Single validation step for 2D data.

    Args:
        student (SAM2VideoPredictor): The student model.
        batch (dict): Batch dict with keys 'image', 'mask', 'pt_list', 'p_label'.
        config (DictConfig): Config with .training.device and .training.out_size.
        return_logits (bool): Whether to return the predicted logits. Default is False.

    Returns:
        if return_logits is True:
            dict: Average metrics {'iou', 'dice', 'hd95'} for the batch.
            torch.Tensor: Predicted logits for the batch.
        else:
            dict: Average metrics {'iou', 'dice', 'hd95'} for the batch.
    """
    device   = config.training.device
    out_size = config.training.out_size

    # 1) Unpack
    imgs       = batch['image'][:, 0].to(device)  # [B,C,H,W]
    masks      = batch['mask'][:, 0].to(device)   # [B,1,H,W]
    batch_size = imgs.size(0)

    # 2) Build sparse prompts exactly as in train
    raw_points = batch['pt_list'][0]
    raw_labels = batch['p_label'][0]
    points, labels = [], []
    for i in range(batch_size):
        if raw_points[i] is not None:
            points.append(raw_points[i].to(device).unsqueeze(0).float())  # [1,N,2]
            labels.append(raw_labels[i].to(device).unsqueeze(0).long())   # [1,N]
    if points:
        sparse_points = torch.cat(points, dim=0)
        sparse_labels = torch.cat(labels, dim=0)
    else:
        sparse_points = sparse_labels = None

    # 3) Prompt‑encode, forward backbone, prepare features & decode mask
    with torch.no_grad():
        sparse_embs, _ = student.sam_prompt_encoder(
            points=(sparse_points, sparse_labels) if sparse_points is not None else None,
            boxes=None, masks=None
        )
        dense_embs = student.sam_prompt_encoder.get_dense_pe()

        backbone_out = student.forward_image(imgs)
        _, vision_feats, vision_pos, _ = student._prepare_backbone_features(backbone_out)

        # No memory bank so add zeros
        vision_feats[-1] = vision_feats[-1] + torch.zeros_like(vision_feats[-1])
        vision_pos[-1]   = vision_pos[-1]   + torch.zeros_like(vision_pos[-1])

        feat_sizes = config.decoder.feat_sizes
        feats = [
            feat.permute(1,2,0).view(batch_size, -1, *sz)
            for feat, sz in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        hires_feats  = feats[:-1]

        logits, _, *_ = student.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=dense_embs,
            sparse_prompt_embeddings=sparse_embs,
            dense_prompt_embeddings=dense_embs,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hires_feats
        )

        # a) Up-sample the raw logits to output size
        logits_up = F.interpolate(logits, size=(out_size, out_size), mode='bilinear', align_corners=False) # [B,1,H,W]

        # b) Compute probabilities and hard predictions for metrics
        probs = torch.sigmoid(logits_up) # [B,1,H,W]
        preds = (probs > 0.5).long().squeeze(1).cpu()
        masks = F.interpolate(masks.float().cpu(), size=(out_size, out_size), mode='nearest').long().squeeze(1)  # [B,H,W]

    # 4) Metrics
    iou_list, dice_list, hd95_list = [], [], []
    for i in range(batch_size):
        iou_list.append(compute_iou(preds[i], masks[i]))
        dice_list.append(compute_dice(preds[i], masks[i]))
        hd95_list.append(compute_hd95(preds[i], masks[i]))

    # Compute averages of all metrics
    metrics = {
        'iou':  sum(iou_list)  / batch_size,
        'dice': sum(dice_list) / batch_size,
        'hd95': sum(hd95_list) / batch_size,
    }

    if return_logits:
        # Return up-sampled raw logits (on CPU) so validate_epoch can visualize them
        return metrics, logits_up.squeeze(1).cpu() # [B,H,W]
    else:
        return metrics