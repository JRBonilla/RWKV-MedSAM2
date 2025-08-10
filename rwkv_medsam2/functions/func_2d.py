# functions/func_2d.py
# Implements the per-batch 2D training and validation steps for
# student-teacher distillation in the RWKV-MedSAM2 model.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .func_metrics import compute_iou, compute_dice, compute_hd95

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

    # 1) Unpack all frames & flatten batch/time
    B,T,C,H,W  = batch['image'].shape
    imgs       = batch['image'].view(B*T, C, H, W).to(device)   # [B*T, C, H, W]
    masks      = batch['mask'].view( B*T, 1, H, W).to(device)   # [B*T, 1, H, W]
    batch_size = B * T

    # 2) Build sparse prompts per flattened frame (B*T total)
    raw_pt_list  = batch['pt_list']    # List[T] of length-B lists of point Tensors
    raw_lbl_list = batch['p_label']    # List[T] of length-B lists of label Tensors
    raw_box_list = batch['bbox']
    raw_m_prompt = batch.get('m_prompt', None)

    # Flatten B,T into B*T order
    pts_all, lbs_all, box_all, msk_all = [], [], [], []
    for t in range(T):
        for b in range(B):
            # Points
            pts = raw_pt_list[t][b] if isinstance(raw_pt_list[t], list) else raw_pt_list[t][b]
            lbs = raw_lbl_list[t][b] if isinstance(raw_lbl_list[t], list) else raw_lbl_list[t][b]
            # Bounding box
            bx = raw_box_list[t][b] if isinstance(raw_box_list[t], list) else raw_box_list[t][b]
            # Mask
            if raw_m_prompt is not None:
                mp = raw_m_prompt[t][b] if isinstance(raw_m_prompt[t], list) else raw_m_prompt[t][b]
            else:
                mp = None

            pts_all.append(pts)
            lbs_all.append(lbs)
            box_all.append(bx)
            msk_all.append(mp)

    # 3) Mixed prompting: encode per-subset and stitch back (student)
    student_sparse_embs, student_dense_embs, _ = _encode_mixed_prompts(
        student.sam_prompt_encoder, pts_all, lbs_all, box_all, msk_all, imgs, device
    )
    image_pe = student.sam_prompt_encoder.get_dense_pe()

    # 4) Enable TF32 matmuls on Ampere+ GPUs while using bfloat16 autocast
    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    student.train()
    optimizer.zero_grad()

    torch.autograd.set_detect_anomaly(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # 5) Forward + memory attention
        backbone_out = student.forward_image(imgs)
        _, vision_feats, vision_pos, _ = student._prepare_backbone_features(backbone_out)

        if memory_bank:
            # Gather saved memory features & positions -> [bank_size, P, C, H, W]
            mem_feats = torch.cat([m[0] for m in memory_bank], dim=0)
            mem_pos   = torch.cat([m[1] for m in memory_bank], dim=0)
            # Sample one entry per batch by similarity
            curr  = vision_feats[-1].permute(1,0,2).reshape(batch_size, -1)
            banks = torch.stack([e[3] for e in memory_bank], dim=0)
            banks = F.normalize(banks, p=2, dim=1)
            sims  = (banks @ F.normalize(curr, p=2, dim=1).t()).t()
            
            # Compute softmax
            probs = F.softmax(sims, dim=1)
            K     = config.memory_bank.capacity

            # a) Draw K indices per example (with replacement)
            idxs = torch.multinomial(probs, num_samples=K, replacement=True) # [B, K]

            # b) Gather K features & positions per example -> [B,K,C,H,W]
            batch_mem_feats = mem_feats[idxs]
            batch_mem_pos   = mem_pos[idxs]
            
            # c) Flatten into KxHxW tokens: [B,K,C,H,W] -> [K*H*W,B,C]
            B,K,C,H,W = batch_mem_feats.shape
            S = H * W
            sel_f = (
                batch_mem_feats         # [B,K,C,H,W]
                    .reshape(B,K,C,S)   # [B,K,C,S]
                    .permute(1,3,0,2)   # [K,S,B,C]
                    .reshape(K*S,B,C)   # [K*S,B,C]
            )
            sel_p = (
                batch_mem_pos           # [B,K,C,H,W]
                    .reshape(B,K,C,S)   # [B,K,C,S]
                    .permute(1,3,0,2)   # [K,S,B,C]
                    .reshape(K*S,B,C)   # [K*S,B,C]
            )

            # d) Memory attention
            updated_feats = student.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos[-1]],
                memory=sel_f,
                memory_pos=sel_p,
                num_obj_ptr_tokens=0
            )
            vision_feats[-1] = updated_feats
        else:
            # If no memory yet, add a zero key/value so memory_attention always has something to attend to
            zeros_f = torch.zeros_like(vision_feats[-1])
            zeros_p = torch.zeros_like(vision_pos[-1])
            vision_feats[-1] = (vision_feats[-1] + zeros_f).clone()
            vision_pos[-1]   = (vision_pos[-1]   + zeros_p).clone()

        # 6) Decode mask
        feats = [
            feat.permute(1,2,0).view(batch_size, -1, *size)
            for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        hires_feats = feats[:-1] # [feat_hr, feat_mr]

        image_embed = image_embed.clone()
        hires_feats = [hf.clone() for hf in hires_feats]
        if student_sparse_embs is not None:
            student_sparse_embs = student_sparse_embs.clone()

        # Resize dense prompt embeddings
        student_dense_embs = F.interpolate(student_dense_embs, size=image_embed.shape[-2:], mode='bilinear', align_corners=False).clone()

        image_pe = image_pe.clone()
        
        student_logits, student_iou, _, student_object_score_logits = student.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=student_sparse_embs,
            dense_prompt_embeddings=student_dense_embs,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hires_feats
        )
        student_pred = F.interpolate(student_logits, size=(out_size, out_size), mode='bilinear', align_corners=False) # [B*T,1,H,W]

        # 7) Teacher distillation
        with torch.no_grad():
            teacher_backbone = teacher.forward_image(imgs)
            # Get both feats and spatial sizes
            _, t_feats, t_pos, t_sizes = teacher._prepare_backbone_features(teacher_backbone)

            # Rebuild the same feats list the student uses
            feats_t = [
                feat.permute(1, 2, 0).view(batch_size, -1, *size)
                for feat, size in zip(t_feats[::-1], t_sizes[::-1])
            ][::-1]
            teacher_embed       = feats_t[-1]
            teacher_hires_feats = feats_t[:-1]

            # Encode the exact same (mixed) prompts for teacher
            teacher_sparse_embs, teacher_dense_embs, _ = _encode_mixed_prompts(
                teacher.sam_prompt_encoder, pts_all, lbs_all, box_all, msk_all, imgs, device
            )
            # Resize dense prompts to match teacher_embed's HxW
            teacher_dense_embs = F.interpolate(teacher_dense_embs, size=teacher_embed.shape[-2:], mode='bilinear', align_corners=False)

            # Get & resize the image positional encoding
            teacher_image_pe = F.interpolate(teacher.sam_prompt_encoder.get_dense_pe(), size=teacher_embed.shape[-2:], mode='bilinear', align_corners=False)

            # Forward through exactly the same mask‐decoder call
            teacher_logits, _, *_ = teacher.sam_mask_decoder(
                image_embeddings=teacher_embed,
                image_pe=teacher_image_pe,
                sparse_prompt_embeddings=teacher_sparse_embs,
                dense_prompt_embeddings=teacher_dense_embs,
                multimask_output=False,
                repeat_image=False,
                high_res_features=teacher_hires_feats
            )
            teacher_pred = F.interpolate(teacher_logits, size=(out_size, out_size), mode='bilinear', align_corners=False) # [B*T,1,H,W]

        # 8) Compute losses
        # a) Segmentation loss (BCE)
        mask_target = masks.to(student_pred.dtype).clone()
        if mask_target.dim() == student_pred.dim() + 1:
          mask_target = mask_target.squeeze(2) # [B*T,1,1,H,W] -> [B*T,1,W,H]
        pw = torch.tensor(pos_weight, device=device, dtype=student_pred.dtype)
        seg_loss = F.binary_cross_entropy_with_logits(student_pred, mask_target, pos_weight=pw)

        # b) Distillation loss (KL-divergence)
        with torch.no_grad():
            teacher_target = torch.sigmoid(teacher_pred) # [B*T,1,H,W] in [0,1]
        dis_loss = F.binary_cross_entropy_with_logits(student_pred, teacher_target, reduction='mean')
        
        # c) Total loss (weighted sum of both losses)
        loss = alpha * seg_loss + (1-alpha) * dis_loss

    # 9) Memory encode and update (IoU-gated + Top-K by dissimilarity)
    new_feats, new_pos = student._encode_new_memory(
        current_vision_feats=vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=F.interpolate(
            student_logits,
            size=(config.model.image_size, config.model.image_size),
            mode='bilinear',
            align_corners=False
        ).to(torch.float32),
        object_score_logits=student_object_score_logits,
        is_mask_from_pts=(student_sparse_embs is not None)
    )
    new_feats = new_feats.to(torch.bfloat16).to(device)
    new_pos   = new_pos[0].to(torch.bfloat16).to(device)

    # Helper to prune to Top-K by total dissimilarity
    def _prune_topk_dissim(bank_entries, cap):
        """
        bank_entries: list of entries where entry[3] is a flat image embedding vector (1D tensor)
        cap: maximum size to keep
        """
        if not bank_entries:
            return bank_entries

        # Stack embedding vectors -> [N, D]
        vecs = torch.stack([e[3] for e in bank_entries], dim=0)
        vecs = vecs.to(device)
        vecs = F.normalize(vecs, p=2, dim=1)

        # Cosine similarity matrix [N, N]
        sim = vecs @ vecs.t()
        # Dissimilarity = 1 - sim, zero out diagonal
        dis = 1.0 - sim
        dis.fill_diagonal_(0.0)
        # Total dissimilarity per entry
        total_dis = dis.sum(dim=1)  # [N]

        Kkeep = min(cap, len(bank_entries))
        # Indices of Top-K by total dissimilarity
        topk_idx = torch.topk(total_dis, k=Kkeep, largest=True).indices.tolist()
        # Keep order stable: sort indices ascending
        topk_idx.sort()
        return [bank_entries[i] for i in topk_idx]

    # Build candidate entries (only if above IoU threshold) and prune globally
    candidates_changed = False
    for i in range(batch_size):
        conf_i = float(student_iou[i, 0].item()) if student_iou.ndim >= 2 else float(student_iou[i].item())
        if conf_i < c_thresh:
            continue  # Gate by confidence

        entry = [
            new_feats[i:i+1].detach(),          # 0: Features   [1,C,H,W]
            new_pos[i:i+1].detach(),            # 1: Pos embeds [1,C,H,W]
            conf_i,                             # 2: IoU scalar
            image_embed[i].reshape(-1).detach() # 3: Flat image embedding vector
        ]

        # Add the new entry to the candidate set
        memory_bank.append(entry)
        candidates_changed = True

    # If memory bank changed, prune to Top-K by dissimilarity
    if candidates_changed:
        memory_bank[:] = _prune_topk_dissim(memory_bank, capacity)

    # 10) Backward and step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

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
        if not idxs: return None
        pts = torch.stack([pts_all[i].to(device).to(torch.float32) for i in idxs], dim=0)  # [k, n, 2]
        lbs = torch.stack([lbs_all[i].to(device).to(torch.long)    for i in idxs], dim=0)  # [k, n]
        return (pts, lbs)

    def _gather_boxes(idxs):
        if not idxs: return None
        return torch.stack([box_all[i].to(device).to(torch.int64).view(4) for i in idxs], dim=0)  # [k,4]

    def _gather_masks(idxs):
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

    # 1) Unpack all frames & flatten batch/time
    B,T,C,H,W  = batch['image'].shape
    imgs       = batch['image'].view(B*T, C, H, W).to(device)   # [B*T, C, H, W]
    masks      = batch['mask'].view( B*T, 1, H, W).to(device)   # [B*T, 1, H, W]
    batch_size = B * T

    # 2) Build sparse prompts over all T frames (flattened to B*T images)
    raw_pt_list  = batch['pt_list']    # List[T] of length-B lists of point Tensors
    raw_lbl_list = batch['p_label']    # List[T] of length-B lists of label Tensors
    points, labels = [], []
    for t in range(T):
        for b in range(B):
            pts = raw_pt_list[t][b]
            lbs = raw_lbl_list[t][b]
            if pts is not None and pts.numel() > 0:
                points.append(pts.to(device).unsqueeze(0).float())  # [1,n,2]
                labels.append(lbs.to(device).unsqueeze(0).long())   # [1,n]
    if points:
        sparse_points = torch.cat(points, dim=0)
        sparse_labels = torch.cat(labels, dim=0)
    else:
        sparse_points = sparse_labels = None

    # 3) Prompt‑encode, forward backbone, prepare features & decode mask
    with torch.no_grad():
        # If no clicks *or* fewer clicks than images, use full-image boxes
        if sparse_points is None or sparse_points.shape[0] < batch_size:
            # Build [batch_size, 4] tensor: [x1,y1,x2,y2] = full image
            full_box = torch.tensor([0, 0, W-1, H-1], device=device, dtype=torch.int64)
            boxes    = full_box.unsqueeze(0).repeat(batch_size, 1)
            sparse_embs, dense_embs = student.sam_prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None
            )
        else:
            sparse_embs, dense_embs = student.sam_prompt_encoder(
                points=(sparse_points, sparse_labels),
                boxes=None,
                masks=None
            )

        image_pe = student.sam_prompt_encoder.get_dense_pe()

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
        hires_feats = feats[:-1]

        dense_embs = F.interpolate(dense_embs, size=image_embed.shape[-2:], mode='bilinear', align_corners=False)
        logits, _, *_ = student.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embs,
            dense_prompt_embeddings=dense_embs,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hires_feats
        )

        # a) Up-sample the raw logits to output size -> flat [B*T,1,H_out,W_out]
        logits_up_flat = F.interpolate(logits, size=(out_size, out_size), mode='bilinear', align_corners=False)
        # b) Reshape flat logits into sequence [B, T, H_out, W_out] for per-frame metrics
        _, _, Hout, Wout = logits_up_flat.shape
        logits_up_seq = logits_up_flat.view(B, T, Hout, Wout)  # [B,T,H,W]

        # c) Compute probabilities and hard predictions: [B,T,H,W]
        probs      = torch.sigmoid(logits_up_seq)
        preds_seq  = (probs > 0.5).long().cpu()

        # d) Up-sample masks to match out_size, then reshape to [B, T, H_out, W_out]
        masks_seq = masks

        # e) Flatten to [B*T, H_out, W_out] for per-frame metrics
        preds_flat = preds_seq.view(B * T, Hout, Wout)
        masks_flat = masks_seq.view(B * T, Hout, Wout)

    # 4) Metrics
    iou_list, dice_list, hd95_list = [], [], []
    for i in range(batch_size):
        iou_list.append(compute_iou(  preds_flat[i], masks_flat[i]))
        dice_list.append(compute_dice(preds_flat[i], masks_flat[i]))
        hd95_list.append(compute_hd95(preds_flat[i], masks_flat[i]))

    # Compute averages of all metrics
    metrics = {
        'iou':  sum(iou_list)  / batch_size,
        'dice': sum(dice_list) / batch_size,
        'hd95': sum(hd95_list) / batch_size,
    }

    if return_logits:
        # Return up-sampled raw logits (on CPU) so validate_epoch can visualize them
        return metrics, logits_up_seq.cpu() # [B,T,H,W]
    else:
        return metrics