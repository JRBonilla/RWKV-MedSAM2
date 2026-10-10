"""Shared frame computation; callers own preparation, state and measurement."""


def predict_frame(model, frame, points, supplied, memory, frame_index, support, frames):
    """
    Encode a frame and run SAM2 tracking with the supplied prompts and memory.

    Args:
        model (SAM2Base): SAM2 model providing image encoding and temporal tracking.
        frame (torch.Tensor): Normalized image frame passed to the encoder.
        points (dict[str, torch.Tensor] | None): Point or box coordinate/label inputs, or
            None when no point prompt is supplied.
        supplied (torch.Tensor | None): Optional support-mask tensor used as the mask
            prompt.
        memory (dict): Conditioned and unconditioned frame outputs used by SAM2 temporal
            memory.
        frame_index (int): Dense selected-sequence index used for temporal memory.
        support (bool): Whether this frame supplies a frozen conditioning prompt.
        frames (int): Total number of frames in the selected inference sequence.

    Returns:
        dict: Tracking output containing predicted masks and memory features.
    """
    encoded = model.forward_image(frame)
    _, features, positions, sizes = model._prepare_backbone_features(encoded)

    return model.track_step(
        frame_idx=frame_index,
        is_init_cond_frame=support,
        current_vision_feats=features,
        current_vision_pos_embeds=positions,
        feat_sizes=sizes,
        point_inputs=points,
        mask_inputs=supplied,
        output_dict=memory,
        num_frames=frames,
        track_in_reverse=False,
        run_mem_encoder=True,
    )
