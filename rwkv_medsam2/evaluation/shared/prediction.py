"""Shared frame computation; callers own preparation, state and measurement."""


def predict_frame(model, frame, points, supplied, memory, frame_index, support, frames):
    encoded = model.forward_image(frame)
    _, features, positions, sizes = model._prepare_backbone_features(encoded)
    return model.track_step(
        frame_idx=frame_index, is_init_cond_frame=support,
        current_vision_feats=features, current_vision_pos_embeds=positions,
        feat_sizes=sizes, point_inputs=points, mask_inputs=supplied,
        output_dict=memory, num_frames=frames, track_in_reverse=False,
        run_mem_encoder=True)
