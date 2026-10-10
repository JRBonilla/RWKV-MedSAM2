"""Focused tests for the opt-in RWKV-MedSAM2 optimization path."""

from __future__ import annotations

import torch

import rwkv_medsam2.functions.func_3d as func_3d
from sam2.utils.misc import load_video_frames_from_data
from rwkv_medsam2.functions.func_3d import validate_step_3d
from rwkv_medsam2.optimized_video_predictor import OptimizedSAM2VideoPredictor


class _TinyPredictor(OptimizedSAM2VideoPredictor):
    """
    Exercise feature-cache logic without constructing the full SAM2 model.
    """

    def __init__(
        self,
        *,
        chunk_size=4,
        training_cache=8,
        gpu_chunks=2,
        cpu_chunks=1,
    ):
        """
        Initialize a small predictor with configurable cache limits for CPU tests.

        Args:
            chunk_size (int): Number of frames in each feature-encoding chunk.
            training_cache (int): Maximum number of cached training frames in the test
                predictor.
            gpu_chunks (int): Maximum GPU feature chunks in the test predictor.
            cpu_chunks (int): Maximum CPU feature chunks in the test predictor.

        Returns:
            None.
        """
        torch.nn.Module.__init__(self)
        self.image_feature_chunk_size = chunk_size
        self.training_feature_cache_frames = training_cache
        self.prediction_gpu_cache_chunks = gpu_chunks
        self.prediction_cpu_cache_chunks = cpu_chunks
        self.prediction_pin_cpu_features = False
        self.encoder_calls = 0

    def forward_image(self, images):
        """
        Produce synthetic backbone features and count image-encoder calls.

        Args:
            images (torch.Tensor): Batch of image tensors to encode or move to the compute
                device.

        Returns:
            dict: Synthetic feature pyramid and positional encodings.
        """
        self.encoder_calls += 1
        count, _, height, width = images.shape

        return {
            "backbone_fpn": [images[:, :1] * 2.0],
            "vision_pos_enc": [
                torch.zeros((count, 1, height, width), device=images.device)
            ],
        }

    def _prepare_backbone_features(self, backbone_out):
        """
        Extract the synthetic feature and positional-encoding lists.

        Args:
            backbone_out (dict): Image-encoder feature pyramid and positional encodings.

        Returns:
            tuple[list, list]: Feature pyramid and positional encodings.
        """
        return (
            backbone_out["backbone_fpn"],
            backbone_out["vision_pos_enc"],
        )


def _state(frame_count, mode):
    """
    Build a deterministic CPU predictor state for cache tests.

    Args:
        frame_count (int): Number of synthetic frames in the predictor state.
        mode (str): Feature-cache mode, either training or prediction.

    Returns:
        dict: Synthetic images, frame count, device, and cache mode.
    """
    return {
        "images": torch.arange(frame_count * 3 * 2 * 2, dtype=torch.float32).reshape(
            frame_count, 3, 2, 2
        ),
        "num_frames": frame_count,
        "device": torch.device("cpu"),
        "cached_features": {},
        "_rwkv_feature_cache_mode": mode,
    }


def test_training_chunk_cache_batches_frames_and_respects_limit():
    """
    Verify batched training encoding and the frame-cache size limit.

    Args:
        None.

    Returns:
        None.
    """
    predictor = _TinyPredictor()
    state = _state(8, "training")

    predictor._get_image_feature(state, 0, 1)
    predictor._get_image_feature(state, 1, 1)
    predictor._get_image_feature(state, 4, 1)

    profile = predictor.optimization_profile(state)["training"]
    assert predictor.encoder_calls == 2
    assert profile["encoder_misses"] == 2
    assert profile["cache_hits"] == 1
    assert profile["maximum_cached_frames"] == 8
    assert len(state["cached_features"]) == 8


def test_prediction_cache_bounds_gpu_and_cpu_chunks():
    """
    Verify prediction feature caches respect GPU and CPU chunk limits.

    Args:
        None.

    Returns:
        None.
    """
    predictor = _TinyPredictor(cpu_chunks=1)
    state = _state(12, "prediction")

    for frame_idx in (0, 4, 8, 0):
        predictor._get_image_feature(state, frame_idx, 1)

    profile = predictor.optimization_profile(state)["prediction"]
    assert predictor.encoder_calls == 3
    assert profile["encoder_misses"] == 3
    assert profile["cpu_hits"] == 1
    assert profile["maximum_gpu_chunks"] <= 2
    assert profile["maximum_cpu_chunks"] <= 1
    assert len(state["_rwkv_prediction_gpu_chunks"]) <= 2
    assert len(state["_rwkv_prediction_cpu_chunks"]) <= 1
    assert profile["position_encodings_shared"]


def test_split_3d_batch_preserves_frame_major_prompt_layout():
    """
    Verify volume selection preserves frame-major prompt alignment.

    Args:
        None.

    Returns:
        None.
    """
    batch = {
        "image": torch.arange(3.0).reshape(3, 1, 1, 1, 1),
        "mask": torch.arange(3.0).reshape(3, 1, 1, 1, 1),
        "pt_list": [["p0", "p1", "p2"], ["q0", "q1", "q2"]],
        "p_label": [["l0", "l1", "l2"], ["m0", "m1", "m2"]],
        "bbox": [["b0", "b1", "b2"], ["c0", "c1", "c2"]],
        "dataset": ["d0", "d1", "d2"],
        "constant": "kept",
    }

    selected = func_3d.split_3d_batch_by_volume(batch, 1)

    assert selected["image"].shape[0] == 1
    assert selected["image"].item() == 1.0
    assert selected["pt_list"] == [["p1"], ["q1"]]
    assert selected["p_label"] == [["l1"], ["m1"]]
    assert selected["bbox"] == [["b1"], ["c1"]]
    assert selected["dataset"] == ["d1"]
    assert selected["constant"] == "kept"


def test_per_volume_backward_averages_gradients_and_syncs_once(monkeypatch):
    """
    Verify per-volume backward averages gradients and synchronizes once per batch.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for temporarily replacing functions and
            attributes.

    Returns:
        None.
    """
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "image": torch.tensor([0.2, 0.4]).reshape(2, 1, 1, 1, 1),
        "mask": torch.zeros((2, 1, 1, 1, 1)),
        "pt_list": [[None, None]],
        "p_label": [[None, None]],
        "bbox": [[None, None]],
    }
    sync_calls = []

    def fake_train_step(
        student,
        teacher,
        active_optimizer,
        small_batch,
        config,
        scaler,
        **controls,
    ):
        """
        Accumulate a synthetic volume loss under the supplied backward controls.

        Args:
            student (SAM2VideoPredictor): Student predictor being optimized.
            teacher (SAM2VideoPredictor | None): Optional teacher predictor used for
                distillation.
            active_optimizer (torch.optim.Optimizer): Optimizer passed through the patched
                training step.
            small_batch (dict): Single-volume collated batch passed to the training-step stub.
            config (OmegaConf.DictConfig): Model, dataset, and training configuration.
            scaler (object | None): Optional mixed-precision gradient scaler.
            **controls (dict): Gradient accumulation and optimizer-update controls passed to the
                stub.

        Returns:
            dict: Successful step status and scalar loss fields.
        """
        assert controls == {
            "zero_grad": False,
            "synchronize": False,
            "clip_gradients": False,
            "optimizer_step": False,
        }
        value = small_batch["image"].reshape(()).float()
        loss = student.weight.reshape(()) * value
        loss.backward()
        scalar = float(loss.detach())

        return {
            "ok": True,
            "loss": scalar,
            "prompt_loss": scalar,
            "non_prompt_loss": None,
            "non_prompt_loss_base": None,
        }

    monkeypatch.setattr(func_3d, "train_step_3d", fake_train_step)
    monkeypatch.setattr(
        func_3d,
        "synchronize_gradients",
        lambda student: sync_calls.append(student),
    )

    output = func_3d.train_step_3d_per_volume(
        model, None, optimizer, batch, object(), None
    )

    assert output["ok"]
    assert len(sync_calls) == 1
    assert abs(output["loss"] - 0.6) < 1e-6
    assert abs(model.weight.item() - 1.97) < 1e-6


class _ValidationPredictor:
    """
    Record initialization controls and yield synthetic masks for validation tests.
    """

    device = torch.device("cpu")

    def __init__(self):
        """
        Initialize a predictor stub that records validation memory settings.

        Args:
            None.

        Returns:
            None.
        """
        self.init_options = None

    def init_state_from_tensor(self, **kwargs):
        """
        Record initialization options and count frames in the supplied tensor.

        Args:
            **kwargs (dict): Tensor-state initialization options, including imgs_tensor and
                storage controls.

        Returns:
            dict: Minimal state containing the number of frames.
        """
        self.init_options = kwargs

        return {"frames": int(kwargs["imgs_tensor"].shape[0])}

    def add_new_points_or_box(self, **kwargs):
        """
        Accept a validation prompt without changing the stub state.

        Args:
            **kwargs (dict): Prompt arguments accepted by the validation stub and otherwise
                ignored.

        Returns:
            None.
        """
        return None

    def propagate_in_video(
        self, state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        """
        Yield deterministic mask logits for each requested test frame.

        Args:
            state (dict): Minimal predictor state returned by the test initializer; ignored by
                this stub.
            start_frame_idx (int): Index of the first frame propagated by the test predictor.
            max_frame_num_to_track (int): Maximum number of frames to propagate.
            reverse (bool): Whether to reverse the stored slice order.

        Returns:
            Iterator[tuple]: Frame index, object IDs, and synthetic mask logits.
        """
        del state, start_frame_idx, reverse
        for frame_idx in range(max_frame_num_to_track):
            value = torch.full((1, 1, 2, 2), float(frame_idx))
            yield frame_idx, [1], value


def test_validate_step_3d_exposes_memory_placement_controls():
    """
    Verify 3D validation forwards storage and output-device controls.

    Args:
        None.

    Returns:
        None.
    """
    predictor = _ValidationPredictor()
    video = torch.zeros((3, 3, 2, 2))
    target = torch.zeros((3, 2, 2))
    boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0]), None, None]
    cache = {
        "chosen_frames": [0],
        "forced_first": True,
        "box": {"bbox": boxes},
    }

    output = validate_step_3d(
        predictor,
        video,
        target,
        prompt_cache=cache,
        prompt_mode="box",
        normalize_coords=True,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        output_device="cpu",
    )

    assert predictor.init_options["offload_video_to_cpu"] is True
    assert predictor.init_options["offload_state_to_cpu"] is True
    assert predictor.init_options["mode"] == "eval"
    assert output["logits"].device.type == "cpu"
    assert output["logits"][:, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_deferred_frame_normalization_matches_formula():
    """
    Verify deferred normalization matches the resident-device ImageNet formula.

    Args:
        None.

    Returns:
        None.
    """
    frames = torch.tensor(
        [
            [
                [[0.0, 255.0], [64.0, 128.0]],
                [[255.0, 0.0], [128.0, 64.0]],
                [[32.0, 96.0], [160.0, 224.0]],
            ]
        ]
    )
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    scaled = load_video_frames_from_data(
        frames,
        img_mean=mean,
        img_std=std,
        offload_video_to_cpu=True,
        defer_normalization_to_device=True,
    )
    expected = load_video_frames_from_data(
        frames,
        img_mean=mean,
        img_std=std,
        offload_video_to_cpu=True,
    )

    predictor = _TinyPredictor()
    state = {
        "device": torch.device("cpu"),
        "_rwkv_deferred_normalization": {"mean": mean, "std": std},
    }
    actual = predictor._images_to_compute_device(state, scaled)

    assert torch.equal(actual, expected)
