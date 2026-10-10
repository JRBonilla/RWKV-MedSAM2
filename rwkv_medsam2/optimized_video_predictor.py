"""Opt-in memory and throughput optimizations for SAM2 video prediction.

The class deliberately adds no parameters or buffers, so checkpoints remain fully
compatible with :class:`sam2.sam2_video_predictor.SAM2VideoPredictor`.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames_from_data


class OptimizedSAM2VideoPredictor(SAM2VideoPredictor):
    """
    SAM2 predictor with batched frame encoding and bounded feature caches.

    Training encodes nearby frames together and keeps a small LRU cache whose
    tensors retain their autograd graphs. Prediction keeps only a few feature
    chunks on the GPU and spills older chunks to a bounded pinned-CPU cache.
    None of these controls change model parameters or mask-generation logic.
    """

    def __init__(
        self,
        image_feature_chunk_size: int = 4,
        training_feature_cache_frames: int = 8,
        prediction_gpu_cache_chunks: int = 2,
        prediction_cpu_cache_chunks: int = 64,
        prediction_pin_cpu_features: bool = True,
        **kwargs,
    ):
        """
        Configure batched frame encoding and bounded training/prediction caches.

        Args:
            image_feature_chunk_size (int): Maximum number of adjacent frames encoded together.
            training_feature_cache_frames (int): Maximum cached training frames; must cover at
                least one encoding chunk.
            prediction_gpu_cache_chunks (int): Maximum prediction feature chunks retained on the
                GPU.
            prediction_cpu_cache_chunks (int): Maximum CPU feature chunks; zero disables storage
                and -1 makes it unbounded.
            prediction_pin_cpu_features (bool): Whether prediction features use pinned CPU
                memory when CUDA is available.
            **kwargs (dict): Model construction options forwarded to SAM2VideoPredictor.

        Returns:
            None.
        """
        super().__init__(**kwargs)
        self.image_feature_chunk_size = int(image_feature_chunk_size)
        self.training_feature_cache_frames = int(training_feature_cache_frames)
        self.prediction_gpu_cache_chunks = int(prediction_gpu_cache_chunks)
        self.prediction_cpu_cache_chunks = int(prediction_cpu_cache_chunks)
        self.prediction_pin_cpu_features = bool(prediction_pin_cpu_features)

        if self.image_feature_chunk_size < 1:
            raise ValueError("image_feature_chunk_size must be at least 1")
        if self.training_feature_cache_frames < self.image_feature_chunk_size:
            raise ValueError(
                "training_feature_cache_frames must be at least "
                "image_feature_chunk_size"
            )
        if self.prediction_gpu_cache_chunks < 1:
            raise ValueError("prediction_gpu_cache_chunks must be at least 1")
        if self.prediction_cpu_cache_chunks < -1:
            raise ValueError(
                "prediction_cpu_cache_chunks must be -1 (unbounded), 0 (disabled), "
                "or a positive limit"
            )

    def init_state(self, *args, **kwargs):
        """
        Initialize a video state and enable prediction feature caching.

        Args:
            *args (tuple): Positional video-state initialization arguments forwarded to the
                parent predictor.
            **kwargs (dict): Keyword video-state initialization options forwarded to the parent
                predictor.

        Returns:
            dict: Parent inference state with prediction cache mode selected.
        """
        state = super().init_state(*args, **kwargs)
        state["_rwkv_feature_cache_mode"] = "prediction"

        return state

    def init_state_from_tensor(
        self,
        imgs_tensor: torch.Tensor,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        mode: str = "train",
    ):
        # The parent path normalizes CPU-offloaded frames on the CPU. Over long
        # propagation sequences, its tiny numerical difference from resident
        # GPU normalization can compound. Keep scaled [0,255] frames on CPU and
        # reproduce the resident normalization arithmetic per GPU chunk.
        """
        Initialize tensor-backed state with cache mode and optional deferred normalization.

        Args:
            imgs_tensor (torch.Tensor): Image sequence with shape [T, C, H, W].
            img_mean (tuple[float, float, float]): Per-channel ImageNet means for RGB
                normalization.
            img_std (tuple[float, float, float]): Per-channel ImageNet standard deviations for
                RGB normalization.
            offload_video_to_cpu (bool): Whether image frames remain in CPU memory during
                propagation.
            offload_state_to_cpu (bool): Whether propagation state is stored in CPU memory.
            mode (str): Execution mode: train retains gradients; eval enables prediction
                caching.

        Returns:
            dict: Training or prediction state, including storage and normalization settings.
        """
        if mode != "eval" or not offload_video_to_cpu:
            state = super().init_state_from_tensor(
                imgs_tensor=imgs_tensor,
                img_mean=img_mean,
                img_std=img_std,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                mode=mode,
            )
            state["_rwkv_feature_cache_mode"] = (
                "training" if mode == "train" else "prediction"
            )

            return state

        with torch.inference_mode():
            images = load_video_frames_from_data(
                imgs_tensor=imgs_tensor.detach().to("cpu"),
                img_mean=img_mean,
                img_std=img_std,
                offload_video_to_cpu=True,
                defer_normalization_to_device=True,
            )
            num_frames, _, video_height, video_width = images.shape
            state = {
                "images": images,
                "num_frames": int(num_frames),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": bool(offload_state_to_cpu),
                "video_height": int(video_height),
                "video_width": int(video_width),
                "device": self.device,
                "storage_device": (
                    torch.device("cpu") if offload_state_to_cpu else self.device
                ),
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": OrderedDict(),
                "obj_idx_to_id": OrderedDict(),
                "obj_ids": [],
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
                "_rwkv_feature_cache_mode": "prediction",
                "_rwkv_deferred_normalization": {
                    "mean": tuple(float(value) for value in img_mean),
                    "std": tuple(float(value) for value in img_std),
                },
            }
            self._get_image_feature(state, frame_idx=0, batch_size=1)

            return state

    @staticmethod
    def optimization_profile(inference_state: dict) -> dict:
        """
        Return cache counters useful for smoke tests and operational logging.

        Args:
            inference_state (dict): Predictor state holding images, feature caches, and
                compute/storage devices.

        Returns:
            dict: Cache mode and copies of training and prediction counters.
        """
        return {
            "mode": inference_state.get("_rwkv_feature_cache_mode"),
            "training": dict(inference_state.get("_rwkv_training_cache_profile", {})),
            "prediction": dict(
                inference_state.get("_rwkv_prediction_cache_profile", {})
            ),
        }

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Retrieve frame features through the active training or prediction cache.

        Args:
            inference_state (dict): Predictor state holding images, feature caches, and
                compute/storage devices.
            frame_idx (int): Frame index whose cached features are requested.
            batch_size (int): Number of samples or objects in the batch.

        Returns:
            tuple: Expanded image, backbone output, vision features, positions, and feature
                sizes.
        """
        mode = inference_state.get("_rwkv_feature_cache_mode")
        if mode is None:
            # The parent initializer warms frame zero before it returns the state.
            mode = "training" if torch.is_grad_enabled() else "prediction"
        if mode == "training":
            return self._get_training_image_feature(
                inference_state, frame_idx, batch_size
            )

        return self._get_prediction_image_feature(
            inference_state, frame_idx, batch_size
        )

    def _images_to_compute_device(
        self, inference_state: dict, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Move images to the compute device and apply deferred normalization if needed.

        Args:
            inference_state (dict): Predictor state holding images, feature caches, and
                compute/storage devices.
            images (torch.Tensor): Batch of image tensors to encode or move to the compute
                device.

        Returns:
            torch.Tensor: Float32 image batch on the state's compute device.
        """
        device = inference_state["device"]
        images = images.to(device=device, non_blocking=True).float()
        normalization = inference_state.get("_rwkv_deferred_normalization")

        if normalization is None:
            return images

        tensors = inference_state.setdefault("_rwkv_deferred_normalization_tensors", {})
        if not tensors:
            tensors["mean"] = torch.tensor(
                normalization["mean"], dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)
            tensors["std"] = torch.tensor(
                normalization["std"], dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)

        images = images / 255.0
        return (images - tensors["mean"]) / tensors["std"]

    def _expand_and_prepare(self, image, backbone_out, batch_size):
        """
        Expand one frame's image and backbone features for the object batch.

        Args:
            image (torch.Tensor): Single-frame image tensor to expand for the object batch.
            backbone_out (dict): Image-encoder feature pyramid and positional encodings.
            batch_size (int): Number of samples or objects in the batch.

        Returns:
            tuple: Expanded image, backbone output, vision features, positions, and feature
                sizes.
        """
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }

        for index, feature in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][index] = feature.expand(
                batch_size, -1, -1, -1
            )
        for index, position in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][index] = position.expand(
                batch_size, -1, -1, -1
            )

        features = self._prepare_backbone_features(expanded_backbone_out)
        return (expanded_image,) + features

    def _get_training_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Encode nearby frames together and reuse graph-preserving cached features.

        Args:
            inference_state (dict): Predictor state holding images, feature caches, and
                compute/storage devices.
            frame_idx (int): Frame index whose cached features are requested.
            batch_size (int): Number of samples or objects in the batch.

        Returns:
            tuple: Expanded image and prepared backbone features for the requested frame.
        """
        frame_idx = int(frame_idx)
        cache = inference_state["cached_features"]
        image, backbone_out = cache.get(frame_idx, (None, None))
        lru = inference_state.setdefault("_rwkv_training_feature_lru", [])
        profile = inference_state.setdefault(
            "_rwkv_training_cache_profile",
            {
                "chunk_size": self.image_feature_chunk_size,
                "frame_limit": self.training_feature_cache_frames,
                "encoder_misses": 0,
                "cache_hits": 0,
                "maximum_cached_frames": 0,
            },
        )

        if backbone_out is None:
            frame_count = int(inference_state["num_frames"])
            start = (frame_idx // self.image_feature_chunk_size) * (
                self.image_feature_chunk_size
            )
            end = min(start + self.image_feature_chunk_size, frame_count)
            device = inference_state["device"]
            images = (
                inference_state["images"][start:end]
                .to(device=device, non_blocking=True)
                .float()
            )
            batched_out = self.forward_image(images)
            count = end - start
            profile["encoder_misses"] += 1

            for offset in range(count):
                key = start + offset
                # Keep these views attached to the batched encoder graph so
                # cached training frames still propagate gradients.
                frame_out = {
                    "backbone_fpn": [
                        feature[offset : offset + 1]
                        for feature in batched_out["backbone_fpn"]
                    ],
                    "vision_pos_enc": [
                        (
                            position[offset : offset + 1]
                            if int(position.shape[0]) == count
                            else position
                        )
                        for position in batched_out["vision_pos_enc"]
                    ],
                }
                cache[key] = (images[offset : offset + 1], frame_out)

                if key in lru:
                    lru.remove(key)

                lru.append(key)
                while len(cache) > self.training_feature_cache_frames:
                    cache.pop(lru.pop(0), None)

            image, backbone_out = cache[frame_idx]
        else:
            profile["cache_hits"] += 1
            if frame_idx in lru:
                lru.remove(frame_idx)

            lru.append(frame_idx)

        profile["maximum_cached_frames"] = max(
            profile["maximum_cached_frames"], len(cache)
        )
        return self._expand_and_prepare(image, backbone_out, batch_size)

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        """
        Compute a tensor's logical storage size in bytes.

        Args:
            tensor (torch.Tensor): Tensor whose byte size is measured.

        Returns:
            int: Element count multiplied by element size.
        """
        return int(tensor.numel() * tensor.element_size())

    def _copy_feature_to_cpu(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Copy a feature tensor to optionally pinned CPU storage.

        Args:
            feature (torch.Tensor): Feature tensor to copy or inspect.

        Returns:
            torch.Tensor: CPU feature copy ready for prediction-cache reuse.
        """
        pin_memory = self.prediction_pin_cpu_features and torch.cuda.is_available()
        result = torch.empty_like(feature, device="cpu", pin_memory=pin_memory)
        # Complete the copy before an evicted GPU tensor can be released or
        # its CPU copy can be reused by a later cache hit.
        result.copy_(feature, non_blocking=False)

        return result

    @torch.inference_mode()
    def _get_prediction_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Retrieve batched prediction features from bounded GPU and CPU caches.

        Args:
            inference_state (dict): Predictor state holding images, feature caches, and
                compute/storage devices.
            frame_idx (int): Frame index whose cached features are requested.
            batch_size (int): Number of samples or objects in the batch.

        Returns:
            tuple: Expanded image and prepared backbone features for the requested frame.
        """
        frame_idx = int(frame_idx)
        frame_count = int(inference_state["num_frames"])
        start = (frame_idx // self.image_feature_chunk_size) * (
            self.image_feature_chunk_size
        )
        end = min(start + self.image_feature_chunk_size, frame_count)
        offset = frame_idx - start
        device = inference_state["device"]

        gpu_chunks = inference_state.setdefault(
            "_rwkv_prediction_gpu_chunks", OrderedDict()
        )
        cpu_chunks = inference_state.setdefault(
            "_rwkv_prediction_cpu_chunks", OrderedDict()
        )
        profile = inference_state.setdefault(
            "_rwkv_prediction_cache_profile",
            {
                "chunk_size": self.image_feature_chunk_size,
                "gpu_chunk_limit": self.prediction_gpu_cache_chunks,
                "cpu_chunk_limit": self.prediction_cpu_cache_chunks,
                "gpu_hits": 0,
                "cpu_hits": 0,
                "encoder_misses": 0,
                "maximum_gpu_chunks": 0,
                "maximum_cpu_chunks": 0,
                "current_cpu_feature_bytes": 0,
                "maximum_cpu_feature_bytes": 0,
                "gpu_to_cpu_bytes": 0,
                "cpu_to_gpu_bytes": 0,
                "position_encodings_shared": False,
            },
        )

        if start in gpu_chunks:
            chunk_features = gpu_chunks.pop(start)
            gpu_chunks[start] = chunk_features
            profile["gpu_hits"] += 1
        elif start in cpu_chunks:
            cpu_features = cpu_chunks.pop(start)
            cpu_chunks[start] = cpu_features
            chunk_features = [
                feature.to(device=device, non_blocking=True) for feature in cpu_features
            ]
            gpu_chunks[start] = chunk_features
            profile["cpu_hits"] += 1
            profile["cpu_to_gpu_bytes"] += sum(
                self._tensor_bytes(feature) for feature in cpu_features
            )
        else:
            images = self._images_to_compute_device(
                inference_state,
                inference_state["images"][start:end],
            )
            backbone_out = self.forward_image(images)
            chunk_features = backbone_out["backbone_fpn"]
            gpu_chunks[start] = chunk_features
            profile["encoder_misses"] += 1

            if "_rwkv_shared_position_encodings" not in inference_state:
                shared_positions = []
                for position in backbone_out["vision_pos_enc"]:
                    reference = position[0:1]
                    if int(position.shape[0]) > 1 and not torch.equal(
                        position, reference.expand_as(position)
                    ):
                        raise RuntimeError(
                            "position encoding differs across images in a feature chunk"
                        )

                    shared_positions.append(reference)

                inference_state["_rwkv_shared_position_encodings"] = shared_positions
                profile["position_encodings_shared"] = True

        while len(gpu_chunks) > self.prediction_gpu_cache_chunks:
            victim_start, victim_features = gpu_chunks.popitem(last=False)
            if victim_start in cpu_chunks:
                cpu_chunks.move_to_end(victim_start)
                continue
            if self.prediction_cpu_cache_chunks == 0:
                continue

            cpu_features = [
                self._copy_feature_to_cpu(feature) for feature in victim_features
            ]
            copied_bytes = sum(self._tensor_bytes(feature) for feature in cpu_features)
            cpu_chunks[victim_start] = cpu_features
            profile["current_cpu_feature_bytes"] += copied_bytes
            profile["gpu_to_cpu_bytes"] += copied_bytes

            while (
                self.prediction_cpu_cache_chunks > 0
                and len(cpu_chunks) > self.prediction_cpu_cache_chunks
            ):
                _, removed = cpu_chunks.popitem(last=False)
                profile["current_cpu_feature_bytes"] -= sum(
                    self._tensor_bytes(feature) for feature in removed
                )

        profile["maximum_gpu_chunks"] = max(
            profile["maximum_gpu_chunks"], len(gpu_chunks)
        )
        profile["maximum_cpu_chunks"] = max(
            profile["maximum_cpu_chunks"], len(cpu_chunks)
        )
        profile["maximum_cpu_feature_bytes"] = max(
            profile["maximum_cpu_feature_bytes"],
            profile["current_cpu_feature_bytes"],
        )

        chunk_features = gpu_chunks[start]
        backbone_out = {
            "backbone_fpn": [
                feature[offset : offset + 1] for feature in chunk_features
            ],
            "vision_pos_enc": inference_state["_rwkv_shared_position_encodings"].copy(),
        }
        image = self._images_to_compute_device(
            inference_state,
            inference_state["images"][frame_idx : frame_idx + 1],
        )

        return self._expand_and_prepare(image, backbone_out, batch_size)
