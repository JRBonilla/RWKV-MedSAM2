"""Distributed-training helpers for custom SAM2 method-based training loops."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    """Initialize NCCL when launched by torchrun and select the local CUDA device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("Headless training requires CUDA.")

    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device("cuda", local_rank))

    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Tear down the active process group without risking an error-path barrier hang."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize ranks when distributed training is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def broadcast_string(value: str, source: int = 0) -> str:
    """Broadcast one string from the source rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    values = [value if dist.get_rank() == source else ""]
    dist.broadcast_object_list(values, src=source)
    return str(values[0])


class DistributedBatchShard(BatchSampler):
    """Shard each deterministic global balanced batch evenly across ranks."""

    def __init__(self, batch_sampler, rank: int, world_size: int):
        self.batch_sampler = batch_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)

        global_batch_size = int(getattr(batch_sampler, "batch_size", 0))
        if global_batch_size <= 0:
            raise ValueError("The wrapped batch sampler must expose a positive batch_size.")
        if global_batch_size % self.world_size != 0:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"world size {self.world_size}."
            )
        self.batch_size = global_batch_size // self.world_size
        self.drop_last = True

    def __iter__(self):
        for global_batch in self.batch_sampler:
            if len(global_batch) % self.world_size != 0:
                raise RuntimeError(
                    f"Balanced batch of size {len(global_batch)} cannot be evenly "
                    f"sharded over {self.world_size} ranks."
                )
            local_batch = global_batch[self.rank :: self.world_size]
            if len(local_batch) != self.batch_size:
                raise RuntimeError(
                    f"Rank {self.rank} received {len(local_batch)} samples; "
                    f"expected {self.batch_size}."
                )
            yield local_batch

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


def synchronize_gradients(model: torch.nn.Module) -> None:
    """Average gradients across ranks for method-based models that bypass DDP.forward."""
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return

    device = parameters[0].device
    present = torch.tensor(
        [1 if parameter.grad is not None else 0 for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX)

    active = [
        parameter
        for parameter, globally_present in zip(parameters, present.tolist())
        if globally_present
    ]
    by_dtype = {}
    for parameter in active:
        by_dtype.setdefault(parameter.dtype, []).append(parameter)

    for dtype_parameters in by_dtype.values():
        gradients = [
            parameter.grad.contiguous()
            if parameter.grad is not None
            else torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in dtype_parameters
        ]
        flat = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        synchronized = torch._utils._unflatten_dense_tensors(flat, gradients)
        for parameter, gradient in zip(dtype_parameters, synchronized):
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
