# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator

logger = init_logger(__name__)


def _dispose_parameter_storage(param: nn.Parameter) -> None:
    with torch.no_grad():
        param.set_(torch.empty(0, dtype=param.dtype, device=param.device))


@dataclass
class _SharedWeightSlot:
    tensor: torch.Tensor
    data_layer_idx: int = -1
    work: dist.Work | None = None


@dataclass
class _SharedWeightLayer:
    layer_idx: int
    parameter: nn.Parameter
    owner_rank: int
    resident_weight: torch.Tensor | None


class SharedWeightSeries:
    """Share one isomorphic weight series with a fixed double buffer.

    Complete layer weights are assigned round-robin to ranks. Non-owner ranks
    keep no resident storage for that layer and receive the weight into one of
    two stable slots before executing the layer.
    """

    def __init__(self, name: str, group: GroupCoordinator) -> None:
        self.name = name
        self.group = group
        self.layers: list[_SharedWeightLayer] = []
        self.slots: list[_SharedWeightSlot] = []
        self._initialized = False

    def register(self, layer_idx: int, parameter: nn.Parameter) -> None:
        if self._initialized:
            raise RuntimeError(f"Cannot register {self.name} layer after initialization")
        owner_rank = layer_idx % self.group.world_size
        self.layers.append(
            _SharedWeightLayer(
                layer_idx=layer_idx,
                parameter=parameter,
                owner_rank=owner_rank,
                resident_weight=None,
            )
        )

    def initialize(self) -> None:
        if self._initialized:
            return
        if not self.layers:
            raise ValueError(f"Shared weight series {self.name!r} has no layers")

        self.layers.sort(key=lambda layer: layer.layer_idx)
        reference = self.layers[0].parameter
        reference_shape = reference.shape
        reference_stride = reference.stride()
        for expected_idx, layer in enumerate(self.layers):
            if layer.layer_idx != expected_idx:
                raise ValueError(
                    f"Shared weight series {self.name!r} requires consecutive layer indices; "
                    f"expected {expected_idx}, got {layer.layer_idx}"
                )
            if layer.parameter.shape != reference_shape or layer.parameter.stride() != reference_stride:
                raise ValueError(
                    f"Shared weight series {self.name!r} requires isomorphic weights; "
                    f"layer {layer.layer_idx} has shape/stride "
                    f"{tuple(layer.parameter.shape)}/{layer.parameter.stride()}, expected "
                    f"{tuple(reference_shape)}/{reference_stride}"
                )

            if layer.owner_rank == self.group.rank_in_group:
                layer.resident_weight = layer.parameter.detach()
            else:
                _dispose_parameter_storage(layer.parameter)

        self.slots = [
            _SharedWeightSlot(
                torch.empty_strided(
                    reference_shape,
                    reference_stride,
                    dtype=reference.dtype,
                    device=reference.device,
                )
            )
            for _ in range(2)
        ]
        self._initialized = True

    def _layer(self, layer_idx: int) -> _SharedWeightLayer:
        return self.layers[layer_idx]

    def _slot(self, layer_idx: int) -> _SharedWeightSlot:
        return self.slots[layer_idx % len(self.slots)]

    def fetch(self, layer_idx: int, *, async_op: bool) -> None:
        if not self._initialized:
            raise RuntimeError(f"Shared weight series {self.name!r} is not initialized")
        layer = self._layer(layer_idx)
        slot = self._slot(layer_idx)

        if slot.work is not None:
            slot.work.wait()
            slot.work = None

        if layer.owner_rank == self.group.rank_in_group:
            assert layer.resident_weight is not None
            communication_tensor = layer.resident_weight
        else:
            with torch.no_grad():
                layer.parameter.set_(slot.tensor)
            communication_tensor = slot.tensor

        slot.data_layer_idx = layer_idx
        if self.group.world_size == 1:
            return
        slot.work = dist.broadcast(
            communication_tensor,
            src=self.group.ranks[layer.owner_rank],
            group=self.group.device_group,
            async_op=async_op,
        )
        if not async_op:
            assert slot.work is None

    def wait(self, layer_idx: int) -> None:
        layer = self._layer(layer_idx)
        if layer.owner_rank == self.group.rank_in_group:
            return
        slot = self._slot(layer_idx)
        if slot.data_layer_idx != layer_idx:
            raise RuntimeError(
                f"Shared weight {self.name} layer {layer_idx} is not prefetched; "
                f"slot contains layer {slot.data_layer_idx}"
            )
        if slot.work is not None:
            slot.work.wait()
            slot.work = None


class SharedAttentionWeightManager:
    """Manage Hunyuan attention QKV/O resident weights and prefetch slots."""

    def __init__(self, group: GroupCoordinator) -> None:
        self.group = group
        self.qkv = SharedWeightSeries("qkv_proj", group)
        self.o = SharedWeightSeries("o_proj", group)
        self.num_layers = 0
        self._initialized = False

    def register_layer(self, layer_idx: int, qkv_proj: nn.Module, o_proj: nn.Module) -> None:
        self.qkv.register(layer_idx, qkv_proj.weight)
        self.o.register(layer_idx, o_proj.weight)
        self.num_layers = max(self.num_layers, layer_idx + 1)

    def initialize(self) -> None:
        if self._initialized:
            return
        self.qkv.initialize()
        self.o.initialize()
        self._initialized = True
        logger.info(
            "Initialized shared attention weights: layers=%d, group_size=%d, qkv_slots=2, o_slots=2",
            self.num_layers,
            self.group.world_size,
        )

    def begin_forward(self) -> None:
        self.initialize()
        self.qkv.fetch(0, async_op=False)
        self.o.fetch(0, async_op=False)

    def wait_layer(self, layer_idx: int) -> None:
        self.qkv.wait(layer_idx)
        self.o.wait(layer_idx)

    @torch.compiler.disable
    def prefetch_next(self, layer_idx: int) -> None:
        next_layer_idx = layer_idx + 1
        if next_layer_idx >= self.num_layers:
            return
        self.qkv.fetch(next_layer_idx, async_op=True)
        self.o.fetch(next_layer_idx, async_op=True)
