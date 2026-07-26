# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed.shared_weight import SharedAttentionWeightManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _make_group(*, rank: int, world_size: int = 2):
    return SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        ranks=list(range(world_size)),
        device_group=object(),
    )


def test_shared_attention_weights_use_stable_double_buffer(monkeypatch):
    broadcasts = []

    def fake_broadcast(tensor, *, src, group, async_op):
        broadcasts.append((tensor, src, group, async_op))
        return _FakeWork() if async_op else None

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    manager = SharedAttentionWeightManager(_make_group(rank=1))
    qkv_layers = [nn.Linear(4, 6, bias=False) for _ in range(3)]
    o_layers = [nn.Linear(6, 4, bias=False) for _ in range(3)]
    for layer_idx, (qkv, o) in enumerate(zip(qkv_layers, o_layers)):
        manager.register_layer(layer_idx, qkv, o)

    manager.begin_forward()
    qkv_slot0_ptr = qkv_layers[0].weight.data_ptr()
    o_slot0_ptr = o_layers[0].weight.data_ptr()
    assert qkv_layers[1].weight.numel() > 0  # rank 1 owns layer 1
    assert qkv_layers[2].weight.numel() == 0

    manager.prefetch_next(0)
    manager.wait_layer(1)
    manager.prefetch_next(1)
    manager.wait_layer(2)

    assert qkv_layers[2].weight.data_ptr() == qkv_slot0_ptr
    assert o_layers[2].weight.data_ptr() == o_slot0_ptr
    assert [entry[1] for entry in broadcasts] == [0, 0, 1, 1, 0, 0]
    assert [entry[3] for entry in broadcasts] == [False, False, True, True, True, True]


def test_shared_attention_weights_reject_non_isomorphic_layers(monkeypatch):
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    manager = SharedAttentionWeightManager(_make_group(rank=0))
    manager.register_layer(0, nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False))
    manager.register_layer(1, nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))

    with pytest.raises(ValueError, match="isomorphic"):
        manager.initialize()
