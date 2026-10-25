# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unsupported real CUDA allocators are rejected before exporting IPC handles."""

from pathlib import Path

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import receive, require_cuda, runtime, spawn_workers
from tests.helpers.mark import hardware_marks

require_cuda()

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime():
    require_cuda(1)


def _unsupported_allocator(endpoint, mode, expected_error, control):
    api, backend, context = runtime(endpoint)
    if mode == "private_pool":
        # Keep the pool alive until rejection has been checked.
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            weight = torch.ones(1024, dtype=torch.float32, device="cuda:0")
        segments = torch.cuda.memory._snapshot()["segments"]
        owners = [
            segment
            for segment in segments
            if segment["address"] <= weight.data_ptr() < segment["address"] + segment["total_size"]
        ]
        assert owners and all(tuple(segment["segment_pool_id"]) != (0, 0) for segment in owners)
    else:
        weight = torch.ones(1024, dtype=torch.float32, device="cuda:0")

    spec = api.WeightSpec("guard-test", "revision", "flat", (api.TensorSpec("weight", "float32", (1024,)),))
    with pytest.raises(api.WeightSharingError) as error:
        backend.prepare_export({"weight": weight}, spec, context)
    assert isinstance(error.value.__cause__, api.UnsupportedError)
    assert expected_error in str(error.value.__cause__)
    assert backend.session_status(context.identity.session_id).exported_storage_ids == ()
    assert not Path(endpoint).exists()
    control.send("checked")


@pytest.mark.parametrize(
    ("mode", "allocator_conf", "expected_error"),
    [
        pytest.param("async_allocator", "backend:cudaMallocAsync", "only the native", id="cudaMallocAsync"),
        pytest.param(
            "expandable_segments",
            "backend:native,expandable_segments:True",
            "expandable segments",
            id="expandable_segments",
        ),
        pytest.param(
            "private_pool", "backend:native,expandable_segments:False", "outside private pools", id="private_pool"
        ),
    ],
)
def test_unsupported_allocator_rejected_before_export(
    tmp_path: Path, monkeypatch, mode, allocator_conf, expected_error
):
    # Allocator configuration is read on import/initialization, so each case
    # needs a fresh spawned process with its environment set before startup.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with spawn_workers(
        (
            _unsupported_allocator,
            (str(tmp_path / "a.sock"), mode, expected_error),
            {"PYTORCH_ALLOC_CONF": allocator_conf},
        )
    ) as workers:
        ((_, control),) = workers
        receive(control, "checked")
