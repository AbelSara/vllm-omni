# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical GPU pairing with different process-local CUDA ordinals."""

import importlib
from pathlib import Path

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import receive, require_cuda, runtime, spawn_workers
from tests.distributed.weight_sharing.helpers import load_weight_sharing
from tests.helpers.mark import hardware_marks

require_cuda()
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=2)]


@pytest.fixture
def gpu_uuids():
    require_cuda(2)
    codec = importlib.import_module(f"{load_weight_sharing().__name__}.torch_cuda_ipc_codec")
    return tuple(codec._device_uuid(index) for index in range(2))


def _spec(api):
    return api.WeightSpec("paired-model", "revision", "tp2-final", (api.TensorSpec("weight", "float32", (8, 8)),))


def _provider(endpoint, rank, control):
    api, backend, context = runtime(endpoint, rank=rank, tp_size=2, device_index=rank)
    weights = {"weight": torch.full((8, 8), rank + 1.0, device=f"cuda:{rank}")}
    exported = backend.prepare_export(weights, _spec(api), context)
    control.send(("ready", context.identity.device_uuid))
    receive(control, "release")
    assert backend.release_export(exported).status == "CLOSED"
    control.send("closed")


def _consumer(endpoint, rank, mismatch, control):
    receive(control, "import")
    device_index = rank if mismatch == "device" else 1 - rank
    api, backend, context = runtime(
        endpoint,
        rank=1 - rank if mismatch == "rank" else rank,
        tp_size=2,
        device_index=device_index,
    )
    if mismatch:
        calls = 0
        original_import = backend._codec.import_storage

        def counted_import(descriptor, index):
            nonlocal calls
            calls += 1
            return original_import(descriptor, index)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(backend._codec, "import_storage", counted_import)
            with pytest.raises(api.WeightSharingError):
                backend.open_import(endpoint, _spec(api), context)
        assert calls == 0
        snapshot = backend.session_status(context.identity.session_id)
        assert snapshot.state == api.SessionState.POISONED
        assert snapshot.imported_storage_ids == ()
        assert snapshot.active_storage_id is None
    else:
        imported = backend.open_import(endpoint, _spec(api), context)
        assert imported["weight"].device.index == device_index
        output = torch.ones((2, 8), device=f"cuda:{device_index}") @ imported["weight"]
        torch.testing.assert_close(output.cpu(), torch.full((2, 8), 8.0 * (rank + 1)))
        assert backend.close_import(imported).status == "CLOSED"
    control.send("done")


def _check_pairs(tmp_path, gpu_uuids, ranks, mismatch=None):
    workers = []
    for rank in ranks:
        endpoint = str(tmp_path / f"rank-{rank}.sock")
        workers.extend(
            [
                (_provider, (endpoint, rank), {"CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids)}),
                (_consumer, (endpoint, rank, mismatch), {"CUDA_VISIBLE_DEVICES": ",".join(reversed(gpu_uuids))}),
            ]
        )
    with spawn_workers(*workers) as started:
        providers, consumers = started[::2], started[1::2]
        for rank, (_, control) in zip(ranks, providers):
            receive(control, ("ready", gpu_uuids[rank]))
        for _, control in consumers:
            control.send("import")
        for process, control in consumers:
            receive(control, "done")
            process.join(timeout=30)
            assert process.exitcode == 0
        for _, control in providers:
            control.send("release")
        for _, control in providers:
            receive(control, "closed")
    assert not list(tmp_path.glob("*.sock"))


def test_rank_pairs_share_with_reversed_visible_device_order(tmp_path: Path, gpu_uuids):
    _check_pairs(tmp_path, gpu_uuids, (0, 1))


@pytest.mark.parametrize("mismatch", ["rank", "device"])
def test_wrong_rank_or_physical_gpu_is_rejected_before_import(tmp_path: Path, gpu_uuids, mismatch):
    _check_pairs(tmp_path, gpu_uuids, (0,), mismatch)
