# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Level Zero IPC between spawned processes on one XPU."""

import gc
import importlib
import sys
import threading
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import receive, spawn_workers
from tests.distributed.weight_sharing.helpers import load_weight_sharing, make_context, make_tensor_specs
from tests.helpers.mark import hardware_marks

if sys.platform != "linux" or not hasattr(torch, "xpu"):
    pytest.skip("Level Zero IPC requires Linux and PyTorch XPU", allow_module_level=True)
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"xpu": "B60"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime():
    if not torch.xpu.is_available():
        pytest.skip("XPU IPC test requires an Intel GPU")


def runtime(endpoint, *, session_id="session"):
    package = load_weight_sharing()
    api = importlib.import_module(f"{package.__name__}.contracts")
    backend = importlib.import_module(f"{package.__name__}.backend")
    codec = importlib.import_module(f"{package.__name__}.torch_xpu_ipc_codec")
    context = make_context(api, endpoint, device_uuid=codec._device_uuid(0), session_id=session_id)
    return api, backend.TorchCudaIpcBackend(codec=codec.TorchXpuIpcCodec()), context


def _bindings(device):
    base = torch.arange(256 * 256, dtype=torch.float32, device=device).remainder_(17).div_(16).reshape(256, 256)
    first = base[:128]
    return {
        "first": first,
        "alias": first,
        "transposed": base[128:].T,
        "independent": first.clone(),
        "fp16": torch.full((32, 64), 0.125, dtype=torch.float16, device=device),
        "bf16": torch.full((32, 64), 0.25, dtype=torch.bfloat16, device=device),
        "scalar": torch.full((), 3.0, device=device),
        "empty": torch.empty(0, device=device),
    }


def _spec(api):
    return api.WeightSpec(
        "small-test-model", "revision", "tp1-final", make_tensor_specs(api, _bindings("cpu"), {"alias": "first"})
    )


def _iteration(context, index):
    return replace(context, identity=msgspec.structs.replace(context.identity, session_id=f"round-{index}"))


def _assert_released(baseline):
    gc.collect()
    current = (torch.xpu.memory_allocated(0), len(list(Path("/proc/self/fd").iterdir())))
    if baseline is not None:
        assert all(value <= limit for value, limit in zip(current, baseline)), (current, baseline)
    assert not any(thread.name == "weight-sharing-session" for thread in threading.enumerate())
    return current if baseline is None else baseline


def _provider(endpoint, rounds, control):
    api, backend, context = runtime(endpoint)
    baseline = None
    spec = _spec(api)
    for index in range(rounds):
        bindings = _bindings("xpu:0")
        exported = backend.prepare_export(bindings, spec, _iteration(context, index))
        del bindings
        snapshot = backend.session_status(exported)
        assert snapshot.storage_count == 6
        assert len(snapshot.exported_storage_ids) == 5
        descriptors = backend._sessions[exported.session_id].control.exports
        # Small storages share caching-allocator segments; each segment sends one fd.
        assert 0 < sum(d.fd_count for d in descriptors) < len(descriptors)
        assert len({(d.ref_counter_handle, d.allocation_offset_bytes) for d in descriptors}) == len(descriptors)
        control.send("ready")
        receive(control, "release")
        assert backend.release_export(exported).status == "CLOSED"
        assert backend.release_export(exported).status == "CLOSED"
        assert not Path(endpoint).exists()
        assert not backend._codec._exported
        baseline = _assert_released(baseline)
        control.send("closed")
        receive(control, "next")


def _check_views(imported):
    assert imported["first"] is imported["alias"]
    assert imported["first"].untyped_storage()._cdata == imported["transposed"].untyped_storage()._cdata
    assert imported["first"].untyped_storage()._cdata != imported["independent"].untyped_storage()._cdata
    assert imported["transposed"].storage_offset() == 128 * 256
    for name, expected in _bindings("cpu").items():
        actual = imported[name]
        assert actual.device.type == "xpu"
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _consumer(endpoint, rounds, control):
    api, backend, context = runtime(endpoint)
    baseline = None
    spec = _spec(api)
    for index in range(rounds):
        receive(control, "import")
        iteration = _iteration(context, index)
        imported = backend.open_import(endpoint, spec, iteration)
        _check_views(imported)
        output = imported["first"] @ torch.ones((256, 4), device="xpu:0")
        external_view = imported["transposed"][:, :8]
        assert backend.close_import(imported).status == "HELD"
        assert backend._codec._mappings
        del external_view
        gc.collect()
        assert backend.close_import(imported).status == "CLOSED"
        # The last carrier's DLPack deleter closes every Level Zero mapping.
        assert not backend._codec._mappings and not backend._codec._received
        expected = _bindings("cpu")["first"].sum(dim=1, keepdim=True).expand(-1, 4)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        del output
        baseline = _assert_released(baseline)
        control.send("closed")


@pytest.mark.parametrize("rounds", [1, 20], ids=["roundtrip", "repeated-lifecycle"])
def test_two_spawned_processes_share_xpu_storage(tmp_path: Path, rounds):
    endpoint = str(tmp_path / "weights.sock")
    with spawn_workers((_provider, (endpoint, rounds)), (_consumer, (endpoint, rounds))) as workers:
        (_, provider), (_, consumer) = workers
        for _ in range(rounds):
            receive(provider, "ready")
            consumer.send("import")
            receive(consumer, "closed")
            provider.send("release")
            receive(provider, "closed")
            provider.send("next")


def _large_spec(api, nbytes):
    return api.WeightSpec("large-test", "revision", "flat", (api.TensorSpec("weight", "float32", (nbytes // 4,)),))


def _large_provider(endpoint, nbytes, control):
    api, backend, context = runtime(endpoint)
    weight = torch.ones(nbytes // 4, device="xpu:0")
    exported = backend.prepare_export({"weight": weight}, _large_spec(api, nbytes), context)
    control.send("ready")
    receive(control, "check")
    # The consumer wrote through its mapping: the memory is shared, not copied.
    assert weight[7].item() == 5.0
    control.send("checked")
    receive(control, "release")
    assert backend.release_export(exported).status == "CLOSED"


def _memory_consumer(endpoint, nbytes, control):
    api, backend, context = runtime(endpoint)
    warmup = torch.ones(1024, device="xpu:0")
    torch.xpu.synchronize(0)
    del warmup
    torch.xpu.empty_cache()
    receive(control, "import")
    before = torch.xpu.memory_allocated(0)
    torch.xpu.reset_peak_memory_stats(0)
    imported = backend.open_import(endpoint, _large_spec(api, nbytes), context)
    peak = torch.xpu.max_memory_allocated(0) - before
    assert peak < nbytes // 4
    assert imported["weight"].sum().item() == nbytes // 4
    imported["weight"][7] = 5.0
    torch.xpu.synchronize(0)
    control.send(peak)
    receive(control, "close")
    assert backend.close_import(imported).status == "CLOSED"
    assert torch.xpu.memory_allocated(0) == before
    control.send("closed")


def test_import_maps_without_a_second_weight_copy(tmp_path: Path, record_property):
    nbytes = 256 * 1024 * 1024
    endpoint = str(tmp_path / "large.sock")
    with spawn_workers((_large_provider, (endpoint, nbytes)), (_memory_consumer, (endpoint, nbytes))) as workers:
        (_, provider), (_, consumer) = workers
        receive(provider, "ready")
        consumer.send("import")
        record_property("import_peak_bytes", receive(consumer))
        provider.send("check")
        receive(provider, "checked")
        consumer.send("close")
        receive(consumer, "closed")
        provider.send("release")


@pytest.mark.parametrize("rounds", [1, 20], ids=["single", "repeated-lifecycle"])
def test_unexposed_export_puts_ipc_handles(tmp_path: Path, rounds):
    api, backend, context = runtime(str(tmp_path / "unexposed.sock"))
    weights = {"weight": torch.arange(16, dtype=torch.float32, device="xpu:0")}
    spec = api.WeightSpec("m", "r", "l", (api.TensorSpec("weight", "float32", (16,), (1,)),))
    for index in range(rounds):
        exported = backend.prepare_export(weights, spec, _iteration(context, index))
        assert backend._codec._exported
        assert backend.release_export(exported).status == "CLOSED"
        assert not backend._codec._exported
    torch.testing.assert_close(weights["weight"].cpu(), torch.arange(16, dtype=torch.float32))
