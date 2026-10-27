# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA references survive uncertain imports and broken metadata sessions."""

import importlib
import os
import weakref
from pathlib import Path

import pytest
import torch

from tests.distributed.weight_sharing.cuda_helpers import (
    assert_export_allocations,
    receive,
    require_cuda,
    runtime,
    spawn_workers,
)
from tests.distributed.weight_sharing.helpers import load_weight_sharing
from tests.helpers.mark import hardware_marks

require_cuda()

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.fixture(autouse=True)
def supported_runtime(ipc_allocator):
    require_cuda(1)


def _spec(api):
    return api.WeightSpec(
        "failure-test", "revision", "final", tuple(api.TensorSpec(name, "float32", (32,)) for name in ("a", "b"))
    )


def _provider(endpoint, fault, control):
    api, backend, context = runtime(endpoint)
    channel = importlib.import_module(f"{load_weight_sharing().__name__}.channel")
    original_send = channel.FramedConnection.send
    dropped = []
    cancellations = []

    def send(connection, message):
        if fault == "provider_exit" and message.op == "bundle":
            os._exit(0)  # No handle has reached the consumer yet.
        lost_op = {"import_ack": "accepted", "release_ack": "released"}.get(fault)
        if message.op == lost_op:
            dropped.append(message.op)
            connection.close()
            raise OSError("injected lost acknowledgement")
        original_send(connection, message)

    def cancel(*args):
        cancellations.append(args)
        raise AssertionError("an uncertain CUDA reference must not be cancelled")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(channel.FramedConnection, "send", send)
        patch.setattr(backend._codec, "cancel_unconsumed", cancel)
        weights = {name: torch.arange(32, dtype=torch.float32, device="cuda:0") for name in ("a", "b")}
        assert_export_allocations(weights)
        sources = [weakref.ref(value) for value in weights.values()]
        exported = backend.prepare_export(weights, _spec(api), context)
        del weights
        control.send("ready")
        receive(control, "inspect")
        record = backend._sessions[exported.session_id]
        assert record.control.release_complete.wait(10), "provider did not observe connection failure"
        snapshot = backend.session_status(exported)
        assert snapshot.state == api.SessionState.POISONED
        assert snapshot.exported_storage_ids == ("storage-0", "storage-1")
        assert all(reference() is not None for reference in sources)
        assert backend.release_export(exported).status == "POISONED"
        assert backend.release_export(exported).status == "POISONED"
        assert not cancellations
        assert len(dropped) == (1 if fault in {"import_ack", "release_ack"} else 0)
        control.send("checked")
        # The parent joins the consumer before allowing the CUDA owner to exit.
        receive(control, "exit")


def _consumer(endpoint, fault, control):
    receive(control, "start")
    api, backend, context = runtime(endpoint)
    original_import = backend._codec.import_storage
    imports = []
    uncertain = []

    def import_storage(descriptor, device_index):
        carrier = original_import(descriptor, device_index)
        imports.append(descriptor.storage_id)
        if fault == "partial_import" and len(imports) == 2:
            # The CUDA operation really consumed its reference before failing.
            # Keep its result alive to model an outcome the backend cannot know.
            uncertain.append(carrier)
            raise RuntimeError("injected failure after CUDA rebuild")
        return carrier

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend._codec, "import_storage", import_storage)
        if fault in {"partial_import", "import_ack"}:
            with pytest.raises(api.WeightSharingError) as error:
                backend.open_import(endpoint, _spec(api), context)
            assert error.value.poisoned
        else:
            imported = backend.open_import(endpoint, _spec(api), context)
            assert backend.session_status(imported).state == api.SessionState.MAPPED
            if fault == "consumer_exit":
                control.send("mapped")
                receive(control, "exit")
                # Exit without destructors or the release protocol, as on a crash.
                os._exit(0)
            with pytest.raises(api.WeightSharingError) as error:
                backend.close_import(imported)
            assert error.value.poisoned

        session_id = context.identity.session_id
        snapshot = backend.session_status(session_id)
        assert snapshot.state == api.SessionState.POISONED
        assert imports == ["storage-0", "storage-1"]
        if fault == "partial_import":
            assert snapshot.imported_storage_ids == ("storage-0",)
            assert snapshot.active_storage_id == "storage-1"
            assert len(uncertain) == 1
        elif fault == "import_ack":
            assert snapshot.imported_storage_ids == ("storage-0", "storage-1")
        else:
            assert snapshot.imported_storage_ids == ()
        record = backend._sessions[session_id]
        if fault != "release_ack":
            assert record.leases and all(not lease.expired() for lease in record.leases)
        result = backend.close_import(session_id)
        assert result.status == "POISONED"
        assert result.local_complete == (fault == "release_ack")
        assert not result.peer_confirmed
        with pytest.raises(api.WeightSharingError):
            backend.open_import(endpoint, _spec(api), context)
        assert imports == ["storage-0", "storage-1"]
        assert backend.session_status(session_id) == snapshot
        control.send("checked")
        receive(control, "exit")


def _run_failure(endpoint, fault):
    with spawn_workers((_provider, (endpoint, fault)), (_consumer, (endpoint, fault))) as workers:
        (_, provider), (consumer_process, consumer) = workers
        receive(provider, "ready")
        consumer.send("start")
        receive(consumer, "mapped" if fault == "consumer_exit" else "checked")
        if fault == "consumer_exit":
            consumer.send("exit")
            consumer_process.join(15)
            assert consumer_process.exitcode == 0
        provider.send("inspect")
        receive(provider, "checked")
        if fault != "consumer_exit":
            consumer.send("exit")
            consumer_process.join(15)
            assert consumer_process.exitcode == 0
        provider.send("exit")


@pytest.mark.parametrize("fault", ["partial_import", "import_ack", "release_ack"])
def test_uncertain_cuda_outcome_retains_resources(tmp_path: Path, fault: str):
    _run_failure(str(tmp_path / "f.sock"), fault)


def test_consumer_exit_does_not_release_provider_weights(tmp_path: Path):
    _run_failure(str(tmp_path / "f.sock"), "consumer_exit")


def _consumer_waiting_for_provider(endpoint, control):
    receive(control, "start")
    api, backend, context = runtime(endpoint)
    imports = []

    def forbidden_import(*args):
        imports.append(args)
        raise AssertionError("must not import from an exited provider")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend._codec, "import_storage", forbidden_import)
        with pytest.raises(api.WeightSharingError) as error:
            backend.open_import(endpoint, _spec(api), context)
        assert error.value.poisoned
        assert isinstance(error.value.__cause__, EOFError)
        assert not imports
        snapshot = backend.session_status(context.identity.session_id)
        assert snapshot.state == api.SessionState.POISONED
        assert snapshot.imported_storage_ids == ()
        assert snapshot.active_storage_id is None
        assert backend.close_import(context.identity.session_id).status == "POISONED"
    control.send("checked")


def test_provider_exit_during_handshake_rejects_import(tmp_path: Path):
    endpoint = str(tmp_path / "f.sock")
    with spawn_workers(
        (_provider, (endpoint, "provider_exit")), (_consumer_waiting_for_provider, (endpoint,))
    ) as workers:
        (provider_process, provider), (_, consumer) = workers
        receive(provider, "ready")
        consumer.send("start")
        receive(consumer, "checked")
        provider_process.join(15)
        assert provider_process.exitcode == 0
