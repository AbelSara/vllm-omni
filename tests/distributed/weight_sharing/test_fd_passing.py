# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SCM_RIGHTS frames and the backend's fd plumbing for fd-based codecs."""

from __future__ import annotations

import importlib
import os
import socket
import sys

import pytest

from tests.distributed.weight_sharing.helpers import (
    FakeCodec,
    FakeStorage,
    SocketPairTransport,
    make_context,
    make_weights,
)

if sys.platform != "linux":
    pytest.skip("SCM_RIGHTS fd passing requires Linux", allow_module_level=True)
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def channel(weight_sharing):
    return importlib.import_module(f"{weight_sharing.__name__}.channel")


def _pair(channel):
    left, right = socket.socketpair()
    limits = channel.Limits(timeout=3)
    return channel.FramedConnection(left, limits), channel.FramedConnection(right, limits)


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_fd_frames_keep_groups_and_order(channel):
    sender, receiver = _pair(channel)
    pipes = [os.pipe() for _ in range(3)]
    try:
        sender.send_fds(((pipes[0][1],), (pipes[1][1], pipes[2][1])))
        first = receiver.recv_fds(1)
        second = receiver.recv_fds(2)
        for fd, (read_end, _write_end) in zip(first + second, pipes):
            assert fd not in {w for _, w in pipes}  # A new descriptor for the same pipe.
            os.write(fd, b"x")
            assert os.read(read_end, 1) == b"x"
            os.close(fd)
    finally:
        for read_end, write_end in pipes:
            os.close(read_end)
            os.close(write_end)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("sent, expected", [(2, 1), (1, 2)], ids=["truncated", "short"])
def test_mismatched_fd_frame_is_rejected_without_leaking(channel, sent, expected):
    sender, receiver = _pair(channel)
    read_end, write_end = os.pipe()
    try:
        sender.send_fds(((write_end,) * sent,))
        before = _open_fds()
        with pytest.raises(channel.ChannelError):
            receiver.recv_fds(expected)
        assert _open_fds() == before
    finally:
        os.close(read_end)
        os.close(write_end)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("groups", [((),), (tuple(range(17)),)], ids=["empty", "too-many"])
def test_invalid_fd_frame_is_not_sent(channel, groups):
    sender, receiver = _pair(channel)
    try:
        with pytest.raises(channel.ChannelError):
            sender.send_fds(groups)
        with pytest.raises(channel.ChannelError):
            receiver.recv_fds(0)
    finally:
        sender.close()
        receiver.close()


class FdCodec(FakeCodec):
    """FakeCodec whose descriptors each reference one pipe fd, like Level Zero dma-bufs."""

    def __init__(self, api):
        super().__init__(api)
        self.pipes: dict[str, tuple[int, int]] = {}
        self.received: dict[str, tuple[int, ...]] = {}
        self.released: list[str] = []

    def export_storage(self, carrier, storage_id):
        descriptor = super().export_storage(carrier, storage_id)
        self.pipes[storage_id] = os.pipe()
        return self.api.StorageExport(*(getattr(descriptor, f) for f in descriptor.__struct_fields__[:-1]), 1)

    def export_fds(self, descriptor):
        self._called("export_fds")
        return (self.pipes[descriptor.storage_id][1],)

    def receive_fds(self, descriptor, fds):
        self._called("receive_fds")
        self.received[descriptor.storage_id] = fds

    def import_storage(self, descriptor, device_index):
        (fd,) = self.received.pop(descriptor.storage_id)
        os.write(fd, b"m")  # Prove the fd refers to the provider's pipe, then release it.
        os.close(fd)
        return super().import_storage(descriptor, device_index)

    def release_consumed(self, descriptor, device_index):
        self._called("release_consumed")
        self.released.append(descriptor.storage_id)

    def close(self):
        for read_end, write_end in self.pipes.values():
            os.close(read_end)
            os.close(write_end)


@pytest.fixture
def fd_runtime(weight_sharing, monkeypatch):
    api = importlib.import_module(f"{weight_sharing.__name__}.contracts")
    backend = importlib.import_module(f"{weight_sharing.__name__}.backend")
    channel = importlib.import_module(f"{weight_sharing.__name__}.channel")
    transport = SocketPairTransport(channel)
    monkeypatch.setattr(backend, "SessionServer", transport.server)
    monkeypatch.setattr(backend, "connect", transport.connect)
    codec = FdCodec(api)
    limits = channel.Limits(timeout=3)
    yield (
        api,
        backend.TorchCudaIpcBackend(codec=codec, limits=limits),
        backend.TorchCudaIpcBackend(codec=codec, limits=limits),
        codec,
    )
    transport.close()
    codec.close()


def test_backend_passes_fds_in_export_order_and_releases_consumed_handles(fd_runtime):
    api, provider, consumer, codec = fd_runtime
    context = make_context(api)
    bindings, spec = make_weights(api, two_storages=True)
    exported = provider.prepare_export(bindings, spec, context)
    imported = consumer.open_import(context.endpoint, spec, context)
    assert isinstance(imported["first"].storage, FakeStorage)
    for read_end, _write_end in codec.pipes.values():
        assert os.read(read_end, 1) == b"m"
    assert len(codec.pipes) == 2 and not codec.received
    assert provider.release_export(exported).status == "HELD"
    assert not codec.released
    del imported
    assert consumer.close_import(context.identity.session_id).status == "CLOSED"
    assert provider.release_export(exported).status == "CLOSED"
    assert codec.released == ["storage-0", "storage-1"]
    assert not codec.cancellations


def test_unexposed_fd_exports_are_cancelled_not_released(fd_runtime):
    api, provider, _consumer, codec = fd_runtime
    bindings, spec = make_weights(api, two_storages=True)
    exported = provider.prepare_export(bindings, spec, make_context(api))
    assert provider.release_export(exported).status == "CLOSED"
    assert codec.cancellations == ["storage-0", "storage-1"]
    assert not codec.released
