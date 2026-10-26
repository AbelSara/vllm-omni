# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Level Zero IPC codec for PyTorch XPU storages; importing this module does not load torch.

PyTorch's XPU reducer has no IPC path, so this codec exports the caching-allocator
segment that holds each storage with ``zeMemGetIpcHandle``. The handle refers to
a dma-buf file descriptor, which the backend passes with SCM_RIGHTS; the consumer
maps it with ``zeMemOpenIpcHandle`` and wraps the mapping through DLPack.
Storages in one segment send its fd once per session and share one mapping.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock, RLock
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from .contracts import (
    CompatibilityError,
    PreparedWeights,
    SessionContext,
    StorageExport,
    UnsupportedError,
    WeightSpec,
)
from .torch_cuda_ipc_codec import TorchCudaIpcCodec, _describe_bindings, _native_blocks

if TYPE_CHECKING:
    import torch


_CODEC_ID = "xpu-ze-ipc-v1"
_ZE_HANDLE_BYTES = 64
# Compute Runtime's handle starts {u64 dma-buf fd, u64 pool offset, u32 pid}. With
# a pid, the importer fetches the fd with pidfd_getfd, which Yama ptrace_scope=1
# forbids between sibling processes. The fd travels by SCM_RIGHTS instead, and
# the importer writes its own fd and a zero pid so the driver uses it directly.
_FD_BYTES = 8
_PID = slice(16, 20)
# Session nonce, provider segment base and segment size identify one shared allocation.
_ALLOCATION_KEY = struct.Struct("<16sQQ")
_ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x13
_ZE_MEMORY_TYPE_DEVICE = 2
_KDL_ONEAPI = 14


class _ZeIpcHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * _ZE_HANDLE_BYTES)]


class _ZeAllocProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("id", ctypes.c_uint64),
        ("pageSize", ctypes.c_uint64),
    ]


class _LevelZero:
    """The few Level Zero calls this codec needs, resolved from the loader."""

    def __init__(self) -> None:
        try:
            lib = ctypes.CDLL("libze_loader.so.1")
        except OSError as exc:
            raise UnsupportedError("XPU weight sharing requires the Level Zero loader") from exc
        handle, ptr, result = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        signatures = {
            "zeInit": (ctypes.c_uint32,),
            "zeDriverGet": (ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(handle)),
            "zeMemGetAllocProperties": (
                handle,
                ptr,
                ctypes.POINTER(_ZeAllocProperties),
                ctypes.POINTER(handle),
            ),
            "zeMemGetAddressRange": (handle, ptr, ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_size_t)),
            "zeMemGetIpcHandle": (handle, ptr, ctypes.POINTER(_ZeIpcHandle)),
            "zeMemPutIpcHandle": (handle, _ZeIpcHandle),
            "zeMemOpenIpcHandle": (handle, handle, _ZeIpcHandle, ctypes.c_uint32, ctypes.POINTER(ptr)),
            "zeMemCloseIpcHandle": (handle, ptr),
        }
        for name, argtypes in signatures.items():
            function = getattr(lib, name, None)
            if function is None:
                raise UnsupportedError(f"the Level Zero loader lacks {name}")
            function.argtypes, function.restype = argtypes, result
        default_context = getattr(lib, "zeDriverGetDefaultContext", None)
        if default_context is None:
            raise UnsupportedError("the Level Zero loader lacks zeDriverGetDefaultContext")
        default_context.argtypes, default_context.restype = (handle,), handle
        self.lib = lib
        self.check(lib.zeInit(0), "zeInit")
        count = ctypes.c_uint32(0)
        self.check(lib.zeDriverGet(ctypes.byref(count), None), "zeDriverGet")
        drivers = (handle * count.value)()
        self.check(lib.zeDriverGet(ctypes.byref(count), drivers), "zeDriverGet")
        self.contexts = [c for c in (default_context(d) for d in drivers) if c]

    @staticmethod
    def check(code: int, operation: str) -> None:
        if code != 0:
            raise UnsupportedError(f"Level Zero {operation} failed (0x{code:x})")

    def owner(self, address: int) -> tuple[int, int]:
        """Return the (context, device) that knows a PyTorch device allocation."""
        for context in self.contexts:
            properties = _ZeAllocProperties(stype=_ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES)
            device = ctypes.c_void_p()
            code = self.lib.zeMemGetAllocProperties(context, address, ctypes.byref(properties), ctypes.byref(device))
            if code == 0 and properties.type == _ZE_MEMORY_TYPE_DEVICE and device.value:
                return context, device.value
        raise UnsupportedError("PyTorch's XPU allocation is not visible in a Level Zero default context")

    def address_range(self, context: int, address: int) -> tuple[int, int]:
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        self.check(
            self.lib.zeMemGetAddressRange(context, address, ctypes.byref(base), ctypes.byref(size)),
            "zeMemGetAddressRange",
        )
        if not base.value or not size.value:
            raise UnsupportedError("invalid Level Zero allocation range")
        return base.value, size.value


_level_zero_state: tuple[int, _LevelZero] | None = None
_level_zero_lock = Lock()


def _level_zero() -> _LevelZero:
    global _level_zero_state
    with _level_zero_lock:
        # Level Zero state does not survive fork; key it by process.
        if _level_zero_state is None or _level_zero_state[0] != os.getpid():
            _level_zero_state = (os.getpid(), _LevelZero())
        return _level_zero_state[1]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", _DLDeleter)]

# DLPack keeps raw pointers into these objects until PyTorch calls the deleter.
_live_dlpack: dict[int, tuple[object, ...]] = {}
_dlpack_lock = RLock()
_DLTENSOR_NAME = ctypes.c_char_p(b"dltensor")


@_DLDeleter
def _dlpack_deleter(managed):  # type: ignore[no-untyped-def]
    with _dlpack_lock:
        entry = _live_dlpack.pop(ctypes.addressof(managed.contents), None)
    if entry is not None:
        cast("Callable[[], None]", entry[-1])()


def _wrap_device_bytes(address: int, nbytes: int, device_index: int, on_free: Callable[[], None]) -> torch.Tensor:
    """A uint8 XPU tensor over mapped device memory; on_free runs when PyTorch frees it."""
    import torch

    shape, strides = (ctypes.c_int64 * 1)(nbytes), (ctypes.c_int64 * 1)(1)
    managed = _DLManagedTensor()
    managed.dl_tensor = _DLTensor(
        address, _DLDevice(_KDL_ONEAPI, device_index), 1, _DLDataType(1, 8, 1), shape, strides, 0
    )
    managed.deleter = _dlpack_deleter
    with _dlpack_lock:
        _live_dlpack[ctypes.addressof(managed)] = (managed, shape, strides, on_free)
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype, capsule_new.argtypes = ctypes.py_object, (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    try:
        return torch.from_dlpack(capsule_new(ctypes.addressof(managed), _DLTENSOR_NAME, None))
    except BaseException:
        with _dlpack_lock:
            _live_dlpack.pop(ctypes.addressof(managed), None)
        raise


def _device_uuid(device_index: int) -> str:
    import torch

    uuid = torch.xpu.get_device_properties(device_index).uuid
    try:
        return f"XPU-{UUID(str(uuid))}"
    except ValueError as exc:
        raise UnsupportedError("a full XPU device UUID is required") from exc


def _allocation_key(descriptor: StorageExport) -> tuple[bytes, int, int]:
    return cast(tuple[bytes, int, int], _ALLOCATION_KEY.unpack(descriptor.ref_counter_handle))


def _validate_descriptor(descriptor: StorageExport) -> None:
    if descriptor.codec != _CODEC_ID:
        raise UnsupportedError(f"unsupported XPU IPC codec: {descriptor.codec}")
    integers = (descriptor.device_index, descriptor.storage_nbytes, descriptor.allocation_offset_bytes)
    if (
        not descriptor.storage_id
        or any(type(value) is not int or value < 0 for value in integers)
        or descriptor.storage_nbytes == 0
        or len(descriptor.handle) != _ZE_HANDLE_BYTES
        or len(descriptor.ref_counter_handle) != _ALLOCATION_KEY.size
        or descriptor.ref_counter_offset != 0
        or descriptor.fd_count not in (0, 1)
    ):
        raise CompatibilityError("invalid XPU IPC storage descriptor")
    _nonce, base, size = _allocation_key(descriptor)
    if not base or descriptor.allocation_offset_bytes + descriptor.storage_nbytes > size:
        raise CompatibilityError("XPU IPC storage exceeds its allocation")


@dataclass
class _Plan:
    nonce: bytes
    context: int
    base: int
    size: int
    offset: int
    first: bool


@dataclass
class _Exported:
    context: int
    handle: _ZeIpcHandle
    fd: int


@dataclass
class _Mapping:
    context: int
    address: int
    references: int = 0


class TorchXpuIpcCodec(TorchCudaIpcCodec):
    """Level Zero counterpart of TorchCudaIpcCodec for native PyTorch XPU storages.

    Reuses the device-neutral manifest, view and lease logic. The provider keeps
    each segment's IPC handle until the peer releases or the export is cancelled;
    the consumer closes a mapping when its last carrier is freed.
    """

    def __init__(self) -> None:
        self._plans: dict[int, _Plan] = {}
        self._exported: dict[tuple[bytes, int, int], _Exported] = {}
        self._received: dict[tuple[bytes, int, int], int] = {}
        self._mappings: dict[tuple[bytes, int, int], _Mapping] = {}
        self._consumer_devices: dict[int, tuple[int, int]] = {}
        # Reentrant: the DLPack deleter may run during garbage collection under this lock.
        self._lock = RLock()

    @staticmethod
    def _runtime_snapshot(context: SessionContext) -> list[dict[str, object]]:
        import torch

        if sys.platform != "linux" or not torch.xpu.is_available():
            raise UnsupportedError("XPU weight sharing requires Linux and an available XPU")
        if torch.accelerator.current_accelerator(check_available=True) != torch.device("xpu"):
            raise UnsupportedError("an available XPU accelerator is required")
        if _device_uuid(context.device_index) != context.identity.device_uuid:
            raise CompatibilityError("XPU device UUID differs")
        torch.accelerator.set_device_index(context.device_index)
        snapshot = torch.xpu.memory_snapshot()
        if not isinstance(snapshot, list) or any(not isinstance(segment, dict) for segment in snapshot):
            raise UnsupportedError("unsupported XPU allocator snapshot")
        return cast(list[dict[str, object]], snapshot)

    def inspect(self, bindings: Mapping[str, object], spec: WeightSpec, context: SessionContext) -> PreparedWeights:
        import torch

        segments = self._runtime_snapshot(context)
        for name, tensor in bindings.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device != torch.device("xpu", context.device_index):
                raise CompatibilityError(f"binding is not on the session's XPU device: {name}")
        prepared = _describe_bindings(bindings, spec)
        # Expandable or private-pool segments are skipped here, so their storages fail closed.
        blocks = _native_blocks(segments, context.device_index)
        level_zero = _level_zero()
        nonce = uuid4().bytes
        seen: set[int] = set()
        for value in prepared.carriers.values():
            carrier = cast("torch.Tensor", value)
            storage = carrier.untyped_storage()
            if not storage.nbytes():
                continue
            if not storage.resizable() or blocks.get(storage.data_ptr(), 0) < storage.nbytes():
                raise UnsupportedError("storage must belong to a live native allocation outside private pools")
            ze_context, _device = level_zero.owner(storage.data_ptr())
            base, size = level_zero.address_range(ze_context, storage.data_ptr())
            offset = storage.data_ptr() - base
            if offset < 0 or offset + storage.nbytes() > size:
                raise UnsupportedError("storage lies outside its Level Zero allocation")
            self._plans[id(carrier)] = _Plan(nonce, ze_context, base, size, offset, base not in seen)
            seen.add(base)
        return prepared

    def validate_runtime(self, context: SessionContext) -> None:
        self._runtime_snapshot(context)

    def export_storage(self, carrier: object, storage_id: str) -> StorageExport:
        import torch

        if type(carrier) is not torch.Tensor:
            raise UnsupportedError("a plain tensor byte carrier is required")
        if (
            carrier.device.type != "xpu"
            or carrier.dtype != torch.uint8
            or carrier.storage_offset() != 0
            or carrier.shape != (carrier.untyped_storage().nbytes(),)
            or carrier.stride() != (1,)
            or carrier.numel() == 0
        ):
            raise UnsupportedError("a nonempty full-storage XPU byte carrier is required")
        plan = self._plans.pop(id(carrier), None)
        if plan is None:
            raise UnsupportedError("XPU storage was not inspected for export")
        key = (plan.nonce, plan.base, plan.size)
        handle = _ZeIpcHandle()
        if plan.first:
            level_zero = _level_zero()
            level_zero.check(
                level_zero.lib.zeMemGetIpcHandle(plan.context, plan.base, ctypes.byref(handle)), "zeMemGetIpcHandle"
            )
            fd = int.from_bytes(bytes(handle.data[:_FD_BYTES]), "little")
            pid = int.from_bytes(bytes(handle.data[_PID]), "little")
            try:
                os.fstat(fd)
                if pid not in (0, os.getpid()):
                    raise OSError("unexpected pid field")
            except OSError as exc:
                level_zero.lib.zeMemPutIpcHandle(plan.context, handle)
                raise UnsupportedError("unsupported Level Zero IPC handle format") from exc
            with self._lock:
                self._exported[key] = _Exported(plan.context, handle, fd)
        # The provider's fd and pid are meaningless to the consumer; it receives its own fd.
        handle_bytes = bytearray(handle.data)
        handle_bytes[:_FD_BYTES] = bytes(_FD_BYTES)
        handle_bytes[_PID] = bytes(4)
        descriptor = StorageExport(
            storage_id=storage_id,
            codec=_CODEC_ID,
            device_index=carrier.device.index or 0,
            handle=bytes(handle_bytes),
            storage_nbytes=carrier.numel(),
            allocation_offset_bytes=plan.offset,
            ref_counter_handle=_ALLOCATION_KEY.pack(*key),
            ref_counter_offset=0,
            event_handle=b"",
            event_sync_required=False,
            fd_count=1 if plan.first else 0,
        )
        _validate_descriptor(descriptor)
        return descriptor

    def export_fds(self, descriptor: StorageExport) -> tuple[int, ...]:
        _validate_descriptor(descriptor)
        with self._lock:
            exported = self._exported.get(_allocation_key(descriptor))
        if exported is None or descriptor.fd_count != 1:
            raise CompatibilityError("no exported XPU IPC handle for this descriptor")
        return (exported.fd,)

    def _put(self, descriptor: StorageExport) -> None:
        if not descriptor.fd_count:
            return
        with self._lock:
            exported = self._exported.pop(_allocation_key(descriptor), None)
        if exported is not None:
            level_zero = _level_zero()
            level_zero.check(level_zero.lib.zeMemPutIpcHandle(exported.context, exported.handle), "zeMemPutIpcHandle")

    def release_consumed(self, descriptor: StorageExport, device_index: int) -> None:
        _validate_descriptor(descriptor)
        self._put(descriptor)

    def cancel_unconsumed(self, descriptor: StorageExport, device_index: int) -> None:
        _validate_descriptor(descriptor)
        self._put(descriptor)

    def receive_fds(self, descriptor: StorageExport, fds: tuple[int, ...]) -> None:
        try:
            _validate_descriptor(descriptor)
            key = _allocation_key(descriptor)
            with self._lock:
                if descriptor.fd_count != 1 or len(fds) != 1 or key in self._received or key in self._mappings:
                    raise CompatibilityError("unexpected XPU IPC file descriptor")
                self._received[key] = fds[0]
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise

    def _consumer_device(self, device_index: int) -> tuple[int, int]:
        import torch

        with self._lock:
            known = self._consumer_devices.get(device_index)
        if known is None:
            # Resolve the Level Zero context/device from a live PyTorch allocation.
            probe = torch.empty(1, dtype=torch.uint8, device=torch.device("xpu", device_index))
            known = _level_zero().owner(probe.data_ptr())
            with self._lock:
                self._consumer_devices[device_index] = known
        return known

    def _release_mapping(self, key: tuple[bytes, int, int]) -> None:
        with self._lock:
            mapping = self._mappings[key]
            mapping.references -= 1
            if mapping.references:
                return
            del self._mappings[key]
        _level_zero().lib.zeMemCloseIpcHandle(mapping.context, mapping.address)

    def import_storage(self, descriptor: StorageExport, device_index: int) -> object:
        _validate_descriptor(descriptor)
        if type(device_index) is not int or device_index < 0:
            raise CompatibilityError("consumer device index must be a nonnegative integer")
        key = _allocation_key(descriptor)
        with self._lock:
            mapping = self._mappings.get(key)
            fd = self._received.pop(key, None)
        if mapping is None:
            if fd is None or descriptor.fd_count != 1:
                raise CompatibilityError("missing XPU IPC file descriptor for a new allocation")
            try:
                context, device = self._consumer_device(device_index)
                handle_bytes = bytearray(descriptor.handle)
                handle_bytes[:_FD_BYTES] = fd.to_bytes(_FD_BYTES, "little")
                handle_bytes[_PID] = bytes(4)
                handle = _ZeIpcHandle.from_buffer_copy(handle_bytes)
                address = ctypes.c_void_p()
                _level_zero().check(
                    _level_zero().lib.zeMemOpenIpcHandle(context, device, handle, 0, ctypes.byref(address)),
                    "zeMemOpenIpcHandle",
                )
            finally:
                # The mapping keeps the dma-buf alive; the fd is no longer needed.
                os.close(fd)
            mapping = _Mapping(context, cast(int, address.value))
            with self._lock:
                self._mappings[key] = mapping
        elif fd is not None:
            os.close(fd)
            raise CompatibilityError("duplicate XPU IPC file descriptor for a mapped allocation")
        with self._lock:
            mapping.references += 1
        try:
            return _wrap_device_bytes(
                mapping.address + descriptor.allocation_offset_bytes,
                descriptor.storage_nbytes,
                device_index,
                lambda: self._release_mapping(key),
            )
        except BaseException:
            self._release_mapping(key)
            raise

    def empty_storage(self, device_index: int) -> object:
        import torch

        return torch.empty(0, dtype=torch.uint8, device=torch.device("xpu", device_index))
