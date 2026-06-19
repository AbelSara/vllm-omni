"""Tests that ``enable_kv_async_prefetch`` drives the prefetch feature, that
direct construction / the AR ``from_vllm_config`` path stay synchronous, and
that companion-KV / device-pool configs disable it."""

from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_direct_construction_defaults_to_sync():
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    assert mgr._async_prefetch is False


def test_prefetch_flag_enables_prefetch():
    od_config = SimpleNamespace(omni_kv_config={"enable_kv_async_prefetch": True})
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_prefetch is True


def test_no_prefetch_stays_sync():
    od_config = SimpleNamespace(omni_kv_config={"enable_kv_async_prefetch": False})
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_prefetch is False


def test_defaults_sync_when_flag_missing():
    od_config = SimpleNamespace(omni_kv_config=None)
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_prefetch is False


def test_prefetch_preserved_with_connector_config():
    od_config = SimpleNamespace(
        omni_kv_config={
            "connector_config": {"type": "mock"},
            "need_recv_cache": True,
            "enable_kv_async_prefetch": True,
        },
    )
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_prefetch is True


def test_prefetch_disabled_with_cfg_companion_collector():
    od_config = SimpleNamespace(
        omni_kv_config={"enable_kv_async_prefetch": True},
        cfg_kv_collect_func=lambda *a, **k: {},
    )
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_prefetch is False


def test_from_vllm_config_ar_path_is_always_sync():
    model_config = SimpleNamespace(omni_kv_config=None)
    vllm_config = SimpleNamespace(kv_transfer_config=None)
    mgr = OmniKVTransferManager.from_vllm_config(vllm_config, model_config)
    assert mgr._async_prefetch is False
