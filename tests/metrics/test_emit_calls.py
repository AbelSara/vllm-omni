"""Verify emit-call wiring in production code paths.

`test_definitions.py` pins the family constants and label shapes; this file
pins that the production side actually calls the observe / inc / set helpers
declared on `OmniPrometheusMetrics`. Two layers:

- ``TestEmitCallSiteStatic`` — source-code inspection via ``inspect.getsource``
  so the test fails fast if a future refactor removes an emit call without
  updating the metrics surface.
- ``TestFailureCounterWiring`` / ``TestPromMetricsPlumbing`` — behavioral
  checks with mock ``OmniPrometheusMetrics`` to verify the call semantics
  (reason taxonomy propagation, kwarg threading).
"""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.metrics import OmniPrometheusMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "emit-calls-test"


# ---------------------------------------------------------------------------
# Static source-level pins — fail fast if emit calls disappear from prod code.
# ---------------------------------------------------------------------------


class TestEmitCallSiteStatic:
    """Source-code inspection of production emit call sites.

    Each test reads the function source via ``inspect.getsource`` so the test
    fails fast if a refactor drops the emit call without updating the metrics
    surface. The string-match is intentionally literal — the call must appear
    in the function body, not just somewhere in the module.
    """

    def test_omni_base_process_single_result_emits_per_stage_metrics(self) -> None:
        from vllm_omni.entrypoints.omni_base import OmniBase

        src = inspect.getsource(OmniBase._process_single_result)
        # Per-stage finish block.
        assert "observe_stage_gen_time(" in src, "missing observe_stage_gen_time emit"
        assert "observe_image_pixels(" in src, "missing observe_image_pixels emit"
        assert "observe_num_inference_steps(" in src, "missing observe_num_inference_steps emit"
        assert "inc_image_count(" in src, "missing inc_image_count emit"
        # Per-step denoise latency emit, scoped to image output_unit_type.
        assert "observe_denoise_step_latency(" in src, "missing observe_denoise_step_latency emit"
        # Finalize-time block.
        assert "set_peak_memory(" in src, "missing set_peak_memory emit"
        assert "observe_queue_wait(" in src, "missing observe_queue_wait emit"

    def test_omni_base_failure_paths_emit_requests_failed(self) -> None:
        from vllm_omni.entrypoints.omni_base import OmniBase

        fire_src = inspect.getsource(OmniBase._fire_failure_counter_if_alive)
        assert "inc_requests_failed(" in fire_src, "_fire_failure_counter_if_alive missing inc_requests_failed emit"
        assert "reason" in fire_src, "_fire_failure_counter_if_alive missing reason parameter"

        log_src = inspect.getsource(OmniBase._log_summary_and_cleanup)
        assert "inc_requests_failed(" in log_src, "_log_summary_and_cleanup missing inc_requests_failed emit"
        assert "reason" in log_src, "_log_summary_and_cleanup missing reason parameter"

    def test_orchestrator_loop_emits_stage_waiting_requests(self) -> None:
        from vllm_omni.engine.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._orchestration_loop)
        assert "set_stage_waiting_requests(" in src, "_orchestration_loop missing set_stage_waiting_requests emit"
        assert "num_waiting_reqs" in src, "_orchestration_loop not reading scheduler_stats.num_waiting_reqs"

    def test_omni_base_emits_image_ttfp_and_stage_in_queue(self) -> None:
        from vllm_omni.entrypoints.omni_base import OmniBase

        src = inspect.getsource(OmniBase._process_single_result)
        assert "observe_image_ttfp(" in src, "missing observe_image_ttfp emit in image path"
        assert "serving_time_to_first_output_ms" in src, "image_ttfp must source serving_time_to_first_output_ms"
        assert "observe_stage_in_queue(" in src, "missing observe_stage_in_queue emit"
        assert "diffusion_engine_exec_time_s" in src, "stage_in_queue must subtract diffusion_engine_exec_time_s"

    def test_stage_pool_build_stage_metrics_plumbs_num_inference_steps(self) -> None:
        from vllm_omni.engine.stage_pool import StagePool

        src = inspect.getsource(StagePool.build_stage_metrics)
        assert "num_inference_steps=num_inference_steps" in src, (
            "build_stage_metrics missing num_inference_steps= kwarg in return"
        )

    def test_scheduler_records_kv_wait_enter_timestamps(self) -> None:
        # _free_request must stamp time.monotonic() at both ENTER paths into
        # waiting_for_transfer_free (active-transfer wait + not-yet-triggered).
        from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

        src = inspect.getsource(OmniARScheduler._free_request)
        assert src.count("_kv_wait_start_ts[request_id] = time.monotonic()") == 2, (
            "_free_request must record kv_wait start ts at both ENTER paths"
        )

    def test_scheduler_emits_kv_wait_output_on_extraction_ack(self) -> None:
        # The kv_extracted_ids pre-process loop must call _emit_kv_wait_output,
        # which carries the wait duration across the process boundary.
        from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

        src = inspect.getsource(OmniARScheduler.update_from_output)
        assert "_emit_kv_wait_output(" in src, "update_from_output missing _emit_kv_wait_output call on extraction ack"
        emit_src = inspect.getsource(OmniARScheduler._emit_kv_wait_output)
        assert "_kv_wait_start_ts.pop" in emit_src, "_emit_kv_wait_output must pop the start ts"
        assert '"kv_wait_s"' in emit_src, "_emit_kv_wait_output must carry kv_wait_s"
        assert '"connector_type"' in emit_src, "_emit_kv_wait_output must carry connector_type"

    def test_orchestrator_loop_emits_kv_wait(self) -> None:
        from vllm_omni.engine.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._orchestration_loop)
        assert "observe_kv_wait(" in src, "_orchestration_loop missing observe_kv_wait emit"
        assert "kv_wait_s" in src, "_orchestration_loop not reading kv_wait_s from kv_transfer_params"


# ---------------------------------------------------------------------------
# Behavioral pins — verify call semantics with mock OmniPrometheusMetrics.
# ---------------------------------------------------------------------------


def _make_omni_base_with_mock_prom() -> tuple[object, MagicMock]:
    """Build a minimal OmniBase shell wired with a mock prom_metrics.

    Mirrors the pattern in ``test_prometheus.py::TestRequestLifecycleGauges``
    — ``object.__new__`` skips ``__init__`` so we don't need a real engine.
    """
    from vllm_omni.entrypoints.omni_base import OmniBase

    obj = object.__new__(OmniBase)
    obj.prom_metrics = MagicMock(spec=OmniPrometheusMetrics)
    obj.request_states = {}
    obj._consumed_metric_messages = {}
    obj.log_stats = True
    return obj, obj.prom_metrics


class TestFailureCounterWiring:
    """`_fire_failure_counter_if_alive` and `_log_summary_and_cleanup` must
    call BOTH the legacy `request_failed()` (which writes the abort bucket of
    `requests_success_total`) and the new `inc_requests_failed(reason)` (which
    writes `requests_failed_total` with the reason taxonomy).
    """

    def test_fire_failure_counter_passes_reason_through(self) -> None:
        obj, prom = _make_omni_base_with_mock_prom()
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done=set()),
        )

        obj._fire_failure_counter_if_alive("req-1", reason="client_disconnect")

        prom.request_failed.assert_called_once()
        prom.inc_requests_failed.assert_called_once_with("client_disconnect")

    def test_fire_failure_counter_default_reason_is_stage_error(self) -> None:
        obj, prom = _make_omni_base_with_mock_prom()
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done=set()),
        )

        obj._fire_failure_counter_if_alive("req-1")

        prom.inc_requests_failed.assert_called_once_with("stage_error")

    def test_fire_failure_counter_skips_when_request_already_succeeded(self) -> None:
        # When the request is in e2e_done (i.e. finalize already fired
        # request_succeeded), the failure path must NOT double-count.
        obj, prom = _make_omni_base_with_mock_prom()
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(e2e_done={"req-1"}),
        )

        obj._fire_failure_counter_if_alive("req-1", reason="client_disconnect")

        prom.request_failed.assert_not_called()
        prom.inc_requests_failed.assert_not_called()

    def test_fire_failure_counter_skips_when_request_state_missing(self) -> None:
        # No request_states entry — already popped by abort path. Fail-safe
        # must NOT raise.
        obj, prom = _make_omni_base_with_mock_prom()

        obj._fire_failure_counter_if_alive("missing-req", reason="oom")

        prom.request_failed.assert_not_called()
        prom.inc_requests_failed.assert_not_called()

    def test_log_summary_and_cleanup_passes_reason_through(self) -> None:
        obj, prom = _make_omni_base_with_mock_prom()
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(
                e2e_done=set(),
                build_and_log_summary=lambda: None,
            ),
        )

        obj._log_summary_and_cleanup("req-1", reason="timeout")

        prom.request_failed.assert_called_once()
        prom.inc_requests_failed.assert_called_once_with("timeout")

    def test_log_summary_and_cleanup_default_reason(self) -> None:
        obj, prom = _make_omni_base_with_mock_prom()
        obj.request_states["req-1"] = SimpleNamespace(
            metrics=SimpleNamespace(
                e2e_done=set(),
                build_and_log_summary=lambda: None,
            ),
        )

        obj._log_summary_and_cleanup("req-1")

        prom.inc_requests_failed.assert_called_once_with("stage_error")


# ---------------------------------------------------------------------------
# Cross-process plumbing pins — verify prom_metrics threads through to the
# Orchestrator (the only emit site that lives outside the API server process).
# ---------------------------------------------------------------------------


class TestPromMetricsPlumbing:
    def test_async_omni_engine_init_accepts_prom_metrics_kwarg(self) -> None:
        from vllm_omni.entrypoints.async_omni import AsyncOmniEngine

        sig = inspect.signature(AsyncOmniEngine.__init__)
        assert "prom_metrics" in sig.parameters, (
            "AsyncOmniEngine.__init__ missing prom_metrics parameter — "
            "OmniBase cannot forward prom_metrics to the Orchestrator"
        )

    def test_orchestrator_init_accepts_prom_metrics_kwarg(self) -> None:
        from vllm_omni.engine.orchestrator import Orchestrator

        sig = inspect.signature(Orchestrator.__init__)
        assert "prom_metrics" in sig.parameters, (
            "Orchestrator.__init__ missing prom_metrics parameter — "
            "AsyncOmniEngine cannot forward prom_metrics for stage_waiting emit"
        )

    def test_orchestrator_init_stores_prom_metrics(self) -> None:
        # Construction without stage_pools raises (stage_pools is required),
        # but the signature accepting prom_metrics is what we need to pin.
        from vllm_omni.engine.orchestrator import Orchestrator

        sig = inspect.signature(Orchestrator.__init__)
        assert sig.parameters["prom_metrics"].default is None, (
            "prom_metrics should default to None so unit-test engines without prom_metrics still construct cleanly"
        )

    def test_omni_base_runs_orchestrator_forwards_prom_metrics(self) -> None:
        # Static check that AsyncOmniEngine._run_orchestrator forwards
        # self._prom_metrics into the Orchestrator construction.
        from vllm_omni.entrypoints.async_omni import AsyncOmniEngine

        src = inspect.getsource(AsyncOmniEngine._run_orchestrator)
        assert "prom_metrics=self._prom_metrics" in src, (
            "AsyncOmniEngine._run_orchestrator not forwarding self._prom_metrics to Orchestrator construction"
        )


# ---------------------------------------------------------------------------
# Observe-method surface pins — verify the OmniPrometheusMetrics API surface
# expected by the emit sites actually exists.
# ---------------------------------------------------------------------------

_EXPECTED_OBSERVE_METHODS: dict[str, tuple[str, ...]] = {
    "observe_stage_gen_time": ("stage", "gen_time_s"),
    "observe_stage_in_queue": ("stage", "in_queue_s"),
    "observe_queue_wait": ("queue_wait_s",),
    "set_stage_waiting_requests": ("stage", "n_waiting"),
    "observe_num_inference_steps": ("n_steps",),
    "inc_image_count": ("n_images",),
    "observe_image_pixels": ("n_pixels",),
    "set_peak_memory": ("stage", "peak_memory_mb"),
    "inc_requests_failed": ("reason",),
    "observe_kv_wait": ("connector_type", "kv_wait_s"),
}


class TestObserveMethodSurface:
    """All 10 observe methods exist on OmniPrometheusMetrics with the
    parameter names the emit sites use. Catches a rename refactor that
    would silently break the emit call.

    Note: ``observe_diffusion_forward`` / ``observe_diffusion_kv_load`` /
    ``observe_image_ttfp`` live on ``OmniModalityMetrics`` alongside the other
    diffusion timing families (see test_modality.py) so they can ride the
    ``_observe_diffusion_finalize`` dispatcher.
    """

    @pytest.mark.parametrize("method,expected_params", list(_EXPECTED_OBSERVE_METHODS.items()))
    def test_observe_method_exists_with_expected_params(self, method: str, expected_params: tuple[str, ...]) -> None:
        func = getattr(OmniPrometheusMetrics, method, None)
        assert func is not None, f"OmniPrometheusMetrics missing {method}"
        sig = inspect.signature(func)
        # Drop 'self' before comparing.
        params = tuple(sig.parameters.keys())[1:]
        # Methods can have extra params with defaults (e.g. n_images=1); we
        # just need the named params to come first and match.
        assert params[: len(expected_params)] == expected_params, (
            f"{method} signature mismatch: expected leading params {expected_params}, got {params}"
        )


class TestEarlyReturnOnLogStatsOff:
    """When --log-stats is off (default), observe methods must be silent no-ops.

    This is the gating contract — emit sites can call unconditionally and the
    helper short-circuits.
    """

    def test_observe_methods_silent_when_log_stats_false(self) -> None:
        prom = OmniPrometheusMetrics(model_name=_MODEL, log_stats=False)
        # None of these should raise or write to the registry.
        prom.observe_stage_gen_time(stage=0, gen_time_s=1.5)
        prom.observe_stage_in_queue(stage=0, in_queue_s=0.2)
        prom.observe_queue_wait(queue_wait_s=0.5)
        prom.set_stage_waiting_requests(stage=0, n_waiting=3)
        prom.observe_num_inference_steps(n_steps=20)
        prom.inc_image_count(n_images=1)
        prom.observe_image_pixels(n_pixels=512 * 512)
        prom.set_peak_memory(stage=0, peak_memory_mb=1024.0)
        prom.inc_requests_failed(reason="oom")
        prom.observe_kv_wait(connector_type="shm", kv_wait_s=0.01)


# ---------------------------------------------------------------------------
# Behavioral pin: emit sites call unconditional observe_* even for text
# stages where image_pixels / num_inference_steps are zero — the helper
# early-returns on <=0. Verify the zero-guard contract.
# ---------------------------------------------------------------------------


class TestZeroGuardContract:
    """Image / diffusion helpers must early-return on zero so the per-stage
    finish block can call unconditionally — text stages contribute zeros
    without polluting the histogram. The guard is ``<= 0`` (not ``< 0``) so
    zero-valued observations don't bump ``_count`` to 1.
    """

    def test_zero_image_pixels_not_observed(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-zero-guard")
        prom.observe_image_pixels(n_pixels=0)
        out = generate_latest(REGISTRY).decode()
        # ``.labels()`` in OmniPrometheusMetrics.__init__ already creates the
        # child sample, so the ``_count`` line exists with value 0.0 even
        # without observations. The guard prevents ``_count`` from bumping
        # to 1.0 — assert the value stays at 0.
        needle = f'vllm_omni:image_pixels_count{{model_name="{_MODEL}-zero-guard"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected image_pixels_count line to exist after OmniPrometheusMetrics construction"
        assert float(count_lines[0].split()[-1]) == 0.0, (
            "zero-pixel observation leaked to registry — guard should early-return on <= 0"
        )

    def test_zero_num_inference_steps_not_observed(self) -> None:
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-zero-steps")
        prom.observe_num_inference_steps(n_steps=0)
        out = generate_latest(REGISTRY).decode()
        needle = f'vllm_omni:num_inference_steps_count{{model_name="{_MODEL}-zero-steps"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected num_inference_steps_count line to exist after construction"
        assert float(count_lines[0].split()[-1]) == 0.0, (
            "zero-step observation leaked to registry — guard should early-return on <= 0"
        )

    def test_positive_image_pixels_observed(self) -> None:
        # Sanity check: positive values DO get observed.
        from prometheus_client import REGISTRY, generate_latest

        prom = OmniPrometheusMetrics(model_name=_MODEL + "-pos-guard")
        prom.observe_image_pixels(n_pixels=512)
        out = generate_latest(REGISTRY).decode()
        needle = f'vllm_omni:image_pixels_count{{model_name="{_MODEL}-pos-guard"}}'
        count_lines = [ln for ln in out.splitlines() if ln.startswith(needle)]
        assert count_lines, "expected image_pixels_count line after positive observation"
        assert float(count_lines[0].split()[-1]) == 1.0, "positive-pixel observation did not increment count to 1"


# ---------------------------------------------------------------------------
# KV-wait emit wiring — scheduler ENTER/EXIT lifecycle + orchestrator dispatch.
# ---------------------------------------------------------------------------


def _make_scheduler_shell() -> object:
    """Minimal OmniARScheduler shell — skips upstream __init__ via object.__new__."""
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    obj = object.__new__(OmniARScheduler)
    obj._kv_wait_start_ts = {}
    obj._omni_kv_config = None
    return obj


class TestKvWaitSchedulerEmit:
    """Scheduler-side half of kv_wait_s: pops start ts, carries the wait across."""

    def test_skips_when_no_start_ts_recorded(self) -> None:
        sched = _make_scheduler_shell()
        outputs: dict[int, list] = {}

        sched._emit_kv_wait_output(outputs, "req-no-wait", req=SimpleNamespace(client_index=0))

        assert outputs == {}, "emit must not append output when no start ts recorded"
        assert "req-no-wait" not in sched._kv_wait_start_ts

    def test_emits_wait_duration_and_connector_type(self) -> None:
        sched = _make_scheduler_shell()

        sched._kv_wait_start_ts["req-1"] = time.monotonic() - 0.25
        outputs: dict[int, list] = {}
        live_req = SimpleNamespace(client_index=3)

        sched._emit_kv_wait_output(outputs, "req-1", req=live_req)

        assert "req-1" not in sched._kv_wait_start_ts, "start ts must be popped after emit"
        assert 3 in outputs and len(outputs[3]) == 1
        params = outputs[3][0].kv_transfer_params
        assert "kv_wait_s" in params and "connector_type" in params
        assert 0.0 < params["kv_wait_s"] < 1.0
        assert params["connector_type"] == "unknown"

    def test_connector_type_resolved_from_omni_kv_config(self) -> None:
        sched = _make_scheduler_shell()
        sched._omni_kv_config = {"connector_config": {"type": "SharedMemoryConnector"}}

        sched._kv_wait_start_ts["req-3"] = time.monotonic() - 0.01
        outputs: dict[int, list] = {}

        sched._emit_kv_wait_output(outputs, "req-3", req=SimpleNamespace(client_index=0))

        eco = outputs[0][0]
        assert eco.kv_transfer_params["connector_type"] == "SharedMemoryConnector"


class TestKvWaitOrchestratorDispatch:
    """Orchestrator reads kv_wait_s from kv_transfer_params and calls observe_kv_wait."""

    def test_orchestrator_static_dispatch_reads_kv_wait_s(self) -> None:
        from vllm_omni.engine.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._orchestration_loop)
        assert "self._prom_metrics is not None" in src
        assert 'kv_params.get("kv_wait_s")' in src
        assert 'kv_params.get("connector_type")' in src
        assert "observe_kv_wait(" in src
