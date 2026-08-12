"""Verify the issue #5811 image / diffusion metric family constants.

Pins:
- 14 new family names share the ``vllm_omni:`` prefix (10 from issue #5811
  + 4 from PR #4755)
- Counter-family constant values don't include ``_total`` (auto-suffixed by
  the prometheus_client at exposition)
- All family names are unique within the new set and against the pre-existing
  families
- New label sets are non-empty and well-formed, matching the issue spec
"""

from __future__ import annotations

import pytest

from vllm_omni.metrics import definitions as defs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# 7 Tier 2 (data already produced) + 3 Tier 3 (new data source) = 10 new
# family constants added by issue #5811.
# ---------------------------------------------------------------------------
_TIER2_FAMILIES = [
    defs.STAGE_GEN_TIME_S,
    defs.REQUEST_QUEUE_WAIT_S,
    defs.STAGE_WAITING_REQUESTS,
    defs.NUM_INFERENCE_STEPS,
    defs.IMAGE_COUNT_METRIC,
    defs.IMAGE_PIXELS_METRIC,
    defs.PEAK_MEMORY_MB,
]

_TIER3_FAMILIES = [
    defs.REQUESTS_FAILED,
    defs.KV_WAIT_S,
    defs.DIFFUSION_FORWARD_S,
]

# PR #4755 — 4 diffusion engine timing histograms (exec / exec_per_step /
# preprocess / postprocess). Distinct from issue #5811's diffusion_forward_s
# (Tier 3) — these come from the diffusion engine's step_streaming emit and
# are wired through OmniModalityMetrics, not OmniPrometheusMetrics.
_PR4755_FAMILIES = [
    defs.DIFFUSION_EXEC_S,
    defs.DIFFUSION_EXEC_PER_STEP_S,
    defs.DIFFUSION_PREPROCESS_S,
    defs.DIFFUSION_POSTPROCESS_S,
]

_NEW_FAMILIES = _TIER2_FAMILIES + _TIER3_FAMILIES + _PR4755_FAMILIES


class TestFamilyCount:
    def test_tier2_has_7_families(self) -> None:
        assert len(_TIER2_FAMILIES) == 7

    def test_tier3_has_3_families(self) -> None:
        assert len(_TIER3_FAMILIES) == 3

    def test_pr4755_has_4_families(self) -> None:
        assert len(_PR4755_FAMILIES) == 4

    def test_total_14_new_families(self) -> None:
        assert len(_NEW_FAMILIES) == 14


class TestPrefix:
    def test_all_new_families_use_vllm_omni_prefix(self) -> None:
        for name in _NEW_FAMILIES:
            assert name.startswith(defs.METRIC_PREFIX), f"family {name!r} missing prefix {defs.METRIC_PREFIX!r}"


class TestUniqueness:
    def test_new_family_names_unique(self) -> None:
        seen: set[str] = set()
        for name in _NEW_FAMILIES:
            assert name not in seen, f"duplicate family name: {name!r}"
            seen.add(name)

    def test_new_families_dont_collide_with_existing(self) -> None:
        existing = {
            defs.NUM_REQUESTS_RUNNING,
            defs.NUM_REQUESTS_WAITING,
            defs.E2E_REQUEST_LATENCY_S,
            defs.REQUESTS_SUCCESS,
            defs.PROMPT_TOKENS,
            defs.GENERATION_TOKENS,
            defs.AUDIO_TTFP_S,
            defs.AUDIO_DURATION_S,
            defs.AUDIO_RTF_METRIC,
            defs.AUDIO_FRAMES_METRIC,
            defs.AUDIO_UNDERRUN_S,
            defs.AUDIO_CONTINUITY_OK_METRIC,
            defs.AUDIO_SKIPPED_REQUESTS_METRIC,
            defs.TRANSFER_SIZE_BYTES,
            defs.TRANSFER_TX_S,
            defs.TRANSFER_RX_S,
            defs.TRANSFER_IN_FLIGHT_S,
        }
        for name in _NEW_FAMILIES:
            assert name not in existing, f"new family {name!r} collides with existing"


class TestCounterSuffix:
    """Counter families in the prometheus_client auto-suffix ``_total`` at
    exposition. The constant value must NOT include ``_total`` so the exposed
    name is correct (``requests_failed_total``, not
    ``requests_failed_total_total``).
    """

    def test_counter_family_constants_no_total_suffix(self) -> None:
        counters = [
            defs.IMAGE_COUNT_METRIC,
            defs.REQUESTS_FAILED,
        ]
        for name in counters:
            assert not name.endswith("_total"), (
                f"counter family {name!r} should not include '_total' — prometheus_client auto-suffixes at exposition"
            )


class TestLabelSets:
    """Issue #5811 label set pins. Each label set matches the issue spec
    exactly — narrower than the original design doc to keep scope tight.
    """

    def test_new_label_sets_nonempty_and_well_formed(self) -> None:
        for labels in (
            defs.STAGE_GEN_TIME_LABELS,
            defs.DIFFUSION_LABELS,
            defs.FAILED_LABELS,
            defs.KV_WAIT_LABELS,
        ):
            assert labels, f"label set {labels!r} is empty"
            assert all(isinstance(x, str) and x for x in labels), (
                f"label set {labels!r} contains non-str or empty entries"
            )

    def test_stage_gen_time_labels_match_issue_spec(self) -> None:
        # Issue #5811 spec: {model_name, stage}. No final_output_type —
        # per-modality slicing goes through e2e_request_latency_s.
        assert defs.STAGE_GEN_TIME_LABELS == ("model_name", "stage")

    def test_diffusion_labels_match_issue_spec(self) -> None:
        # Used by stage_waiting_requests + peak_memory_mb (per-stage gauges).
        assert defs.DIFFUSION_LABELS == ("model_name", "stage")

    def test_failed_labels_carry_reason(self) -> None:
        # requests_failed_total carries a `reason` taxonomy.
        assert defs.FAILED_LABELS == ("model_name", "reason")

    def test_kv_wait_labels_carry_connector_type(self) -> None:
        # kv_wait_s slices by physical transport backend.
        assert defs.KV_WAIT_LABELS == ("model_name", "connector_type")

    def test_stage_labels_carry_replica(self) -> None:
        # Used by diffusion_forward_s (Tier 3) which is per-(stage, replica).
        assert defs.STAGE_LABELS == ("model_name", "stage", "replica")
