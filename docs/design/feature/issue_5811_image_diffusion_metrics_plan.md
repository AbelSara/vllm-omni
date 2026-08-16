# Issue #5811: Image and Diffusion Prometheus Metrics Plan

Issue: https://github.com/vllm-project/vllm-omni/issues/5811

## Goal

Complete and harden the Prometheus metric surface for text-to-image and
image-to-image serving. The implementation must expose stable service-level
metrics, avoid recording unrelated AR/text stages as image workloads, document
the timing semantics, and include deterministic CPU-level regression coverage.

## Final metric semantics

- `stage_gen_time_s`: generic stage submit-to-output latency, labelled with
  the bounded `stage_type` (`llm` or `diffusion`). It includes time waiting
  inside the stage and the stage execution/post-processing time. Diffusion
  dashboards select `stage_type="diffusion"`.
- `request_queue_wait_s`: request enqueue-to-initial-stage-submit latency in
  the orchestration layer.
- `stage_waiting_requests`: sum of the latest output-driven scheduler waiting
  snapshots across a stage's replicas. Diffusion snapshots piggyback on the
  existing output metrics payload, so the gauge may lag until the next output.
  An observed empty queue resets that replica's contribution to zero.
- `image_pixels`: output pixel-count distribution for image outputs only.
- `num_inference_steps`: configured diffusion step-count distribution;
  missing or zero values are not observed.
- `image_count_total`: number of generated image output units.
- `peak_memory_mb`: latest positive stage peak-memory observation.
- `requests_failed_total`: failed requests partitioned by a bounded reason
  taxonomy.
- `kv_wait_s`: cross-stage KV-transfer wait, labelled by connector type.
- `diffusion_forward_s`: forward-only diffusion execution reported by the
  `.diffuse` profiler stage.
- `diffusion_kv_load_s`: diffusion-side KV receive/load time, kept separate
  from `diffusion_forward_s`.
- `diffusion_exec_s`: total diffusion-engine execution time available to the
  service layer.

## Implementation checklist

- [x] Keep `stage_gen_time_s` generic and add a bounded `stage_type` label so
      LLM and diffusion stages can be selected explicitly.
- [x] Restrict image count and pixel count to image outputs, and inference-step
      observations to diffusion stages, through a directly testable workload
      metrics helper.
- [x] Keep image TTFP and denoise-step metrics scoped to image outputs.
- [x] Preserve valid zero queue-wait observations without exporting missing
      `queue_wait_ms` data as synthetic zero samples.
- [x] Preserve valid zero in-stage-wait observations while clamping
      clock/rounding artifacts below zero.
- [x] Reuse the existing diffusion output metrics payload to carry scheduler
      waiting snapshots without adding an IPC message type or channel.
- [x] Aggregate the latest per-replica waiting snapshots at stage scope and
      remove a dead replica's contribution.
- [x] Publish each stage's latest positive `peak_memory_mb` when that stage
      finishes, before non-final stage results return from the service layer.
- [x] Guarantee at-most-once failure counter updates with a request-scoped
      `ClientRequestState` guard and one centralized recording helper.
- [x] Bound failure reasons to `client_abort`, `client_disconnect`,
      `stage_error`, and `unknown`; normalize every other value to `unknown`.
- [x] Clean KV-wait timestamps on extraction success and scheduler terminal
      paths; aborted/failed waits are discarded without histogram emission.
- [x] Keep all duration values in seconds at the Prometheus boundary.
- [x] Add L1 `core_model` + `cpu` behavior tests, including Prometheus
      exposition assertions and negative/boundary cases.
- [ ] Update `docs/design/metrics.md` and `docs/usage/metrics.md` with the new
      families, labels, units, timing relationships, and PromQL examples.
- [x] Run targeted tests, `git diff --check`, and formatting/lint checks that
      are available in the local environment.

## Test plan

Level: L1, deterministic CPU logic tests.

Markers:

```python
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
```

Target files:

- `tests/metrics/test_definitions.py`
- `tests/metrics/test_modality.py`
- `tests/metrics/test_emit_calls.py`
- `tests/diffusion/test_diffusion_engine_metrics.py`

Required cases:

1. Every family is exposed with the documented name and labels.
2. Counters expose exactly one Prometheus `_total` suffix.
3. Zero queue waits are recorded and negative values are clamped.
4. Text/AR stage results do not populate image/diffusion histograms.
5. Image outputs populate count, pixels, inference steps, and timing exactly
   once even when a metric message is replayed.
6. Failure paths increment both the compatibility abort bucket and the new
   reason-labelled counter at most once.
7. KV-wait timing starts once, emits once, and is cleaned on terminal paths.
8. Diffusion forward and KV-load timings remain separate.

No Buildkite edit is expected: the existing L1 CPU test sweep collects
`core_model and cpu` tests under `tests/`.

## Completion criteria

- The issue's seven directly exportable metrics and three new-source metrics
  have reachable production emit paths.
- Metric scope and units match this document and the public metrics docs.
- L1 tests validate behavior and Prometheus exposition, rather than only
  checking source-code strings.
- The relevant targeted suite passes in a Linux development environment with
  the pinned vLLM version and PyTorch installed.
