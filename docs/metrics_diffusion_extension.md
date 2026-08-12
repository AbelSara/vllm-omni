# Diffusion / Image / Video 指标扩展 PR 规划

本文档归纳 omni 模态 Prometheus 指标的两部分工作：

- **Part A** — [Issue #5811](https://github.com/vllm-project/vllm-omni/issues/5811)：image 服务层指标落地，**本次 PR 范围**。
- **Part B** — 拓展规划（video 模态、cache backend、其他服务层 family），**后续 PR**。

所有命名遵循 `vllm_omni/metrics/definitions.py:22` 的 `METRIC_PREFIX = "vllm_omni:"` 约定（与现有 17 个 family 一致；`docs/design/metrics.md` 中 `vllm:omni_*` 写法是历史遗留，本次 PR 一并修正）。

---

## Part A — Issue #5811 实现状态（本次 PR 范围）

issue 来源：<https://github.com/vllm-project/vllm-omni/issues/5811>

issue 范围聚焦 image（text-to-image / image-to-image）服务层指标，分三层。issue 明确 out-of-scope：组件级 profiling、需要 GPU sync 的测量。

### Tier 1 — 已有 family 复用（零代码改动）

下列 family 在本仓库已落地，本次 PR 不动：

| Family | Type | Labels |
|---|---|---|
| `vllm_omni:e2e_request_latency_s` | Histogram | `model_name` |
| `vllm_omni:num_requests_running` / `num_requests_waiting` | Gauge | `model_name` |
| `vllm_omni:requests_success_total` | Counter | `model_name, finished_reason` |
| `vllm_omni:prompt_tokens_total` / `generation_tokens_total` | Counter | `model_name` |
| `vllm_omni:transfer_size_bytes` / `transfer_tx_s` / `transfer_rx_s` / `transfer_in_flight_s` | per definitions | `model_name, from_stage, from_replica, to_stage, to_replica` |

加上上游 `vllm:*` AR stage 指标（TTFT / TPOT / ITL / KV cache 等），通过 `OmniPrometheusStatLogger` 包装后复用。

### Tier 2 — 新 family，数据已存在（本次 PR 落地，含 emit 调用点连线）

7 个 family，从已有 stats / scheduler / pipeline timings 导出。本次 PR 落地 family 声明、`OmniPrometheusMetrics` observe 方法，**并完成生产侧 emit 调用点连线**。

| Family | Type | Labels | Source | Emit 调用点 |
|---|---|---|---|---|
| `vllm_omni:stage_gen_time_s` | Histogram | `model_name, stage` | `StageRequestStats.stage_gen_time_ms` (stats.py) | `omni_base.py` per-stage finish block |
| `vllm_omni:request_queue_wait_s` | Histogram | `model_name` | `pipeline_timings["queue_wait_ms"]` (orchestrator.py) | `omni_base.py` finalize guard |
| `vllm_omni:stage_waiting_requests` | Gauge | `model_name, stage` | `SchedulerStats.num_waiting_reqs`（跨进程载体，来自 `BaseScheduler.num_waiting_requests()`） | `orchestrator.py` `_orchestration_loop` |
| `vllm_omni:image_pixels` | Histogram | `model_name` | image output shape | `omni_base.py` per-stage finish block |
| `vllm_omni:num_inference_steps` | Histogram | `model_name` | `sampling_params.num_inference_steps` | `omni_base.py` per-stage finish block |
| `vllm_omni:image_count` | Counter | `model_name` | `output_unit_count` (stage_pool.py) | `omni_base.py` per-stage finish block |
| `vllm_omni:peak_memory_mb` (optional) | Gauge | `model_name, stage` | `engine_outputs.peak_memory_mb` (omni_base.py) | `omni_base.py` finalize guard |

`stage_pool.py:build_stage_metrics` 同时把 `num_inference_steps` 写入 `StageRequestMetrics`（`stats.py` 新增字段），让 `omni_base.py:_process_single_result` 能从 `_m.num_inference_steps` 读出。`prom_metrics` 经 `AsyncOmniEngine` → `Orchestrator` 透传，供 `_orchestration_loop` 内 emit `set_stage_waiting_requests`。

### Tier 3 — 新 family，需补数据源

3 个 family。本次 PR 注册 family 声明和 `OmniPrometheusMetrics` observe 方法。`requests_failed_total` 的 emit 调用点在本次 PR 内完成；`kv_wait_s` / `diffusion_forward_s` 因依赖外部 block rework，emit 调用留作 follow-up。

| Family | Type | Labels | 状态 | Blocker |
|---|---|---|---|---|
| `vllm_omni:requests_failed_total` | Counter | `model_name, reason` | **本次 PR 落地**（emit 已连线） | `reason` taxonomy 已在本节末尾锁定；`omni_base.py` / `async_omni.py` 失败路径加 `inc_requests_failed(reason)` |
| `vllm_omni:kv_wait_s` | Histogram | `model_name, connector_type` | follow-up | 依赖 KV manager block rework（记录 waiting 时间戳） |
| `vllm_omni:diffusion_forward_s` | Histogram | `model_name, stage, replica` | follow-up | 需在 diffusion runner 内部加 sub-timer 拆 forward vs preprocess/postprocess/KV load；与 issue out-of-scope "需要 GPU sync 的测量" 边界冲突，需先评估 |

`requests_failed_total` 的 `reason` 取值见本节末尾 taxonomy；`omni_base.py:_fire_failure_counter_if_alive` / `_log_summary_and_cleanup` 接受 `reason` 参数，`async_omni.py` 在 `CancelledError` / `Exception` 分支分别传 `client_disconnect` / `stage_error`。

### issue #5811 配套约定

- bucket：issue 未指定，Histogram 用 prometheus_client 默认 bucket；time-bearing family（`*_s`）用 `SECONDS_BUCKETS`
- Counter `_total` 后缀由 prometheus_client 自动追加，常量不带后缀
- 所有 family 受 `--log-stats` 闸门控制（默认 off）
- `OmniPrometheusMetrics.observe_*` 在 `log_stats=False` 时 early-return
- 标签维度严格按 issue 表（不引入 `replica` / `final_output_type` 等额外维度）

### issue #5811 落地动作清单（本次 PR 已完成）

1. `vllm_omni/metrics/definitions.py` — 新增 10 个 family 名 + 4 个 label set（`STAGE_GEN_TIME_LABELS` / `DIFFUSION_LABELS` / `FAILED_LABELS` / `KV_WAIT_LABELS`）
2. `vllm_omni/metrics/prometheus.py` — `OmniPrometheusMetrics` 加 10 个 observe 方法：`observe_stage_gen_time` / `observe_queue_wait` / `set_stage_waiting_requests` / `observe_num_inference_steps` / `inc_image_count` / `observe_image_pixels` / `set_peak_memory` / `inc_requests_failed` / `observe_kv_wait` / `observe_diffusion_forward`
3. `vllm_omni/metrics/stats.py` — `StageRequestStats` 加 `num_inference_steps: int = 0` 字段，供 `_process_single_result` 读取
4. `vllm_omni/engine/stage_pool.py` — `build_stage_metrics` 返回的 `StageRequestMetrics(...)` 写入 `num_inference_steps=num_inference_steps`
5. `vllm_omni/entrypoints/omni_base.py` — `_process_single_result` 内：
   - per-stage finish block 调 `observe_stage_gen_time` / `observe_image_pixels` / `observe_num_inference_steps` / `inc_image_count`
   - finalize guard 调 `set_peak_memory` / `observe_queue_wait`
   - `_fire_failure_counter_if_alive` / `_log_summary_and_cleanup` 加 `reason` 参数，在失败路径调 `inc_requests_failed(reason)`
   - 构造顺序调整：`OmniPrometheusMetrics` 在 `AsyncOmniEngine` 之前构造并经 kwarg 透传
6. `vllm_omni/entrypoints/async_omni.py` — `AsyncOmniEngine.__init__` 加 `prom_metrics` 参数；`_run_orchestrator` 透传给 `Orchestrator`；`CancelledError` / `Exception` 失败分支分别传 `client_disconnect` / `stage_error`
7. `vllm_omni/engine/orchestrator.py` — `Orchestrator.__init__` 加 `prom_metrics` 参数；`_orchestration_loop` 在 `_stat_logger.record()` 后从 `raw_outputs.scheduler_stats.num_waiting_reqs` 调 `set_stage_waiting_requests`
8. `tests/metrics/test_definitions.py` — 锁定 10 个 family 常量、label set 形状、Counter 后缀、唯一性
9. `tests/metrics/test_emit_calls.py` — 验证 emit 调用点接线（mock `OmniPrometheusMetrics`，断言 `observe_*` / `inc_*` / `set_*` 在预期路径被调用）

### Tier 3 follow-up（不在本次 PR 范围）

- `kv_wait_s` — 依赖 KV manager block rework；当前 KV 等待路径无时间戳记录
- `diffusion_forward_s` — 需在 diffusion runner 内部加 sub-timer 拆 forward vs preprocess/postprocess/KV load；且 forward-only 计时需要 GPU sync，与 issue out-of-scope "需要 GPU sync 的测量" 边界冲突，需先评估

---

## PR #4755 — Diffusion 引擎计时直方图（本次 PR 一并落地）

PR 来源：<https://github.com/vllm-project/vllm-omni/pull/4755>

PR #4755 给 diffusion 引擎加了 4 个计时直方图。引擎侧 emit 已经在
`vllm_omni/diffusion/diffusion_engine.py:step_streaming` 完成
（`preprocess_time_ms` / `diffusion_engine_exec_time_ms` /
`diffusion_engine_total_time_ms` / `postprocess_time_ms`）；本 PR 落地
Prometheus family 声明、`OmniModalityMetrics` observe 方法、以及
`OrchestratorAggregator.accumulate_diffusion_metrics` →
`_observe_diffusion_finalize` 的端到端 wiring。

### Family 列表

| Family | Type | Labels | Buckets | Source |
|---|---|---|---|---|
| `vllm_omni:diffusion_exec_s` | Histogram | `model_name, stage, replica` | `SECONDS_BUCKETS` | `diffusion_engine_exec_time_ms` / 1000 |
| `vllm_omni:diffusion_exec_per_step_s` | Histogram | `model_name, stage, replica` | `SECONDS_FAST_BUCKETS` | `diffusion_exec_s / num_inference_steps` |
| `vllm_omni:diffusion_preprocess_s` | Histogram | `model_name, stage, replica` | `SECONDS_FAST_BUCKETS` | `preprocess_time_ms` / 1000 |
| `vllm_omni:diffusion_postprocess_s` | Histogram | `model_name, stage, replica` | `SECONDS_FAST_BUCKETS` | `postprocess_time_ms` / 1000 |

Bucket 选择：
- `exec_s` 用 `SECONDS_BUCKETS`（上限 300 s），因为视频生成可超过
  `SECONDS_FAST_BUCKETS` 的 60 s 上限
- 其余 3 个 sub-second，用 `SECONDS_FAST_BUCKETS`
- `per_step_s` 不在引擎侧 emit（PR commit 5318b1c 移除），改在 finalize
  dispatcher 派生 `exec / num_inference_steps`，避免重复 emit

### Wiring 路径

1. **引擎 emit**（已有，不修改）— `diffusion_engine.step_streaming` 把
   `_ms` keys 写入 `request_output.metrics`
2. **累加器**（`vllm_omni/metrics/stats.py`）—
   `OrchestratorAggregator.accumulate_diffusion_metrics` 通过 `_MS_TO_S`
   map 把 `_ms` key 转为 `_s` key 累加到 per-request bucket
   - `diffusion_engine_exec_time_ms` → `diffusion_engine_exec_time_s`
   - `preprocess_time_ms` → `preprocess_time_s`
   - `postprocess_time_ms` → `postprocess_time_s`
   - `diffusion_engine_total_time_ms` → `diffusion_engine_total_time_s`
     （保留在 dict 里保持引擎 emit 对称，但不 route 到任何 Prometheus
     family — exec/preprocess/postprocess 已覆盖 PR #4755 计时面）
   - `StageRequestStats.diffusion_metrics` 类型从 `dict[str, int]` 改为
     `dict[str, float]`；`_as_stage_request_stats` 对 `_s` key 用
     `float(v)`、其他 key 用 `int(v)`，避免计数列在 log table 渲染成
     `1.0` / `2.0`
3. **Family + observe API**（`vllm_omni/metrics/modality.py`）— 4 个
   Histogram family + `OmniModalityMetrics.observe_diffusion_exec` /
   `observe_diffusion_exec_per_step` / `observe_diffusion_preprocess` /
   `observe_diffusion_postprocess`，受 `--log-stats` 闸门控制
4. **Finalize dispatcher**（`vllm_omni/metrics/modality.py`）—
   `_observe_diffusion_finalize` 从 `stage_metrics.diffusion_metrics` 读
   `_s` keys 并调对应 observe；`observe_modality_at_finalize` 在
   `stage_metrics.diffusion_metrics` 非空时（无视 `output_type`）路由到
   diffusion path，然后再按 `output_type == "audio"` 决定是否走 audio path
   - audio diffusion 阶段（stable_audio）：两条 path 都触发
   - image / video diffusion 阶段：只触发 diffusion path（其模态计数走
     `OmniPrometheusMetrics` 的 per-stage finish block，不在此 dispatcher）
   - text path：不触发（diffusion_metrics 为 None/空，early return）

### 与 issue #5811 `diffusion_forward_s` 的关系

两者不重叠：
- `diffusion_forward_s`（issue #5811 Tier 3，follow-up）— forward-only 计时
  （排除 preprocess/postprocess/KV load），需在 diffusion runner 内部加
  sub-timer 拆分；与 issue out-of-scope "GPU sync 测量" 边界冲突，需评估
- `diffusion_exec_s`（PR #4755）— E2E 引擎执行计时（含 forward，不含
  preprocess/postprocess），数据已在 `step_streaming` emit 中

### PR #4755 落地动作清单（本次 PR 已完成）

1. `vllm_omni/metrics/definitions.py` — 新增 4 个 family 常量
   `DIFFUSION_EXEC_S` / `DIFFUSION_EXEC_PER_STEP_S` /
   `DIFFUSION_PREPROCESS_S` / `DIFFUSION_POSTPROCESS_S`
2. `vllm_omni/metrics/modality.py` — 4 个 Histogram family 声明 +
   `OmniModalityMetrics` 4 个 observe 方法 + `_observe_diffusion_finalize`
   helper + `observe_modality_at_finalize` dispatcher 路由更新
3. `vllm_omni/metrics/stats.py` —
   `StageRequestStats.diffusion_metrics` 类型改 `dict[str, float]`；
   `accumulate_diffusion_metrics` 加 `_MS_TO_S` map + `_s` key 累加；
   `_as_stage_request_stats` 对 `_s` key 用 `float(v)`
4. `tests/metrics/test_definitions.py` — 加 `_PR4755_FAMILIES` 列表 +
   `test_pr4755_has_4_families` + 总数从 10 改为 14
5. `tests/metrics/test_modality.py` — `_EXPECTED_FAMILIES` 加 4 个
   diffusion family + `_StubModMetrics` 加 4 个 diffusion observe 方法 +
   `TestObserveDiffusionFinalize` 7 个行为测试（含 audio diffusion 双 path
   触发、video 单 path、`total_time_s` 不 route 等）+ `TestBucketSelection`
   4 个 bucket 选择 pin

---


## Part B — 拓展规划（后续 PR，不在本次 PR 范围）

下列内容原属本仓库早期设计草案，超出 issue #5811 范围，本次 PR 不实现，保留作为后续 PR 的规划。

### B.1 Video 模态（镜像 audio 设计）

| Family | Type | Labels | Source | Mirrors |
|---|---|---|---|---|
| `vllm_omni:video_ttff_s` | Histogram | `model_name, stage, replica` | `serving_video_output_stream.py` 首 chunk | `audio_ttfp_s` |
| `vllm_omni:video_duration_s` | Histogram | `model_name, stage, replica` | `frames / fps`（fps 在 `serving_video.py` 已解析） | `audio_duration_s` |
| `vllm_omni:video_rtf` | Histogram | `model_name, stage, replica` | `stage_gen_time_s / video_duration_s` | `audio_rtf` |
| `vllm_omni:video_frames_total` | Counter | `model_name, stage, replica` | runner 输出帧数 | `audio_frames_total` |
| `vllm_omni:video_underrun_s` | Histogram | `model_name, stage, replica` | chunk 到达时间 + 复用 `compute_continuity_stats` | `audio_underrun_s` |
| `vllm_omni:video_continuity_ok_total` | Counter | `model_name, stage, replica, threshold_ms` | 同 underrun | `audio_continuity_ok_total` |
| `vllm_omni:video_skipped_requests_total` | Counter | `model_name, stage, replica, reason` | 空 chunk / 编码失败 | `audio_skipped_requests_total` |

### B.2 Image 拓展（超出 issue #5811 的 image 服务层最小集）

| Family | Type | Labels | Source | Mirrors |
|---|---|---|---|---|
| `vllm_omni:image_ttfp_s` | Histogram | `model_name, stage, replica` | serving_image 首 chunk hook | `audio_ttfp_s` |
| `vllm_omni:image_skipped_requests_total` | Counter | `model_name, stage, replica, reason` | 空 image 输出路径 | `audio_skipped_requests_total` |
| `vllm_omni:image_pixels_total` | Counter | `model_name, stage, replica` | `StageRequestStats.image_pixels` (stats.py) | `audio_frames_total` |

### B.3 Diffusion 派生 / 容量指标

| Family | Type | Labels | Source | 备注 |
|---|---|---|---|---|
| `vllm_omni:diffusion_steps_per_second` | Histogram | `model_name, stage, replica` | `num_inference_steps / stage_gen_time_s` | 派生 perf rate；issue 用 `diffusion_forward_s` 代替 |
| `vllm_omni:stage_running_requests` | Gauge | `model_name, stage, replica` | `BaseScheduler.num_running_requests()` | `stage_waiting_requests` 对偶；issue 范围内未要求 |
| `vllm_omni:kv_transfer_active` | Gauge | `model_name, stage` | `len(active_kv_transfers)` (`omni_ar_scheduler.py`) | issue 用 `kv_wait_s`（直方图）代替 |
| `vllm_omni:diffusion_batch_size` | Histogram | `model_name, stage, replica` | runner 实际 batch size | `StageRequestMetrics.batch_size=1` 硬编码需先解；runner 需通过 `engine_outputs.metrics` 暴露 |

### B.4 Cache backend 可观测性

需要在 cache backend 加 emit hook。`backend ∈ {cache_dit, tea_cache, mag_cache, step_cache}` 区分四种加速后端。

| Family | Type | Labels | Source |
|---|---|---|---|
| `vllm_omni:cache_hit_total` | Counter | `model_name, stage, replica, backend` | `MagCacheBackend.log_cache_hit` (hook.py)；其他 backend 加同名 hook |
| `vllm_omni:cache_miss_total` | Counter | `model_name, stage, replica, backend` | `log_cache_miss` (hook.py) |
| `vllm_omni:cache_skip_steps_total` | Counter | `model_name, stage, replica, backend` | `StepCacheState` countdown→0 瞬间 (state.py) |

`cache_hit_ratio` 不作为 Python-side Gauge — 通过 PromQL 计算：

```promql
rate(vllm_omni:cache_hit_total[5m])
  / (rate(vllm_omni:cache_hit_total[5m])
     + rate(vllm_omni:cache_miss_total[5m]))
```

### B.5 现有 family 的 label 扩展（label-only change）

| Family | Change | Reason |
|---|---|---|
| `vllm_omni:e2e_request_latency_s` | add `final_output_type ∈ {text, audio, image, video}` | 一条 PromQL 切 per-modality e2e |
| `vllm_omni:transfer_*` | add `connector_type ∈ {shm, mooncake, yuanrong, transfer_engine}` | 区分物理传输后端 |

⚠️ label-only change 仍可能 breaking 现有 dashboard。需在落地 PR 描述里明确标注。

---

## 失败原因 taxonomy（Tier 3 `requests_failed_total` 落地前必须锁定）

| reason | 描述 |
|---|---|
| `client_disconnect` | 客户端主动断连 |
| `scheduler_abort` | 调度器 preempt（KV 压力等） |
| `stage_error` | stage 执行异常 |
| `oom` | OOM 恢复路径 |
| `timeout` | 请求超时 |
| `safety_filter` | 内容审核 reject（diffusion 专属，不归 `stage_error`） |

---

## 共享约束（贯穿 Part A 和 Part B）

- 所有 time-bearing 指标 `_s` 后缀，bucket 选 `SECONDS_BUCKETS` 或 `SECONDS_FAST_BUCKETS`
- Counter `_total` 后缀由 prometheus_client 自动追加
- 所有 family 受 `--log-stats` 闸门控制（默认 off）
- `OmniModalityMetrics.observe_*` / `OmniPrometheusMetrics.observe_*` 在 `log_stats=False` 时 early-return
- replica_resolver 失败时 fail-safe 跳过，不臆造 label
