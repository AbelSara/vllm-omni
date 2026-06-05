# DreamZero Offline Benchmark

This directory contains a small offline benchmark for DreamZero. It runs local
`Omni` inference, measures per-request generation latency, decodes the predicted
video latents, and writes artifacts for visual and numeric checks.

The playback `--fps` option only controls the MP4/GIF frame rate. The real
inference FPS is reported in the JSON summary as `model_video_fps`.

## Assets

Download the example camera videos before running the benchmark:

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets \
  --repo-type dataset \
  --local-dir outputs/dreamzero/assets
```

The bundled assets currently contain enough frames for the default two
action-producing requests: one initial single-frame request plus one 4-frame
chunk. Use longer videos with `--num-requests N` for longer steady-state runs.

## Run

Two-GPU CFG-parallel run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_2_tp1_cfg2 \
  --save-input-video \
  --save-side-by-side
```

Single-GPU baseline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_tp1_cfg1_baseline \
  --save-side-by-side
```

Optional action parity check against a previous run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_2_tp1_cfg2_checked \
  --reference-actions outputs/dreamzero/benchmark/dreamzero_gpu1_tp1_cfg1_baseline_actions.npz \
  --accuracy-atol 1e-3
```

The script defaults `DIFFUSION_ATTENTION_BACKEND` to `TORCH_SDPA` unless the
environment already sets another backend.

## Main Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--deploy-config` | required | DreamZero deploy YAML. |
| `--model` | `GEAR-Dreams/DreamZero-DROID` | Hugging Face model id or local model path. |
| `--video-dir` | `outputs/dreamzero/assets` | Directory with the three camera MP4 files. |
| `--num-requests` | `2` | Total generate calls, including the initial request. |
| `--fps` | `5` | Playback FPS for written video files, not inference FPS. |
| `--save-input-video` | off | Also write the stitched input camera video. |
| `--save-side-by-side` | off | Write input and prediction side by side. |
| `--save-gif` | off | Also write a GIF of the prediction. |
| `--reference-actions` | unset | Compare generated actions with a previous NPZ. |
| `--accuracy-atol` | `1e-3` | Max absolute action error allowed for reference comparison. |

## Output

For `--output-stem dreamzero_gpu1_2_tp1_cfg2`, the benchmark writes:

- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2.mp4`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_side_by_side.mp4`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_actions.npz`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_summary.json`

Useful summary fields:

| Field | Meaning |
| --- | --- |
| `latency_s.first_request` | First request wall time, including initial cache setup. |
| `latency_s.steady_state` | Stats over requests after the first one. |
| `latency_s.model_generate_total` | Sum of `omni.generate(...)` request times. |
| `latency_s.decode_video` | VAE decode latency for the concatenated prediction latents. |
| `throughput.model_request_hz` | Requests per second over model generation time. |
| `throughput.model_video_fps` | Decoded video frames per second over model generation time. |
| `throughput.model_plus_decode_video_fps` | Decoded video frames per second including VAE decode. |
| `throughput.model_action_hz` | Generated actions per second over model generation time. This is not robot control Hz. |

## Current Local Result

Measured on June 5, 2026 with:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver `590.48.01`
- TP1/CFG2 run used GPUs `1,2`
- TP1/CFG1 baseline used GPU `1`
- CPU: 2x AMD EPYC 9355 32-Core Processor, 128 logical CPUs
- Host memory: 1.5 TiB
- Backend: `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`
- Mode: eager, no torch compile
- Workload: 2 requests, 17 decoded output frames

| Mode | GPUs | First latency | Steady latency | Total generate | Decode | Model video FPS | Model+decode video FPS | Model action Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 CFG1 | 1 | 7.434s | 8.047s | 15.481s | 0.411s | 1.098 | 1.070 | 3.101 |
| TP1 CFG2 | 2 | 4.655s | 4.102s | 8.758s | 0.411s | 1.941 | 1.854 | 5.481 |

The generated prediction videos were readable by OpenCV:

| Artifact | Frames | FPS | Size |
| --- | ---: | ---: | --- |
| `dreamzero_gpu1_2_tp1_cfg2.mp4` | 17 | 5.0 | 640x352 |
| `dreamzero_gpu1_2_tp1_cfg2_side_by_side.mp4` | 17 | 5.0 | 1280x352 |

Action parity check between TP1/CFG2 and TP1/CFG1 baseline:

| Chunk | Shape | Max abs error | RMSE | Passed |
| ---: | --- | ---: | ---: | --- |
| 0 | 24x8 | 0.0 | 0.0 | yes |
| 1 | 24x8 | 0.0 | 0.0 | yes |
