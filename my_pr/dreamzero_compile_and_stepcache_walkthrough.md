# DreamZero 代码走读：Graph 编译 与 StepCache

> 目标：逐段走读 DreamZero 的 **torch.compile / CUDA Graph 编译** 与 **StepCache（速度向量相似度跳步）** 两条优化路径，解释每一步「在做什么 / 为什么这么做」。
>
> 涉及文件：
> - `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py`（编译入口、去噪循环）
> - `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`（被编译的 DiT）
> - `vllm_omni/diffusion/models/dreamzero/wan_vae_feat_cache_patch.py`（VAE 编译补丁）
> - `vllm_omni/diffusion/cache/stepcache/{config,state,backend,__init__}.py`（StepCache）

---

## 第一部分：Graph 编译

### 0. 背景：为什么 DreamZero 的编译要"分而治之"

DreamZero 是一个 **流式（streaming）世界模型**：每来一帧观测，就做一次「prefill KV cache → 去噪循环」的 decode-like 推理。这种推理 host 侧 launch 开销占比很高，非常适合用 CUDA Graph 消除 launch 开销。

但它内部有两类性质完全不同的模块：

| 模块 | 形状 | 是否有原地复用 | 编译策略 |
|---|---|---|---|
| text/image 编码器、VAE | 固定 | 无 | `reduce-overhead`（开 CUDA Graph） |
| DiT block | 固定 | **有**（adaLN modulation 切片复用） | `default`（只做 Inductor 融合，不录图） |

所以编译入口 `setup_compile()` 不是"一把梭整图编译"，而是按模块分别处理。

---

### 1. `setup_compile()` — 编译入口

`pipeline_dreamzero.py:360-423`

```python
def setup_compile(self) -> None:
    if not torch.cuda.is_available():
        logger.info("DreamZero setup_compile skipped: CUDA not available.")
        return

    from vllm_omni.diffusion.models.dreamzero.wan_vae_feat_cache_patch import (
        apply_wan_vae_feat_cache_tensor_patch,
    )
    apply_wan_vae_feat_cache_tensor_patch()          # ① VAE 补丁，见 §3

    compile_ro  = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}
    dit_compile = {"mode": "default",         "fullgraph": True, "dynamic": False}
```

逐项解释这三个编译参数：

- **`mode="reduce-overhead"`**：torch.compile 在 Inductor 融合之上额外启用 **CUDA Graph**，把整段 kernel 序列录制成一张图重放，消除每个 kernel 的 host launch 开销。代价是它会用**静态输出 buffer**（下面会反复提到这个坑）。
- **`mode="default"`**：只做 Inductor 算子融合 / codegen，**不录 CUDA Graph**。
- **`fullgraph=True`**：要求整段无 graph break（不允许回退到 Python）。一旦有动态控制流就直接报错——所以 VAE 才需要打补丁（§3）。
- **`dynamic=False`**：固定 shape 特化，生成最快的 kernel。DreamZero 每步形状固定，所以可以关掉动态 shape。

#### ①-④ 四类模块分别编译

```python
# ② 文本编码器（UMT5）
self.text_encoder.forward = torch.compile(self.text_encoder.forward, **compile_ro)

# ③ 图像编码器（CLIP visual）
self.image_encoder.model.visual.forward = torch.compile(
    self.image_encoder.model.visual.forward, **compile_ro)

# ④ VAE encode
self.vae.encode = torch.compile(self.vae.encode, **compile_ro)
```

这三个都用 `reduce-overhead`：输入 shape 固定、没有「输出 buffer 被后续步骤复用」的问题，开 CUDA Graph 安全且收益大。

注意每个 `torch.compile` 都包在 `try/except` 里——**编译失败只 warning + 跳过，不让整个 pipeline 崩**。这是工程上很重要的容错。

#### ⑤ DiT —— 逐 block 编译，且故意不用 reduce-overhead

`pipeline_dreamzero.py:404-421`

```python
compiled_blocks = 0
for block in self.transformer.blocks:
    try:
        block.forward = torch.compile(block.forward, **dit_compile)  # mode="default"!
        compiled_blocks += 1
    except Exception as exc:
        logger.warning("... block %d compile failed ...", compiled_blocks)
        break                       # 失败就停，剩余 block 保持 eager
```

两个关键决策：

1. **逐 block 编译，而不是整个 transformer 一次编译**：粒度小，单 block 用 `fullgraph=True` 更容易成功；某个 block 失败时 `break`，前面已编译的保留，后面的保持 eager，**部分编译也能跑**。

2. **DiT 用 `mode="default"`，不用 `reduce-overhead`**。原因见 `:379-380` 的注释：

   > DiT blocks: default avoids CUDAGraph overwrite on modulation tensors

   DiT 每个 block 用 6 参数 adaLN 调制（`causal_wan_model.py:654`）：
   ```python
   e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)   # 切成 e[0..5]
   ...
   x = x + (y * e[2].squeeze(2))                            # 复用切片
   ```
   这些 `e[i]` 切片张量如果落在 CUDA Graph 的静态 buffer 上，会被下一步重放时**覆写**，导致结果错乱。所以 DiT 只融合、不录图。

最后 `setup_compile()` 调 `self.warmup_compile()`（§4）。

---

### 2. `_cudagraph_mark_step_begin()` — CUDA Graph 的"翻页"信号

`pipeline_dreamzero.py:514-519`

```python
@staticmethod
def _cudagraph_mark_step_begin() -> None:
    try:
        torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        pass
```

`reduce-overhead` 用静态 buffer 复用同一块显存。`cudagraph_mark_step_begin()` 是告诉 CUDA Graph runtime「**新的一步开始了，可以轮换/复用静态 buffer 了**」。

调用点遍布所有进入编译区之前的位置：
- `_predict_noise_eager`（`:332`，每次 DiT forward 前）
- `_encode_text`（`:527`）
- `_encode_image`（`:556`）
- `decode_video_latents`（`:604`）
- warmup 里 VAE decode 前（`:459`）

漏调用会导致跨步的静态 buffer 数据竞争。

---

### 3. `predict_noise` 里的 clone —— 编译与 StepCache 的交点

`pipeline_dreamzero.py:310-328`

```python
def predict_noise(self, **kwargs):
    video_pred, action_pred = self._predict_noise_eager(kwargs)

    if is_stepcache_active(self):
        video_pred = video_pred.clone()              # ★ 关键
        if action_pred is not None:
            action_pred = action_pred.clone()
    ...
```

为什么开 StepCache 时要 `clone()`？

- `reduce-overhead` 的 CUDA Graph 返回的 tensor 背后是**会被下一步重放覆写的静态 buffer**。
- StepCache 要把某一步的预测**缓存下来、复用到后面好几步**（§6）。
- 如果不 clone，缓存的是一个会被覆写的视图，下一步就被冲掉了 → 必须 deep copy 到独立显存。

> 注：DiT block 输出本身在 `causal_wan_model.py:1042-1044` 也有 `.clone()`，是为了避免 unpatchify 原地操作影响 KV cache 写回。两处 clone 目的不同。

---

### 4. `warmup_compile()` — 把编译/捕获开销挪到计时之外

`pipeline_dreamzero.py:425-465`

```python
def warmup_compile(self) -> None:
    if not torch.cuda.is_available():
        return
    device = next(self.text_encoder.parameters()).device
    with torch.inference_mode():
        # 跑一遍 dummy 输入，触发三条编译路径的实际 compile + CUDA Graph capture
        try:
            text_tokens = torch.zeros(1, 16, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(text_tokens)
            self._encode_text(text_tokens, attention_mask)
        except Exception as exc: ...
        try:
            image = torch.zeros(1, 1, 3, 180, 320, dtype=torch.bfloat16, device=device)
            self._encode_image(image, self.num_frames, 180, 320)
        except Exception as exc: ...
        try:
            dummy_latent = torch.zeros(1, 16, self.num_frame_per_block, ...)
            ...
            self._cudagraph_mark_step_begin()
            self.vae.decode(dummy_latent_denorm, return_dict=False)
        except Exception as exc: ...
    torch.accelerator.synchronize(device)
```

torch.compile 是 **lazy** 的：第一次真正调用才编译 + 捕获图，会有几百 ms~秒级的首次延迟。warmup 用 dummy 输入提前触发，使首帧真实推理不被编译开销污染。每条路径独立 `try/except`，单条 warmup 失败不影响其它。

部署侧前提：`dreamzero.yaml:15` 的 `enforce_eager: false`——`true` 会强制 eager、整个编译路径被关掉。

---

### 5. VAE 编译补丁 `wan_vae_feat_cache_patch.py`

`fullgraph=True` 不允许 graph break，而 Wan VAE 原始 `feat_cache` 用 Python `None` / list 作为「首帧标记」，属于动态控制流，无法编译。补丁把标记替换成 **scalar int8 张量**：

```python
def is_marker(entry):  return entry.numel() == 1 and entry.dtype == torch.int8
def make_marker(x):    return torch.zeros(1, device=x.device, dtype=torch.int8)   # 标记值 0
```

这样 feat_cache 全程是张量，控制流静态化，且 `setup_compile` 里编译的是 `decoder.forward`（无 Python 帧循环）而非 `_decode`。

---

### Graph 编译小结（数据流视角）

```
一次 forward(req):
  _encode_text        ─ compiled(reduce-overhead, CUDA Graph)
  _encode_image       ─ compiled(reduce-overhead, CUDA Graph)
  vae.encode/_encode  ─ compiled(reduce-overhead, CUDA Graph)
  _prefill_kv_cache ─┐
  diffuse 去噪循环  ─┴─ 每步调 DiT block.forward ─ compiled(default, 仅 Inductor 融合)
  decode_video_latents ─ VAE decode（feat_cache 补丁后可编译）

横切关注点:
  每进编译区前  → _cudagraph_mark_step_begin()
  StepCache 开  → predict_noise 输出 .clone()（防静态 buffer 被覆写）
  启动时        → warmup_compile() 预热
```

---

## 第二部分：StepCache（速度向量相似度跳步）

### 6. 它解决什么问题

StepCache **不是** block 级的 Cache-DiT（那个缓存单次 forward 内部某些 block 的输出）。StepCache 跳过的是 **整个 DiT forward**，作用在去噪循环的 **scheduler step 之间**。

直觉（`stepcache/__init__.py:4-8`、`config.py:19-25`）：flow-matching 去噪中，相邻 step 预测出的 **velocity（flow）向量** 在接近收敛时方向几乎不变。于是测连续两步 velocity 的**余弦相似度**，足够高就复用上一步预测、跳过接下来若干步的 DiT。

---

### 7. 配置 `StepCacheConfig`

`stepcache/config.py:17-51`

```python
@dataclass(frozen=True)
class StepCacheConfig:
    enabled: bool = True
    min_history_steps: int = 2                       # 前 2 步必跑（无历史可比）
    max_history: int = 2                             # 历史只留 2 个
    sim_thresholds:  tuple[float, ...] = (0.95, 0.93)  # 相似度阈值（降序）
    skip_countdowns: tuple[int, ...]   = (4, 2)        # 命中后跳几步
```

`sim_thresholds` 与 `skip_countdowns` 一一对应：**越相似 → 跳越多步**。
- sim > 0.95 → 接下来跳 4 步
- sim > 0.93 → 接下来跳 2 步

`from_diffusion_cache_config()` 从 `DiffusionCacheConfig`（YAML 来的）构建，并强校验两个 tuple 等长：

```python
if len(thresholds) != len(countdowns):
    raise ValueError("velocity_sim_thresholds and velocity_skip_countdowns must have the same length; ...")
```

部署侧（`dreamzero.yaml:26-30`）：
```yaml
cache_backend: step_cache
cache_config:
  step_cache_dit_enabled: true
  velocity_sim_thresholds: [0.95, 0.93]
  velocity_skip_countdowns: [4, 2]
```

`frozen=True`：配置不可变，运行期只改 state，不改 config。

---

### 8. 状态机 `StepCacheState.should_run_step` —— 核心决策

`stepcache/state.py:13-47`

```python
class StepCacheState:
    def __init__(self, config):
        self.config = config
        self.skip_countdown = 0           # 唯一的可变状态

    def reset(self):
        self.skip_countdown = 0

    def should_run_step(self, prev_predictions):
        if not self.config.enabled:
            return True                                   # (a) 关了就每步都跑

        if len(prev_predictions) < self.config.min_history_steps:
            return True                                   # (b) 前 2 步必跑

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False                                  # (c) 倒计时中 → 跳，递减
        if self.skip_countdown == 1:
            self.skip_countdown = 0
            return True                                   # (d) 倒计时最后一格 → 这步跑（刷新缓存）

        # (e) 不在倒计时 → 现算相似度
        v_last = prev_predictions[-1][0].flatten(1).float()
        v_prev = prev_predictions[-2][0].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        for threshold, countdown in zip(self.config.sim_thresholds, self.config.skip_countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False                              # (f) 命中阈值 → 这步也跳，设倒计时

        return True                                       # (g) 都不命中 → 跑
```

逐分支：

- **(a)** 全局开关。
- **(b)** 没有「连续两步」就无法算相似度，强制跑——所以 `min_history_steps=2`。
- **(c)(d)** 倒计时机制：一旦决定跳 N 步，就靠 `skip_countdown` 连续返回 False，跳到最后一格 (d) 返回 True 重新跑一次刷新缓存。**注意 (d) 跑完后没有立刻重算相似度**，下一次进来才走到 (e)。
- **(e)** 取历史里最后两个 velocity，`flatten(1)` 成 `[B, -1]`，`.float()` 升精度算余弦，再 `.mean()` 跨 batch/标量化。
- **(f)** 阈值表**降序**遍历，命中最高的那档就设对应倒计时并跳过当前步。
- **(g)** 相似度不够，老实跑。

`trim_history`（`:49-52`）把历史裁到 `max_history=2`，内存恒定。

---

### 9. 在去噪循环里的接线 `diffuse()`

`pipeline_dreamzero.py:783-888`（节选关键路径）

```python
# 循环外初始化
_cached_flow_pred = None
_cached_flow_pred_action = None
_prev_predictions: list[tuple[torch.Tensor]] = []
_step_cache = get_stepcache_state(self) if is_stepcache_active(self) else None

for index in range(len(timesteps_video)):
    ...
    # ① 问 StepCache：这步要不要真跑 DiT？
    run_dit = _step_cache is None or _step_cache.should_run_step(_prev_predictions)

    if run_dit:
        # ② 真跑：构造 pos/neg kwargs → CFG 并行 → DiT forward
        noise_pred = self.predict_noise_maybe_with_cfg(...)
        flow_pred, flow_pred_action = noise_pred
        _cached_flow_pred = flow_pred                  # ③ 缓存（已在 predict_noise 里 clone 过）
        _cached_flow_pred_action = flow_pred_action

        _prev_predictions.append((flow_pred,))         # ④ 只把 video velocity 入历史
        if _step_cache is not None:
            _step_cache.trim_history(_prev_predictions)
    else:
        # ⑤ 跳过：复用缓存的预测，省一次完整 DiT forward
        assert _cached_flow_pred is not None
        flow_pred = _cached_flow_pred
        flow_pred_action = _cached_flow_pred_action

    # ⑥ 无论跑没跑 DiT，scheduler.step 每步都执行
    noise_pred_tuple = (flow_pred.transpose(1, 2), flow_pred_action)
    step_output = video_action_scheduler.step(noise_pred_tuple, t, latents, ...)
    noisy_input, noisy_input_action = step_output[0]
    noisy_input, noisy_input_action = self._synchronize_cfg_parallel_step_output(...)
```

要点：

- **③ 缓存的 clone 来源**：`flow_pred` 是 `predict_noise` 返回的，那里已经在 `is_stepcache_active` 时 clone 过（§3）。所以这里直接存引用即可，复用时不会被 CUDA Graph 静态 buffer 覆写。
- **④ 历史只存 video velocity**：`_prev_predictions.append((flow_pred,))`，相似度判断只看视频流；action velocity 不参与判断，但跳步时也跟着 video 一起复用（⑤）。
- **⑥ 省的只是 DiT，不省 scheduler**：被跳过的步里，scheduler 仍然用「缓存的 velocity」做一次积分推进 latent。所以跳步**不改变去噪步数**，只是这些步重用上一次的速度场——省下的是最贵的 DiT forward（含 CFG 的两次前向）。

---

### 10. 后端装配 `StepCacheBackend`

`stepcache/backend.py:86-129`

```python
class StepCacheBackend(CacheBackend):
    def enable(self, pipeline):
        pipeline_type = pipeline.__class__.__name__
        if pipeline_type in CUSTOM_STEPCACHE_ENABLERS:          # 仅 DreamZeroPipeline
            CUSTOM_STEPCACHE_ENABLERS[pipeline_type](pipeline, self.config)
        else:
            raise ValueError(f"step_cache backend does not support {pipeline_type}. ...")
        self.enabled = True

    def refresh(self, pipeline, num_inference_steps, verbose=True):
        state = get_stepcache_state(pipeline)
        if state is not None:
            state.reset()                                       # 每次新生成清倒计时
```

`enable_dreamzero_stepcache`（`:63-77`）把 config/state 挂到 pipeline 的私有属性上：

```python
def enable_dreamzero_stepcache(pipeline, config):
    cache_config = StepCacheConfig.from_diffusion_cache_config(config)
    setattr(pipeline, STEP_CACHE_CONFIG_ATTR, cache_config)     # "_stepcache_config"
    setattr(pipeline, STEP_CACHE_STATE_ATTR, StepCacheState(cache_config))  # "_stepcache_state"
```

之后 `is_stepcache_active(pipeline)`（`:35-38`）就靠读 `_stepcache_config.enabled` 判断。`CUSTOM_STEPCACHE_ENABLERS` 只注册了 `DreamZeroPipeline`（`:79-81`），别的 pipeline 用 `step_cache` 后端会直接报错。

---

### StepCache 小结（生命周期视角）

```
启动:   backend.enable(pipeline)
          → enable_dreamzero_stepcache → 挂 _stepcache_config / _stepcache_state

每次生成: backend.refresh → state.reset()（skip_countdown=0）

去噪循环每步:
  should_run_step(history)?
    ├─ True  → 跑 DiT → clone 缓存 → 入历史(只 video velocity) → trim
    └─ False → 复用缓存 velocity（video+action）
  → scheduler.step 照常推进 latent（每步都做）

判据: cosine_sim(最近两步 video velocity)
        > 0.95 → 跳 4 步
        > 0.93 → 跳 2 步
```

---

## 附：两条优化的协作关系

1. **编译让"真跑"更快**：DiT block 被 Inductor 融合，编码器/VAE 走 CUDA Graph。
2. **StepCache 让"跑得更少"**：相邻步速度场相似时整步跳过 DiT。
3. **交点是那个 `.clone()`**（`pipeline_dreamzero.py:314-317`）：CUDA Graph 的静态输出 buffer 与 StepCache 的"跨步复用"天然冲突，必须 clone 才能让两者共存。这是把两套机制接到一起时最容易踩的坑。
```
