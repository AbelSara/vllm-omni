# DreamZero KV Cache 生命周期：建立 / 使用 / 更新

> 本文走读 DreamZero 自注意力 KV cache 的完整生命周期。
> 涉及文件：
> - `vllm_omni/diffusion/models/dreamzero/state_dreamzero.py`（cache 数据结构与读写接口）
> - `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`（自注意力读写 cache）
> - `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py`（prefill 触发与写回）
>
> 维度约定（14B 配置）：`dim=5120`、`num_heads=40`、`head_dim=128`、`num_layers=40`、
> `frame_seqlen=220`、`max_attention_size = 21 × 220 = 4620`。

---

## 0. 一句话概览

KV cache 存的是**历史视频帧经 RoPE 后的 K 和原始 V**（不含 action/state token，滑窗截断到 4620），
按 **40 层 × CFG 两分支** 组织，存在 `DreamZeroState.kv_cache` / `kv_cache_neg`（GPU 显存，每 session 一份）。
**只在 prefill 写入干净观测**，去噪循环**只读不写**——这就是世界模型"累积历史、查询过去预测未来"的机制。

---

## 1. 数据结构

`state_dreamzero.py:78-81`：

```python
self.kv_cache: list[torch.Tensor] | None         # cond（正）分支
self.kv_cache_neg: list[torch.Tensor] | None     # uncond（负）分支，给 CFG
self.crossattn_cache / _neg                       # 另一套，cross-attn 用（不在本文范围）
```

- 一个 session 一份 state，KV cache 挂在 state 上。
- `kv_cache` 和 `kv_cache_neg` 是 **CFG 两个分支各自独立的历史**。
- 每个是 **长度 40 的 list**（每层一个张量）。

单层张量形状：

| 维度 | 含义 |
| --- | --- |
| `2` | K / V |
| `B` | batch |
| `seq_history` | 累积的历史 token 数（从 0 增长，最多 4620）|
| `num_heads`（tp 切分后）| 注意力头数 |
| `head_dim = 128` | 每头维度 |

---

## 2. 建立（create）

`state_dreamzero.py:117-135`，**仅在 session 第一帧**（`current_start_frame == 0`）调用：

```python
def create_kv_caches(self, batch_size, dtype, device, num_layers, num_heads, head_dim):
    self.kv_cache = [
        torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(num_layers)        # 40 层各一个
    ]
    self.kv_cache_neg = [ ... 同上 ... ]
    self.crossattn_cache = [{"is_init": False, "k": None, "v": None,
                             "k_img": None, "v_img": None} for _ in range(num_layers)]
    self.crossattn_cache_neg = [ ... ]
```

- 初始 `seq_history = 0`（空历史）。
- 调用点：`pipeline_dreamzero.py` 的 `_prefill_kv_cache`，在 `current_start_frame == 0` 分支里。

---

## 3. 写入 / 更新（write）

### 3.1 触发：prefill

`pipeline_dreamzero.py:479-600`，`_prefill_kv_cache`：

```
current_start_frame == 0（首帧）:
  ├─ create_kv_caches()                     # 建空 cache
  ├─ hidden_states = 首帧 image_latents（干净观测）
  ├─ timestep = 0（无噪声）
  └─ predict_noise_maybe_with_cfg(update_kv_cache=True)
       └─ current_start_frame = 1

后续 chunk（current_start_frame != 1）:
  ├─ current_ref = 当前观测帧的 VAE latent（每步新帧）
  └─ predict_noise(update_kv_cache=True)    # 把新帧写入 cache
```

要点：prefill 用 **干净观测帧 + timestep=0**，所以 cache 里存的是纯净的真实观测历史。

### 3.2 实际写回

`pipeline_dreamzero.py:323-327`（`predict_noise` 内，作为副作用）：

```python
if kwargs.get("update_kv_cache", False) and updated_kv_caches:
    state = kwargs.get("dreamzero_state", self.state)
    is_neg = kwargs.get("is_negative", False)
    for i, kv in enumerate(updated_kv_caches):
        state.update_kv_cache(i, kv, is_negative=is_neg)    # 逐层写回
```

`state_dreamzero.py:137-147`：

```python
def update_kv_cache(self, layer_index, updated_kv, is_negative=False):
    cache = self.kv_cache_neg if is_negative else self.kv_cache
    cache[layer_index] = updated_kv.clone()    # 整层替换为新的 [2, B, S_kept, n, d]
```

---

## 4. 读取 / 使用（read）

`causal_wan_model.py:555-572`，`CausalWanSelfAttention.forward` 里：

```python
updated_k = kv_cache[0]                          # 读历史 K  [B, S_hist, n, 128]
updated_v = kv_cache[1]                          # 读历史 V
new_k = torch.cat([updated_k, roped_key], dim=1) # 拼当前帧 K → [B, S_hist+S, n, 128]
new_v = torch.cat([updated_v, v], dim=1)
new_k = new_k[:, -self.max_attention_size:]      # 滑窗截断（4620）
new_v = new_v[:, -self.max_attention_size:]

# action/state token 单独拼上（它们参与注意力，但不写入持久 cache）
if action_register_length is not None:
    q_cat = torch.cat([roped_query, roped_action_query], dim=1)
    k_cat = torch.cat([new_k, roped_action_key], dim=1)
    v_cat = torch.cat([new_v, action_v], dim=1)
else:
    q_cat, k_cat, v_cat = roped_query, new_k, new_v

x = self.attn(q_cat, k_cat, v_cat)               # query 对 [历史+当前(+动作)] 做注意力
updated_kv_cache = torch.stack([new_k, new_v], dim=0)   # 准备写回的 [2, B, S_kept, n, 128]
```

存进 cache 的注意事项：

1. **K 是 RoPE 之后的**（`roped_key`，位置编码已烤进去），V 是原始的。
2. **不含 action/state token**（`causal_wan_model.py:547-553` 把它们从 `roped_key`/`v` 里剥离）；
   动作/状态只参与当前步注意力（`k_cat`/`v_cat` 临时拼上），不写进持久 cache。
3. **滑动窗口**：只保留最近 `max_attention_size = 4620` 个 token，更老的历史被丢弃。

---

## 5. 读写时机的关键区别

| 阶段 | `update_kv_cache` | 行为 |
| --- | --- | --- |
| **prefill** | `True` | 干净观测帧 → 算 K/V → **写回 cache** |
| **denoise 循环** | `False`（`diffuse` 的 common_kwargs，`causal_wan_model` 调用处）| 噪声 query → **只读 cache** → 不写回 |

去噪时 `self_attn` 也返回 `updated_kv_cache`，但因 `update_kv_cache=False`，
`predict_noise` 不写回——所以**预测过程的噪声状态不会污染历史**，cache 始终是纯净观测。

---

## 6. 失效 / 重建（reset）

`state_dreamzero.py:72-86` `reset()` 把 `kv_cache` 置 None、`current_start_frame=0`。

触发 reset（`pipeline_dreamzero.py:811-817`）：

| 来源 | 条件 |
| --- | --- |
| 显式 | `extra_args["reset"]`（OpenPI 切会话）|
| 自动 `should_reset`（`state_dreamzero.py:88-111`）| `language is None` / 指令变化 / 首次后又来单帧 / `current_start_frame >= local_attn_size` |
| 换 session | 不同 `session_id` 取到不同 state，新 state 的 cache 本就是 None |

reset 后下一次 forward 的 `_prefill_kv_cache` 看到 `current_start_frame==0` → 重新 `create_kv_caches`。

---

## 7. 生命周期时间线

```
session 起 → create_kv_caches (空, seq=0)
  chunk_0: prefill 首帧 → 写入 (seq=220)
           denoise×16:    只读
  chunk_1: prefill 新帧 → cat 追加 → 写入 (seq=440)
           denoise×16:    只读
  ...      持续增长，到 4620 后滑窗丢旧帧
  reset  → 重建空 cache
```

---

## 8. 与 cross-attn cache 的对比（别混淆）

| | 自注意力 KV cache | cross-attn cache |
| --- | --- | --- |
| 存什么 | 历史视频帧的 K（RoPE 后）/V | 固定 context（文本/图像）的 k/v/k_img/v_img |
| 存哪里 | `state.kv_cache` / `kv_cache_neg` | `state.crossattn_cache` 每层的 dict |
| 增长吗 | **每个 chunk 增长**（追加新帧）| **不增长**（session 内固定）|
| 写时机 | prefill 时写 | 首次 forward 写（`is_init`）|
| 含义 | 世界模型的"记忆 / 历史" | 条件的"一次性编码" |
