# DreamZero 整体计算过程：输入尺寸 / 语义 / 算子 / 矩阵变换

> 本文走读 DreamZero 一次 transformer 前向的完整计算链路，标注每步的输入/输出张量尺寸、
> 语义、使用的算子，以及矩阵乘的尺寸变换。
> 涉及文件：
> - `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py`（预处理 / 编码 / 去噪循环）
> - `vllm_omni/diffusion/models/dreamzero/causal_wan_model.py`（transformer 本体）
>
> 维度约定（14B 配置，tp=1）：`dim=5120`、`num_heads=40`、`head_dim=128`、`num_layers=40`、
> `ffn_dim=13824`、`frame_seqlen=220`、`text_len=512`、`clip_len=257`、VAE 通道 `16`、
> `action_horizon=24`、`action_dim=32`（内部）→ `8`（输出）。

---

## 0. 维度图例

| 符号 | 值 | 含义 |
| --- | --- | --- |
| `B` | 1（CFG 时 cond/uncond 各 1）| batch |
| `S` | nfpb×220 (+动作+状态) | 当前 chunk 的 token 序列长度 |
| `dim` | 5120 | 隐藏维 |
| `n / d` | 40 / 128 | 头数 / 每头维（tp=1）|
| `L` | 40 | 层数 |

---

## 1. 输入（一次 transformer 前向的入参）

| 张量 | 尺寸 | 语义 |
| --- | --- | --- |
| `x`（noise_obs ⊕ y）| `[B, 16+, nfpb, 22, 40]` | 待去噪视频 latent（16 VAE 通道）+ 图像条件 |
| `context` | `[B, 512, 4096]` | UMT5 文本编码 |
| `clip_feature` | `[B, 257, clip_dim]` | CLIP 图像特征 |
| `timestep` | `[B, F]` | 当前去噪步 |
| `kv_cache` | list[40] of `[2,B,S_hist,n,128]` | 自注意力历史 |
| `crossattn_cache` | list[40] of dict | cross-attn 固定 k/v |

来源（`pipeline_dreamzero.forward`）：
- 文本：`tokenizer`（800-809，固定 512）→ `_encode_text`（827，UMT5）→ `prompt_embeds [B,512,4096]`
- 图像（CLIP）：`_encode_image`（850，仅首帧）→ `clip_feas`
- 视频 latent：`noise_obs`（879）+ 当前观测 `_encode_vae_latents`（875，每步）

---

## 2. Patch 化（Conv3d）— `causal_wan_model.py:_forward_inference:1002`

| 算子 | 输入 → 输出 | 说明 |
| --- | --- | --- |
| `Conv3dLayer(patch=(1,2,2))` | `[B,16,nfpb,22,40]` → `[B,5120,nfpb,11,20]` | 2×2 空间 patch，投影到 5120 |
| `flatten(2).transpose(1,2)` | → `[B, S, 5120]` | S = nfpb×11×20 = nfpb×220 个 token |

---

## 3. 时间嵌入 — `_forward_blocks:940-943`

| 算子 | 输入 → 输出 |
| --- | --- |
| `sinusoidal_embedding_1d` | timestep `[B,S]` → `[B·S, 256]` |
| `time_embedding`（Linear+SiLU+Linear）| `[*,256]` → `[*,5120]` |
| `time_projection`（SiLU+Linear）| `[B,F,5120]` → `[B,F,6×5120]` → `[B,F,6,5120]` |

`6` = AdaLN 风格的 6 个调制参数（scale/shift/gate ×2），每 block 用。

---

## 4. 动作 / 状态 token 注入 — `_forward_blocks:917-926`

| 算子 | 输入 → 输出 | 语义 |
| --- | --- | --- |
| `action_encoder` | noise_action `[B,24,32]` → `[B, action_len, 5120]` | 动作变 token |
| `state_encoder` | state `[B,1,64]` → `[B,1,5120]` | 状态变 token |
| `cat` | `[B,S_video,5120]` ⊕ `[B,action_reg,5120]` → `[B,S,5120]` | 拼到视频 token 后面 |

之后整条序列 `S = 视频 + 动作 + 状态` 一起过 transformer——这是 DreamZero
把"预测视频"和"预测动作"统一在一个模型里的方式。

---

## 5. Context 拼装（文本 + 图像）— `_forward_blocks:945-948`

| 算子 | 输入 → 输出 |
| --- | --- |
| `text_embedding`（Linear）| `[B,512,4096]` → `[B,512,5120]` |
| `img_emb`（MLPProj）| `[B,257,clip]` → `[B,257,5120]` |
| `cat` | → `[B, 769, 5120]`（257 图像 + 512 文本，图像在前）|

---

## 6. × 40 层 Transformer Block

每层 `x [B,S,5120]` → `x [B,S,5120]`，三步：自注意力 → cross-attn → FFN，
都带 AdaLN 调制（用第 3 步的 6 个参数做 scale/shift/gate）。

### 6.1 自注意力（因果 + KV cache）— `causal_wan_model.py:504-575`

| # | 算子 | 矩阵变换 |
| --- | --- | --- |
| 1 | `qkv = QKVParallelLinear(x)` | `[B,S,5120]` —GEMM→ `[B,S,3×5120]` |
| 2 | `split` | → q,k,v 各 `[B,S,5120]` |
| 3 | `fused_qk_rms_norm(norm_q,norm_k,q,k)` | q/k RMSNorm（含 **1 次 f32 all_reduce**）|
| 4 | `unflatten(2,(40,128))` | `[B,S,5120]` → `[B,S,40,128]` |
| 5 | `causal_rope_action_apply` | RoPE（视频/动作/状态用不同频率表），形状不变 |
| 6 | 剥离 action/state 尾部 | video 部分 vs action 部分分开 |
| 7 | `cat(KV历史, 新k/v)` | `[B,S_hist,40,128]` ⊕ `[B,S,…]` → `[B,S_hist+S,40,128]` |
| 8 | 滑窗 `[:, -4620:]` | 截断到 max_attention_size |
| 9 | `attn(q,k,v)` FlashAttn | 见下方注意力核 |
| 10 | `flatten(2)` | `[B,S,40,128]` → `[B,S,5120]` |
| 11 | `o = RowParallelLinear` | GEMM → `[B,S,5120]`（含 **1 次 bf16 all_reduce**）|

**注意力核（step 9）**：
```
q·kᵀ : [B,40,S,128] × [B,40,128,S_kv] → [B,40,S,S_kv]
softmax(/√128)
   ·v : [B,40,S,S_kv] × [B,40,S_kv,128] → [B,40,S,128]
```
`attn` 用 `causal=False`——因果性靠 KV cache 结构保证（query 只看历史+自己，未来帧还没进 cache）。

### 6.2 Cross-attention（文本 + 图像）— `causal_wan_model.py:393-424`

| # | 算子 | 矩阵变换 |
| --- | --- | --- |
| 1 | 拆 context | `[B,769,5120]` → img `[B,257,5120]` + txt `[B,512,5120]` |
| 2 | `q = ColumnParallel(x)` + `norm_q` | `[B,S,5120]` → `[B,S,40,128]`（**每步算**）|
| 3 | `k,v`（文本）+ `k_img,v_img`（图像）| 文本 `[B,512,40,128]`，图像 `[B,257,40,128]`（**缓存**）|
| 4 | `attn(q,k,v)` 文本 | → `[B,S,40,128]` |
| 5 | `attn(q,k_img,v_img)` 图像 | → `[B,S,40,128]` |
| 6 | `x + img_x` | 两路注意力相加 → `[B,S,5120]` |
| 7 | `o = RowParallel` | GEMM → `[B,S,5120]`（含 **1 次 bf16 all_reduce**）|

同一个 query 分别对文本和图像做两次独立 attention 再相加——这是 I2V"同时看指令和画面"的融合方式。
`k/v/k_img/v_img` 只依赖 session 内不变的 context，故缓存（首次算、后续复用）。

### 6.3 FFN — `causal_wan_model.py:617-620`

| # | 算子 | 矩阵变换 |
| --- | --- | --- |
| 1 | `ColumnParallelLinear` | `[B,S,5120]` —GEMM→ `[B,S,13824]` |
| 2 | `GELU(tanh)` | 逐元素 |
| 3 | `RowParallelLinear` | `[B,S,13824]` —GEMM→ `[B,S,5120]`（含 **1 次 bf16 all_reduce**）|

> **每 block 的 bf16 all_reduce**：自注意力 `o` + cross-attn `o` + FFN `fc2` = **3 次**。
> × 40 层 × 前向次数 = profiling 里 4114 次 bf16 all_reduce 的来源。

---

## 7. 输出头 + 拆分 — `_forward_blocks:957-975`

| 算子 | 输入 → 输出 | 语义 |
| --- | --- | --- |
| `head`（norm + Linear）| 视频 token `[B,S_v,5120]` → `[B,S_v,16×4]` | 视频噪声预测 |
| `unpatchify` | → `[B,16,nfpb,22,40]` | 还原成 latent 形状 |
| `action_decoder` | 动作 token `[B,24,5120]` → `[B,24,32]` | 动作噪声预测 |

返回 `(video_noise_pred, action_noise_pred, updated_kv_caches)`。

---

## 8. 外层：去噪循环与后处理（`pipeline_dreamzero.py`）

```
prefill:    干净观测 → transformer(timestep=0, update_kv_cache=True)
              └─ 自注意力把新帧 K/V 写进 cache
denoise×16: 噪声 → transformer(timestep>0, update_kv_cache=False)
              └─ 自注意力读 cache 历史（不写回）
              └─ cross-attn 读固定 context（缓存命中）
              └─ scheduler 去噪一步：视频用标准 CFG，动作只取 cond 分支
```

去噪 16 步后：
- `noise_obs` → 干净视频 latent
- `noise_action` → 干净归一化动作

后处理（`pipeline_dreamzero.py:963-980`）：
- `_denormalize_action`：q99 反归一化 `[B,24,32]`
- 相对→绝对：关节动作前 7 维 += 当前关节状态
- `transform_action_output` → `actions [24, 8]`（7 关节 + 1 夹爪）

---

## 9. 最终输出

```python
DiffusionOutput(output={
    "actions": [24, 8],          # 主输出：未来 24 步控制（部署用）
    "video":   [B, nfpb, 16, 22, 40],   # 副产物：视频 latent（VAE 解码后可视化）
})
```

---

## 9.5 计算量（FLOPs）与通信量（TP / CFG 并行）

### 记号

| 符号 | 含义 | 例值 |
| --- | --- | --- |
| `B` | 每卡 batch | 1 |
| `S` | query token 数（当前 chunk，nfpb×220 + 动作/状态）| ~600 |
| `S_kv` | 自注意力 KV 长度（历史+当前，≤4620）| 4620 |
| `dim` | 隐藏维 | 5120 |
| `ffn` | FFN 中间维 | 13824 |
| `text` / `clip` | 文本 / 图像 context 长度 | 512 / 257 |
| `T` | TP size | 2 |
| `bf16` | 每元素字节 | 2 B |

GEMM FLOPs 通式：`[B·S, K] × [K, N]` = **2·B·S·K·N**（乘加各算 1）。

### 9.5.1 每个算子的计算量（单层 block）

| 算子 | 单卡 FLOPs | TP=T 每卡 | TP 怎么切 | 是否缓存 |
| --- | --- | --- | --- | --- |
| 自注意力 QKV | `6·B·S·dim²` | `/T` | 输出按头切（Column）| 否 |
| 自注意力 核 (qkᵀ+sv) | `4·B·S·S_kv·dim` | `/T` | 按头切 | — |
| 自注意力 o | `2·B·S·dim²` | `/T` | 输入按头切（Row）| 否 |
| cross q | `2·B·S·dim²` | `/T` | Column | 否（每步算）|
| cross k/v（文本）| `4·B·text·dim²` | `/T` | Column | **是**（首次）|
| cross k_img/v_img（图像）| `4·B·clip·dim²` | `/T` | Column | **是**（首次）|
| cross 核（文本+图像）| `4·B·S·(text+clip)·dim` | `/T` | 按头切 | — |
| cross o | `2·B·S·dim²` | `/T` | Row | 否 |
| FFN fc1 | `2·B·S·dim·ffn` | `/T` | Column | 否 |
| FFN fc2 | `2·B·S·ffn·dim` | `/T` | Row | 否 |
| qk-norm / RoPE / GELU | `~O(B·S·dim)` | 基本不变 | 逐元素 | — |

> **TP 把每个 GEMM 的计算量切成 1/T**（按头 / 按中间维切分）。
> **缓存命中后**，cross 的 k/v/k_img/v_img 投影在稳态步不再计算（只首次算）。

单层 GEMM 计算量（代入 B·S=1，单 token，单卡）：

```
QKV      6·dim²            = 6 × 5120²        ≈ 157.3 MFLOP
self o   2·dim²            = 2 × 5120²        ≈  52.4 MFLOP
cross q  2·dim²                               ≈  52.4 MFLOP
cross o  2·dim²                               ≈  52.4 MFLOP
FFN fc1  2·dim·ffn         = 2 × 5120 × 13824 ≈ 141.6 MFLOP
FFN fc2  2·ffn·dim                            ≈ 141.6 MFLOP
─────────────────────────────────────────────────────────
GEMM 小计/token/block                          ≈ 597.7 MFLOP
注意力/token/block: self 4·S_kv·dim + cross 4·(text+clip)·dim
                  = 4×4620×5120 + 4×769×5120  ≈  94.6 + 15.7 = 110.3 MFLOP
─────────────────────────────────────────────────────────
合计/token/block                               ≈ 708 MFLOP
× 40 层                                         ≈ 28.3 GFLOP / token
```

### 9.5.2 通信量（TP：all_reduce）

TP 下，每层有 **3 次 bf16 all_reduce**（self o、cross o、FFN fc2），各 reduce 一个 `[B,S,dim]` 张量：

| 量 | 公式 | 例值（B=1,S=600）|
| --- | --- | --- |
| 单次 all_reduce 张量大小 `M` | `B·S·dim·2` B | 600×5120×2 ≈ **6.14 MB** |
| 单层 bf16 通信 | `3·M` | ≈ 18.4 MB |
| **单次前向（40 层）** | `120·M` | ≈ **737 MB** |
| qk-norm f32 all_reduce | `B·S·(小常数)·4` B | 每次几 KB，**可忽略** |

Ring all_reduce 每卡实际线上传输 ≈ `2·M·(T-1)/T`；T=2 时 ≈ `M`。
所以 TP2 单次前向每卡 all_reduce 线上流量 ≈ **737 MB**，且**全暴露在关键路径**（同步）。

> 对应 profiling：120 bf16 all_reduce/前向 × 34 前向 ≈ **4080 次** ≈ 实测 4114。
> 小消息下 PCIe 是**延迟受限**（非带宽受限），所以实测耗时远高于"流量÷带宽"。

### 9.5.3 通信量（CFG 并行：all_gather）

CFG 并行切的是 **cond/uncond 维度**：一卡跑 cond、一卡跑 uncond，**每卡跑一次完整前向，层内零通信**。
只在**每个去噪步末尾**合并两分支的噪声预测做一次 all_gather：

| 量 | 公式 | 例值（nfpb=3）|
| --- | --- | --- |
| 单次 all_gather 张量 | 视频 latent `[B,16,nfpb,22,40]` + 动作 `[B,24,32]` | 16×3×22×40×2 ≈ **845 KB** |
| 单步通信 | 1 次 all_gather | < 1 MB |
| 单次前向（16 步）| 16 次 | ≈ 13 MB |

> 对应 profiling：CFG2 共 33 次 all_gather、总 27 ms。

### 9.5.4 TP vs CFG 对比（每次前向，单卡）

| | 计算量/卡 | 通信次数/前向 | 通信量/前向 | 精度 |
| --- | --- | --- | --- | --- |
| **单卡** | 全量 | 0 | 0 | 基准 |
| **TP=2** | 全量 **/2** | 120 次 all_reduce | ≈ 737 MB | bf16 求和顺序变 → ~0.065 误差 |
| **CFG=2** | 全量（半 batch）| 16 次 all_gather | ≈ 13 MB | all_gather 确定性 → **0 误差** |

**结论**：
- TP 把每个 GEMM 算量切 1/T，但代价是**每层 3 次 all_reduce**（120 次/前向、~737 MB），且引入 bf16 求和精度漂移。
- CFG 不切单卡算量（每卡跑完整前向），但通信**只在步末一次 all_gather**（~57× 少于 TP），且零精度损失。
- 这就是为什么 DreamZero 这种**短序列 decode-like** 负载上，**CFG 通信开销远小于 TP**，且精度更好。

---

## 10. 串起来的一句话

> DreamZero 把相机图像（CLIP 语义锚点 + VAE 结构）、状态、文本指令编码成条件，
> 从噪声出发，让**视频 latent 和动作并排作为 token** 在同一个 40 层因果 transformer 里做 16 步 diffusion 去噪：
> 自注意力看历史（KV cache，世界模型）、cross-attn 看图像和文本（条件），
> 最后视频 token 解码出预测画面、动作 token 解码出未来 24 步控制指令。
> 视频是副产物，**动作才是部署时要的输出**。
