# Session-Aware Cache 设计方案（diffusion 通用层 + DreamZero 接入）

> 对应 RFC workstream：*KV preallocation / host-side cleanup*，并为 *torch.compile / CUDA Graph* 工作流打基础。
> 架构参考 SGLang [PR #19171](https://github.com/sgl-project/sglang/pull/19171)
> （`SessionAwareCache` + streaming session），对照分析见 §3.1。
>
> **定位更新**：《vLLM-Omni Diffusion 统一 KV Cache 管理》RFC（BDE，World Model
> RFC #1987 的 KV 条目）发布后，本方案与其的融合分析见 §10——结论：本方案
> 即 BDE 留白的 workstream B / WP-7（session 生命周期），可先行落地；
> 存储层（§4.2/4.3）在 BDE paged 栈就绪后由其接管。
>
> 关联文档：`dreamzero_kvcache_lifecycle.md`（现有 KV cache 生命周期走读）、`dreamzero_compute_flow.md`。
>
> 涉及文件：
> - **通用层（新增）**：`vllm_omni/diffusion/cache/session_state_cache.py`、
>   `vllm_omni/diffusion/models/interface.py`、
>   `vllm_omni/diffusion/worker/diffusion_model_runner.py`
> - **DreamZero 接入**：`vllm_omni/diffusion/models/dreamzero/{pipeline_dreamzero.py, state_dreamzero.py, causal_wan_model.py}`
> - **控制面**：`vllm_omni/entrypoints/openpi/{serving.py, connection.py}`
> - **配置**：`vllm_omni/diffusion/data.py`（`OmniDiffusionConfig`）、`vllm_omni/deploy/dreamzero*.yaml`

---

## 1. 背景与动机

DreamZero 是流式闭环的自回归视频-动作世界模型（40 层 Causal-WAN DiT，~14B）。
每个机器人控制步是一次 forward：新相机帧 + 持久 KV cache → (video, action)。
**单步延迟直接决定控制频率**，而 session（一台机器人一次连续的控制回合）的
KV cache 是跨步持久状态，它的分配、增长、清理方式同时影响：

1. **单步延迟**：当前 prefill 写 cache 的路径每步做全窗 `cat` + `clone`；
2. **显存安全**：多 session 并发时没有任何准入控制，OOM 只能靠运气；
3. **后续优化的可行性**：CUDA Graph / torch.compile 需要稳定的张量地址，
   现在每步重分配的 cache 是硬阻碍。

### 1.1 现状盘点（代码事实）

| 能力 | 现状 | 代码位置 |
| --- | --- | --- |
| session → state 映射 | 有，`OrderedDict` LRU，上限 `MAX_DREAMZERO_SESSIONS=64` | `pipeline_dreamzero.py:54,267-278` |
| KV cache 分配 | **lazy**：seq 维从 0 开始，每步 `cat` 重新分配 | `state_dreamzero.py:117-135`，`causal_wan_model.py:478-495` |
| KV cache 更新 | `cat`（全窗拷贝）→ `stack`（再拷贝）→ `update_kv_cache` 里 `.clone()`（第三次拷贝） | `causal_wan_model.py:480-495`，`state_dreamzero.py:137-147` |
| 滑动窗口 | `[:, -max_attention_size:]` 截断（4620 token = 21 帧） | `causal_wan_model.py:482-483` |
| 显存预算 / 准入控制 | **无**。LRU 只按"个数"淘汰，且只在第 65 个 session 到来时触发 | `pipeline_dreamzero.py:273-276` |
| 显式释放 | **无**。`ServingRealtimeRobotOpenPI.reset()` 是空操作；WebSocket 断连不释放任何 GPU 状态 | `serving.py:119-120`，`connection.py:123-126` |
| TTL / 空闲回收 | 无 | — |
| 指标 / 可观测性 | 无 | — |

### 1.2 量化问题

**(a) 显存：单 session 满窗占用（14B 配置，bf16）**

每层每分支：`2(K/V) × B × 4620 × num_heads/TP × 128 × 2B`。

| TP | 单层单分支 | ×40 层 ×2 分支（pos+neg） | + cross-attn cache（512 文本 token×2 分支） | 合计/每 rank |
| --- | --- | --- | --- | --- |
| 1 | ~90 MiB | ~7.0 GiB | ~0.8 GiB | **~7.9 GiB** |
| 4 | ~22.5 MiB | ~1.76 GiB | ~0.2 GiB | **~2.0 GiB** |

`MAX_DREAMZERO_SESSIONS=64` 与现实显存预算严重脱节：TP=1 下 3 个活跃满窗
session 就可能挤掉权重以外的全部空间。**没有按字节的预算与准入控制，
多 session 服务实际上不可用。**

**(b) 带宽 / 分配器：每个 prefill 步的浪费**

稳态（窗已满）下每步每分支每层：
- `cat`：重新分配并拷贝整窗 4620 token（~90 MiB 写）；
- `stack`：再拷贝一次（~90 MiB）；
- `update_kv_cache` 的 `.clone()`：第三次（~90 MiB）。

40 层 × 2 分支 ≈ **每步 ~21 GiB 纯冗余 D2D 流量 + 240 次新分配**，
而真正新增的信息只有 880 token（4 帧 × 220）≈ 整窗的 19%。
同时每步分配/释放导致 allocator 抖动，且张量地址不稳定，CUDA Graph 无法捕获。

**(c) 生命周期：泄漏式持有**

机器人断开 / 回合结束后，7.9 GiB 的 state 仍挂在 `_states` 里，
直到第 65 个不同 session_id 出现才被 LRU 顶掉。`endpoint=reset` 的
WebSocket 消息到达 serving 层后被丢弃（空操作），真正的 reset 依赖
"下一次 infer 捎带 `reset=True`"——如果客户端不再发请求，显存永远不还。

---

## 2. 目标与非目标

### 目标

1. **通用 SessionStateManager（diffusion runner 层，模型无关）**：集中管理
   session 生命周期——创建、触摸（touch）、显式释放、TTL 空闲回收、
   按显存字节预算的 LRU 淘汰与准入控制；DreamZero 通过 protocol 接入，
   而非自管（架构对照与分层判定见 §3）。
2. **KV 预分配 + 原地更新**：每 session 一次性分配固定容量的 KV slab，
   prefill 只原地写入新 token；消除 `cat/stack/clone` 三连拷贝与每步分配。
3. **Slab 复用池**：session 释放后 slab 归还 free pool，新 session 直接复用
   （同一部署内形状完全一致），避免反复 `cudaMalloc`/`cudaFree` 与碎片。
4. **Host-side cleanup 打通**：`endpoint=reset`、WebSocket 断连、TTL 三条路径
   都能真正释放 worker 内的 GPU 状态。
5. **可观测**：session 数、占用字节、淘汰/复用计数、命中率等指标。
6. **CFG-Parallel / TP 安全**：多 rank 下淘汰决策确定性一致。
7. **为 CUDA Graph 铺路**：cache 地址在 session 生命周期内固定。

### 非目标（本期不做，仅留接口）

- KV cache 跨节点迁移 / KV connector 接入（与 `kv_transfer` 工作流分开）。
- 本方案内不自建 paged 化。~~把 DreamZero cache 迁到 vLLM paged KV block
  manager~~——这条原判断已被 BDE RFC 推翻（其 `BDERequestAdapter` +
  `ChunkWindowManager` 兼容层吸收了语义差异）；迁移路径与时序见 §10，
  但 paged 后端落地前，本方案的存储层仍是有效的过渡实现。
- CPU offload 二级缓存（设计预留 hook，作为 Phase 4 可选项）。
- 改变滑窗 / reset 语义（`should_reset` 逻辑保持不变）。

---

## 3. 总体架构：diffusion 通用 session 层 + 模型私有 cache 实现

### 3.1 参考实现对照：SGLang `SessionAwareCache`（PR #19171）

SGLang 为 streaming session 做了同主题的优化，其结构值得对照：

| SGLang 机制 | 做法 | 在 vLLM-Omni diffusion 侧的对应 |
| --- | --- | --- |
| `SessionAwareCache` | **装饰器**包在任意 `BasePrefixCache`（RadixCache 等）外面；streaming 请求绕过 radix 前缀匹配，直接继承上一请求的 KV | 无 radix/前缀匹配可绕过；对应物是"绕过 lazy 重建，直接复用 session 持久 state" |
| `SessionSlot` | 持有 `req_pool_idx` / `kv_committed_len` / `kv_allocated_len`，请求结束时把 KV **所有权**从 req 转移到 slot（仅元数据，零拷贝） | `SessionEntry` 持有 state 对象（slab 所有权），session 内跨请求天然零拷贝 |
| 全局 KV 池 | 所有 AR 模型共享 `token_to_kv_pool` + `req_to_token_pool`，slot 只记池内索引 | **不存在**：DiT 各 pipeline 自持张量，多数 DiT 根本没有 KV cache → 用「opaque state + 类型化 slab pool」替代「池内索引」 |
| `release_session()` / `reap_timed_out_sessions()` | scheduler 统一的显式释放 + 超时回收 | manager 的 release 控制面 + 逻辑时钟 TTL（§4.4/§4.6） |
| `session_held_tokens()` | session 持有量上报，供 idle memory 泄漏自检扣除 | `stats()` 的 bytes 账本（§4.7） |

**结论：模式可全局化，机制不能照搬。**

- SGLang 能把 session 层做成全局装饰器，前提是所有模型共享同一套 paged
  KV 池和统一的 `BasePrefixCache` 接口——session 层只搬"指针"，不碰布局。
- vLLM-Omni 的两条路径都不具备这个前提：
  - **AR 路径**的 KV 池在 vLLM core（paged block manager + prefix caching），
    在那里加 session 层意味着 patch vLLM 核心类，属 `patch.py` 高危区，
    且 DreamZero 根本不走该路径——**不做**（见 §8 风险表）；
  - **diffusion 路径**没有共享 KV 池，DreamZero 的 cache 是连续滑窗布局、
    attention 非 paged——SGLang 的"索引转移"无对应物。
- 但 SGLang 设计中**与池无关的部分**——session 生命周期、所有权槽位、
  显式释放/超时回收、持有量自检——全部可以提升为 diffusion 通用层。
  挂载点现成：`DiffusionModelRunner` 已经承载了三个跨模型横切组件
  （`prompt_embed_cache`、`cache_backend`、`kv_transfer_manager`，
  见 `diffusion_model_runner.py:76-84`），session 层是第四个。

### 3.2 分层判定

| 能力 | 归属 | 理由 |
| --- | --- | --- |
| session → state 映射、逻辑时钟、TTL/LRU/准入、bytes 账本、stats | **通用层**（runner 级） | 与模型无关，纯请求流的函数 |
| slab 分配池（shape+dtype → free list） | **通用层** | 张量复用与碎片治理对任何持久 state 模型同样适用 |
| release 控制请求解析、三条清理路径 | **通用层** | runner 在 `pipeline.forward()` 之前统一拦截，不进模型代码 |
| state 内容（KV slab 布局、crossattn、帧缓冲）、`estimated_bytes` 计算 | **模型私有**（protocol 实现） | 布局/语义由模型定义 |
| attention 原地 append、滑窗 memmove、`should_reset` | **模型私有** | DreamZero 的 Causal-WAN 语义 |

### 3.3 架构图

```
                    ┌─────────────────────────────────────────────┐
 WebSocket 层        │  RobotRealtimeConnection                    │
 (connection.py)    │   · infer(session_id)                       │
                    │   · endpoint=reset ──┐                      │
                    │   · disconnect ──────┤                      │
                    └──────────┬───────────┼──────────────────────┘
                               │ infer 请求 │ release 控制请求
                               ▼           ▼      (同一 generate() 通道，
                    ┌─────────────────────────────────────────────┐  天然与在途 step 串行)
 Serving 层          │  ServingRealtimeRobotOpenPI                 │
                    └──────────┬──────────────────────────────────┘
                               ▼
 ╔═══════════════════════════════════════════════════════════════╗
 ║ 通用层 (模型无关, 每个 TP/CFG rank 各一份)                        ║
 ║                                                               ║
 ║  DiffusionModelRunner.execute_model()                         ║
 ║   ├ extra_args.control == release_session                    ║
 ║   │    → manager.release(sid)，不进 pipeline                  ║
 ║   └ 推理请求且 supports_session_state(pipeline):               ║
 ║        req.session_state = manager.acquire(sid)               ║
 ║                  │                                            ║
 ║                  ▼                                            ║
 ║  SessionStateManager (diffusion/cache/session_state_cache.py)║
 ║   · sessions: OrderedDict[sid, SessionEntry]                  ║
 ║   · 预算账本 bytes_used / bytes_budget                         ║
 ║   · 逻辑时钟 (请求序号, 跨 rank 确定性)                          ║
 ║   · TTL 扫描 / LRU 淘汰 / 准入控制 / stats                      ║
 ║   · TensorSlabPool (shape+dtype → free list)                  ║
 ║   · 未命中时: pipeline.create_session_state(sid, pool)         ║
 ╚══════════════════╤════════════════════════════════════════════╝
                    ▼  SessionState protocol (estimated_bytes/release/reset)
 ┌─────────────────────────────────────────────────────────────┐
 │ 模型私有层                                                    │
 │                                                             │
 │  DreamZeroPipeline.forward(req)                             │
 │   └ state = req.session_state   (不再自管 _states)            │
 │              │                                              │
 │              ▼                                              │
 │  DreamZeroState (实现 SessionState)                          │
 │   · kv: PreallocKVCache (pos / neg 按需各一, slab 取自 pool)  │
 │   · crossattn / clip_feas / ys / 帧缓冲                      │
 │              │                                              │
 │              ▼                                              │
 │  CausalWanSelfAttention.forward()                           │
 │   · prefill: kv.append() 原地写                              │
 │   · denoise: kv.view() 只读 + 临时 cat 当前帧                 │
 └─────────────────────────────────────────────────────────────┘
```

### 3.4 新增 / 改动文件

| 文件 | 内容 |
| --- | --- |
| `vllm_omni/diffusion/cache/session_state_cache.py`（新增） | `SessionCacheConfig`、`SessionState`（Protocol）、`SessionEntry`、`SessionStateManager`、`TensorSlabPool`。与同目录 `prompt_embed_cache.py` 同级，沿用既有"runner 级横切缓存"先例 |
| `vllm_omni/diffusion/models/interface.py` | 新增 `SupportsSessionState` 接口 + `supports_session_state()` 检查器（先例：`supports_step_execution`） |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | `execute_model()` 拦截控制请求、acquire 并把 state 挂到请求上；stats 周期日志 |
| `vllm_omni/diffusion/models/dreamzero/session_kv.py`（新增） | `PreallocKVCache`（DreamZero 私有布局，slab 来自通用 pool） |
| `vllm_omni/diffusion/models/dreamzero/{state_dreamzero,pipeline_dreamzero,causal_wan_model}.py` | `DreamZeroState` 实现 protocol；删除 `_states`/`MAX_DREAMZERO_SESSIONS`；attention 原地路径 |

收益（相对 DreamZero 独立实现）：未来任何流式/因果 DiT（CausVid、
Self-Forcing 风格的 streaming 世界模型）实现同一 protocol 即获得全部
生命周期/预算/清理能力；配置、指标、控制面跨模型统一；DreamZero 的
pipeline 代码反而变薄（删掉自管 LRU）。

---

## 4. 详细设计

### 4.1 配置：`SessionCacheConfig`

通用层组件，配置也放在通用位置：`OmniDiffusionConfig` 新增一级字段
`session_cache`（`vllm_omni/diffusion/data.py`），deploy YAML 的 stage
条目直接声明；兼容读取 `model_config.session_cache` 作为 fallback
（与 `enable_prompt_embed_cache` 等既有 od_config 开关同级，而不是
藏在模型私有 `model_config` 里）。

```yaml
# vllm_omni/deploy/dreamzero.yaml
stages:
  - stage_id: 0
    session_cache:
        enabled: true              # 总开关；false 时完全走旧路径（回滚开关）
        max_sessions: 8            # 硬上限（个数）；默认从预算推导，二者取小
        memory_fraction: 0.25      # KV slab 可用的 GPU 显存比例（按 rank）
        # 或 memory_budget_gb: 16  # 与 memory_fraction 二选一，显式字节预算
        idle_ttl_s: 120            # 空闲超时；0 = 不启用 TTL
        preallocate: true          # 预分配 slab（Phase 2 行为）；false = 仅生命周期管理
        admission_policy: evict_lru   # 预算不足时: evict_lru | reject
```

```python
@dataclass
class SessionCacheConfig:
    enabled: bool = True
    max_sessions: int | None = None      # None → 由预算推导
    memory_fraction: float | None = 0.25
    memory_budget_bytes: int | None = None
    idle_ttl_s: float = 120.0
    preallocate: bool = True
    admission_policy: str = "evict_lru"  # or "reject"

    def resolve_budget(self, device: torch.device) -> int:
        if self.memory_budget_bytes is not None:
            return self.memory_budget_bytes
        total = torch.accelerator.get_device_properties(device).total_memory  # 经平台抽象
        return int(total * (self.memory_fraction or 0.25))
```

`max_sessions` 与预算的关系：`effective_max = min(max_sessions, budget // bytes_per_session)`。
`bytes_per_session` 在模型加载后即可静态算出（见 4.3），启动时打印一行
日志：`session_cache: budget=16.0GiB, per_session=2.0GiB, effective_max_sessions=8`，
让部署者立刻看到容量。

### 4.2 `PreallocKVCache`：固定容量 slab + 原地滑窗（DreamZero 私有）

模型私有层组件（`dreamzero/session_kv.py`），slab 由通用 `TensorSlabPool`
提供。替换现有"每层一个 `[2,B,S_grow,n,d]` 增长张量"的结构。

**布局**：每个分支（pos / neg）一个 slab：

```
slab: Tensor[num_layers, 2, B, capacity, n_heads_tp, head_dim]   (一次分配)
capacity = max_attention_size (+ 可选少量 headroom，见溢出策略)
seq_len: int        # 当前有效 token 数（所有层一致，单一标量）
```

单张大张量而不是 40 个小张量：一次 `cudaMalloc`、地址连续、
便于整体归还 pool / 将来整体 offload。各层通过 `slab[i]` 取 view。

```python
class PreallocKVCache:
    """每 session 每分支一个。K 存 RoPE 后值、V 存原始值，与现行为一致。"""

    def __init__(self, slab: torch.Tensor):   # slab 来自通用 TensorSlabPool
        self.slab = slab                       # [L, 2, B, cap, n, d]
        self.seq_len = 0
        self.capacity = slab.shape[3]

    def layer_view(self, layer: int) -> torch.Tensor:
        # 只读历史: [2, B, seq_len, n, d]，attention 读这个
        return self.slab[layer, :, :, : self.seq_len]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        # prefill 原地写入新 token（k/v: [B, S_new, n, d]）
        s = k.shape[1]
        self.slab[layer, 0, :, self.seq_len : self.seq_len + s].copy_(k)
        self.slab[layer, 1, :, self.seq_len : self.seq_len + s].copy_(v)

    def commit(self, s_new: int) -> None:
        # 所有层 append 完后由 pipeline 统一推进（各层 seq 一致）
        self.seq_len = min(self.seq_len + s_new, self.capacity)
```

**滑窗溢出策略**（窗满后每步需要丢最老的 880 token）：

| 方案 | 做法 | 代价 | 结论 |
| --- | --- | --- | --- |
| A. 共享 scratch 搬移 | 溢出时 `tail → scratch → slab 头部` 两段拷贝；scratch 全局一个（按最大单层尺寸 ~90 MiB，跨层/跨 session 复用） | 每步每层 2 次窗内拷贝（~146 MiB/层/分支），仍比现状 3 次全窗拷贝+分配少 1/3 且零分配 | 备选 |
| B. 前向分块 memmove | 同一 slab 内 `dst=[0:W-s]` ← `src=[s:W]`，按 `chunk ≤ s`（880）从左到右分块拷贝，块间无重叠，安全无 scratch | 同 A 的流量但无 scratch、单 kernel 循环 | **推荐** |
| C. 真环形缓冲 | 写指针取模，读侧需要 gather 或两段 attention | 改 attention 接口，且 K 已含绝对位置 RoPE，两段拼接还是要物化 | 否决 |
| D. 双 slab ping-pong | 翻倍显存 | 显存是主要矛盾，不可接受 | 否决 |

注意：未满窗期间（前 5 步）完全没有搬移，只有新 token 的原地写；
满窗后每步一次 B 方案 memmove。**稳态每步 D2D 流量从 ~21 GiB 降到
~6 GiB（A/B 搬移 + 新 token 写入），且分配次数从 240 → 0。**
若再加少量 headroom（capacity = W + k·880），可把 memmove 频率摊薄到每
k 步一次，作为后续微调项。

**RoPE 正确性**：现实现把 RoPE 后的 K 存进 cache，滑窗截断保留绝对位置
（`causal_wan_model.py:482`），方案 B 的 memmove 是纯位置平移、不改内容，
语义与现状逐 bit 一致。

**CFG-Parallel 下的分支按需分配**：现状 `create_kv_caches` 同时建 pos/neg
两套；CFG-parallel 时每个 rank 实际只写自己负责的分支，另一套永远 seq=0
（lazy 模式下无成本，但预分配会浪费一半显存）。因此 slab 采用
**首次 append 时才从 pool 取**（alloc-on-first-write），无需感知 rank 角色，
单 rank CFG（两分支都本地算）也自然正确。

### 4.3 `TensorSlabPool`：slab 复用池与账本（通用层）

通用组件：按 `(shape, dtype)` 键化的 free list 集合，预算账本跨所有键
共享。模型层只声明自己要什么形状的 slab，不感知预算与复用：

```python
class TensorSlabPool:
    """Keyed slab pool. 同一部署内 DreamZero 只会用到一个键
    （[L, 2, B, cap, n_tp, d], bf16），但接口对未来模型开放。"""

    def __init__(self, device, budget_bytes):
        self.device = device
        self.budget_bytes = budget_bytes
        self.allocated_bytes = 0
        self._free: dict[tuple, list[torch.Tensor]] = defaultdict(list)

    def try_acquire(self, shape, dtype) -> torch.Tensor | None:
        key = (tuple(shape), dtype)
        if self._free[key]:
            return self._free[key].pop()
        slab_bytes = math.prod(shape) * dtype.itemsize
        if self.allocated_bytes + slab_bytes > self.budget_bytes:
            return None                       # 触发上层淘汰或拒绝
        self.allocated_bytes += slab_bytes
        return torch.empty(shape, dtype=dtype, device=self.device)

    def release(self, slab: torch.Tensor) -> None:
        self._free[(tuple(slab.shape), slab.dtype)].append(slab)  # 不 free，留给下个 session
```

- **不调用 `empty_cache`**，slab 常驻、地址稳定——这正是 CUDA Graph 需要的。
- cross-attn cache（每层 k/v，512 文本 token，session 内只写一次）体量小
  （TP=1 两分支 ~0.8 GiB），同样纳入账本但可以继续 lazy 分配；
  Phase 2 顺手并入 slab 末尾的固定区段亦可，非关键路径。
- `bytes_per_session`（用于准入计算）= kv slab ×(1 或 2 个分支，按 CFG 模式)
  + crossattn 估算 + ys/clip_feas 估算（后两者 < 100 MiB）。

### 4.4 `SessionStateManager`：生命周期与淘汰（通用层）

模型通过 protocol 接入（对应 SGLang 的 `BasePrefixCache` 接口角色——
通用层只认接口，不认布局）：

```python
# diffusion/cache/session_state_cache.py
class SessionState(Protocol):
    """模型私有 state 必须实现的最小接口（DreamZeroState 实现它）。"""
    def estimated_bytes(self) -> int: ...     # 准入/账本用（含未来增长上限）
    def release(self) -> None: ...            # 归还 slab 到 pool、断开引用
    def reset(self) -> None: ...              # 复用 entry 时清空内容

# diffusion/models/interface.py
@runtime_checkable
class SupportsSessionState(Protocol):
    def create_session_state(
        self, session_id: str, slab_pool: TensorSlabPool
    ) -> SessionState: ...

def supports_session_state(pipeline) -> bool: ...   # 先例: supports_step_execution
```

manager 本体（对应 SGLang `SessionAwareCache.slots` + `release_session`
+ `reap_timed_out_sessions` 的合体）：

```python
@dataclass
class SessionEntry:
    state: SessionState          # opaque，通用层不读内容
    created_tick: int
    last_used_tick: int          # 逻辑时钟（见下）
    last_used_wall: float        # 仅供日志告警
    steps: int = 0

class SessionStateManager:
    def __init__(self, cfg: SessionCacheConfig, pool: TensorSlabPool,
                 state_factory):          # = pipeline.create_session_state
        self._sessions: OrderedDict[str, SessionEntry] = OrderedDict()
        self._tick = 0           # 每个到达 runner 的请求 +1

    # ---- 数据面 ----
    def acquire(self, session_id: str) -> SessionState: ...
        # 命中: move_to_end + touch
        # 未命中: 准入检查(见下) → state_factory 新建 entry
        #         (state 内 slab 仍 lazy-on-first-write)

    # ---- 控制面 ----
    def release(self, session_id: str) -> bool: ...
        # state.reset() + 归还 slab 到 pool + 删除 entry
    def sweep_ttl(self) -> int: ...
        # 每次 acquire 时顺带扫描: last_used_tick 落后超过阈值的 entry → release
        # (逻辑时钟，见下；wall-clock 仅用于日志告警)
    def evict_for_admission(self) -> bool: ...
        # pool.try_acquire 失败时: 按 last_used_tick 淘汰最旧的"非当前"session
        # admission_policy == "reject" 时改为抛 SessionAdmissionError

    def stats(self) -> dict: ...
        # num_sessions / bytes_used / pool_free / evictions{lru,ttl,explicit} / reuse_hits
```

**跨 rank 确定性（关键正确性约束）**：TP 与 CFG-parallel 下，每个 rank
各持一份 manager。所有 rank 看到**同一请求流**，因此：

- LRU 序、准入触发点、`evict_for_admission` 的受害者选择全部以
  **逻辑时钟 `_tick`（请求序号）** 为准，绝不用 wall clock 排序——
  否则两个 rank 可能淘汰不同 session，导致某 session 在 rank0 还有 pos
  历史、在 rank1 的 neg 历史已被清空重建，CFG 两分支历史错位，输出悄悄劣化。
- TTL 同理不能用 wall clock 做决策：若某 rank 因时钟偏差提前回收了一个
  session，该 session 的下一个请求在这个 rank 上会从空 cache 重建、而在
  其他 rank 上继续旧历史——两分支历史错位且不会自愈。因此 TTL 规则也
  定义在逻辑时钟上：「当前请求到达时，若某 entry 的
  `last_used_tick < _tick - K` 则回收」，其中 K 直接配置为请求数（或由
  `idle_ttl_s × 估算 QPS` 换算后固化进配置）。纯逻辑时钟，rank 间严格
  一致；wall-clock 的 `idle_ttl_s` 仅用于日志告警，不参与决策。

> 设计裁定：**所有改变 cache 内容/存在性的决策只依赖请求流的纯函数**。
> 这是多 rank 一致性的充分条件，评审时按此标准检查每条淘汰路径。

DreamZero 自己的 `MAX_DREAMZERO_SESSIONS=64` 与 `_states` OrderedDict
（`pipeline_dreamzero.py:54,203-205,267-278`）整体删除，个数硬上限由
`SessionCacheConfig.max_sessions` 承担，实际生效上限由预算推导（见 4.1）。

### 4.5 Runner / attention / pipeline 改造点

**`diffusion_model_runner.py` `execute_model`**（`:278`，通用层接入点）：

```python
# pipeline.forward(req) 之前（与 prompt_embed_cache / kv_transfer 同级的横切逻辑）:
extra_args = req.sampling_params.extra_args or {}
if self.session_manager is not None:
    control = extra_args.get("control")
    if control == "release_session":
        ok = self.session_manager.release(str(extra_args.get("session_id") or "default"))
        return DiffusionOutput(output={"control_result": {"released": ok}})
    if "session_id" in extra_args:
        req.session_state = self.session_manager.acquire(str(extra_args["session_id"]))
```

manager 在 `load_model()` 末尾创建：`supports_session_state(self.pipeline)`
且配置 `enabled` 时，以 `pipeline.create_session_state` 为 factory 实例化
（不支持的 pipeline 零开销，路径不存在）。

**`pipeline_dreamzero.py` `forward`**（`:756-758`）：
`self._get_or_create_state(session_id)` 替换为
`state = getattr(req, "session_state", None) or self._fallback_state`
（fallback 仅供 `enabled=false` 回滚路径与裸调用测试）。

**`causal_wan_model.py` `CausalWanSelfAttention.forward`**（`:420-499`）：

接口从 `kv_cache: Tensor`（`[2,B,S,n,d]`）改为接受
`kv: PreallocKVCache 的 LayerHandle`（保留旧 Tensor 路径作回滚分支）：

```python
hist = kv.layer_view(layer_idx)              # [2, B, seq, n, d] 只读 view
new_k = torch.cat([hist[0], roped_key], 1)   # 仅当前步的临时拼接(880 token)
new_v = torch.cat([hist[1], v], 1)
new_k = new_k[:, -self.max_attention_size:]
new_v = new_v[:, -self.max_attention_size:]
...
x = self.attn(q_cat, k_cat, v_cat)
if update_kv_cache:                          # prefill 才走
    kv.append(layer_idx, roped_key, v)       # 原地写新 token；溢出走 4.2-B memmove
# 不再 stack / 不再返回 updated_kv_cache 给上层 clone
```

- denoise 循环（`update_kv_cache=False`）行为不变：只读 view + 临时 cat，
  临时张量仍每步分配（小，880+4620 token），后续 CUDA Graph 工作流再静态化。
- `pipeline_dreamzero.py` 的 `predict_noise`（`:306-341`）删除
  `updated_kv_caches` 回写循环与 `update_kv_cache()` 的 `.clone()` 路径，
  prefill 结束后调用一次 `kv.commit(s_new)`。
- `_prefill_kv_cache`（`:479-600`）中 `create_kv_caches` 删除——
  `DreamZeroState` 由 `create_session_state(sid, slab_pool)` 构造时持有
  pool 引用，真正的 slab 在首次 append 时取（4.2 的 alloc-on-first-write）。

**`state_dreamzero.py`**（实现 `SessionState` protocol）：

- `kv_cache/kv_cache_neg: list[Tensor]` → `kv_pos/kv_neg: PreallocKVCache | None`；
- 新增 `estimated_bytes()`（slab×分支数 + crossattn/ys/clip 估算）与
  `release()`（slab 归还 pool、引用置 None）；
- `reset()` 复用 `release()` 的归还路径后重新初始化（state 可能被复用）；
- `get_kv_caches()/update_kv_cache()` 保留薄兼容层（旧测试与
  `enabled=false` 回滚路径使用）。

### 4.6 控制面：host-side cleanup 三条路径

释放请求与推理请求走**同一个 `generate()` 通道**，作为带
`extra_args={"control": "release_session", "session_id": sid}` 的特殊请求，
由**通用层 runner 在 `pipeline.forward()` 之前拦截**（代码见 4.5），
完全不进模型代码——这是 SGLang 把 `_close_session` 放在 scheduler 而非
模型层的同款决策。理由：天然与该 session 在途的 step 串行（diffusion
engine 按请求顺序执行），无需新增 RPC 面，也自动广播到所有 TP/CFG rank
（每个 rank 的 runner 都会执行，满足 4.4 的确定性要求），且对任何
实现了 protocol 的 pipeline 自动生效。

三个触发点：

1. **`endpoint=reset`**（`connection.py:123-126`）：
   `serving.reset(obs)` 从空操作改为向引擎提交 release 控制请求
   （需要把 `reset` 改 async 或 fire-and-forget 提交）。
2. **WebSocket 断连 / idle 超时**（`connection.py:96-105, 153`）：
   connection 记录本连接触摸过的 session_id 集合，在 `finally` 中逐个
   提交 release。注意：若部署上允许 session 跨连接迁移（断线重连），
   此路径应配置开关 `release_on_disconnect`（默认 true；重连场景反正
   也能从首帧重建，只损失一次 prefill 历史）。
3. **TTL**（4.4）：兜底，覆盖客户端异常退出且连接未感知的情况。

### 4.7 可观测性

- `manager.stats()` 暴露：`num_sessions`、`bytes_used/budget`、
  `pool_free_slabs`、`evictions{explicit,ttl,lru}`、`slab_reuse_hits`、
  `admission_rejects`；
- 每 N 步（如 100）`logger.info` 一行摘要；淘汰/拒绝事件逐条 `info/warning`；
- 预留接入 `/metrics`（Prometheus）的字段命名，本期不接。

---

## 5. 与 RFC 其他工作流的交互

| 工作流 | 交互 |
| --- | --- |
| torch.compile / CUDA Graph | 本方案后，prefill 与 denoise 读到的 cache 地址在 session 内固定（slab 不迁移）；denoise 的 K/V 拼接形状仍随 `seq_len` 变化，graph 化需配合"窗满前按 seq 桶 / 窗满后单一形状"的捕获策略——窗满后（稳态，最重要的工况）形状恒定，本方案直接使其可捕获 |
| 量化 | KV slab dtype 由配置决定，预留 fp8 KV 的 itemsize 计算路径（账本按 dtype 算），本期不实现 |
| Context parallel | head 维已按 TP 切（`tp_num_heads`），slab 形状随并行布局自动缩小；序列维切分（USP）若引入需重审 slab 布局，账本接口不变 |
| Host-side cleanup | 即本方案 4.6 |
| 共享 benchmark | 见 §7 |
| Page-attention & KV cache management（BDE RFC，#1987 条目 F） | 本方案 = 其 workstream B / WP-7 的设计输入；存储层最终由 BDE paged 栈接管，融合分析与迁移时序见 §10 |

---

## 6. 实施计划（4 个 Phase，均独立可合入、可回滚）

### Phase 1 — 通用 session 层 + DreamZero 生命周期接入（低风险，先行合入）
- 新增 `diffusion/cache/session_state_cache.py`：`SessionCacheConfig`、
  `SessionState`/`SessionEntry`、`SessionStateManager`、`TensorSlabPool`
  （此阶段 DreamZeroState 内部仍是现有 lazy cache，manager 只管映射、
  tick、TTL、按 `estimated_bytes()` 的 LRU 淘汰与硬上限）；
- `models/interface.py` 加 `SupportsSessionState`；runner `execute_model`
  接入 acquire/控制拦截；
- `pipeline_dreamzero.py` 实现 `create_session_state`、删除自管
  `_states`/`MAX_DREAMZERO_SESSIONS`；
- 4.6 控制面三条路径打通；
- stats/日志。
- **风险**：低。不碰计算路径，淘汰策略错误最多导致多一次首帧重建。

### Phase 2 — KV 预分配 + 原地滑窗（核心性能项，DreamZero 私有层）
- `dreamzero/session_kv.py` 的 `PreallocKVCache` + attention/pipeline
  改造（4.2/4.5），slab 经由通用 `TensorSlabPool`（4.3）；
- `session_cache.enabled/preallocate` 开关，默认 **off** 合入，
  数值 parity 通过后翻默认；
- 溢出 memmove 用方案 B（前向分块），保留方案 A 注释备选。
- **风险**：中。靠逐 bit parity 测试兜底（见 §7）。

### Phase 3 — 默认开启 + 调优
- headroom 摊薄 memmove 频率的实验；cross-attn cache 并入 slab；
- 默认 `enabled: true, preallocate: true`，更新 deploy YAML 与文档。

### Phase 4（可选）— CPU offload 二级缓存
- 淘汰时 slab D2H 到 pinned host 池、重激活时 H2D 恢复（可复用本仓库
  已有的 KV 异步 H2D 专用 stream 基建，见 `hy-img-kvconnector-opt` 分支
  Phase 2/3 工作）；适合"机器人暂停后恢复"场景。默认关闭。

---

## 7. 测试与验收

### 单元测试
**通用层**（`tests/diffusion/test_session_state_cache.py`，新增，CPU 可跑）：
- manager：LRU 序 / tick 单调 / TTL（逻辑时钟版）回收 / 显式 release /
  准入 evict_lru 与 reject 两策略 / pool 复用计数 / 预算账本守恒
  （state 用假实现，不依赖 DreamZero）；
- CFG-parallel 模拟：两个 manager 喂同一请求流，断言淘汰决策序列相同。

**DreamZero 私有层**（`tests/dreamzero/test_session_kv.py`，新增）：
- `PreallocKVCache`：append→view 与基线 `cat` 路径**逐 bit 一致**
  （含未满窗、恰好满窗 4620、溢出 memmove 三个相位）；
  方案 B memmove 与 `cat[-W:]` 等价性的随机化测试。

### 既有测试回归
- `tests/dreamzero/test_pipeline_state.py`（state 行为）；
- `tests/dreamzero/upstream/test_openpi_e2e_source_parity.py`（数值对源仓库 parity，
  Phase 2 的硬验收线）；
- `tests/e2e/online_serving/test_dreamzero_expansion.py`（serving 链路）。

### 新增 e2e
- 多 session 交错（A/B/A/B…）：互不污染、各自滑窗正确；
- 断连释放：连接关闭后 `stats().num_sessions` 归零、slab 回 pool；
- 预算压满：第 N+1 个 session 触发 LRU 淘汰，被淘汰 session 再来时从
  首帧重建且输出合法。

### Benchmark（接 RFC 共享 benchmark）
| 指标 | 基线 | 预期 |
| --- | --- | --- |
| 单步延迟（稳态满窗，TP1/TP4 × CFG1/CFG2） | 现状 | prefill 段下降（D2D 流量 ~21→~6 GiB/步 + 零分配）；端到端收益取决于 prefill 占比，benchmark 给实数 |
| 每步 `cudaMalloc` 次数（torch profiler） | ~240 | 0（稳态） |
| 显存峰值 vs session 数曲线 | 无界 | 阶梯状、封顶于 budget |
| 8-session 交错吞吐 | 易 OOM | 稳定运行 |

---

## 8. 风险与开放问题

| 风险 / 问题 | 应对 |
| --- | --- |
| memmove（方案 B）与原 `cat[-W:]` 在数值上必须严格等价 | 纯拷贝无算术，逐 bit 测试覆盖三个相位 |
| `_states` 中 `"default"` session 的特殊性（`pipeline.__init__` 即创建，`self.state` 兜底引用） | manager 化后 `self.state` 仅作 `predict_noise` 缺省参数的兼容兜底，标注 deprecated，所有调用点显式传 state |
| 多 rank 淘汰一致性被未来改动破坏 | §4.4 的设计裁定写入代码注释 + CFG 模拟单测锁住 |
| `release_on_disconnect` 与断线重连体验冲突 | 配置开关，默认 true，文档说明重连代价 = 一次首帧 prefill |
| AR 路径（vLLM core paged KV）做不做 session 层 | 本期不做：那是 SGLang 模式的真正对应物，但要 patch vLLM 核心类（`patch.py` 高危区），且 DreamZero 不走该路径。protocol 命名保持模型无关，将来如需可加 AR 适配器 |
| `SessionState` protocol 过度泛化 / 第二个接入模型出现前接口跑偏 | protocol 只保留三个方法（estimated_bytes/release/reset），有第二个模型接入时再扩展；不预设布局相关接口 |
| runner 拦截控制请求改变了 `execute_model` 的"必进 pipeline"假设 | 控制请求返回合法 `DiffusionOutput`，对 engine/scheduler 透明；e2e 用例覆盖 |
| batch>1 的未来场景 | `TensorSlabPool` 已按 shape+dtype 键化（4.3），无单一形状假设 |
| denoise 临时 cat 仍有每步小分配 | 量级小（<3% 流量），归入 CUDA Graph 工作流统一静态化，本期不做 |
| `should_reset` 的 `local_attn_size` 自动 reset 与 TTL/LRU 叠加 | reset 复用 release 的归还路径，slab 回 pool 后立即重取（同形状必命中），无额外开销 |

---

## 9. 工作量估算

| 项 | 估算 |
| --- | --- |
| Phase 1（通用层 manager/pool/protocol + DreamZero 接入 + 控制面 + 测试） | ~4 人日 |
| Phase 2（prealloc + attention 改造 + parity 测试） | ~5 人日 |
| Phase 3（调优 + 翻默认 + 文档） | ~2 人日 |
| Phase 4（CPU offload，可选） | ~4 人日 |

---

## 10. 与《统一 KV Cache 管理》RFC（BDE）的融合

> BDE = World Model RFC #1987 的条目 F（"Page-attention and KV cache
> management for Autoregressive Diffusion"）的设计方案：复用 vLLM 主线
> `KVCacheManager` / `BlockPool` / `SlidingWindowManager` / `BlockTables` /
> paged attention，diffusion 侧只补 `BDERequestAdapter`、
> `ChunkWindowSpec/Manager`、scheduler admission deltas、runner
> slot_mapping、blockwise-causal paged backend 五个兼容层。

### 10.1 关系判定：互补，且接缝是 BDE 自己声明的

两个方案切的是同一问题的不同层，几乎没有重叠的"竞争面"：

- **BDE 解决"KV 字节放哪、怎么驱逐、怎么准入"**——对应本方案的
  §4.2（PreallocKVCache）、§4.3（TensorSlabPool）和 §4.4 中按字节预算
  的准入部分。它的 BlockPool/refcount/prefix-index 比我们的 slab pool
  能力强一个量级（跨请求 prefix 复用、preemption、全局预算）。
- **本方案解决"session 是什么、活多久、谁来杀"**——BDE 把这块**显式
  留白**：WP-7（"Multiturn session KV lifetime, Open Q6"）依赖
  workstream B，要求 B"暴露 committed-chunk count（→
  `num_computed_tokens`）并定义 session 范围的 free()/reset/preempt
  语义"；其 FlashDreams 调研结论（"实时 world-model KV 生命周期默认
  应当是 session-scoped，而不是 per-turn"）正是本方案的出发点。
- 结论：**本方案上移为 BDE 的 session 层（workstream B / WP-7 的设计），
  存储层让位给 vLLM 栈**。SGLang 对照（§3.1）得出的"所有权槽位 +
  显式释放 + 持有量自检"模式不变，只是槽位里装的东西从 slab 换成
  adapter + block ids。

### 10.2 组件映射：融合后各归何处

| 本方案组件 | 融合后 | 说明 |
| --- | --- | --- |
| `SessionStateManager`（runner 级，每 rank 一份） | **上移到 scheduler/engine 侧单点**，`SessionEntry` 持有长生命周期的 `BDERequestAdapter` 而非 slab | session 的 KV 所有权用 adapter 的 `request_id` 在 BlockPool 中记账；acquire = 复用既有 adapter，release = `kv_cache_manager.free(adapter)` |
| `TensorSlabPool` + bytes 账本 + 准入 | **删除**，由 `BlockPool` + `gpu_memory_fraction` + `allocate_slots()` 返回 `None` 的背压取代 | 预算从"每 rank 自算"变成全局共享、scheduler 单点判定 |
| `PreallocKVCache` 原地滑窗 + 方案 B memmove | **被逻辑驱逐取代**：`ChunkWindowManager` 改 block-table 可见性，物理零搬移 | 即 BDE 引用 StreamDiffusionV2 的论点："window eviction 是 block 上的逻辑可见性，不是 dense tensor roll"——比我们的 memmove 更优 |
| 逻辑时钟跨 rank 确定性（§4.4 设计裁定） | **大幅简化**：分配/驱逐决策集中在 scheduler 单点，worker 只接收 block ids | 本方案最大的复杂度来源直接消失；CFG-parallel 两分支用同一 block 几何、rank-local 内容天然一致。session 级 LRU/TTL 策略仍用请求序号，但只在单点执行 |
| `SessionState` protocol | **保留但瘦身**：只管非 paged-KV 的 session 残余——crossattn cache、`clip_feas`/`ys`、帧缓冲、`language` | 这些不是 self-attn KV，不进 BlockPool；worker 侧仍需按 sid 的 state map，但只作为 scheduler 决策的确定性跟随者 |
| 控制面三条路径（§4.6） | **触发源不变**（OpenPI reset / 断连 / TTL），落点从 runner 拦截改为 scheduler 的 session free 入口 | BDE 的 `finish_requests → free()` 给了正规出口；v1 仍可走 generate 通道过渡 |
| 逐 bit parity 测试（§7） | **改造成 WP-6 的 parity gate oracle** | 三相位（未满窗/恰满/溢出）测试矩阵直接复用 |
| CUDA Graph 地址稳定收益 | 部分保留：paged pool 地址稳定，但 block table/slot_mapping 每 chunk 重建 | graph 化策略移交 BDE 的 paged backend 讨论 |

### 10.3 本方案补齐 BDE 的缺口（作为 workstream B 的交付）

1. **WP-1 的数据来源**：`num_computed_tokens` = SessionEntry 的
   committed-chunk 计数——本方案 §1.1 的生命周期走读已给出精确映射
   （first-frame prefill = 220 token 特殊 prefix；常规 chunk = 880；
   action/state register 不入持久 KV），与 BDE 术语节一致。
2. **WP-7 整体**：session→adapter 的 lifetime 绑定、TTL/LRU、
   `release_on_disconnect` 开关、三条清理路径、stats 字段
   （`num_sessions/bytes_used/evictions{explicit,ttl,lru}`）——直接移植。
3. **准入失败时的 session 级受害者选择**：vLLM scheduler 只会 defer/preempt
   *请求*，不会替你挑一个 *idle session* 释放——`allocate_slots` 返回
   `None` 时需要本方案的 `evict_for_admission`（按 last_used 挑非活跃
   session 调 `free()`）作为钩子挂进 admission 路径，否则 idle session
   会永久占住 block 直到饿死新 session。这是融合后本方案仍然独有的逻辑。

### 10.4 DreamZero 迁移的语义差异（喂给 WP-6 的具体问题）

**(a) 窗口粒度：`ChunkWindowSpec` 的断言对 DreamZero 不成立。**
实测：`4620 = 21×220 = 5.25 × 880`，不是 chunk（880）的整数倍——
`sliding_window == window_chunks × chunk_size` 断言直接失败。
且现状 `[-4620:]` 是 token 粒度滑窗，会"半驱逐" chunk（满窗后第一次
驱逐丢掉 first frame 220 + chunk1 的前 660），chunk 对齐的驱逐
（保 5 chunk=4400，或 sink 保 first frame）都与现状**不逐 bit 等价**。
解法（已验证数学）：所有驱逐量都恰好对齐**帧边界**（220 的倍数），
因此取 `block_size = frame_seqlen = 220`（一帧一 block），窗口按 token
声明为 4620、驱逐 snap 到 block 而非 chunk，即与现状逐 token 等价。
→ **对 BDE 的修改建议**：`ChunkWindowSpec` 放宽为 token 粒度
`sliding_window` + block 粒度 snap；`sink_chunks` 改为按 token/block 计
（first-frame prefix 是 220，不是 880，二者大小不一）。

**(b) block_size 整除性**：`gcd(220, 880, 4620) = 220`，可选 block_size ∈
{4,5,10,11,20,22,44,55,110,220}；常见 paged kernel 的 16/32 整除 880 但
不整除 220。后端选型需支持任意 page size（如 FlashInfer 风格）或上述
因子之一。

**(c) CFG 两分支**：每 session 两条独立 KV 序列。CFG-parallel 下两分支
chunk 进度恒同 → 单点分配一套 block ids、两 rank 各存本分支内容即可；
单 rank CFG 则需要两个 adapter（`request_id = sid#pos / sid#neg`）。
本方案 §4.2 的 alloc-on-first-write 对应"neg adapter 仅在实际写入时创建"。

**(d) denoise 不落盘语义**：DreamZero 的 denoise 步**从不 commit** KV
（预测不污染历史），commit 点是**下一个请求**的 observation prefill。
与 BDE 的 T>1 规则（同 chunk 重复 step 覆盖同一组 slot、commit 时才
推进 `num_computed_tokens`）兼容，但要求 in-flight chunk 的 slot 跨
*请求*保留——session-scoped adapter 恰好覆盖；纯 per-request adapter
做不到。这再次说明 session 层是前置依赖而非附属。

**(e) 残余 state**：crossattn cache（512 文本 token，session 内只写一次）、
`clip_feas`/`ys`、帧缓冲不是 self-attn KV，v1 不进 BlockPool，留在瘦身后
的 `SessionState`（10.2）。

### 10.5 融合后的实施时序（修订 §6 的解读）

```
时间 ──────────────────────────────────────────────────────▶
本方案 Phase 1 (session 层+控制面)  ████████
  = BDE workstream B / WP-7，不被 BDE 阻塞，先行落地（用现 lazy cache）
本方案 Phase 2 (slab 原地滑窗)              ▒▒▒▒▒▒  ← 条件启动（见下）
BDE Phase 0-1 (scaffolding+adapter+paged backend)   ████████████
BDE Phase 2 (chunk window + admission)                    ████████
WP-6 DreamZero 迁移 + parity/质量 gate                          ████
  → gate 通过后：删除 slab 路径(若启动过) + SessionState 瘦身
```

- **Phase 1 无条件先做**：它是两个方案共同的前置（BDE 的 WP-1 都要从它
  拿 committed-chunk 计数），且收益独立（OOM 治理、泄漏修复）。落地时
  `SessionState.release()` 的实现先是"slab 归还/引用置空"，BDE 就绪后
  换成 `free(adapter)`——protocol 接口不变。
- **Phase 2（slab）改为条件启动**：若 BDE WP-5（paged backend，跨
  attention/worker/scheduler 三个 owner area 的长杆）能赶上 DreamZero P0
  的延迟节点，直接跳过 slab 走 WP-6；若赶不上，slab 作为过渡上线，
  **sunset 条件 = WP-6 parity gate 通过即删**。slab 的逐 bit parity 测试
  无论哪条路都要写（它就是 WP-6 的 oracle），所以测试投入不浪费。
- 本方案 Phase 4（CPU offload）取消，并入 BDE Phase 4（vLLM KV
  connector 路径）。

### 10.6 对 BDE RFC 的反馈清单（反馈周期内提出）

1. `ChunkWindowSpec`：窗口改 token 粒度 + block 粒度 snap（10.4-a 的
   断言失败 + 逐 token 等价方案，附数学验证）；sink 按 token/block 计。
2. WP-7 / workstream B：引用本方案 §4.4/§4.6/§4.7 作为 session 层设计
   输入；特别是 10.3-3 的"准入失败时 session 级受害者选择"钩子，
   建议进入 BDE 的 scheduler admission 设计而非留作开放问题。
3. CFG 分支的 adapter keying 约定（`sid#branch`）与 CFG-parallel 下
   "同 ids、rank-local 内容"的分配语义，建议写进组件 3。
4. DreamZero 的 `block_size` 约束（10.4-b）作为 paged backend 选型输入。
5. T>1 语义补充：commit 点可能在**下一个请求**（world-model 的
   observation prefill），明确 in-flight chunk slot 的跨请求保留依赖
   session-scoped adapter。
