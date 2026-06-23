# KV Transfer Manager: Prefetch/Sync Path Refactoring

## 目标

将 `receive_multi_kv_cache_distributed` 中的 prefetch 逻辑与 sync 逻辑分离，
拆成 **fetch → distribute → apply** 三步，两条路径复用 distribute 和 apply。

同时简化 `_prefetch_payload`：利用 RDMA pool buffer 已是 pinned memory 的特性，
在 `receive_kv_cache_for_request` 内直接完成 H2D，去掉 `_h2d_in_background` 和 `_to_owned_pinned`。

## 当前问题

`receive_multi_kv_cache_distributed` 混合了三种职责：

1. 获取 KV payload（sync receive 或 prefetch consume）
2. 跨 rank 分发（broadcast / CFG send / SP broadcast）
3. 应用到 req（attach past_key_values）

`prefetched` 和 `prefetch_failed` 参数把 prefetch 逻辑耦合进了 sync 路径，
导致每个 branch 都有 3-way 分支（prefetch_failed / prefetched / sync），代码重复且难以维护。

此外，当前 Branch 3/4 OWNER prefetch 路径调用 `_record_stream_for_prefetched` + `apply_kv_cache_to_request`，
跳过 H2D fallback（Branch 1/2 用 `apply_prefetched_kv`，包含 H2D fallback），行为不一致。
如果后台 H2D 失败，CPU tensor 进入 collective broadcast，所有 rank 拿到 CPU tensor。

## 新设计

### 三步拆分

| 步骤 | 职责 | 同步路径 | Prefetch 路径 |
|------|------|----------|---------------|
| fetch | 获取 KV payload，标记 received | `receive_multi_kv_cache` | `consume_prefetched_kv`（miss → `receive_multi_kv_cache`） |
| distribute | OWNER 广播给 FOLLOWER | `distribute_kv_cache` | `distribute_kv_cache` |
| apply | 挂到 req 上 | `apply_kv_cache_to_request` | `apply_kv_cache_to_request` |

### 关键约束：OWNER 必须先 apply 再 distribute

`distribute_kv_cache` 内部 OWNER 调用 `_collect_request_kv_payload(req)` /
`_build_cfg_rank_local_payloads(req, ...)` 时，是从 `req` 上读取 `past_key_values`。
如果 OWNER 没先 apply，collect 读不到数据，广播就是空的。

因此 OWNER 和 FOLLOWER 的 apply 时机不同：

```
OWNER:     fetch → apply → distribute
FOLLOWER:                      distribute → apply
```

`distribute_kv_cache` 返回值反映这一差异：
- OWNER：apply 已在 distribute 前完成，返回 `None`（调用方无需再 apply）
- FOLLOWER：返回 `dict`，调用方负责 apply
- RANK_LOCAL / pure TP：不需要 distribute，返回 `None`，调用方自己 apply

### 两条路径

**同步路径**（入口：`receive_multi_kv_cache_distributed`）：

```
OWNER:     receive_multi_kv_cache  →  distribute_kv_cache
                  (fetch+apply)          (distribute, 返回 None)
FOLLOWER:  （不 fetch）  ─────────→  distribute_kv_cache  →  apply
                                       (distribute, 返回 dict)  (apply)
RANK_LOCAL: receive_multi_kv_cache
                  (fetch+apply, 无需 distribute)
```

> **apply 时机说明**：`receive_multi_kv_cache` → `receive_kv_cache` 内部已调用
> `apply_kv_cache_to_request`，所以 **fetch 这一步就把 KV 挂到了 req 上**（OWNER / RANK_LOCAL 都如此，
> 故标注 `fetch+apply`，**不存在第二次 apply**）。OWNER 的 `distribute_kv_cache` 从 `req.past_key_values`
> 读数据广播、返回 `None`，入口的 `if kv_payload is not None` 为假 → 不再额外 apply。
> 只有 FOLLOWER 自己不 fetch，由 `distribute_kv_cache` 返回的 payload 在外部 apply 一次。
> （对比 prefetch 路径的 `consume → apply`：`consume_prefetched_kv` **不** apply，故那里 apply 是独立一步。）
>
> **两个 apply 不是同一个函数，按 payload 形态分工，不可互换**：
> - `apply_kv_cache_to_request(req, data)` —— OWNER/RANK_LOCAL 的 fetch 内部用。输入是**连接器原始格式**
>   （顶层 `"layer_blocks"` 键），只设 primary `past_key_values` + `metadata`，**不搬设备**（数据已在 device）。
> - `_apply_request_kv_payload(req, kv_payload, target_device)` —— FOLLOWER 外部用。输入是
>   `_collect_request_kv_payload` 的**广播格式**（`past_key_values` 为现成 SimpleNamespace + `sp.*` 前缀键，
>   含 CFG companion KV），**会 `_move_to_device`** 并还原全部 `sp.*`。
>
> 形态转换链：原始 `{layer_blocks}` --apply_kv_cache_to_request--> `req.past_key_values`
> --_collect_request_kv_payload--> 广播格式 `{past_key_values, sp.*}` --broadcast--> FOLLOWER
> --_apply_request_kv_payload--> req。喂错格式会找不到对应键（`layer_blocks` vs `past_key_values`）。

**Prefetch 路径**（入口：`consume_and_distribute_kv_cache`）：

```
OWNER (hit):   consume → apply → distribute
                            (apply)  (返回 None)
OWNER (miss):  consume → receive_multi_kv_cache → apply → distribute
                          (sync fetch)            (apply)  (返回 None)
OWNER (error): consume → KVPrefetchConsumeError → received=False → distribute
                          (不回退 sync)                          (广播空信号, 返回 None)
FOLLOWER:                                            distribute → apply
                                                      (返回 dict)  (apply)
RANK_LOCAL (hit):   consume → apply
RANK_LOCAL (miss):  consume → receive_multi_kv_cache → apply
```

### Prefetch 路径不再需要 prepare 步骤

旧设计中 prefetch 路径多了一步 `prepare_prefetched_kv`（H2D fallback + record_stream），
原因是 `_prefetch_payload` 里 `receive_kv_cache_for_request(target_device=None, pin=True)` 只做 pinned copy-out，
H2D 由单独的 `_h2d_in_background` 完成，可能失败导致 CPU tensor 留在 payload 中。

新设计改为 `receive_kv_cache_for_request(target_device=device, pin=False)`，在 `receive_kv_cache_for_request`
内部直接完成 H2D，数据消费时已在 GPU 上。因此：

1. **H2D fallback 不再需要**：数据从 `consume_prefetched_kv` 返回时已在 target device 上
2. **record_stream 仍需要**：GPU tensor 由 `_bg_copy_stream`（后台线程）放置，
   主线程 collective 使用前需 `record_stream(current_stream)` 防止 allocator 回收
3. **bg `_bg_copy_stream.synchronize()` 不可省**：`record_stream` 只防 allocator 复用显存，
   **不建立跨流执行顺序**。`_bg_copy_stream` 上的 H2D 与主线程 default stream 之间没有 event/wait，
   若 bg 不 synchronize 就把 GPU tensor 交给 future，主线程 forward 可能在 H2D 完成前读到 KV → 数值错乱。
   RDMA pinned 源的 H2D 是真异步，最容易踩中；SHM pageable 的 `.to` 恰好同步、侥幸安全。
   因此 bg 线程返回前必须 `self._bg_copy_stream.synchronize()`，让交付的 GPU tensor 是 CPU-observed 完成态
   （等价于原 `_h2d_in_background` 末尾那行 `stream.synchronize()`）。

但 `record_stream` 逻辑可以内联到 `consume_and_distribute_kv_cache` 中，
不再需要独立的 `prepare_prefetched_kv` 方法。

### _prefetch_payload 简化

#### 关键发现：RDMA pool buffer 已是 pinned，SHM 不是

RDMA connector（Mooncake / Mori）创建 pool 时：

```python
pool = torch.empty(pool_size, dtype=torch.uint8).pin_memory()  # cudaMallocHost
```

`ManagedBuffer.tensor = pool[offset:offset+size]` 是 pool 的 slice view，**也是 pinned 的**。
`receive_kv_cache_for_request` 内部 `from_bytes(memoryview(buf_tensor.numpy()))` 通过
`torch.frombuffer` 创建的 tensor，**PyTorch 能正确追踪其 pinned 状态**（实测 `is_pinned()` 返回 True）。

因此 RDMA 路径上 `pin=True` + `_to_owned_pinned` 的 copy-out 步骤是多余的：
- 旧路径：view → `_to_owned_pinned`（pinned copy，冗余）→ `_h2d_in_background`（async H2D）
- 新路径：`target_device=device` → `receive_kv_cache_for_request` 内部 H2D 直接从 pinned 池视图发起

**SHM connector 例外**：`SharedMemoryConnector` 从 `multiprocessing.shared_memory`
（`/dev/shm`，**可分页内存**）反序列化，既不是 `ManagedBuffer`、也不是 pinned 池切片，
`is_pinned()` 为 False。对它 `.to(device)` 退化为**同步 pageable 拷贝**——不走侧流、持 driver 锁更久、
给 forward 带来抖动，且冷启动那一次裸露。

#### 方案 A：`is_pinned()` 门控，统一走 async 侧流

H2D 真异步（`non_blocking=True` 在侧流上与 forward 重叠）的前提是**源为 pinned**。
所以按源是否 pinned 分两路，两条 connector 都拿到侧流重叠：

- **RDMA（源已 pinned）**：直接 `src.to(device, non_blocking=True)`，零额外拷贝。
- **SHM（源 pageable）**：先 `src = tensor.pin_memory()` 暂存到 pinned，再 `src.to(device, non_blocking=True)`。
  暂存是一次 CPU→CPU 拷 + `cudaHostAlloc`，藏在 bg 线程的 forward(N) 窗口内（实测 `cudaHostAlloc`
  抖动明显再升级为复用 pinned staging buffer）。

> **生命周期红线**：`non_blocking=True` 的 H2D 源（SHM 的 pinned 暂存 / RDMA 的池视图）
> **必须存活到 stream synchronize 之后**。而 RDMA 池 buffer 在 `receive_kv_cache_for_request`
> 的 `finally` 里 `release`，所以 **synchronize 必须放在 `receive_kv_cache_for_request` 内、release 之前**，
> 不能推迟到 `_prefetch_payload`（那时池已归还、SHM 暂存也已出作用域 → use-after-free）。

#### 新 _prefetch_payload

> **设备无关红线（NPU/GPU 双路径）**：预取的 stream / device / current_stream 一律走
> `current_omni_platform`（与同栈 `diffusion/offloader/layerwise_backend.py` 同款），
> **不得写死 `torch.cuda.*`**。`torch.cuda.is_available()` 在 NPU 上返回 False、`torch.cuda.Stream()`
> 在 NPU 上不存在——写死会让 NPU 静默跳过后台 H2D（不崩，但丢掉"掩盖 H2D"）。
> `pin_memory()` / `record_stream()` / `.to(non_blocking=True)` 是设备无关张量方法，torch_npu 支持，无需改。

```python
def _prefetch_payload(self, request_id, role, sender_info, target_device):
    """Background-thread body: receive + async H2D on bg_copy_stream."""
    # bg 线程不继承主线程的当前卡（默认 0 号），TP 下每 rank 一张卡，必须显式设。
    if target_device is not None and target_device.type != "cpu":
        torch.accelerator.set_device_index(target_device.index)
    with current_omni_platform.device(target_device):
        if self._bg_copy_stream is None:
            self._bg_copy_stream = current_omni_platform.Stream()
        with current_omni_platform.stream(self._bg_copy_stream):
            # H2D（is_pinned() 门控 + non_blocking）与 synchronize 都在
            # receive_kv_cache_for_request 内完成：源（pinned 暂存 / 池视图）
            # 必须在 release 前已 sync，故不能把 sync 推迟到这里。
            data, size = self.receive_kv_cache_for_request(
                request_id,
                target_device=target_device,
                sender_info=sender_info,
                pin=False,
            )
    if data is None:
        return None, 0
    return data, size
```

- `pin=False`：不需要 pinned copy-out；RDMA 池本身 pinned，SHM 由内部 `is_pinned()` 门控按需 pin
- `target_device=target_device`：`receive_kv_cache_for_request` 内部完成 H2D（方案 A）+ synchronize
- `current_omni_platform.Stream() / .stream(...)`：H2D 在专用 stream 上执行，
  不阻塞主线程 default stream，主线程计算可与 H2D overlap；**设备无关**，CUDA/NPU/XPU 通用
- `set_device_index` + `current_omni_platform.device(...)`：bg 线程显式落到 `target_device` 那张卡
- synchronize **在 `receive_kv_cache_for_request` 内、pool release 之前**完成（见下方 §2）：
  既保证交付的 GPU tensor 是 CPU-observed 完成态（`record_stream` 只防 allocator 复用、不建立跨流顺序），
  又保证 `non_blocking` H2D 的源存活到拷贝结束

#### 可删除的代码

| 方法 / 参数 | 原因 |
|-------------|------|
| `_h2d_in_background` | H2D 由 `receive_kv_cache_for_request` 内部 `.to(device)` 完成 |
| `_to_owned_pinned` | pool buffer 已 pinned，不需要 copy-out 到新 pinned tensor |
| `receive_kv_cache_for_request` 的 `pin` 参数 | 不再需要 pinned copy-out 路径 |
| `prepare_prefetched_kv` 的 H2D fallback 分支 | 数据消费时已在 GPU 上 |

## 修改清单

### `kv_transfer_manager.py`

#### 0. 跨切面：设备无关（NPU / GPU 双路径）

预取路径所有设备 API 一律走 `current_omni_platform`（顶部 `from vllm_omni.platforms import current_omni_platform`），
**禁止写死 `torch.cuda.*`**。NPU 上 `torch.cuda.is_available()==False`、无 `torch.cuda.Stream`，
写死会让 NPU 静默跳过后台 H2D（不崩，但"掩盖 H2D"失效）。逐处映射：

| 写死（现状） | 设备无关 |
|---|---|
| `torch.cuda.Stream()` | `current_omni_platform.Stream()` |
| `torch.cuda.stream(s)` | `current_omni_platform.stream(s)` |
| `torch.cuda.current_stream()` | `current_omni_platform.current_stream()` |
| `torch.cuda.is_available()` | `current_omni_platform.get_device_count() >= 1` / 按 `device.type != "cpu"` |
| `torch.cuda.mem_get_info(device)` | `current_omni_platform.get_free_memory(device)` |
| `torch.cuda.device(device)` | `current_omni_platform.device(device)`（bg 线程另加 `torch.accelerator.set_device_index`） |
| 类型注解 `torch.cuda.Stream` | `current_omni_platform.Stream` |

涉及方法：`_prefetch_payload`（§1）、`receive_kv_cache_for_request` 的 H2D 段（§2）、
`_has_free_mem_for_prefetch`（`type != "cuda"` 判断 + `mem_get_info`）、保留的 `_record_stream_for_prefetched`
（`current_stream` + `is_available`）。`pin_memory()` / `record_stream()` / `.to(non_blocking=True)`
本就设备无关，不动。参照同栈 `diffusion/offloader/layerwise_backend.py`（已用同款抽象）。

#### 1. 简化 `_prefetch_payload`

见上方 "新 _prefetch_payload" 章节。

#### 2. 简化 `receive_kv_cache_for_request`

移除 `pin` 参数和 `_to_owned_pinned` 分支，新增 `async_h2d` 开关按路径分流——
**prefetch 路径（bg 线程）走方案 A（`is_pinned()` 门控 + non_blocking + sync）；同步路径走阻塞 `.to()`、不 sync**：

```python
# 签名加 async_h2d: bool = False；H2D 段改为：
if isinstance(data, dict) and "layer_blocks" in data:
    layer_blocks = data["layer_blocks"]
    cache_lists = [
        layer_blocks.get("key_cache", []),
        layer_blocks.get("value_cache", []),
    ]
    staging: list[torch.Tensor] = []   # 持 pinned 暂存活到 sync 之后（防 UAF）
    did_h2d = False
    for cache_list in cache_lists:
        for i, tensor in enumerate(cache_list):
            if not isinstance(tensor, torch.Tensor):
                continue
            if target_device is not None and tensor.device != target_device:
                if async_h2d:
                    # 方案 A：async 侧流 H2D 需 pinned 源。
                    # RDMA 池视图已 pinned -> 直接发；SHM pageable -> 先暂存到 pinned。
                    src = tensor if tensor.is_pinned() else tensor.pin_memory()
                    if src is not tensor:
                        staging.append(src)
                    cache_list[i] = src.to(target_device, non_blocking=True)
                    did_h2d = True
                else:
                    # 同步路径：阻塞拷贝，返回即完成、源可安全释放，无需 sync
                    cache_list[i] = tensor.to(target_device).contiguous()
            elif needs_buffer_detach:
                # 无 H2D（target_device=None）时 copy-out 以释放 pool buffer
                cache_list[i] = tensor.clone()
    if did_h2d:  # 仅 async 路径：sync 先于 finally 的 release / staging 出作用域
        current_omni_platform.current_stream().synchronize()
    del staging
```

- 删除 `_to_owned_pinned` 调用与 `pin` 参数；新增 `async_h2d` 形参
- **prefetch 路径**（`_prefetch_payload` 传 `async_h2d=True`，在 `_bg_copy_stream` 上）：`is_pinned()` 门控 +
  `non_blocking` + **sync 落在 bg 线程、先于 release/staging 释放**，杜绝 UAF + 保证消费前 CPU-observed 完成
- **同步路径**（主线程，`async_h2d=False`）：阻塞 `tensor.to(...)`，无 staging、无 sync——无重叠收益、阻塞即简单正确
- synchronize 不能挪到调用方：它必须先于 `receive_kv_cache_for_request` 内 `finally` 的 pool release

#### 3. 删除 `_h2d_in_background`

功能已由 `receive_kv_cache_for_request(target_device=device)` 内部 `.to(device)` 替代。

#### 4. 删除 `_to_owned_pinned`

不再需要 pinned copy-out。

#### 5. 新增 `distribute_kv_cache`（从 `receive_multi_kv_cache_distributed` 拆出）

```python
def distribute_kv_cache(self, req, target_device=None, *, received=False) -> dict | None:
```

- 只做广播/分发，**不做 apply**
- 返回 `kv_payload`（dict 或 None），调用方自己决定是否 apply
- RANK_LOCAL / pure TP：不需要分发，返回 `None`（调用方直接从 req 上取数据即可）
- OWNER：`received=True` 时 collect payload 并广播给 FOLLOWER；`received=False` 时广播空信号避免 FOLLOWER 死锁
- FOLLOWER：从 group 接收 payload
- 调用方拿到返回值后调用 `_apply_request_kv_payload`（注意：FOLLOWER 的 payload 是 broadcast 格式，必须用 `_apply_request_kv_payload` 而非 `apply_kv_cache_to_request`）

**返回值语义**：

| 场景 | 返回值 | 含义 |
|------|--------|------|
| RANK_LOCAL / pure TP | `None` | 无分发，调用方自己 apply |
| OWNER（收到数据） | `None` | OWNER 已通过 fetch+apply 将 payload 挂到 req 上，无需再次 apply |
| FOLLOWER（从 group 收到） | `dict` | FOLLOWER 收到的 payload，需 apply |
| 任何 rank 收到 None payload | `None` | 广播了空数据，表示失败 |

**完整实现**（从 `receive_multi_kv_cache_distributed` Branch 3/4 提取）：

```python
def distribute_kv_cache(self, req, target_device=None, *, received=False):
    """Broadcast/distribute KV from OWNER to FOLLOWER.
    Returns payload for FOLLOWER to apply, or None (caller doesn't need to apply).

    OWNER must apply KV before calling this method; `_collect_request_kv_payload`
    reads from `req.past_key_values`.
    """
    pt = self._topo_config

    # RANK_LOCAL / pure TP: no distribution needed
    if pt.is_rank_local or (pt.tp_active and not pt.has_cfg and not pt.has_sp):
        return None

    # ── Branch 3: TP + CFG/SP ──
    if pt.tp_active and (pt.has_cfg or pt.has_sp):
        kv_payload = None
        if pt.is_owner:
            if received:
                if pt.has_cfg:
                    cfg_rank_payloads = self._build_cfg_rank_local_payloads(req, pt.cfg_size)
                    kv_payload = cfg_rank_payloads[0]
                    for dst in range(1, pt.cfg_size):
                        try:
                            self._send_kv_payload(pt.cfg_group, cfg_rank_payloads[dst], dst, target_device)
                        except Exception:
                            continue
                elif pt.has_sp:
                    kv_payload = self._collect_request_kv_payload(req)
            else:
                # received=False: broadcast empty signal to avoid FOLLOWER deadlock
                if pt.has_cfg:
                    for dst in range(1, pt.cfg_size):
                        try:
                            self._send_kv_payload(pt.cfg_group, None, dst, target_device)
                        except Exception:
                            continue
        elif pt.sp_rank == 0 and pt.has_cfg:
            kv_payload = self._recv_kv_payload(pt.cfg_group, 0, target_device)

        if pt.has_sp and pt.sp_group is not None:
            kv_payload = self._broadcast_kv_payload(pt.sp_group, kv_payload, target_device, src=0)

        if not kv_payload:
            return None
        return kv_payload

    # ── Branch 4: TP inactive, world broadcast ──
    kv_payload = None
    if pt.is_owner:
        if received:
            kv_payload = self._collect_request_kv_payload(req)
    kv_payload = self._broadcast_kv_payload(pt.world, kv_payload, target_device, src=0)
    if not kv_payload:
        return None
    return kv_payload
```

#### 6. 简化 `receive_multi_kv_cache_distributed`

```python
def receive_multi_kv_cache_distributed(self, req, cfg_kv_collect_func=None, target_device=None) -> bool:
    """Sync receive + distribute + apply."""
    received = self.receive_multi_kv_cache(req, cfg_kv_collect_func, target_device)
    kv_payload = self.distribute_kv_cache(req, target_device, received=received)
    if kv_payload is not None:
        self._apply_request_kv_payload(req, kv_payload, target_device)
    return received or kv_payload is not None
```

- 无 prefetch 参数，纯粹的 sync 入口
- 调用点 2（`_update_states`）无需改动

#### 7. 新增 `consume_and_distribute_kv_cache`

```python
def consume_and_distribute_kv_cache(self, req, target_device=None) -> bool:
    """Consume prefetched KV → apply → distribute; falls back to sync on miss."""
    received = False
    if self._async_prefetch:
        try:
            data, _ = self.consume_prefetched_kv(req)
            if data is not None:
                self._record_stream_for_prefetched(data)
                self.apply_kv_cache_to_request(req, data)
                received = True
        except KVPrefetchConsumeError:
            logger.exception("KV prefetch consumed payload for %s but failed",
                             req.request_id if hasattr(req, 'request_id') else req)
            # payload 已消耗，不回退 sync（sync 也拿不到数据）
            # received 保持 False，distribute_kv_cache 会广播空信号给 FOLLOWER 避免死锁
    if not received:
        received = self.receive_multi_kv_cache(req, None, target_device)
    kv_payload = self.distribute_kv_cache(req, target_device, received=received)
    if kv_payload is not None:
        self._apply_request_kv_payload(req, kv_payload, target_device)
    return received or kv_payload is not None
```

- 调用点 1（`execute_model`）使用此入口
- **关键**：`KVPrefetchConsumeError` 后不回退 sync，`received=False` 让 `distribute_kv_cache` 广播空信号避免 FOLLOWER 死锁
- `_prefetch_payload` 已在后台 stream 完成 H2D，`consume_prefetched_kv` 返回的 tensor 已在 GPU 上
- 仍需 `_record_stream_for_prefetched`：GPU tensor 由 `_bg_copy_stream` 放置，
  主线程 default stream 使用前需 record_stream 防止 allocator 回收

#### 8. `consume_prefetched_kv` 移除 `target_device` 参数

当前 `consume_prefetched_kv(req_or_rid, target_device)`（1261 行）的 `target_device` 未使用，
只弹出 future 并调用 `fut.result()`。移除该参数。

#### 9. 删除 `apply_prefetched_kv`

旧设计中改为 deprecated wrapper，新设计中不再需要：
- H2D 已在 `_prefetch_payload` 完成，不需要 `prepare_prefetched_kv` 的 H2D fallback
- record_stream 由 `_record_stream_for_prefetched` 处理
- apply 由 `apply_kv_cache_to_request` 处理
- 当前调用点只有 Branch 1/2 的 prefetch 命中路径，重构后走 `consume_and_distribute_kv_cache`

#### 10. 新增 `recv_role` property

```python
@property
def recv_role(self) -> ReceiveRole:
    return self._topo_config.role
```

### `diffusion_model_runner.py`

#### 调用点 1：`execute_model`（~L293-314）

从 ~25 行缩减为：

```python
kv_recv_t0 = time.perf_counter()
self.kv_transfer_manager.consume_and_distribute_kv_cache(req, target_device=target_device)
kv_recv_ms = (time.perf_counter() - kv_recv_t0) * 1000
logger.debug("KV recv for %s %.1fms", req.request_id, kv_recv_ms)
```

#### 调用点 2：`_update_states`（~L422-426）

不变，继续使用 `receive_multi_kv_cache_distributed`。

### `test_kv_async_prefetch.py`

- `consume_prefetched_kv` 去掉 `target_device` 参数
- `mgr._recv_role()` → `mgr.recv_role`
- 删除 `apply_prefetched_kv` 相关测试，改为 `apply_kv_cache_to_request` + `_record_stream_for_prefetched`
- 对应断言更新

## 问题跟踪

详见 `prefetch_kv_issues.md` 和 `prefetch_kv_solutions.md`。

| # | 问题 | 状态 |
|---|------|------|
| 1 | KVPrefetchConsumeError 后回退 sync 矛盾 | 已解决：`consume_failed` 标记阻断回退 |
| 2 | distribute_kv_cache 返回 None 歧义 | 已解决：None 统一表示"调用方不需要额外 apply" |
| 3 | distribute_kv_cache 缺实现 | 已解决：完整实现见修改清单 |
| 4 | OWNER 路径跳过 H2D fallback | 已解决：H2D 在 `_prefetch_payload` 完成，消费时已在 GPU 上 |
| 5 | prepare_prefetched_kv 签名含糊 | 已关闭：不再需要 `prepare_prefetched_kv` |
| 6 | prefetch 路径 apply 时机 | 已验证，无需改动 |
| 7 | 测试 _recv_role() 不存在 | 已解决：新增 `recv_role` property |
| 8 | consume_prefetched_kv target_device 未使用 | 已解决：移除该参数 |
| 9 | _record_stream_for_prefetched 应删除 | 已关闭：仍需保留，用于 bg stream → default stream 的 record_stream |
| 10 | _h2d_in_background 和 _to_owned_pinned 冗余 | 已解决：pool buffer 已 pinned，`receive_kv_cache_for_request(target_device=device)` 内部完成 H2D |
| 11 | 新 _prefetch_payload 丢了 bg stream synchronize | 已解决：synchronize 放在 `receive_kv_cache_for_request` 内、pool release 之前（§2）。record_stream 不建立跨流顺序，缺它则 default stream 可能读到未拷完 KV（RDMA pinned 真异步最易踩）；放 release 前同时杜绝 non_blocking 源 use-after-free |
| 12 | SHM connector 源非 pinned，`.to` 退化同步、不走侧流 | 已解决（方案 A）：H2D 按 `is_pinned()` 门控——RDMA 直发 `non_blocking=True`，SHM 先 `pin_memory()` 暂存再 async；两条 connector 都拿到侧流重叠。后续若 `cudaHostAlloc` 抖动明显，升级为复用 pinned staging buffer（方案 B） |
| 13 | 预取路径写死 `torch.cuda.*`，NPU 静默跳过后台 H2D | 已解决（设计层）：所有 stream/device/mem API 走 `current_omni_platform`（§0 跨切面表），CUDA/NPU/XPU 通用；`pin_memory`/`record_stream`/`non_blocking` 设备无关不动。待代码落地 + NPU 实测 |

### 代码复审补充发现（2026 复审）

| # | 问题 | 状态 |
|---|------|------|
| 14 | **D2D CFG 点对点 send/recv 失败不对称 → 进程组 desync**：`_send_kv_payload` 真·NCCL 发送异常时 `combined.zero_()` 补发 0 sentinel 再 `raise`；分发调用点 owner 侧 `except Exception: continue` 吞掉、照常 return True，而 follower 侧 `_recv_kv_payload`→`_unpack_kv_payload` 对 0 blob 抛 `RuntimeError` 且**未被 try 包**→ 仅该 follower 崩、其余继续 → 下次集合通信 hang。（SP/world `broadcast` 路径对称、不受影响；owner 主动发 `None` 走 object 通道、安全） | 待定方向：① receiver 把 0/corrupt 当优雅 miss 返回 `None`（消除 hang，残留 owner 有/follower 无的不一致，推荐起步）；② 全 rank all-reduce success flag 一致 abort（最正确，较重）；③ 维持崩（最差） |
| 16 | `receive_kv_cache_for_request` 轮询 backoff 上限 0.5s → 冷启动/sync 路径最坏 +0.5s 尾延迟（bg 预取被 forward 遮住不受影响） | 待定：冷启动尾延迟敏感则下调上限（如 0.1s） |
| 19 | `_h2d_in_background` docstring "dedicated **CUDA** stream" 设备相关措辞 | 随 #13 一起中性化 |
| 20 | **重构后的同步入口无条件调 `receive_multi_kv_cache`，FOLLOWER 会误 fetch**：扁平化丢了当前代码"follower 从不 fetch"的 role 守卫 → follower 进 `connector.get()` 会偷走 owner payload（违反 consume-once）或卡到 `recv_timeout` 超时 | 待解决：fetch 前按 role 门控——`received = False; if not topo_config.is_follower: received = receive_multi_kv_cache(...)`（与 prefetch 路径 follower 守卫一致）。同步路径图的 OWNER 行已修正为 `fetch+apply`、FOLLOWER 标注「不 fetch」 |
| 22 | **CFG follower 收 None（`_build_cfg_rank_local_payloads` 补 None）是否影响 HunyuanImage-3 uncond** | **遗留（旧逻辑，暂不动）**。已查清：纯 cfg-parallel（tp=1）走 **Branch 4 world 全量广播**，cfg1 拿到完整 pos AR KV → **一直正确**。None 路径仅在 **Branch 3 = TP×CFG（tp_active ∧ has_cfg）+ plain CFG（branch_roles 空）** 触及，是**未测试死角**而非在跑 bug。代码层确认：该组合下 cfg_rank1 的 `ar_kv_data` 为空 → uncond 拿不到 pos 前缀。**仅当未来要支持 TP×CFG 才需修**（follower 填 `main_payload` 而非 None + TP×CFG parity）。重构 `distribute_kv_cache` 须**原样保留 Branch 3/4 语义**，勿改纯 cfg-parallel 的 Branch 4 全量广播 |

> 已落地并移除：#15（info→debug）、#17（删多余空行）、#18（`_prefetch_payload` 误导注释修正）、#21（`_to_dev` 改用 `_move_to_device`）。

### 实现状态（2026-06-19：全量落地，待 GPU+NPU 验证）

`prefetch_kv.md` 的重构 + 设备无关 + 复审修复**已全部写入代码**（`kv_transfer_manager.py` / `diffusion_model_runner.py` / `tests/distributed/omni_connectors/test_kv_async_prefetch.py`），py_compile 通过；GPU/NPU parity 由使用者验证。

- **Phase 0**：#14 — `_unpack_kv_payload` 损坏 blob → `warning + return None`（消除 desync hang）。
- **Phase 1**：方案 A H2D（`is_pinned()` 门控 + `pin_memory` 暂存 + `non_blocking` + staging 防 UAF + release 前 `synchronize`）；删 `_h2d_in_background` / `_to_owned_pinned`；`receive_kv_cache_for_request` 去 `pin` 参数（#1/#2/#3/#4/#10/#11/#12）。
- **Phase 2**：新增 `recv_role` property、`distribute_kv_cache`、`consume_and_distribute_kv_cache`；`receive_multi_kv_cache_distributed` 改纯 sync + **follower fetch 守卫(#20)**；删 `apply_prefetched_kv`；`consume_prefetched_kv` 去 `target_device`(#5/#6/#7/#8/#9)；runner 单入口；测试改 `recv_role` / 去 `pin` / 去 `apply_prefetched_kv`。
- **Phase 3（设备无关 #13/#19）**：H2D/stream/record_stream 全走 `current_omni_platform.*` + `torch.accelerator.set_device_index`；`_has_free_mem_for_prefetch` 在非 CUDA 上跳过限流（无 total 查询，加注说明）。

**与设计文档的两处有意偏差**（实现时为保旧行为/健壮性）：
1. `distribute_kv_cache` 对 **OWNER 也返回 payload**（非 doc 表里的"返回 None"）——旧代码 Branch 3 OWNER 末尾会 `_apply_request_kv_payload(cfg_rank_payloads[0])`（apply 自己的 branch-local payload，plain CFG 下幂等）。返回 None 会丢这次 apply，故保留返回、由入口统一 apply。
2. `_build_topo_config` 在 **world group 不可用时默认 RANK_LOCAL**（原为 raise）——让 `recv_role` 在未初始化拓扑下安全退化（自收、无集合通信），与 #7 的 `_recv_role` 安全语义一致。
