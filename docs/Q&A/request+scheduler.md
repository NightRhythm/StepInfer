# 请求与调度器

## 总体

### 1.请求与调度器和 Continuous Batching 有什么关系？

在传统的静态批处理中，所有请求同生同死打包成一个张量，**根本不需要 Request 和 Scheduler**。调度器诞生的唯一目的，就是将调度颗粒度下沉到**“单步迭代（Iteration-level）”**，实现请求的随时加入与提前退出。调度器就是 Continuous Batching 的执行大脑，Request 就是其调度的原子数据载体。

---

## 请求（Request）

### 1.一条纯粹的 `Request` 应该包含哪些核心字段？

1. `request_id`: 唯一标识；

2. `prompt` / `prompt_token_ids`: 原始文本与分词后的输入；

3. `status`: 请求状态机（`WAITING` / `RUNNING` / `FINISHED`）；

4. `max_new_tokens`: 生成上限阈值；

5. `arrival_time`: 到达时间戳（用于 FIFO 排队和延迟统计）。

### 2.模型每步生成的 Token 存在哪里？为什么不与 Prompt 混存？

单独保存在 `generated_token_ids: list[int]` 中；核心价值在于：

1. 便于向调用者增量/流式返回新生成内容；

2. 区分阶段: 若 `len(generated_token_ids) == 0` 则为 **Prefill**，若 `> 0` 则为 **Decode**。

### 3.判断请求完成 (FINISHED)的依据是什么？

**命中 EOS Token** 或 **达到 `max_new_tokens` 上限**。需记录 `stop_reason`（如 `"stop"` 或 `"length"`），供上层应用识别是正常回答完毕还是被截断。

---

## 批处理演进

### 1.现有的静态批处理中，为什么某个请求提前结束，不能直接把 Batch 从 8 变成 7，而必须塞 PAD 陪跑？

Attention 矩阵乘法完全支持 8 变 7，但现有 `LayerKVCache` 是**连续大张量**，且仅有**一个全局标量指针 `self.current_len`**。

若强行缩减 Batch，后续请求写入 Cache 时会错位覆盖已结束请求的槽位（**槽位踩踏/串台**）。必须依靠调度器维护“请求-槽位映射”才能安全解耦。

### 2.Continuous Batching 出来之前可以实现提前返回吗？

**可以！通过“缩 Batch 策略”完全可以做到。**

1. **提前返回与缩 Batch**: 在每个 Decode 迭代步中，一旦某个请求命中 EOS 或达到最大长度，即可立即将其结果返回给用户，并将其移出当前计算批次。此时 Batch 维度直接缩减（如 $8 \to 7 \to 6$），剩下的请求继续解码，完全不需要补 PAD 陪跑。

2. **该策略的局限（只出不进）**: 这种弱化版调度只能单向缩小批次，**无法优雅解决“新请求半途插队加入”的问题**。因为新请求是 Prefill 阶段（带着长 Prompt，维度为非 1），而正在运行的是 Decode（维度固定为 1）。若想通过补 PAD 对齐让新请求半途插入，老请求的计算量会暴增几十上百倍。

3. **Continuous Batching 的真正突破**: 不仅支持已完成请求的提前退场，更核心的是打破了 Prefill 与 Decode 的边界，实现了**“随时退场 + 随时插入新请求”的双向动态批处理**。

### 3.纯 PyTorch 下，当前维度很固定（Prefill 为 `[B, N, T, H]`，Decode 为 `[B, N, 1, H]`），提前结束只是 B 减小，新请求半路加入改变那个 1 怎么解决？

采用**分阶段解耦调度**：

1. 新请求不强行与老请求拼进同一步前向，调度器专门给新请求单独执行一次 Prefill 前向；

2. 跑完首步后，新请求也蜕变为“只需 1 个 Token”的状态，下一步全员即可整齐划一以 `[B+1, N, 1, H]` 继续解码。

---

## 请求队列（RequestQueue）

### 1.队列用“一锅端”还是“分状态管理”？

按状态分离，避免每步全局 $O(N)$ 遍历过滤：

1. `waiting`: 使用 `collections.deque`，满足严格的 FIFO 先进先出；

2. `running`: 使用 `list` 或 `dict`，便于在任意请求结束时快速查找并移出。

### 2.当请求队列满载时，应该挂起等待还是直接报错？

**直接抛异常快速失败 (Backpressure / 429 机制)**。

避免在引擎外部无休止挂起导致客户端 HTTP 连接超时雪崩。

---

## 调度器（Scheduler）

### 1.调度器每一步做决策时，放多少个 WAITING 进 RUNNING 依据什么？

1. **并发容量配额**: $\text{available\_slots} = \text{max\_batch\_size} - \text{len(running)}$；

2. **上下文窗口卡控**: 检查 Prompt 长度是否超出全局 `max_seq_len`。

### 2.调度的产物 ScheduleResult 装什么给执行引擎？

1. 直接装载被选中的请求实体 `requests: list[Request]`；

2. 附带阶段标记 `is_prefill: bool`（指示当前步是跑 Prefill 还是 Decode）。

### 3.单步执行后，如何“打扫战场”？已完成的请求怎么处理？

1. 遍历将采样的 `next_tokens` 追加到各个 Request 的 `generated_token_ids`；

2. 调度器检查请求的终止条件（EOS / max_tokens）；

3. 命中完成态的请求，调度器**当场从 `running` 集合移出，直接返回给调用方，引擎内部不作常驻留存**，杜绝内存泄漏。
