# 从一次请求读完整条执行链

这份导读针对本项目的小模型。下面的公式和状态不能当成完整 DeepSeek-V4.1 的结构说明。

## 1. 先看没有 cache 的模型

入口是 `MiniDeepSeek.forward(tokens)`。输入 `[T]`，embedding 得到 `[T,D]`。
每层执行 pre-norm attention、残差连接、pre-norm MoE、残差连接，最后归一化并投影到词表。

默认配置：`D=64`、`H=4`、每头内容维 `d=16`、RoPE 维 `r=8`、
KV latent 维 `c=16`、indexer 维 `i=16`，两层、四个路由专家、每 token 选两个。

在 attention 中：

```text
x[T,D] ── q_proj ──> q[T,H,d+r]
       ├─ kv_down + norm ──> latent[T,c]
       ├─ k_rope ──> rotary_key[T,r]
       └─ index_k ──> index_key[T,i]

latent ── kv_up ──> K_content[T,H,d], V[T,H,d]
K = concat(K_content, broadcast(rotary_key))
Q 的 RoPE 部分与 rotary_key 使用绝对 token 位置旋转。
```

Cache 只保存 `latent + rotary_key + index_key`，每 token 每层为 `c+r+i=40` 个浮点值。
作为对照，存储所有头的 K_content/V 加共享 rotary/index key 为
`2*H*d+r+i=152` 个浮点值。这是该教学配置的缓存元素比较，
不是完整模型的显存节省比例；权重、激活、页空洞和临时展开 K/V 仍占内存。

为了直观展示数学，读取时完整展开 K/V。生产实现可以改变权重与计算布局来避免这项开销；
本项目没有做这种优化。

## 2. 稀疏索引先选择，再计算 attention

对于 query 位置 `p`：

1. 因果范围为 `key_position <= p`。
2. 最近 `window` 个 token 必须可见。
3. 对更早的历史计算 `relu(index_q @ index_k.T)`，选择至多 `sparse_topk` 个。
4. 用历史选择与窗口的并集作为 attention mask。

未来 token 必须在选择前排除。否则未来分数可能挤掉历史候选，即使最后把未来位置 mask 掉，
也已经改变了历史候选集合。测试 `test_future_tokens_cannot_change_prefix_logits` 专门检查此问题。

另一个不明显的问题是分数相同：ReLU 会产生大量 0。这里采用稳定排序，优先较早的位置，
使分块长度不同也不会改变 tie-break。普通 `topk` 不承诺稳定的相同值索引。

`sparse_topk=0` 表示完整 causal attention。
当前实现仍计算完整 score 矩阵，因此复杂度仍含 `O(T*S)`；
这是稀疏语义参考，不能声称减少了实际 attention FLOPs。

## 3. MoE 的 dispatch 和 combine

Gate 输出 `[T,E]`，选择每 token 的 `K` 个专家；仅在被选专家之间归一化路由权重。
所有 token 通过共享专家，再将各路由专家的结果乘权重后 `index_add` 回原 token 行。

`MoE.last_load` 记录本次每个路由专家接收的 token 数，用于观察负载分布。
它不是负载均衡算法。这里的专家都在同一设备，没有跨卡通信。

## 4. 从逻辑位置到物理页

假设 `block_size=4`、请求的页表为 `[7,1,5]`：

| 逻辑 token 位置 | 逻辑页 | 页内 offset | 物理页 |
|---|---:|---:|---:|
| 0..3 | 0 | 0..3 | 7 |
| 4..7 | 1 | 0..3 | 1 |
| 8..11 | 2 | 0..3 | 5 |

写地址是 `cache[layer, pages[position // B], position % B]`。
缓存本体形状为 `[layers, physical_pages, block_size, cache_width]`。
测试刻意使用不连续物理页，防止把逻辑页号误当成物理页号。

`PagePool` 管引用与容量，`PagedKV` 管实际张量，两者职责分开。
完整 prompt 页可共享；还会继续写的尾页保持私有，因此当前实现无需 copy-on-write。
前缀 key 包含该页之前的**所有 token**，相同局部 token 块不一定具有相同的 KV。
使用完整 token tuple 是为了消除哈希碰撞问题，长上下文会增加 CPU 内存成本。

一个 prefix entry 持有一份引用，每个请求再持有自己的引用。
请求结束只释放自己的引用。缺少空闲页时，先回收只有 prefix cache 持有的旧页。
`close()` 清空 prefix cache，最终所有页都应回到 free list。

首版在准入时预留 `prompt + max_tokens` 的容量。这样等待请求不会拿着半份资源阻塞，
但会降低实际可同时服务的请求数。它不模拟生产引擎的动态增长和抢占。

## 5. 调度与 token 状态

每个请求维护：

- `tokens`：prompt 与已经发给用户的输出；
- `computed`：已有完整各层 KV 的 token 数；
- `lease.pages`：预留的物理页表。

Prefill 过程中，`computed` 逐 chunk 增加，到 prompt 末尾才从最后一个 logits 采样。
刚采样出来的 token 尚未执行模型，所以 decode 开始时始终有：

```text
len(tokens) = computed + 1
```

一个普通 decode step 计算这个 pending token，得到新的 token；该关系保持成立。
EOS 或 `max_tokens` 结束后释放 lease。最后一个输出 token 不必再计算 KV。

每次 `step()` 先准入等待请求，给已有 decode 请求基础预算，再分配草稿预算，
剩下的预算给 chunked prefill。decode 优先可能延迟 prefill；
有限输出长度保证最终释放，但这不是生产公平性或延迟 SLA 策略。

投影和专家计算在合并的 token 张量上执行；attention 切片回每个请求。
`examples/trace_scheduler.py` 打印每步实际调度计划，并在运行中插入新请求。
这与先跑完第一个请求再开始第二个请求不同。

## 6. 草稿验证与回退

当前 drafter 用历史重复 n-gram 提出后续 token，并用观察到的后继频率估计置信度。
预算分配使用累计置信度，不能分配第 3 个草稿位置却跳过第 2 个。
没有训练草稿模型；它只是学习“提议—验证—提交”协议的便宜方式。

假设 pending token 为 `a`，草稿是 `[b,c,d]`：

```text
target 本次输入： [a, b, c, d]
target 预测：     [b, x, ..., ...]
```

第一个草稿 `b` 被接受，`c` 被拒绝，发出 `[b,x]`。
KV 仅提交 `a,b` 两个输入位置，因此 `computed += 2`。
`x` 是新的 pending token。旧的 `c,d` 数据无需物理清零：
下一轮将从 `x` 的逻辑位置覆盖，并且所有读取受 `computed + 本轮输入长度` 限制。

若全部接受，则发出三个草稿与一个 bonus token，提交四个输入位置。
输出剩余长度会限制草稿数量；EOS 可以截断整个提交结果。

测试中的 `OracleDraft` 仅用于强制覆盖全部接受、首个拒绝和中间拒绝路径，
**不用于 benchmark**。真实 demo 使用 `NGramDraft`。

本项目仅支持 greedy speculation。对 temperature 采样，直接比较 draft 和 target token
不能保证目标分布；需要接受/拒绝采样与 residual distribution，故当前直接拒绝该配置。

## 7. 推荐练习

1. 运行 `trace_scheduler.py`，把 `token_budget` 从 6 调到 3（同时设置 `max_seqs<=3`），观察 prefill 分块。
2. 用相同 prompt 连续生成两次，观察 `prefix_hit_tokens`，再关闭 `prefix_cache`。
3. 对比 `sparse_topk=0` 和小 Top-K，保持各自的全序列与分页结果一致；二者输出本来就不必相同。
4. 用 `MoE.last_load` 计算负载差异；不要把 token 平衡直接当作运行时间平衡。
5. 在测试中改变 rejection 位置，检查 `computed` 和下一步输出。
6. 训练小 checkpoint 后交替运行有/无 speculation benchmark，记录负收益和接受率。

更进一步：实现动态页增长与安全抢占、为 sampler 加 top-p、写真正只计算选中 key 的稀疏 reference。
每次改动先保持对应的数值/状态测试，再考虑 kernel 或图执行。
