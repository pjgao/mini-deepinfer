# 与 DeepInfer 的对应关系

参考版本固定为 [`d585e6542c0571176272d703f03c4807ae901284`](https://github.com/deepseek-ai/DeepInfer/tree/d585e6542c0571176272d703f03c4807ae901284)。
下表是阅读路线，不表示二进制、权重格式、数值语义或 API 完全兼容。
本项目代码独立编写；没有复制上游执行器或厂商算子。

| 本项目 | DeepInfer 参考路径 | 保留的思想 | 简化/缺失 |
|---|---|---|---|
| `LLM` / `engine.py` | [`entrypoints/llm.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/entrypoints/llm.py)、[`engine/llm_engine.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/engine/llm_engine.py) | 请求、调度、执行、输出闭环 | 无多进程、在线服务、VE/PD 分离 |
| `step()` / token budget | [`csrc/engine/scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/csrc/engine/scheduler.hpp)、`zos.cpp` | 调度与执行输入对应 | Python 同步实现，未实现 zero-overhead scheduler |
| `PagePool` / `PagedKV` | [`worker/cache_engine.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/worker/cache_engine.py)、`csrc/engine/block_manager.hpp` | 页表、KV 生命周期 | float32、最坏容量预留、无外部缓存系统 |
| `LatentAttention` | [`models/v4_modules/attn.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/model_executor/models/v4_modules/attn.py) | 降低缓存维度、独立位置处理 | 自定义小型结构，不是 V4.1 精确重现 |
| 简化 indexer | [`models/v4_modules/index_attn.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/model_executor/models/v4_modules/index_attn.py) | 根据索引分数选择历史范围 | 无多头权重、候选块、压缩和跨层索引共享 |
| `MoE` | [`models/v4_modules/mlp.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/model_executor/models/v4_modules/mlp.py) | 路由专家、共享专家、dispatch/combine | 简化 gate；无 EP、EPLB 与 MegaMoE |
| `allocate_drafts()` | [`csrc/engine/mtp_varlen.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/csrc/engine/mtp_varlen.hpp) | 按置信度分配验证预算 | n-gram 频率代理，不是学习置信度头或官方调度策略 |
| `verify_greedy()` | `layers/sampler.py`、`worker/model_runner.py` | 验证草稿、接受前缀、回退状态 | 仅 greedy；没有保持随机分布的 speculative sampling |
| `backend.py` | [`platform.py`](https://github.com/deepseek-ai/DeepInfer/blob/d585e6542c0571176272d703f03c4807ae901284/deepinfer/platform.py) | 将设备差异集中管理 | 不使用 transfer_to_npu、HCCL、厂商融合算子 |

## 从其他教育项目借鉴什么

- nano-vllm：让入口、sequence、scheduler、block manager 的责任清楚；读者能够沿一次请求读完整个引擎。
- mini-sglang：用独立模块展示服务机制，并通过小例子观察调度和缓存效果。

两者都有自己的模型覆盖、kernel 与性能目标。本项目选择 CPU 可运行的张量参考实现，
不会沿用它们的性能数据，也不声称功能等价。固定参考 revision 见根目录 `provenance.json`。

## 不应误读的名称

- “latent KV”指本项目的低秩缓存机制，不意味着支持所有 MLA 模型。
- “稀疏 attention”描述选择语义，目前仍构造 dense score 矩阵。
- “连续批处理”表示每步可加入/退出请求，不意味着调度与设备异步重叠。
- “置信度预算”指简化的 prefix 概率分配，不是 DeepInfer 的 DSpark 训练方案。
- “NPU backend”指 torch_npu eager 路径，不意味着使用 Ascend 950 专用 DeepGEMM/FlashMLA。
