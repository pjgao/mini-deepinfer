# 验证记录

验证日期：2026-10-21。报告仅覆盖下面列出的环境和功能。

| 项目 | 环境 / 方法 | 结果 |
|---|---|---|
| CPU 单元/集成测试 | WSL2 Ubuntu，Python 3.12.3，PyTorch 2.14.1+cpu，1 CPU thread | 54 tests passed |
| 格式/静态检查 | Ruff lint（含 UP、B、SIM）与 format check | passed |
| 设备验证脚本 | `validate_device.py --device cpu` | 全序列、分页/分块、多请求生成、前缀复用通过 |
| 小模型训练 | seed=42，160 steps，仓库内合成语料，dense causal attention | loss 5.7958 → 0.1091 |
| checkpoint 重新加载 | 本项目 `config.json + model.pt` | 可生成训练语句并正常 EOS |
| 包构建与独立安装 | sdist + wheel；替换 editable 安装，在仓库外执行 CLI | passed |
| 推测解码 | greedy、n-gram 草稿；另有测试强制不同拒绝位置 | 与目标模型输出一致；报告提出与接受数量 |
| CUDA | 提供 eager 入口和验证脚本 | 尚未进行设备实测 |
| Ascend 950 | 提供 torch_npu eager 入口和验证脚本 | 尚未进行设备实测 |
| 官方 V4.1 权重、厂商 kernel、分布式 EP | 不在首版范围 | 不支持 |

这里的“CPU 设备验证”是在相同 CPU 上对照不同执行路径，不是跨硬件对照。
CUDA/NPU 支持状态必须在真实设备运行后更新。

测试还覆盖同分值下的 Top-K 采样、草稿或采样异常后的缓存释放、批量提交失败时的请求取消、
模型返回数量不匹配、空批次处理，以及配置中的非整数计数。模型计算重构后，
CPU 对照脚本的最大 logits 误差为 0；160 步训练仍得到下述损失和续写结果。

训练后输入 `A KV cache`，得到补全文本 ` stores previous keys and values.`，随后 EOS。
这句话出现在训练语料中，只证明小模型训练/保存/加载/生成链路工作，
不属于泛化能力或正式模型精度结果。

## 执行命令（uv 项目）

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python examples/validate_device.py --device cpu --output runs/cpu-validation.json
uv run python examples/train_tiny.py --steps 160 --output checkpoints/tiny
uv run mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --max-tokens 40 --draft-tokens 3
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --output runs/baseline.json
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --draft-tokens 3 --output runs/spec.json
uv build
```

项目使用 `uv.lock` 固定依赖，默认 Python 由 `.python-version` 选择。
构建统一使用 `uv build`，无需额外安装 Python build 包或系统 ensurepip。

一次 CPU benchmark 冒烟运行中，有/无草稿均实际发出 256 token。
草稿配置提出 390 个 token、接受 40 个；这些是计数验证。
单轮耗时只有约 0.16–0.17 秒，**不足以建立性能收益结论**，因此不作为加速比宣传。
正式性能实验需要更长测量窗口、多轮交替运行与环境记录。

GitHub Actions 在 Python 3.10 / 3.12 上执行 `uv sync --locked` 和锁定环境检查；
Python 3.10 另行覆盖 PyTorch 2.5.1 的兼容性测试，不改动锁文件。
请以 [当前 workflow 结果](https://github.com/pjgao/mini-deepinfer/actions/workflows/ci.yml) 为准；
本地通过不意味着远程 CI 已通过。

没有在本仓库公开内部机器地址、共享设备清单或私有业务数据。
