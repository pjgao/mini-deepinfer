# 昇腾执行与验证

## 两个不同的目标

**运行本项目：** 普通 PyTorch eager 算子，通过 `torch_npu` 在指定 NPU 执行。
不依赖 FlashMLA、DeepGEMM-Ascend、DeepEP-Ascend 或 TileLang。
各代设备仍需实测；提供 `npu:0` 参数不等于已验证。

**移植上游高性能路径：** DeepInfer 的固定参考版本声明 Ascend 950 + CANN 9.2，
还需要对应驱动、固件、低精度格式和通信环境。这不属于当前 mini 实现的支持范围。
不要把 CPU eager 的绿灯当作 950 厂商 kernel 的验证。

## 运行前确认

1. 确认真实设备型号、驱动、CANN、PyTorch 和 torch_npu 版本。
2. 确认设备分配与现有进程；使用获准的空闲设备。
3. 在单独目录/环境安装，保留原来的 torch/torch_npu 配对。
4. 如果 SSH 别名或设备型号与预期不同，先解决环境身份问题。

已有可用 NPU 环境内：

```bash
uv pip install --python /path/to/npu-env/bin/python -e . --no-deps
ASCEND_RT_VISIBLE_DEVICES=0 /path/to/npu-env/bin/python examples/validate_device.py \
  --device npu:0 --output runs/npu-validation.json
```

替换解释器路径为现有 NPU 环境的实际路径。这里使用 uv 的 pip 接口，只安装项目本身；
不要在厂商环境中执行 `uv sync` 或普通 `uv run`，它们会同步默认 CPU 锁文件。
可见设备列表应由当前机器分配决定。`npu:0` 是可见列表里的逻辑编号。
`validate_device.py` 只分配一个很小的 float32 模型和页缓存，不下载权重、不启动服务、
不修改驱动，也不清理其他进程。失败时会保留异常并返回非零状态，不会静默回退 CPU。

## 验证信号

脚本使用同一份权重，依次比较：

- CPU 与设备的无缓存 logits；
- 设备上分页/分块 logits 与 CPU reference；
- 多请求 greedy 输出；
- 同样请求第二轮的前缀缓存复用。

logits 容差是 `atol=rtol=1e-3`，greedy token 要求完全相同。
如果 logits 很接近但 argmax 改变，需要检查数值 margin；脚本不会因此自动放宽验收。
这个固定小样本通过也只证明脚本覆盖的 float32 eager 路径。

进一步的 NPU 验证应包含更长输入、页边界、不同 batch、不同精度、EOS、草稿拒绝和容量压力。
在新增 NPU kernel 之前，应让同样的 reference 测试仍然可运行。

## 可能需要适配的 eager 算子

- 稳定 argsort：用来保证相同 indexer 分数的 tie-break 与 chunk 长度无关。
- 页表高级索引与 `index_add`：部分 shape 下可能产生额外转换或同步。
- `torch.where` 专家分发：有动态 shape 和 Python 循环，不适合直接当成融合/图执行方案。
- CPU sampler：随机采样刻意使用每请求 CPU generator，会同步 logits。

这些是待设备验证的兼容性与性能点，不是已经复现的 NPU 故障。

## 若继续接入厂商算子

建议保持 `model.forward()` 为数学参考，先替换一个算子，再逐项验证：

1. tensor shape、stride、dtype 与量化 scale 的编码；
2. cache 写入、读取、页索引与拒绝后的覆盖；
3. stream 生命周期、异步依赖和图捕获限制；
4. 完整输出与质量，最后才测性能。

尤其不要仅凭函数名相同，直接混用不同平台的 FP8/FP4 scale packing。
当前 mini 仅使用浮点缓存，尚未定义量化缓存协议。
