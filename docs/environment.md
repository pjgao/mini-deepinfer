# uv 环境与依赖管理

项目使用 `pyproject.toml` 声明依赖，`uv.lock` 固定解析结果，
`.python-version` 默认选择 Python 3.12。安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后：

```bash
uv sync --locked
uv run mini-deepinfer --device cpu --max-tokens 8
uv run pytest -q
uv run ruff check .
uv build
```

`uv sync` 会创建 `.venv`、以 editable 方式安装本项目，并默认安装 `dev` 依赖组。
`--locked` 会检查锁文件是否与项目声明一致，发现不一致时失败。
`uv run` 使用项目环境，必要时自动同步；不需要 `activate`。
sdist 和 wheel 输出到 `dist/`，不提交环境或构建产物。

## Python 与 PyTorch

代码要求 Python 3.10+；默认 3.12。临时选择版本可以运行
`uv sync --locked --python 3.10`，后续命令也使用 `uv run --python 3.10 ...`；
若要持久修改默认版本，运行 `uv python pin 3.10`。

CPU 教学环境在 Linux/Windows 使用专属 PyTorch CPU 索引，macOS 使用 PyPI。
CPU 索引只用于 `torch`，其他依赖从 PyPI 解析，避免为 CPU 示例下载 CUDA 运行库。
锁文件可以包含不同 Python/平台对应的不同版本，不代表所有组合都已实测；
当前验证范围见 [validation.md](validation.md)。

## 更新依赖

```bash
uv add <package>
uv add --dev <development-tool>
uv lock --upgrade-package torch
uv sync --locked
uv run pytest -q
```

依赖变更时一起提交 `pyproject.toml` 和 `uv.lock`；按需提交 `.python-version`。
仅运行示例时可用 `uv sync --locked --no-dev`，随后使用
`uv run --no-dev mini-deepinfer --max-tokens 8`，避免自动补装开发工具。

## 已有 CUDA / Ascend 环境

设备环境由适合硬件的 PyTorch/CANN/torch_npu 配套决定，默认 CPU 锁文件不管理这些配套。
先准备独立且已验证的设备环境，再仅安装本项目：

```bash
uv pip install --python /path/to/device-env/bin/python -e . --no-deps
/path/to/device-env/bin/python examples/validate_device.py --device cuda:0
# Ascend 需已有匹配的 torch_npu / CANN，并先确认获准的空闲卡：
ASCEND_RT_VISIBLE_DEVICES=0 /path/to/device-env/bin/python examples/validate_device.py --device npu:0
```

将解释器路径替换为实际值。直接使用该解释器运行示例，避免 `uv sync` 或普通 `uv run`
将设备环境改回 CPU 配置。`--no-deps` 不会检查运行依赖是否完整，需先确认环境中存在
符合本项目版本要求的 `torch`，NPU 还需可导入配套 `torch_npu`。
详见 [昇腾说明](ascend.md)。

uv 配置参考：[PyTorch 集成](https://docs.astral.sh/uv/guides/integration/pytorch/)、
[GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)。
