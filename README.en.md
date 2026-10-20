# mini-deepinfer

[中文](README.md) · [Walkthrough (Chinese)](docs/walkthrough.md) · [Upstream map](docs/upstream-map.md)

A small, executable **DeepInfer-inspired teaching engine**, organized in the spirit
of nano-vllm and mini-sglang. Core computation is ordinary PyTorch, not a wrapper
around the production engine.

**Scope:** the default model is tiny and randomly initialized. It is not useful for
chat. You can train and load this project's own checkpoints. Official DeepSeek-V4.1
checkpoints are **not supported**. This is an independent project, not affiliated
with DeepSeek. Check [validation status](docs/validation.md) before assuming a
backend has been tested.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```bash
git clone https://github.com/pjgao/mini-deepinfer.git
cd mini-deepinfer
uv sync --locked
uv run mini-deepinfer --device cpu --max-tokens 8
uv run python examples/trace_scheduler.py
uv run pytest -q
```

Python 3.10+; `.python-version` selects 3.12 by default. No manual environment
activation is needed on Linux, Windows or WSL2. The lock uses CPU PyTorch on
Linux/Windows and PyPI PyTorch on macOS. For an existing CUDA/NPU environment,
install with `uv pip install --python /path/to/env/bin/python -e . --no-deps`
and invoke that interpreter directly; do not sync the CPU lock into it.
The NPU path requires a matching `torch_npu`/CANN stack and never silently falls
back to CPU. See [environment management](docs/environment.md).

```python
from mini_deepinfer import LLM, SamplingParams

with LLM(device="cpu") as engine:
    outputs = engine.generate(
        ["What is a KV cache?", "What is an expert?"],
        SamplingParams(max_tokens=16, temperature=0),
    )
    print(outputs[0].token_ids)
```

## What is implemented

- Low-rank latent KV, decoupled RoPE and an explicit expanded-KV attention reference.
- A simplified sparse indexer selecting old tokens, unioned with a sliding window.
- Real top-k routed experts and a shared expert, packed across requests on one device.
- Paged KV allocation, reference counts and immutable full-page prefix sharing.
- Chunked prefill, token budgets, continuous batching, cancellation and EOS.
- N-gram drafting, confidence-budgeted verification and exact greedy rollback.
- Per-request seeded sampling, tiny-model training and safe state-dict loading.

This does **not** reproduce full V4.1, learned DSpark heads, indexer training,
YOCO bounded replay, quantized KV, distributed EP/TP/CP, EPLB, MegaMoE, async
overlap scheduling, graph capture, structured output or an online API server.
Attention gathers pages in Python; sparsity is a semantic reference, not an
efficient sparse kernel.

## Train a tiny checkpoint

```bash
uv run python examples/train_tiny.py --steps 160 --output checkpoints/tiny
uv run mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --max-tokens 40
uv run mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --draft-tokens 3
```

The synthetic-corpus exercise defaults to dense causal attention: a hard top-k
indexer does not learn without a separate training objective. Falling training
loss is not a held-out model-quality result.

## Measure and verify

```bash
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --output runs/baseline.json
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --draft-tokens 3 --output runs/spec.json
uv run python examples/validate_device.py --device cpu
uv run ruff check .
uv build
```

Throughput counts **emitted tokens**, including prefill and host overhead, with
device synchronization. Proposed/accepted draft counts are reported separately.
No speedup is promised. Compare identical workloads and alternate repeated runs.

The tests check full-vs-paged logits, chunk boundaries, causality, request
isolation, prefix sharing, memory pressure, cancellation, sampling and speculative
accept/reject paths. CPU CI does not establish CUDA/NPU correctness.

## Read the engine

`model.py` → `cache.py` → `engine.py` → `speculative.py` → `tests/`.
See the [walkthrough](docs/walkthrough.md), [Ascend notes](docs/ascend.md) and
[source mapping](docs/upstream-map.md). Exact reference revisions are recorded in
[provenance.json](provenance.json).

Apache-2.0. Independently written teaching implementation; references and
acknowledgments are in [NOTICE](NOTICE).
