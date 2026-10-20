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

```bash
git clone https://github.com/pjgao/mini-deepinfer.git
cd mini-deepinfer
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
mini-deepinfer --device cpu --max-tokens 8
python examples/trace_scheduler.py
python -m pytest -q
```

Python 3.10+. On Windows, activate with `.venv\Scripts\Activate.ps1`; WSL2 is also
supported. Install the appropriate torch build for CUDA or NPU yourself. The NPU
path requires a matching `torch_npu`/CANN stack and never silently falls back to CPU.

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
python examples/train_tiny.py --steps 160 --output checkpoints/tiny
mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --max-tokens 40
mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --draft-tokens 3
```

The synthetic-corpus exercise defaults to dense causal attention: a hard top-k
indexer does not learn without a separate training objective. Falling training
loss is not a held-out model-quality result.

## Measure and verify

```bash
python examples/benchmark.py --checkpoint checkpoints/tiny --output runs/baseline.json
python examples/benchmark.py --checkpoint checkpoints/tiny --draft-tokens 3 --output runs/spec.json
python examples/validate_device.py --device cpu
ruff check .
python -m build
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
