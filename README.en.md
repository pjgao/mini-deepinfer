# mini-deepinfer

[![CPU correctness](https://github.com/pjgao/mini-deepinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/pjgao/mini-deepinfer/actions/workflows/ci.yml)
[中文](README.md) · [Walkthrough (Chinese)](docs/walkthrough.md) · [Design references](docs/upstream-map.md) · [Environment setup](docs/environment.md)

mini-deepinfer is a small language model inference engine written in PyTorch for
studying model computation, KV caching and request scheduling. It draws on
[DeepInfer](https://github.com/deepseek-ai/DeepInfer), with a compact code structure
inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and
[mini-sglang](https://github.com/sgl-project/mini-sglang).

The repository includes a model, an inference engine, training examples and tests.
It runs on a CPU without downloading model weights or compiling custom kernels.
It is intended for readers familiar with Python and basic PyTorch who want to
read or modify an inference engine. The default model has random weights; the
training example produces a small checkpoint of its own.

## Features

| Component | Implementation |
|---|---|
| Model | Compressed latent KV, separate RoPE keys, and causal sparse attention combining an indexer with a sliding window |
| MoE | Top-K routed experts and one shared expert per layer |
| KV cache | Paged storage, logical-to-physical page mapping, reference counting and reclamation |
| Prefix reuse | Exact token matching for complete pages, LRU eviction, and private partial tail pages |
| Scheduling | Continuous batching, chunked prefill, a per-step token budget, request insertion and cancellation |
| Speculative decoding | N-gram drafts, confidence-based budget allocation, target verification and rollback |
| Sampling | Greedy, temperature and Top-K sampling, with a separate random generator per request |
| Training | A synthetic text corpus, checkpoint saving and state-dict loading, CLI and Python interfaces |

Speculative decoding supports greedy sampling only. The engine runs synchronously
in one process on one device, using ordinary PyTorch operators.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
Python 3.10+ is supported; `.python-version` selects 3.12 by default.

```bash
git clone https://github.com/pjgao/mini-deepinfer.git
cd mini-deepinfer
uv sync --locked
uv run mini-deepinfer --device cpu --max-tokens 8
uv run python examples/trace_scheduler.py
uv run pytest -q
```

The CLI prints generated token IDs, text, finish reasons and engine statistics.
Meaningless text is expected from random weights. Use the training example below
to try text completion with a trained checkpoint.

`trace_scheduler.py` prints each scheduling step, including request IDs,
prefill/decode work, free pages and prefix hits. It shows how requests enter and
leave the batch and how cached pages are reused.

No manual environment activation is needed on Linux, Windows or WSL2. The lock
uses CPU PyTorch on Linux/Windows and PyPI PyTorch on macOS. See
[environment setup](docs/environment.md) for dependency updates and accelerator
environments.

## Python API

`LLM.generate()` accepts a list of prompts and sampling parameters and returns
the generated output for each request:

```python
from mini_deepinfer import LLM, SamplingParams

with LLM(device="cpu") as engine:
    outputs = engine.generate(
        ["What is a KV cache?", "What is an expert?"],
        SamplingParams(max_tokens=16, temperature=0),
    )
    print(outputs[0].token_ids)
```

For step-by-step scheduling, use `add_request()`, `step()` and `cancel()`.
The `token_budget` and `prefill_chunk` constructor arguments control the amount
of work in each step. `engine.stats` records prefill, decode and output token counts,
prefix hits, and proposed and accepted drafts.

## Model and execution

![mini-deepinfer architecture: two pre-norm layers with latent attention and MoE](docs/assets/model-architecture.png)

The default model has 2 layers, a hidden dimension of 64 and 4 attention heads.
Each layer has 4 routed experts, selects 2 per token, and includes one shared
expert. The tokenizer encodes UTF-8 bytes using a vocabulary of 256 byte values
plus BOS, EOS and PAD. The default sequence limit is 512 tokens, including the
prompt and generated output. These settings are defined in
[`ModelConfig`](mini_deepinfer/config.py).

For each token, each layer caches a 16-dimensional KV latent, an 8-dimensional
RoPE key and a 16-dimensional index key. Attention expands K/V when reading the
cache and takes the union of a single-head indexer's Top-K historical positions
and a recent-token window.

Before admitting a request, the engine reserves enough cache pages for its prompt
and maximum output length. Each step packs tokens from several requests for
projection and MoE computation; attention is computed separately for each request.
When a request reaches EOS or its length limit, the engine releases its page
references while retaining reusable complete prefix pages.

The [walkthrough](docs/walkthrough.md) covers scheduling states and cache behavior.
The [design references](docs/upstream-map.md) connect these modules to DeepInfer.

## Train a small model

The training script uses the repository's synthetic text corpus and demonstrates
forward and backward computation, checkpoint saving, and generation after loading:

```bash
uv run python examples/train_tiny.py --steps 160 --output checkpoints/tiny
uv run mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --max-tokens 40
uv run mini-deepinfer --checkpoint checkpoints/tiny --prompt "A KV cache" --draft-tokens 3
```

Use this exercise to inspect training loss, text completion and draft acceptance.
The corpus is small, so the results mainly reflect fitting those training texts.

Training defaults to dense causal attention. The hard Top-K indexer requires a
separate training objective, which is not implemented here. The default random
model still exercises the sparse attention and cache computations.

## Measure performance

The benchmark writes JSON reports for ordinary and speculative decoding. It records
elapsed time, emitted tokens, tokens per second, and proposed and accepted drafts.
Timing includes prefill, scheduling, model computation and sampling, with device
synchronization before and after measurement.

```bash
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --output runs/baseline.json
uv run python examples/benchmark.py --checkpoint checkpoints/tiny --draft-tokens 3 --output runs/spec.json
uv run python examples/validate_device.py --device cpu
uv run ruff check .
uv build
```

Python control flow, cache gathering and stable sorting make the implementation
easy to inspect but can add overhead. Draft verification also costs time, so
speculative decoding may be slower. Compare identical weights, requests, devices,
thread counts, context lengths and cache settings, and alternate multiple runs.

## Device support

| Device | Setup | Validation |
|---|---|---|
| CPU | Default uv environment | Tests, generation, training and cache consistency checks pass |
| CUDA | An appropriate PyTorch build and `device="cuda:0"` | Execution interface available; device testing pending |
| Ascend NPU | Compatible PyTorch, `torch_npu`, drivers and CANN | Execution interface available; device testing pending |

Detailed environments and results are listed in the
[validation record](docs/validation.md). Numerical and output agreement must be
checked on each device.

For an existing CUDA/NPU environment, install only this project with
`uv pip install --python /path/to/env/bin/python -e . --no-deps` and invoke that
interpreter directly. Keep the installed accelerator packages; do not sync the
default CPU lock into that environment. NPU execution uses PyTorch eager mode
and reports errors rather than falling back to CPU. See the
[Ascend notes](docs/ascend.md) for device selection and validation commands.

## Scope

mini-deepinfer has its own model architecture, tokenizer and checkpoint format.
It does not load official DeepSeek-V4.1 weights or depend on the full DeepInfer
runtime. This is an independent educational project with no affiliation to DeepSeek.

The model uses a simplified single-head indexer and expands K/V when reading the cache.
It does not implement matrix absorption, V4.1-specific compression, YOCO,
cross-layer cache sharing, KV quantization, or trained DSpark draft heads.
Page allocation reserves the maximum request capacity and does not support
preemption. Prefix matching uses complete token pages rather than a radix tree.

Distributed TP/EP/CP, EPLB, MegaMoE, asynchronous CPU/device overlap, graph capture,
fused or vendor-specific kernels, structured output constraints and an online API
server are outside the implementation. Sparse attention illustrates the algorithm
and cache behavior without an optimized sparse kernel. The project is intended
for source-code study and small experiments, rather than production serving or
performance comparisons with production engines.

## Source guide

Start with the full-sequence model computation, then follow its use of the cache
and scheduler.

| File | What to read |
|---|---|
| [`config.py`](mini_deepinfer/config.py) | Model settings and sampling parameters |
| [`model.py`](mini_deepinfer/model.py) | Full-sequence `forward()`, paged `forward_batch()`, attention and MoE |
| [`cache.py`](mini_deepinfer/cache.py) | Page allocation, reference counts, prefix sharing and reclamation |
| [`engine.py`](mini_deepinfer/engine.py) | Request handling from admission to completion |
| [`speculative.py`](mini_deepinfer/speculative.py) | Draft generation, allocation, verification and rollback |
| [`tests/`](tests/) | Functional tests and numerical comparisons |

## Tests and contributions

Run `uv run pytest -q` for the test suite, `uv run ruff check .` for linting and
`uv build` to create source and wheel distributions.

Tests compare paged and chunked logits with full-sequence computation. They cover
causality, request isolation, prefix sharing, memory pressure, cancellation, EOS,
sampling, and output agreement after draft acceptance or rejection.
GitHub Actions runs the CPU tests; CUDA/NPU tests require the corresponding device.

Contributions that improve readability, examples or tests are welcome. Performance
changes should include correctness comparisons and measurement conditions.
Discuss larger distributed or custom-kernel additions in an issue first.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## References and license

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer): model execution and engine design.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): compact inference engine organization.
- [mini-sglang](https://github.com/sgl-project/mini-sglang): teaching-oriented runtime and scheduling structure.

Reference revisions are recorded in [provenance.json](provenance.json).
The code is independently written and licensed under Apache-2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
