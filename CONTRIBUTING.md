# Contributing

Keep the engine readable and the scope honest.

1. Read `docs/walkthrough.md` and the explicit simplifications in `docs/upstream-map.md`.
2. Install `python -m pip install -e ".[dev]"` after selecting your torch build.
3. Run `ruff check .`, `python -m pytest -q` and the relevant example.
4. For a new backend/kernel, keep a portable reference and test it against real outputs.
5. Record the tested device and software versions. Do not infer NPU/CUDA support from CPU CI.

Performance changes should preserve output/quality and state their measurement
scope. Report emitted tokens, not maximum draft capacity. Keep negative results.
Do not check in model weights, private prompts, credentials, device inventories,
internal hostnames or benchmark traces from shared infrastructure.

This release focuses on a single-process teaching engine. Before adding a large
runtime dependency, distributed stack or model family, discuss whether it belongs
in an optional example or a separate backend.

## Useful extensions

- A real sparse attention reference that gathers only selected keys.
- Dynamic page growth and preemption with progress guarantees.
- A separately tested stochastic speculative sampler.
- A trained indexer objective with explicit quality evaluation.
- A genuine checkpoint adapter with architecture/tokenizer parity tests.

These are future directions, not current features.
