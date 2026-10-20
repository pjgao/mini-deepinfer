# Contributing

Keep the engine readable and the scope honest.

1. Read `docs/walkthrough.md` and the explicit simplifications in `docs/upstream-map.md`.
2. Install uv, then run `uv sync --locked` for the reproducible CPU development environment.
3. Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`, `uv build` and the relevant example.
4. For a new backend/kernel, keep a portable reference and test it against real outputs.
5. Record the tested device and software versions. Do not infer NPU/CUDA support from CPU CI.

Performance changes should preserve output/quality and state their measurement
scope. Report emitted tokens, not maximum draft capacity. Keep negative results.
Do not check in model weights, private prompts, credentials, device inventories,
internal hostnames or benchmark traces from shared infrastructure.

Commit `pyproject.toml` and `uv.lock` together when changing dependencies.
Use `uv add --dev <package>` for development tools; see
[environment management](docs/environment.md) for dependency updates and existing accelerator environments.

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
