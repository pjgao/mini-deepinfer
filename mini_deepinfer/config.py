import math
from dataclasses import dataclass


def _validate_count(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259  # bytes 0..255, BOS=256, EOS=257, PAD=258
    dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 16
    kv_rank: int = 16
    rope_dim: int = 8
    index_dim: int = 16
    num_experts: int = 4
    experts_per_token: int = 2
    expert_dim: int = 64
    window: int = 8
    sparse_topk: int = 8
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "dim",
            "num_layers",
            "num_heads",
            "head_dim",
            "kv_rank",
            "rope_dim",
            "index_dim",
            "num_experts",
            "experts_per_token",
            "expert_dim",
            "window",
            "max_seq_len",
        )
        for name in positive:
            _validate_count(name, getattr(self, name), minimum=1)
        _validate_count("sparse_topk", self.sparse_topk)
        if self.rope_dim % 2:
            raise ValueError("rope_dim must be even")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds num_experts")

    @property
    def cache_width(self) -> int:
        return self.kv_rank + self.rope_dim + self.index_dim


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0
    eos_token_id: int | None = 257

    def __post_init__(self) -> None:
        _validate_count("max_tokens", self.max_tokens)
        _validate_count("top_k", self.top_k)
        if self.temperature < 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and nonnegative")
