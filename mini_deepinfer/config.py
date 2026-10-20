from dataclasses import dataclass


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

    def __post_init__(self):
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
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rope_dim % 2:
            raise ValueError("rope_dim must be even")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds num_experts")
        if self.sparse_topk < 0:
            raise ValueError("sparse_topk must be nonnegative (0 = dense causal)")

    @property
    def cache_width(self):
        return self.kv_rank + self.rope_dim + self.index_dim


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0
    eos_token_id: int | None = 257

    def __post_init__(self):
        import math

        if self.max_tokens < 0:
            raise ValueError("max_tokens must be nonnegative")
        if self.temperature < 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
