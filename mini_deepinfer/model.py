"""A small model with latent KV, sparse attention and shared/routed experts.

Compressed latent KV + decoupled RoPE + sparse indexer + shared/routed experts.
The reference deliberately expands latent KV before attention to expose the math.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cache import PagedKV
from .config import ModelConfig

TokenBatch = tuple[list[int], int, list[int]]  # token IDs, absolute start, physical pages


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """x: [tokens, heads, even rope_dim], positions: [tokens]."""
    freq = 10000.0 ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    angle = positions.float()[:, None, None] * freq[None, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2)


class LatentAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.dim, config.num_heads * (config.head_dim + config.rope_dim), bias=False)
        self.kv_down = nn.Linear(config.dim, config.kv_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_rank)
        self.kv_up = nn.Linear(config.kv_rank, 2 * config.num_heads * config.head_dim, bias=False)
        self.k_rope = nn.Linear(config.dim, config.rope_dim, bias=False)
        self.index_q = nn.Linear(config.dim, config.index_dim, bias=False)
        self.index_k = nn.Linear(config.dim, config.index_dim, bias=False)
        self.out = nn.Linear(config.num_heads * config.head_dim, config.dim, bias=False)

    def project(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        q = self.q(x).view(-1, config.num_heads, config.head_dim + config.rope_dim)
        q_plain, q_rope = q.split([config.head_dim, config.rope_dim], -1)
        q = torch.cat((q_plain, rope(q_rope, positions)), -1)
        latent = self.kv_norm(self.kv_down(x))
        rope_key = rope(self.k_rope(x)[:, None, :], positions).squeeze(1)
        packed = torch.cat((latent, rope_key, self.index_k(x)), -1)
        return q, self.index_q(x), packed

    def attend(
        self, q: torch.Tensor, index_q: torch.Tensor, packed: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        config = self.config
        latent, rope_key, index_k = packed.split([config.kv_rank, config.rope_dim, config.index_dim], -1)
        k, v = self.kv_up(latent).view(-1, config.num_heads, 2 * config.head_dim).chunk(2, -1)
        k = torch.cat((k, rope_key[:, None, :].expand(-1, config.num_heads, -1)), -1)
        key_pos = torch.arange(len(packed), device=q.device)
        causal = key_pos[None, :] <= positions[:, None]
        mask = causal
        if config.sparse_topk:
            # Select old tokens, then union with the sliding window. Future keys
            # must be masked BEFORE top-k, otherwise prefill can leak information.
            recent = causal & (key_pos[None, :] > positions[:, None] - config.window)
            old = causal & ~recent
            scores = (index_q.float() @ index_k.float().T).relu()
            scores = scores.masked_fill(~old, -torch.inf)
            # Stable ties select the earlier position regardless of chunk length.
            chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, : config.sparse_topk]
            selected = torch.zeros_like(old).scatter_(1, chosen, True) & old
            mask = recent | selected
        score = torch.einsum("thd,shd->hts", q.float(), k.float())
        score = score / (config.head_dim + config.rope_dim) ** 0.5
        probs = score.masked_fill(~mask[None, :, :], -torch.inf).softmax(-1).to(v.dtype)
        out = torch.einsum("hts,shd->thd", probs, v)
        return self.out(out.flatten(1))


class Expert(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.dim, 2 * config.expert_dim, bias=False)
        self.down = nn.Linear(config.expert_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.top_k = config.experts_per_token
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(config) for _ in range(config.num_experts))
        self.shared = Expert(config)
        self.last_load: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score, expert_ids = self.gate(x).float().topk(self.top_k, dim=-1)
        weights = score.softmax(-1).to(x.dtype)
        out = self.shared(x)
        loads = []
        # Gather each expert's tokens, then add weighted outputs at their original positions.
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(expert_ids == expert_id)
            loads.append(len(tokens))
            if tokens.numel():
                out = out.index_add(0, tokens, expert(x[tokens]) * weights[tokens, slots, None])
        self.last_load = loads
        return out


class Layer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attn = LatentAttention(config)
        self.moe = MoE(config)


class MiniDeepSeek(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Uncached full-sequence reference, also used by the tiny training demo."""
        x = self.embedding(tokens)
        positions = torch.arange(len(tokens), device=tokens.device)
        for layer in self.layers:
            q, index_query, packed = layer.attn.project(layer.attn_norm(x), positions)
            x = x + layer.attn.attend(q, index_query, packed, positions)
            x = x + layer.moe(layer.ffn_norm(x))
        return self.head(self.norm(x))

    def forward_batch(self, batches: Sequence[TokenBatch], cache: PagedKV) -> list[torch.Tensor]:
        """batches = [(new token IDs, absolute start, page table), ...].

        Projections and MoE operate on all packed tokens together. Attention uses
        explicit per-request slices, making variable-length boundaries visible.
        """
        if not batches:
            return []
        if any(not tokens for tokens, _, _ in batches):
            raise ValueError("Each request in a batch must contain nonempty token IDs")
        device = self.embedding.weight.device
        lengths = [len(tokens) for tokens, _, _ in batches]
        ids = torch.tensor([t for tokens, _, _ in batches for t in tokens], device=device, dtype=torch.long)
        positions = torch.tensor(
            [p for tokens, start, _ in batches for p in range(start, start + len(tokens))],
            device=device,
            dtype=torch.long,
        )
        x = self.embedding(ids)
        for layer_id, layer in enumerate(self.layers):
            q, index_query, packed = layer.attn.project(layer.attn_norm(x), positions)
            chunks = []
            offset = 0
            for length, (_, start, pages) in zip(lengths, batches, strict=True):
                end = offset + length
                cache.write(layer_id, pages, start, packed[offset:end])
                history = cache.read(layer_id, pages, start + length)
                chunks.append(
                    layer.attn.attend(q[offset:end], index_query[offset:end], history, positions[offset:end])
                )
                offset = end
            x = x + torch.cat(chunks)
            x = x + layer.moe(layer.ffn_norm(x))
        return list(self.head(self.norm(x)).split(lengths))

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, directory: str | Path) -> "MiniDeepSeek":
        path = Path(directory)
        model = cls(ModelConfig(**json.loads((path / "config.json").read_text(encoding="utf-8"))))
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        return model
