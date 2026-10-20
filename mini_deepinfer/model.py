"""A tiny DeepSeek-inspired model, NOT a V4.1 checkpoint implementation.

Compressed latent KV + decoupled RoPE + sparse indexer + shared/routed experts.
The reference deliberately expands latent KV before attention to expose the math.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def rope(x, positions):
    """x: [tokens, heads, even rope_dim], positions: [tokens]."""
    freq = 10000.0 ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    angle = positions.float()[:, None, None] * freq[None, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2)


class LatentAttention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.dim, c.num_heads * (c.head_dim + c.rope_dim), bias=False)
        self.kv_down = nn.Linear(c.dim, c.kv_rank, bias=False)
        self.kv_norm = RMSNorm(c.kv_rank)
        self.kv_up = nn.Linear(c.kv_rank, 2 * c.num_heads * c.head_dim, bias=False)
        self.k_rope = nn.Linear(c.dim, c.rope_dim, bias=False)
        self.index_q = nn.Linear(c.dim, c.index_dim, bias=False)
        self.index_k = nn.Linear(c.dim, c.index_dim, bias=False)
        self.out = nn.Linear(c.num_heads * c.head_dim, c.dim, bias=False)

    def project(self, x, positions):
        c = self.c
        q = self.q(x).view(-1, c.num_heads, c.head_dim + c.rope_dim)
        q_plain, q_rope = q.split([c.head_dim, c.rope_dim], -1)
        q = torch.cat((q_plain, rope(q_rope, positions)), -1)
        latent = self.kv_norm(self.kv_down(x))
        kr = rope(self.k_rope(x)[:, None, :], positions).squeeze(1)
        packed = torch.cat((latent, kr, self.index_k(x)), -1)
        return q, self.index_q(x), packed

    def attend(self, q, index_q, packed, positions):
        c = self.c
        latent, kr, index_k = packed.split([c.kv_rank, c.rope_dim, c.index_dim], -1)
        k, v = self.kv_up(latent).view(-1, c.num_heads, 2 * c.head_dim).chunk(2, -1)
        k = torch.cat((k, kr[:, None, :].expand(-1, c.num_heads, -1)), -1)
        key_pos = torch.arange(len(packed), device=q.device)
        causal = key_pos[None, :] <= positions[:, None]
        mask = causal
        if c.sparse_topk:
            # Select old tokens, then union with the sliding window. Future keys
            # must be masked BEFORE top-k, otherwise prefill can leak information.
            recent = causal & (key_pos[None, :] > positions[:, None] - c.window)
            old = causal & ~recent
            scores = (index_q.float() @ index_k.float().T).relu()
            scores = scores.masked_fill(~old, -torch.inf)
            # Stable ties select the earlier position regardless of chunk length.
            chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, : c.sparse_topk]
            selected = torch.zeros_like(old).scatter_(1, chosen, True) & old
            mask = recent | selected
        score = torch.einsum("thd,shd->hts", q.float(), k.float())
        score = score / (c.head_dim + c.rope_dim) ** 0.5
        probs = score.masked_fill(~mask[None, :, :], -torch.inf).softmax(-1).to(v.dtype)
        out = torch.einsum("hts,shd->thd", probs, v)
        return self.out(out.flatten(1))


class Expert(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up = nn.Linear(c.dim, 2 * c.expert_dim, bias=False)
        self.down = nn.Linear(c.expert_dim, c.dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k = c.experts_per_token
        self.gate = nn.Linear(c.dim, c.num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(c) for _ in range(c.num_experts))
        self.shared = Expert(c)
        self.last_load = None

    def forward(self, x):
        score, expert_ids = self.gate(x).float().topk(self.top_k, dim=-1)
        weights = score.softmax(-1).to(x.dtype)
        out = self.shared(x)
        loads = []
        # Real dispatch/gather/combine on one device; no EP or fused kernel claim.
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(expert_ids == expert_id)
            loads.append(len(tokens))
            if len(tokens):
                out = out.index_add(0, tokens, expert(x[tokens]) * weights[tokens, slots, None])
        self.last_load = loads
        return out


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn, self.moe = LatentAttention(c), MoE(c)


class MiniDeepSeek(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        """Uncached full-sequence reference, also used by the tiny training demo."""
        x = self.embedding(tokens)
        positions = torch.arange(len(tokens), device=tokens.device)
        for layer in self.layers:
            q, iq, packed = layer.attn.project(layer.attn_norm(x), positions)
            x = x + layer.attn.attend(q, iq, packed, positions)
            x = x + layer.moe(layer.ffn_norm(x))
        return self.head(self.norm(x))

    def forward_batch(self, batches, cache):
        """batches = [(new token IDs, absolute start, page table), ...].

        Projections and MoE operate on all packed tokens together. Attention uses
        explicit per-request slices, making variable-length boundaries visible.
        """
        device = self.embedding.weight.device
        lengths = [len(tokens) for tokens, _, _ in batches]
        ids = torch.tensor([t for tokens, _, _ in batches for t in tokens], device=device)
        positions = torch.tensor(
            [p for tokens, start, _ in batches for p in range(start, start + len(tokens))], device=device
        )
        x = self.embedding(ids)
        for layer_id, layer in enumerate(self.layers):
            q, iq, packed = layer.attn.project(layer.attn_norm(x), positions)
            chunks, offset = [], 0
            for length, (_, start, pages) in zip(lengths, batches):
                end = offset + length
                cache.write(layer_id, pages, start, packed[offset:end])
                history = cache.read(layer_id, pages, start + length)
                chunks.append(
                    layer.attn.attend(q[offset:end], iq[offset:end], history, positions[offset:end])
                )
                offset = end
            x = x + torch.cat(chunks)
            x = x + layer.moe(layer.ffn_norm(x))
        return list(self.head(self.norm(x)).split(lengths))

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        model = cls(ModelConfig(**json.loads((path / "config.json").read_text(encoding="utf-8"))))
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        return model
