"""Train a tiny dense-attention MoE on a public, in-repository synthetic corpus.

This is a learning exercise, not a useful language model or DeepSeek distillation.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from mini_deepinfer import MiniDeepSeek, ModelConfig
from mini_deepinfer.backend import resolve_device
from mini_deepinfer.tokenizer import ByteTokenizer

CORPUS = """A KV cache stores previous keys and values.
A page table maps logical tokens to physical pages.
Prefill reads the prompt. Decode generates the next token.
A shared expert processes every token.
A routed expert processes selected tokens.
The target model verifies draft tokens.
The scheduler gives each request a token budget.
mini-deepinfer is a small engine for learning.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--output", default="checkpoints/tiny")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    # Discrete index selection does not train an indexer. Start with dense causal
    # attention; the sparse path remains available in the default random demo.
    model = MiniDeepSeek(ModelConfig(sparse_topk=0)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tokenizer = ByteTokenizer()
    # Use the same BOS framing as inference; each line has an explicit EOS target.
    data = [tokenizer.encode(line) + [tokenizer.eos_token_id] for line in CORPUS.splitlines()]
    losses = []
    for step in range(args.steps):
        tokens = torch.tensor(rng.choice(data), device=device)
        loss = F.cross_entropy(model(tokens[:-1]), tokens[1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 40 == 0:
            print(f"step={step + 1} loss={losses[-1]:.4f}")
    model.eval().cpu().save(args.output)
    report = {
        "seed": args.seed,
        "steps": args.steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "purpose": "synthetic corpus overfit demonstration; no held-out quality claim",
    }
    Path(args.output, "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved mini checkpoint to {args.output}")


if __name__ == "__main__":
    main()
