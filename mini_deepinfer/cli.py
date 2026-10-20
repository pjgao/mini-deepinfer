import argparse
import json
from dataclasses import asdict

import torch

from .config import SamplingParams
from .engine import LLM
from .model import MiniDeepSeek


def main():
    parser = argparse.ArgumentParser(
        description="Run the tiny educational engine (random weights by default)."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple requests")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--draft-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = MiniDeepSeek.load(args.checkpoint) if args.checkpoint else None
    with LLM(model, device=args.device, draft_tokens=args.draft_tokens) as llm:
        print(
            "Tiny teaching model; no official DeepSeek weights."
            if model is None
            else "Local mini checkpoint."
        )
        params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
        for out in llm.generate(args.prompt or ["Hello, mini-deepinfer!", "What is a KV cache?"], params):
            print(json.dumps(asdict(out), ensure_ascii=False))
        print(json.dumps(asdict(llm.stats), indent=2))


if __name__ == "__main__":
    main()
