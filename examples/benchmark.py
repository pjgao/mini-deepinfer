"""End-to-end offline benchmark: count emitted tokens, never speculative capacity."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mini_deepinfer import LLM, MiniDeepSeek, SamplingParams
from mini_deepinfer.backend import synchronize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--draft-tokens", type=int, default=0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.requests <= 0 or args.max_tokens <= 0:
        parser.error("--requests and --max-tokens must be positive")
    torch.set_num_threads(1)
    prompts = ["A KV cache stores", "The target model", "A shared expert", "The scheduler"] * args.requests
    prompts = prompts[: args.requests]
    params = SamplingParams(max_tokens=args.max_tokens, eos_token_id=None)
    model = MiniDeepSeek.load(args.checkpoint) if args.checkpoint else None
    with LLM(model, device=args.device, draft_tokens=args.draft_tokens, prefix_cache=False) as llm:
        llm.generate([prompts[0]], SamplingParams(max_tokens=4, eos_token_id=None))
        llm.stats = type(llm.stats)()
        synchronize(llm.device)
        begin = time.perf_counter()
        outputs = llm.generate(prompts, params)
        synchronize(llm.device)
        elapsed = time.perf_counter() - begin
        count = sum(len(o.token_ids) for o in outputs)
        report = {
            "scope": "offline end-to-end, includes prefill and host work; not an online SLA",
            "device": str(llm.device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "requests": args.requests,
            "draft_tokens": args.draft_tokens,
            "emitted_tokens": count,
            "elapsed_seconds": elapsed,
            "tokens_per_second": count / elapsed,
            "accepted_fraction": (
                llm.stats.accepted_tokens / llm.stats.proposed_tokens if llm.stats.proposed_tokens else None
            ),
            "kv_bytes_reserved": llm.cache.nbytes,
            "stats": asdict(llm.stats),
        }
        print(json.dumps(report, indent=2))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
