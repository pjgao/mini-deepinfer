"""Bounded CPU/device comparison, using identical locally initialized weights.

Call only after selecting an authorized idle device. Does not probe other devices,
change drivers, start services or allocate a full-size model.
"""

import argparse
import copy
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from mini_deepinfer import LLM, MiniDeepSeek, ModelConfig, SamplingParams
from mini_deepinfer.backend import resolve_device, synchronize
from mini_deepinfer.cache import PagedKV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = resolve_device(args.device)
    c = ModelConfig(vocab_size=32, max_seq_len=128)
    cpu_model = MiniDeepSeek(c).eval()
    target = copy.deepcopy(cpu_model).to(device)
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 4, 5, 6])
    with torch.inference_mode():
        expected = cpu_model(ids)
        actual = target(ids.to(device)).cpu()
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        cache = PagedKV(c, 8, 4, device, torch.float32)
        parts = [
            target.forward_batch([(ids[i : i + 3].tolist(), i, [5, 1, 7])], cache)[0].cpu()
            for i in range(0, len(ids), 3)
        ]
        torch.testing.assert_close(torch.cat(parts), expected, atol=1e-3, rtol=1e-3)
    p = SamplingParams(max_tokens=16, eos_token_id=None)
    prompts = [ids.tolist(), [7, 8, 9]]
    with LLM(copy.deepcopy(cpu_model)) as cpu_llm:
        gold = [o.token_ids for o in cpu_llm.generate(prompts, p)]
    with LLM(target, device=str(device), draft_tokens=3, block_size=4) as target_llm:
        got = [o.token_ids for o in target_llm.generate(prompts, p)]
        if got != gold:
            raise AssertionError(
                "Greedy output differs from CPU; inspect numeric margins before claiming parity"
            )
        again = [o.token_ids for o in target_llm.generate(prompts, p)]
        assert again == gold, "Prefix reuse changed output"
        stats = asdict(target_llm.stats)
    synchronize(device)
    report = {
        "passed": True,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "max_abs_logit_error": float((actual - expected).abs().max()),
        "checks": ["full_logits", "paged_chunked_logits", "greedy_generation", "prefix_reuse"],
        "scope": "tiny float32 eager reference only; no vendor kernels, distributed EP or full V4.1",
        "stats": stats,
    }
    if device.type == "npu":
        import torch_npu

        report.update(torch_npu=torch_npu.__version__, hardware=torch.npu.get_device_name(device))
    elif device.type == "cuda":
        report["hardware"] = torch.cuda.get_device_name(device)
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
