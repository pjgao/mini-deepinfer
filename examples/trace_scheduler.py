"""Watch real requests arrive, prefill in chunks, decode and release pages."""

import json

import torch

from mini_deepinfer import LLM, SamplingParams

torch.set_num_threads(1)
params = SamplingParams(max_tokens=4, eos_token_id=None)
with LLM(block_size=4, max_seqs=2, token_budget=6, prefill_chunk=4) as llm:
    llm.add_request("KV cache", params)
    for tick in range(100):
        if tick == 2:
            llm.add_request("MoE", params)
        if tick == 6:
            llm.add_request("KV cache", params)
        updates = llm.step()
        print(
            json.dumps(
                {
                    "tick": tick,
                    "schedule": llm.last_schedule,
                    "active": len(llm.active),
                    "waiting": len(llm.waiting),
                    "free_pages": len(llm.pool.free),
                    "prefix_hits": llm.stats.prefix_hit_tokens,
                    "updates": [
                        {"id": o.request_id, "generated": len(o.token_ids), "done": o.finish_reason}
                        for o in updates
                    ],
                }
            )
        )
        if tick >= 6 and not llm.has_pending:
            break
