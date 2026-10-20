"""Admission -> chunked prefill / decode -> sample / verify -> retire.

One synchronous step packs tokens across requests. CPU scheduling overlap,
distributed EP/TP and graph capture are deliberately left out of this reference.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from .backend import resolve_device
from .cache import Lease, PagedKV, PagePool
from .config import ModelConfig, SamplingParams
from .model import MiniDeepSeek
from .speculative import Draft, NGramDraft, allocate_drafts, verify_greedy
from .tokenizer import ByteTokenizer


@dataclass
class Output:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    finish_reason: str | None


@dataclass
class Stats:
    steps: int = 0
    prefill_tokens: int = 0
    target_decode_tokens: int = 0
    output_tokens: int = 0
    prefix_hit_tokens: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    params: SamplingParams
    generator: torch.Generator
    tokens: list[int] = field(default_factory=list)
    computed: int = 0
    lease: Lease | None = None
    finish_reason: str | None = None

    @property
    def generated(self) -> list[int]:
        return self.tokens[len(self.prompt) :]

    @property
    def generated_count(self) -> int:
        return len(self.tokens) - len(self.prompt)


@dataclass
class ScheduledRequest:
    request: Request
    tokens: list[int]
    is_decode: bool


class LLM:
    def __init__(
        self,
        model: MiniDeepSeek | None = None,
        *,
        config: ModelConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
        num_pages: int = 128,
        block_size: int = 16,
        max_seqs: int = 8,
        token_budget: int = 64,
        prefill_chunk: int = 32,
        prefix_cache: bool = True,
        draft_tokens: int = 0,
    ) -> None:
        if max_seqs <= 0 or token_budget < max_seqs or prefill_chunk <= 0 or draft_tokens < 0:
            raise ValueError("Require 0 < max_seqs <= token_budget, prefill_chunk > 0, draft_tokens >= 0")
        self.device = resolve_device(device)
        if model is None:
            # Isolate initialization from the caller's CPU random state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = MiniDeepSeek(config or ModelConfig())
        elif config is not None and config != model.config:
            raise ValueError("config disagrees with model.config")
        self.model = model.to(self.device).eval()
        self.config = model.config
        self.tokenizer = ByteTokenizer()
        self.pool = PagePool(num_pages, block_size, prefix_cache)
        self.cache = PagedKV(self.config, num_pages, block_size, self.device, next(model.parameters()).dtype)
        self.max_seqs = max_seqs
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.draft_tokens = draft_tokens
        self.drafter = NGramDraft()
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.stats = Stats()
        self.last_schedule: list[dict] = []
        self._next_id = 0
        self.closed = False

    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None) -> str:
        if self.closed:
            raise RuntimeError("Engine is closed")
        params = params or SamplingParams()
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids or any(type(t) is not int or not 0 <= t < self.config.vocab_size for t in ids):
            raise ValueError("Prompt must contain valid token IDs")
        capacity = len(ids) + params.max_tokens
        if capacity > self.config.max_seq_len:
            raise ValueError("Prompt + max_tokens exceeds model max_seq_len")
        if params.max_tokens and capacity > self.pool.num_pages * self.pool.block_size:
            raise ValueError("Request cannot fit in the KV pool")
        if params.eos_token_id is not None and not 0 <= params.eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id outside vocabulary; pass None for custom vocabularies")
        if self.draft_tokens and params.temperature != 0:
            raise ValueError("Speculation supports greedy sampling only; set temperature=0")
        request_id = str(self._next_id)
        self._next_id += 1
        req = Request(request_id, ids, params, torch.Generator().manual_seed(params.seed), ids.copy())
        self.requests[request_id] = req
        if params.max_tokens == 0:
            req.finish_reason = "length"
        else:
            self.waiting.append(req)
        return request_id

    def _admit(self) -> None:
        while self.waiting and len(self.active) < self.max_seqs:
            req = self.waiting[0]
            lease = self.pool.acquire(req.prompt, len(req.prompt) + req.params.max_tokens)
            if lease is None:
                break  # FIFO: running requests will release reserved pages.
            self.waiting.popleft()
            req.lease, req.computed = lease, lease.cached_tokens
            self.stats.prefix_hit_tokens += lease.cached_tokens
            self.active.append(req)

    def _sample(self, logits: torch.Tensor, req: Request) -> int:
        # CPU generator gives each request its own sampling stream on every backend.
        if req.params.temperature == 0:
            return int(logits.argmax().item())
        scores = logits.float().cpu() / req.params.temperature
        if req.params.top_k:
            values, indices = scores.topk(min(req.params.top_k, len(scores)))
            scores = torch.full_like(scores, -torch.inf).scatter_(0, indices, values)
        return int(torch.multinomial(scores.softmax(-1), 1, generator=req.generator).item())

    def _finish(self, req: Request, reason: str) -> None:
        req.finish_reason = reason
        if req.lease is not None:
            self.pool.release(req.lease)
            req.lease = None
        if req in self.active:
            self.active.remove(req)

    def _emit(self, req: Request, tokens: list[int]) -> int:
        count = 0
        for token in tokens:
            req.tokens.append(token)
            count += 1
            self.stats.output_tokens += 1
            if token == req.params.eos_token_id:
                self._finish(req, "eos")
                break
            if req.generated_count >= req.params.max_tokens:
                self._finish(req, "length")
                break
        return count

    @torch.inference_mode()
    def step(self) -> list[Output]:
        if self.closed:
            raise RuntimeError("Engine is closed")
        try:
            return self._step()
        except Exception:
            # Planning, model execution and sampling all own live page leases.
            self.close()
            raise

    def _schedule(self) -> list[ScheduledRequest]:
        decode = [request for request in self.active if request.generated_count]
        prefill = [request for request in self.active if not request.generated_count]
        proposals = [
            self.drafter.propose(
                request.tokens,
                min(self.draft_tokens, request.params.max_tokens - request.generated_count - 1),
            )
            if self.draft_tokens
            else Draft([], [])
            for request in decode
        ]
        counts = allocate_drafts(proposals, self.token_budget - len(decode))
        plans = [
            ScheduledRequest(request, [request.tokens[-1], *proposal.tokens[:count]], is_decode=True)
            for request, proposal, count in zip(decode, proposals, counts, strict=True)
        ]
        remaining = self.token_budget - sum(len(plan.tokens) for plan in plans)
        for request in prefill:
            length = min(self.prefill_chunk, remaining, len(request.prompt) - request.computed)
            if length:
                tokens = request.prompt[request.computed : request.computed + length]
                plans.append(ScheduledRequest(request, tokens, is_decode=False))
                remaining -= length
        return plans

    def _step(self) -> list[Output]:
        self._admit()
        self.last_schedule = []
        if not self.active:
            return []
        plans = self._schedule()
        batches = []
        for plan in plans:
            request = plan.request
            assert request.lease is not None
            batches.append((plan.tokens, request.computed, request.lease.pages))
        self.last_schedule = [
            {
                "request_id": plan.request.request_id,
                "phase": "decode" if plan.is_decode else "prefill",
                "start": plan.request.computed,
                "tokens": len(plan.tokens),
                "drafts": len(plan.tokens) - 1 if plan.is_decode else 0,
            }
            for plan in plans
        ]
        results = self.model.forward_batch(batches, self.cache)
        if len(results) != len(plans):
            raise ValueError("Model must return one logits tensor per scheduled request")
        self.stats.steps += 1
        updates = []
        for plan, logits in zip(plans, results, strict=True):
            req, ids = plan.request, plan.tokens
            if plan.is_decode:
                self.stats.target_decode_tokens += len(ids)
                self.stats.proposed_tokens += len(ids) - 1
                if self.draft_tokens:
                    emitted, accepted = verify_greedy(ids[1:], logits.argmax(-1).tolist())
                else:
                    emitted, accepted = [self._sample(logits[-1], req)], 0
                # Only input + accepted prefix is valid. The correction/bonus is
                # pending; rejected suffix slots will be overwritten next step.
                req.computed += 1 + accepted
                emitted_count = self._emit(req, emitted)
                self.stats.accepted_tokens += min(accepted, emitted_count)
                updates.append(self.output(req.request_id))
            else:
                req.computed += len(ids)
                self.stats.prefill_tokens += len(ids)
                assert req.lease is not None
                self.pool.remember(req.lease, req.prompt, req.computed)
                if req.computed == len(req.prompt):
                    self._emit(req, [self._sample(logits[-1], req)])
                    updates.append(self.output(req.request_id))
        self.pool.check_invariants()
        return updates

    @property
    def has_pending(self) -> bool:
        return bool(self.waiting or self.active)

    def output(self, request_id: str) -> Output:
        request = self.requests[request_id]
        generated = request.generated
        return Output(
            request_id=request.request_id,
            prompt_token_ids=request.prompt.copy(),
            token_ids=generated,
            text=self.tokenizer.decode(generated),
            finish_reason=request.finish_reason,
        )

    def cancel(self, request_id: str) -> None:
        req = self.requests[request_id]
        if req.finish_reason is not None:
            return
        if req in self.waiting:
            self.waiting.remove(req)
        self._finish(req, "cancelled")

    def generate(
        self,
        prompts: str | Iterable[str | list[int]],
        params: SamplingParams | None = None,
    ) -> list[Output]:
        if self.closed:
            raise RuntimeError("Engine is closed")
        if isinstance(prompts, str):
            prompts = [prompts]
        ids = []
        try:
            for prompt in prompts:
                ids.append(self.add_request(prompt, params))
        except Exception:
            for request_id in ids:
                self.cancel(request_id)
            raise
        while any(self.requests[i].finish_reason is None for i in ids):
            self.step()
        return [self.output(i) for i in ids]

    def close(self) -> None:
        for req in [*self.waiting, *self.active]:
            self.cancel(req.request_id)
        self.pool.clear_prefixes()
        self.pool.check_invariants()
        self.closed = True

    def __enter__(self) -> "LLM":
        if self.closed:
            raise RuntimeError("Engine is closed")
        return self

    def __exit__(self, *_) -> None:
        self.close()
