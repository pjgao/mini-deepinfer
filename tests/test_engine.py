import copy

import pytest
import torch

from mini_deepinfer import LLM, SamplingParams
from mini_deepinfer.speculative import Draft


def params(n=12, **kwargs):
    return SamplingParams(max_tokens=n, eos_token_id=None, **kwargs)


@torch.inference_mode()
def reference_greedy(model, prompt, count):
    tokens = prompt.copy()
    for _ in range(count):
        tokens.append(int(model(torch.tensor(tokens))[-1].argmax()))
    return tokens[len(prompt) :]


def test_engine_matches_uncached_reference_with_small_budget(model):
    prompts = [[1, 2, 3, 4, 5], [6, 7], [8] * 12]
    expected = [reference_greedy(model, p, 12) for p in prompts]
    with LLM(model, token_budget=4, max_seqs=2, prefill_chunk=2, block_size=4) as llm:
        outputs = llm.generate(prompts, params())
        assert [o.token_ids for o in outputs] == expected
        assert all(o.finish_reason == "length" for o in outputs)
    assert len(llm.pool.free) == llm.pool.num_pages


def test_prefix_reuse_preserves_output_and_cache(model):
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with LLM(model, block_size=4) as llm:
        a = llm.generate([prompt], params())[0]
        b = llm.generate([prompt], params())[0]
        assert a.token_ids == b.token_ids
        assert llm.stats.prefix_hit_tokens == 8


def test_capacity_queue_makes_progress(model):
    with LLM(model, num_pages=4, block_size=4, max_seqs=4, token_budget=8) as llm:
        outputs = llm.generate([[1, 2, 3]] * 4, params())
        assert len(outputs) == 4
        assert all(len(o.token_ids) == 12 for o in outputs)


def test_new_arrivals_and_cancellation_release_pages(model):
    with LLM(model, max_seqs=1, token_budget=4, prefill_chunk=2) as llm:
        a = llm.add_request([1, 2, 3], params())
        llm.step()
        b = llm.add_request([3, 4], params())
        llm.cancel(a)
        assert llm.output(a).finish_reason == "cancelled"
        while llm.has_pending:
            llm.step()
        assert llm.output(b).finish_reason == "length"
        c = llm.add_request([1, 2], params())
        llm.cancel(c)  # waiting cancellation
        assert not llm.has_pending
    assert all(n == 0 for n in llm.pool.refs)


def test_seeded_sampling_independent_of_batch_composition(model):
    p = params(20, temperature=0.8, seed=42)
    with LLM(copy.deepcopy(model), prefill_chunk=2) as llm:
        single = llm.generate([[1, 2, 3]], p)[0].token_ids
    with LLM(model, prefill_chunk=7) as llm:
        batch = llm.generate([[9, 8], [1, 2, 3], [3] * 20], p)[1].token_ids
    assert single == batch


class OracleDraft:
    """Test-only drafter: force acceptance and a selected rejection position."""

    def __init__(self, model, reject_at=None):
        self.model, self.reject_at = model, reject_at

    def propose(self, history, limit):
        tokens = reference_greedy(self.model, history, limit)
        if self.reject_at is not None and len(tokens) > self.reject_at:
            tokens[self.reject_at] = (tokens[self.reject_at] + 1) % self.model.config.vocab_size
        return Draft(tokens, [1.0] * len(tokens))


@pytest.mark.parametrize("reject_at", [None, 0, 1, 2])
def test_speculation_acceptance_rejection_and_rollback(model, reject_at):
    prompt = [1, 2, 3, 4]
    expected = reference_greedy(model, prompt, 17)
    with LLM(model, draft_tokens=3, block_size=4, token_budget=12) as llm:
        llm.drafter = OracleDraft(model, reject_at)
        got = llm.generate([prompt], params(17))[0]
        assert got.token_ids == expected
        assert llm.stats.proposed_tokens > 0
        assert llm.stats.accepted_tokens > 0 if reject_at != 0 else llm.stats.accepted_tokens == 0


def test_real_ngram_speculation_matches_greedy(model):
    prompt = [1, 2, 3] * 4
    expected = reference_greedy(model, prompt, 20)
    with LLM(model, draft_tokens=4) as llm:
        assert llm.generate([prompt], params(20))[0].token_ids == expected


def test_speculation_shares_budget_across_live_requests(model):
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    expected = [reference_greedy(model, p, 13) for p in prompts]
    with LLM(model, draft_tokens=3, max_seqs=3, token_budget=7, prefill_chunk=3) as llm:
        llm.drafter = OracleDraft(model, reject_at=1)
        ids = [llm.add_request(p, params(13)) for p in prompts]
        while llm.has_pending:
            llm.step()
            assert sum(p["tokens"] for p in llm.last_schedule) <= 7
            for req in llm.active:
                if req.generated:
                    assert req.computed == len(req.tokens) - 1
        assert [llm.output(i).token_ids for i in ids] == expected


def test_eos_inside_verified_prefix_truncates_stats(model):
    prompt = [1, 2, 3]
    sequence = reference_greedy(model, prompt, 12)
    eos = next((t for t in sequence[1:] if t != sequence[0]), None)
    if eos is None:
        pytest.skip("Model emitted a constant sequence")
    expected = sequence[: sequence.index(eos) + 1]
    with LLM(model, draft_tokens=4) as llm:
        llm.drafter = OracleDraft(model)
        out = llm.generate([prompt], SamplingParams(max_tokens=12, eos_token_id=eos))[0]
        assert out.token_ids == expected and out.finish_reason == "eos"
        assert llm.stats.output_tokens == len(expected)
        assert llm.stats.accepted_tokens <= len(expected) - 1


def test_eos_and_zero_output(model):
    prompt = [1, 2]
    eos = reference_greedy(model, prompt, 1)[0]
    with LLM(model) as llm:
        out = llm.generate([prompt], SamplingParams(max_tokens=8, eos_token_id=eos))[0]
        assert out.token_ids == [eos]
        assert out.finish_reason == "eos"
        zero = llm.generate([prompt], params(0))[0]
        assert zero.token_ids == [] and zero.finish_reason == "length"


def test_invalid_requests(model):
    with LLM(model, num_pages=2, block_size=4, draft_tokens=2) as llm:
        for prompt, p in [
            ([], params(1)),
            ([999], params(1)),
            ([1], params(9)),
            ([1], params(1, temperature=1)),
            ([1], SamplingParams()),
        ]:
            with pytest.raises(ValueError):
                llm.add_request(prompt, p)


def test_forward_failure_cleans_resources(model, monkeypatch):
    llm = LLM(model)
    llm.add_request([1, 2], params())

    def fail(*_):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model, "forward_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        llm.step()
    assert llm.closed and all(n == 0 for n in llm.pool.refs)


def test_top_k_limits_candidates_when_logits_are_tied(model):
    with LLM(model) as llm:
        request_id = llm.add_request([1], params(1, temperature=1, top_k=1))
        request = llm.requests[request_id]
        logits = torch.zeros(model.config.vocab_size)
        expected = logits.topk(1).indices.item()
        assert {llm._sample(logits, request) for _ in range(32)} == {expected}


def test_sampling_failure_releases_requests_and_prefixes(model, monkeypatch):
    llm = LLM(model, block_size=2)
    llm.add_request([1, 2, 3, 4], params())

    def fail(*_):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(llm, "_sample", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        llm.step()
    assert llm.closed
    assert not llm.has_pending
    assert not llm.pool.prefixes
    assert all(count == 0 for count in llm.pool.refs)


def test_drafter_failure_releases_active_requests(model, monkeypatch):
    llm = LLM(model, draft_tokens=2)
    llm.add_request([1, 2, 3], params())
    llm.step()

    def fail(*_):
        raise RuntimeError("draft failed")

    monkeypatch.setattr(llm.drafter, "propose", fail)
    with pytest.raises(RuntimeError, match="draft failed"):
        llm.step()
    assert llm.closed and not llm.has_pending
    assert all(count == 0 for count in llm.pool.refs)


def test_incomplete_model_result_fails_instead_of_stalling(model, monkeypatch):
    llm = LLM(model)
    llm.add_request([1, 2], params())
    monkeypatch.setattr(model, "forward_batch", lambda *_: [])
    with pytest.raises(ValueError):
        llm.step()
    assert llm.closed and not llm.has_pending
    assert all(count == 0 for count in llm.pool.refs)


def test_invalid_generate_batch_cancels_only_its_new_requests(model):
    with LLM(model) as llm:
        existing = llm.add_request([7, 8], params(1))
        with pytest.raises(ValueError, match="valid token IDs"):
            llm.generate([[1, 2], []], params(2))
        assert [request.request_id for request in llm.waiting] == [existing]
        assert llm.output(existing).finish_reason is None
        while llm.has_pending:
            llm.step()
        assert llm.output(existing).finish_reason == "length"
