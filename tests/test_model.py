from dataclasses import replace

import pytest
import torch

from mini_deepinfer import MiniDeepSeek
from mini_deepinfer.cache import PagedKV


@pytest.mark.parametrize("chunk", [1, 3, 7, 19])
@pytest.mark.parametrize("sparse", [0, 2])
@torch.inference_mode()
def test_cached_logits_match_full_sequence(model, chunk, sparse):
    model.config = replace(model.config, sparse_topk=sparse)
    for layer in model.layers:
        layer.attn.config = model.config
    ids = torch.randint(0, 32, (19,))
    expected = model(ids)
    cache = PagedKV(model.config, 8, 4, "cpu", torch.float32)
    # Non-contiguous physical pages catch accidental logical=physical indexing.
    pages = [7, 1, 5, 3, 0]
    actual = []
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk].tolist()
        actual.extend(model.forward_batch([(part, start, pages)], cache))
    torch.testing.assert_close(torch.cat(actual), expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_future_tokens_cannot_change_prefix_logits(model):
    a = torch.randint(0, 32, (24,))
    b = a.clone()
    b[9:] = (b[9:] + 1) % 32
    torch.testing.assert_close(model(a)[:9], model(b)[:9], atol=0, rtol=0)


@torch.inference_mode()
def test_packed_requests_do_not_mix_attention(model):
    a, b = [1, 2, 3, 4, 5], [9, 8, 7]
    cache = PagedKV(model.config, 8, 4, "cpu", torch.float32)
    aa, bb = model.forward_batch([(a, 0, [3, 1]), (b, 0, [5])], cache)
    torch.testing.assert_close(aa, model(torch.tensor(a)), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(bb, model(torch.tensor(b)), atol=2e-6, rtol=2e-5)


def test_training_gradient_and_checkpoint_roundtrip(model, tmp_path):
    ids = torch.tensor([1, 2, 3, 4, 5])
    loss = torch.nn.functional.cross_entropy(model(ids)[:-1], ids[1:])
    loss.backward()
    assert model.layers[0].attn.kv_down.weight.grad.abs().sum() > 0
    assert model.layers[0].moe.shared.down.weight.grad.abs().sum() > 0
    model.save(tmp_path)
    restored = MiniDeepSeek.load(tmp_path)
    torch.testing.assert_close(model(ids), restored(ids), atol=0, rtol=0)


def test_empty_batch_returns_no_results(model):
    cache = PagedKV(model.config, 8, 4, "cpu", torch.float32)
    assert model.forward_batch([], cache) == []


def test_empty_request_in_batch_is_rejected_before_cache_write(model):
    cache = PagedKV(model.config, 8, 4, "cpu", torch.float32)
    before = cache.data.clone()
    with pytest.raises(ValueError, match="nonempty"):
        model.forward_batch([([1, 2], 0, [1]), ([], 0, [2])], cache)
    torch.testing.assert_close(cache.data, before, atol=0, rtol=0)
