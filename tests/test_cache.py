import pytest

from mini_deepinfer.cache import PagePool


def test_full_prefix_sharing_partial_page_private():
    p = PagePool(12, 4)
    a = p.acquire(list(range(10)), 16)
    p.remember(a, list(range(10)), 10)
    b = p.acquire(list(range(10)), 16)
    assert b.cached_tokens == 8
    assert a.pages[:2] == b.pages[:2]
    assert set(a.pages[2:]).isdisjoint(b.pages[2:])
    p.release(a)
    p.release(b)
    p.clear_prefixes()
    p.check_invariants()
    assert len(p.free) == 12


def test_exact_block_prompt_keeps_last_token_uncached():
    p = PagePool(8, 4)
    a = p.acquire(list(range(8)), 12)
    p.remember(a, list(range(8)), 8)
    p.release(a)
    b = p.acquire(list(range(8)), 12)
    assert b.cached_tokens == 4


def test_pressure_eviction_and_no_partial_admission():
    p = PagePool(4, 4)
    a = p.acquire(list(range(8)), 16)
    p.remember(a, list(range(8)), 8)
    assert p.acquire([10, 11], 4) is None
    p.check_invariants()
    p.release(a)
    b = p.acquire([10, 11], 16)
    assert b is not None
    assert not p.prefixes
    p.release(b)
    assert len(p.free) == 4


def test_prefix_key_includes_all_earlier_tokens():
    p = PagePool(10, 2)
    a = p.acquire([1, 2, 3, 4, 5], 6)
    p.remember(a, [1, 2, 3, 4, 5], 5)
    b = p.acquire([9, 2, 3, 4, 5], 6)
    assert b.cached_tokens == 0


def test_double_release_and_oversized_request():
    p = PagePool(2, 4)
    with pytest.raises(ValueError):
        p.acquire([1], 9)
    a = p.acquire([1], 4)
    p.release(a)
    with pytest.raises(ValueError):
        p.release(a)
