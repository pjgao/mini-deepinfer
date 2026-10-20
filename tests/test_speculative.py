import pytest

from mini_deepinfer.speculative import Draft, NGramDraft, allocate_drafts, verify_greedy


@pytest.mark.parametrize(
    ("draft", "pred", "expected", "accepted"),
    [
        ([2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5], 3),
        ([2, 3, 4], [9, 3, 4, 5], [9], 0),
        ([2, 3, 4], [2, 9, 4, 5], [2, 9], 1),
        ([2, 3, 4], [2, 3, 9, 5], [2, 3, 9], 2),
        ([], [7], [7], 0),
    ],
)
def test_verify(draft, pred, expected, accepted):
    assert verify_greedy(draft, pred) == (expected, accepted)


def test_confidence_budget_is_prefix_closed():
    proposals = [Draft([1, 2, 3], [0.9, 0.9, 0.9]), Draft([4, 5], [0.8, 0.1])]
    assert allocate_drafts(proposals, 3) == [2, 1]
    assert allocate_drafts(proposals, 100) == [3, 2]
    assert allocate_drafts(proposals, 0) == [0, 0]


def test_ngram_uses_observed_history():
    draft = NGramDraft().propose([1, 2, 3, 1, 2, 3, 1, 2], 3)
    assert draft.tokens == [3, 1, 2]
    assert draft.confidence == [1, 1, 1]
    assert NGramDraft().propose([1, 2, 3], 4).tokens == []
