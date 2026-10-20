"""Greedy speculative verification and confidence-budgeted n-gram drafting.

The drafter estimates confidence from observed suffix frequencies.
Target verification is exact for greedy decoding. Stochastic speculation would
require rejection sampling and a residual distribution; it is intentionally absent.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass
class Draft:
    tokens: list[int]
    confidence: list[float]


class NGramDraft:
    def __init__(self, max_order: int = 4) -> None:
        self.max_order = max_order

    def propose(self, history: list[int], limit: int) -> Draft:
        work = history.copy()
        tokens = []
        confidence = []
        for _ in range(limit):
            followers = Counter()
            for order in range(min(self.max_order, len(work)), 0, -1):
                suffix = work[-order:]
                # Only observed history supplies evidence, never generated drafts.
                followers = Counter(
                    history[i + order]
                    for i in range(len(history) - order)
                    if history[i : i + order] == suffix
                )
                if followers:
                    break
            if not followers:
                break
            token, count = followers.most_common(1)[0]
            tokens.append(token)
            confidence.append(count / sum(followers.values()))
            work.append(token)
        return Draft(tokens, confidence)


def allocate_drafts(proposals: list[Draft], budget: int) -> list[int]:
    """Give a shared verification budget to the highest-confidence prefix first.

    Prefix products estimate the chance of reaching each draft position. Allocation
    can never select position j without also selecting 0..j-1.
    """
    allocated = [0] * len(proposals)
    probability = [1.0] * len(proposals)
    for _ in range(max(0, budget)):
        candidates = [
            (probability[i] * p.confidence[allocated[i]], -i, i)
            for i, p in enumerate(proposals)
            if allocated[i] < len(p.tokens)
        ]
        if not candidates:
            break
        score, _, best = max(candidates)
        if score <= 0:
            break
        allocated[best] += 1
        probability[best] = score
    return allocated


def verify_greedy(draft: list[int], predictions: list[int]) -> tuple[list[int], int]:
    """Input to target is [last_pending_token, *draft].

    predictions[j] predicts draft[j]; predictions[len(draft)] is the bonus.
    Return accepted draft prefix plus the correction/bonus, and prefix length.
    """
    if len(predictions) != len(draft) + 1:
        raise ValueError("Need one target prediction per draft token plus a bonus")
    for i, token in enumerate(draft):
        if token != predictions[i]:
            return [*draft[:i], predictions[i]], i
    return [*draft, predictions[-1]], len(draft)
