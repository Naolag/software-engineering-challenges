"""
Speculative decoding: alternative design.

Differences from Solution.py (same output distribution, different structure):
  * Class-based decoder with a single-responsibility `_run_round` method.
  * Acceptance uniforms are pre-drawn for the whole round, and the accept mask
    is computed in one pass; the first rejection is then located with a scan.
  * Sampling uses prefix sums + bisect (O(V) build, O(log V) lookup) instead
    of a linear scan with a round-off guard.
  * Verification results are returned as an immutable RoundResult, which makes
    each round independently testable.

Because the RNG is consumed in a different order, the same seed yields
different token streams than Solution.py. Only the distribution is equal.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, List, Optional, Sequence, Tuple

Distribution = Sequence[float]
Model = Callable[[Sequence[int]], Distribution]


@dataclass(frozen=True)
class Stats:
    rounds: int
    drafted: int
    accepted: int
    generated: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.generated / self.rounds if self.rounds else 0.0


@dataclass(frozen=True)
class RoundResult:
    tokens: Tuple[int, ...]   # tokens emitted this round (accepted + 1 extra)
    drafted: int
    accepted: int


class SpeculativeDecoder:
    SUM_TOLERANCE = 1e-6
    RESIDUAL_EPS = 1e-12

    def __init__(
        self,
        target: Model,
        draft: Model,
        vocab_size: int,
        k: int = 4,
        eos_token: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.target, self.draft = target, draft
        self.vocab_size, self.k, self.eos = vocab_size, k, eos_token
        self.rng = rng if rng is not None else random.Random()

    # -- helpers ----------------------------------------------------------
    def _checked(self, dist: Distribution, name: str) -> Distribution:
        if len(dist) != self.vocab_size:
            raise ValueError(f"{name}: expected {self.vocab_size} probabilities, got {len(dist)}")
        if any((not math.isfinite(p)) or p < 0.0 for p in dist):
            raise ValueError(f"{name}: probabilities must be finite and non-negative")
        if abs(sum(dist) - 1.0) > self.SUM_TOLERANCE:
            raise ValueError(f"{name}: probabilities must sum to 1")
        return dist

    def _sample(self, weights: Sequence[float]) -> int:
        cumulative = list(accumulate(weights))
        total = cumulative[-1] if cumulative else 0.0
        if total <= 0.0:
            raise ValueError("cannot sample from an all-zero weight vector")
        # bisect_right skips zero-weight entries because their prefix sum equals
        # their predecessor's, so a zero-probability token is never returned.
        return bisect_right(cumulative, self.rng.random() * total)

    def _sample_residual(self, p: Distribution, q: Distribution) -> int:
        residual = [max(0.0, a - b) for a, b in zip(p, q)]
        return self._sample(residual if sum(residual) > self.RESIDUAL_EPS else p)

    # -- one round --------------------------------------------------------
    def _run_round(self, sequence: List[int], remaining: int) -> RoundResult:
        gamma = min(self.k, remaining - 1)

        draft_tokens: List[int] = []
        draft_dists: List[Distribution] = []
        scratch = list(sequence)
        for _ in range(gamma):
            q = self._checked(self.draft(scratch), "draft")
            token = self._sample(q)
            draft_dists.append(q)
            draft_tokens.append(token)
            scratch.append(token)
            if token == self.eos:
                break

        n = len(draft_tokens)
        target_dists = [
            self._checked(self.target(sequence + draft_tokens[:i]), "target")
            for i in range(n + 1)
        ]

        # Decide every position up front, then find the first rejection.
        uniforms = [self.rng.random() for _ in range(n)]
        accept_mask = [
            u * draft_dists[i][draft_tokens[i]] < target_dists[i][draft_tokens[i]]
            for i, u in enumerate(uniforms)
        ]
        first_reject = next((i for i, ok in enumerate(accept_mask) if not ok), None)

        if first_reject is None:
            extra = self._sample(target_dists[n])           # bonus token
            return RoundResult(tuple(draft_tokens) + (extra,), n, n)

        extra = self._sample_residual(target_dists[first_reject], draft_dists[first_reject])
        return RoundResult(tuple(draft_tokens[:first_reject]) + (extra,), n, first_reject)

    # -- public API -------------------------------------------------------
    def generate(self, prompt: Sequence[int], max_new_tokens: int) -> Tuple[List[int], Stats]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        sequence, output = list(prompt), []
        rounds = drafted = accepted = 0

        while len(output) < max_new_tokens:
            result = self._run_round(sequence, max_new_tokens - len(output))
            rounds += 1
            drafted += result.drafted
            accepted += result.accepted

            for token in result.tokens:
                output.append(token)
                sequence.append(token)
                if token == self.eos:
                    return output, Stats(rounds, drafted, accepted, len(output))

        return output, Stats(rounds, drafted, accepted, len(output))


def speculative_decode(
    target: Model,
    draft: Model,
    prompt: Sequence[int],
    max_new_tokens: int,
    k: int = 4,
    *,
    vocab_size: int,
    eos_token: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], Stats]:
    """Same signature as Solution.speculative_decode."""
    decoder = SpeculativeDecoder(target, draft, vocab_size, k, eos_token, rng)
    return decoder.generate(prompt, max_new_tokens)
