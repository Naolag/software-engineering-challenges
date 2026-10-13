"""
Speculative decoding with exact distribution preservation.

Primary solution: a single sequential loop per round with early exit on the
first rejection.

References: Leviathan et al. 2023 ("Fast Inference from Transformers via
Speculative Decoding"); Chen et al. 2023 ("Accelerating LLM Decoding with
Speculative Sampling").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

Distribution = Sequence[float]
Model = Callable[[Sequence[int]], Distribution]

_SUM_TOLERANCE = 1e-6
_RESIDUAL_EPS = 1e-12


@dataclass(frozen=True)
class Stats:
    rounds: int          # number of (conceptually parallel) target verification passes
    drafted: int         # tokens proposed by the draft model
    accepted: int        # drafted tokens accepted by the target model
    generated: int       # tokens returned to the caller

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.generated / self.rounds if self.rounds else 0.0


def _validate(dist: Distribution, vocab_size: int, name: str) -> None:
    if len(dist) != vocab_size:
        raise ValueError(f"{name}: expected {vocab_size} probabilities, got {len(dist)}")
    total = 0.0
    for p in dist:
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"{name}: probabilities must be finite and non-negative")
        total += p
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"{name}: probabilities must sum to 1 (got {total})")


def _sample(weights: Sequence[float], rng: random.Random) -> int:
    """Inverse-CDF sample from (possibly unnormalized) non-negative weights.

    Never returns an index whose weight is zero, even under float rounding.
    """
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("cannot sample from an all-zero weight vector")
    threshold = rng.random() * total
    cumulative = 0.0
    last_positive = -1
    for index, weight in enumerate(weights):
        if weight > 0.0:
            last_positive = index
            cumulative += weight
            if threshold < cumulative:
                return index
    return last_positive  # float round-off guard


def _sample_residual(p: Distribution, q: Distribution, rng: random.Random) -> int:
    """Sample from norm(max(0, p - q)); falls back to p if the residual vanishes."""
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    if sum(residual) <= _RESIDUAL_EPS:
        return _sample(p, rng)
    return _sample(residual, rng)


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
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = rng if rng is not None else random.Random()

    sequence = list(prompt)  # copy: never mutate the caller's prompt
    output: List[int] = []
    rounds = drafted = accepted = 0

    while len(output) < max_new_tokens:
        remaining = max_new_tokens - len(output)
        # A round can emit up to gamma + 1 tokens (accepted drafts + 1 bonus or
        # corrected token), so gamma <= remaining - 1 guarantees no overshoot.
        gamma = min(k, remaining - 1)

        # --- Draft phase: autoregressive, cheap model ---------------------
        draft_tokens: List[int] = []
        draft_dists: List[Distribution] = []
        scratch = list(sequence)
        for _ in range(gamma):
            q = draft(scratch)
            _validate(q, vocab_size, "draft")
            token = _sample(q, rng)
            draft_dists.append(q)
            draft_tokens.append(token)
            scratch.append(token)
            if token == eos_token:
                break  # no point drafting past EOS

        # --- Verify phase: one conceptual parallel target pass ------------
        # target_dists[i] is conditioned on sequence + draft_tokens[:i].
        rounds += 1
        drafted += len(draft_tokens)
        target_dists: List[Distribution] = []
        for i in range(len(draft_tokens) + 1):
            p = target(sequence + draft_tokens[:i])
            _validate(p, vocab_size, "target")
            target_dists.append(p)

        finished = False
        for i, token in enumerate(draft_tokens):
            p, q = target_dists[i], draft_dists[i]
            # Accept with probability min(1, p/q) without dividing: q[token] > 0
            # is guaranteed because the token was sampled from q.
            if rng.random() * q[token] < p[token]:
                accepted += 1
                output.append(token)
                sequence.append(token)
                if token == eos_token:
                    finished = True
                    break
            else:
                corrected = _sample_residual(p, q, rng)
                output.append(corrected)
                sequence.append(corrected)
                finished = corrected == eos_token
                break
        else:
            # Every draft token was accepted: take the free "bonus" token from
            # the target distribution that follows the last accepted token.
            bonus = _sample(target_dists[len(draft_tokens)], rng)
            output.append(bonus)
            sequence.append(bonus)
            finished = bonus == eos_token

        if finished:
            break

    return output, Stats(rounds, drafted, accepted, len(output))
