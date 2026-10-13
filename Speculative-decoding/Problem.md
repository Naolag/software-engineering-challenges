# Speculative Decoding with Exact Distribution Preservation

**Category:** Algorithms / probability / LLM inference
**Difficulty:** Hard
**Language:** Python 3.9+ (standard library only)

## Background

Autoregressive LLM decoding is slow because each token needs one full forward pass of a large **target** model, and those passes are sequential. **Speculative decoding** speeds this up with a small, fast **draft** model:

1. The draft model cheaply proposes up to `k` tokens.
2. The target model scores all proposed positions in a **single parallel pass**.
3. Each proposed token is accepted or rejected so that the final output is **statistically identical to sampling from the target model alone**. It is a pure speedup with no quality trade-off.

## Task

Implement:

```python
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
) -> Tuple[List[int], Stats]
```

- `Model` is a callable `model(context: Sequence[int]) -> Sequence[float]` returning a probability distribution over the next token (length `vocab_size`, non-negative, sums to 1).
- `target` and `draft` are pure functions of the context. Calling them twice with the same context gives the same answer.
- Return the generated tokens (excluding the prompt) and a `Stats` object with `rounds`, `drafted`, `accepted`, and `generated`.

## Requirements

1. **Exactness.** For any `target`, `draft`, `prompt` and `k`, the distribution over returned token sequences must equal the distribution obtained by sampling one token at a time from `target`. This is the central requirement.
2. **Parallel verification.** Each round may draft up to `k` tokens, then evaluate `target` at the `len(drafted) + 1` positions it needs. Count this as **one** round in `Stats.rounds`.
3. **Budget.** Never return more than `max_new_tokens` tokens. If no `eos_token` is hit, return exactly `max_new_tokens`.
4. **EOS.** If `eos_token` is generated, it is included as the last token and generation stops.
5. **Robustness.**
   - Never emit a token that the target model assigns probability 0.
   - Handle models whose supports are disjoint, and models that are identical.
   - Do not mutate `prompt`.
   - Raise `ValueError` for `k < 1`, a negative `max_new_tokens`, and malformed distributions (wrong length, negative or non-finite values, sum not 1).
6. **Determinism.** The same `rng` seed must give the same output.
7. Do not use NumPy, SciPy or any ML library.

## Examples

```python
target = lambda ctx: [0.5, 0.5, 0.0, 0.0]
draft  = lambda ctx: [0.0, 0.0, 0.5, 0.5]       # disjoint support
tokens, stats = speculative_decode(target, draft, [0], 10, 4, vocab_size=4)
# every token is 0 or 1; stats.accepted == 0

tokens, stats = speculative_decode(target, target, [0], 12, 4, vocab_size=4)
# identical models: every draft token is accepted, stats.acceptance_rate == 1.0
```

## Constraints

- `1 <= vocab_size <= 50_000`, `0 <= max_new_tokens <= 10_000`, `1 <= k <= 32`
- Distributions may contain exact zeros and extremely small values.

## Hints

<details>
<summary>Hint 1: acceptance</summary>
Accept a drafted token <code>x</code> with probability <code>min(1, p(x) / q(x))</code>, where <code>p</code> is the target and <code>q</code> the draft distribution.
</details>

<details>
<summary>Hint 2: what happens on rejection</summary>
Sampling from <code>p</code> again is <b>not</b> correct. Work out which distribution makes the total probability of each token equal <code>p(x)</code>.
</details>

<details>
<summary>Hint 3: the end of a round</summary>
If every drafted token is accepted, the target pass already produced the distribution for the next position. Think about how to use it, and how many tokens a round may emit when the remaining budget is small.
</details>

## How solutions are graded

| Area | What is checked |
|---|---|
| Exactness (50%) | Chi-square goodness of fit against the exact target joint distribution |
| Edge cases (25%) | Zero-probability tokens, disjoint support, EOS, budget limits, k larger than the remaining budget |
| Input validation (10%) | `ValueError` on bad arguments and malformed distributions |
| Code quality (15%) | Readability, naming, separation of concerns, no hidden global state |
