"""
Test suite for the Speculative Decoding challenge.

Run:  python -m unittest tests -v     (standard library only)

The key test is statistical: the output distribution of speculative decoding
must equal the target model's distribution. It is a chi-square goodness-of-fit
test against the exact joint distribution, and a mutation test proves the
suite can detect a plausible, subtly wrong implementation.
"""

import itertools
import math
import random
import unittest
from unittest import mock

import AlternativeSolution
import Solution

VOCAB = 4


# --------------------------------------------------------------------------
# Toy context-dependent language models (deterministic functions of context)
# --------------------------------------------------------------------------
def make_model(a, b, scale, offset=0.0, vocab=VOCAB):
    def model(context):
        last = context[-1] if context else 0
        logits = [
            scale * math.sin(a * (j + 1) * (last + 2) + b * len(context) + offset)
            for j in range(vocab)
        ]
        peak = max(logits)
        exps = [math.exp(x - peak) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]
    return model


TARGET = make_model(a=1.7, b=0.3, scale=2.0)
DRAFT = make_model(a=1.3, b=0.5, scale=1.2, offset=0.9)   # related but different


def exact_joint(model, prompt, length, vocab=VOCAB):
    """Exact probability of every length-`length` continuation under `model`."""
    joint = {}
    for seq in itertools.product(range(vocab), repeat=length):
        prob, ctx = 1.0, list(prompt)
        for token in seq:
            prob *= model(ctx)[token]
            ctx.append(token)
        joint[seq] = prob
    return joint


def chi_square_stat(counts, expected_probs, n, min_expected=5.0):
    """Pearson chi-square with sparse cells pooled. Returns (stat, df)."""
    stat, df = 0.0, -1
    pooled_obs = pooled_exp = 0.0
    for key, prob in expected_probs.items():
        expected = prob * n
        if expected < min_expected:
            pooled_obs += counts.get(key, 0)
            pooled_exp += expected
        else:
            stat += (counts.get(key, 0) - expected) ** 2 / expected
            df += 1
    if pooled_exp >= min_expected:
        stat += (pooled_obs - pooled_exp) ** 2 / pooled_exp
        df += 1
    return stat, df


def chi_square_critical(df, z=3.09):
    """Wilson-Hilferty approximation of the chi-square quantile (p = 0.001)."""
    c = 2.0 / (9.0 * df)
    return df * (1 - c + z * math.sqrt(c)) ** 3


def sample_counts(decode, n, length, k, seed):
    rng = random.Random(seed)
    counts = {}
    for _ in range(n):
        tokens, _ = decode(TARGET, DRAFT, [1], length, k, vocab_size=VOCAB, rng=rng)
        key = tuple(tokens)
        counts[key] = counts.get(key, 0) + 1
    return counts


def fixed(dist):
    return lambda context: list(dist)


class SpeculativeDecodingTests:
    """Mixin: concrete classes below bind `decode` to a specific solution."""

    decode = None  # set by subclasses (staticmethod)

    def run_decode(self, *args, **kwargs):
        return type(self).decode(*args, **kwargs)

    # ---- correctness of the output distribution --------------------------
    def test_output_distribution_matches_target(self):
        n, length = 60000, 3
        counts = sample_counts(type(self).decode, n, length, k=3, seed=12345)
        stat, df = chi_square_stat(counts, exact_joint(TARGET, [1], length), n)
        self.assertLess(stat, chi_square_critical(df),
                        f"chi-square {stat:.1f} exceeds critical value (df={df})")

    def test_mutation_resampling_from_target_is_detected(self):
        """A plausible bug (resample from p, not the residual) must FAIL the test."""
        buggy = lambda p, q, rng: Solution._sample(p, rng)
        n, length = 60000, 3
        with mock.patch.object(Solution, "_sample_residual", buggy):
            counts = sample_counts(Solution.speculative_decode, n, length, k=3, seed=12345)
        stat, df = chi_square_stat(counts, exact_joint(TARGET, [1], length), n)
        self.assertGreater(stat, chi_square_critical(df),
                           "test suite failed to detect the residual-sampling bug")

    # ---- efficiency behaviour ------------------------------------------
    def test_identical_models_accept_everything(self):
        tokens, stats = self.run_decode(TARGET, TARGET, [1], 40, 4,
                                        vocab_size=VOCAB, rng=random.Random(7))
        self.assertEqual(len(tokens), 40)
        self.assertEqual(stats.accepted, stats.drafted)
        self.assertAlmostEqual(stats.acceptance_rate, 1.0)
        self.assertGreater(stats.tokens_per_round, 4.0)   # about k + 1 per round

    def test_fewer_target_rounds_than_tokens_when_draft_is_good(self):
        _, stats = self.run_decode(TARGET, DRAFT, [1], 200, 4,
                                   vocab_size=VOCAB, rng=random.Random(3))
        self.assertLess(stats.rounds, 200)
        self.assertGreater(stats.acceptance_rate, 0.2)

    # ---- length and termination ------------------------------------------
    def test_never_exceeds_budget_and_hits_it_exactly_without_eos(self):
        for budget in (0, 1, 2, 3, 5, 17):
            for k in (1, 2, 4, 9):
                tokens, stats = self.run_decode(TARGET, DRAFT, [1], budget, k,
                                                vocab_size=VOCAB, rng=random.Random(budget * 31 + k))
                self.assertEqual(len(tokens), budget, f"budget={budget}, k={k}")
                self.assertEqual(stats.generated, budget)

    def test_eos_terminates_and_is_last_token(self):
        eos = 3
        for seed in range(200):
            tokens, _ = self.run_decode(TARGET, DRAFT, [1], 50, 4, vocab_size=VOCAB,
                                        eos_token=eos, rng=random.Random(seed))
            if eos in tokens:
                self.assertEqual(tokens.index(eos), len(tokens) - 1)
            self.assertLessEqual(len(tokens), 50)

    # ---- degenerate distributions ----------------------------------------
    def test_disjoint_support_always_rejects_but_stays_correct(self):
        target = fixed([0.5, 0.5, 0.0, 0.0])
        draft = fixed([0.0, 0.0, 0.5, 0.5])
        tokens, stats = self.run_decode(target, draft, [0], 60, 4,
                                        vocab_size=VOCAB, rng=random.Random(1))
        self.assertTrue(set(tokens) <= {0, 1})
        self.assertEqual(stats.accepted, 0)

    def test_never_emits_zero_probability_tokens(self):
        target = fixed([0.0, 0.7, 0.0, 0.3])
        draft = fixed([0.25, 0.25, 0.25, 0.25])
        for seed in range(50):
            tokens, _ = self.run_decode(target, draft, [0], 30, 4,
                                        vocab_size=VOCAB, rng=random.Random(seed))
            self.assertTrue(set(tokens) <= {1, 3})

    def test_point_mass_models(self):
        tokens, _ = self.run_decode(fixed([0, 1, 0, 0]), fixed([0, 1, 0, 0]), [0], 10, 3,
                                    vocab_size=VOCAB, rng=random.Random(0))
        self.assertEqual(tokens, [1] * 10)

    # ---- interface hygiene -----------------------------------------------
    def test_does_not_mutate_prompt(self):
        prompt = [1, 2, 3]
        self.run_decode(TARGET, DRAFT, prompt, 10, 3, vocab_size=VOCAB, rng=random.Random(0))
        self.assertEqual(prompt, [1, 2, 3])

    def test_deterministic_for_a_given_seed(self):
        a, _ = self.run_decode(TARGET, DRAFT, [1], 30, 4, vocab_size=VOCAB, rng=random.Random(99))
        b, _ = self.run_decode(TARGET, DRAFT, [1], 30, 4, vocab_size=VOCAB, rng=random.Random(99))
        self.assertEqual(a, b)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.run_decode(TARGET, DRAFT, [1], 5, 0, vocab_size=VOCAB)           # k < 1
        with self.assertRaises(ValueError):
            self.run_decode(TARGET, DRAFT, [1], -1, 2, vocab_size=VOCAB)          # negative budget

    def test_rejects_malformed_distributions(self):
        bad_sum = fixed([0.5, 0.5, 0.5, 0.0])
        bad_len = fixed([0.5, 0.5])
        negative = fixed([1.5, -0.5, 0.0, 0.0])
        for bad in (bad_sum, bad_len, negative):
            with self.assertRaises(ValueError):
                self.run_decode(bad, DRAFT, [1], 5, 2, vocab_size=VOCAB, rng=random.Random(0))
            with self.assertRaises(ValueError):
                self.run_decode(TARGET, bad, [1], 5, 2, vocab_size=VOCAB, rng=random.Random(0))


class TestSolution(SpeculativeDecodingTests, unittest.TestCase):
    decode = staticmethod(Solution.speculative_decode)


class TestAlternativeSolution(SpeculativeDecodingTests, unittest.TestCase):
    decode = staticmethod(AlternativeSolution.speculative_decode)

    def test_mutation_resampling_from_target_is_detected(self):
        self.skipTest("mutation test patches Solution internals; covered in TestSolution")


if __name__ == "__main__":
    unittest.main()
