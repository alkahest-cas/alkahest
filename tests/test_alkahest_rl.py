"""Tests for alkahest.rl (core + integration verifier; verifiers env optional)."""

from __future__ import annotations

import random

import alkahest as ak
import pytest
from alkahest import ExprPool, diff, simplify, unicode_str
from alkahest.rl.core.curriculum import CurriculumScheduler
from alkahest.rl.core.rubric import Rubric
from alkahest.rl.envs.integration.env import _make_row, build_dataset
from alkahest.rl.envs.integration.grammar import random_elementary
from alkahest.rl.envs.integration.verifier import IntegrationVerifier, _model_declined


class TestCurriculumScheduler:
    def test_starts_at_tier_zero(self):
        c = CurriculumScheduler(n_tiers=3)
        assert c.current_tier == 0

    def test_advances_on_high_pass_rate(self):
        c = CurriculumScheduler(n_tiers=3, advance_threshold=0.7, window=8)
        for _ in range(4):
            c.record(1.0)
        assert c.current_tier == 1


class TestIntegrationGrammar:
    def test_tier_zero_produces_diffable_expr(self):
        pool = ExprPool()
        rng = random.Random(0)
        f_antideriv = random_elementary(pool, 0, rng)
        x = pool.symbol("x")
        f = simplify(diff(f_antideriv, x).value).value
        assert f is not None
        assert len(unicode_str(f)) > 0


class TestIntegrationVerifier:
    def test_correct_polynomial_antiderivative(self):
        pool = ExprPool()
        x = pool.symbol("x")
        f = pool.integer(2) * x
        verifier = IntegrationVerifier()
        reward = verifier.verify(
            "x^2",
            {"f_expr": f, "is_elementary": True, "pool": pool},
        )
        assert reward == 1.0

    def test_decline_nonelementary(self):
        pool = ExprPool()
        x = pool.symbol("x")
        f = ak.exp(x * x)
        verifier = IntegrationVerifier()
        reward = verifier.verify(
            "no elementary form",
            {"f_expr": f, "is_elementary": False, "pool": pool},
        )
        assert reward == 1.0

    def test_unnecessary_decline_on_elementary(self):
        pool = ExprPool()
        x = pool.symbol("x")
        f = pool.integer(2) * x
        verifier = IntegrationVerifier()
        reward = verifier.verify(
            "no elementary form",
            {"f_expr": f, "is_elementary": True, "pool": pool},
        )
        assert reward == pytest.approx(-0.2)


class TestModelDeclined:
    def test_detects_markers(self):
        assert _model_declined("This has no elementary closed form.")
        assert not _model_declined("x^2 + 1")


class TestBuildDataset:
    def test_small_dataset_has_expected_keys(self):
        pytest.importorskip("datasets")
        ds = build_dataset(tier=0, neg_frac=0.0, n=3, seed=1)
        row = ds[0]
        assert "prompt" in row
        assert "f_str" in row
        assert row["is_elementary"] is True

    def test_make_row_elementary(self):
        rng = random.Random(7)
        row = _make_row(0, False, rng)
        assert row["is_elementary"] is True
        assert row["tier"] == 0
        assert "f_str" in row


class TestRubric:
    def test_weighted_score(self):
        def r_a(**_):
            return 1.0

        def r_b(**_):
            return 0.0

        rubric = Rubric(funcs=[r_a, r_b], weights=[1.0, 1.0])
        assert rubric.score() == 0.5


def test_load_environment_smoke():
    pytest.importorskip("verifiers")
    from alkahest.rl.envs.integration.env import load_environment

    env = load_environment(
        difficulty_tier=0,
        n_train=4,
        n_eval=2,
        hard_negative_fraction=0.0,
        adaptive=True,
    )
    assert env is not None
    assert hasattr(env, "curriculum")
    assert env.curriculum.n_tiers == 5
