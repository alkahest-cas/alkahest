"""V1-15: E-graph default rule completeness.

Verifies that:
1. simplify_egraph reduces sin(x)**2 + cos(x)**2 → 1 by default.
2. simplify_egraph does **not** reduce exp(log(x)) → x (needs positivity).
3. simplify_egraph reduces log(exp(x)) → x by default.
4. Existing egraph proptests are not broken (basic identities still hold).
5. Opt-out: EgraphConfig(include_trig_rules=False) does NOT apply trig rules.

Trig / log-exp tests are skipped when the `egraph` Cargo feature is absent
(HAS_EGRAPH == False), because simplify_egraph falls back to the rule-based
engine which does not carry those identities.
"""

from __future__ import annotations

import alkahest
import pytest
from alkahest import (
    EgraphConfig,
    ExprPool,
    cos,
    exp,
    log,
    simplify_egraph,
    simplify_egraph_with,
    sin,
)

needs_egraph = pytest.mark.skipif(
    not alkahest.HAS_EGRAPH,
    reason="egraph Cargo feature not enabled in this build",
)


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


class TestEgraphTrigIdentity:
    @needs_egraph
    def test_sin2_plus_cos2_reduces_to_one(self, pool, x):
        expr = sin(x) ** 2 + cos(x) ** 2
        result = simplify_egraph(expr)
        assert str(result.value) == "1"

    @needs_egraph
    def test_cos2_plus_sin2_reduces_to_one(self, pool, x):
        expr = cos(x) ** 2 + sin(x) ** 2
        result = simplify_egraph(expr)
        assert str(result.value) == "1"


class TestEgraphLogExpCancellation:
    @needs_egraph
    def test_exp_of_log_stays_without_assumptions(self, pool, x):
        expr = exp(log(x))
        result = simplify_egraph(expr)
        assert str(result.value) == "exp(log(x))"

    @needs_egraph
    def test_log_of_exp(self, pool, x):
        expr = log(exp(x))
        result = simplify_egraph(expr)
        assert str(result.value) == "x"


class TestEgraphExistingRulesUnaffected:
    def test_add_zero(self, pool, x):
        expr = x + pool.integer(0)
        result = simplify_egraph(expr)
        assert str(result.value) == "x"

    def test_mul_one(self, pool, x):
        expr = x * pool.integer(1)
        result = simplify_egraph(expr)
        assert str(result.value) == "x"

    def test_mul_zero(self, pool, x):
        expr = x * pool.integer(0)
        result = simplify_egraph(expr)
        assert str(result.value) == "0"

    def test_const_fold(self, pool):
        expr = pool.integer(3) + pool.integer(4)
        result = simplify_egraph(expr)
        assert str(result.value) == "7"


class TestEgraphConfig:
    def test_default_config_has_trig_enabled(self):
        cfg = EgraphConfig()
        assert cfg.include_trig_rules is True

    def test_default_config_has_log_exp_enabled(self):
        cfg = EgraphConfig()
        assert cfg.include_log_exp_rules is True

    @needs_egraph
    def test_opt_out_trig_rules_does_not_simplify(self, pool, x):
        cfg = EgraphConfig(include_trig_rules=False)
        expr = sin(x) ** 2 + cos(x) ** 2
        result = simplify_egraph_with(expr, cfg)
        assert str(result.value) != "1"

    @needs_egraph
    def test_opt_out_log_exp_rules_does_not_simplify(self, pool, x):
        cfg = EgraphConfig(include_log_exp_rules=False)
        expr = log(exp(x))
        result = simplify_egraph_with(expr, cfg)
        assert str(result.value) == "log(exp(x))"

    def test_custom_node_limit_attr(self):
        cfg = EgraphConfig(node_limit=50_000)
        assert cfg.node_limit == 50_000

    def test_simplify_egraph_with_default_config_still_works(self, pool, x):
        cfg = EgraphConfig()
        expr = x + pool.integer(0)
        result = simplify_egraph_with(expr, cfg)
        assert str(result.value) == "x"
