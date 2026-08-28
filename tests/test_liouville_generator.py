"""Tests for the LIOUVILLE integrable-expression generator (tiers 3 and 4).

The generator's whole value is that every ``(f, F)`` pair it emits is correct by
construction *and* checked before emission — a wrong pair silently poisons
training data, benchmark scores and RL rewards alike. These tests pin:

* every emitted pair verifies (``d/dx F == f``), over a decent sample;
* the same seed reproduces the same corpus, byte for byte;
* each tier really exercises the differential-field structure it claims (asserted
  on generator kinds, never on exact strings);
* degenerate parameter draws are rejected as :class:`DegenerateSample` rather
  than escaping as an invalid pair;
* tier 2 differentiates at all (regression: it used to build ``d^(1/2)`` as a
  ``pow`` with a non-integer exponent, which ``ak.diff`` rejects with
  ``E-DIFF-002``, so every tier-2 row raised).

Only ``alkahest`` is needed — no ``verifiers`` / ``datasets`` extra.
"""

from __future__ import annotations

import random

import alkahest as ak
import pytest
from alkahest import ExprPool
from alkahest.rl.envs.integration.corpus import (
    bwd_baseline,
    const_skeleton,
    domain_guards,
    generate_corpus,
    has_canonical_argument_order,
    read_expr,
    summarise,
    verify_pair,
)
from alkahest.rl.envs.integration.env import _make_row
from alkahest.rl.envs.integration.grammar import (
    TIER_STRUCTURE,
    TIERS,
    DegenerateSample,
    GeneratedPair,
    build_tower,
    liouville_pair,
    node_count,
    random_bwd_pair,
    random_elementary,
)

LIOUVILLE_TIERS = (3, 4)
_SEED = 20260824


# ─────────────────────────────────────────────────────────────────────────────
# Correctness of every emitted pair
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_every_generated_pair_verifies(tier):
    """``d/dx F == f`` for every pair that passes the gate, over a fixed-seed sample.

    A *failed* verification is allowed — it means the pair is discarded, which is
    the whole point of the gate — but it must be rare, and no pair may verify
    "by accident" (``ok`` without either route succeeding).
    """
    rng = random.Random(_SEED + tier)
    n_drawn = verified = 0
    for _ in range(40):
        pool = ExprPool()
        try:
            pair = liouville_pair(pool, tier, rng)
        except DegenerateSample:
            continue
        n_drawn += 1
        verification = verify_pair(pair, pool)
        if verification.ok:
            verified += 1
            assert verification.symbolic or verification.numeric, (
                f"tier {tier} pair admitted with no proof at all: {verification}"
            )
        else:
            assert verification.reason, "a rejection must say why"
    assert n_drawn >= 35
    assert verified / n_drawn >= 0.85, (
        f"tier {tier} discard rate too high: {n_drawn - verified}/{n_drawn}"
    )


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_corpus_emits_only_verified_pairs(tier):
    records, stats = generate_corpus(tier, 12, seed=_SEED + tier)
    assert stats.emitted == 12
    assert all(r["verification"]["ok"] for r in records)
    # Every row must be readable back from its serialised form, or the corpus is
    # not reusable outside the process that produced it.
    assert all(r["parse_roundtrip"] for r in records)


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_verifier_rejects_a_deliberately_wrong_pair(tier):
    """The verifier must actually be able to say no."""
    pool = ExprPool()
    rng = random.Random(_SEED)
    pair = liouville_pair(pool, tier, rng)
    x = pool.symbol("x")
    pair.integrand = pair.integrand + x * pool.integer(3) + pool.integer(1)
    assert not verify_pair(pair, pool).ok


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_same_seed_same_pairs(tier):
    def draw():
        rng = random.Random(4242)
        out = []
        for _ in range(6):
            pool = ExprPool()
            pair = liouville_pair(pool, tier, rng)
            out.append((str(pair.integrand), str(pair.integral)))
        return out

    assert draw() == draw()


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_corpus_is_reproducible(tier):
    first, _ = generate_corpus(tier, 6, seed=99)
    second, _ = generate_corpus(tier, 6, seed=99)
    assert [r["integrand"] for r in first] == [r["integrand"] for r in second]
    assert [r["integral"] for r in first] == [r["integral"] for r in second]


# ─────────────────────────────────────────────────────────────────────────────
# Structural claims per tier
# ─────────────────────────────────────────────────────────────────────────────


def _sample(tier, n=30, seed=_SEED):
    """``n`` sampled pairs, each paired with the pool that owns its expressions."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pool = ExprPool()
        try:
            out.append((pool, liouville_pair(pool, tier, rng)))
        except DegenerateSample:
            continue
    return out


def test_tiers_table_covers_the_liouville_tiers():
    for tier in LIOUVILLE_TIERS:
        assert tier in TIERS
        assert TIER_STRUCTURE[tier]


def test_tier3_is_a_single_transcendental_tower():
    pairs = _sample(3)
    assert pairs
    for _pool, p in pairs:
        assert p.params["tower_height"] == 1
        assert p.params["top_generator_kind"] in ("exp", "log")
        # "rational in theta", not "polynomial in theta": there is a real
        # denominator, and it is not square-free whenever there is room for it.
        assert p.params["denominator_degree_in_theta"] >= 1
    kinds = {p.params["top_generator_kind"] for _pool, p in pairs}
    assert kinds == {"exp", "log"}, f"tier 3 should exercise both monomials, got {kinds}"
    assert any(max(p.params["multiplicities"]) > 1 for _pool, p in pairs), (
        "no non-square-free denominator sampled — the functional parameters of the "
        "Liouville ansatz are never exercised"
    )


def test_tier4_is_a_two_generator_tower():
    pairs = _sample(4, n=40)
    assert pairs
    for _pool, p in pairs:
        assert p.params["tower_height"] == 2
        assert len(p.params["tower_kinds"]) == 3
        assert p.params["tower_kinds"][0] == "x"
    tops = {p.params["top_generator_kind"] for _pool, p in pairs}
    assert "sqrt" in tops, "tier 4 never sampled the mixed algebraic+transcendental layer"
    assert tops & {"exp", "log"}, "tier 4 never sampled a transcendental top generator"
    towers = {tuple(p.params["tower_kinds"]) for _pool, p in pairs}
    assert len(towers) >= 3, f"tier 4 tower diversity too low: {towers}"


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_generators_actually_occur_in_the_emitted_pair(tier):
    """A tier is a claim about structure — the generators must survive to the output."""
    for _pool, pair in _sample(tier, n=15):
        for gen in pair.tower.split(" < ")[1:]:
            head = gen.split("(")[0].strip()
            assert head in str(pair.integrand) or head in str(pair.integral)


def test_tier4_can_carry_algebraic_number_coefficients():
    """risch.md Gap E: Q(sqrt(d)) constants in the logarithmic part."""
    pairs = _sample(4, n=60, seed=11)
    assert any("sqrt(" in str(p.integral) for _pool, p in pairs)


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_pairs_carry_domain_guards(tier):
    for _pool, pair in _sample(tier, n=8):
        guards = domain_guards(pair.integral, pair.integrand)
        assert guards, "an expression with logs/denominators must produce guards"
        assert all(kind in ("positive", "nonzero") for _e, kind in guards)


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate parameter handling
# ─────────────────────────────────────────────────────────────────────────────


def test_exhausted_attempts_raise_degenerate_sample(monkeypatch):
    from alkahest.rl.envs.integration import grammar

    def always_degenerate(*_args, **_kwargs):
        raise DegenerateSample("forced")

    monkeypatch.setattr(grammar, "_liouville_once", always_degenerate)
    with pytest.raises(DegenerateSample):
        liouville_pair(ExprPool(), 3, random.Random(0), max_attempts=3)


def test_generate_corpus_reports_rather_than_hangs(monkeypatch):
    """A generator that can never succeed must return short, not spin forever."""
    from alkahest.rl.envs.integration import grammar

    monkeypatch.setattr(
        grammar,
        "_liouville_once",
        lambda *_a, **_k: (_ for _ in ()).throw(DegenerateSample("forced")),
    )
    records, stats = generate_corpus(3, 5, seed=1, max_draws_per_pair=2)
    assert records == []
    assert stats.emitted == 0
    assert stats.generator_failures > 0


def test_build_tower_rejects_non_liouville_tiers():
    with pytest.raises(ValueError):
        build_tower(ExprPool(), 1, random.Random(0))


def test_unknown_tier_still_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        random_elementary(ExprPool(), 5, random.Random(0))


# ─────────────────────────────────────────────────────────────────────────────
# The RL entry point keeps working for every tier
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_random_elementary_differentiates(tier):
    """Regression: tier 2 used to raise E-DIFF-002 on ``d^(1/2)``."""
    rng = random.Random(_SEED + tier)
    for _ in range(4):
        pool = ExprPool()
        x = pool.symbol("x")
        integral = random_elementary(pool, tier, rng)
        integrand = ak.simplify(ak.diff(integral, x).value).value
        assert node_count(integrand) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers used by the quality report
# ─────────────────────────────────────────────────────────────────────────────


def test_node_count_matches_hand_count():
    pool = ExprPool()
    x = pool.symbol("x")
    # add(mul(2, x), 1) -> add, mul, 2, x, 1
    expr = pool.add([pool.mul([pool.integer(2), x]), pool.integer(1)])
    assert node_count(expr) == 5
    assert node_count(x) == 1


def test_const_skeleton_collapses_coefficients_but_not_structure():
    pool = ExprPool()
    x = pool.symbol("x")
    a = pool.integer(2) * ak.log(x + pool.integer(3))
    b = pool.integer(7) * ak.log(x + pool.integer(5))
    c = pool.integer(2) * ak.exp(x + pool.integer(3))
    assert const_skeleton(a) == const_skeleton(b)
    assert const_skeleton(a) != const_skeleton(c)


def test_skeleton_is_argument_order_insensitive():
    pool = ExprPool()
    x = pool.symbol("x")
    lhs = pool.add([x, ak.log(x), pool.integer(2)])
    rhs = pool.add([pool.integer(2), ak.log(x), x])
    assert const_skeleton(lhs) == const_skeleton(rhs)


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_no_term_ordering_artifact(tier):
    """Davis (arXiv:1912.05752): BWD data carries the differentiator's term order.

    It cannot here — ``ExprPool.add``/``mul`` sort their children canonically — so
    permuting every argument list must be a no-op.
    """
    rng = random.Random(5)
    for pool, pair in _sample(tier, n=8):
        # NB the pair's *own* pool: rebuilding an Expr in a foreign pool panics
        # (``ExprPool: ExprId out of range``) rather than raising.
        assert has_canonical_argument_order(pool, pair.integrand, rng)


def test_bwd_baseline_is_generated_and_verified():
    records = bwd_baseline(6, seed=3)
    assert len(records) == 6
    assert all(r["method"] == "BWD" for r in records)
    stats = summarise(records)
    assert stats["n"] == 6
    assert stats["integrand_nodes"]["median"] >= 1


def test_random_bwd_pair_is_a_real_pair():
    pool = ExprPool()
    pair = random_bwd_pair(pool, random.Random(0))
    assert verify_pair(pair, pool).symbolic or verify_pair(pair, pool).numeric


# ─────────────────────────────────────────────────────────────────────────────
# Findings the generator turned up in the integrator itself
# ─────────────────────────────────────────────────────────────────────────────

#: Nested-exponential integrands the LIOUVILLE generator produced, each with a
#: machine-verified elementary antiderivative, on which ``integrate`` returns a
#: *certified* ``E-INT-004`` ("no elementary antiderivative exists"). That is a
#: false NonElementary — the silent-error class, not a mere decline. It
#: reproduces on stock ``main`` (a1f26bf) as well as on the routed build, and
#: hits about 2.3% of verified tier-4 nested-exp pairs (7/300).
# Integrands that the nested-exp path once falsely certified as E-INT-004.
# The pole-order argument ruled out a *rational* solution and concluded
# non-elementary without examining the logarithmic part, so Liouville's
# `v + Σcᵢlog(uᵢ)` was only half-tested. Fixed; kept as a regression guard.
FORMERLY_FALSE_NONELEMENTARY = [
    ("exp(x)*exp(exp(x))/(exp(exp(x))+1)", "log(exp(exp(x))+1)"),
    ("exp(x)*exp(exp(x))/(exp(exp(x))+1)^2", "-1/(exp(exp(x))+1)"),
    ("-3*exp(x)*exp(exp(x))/(1+exp(exp(x)))^2", "3/(1+exp(exp(x)))"),
    ("-4*exp(x)*exp(exp(x))/(exp(exp(x))-2)^2", "4/(exp(exp(x))-2)"),
]


@pytest.mark.parametrize(("f_str", "big_f_str"), FORMERLY_FALSE_NONELEMENTARY)
def test_false_nonelementary_witnesses_are_really_elementary(f_str, big_f_str):
    """The witness pairs are correct, whatever ``integrate`` says about them."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = read_expr(f_str, pool)
    big_f = read_expr(big_f_str, pool)
    residual = ak.simplify(ak.diff(big_f, x).value - f).value
    for point in (0.2, 0.8, 1.3):
        assert abs(ak.eval_expr(residual, {x: point})) < 1e-9


@pytest.mark.parametrize(("f_str", "big_f_str"), FORMERLY_FALSE_NONELEMENTARY)
def test_integrate_does_not_falsely_certify_nonelementary(f_str, big_f_str):
    del big_f_str
    pool = ExprPool()
    x = pool.symbol("x")
    f = read_expr(f_str, pool)
    code = None
    try:
        with ak.context(pool=pool, budget=ak.Budget(wall_ms=5000)):
            ak.integrate(f, x)
    except ak.IntegrationError as exc:
        code = getattr(exc, "code", "")
    assert code != "E-INT-004", f"false NonElementary on {f_str}"


def test_read_expr_yields_a_differentiable_expression():
    """``parse`` alone does not: ``(x+1)^-1`` comes back with an unfolded exponent.

    ``parse("(x+1)^-1")`` builds ``pow((x+1), mul(1, -1))``; ``diff`` then raises
    ``E-DIFF-002``. Every integrand in the corpus prints with ``^-1``, so a reader
    has to fold it — which is what ``read_expr`` does.
    """
    pool = ExprPool()
    x = pool.symbol("x")
    raw = ak.parse("(x+1)^-1", pool, {"x": x})
    with pytest.raises(ak.DiffError):
        ak.diff(raw, x)
    folded = read_expr("(x+1)^-1", pool)
    assert ak.diff(folded, x).value is not None


# ─────────────────────────────────────────────────────────────────────────────
# The dataset builder must not emit an unverified row
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tier", LIOUVILLE_TIERS)
def test_dataset_rows_carry_a_verified_antiderivative(tier):
    """``_make_row`` gates tier 3/4 rows through the verifier before emitting.

    The row is re-read from its *serialised* form and re-verified, so this also
    covers the string round-trip the dataset actually ships.
    """
    rng = random.Random(_SEED)
    for _ in range(5):
        row = _make_row(tier, False, rng)
        assert row["is_elementary"] is True
        assert row["tier"] == tier
        pool = ExprPool()
        pair = GeneratedPair(
            integrand=read_expr(row["f_str"], pool),
            integral=read_expr(row["F_str"], pool),
            tier=tier,
            method="LIOUVILLE",
            tower="",
        )
        assert verify_pair(pair, pool).ok, f"dataset row does not verify: {row['f_str']}"


def test_dataset_hard_negative_rows_are_unchanged():
    """The curated non-elementary path must keep its shape (no ``F_str``)."""
    row = _make_row(3, True, random.Random(1))
    assert row["is_elementary"] is False
    assert "F_str" not in row


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_dataset_rows_generate_for_every_tier(tier):
    row = _make_row(tier, False, random.Random(3 + tier))
    assert row["f_str"]
    assert row["prompt"][0]["content"].startswith("Find ∫")
