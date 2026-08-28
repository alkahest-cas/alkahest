"""Random *integrable* expression generators, graded by differential-field structure.

Tiers 0–2 are the original small hand-written grammars: sample an antiderivative
``F`` directly and let the caller differentiate it (a plain BWD generator in the
sense of Lample & Charton, ICLR 2020).

Tiers 3–4 implement the **LIOUVILLE** generator of Barket, England & Gerhard
(*The Liouville Generator for Producing Integrable Expressions*, CASC 2024,
`arXiv:2406.11631 <https://arxiv.org/abs/2406.11631>`_), which is modelled on the
Parallel Risch (Risch–Norman) algorithm run *backwards*.  Liouville's theorem says
that if ``f`` in a differential field ``D`` has an elementary antiderivative then

.. math::

    f = v_0' + \\sum_i c_i \\frac{v_i'}{v_i}
    \\quad\\Longrightarrow\\quad
    F = v_0 + \\sum_i c_i \\log(v_i),

with ``v_i in D`` and ``c_i`` constants.  The generator therefore *starts* from
that shape:

1. build a tower ``Q(theta_0=x, theta_1, ..., theta_n)`` of monomial extensions;
2. sample ``q_1, ..., q_r`` in ``theta_n`` with coefficients in the lower field and
   set ``D = prod q_i^{m_i}`` — deliberately **not** square-free, which is what
   turns the free parameters of the ansatz from constants into *functions*;
3. sample the rational part directly in partial fraction form,
   ``F_hat = sum_i sum_{k<=m_i} n_{ik} / q_i^k`` — normalising it recovers the
   paper's ``N/D`` with ``deg N < deg D`` without needing a square-free
   factorisation over the tower, which the kernel does not have;
4. ``A = sum_i c_i log(a_i)`` with the ``a_i`` drawn from the factors ``{q_i}`` of
   ``D`` (these produce degree-1 denominators in the partial-fraction form of the
   integrand — the thing BWD essentially never produces);
5. ``B = sum_i d_i log(b_i)`` with the ``b_i`` fresh polynomials of the field;
6. emit ``F = Normalise(F_hat + A) + B`` and ``f = Normalise(F_hat' + A') + B'``.

Normalising *both* sides is what balances integrand and integral length (§5.4 of
the paper); it is the fix for the BWD length bias.

Nothing here verifies the emitted pair — that is
:mod:`alkahest.rl.envs.integration.corpus`, and no pair should reach a dataset
without going through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import alkahest as ak
from alkahest import Expr, ExprPool

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

TIERS = {
    0: "rational",
    1: "exp_log",
    2: "algebraic_coeff",
    3: "rational_exp",
    4: "nested_tower",
}

#: What each tier is *for*, in terms of the Risch register in
#: ``temp-alkahest/planning/risch.md``.  The tier number is a statement about
#: which differential-field structure the integrand exercises, not about size.
TIER_STRUCTURE = {
    0: "Q(x) — pure rational, base case of the Risch recursion.",
    1: "Q(x)(theta), theta = exp(u) or log(u) — one monomial extension, polynomial part only.",
    2: "Q(sqrt(d))(x)(theta) — algebraic-number constant coefficients (risch.md Gap E).",
    3: (
        "Q(x)(theta), theta = exp(u) or log(u), *rational* in theta: a non-square-free "
        "denominator D = prod q_i^{m_i} in theta with Q(x) coefficients, plus a "
        "Liouvillian log part. Exercises Hermite reduction and the Rothstein-Trager "
        "residue/log part on a transcendental monomial (risch.md exp case / primitive case)."
    ),
    4: (
        "Q(x)(theta_1)(theta_2) — two generators, theta_2 built over theta_1: nested "
        "exp/log, log-over-exp, an independent second monomial, or an algebraic "
        "sqrt layer over a transcendental (risch.md Gap B tower composition, Gap C "
        "mixed algebraic+transcendental), optionally with Q(sqrt(d)) coefficients (Gap E)."
    ),
}

# Rational exponents u^(p/q) with q > 2 are deliberately absent: ``ak.diff`` raises
# E-DIFF-002 ("cannot differentiate power with non-integer exponent") and
# ``ak.diff_forward`` raises E-DIFF-004, so no pair with a cube root can be built or
# verified from Python today.  The q = 2 slice is reachable through the ``sqrt``
# primitive and is what tier 4's algebraic recipes use.

__all__ = [
    "TIERS",
    "TIER_STRUCTURE",
    "DegenerateSample",
    "GeneratedPair",
    "Tower",
    "build_tower",
    "liouville_pair",
    "node_count",
    "random_bwd_pair",
    "random_elementary",
]


class DegenerateSample(ValueError):
    """A sampled parameter set collapsed (constant integrand, zero denominator, ...).

    Raised internally and retried; it never escapes :func:`liouville_pair`.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Size measurement
# ─────────────────────────────────────────────────────────────────────────────


def node_count(expr: Expr) -> int:
    """Number of nodes in the *tree* expansion of ``expr``.

    This is the prefix-notation token count Barket et al. plot, and the closest
    available analogue of SymPy's ``count_ops``.  The kernel exposes no
    ``leaf_count``/``node_count`` on :class:`~alkahest.Expr`, so we walk
    :meth:`Expr.node` ourselves.  Shared subexpressions are counted once per
    occurrence (the DAG is expanded), which is what a tokenizer would see.
    """
    total = 0
    stack: list[object] = [expr]
    while stack:
        item = stack.pop()
        if isinstance(item, Expr):
            total += 1
            stack.extend(item.node()[1:])
        elif isinstance(item, list):
            stack.extend(item)
        # bare strings (tags, symbol names, integer literals) belong to the node
        # already counted above
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Differential towers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Tower:
    """A differential tower ``Q(theta_0=x, theta_1, ..., theta_n)``.

    Attributes:
        gens: the generators, ``gens[0]`` is always ``x``.
        kinds: parallel list of ``"x" | "exp" | "log" | "sqrt"`` labels.
        guards: ``(expr, "positive" | "nonzero")`` conditions that must hold for
            the tower (and hence the generated pair) to be real-valued.  Used by
            the corpus verifier to pick in-domain sample points.
        description: human-readable tower, e.g. ``"x < exp(x) < log(exp(x) + 2)"``.
    """

    gens: list[Expr]
    kinds: list[str]
    guards: list[tuple[Expr, str]] = field(default_factory=list)
    description: str = ""

    @property
    def top(self) -> Expr:
        return self.gens[-1]

    @property
    def lower(self) -> list[Expr]:
        return self.gens[:-1]

    @property
    def top_is_algebraic(self) -> bool:
        return self.kinds[-1] == "sqrt"


def _sum(pool: ExprPool, terms: list[Expr]) -> Expr:
    """``pool.add`` that does not wrap a single term in a 1-ary ``add`` node.

    ``ExprPool.add([t])`` builds a genuine one-child ``add``, and ``simplify``
    does not flatten it — so ``str(e)`` prints ``(t)`` and ``parse(str(e))``
    comes back as a *different* expression.  A corpus that cannot be read back
    is useless, so every sum here goes through this.
    """
    if not terms:
        raise DegenerateSample("empty sum")
    return terms[0] if len(terms) == 1 else pool.add(terms)


def _prod(pool: ExprPool, factors: list[Expr]) -> Expr:
    """``pool.mul`` with the same 1-ary guard as :func:`_sum`."""
    if not factors:
        raise DegenerateSample("empty product")
    return factors[0] if len(factors) == 1 else pool.mul(factors)


def _small_int(rng: Random, lo: int = -4, hi: int = 4, *, nonzero: bool = True) -> int:
    while True:
        v = rng.randint(lo, hi)
        if v or not nonzero:
            return v


def _pos_int(rng: Random, lo: int = 1, hi: int = 5) -> int:
    return rng.randint(lo, hi)


def _log_safe_inner(pool: ExprPool, x: Expr, rng: Random) -> tuple[Expr, list[tuple[Expr, str]]]:
    """An argument for ``log`` that is positive on ``x > 0``."""
    which = rng.random()
    if which < 0.4:
        return x, [(x, "positive")]
    if which < 0.7:
        c = pool.integer(_pos_int(rng, 1, 4))
        return x + c, [(x + c, "positive")]
    c = pool.integer(_pos_int(rng, 1, 4))
    inner = x * x + c
    return inner, [(inner, "positive")]


def _exp_inner(pool: ExprPool, x: Expr, rng: Random) -> tuple[Expr, list[tuple[Expr, str]]]:
    which = rng.random()
    if which < 0.45:
        return x, []
    if which < 0.7:
        return pool.integer(_small_int(rng, -3, 3)) * x, []
    if which < 0.9:
        return x * x, []
    return x / (x + pool.integer(1)), [(x + pool.integer(1), "nonzero")]


def _single_transcendental(pool: ExprPool, x: Expr, rng: Random) -> Tower:
    """Tier 3 tower: ``Q(x)(theta)`` with one exp or log monomial."""
    if rng.random() < 0.5:
        inner, guards = _exp_inner(pool, x, rng)
        theta = ak.exp(inner)
        return Tower([x, theta], ["x", "exp"], guards, f"x < exp({inner})")
    inner, guards = _log_safe_inner(pool, x, rng)
    theta = ak.log(inner)
    # log(u) vanishes where u == 1; the denominators built on it must not.
    return Tower([x, theta], ["x", "log"], guards, f"x < log({inner})")


def _nested_tower(pool: ExprPool, x: Expr, rng: Random) -> Tower:
    """Tier 4 tower: two generators, the second built over the first."""
    one = pool.integer(1)
    recipes: list[Callable[[], Tower]] = []

    def log_over_exp() -> Tower:
        inner, g = _exp_inner(pool, x, rng)
        t1 = ak.exp(inner)
        c = pool.integer(_pos_int(rng, 1, 3))
        arg = t1 + c
        t2 = ak.log(arg)
        return Tower(
            [x, t1, t2],
            ["x", "exp", "log"],
            [*g, (arg, "positive")],
            f"x < exp({inner}) < log(exp({inner}) + {c})",
        )

    def nested_exp() -> Tower:
        t1 = ak.exp(x)
        t2 = ak.exp(t1)
        return Tower([x, t1, t2], ["x", "exp", "exp"], [], "x < exp(x) < exp(exp(x))")

    def nested_log() -> Tower:
        inner, g = _log_safe_inner(pool, x, rng)
        t1 = ak.log(inner)
        c = pool.integer(_pos_int(rng, 2, 4))
        arg = t1 + c
        t2 = ak.log(arg)
        return Tower(
            [x, t1, t2],
            ["x", "log", "log"],
            [*g, (arg, "positive")],
            f"x < log({inner}) < log(log({inner}) + {c})",
        )

    def exp_of_x_log_x() -> Tower:
        t1 = ak.log(x)
        t2 = ak.exp(x * t1)  # x^x
        return Tower(
            [x, t1, t2], ["x", "log", "exp"], [(x, "positive")], "x < log(x) < exp(x*log(x))"
        )

    def independent_pair() -> Tower:
        t1 = ak.exp(x)
        t2 = ak.log(x)
        return Tower(
            [x, t1, t2],
            ["x", "exp", "log"],
            [(x, "positive")],
            "x < exp(x) < log(x)  (algebraically independent)",
        )

    def sqrt_over_exp() -> Tower:
        t1 = ak.exp(x)
        c = pool.integer(_pos_int(rng, 1, 4))
        rad = x + c
        t2 = ak.sqrt(rad)
        return Tower(
            [x, t1, t2], ["x", "exp", "sqrt"], [(rad, "positive")], f"x < exp(x) < sqrt(x + {c})"
        )

    def sqrt_over_log() -> Tower:
        t1 = ak.log(x)
        c = pool.integer(_pos_int(rng, 1, 4))
        rad = x * x + c
        t2 = ak.sqrt(rad)
        return Tower(
            [x, t1, t2],
            ["x", "log", "sqrt"],
            [(x, "positive"), (rad, "positive")],
            f"x < log(x) < sqrt(x^2 + {c})",
        )

    del one
    recipes = [
        log_over_exp,
        nested_exp,
        nested_log,
        exp_of_x_log_x,
        independent_pair,
        sqrt_over_exp,
        sqrt_over_log,
    ]
    weights = [0.22, 0.10, 0.14, 0.10, 0.18, 0.13, 0.13]
    return _weighted_choice(rng, recipes, weights)()


def _weighted_choice(rng: Random, items: Sequence, weights: Sequence[float]):
    total = math.fsum(weights)
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def build_tower(pool: ExprPool, tier: int, rng: Random) -> Tower:
    """Sample the differential tower a tier's integrands live in."""
    x = pool.symbol("x")
    if tier == 3:
        return _single_transcendental(pool, x, rng)
    if tier == 4:
        return _nested_tower(pool, x, rng)
    msg = f"build_tower is only defined for the LIOUVILLE tiers (3, 4), got {tier}"
    raise ValueError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# LIOUVILLE generator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GeneratedPair:
    """One ``(integrand, integral)`` pair plus everything needed to check it."""

    integrand: Expr
    integral: Expr
    tier: int
    method: str
    tower: str
    guards: list[tuple[Expr, str]] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _lower_field_element(
    pool: ExprPool,
    tower: Tower,
    rng: Random,
    *,
    allow_gens: bool = True,
) -> Expr:
    """A small element of ``Q(theta_0, ..., theta_{n-1})``.

    Barket et al. §4.2: *"the coefficient of each q_i should usually have one and
    very rarely more than one of theta_0, ..., theta_{n-1}"*.
    """
    x = tower.gens[0]
    lower_nontrivial = [g for g, k in zip(tower.lower, tower.kinds) if k != "x"]
    c = pool.integer(_small_int(rng))
    r = rng.random()
    if r < 0.45 or not allow_gens:
        return c
    if r < 0.70:
        return c * x ** pool.integer(rng.randint(1, 2))
    if r < 0.92 and lower_nontrivial:
        return c * rng.choice(lower_nontrivial)
    if lower_nontrivial and rng.random() < 0.5:
        # the "very rarely more than one" case
        return c * x * rng.choice(lower_nontrivial)
    return c * x


def _denominator_factors(
    pool: ExprPool,
    tower: Tower,
    rng: Random,
    r_max: int,
) -> list[tuple[Expr, int]]:
    """Sample ``[(q_i, m_i)]`` with the ``q_i`` pairwise coprime and square-free.

    Barket et al. Step 1–2 generate ``q_1 ... q_r`` and then run a square-free
    factorisation.  We construct the already-factored form directly — monic
    factors that are linear in ``theta_n`` with *distinct* trailing coefficients
    are pairwise coprime and square-free by construction, so the factorisation is
    a no-op and we never need a square-free algorithm over a tower.

    At least one multiplicity is > 1 whenever ``r_max > 1``: a non-square-free
    denominator is exactly what makes the free parameters of the Liouville ansatz
    *functional* rather than constant (§2.2 of the paper), and functional
    parameters are what let us dial the length balance.
    """
    theta = tower.top
    r = rng.randint(1, r_max)
    factors: list[tuple[Expr, int]] = []
    used_shifts: list[str] = []
    quadratic_allowed = not tower.top_is_algebraic
    for _ in range(r):
        for _attempt in range(12):
            if quadratic_allowed and rng.random() < 0.15:
                c = pool.integer(_pos_int(rng, 1, 4))
                q = theta ** pool.integer(2) + c
                key = f"quad:{c}"
            else:
                shift = _lower_field_element(pool, tower, rng)
                q = theta + shift
                key = f"lin:{shift}"
            if key not in used_shifts:
                used_shifts.append(key)
                factors.append((q, 1))
                break
        else:  # pragma: no cover - exhausted distinct shifts
            raise DegenerateSample("could not sample distinct denominator factors")
    if not factors:  # pragma: no cover - r >= 1
        raise DegenerateSample("empty denominator")

    # Multiplicities: the paper uses q_1^1 q_2^2 ... q_r^r; we randomise but keep
    # at least one repeated factor when there is room for one.
    mults = [rng.randint(1, min(3, r_max)) for _ in factors]
    if r_max > 1 and max(mults) == 1:
        mults[rng.randrange(len(mults))] = 2
    return [(q, m) for (q, _), m in zip(factors, mults)]


def _factor_degree(q: Expr) -> int:
    """Degree of a sampled denominator factor in ``theta_n`` (it is linear or quadratic)."""
    node = q.node()
    if node[0] == "add":
        for child in node[1]:
            if isinstance(child, Expr) and child.node()[0] == "pow":
                return 2
    return 1


def _degree_in_top(factors: Sequence[tuple[Expr, int]]) -> int:
    """Degree of ``prod q_i^{m_i}`` in ``theta_n``."""
    return sum(_factor_degree(q) * m for q, m in factors)


def _numerator_over(
    pool: ExprPool,
    tower: Tower,
    rng: Random,
    factor_degree: int,
) -> Expr:
    """A numerator for one partial fraction: degree ``< factor_degree`` in ``theta_n``.

    Barket et al. §4.2: *"most of the numerators created should be of lower
    degree in theta_n than the denominator"* — a numerator of degree >= deg(D)
    splits off a polynomial part, which is exactly the BWD-flavoured piece we are
    trying to under-represent.  Generating the rational part already in partial
    fraction form (Algorithm 2, Step 10) makes that automatic, and the
    coefficients here *are* the "functional parameters" of §2.2: they range over
    the whole lower field, so their size is a free knob on the length balance.
    """
    theta = tower.top
    terms: list[Expr] = []
    for i in range(factor_degree):
        if i and rng.random() < 0.4:
            continue
        coeff = _lower_field_element(pool, tower, rng)
        terms.append(coeff if i == 0 else coeff * theta ** pool.integer(i))
    if not terms:
        terms.append(_lower_field_element(pool, tower, rng, allow_gens=False))
    return _sum(pool, terms)


def _constant(pool: ExprPool, tier: int, rng: Random) -> Expr:
    """A constant ``c_i`` for the log part, in the constant field."""
    if tier >= 4 and rng.random() < 0.18:
        # Gap E: algebraic-number coefficients, Q(sqrt(d)).
        d = rng.choice([2, 3, 5, 6, 7])
        return pool.integer(_small_int(rng, -3, 3)) * ak.sqrt(pool.integer(d))
    if rng.random() < 0.2:
        num = _small_int(rng, -5, 5)
        den = _pos_int(rng, 2, 4)
        return pool.rational(num, den)
    return pool.integer(_small_int(rng, -4, 4))


def _fresh_log_argument(pool: ExprPool, tower: Tower, rng: Random) -> Expr:
    """A ``b_i`` for the *B* part: a field element that is **not** a factor of D."""
    x = tower.gens[0]
    theta = tower.top
    r = rng.random()
    if r < 0.35:
        c = pool.integer(_pos_int(rng, 1, 5))
        return x + c
    if r < 0.6:
        c = pool.integer(_pos_int(rng, 1, 5))
        return x * x + c
    if r < 0.85:
        c = pool.integer(_pos_int(rng, 1, 5))
        return theta ** pool.integer(2) + c
    coeff = pool.integer(_pos_int(rng, 1, 3))
    c = pool.integer(_pos_int(rng, 1, 4))
    return coeff * theta + c


def _pow_expr(pool: ExprPool, base: Expr, exponent: int) -> Expr:
    return base ** pool.integer(exponent)


def _safe(fn: Callable[[], Expr], fallback: Expr) -> Expr:
    try:
        return fn()
    except Exception:
        return fallback


def _normalise(expr: Expr) -> Expr:
    """``Normalise`` of Algorithm 2 — put over a common denominator, reduced.

    ``together`` already cancels the gcd, so a separate ``cancel`` is only a
    belt-and-braces pass; both are wrapped because either can decline on a tower
    the rational-function layer does not recognise (a sqrt generator, mostly), in
    which case we keep the un-normalised form rather than fail the sample.
    """
    out = _safe(lambda: ak.simplify(expr).value, expr)
    out = _safe(lambda: ak.together(out), out)
    out = _safe(lambda: ak.cancel(out), out)
    return _safe(lambda: ak.simplify(out).value, out)


def _simplified(expr: Expr) -> Expr:
    return _safe(lambda: ak.simplify(expr).value, expr)


def _diff(expr: Expr, x: Expr) -> Expr:
    return ak.diff(expr, x).value


#: Cap on ``deg_{theta_n} D``.  ``Normalise((N/D)')`` has a denominator of degree
#: up to ``deg D + deg rad(D)``, so the integrand grows roughly quadratically in
#: this number; 5 keeps pairs in the size range a CAS user would actually type.
_MAX_DENOMINATOR_DEGREE = 5
_MAX_DENOMINATOR_DEGREE_ALGEBRAIC = 3


def _liouville_once(
    pool: ExprPool,
    tier: int,
    rng: Random,
    *,
    normalise: bool,
) -> GeneratedPair:
    x = pool.symbol("x")
    tower = build_tower(pool, tier, rng)

    algebraic_top = tower.top_is_algebraic
    r_max = 2 if algebraic_top else rng.randint(1, 3)
    factors = _denominator_factors(pool, tower, rng, r_max)
    deg_d = _degree_in_top(factors)
    cap = _MAX_DENOMINATOR_DEGREE_ALGEBRAIC if algebraic_top else _MAX_DENOMINATOR_DEGREE
    if deg_d > cap:
        raise DegenerateSample(f"denominator degree {deg_d} exceeds cap {cap}")

    # Rational part, generated *already* in partial fraction form (Algorithm 2,
    # Step 10): F_hat = sum_i sum_{k<=m_i} n_{ik} / q_i^k.  Normalising this is
    # exactly the ``N/D`` of Step 7 with deg N < deg D.
    pf_terms: list[Expr] = []
    for q, m in factors:
        qdeg = _factor_degree(q)
        for k in range(1, m + 1):
            if k < m and rng.random() < 0.35:
                continue
            n_ik = _numerator_over(pool, tower, rng, qdeg)
            pf_terms.append(n_ik * _pow_expr(pool, q, -k))
    if not pf_terms:  # pragma: no cover - every factor contributes its top power
        raise DegenerateSample("empty rational part")
    f_hat = _sum(pool, pf_terms)

    # Step 4 — logs whose arguments are factors of D.  These are what make
    # normalisation *shrink* the integrand (A' shares D's factors), so we want at
    # least one most of the time; see §5.4 of the paper.
    s = len(factors)
    j = rng.randint(1, s) if rng.random() < 0.8 else 0
    a_terms: list[Expr] = []
    for idx in rng.sample(range(s), j) if j else []:
        q, _m = factors[idx]
        a_terms.append(_constant(pool, tier, rng) * ak.log(q))

    # Step 5 — logs with fresh arguments from the field.  In the partial fraction
    # view these are the degree-1 denominators BWD essentially never produces.
    k_fresh = _weighted_choice(rng, [0, 1, 2], [0.25, 0.5, 0.25])
    b_terms: list[Expr] = []
    b_args: list[Expr] = []
    for _ in range(k_fresh):
        arg = _fresh_log_argument(pool, tower, rng)
        b_args.append(arg)
        b_terms.append(_constant(pool, tier, rng) * ak.log(arg))

    a_part = _sum(pool, a_terms) if a_terms else pool.integer(0)
    b_part = _sum(pool, b_terms) if b_terms else pool.integer(0)

    f_hat_plus_a = f_hat + a_part
    d_f_hat_plus_a = _diff(f_hat_plus_a, x)
    d_b = _diff(b_part, x) if b_terms else pool.integer(0)

    if normalise:
        integral = _normalise(f_hat_plus_a) + b_part
        integrand = _normalise(d_f_hat_plus_a) + _simplified(d_b)
    else:
        integral = f_hat_plus_a + b_part
        integrand = _simplified(d_f_hat_plus_a) + _simplified(d_b)

    integral = _simplified(integral)
    integrand = _simplified(integrand)

    if _is_x_free(integrand, x):
        raise DegenerateSample("integrand does not depend on x")
    if str(integrand).strip() in ("0", "0/1"):
        raise DegenerateSample("zero integrand")
    missing = [
        str(g) for g in tower.gens[1:] if not _contains(integrand, g) or not _contains(integral, g)
    ]
    if missing:
        # A tier is a claim about differential-field structure; a pair whose
        # generators cancelled away does not exercise the tier it came from.
        raise DegenerateSample(f"generators absent from the emitted pair: {missing}")

    guards: list[tuple[Expr, str]] = list(tower.guards)
    for q, _m in factors:
        guards.append((q, "nonzero"))
        if any(_contains(a, q) for a in a_terms):
            guards.append((q, "positive"))  # q appears as a log argument in A
    for arg in b_args:
        guards.append((arg, "positive"))

    return GeneratedPair(
        integrand=integrand,
        integral=integral,
        tier=tier,
        method="LIOUVILLE",
        tower=tower.description,
        guards=guards,
        params={
            "n_denominator_factors": s,
            "denominator_degree_in_theta": deg_d,
            "multiplicities": [m for _q, m in factors],
            "n_logs_from_D": j,
            "n_fresh_logs": k_fresh,
            "normalised": normalise,
            "top_generator_kind": tower.kinds[-1],
            "tower_kinds": list(tower.kinds),
            "tower_height": len(tower.gens) - 1,
        },
    )


def _contains(expr: Expr, sub: Expr) -> bool:
    stack: list[object] = [expr]
    while stack:
        item = stack.pop()
        if isinstance(item, Expr):
            if item == sub:
                return True
            stack.extend(item.node()[1:])
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _is_x_free(expr: Expr, x: Expr) -> bool:
    return not _contains(expr, x)


def liouville_pair(
    pool: ExprPool,
    tier: int,
    rng: Random,
    *,
    normalise: bool = True,
    max_attempts: int = 24,
) -> GeneratedPair:
    """Generate one ``(integrand, integral)`` pair by the LIOUVILLE method.

    Args:
        pool: expression pool the result is built in.
        tier: 3 or 4 — selects the differential tower (see :data:`TIER_STRUCTURE`).
        rng: seeded :class:`random.Random`; the same seed reproduces the pair.
        normalise: put both sides over a common denominator (Algorithm 2, Step 8).
            ``False`` yields the partial-fraction form of Step 10, which exposes
            the degree-1 denominators BWD cannot produce.
        max_attempts: degenerate parameter draws are retried this many times.

    Raises:
        DegenerateSample: every attempt collapsed.  Callers should treat this as
            "resample", not as a bug.
    """
    last: Exception | None = None
    for _ in range(max_attempts):
        try:
            return _liouville_once(pool, tier, rng, normalise=normalise)
        except DegenerateSample as exc:
            last = exc
        except ak.AlkahestError as exc:
            last = exc
    msg = f"LIOUVILLE generator failed {max_attempts} times at tier {tier}: {last}"
    raise DegenerateSample(msg)


# ─────────────────────────────────────────────────────────────────────────────
# BWD baseline (for measuring the length bias LIOUVILLE claims to fix)
# ─────────────────────────────────────────────────────────────────────────────

_BWD_UNARY = ("exp", "log", "sin", "cos", "sqrt")
_BWD_BINARY = ("+", "-", "*", "/")


def _bwd_expr(pool: ExprPool, x: Expr, rng: Random, depth: int) -> Expr:
    if depth <= 0 or rng.random() < 0.3:
        if rng.random() < 0.55:
            return x
        return pool.integer(_small_int(rng, -5, 5))
    if rng.random() < 0.35:
        op = rng.choice(_BWD_UNARY)
        inner = _bwd_expr(pool, x, rng, depth - 1)
        if op == "exp":
            return ak.exp(inner)
        if op == "log":
            return ak.log(inner * inner + pool.integer(1))
        if op == "sqrt":
            return ak.sqrt(inner * inner + pool.integer(1))
        return ak.sin(inner) if op == "sin" else ak.cos(inner)
    op = rng.choice(_BWD_BINARY)
    lhs = _bwd_expr(pool, x, rng, depth - 1)
    rhs = _bwd_expr(pool, x, rng, depth - 1)
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    return lhs * _pow_expr(pool, rhs * rhs + pool.integer(1), -1)


def random_bwd_pair(pool: ExprPool, rng: Random, *, depth: int = 3) -> GeneratedPair:
    """Lample & Charton **BWD** baseline: random ``F``, emit ``(F', F)``.

    Only used to measure the length bias that LIOUVILLE is supposed to remove.
    """
    x = pool.symbol("x")
    for _ in range(24):
        try:
            integral = _simplified(_bwd_expr(pool, x, rng, depth))
            integrand = _simplified(_diff(integral, x))
        except ak.AlkahestError:
            continue
        if _is_x_free(integrand, x) or str(integrand).strip() in ("0", "0/1"):
            continue
        return GeneratedPair(
            integrand=integrand,
            integral=integral,
            tier=-1,
            method="BWD",
            tower="unstructured",
            guards=[],
            params={"depth": depth},
        )
    raise DegenerateSample("BWD baseline collapsed")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point used by the RL environment
# ─────────────────────────────────────────────────────────────────────────────


def random_elementary(pool: ExprPool, tier: int, rng: Random) -> Expr:
    """Build a random antiderivative *F* at the given Risch tier.

    The caller computes ``f = diff(F, x)`` to obtain the integrand.

    Tiers 3 and 4 route through the LIOUVILLE generator; the integral it returns
    is already normalised, so the caller's ``diff`` reproduces the generator's
    integrand up to normalisation.  Use :func:`liouville_pair` directly when you
    want both sides in the exact form Algorithm 2 emits.
    """
    x = pool.symbol("x")
    if tier == 0:
        return _rational(pool, x, rng)
    if tier == 1:
        return _exp_log(pool, x, rng)
    if tier == 2:
        return _algebraic_coeff(pool, x, rng)
    if tier in (3, 4):
        return liouville_pair(pool, tier, rng).integral
    msg = f"Tier {tier} grammar not yet implemented"
    raise NotImplementedError(msg)


def _rational(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    """Random polynomial in *x* with small integer coefficients."""
    degree = rng.randint(1, 4)
    terms: list[Expr] = []
    zero = pool.integer(0)
    for i in range(degree + 1):
        c = pool.integer(rng.randint(-3, 3))
        if c == zero:
            continue
        terms.append(c * x ** pool.integer(i))
    if not terms:
        return x
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _exp_log(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    inner = _rational(pool, x, rng)
    choice = rng.choice(["exp", "log", "product"])
    if choice == "exp":
        return ak.exp(inner)
    if choice == "log":
        # Keep the logarithm argument positive on a typical evaluation domain.
        arg = ak.simplify(inner * inner + pool.integer(1)).value
        return ak.log(arg)
    return _rational(pool, x, rng) * ak.exp(inner)


def _algebraic_coeff(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    d = rng.choice([2, 3, 5, 7])
    # NB: ``pool.integer(d) ** (1/2)`` builds a ``pow`` with a non-integer
    # exponent, which ``ak.diff`` refuses (E-DIFF-002) and ``ak.diff_forward``
    # refuses (E-DIFF-004) — that made every tier-2 row unusable.  The ``sqrt``
    # primitive is a plain function node and differentiates fine.
    sqrt_d = ak.sqrt(pool.integer(d))
    return sqrt_d * _rational(pool, x, rng) + _exp_log(pool, x, rng)
