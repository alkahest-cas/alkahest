"""Verify LIOUVILLE-generated pairs and emit them as a reusable JSON corpus.

Two jobs, deliberately in one module:

* **Verification.**  A generator that emits a wrong ``(f, F)`` pair silently
  poisons training data, benchmarks and RL rewards alike, so every pair is
  checked *twice* before it is allowed out:

  1. symbolically — ``simplify(diff(F) - f) == 0``, with ``together``/``cancel``
     and the e-graph simplifier as fallbacks.  This catches a bug in the
     normalisation chain (``together``/``cancel``/``simplify``), which is where
     the generator's own emitted form could drift from the constructed one.
  2. numerically and *independently of* :func:`alkahest.diff` — a fourth-order
     central difference of the emitted ``F`` is compared against the emitted
     ``f`` at several in-domain points.  This is the check that would catch a
     wrong differentiation rule, which the symbolic check cannot: the symbolic
     check uses ``diff`` on both sides.

  Sample points are chosen so that every ``log`` argument is positive, every
  ``sqrt`` radicand is positive and every negative-power base is bounded away
  from zero — the guards are read off the emitted expressions themselves, not
  taken on trust from the generator.

* **Corpus emission.**  ``python -m alkahest.rl.envs.integration.corpus`` writes
  a JSON file of verified pairs plus the quality statistics Barket et al. report
  (length balance against a BWD baseline, uniqueness after coefficient erasure)
  and Alkahest's own solvability profile.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import alkahest as ak
from alkahest import Expr, ExprPool
from alkahest.rl.envs.integration.grammar import (
    TIER_STRUCTURE,
    TIERS,
    DegenerateSample,
    GeneratedPair,
    liouville_pair,
    node_count,
    random_bwd_pair,
    random_elementary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

__all__ = [
    "CorpusStats",
    "Verification",
    "bwd_baseline",
    "const_skeleton",
    "domain_guards",
    "generate_corpus",
    "has_canonical_argument_order",
    "load_pairs",
    "read_expr",
    "record_from_pair",
    "solvability",
    "summarise",
    "verify_pair",
]

#: Points are searched for in this window.  It straddles zero on purpose: a
#: tower without ``log(x)`` in it is perfectly real for negative ``x``, and some
#: log arguments (``sqrt(x+4) - 2*exp(x)``, say) are *only* positive there.  The
#: upper end is kept modest because tier-4 towers include ``exp(exp(x))``.
_SAMPLE_LO = -3.5
_SAMPLE_HI = 3.5
_N_GRID = 141
#: Step for the fourth-order central difference.  Truncation error is O(h^4) and
#: float round-off is O(eps/h), so 1e-3 sits near the optimum (~1e-12 relative).
_FD_H = 1e-3
_NUM_TOL = 1e-5
_MIN_NUM_POINTS = 3
_MAX_MAGNITUDE = 1e14


# ─────────────────────────────────────────────────────────────────────────────
# Domain analysis
# ─────────────────────────────────────────────────────────────────────────────


def domain_guards(*exprs: Expr) -> list[tuple[Expr, str]]:
    """Read the real-domain conditions straight off the emitted expressions.

    ``log(u)`` needs ``u > 0``; ``sqrt(u)`` needs ``u > 0`` (strictly, because
    ``sqrt(u)^-1`` routinely appears after differentiation); ``b^-k`` needs
    ``b != 0``.  Deriving these from the *emitted* form rather than from the
    generator's bookkeeping means a normalisation step that introduces a new
    denominator cannot slip a domain trap past us.
    """
    guards: list[tuple[Expr, str]] = []
    seen: set = set()
    stack: list[object] = list(exprs)
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
            continue
        if not isinstance(item, Expr):
            continue
        node = item.node()
        tag = node[0]
        if tag == "func" and node[1] in ("log", "sqrt"):
            for arg in node[2]:
                key = (str(arg), "positive")
                if key not in seen:
                    seen.add(key)
                    guards.append((arg, "positive"))
        elif tag == "pow":
            base, exponent = node[1], node[2]
            if _negative_exponent(exponent):
                key = (str(base), "nonzero")
                if key not in seen:
                    seen.add(key)
                    guards.append((base, "nonzero"))
        stack.extend(node[1:])
    return guards


def _negative_exponent(exponent: Expr) -> bool:
    if not isinstance(exponent, Expr):
        return False
    try:
        return float(ak.eval_expr(exponent, {})) < 0
    except Exception:
        return False


def _eval(expr: Expr, x: Expr, value: float) -> float | None:
    try:
        out = ak.eval_expr(expr, {x: value})
    except Exception:
        return None
    try:
        out = float(out)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or abs(out) > _MAX_MAGNITUDE:
        return None
    return out


def _guards_hold(guards: Sequence[tuple[Expr, str]], x: Expr, value: float) -> bool:
    for expr, kind in guards:
        v = _eval(expr, x, value)
        if v is None:
            return False
        if kind == "positive" and v <= 1e-3:
            return False
        if kind == "nonzero" and abs(v) <= 1e-3:
            return False
    return True


def _candidate_grid() -> list[float]:
    step = (_SAMPLE_HI - _SAMPLE_LO) / (_N_GRID - 1)
    return [_SAMPLE_LO + i * step for i in range(_N_GRID)]


def _in_domain_points(
    guards: Sequence[tuple[Expr, str]],
    x: Expr,
    n_wanted: int,
) -> tuple[list[float], float]:
    """Grid points where all guards hold, together with the fraction that do."""
    grid = _candidate_grid()
    good = [v for v in grid if _guards_hold(guards, x, v)]
    fraction = len(good) / len(grid)
    if not good:
        return [], fraction
    # Spread the chosen points over the valid set rather than clustering.
    if len(good) <= n_wanted:
        return good, fraction
    stride = len(good) / n_wanted
    return [good[int(i * stride)] for i in range(n_wanted)], fraction


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Verification:
    ok: bool
    symbolic: bool
    numeric: bool
    n_points: int
    max_rel_error: float | None
    in_domain_fraction: float
    flags: list[str] = field(default_factory=list)
    reason: str = ""


def _is_zero(expr: Expr) -> bool:
    return str(expr).strip() in ("0", "0/1", "-0", "0.0000000000000000")


def _residual_is_zero(residual: Expr) -> bool:
    candidates = [residual]
    for fn in (
        lambda e: ak.simplify(e).value,
        ak.together,
        ak.cancel,
        lambda e: ak.simplify_expanded(e).value,
    ):
        try:
            candidates.append(fn(candidates[-1]))
        except Exception:
            break
    if any(_is_zero(c) for c in candidates):
        return True
    if getattr(ak, "HAS_EGRAPH", False):
        for c in list(candidates):
            try:
                if _is_zero(ak.simplify_egraph(c)):
                    return True
            except Exception:
                pass
    return False


def _residual_vanishes(residual: Expr, reference: Expr, x: Expr) -> bool | None:
    """Tri-state "is this residual zero": ``True`` / ``False`` / ``None`` (undecided).

    ``simplify``/``together``/``cancel`` are incomplete — they fail on residuals as
    tame as ``(2x-2)/(x^2-2x+5/4) - (8x-8)/(1+(2-2x)^2)``, which arises whenever
    ``integrate`` returns ``log(u)`` where the reference answer is ``log(c*u)``.
    Reporting that as a *wrong* answer would be a serious slander, so the symbolic
    verdict is backed by a numeric one and "cannot tell" stays "cannot tell".
    """
    if _residual_is_zero(residual):
        return True
    guards = domain_guards(residual, reference)
    points, _fraction = _in_domain_points(guards, x, 6)
    checked = 0
    for x0 in points:
        r = _eval(residual, x, x0)
        if r is None:
            continue
        ref = _eval(reference, x, x0)
        scale = max(1.0, abs(ref) if ref is not None else 1.0)
        if abs(r) > 1e-5 * scale:
            return False
        if abs(r) <= 1e-8 * scale:
            checked += 1
    return True if checked >= _MIN_NUM_POINTS else None


def _fd_once(f_integral: Expr, x: Expr, x0: float, h: float) -> float | None:
    vals = []
    for offset in (-2, -1, 1, 2):
        v = _eval(f_integral, x, x0 + offset * h)
        if v is None:
            return None
        vals.append(v)
    fm2, fm1, fp1, fp2 = vals
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)


def _finite_difference(f_integral: Expr, x: Expr, x0: float) -> tuple[float, float] | None:
    """Richardson-extrapolated derivative of ``f_integral`` at ``x0``, with an error bar.

    Two fourth-order central differences at ``h`` and ``h/2`` are combined; their
    disagreement is the error estimate.  Carrying the error bar matters: near the
    left edge of the sample window the higher derivatives of a tier-4 tower are
    enormous, and a bare ``h = 1e-3`` stencil is wrong by ~1e-3 relative there.
    Treating that as a *mismatch* would discard perfectly good pairs, so an
    inconclusive point is skipped instead.
    """
    d_h = _fd_once(f_integral, x, x0, _FD_H)
    d_half = _fd_once(f_integral, x, x0, _FD_H / 2.0)
    if d_h is None or d_half is None:
        return None
    richardson = (16.0 * d_half - d_h) / 15.0
    return richardson, abs(d_half - d_h) / 15.0


def verify_pair(
    pair: GeneratedPair,
    pool: ExprPool,
    *,
    n_points: int = 6,
    tol: float = _NUM_TOL,
) -> Verification:
    """Check ``d/dx integral == integrand``, symbolically and numerically.

    The numeric leg deliberately does not call :func:`alkahest.diff`: it compares
    the emitted integrand against a fourth-order finite difference of the emitted
    integral, so a wrong differentiation rule cannot verify itself.
    """
    x = pool.symbol("x")
    flags: list[str] = []

    guards = list(pair.guards) + domain_guards(pair.integral, pair.integrand)
    points, fraction = _in_domain_points(guards, x, n_points + 2)
    if fraction < 0.3:
        flags.append("narrow_domain")
    if _has_sign_change(guards, x):
        flags.append("guard_sign_change")

    symbolic = False
    try:
        residual = ak.diff(pair.integral, x).value - pair.integrand
        symbolic = _residual_is_zero(residual)
    except Exception as exc:
        flags.append(f"diff_failed:{type(exc).__name__}")

    numeric = False
    checked = 0
    max_rel: float | None = None
    inconclusive = 0
    for x0 in points:
        fd_pair = _finite_difference(pair.integral, x, x0)
        if fd_pair is None:
            continue
        fd, fd_err = fd_pair
        direct = _eval(pair.integrand, x, x0)
        if direct is None:
            continue
        scale = max(1.0, abs(fd), abs(direct))
        if fd_err / scale > tol:
            # The stencil itself cannot resolve the derivative here.
            inconclusive += 1
            continue
        checked += 1
        rel = abs(fd - direct) / scale
        max_rel = rel if max_rel is None else max(max_rel, rel)
        if rel > tol:
            return Verification(
                ok=False,
                symbolic=symbolic,
                numeric=False,
                n_points=checked,
                max_rel_error=max_rel,
                in_domain_fraction=fraction,
                flags=flags,
                reason=f"numeric mismatch at x={x0:.4f}: rel={rel:.3e}",
            )
    numeric = checked >= _MIN_NUM_POINTS
    if inconclusive:
        flags.append(f"fd_inconclusive:{inconclusive}")

    if not symbolic and not numeric:
        reason = "no symbolic proof and too few conclusive in-domain sample points"
        return Verification(
            ok=False,
            symbolic=symbolic,
            numeric=numeric,
            n_points=checked,
            max_rel_error=max_rel,
            in_domain_fraction=fraction,
            flags=flags,
            reason=reason,
        )
    if not symbolic:
        flags.append("numeric_only")
    return Verification(
        ok=True,
        symbolic=symbolic,
        numeric=numeric,
        n_points=checked,
        max_rel_error=max_rel,
        in_domain_fraction=fraction,
        flags=flags,
    )


def _has_sign_change(guards: Sequence[tuple[Expr, str]], x: Expr) -> bool:
    """Does any ``log`` argument / denominator flip sign inside the window?

    A ``True`` here is not a wrong pair — ``d/dx log(u) = u'/u`` holds either side
    of ``u = 0`` — but the *integral* is then only piecewise real, so downstream
    consumers should know.
    """
    grid = _candidate_grid()
    for expr, _kind in guards:
        seen_pos = seen_neg = False
        for v in grid:
            out = _eval(expr, x, v)
            if out is None:
                continue
            if out > 0:
                seen_pos = True
            elif out < 0:
                seen_neg = True
            if seen_pos and seen_neg:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_TAGS = ("integer", "float", "rational")


def const_skeleton(expr: Expr, *, erase_exponents: bool = True) -> str:
    """Canonical string with every numeric literal replaced by ``CONST``.

    Two expressions with the same skeleton are "similar" in the sense of Lample &
    Charton / Barket et al.: they differ only in their coefficients.  ``add`` and
    ``mul`` children are sorted so the skeleton is order-insensitive.

    ``erase_exponents=False`` keeps integer exponents intact, which is the looser
    reading of "differ only by coefficients" and always reports *higher*
    uniqueness; the strict default erases them too.
    """
    node = expr.node()
    tag = node[0]
    if tag in _NUMERIC_TAGS:
        return "CONST"
    if tag == "symbol":
        return f"var({node[1]})"
    if tag == "func":
        args = " ".join(const_skeleton(a, erase_exponents=erase_exponents) for a in node[2])
        return f"({node[1]} {args})"
    if tag in ("add", "mul"):
        parts = sorted(const_skeleton(a, erase_exponents=erase_exponents) for a in node[1])
        return f"({tag} {' '.join(parts)})"
    if tag == "pow":
        base = const_skeleton(node[1], erase_exponents=erase_exponents)
        if erase_exponents:
            exponent = const_skeleton(node[2], erase_exponents=True)
        else:
            exponent = str(node[2]).replace(" ", "")
        return f"(pow {base} {exponent})"
    children = " ".join(
        const_skeleton(a, erase_exponents=erase_exponents) for a in node[1:] if isinstance(a, Expr)
    )
    return f"({tag} {children})"


def _rebuild_shuffled(pool: ExprPool, expr: Expr, rng: Random) -> Expr:
    node = expr.node()
    tag = node[0]
    if tag in ("add", "mul"):
        kids = [_rebuild_shuffled(pool, c, rng) for c in node[1]]
        rng.shuffle(kids)
        return pool.add(kids) if tag == "add" else pool.mul(kids)
    if tag == "pow":
        return _rebuild_shuffled(pool, node[1], rng) ** _rebuild_shuffled(pool, node[2], rng)
    if tag == "func":
        return pool.func(node[1], [_rebuild_shuffled(pool, a, rng) for a in node[2]])
    return expr


def has_canonical_argument_order(pool: ExprPool, expr: Expr, rng: Random) -> bool:
    """Is ``expr`` invariant under permuting every ``add``/``mul`` argument list?

    This is the check for Davis's criticism of BWD data (arXiv:1912.05752): if a
    generator's output carried a term order inherited from the differentiator, a
    model could key on it.  In Alkahest it cannot — ``ExprPool.add``/``mul`` sort
    their children canonically at construction — but the property is asserted
    rather than assumed.
    """
    try:
        return _rebuild_shuffled(pool, expr, rng) == expr
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Solvability probe
# ─────────────────────────────────────────────────────────────────────────────


def _try_integrate(
    expr: Expr, pool: ExprPool, wall_ms: int, max_steps: int
) -> tuple[Expr | None, str | None]:
    x = pool.symbol("x")
    try:
        with ak.context(pool=pool, budget=ak.Budget(wall_ms=wall_ms, max_steps=max_steps)):
            return ak.integrate(expr, x).value, None
    except ak.BudgetExceededError as exc:
        return None, getattr(exc, "code", "E-BUDGET-001")
    except Exception as exc:
        return None, getattr(exc, "code", type(exc).__name__)


def solvability(
    integrand: Expr,
    pool: ExprPool,
    *,
    wall_ms: int = 3000,
    max_steps: int = 5_000_000,
) -> dict[str, object]:
    """Can Alkahest's own :func:`alkahest.integrate` close this integrand?

    Returns ``solved`` / ``code`` / ``verified`` / ``seconds``, plus
    ``solved_termwise``: whether every *top-level additive term* integrates on
    its own.  The gap between the two isolates how much of the failure rate is a
    missing sum-splitting fallback (``risch.md`` §5.1, "route commitment with no
    fallback") rather than a genuinely harder integrand — which matters here,
    because ``Normalise(F' + A') + B'`` is a sum by construction.

    ``verified`` re-differentiates whatever ``integrate`` returned: a ``solved``
    row with ``verified=False`` would be a silent-error-gate escape and is worth
    shouting about.
    """
    x = pool.symbol("x")
    t0 = time.time()
    value, code = _try_integrate(integrand, pool, wall_ms, max_steps)
    elapsed = time.time() - t0

    verified: bool | None = None
    if value is not None:
        try:
            verified = _residual_vanishes(ak.diff(value, x).value - integrand, integrand, x)
        except Exception:
            verified = None

    node = integrand.node()
    terms = list(node[1]) if node[0] == "add" else [integrand]
    if value is not None:
        termwise: bool | None = True
    else:
        termwise = all(_try_integrate(t, pool, wall_ms, max_steps)[0] is not None for t in terms)

    return {
        "solved": value is not None,
        "code": code,
        "verified": verified,
        "seconds": elapsed,
        "n_terms": len(terms),
        "solved_termwise": termwise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus assembly
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CorpusStats:
    tier: int
    requested: int
    emitted: int
    discarded: int
    discard_reasons: dict[str, int] = field(default_factory=dict)
    generator_failures: int = 0
    seconds: float = 0.0


def record_from_pair(
    pair: GeneratedPair,
    verification: Verification,
    pool: ExprPool,
    rng: Random,
    *,
    probe_solvability: bool,
    solver_wall_ms: int,
) -> dict:
    record = {
        "integrand": str(pair.integrand),
        "integral": str(pair.integral),
        "tier": pair.tier,
        "method": pair.method,
        "tower": pair.tower,
        "integrand_nodes": node_count(pair.integrand),
        "integral_nodes": node_count(pair.integral),
        "params": pair.params,
        "verification": asdict(verification),
        "skeleton": const_skeleton(pair.integrand),
        "skeleton_keep_exponents": const_skeleton(pair.integrand, erase_exponents=False),
        "canonical_arg_order": has_canonical_argument_order(pool, pair.integrand, rng),
        "parse_roundtrip": _roundtrips(pair.integrand, pool),
    }
    if probe_solvability:
        record["solvability"] = solvability(pair.integrand, pool, wall_ms=solver_wall_ms)
    return record


def read_expr(text: str, pool: ExprPool) -> Expr:
    """Read one serialised expression back from a corpus record.

    **Use this rather than a bare** :func:`alkahest.parse`.  ``parse`` leaves a
    negative exponent as an unfolded product — ``parse("(x+1)^-1")`` builds
    ``pow((x+1), mul(1, -1))``, not ``pow((x+1), -1)`` — and
    :func:`alkahest.diff` then refuses the whole expression with ``E-DIFF-002``
    ("cannot differentiate power with non-integer exponent").  Every integrand in
    this corpus prints with ``^-1``, so a naive reader gets a corpus it cannot
    differentiate.  A ``simplify`` pass folds the exponent and fixes it.
    """
    x = pool.symbol("x")
    return ak.simplify(ak.parse(text, pool, {"x": x})).value


def load_pairs(path: Path) -> list[tuple[Expr, Expr, ExprPool, dict]]:
    """Read a corpus JSON file back into ``(integrand, integral, pool, record)``."""
    payload = json.loads(Path(path).read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    out = []
    for record in records:
        pool = ExprPool()
        out.append(
            (
                read_expr(record["integrand"], pool),
                read_expr(record["integral"], pool),
                pool,
                record,
            )
        )
    return out


def _roundtrips(expr: Expr, pool: ExprPool) -> bool:
    """Does ``read_expr(str(expr))`` come back as the same expression?

    A corpus that cannot be read back is useless, so this is recorded per row.
    """
    try:
        return read_expr(str(expr), pool) == ak.simplify(expr).value
    except Exception:
        return False


def generate_corpus(
    tier: int,
    n: int,
    seed: int,
    *,
    normalise: bool = True,
    probe_solvability: bool = False,
    solver_wall_ms: int = 3000,
    max_draws_per_pair: int = 6,
) -> tuple[list[dict], CorpusStats]:
    """Generate ``n`` **verified** pairs at ``tier``.

    Unverified pairs are discarded, never emitted; the discard reasons are
    returned so a high rate is visible rather than silent.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    stats = CorpusStats(tier=tier, requested=n, emitted=0, discarded=0)
    t0 = time.time()
    draws = 0
    budget = n * max_draws_per_pair
    while len(records) < n and draws < budget:
        draws += 1
        pool = ExprPool()
        try:
            if tier in (3, 4):
                pair = liouville_pair(pool, tier, rng, normalise=normalise)
            else:
                pair = _legacy_pair(pool, tier, rng)
        except DegenerateSample:
            stats.generator_failures += 1
            continue
        except Exception as exc:
            stats.generator_failures += 1
            _bump(stats.discard_reasons, f"generator:{type(exc).__name__}")
            continue
        verification = verify_pair(pair, pool)
        if not verification.ok:
            stats.discarded += 1
            _bump(stats.discard_reasons, verification.reason or "unverified")
            continue
        records.append(
            record_from_pair(
                pair,
                verification,
                pool,
                rng,
                probe_solvability=probe_solvability,
                solver_wall_ms=solver_wall_ms,
            )
        )
    stats.emitted = len(records)
    stats.seconds = time.time() - t0
    return records, stats


def _legacy_pair(pool: ExprPool, tier: int, rng: Random) -> GeneratedPair:
    """Wrap the tier 0–2 grammars in the same ``(integrand, integral)`` shape."""
    x = pool.symbol("x")
    integral = random_elementary(pool, tier, rng)
    integrand = ak.simplify(ak.diff(integral, x).value).value
    return GeneratedPair(
        integrand=integrand,
        integral=integral,
        tier=tier,
        method="BWD-grammar",
        tower=TIERS.get(tier, "?"),
        guards=[],
        params={},
    )


def bwd_baseline(n: int, seed: int, *, depths: Sequence[int] = (3, 4, 5)) -> list[dict]:
    """Lample & Charton BWD pairs, for the length-balance comparison only.

    Depths are mixed so the baseline spans a size range comparable to the
    LIOUVILLE output; comparing a shallow BWD sample against a deep LIOUVILLE
    sample would flatter the latter for the wrong reason.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    guard = 0
    while len(out) < n and guard < 50 * n:
        guard += 1
        pool = ExprPool()
        try:
            pair = random_bwd_pair(pool, rng, depth=rng.choice(list(depths)))
        except DegenerateSample:
            continue
        verification = verify_pair(pair, pool)
        if not verification.ok:
            continue
        out.append(
            {
                "integrand": str(pair.integrand),
                "integral": str(pair.integral),
                "tier": -1,
                "method": "BWD",
                "integrand_nodes": node_count(pair.integrand),
                "integral_nodes": node_count(pair.integral),
                "skeleton": const_skeleton(pair.integrand),
                "skeleton_keep_exponents": const_skeleton(pair.integrand, erase_exponents=False),
                "depth": pair.params.get("depth"),
            }
        )
    return out


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def summarise(records: Sequence[dict]) -> dict:
    """Length balance + uniqueness for a set of records."""
    if not records:
        return {}
    f_sizes = [r["integrand_nodes"] for r in records]
    g_sizes = [r["integral_nodes"] for r in records]
    ratios = [f / g for f, g in zip(f_sizes, g_sizes) if g]
    skeletons = [r["skeleton"] for r in records]
    skeletons_loose = [r["skeleton_keep_exponents"] for r in records]
    return {
        "n": len(records),
        "integrand_nodes": _describe(f_sizes),
        "integral_nodes": _describe(g_sizes),
        "size_ratio_integrand_over_integral": _describe(ratios),
        "unique_fraction_const_erased": len(set(skeletons)) / len(skeletons),
        "unique_fraction_exponents_kept": len(set(skeletons_loose)) / len(skeletons_loose),
        "unique_fraction_verbatim": len({r["integrand"] for r in records}) / len(records),
    }


def _describe(values: Sequence[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p25": q(0.25),
        "median": q(0.5),
        "mean": sum(ordered) / n,
        "p75": q(0.75),
        "p90": q(0.9),
        "max": ordered[-1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m alkahest.rl.envs.integration.corpus",
        description="Generate a verified corpus of (integrand, integral) pairs.",
    )
    parser.add_argument("--tiers", default="3,4", help="comma-separated tiers (default 3,4)")
    parser.add_argument("-n", "--per-tier", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("temp-alkahest/testing/autogen/liouville_corpus.json"),
    )
    parser.add_argument(
        "--partial-fraction", action="store_true", help="emit the un-normalised form"
    )
    parser.add_argument(
        "--solvability", action="store_true", help="probe ak.integrate on each pair"
    )
    parser.add_argument("--solver-wall-ms", type=int, default=3000)
    parser.add_argument("--bwd-baseline", type=int, default=0, help="also emit N BWD pairs")
    args = parser.parse_args(argv)

    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
    payload: dict[str, object] = {
        "generator": "LIOUVILLE (Barket, England & Gerhard, arXiv:2406.11631)",
        "seed": args.seed,
        "normalised": not args.partial_fraction,
        "tier_structure": {str(t): TIER_STRUCTURE.get(t, "") for t in tiers},
        "records": [],
        "stats": {},
        "summary": {},
    }
    all_records: list[dict] = []
    for tier in tiers:
        records, stats = generate_corpus(
            tier,
            args.per_tier,
            args.seed + tier,
            normalise=not args.partial_fraction,
            probe_solvability=args.solvability,
            solver_wall_ms=args.solver_wall_ms,
        )
        all_records.extend(records)
        payload["stats"][str(tier)] = asdict(stats)  # type: ignore[index]
        payload["summary"][str(tier)] = summarise(records)  # type: ignore[index]
        print(
            f"tier {tier}: emitted {stats.emitted}/{stats.requested} "
            f"(discarded {stats.discarded}, generator failures {stats.generator_failures}) "
            f"in {stats.seconds:.1f}s",
            file=sys.stderr,
        )
    payload["records"] = all_records

    if args.bwd_baseline:
        baseline = bwd_baseline(args.bwd_baseline, args.seed + 999)
        payload["bwd_baseline"] = baseline
        payload["bwd_summary"] = summarise(baseline)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(all_records)} pairs to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
