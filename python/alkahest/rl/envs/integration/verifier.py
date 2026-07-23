from __future__ import annotations

from typing import TYPE_CHECKING

import alkahest as ak
from alkahest.rl.core.verifier import BaseVerifier

if TYPE_CHECKING:
    from alkahest import Expr, ExprPool

_N_SPOT_CHECKS = 8
_SPOT_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23]


class IntegrationVerifier(BaseVerifier):
    """Reward model integration attempts.

    Soundness note (directional issue #3): the "honest refusal" reward is gated
    exclusively on the ``is_elementary`` label carried by the row — which comes
    from the curated known-non-elementary corpus in ``env._known_nonelementary_forms``
    — and NEVER on the engine's own ``integrate()`` classification. We therefore
    reward a refusal only when the integrand is *provably* non-elementary, and
    never when the engine merely gave up (``E-INT-001``) or emitted a false
    ``NonElementary`` verdict (``E-INT-004``; cf. bug B2, fixed in #219). A model
    answer is otherwise scored by differentiating it and comparing to the
    integrand, independent of any engine verdict.
    """

    def verify(self, completion: str, metadata: dict) -> float:
        is_elementary: bool = metadata["is_elementary"]
        f_expr: Expr = metadata["f_expr"]
        pool: ExprPool = metadata["pool"]
        x = pool.symbol("x")

        declined = _model_declined(completion)

        if not is_elementary:
            if declined:
                return 1.0
            reward = _verify_antiderivative(completion, f_expr, pool, x)
            return reward if reward > 0 else -0.5

        if declined:
            return -0.2

        return _verify_antiderivative(completion, f_expr, pool, x)


def _verify_antiderivative(
    completion: str,
    f: Expr,
    pool: ExprPool,
    x: Expr,
) -> float:
    try:
        candidate = ak.parse(completion, pool, {"x": x})
    except Exception:
        return 0.0

    try:
        diff_c = ak.diff(candidate, x)
        residual = ak.simplify(ak.simplify(diff_c.value).value - f).value
        if _is_zero(residual):
            return 1.0

        if _is_constant(residual, x, pool):
            return 1.0

        try:
            residual2 = ak.simplify_egraph(residual)
            if _is_zero(residual2):
                return 1.0
        except (AttributeError, Exception):
            pass

        hits = 0
        for p in _SPOT_PRIMES[:_N_SPOT_CHECKS]:
            try:
                pt = {x: ak.ArbBall(p / (p + 1), 1e-30)}
                ball = ak.interval_eval(residual, pt)
                if _ball_contains_zero(ball):
                    hits += 1
            except Exception:
                pass
        if hits == _N_SPOT_CHECKS:
            return 0.9

    except Exception:
        pass

    return 0.0


def _is_zero(expr: Expr) -> bool:
    s = str(expr).strip()
    return s in ("0", "0/1", "")


def _is_constant(expr: Expr, x: Expr, pool: ExprPool) -> bool:
    """Check ``diff(expr, x) == 0`` symbolically."""
    try:
        d = ak.diff(expr, x)
        return _is_zero(ak.simplify(d.value).value)
    except Exception:
        return False


def _ball_contains_zero(ball) -> bool:
    try:
        lo, hi = float(ball.lo), float(ball.hi)
        return lo <= 0.0 <= hi
    except Exception:
        return False


def _model_declined(text: str) -> bool:
    markers = [
        "no elementary",
        "nonelementary",
        "non-elementary",
        "cannot be expressed",
        "no closed form",
    ]
    t = text.lower()
    return any(m in t for m in markers)
