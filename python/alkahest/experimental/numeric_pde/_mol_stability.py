"""Von Neumann stability hook for :class:`MolSystem` (PDE milestone M7b).

For each scalar equation ``u_t = F`` the linear constant-coefficient part
``L u = Σ_α c_α ∂^α u`` of ``F`` (its linearisation at ``u = 0``, kept only
when every coefficient is an exact constant) is discretised with the same
interior stencils the system uses, and the fully discrete scheme of each
fixed-step method — explicit Euler ``u^{n+1} = (1 + Δt L_h) u^n`` and
classical RK4 ``u^{n+1} = Σ_{k≤4} (Δt L_h)^k / k! u^n`` — is handed to
:func:`von_neumann`, which proves the exact condition on ``Δt``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ...exceptions import NumericPdeError
from ._exact import expr_free_symbols
from ._fd import fd_weights


def _frac(e):
    n = e.node()
    if n[0] == "integer":
        return Fraction(int(n[1]))
    if n[0] == "rational":
        return Fraction(int(n[1]), int(n[2]))
    return None


def _exact_const(e):
    """An exact Fraction for a rational-constant Expr (after expansion)."""
    import alkahest

    f = _frac(e)
    if f is not None:
        return f
    try:
        s = alkahest.simplify(e)
        s = s.value if hasattr(s, "value") else s
    except Exception:
        return None
    return _frac(s)


def _conv(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = ka + kb
            out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v != 0}


def stability_report(ms):
    import alkahest

    from ._mol import _central_halfwidth, _jet_map, _upwind_window

    sys = ms._sys
    pool = sys.pool
    dims = sys.dims
    D = len(dims)
    report = {}
    dt = pool.symbol("dt")
    for f, (g, alg) in sys.ctx["rhs"].items():
        name = sys.field_names[f]
        if g is None:
            continue
        entry: dict[str, Any] = {"dims": D}
        report[name] = entry
        if alg:
            entry["skipped"] = "algebraic equation"
            continue
        if f in getattr(sys, "fv", {}):
            entry["skipped"] = "conservative WENO flux (nonlinear: not analysed)"
            continue
        jets = _jet_map(sys.ctx, g)
        if any(fld != f for _, fld, _ in jets):
            entry["skipped"] = "coupled to other fields (systems are not analysed)"
            continue
        if any(d.kind != "fd" or not d.uniform for d in dims):
            entry["skipped"] = "von Neumann analysis needs uniform finite-difference grids"
            continue
        zero = {s: pool.integer(0) for s, _, _ in jets}
        stencil = {}
        lin_terms = []
        ok = True
        for sym, _, sa in jets:
            c = alkahest.subs(alkahest.diff(g, sym).value, zero)
            names = expr_free_symbols(c)
            if names:
                ok = False
                break
            cf = _exact_const(c)
            if cf is None:
                ok = False
                break
            if cf == 0:
                continue
            lin_terms.append(f"{cf}*{sym}")
            # Tensor-product interior stencil of this jet.
            part = {(0,) * D: Fraction(1)}
            for k, m in enumerate(sa):
                if m == 0:
                    continue
                h = dims[k].h
                if not isinstance(h, Fraction):
                    ok = False
                    break
                if m == 1 and sys.scheme in ("upwind", "weno5"):
                    # The sign of the advective coefficient picks the side.
                    if cf > 0:
                        offs = _upwind_window(sys.order, +1)
                    else:
                        offs = _upwind_window(sys.order, -1)
                else:
                    q = _central_halfwidth(m, sys.order)
                    offs = list(range(-q, q + 1))
                w = fd_weights(m, offs).weights
                one = {}
                for o, wv in zip(offs, w):
                    key = tuple(o if j == k else 0 for j in range(D))
                    one[key] = Fraction(wv) / h**m
                part = _conv(part, one)
            if not ok:
                break
            for key, v in part.items():
                stencil[key] = stencil.get(key, 0) + cf * v
        if not ok:
            entry["skipped"] = "coefficients are not exact constants (variable or float)"
            continue
        stencil = {k: v for k, v in stencil.items() if v != 0}
        entry["linear_part"] = " + ".join(lin_terms) if lin_terms else "0"
        entry["stencil"] = {k: str(v) for k, v in sorted(stencil.items())}
        if not stencil:
            entry["skipped"] = "no linear part"
            continue
        # Collapse to the diagonal mode in several dimensions.
        one_d = {}
        for k, v in stencil.items():
            s = sum(k)
            one_d[s] = one_d.get(s, 0) + v
        status_override = "necessary" if D > 1 else None
        methods = {}
        # Explicit Euler: the exact scheme handed to von_neumann (proved).
        # Its cylindrical decomposition grows quickly with the stencil
        # width, so wider stencils use the exact real-axis criterion
        # |1 + dt λ| ≤ 1 on the symbol's range instead.
        if max(abs(k) for k in one_d) <= 1:
            methods["euler"] = _euler_von_neumann(one_d, pool, dt, status_override)
        else:
            methods["euler"] = _poly_condition(one_d, status_override, "euler")
        # Classical RK4: G = P(dt λ(θ)) with P the degree-4 Taylor polynomial.
        methods["rk4"] = _poly_condition(one_d, status_override, "rk4")
        entry["methods"] = methods
    return report


def _euler_von_neumann(one_d, pool, dt, status_override):
    from ._scheme import Scheme, von_neumann

    coeffs = {(1, 0): pool.integer(1)}
    for o in set(one_d) | {0}:
        v = Fraction(one_d.get(o, 0))
        coeffs[(0, o)] = (
            pool.integer(-1 if o == 0 else 0) - pool.rational(v.numerator, v.denominator) * dt
        )
    try:
        res = von_neumann(Scheme(coeffs, dt=dt, pool=pool))
    except NumericPdeError as e:
        return {"error": str(e)}
    max_dt = None
    for lo, hi, _lc, _hc in res["intervals"]:
        lo_f = None if lo is None else _to_float(lo)
        hi_f = None if hi is None else _to_float(hi)
        if (lo_f is None or lo_f <= 0) and (hi_f is None or hi_f > 0):
            max_dt = hi_f if hi_f is not None else float("inf")
    return {
        "condition": str(res["stability_condition"]),
        "proof_status": status_override or res["proof_status"],
        "max_dt": 0.0 if max_dt is None else max_dt,
        "G": str(res["G"]) if res["G"] is not None else None,
        "method": "von_neumann",
    }


def _symbol_range(one_d):
    """For a symmetric stencil (real symbol ``λ(θ) = Σ_k c_k e^{ikθ}``):
    ``(λ_min, λ_max, exact)``; else None. With ``c = cos θ``,
    ``λ = c0 + Σ 2 c_k T_k(c)``; up to ``k = 2`` it is a quadratic in ``c``
    and the extremes are exact rationals."""
    import numpy as np

    if not all(one_d.get(-k, 0) == v for k, v in one_d.items()):
        return None
    K = max(abs(k) for k in one_d)
    c0 = Fraction(one_d.get(0, 0))
    c1 = Fraction(one_d.get(1, 0))
    c2 = Fraction(one_d.get(2, 0))
    if K <= 2:
        # λ(c) = (c0 - 2 c2) + 2 c1 c + 4 c2 c²
        a, b, cc = 4 * c2, 2 * c1, c0 - 2 * c2
        pts = [Fraction(-1), Fraction(1)]
        if a != 0:
            v = -b / (2 * a)
            if -1 < v < 1:
                pts.append(v)
        vals = [a * p * p + b * p + cc for p in pts]
        return (min(vals), max(vals), True)
    coeffs = np.zeros(K + 1)
    coeffs[0] = float(c0)
    for k in range(1, K + 1):
        coeffs[k] = 2 * float(one_d.get(k, 0))
    cheb = np.polynomial.chebyshev.Chebyshev(coeffs)
    crit = [r.real for r in cheb.deriv().roots() if abs(r.imag) < 1e-12 and -1 <= r.real <= 1]
    vals = [float(cheb(p)) for p in [-1.0, 1.0, *crit]]
    return (min(vals), max(vals), False)


# Negative real root of z^3 + 4 z^2 + 12 z + 24, where the RK4 polynomial
# P(z) = 1 + z + z²/2 + z³/6 + z⁴/24 equals 1: RK4's real stability interval
# is [β, 0]; explicit Euler's is [-2, 0].
_RK4_BETA = -2.785293563405282
_REAL_BOUND = {"euler": -2.0, "rk4": _RK4_BETA}


def _amplification(method, z):
    if method == "euler":
        return 1 + z
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def _poly_condition(one_d, status_override, method):
    """Stability of ``u^{n+1} = P(dt L) u^n``: exact on the real axis for a
    symmetric stencil (the symbol's range is a real interval); by sampling
    θ and bisecting on ``dt`` otherwise (numerical evidence only)."""
    import numpy as np

    rng = _symbol_range(one_d)
    if rng is not None:
        lo, hi, exact = rng
        status = status_override or ("proved" if exact else "numeric")
        if hi > 0:
            return {"condition": "unstable (a growing mode)", "proof_status": status, "max_dt": 0.0}
        beta = _REAL_BOUND[method]
        max_dt = beta / float(lo) if lo < 0 else float("inf")
        return {
            "condition": f"dt * ({lo}) >= {beta:.15g}" if exact else f"dt <= {max_dt:.12g}",
            "proof_status": status,
            "max_dt": max_dt,
            "symbol_range": (float(lo), float(hi)),
            "method": "real-axis stability interval",
        }
    th = np.linspace(0, 2 * np.pi, 4097)
    lam = sum(float(v) * np.exp(1j * k * th) for k, v in one_d.items())

    def ok(dt):
        return np.max(np.abs(_amplification(method, dt * lam))) <= 1 + 1e-12

    scale = max(float(np.abs(lam).max()), 1e-300)
    if not ok(1e-9 / scale):
        return {"condition": "unstable for every dt > 0", "proof_status": "numeric", "max_dt": 0.0}
    lo, hi = 0.0, 4.0 / scale
    if ok(hi):
        return {"condition": "dt >= 0", "proof_status": "numeric", "max_dt": float("inf")}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return {
        "condition": f"dt <= {lo:.12g}",
        "proof_status": "numeric",
        "max_dt": lo,
        "method": "sampled symbol",
    }


def _to_float(v):
    if isinstance(v, (int, float, Fraction)):
        return float(v)
    try:
        import alkahest

        return float(alkahest.eval_expr(v, {}))
    except Exception:
        n = v.node() if hasattr(v, "node") else None
        if n and n[0] == "rational":
            return int(n[1]) / int(n[2])
        if n and n[0] == "integer":
            return float(int(n[1]))
        raise
