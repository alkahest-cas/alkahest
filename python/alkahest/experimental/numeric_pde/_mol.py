"""Method of lines: a symbolic ``PDE`` problem → a compiled ODE/DAE system.

PDE milestone M7b. :func:`method_of_lines` reads an
``alkahest.experimental.PDE(...).on(domain).bc(...).ic(...)`` problem in one
to three space dimensions, replaces every spatial ``Derivative`` by exact
finite-difference stencils (:func:`fd_weights`, any order; upwind-biased or
WENO-5 for advective terms) or by Chebyshev / Fourier collocation, closes the
boundaries (Dirichlet, Neumann, Robin, periodic — eliminated, or kept as
algebraic constraints that make the system a DAE), and compiles the result.

The core design decision is that **nothing is generated per grid point**.
For every *class* of points that share a stencil (the interior bulk, each
near-boundary layer, each boundary face) one scalar kernel expression is
built in the values at fixed offsets, compiled once through the JIT
(:func:`alkahest.compile_expr` on a list → ``CompiledVecFn``), and applied by
the native runtime (``alkahest_cas::pde::mol``) over the index box of that
class, reading neighbours through shifted slices of the field arrays. The
compile cost is independent of the grid size and a 10⁶-point grid is 10⁶
evaluations of a handful of kernels, not 10⁶ expression nodes. The exact
sparse Jacobian is built the same way: one compiled kernel of symbolic
partial derivatives per class, composed through the boundary closures.
"""

from __future__ import annotations

import math
import time as _time
import warnings
from fractions import Fraction
from itertools import product as _iproduct
from typing import Any

from ...exceptions import NumericPdeError
from ._exact import expr_free_symbols as _efs
from ._fd import fd_weights

__all__ = ["MolResult", "MolStabilityWarning", "MolSystem", "method_of_lines"]


class MolStabilityWarning(UserWarning):
    """The requested fixed time step violates the proved von Neumann
    condition of the generated scheme."""


def _err(code, msg, remediation=None):
    return NumericPdeError(msg, code=code, remediation=remediation)


def _ak():
    import alkahest

    return alkahest


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a test dependency
        raise _err("E-PDE-370", "method_of_lines needs NumPy") from exc
    return np


# ---------------------------------------------------------------------------
# Small expression helpers
# ---------------------------------------------------------------------------


_CONSTANTS = frozenset(("pi", "E", "I"))


def expr_free_symbols(e):
    """Free symbol names of ``e`` (the constants ``pi``, ``E``, ``I`` excluded)."""
    return {k: v for k, v in _efs(e).items() if k not in _CONSTANTS}


def _is_zero(e):
    n = e.node()
    if n[0] == "integer":
        return int(n[1]) == 0
    if n[0] == "float":
        return float(n[1]) == 0.0
    return False


def _num(e):
    """A Python number for a numeric constant ``Expr`` (else ``None``)."""
    n = e.node()
    if n[0] == "integer":
        return int(n[1])
    if n[0] == "rational":
        return Fraction(int(n[1]), int(n[2]))
    if n[0] == "float":
        return float(n[1])
    try:
        ak = _ak()
        v = ak.eval_expr(e, {})
        return float(v)
    except Exception:
        return None


def _const(pool, v):
    if isinstance(v, bool):
        raise TypeError("a bool is not a number")
    if isinstance(v, int):
        return pool.integer(v)
    if isinstance(v, Fraction):
        return pool.rational(v.numerator, v.denominator)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 2**53:
            return pool.integer(int(v))
        return pool.float(v)
    return v


def _exact(v):
    """An exact rational for an int / Fraction / decimal-looking float."""
    if isinstance(v, (int, Fraction)) and not isinstance(v, bool):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(repr(v))
    return None


def _diff(e, s):
    return _ak().diff(e, s).value


def _subs(e, m):
    return _ak().subs(e, m) if m else e


def _contains(e, syms):
    names = expr_free_symbols(e)
    return any(str(s) in names for s in syms)


def _max2(pool, a, b):
    """``max(a, b)`` as ``(a + b + |a - b|) / 2`` (compiles to straight-line
    code, and differentiates to the partials of the larger argument)."""
    return pool.rational(1, 2) * (a + b + _ak().abs(a - b))


# ---------------------------------------------------------------------------
# Problem analysis
# ---------------------------------------------------------------------------


class _Dim:
    """One space dimension of the grid."""

    def __init__(self, var, lo, hi):
        self.var = var
        self.name = str(var)
        self.lo_expr, self.hi_expr = lo, hi
        self.lo_x: Any = None  # exact (Fraction) bound when rational, else float
        self.hi_x: Any = None
        self.lo_f = 0.0
        self.hi_f = 0.0
        self.specs: set[tuple[int, Any]] = set()  # (derivative order, stencil kind) used
        self.periodic = False
        self.kind = "fd"  # "fd" | "cheb" | "fourier"
        self.n = 0
        self.nodes: Any = None
        self.uniform = True
        self.h: Any = None  # exact or float spacing (uniform fd)
        self.pad = 0
        self.reach = 1
        self.D = {}  # spectral: order -> numpy matrix

    @property
    def length(self):
        return self.hi_f - self.lo_f

    def __repr__(self):
        return f"_Dim({self.name}, [{self.lo_f}, {self.hi_f}], n={self.n}, {self.kind})"


def _domain_dims(domain):
    if domain is None:
        raise _err(
            "E-PDE-359",
            "method_of_lines needs a domain: PDE(...).on(Interval(x, a, b)) or a Box",
        )
    spec = domain.spec()
    out = []

    def walk(s):
        kind = s[0]
        if kind == "interval":
            out.append((s[1], s[2], s[3]))
        elif kind == "box":
            for v, (lo, hi) in zip(s[1], s[2]):
                out.append((v, lo, hi))
        elif kind == "product":
            for part in s[1]:
                walk(part)
        else:
            raise _err(
                "E-PDE-359",
                f"method_of_lines supports intervals and boxes, not a {kind!r} domain",
                "map a disk or ball to polar coordinates, or truncate a half-line to an interval",
            )

    walk(spec)
    if not 1 <= len(out) <= 3:
        raise _err("E-PDE-359", f"method_of_lines supports 1-3 space dimensions, got {len(out)}")
    pool = out[0][0]._pool
    dims = []
    for v, lo, hi in out:
        lo_e = _const(pool, lo) if not hasattr(lo, "node") else lo
        hi_e = _const(pool, hi) if not hasattr(hi, "node") else hi
        d = _Dim(v, lo_e, hi_e)
        lo_n, hi_n = _num(lo_e), _num(hi_e)
        if lo_n is None or hi_n is None:
            raise _err(
                "E-PDE-359",
                f"the bounds of {v} must be numbers, got [{lo}, {hi}]",
                "substitute values for symbolic bounds first",
            )
        d.lo_x = Fraction(lo_n) if isinstance(lo_n, (int, Fraction)) else float(lo_n)
        d.hi_x = Fraction(hi_n) if isinstance(hi_n, (int, Fraction)) else float(hi_n)
        d.lo_f, d.hi_f = float(lo_n), float(hi_n)
        if not d.hi_f > d.lo_f:
            raise _err("E-PDE-359", f"empty interval for {v}: [{lo}, {hi}]")
        dims.append(d)
    return dims


def _applied_parts(e):
    """``(name, args)`` for an applied undefined function node, else None."""
    n = e.node()
    if n[0] == "func" and n[1] not in ("Derivative",):
        return n[1], list(n[2])
    return None


class _Unknown:
    def __init__(self, idx, expr, name, args):
        self.idx = idx
        self.expr = expr
        self.name = name
        self.args = args  # the argument symbols, in slot order
        self.time_order = 0
        self.fields = []  # field index per time-derivative level


def _analyse(problem, params):
    if not hasattr(problem, "equations") or not hasattr(problem, "unknowns"):
        raise TypeError("method_of_lines takes an alkahest.experimental.PDE problem")
    dims = _domain_dims(problem.domain)
    pool = dims[0].var._pool
    space = [d.var for d in dims]
    indep = list(problem.independent)
    others = [v for v in indep if all(v != s for s in space)]
    if len(others) != 1:
        raise _err(
            "E-PDE-359",
            "method_of_lines needs exactly one time variable besides the domain's "
            f"space variables; independent variables {indep}, domain variables {space}",
        )
    tvar = others[0]
    unknowns = []
    for i, u in enumerate(problem.unknowns):
        parts = _applied_parts(u)
        if parts is None:
            raise _err("E-PDE-359", f"unknown {u} is not an applied function")
        name, args = parts
        if any(all(a != v for v in indep) for a in args) or len(args) != len(indep):
            raise _err(
                "E-PDE-359",
                f"unknown {u} must take every independent variable ({indep}) as a bare argument",
            )
        unknowns.append(_Unknown(i, u, name, args))
    max_order = max(problem.order, 1)
    from alkahest.experimental import JetSpace

    js = JetSpace(indep, list(problem.unknowns), max_order)
    t_pos = next(i for i, v in enumerate(indep) if v == tvar)
    s_pos = [next(i for i, v in enumerate(indep) if v == s) for s in space]

    # Parameter substitution.
    psubs = {}
    for k, v in (params or {}).items():
        key = k if hasattr(k, "node") else pool.symbol(str(k))
        ex = _exact(v)
        psubs[key] = _const(pool, ex if ex is not None else v) if not hasattr(v, "node") else v

    eqs = []
    for eq in problem.equations:
        je = js.to_jets(_subs(eq, psubs))
        jets = js.jets_in(je)
        info = []
        for func, alpha, sym in jets:
            ui = next(j for j, u in enumerate(unknowns) if u.expr == func)
            sa = tuple(alpha[p] for p in s_pos)
            info.append((ui, sa, alpha[t_pos], sym))
        eqs.append((je, info))

    # Leading time derivative per equation.
    leading = {}
    for k, (je, info) in enumerate(eqs):
        tj = [(ui, sa, ta, sym) for ui, sa, ta, sym in info if ta > 0]
        if not tj:
            continue
        top = max(ta for _, _, ta, _ in tj)
        lead = [j for j in tj if j[2] == top]
        if len(lead) != 1 or any(lead[0][1]):
            raise _err(
                "E-PDE-360",
                f"equation {k} must contain one highest time derivative of one unknown, with "
                f"no space derivative on it; found {[str(j[3]) for j in lead]}",
                "write the system as u_t = F(...) (or u_tt = F) per unknown",
            )
        if top > 2:
            raise _err(
                "E-PDE-360", f"equation {k} is of order {top} in time; at most 2 is supported"
            )
        ui, _, _, sym = lead[0]
        if ui in leading.values():
            raise _err(
                "E-PDE-360",
                f"two equations lead with a time derivative of {unknowns[ui].expr}",
            )
        coef = _diff(je, sym)
        if _contains(coef, [j[3] for j in tj]):
            raise _err(
                "E-PDE-360",
                f"equation {k} is not linear in {sym} (or its coefficient involves "
                "time derivatives)",
            )
        leading[k] = ui
        unknowns[ui].time_order = top
    lead_of = {ui: k for k, ui in leading.items()}
    alg_eqs = [k for k in range(len(eqs)) if k not in leading]
    free_u = [u.idx for u in unknowns if u.idx not in lead_of]
    if len(alg_eqs) != len(free_u):
        raise _err(
            "E-PDE-360",
            f"{len(alg_eqs)} equation(s) without a time derivative for {len(free_u)} unknown(s) "
            "without one; the system is not square",
        )
    for k, ui in zip(alg_eqs, free_u):
        lead_of[ui] = k

    # Fields: u, u_t, ... per unknown.
    fields = []  # (unknown idx, time level)
    for u in unknowns:
        levels = max(u.time_order, 1)
        u.fields = []
        for lv in range(levels):
            u.fields.append(len(fields))
            fields.append((u.idx, lv))

    # Right-hand sides in jets.
    rhs = {}  # field -> (expr in jets, algebraic?)
    for u in unknowns:
        k = lead_of[u.idx]
        je, info = eqs[k]
        if u.time_order == 0:
            rhs[u.fields[0]] = (je, True)
            continue
        lsym = next(
            sym for ui, sa, ta, sym in info if ui == u.idx and ta == u.time_order and not any(sa)
        )
        coef = _diff(je, lsym)
        rest = _subs(je, {lsym: pool.integer(0)})
        g = (pool.integer(-1) * rest) * coef ** pool.integer(-1)
        if u.time_order == 1:
            rhs[u.fields[0]] = (g, False)
        else:
            rhs[u.fields[0]] = (None, False)  # u' = v (filled in later)
            rhs[u.fields[1]] = (g, False)
    # Check the remaining jets: time derivatives of order < T only.
    for f, (g, _) in rhs.items():
        if g is None:
            continue
        for func, alpha, sym in js.jets_in(g):
            ui = next(j for j, u in enumerate(unknowns) if u.expr == func)
            ta = alpha[t_pos]
            T = unknowns[ui].time_order
            if ta > 0 and ta >= max(T, 1):
                raise _err(
                    "E-PDE-360",
                    f"{sym} appears on the right-hand side of the equation for "
                    f"{unknowns[fields[f][0]].expr}; only lower time derivatives may",
                )
    ctx = {
        "pool": pool,
        "dims": dims,
        "tvar": tvar,
        "indep": indep,
        "t_pos": t_pos,
        "s_pos": s_pos,
        "js": js,
        "unknowns": unknowns,
        "fields": fields,
        "rhs": rhs,
        "psubs": psubs,
    }
    return ctx


def _jet_map(ctx, expr):
    """``[(sym, field, spatial alpha)]`` for every jet of ``expr``."""
    out = []
    unknowns = ctx["unknowns"]
    for func, alpha, sym in ctx["js"].jets_in(expr):
        ui = next(j for j, u in enumerate(unknowns) if u.expr == func)
        ta = alpha[ctx["t_pos"]]
        sa = tuple(alpha[p] for p in ctx["s_pos"])
        out.append((sym, unknowns[ui].fields[ta], sa))
    return out


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class _BC:
    def __init__(self, dim, side, expr, field_ids, raw):
        self.dim = dim
        self.side = side  # 0 = lo, 1 = hi
        self.expr = expr  # residual in jets (+ tangential coordinates, t)
        self.fields = field_ids
        self.raw = raw


def _boundary_value(e, ctx):
    """For an applied unknown / Derivative node evaluated on the boundary:
    ``(unknown idx, (dim, side), alpha over indep)`` — or None."""
    n = e.node()
    if n[0] != "func":
        return None
    if n[1] == "Derivative":
        inner = n[2][0]
        orders = [int(o.node()[1]) for o in n[2][1:]]
    else:
        inner = e
        orders = None
    parts = _applied_parts(inner)
    if parts is None:
        return None
    name, args = parts
    unknowns = ctx["unknowns"]
    u = next((u for u in unknowns if u.name == name and len(u.args) == len(args)), None)
    if u is None:
        return None
    if orders is None:
        orders = [0] * len(args)
    faces = []
    alpha = [0] * len(ctx["indep"])
    for slot, a in enumerate(args):
        var = u.args[slot]
        pos = next(i for i, v in enumerate(ctx["indep"]) if v == var)
        alpha[pos] = orders[slot]
        if a == var:
            continue
        di = next((i for i, d in enumerate(ctx["dims"]) if d.var == var), None)
        if di is None:
            return ("time", u.idx, a)
        val = _num(a)
        d = ctx["dims"][di]
        if val is None:
            return ("bad", e)
        if abs(val - d.lo_f) <= 1e-12 * max(1.0, abs(d.lo_f)):
            faces.append((di, 0))
        elif abs(val - d.hi_f) <= 1e-12 * max(1.0, abs(d.hi_f)):
            faces.append((di, 1))
        else:
            return ("bad", e)
    return ("bnd", u.idx, faces, tuple(alpha))


def _is_periodicity(r, occ, dim, dims):
    """Whether ``r = 0`` says exactly ``A(lo) = A(hi)``: mapping every
    occurrence on the hi face to the lo face must cancel it."""
    import alkahest

    d = dims[dim]
    m = {}
    for e, bv in occ:
        if bv[2] == [(dim, 1)]:
            m[e] = alkahest.subs(e, {d.hi_expr: d.lo_expr})
            if m[e] == e:
                return False
    z = _subs(r, m)
    if _is_zero(z):
        return True
    try:
        s = alkahest.simplify(z)
        s = s.value if hasattr(s, "value") else s
    except Exception:
        return False
    return _is_zero(s)


def _parse_bcs(problem, ctx):
    dims = ctx["dims"]
    unknowns = ctx["unknowns"]
    psubs = ctx["psubs"]
    bcs = []
    periodic = set()
    for lhs, rhs in problem.bcs:
        r = _subs(lhs - rhs, psubs)
        occ = []
        stack = [r]
        seen = set()
        while stack:
            e = stack.pop()
            if e in seen:
                continue
            seen.add(e)
            bv = _boundary_value(e, ctx)
            if bv is not None:
                if bv[0] == "bad":
                    raise _err(
                        "E-PDE-362",
                        f"boundary condition {lhs} = {rhs}: {bv[1]} is not evaluated on the "
                        "boundary of the domain",
                    )
                if bv[0] == "time":
                    raise _err(
                        "E-PDE-362",
                        f"boundary condition {lhs} = {rhs} fixes the time argument; use .ic()",
                    )
                occ.append((e, bv))
                continue
            n = e.node()
            if n[0] in ("add", "mul"):
                stack.extend(n[1])
            elif n[0] == "pow":
                stack.extend([n[1], n[2]])
            elif n[0] == "func":
                stack.extend(a for a in n[2] if hasattr(a, "node"))
        if not occ:
            raise _err("E-PDE-362", f"boundary condition {lhs} = {rhs} involves no unknown")
        face_sets = {tuple(sorted(bv[2])) for _, bv in occ}
        if len(face_sets) == 2 and all(len(fs) == 1 for fs in face_sets):
            (fa,), (fb,) = sorted(face_sets)
            if fa[0] == fb[0] and {fa[1], fb[1]} == {0, 1}:
                # u(lo) = u(hi) (or the same for a derivative): periodic.
                if not _is_periodicity(r, occ, fa[0], dims):
                    raise _err(
                        "E-PDE-362",
                        f"boundary condition {lhs} = {rhs} relates the two ends of "
                        f"{dims[fa[0]].name} but is not a periodicity condition u(lo) = u(hi)",
                        "write periodicity as u(lo, t) = u(hi, t); other couplings of the "
                        "two ends are not supported",
                    )
                periodic.add(fa[0])
                continue
        if len(face_sets) != 1 or len(next(iter(face_sets))) != 1:
            raise _err(
                "E-PDE-362",
                f"boundary condition {lhs} = {rhs} must hold on one face (or relate the two "
                "ends of one variable, for periodicity)",
            )
        (dim, side) = next(iter(face_sets))[0]
        # Rewrite with jets: unknown -> u symbol, Derivative -> jet.
        js = ctx["js"]
        rep = {}
        fids = set()
        for e, (_, ui, _, alpha) in occ:
            u = unknowns[ui]
            ta = alpha[ctx["t_pos"]]
            if ta > 0:
                raise _err(
                    "E-PDE-362",
                    f"boundary condition {lhs} = {rhs} contains a time derivative",
                )
            rep[e] = js.jet(u.expr, list(alpha))
            fids.add(u.fields[0])
        # Tangential coordinates stay as the space symbols; the normal one is
        # the face value.
        expr = _subs(r, rep)
        expr = _subs(expr, {dims[dim].var: dims[dim].hi_expr if side else dims[dim].lo_expr})
        bcs.append(_BC(dim, side, expr, sorted(fids), (lhs, rhs)))
    for d in periodic:
        dims[d].periodic = True
    for b in bcs:
        if dims[b.dim].periodic:
            raise _err(
                "E-PDE-362",
                f"{dims[b.dim].name} is periodic, but a boundary condition also fixes "
                f"{b.raw[0]} = {b.raw[1]}",
            )
    return bcs


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


_FD_SCHEMES = ("central", "upwind", "weno5", "weno5_fv")
_SCHEMES = (*_FD_SCHEMES, "chebyshev", "fourier")


def _setup_grid(dims, grid, scheme):
    np = _np()
    if isinstance(grid, int) and not isinstance(grid, bool):
        grid = {d.var: grid for d in dims}
    if not isinstance(grid, dict):
        raise _err("E-PDE-361", "grid must be an int or a dict {variable: points or node array}")
    names = {str(k): v for k, v in grid.items()}
    for d in dims:
        if d.name not in names:
            raise _err("E-PDE-361", f"grid gives no resolution for {d.name}")
        g = names[d.name]
        d.kind = "cheb" if scheme == "chebyshev" else "fourier" if scheme == "fourier" else "fd"
        if d.kind == "fourier" and not d.periodic:
            raise _err(
                "E-PDE-365",
                f"Fourier collocation needs a periodic {d.name} (add .bc(u(lo, t), u(hi, t)))",
            )
        if d.kind == "cheb" and d.periodic:
            raise _err("E-PDE-365", f"Chebyshev collocation on a periodic {d.name}; use fourier")
        if isinstance(g, int) and not isinstance(g, bool):
            n = g
            if n < (4 if not d.periodic else 3):
                raise _err("E-PDE-361", f"{d.name}: {n} grid points is too few")
            d.n = n
            if d.kind == "cheb":
                N = n - 1
                xi = np.cos(np.pi * np.arange(N + 1) / N)[::-1]
                d.nodes = d.lo_f + (d.hi_f - d.lo_f) * (xi + 1) / 2
                d.uniform = False
            elif d.periodic:
                h = (
                    Fraction(d.hi_x - d.lo_x) / n
                    if isinstance(d.lo_x, Fraction) and isinstance(d.hi_x, Fraction)
                    else (d.hi_f - d.lo_f) / n
                )
                d.h = h
                d.nodes = d.lo_f + np.arange(n) * float(h)
            else:
                h = (
                    Fraction(d.hi_x - d.lo_x) / (n - 1)
                    if isinstance(d.lo_x, Fraction) and isinstance(d.hi_x, Fraction)
                    else (d.hi_f - d.lo_f) / (n - 1)
                )
                d.h = h
                d.nodes = d.lo_f + np.arange(n) * float(h)
                d.nodes[-1] = d.hi_f
        else:
            if d.kind != "fd" or d.periodic:
                raise _err(
                    "E-PDE-361",
                    f"{d.name}: explicit node arrays are for finite differences on a "
                    "non-periodic interval",
                )
            nodes = np.asarray(g, dtype=float).ravel()
            if nodes.size < 4 or np.any(np.diff(nodes) <= 0):
                raise _err("E-PDE-361", f"{d.name}: nodes must be strictly increasing (≥ 4)")
            tol = 1e-12 * max(1.0, abs(d.lo_f), abs(d.hi_f))
            if abs(nodes[0] - d.lo_f) > tol or abs(nodes[-1] - d.hi_f) > tol:
                raise _err(
                    "E-PDE-361",
                    f"{d.name}: nodes must start at {d.lo_f} and end at {d.hi_f}",
                )
            d.n = nodes.size
            d.nodes = nodes
            steps = np.diff(nodes)
            d.uniform = bool(np.allclose(steps, steps[0], rtol=1e-13, atol=0))
            d.h = (d.hi_f - d.lo_f) / (d.n - 1) if d.uniform else None
            if d.uniform:
                hx = (
                    Fraction(d.hi_x - d.lo_x)
                    if isinstance(d.lo_x, Fraction) and isinstance(d.hi_x, Fraction)
                    else None
                )
                if hx is not None:
                    d.h = hx / (d.n - 1)
        if d.kind == "cheb":
            from ._spectral import cheb_diff_matrix

            D, _ = cheb_diff_matrix(d.n - 1, numeric=True)
            D = D[::-1, ::-1] * (2.0 / (d.hi_f - d.lo_f))
            d.D = {1: D}
        elif d.kind == "fourier":
            from ._spectral import fourier_diff_matrix

            scale = 2 * np.pi / (d.hi_f - d.lo_f)
            D1, _ = fourier_diff_matrix(d.n, 1, numeric=True)
            D2, _ = fourier_diff_matrix(d.n, 2, numeric=True)
            d.D = {1: D1 * scale, 2: D2 * scale**2}


def _spectral_matrix(d, m):
    np = _np()
    if m in d.D:
        return d.D[m]
    if d.kind == "fourier" and m % 2 == 0:
        M = np.linalg.matrix_power(d.D[2], m // 2)
    else:
        M = np.linalg.matrix_power(d.D[1], m)
    d.D[m] = M
    return M


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------


def _central_halfwidth(m, p):
    return (m + p - 1) // 2


def _upwind_window(p, sign):
    """Offsets of the order-``p`` first-derivative stencil biased against the
    flow: ``sign < 0`` backward (flow to the right), ``sign > 0`` forward."""
    lo = -(p // 2 + 1)
    offs = list(range(lo, lo + p + 1))
    return offs if sign < 0 else [-o for o in reversed(offs)]


def _fit(offs, lo, hi):
    """Slide the window ``offs`` (consecutive integers) into ``[lo, hi]``."""
    if lo is None:
        return offs
    k = len(offs)
    if k > hi - lo + 1:
        raise _err("E-PDE-361", "the grid is too coarse for the requested stencil")
    a = offs[0]
    if a < lo:
        a = lo
    if a + k - 1 > hi:
        a = hi - k + 1
    return list(range(a, a + k))


class _Weights:
    """A 1-D stencil: offsets with constant weights (uniform) or per-point
    weight arrays (non-uniform)."""

    def __init__(self, offsets, consts=None, arrays=None):
        self.offsets = offsets
        self.consts = consts
        self.arrays = arrays


def _fornberg(z, x, m):
    """Numerical Fornberg weights for derivative ``m`` at ``z`` on nodes ``x``."""
    n = len(x)
    c = [[0.0] * (m + 1) for _ in range(n)]
    c1, c4 = 1.0, x[0] - z
    c[0][0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i][k] = c1 * (k * c[i - 1][k - 1] - c5 * c[i - 1][k]) / c2
                c[i][0] = -c1 * c5 * c[i - 1][0] / c2
            for k in range(mn, 0, -1):
                c[j][k] = (c4 * c[j][k] - k * c[j][k - 1]) / c3
            c[j][0] = c4 * c[j][0] / c3
        c1 = c2
    return [c[j][m] for j in range(n)]


_WCACHE: dict = {}


def _fd_const(pool, d, m, offs):
    key = (m, tuple(offs))
    st = _WCACHE.get(key)
    if st is None:
        st = fd_weights(m, list(offs)).weights
        _WCACHE[key] = st
    h = d.h
    out = []
    for w in st:
        if isinstance(h, Fraction):
            out.append(_const(pool, Fraction(w) / h**m))
        else:
            out.append(pool.float(float(w) / float(h) ** m))
    return out


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------


class _Builder:
    """Accumulates the inputs of one kernel."""

    def __init__(self, sys):
        self.sys = sys
        self.pool = sys.pool
        self.inputs = []  # program input tuples
        self.syms = []
        self.index = {}

    def field(self, f, off):
        key = ("field", f, tuple(off))
        if key not in self.index:
            name = f"_v{f}_" + "_".join(("m" if o < 0 else "p") + str(abs(o)) for o in off)
            s = self.pool.symbol(name)
            self.index[key] = s
            self.inputs.append(("field", f, list(off)))
            self.syms.append(s)
        return self.index[key]

    def coef(self, c):
        key = ("coef", c)
        if key not in self.index:
            s = self.pool.symbol(f"_c{c}")
            self.index[key] = s
            self.inputs.append(("coef", c))
            self.syms.append(s)
        return self.index[key]

    def time(self):
        key = ("time",)
        if key not in self.index:
            s = self.pool.symbol("_t")
            self.index[key] = s
            self.inputs.append(("time",))
            self.syms.append(s)
        return self.index[key]


class _Avail:
    """Per dimension: the offsets available at every point of a class, as
    ``(lo, hi)`` (``None`` = unbounded: periodic or spectral)."""

    def __init__(self, ranges):
        self.ranges = ranges


def _class_ranges(d, a, b, sig):
    """Split grid indices ``[a, b)`` of a non-periodic fd dim into runs of
    points that use the same stencil windows: ``sig(lo, hi)`` is the tuple
    of windows chosen with available offsets ``[lo, hi]``. Only the ``reach``
    points next to each end can differ from the bulk, so this is O(reach),
    not O(n)."""
    R = d.reach
    out = []

    def push(i0, i1):
        lo, hi = -i0, d.n - 1 - i0
        key = sig(lo, hi)
        if out and out[-1][3] == key and out[-1][1] == i0:
            out[-1][1] = i1
        else:
            out.append([i0, i1, (lo, hi), key])

    left_end = min(b, a + R)
    right_start = max(left_end, b - R)
    for i in range(a, left_end):
        push(i, i + 1)
    if right_start > left_end:
        push(left_end, right_start)
    for i in range(right_start, b):
        push(i, i + 1)
    return [(lo, hi, av) for lo, hi, av, _ in out]


class _System:
    """The discretisation: everything needed to build the native program."""

    def __init__(
        self,
        problem,
        grid,
        scheme,
        order,
        bc_mode,
        params,
        jacobian,
        weno_eps,
        flux=None,
        wave_speed=None,
        flux_splitting="rusanov",
        weno_variant="js",
    ):
        t0 = _time.perf_counter()
        if scheme not in _SCHEMES:
            raise _err("E-PDE-365", f"unknown scheme {scheme!r}; one of {', '.join(_SCHEMES)}")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise _err("E-PDE-365", f"order must be a positive integer, got {order!r}")
        if bc_mode not in ("eliminate", "algebraic"):
            raise _err("E-PDE-365", "bc_mode is 'eliminate' or 'algebraic'")
        self.problem = problem
        self.scheme = scheme
        self.order = order
        self.bc_mode = bc_mode
        self.want_jac = jacobian
        self.weno_eps = weno_eps
        ctx = _analyse(problem, params)
        self.ctx = ctx
        self.pool = ctx["pool"]
        self.dims = ctx["dims"]
        self.tvar = ctx["tvar"]
        self.unknowns = ctx["unknowns"]
        self.fields = list(ctx["fields"])  # (unknown, level); aux fields appended
        self.field_names = []
        for ui, lv in self.fields:
            nm = self.unknowns[ui].name
            self.field_names.append(nm if lv == 0 else nm + "_" + "t" * lv)
        self.bcs = _parse_bcs(problem, ctx)
        if scheme == "weno5" and any(not d.periodic and False for d in self.dims):
            pass
        _setup_grid(self.dims, grid, scheme)
        if scheme in ("weno5", "weno5_fv") and any(not d.uniform for d in self.dims):
            raise _err("E-PDE-365", "WENO-5 needs a uniform grid")
        self.fv = {}
        if scheme == "weno5_fv":
            self._setup_fv(flux, wave_speed, flux_splitting, weno_variant)
        elif flux is not None or wave_speed is not None:
            raise _err(
                "E-PDE-365",
                "flux= and wave_speed= are options of scheme='weno5_fv'",
            )
        self._check_free_symbols()
        self._reach()
        self._layout()
        self.coefs = []  # numpy arrays (padded)
        self._coef_cache = {}
        self.ops = []
        self.kernel_exprs = []  # (op index, expr, builder) for export
        self.aux = {}  # (field, alpha) -> aux field
        self._roles()
        self._build_ops()
        self.discretize_time = _time.perf_counter() - t0

    # -- checks ----------------------------------------------------------------

    def _check_free_symbols(self):
        allowed = {d.name for d in self.dims} | {str(self.tvar)}
        js = self.ctx["js"]
        for f, (g, _) in self.ctx["rhs"].items():
            if g is None:
                continue
            jets = {str(s) for _, _, s in js.jets_in(g)}
            extra = set(expr_free_symbols(g)) - allowed - jets
            if extra:
                raise _err(
                    "E-PDE-366",
                    f"the equation for {self.field_names[f]} has free symbols {sorted(extra)}",
                    "pass their values: method_of_lines(..., params={nu: 0.01})",
                )
        for b in self.bcs:
            jets = {str(s) for _, _, s in js.jets_in(b.expr)}
            extra = set(expr_free_symbols(b.expr)) - allowed - jets
            if extra:
                raise _err(
                    "E-PDE-366",
                    f"boundary condition {b.raw[0]} = {b.raw[1]} has free symbols {sorted(extra)}",
                    "pass their values with params=",
                )

    def _reach(self):
        """Per dimension: ``pad`` (periodic halo = the widest stencil reach)
        and ``reach`` (the offset window used to group points into stencil
        classes; wide enough for the one-sided windows near a boundary)."""
        p = self.order
        need = [1] * len(self.dims)
        window = [1] * len(self.dims)
        for d in self.dims:
            d.specs = set()
        exprs = [g for g, _ in self.ctx["rhs"].values() if g is not None]
        exprs += [b.expr for b in self.bcs]
        for e in exprs:
            for _, _, sa in _jet_map(self.ctx, e):
                for k, m in enumerate(sa):
                    if m == 0:
                        continue
                    r = _central_halfwidth(m, p)
                    w = m + p
                    self.dims[k].specs.add((m, "central"))
                    if m == 1 and self.scheme in ("upwind", "weno5"):
                        self.dims[k].specs.add((1, ("upwind", -1)))
                        self.dims[k].specs.add((1, ("upwind", 1)))
                    if m == 1 and self.scheme == "weno5":
                        self.dims[k].specs.add((1, "weno"))
                    if m == 1 and self.scheme == "upwind":
                        r = max(r, max(abs(o) for o in _upwind_window(p, -1)))
                        w = max(w, p + 2)
                    if m == 1 and self.scheme == "weno5":
                        r = max(r, 3)
                        w = max(w, 7)
                    need[k] = max(need[k], r)
                    window[k] = max(window[k], r, w)
        for spec in self.fv.values():
            for k in spec["dims"]:
                need[k] = max(need[k], 3)
                window[k] = max(window[k], 7)
        for d, r, w in zip(self.dims, need, window):
            d.reach = w if d.kind == "fd" else 0
            d.pad = r if (d.periodic and d.kind == "fd") else 0

    def _layout(self):
        np = _np()
        self.shape = [d.n + 2 * d.pad for d in self.dims]
        D = len(self.dims)
        self.strides = [1] * D
        for k in range(D - 2, -1, -1):
            self.strides[k] = self.strides[k + 1] * self.shape[k + 1]
        self.size = int(np.prod(self.shape))

    def flat(self, idx):
        return sum((i + d.pad) * s for i, d, s in zip(idx, self.dims, self.strides))

    # -- coefficient arrays ------------------------------------------------------

    def coord_coef(self, k):
        key = ("coord", k)
        if key in self._coef_cache:
            return self._coef_cache[key]
        np = _np()
        d = self.dims[k]
        line = np.empty(self.shape[k])
        if d.pad:
            h = float(d.h)
            line[:] = d.lo_f + (np.arange(self.shape[k]) - d.pad) * h
        else:
            line[:] = d.nodes
        shp = [1] * len(self.dims)
        shp[k] = self.shape[k]
        arr = np.broadcast_to(line.reshape(shp), self.shape).ravel().copy()
        self.coefs.append(arr)
        self._coef_cache[key] = len(self.coefs) - 1
        return len(self.coefs) - 1

    def weight_coefs(self, k, m, window_of_i, irange):
        """Per-point Fornberg weights along dim ``k`` for points with grid
        index in ``irange``; ``window_of_i(i)`` gives the offsets."""
        np = _np()
        d = self.dims[k]
        offs0 = window_of_i(irange[0])
        cols = [np.zeros(self.shape[k]) for _ in offs0]
        for i in range(*irange):
            offs = window_of_i(i)
            w = _fornberg(d.nodes[i], [d.nodes[i + o] for o in offs], m)
            for c, v in zip(cols, w):
                c[i + d.pad] = v
        idx = []
        shp = [1] * len(self.dims)
        shp[k] = self.shape[k]
        for c in cols:
            self.coefs.append(np.broadcast_to(c.reshape(shp), self.shape).ravel().copy())
            idx.append(len(self.coefs) - 1)
        return idx

    # -- roles -----------------------------------------------------------------

    def _roles(self):
        """For every field and face: ``"fill"``, ``"alg"`` or ``"pde"``."""
        self.face_bc = {}
        # Assign every BC to the field whose boundary value it determines.
        for b in self.bcs:
            cands = [f for f in b.fields if (f, b.dim, b.side) not in self.face_bc]
            if not cands:
                raise _err(
                    "E-PDE-362",
                    f"too many boundary conditions on the {'lo' if b.side == 0 else 'hi'} "
                    f"face of {self.dims[b.dim].name}",
                )
            self.face_bc[(cands[0], b.dim, b.side)] = b
        self.status = {}
        for f, (ui, lv) in enumerate(self.fields):
            for k, d in enumerate(self.dims):
                if d.periodic:
                    continue
                for side in (0, 1):
                    if lv > 0:
                        base = self.unknowns[ui].fields[0]
                        st = self.status.get((base, k, side))
                        if st == "fill":
                            self.status[(f, k, side)] = "fill_dt"
                        elif st == "alg":
                            raise _err(
                                "E-PDE-365",
                                "bc_mode='algebraic' is not supported for equations of "
                                "second order in time",
                                "use bc_mode='eliminate'",
                            )
                        else:
                            self.status[(f, k, side)] = "pde"
                        continue
                    if (f, k, side) in self.face_bc:
                        self.status[(f, k, side)] = "alg" if self.bc_mode == "algebraic" else "fill"
                    else:
                        self.status[(f, k, side)] = "pde"
        # A field with algebraic BCs (or an algebraic equation) makes a DAE.

    def role(self, f, cat):
        """Role of the points of field ``f`` with category tuple ``cat``
        (per dim: 0 lo face, 1 interior, 2 hi face)."""
        owner = None
        for k in range(len(self.dims) - 1, -1, -1):
            if cat[k] != 1:
                owner = (k, 0 if cat[k] == 0 else 1)
                break
        if owner is None:
            return "pde", None
        return self.status[(f, owner[0], owner[1])], owner

    # -- discretising jets -----------------------------------------------------

    def _window(self, m, kind, lo, hi):
        """Offsets of the 1-D stencil for derivative order ``m`` and
        ``kind`` (``"central"`` or ``("upwind", sign)``) with available
        offsets ``[lo, hi]`` (``None``: unbounded)."""
        p = self.order
        if kind == "central":
            q = _central_halfwidth(m, p)
            want = list(range(-q, q + 1))
            if lo is not None and (want[0] < lo or want[-1] > hi):
                # The most centred window of m + p points that fits (fully
                # one-sided only at the boundary itself): such windows keep
                # the semi-discrete operator stable up to order 8 at least.
                k = m + p
                want = _fit(list(range(-(k // 2), k - k // 2)), lo, hi)
            return want
        return _fit(_upwind_window(p, kind[1]), lo, hi)

    def _signature(self, k, lo, hi):
        sig = []
        specs: list[tuple[int, Any]] = sorted(self.dims[k].specs, key=repr)
        for m, kind in specs:
            if kind == "weno":
                sig.append(lo is None or (lo <= -3 and hi >= 2, lo <= -2 and hi >= 3))
                continue
            try:
                sig.append(tuple(self._window(m, kind, lo, hi)))
            except NumericPdeError:
                sig.append(None)
        return tuple(sig)

    def _stencil(self, k, m, avail, kind, irange):
        """1-D stencil along dim ``k`` for derivative order ``m``.
        ``kind``: 'central' | ('upwind', sign) | 'onesided'(normal BC)."""
        d = self.dims[k]
        lo, hi = avail
        want = self._window(m, kind, lo, hi)
        if d.uniform:
            return _Weights(want, consts=_fd_const(self.pool, d, m, want))
        arrays = self.weight_coefs(k, m, lambda i: want, irange)
        return _Weights(want, arrays=arrays)

    def _apply(self, b, f, alpha, avail, kinds, irange_of):
        """``Σ w u(f, offset)`` for the tensor stencil of ``alpha`` (fd dims)
        applied to field ``f`` (an aux field already carries spectral dims)."""
        pool = self.pool
        D = len(self.dims)
        per = []
        for k in range(D):
            m = alpha[k]
            if m == 0 or self.dims[k].kind != "fd":
                per.append([(0, pool.integer(1))])
                continue
            st = self._stencil(k, m, avail.ranges[k], kinds[k], irange_of(k))
            terms = []
            for j, o in enumerate(st.offsets):
                w = st.consts[j] if st.consts is not None else b.coef(st.arrays[j])
                terms.append((o, w))
            per.append(terms)
        acc = []
        for combo in _iproduct(*per):
            off = [c[0] for c in combo]
            w = [c[1] for c in combo]
            v = b.field(f, off)
            acc.append(pool.mul([*w, v]) if len(w) > 0 else v)
        return pool.add(acc) if len(acc) > 1 else acc[0]

    def _aux_field(self, f, salpha):
        """Field holding the spectral part of ``∂^alpha`` of field ``f``."""
        key = (f, salpha)
        if key in self.aux:
            return self.aux[key]
        np = _np()
        cur = f
        partial = [0] * len(self.dims)
        for k, m in enumerate(salpha):
            if m == 0:
                continue
            partial[k] = m
            pk = (f, tuple(partial))
            if pk in self.aux:
                cur = self.aux[pk]
                continue
            d = self.dims[k]
            M = _spectral_matrix(d, m)
            new = len(self.fields)
            self.fields.append((None, f"aux{new}"))
            self.field_names.append(f"_aux_{self.field_names[f]}_{d.name}{m}")
            region = [(0, s) for s in self.shape]
            self.ops.append(
                {
                    "kind": "dense",
                    "src": cur,
                    "dst": new,
                    "axis": k,
                    "region": region,
                    "src_lo": 0,
                    "src_len": d.n,
                    "matrix": np.ascontiguousarray(M, dtype=float).ravel(),
                }
            )
            self.aux[pk] = new
            cur = new
        self.aux[key] = cur
        return cur

    def _discretise(self, b, expr, avail, irange_of, boundary=None):
        """Replace every jet of ``expr`` by stencils. ``boundary`` = (dim,
        side) makes normal derivatives one-sided (BC closure)."""
        pool = self.pool
        D = len(self.dims)
        jets = _jet_map(self.ctx, expr)
        out = expr
        # Advective upwinding first (fd dims, first derivatives).
        if self.scheme in ("upwind", "weno5") and boundary is None:
            for sym, f, sa in jets:
                if sum(sa) != 1:
                    continue
                k = sa.index(1)
                if self.dims[k].kind != "fd":
                    continue
                a = _diff(out, sym)
                if _contains(a, [sym]) or _is_zero(a):
                    continue
                rest = _subs(out, {sym: pool.integer(0)})
                dm = self._upwind_derivative(b, f, k, avail, irange_of, -1)
                dp = self._upwind_derivative(b, f, k, avail, irange_of, +1)
                half = pool.rational(1, 2)
                absa = _ak().abs(a)
                out = rest + half * (a - absa) * dm + half * (a + absa) * dp
        rep = {}
        for sym, f, sa in _jet_map(self.ctx, out):
            src = f
            if any(m and self.dims[k].kind != "fd" for k, m in enumerate(sa)):
                src = self._aux_field(
                    f, tuple(m if self.dims[k].kind != "fd" else 0 for k, m in enumerate(sa))
                )
            fa = tuple(m if self.dims[k].kind == "fd" else 0 for k, m in enumerate(sa))
            kinds = ["central"] * D
            rep[sym] = self._apply(b, src, fa, avail, kinds, irange_of)
        return self._coords_time(b, _subs(out, rep))

    def _upwind_derivative(self, b, f, k, avail, irange_of, sign):
        lo, hi = avail.ranges[k]
        if self.scheme == "weno5":
            offs = list(range(-3, 3)) if sign < 0 else list(range(-2, 4))
            if lo is None or (offs[0] >= lo and offs[-1] <= hi):
                return self._weno(b, f, k, sign)
        alpha = [0] * len(self.dims)
        alpha[k] = 1
        kinds = ["central"] * len(self.dims)
        kinds[k] = ("upwind", sign)
        return self._apply(b, f, tuple(alpha), avail, kinds, irange_of)

    def _weno(self, b, f, k, sign):
        """Jiang–Peng WENO-5 one-sided derivative (uniform grid)."""
        pool = self.pool
        D = len(self.dims)
        d = self.dims[k]
        h = d.h
        inv_h = _const(pool, 1 / Fraction(h)) if isinstance(h, Fraction) else pool.float(1 / h)

        def u(o):
            off = [0] * D
            off[k] = o
            return b.field(f, off)

        if sign < 0:
            v = [(u(j) - u(j - 1)) * inv_h for j in (-2, -1, 0, 1, 2)]
        else:
            v = [(u(j + 1) - u(j)) * inv_h for j in (2, 1, 0, -1, -2)]
        r = pool.rational
        v1, v2, v3, v4, v5 = v
        phi1 = r(1, 3) * v1 - r(7, 6) * v2 + r(11, 6) * v3
        phi2 = r(-1, 6) * v2 + r(5, 6) * v3 + r(1, 3) * v4
        phi3 = r(1, 3) * v3 + r(5, 6) * v4 - r(1, 6) * v5
        two = pool.integer(2)
        s1 = (
            r(13, 12) * (v1 - two * v2 + v3) ** two
            + r(1, 4) * (v1 - pool.integer(4) * v2 + pool.integer(3) * v3) ** two
        )
        s2 = r(13, 12) * (v2 - two * v3 + v4) ** two + r(1, 4) * (v2 - v4) ** two
        s3 = (
            r(13, 12) * (v3 - two * v4 + v5) ** two
            + r(1, 4) * (pool.integer(3) * v3 - pool.integer(4) * v4 + v5) ** two
        )
        eps = pool.float(self.weno_eps)
        m2 = pool.integer(-2)
        a1 = r(1, 10) * (eps + s1) ** m2
        a2 = r(6, 10) * (eps + s2) ** m2
        a3 = r(3, 10) * (eps + s3) ** m2
        return (a1 * phi1 + a2 * phi2 + a3 * phi3) * (a1 + a2 + a3) ** pool.integer(-1)

    # -- conservative WENO flux (scheme="weno5_fv") ------------------------------

    def _unknown_of(self, key):
        for u in self.unknowns:
            if (hasattr(key, "node") and u.expr == key) or (isinstance(key, str) and key == u.name):
                return u.idx
        raise _err(
            "E-PDE-373",
            f"flux= key {key!r} is not an unknown of the problem "
            f"({', '.join(str(u.expr) for u in self.unknowns)})",
        )

    def _flux_per_dim(self, val, what):
        """``val`` (one expression in 1-D, or a list / dict over the space
        variables) as a list with one entry (or ``None``) per dimension."""
        D = len(self.dims)
        if isinstance(val, dict):
            out = [None] * D
            for k2, v in val.items():
                for k, d in enumerate(self.dims):
                    if (hasattr(k2, "node") and k2 == d.var) or k2 == d.name:
                        out[k] = v
                        break
                else:
                    raise _err("E-PDE-373", f"{what}: {k2!r} is not a space variable")
            return out
        if isinstance(val, (list, tuple)):
            if len(val) != D:
                raise _err("E-PDE-373", f"{what}: give one entry per space dimension ({D})")
            return list(val)
        if D != 1:
            raise _err(
                "E-PDE-373",
                f"{what}: in {D} dimensions give a list or a dict over the space variables",
            )
        return [val]

    def _fv_jets(self, e, what):
        """``e`` (an expression in the unknowns, or a number) in jets; it may
        depend on the unknowns' values only."""
        pool = self.pool
        js = self.ctx["js"]
        if not hasattr(e, "node"):
            ex = _exact(e)
            if ex is None:
                raise _err("E-PDE-373", f"{what} {e!r} is not an expression or a number")
            return _const(pool, ex)
        je = js.to_jets(_subs(e, self.ctx["psubs"]))
        jets = js.jets_in(je)
        if any(any(a) for _, a, _ in jets):
            raise _err(
                "E-PDE-373",
                f"{what} {e} involves derivatives; it must be a function of the unknowns",
            )
        names = set(expr_free_symbols(je)) - {str(s) for _, _, s in jets}
        coords = {d.name for d in self.dims} | {str(self.tvar)}
        if names & coords:
            raise _err(
                "E-PDE-373",
                f"{what} {e} depends on {sorted(names & coords)}; only fluxes of the "
                "unknowns (no explicit space or time dependence) are supported",
            )
        if names:
            raise _err(
                "E-PDE-366",
                f"{what} {e} has free symbols {sorted(names)}",
                "pass their values with params=",
            )
        return je

    def _auto_flux(self):
        """A scalar conservation law ``u_t + Σ a_k(u) u_{x_k} = rest``: the
        flux ``F_k = ∫ a_k du``, checked by differentiating it back."""
        ak = _ak()
        cand = [u for u in self.unknowns if u.time_order == 1]
        if len(self.unknowns) != 1 or len(cand) != 1:
            raise _err(
                "E-PDE-373",
                "scheme='weno5_fv' needs flux= for a system (the flux of every "
                "conservative unknown)",
                "method_of_lines(..., scheme='weno5_fv', flux={u(x, t): F})",
            )
        u = cand[0]
        f = u.fields[0]
        g, _ = self.ctx["rhs"][f]
        u0 = self.ctx["js"].to_jets(u.expr)
        D = len(self.dims)
        out = [None] * D
        for sym, fld, sa in _jet_map(self.ctx, g):
            if fld != f or sum(sa) != 1:
                continue
            k = sa.index(1)
            a = ak.simplify(self.pool.integer(-1) * _diff(g, sym)).value
            names = set(expr_free_symbols(a)) - {str(u0)}
            if names:
                raise _err(
                    "E-PDE-373",
                    f"the advective coefficient {a} of {sym} is not a function of {u.expr} "
                    "alone; pass flux= explicitly",
                )
            try:
                F = ak.integrate(a, u0).value
                ok = _is_zero(ak.simplify(_diff(F, u0) - a).value)
            except Exception:
                ok = False
            if not ok:
                raise _err(
                    "E-PDE-373",
                    f"could not integrate the advective coefficient {a} to a flux; "
                    "pass flux= explicitly",
                )
            out[k] = F
        if all(F is None for F in out):
            raise _err(
                "E-PDE-373",
                f"the equation for {u.expr} has no first-derivative term to write in flux form",
            )
        return {u.idx: out}

    def _setup_fv(self, flux, wave_speed, splitting, variant):
        ak = _ak()
        pool = self.pool
        ctx = self.ctx
        js = ctx["js"]
        D = len(self.dims)
        if splitting not in ("rusanov", "lax_friedrichs"):
            raise _err("E-PDE-365", "flux_splitting is 'rusanov' or 'lax_friedrichs'")
        if variant not in ("js", "z"):
            raise _err("E-PDE-365", "weno_variant is 'js' (Jiang–Shu) or 'z' (Borges et al.)")
        self.fv_splitting = splitting
        self.fv_variant = variant
        self.fv_u0 = {u.idx: js.to_jets(u.expr) for u in self.unknowns}
        if flux is None:
            per_unknown = self._auto_flux()
        else:
            if not isinstance(flux, dict) or not flux:
                raise _err("E-PDE-373", "flux= must be a dict {unknown: flux}")
            per_unknown = {}
            for key, val in flux.items():
                ui = self._unknown_of(key)
                u = self.unknowns[ui]
                if u.time_order != 1:
                    raise _err(
                        "E-PDE-373",
                        f"{u.expr} is not first order in time; only u_t + div F(u) = ... "
                        "takes a conservative flux",
                    )
                per = self._flux_per_dim(val, f"flux of {u.expr}")
                per_unknown[ui] = [
                    None if F is None else self._fv_jets(F, f"the flux of {u.expr}") for F in per
                ]
        for ui, per in per_unknown.items():
            u = self.unknowns[ui]
            f = u.fields[0]
            g, _ = ctx["rhs"][f]
            dims = [k for k, F in enumerate(per) if F is not None]
            if not dims:
                continue
            terms = [js.total_derivative(per[k], self.dims[k].var) for k in dims]
            div = pool.add(terms) if len(terms) > 1 else terms[0]
            rest = ak.simplify(g + div).value
            for _sym, _fld, sa in _jet_map(ctx, rest):
                if sum(sa) == 1 and sa.index(1) in dims:
                    raise _err(
                        "E-PDE-373",
                        f"the equation for {u.expr} is not u_t + div F = (no first "
                        f"derivatives): with this flux the remainder is {rest}",
                        "include every advective term in the flux, or use scheme='weno5'",
                    )
            self.fv[f] = {"unknown": ui, "F": per, "dims": dims, "rest": rest, "phi": {}}
        if not self.fv:
            raise _err("E-PDE-373", "flux= gives no flux in any space dimension")
        # Wave speeds per dimension: the splitting constant α.
        fvf = list(self.fv)
        self.fv_speed = {}
        user = None if wave_speed is None else self._flux_per_dim(wave_speed, "wave_speed")
        for k in range(D):
            if not any(k in s["dims"] for s in self.fv.values()):
                continue
            if user is not None and user[k] is not None:
                s = self._fv_jets(user[k], "wave_speed")
            elif splitting == "lax_friedrichs":
                raise _err(
                    "E-PDE-373",
                    "flux_splitting='lax_friedrichs' needs a constant wave_speed= "
                    "(a bound on the largest |eigenvalue| of the flux Jacobian)",
                    "pass wave_speed=, or use flux_splitting='rusanov' (local speeds)",
                )
            else:
                # Spectral radius of ∂F_k/∂U: |F'| for one field, else the
                # Gershgorin (max row sum) bound — never below it.
                rows = []
                for f in fvf:
                    F = self.fv[f]["F"][k]
                    if F is None:
                        continue
                    row = []
                    for f2 in fvf:
                        dv = _diff(F, self.fv_u0[self.fv[f2]["unknown"]])
                        if not _is_zero(dv):
                            row.append(ak.abs(dv))
                    if row:
                        rows.append(pool.add(row) if len(row) > 1 else row[0])
                s = rows[0] if rows else pool.integer(0)
                for r in rows[1:]:
                    s = _max2(pool, s, r)
            if splitting == "lax_friedrichs":
                v = _num(s)
                if v is None:
                    raise _err(
                        "E-PDE-373",
                        "flux_splitting='lax_friedrichs' needs a numeric wave_speed",
                    )
                s = _const(pool, _exact(v) if _exact(v) is not None else v)
            self.fv_speed[k] = s

    def _fv_sub(self, b, k, o):
        """Jets of every unknown's value → its builder input at offset ``o``
        along dim ``k``."""
        off = [0] * len(self.dims)
        off[k] = o
        return {s: b.field(self.unknowns[ui].fields[0], off) for ui, s in self.fv_u0.items()}

    def _weno_recon(self, v):
        """WENO reconstruction at ``i+1/2`` biased towards ``v[len // 2]``:
        ``v`` holds 5 values (WENO-5), 3 (WENO-3) or 1 (first order)."""
        pool = self.pool
        if len(v) == 1:
            return v[0]
        r = pool.rational
        two = pool.integer(2)
        eps = pool.float(self.weno_eps)
        if len(v) == 3:
            a, b_, c = v
            q = [r(-1, 2) * a + r(3, 2) * b_, r(1, 2) * b_ + r(1, 2) * c]
            beta = [(b_ - a) ** two, (c - b_) ** two]
            d = [r(1, 3), r(2, 3)]
            tau = _ak().abs(beta[0] - beta[1])
        else:
            v1, v2, v3, v4, v5 = v
            q = [
                r(1, 3) * v1 - r(7, 6) * v2 + r(11, 6) * v3,
                r(-1, 6) * v2 + r(5, 6) * v3 + r(1, 3) * v4,
                r(1, 3) * v3 + r(5, 6) * v4 - r(1, 6) * v5,
            ]
            four, three = pool.integer(4), pool.integer(3)
            beta = [
                r(13, 12) * (v1 - two * v2 + v3) ** two
                + r(1, 4) * (v1 - four * v2 + three * v3) ** two,
                r(13, 12) * (v2 - two * v3 + v4) ** two + r(1, 4) * (v2 - v4) ** two,
                r(13, 12) * (v3 - two * v4 + v5) ** two
                + r(1, 4) * (three * v3 - four * v4 + v5) ** two,
            ]
            d = [r(1, 10), r(6, 10), r(3, 10)]
            tau = _ak().abs(beta[0] - beta[2])
        m1 = pool.integer(-1)
        if self.fv_variant == "js":
            al = [dk * (eps + bk) ** pool.integer(-2) for dk, bk in zip(d, beta)]
        else:
            al = [
                dk * (pool.integer(1) + (tau * (bk + eps) ** m1) ** two) for dk, bk in zip(d, beta)
            ]
        num = pool.add([a_ * qk for a_, qk in zip(al, q)])
        return num * pool.add(al) ** m1

    def _emit_fv_fluxes(self):
        """Fill one auxiliary field per (conservative field, dimension) with
        the numerical flux ``F̂_{j+1/2}`` (stored at index ``j``): WENO-5
        reconstructions of the split fluxes ``F± = (F ± α u)/2`` (Shu–Osher
        conservative form), WENO-3 and then first order where the stencil
        would leave a non-periodic grid."""
        if not self.fv:
            return
        pool = self.pool
        D = len(self.dims)
        half = pool.rational(1, 2)
        plus = {5: [-2, -1, 0, 1, 2], 3: [-1, 0, 1], 1: [0]}
        minus = {5: [3, 2, 1, 0, -1], 3: [2, 1, 0], 1: [1]}
        for f, spec in self.fv.items():
            for k in spec["dims"]:
                d = self.dims[k]
                new = len(self.fields)
                self.fields.append((None, f"flux{new}"))
                self.field_names.append(f"_flux_{self.field_names[f]}_{d.name}")
                spec["phi"][k] = new
                n = d.n
                if d.periodic:
                    classes = [(-1, n, 5, 5)]
                else:
                    classes = []
                    for j in range(n - 1):
                        po = 5 if j >= 2 and j + 2 <= n - 1 else 3 if j >= 1 else 1
                        mo = 5 if j >= 1 and j + 3 <= n - 1 else 3 if j + 2 <= n - 1 else 1
                        if classes and classes[-1][2:] == (po, mo) and classes[-1][1] == j:
                            classes[-1] = (classes[-1][0], j + 1, po, mo)
                        else:
                            classes.append((j, j + 1, po, mo))
                F = spec["F"][k]
                for j0, j1, po, mo in classes:
                    b = _Builder(self)
                    P, M = plus[po], minus[mo]
                    offs = sorted(set(P) | set(M))
                    sub = {o: self._fv_sub(b, k, o) for o in offs}
                    fv_ = {o: _subs(F, sub[o]) for o in offs}
                    uv = {}
                    for o in offs:
                        off = [0] * D
                        off[k] = o
                        uv[o] = b.field(f, off)
                    s = self.fv_speed[k]
                    if self.fv_splitting == "rusanov":
                        alpha = None
                        for o in offs:
                            so = _subs(s, sub[o])
                            alpha = so if alpha is None else _max2(pool, alpha, so)
                    else:
                        alpha = s
                    fp = [half * (fv_[o] + alpha * uv[o]) for o in P]
                    fm = [half * (fv_[o] - alpha * uv[o]) for o in M]
                    expr = self._weno_recon(fp) + self._weno_recon(fm)
                    ranges = [(j0, j1) if e == k else (0, self.dims[e].n) for e in range(D)]
                    self._emit(b, expr, ("fill", new), ranges, "flux")

    def _fv_rhs(self, b, f, cat, avail, ranges):
        """``-Σ_k (F̂_{j+1/2} - F̂_{j-1/2}) / h_k`` plus the discretised
        remainder; at a non-periodic face without a boundary condition the
        outer interface carries the physical flux (transmissive)."""
        pool = self.pool
        spec = self.fv[f]
        D = len(self.dims)
        terms = []
        for k in spec["dims"]:
            d = self.dims[k]
            phi = spec["phi"][k]
            h = d.h
            inv_h = _const(pool, 1 / Fraction(h)) if isinstance(h, Fraction) else pool.float(1 / h)
            off_l = [0] * D
            off_l[k] = -1

            def physical(k=k):
                return _subs(spec["F"][k], self._fv_sub(b, k, 0))

            right = physical() if (not d.periodic and cat[k] == 2) else b.field(phi, [0] * D)
            left = physical() if (not d.periodic and cat[k] == 0) else b.field(phi, off_l)
            terms.append((right - left) * inv_h)
        div = pool.add(terms) if len(terms) > 1 else terms[0]
        rest = spec["rest"]
        if _is_zero(rest):
            return pool.integer(-1) * div
        return self._discretise(b, rest, avail, self._irange_of(ranges)) - div

    # -- ops ---------------------------------------------------------------------

    def _cat_ranges(self, k, cat):
        d = self.dims[k]
        if d.periodic:
            return (0, d.n)
        return {0: (0, 1), 1: (1, d.n - 1), 2: (d.n - 1, d.n)}[cat]

    def _boxes(self, cat):
        """Split the category box into availability classes: list of
        (index ranges, _Avail)."""
        per = []
        for k, d in enumerate(self.dims):
            a, bnd = self._cat_ranges(k, cat[k])
            if d.periodic or d.kind != "fd":
                per.append([(a, bnd, (None, None))])
            elif cat[k] != 1:
                i = a
                per.append([(a, bnd, (-i, d.n - 1 - i))])
            else:
                per.append(_class_ranges(d, a, bnd, lambda lo, hi, k=k: self._signature(k, lo, hi)))
        out = []
        for combo in _iproduct(*per):
            out.append(([(c[0], c[1]) for c in combo], _Avail([c[2] for c in combo])))
        return out

    def _region(self, ranges):
        return [(a + d.pad, b + d.pad) for (a, b), d in zip(ranges, self.dims)]

    def _emit(self, b, expr, target, ranges, kind):
        """Append a kernel op (with its Jacobian kernel)."""
        pool = self.pool
        if not b.syms:
            # A constant kernel still needs one input: read time.
            b.time()
        op = {
            "kind": "kernel",
            "expr": expr,
            "syms": list(b.syms),
            "inputs": list(b.inputs),
            "target": target,
            "region": self._region(ranges),
            "role": kind,
        }
        if self.want_jac:
            jac, jin = [], []
            for i, (inp, s) in enumerate(zip(b.inputs, b.syms)):
                if inp[0] != "field":
                    continue
                dv = _diff(expr, s)
                if not _is_zero(dv):
                    jac.append(dv)
                    jin.append(i)
            op["jac"] = jac if jac else None
            op["jac_inputs"] = jin
        self.ops.append(op)
        _ = pool
        return op

    def _build_ops(self):
        D = len(self.dims)
        cats = list(_iproduct(*[(1,) if d.periodic else (0, 1, 2) for d in self.dims]))
        n_unknown_fields = len(self.fields)
        # 1. roles per field/category
        plan = []
        for f in range(n_unknown_fields):
            for cat in cats:
                role, owner = self.role(f, cat)
                plan.append((f, cat, role, owner))
        # States, in field-major, row-major order.
        np = _np()
        # Category of every grid index per dim (0 lo face, 1 interior, 2 hi).
        cat_axes = []
        for d in self.dims:
            c = np.ones(d.n, dtype=np.int64)
            if not d.periodic:
                c[0], c[-1] = 0, 2
            cat_axes.append(c)
        grids = np.meshgrid(*[np.arange(d.n) for d in self.dims], indexing="ij")
        cats_grid = np.meshgrid(*cat_axes, indexing="ij")
        flat_all = sum((g + d.pad) * s for g, d, s in zip(grids, self.dims, self.strides)).ravel()
        code_all = sum(c * 3**k for k, c in enumerate(cats_grid)).ravel()
        s_fields, s_flats, s_alg = [], [], []
        for f in range(n_unknown_fields):
            algebraic_field = self.ctx["rhs"].get(f, (None, False))[1]
            keep = np.zeros(3 ** len(self.dims), dtype=bool)
            alg = np.zeros(3 ** len(self.dims), dtype=bool)
            for f2, cat, role, _owner in plan:
                if f2 == f and role in ("pde", "alg"):
                    code = sum(c * 3**k for k, c in enumerate(cat))
                    keep[code] = True
                    alg[code] = role == "alg" or algebraic_field
            m = keep[code_all]
            s_flats.append(flat_all[m])
            s_fields.append(np.full(int(m.sum()), f, dtype=np.int64))
            s_alg.append(alg[code_all][m])
        self.state_field = np.concatenate(s_fields) if s_fields else np.zeros(0, np.int64)
        self.state_flat = np.concatenate(s_flats) if s_flats else np.zeros(0, np.int64)
        self.state_alg = np.concatenate(s_alg) if s_alg else np.zeros(0, bool)
        # 2. periodic halo copies (for every field incl. velocity fields)
        halo_ops = []
        for f in range(n_unknown_fields):
            for k, d in enumerate(self.dims):
                if not d.pad:
                    continue
                for side in (0, 1):
                    rng = []
                    for e in range(D):
                        if e == k:
                            rng.append((0, d.pad) if side == 0 else (d.pad + d.n, 2 * d.pad + d.n))
                        else:
                            rng.append((0, self.shape[e]))
                    off = [0] * D
                    off[k] = d.n if side == 0 else -d.n
                    halo_ops.append((f, rng, off))
        self._halo_ops = halo_ops

        def emit_halos():
            for f, rng, off in halo_ops:
                b = _Builder(self)
                v = b.field(f, off)
                op = self._emit(
                    b,
                    v,
                    ("fill", f),
                    [(a - dd.pad, c - dd.pad) for (a, c), dd in zip(rng, self.dims)],
                    "halo",
                )
                op["region"] = rng

        emit_halos()
        # 3. boundary fills, ordered by owner dim
        fills = [(f, cat, owner) for f, cat, role, owner in plan if role in ("fill", "fill_dt")]
        fills.sort(key=lambda x: (x[2][0], x[0]))
        for f, cat, owner in fills:
            self._emit_fill(f, cat, owner)
        if any(d.pad for d in self.dims) and fills:
            emit_halos()
        # 3b. conservative WENO interface fluxes (scheme="weno5_fv")
        self._emit_fv_fluxes()
        # 4. rhs kernels (dense aux ops get inserted on demand before these)
        for f, cat, role, owner in plan:
            if role == "pde":
                self._emit_pde(f, cat)
            elif role == "alg":
                self._emit_alg_bc(f, cat, owner)
        # Move dense ops to just after the fills: they were appended lazily.
        dense = [o for o in self.ops if o["kind"] == "dense"]
        rest = [o for o in self.ops if o["kind"] != "dense"]
        first_rhs = next((i for i, o in enumerate(rest) if o["target"][0] == "rhs"), len(rest))
        self.ops = rest[:first_rhs] + dense + rest[first_rhs:]

    def _irange_of(self, ranges):
        return lambda k: ranges[k]

    def _emit_pde(self, f, cat):
        g, _alg = self.ctx["rhs"][f]
        ui, lv = self.fields[f]
        u = self.unknowns[ui]
        for ranges, avail in self._boxes(cat):
            b = _Builder(self)
            if g is None:
                # u' = u_t: the next level's value.
                expr = b.field(u.fields[lv + 1], [0] * len(self.dims))
            elif f in self.fv:
                expr = self._fv_rhs(b, f, cat, avail, ranges)
            else:
                expr = self._discretise(b, g, avail, self._irange_of(ranges))
            self._emit(b, expr, ("rhs", f), ranges, "pde")

    def _bc_disc(self, b, bc, f, avail, ranges):
        """Discretise a BC residual at the face points (normal derivatives
        one-sided into the domain)."""
        return self._discretise(
            b, bc.expr, avail, self._irange_of(ranges), boundary=(bc.dim, bc.side)
        )

    def _emit_alg_bc(self, f, cat, owner):
        bc = self.face_bc[(f, owner[0], owner[1])]
        for ranges, avail in self._boxes(cat):
            b = _Builder(self)
            expr = self._bc_residual(b, bc, f, avail, ranges)
            self._emit(b, expr, ("rhs", f), ranges, "bc")

    def _emit_fill(self, f, cat, owner):
        pool = self.pool
        ui, lv = self.fields[f]
        if lv > 0:
            # Velocity boundary value: d/dt of the position fill.
            base = self.unknowns[ui].fields[lv - 1]
            for ranges, avail in self._boxes(cat):
                b0 = _Builder(self)
                expr0, _ = self._fill_expr(b0, base, cat, owner, ranges, avail)
                b = _Builder(self)
                acc = []
                for inp, s in zip(b0.inputs, b0.syms):
                    if inp[0] == "field":
                        dv = _diff(expr0, s)
                        if not _is_zero(dv):
                            vel = self.unknowns[ui].fields[lv]
                            acc.append(dv * b.field(vel, inp[2]))
                dt = _diff(expr0, b0.time()) if ("time",) in b0.index else None
                if dt is not None and not _is_zero(dt):
                    acc.append(dt)
                expr = pool.add(acc) if len(acc) > 1 else acc[0] if acc else pool.integer(0)
                # Re-home the remaining symbols of expr0's partials into b.
                expr = self._rehome(expr, b0, b)
                self._emit(b, expr, ("fill", f), ranges, "fill")
            return
        for ranges, avail in self._boxes(cat):
            b = _Builder(self)
            expr, _ = self._fill_expr(b, f, cat, owner, ranges, avail)
            self._emit(b, expr, ("fill", f), ranges, "fill")

    def _rehome(self, expr, b0, b):
        rep = {}
        for inp, s in zip(b0.inputs, b0.syms):
            if inp[0] == "field":
                rep[s] = b.field(inp[1], inp[2])
            elif inp[0] == "coef":
                rep[s] = b.coef(inp[1])
            else:
                rep[s] = b.time()
        return _subs(expr, rep)

    def _coords_time(self, b, out):
        cs = {}
        names = expr_free_symbols(out)
        for k, d in enumerate(self.dims):
            if d.name in names:
                cs[d.var] = b.coef(self.coord_coef(k))
        if str(self.tvar) in names:
            cs[self.tvar] = b.time()
        return _subs(out, cs)

    def _bc_cheb(self, b, bc, row, base):
        """A BC residual on a Chebyshev face: a normal derivative of order m
        is row ``row`` of ``D^m`` applied to the whole line; offsets are
        relative to grid index ``base`` along the face's dimension."""
        pool = self.pool
        k = bc.dim
        d = self.dims[k]
        D = len(self.dims)
        rep = {}
        for sym, fld, sa in _jet_map(self.ctx, bc.expr):
            if any(m for j, m in enumerate(sa) if j != k):
                raise _err(
                    "E-PDE-363",
                    f"boundary condition {bc.raw[0]} = {bc.raw[1]} has a tangential derivative; "
                    "not supported with Chebyshev collocation",
                )
            m = sa[k]
            if m == 0:
                off = [0] * D
                off[k] = row - base
                rep[sym] = b.field(fld, off)
                continue
            M = _spectral_matrix(d, m)
            terms = []
            for j in range(d.n):
                if M[row, j] != 0.0:
                    off = [0] * D
                    off[k] = j - base
                    terms.append(pool.float(float(M[row, j])) * b.field(fld, off))
            rep[sym] = pool.add(terms) if len(terms) > 1 else terms[0]
        return self._coords_time(b, _subs(bc.expr, rep))

    def _bc_residual(self, b, bc, f, avail, ranges):
        d = self.dims[bc.dim]
        if d.kind == "cheb":
            row = 0 if bc.side == 0 else d.n - 1
            return self._bc_cheb(b, bc, row, row)
        return self._bc_disc(b, bc, f, avail, ranges)

    def _fill_expr(self, b, f, cat, owner, ranges, avail):
        """The boundary value of field ``f`` at the face points, solved from
        the discretised BC (linear in that value)."""
        pool = self.pool
        D = len(self.dims)
        k, side = owner
        bc = self.face_bc[(f, k, side)]
        d = self.dims[k]
        zero = [0] * D
        for _, _, sa in _jet_map(self.ctx, bc.expr):
            if any(m for j, m in enumerate(sa) if j != k):
                raise _err(
                    "E-PDE-363",
                    f"boundary condition {bc.raw[0]} = {bc.raw[1]} has a tangential "
                    "derivative, which cannot be eliminated",
                    "use bc_mode='algebraic'",
                )
        nonlinear = _err(
            "E-PDE-363",
            f"boundary condition {bc.raw[0]} = {bc.raw[1]} cannot be solved for the "
            "boundary value (it is nonlinear in it, or does not involve it)",
            "use bc_mode='algebraic'",
        )
        if d.kind == "cheb":
            row = 0 if side == 0 else d.n - 1
            r0 = self._bc_cheb(b, bc, row, row)
            other = (f, k, 1 - side)
            other_bc = self.face_bc.get(other)
            x0 = b.field(f, zero)
            if other_bc is not None and self.status.get(other) == "fill":
                orow = d.n - 1 - row
                r1 = self._bc_cheb(b, other_bc, orow, row)
                off = [0] * D
                off[k] = orow - row
                x1 = b.field(f, off)
                a00, a01 = _diff(r0, x0), _diff(r0, x1)
                a10, a11 = _diff(r1, x0), _diff(r1, x1)
                if any(_contains(a, [x0, x1]) for a in (a00, a01, a10, a11)):
                    raise nonlinear
                z = {x0: pool.integer(0), x1: pool.integer(0)}
                c0, c1 = _subs(r0, z), _subs(r1, z)
                det = a00 * a11 - a01 * a10
                if _is_zero(det):
                    raise nonlinear
                return (a01 * c1 - a11 * c0) * det ** pool.integer(-1), x0
            r = r0
        else:
            r = self._bc_disc(b, bc, f, avail, ranges)
            x0 = b.field(f, zero)
        a = _diff(r, x0)
        if _is_zero(a) or _contains(a, [x0]):
            raise nonlinear
        c = _subs(r, {x0: pool.integer(0)})
        return (pool.integer(-1) * c) * a ** pool.integer(-1), x0


# ---------------------------------------------------------------------------
# Public objects
# ---------------------------------------------------------------------------


class MolResult:
    """Solution of :meth:`MolSystem.integrate`.

    Attributes
    ----------
    t : numpy.ndarray
        Output times.
    y : numpy.ndarray
        States, shape ``(len(t), n_states)``.
    stats : dict
        Work counters (steps, rejected steps, rhs/Jacobian evaluations,
        factorisations) and ``wall_time``.
    """

    def __init__(self, system, t, y, stats):
        self.system = system
        self.t = t
        self.y = y
        self.stats = stats

    def field(self, name=None, i=-1):
        """The grid values of an unknown at output ``i`` (boundary values
        included), shaped like :attr:`MolSystem.grid`."""
        return self.system.field(self.y[i], name, t=float(self.t[i]))

    def __repr__(self):
        return (
            f"MolResult(t=[{self.t[0]}, ..., {self.t[-1]}] ({len(self.t)}), "
            f"n_states={self.y.shape[1]})"
        )


class MolSystem:
    """A semi-discrete PDE: ``M y' = f(t, y)`` with compiled stencil kernels.

    Built by :func:`method_of_lines`. Attributes: :attr:`grid` (node
    coordinates per space variable), :attr:`n_states`, :attr:`y0`,
    :attr:`t0`, :attr:`algebraic` (mass-matrix zeros: algebraic boundary
    rows), :attr:`compile_time` / :attr:`discretize_time` (seconds).
    """

    def __init__(self, sys, program, compile_time):
        np = _np()
        self._sys = sys
        self._prog = program
        self.compile_time = compile_time
        self.discretize_time = sys.discretize_time
        self.n_states = program.n_states
        self.grid = {d.name: np.asarray(d.nodes) for d in sys.dims}
        self.algebraic = np.asarray(sys.state_alg, dtype=bool)
        self.fields = list(sys.field_names)
        self.scheme = sys.scheme
        self.order = sys.order
        self.bc_mode = sys.bc_mode
        self.n_kernels = sum(1 for o in sys.ops if o["kind"] == "kernel")
        self._state_flat = sys.state_flat
        self._state_field = sys.state_field
        self.t0, self.y0 = self._initial_state()
        self._stability = None
        self._jax = None

    # -- evaluation --------------------------------------------------------------

    def rhs(self, t, y, out=None):
        """``f(t, y)`` (one native pass of the stencil kernels)."""
        np = _np()
        y = np.ascontiguousarray(y, dtype=float)
        if out is None:
            out = np.empty(self.n_states)
        self._prog.rhs_into(float(t), y, out)
        return out

    def jacobian(self, t, y, format="csr"):
        """The exact sparse Jacobian ``∂f/∂y`` (compiled symbolic partials of
        each kernel, composed through the boundary closures). ``format`` is
        ``"csr"`` (a :class:`scipy.sparse.csr_matrix`, needs SciPy),
        ``"dense"`` or ``"raw"`` (``(indptr, indices, data)``)."""
        np = _np()
        if not self._sys.want_jac:
            raise _err("E-PDE-372", "this system was built with jacobian=False")
        bp, bi, bd = self._prog.jacobian(float(t), np.ascontiguousarray(y, dtype=float))
        indptr = np.frombuffer(bp, dtype="<u8").astype(np.int64)
        indices = np.frombuffer(bi, dtype="<u8").astype(np.int64)
        data = np.frombuffer(bd, dtype="<f8").copy()
        n = self.n_states
        if format == "raw":
            return indptr, indices, data
        if format == "dense":
            m = np.zeros((n, n))
            for r in range(n):
                m[r, indices[indptr[r] : indptr[r + 1]]] = data[indptr[r] : indptr[r + 1]]
            return m
        if format == "csr":
            try:
                import scipy.sparse as sp
            except ImportError as exc:
                raise _err(
                    "E-PDE-371", "format='csr' needs SciPy; use format='raw' or 'dense'"
                ) from exc
            return sp.csr_matrix((data, indices, indptr), shape=(n, n))
        raise ValueError("format is 'csr', 'dense' or 'raw'")

    def field(self, y, name=None, t=None):
        """Grid values of unknown ``name`` (default: the first) for state
        ``y``, boundary nodes included, shape ``(n_x[, n_y[, n_z]])``."""
        np = _np()
        sys = self._sys
        if name is None:
            f = 0
        else:
            if name not in self.fields:
                raise KeyError(f"no field {name!r}; fields are {self.fields}")
            f = self.fields.index(name)
        t = self.t0 if t is None else t
        full = self._prog.fields(float(t), np.ascontiguousarray(y, dtype=float))
        arr = np.frombuffer(full[f], dtype="<f8").reshape(sys.shape)
        sl = tuple(slice(d.pad, d.pad + d.n) for d in sys.dims)
        return arr[sl].copy()

    def state_coordinates(self):
        """Coordinates of every state component: ``(n_states, dim)``."""
        np = _np()
        sys = self._sys
        out = np.empty((self.n_states, len(sys.dims)))
        idx = np.unravel_index(sys.state_flat, sys.shape)
        for k, d in enumerate(sys.dims):
            out[:, k] = np.asarray(d.nodes)[idx[k] - d.pad]
        return out

    def state_fields(self):
        """Field name of every state component."""
        return [self.fields[f] for f in self._state_field]

    def _initial_state(self):
        np = _np()
        ak = _ak()
        sys = self._sys
        ctx = sys.ctx
        pool = sys.pool
        ics = {}
        t0 = None
        for lhs, rhs in sys.problem.ics:
            bv = _boundary_value(lhs, ctx)
            if bv is None or bv[0] != "time":
                n = lhs.node()
                raise _err("E-PDE-364", f"initial condition {lhs} = {rhs} is not u(x, t0) = g(x)")
            _, ui, tval = bv
            tv = _num(tval)
            if tv is None:
                raise _err("E-PDE-364", f"initial time {tval} is not a number")
            if t0 is not None and abs(tv - t0) > 1e-14:
                raise _err("E-PDE-364", "initial conditions are given at different times")
            t0 = float(tv)
            n = lhs.node()
            level = 0
            if n[1] == "Derivative":
                orders = [int(o.node()[1]) for o in n[2][1:]]
                u = sys.unknowns[ui]
                for slot, o in enumerate(orders):
                    var = u.args[slot]
                    if o and var != sys.tvar:
                        raise _err("E-PDE-364", f"initial condition {lhs}: space derivative")
                    if var == sys.tvar:
                        level = o
            ics[(ui, level)] = _subs(rhs, ctx["psubs"])
        coords = self.state_coordinates()
        y0 = np.zeros(self.n_states)
        for f, (ui, lv) in enumerate(sys.fields):
            if ui is None:
                continue
            mask = self._state_field == f
            if not mask.any():
                continue
            g = ics.get((ui, lv))
            u = sys.unknowns[ui]
            if g is None:
                if u.time_order == 0:
                    continue  # algebraic: consistent initialisation solves it
                raise _err(
                    "E-PDE-364",
                    f"no initial condition for {u.expr}"
                    + (" (and its time derivative)" if lv else ""),
                )
            names = expr_free_symbols(g)
            extra = set(names) - {d.name for d in sys.dims}
            if extra:
                raise _err("E-PDE-366", f"initial condition {g} has free symbols {sorted(extra)}")
            vars_ = [d.var for d in sys.dims]
            fn = ak.compile_expr(g, vars_)
            cols = [np.ascontiguousarray(coords[mask, k]) for k in range(len(sys.dims))]
            y0[mask] = ak.numpy_eval(fn, *cols)
        if t0 is None:
            raise _err("E-PDE-364", "the problem has no initial condition")
        _ = pool
        return t0, y0

    # -- integration -------------------------------------------------------------

    def integrate(
        self,
        t_span,
        method="bdf",
        *,
        t_eval=None,
        rtol=1e-6,
        atol=1e-9,
        dt=None,
        y0=None,
        max_step=math.inf,
        max_steps=10_000_000,
        max_order=5,
        linear_solver="auto",
        threads=None,
        gmres_restart=None,
        gmres_max_iters=None,
    ):
        """Integrate from :attr:`t0`.

        ``t_span`` is the end time or ``(t0, t1)``; ``t_eval`` the output
        times (default ``[t1]``). ``method``:

        * ``"bdf"`` — native variable-order BDF (1–5) with the exact sparse
          Jacobian (banded LU after reverse Cuthill–McKee); the stiff default,
          and the only one for DAE systems (``bc_mode="algebraic"``);
        * ``"rk45"`` — native Dormand–Prince 5(4);
        * ``"euler"``, ``"rk4"`` — fixed step ``dt``; checked against the
          proved von Neumann condition (:meth:`stability_report`), with a
          :class:`MolStabilityWarning` and the stable ``dt`` when it is violated;
        * ``"alkahest_rk45"`` — :func:`alkahest.experimental.ode_integrate_rk45`
          on :meth:`to_ode` (one expression per state: small grids only);
        * ``"scipy_bdf"``, ``"scipy_radau"``, ``"scipy_rk45"``, ... —
          :func:`scipy.integrate.solve_ivp` with :meth:`rhs` and the sparse
          :meth:`jacobian`.

        ``linear_solver`` (native BDF) is ``"auto"`` (default: the banded
        direct solver for small or narrow-band systems, ILU(0)-preconditioned
        restarted GMRES on the exact sparse Jacobian for large 2-D/3-D grids),
        ``"direct"`` or ``"krylov"``; ``gmres_restart`` / ``gmres_max_iters``
        tune GMRES. ``threads`` runs a native method on a dedicated pool of
        that many threads (default: all cores, or ``RAYON_NUM_THREADS``); the
        result is bitwise identical for every thread count. ``stats`` reports
        ``linear_solver``, ``linear_solves``, ``krylov_iters``,
        ``krylov_failures``, ``preconditioner_setups`` and
        ``direct_fallbacks``.
        """
        np = _np()
        if linear_solver not in ("auto", "direct", "krylov"):
            raise _err(
                "E-PDE-372",
                f"linear_solver is 'auto', 'direct' or 'krylov', not {linear_solver!r}",
            )
        if threads is not None and (int(threads) != threads or threads < 1):
            raise _err("E-PDE-372", f"threads must be a positive integer, not {threads!r}")
        if isinstance(t_span, (tuple, list)):
            t0, t1 = float(t_span[0]), float(t_span[1])
        else:
            t0, t1 = self.t0, float(t_span)
        if t_eval is None:
            t_eval = np.array([t1])
        t_eval = np.ascontiguousarray(t_eval, dtype=float)
        y_init = self.y0 if y0 is None else np.ascontiguousarray(y0, dtype=float)
        start = _time.perf_counter()
        if method in ("euler", "rk4"):
            if dt is None or not dt > 0:
                raise _err("E-PDE-372", f"method={method!r} needs a time step dt > 0")
            self._check_dt(method, dt)
        if method in ("bdf", "rk45", "euler", "rk4"):
            if method != "bdf" and self.algebraic.any():
                raise _err(
                    "E-PDE-369",
                    "this system has algebraic boundary rows (a DAE); integrate it "
                    "with method='bdf'",
                )
            try:
                tt, yy, stats, band = self._prog.integrate(
                    t0,
                    y_init,
                    t_eval,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    h0=dt,
                    max_step=max_step,
                    max_steps=max_steps,
                    max_order=max_order,
                    linear_solver=linear_solver,
                    threads=None if threads is None else int(threads),
                    gmres_restart=gmres_restart,
                    gmres_max_iters=gmres_max_iters,
                )
            except RuntimeError as exc:
                msg = str(exc)
                if "index > 1" in msg:
                    raise self._index_error(msg) from exc
                raise _err("E-PDE-368", f"integration failed: {msg}") from exc
            y = np.frombuffer(yy, dtype="<f8").reshape(len(tt), self.n_states).copy()
            stats = dict(stats)
            stats["band"] = band
        elif method == "alkahest_rk45":
            tt, y, stats = self._alkahest_rk45(t0, t_eval, y_init, rtol, atol)
        elif method.startswith("scipy_"):
            tt, y, stats = self._scipy(method[6:], t0, t_eval, y_init, rtol, atol, max_step)
        else:
            raise _err("E-PDE-372", f"unknown method {method!r}")
        stats["wall_time"] = _time.perf_counter() - start
        return MolResult(self, np.asarray(tt, dtype=float), y, stats)

    def _index_error(self, msg):
        detail = ""
        if self.n_states <= 400:
            try:
                ak = _ak()
                red = ak.dae_index_reduce(self.to_dae())
                detail = f"; dae_index_reduce: {red!r}"
            except Exception as e:  # pragma: no cover - diagnostic only
                detail = f"; dae_index_reduce refused: {e}"
        return _err(
            "E-PDE-369",
            "the boundary constraints do not determine the algebraic boundary values "
            f"(DAE index > 1){detail}",
            "use bc_mode='eliminate', or reformulate the boundary condition",
        )

    def _scipy(self, name, t0, t_eval, y0, rtol, atol, max_step):
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:
            raise _err("E-PDE-371", "the scipy_* methods need SciPy") from exc
        if self.algebraic.any():
            raise _err("E-PDE-369", "solve_ivp does not integrate DAEs; use method='bdf'")
        methods = {
            "bdf": "BDF",
            "radau": "Radau",
            "lsoda": "LSODA",
            "rk45": "RK45",
            "dop853": "DOP853",
        }
        if name not in methods:
            raise _err("E-PDE-372", f"unknown SciPy method {name!r}")
        kw: dict[str, Any] = {}
        if methods[name] in ("BDF", "Radau", "LSODA") and self._sys.want_jac:
            kw["jac"] = lambda t, y: self.jacobian(t, y, format="csr")
        nfev = [0]

        def f(t, y):
            nfev[0] += 1
            return self.rhs(t, y)

        sol = solve_ivp(
            f,
            (t0, float(t_eval[-1])),
            y0,
            method=methods[name],
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            **kw,
        )
        if not sol.success:
            raise _err("E-PDE-368", f"solve_ivp failed: {sol.message}")
        return (
            sol.t,
            sol.y.T.copy(),
            {"rhs_evals": int(sol.nfev), "jac_evals": int(sol.njev), "steps": len(sol.t)},
        )

    def _alkahest_rk45(self, t0, t_eval, y0, rtol, atol):
        np = _np()
        ak = _ak()
        from alkahest.experimental import ode_integrate_rk45

        ode = self.to_ode()
        ys = []
        t_prev = t0
        y = list(y0)
        out_t = []
        for te in t_eval:
            if te > t_prev:
                traj = ode_integrate_rk45(
                    ode, y, t_prev, float(te), rtol=rtol, atol=atol, h_max=float(te - t_prev)
                )
                y = list(traj.y_final())
                t_prev = float(te)
            ys.append(list(y))
            out_t.append(te)
        _ = ak
        return np.array(out_t), np.array(ys), {}

    # -- symbolic forms ------------------------------------------------------------

    def _pointwise(self, max_states):
        """One symbolic expression per state (substituting the fills), for
        small systems only."""
        ak = _ak()
        sys = self._sys
        pool = sys.pool
        if self.n_states > max_states:
            raise _err(
                "E-PDE-372",
                f"to_ode/to_dae build one expression per state; {self.n_states} states "
                f"exceed the limit of {max_states}",
                "coarsen the grid (the compiled stencil kernels have no such limit)",
            )
        np = _np()
        ys = [pool.symbol(f"y{i}") for i in range(self.n_states)]
        tsym = sys.tvar
        node = {}
        pairs = list(zip(sys.state_field.tolist(), sys.state_flat.tolist()))
        for i, (f, p) in enumerate(pairs):
            node[(f, p)] = ys[i]
        rhs = [None] * self.n_states
        state_index = {(f, p): i for i, (f, p) in enumerate(pairs)}
        strides = sys.strides

        def shift(off):
            return sum(o * s for o, s in zip(off, strides))

        def points(region):
            for idx in _iproduct(*[range(a, b) for a, b in region]):
                yield sum(i * s for i, s in zip(idx, strides))

        for op in sys.ops:
            if op["kind"] == "dense":
                k = op["axis"]
                M = np.asarray(op["matrix"]).reshape(-1, op["src_len"])
                st = strides[k]
                rlo = op["region"][k][0]
                for p in points(op["region"]):
                    i = (p // st) % sys.shape[k] - rlo
                    base = p - (rlo + i) * st
                    acc = []
                    for j in range(op["src_len"]):
                        q = base + (op["src_lo"] + j) * st
                        v = node.get((op["src"], q))
                        if v is not None and M[i, j] != 0:
                            acc.append(pool.float(float(M[i, j])) * v)
                    node[(op["dst"], p)] = pool.add(acc) if acc else pool.integer(0)
                continue
            for p in points(op["region"]):
                rep = {}
                for inp, s in zip(op["inputs"], op["syms"]):
                    if inp[0] == "field":
                        q = p + shift(inp[2])
                        rep[s] = node.get((inp[1], q), pool.integer(0))
                    elif inp[0] == "coef":
                        rep[s] = pool.float(float(sys.coefs[inp[1]][p]))
                    else:
                        rep[s] = tsym
                val = ak.subs(op["expr"], rep)
                kind, f = op["target"]
                if kind == "rhs":
                    rhs[state_index[(f, p)]] = val
                else:
                    node[(f, p)] = val
        return ys, rhs

    def to_ode(self, max_states=2000):
        """The semi-discrete system as an :class:`alkahest.ODE` (one
        expression per state; for cross-checks on small grids)."""
        ak = _ak()
        if self.algebraic.any():
            raise _err("E-PDE-369", "a DAE system; use to_dae()")
        ys, rhs = self._pointwise(max_states)
        return ak.ODE(ys, rhs, self._sys.tvar)

    def to_dae(self, max_states=2000):
        """The semi-discrete system as an :class:`alkahest.DAE` (algebraic
        boundary rows as constraints), for :func:`alkahest.pantelides` /
        :func:`alkahest.dae_index_reduce`."""
        ak = _ak()
        pool = self._sys.pool
        ys, rhs = self._pointwise(max_states)
        dys = [pool.symbol(f"dy{i}") for i in range(len(ys))]
        eqs = []
        diff_vars, diff_d, alg_vars = [], [], []
        for i, r in enumerate(rhs):
            if self.algebraic[i]:
                eqs.append(r)
            else:
                eqs.append(dys[i] - r)
        order = [i for i in range(len(ys)) if not self.algebraic[i]] + [
            i for i in range(len(ys)) if self.algebraic[i]
        ]
        for i in order:
            if self.algebraic[i]:
                alg_vars.append(ys[i])
            else:
                diff_vars.append(ys[i])
                diff_d.append(dys[i])
        return ak.DAE.new(eqs, diff_vars + alg_vars, diff_d, self._sys.tvar)

    # -- stability -----------------------------------------------------------------

    def stability_report(self):
        """Von Neumann analysis of the linear constant-coefficient part of
        each scalar equation, discretised as generated (interior stencil),
        for the fixed-step methods: a dict per field with the scheme's
        amplification factor and the proved condition on ``dt`` for explicit
        Euler and classical RK4, plus the largest stable ``dt``.

        One space dimension is decided exactly (``proof_status`` from
        :func:`von_neumann`). In 2-D/3-D the diagonal Fourier modes
        ``θ_x = θ_y (= θ_z)`` are analysed, which gives a *necessary*
        condition (``proof_status = "necessary"``).
        """
        if self._stability is None:
            from ._mol_stability import stability_report

            self._stability = stability_report(self)
        return self._stability

    def _check_dt(self, method, dt):
        try:
            rep = self.stability_report()
        except NumericPdeError:
            return
        for name, entry in rep.items():
            m = entry.get("methods", {}).get(method)
            if not m or m.get("max_dt") is None:
                continue
            if dt > m["max_dt"] * (1 + 1e-12):
                warnings.warn(
                    MolStabilityWarning(
                        f"dt = {dt:g} violates the {m['proof_status']} von Neumann condition "
                        f"{m['condition']} for {name} with {method}; a stable step is "
                        f"dt <= {m['max_dt']:.6g}"
                    ),
                    stacklevel=3,
                )

    # -- export ----------------------------------------------------------------------

    def to_jax(self):
        """A pure-JAX ``f(t, y)`` (gathers plus the kernel expressions as
        ``jax.numpy``), traceable by ``jax.jit`` / ``jax.grad`` and lowerable
        to StableHLO."""
        if self._jax is None:
            from ._mol_jax import build_jax_rhs

            self._jax = build_jax_rhs(self)
        return self._jax

    def to_stablehlo(self):
        """StableHLO (MLIR text) of the right-hand side, lowered through
        :meth:`to_jax` and ``jax.jit``."""
        np = _np()
        f = self.to_jax()
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover
            raise _err("E-PDE-370", "to_stablehlo needs JAX") from exc
        lowered = jax.jit(f).lower(jnp.asarray(0.0), np.asarray(self.y0))
        return lowered.as_text()

    def __repr__(self):
        g = " × ".join(f"{k}:{v.size}" for k, v in self.grid.items())
        dae = ", DAE" if self.algebraic.any() else ""
        return (
            f"MolSystem({self.n_states} states, grid {g}, scheme={self.scheme}, "
            f"order={self.order}, {self.n_kernels} kernels{dae})"
        )


def method_of_lines(
    problem,
    grid,
    *,
    scheme="central",
    order=2,
    bc_mode="eliminate",
    params=None,
    jacobian=True,
    weno_eps=1e-6,
    flux=None,
    wave_speed=None,
    flux_splitting="rusanov",
    weno_variant="js",
):
    """Discretise a PDE problem in space and compile the semi-discrete system.

    Parameters
    ----------
    problem : alkahest.experimental.PDE
        ``PDE(eqs, unknowns).on(domain).bc(...).ic(...)`` with a domain of
        1–3 intervals (``Interval``, ``Box``, products), one time variable,
        each unknown's equation solvable for ``u_t`` (or ``u_tt``) — or free
        of time derivatives (an algebraic field). Periodicity is a BC
        ``u(lo, t) = u(hi, t)``.
    grid : int or dict
        Points per space variable (``{x: 101, y: 65}``; an int for all), or
        a strictly increasing node array per variable for a stretched grid.
        Non-periodic counts include both boundary nodes; periodic counts
        exclude the duplicate end point.
    scheme : str
        ``"central"`` (default), ``"upwind"`` (advective first-derivative
        terms biased by the sign of their coefficient), ``"weno5"`` (the
        same, with Jiang–Peng WENO-5 derivatives; uniform grids),
        ``"chebyshev"`` (Gauss–Lobatto collocation), ``"fourier"``
        (periodic collocation) or ``"weno5_fv"`` — a **conservative** WENO-5
        flux (Shu–Osher flux-difference form, uniform grids) for
        conservation laws ``u_t + div F(u) = rest``: the semi-discrete update
        is ``-(F̂_{j+1/2} - F̂_{j-1/2}) / h``, with ``F̂`` the WENO-5
        reconstructions of the split fluxes ``(F ± α u) / 2``, so the discrete
        total ``Σ h u_j`` changes only through the boundary fluxes (exactly
        conserved, to rounding, on a periodic grid). ``rest`` (diffusion,
        sources) uses central stencils of ``order``.
    order : int
        Accuracy order of the finite-difference stencils (any; the stencils
        are exact Fornberg weights, one-sided near boundaries).
    bc_mode : str
        ``"eliminate"`` (boundary values solved from the discrete BC — must be
        linear in the boundary value) or ``"algebraic"`` (boundary nodes are
        states with the BC as an algebraic equation: a DAE, integrated by
        the BDF method).
    params : dict
        Values of the problem's free symbols.
    jacobian : bool
        Also compile the exact sparse Jacobian kernels (default ``True``).
    weno_eps : float
        The ``ε`` of the WENO smoothness weights.
    flux : dict, optional
        ``scheme="weno5_fv"``: the flux of every conservative unknown,
        ``{u(x, t): F}`` in 1-D or ``{u(x, y, t): {x: F, y: G}}`` (a list in
        domain order also works), each a function of the unknowns' values.
        The equation minus ``-div F`` must leave no first derivative along a
        flux direction (checked symbolically; ``E-PDE-373`` otherwise).
        Default: for a single scalar equation ``u_t + a(u) u_x = ...`` the
        flux ``∫ a du``, verified by differentiation.
    wave_speed : expression, number or dict, optional
        The splitting constant: a bound on the spectral radius of ``∂F/∂u``
        (per space variable as for ``flux``). Default: ``|F'(u)|`` for one
        unknown, the Gershgorin bound ``max_i Σ_j |∂F_i/∂u_j|`` for a system.
    flux_splitting : str
        ``"rusanov"`` (default; local Lax–Friedrichs: ``α`` is the largest
        wave speed over the six cells of each interface stencil) or
        ``"lax_friedrichs"`` (global: ``α = wave_speed``, a number).
    weno_variant : str
        ``"js"`` (Jiang–Shu weights, default) or ``"z"`` (Borges et al.
        WENO-Z, which keeps fifth order at critical points).

    Returns
    -------
    MolSystem

    Raises
    ------
    NumericPdeError
        ``E-PDE-359 … E-PDE-373`` (see :class:`alkahest.exceptions.NumericPdeError`).
    """
    sys = _System(
        problem,
        grid,
        scheme,
        order,
        bc_mode,
        params,
        jacobian,
        weno_eps,
        flux=flux,
        wave_speed=wave_speed,
        flux_splitting=flux_splitting,
        weno_variant=weno_variant,
    )
    from alkahest.alkahest import _MolProgram

    t0 = _time.perf_counter()
    ops = []
    for o in sys.ops:
        o2 = {k: v for k, v in o.items() if k != "role"}
        ops.append(o2)
    try:
        prog = _MolProgram(
            list(sys.shape),
            len(sys.fields),
            list(sys.coefs),
            sys.state_field.tolist(),
            sys.state_flat.tolist(),
            sys.state_alg.tolist(),
            ops,
        )
    except ValueError as exc:
        raise _err("E-PDE-367", f"building the native program failed: {exc}") from exc
    except Exception as exc:
        if type(exc).__name__ in ("JitError",):
            raise _err("E-PDE-367", f"kernel compilation failed: {exc}") from exc
        raise
    compile_time = _time.perf_counter() - t0
    return MolSystem(sys, prog, compile_time)
