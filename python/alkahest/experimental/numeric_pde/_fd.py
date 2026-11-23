"""Exact finite-difference stencils: Fornberg weights, truncation errors,
multi-dimensional (tensor-product and compact) stencils, and kernel hooks."""

from __future__ import annotations

from fractions import Fraction
from typing import Any
from itertools import product as _iproduct
from math import factorial

from ...exceptions import NumericPdeError
from ._exact import _num_of, frac_to_expr

__all__ = [
    "FDStencil",
    "Stencil",
    "fd_weights",
    "laplacian_stencil",
    "stencil",
    "truncation_error",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pool(*candidates):
    import alkahest

    for c in candidates:
        if c is not None and hasattr(c, "_pool"):
            return c._pool
    p = alkahest.active_pool()
    return p if p is not None else alkahest.default_pool()


def _as_exact(v):
    """Fraction for an exact number (int, Fraction, rational Expr, str), else
    the Expr itself; floats are refused."""
    if isinstance(v, bool):
        raise NumericPdeError(f"invalid offset {v!r}", code="E-PDE-350")
    if isinstance(v, int | Fraction):
        return Fraction(v)
    if isinstance(v, float):
        raise NumericPdeError(
            f"floating-point offset {v!r}: pass an exact rational "
            "(fractions.Fraction or pool.rational) so the weights are exact",
            code="E-PDE-351",
        )
    if isinstance(v, str):
        try:
            return Fraction(v)
        except ValueError as exc:
            raise NumericPdeError(f"invalid offset {v!r}", code="E-PDE-350") from exc
    if hasattr(v, "node"):
        q = _num_of(v.node())
        if q is not None:
            return q
        if v.node()[0] == "float":
            raise NumericPdeError(
                f"floating-point offset {v}: pass an exact rational", code="E-PDE-351"
            )
        return v
    raise NumericPdeError(f"invalid offset {v!r}", code="E-PDE-350")


def _cancel(e):
    import alkahest

    return alkahest.cancel(e)


def _is_zero_expr(e):
    n = _cancel(e).node()
    return n[0] == "integer" and n[1] == "0"


def _to_expr(pool, v):
    if isinstance(v, Fraction | int):
        return frac_to_expr(pool, v)
    return v


def _shift(i, s):
    """Canonical ``i + s`` (``j + 0`` is ``j``), so equal grid values are equal Exprs."""
    import alkahest

    return alkahest.simplify(i + s).value


def _nth_derivative(u, x, n):
    import alkahest

    out = u
    for _ in range(n):
        out = alkahest.diff(out, x).value
    return out


def _mixed_derivative(u, xs, alpha):
    out = u
    for x, a in zip(xs, alpha):
        out = _nth_derivative(out, x, a)
    return out


def _default_function(pool, dim, u, vars):
    names = ("x", "y", "z", "w")
    if vars is None:
        if u is not None:
            n = u.node()
            if n[0] == "func":
                vars = list(n[2])
        if vars is None:
            vars = [pool.symbol(names[i] if i < 4 else f"x{i}") for i in range(dim)]
    vars = list(vars)
    if len(vars) != dim:
        raise NumericPdeError(
            f"expected {dim} independent variable(s), got {len(vars)}", code="E-PDE-350"
        )
    if u is None:
        u = pool.func("u", vars)
    elif isinstance(u, str):
        u = pool.func(u, vars)
    return u, vars


# ---------------------------------------------------------------------------
# Fornberg's algorithm
# ---------------------------------------------------------------------------


def fornberg(m, nodes, x0=0):
    """Weights ``C[k][j]`` for derivative orders ``k = 0..m`` at ``x0`` on
    ``nodes`` (Fornberg 1988). Works over any field: :class:`Fraction` for
    rational nodes, alkahest ``Expr`` for symbolic ones."""
    n = len(nodes)
    zero = nodes[0] * 0
    C = [[zero for _ in range(n)] for _ in range(m + 1)]
    c1 = zero + 1
    c4 = nodes[0] - x0
    C[0][0] = zero + 1
    for i in range(1, n):
        mn = min(i, m)
        c2 = zero + 1
        c5 = c4
        c4 = nodes[i] - x0
        for j in range(i):
            c3 = nodes[i] - nodes[j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[k][i] = c1 * (k * C[k - 1][i - 1] - c5 * C[k][i - 1]) / c2
                C[0][i] = -c1 * c5 * C[0][i - 1] / c2
            for k in range(mn, 0, -1):
                C[k][j] = (c4 * C[k][j] - k * C[k - 1][j]) / c3
            C[0][j] = c4 * C[0][j] / c3
        c1 = c2
    return C


# ---------------------------------------------------------------------------
# 1-D stencils
# ---------------------------------------------------------------------------


class FDStencil:
    """A one-dimensional finite-difference approximation

    ``u^{(m)}(x) ≈ h^{-m} Σ_j w_j u(x + s_j h)``

    Attributes
    ----------
    order : int
        The derivative order ``m``.
    offsets : tuple
        The offsets ``s_j`` in units of ``h`` (rationals, or ``Expr`` for a
        symbolic non-uniform grid).
    weights : tuple
        The dimensionless weights ``w_j`` (exact :class:`Fraction` or
        ``Expr``).
    h : Expr
        The grid-spacing symbol.
    error_order : int or None
        The order of accuracy ``p``; ``None`` when the stencil is exact for
        every smooth function (only possible for ``m = 0`` at a node).
    error_constant : Fraction or Expr
        ``C`` in the leading truncation error
        ``τ = u^{(m)}(x) − h^{-m} Σ w_j u(x + s_j h) = C h^p u^{(m+p)}(x) + O(h^{p+1})``.
        For the central second difference ``C = −1/12``.
    """

    def __init__(self, order, offsets, weights, h, error_order, error_constant, pool):
        self.order = order
        self.offsets = tuple(offsets)
        self.weights = tuple(weights)
        self.h = h
        self.error_order = error_order
        self.error_constant = error_constant
        self._pool = pool

    # -- presentation ---------------------------------------------------------
    @property
    def dim(self):
        return 1

    @property
    def exact(self):
        return all(isinstance(w, Fraction) for w in self.weights)

    @property
    def weights_expr(self):
        """The weights ``w_j / h^m`` as alkahest expressions."""
        p = self._pool
        scale = self.h ** (-self.order) if self.order else None
        out = []
        for w in self.weights:
            e = _to_expr(p, w)
            out.append(_cancel(e * scale) if scale is not None else e)
        return out

    def __repr__(self):
        ws = ", ".join(str(w) for w in self.weights)
        offs = ", ".join(str(s) for s in self.offsets)
        return (
            f"FDStencil(order={self.order}, offsets=[{offs}], weights=[{ws}], "
            f"error_order={self.error_order}, error_constant={self.error_constant})"
        )

    def as_dict(self):
        return dict(zip(self.offsets, self.weights))

    def leading_error(self, u=None, x=None):
        """``(C, p, m + p)`` for ``τ = C h^p u^{(m+p)} + O(h^{p+1})``."""
        if self.error_order is None:
            return (Fraction(0), None, None)
        return (self.error_constant, self.error_order, self.order + self.error_order)

    def truncation_error(self, u=None, x=None):
        """The leading truncation error ``C h^p u^{(m+p)}(x)`` as an ``Expr``.

        Sign convention: ``τ = exact − approximation``. ``u`` is an applied
        function such as ``u(x)`` (or a name); the default is ``u(x)``.
        """
        p = self._pool
        u, (x,) = _default_function(p, 1, u, None if x is None else [x])
        if self.error_order is None:
            return p.integer(0)
        c = _to_expr(p, self.error_constant)
        d = _nth_derivative(u, x, self.order + self.error_order)
        return _cancel(c * self.h**self.error_order) * d

    # -- application / codegen ------------------------------------------------
    def apply(self, u, index=None):
        """``h^{-m} Σ w_j u(index + s_j)`` as an ``Expr``.

        ``u`` is a function name (``"u"`` gives applied ``u(j + s)``) or a
        callable taking the offset and returning an ``Expr``.
        """
        p = self._pool
        if isinstance(u, str):
            name = u
            if index is None:
                index = p.symbol("j")

            def u(s, _n=name, _j=index):
                return p.func(_n, [_shift(_j, _to_expr(p, s))])

        terms = [_to_expr(p, w) * u(s) for s, w in zip(self.offsets, self.weights) if w != 0]
        tot = p.add(terms) if len(terms) > 1 else terms[0]
        return tot * self.h ** (-self.order) if self.order else tot

    def apply_function(self, f, x):
        """Apply the stencil to a smooth expression ``f(x)``: ``h^{-m} Σ w_j f(x + s_j h)``."""
        import alkahest

        p = self._pool
        terms = [
            _to_expr(p, w) * alkahest.subs(f, {x: x + _to_expr(p, s) * self.h})
            for s, w in zip(self.offsets, self.weights)
            if w != 0
        ]
        tot = p.add(terms) if len(terms) > 1 else terms[0]
        return tot * self.h ** (-self.order) if self.order else tot

    def _as_multi(self):
        return Stencil(
            1,
            {(s,): w for s, w in zip(self.offsets, self.weights)},
            self.order,
            {(self.order,): Fraction(1)},
            self.h,
            self._pool,
        )

    def kernel(self, name="u"):
        """A compiled kernel; see :meth:`Stencil.kernel`."""
        return self._as_multi().kernel(name)

    def emit_c(self, fn_name="stencil", name="u"):
        """C99 source of a function evaluating the stencil from neighbour values."""
        return self._as_multi().emit_c(fn_name, name)


def _xpow(w, s, n):
    """``w * s**n`` where either may be an Expr (``s**0`` is 1 even for Expr)."""
    if n == 0:
        return w
    if isinstance(s, Fraction):
        return w * s**n
    return w * s**n if n > 1 else w * s


def _moments_error(order, offsets, weights, exact):
    """Leading-error data from the moments ``M_n = Σ w_j s_j^n / n!``."""
    n_pts = len(offsets)
    cap = 2 * n_pts + order + 3
    for n in range(cap + 1):
        if exact:
            M = sum((w * s**n for s, w in zip(offsets, weights)), Fraction(0)) / factorial(n)
            target = Fraction(1) if n == order else Fraction(0)
            if target != M:
                if n <= order:
                    raise AssertionError("Fornberg weights failed their moment check")
                return n - order, -M
        else:
            # weights/offsets are Expr here (at least one of each pair)
            acc: Any = None
            for s, w in zip(offsets, weights):
                t = _xpow(w, s, n)
                acc = t if acc is None else acc + t
            if n == order:
                acc = acc - factorial(n)
            M = _cancel(acc)
            if not _is_zero_expr(M):
                if n <= order:
                    raise AssertionError("Fornberg weights failed their moment check")
                pool = M._pool
                val = _cancel(pool.integer(-1) * M * pool.rational(1, factorial(n)))
                return n - order, val
    return None, Fraction(0)


def fd_weights(order, offsets, h=None, x0=0):
    """Exact finite-difference weights by Fornberg's algorithm.

    Parameters
    ----------
    order : int
        Derivative order ``m ≥ 0``.
    offsets : sequence
        Stencil offsets in units of ``h``: integers (uniform grid), halves
        (staggered grid), one-sided sets such as ``[0, 1, 2]``, arbitrary
        rationals, or symbolic ``Expr`` (non-uniform grid).
    h : Expr, optional
        The spacing symbol (default: a symbol ``h``).
    x0 : number or Expr
        Evaluation point relative to the offsets (default 0).

    Returns
    -------
    FDStencil
        Weights plus the leading truncation error ``C h^p u^{(m+p)}`` with the
        convention ``τ = exact − approximation``.

    Raises
    ------
    NumericPdeError
        ``E-PDE-350`` for repeated offsets, fewer than ``m + 1`` points, or a
        negative order; ``E-PDE-351`` for floating-point offsets.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise NumericPdeError(f"derivative order must be an integer >= 0, got {order!r}")
    offs = [_as_exact(s) for s in offsets]
    x0e = _as_exact(x0)
    if len(offs) < order + 1:
        raise NumericPdeError(
            f"a derivative of order {order} needs at least {order + 1} points, got {len(offs)}",
            code="E-PDE-350",
        )
    exact = all(isinstance(s, Fraction) for s in offs) and isinstance(x0e, Fraction)
    pool = _pool(h, *[s for s in offs if not isinstance(s, Fraction)], x0e)
    if h is None:
        h = pool.symbol("h")
    if exact:
        if len(set(offs)) != len(offs):
            raise NumericPdeError(f"repeated stencil offsets in {offsets}", code="E-PDE-350")
        rel = [s - x0e for s in offs]
        C = fornberg(order, offs, x0e)
        w = C[order]
    else:
        nodes = [_to_expr(pool, s) for s in offs]
        x0x = _to_expr(pool, x0e)
        for i in range(len(nodes)):
            for j in range(i):
                if _is_zero_expr(nodes[i] - nodes[j]):
                    raise NumericPdeError(
                        f"repeated stencil offsets in {offsets}", code="E-PDE-350"
                    )
        C = fornberg(order, nodes, x0x)
        w = [_cancel(c) for c in C[order]]
        rel = [_cancel(s - x0x) for s in nodes]
    p, cst = _moments_error(order, rel, w, exact)
    return FDStencil(order, offs, w, h, p, cst, pool)


def _central_offsets(order, accuracy):
    if accuracy % 2:
        raise NumericPdeError(
            "a central stencil has even order of accuracy; pass an even accuracy",
            code="E-PDE-350",
        )
    npts = 2 * ((order + 1) // 2) - 1 + accuracy
    half = npts // 2
    return list(range(-half, half + 1))


# ---------------------------------------------------------------------------
# multi-dimensional stencils
# ---------------------------------------------------------------------------


def _multinomial_power(k, n):
    """Expansion of ``(Σ_i k_i D_i)^n / n!`` as {multi-index: Fraction}."""
    d = len(k)
    out = {}

    def rec(i, remaining, alpha, coef):
        if i == d - 1:
            a = (*alpha, remaining)
            c = coef * k[i] ** remaining / factorial(remaining)
            if c:
                out[a] = out.get(a, 0) + c
            return
        for a in range(remaining + 1):
            c = coef * k[i] ** a / factorial(a) if a else coef
            if c:
                rec(i + 1, remaining - a, (*alpha, a), c)

    rec(0, n, (), Fraction(1))
    return {a: c for a, c in out.items() if c}


class Stencil:
    """A d-dimensional stencil ``L_h u(x) = h^{-m} Σ_k w_k u(x + k h)``
    approximating the constant-coefficient operator ``Σ_α t_α ∂^α`` of order
    ``m``.

    Attributes
    ----------
    dim : int
    weights : dict
        ``{offset tuple: Fraction}``.
    order : int
        ``m``, the order of the target operator (power of ``h`` divided out).
    target : dict
        ``{multi-index α: t_α}``.
    error_order : int or None
    error_terms : dict
        ``{α: e_α}`` with ``τ = exact − approx = h^p Σ_α e_α ∂^α u + O(h^{p+1})``.
    """

    def __init__(self, dim, weights, order, target, h, pool, name=None):
        self.dim = dim
        self.weights = {tuple(Fraction(x) for x in k): Fraction(w) for k, w in weights.items() if w}
        self.order = order
        self.target = {tuple(a): Fraction(c) for a, c in target.items() if c}
        self.h = h
        self._pool = pool
        self.name = name
        self.error_order, self.error_terms = self._analyse()

    def __repr__(self):
        nm = f"{self.name}, " if self.name else ""
        return (
            f"Stencil({nm}dim={self.dim}, points={len(self.weights)}, order={self.order}, "
            f"error_order={self.error_order})"
        )

    def symbol_part(self, n):
        """Degree-``n`` part ``Σ_k w_k (k·D)^n / n!`` of the stencil's symbol."""
        acc = {}
        for k, w in self.weights.items():
            for a, c in _multinomial_power(k, n).items():
                acc[a] = acc.get(a, 0) + w * c
        return {a: c for a, c in acc.items() if c}

    def _analyse(self):
        m = self.order
        for n in range(m + 1):
            part = self.symbol_part(n)
            want = self.target if n == m else {}
            if part != want:
                raise NumericPdeError(
                    f"stencil is not consistent with its target operator at degree {n}",
                    code="E-PDE-350",
                )
        maxoff = max((max((abs(x) for x in k), default=0) for k in self.weights), default=0)
        cap = m + 2 * len(self.weights) + 4
        if maxoff == 0:
            cap = m + 1
        for n in range(m + 1, cap + 1):
            part = self.symbol_part(n)
            if part:
                return n - m, {a: -c for a, c in part.items()}
        return None, {}

    def truncation_error(self, u=None, vars=None):
        """Leading truncation error ``h^p Σ_α e_α ∂^α u`` (``exact − approx``)."""
        p = self._pool
        u, xs = _default_function(p, self.dim, u, vars)
        if self.error_order is None:
            return p.integer(0)
        terms = []
        for a, c in sorted(self.error_terms.items(), reverse=True):
            terms.append(frac_to_expr(p, c) * _mixed_derivative(u, xs, a))
        tot = p.add(terms) if len(terms) > 1 else terms[0]
        return self.h**self.error_order * tot

    def apply(self, u, index=None):
        """``h^{-m} Σ w_k u(index + k)`` as an ``Expr``; ``u`` is a name or a
        callable taking the offset tuple."""
        p = self._pool
        if isinstance(u, str):
            name = u
            if index is None:
                index = [p.symbol(n) for n in ("i", "j", "k", "l")[: self.dim]]
            index = list(index)

            def u(off, _n=name, _ix=index):
                return p.func(_n, [_shift(i, frac_to_expr(p, s)) for i, s in zip(_ix, off)])

        terms = [frac_to_expr(p, w) * u(k) for k, w in sorted(self.weights.items())]
        tot = p.add(terms) if len(terms) > 1 else terms[0]
        return tot * self.h ** (-self.order) if self.order else tot

    # -- codegen --------------------------------------------------------------
    def _value_symbols(self, name):
        p = self._pool
        syms = []
        offs = sorted(self.weights)
        for k in offs:
            tag = "_".join(
                (f"m{-x}" if x < 0 else f"p{x}").replace("/", "o") if x != 0 else "0" for x in k
            )
            syms.append(p.symbol(f"{name}_{tag}"))
        return offs, syms

    def _kernel_expr(self, name):
        p = self._pool
        offs, syms = self._value_symbols(name)
        terms = [frac_to_expr(p, self.weights[k]) * s for k, s in zip(offs, syms)]
        tot = p.add(terms) if len(terms) > 1 else terms[0]
        expr = tot * self.h ** (-self.order) if self.order else tot
        return expr, offs, syms

    def kernel(self, name="u"):
        """Compile the stencil with :func:`alkahest.compile_expr`.

        Returns a :class:`StencilKernel` that applies the stencil to every
        interior point of a NumPy array (integer offsets only).
        """
        import alkahest

        for k in self.weights:
            if any(x.denominator != 1 for x in k):
                raise NumericPdeError(
                    "kernel() needs integer offsets; a staggered stencil maps between "
                    "two grids — use apply() and compile the expression yourself",
                    code="E-PDE-358",
                )
        expr, offs, syms = self._kernel_expr(name)
        try:
            fn = alkahest.compile_expr(expr, [*list(syms), self.h])
        except Exception as exc:  # pragma: no cover - JIT always has an interpreter tier
            raise NumericPdeError(f"kernel compilation failed: {exc}", code="E-PDE-358") from exc
        return StencilKernel(self, fn, expr, offs, syms)

    def emit_c(self, fn_name="stencil", name="u"):
        """C99 source for ``double fn(neighbour values…, double h)`` via
        :func:`alkahest.emit_c_expr`."""
        import alkahest

        expr, _offs, syms = self._kernel_expr(name)
        return alkahest.emit_c_expr(expr, [*list(syms), self.h], fn_name=fn_name)


class StencilKernel:
    """A stencil compiled to native code. Call it with a NumPy array of grid
    values (``dim`` axes) and the spacing ``h``; it returns the stencil applied
    at every point whose neighbours all exist (the array shrinks by the
    stencil's reach on each side)."""

    def __init__(self, stencil, fn, expr, offsets, symbols):
        self.stencil = stencil
        self.compiled = fn
        self.expr = expr
        self.offsets = [tuple(int(x) for x in k) for k in offsets]
        self.symbols = symbols

    def __call__(self, values, h):
        import numpy as np

        import alkahest

        values = np.asarray(values, dtype=float)
        d = self.stencil.dim
        if values.ndim != d:
            raise NumericPdeError(
                f"expected a {d}-dimensional array, got {values.ndim}", code="E-PDE-358"
            )
        lo = [min(k[i] for k in self.offsets) for i in range(d)]
        hi = [max(k[i] for k in self.offsets) for i in range(d)]
        shape = [values.shape[i] - (hi[i] - lo[i]) for i in range(d)]
        if any(s <= 0 for s in shape):
            raise NumericPdeError("grid too small for the stencil", code="E-PDE-358")
        cols = []
        for k in self.offsets:
            sl = tuple(slice(k[i] - lo[i], k[i] - lo[i] + shape[i]) for i in range(d))
            cols.append(np.ascontiguousarray(values[sl]).ravel())
        hs = np.full(cols[0].shape, float(h))
        out = alkahest.numpy_eval(self.compiled, *cols, hs)
        return np.asarray(out).reshape(shape)


def _tensor(stencils, h, pool):
    weights = {}
    for combo in _iproduct(*[list(zip(s.offsets, s.weights)) for s in stencils]):
        k = tuple(c[0] for c in combo)
        w = Fraction(1)
        for c in combo:
            w *= c[1]
        if w:
            weights[k] = weights.get(k, 0) + w
    order = sum(s.order for s in stencils)
    alpha = tuple(s.order for s in stencils)
    return Stencil(len(stencils), weights, order, {alpha: 1}, h, pool, name=f"d{alpha}")


def _laplacian_target(dim):
    return {tuple(2 if i == j else 0 for i in range(dim)): 1 for j in range(dim)}


_LAPLACIANS = {
    # (dim, points): (denominator, {class: weight})  class = number of non-zero coords
    (1, 3): (1, {0: -2, 1: 1}),
    (2, 5): (1, {0: -4, 1: 1}),
    (2, 9): (6, {0: -20, 1: 4, 2: 1}),  # compact / Mehrstellen
    (3, 7): (1, {0: -6, 1: 1}),
    (3, 19): (6, {0: -24, 1: 2, 2: 1}),  # Mehrstellen
    (3, 27): (30, {0: -128, 1: 14, 2: 3, 3: 1}),  # compact 27-point
}


def laplacian_stencil(dim=2, points=None, h=None):
    """Standard and compact Laplacian stencils with their truncation errors.

    ``(dim, points)`` is one of ``(1, 3)``, ``(2, 5)``, ``(2, 9)`` (compact
    Mehrstellen), ``(3, 7)``, ``(3, 19)`` (Mehrstellen) or ``(3, 27)``.
    The compact stencils are second order for ``Δu`` but have leading error
    ``−h²/12 Δ²u``, so with the right-hand side corrected to
    ``f + h²/12 Δf`` they solve Poisson's equation to fourth order.
    """
    if points is None:
        points = {1: 3, 2: 5, 3: 7}.get(dim)
    key: Any = (dim, points)
    if key not in _LAPLACIANS:
        raise NumericPdeError(
            f"no {points}-point Laplacian in {dim} dimension(s); available: {sorted(_LAPLACIANS)}",
            code="E-PDE-350",
        )
    den, cls = _LAPLACIANS[key]
    pool = _pool(h)
    if h is None:
        h = pool.symbol("h")
    weights = {}
    for k in _iproduct((-1, 0, 1), repeat=dim):
        nz = sum(1 for x in k if x)
        if nz in cls:
            weights[k] = Fraction(cls[nz], den)
    name = f"laplacian-{points}pt" + (" (Mehrstellen)" if points in (9, 19, 27) else "")
    return Stencil(dim, weights, 2, _laplacian_target(dim), h, pool, name=name)


def stencil(
    derivative, offsets=None, *, accuracy=None, kind="central", h=None, dim=None, points=None
):
    """Build a finite-difference stencil.

    * ``stencil(m, offsets)`` — the 1-D stencil on the given offsets
      (same as :func:`fd_weights`).
    * ``stencil(m, accuracy=p, kind="central" | "forward" | "backward")`` —
      the smallest stencil of that kind with order of accuracy ``p``.
    * ``stencil((a1, a2, ...), offsets=[offs1, offs2, ...])`` or with
      ``accuracy=p`` — the tensor-product stencil for the mixed partial
      ``∂^α``, built from 1-D stencils per axis.
    * ``stencil("laplacian", dim=2, points=9)`` — see :func:`laplacian_stencil`.
    """
    if isinstance(derivative, str):
        if derivative.lower() != "laplacian":
            raise NumericPdeError(f"unknown stencil {derivative!r}", code="E-PDE-350")
        return laplacian_stencil(dim or 2, points, h=h)
    if isinstance(derivative, tuple | list):
        alpha = list(derivative)
        pool = _pool(h)
        if h is None:
            h = pool.symbol("h")
        if offsets is None:
            per = [stencil(a, accuracy=accuracy or 2, kind=kind, h=h) for a in alpha]
        else:
            if len(offsets) != len(alpha):
                raise NumericPdeError(
                    "give one offset list per axis of the multi-index", code="E-PDE-350"
                )
            per = [fd_weights(a, o, h=h) for a, o in zip(alpha, offsets)]
        if not all(s.exact for s in per):
            raise NumericPdeError("tensor-product stencils need rational offsets", code="E-PDE-350")
        return _tensor(per, h, pool)
    m = derivative
    if offsets is None:
        p = accuracy if accuracy is not None else 2
        if not isinstance(p, int) or p < 1:
            raise NumericPdeError("accuracy must be a positive integer", code="E-PDE-350")
        if kind == "central":
            offsets = _central_offsets(m, p)
        elif kind == "forward":
            offsets = list(range(m + p))
        elif kind == "backward":
            offsets = [-k for k in range(m + p)][::-1]
        else:
            raise NumericPdeError(f"unknown stencil kind {kind!r}", code="E-PDE-350")
    return fd_weights(m, offsets, h=h)


def truncation_error(st, u=None, vars=None):
    """Leading truncation error (``exact − approximation``) of a stencil as an
    ``Expr`` in derivatives of ``u``."""
    if isinstance(st, FDStencil):
        x = None if vars is None else (vars[0] if isinstance(vars, list | tuple) else vars)
        return st.truncation_error(u, x)
    return st.truncation_error(u, vars)
