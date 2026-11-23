"""alkahest.fex — a fast, exact evaluation engine for Finite Expression Method search.

**Experimental**: names and signatures may change between minor versions.

The Finite Expression Method (FEX; Liang & Yang, arXiv 2206.10121, JMLR 26,
2025) looks for a closed-form solution of a PDE inside a fixed binary tree of
operator *slots*. A candidate is an opcode vector ``e`` (one operator per
slot) plus parameters ``theta``; leaves are ``sum_i alpha_i * phi(x_i) + beta``
over all ``d`` coordinates and inner unary slots are ``alpha * op(s) + beta``.
Scoring a candidate means fitting ``theta`` to the PDE residual, and that
dominates the cost of a search. This module makes it fast and exact:

* :class:`Template` — the tree shape, operator tables and dimension;
  :meth:`Template.instantiate` turns a candidate into an ordinary
  :class:`~alkahest.Expr`.
* :class:`Problem` — the PDE: Poisson, linear conservation law (transport),
  cubic Schrödinger, Rayleigh-quotient eigenproblem, or any operator written
  as an expression in the jet symbols of a :class:`Jet`.
* :class:`Evaluator` — compiled once per (template, problem, points). It
  computes ``u``, ``grad u`` and ``lap u`` of every candidate exactly, with a
  fused Forward-Laplacian pass (O(d) per node, not ``d`` autograd passes),
  the PDE and boundary residuals, and the full Jacobian with respect to
  ``theta`` in the same pass, for a whole batch of candidates in parallel,
  in float64. Scoring a new candidate never compiles anything.
* :func:`fit` — Levenberg–Marquardt on the residual with the exact Jacobian
  (default), or Adam / BFGS (the paper's 20 + 20 coarse fit).
* :func:`canonical_key` — a normal form of a skeleton under the slot algebra
  (``0·x``, id chains, commutativity, ``x + 0·x²`` padding, absorbed
  constants), used to avoid scoring the same function family twice.
* :func:`search` — a small reference controller (risk-seeking policy
  gradient over per-slot categorical distributions, as in the paper) so a
  search runs end to end.

Candidates whose value overflows, divides by zero or takes ``log`` of a
non-positive number get loss ``+inf`` — never NaN. Malformed input raises
:class:`FexError` (``E-FEX-001`` … ``E-FEX-008``).

The dimension ``d`` is concrete. Arrays are NumPy ``float64`` (opcodes
``int64``); anything exposing ``__dlpack__`` is accepted as input.

Example
-------
>>> import numpy as np
>>> from alkahest import fex
>>> d = 5
>>> tpl = fex.Template("depth2_sub", dim=d)
>>> prob = fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(xi**2 for xi in x))
>>> rng = np.random.default_rng(0)
>>> ev = fex.Evaluator(tpl, prob, fex.sample_box(500, d, rng=rng),
...                    fex.sample_box_boundary(200, d, rng=rng))
>>> e = [3, 0, 0, 2]          # id(L[sq] + L[0]) = a * sum(x_i^2) + b
>>> res = fex.fit(ev, e, rng=rng)
>>> res.loss < 1e-20
True
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .alkahest import ExprPool, FexError, _FexEvaluator, _FexProblem, _FexTemplate

__all__ = [
    "PAPER_BINARY",
    "PAPER_UNARY",
    "Evaluator",
    "FexError",
    "FitResult",
    "Jet",
    "Problem",
    "SearchResult",
    "Template",
    "canonical_key",
    "fit",
    "sample_box",
    "sample_box_boundary",
    "sample_initial_slice",
    "search",
]

#: The FEX paper's unary table, in the reference implementation's order.
PAPER_UNARY = ["0", "1", "id", "sq", "cube", "quartic", "exp", "sin", "cos"]
#: The FEX paper's binary table, in the reference implementation's order.
PAPER_BINARY = ["+", "*", "-"]

_DEPTH_PRESETS = {1: "depth1", 2: "depth2", 3: "depth3"}


def _f64(a: Any) -> np.ndarray:
    if hasattr(a, "__dlpack__") and not isinstance(a, np.ndarray):
        a = np.from_dlpack(a)
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a: Any) -> np.ndarray:
    if hasattr(a, "__dlpack__") and not isinstance(a, np.ndarray):
        a = np.from_dlpack(a)
    return np.ascontiguousarray(a, dtype=np.int64)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class Template:
    """A FEX template: tree shape, operator tables and dimension.

    Parameters
    ----------
    shape : str
        ``L`` (leaf), ``U(child)`` and ``B(left,right)``, e.g. ``"U(B(L,L))"``,
        or a preset of the reference implementation: ``depth1``,
        ``depth2_sub``, ``depth2_rml``, ``depth2``, ``depth3``.
    unary, binary : list of str
        Operator tables. Unary: ``0 1 id sq cube quartic exp sin cos log abs``;
        binary: ``+ - * /``. Defaults are the paper's tables in the reference
        order, so the reference code's action vectors are opcode vectors here.
    dim : int
        Number of coordinates each leaf sums over (time included).

    Slots are numbered in-order (left subtree, node, right subtree). The
    parameters of each slot follow in the same order: ``d + 1`` per leaf
    (weights, then bias), ``2`` per inner unary slot (``alpha, beta``), none
    per binary slot.
    """

    def __init__(
        self,
        shape: str = "depth2_sub",
        unary: Sequence[str] = PAPER_UNARY,
        binary: Sequence[str] = PAPER_BINARY,
        dim: int = 1,
    ) -> None:
        self._native = _FexTemplate(shape, list(unary), list(binary), int(dim))

    @classmethod
    def binary_tree(
        cls,
        depth: int,
        unary: Sequence[str] = PAPER_UNARY,
        binary: Sequence[str] = PAPER_BINARY,
        dim: int = 1,
    ) -> Template:
        """The reference implementation's ``depthN`` tree (``N`` = 1, 2, 3)."""
        try:
            preset = _DEPTH_PRESETS[depth]
        except KeyError:
            raise ValueError("depth must be 1, 2 or 3") from None
        return cls(preset, unary, binary, dim)

    shape = property(lambda self: self._native.shape, doc="Canonical shape string.")
    dim = property(lambda self: self._native.dim, doc="Dimension d.")
    n_slots = property(lambda self: self._native.n_slots, doc="Opcode vector length.")
    n_params = property(lambda self: self._native.n_params, doc="Parameter vector length.")
    unary = property(lambda self: self._native.unary, doc="Unary table.")
    binary = property(lambda self: self._native.binary, doc="Binary table.")
    slot_arities = property(
        lambda self: self._native.slot_arities, doc="Table size per slot."
    )
    slot_kinds = property(
        lambda self: self._native.slot_kinds, doc="'leaf', 'unary' or 'binary' per slot."
    )
    param_offsets = property(
        lambda self: self._native.param_offsets, doc="First parameter index per slot."
    )
    n_skeletons = property(
        lambda self: self._native.n_skeletons, doc="Number of distinct opcode vectors."
    )

    def __repr__(self) -> str:
        return (
            f"Template({self.shape!r}, dim={self.dim}, n_slots={self.n_slots}, "
            f"n_params={self.n_params})"
        )

    def describe(self, e: Sequence[int]) -> str:
        """A readable rendering of skeleton ``e``, e.g. ``id((L[sq] + L[0]))``."""
        return self._native.describe([int(v) for v in e])

    def canonical_key(self, e: Sequence[int]) -> str:
        """See :func:`canonical_key`."""
        return self._native.canonical_key([int(v) for v in e])

    def redundant_params(self, e: Sequence[int]) -> list[float | None]:
        """Parameters that can be pinned (to the returned value) without
        shrinking the function family; ``None`` for free parameters."""
        return self._native.redundant_params([int(v) for v in e])

    def canonical_skeletons(self, limit: int = 1_000_000) -> list[tuple[str, list[int]]]:
        """One ``(key, e)`` per canonical family over every skeleton."""
        return self._native.canonical_skeletons(limit)

    def coords(self, pool: ExprPool | None = None) -> list[Any]:
        """Fresh coordinate symbols ``x0 … x{d-1}``."""
        pool = pool if pool is not None else ExprPool()
        return [pool.symbol(f"x{i}") for i in range(self.dim)]

    def instantiate(
        self, e: Sequence[int], theta: Sequence[float], coords: Sequence[Any] | None = None
    ) -> Any:
        """The candidate as an :class:`~alkahest.Expr` in ``coords``.

        Exactly-zero leaf weights are left out; every other parameter is a
        float.
        """
        coords = list(coords) if coords is not None else self.coords()
        return self._native.instantiate(
            [int(v) for v in e], [float(t) for t in np.asarray(theta).ravel()], coords
        )

    def instantiate_symbolic(
        self, e: Sequence[int], params: Sequence[Any], coords: Sequence[Any]
    ) -> Any:
        """The skeleton with each parameter replaced by the given expression."""
        return self._native.instantiate_symbolic([int(v) for v in e], list(params), list(coords))

    def same_function(
        self,
        e1: Sequence[int],
        theta1: Sequence[float],
        e2: Sequence[int],
        theta2: Sequence[float],
        coords: Sequence[Any] | None = None,
        max_den: int = 1000,
        tol: float = 1e-9,
    ) -> bool | None:
        """Decide whether two fitted candidates are the same function.

        Parameters are snapped to nearby rationals (denominator ``<= max_den``,
        within ``tol``), both candidates are instantiated exactly, and the
        difference goes to Alkahest's zero test. ``True`` is a proof, ``False``
        a proof of difference, ``None`` undecided.
        """
        coords = list(coords) if coords is not None else self.coords()
        return self._native.same_function(
            [int(v) for v in e1],
            [float(t) for t in theta1],
            [int(v) for v in e2],
            [float(t) for t in theta2],
            coords,
            max_den,
            tol,
        )

    def random_skeletons(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``n`` uniformly random opcode vectors, shape ``(n, n_slots)``."""
        rng = rng if rng is not None else np.random.default_rng()
        ar = self.slot_arities
        return np.stack([rng.integers(0, k, size=n) for k in ar], axis=1).astype(np.int64)

    def init_params(
        self, n: int = 1, rng: np.random.Generator | None = None, scale: float = 0.1
    ) -> np.ndarray:
        """``N(0, scale²)`` parameters, shape ``(n, n_params)`` (the reference
        implementation's ``reset_params``)."""
        rng = rng if rng is not None else np.random.default_rng()
        return rng.normal(0.0, scale, size=(n, self.n_params))


def canonical_key(template: Template, e: Sequence[int]) -> str:
    """Normal form of skeleton ``e`` under the slot algebra.

    Equal keys mean equal function families (as ``theta`` ranges over all real
    values): ``0`` and ``1`` slots are free constants, ``id`` over a leaf is the
    leaf, ``+`` and ``*`` are commutative, ``a - b`` is ``a + b`` when ``b`` can
    be negated, constants are absorbed by neighbouring free shifts and scales,
    ``cos`` of a shift-closed argument is ``sin``. Not every equivalence is
    found, but no two different families share a key.
    """
    return template.canonical_key(e)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


class Jet:
    """Symbols for writing a PDE operator: ``u``, ``lap`` (Δu over the spatial
    coordinates), ``grad[i]`` (∂u/∂x_i), ``ut`` (∂u/∂t, when ``time`` is set)
    and the coordinates ``x[i]``, all in one :class:`~alkahest.ExprPool`."""

    def __init__(self, dim: int, time: int | None = None, pool: ExprPool | None = None):
        self.pool = pool if pool is not None else ExprPool()
        self.dim = int(dim)
        self.time = time
        self.u = self.pool.symbol("u")
        self.lap = self.pool.symbol("lap_u")
        self.grad = [self.pool.symbol(f"u_x{i}") for i in range(self.dim)]
        self.x = [self.pool.symbol(f"x{i}") for i in range(self.dim)]
        self.ut = self.pool.symbol("u_t") if time is not None else None


def _source(v: Any, jet: Jet) -> Any:
    """A number, an Expr in jet.x, a callable of the coordinate list, or None."""
    if v is None:
        return None
    if callable(v):
        return v(jet.x)
    return v


class Problem:
    """A PDE problem for FEX scoring: interior operator ``F(u) = rhs``,
    boundary data ``u = bc``, boundary weight ``bc_weight`` (λ, the paper uses
    100), and a loss.

    Source terms (``rhs``, ``bc``, ``potential``) are numbers, expressions in
    the problem's coordinates (``problem.x``), or callables taking the list of
    coordinate symbols and returning an expression.

    The least-squares loss is ``mean(r_int²) + λ·mean(r_bd²)`` — the reference
    implementation's ``mse(pde) + 100·mse(bd)``.
    """

    def __init__(self, native: _FexProblem, jet: Jet, exact: Callable | None = None):
        self._native = native
        self.jet = jet
        self.exact = exact

    dim = property(lambda self: self._native.dim)
    time = property(lambda self: self._native.time)
    loss = property(lambda self: self._native.loss)
    order = property(lambda self: self._native.order, doc="Highest jet order the operator reads.")
    operator_kind = property(lambda self: self._native.operator_kind)

    @property
    def x(self) -> list[Any]:
        """The coordinate symbols."""
        return self.jet.x

    @classmethod
    def _make(cls, dim, kind, jet, rhs, bc, potential, exact, **kw) -> Problem:
        native = _FexProblem(
            dim,
            kind,
            rhs=_source(rhs, jet),
            bc=_source(bc, jet),
            potential=_source(potential, jet),
            coords=jet.x,
            **kw,
        )
        return cls(native, jet, exact)

    @classmethod
    def poisson(
        cls,
        dim: int,
        rhs: Any = 0.0,
        bc: Any = None,
        *,
        bc_weight: float = 100.0,
        exact: Callable | None = None,
        jet: Jet | None = None,
    ) -> Problem:
        """``−Δu = rhs`` with ``u = bc`` on the boundary points."""
        jet = jet or Jet(dim)
        return cls._make(dim, "poisson", jet, rhs, bc, None, exact, bc_weight=bc_weight)

    @classmethod
    def transport(
        cls,
        coeffs: Sequence[float],
        rhs: Any = 0.0,
        bc: Any = None,
        *,
        time: int | None = 0,
        bc_weight: float = 100.0,
        exact: Callable | None = None,
        jet: Jet | None = None,
    ) -> Problem:
        """``Σ c_i ∂u/∂x_i = rhs`` (linear conservation law); ``bc`` is usually
        the initial condition on points with ``x[time] = 0``."""
        dim = len(coeffs)
        jet = jet or Jet(dim, time)
        return cls._make(
            dim,
            "transport",
            jet,
            rhs,
            bc,
            None,
            exact,
            grad_coeffs=[float(c) for c in coeffs],
            time=time,
            bc_weight=bc_weight,
        )

    @classmethod
    def cubic_schrodinger(
        cls,
        dim: int,
        potential: Any,
        rhs: Any = 0.0,
        bc: Any = None,
        *,
        loss: str = "normalized",
        power: float = 3.0,
        bc_weight: float = 0.0,
        exact: Callable | None = None,
        jet: Jet | None = None,
    ) -> Problem:
        """``−Δu + u³ + V(x)·u = rhs``. The default loss divides the mean square
        residual by ``mean(|u|^power)`` (the reference implementation's
        normalisation, which excludes ``u = 0``)."""
        jet = jet or Jet(dim)
        return cls._make(
            dim,
            "cubic_schrodinger",
            jet,
            rhs,
            bc,
            potential,
            exact,
            loss=loss,
            power=power,
            bc_weight=bc_weight,
        )

    @classmethod
    def eigen(
        cls,
        dim: int,
        potential: Any,
        bc: Any = None,
        *,
        bc_weight: float = 100.0,
        exact: Callable | None = None,
        jet: Jet | None = None,
    ) -> Problem:
        """Rayleigh quotient ``mean(|∇u|² + V u²) / mean(u²)`` for
        ``−Δu + V u = λu`` (minimised by the ground state)."""
        jet = jet or Jet(dim)
        return cls._make(
            dim, "poisson", jet, None, bc, potential, exact, loss="rayleigh", bc_weight=bc_weight
        )

    @classmethod
    def from_operator(
        cls,
        jet: Jet,
        operator: Any,
        rhs: Any = 0.0,
        bc: Any = None,
        *,
        potential: Any = None,
        bc_weight: float = 100.0,
        loss: str = "least_squares",
        power: float = 3.0,
        exact: Callable | None = None,
    ) -> Problem:
        """Any operator, written as an expression in ``jet.u``, ``jet.lap``,
        ``jet.grad[i]``, ``jet.ut`` and ``jet.x[i]`` (or a callable of the jet).

        Its partial derivatives in the jet are taken symbolically and compiled
        once; a constant-coefficient linear operator is recognised and uses the
        fast path.
        """
        op = operator(jet) if callable(operator) else operator
        return cls._make(
            jet.dim,
            "expr",
            jet,
            rhs,
            bc,
            potential,
            exact,
            operator=op,
            u=jet.u,
            lap=jet.lap,
            grad=jet.grad,
            ut=jet.ut,
            time=jet.time,
            bc_weight=bc_weight,
            loss=loss,
            power=power,
        )


# ---------------------------------------------------------------------------
# Point samplers (the reference implementation's)
# ---------------------------------------------------------------------------


def sample_box(
    n: int, dim: int, lo: float = -1.0, hi: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """``n`` uniform points in ``[lo, hi]^dim``."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(lo, hi, size=(n, dim))


def sample_box_boundary(
    n: int, dim: int, lo: float = -1.0, hi: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """``n`` points on the boundary of ``[lo, hi]^dim``: uniform points with one
    random coordinate set to ``lo`` (first half) or ``hi`` (second half)."""
    rng = rng if rng is not None else np.random.default_rng()
    pts = rng.uniform(lo, hi, size=(n, dim))
    half = n // 2
    pts[np.arange(half), rng.integers(0, dim, size=half)] = lo
    pts[np.arange(half, n), rng.integers(0, dim, size=n - half)] = hi
    return pts


def sample_initial_slice(
    n: int,
    dim: int,
    lo: float = 0.0,
    hi: float = 1.0,
    time: int = 0,
    t0: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``n`` uniform points in ``[lo, hi]^dim`` with ``x[time] = t0``."""
    pts = sample_box(n, dim, lo, hi, rng)
    pts[:, time] = t0
    return pts


# ---------------------------------------------------------------------------
# Evaluator and fitting
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Outcome of fitting one candidate."""

    e: list[int]
    theta: np.ndarray
    loss: float
    iterations: int = 0
    evaluations: int = 0

    @property
    def score(self) -> float:
        """The FEX reward ``1 / (1 + sqrt(loss))``."""
        return 1.0 / (1.0 + math.sqrt(self.loss)) if math.isfinite(self.loss) else 0.0


class Evaluator:
    """Compiled scoring kernel for one template, problem and point sets.

    Parameters
    ----------
    template : Template
    problem : Problem
    points : array (n, d)
        Interior collocation points.
    boundary : array (m, d), optional
        Boundary (or initial-condition) points.

    Quantities that depend only on the points — ``rhs``, ``bc``, the potential,
    and ``phi(x_i), phi'(x_i), phi''(x_i)`` for each leaf operator — are
    computed once here and shared by every candidate.
    """

    def __init__(self, template: Template, problem: Problem, points: Any, boundary: Any = None):
        self.template = template
        self.problem = problem
        self.points = _f64(points)
        self.boundary = None if boundary is None else _f64(boundary)
        self._native = _FexEvaluator(template._native, problem._native, self.points, self.boundary)

    n_residuals = property(lambda self: self._native.n_residuals)
    n_interior = property(lambda self: self._native.n_interior)
    n_boundary = property(lambda self: self._native.n_boundary)

    def _batch(self, e: Any, theta: Any) -> tuple[np.ndarray, np.ndarray, bool]:
        e = _i64(e)
        theta = _f64(theta)
        single = e.ndim == 1
        e = e.reshape(-1, self.template.n_slots)
        theta = theta.reshape(e.shape[0], self.template.n_params)
        return e, theta, single

    def residual_and_jacobian(
        self, e: Any, theta: Any, jacobian: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Residuals ``R`` and Jacobians ``J = ∂R/∂θ``.

        ``e`` is ``(batch, n_slots)`` (or one opcode vector), ``theta`` is
        ``(batch, n_params)``. Returns ``R`` of shape ``(batch, M)`` and ``J`` of
        shape ``(batch, M, n_params)`` (squeezed for a single candidate), with
        ``M = n_interior + n_boundary`` rows weighted so that ``‖R‖²`` is the
        loss. An invalid candidate has ``R = +inf`` and ``J = 0``.
        """
        e, theta, single = self._batch(e, theta)
        b, m, p = e.shape[0], self.n_residuals, self.template.n_params
        r = np.empty((b, m))
        j = np.empty((b, m, p)) if jacobian else None
        self._native.residual_and_jacobian_into(e, theta, r, j)
        if single:
            return r[0], (j[0] if j is not None else None)
        return r, j

    def loss(self, e: Any, theta: Any) -> np.ndarray | float:
        """Loss per candidate (``+inf`` for an invalid one)."""
        e, theta, single = self._batch(e, theta)
        out = np.empty(e.shape[0])
        self._native.loss_into(e, theta, out, None)
        return float(out[0]) if single else out

    def loss_and_grad(self, e: Any, theta: Any) -> tuple[Any, np.ndarray]:
        """Loss and gradient in ``theta`` per candidate."""
        e, theta, single = self._batch(e, theta)
        out = np.empty(e.shape[0])
        g = np.empty_like(theta)
        self._native.loss_into(e, theta, out, g)
        return (float(out[0]), g[0]) if single else (out, g)

    def jets(self, e: Sequence[int], theta: Any, points: Any) -> tuple[np.ndarray, ...]:
        """``(u, grad_u, lap_u)`` of one candidate at arbitrary points."""
        pts = _f64(points).reshape(-1, self.template.dim)
        n = pts.shape[0]
        u, g, lap = np.empty(n), np.empty((n, self.template.dim)), np.empty(n)
        self._native.jets_into(
            [int(v) for v in e], [float(t) for t in _f64(theta).ravel()], pts, u, g, lap
        )
        return u, g, lap

    def fit(
        self,
        e: Any,
        theta0: Any = None,
        *,
        method: str = "lm",
        max_iter: int = 50,
        lr: float = 1e-3,
        adam_iters: int = 20,
        bfgs_iters: int = 20,
        tol: float = 1e-30,
        freeze_redundant: bool = True,
        rng: np.random.Generator | None = None,
    ) -> list[FitResult]:
        """Fit a batch of candidates in parallel (see :func:`fit`)."""
        e = _i64(e).reshape(-1, self.template.n_slots)
        if theta0 is None:
            theta0 = self.template.init_params(e.shape[0], rng)
        theta0 = _f64(theta0).reshape(e.shape[0], self.template.n_params)
        out = self._native.fit_batch(
            e, theta0, method, max_iter, lr, adam_iters, bfgs_iters, tol, freeze_redundant
        )
        return [
            FitResult([int(v) for v in e[i]], np.asarray(th), loss, it, ev)
            for i, (th, loss, it, ev) in enumerate(out)
        ]

    def relative_l2(
        self, e: Sequence[int], theta: Any, exact: Callable | None = None, points: Any = None
    ) -> float:
        """Relative L² error ``‖u − u*‖ / ‖u*‖`` against ``exact`` (a function of
        an ``(n, d)`` array), on ``points`` (default: the interior points)."""
        exact = exact or self.problem.exact
        if exact is None:
            raise ValueError("no exact solution given")
        pts = self.points if points is None else _f64(points)
        u, _, _ = self.jets(e, theta, pts)
        ref = np.asarray(exact(pts), dtype=np.float64).ravel()
        return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def fit(
    evaluator: Evaluator,
    e: Sequence[int],
    theta0: Any = None,
    *,
    method: str = "lm",
    restarts: int = 1,
    max_iter: int = 50,
    rng: np.random.Generator | None = None,
    **kw: Any,
) -> FitResult:
    """Fit one candidate's parameters; return the best of ``restarts`` starts.

    ``method`` is ``"lm"`` (Levenberg–Marquardt with the exact Jacobian; the
    default), ``"adam"``, ``"bfgs"`` or ``"adam_bfgs"`` (the paper's 20 Adam
    + 20 BFGS coarse fit; set ``adam_iters`` / ``bfgs_iters`` / ``lr``).
    Restarts run in parallel.
    """
    rng = rng if rng is not None else np.random.default_rng()
    tpl = evaluator.template
    starts = tpl.init_params(restarts, rng)
    if theta0 is not None:
        starts[0] = _f64(theta0).ravel()
    es = np.tile(_i64(e).ravel(), (restarts, 1))
    res = evaluator.fit(es, starts, method=method, max_iter=max_iter, **kw)
    return min(res, key=lambda r: r.loss)


# ---------------------------------------------------------------------------
# Reference controller
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Outcome of :func:`search`."""

    e: list[int]
    theta: np.ndarray
    loss: float
    key: str
    expression: Any
    pool: list[FitResult]
    candidates_scored: int
    unique_skeletons: int
    seconds: float
    history: list[float] = field(default_factory=list)


def search(
    template: Template,
    problem: Problem,
    points: Any,
    boundary: Any = None,
    *,
    epochs: int = 200,
    batch: int = 10,
    random_steps: int = 3,
    greedy: float = 0.1,
    percentile: float = 0.5,
    lr: float = 0.05,
    score_method: str = "lm",
    score_iters: int = 20,
    score_restarts: int = 2,
    pool_size: int = 10,
    finetune_iters: int = 200,
    finetune_restarts: int = 4,
    target_loss: float | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> SearchResult:
    """Run a FEX search end to end with the paper's controller.

    Each slot has a categorical distribution (softmax of learned logits —
    the reference controller's network has a constant input, so it is exactly
    this). Every epoch samples ``batch`` skeletons (uniformly for the first
    ``random_steps`` epochs, with ``greedy``-probability uniform resampling of
    each slot afterwards), scores each by a short fit, and applies the
    risk-seeking policy gradient: only the top ``percentile`` of the batch
    contributes, with advantage ``reward − reward at the quantile``. The reward
    is ``1/(1 + sqrt(loss))``.

    Skeletons are deduplicated by :func:`canonical_key`: a family already
    scored reuses its score. New skeletons in a batch are fitted together, in
    parallel. The best ``pool_size`` families are then fine-tuned
    (Levenberg–Marquardt, ``finetune_iters`` steps, ``finetune_restarts``
    starts).
    """
    rng = rng if rng is not None else np.random.default_rng()
    t0 = time.perf_counter()
    ev = Evaluator(template, problem, points, boundary)
    arities = template.slot_arities
    logits = [np.zeros(k) for k in arities]
    m1 = [np.zeros(k) for k in arities]
    m2 = [np.zeros(k) for k in arities]
    adam_t = 0
    cache: dict[str, FitResult] = {}
    scored = 0
    history: list[float] = []

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    for step in range(epochs):
        probs = [softmax(z) for z in logits]
        es = np.empty((batch, len(arities)), dtype=np.int64)
        for s, k in enumerate(arities):
            if step < random_steps:
                es[:, s] = rng.integers(0, k, size=batch)
            else:
                es[:, s] = rng.choice(k, size=batch, p=probs[s])
                flip = rng.random(batch) < greedy
                es[flip, s] = rng.integers(0, k, size=int(flip.sum()))
        keys = [template.canonical_key(e) for e in es]
        new = [i for i, k in enumerate(keys) if k not in cache]
        # Deduplicate within the batch too.
        first: dict[str, int] = {}
        for i in new:
            first.setdefault(keys[i], i)
        todo = list(first.values())
        if todo:
            ee = np.repeat(es[todo], score_restarts, axis=0)
            th0 = template.init_params(len(ee), rng)
            res = ev.fit(ee, th0, method=score_method, max_iter=score_iters)
            scored += len(ee)
            for j, i in enumerate(todo):
                best = min(res[j * score_restarts : (j + 1) * score_restarts], key=lambda r: r.loss)
                cache[keys[i]] = best
        losses = np.array([cache[k].loss for k in keys])
        rewards = np.where(np.isfinite(losses), 1.0 / (1.0 + np.sqrt(np.abs(losses))), 0.0)
        best_loss = min(r.loss for r in cache.values())
        history.append(best_loss)
        if verbose:
            print(f"step {step}: best loss {best_loss:.3e}, unique {len(cache)}")
        # Risk-seeking policy gradient over the top quantile.
        order = np.argsort(-rewards)
        nq = max(1, int(batch * percentile))
        base = rewards[order[min(nq, batch - 1)]]
        grads = [np.zeros(k) for k in arities]
        for idx in order[:nq]:
            adv = rewards[idx] - base
            if adv == 0.0:
                continue
            for s in range(len(arities)):
                g = -probs[s]
                g[es[idx, s]] += 1.0
                grads[s] += adv * g / nq
        adam_t += 1
        for s in range(len(arities)):
            m1[s] = 0.9 * m1[s] + 0.1 * grads[s]
            m2[s] = 0.999 * m2[s] + 0.001 * grads[s] ** 2
            mh = m1[s] / (1 - 0.9**adam_t)
            vh = m2[s] / (1 - 0.999**adam_t)
            logits[s] += lr * mh / (np.sqrt(vh) + 1e-8)
        if target_loss is not None and best_loss <= target_loss:
            break

    top = sorted(cache.items(), key=lambda kv: kv[1].loss)[:pool_size]
    finals: list[tuple[str, FitResult]] = []
    for key, r in top:
        es = np.tile(np.asarray(r.e, dtype=np.int64), (finetune_restarts, 1))
        th0 = template.init_params(finetune_restarts, rng)
        th0[0] = r.theta
        res = ev.fit(es, th0, method="lm", max_iter=finetune_iters)
        finals.append((key, min(res, key=lambda x: x.loss)))
    finals.sort(key=lambda kv: kv[1].loss)
    key, best = finals[0]
    return SearchResult(
        e=best.e,
        theta=best.theta,
        loss=best.loss,
        key=key,
        expression=template.instantiate(best.e, best.theta, problem.x),
        pool=[r for _, r in finals],
        candidates_scored=scored,
        unique_skeletons=len(cache),
        seconds=time.perf_counter() - t0,
        history=history,
    )
