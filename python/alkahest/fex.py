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
* :func:`finalize` — turns a fitted candidate into an exact closed form
  (filtering, tied weights, snapping to rationals and constants such as
  ``π/4``, refitting under a loss gate), lifts it to symbolic dimension ``d``
  with ``Sum`` / ``Indexed`` and verifies it exactly — for every ``d`` — with
  the statuses of :func:`alkahest.experimental.verify_pde_solution`, plus a
  Lean certificate at the fitted ``d`` when cheap;
  :func:`recovery_metric` is the skeleton-equivalence recovery criterion.

Candidates whose value overflows, divides by zero or takes ``log`` of a
non-positive number get loss ``+inf`` — never NaN. Malformed input raises
:class:`FexError` (``E-FEX-001`` … ``E-FEX-011``).

Search and fitting run at a concrete dimension ``d``; :func:`finalize`
lifts the result to symbolic ``d``. Arrays are NumPy ``float64`` (opcodes
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np

from .alkahest import ExprPool, FexError, _FexEvaluator, _FexProblem, _FexTemplate

__all__ = [
    "PAPER_BINARY",
    "PAPER_UNARY",
    "BidiagonalDiffusion",  # noqa: F822 (module __getattr__, from _fex_pide)
    "Evaluator",
    "FexError",
    "FitResult",
    "GaussianJump",  # noqa: F822 (module __getattr__, from _fex_pide)
    "Jet",
    "Problem",
    "RadialDrift",  # noqa: F822 (module __getattr__, from _fex_pide)
    "SearchResult",
    "Template",
    "TorchEvaluator",  # noqa: F822 (module __getattr__: torch is imported on first use)
    "analyze",
    "canonical_key",
    "family_symbols",
    "finalize",
    "fit",
    "gaussian_expectation",  # noqa: F822 (module __getattr__, from _fex_pide)
    "group_weights",  # noqa: F822 (module __getattr__, from _fex_pide)
    "lift_candidate",
    "recovery_metric",
    "reduce_problem",
    "sample_box",
    "sample_box_boundary",
    "sample_initial_slice",
    "search",
    "snap_candidates",
    "solve",
    "taylor_moment",  # noqa: F822 (module __getattr__, from _fex_pide)
    "transform_problem",
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
    slot_arities = property(lambda self: self._native.slot_arities, doc="Table size per slot.")
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
        # What the problem was built from, in exact form where it was given
        # as expressions: read by :func:`finalize` to verify a solution.
        self.spec: dict[str, Any] = {}

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
        rhs, bc, potential = _source(rhs, jet), _source(bc, jet), _source(potential, jet)
        native = _FexProblem(
            dim,
            kind,
            rhs=rhs,
            bc=bc,
            potential=potential,
            coords=jet.x,
            **kw,
        )
        out = cls(native, jet, exact)
        out.spec = {"kind": kind, "rhs": rhs, "bc": bc, "potential": potential, **kw}
        return out

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

    @classmethod
    def pide(
        cls,
        dim: int,
        *,
        time: int = 0,
        drift: Any = None,
        diffusion: Any = None,
        jump: Any = None,
        rhs: Any = 0.0,
        terminal: Any = None,
        T: Any = 1,
        bc_weight: float = 100.0,
        extra: Any = None,
        exact: Callable | None = None,
        jet: Jet | None = None,
    ) -> Problem:
        """A partial integro-differential equation (FEX-PG, Hardwick–Liang–Yang):

        ``u_t + b·∇u + ½Tr(σσᵀ∇²u) + λ∫(u(x+G) − u − G·∇u) φ(z)dz = rhs``,
        ``u(T, x) = terminal``, with ``dim`` coordinates including time
        (index ``time``).

        ``drift`` is ``None``, a list of expressions (one per spatial
        coordinate) or :class:`RadialDrift`; ``diffusion`` is ``None``, a
        number ``θ`` (``σ = θI``), :class:`BidiagonalDiffusion` or a matrix;
        ``jump`` is a :class:`GaussianJump`. The jump integral enters through
        its Taylor-moment replacement (:func:`taylor_moment`): a local part in
        the operator and a nonlocal part evaluated at shifted points.
        ``extra`` adds any further operator term (an expression in the jet
        symbols; not verified by :func:`finalize`). Boundary points for
        ``terminal`` are the slice ``x[time] = T``
        (:func:`sample_initial_slice` with ``t0=T``).
        """
        from . import _fex_pide as _pd

        jet = jet or Jet(dim, time)
        if jet.time != time:
            raise _pd._err("the jet's time coordinate must be the PIDE's")
        op, kw, desc = _pd.build(
            dim,
            time=time,
            drift=drift,
            diffusion=diffusion,
            jump=jump,
            rhs=rhs,
            terminal=terminal,
            T=T,
            bc_weight=bc_weight,
            extra=extra,
            jet=jet,
        )
        rhs_e = _source(rhs, jet)
        if rhs_e is not None and _pd.exact_number(rhs_e) is not None:
            rhs_e = _pd.to_expr(rhs_e, jet.pool)
        out = cls._make(
            dim,
            "expr",
            jet,
            rhs_e,
            terminal,
            None,
            exact,
            operator=op,
            u=jet.u,
            lap=jet.lap,
            grad=jet.grad,
            ut=jet.ut,
            time=time,
            bc_weight=bc_weight,
            loss="least_squares",
            **{k: v for k, v in kw.items() if v is not None},
        )
        out.spec["pide"] = desc
        return out


_PIDE_NAMES = (
    "BidiagonalDiffusion",
    "GaussianJump",
    "RadialDrift",
    "gaussian_expectation",
    "group_weights",
    "taylor_moment",
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


def _local_method(ev: Evaluator) -> str:
    """Levenberg–Marquardt for a residual loss, BFGS otherwise (a Rayleigh
    quotient or a normalised loss has no residual vector)."""
    return "lm" if ev.problem.loss in ("least_squares", "centered") else "bfgs"


def _prune(
    ev: Evaluator,
    best: FitResult,
    cutoff: float,
    iters: int,
    restarts: int,
    rng: np.random.Generator,
) -> FitResult:
    """Greedily replace leaves of ``best`` by the constant ``0`` while the
    refitted loss stays at most ``cutoff``."""
    tpl = ev.template
    zero = tpl.unary.index("0")
    leaves = [i for i, k in enumerate(tpl.slot_kinds) if k == "leaf"]
    changed = True
    while changed:
        changed = False
        for i in leaves:
            if best.e[i] == zero:
                continue
            e = list(best.e)
            e[i] = zero
            es = np.tile(np.asarray(e, dtype=np.int64), (restarts, 1))
            th0 = tpl.init_params(restarts, rng)
            th0[0] = best.theta
            r = min(ev.fit(es, th0, method=_local_method(ev), max_iter=iters), key=lambda x: x.loss)
            if r.loss <= cutoff:
                best = r
                changed = True
    return best


@dataclass
class SearchResult:
    """Outcome of :func:`search` (and of :func:`solve`)."""

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
    template: Template | None = None
    points: np.ndarray | None = None
    boundary: np.ndarray | None = None
    #: Epochs run, and the wall time (s) at which the best loss first reached
    #: ``target_loss`` (``None`` if it never did).
    epochs: int = 0
    seconds_to_target: float | None = None
    #: :func:`solve` only: the proposal that produced the result, the
    #: analysis, the search in the transformed / reduced variables, and a
    #: log of every proposal tried.
    proposal: Any = None
    analysis: Any = None
    sub_result: Any = None
    attempts: list[dict] = field(default_factory=list)


_SEARCH_REMEDIATION = (
    "give slot_ops as {'leaf' | 'unary' | 'binary': [operator names]} or one list of "
    "names per slot, naming operators of the template's tables and leaving every slot at "
    "least one; baseline is 'quantile', 'ewma' or 'mean'; population, pqt_size and "
    "curriculum are non-negative integers"
)


def _search_error(message: str) -> FexError:
    exc = FexError(f"[E-FEX-013] {message}\nRemediation: {_SEARCH_REMEDIATION}")
    exc.code = "E-FEX-013"
    exc.remediation = _SEARCH_REMEDIATION
    exc.span = None
    return exc


def _slot_masks(template: Template, slot_ops: Any) -> list[np.ndarray] | None:
    """Boolean allowed-opcode masks per slot (``None`` = everything)."""
    if slot_ops is None:
        return None
    kinds = template.slot_kinds
    tables = {"leaf": template.unary, "unary": template.unary, "binary": template.binary}
    if isinstance(slot_ops, dict):
        bad = set(slot_ops) - set(tables)
        if bad:
            raise _search_error(f"unknown slot kinds {sorted(bad)} in slot_ops")
        per_slot = [slot_ops.get(k) for k in kinds]
    else:
        per_slot = list(slot_ops)
        if len(per_slot) != len(kinds):
            raise _search_error(f"slot_ops has {len(per_slot)} entries for {len(kinds)} slots")
    masks = []
    for k, ops in zip(kinds, per_slot):
        table = tables[k]
        if ops is None:
            masks.append(np.ones(len(table), dtype=bool))
            continue
        unknown = [o for o in ops if o not in table]
        if unknown:
            raise _search_error(f"operators {unknown} are not in the {k} table {table}")
        m = np.array([o in ops for o in table])
        if not m.any():
            raise _search_error(f"no operator allowed in a {k} slot")
        masks.append(m)
    return masks


def _shape_tree(shape: str) -> list[dict]:
    """Per slot (in-order): kind, depth from the root, children slots."""
    out: list[dict] = []
    pos = 0

    def parse(depth: int) -> int:
        nonlocal pos
        c = shape[pos]
        if c == "L":
            pos += 1
            out.append({"kind": "leaf", "depth": depth, "children": []})
            return len(out) - 1
        pos += 2  # "U(" or "B("
        if c == "U":
            ch = parse(depth + 1)
            pos += 1
            out.append({"kind": "unary", "depth": depth, "children": [ch]})
            return len(out) - 1
        left = parse(depth + 1)
        pos += 1  # ","
        out.append({"kind": "binary", "depth": depth, "children": []})
        me = len(out) - 1
        right = parse(depth + 1)
        pos += 1  # ")"
        out[me]["children"] = [left, right]
        return me

    parse(0)
    return out


def _curriculum_masks(template: Template) -> list[list[np.ndarray] | None]:
    """Depth curriculum: stage ``k`` lets the ``k`` innermost layers of inner
    unary slots take any operator and pins the rest to ``id`` (stage 0 is a
    combination of leaves). The last stage is unrestricted (``None``)."""
    tree = _shape_tree(template.shape)
    unary_depths = sorted({n["depth"] for n in tree if n["kind"] == "unary"}, reverse=True)
    if "id" not in template.unary or not unary_depths:
        return [None]
    idx = template.unary.index("id")
    stages: list[list[np.ndarray] | None] = []
    for k in range(len(unary_depths)):
        free = set(unary_depths[:k])
        masks = []
        for n, ar in zip(tree, template.slot_arities):
            m = np.ones(ar, dtype=bool)
            if n["kind"] == "unary" and n["depth"] not in free:
                m[:] = False
                m[idx] = True
            masks.append(m)
        stages.append(masks)
    stages.append(None)
    return stages


def _combine(a: list[np.ndarray] | None, b: list[np.ndarray] | None) -> list[np.ndarray] | None:
    if a is None:
        return b
    if b is None:
        return a
    out = []
    for x, y in zip(a, b):
        z = x & y
        out.append(z if z.any() else x)
    return out


def _prior_logits(template: Template, prior: Any) -> list[np.ndarray]:
    """Initial logits: zero, plus ``prior`` given as per-slot arrays or as
    ``{kind: {operator: logit}}``."""
    ar = template.slot_arities
    z = [np.zeros(k) for k in ar]
    if prior is None:
        return z
    if isinstance(prior, dict):
        tables = {"leaf": template.unary, "unary": template.unary, "binary": template.binary}
        for s, k in enumerate(template.slot_kinds):
            for name, w in (prior.get(k) or {}).items():
                if name in tables[k]:
                    z[s][tables[k].index(name)] += float(w)
        return z
    pr = list(prior)
    if len(pr) != len(ar):
        raise _search_error(f"prior has {len(pr)} entries for {len(ar)} slots")
    return [zi + np.asarray(p, dtype=float).reshape(-1) for zi, p in zip(z, pr)]


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
    score_method: str | None = None,
    score_iters: int = 20,
    score_restarts: int = 2,
    pool_size: int = 10,
    finetune_iters: int = 200,
    finetune_restarts: int = 4,
    target_loss: float | None = None,
    tie_loss: float = 1e-20,
    prune: bool = True,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    slot_ops: Any = None,
    prior: Any = None,
    entropy_weight: float = 0.0,
    baseline: str = "quantile",
    baseline_decay: float = 0.9,
    pqt_size: int = 0,
    pqt_weight: float = 1.0,
    population: int = 1,
    restart_patience: int | None = None,
    curriculum: int | None = None,
    seeds: Sequence[Sequence[int]] | None = None,
    scorer: Any = None,
    stall_epochs: int | None = None,
) -> SearchResult:
    """Run a FEX search end to end.

    **Reference controller** (the defaults; #582's behaviour). Each slot has
    a categorical distribution (softmax of learned logits — the reference
    controller's network has a constant input, so it is exactly this).
    Every epoch samples ``batch`` skeletons (uniformly for the first
    ``random_steps`` epochs, with ``greedy``-probability uniform resampling of
    each slot afterwards), scores each by a short fit, and applies the
    risk-seeking policy gradient: only the top ``percentile`` of the batch
    contributes, with advantage ``reward − reward at the quantile``. The
    reward is ``1/(1 + sqrt(loss))``.

    Skeletons are deduplicated by :func:`canonical_key`: a family already
    scored reuses its score. New skeletons in a batch are fitted together, in
    parallel. The best ``pool_size`` families are then fine-tuned
    (Levenberg–Marquardt, ``finetune_iters`` steps, ``finetune_restarts``
    starts). Among fine-tuned families whose loss is at most ``tie_loss``
    the one with the fewest free parameters is kept, and with ``prune`` each
    of its leaves is then tried as the constant ``0`` (refitted; kept while
    the loss stays within that cutoff), so a superset family such as
    ``L[sq] + L[cos]`` that fits with zero ``cos`` weights reduces to
    ``L[sq]``.

    **Controller options** (F4; each off by default):

    * ``slot_ops`` — restrict slots to operator subsets, as
      ``{"leaf": [...], "unary": [...], "binary": [...]}`` or one list per
      slot (e.g. a :class:`~alkahest._fex_analyze.Proposal`'s ``slot_ops``).
      Sampling, the greedy resampling and the uniform warm-up all respect it.
    * ``prior`` — initial logits, per slot or as ``{kind: {op: logit}}``
      (a soft operator-subset prior instead of a hard restriction).
    * ``entropy_weight`` — entropy bonus ``λ_H·H(π)`` in the policy gradient
      (Petersen et al., DSO, ICLR 2021).
    * ``baseline`` — ``"quantile"`` (risk-seeking: advantage over the
      batch's ``percentile`` reward; the reference), ``"ewma"`` (risk-seeking
      over an exponential moving average of that quantile, decay
      ``baseline_decay``) or ``"mean"`` (plain REINFORCE over the whole
      batch with an EWMA mean-reward baseline).
    * ``pqt_size`` / ``pqt_weight`` — priority-queue training: the best
      ``pqt_size`` families seen so far are kept and their log-likelihood,
      weighted by ``pqt_weight``, is added to every update (DSO's PQT).
    * ``population`` / ``restart_patience`` — ``population`` independent
      controllers sample ``batch`` skeletons each per epoch and share one
      score cache; a controller whose best reward has not improved for
      ``restart_patience`` epochs is restarted from the best controller's
      logits plus noise (from the prior, with a single controller).
    * ``curriculum`` — depth curriculum: ``curriculum`` epochs with every
      inner unary slot pinned to ``id``, then one more layer of inner unary
      slots freed every ``curriculum`` epochs until the full tree is open.
    * ``seeds`` — skeletons scored before the first epoch (e.g. ansätze from
      :func:`analyze`); they enter the cache and the priority queue.
    * ``scorer`` — an object with ``fit(es, theta0, method=, max_iter=)``
      returning :class:`FitResult` lists (e.g. :class:`TorchEvaluator` on a
      GPU) used for the coarse scoring; fine-tuning stays on the CPU
      evaluator.
    * ``stall_epochs`` — stop early once no new skeleton family has been
      scored for that many epochs (a restricted search space is exhausted
      long before ``epochs``).

    Raises :class:`FexError` ``E-FEX-013`` for malformed controller options.
    """
    rng = rng if rng is not None else np.random.default_rng()
    t0 = time.perf_counter()
    ev = Evaluator(template, problem, points, boundary)
    if score_method is None:
        score_method = _local_method(ev)
    if baseline not in ("quantile", "ewma", "mean"):
        raise _search_error(f"unknown baseline {baseline!r}")
    if population < 1 or pqt_size < 0 or (curriculum is not None and curriculum < 1):
        raise _search_error("population must be >= 1, pqt_size >= 0 and curriculum >= 1")
    arities = template.slot_arities
    hard = _slot_masks(template, slot_ops)
    stages = _curriculum_masks(template) if curriculum else [None]
    init = _prior_logits(template, prior)
    members: list[dict[str, Any]] = []
    for _ in range(population):
        members.append(
            {
                "logits": [z.copy() for z in init],
                "m1": [np.zeros(k) for k in arities],
                "m2": [np.zeros(k) for k in arities],
                "t": 0,
                "best": 0.0,
                "since": 0,
            }
        )
    scorer = scorer if scorer is not None else ev
    cache: dict[str, FitResult] = {}
    pq: dict[str, tuple[float, np.ndarray]] = {}
    scored = 0
    history: list[float] = []
    ewma: float | None = None
    t_target: float | None = None
    plain = (
        hard is None
        and prior is None
        and curriculum is None
        and population == 1
        and pqt_size == 0
        and entropy_weight == 0.0
        and baseline == "quantile"
        and restart_patience is None
        and not seeds
    )

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def score(es: np.ndarray) -> list[str]:
        nonlocal scored
        keys = [template.canonical_key(e) for e in es]
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in cache:
                first.setdefault(k, i)
        todo = list(first.values())
        if todo:
            ee = np.repeat(es[todo], score_restarts, axis=0)
            th0 = template.init_params(len(ee), rng)
            res = scorer.fit(ee, th0, method=score_method, max_iter=score_iters)
            scored += len(ee)
            for j, i in enumerate(todo):
                best = min(res[j * score_restarts : (j + 1) * score_restarts], key=lambda r: r.loss)
                cache[keys[i]] = best
        return keys

    def reward_of(loss: float) -> float:
        return 1.0 / (1.0 + math.sqrt(abs(loss))) if math.isfinite(loss) else 0.0

    def push_pq(keys: list[str], es: np.ndarray) -> None:
        if pqt_size <= 0:
            return
        for k, e in zip(keys, es):
            pq[k] = (reward_of(cache[k].loss), np.array(e))
        if len(pq) > pqt_size:
            for k, _ in sorted(pq.items(), key=lambda kv: kv[1][0])[: len(pq) - pqt_size]:
                del pq[k]

    if seeds:
        es0 = np.asarray([list(e) for e in seeds], dtype=np.int64).reshape(-1, len(arities))
        push_pq(score(es0), es0)

    step = 0
    stall, n_seen = 0, len(cache)
    for step in range(epochs):
        stage = stages[min(step // curriculum, len(stages) - 1)] if curriculum else None
        masks = _combine(hard, stage)
        if plain:
            # #582's controller, unchanged (same random stream).
            mem = members[0]
            probs = [softmax(z) for z in mem["logits"]]
            es = np.empty((batch, len(arities)), dtype=np.int64)
            for s, k in enumerate(arities):
                if step < random_steps:
                    es[:, s] = rng.integers(0, k, size=batch)
                else:
                    es[:, s] = rng.choice(k, size=batch, p=probs[s])
                    flip = rng.random(batch) < greedy
                    es[flip, s] = rng.integers(0, k, size=int(flip.sum()))
            samples = [(mem, probs, es)]
        else:
            samples = []
            for mem in members:
                probs = []
                for s, z in enumerate(mem["logits"]):
                    zz = z.copy()
                    if masks is not None:
                        zz[~masks[s]] = -np.inf
                    probs.append(softmax(zz))
                es = np.empty((batch, len(arities)), dtype=np.int64)
                for s, k in enumerate(arities):
                    allowed = np.arange(k) if masks is None else np.flatnonzero(masks[s])
                    if step < random_steps:
                        es[:, s] = rng.choice(allowed, size=batch)
                    else:
                        es[:, s] = rng.choice(k, size=batch, p=probs[s])
                        flip = rng.random(batch) < greedy
                        es[flip, s] = rng.choice(allowed, size=int(flip.sum()))
                samples.append((mem, probs, es))
        all_es = np.concatenate([x[2] for x in samples])
        keys_all = score(all_es)
        push_pq(keys_all, all_es)
        best_loss = min(r.loss for r in cache.values())
        history.append(best_loss)
        if target_loss is not None and best_loss <= target_loss and t_target is None:
            t_target = time.perf_counter() - t0
        if verbose:
            print(f"step {step}: best loss {best_loss:.3e}, unique {len(cache)}")
        off = 0
        for mem, probs, es in samples:
            keys = keys_all[off : off + len(es)]
            off += len(es)
            losses = np.array([cache[k].loss for k in keys])
            rewards = np.where(np.isfinite(losses), 1.0 / (1.0 + np.sqrt(np.abs(losses))), 0.0)
            order = np.argsort(-rewards)
            nb = len(es)
            nq = max(1, int(nb * percentile))
            q = rewards[order[min(nq, nb - 1)]]
            if baseline == "quantile":
                base, chosen, norm = q, order[:nq], nq
            elif baseline == "ewma":
                ewma = q if ewma is None else baseline_decay * ewma + (1 - baseline_decay) * q
                base, chosen, norm = ewma, order[:nq], nq
            else:
                mean = float(rewards.mean())
                ewma = mean if ewma is None else baseline_decay * ewma + (1 - baseline_decay) * mean
                base, chosen, norm = ewma, order, nb
            grads = [np.zeros(k) for k in arities]
            for idx in chosen:
                adv = rewards[idx] - base
                if adv == 0.0:
                    continue
                for s in range(len(arities)):
                    g = -probs[s].copy()
                    g[es[idx, s]] += 1.0
                    grads[s] += adv * g / norm
            if pqt_size and pq:
                for _r, e in pq.values():
                    for s in range(len(arities)):
                        g = -probs[s].copy()
                        g[e[s]] += 1.0
                        grads[s] += pqt_weight * g / len(pq)
            if entropy_weight:
                for s in range(len(arities)):
                    p = probs[s]
                    lp = np.log(np.where(p > 0, p, 1.0))
                    h = -float((p * lp).sum())
                    grads[s] += entropy_weight * (-p * (lp + h))
            mem["t"] += 1
            t = mem["t"]
            for s in range(len(arities)):
                mem["m1"][s] = 0.9 * mem["m1"][s] + 0.1 * grads[s]
                mem["m2"][s] = 0.999 * mem["m2"][s] + 0.001 * grads[s] ** 2
                mh = mem["m1"][s] / (1 - 0.9**t)
                vh = mem["m2"][s] / (1 - 0.999**t)
                mem["logits"][s] += lr * mh / (np.sqrt(vh) + 1e-8)
            top = float(rewards.max()) if len(rewards) else 0.0
            if top > mem["best"] + 1e-12:
                mem["best"], mem["since"] = top, 0
            else:
                mem["since"] += 1
        if restart_patience is not None:
            lead = max(members, key=lambda m: m["best"])
            for mem in members:
                if mem["since"] < restart_patience:
                    continue
                if population > 1 and mem is not lead:
                    mem["logits"] = [z + rng.normal(0.0, 0.5, z.shape) for z in lead["logits"]]
                else:
                    mem["logits"] = [z.copy() for z in init]
                mem["m1"] = [np.zeros(k) for k in arities]
                mem["m2"] = [np.zeros(k) for k in arities]
                mem["t"], mem["since"], mem["best"] = 0, 0, 0.0
        if target_loss is not None and best_loss <= target_loss:
            break
        if stall_epochs is not None:
            stall = 0 if len(cache) > n_seen else stall + 1
            n_seen = len(cache)
            if stall >= stall_epochs:
                break

    top = sorted(cache.items(), key=lambda kv: kv[1].loss)[:pool_size]
    finals: list[tuple[str, FitResult]] = []
    for key, r in top:
        es = np.tile(np.asarray(r.e, dtype=np.int64), (finetune_restarts, 1))
        th0 = template.init_params(finetune_restarts, rng)
        th0[0] = r.theta
        res = ev.fit(es, th0, method=_local_method(ev), max_iter=finetune_iters)
        finals.append((key, min(res, key=lambda x: x.loss)))

    # Among families that solve the problem to within `tie_loss`, prefer the
    # one with the fewest free parameters (a superset family such as
    # L[sq] + L[cube] also fits ½Σx² exactly, with zero cube weights).
    def free_params(r: FitResult) -> int:
        return sum(1 for v in template.redundant_params(r.e) if v is None)

    finals.sort(key=lambda kv: kv[1].loss)
    cutoff = max(tie_loss, finals[0][1].loss)
    tied = [kv for kv in finals if kv[1].loss <= cutoff]
    key, best = min(tied, key=lambda kv: (free_params(kv[1]), len(kv[0]), kv[1].loss))
    if prune and "0" in template.unary:
        best = _prune(ev, best, cutoff, finetune_iters, finetune_restarts, rng)
        key = template.canonical_key(best.e)
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
        template=template,
        points=ev.points,
        boundary=ev.boundary,
        epochs=step + 1,
        seconds_to_target=t_target,
    )


# ---------------------------------------------------------------------------
# Symbolic pruning and ansätze (F4, R9)
# ---------------------------------------------------------------------------


def analyze(
    problem: Problem,
    *,
    seed: int = 0,
    unary: Sequence[str] | None = None,
    binary: Sequence[str] | None = None,
    shape: str = "depth2_sub",
) -> Any:
    """Analyse ``problem`` symbolically before a search and propose
    restricted searches (operator subsets, ``u = e^v`` / ``log``
    substitutions, variable reductions, radial, separable, plane-wave and
    scaling ansätze). Returns an :class:`~alkahest._fex_analyze.Analysis`;
    see :mod:`alkahest._fex_analyze` for each test."""
    from ._fex_analyze import analyze as _analyze

    return _analyze(
        problem,
        seed=seed,
        unary=None if unary is None else list(unary),
        binary=None if binary is None else list(binary),
        shape=shape,
    )


def transform_problem(problem: Problem, kind: str = "exp", *, loss: str | None = None) -> Problem:
    """The problem for ``v`` under ``u = e^v`` (``"exp"``) or ``u = log v``
    (``"log"``), derived through the jet layer. Raises ``E-FEX-012`` when it
    does not apply."""
    from ._fex_analyze import transform_problem as _tp

    return _tp(problem, kind, loss=loss)


def reduce_problem(problem: Problem, variables: Sequence[int]) -> Problem:
    """The problem for solutions depending only on ``variables``. Raises
    ``E-FEX-012`` when the operator or data still mention a dropped
    coordinate."""
    from ._fex_analyze import reduce_problem as _rp

    return _rp(problem, list(variables))


def _scale_output(tpl: Template, e: list[int], theta: np.ndarray, slot: int, c: float) -> bool:
    tree = _shape_tree(tpl.shape)
    offs = list(tpl.param_offsets)
    n = tree[slot]
    if n["kind"] == "leaf":
        theta[offs[slot] : offs[slot] + tpl.dim + 1] *= c
        return True
    if n["kind"] == "unary":
        theta[offs[slot] : offs[slot] + 2] *= c
        return True
    op = tpl.binary[e[slot]]
    left, right = n["children"]
    if op in ("+", "-"):
        return _scale_output(tpl, e, theta, left, c) and _scale_output(tpl, e, theta, right, c)
    if op in ("*", "/"):
        return _scale_output(tpl, e, theta, left, c)
    return False


def lift_candidate(
    template: Template,
    e: Sequence[int],
    theta: Any,
    *,
    transform: str | None = None,
    variables: Sequence[int] | None = None,
    dim: int | None = None,
) -> tuple[Template, list[int], np.ndarray]:
    """Map a candidate found for a transformed and / or reduced problem back
    to the original unknown and coordinates.

    ``variables`` places the reduced template's leaf weights at those
    coordinates of a ``dim``-dimensional template (zero elsewhere).
    ``transform="exp"`` turns ``v`` into ``u = e^v``: when the root slot is
    ``id`` its affine map is folded into the subtree and the root becomes
    ``exp`` (same shape); otherwise the tree is wrapped in a new ``exp``
    root ``U(…)``. ``"log"`` wraps in ``log``.
    """
    e = [int(v) for v in e]
    theta = np.asarray(theta, dtype=np.float64).copy()
    tpl = template
    if variables is not None:
        variables = [int(v) for v in variables]
        d = int(dim) if dim is not None else max(variables) + 1
        new = Template(tpl.shape, tpl.unary, tpl.binary, d)
        th = np.zeros(new.n_params)
        o_old, o_new = list(tpl.param_offsets), list(new.param_offsets)
        for s, k in enumerate(tpl.slot_kinds):
            a, b = o_old[s], o_new[s]
            if k == "leaf":
                for j, v in enumerate(variables):
                    th[b + v] = theta[a + j]
                th[b + d] = theta[a + tpl.dim]
            elif k == "unary":
                th[b : b + 2] = theta[a : a + 2]
        tpl, theta = new, th
    if transform is None:
        return tpl, e, theta
    outer = {"exp": "exp", "log": "log"}[transform]
    unary = list(tpl.unary)
    if outer not in unary:
        unary.append(outer)
    remap = Template(tpl.shape, unary, tpl.binary, tpl.dim)
    root = len(e) - 1
    kinds = tpl.slot_kinds
    if transform == "exp" and kinds[root] == "unary" and tpl.unary[e[root]] == "id":
        child = _shape_tree(tpl.shape)[root]["children"][0]
        o = tpl.param_offsets[root]
        a_r, b_r = float(theta[o]), float(theta[o + 1])
        th = theta.copy()
        if _scale_output(tpl, e, th, child, a_r):
            th[o], th[o + 1] = math.exp(b_r), 0.0
            e2 = [unary.index(tpl.unary[v]) if k != "binary" else v for v, k in zip(e, kinds)]
            e2[root] = unary.index("exp")
            return remap, e2, th
    wrapped = Template(f"U({tpl.shape})", unary, tpl.binary, tpl.dim)
    e2 = [unary.index(tpl.unary[v]) if k != "binary" else v for v, k in zip(e, kinds)]
    e2.append(unary.index(outer))
    return wrapped, e2, np.concatenate([theta, [1.0, 0.0]])


def _saturated(
    tpl: Template, e: Sequence[int], theta: Any, coords: list[Any], pts: np.ndarray
) -> bool:
    """Whether a ``tanh`` slot's argument exceeds 18 in magnitude at some
    sample point (``tanh`` is then ±1 to double precision there): such a
    candidate can fit a jump with an exactly zero collocation residual."""
    if "tanh" not in tpl.unary or not any(
        tpl.unary[v] == "tanh" for v, k in zip(e, tpl.slot_kinds) if k != "binary"
    ):
        return False
    import alkahest as _alk

    expr = tpl.instantiate(e, theta, coords)
    stack, seen = [expr], set()
    cols = [np.ascontiguousarray(pts[:, i]) for i in range(pts.shape[1])]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        nd = n.node()
        if nd[0] == "func" and nd[1] == "tanh":
            f = _alk.compile_expr(nd[2][0], list(coords))
            v = np.asarray(_alk.numpy_eval(f, *cols))
            if not np.all(np.isfinite(v)) or float(np.max(np.abs(v))) > 18.0:
                return True
        for a in nd[1:]:
            if isinstance(a, list):
                stack.extend(c for c in a if hasattr(c, "node"))
            elif hasattr(a, "node"):
                stack.append(a)
    return False


def _accepts(loss: float, accept_loss: float, theta: Any = None, max_param: float = 1e4) -> bool:
    """Exact in floating point, with parameters below ``max_param``: a
    saturated ``tanh`` / ``exp`` with huge weights is a step function whose
    collocation residual can be exactly zero at every sample point."""
    if not (math.isfinite(loss) and loss <= accept_loss):
        return False
    return theta is None or float(np.max(np.abs(theta), initial=0.0)) <= max_param


def _flip_leaf_signs(
    tpl: Template, e: list[int], theta: np.ndarray, rng, p: float = 0.5
) -> np.ndarray:
    """``theta`` with each leaf weight set to minus its magnitude with
    probability ``p`` — starts for discrete solution sets (a sign per
    coordinate); ``p = 1`` is the all-decaying start."""
    th = np.array(theta, dtype=np.float64)
    for s, k in enumerate(tpl.slot_kinds):
        if k == "leaf":
            o = tpl.param_offsets[s]
            w = th[o : o + tpl.dim]
            flip = rng.random(tpl.dim) < p
            w[flip] = -np.abs(w[flip])
    return th


def _e_back(
    tpl_v: Template, tpl_u: Template, e_u: Sequence[int], transform: str | None
) -> list[int]:
    """Opcodes of a lifted skeleton in the searched template's tables (a
    folded ``exp`` root goes back to ``id``)."""
    folded = transform == "exp" and tpl_u.n_slots == tpl_v.n_slots
    out = []
    for s, k in enumerate(tpl_v.slot_kinds):
        if k == "binary":
            out.append(tpl_v.binary.index(tpl_u.binary[e_u[s]]))
        else:
            name = "id" if (folded and s == tpl_v.n_slots - 1) else tpl_u.unary[e_u[s]]
            out.append(tpl_v.unary.index(name))
    return out


def _unlift(
    tpl_v: Template,
    tpl_u: Template,
    theta_u: np.ndarray,
    transform: str | None,
    variables: Sequence[int] | None,
) -> np.ndarray | None:
    """Inverse of :func:`lift_candidate` on parameters (the skeleton is the
    searched one). ``None`` when the lifted root scale is not positive."""
    th = np.asarray(theta_u, dtype=np.float64).copy()
    if transform is not None:
        if tpl_u.n_slots == tpl_v.n_slots:  # folded: root exp with (α, 0)
            o = tpl_u.param_offsets[-1]
            if not th[o] > 0:
                return None
            th[o], th[o + 1] = 1.0, math.log(th[o])
        else:  # wrapped in U(…)
            th = th[:-2]
    if variables is None:
        return th
    out = np.zeros(tpl_v.n_params)
    d_full = tpl_u.dim
    for s, k in enumerate(tpl_v.slot_kinds):
        a, b = tpl_v.param_offsets[s], tpl_u.param_offsets[s]
        if k == "leaf":
            for j, v in enumerate(variables):
                out[a + j] = th[b + v]
            out[a + tpl_v.dim] = th[b + d_full]
        elif k == "unary":
            out[a : a + 2] = th[b : b + 2]
    return out


def _fold_constant_products(
    tpl: Template, e: Sequence[int], theta: Any
) -> tuple[list[int], np.ndarray]:
    """``c · X`` with ``c`` a constant leaf (``L[0]`` / ``L[1]``) becomes
    ``X·c + 0`` (``X`` rescaled, the product turned into a sum with the
    constant leaf zeroed): the same function, with the coefficients where
    :func:`finalize` can snap and lift them."""
    e = [int(v) for v in e]
    th = np.asarray(theta, dtype=np.float64).copy()
    if "+" not in tpl.binary:
        return e, th
    tree = _shape_tree(tpl.shape)
    offs = list(tpl.param_offsets)
    d = tpl.dim
    for s, node in enumerate(tree):
        if node["kind"] != "binary" or tpl.binary[e[s]] != "*":
            continue
        left, right = node["children"]
        for c_slot, other in ((left, right), (right, left)):
            if tree[c_slot]["kind"] != "leaf" or tpl.unary[e[c_slot]] not in ("0", "1"):
                continue
            o = offs[c_slot]
            val = th[o + d] + (th[o : o + d].sum() if tpl.unary[e[c_slot]] == "1" else 0.0)
            trial = th.copy()
            if _scale_output(tpl, e, trial, other, float(val)):
                trial[o : o + d + 1] = 0.0
                th = trial
                e[s] = tpl.binary.index("+")
                break
    return e, th


def solve(
    problem: Problem,
    points: Any,
    boundary: Any = None,
    *,
    shape: str = "depth2_sub",
    unary: Sequence[str] = PAPER_UNARY,
    binary: Sequence[str] = PAPER_BINARY,
    analysis: Any = True,
    proposals: Sequence[Any] | None = None,
    max_proposals: int | None = None,
    accept_loss: float = 1e-20,
    alternatives: int = 8,
    select: str = "auto",
    max_param: float = 1e4,
    restrict: str = "hard",
    prior_strength: float = 2.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    **search_kw: Any,
) -> SearchResult:
    """FEX with symbolic pruning: :func:`analyze` the problem, then
    :func:`search` each proposal in turn — the transformed and / or reduced
    problem, with the proposal's operator tables and slot restrictions —
    map the result back with :func:`lift_candidate`, and stop at the first
    proposal whose search loss is at most ``accept_loss``.

    The returned :class:`SearchResult` is in the original unknown and
    coordinates (so :func:`finalize` takes it with the original problem).
    Within a proposal, candidates exact in the searched variables come
    first and the *original* loss decides among them. For a substitution,
    each exact family is also refitted on the original loss (from the lifted
    parameters and ``alternatives − 1`` sign-flipped starts) and re-solved
    in the searched variables from there: ``u = e^v`` turns
    ``−Δu + Vu = λu`` into an equation that growing solutions such as
    ``e^{+|x|²/2}`` solve too, and the original loss (the Rayleigh quotient
    with the boundary penalty) picks the ground state. When no proposal is
    exact, ``select="original"`` returns the smallest original loss and
    ``select="first"`` the first (most specific) proposal's result;
    A candidate counts as exact only with every parameter below
    ``max_param`` in magnitude and no ``tanh`` argument above 18 at the
    sample points: a saturated ``tanh`` is a step whose collocation residual
    vanishes at every sample point.
    ``"auto"`` (default) is ``"first"`` for a Rayleigh-quotient problem —
    a Monte Carlo Rayleigh quotient in high dimension can be driven below
    the true eigenvalue, so it cannot rank proposals — and ``"original"``
    otherwise.

    ``restrict="hard"`` (default) searches each proposal's operator tables
    and slot restrictions only; ``restrict="soft"`` searches the full tables
    with the proposal as a prior instead (the proposed operators' initial
    logits raised by ``prior_strength``), the LLM-free counterpart of the
    operator-set prediction of Bhatnagar et al. (arXiv 2503.09986).

    ``unary`` / ``binary`` are the full operator tables (the last proposal
    searches them unrestricted) and ``shape`` the tree of that search.
    ``analysis=False`` runs the unrestricted search only; ``proposals``
    overrides the analysis' list. ``search_kw`` goes to :func:`search`
    (e.g. ``epochs``, controller options).
    """
    from ._fex_analyze import Proposal
    from ._fex_analyze import analyze as _analyze

    rng = rng if rng is not None else np.random.default_rng()
    t0 = time.perf_counter()
    if select == "auto":
        select = "first" if problem.loss == "rayleigh" else "original"
    if restrict not in ("hard", "soft"):
        raise _search_error(f"unknown restrict {restrict!r}; use 'hard' or 'soft'")
    if select not in ("first", "original"):
        raise _search_error(f"unknown select {select!r}; use 'auto', 'first' or 'original'")
    pts = _f64(points)
    bd = None if boundary is None else _f64(boundary)
    ana = None
    if proposals is None:
        if analysis is True:
            ana = _analyze(problem, unary=list(unary), binary=list(binary), shape=shape)
            proposals = ana.proposals
        elif analysis:
            ana = analysis
            proposals = ana.proposals
        else:
            proposals = [
                Proposal(
                    "paper",
                    "no analysis",
                    list(unary),
                    list(binary),
                    shape=shape,
                )
            ]
    proposals = list(proposals)
    if max_proposals is not None:
        proposals = proposals[:max_proposals]
    ev_cache: dict[tuple, Evaluator] = {}

    def ev_for(tpl: Template) -> Evaluator:
        k = (tpl.shape, tuple(tpl.unary), tuple(tpl.binary))
        if k not in ev_cache:
            ev_cache[k] = Evaluator(tpl, problem, pts, bd)
        return ev_cache[k]

    def original_loss(tpl: Template, e: list[int], th: np.ndarray) -> float:
        return float(ev_for(tpl).loss(e, th))

    attempts: list[dict] = []
    best: tuple | None = None
    scored = 0
    unique = 0
    for prop in proposals:
        ta = time.perf_counter()
        sub = problem
        try:
            if prop.transform:
                sub = transform_problem(sub, prop.transform, loss=prop.loss)
            if prop.variables is not None:
                sub = reduce_problem(sub, prop.variables)
        except FexError as exc:
            attempts.append({"proposal": prop.name, "skipped": str(exc)})
            continue
        sp = pts if prop.variables is None else pts[:, prop.variables]
        sb = None
        if bd is not None and (sub.spec.get("bc") is not None):
            sb = bd if prop.variables is None else bd[:, prop.variables]
        kw = dict(search_kw)
        kw.setdefault("target_loss", accept_loss)
        if restrict == "soft" and prop.slot_ops:
            # The proposal as a prior: the full tables, with the proposed
            # operators' logits raised by `prior_strength`.
            table = list(dict.fromkeys([*unary, *prop.unary]))
            tpl = Template(prop.shape, table, list(binary), sub.dim)
            kw.setdefault(
                "prior",
                {k: dict.fromkeys(v, prior_strength) for k, v in prop.slot_ops.items()},
            )
            slot_ops = None
        else:
            tpl = Template(prop.shape, prop.unary, prop.binary, sub.dim)
            slot_ops = prop.slot_ops or None
        t_search = time.perf_counter()
        res = search(tpl, sub, sp, sb, slot_ops=slot_ops, rng=rng, **kw)
        scored += res.candidates_scored
        unique += res.unique_skeletons
        cands = [(res.e, res.theta, res.loss)]
        cands += [(r.e, r.theta, r.loss) for r in res.pool]
        scored_here = []

        def consider(e_v, th_v, l_v, tpl=tpl, prop=prop, scored_here=scored_here):
            if not math.isfinite(l_v):
                return None
            tu, eu, thu = lift_candidate(
                tpl,
                e_v,
                th_v,
                transform=prop.transform,
                variables=prop.variables,
                dim=problem.dim,
            )
            if _saturated(tu, eu, thu, problem.x, pts):
                return None
            lo = original_loss(tu, eu, thu)
            item = (lo, l_v, tu, eu, thu)
            scored_here.append(item)
            return item

        for c in cands:
            consider(*c)
        if prop.transform and alternatives:
            # A substitution can admit solutions the original problem rules
            # out (u = e^v: every zero-free eigenfunction-like solution, the
            # growing ones included). For each family that solves the
            # searched problem: refit its lifted form on the original loss,
            # map back, and re-solve the searched problem from there — the
            # local solution nearest to what the original loss prefers.
            ev_sub = Evaluator(tpl, sub, sp, sb)
            cut = max(accept_loss, res.loss) * 100
            seen = set()
            for lo, l_v, tu, eu, thu in list(scored_here):
                k = tuple(eu)
                if l_v > cut or k in seen or _accepts(lo, accept_loss):
                    # Already a solution of the original problem: nothing
                    # to choose between.
                    continue
                seen.add(k)
                ev_o = ev_for(tu)
                starts = [thu, _flip_leaf_signs(tu, eu, thu, rng, p=1.0)] + [
                    _flip_leaf_signs(tu, eu, thu, rng) for _ in range(max(0, alternatives - 2))
                ]
                es_o = np.tile(np.asarray(eu, dtype=np.int64), (len(starts), 1))
                fits_o = ev_o.fit(
                    es_o, np.asarray(starts), method=_local_method(ev_o), max_iter=100
                )
                e_back = np.asarray(_e_back(tpl, tu, eu, prop.transform), dtype=np.int64)
                back = [_unlift(tpl, tu, th, prop.transform, prop.variables) for th in starts[1:]]
                back += [_unlift(tpl, tu, f.theta, prop.transform, prop.variables) for f in fits_o]
                back = [b for b in back if b is not None]
                if not back:
                    continue
                refits = ev_sub.fit(
                    np.tile(e_back, (len(back), 1)),
                    np.asarray(back),
                    method=_local_method(ev_sub),
                    max_iter=200,
                )
                for r2 in refits:
                    consider(r2.e, r2.theta, r2.loss)
        # Exact in the searched variables first, then the original loss.
        exact = [c for c in scored_here if _accepts(c[1], accept_loss, c[4], max_param)]

        def n_free(c) -> int:
            return sum(1 for v in c[2].redundant_params(c[3]) if v is None)

        def rank(c) -> float:
            # For an eigenproblem the Monte Carlo Rayleigh quotient cannot
            # rank exact solutions of the substituted equation in high
            # dimension (it is dominated by the few samples where u is
            # large); the boundary condition u = 0 can. The ground state
            # decays towards every face: compare |u| at the face centres of
            # the box spanned by the boundary points with |u| at its centre.
            if problem.loss == "rayleigh" and bd is not None and len(bd):
                lo_b, hi_b = bd.min(axis=0), bd.max(axis=0)
                mid = 0.5 * (lo_b + hi_b)
                probe = [mid]
                for i in range(len(mid)):
                    for v in (lo_b[i], hi_b[i]):
                        q = mid.copy()
                        q[i] = v
                        probe.append(q)
                u = np.abs(ev_for(c[2]).jets(c[3], c[4], np.asarray(probe))[0])
                return float(u[1:].max() / u[0]) if u[0] > 0 else math.inf
            return c[0]

        if exact:
            # Simplest family first (fewest free parameters), then the
            # original loss (which separates e.g. decaying from growing).
            pick = min(exact, key=lambda c: (n_free(c), rank(c), c[1]))
        elif problem.loss == "rayleigh":
            # No exact candidate: among the near-best fits of the searched
            # problem, the one that decays (see `rank`).
            best_v = min(c[1] for c in scored_here)
            near = [c for c in scored_here if c[1] <= 10 * best_v + 1e-300]
            pick = min(near, key=lambda c: (rank(c), c[1]))
        else:
            pick = min(scored_here, key=lambda c: (c[0], c[1]))
        lo, l_v, tu, eu, thu = pick
        attempts.append(
            {
                "proposal": prop.name,
                "reason": prop.reason,
                "transform": prop.transform,
                "variables": prop.variables,
                "unary": prop.unary,
                "slot_ops": prop.slot_ops,
                "search_loss": l_v,
                "original_loss": lo,
                "key": tu.canonical_key(eu),
                "epochs": res.epochs,
                "candidates_scored": res.candidates_scored,
                "seconds": time.perf_counter() - ta,
            }
        )
        if verbose:
            print(attempts[-1])
        ok_here = _accepts(l_v, accept_loss, thu, max_param)
        hit = None
        if ok_here:
            # When the coarse search hit the target, that moment; otherwise
            # (exact only after fine-tuning) now.
            hit = (
                (t_search - t0) + res.seconds_to_target
                if res.seconds_to_target is not None
                else time.perf_counter() - t0
            )
        cand = (ok_here, lo, prop, res, tu, eu, thu, l_v, hit)
        if (
            best is None
            or (cand[0] and not best[0])
            or (select == "original" and cand[0] == best[0] and cand[1] < best[1])
        ):
            best = cand
        if ok_here:
            break
    if best is None:
        raise _search_error("no proposal could be searched")
    _ok, lo, prop, res, tu, eu, thu, l_v, hit = best
    e2, th2 = _fold_constant_products(tu, eu, thu)
    if e2 != list(eu):
        lo2 = original_loss(tu, e2, th2)
        if math.isfinite(lo2) and lo2 <= max(lo * (1 + 1e-6), lo + 1e-28):
            eu, thu, lo = e2, th2, lo2
    return SearchResult(
        e=eu,
        theta=thu,
        loss=lo,
        key=tu.canonical_key(eu),
        expression=tu.instantiate(eu, thu, problem.x),
        pool=res.pool,
        candidates_scored=scored,
        unique_skeletons=unique,
        seconds=time.perf_counter() - t0,
        history=res.history,
        template=tu,
        points=pts,
        boundary=bd,
        epochs=res.epochs,
        seconds_to_target=hit,
        proposal=prop,
        analysis=ana,
        sub_result=res,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Finalising a solution (F3)
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    # The torch evaluator is imported on first use (torch is optional).
    if name == "TorchEvaluator":
        from ._fex_torch import TorchEvaluator

        return TorchEvaluator
    if name in _PIDE_NAMES:
        from . import _fex_pide

        return getattr(_fex_pide, name)
    raise AttributeError(f"module 'alkahest.fex' has no attribute {name!r}")


def finalize(result: Any, problem: Problem, **kw: Any) -> dict:
    """Turn a fitted candidate into an exact, verified closed form — for the
    fitted dimension and, when it lifts, for symbolic ``d``.

    Coefficients are pinned, filtered (``|θ| < tau`` → 0), grouped (equal leaf
    weights tied) and snapped to exact constants (continued fractions, π, √2,
    √3, e, log 2 and products / quotients of two, PSLQ and LLL relations),
    refitting after every step and keeping it only while the loss stays
    within the gate. Tied leaves ``Σᵢ α φ(xᵢ)`` lift to
    ``α·Sum(φ(Indexed(x, i)), i, 1, d)``, the operator is written for
    symbolic ``d``, and the residuals — PDE, and the boundary condition on
    every face of the box (or the initial condition) — go through the exact
    zero test with the statuses of
    :func:`alkahest.experimental.verify_pde_solution`.

    Returns a dict with ``expr``, ``expr_symbolic_d``, ``snapped_params``,
    ``verification`` (``concrete``, ``symbolic_d``, ``interval``) and
    ``certificate``; see :func:`alkahest._fex_finalize.finalize` for every
    option. Raises :class:`FexError` ``E-FEX-009`` / ``E-FEX-010``.
    """
    from ._fex_finalize import finalize as _finalize

    return _finalize(result, problem, **kw)


def recovery_metric(found: Any, truth: Any) -> dict:
    """Skeleton equivalence after simplification (the recovery criterion of
    SSDE and SymPlex), plus whether ``found − truth`` is proved zero. Raises
    :class:`FexError` ``E-FEX-011`` unless both are expressions."""
    from ._fex_finalize import recovery_metric as _rm

    return _rm(found, truth)


def family_symbols(problem: Problem) -> Any:
    """``d``, ``i``, ``x``, ``t`` and ``sum(...)`` for writing the
    ``family=`` of :func:`finalize` in symbolic dimension."""
    from ._fex_finalize import family_symbols as _fs

    return _fs(problem)


def snap_candidates(value: float, *, max_den: int = 1000, rel_tol: float = 1e-7) -> list:
    """Exact constants near ``value``, simplest first, as ``(constant, how)``
    with ``how`` in ``rational`` / ``constant`` / ``pslq`` / ``lll``."""
    from ._fex_finalize import snap_candidates as _sc

    return _sc(value, max_den=max_den, rel_tol=rel_tol)
