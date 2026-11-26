"""Symbolic analysis of a FEX problem before the search (milestone F4, R9).

:func:`analyze` reads a :class:`alkahest.fex.Problem` — its operator written
in the jet symbols ``u, lap, grad[i], ut`` and the coordinates, and its
source terms — and proposes how to search it:

* **operator vocabulary**: the unary functions in the source terms, split
  into those applied to a single coordinate (candidate *leaf* operators,
  ``cos(x_i)``, ``x_i²``) and those applied to a combination of coordinates
  (candidate *inner* operators, ``exp(Σ…)``, ``sin(Σ…)``);
* **separability**: an operator without mixed terms and additively
  separable data give a sum of one-variable functions (``u = Σ fᵢ(xᵢ)``),
  and for a linear homogeneous operator a product (``log u`` additive);
* **variable dependence** (Song–Cameron–Yang): the smallest set of
  coordinates ``S`` such that, with ``∂u/∂x_j = 0`` for ``j ∉ S``, neither the
  operator nor the data mention ``x_j`` — the problem then reduces to
  ``|S|`` dimensions;
* **dispersion relation** of a linear constant-coefficient operator
  ``P(k)`` (``u = e^{k·x}``): real zeros give exponentials, zeros of
  ``P(ik)`` give sines and cosines, a first-order operator gives any profile
  of a linear form (characteristics);
* **scaling symmetry**: the weight ``a`` with ``x → s x``, ``u → s^a u``
  leaving the equation invariant, for homogeneous polynomial data, gives the
  degree of a polynomial solution;
* **radial symmetry**: rotation-invariant operator and data propose
  ``g(|x|²)``;
* **dependent-variable substitutions** ``u = e^v`` (linear homogeneous
  operators, eigenproblems — ``−Δu + V u = λu`` becomes
  ``−Δv − |∇v|² + V = λ`` with ``λ`` fitted as the mean residual — and
  operators whose data contain ``exp``) and ``u = log w``. The substituted
  operator is derived with M0's jet layer: ``D(φ(v), x)`` and
  ``D(φ(v), x, x)`` of an applied unknown ``v(x)``, converted to jets with
  :func:`alkahest.experimental.functions_to_jets`, give the images of
  ``u_x``, ``u_xx`` (hence ``Δu``) and ``u_t``.

Every test that decides a proposal is either exact (symbolic substitution
and simplification) or a numerical *heuristic* used only to rank proposals:
nothing here certifies anything. A proposal only restricts where the search
looks; :func:`alkahest.fex.finalize` still verifies whatever is found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import alkahest as _ak

__all__ = ["Analysis", "Proposal", "analyze", "reduce_problem", "transform_problem"]

_REMEDIATION = {
    "E-FEX-012": (
        "propose only the transforms analyze() lists for this problem: u = e^v needs an "
        "operator that is homogeneous in u (rhs = 0) or an eigenproblem, and boundary data "
        "that are positive or absent; a variable reduction needs operator and data free of "
        "the dropped coordinates once their derivatives vanish"
    ),
}


def _err(code: str, message: str) -> Exception:
    from .alkahest import FexError

    rem = _REMEDIATION[code]
    exc = FexError(f"[{code}] {message}\nRemediation: {rem}")
    exc.code = code
    exc.remediation = rem
    exc.span = None
    return exc


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _children(e: Any) -> list[Any]:
    nd = e.node()
    out = []
    for a in nd[1:]:
        if isinstance(a, (list, tuple)):
            out.extend(c for c in a if isinstance(c, _ak.Expr))
        elif isinstance(a, _ak.Expr):
            out.append(a)
    return out


def _nodes(e: Any):
    stack, seen = [e], set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        yield n
        stack.extend(_children(n))


def _symbols(e: Any) -> set:
    return {n for n in _nodes(e) if n.node()[0] == "symbol"}


def _is_expr(v: Any) -> bool:
    return isinstance(v, _ak.Expr)


def _pow_parts(e: Any) -> tuple[Any, int | None]:
    nd = e.node()
    base, ex = (nd[1], nd[2]) if len(nd) > 2 else tuple(nd[1])
    k = None
    if isinstance(ex, int):
        k = ex
    elif _is_expr(ex) and ex.node()[0] == "integer":
        k = int(ex.node()[1])
    return base, k


_POW_NAME = {1: "id", 2: "sq", 3: "cube", 4: "quartic"}
_FUNC_NAME = {"exp": "exp", "sin": "sin", "cos": "cos", "log": "log", "tanh": "tanh", "abs": "abs"}


def _single_coordinate(e: Any, coords: set) -> Any | None:
    """The coordinate if ``e`` is an affine function of exactly one
    coordinate (``c·x_i + b``), else ``None``."""
    syms = _symbols(e) & coords
    if len(syms) != 1:
        return None
    (x,) = syms
    if e == x:
        return x
    # Affine in x: every node on the way is add / mul by numbers.
    for n in _nodes(e):
        k = n.node()[0]
        if k in ("func", "pow"):
            return None
    return x


def _vocabulary(exprs: list[Any], coords: list[Any]) -> tuple[set, set]:
    """(leaf ops, inner ops) appearing in ``exprs``."""
    cs = set(coords)
    leaf, inner = set(), set()
    for ex in exprs:
        if not _is_expr(ex):
            continue
        for n in _nodes(ex):
            nd = n.node()
            if nd[0] == "func" and nd[1] in _FUNC_NAME:
                arg = nd[2][0]
                if not (_symbols(arg) & cs):
                    continue
                (leaf if _single_coordinate(arg, cs) is not None else inner).add(_FUNC_NAME[nd[1]])
            elif nd[0] == "pow":
                base, k = _pow_parts(n)
                if k is None or not (_symbols(base) & cs):
                    continue
                name = _POW_NAME.get(abs(k))
                if name is None:
                    continue
                if base in cs:
                    leaf.add(name)
                else:
                    inner.add(name)
            elif nd[0] == "symbol" and n in cs:
                leaf.add("id")
    return leaf, inner


def _term_lists(e: Any, cap: int = 4096) -> list[list[Any]]:
    """``e`` as a sum of products (each a list of factors), distributing
    products and small positive integer powers over sums; float
    coefficients are fine (unlike ``expand``)."""
    nd = e.node()
    if nd[0] == "add":
        out: list[list[Any]] = []
        for c in nd[1]:
            out.extend(_term_lists(c, cap))
        return out
    if nd[0] == "mul":
        out = [[]]
        for c in nd[1]:
            parts = _term_lists(c, cap)
            if len(out) * len(parts) > cap:
                return [[e]]
            out = [a + b for a in out for b in parts]
        return out
    if nd[0] == "pow":
        base, k = _pow_parts(e)
        if k is not None and 2 <= k <= 4 and base.node()[0] == "add":
            out = [[]]
            parts = _term_lists(base, cap)
            for _ in range(k):
                if len(out) * len(parts) > cap:
                    return [[e]]
                out = [a + b for a in out for b in parts]
            return out
    return [[e]]


def _additive_terms(e: Any) -> list[Any]:
    out = []
    for fs in _term_lists(e):
        t = fs[0] if fs else 1
        for f in fs[1:]:
            t = t * f
        out.append(t)
    return out


def _additively_separable(e: Any, coords: list[Any]) -> bool:
    if not _is_expr(e):
        return True
    cs = set(coords)
    return all(len(_symbols(t) & cs) <= 1 for t in _additive_terms(e))


def _simplify(e: Any) -> Any:
    r = _ak.simplify(e)
    return getattr(r, "value", r)


def _subs(e: Any, mapping: dict) -> Any:
    if not _is_expr(e):
        return e
    return _ak.subs(e, mapping)


def _numeric(e: Any, env: dict[Any, float]) -> float:
    try:
        return float(_ak.eval_expr(e, env))
    except Exception:
        return math.nan


# ---------------------------------------------------------------------------
# The operator of a problem, in jet symbols
# ---------------------------------------------------------------------------


@dataclass
class OperatorView:
    """``F(u, ∇u, Δu, u_t, x) = rhs`` in a problem's :class:`~alkahest.fex.Jet`."""

    op: Any
    rhs: Any
    bc: Any
    eigen: bool
    loss: str
    jet: Any

    @property
    def jets(self) -> list[Any]:
        j = self.jet
        out = [j.u, j.lap, *j.grad]
        if j.ut is not None:
            out.append(j.ut)
        return out


def operator_view(problem: Any) -> OperatorView:
    """The problem's operator as an expression in its jet symbols."""
    spec = problem.spec or {}
    jet = problem.jet
    kind = spec.get("kind")
    V = spec.get("potential")
    rhs = spec.get("rhs")
    bc = spec.get("bc")
    loss = problem.loss
    eigen = loss == "rayleigh"
    if kind == "poisson":
        op = -jet.lap
        if V is not None and not (isinstance(V, (int, float)) and V == 0):
            op = op + V * jet.u
    elif kind == "transport":
        cs = spec.get("grad_coeffs") or []
        op = sum((c * g for c, g in zip(cs, jet.grad) if c != 0), 0 * jet.u)
    elif kind == "cubic_schrodinger":
        op = -jet.lap + jet.u**3
        if V is not None:
            op = op + V * jet.u
    elif kind == "expr":
        op = spec["operator"]
    else:
        op = None
    if rhs is None:
        rhs = 0
    return OperatorView(op=op, rhs=rhs, bc=bc, eigen=eigen, loss=loss, jet=jet)


def _jet_degree(op: Any, jets: list[Any]) -> int | None:
    """Total polynomial degree of ``op`` in the jet symbols, or ``None`` when
    a jet sits inside a non-polynomial function."""
    js = set(jets)
    for n in _nodes(op):
        nd = n.node()
        if nd[0] == "func" and (_symbols(n) & js):
            return None
        if nd[0] == "pow":
            base, k = _pow_parts(n)
            if (_symbols(base) & js) and (k is None or k < 0):
                return None
    best = 0
    for fs in _term_lists(op):
        deg = 0
        for n in fs:
            if n in js:
                deg += 1
            elif n.node()[0] == "pow":
                base, k = _pow_parts(n)
                if base in js:
                    deg += k or 0
        best = max(best, deg)
    return best


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    """One way to search a problem.

    ``unary`` / ``binary`` are the operator tables of the template to use;
    ``slot_ops`` restricts slots by kind (``"leaf"``, ``"unary"`` — inner
    unary slots — and ``"binary"``) to subsets of those tables. ``transform``
    is ``None``, ``"exp"`` (search ``v`` with ``u = e^v``) or ``"log"``
    (``u = log w``); ``variables`` the coordinates the solution depends on
    (``None`` = all).
    """

    name: str
    reason: str
    unary: list[str]
    binary: list[str]
    slot_ops: dict[str, list[str]] = field(default_factory=dict)
    shape: str = "depth2_sub"
    transform: str | None = None
    variables: list[int] | None = None
    loss: str | None = None
    priority: float = 0.0


@dataclass
class Analysis:
    """What :func:`analyze` found, and the proposals in the order to try."""

    linear: bool | None
    homogeneous: bool
    eigen: bool
    constant_coefficients: bool | None
    leaf_vocabulary: list[str]
    inner_vocabulary: list[str]
    separable: bool
    radial: bool
    depends_on: list[int]
    dispersion: dict | None
    scaling_degree: int | None
    transforms: list[str]
    proposals: list[Proposal]
    notes: list[str]

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["proposals"] = [p.__dict__ for p in self.proposals]
        return out


def _order_ops(ops: set, table: list[str]) -> list[str]:
    return [o for o in table if o in ops]


_ALL_UNARY = ["0", "1", "id", "sq", "cube", "quartic", "exp", "sin", "cos", "log", "abs", "tanh"]


def _data_rotation_invariant(exprs: list[Any], coords: list[Any], rng) -> bool:
    d = len(coords)
    if d < 2:
        return False
    for ex in exprs:
        if not _is_expr(ex) or not (_symbols(ex) & set(coords)):
            continue
        for _ in range(3):
            x = rng.uniform(-0.7, 0.7, d)
            q, _r = np.linalg.qr(rng.normal(size=(d, d)))
            y = q @ x
            a = _numeric(ex, dict(zip(coords, x)))
            b = _numeric(ex, dict(zip(coords, y)))
            if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > 1e-9 * (
                1 + abs(a) + abs(b)
            ):
                return False
    return True


def _homogeneous_degree(ex: Any, coords: list[Any], rng) -> int | None:
    """``k`` with ``f(s x) = s^k f(x)`` (checked at random points, s = 2, 3)."""
    if not _is_expr(ex):
        return 0 if isinstance(ex, (int, float)) and ex != 0 else None
    if not (_symbols(ex) & set(coords)):
        return 0
    d = len(coords)
    ks = set()
    for s in (2.0, 3.0):
        x = rng.uniform(0.2, 0.8, d)
        a = _numeric(ex, dict(zip(coords, x)))
        b = _numeric(ex, dict(zip(coords, s * x)))
        if not (math.isfinite(a) and math.isfinite(b)) or a == 0 or b / a <= 0:
            return None
        k = math.log(b / a) / math.log(s)
        if abs(k - round(k)) > 1e-8:
            return None
        ks.add(round(k))
    return ks.pop() if len(ks) == 1 else None


def _dependence(view: OperatorView, extra: list[Any]) -> list[int]:
    """Smallest coordinate set closed under the operator (see module doc)."""
    jet = view.jet
    coords = list(jet.x)
    d = len(coords)
    time = getattr(jet, "time", None)
    data = [view.rhs, view.bc, *extra]
    s = set()
    for ex in data:
        if _is_expr(ex):
            s |= {i for i, c in enumerate(coords) if c in _symbols(ex)}
    if time is not None:
        s.add(time)
    if view.op is None:
        return list(range(d))
    for _ in range(d + 1):
        zero = {jet.grad[j]: 0 for j in range(d) if j not in s}
        red = _subs(view.op, zero) if zero else view.op
        red = _simplify(red)
        new = {i for i, c in enumerate(coords) if c in _symbols(red)} | s
        if new == s:
            break
        s = new
    return sorted(s)


def _dispersion(view: OperatorView, rng) -> dict | None:
    """``P(k)`` for a linear constant-coefficient homogeneous operator."""
    jet = view.jet
    coords = list(jet.x)
    if view.op is None:
        return None
    if _symbols(view.op) & set(coords):
        return None
    deg = _jet_degree(view.op, view.jets)
    if deg != 1:
        return None
    d = len(coords)
    pool = jet.pool
    ks = [pool.symbol(f"k{i}") for i in range(d)]
    time = jet.time
    mp = {jet.u: 1, jet.lap: sum(ks[i] ** 2 for i in range(d) if i != time)}
    for i in range(d):
        mp[jet.grad[i]] = ks[i]
    if jet.ut is not None and time is not None:
        mp[jet.ut] = ks[time]
    P = _simplify(_subs(view.op, mp))
    order = 2 if any(_numeric(P, dict(zip(ks, [0.0] * d))) != 0 for _ in [0]) else None
    # Real zeros of P and of P(i k) by sign changes at random directions
    # (a heuristic for ranking proposals, nothing more).
    vals = []
    for _ in range(64):
        k = rng.normal(size=d)
        vals.append(_numeric(P, dict(zip(ks, k))))
    vals = np.array(vals)
    real = bool(np.nanmin(vals) < 0 < np.nanmax(vals)) or bool(np.any(np.abs(vals) < 1e-12))
    first_order = view.jet.lap not in _symbols(view.op)
    # P(i k): for a real polynomial, the even part is real and the odd part
    # imaginary; a zero needs both to vanish. Second-order even operators
    # (lap only): P(ik) = −P(k).
    imag = None
    if not first_order:
        neg = _simplify(_subs(P, {k: -k for k in ks}))
        even = _simplify((P + neg) / 2)
        odd = _simplify((P - neg) / 2)
        ev = np.array([_numeric(even, dict(zip(ks, rng.normal(size=d)))) for _ in range(64)])
        odd_zero = all(
            abs(_numeric(odd, dict(zip(ks, rng.normal(size=d))))) < 1e-12 for _ in range(8)
        )
        imag = bool(odd_zero and np.nanmin(ev) < 0 < np.nanmax(ev))
    del order
    return {
        "symbol": P,
        "wavevector": ks,
        "first_order": first_order,
        "real_zeros": real,
        "imaginary_zeros": imag,
    }


def _separable_operator(view: OperatorView) -> bool:
    """No coefficient couples coordinates: every additive term of the
    operator mentions at most one coordinate, and only through ``V``-type
    terms (the jets ``lap`` and ``grad[i]`` never multiply each other)."""
    if view.op is None:
        return False
    coords = list(view.jet.x)
    return _additively_separable(view.op, coords)


def analyze(
    problem: Any,
    *,
    seed: int = 0,
    unary: list[str] | None = None,
    binary: list[str] | None = None,
    shape: str = "depth2_sub",
) -> Analysis:
    """Analyse ``problem`` symbolically and propose restricted searches.

    Returns an :class:`Analysis` whose ``proposals`` are ordered from most to
    least specific; the last one is always the unrestricted search over
    ``unary`` / ``binary`` (default: the paper's tables).
    """
    rng = np.random.default_rng(seed)
    view = operator_view(problem)
    jet = view.jet
    coords = list(jet.x)
    d = len(coords)
    notes: list[str] = []
    V = (problem.spec or {}).get("potential")
    data = [view.rhs, view.bc, V]
    # Coefficient parts of the operator (everything that is not a jet).
    if view.op is not None and problem.spec.get("kind") == "expr":
        data.append(_subs(view.op, dict.fromkeys(view.jets, 0)))
    leaf_v, inner_v = _vocabulary(data, coords)
    rhs_zero = isinstance(view.rhs, (int, float)) and view.rhs == 0
    deg = _jet_degree(view.op, view.jets) if view.op is not None else None
    linear = None if deg is None else deg <= 1
    homogeneous = bool(rhs_zero)
    const_coeff = None if view.op is None else not (_symbols(view.op) & set(coords))
    sep = (
        all(_additively_separable(e, coords) for e in data if e is not None)
        and _separable_operator(view)
        and (jet.time is None)
    )
    radial = (
        view.op is not None
        and jet.time is None
        and not any(
            g in _symbols(view.op) for g in jet.grad
        )  # |∇u|² written per coordinate is not detected
        and _data_rotation_invariant([e for e in data if e is not None], coords, rng)
    )
    depends = _dependence(view, [V])
    disp = _dispersion(view, rng) if (linear and homogeneous) else None
    scaling = None
    if view.op is not None and const_coeff and linear and problem.spec.get("kind") == "poisson":
        k = _homogeneous_degree(view.rhs, coords, rng)
        kb = _homogeneous_degree(view.bc, coords, rng) if view.bc is not None else None
        if k is not None and (kb is None or kb == k + 2):
            scaling = k + 2
            notes.append(
                f"scaling x→s·x, u→s^{scaling}·u leaves −Δu = rhs invariant (rhs of degree {k})"
            )
    transforms: list[str] = []
    if view.eigen:
        transforms.append("exp")
        notes.append("eigenproblem: u = e^v turns −Δu + Vu = λu into −Δv − |∇v|² + V = λ")
    elif homogeneous and deg is not None and "exp" in (leaf_v | inner_v) and jet.time is None:
        transforms.append("exp")
        notes.append("homogeneous operator with exp in its data: try u = e^v")
    if "log" in (leaf_v | inner_v) and not view.eigen:
        transforms.append("log")

    # Vocabulary of the solution.
    leaves = {"0", "id"} | {o for o in leaf_v if o in _ALL_UNARY}
    inner = {"id"} | {o for o in inner_v if o in _ALL_UNARY}
    if scaling is not None and scaling in _POW_NAME:
        leaves.add(_POW_NAME[scaling])
    proposals: list[Proposal] = []
    full_unary = list(unary or ["0", "1", "id", "sq", "cube", "quartic", "exp", "sin", "cos"])
    full_binary = list(binary or ["+", "*", "-"])

    def prop(name, reason, leaf_ops, inner_ops, binary=("+", "*", "-"), **kw):
        leaf_ops = set(leaf_ops)
        inner_ops = set(inner_ops)
        table = _order_ops(leaf_ops | inner_ops | {"0", "id"}, _ALL_UNARY)
        proposals.append(
            Proposal(
                name=name,
                reason=reason,
                unary=table,
                binary=[b for b in full_binary if b in binary] or list(full_binary),
                slot_ops={
                    "leaf": _order_ops(leaf_ops, table),
                    "unary": _order_ops(inner_ops, table),
                },
                shape=shape,
                **kw,
            )
        )

    red = depends if len(depends) < d else None
    if red is not None:
        notes.append(
            f"u depends only on {[str(coords[i]) for i in red]}: with ∂u/∂x_j = 0 for the "
            "others, neither the operator nor the data mention them"
        )
    for tr in transforms:
        if tr == "exp":
            # v = log u: products of one-variable functions become sums; the
            # data's leaf functions and polynomial growth (WKB: V ~ x^m gives
            # v ~ x^{(m+2)/2}) are the leaf vocabulary of v.
            vleaves = {"0", "id"} | {o for o in leaf_v if o in _ALL_UNARY}
            pdeg = max((k for k, n in _POW_NAME.items() if n in leaf_v), default=0)
            if pdeg:
                for k in range(2, max(2, (pdeg + 2) // 2) + 1):
                    vleaves.add(_POW_NAME[min(k, 4)])
            if sep:
                prop(
                    "exp-separable",
                    "u = e^v with a separable operator and data: v = Σ fᵢ(xᵢ), a sum of leaves",
                    vleaves,
                    {"id"},
                    binary=("+",),
                    transform="exp",
                    variables=red,
                    loss="centered" if view.eigen else None,
                    priority=3,
                )
            prop(
                "exp-vocabulary",
                "u = e^v; v built from the data's functions",
                vleaves,
                {"id"} | {o for o in inner_v if o in _ALL_UNARY},
                transform="exp",
                variables=red,
                loss="centered" if view.eigen else None,
                priority=2,
            )
        if tr == "log":
            prop(
                "log-vocabulary",
                "u = log w; w built from the data's functions",
                leaves | {"exp"},
                inner | {"exp"},
                transform="log",
                variables=red,
                priority=1,
            )
    if disp is not None and disp["first_order"]:
        prop(
            "characteristics",
            "first-order linear constant-coefficient operator: u = g(k·x) with P(k) = 0",
            {"0", "id"},
            inner | {"exp", "sin", "cos", "sq"},
            variables=red,
            priority=2.5,
        )
    if disp is not None and not disp["first_order"]:
        waves = set()
        if disp["real_zeros"]:
            waves.add("exp")
        if disp["imaginary_zeros"]:
            waves |= {"sin", "cos"}
        if waves:
            prop(
                "plane-waves",
                f"dispersion relation P(k) = {disp['symbol']}: "
                + ("real zeros (exponentials) " if disp["real_zeros"] else "")
                + ("imaginary zeros (sin / cos)" if disp["imaginary_zeros"] else ""),
                leaves | waves,
                inner | waves,
                variables=red,
                priority=1.5,
            )
    if scaling is not None:
        prop(
            "scaling",
            f"homogeneous of degree {scaling} under the scaling symmetry",
            {"0", _POW_NAME.get(scaling, "sq")} | ({"id"} if scaling == 1 else set()),
            {"id"},
            binary=("+", "-"),
            variables=red,
            priority=2.8,
        )
    if radial:
        prop(
            "radial",
            "rotation-invariant operator and data: u = g(|x|²)",
            {"0", "sq"},
            inner | {"exp", "sin", "cos", "sq"},
            variables=red,
            priority=2.2,
        )
    if sep and linear and not view.eigen and not homogeneous:
        prop(
            "separable-sum",
            "linear operator, separable data: u = Σ fᵢ(xᵢ)",
            leaves,
            {"id"},
            binary=("+", "-"),
            variables=red,
            priority=2.6,
        )
    prop(
        "vocabulary",
        "the data's functions (leaf and inner)",
        leaves,
        inner,
        variables=red,
        priority=1.0,
    )
    if red is not None:
        proposals.append(
            Proposal(
                name="reduced",
                reason="variable dependence only, paper operator tables",
                unary=list(full_unary),
                binary=list(full_binary),
                variables=red,
                priority=1.2,
                shape=shape,
            )
        )
    proposals.append(
        Proposal(
            name="paper",
            reason="no restriction: the paper's operator tables",
            unary=list(full_unary),
            binary=list(full_binary),
            priority=0.0,
            shape=shape,
        )
    )
    proposals.sort(key=lambda p: -p.priority)
    # Drop exact duplicates (same tables, transform, variables).
    seen, uniq = set(), []
    for p in proposals:
        k = (
            tuple(p.unary),
            tuple(p.binary),
            tuple(sorted((k, tuple(v)) for k, v in p.slot_ops.items())),
            p.transform,
            tuple(p.variables or ()),
            p.shape,
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return Analysis(
        linear=linear,
        homogeneous=homogeneous,
        eigen=view.eigen,
        constant_coefficients=const_coeff,
        leaf_vocabulary=sorted(leaf_v),
        inner_vocabulary=sorted(inner_v),
        separable=sep,
        radial=bool(radial),
        depends_on=depends,
        dispersion=None if disp is None else {k: v for k, v in disp.items() if k != "wavevector"},
        scaling_degree=scaling,
        transforms=transforms,
        proposals=uniq,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Transforms of the dependent variable (through M0's jet layer)
# ---------------------------------------------------------------------------


def _chain_rule(kind: str, pool: Any) -> tuple[Any, Any, Any, Any, Any]:
    """``(phi(v), phi'(v), A(v), B(v), v)`` with ``D(phi(v), x, x) = A·v_xx + B·v_x²``,
    derived with M0's jet layer on an applied unknown ``v(x)``."""
    from .experimental import D, Function, functions_to_jets

    x = pool.symbol("_fex_x")
    vf = Function("_fex_v")(x)
    phi = {"exp": _ak.exp, "log": _ak.log}[kind]
    u = phi(vf)
    (d1, d2), names = functions_to_jets([D(u, x), D(u, x, x)], [vf])
    inv = {str(k): v for k, v in names.items()}
    v = inv["_fex_v(_fex_x)"]
    vx = inv["Derivative(_fex_v(_fex_x), _fex_x)"]
    vxx = inv["Derivative(_fex_v(_fex_x), _fex_x, _fex_x)"]
    first = _simplify(_subs(d1, {vx: 1}))
    a = _simplify(_subs(d2, {vxx: 1, vx: 0}))
    b = _simplify(_subs(d2, {vxx: 0, vx: 1}))
    # The second derivative is A·v_xx + B·v_x² exactly (chain rule).
    chk = _simplify(d2 - a * vxx - b * vx**2)
    if not (chk.node()[0] == "integer" and chk.node()[1] == "0"):
        raise _err("E-FEX-012", f"unexpected chain rule for {kind}: residual {chk}")
    return _subs(phi(v), {}), first, a, b, v


def transform_problem(problem: Any, kind: str = "exp", *, loss: str | None = None) -> Any:
    """The problem for ``v`` under ``u = e^v`` (``kind="exp"``) or
    ``u = log v`` (``kind="log"``), as a new :class:`~alkahest.fex.Problem`
    on the same coordinates.

    For ``exp`` a homogeneous operator is divided by ``e^v`` (so a linear
    operator becomes one in ``v`` without exponentials). An eigenproblem
    (Rayleigh loss) becomes ``−Δv − |∇v|² + V = λ`` with the centred loss,
    whose fitted mean residual is ``λ``. Boundary data ``g`` become
    ``log g`` (``exp``) or ``e^g`` (``log``); a zero boundary condition
    (unreachable by ``e^v``) is dropped and the solution is chosen later on
    the original problem.

    Raises :class:`FexError` ``E-FEX-012`` when the transform does not apply.
    """
    from . import fex as _fex

    if kind not in ("exp", "log"):
        raise _err("E-FEX-012", f"unknown transform {kind!r}; use 'exp' or 'log'")
    view = operator_view(problem)
    if view.op is None:
        raise _err("E-FEX-012", "the problem's operator is not available symbolically")
    jet = view.jet
    rhs_zero = isinstance(view.rhs, (int, float)) and view.rhs == 0
    if kind == "exp" and not (rhs_zero or view.eigen):
        raise _err("E-FEX-012", "u = e^v needs a homogeneous problem (rhs = 0) or an eigenproblem")
    pool = jet.pool
    new = _fex.Jet(jet.dim, jet.time, pool=pool)
    # The new jet shares the coordinates of the old one.
    new.x = list(jet.x)
    phi_v, d1, a, b, vsym = _chain_rule(kind, pool)
    V = new.u
    phi = _subs(phi_v, {vsym: V})
    d1v = _subs(d1, {vsym: V})
    av = _subs(a, {vsym: V})
    bv = _subs(b, {vsym: V})
    time = jet.time
    spatial = [i for i in range(jet.dim) if i != time]
    grad2 = sum((new.grad[i] ** 2 for i in spatial), 0 * V)
    mp = {jet.u: phi, jet.lap: av * new.lap + bv * grad2}
    for i in range(jet.dim):
        mp[jet.grad[i]] = d1v * new.grad[i]
    if jet.ut is not None:
        mp[jet.ut] = d1v * new.ut
    op = _subs(view.op, mp)
    op = _simplify(op / phi) if kind == "exp" else _simplify(op)
    bc = view.bc
    bc_weight = (problem.spec or {}).get("bc_weight", 100.0)
    if bc is not None:
        zero = isinstance(bc, (int, float)) and bc == 0
        if zero or (_is_expr(bc) and not (_symbols(bc) & set(jet.x)) and _numeric(bc, {}) == 0):
            bc, bc_weight = None, 0.0
        else:
            bc = _simplify(_ak.log(bc) if kind == "exp" else _ak.exp(bc))
    if view.eigen:
        loss = loss or "centered"
        rhs = 0.0
        bc, bc_weight = None, 0.0
    else:
        loss = loss or "least_squares"
        rhs = view.rhs if kind != "exp" else 0.0
    out = _fex.Problem.from_operator(new, op, rhs=rhs, bc=bc, bc_weight=bc_weight, loss=loss)
    out.spec["transform"] = kind
    out.spec["parent"] = problem
    return out


def reduce_problem(problem: Any, variables: list[int]) -> Any:
    """The problem restricted to solutions that depend only on the
    coordinates ``variables`` (the others' derivatives set to zero), as a new
    problem of dimension ``len(variables)`` whose coordinates are the kept
    ones.

    Raises :class:`FexError` ``E-FEX-012`` when the operator or the data
    still mention a dropped coordinate.
    """
    from . import fex as _fex

    view = operator_view(problem)
    if view.op is None:
        raise _err("E-FEX-012", "the problem's operator is not available symbolically")
    jet = view.jet
    keep = sorted({int(i) for i in variables})
    if not keep or keep[-1] >= jet.dim:
        raise _err("E-FEX-012", f"variables {variables} out of range for dimension {jet.dim}")
    drop = [j for j in range(jet.dim) if j not in keep]
    time = jet.time
    if time is not None and time not in keep:
        raise _err("E-FEX-012", "the time coordinate cannot be dropped")
    new = _fex.Jet(len(keep), None if time is None else keep.index(time), pool=jet.pool)
    new.x = [jet.x[i] for i in keep]
    mp = {jet.u: new.u, jet.lap: new.lap}
    for k, i in enumerate(keep):
        mp[jet.grad[i]] = new.grad[k]
    for j in drop:
        mp[jet.grad[j]] = 0
    if jet.ut is not None:
        mp[jet.ut] = new.ut
    op = _simplify(_subs(view.op, mp))
    dropped = {jet.x[j] for j in drop}
    for name, ex in (("operator", op), ("rhs", view.rhs), ("boundary data", view.bc)):
        if _is_expr(ex) and (_symbols(ex) & dropped):
            raise _err(
                "E-FEX-012",
                f"the {name} still depends on a dropped coordinate after the reduction",
            )
    spec = problem.spec or {}
    loss = problem.loss
    if loss == "rayleigh":
        raise _err("E-FEX-012", "reduce the eigenproblem after the u = e^v transform")
    kw = {}
    if loss == "normalized":
        kw["power"] = spec.get("power", 3.0)
    out = _fex.Problem.from_operator(
        new,
        op,
        rhs=view.rhs,
        bc=view.bc,
        bc_weight=spec.get("bc_weight", 100.0),
        loss=loss,
        **kw,
    )
    out.spec["reduced_from"] = problem
    out.spec["variables"] = keep
    for k in ("transform", "parent"):
        if k in spec:
            out.spec[k] = spec[k]
    return out
