"""alkahest.tensor — tensor calculus on a coordinate chart (F27).

Two layers:

**Components.** :class:`Metric` takes an ``n x n`` symmetric matrix of
expressions and ``n`` coordinate symbols and computes, on demand and cached,
the inverse metric, the Christoffel symbols, the Riemann, Ricci, Einstein and
Weyl tensors, the Ricci and Kretschmann scalars and the geodesic equations.
Entries may contain parameters, ``sin``/``cos``/``sinh``/``cosh``/``exp``/
``log``/radicals and **undefined functions** of the coordinates such as the
FLRW scale factor ``a(t)``.  Every component is computed in an exact field —
integer polynomials over the coordinates, the parameters and one generator per
transcendental atom, with ``sin² = 1 − cos²`` applied and denominators made
free of ``sin`` and radicals — whose normal form is canonical: a component
that vanishes identically comes out as the literal ``0`` (that is how
``Metric.is_ricci_flat`` *proves* the Kerr metric is a vacuum solution), and a
non-zero one is never reported as zero.  Nothing is sampled, no float enters.
Only independent components are computed (``Γ_{abc}`` for ``b ≤ c``,
``R_{abcd}`` for ``a < b, c < d, (a,b) ≤ (c,d)``); the rest follow by symmetry.

**Abstract indices.** A :class:`Tensor` carries its components and the
variance of each slot (``"u"`` up, ``"d"`` down).  Indexing it with
:class:`TensorIndex` objects — ``+a`` up, ``-a`` down — gives an
:class:`IndexedTensor`; a repeated index (once up, once down) is summed
(Einstein convention), a slot asked for in the other variance is raised or
lowered with the metric, and ``+``/``-``/``*`` combine them::

    a, b = tensor_indices("a b")
    G = m.ricci()[-a, -b] - m.ricci_scalar() * m.as_tensor()[-a, -b] / 2
    G.components          # == m.einstein().components

Conventions (MTW / Carroll / Wald):
``Γ^a_{bc} = ½ g^{ad}(∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})``,
``R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce}Γ^e_{db} − Γ^a_{de}Γ^e_{cb}``,
``R_{bd} = R^a_{bad}``, ``G_{ab} = R_{ab} − ½ R g_{ab}``, ``K = R_{abcd}R^{abcd}``.
The round 2-sphere of radius ``ρ`` has ``R = 2/ρ²``.

Refusals are :class:`TensorError` with a stable ``E-TENSOR-NNN`` code
(``001`` shape, ``002`` not symmetric, ``003`` degenerate, ``004`` a float,
``005`` an unrepresentable node, ``006`` a missing derivative, ``007`` a
vanishing denominator, ``008`` dimension too small, ``009`` an index error);
a tripped ``Budget`` raises ``BudgetExceededError``.

Example
-------
>>> import alkahest as ak
>>> p = ak.ExprPool()
>>> t, r, th, ph, M = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M"))
>>> f = 1 - 2 * M / r
>>> m = ak.Metric([[-f, 0, 0, 0], [0, 1 / f, 0, 0], [0, 0, r**2, 0],
...                [0, 0, 0, r**2 * ak.sin(th) ** 2]], [t, r, th, ph])
>>> m.is_ricci_flat()
True
>>> print(m.kretschmann())
(48 * M^2 * r^-6)
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any

from .alkahest import (
    Expr,
    TensorError,
    _tensor_canonical,
    _tensor_derivative,
    _TensorMetric,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "IndexedTensor",
    "Metric",
    "Tensor",
    "TensorError",
    "TensorIndex",
    "tensor_indices",
]

_INDEX_REMEDIATION = (
    "each index may appear at most twice in a term, once up (+a) and once down (-a); "
    "the free indices of every summand must agree"
)


def _err(code: str, message: str, remediation: str | None = None) -> TensorError:
    full = f"[{code}] {message}"
    if remediation:
        full += f"\nRemediation: {remediation}"
    e = TensorError(full)
    e.code = code
    e.remediation = remediation
    e.span = None
    return e


def _is_zero(e: Any) -> bool:
    """``True`` for a literal zero (an ``int`` or the ``Expr`` ``0``)."""
    if isinstance(e, Expr):
        return e.node_tag() == "integer" and str(e) == "0"
    return e == 0


def _scalar_ok(x: Any) -> bool:
    return isinstance(x, (Expr, int, Rational)) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------


class TensorIndex:
    """An abstract index.  ``+a`` (or ``a``) is contravariant, ``-a`` covariant."""

    __slots__ = ("name", "up")

    def __init__(self, name: str, up: bool = True):
        if not isinstance(name, str) or not name:
            raise TypeError("an index name must be a non-empty string")
        self.name = name
        self.up = bool(up)

    def __neg__(self) -> TensorIndex:
        return TensorIndex(self.name, not self.up)

    def __pos__(self) -> TensorIndex:
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TensorIndex) and (self.name, self.up) == (other.name, other.up)

    def __hash__(self) -> int:
        return hash((self.name, self.up))

    def __repr__(self) -> str:
        return self.name if self.up else f"-{self.name}"


def tensor_indices(names: str | Iterable[str]) -> tuple[TensorIndex, ...]:
    """Make indices from ``"a b c"`` (spaces or commas) or a list of names.

    Returns a tuple of contravariant :class:`TensorIndex`; negate one for the
    covariant position.
    """
    if isinstance(names, str):
        names = names.replace(",", " ").split()
    return tuple(TensorIndex(n) for n in names)


# ---------------------------------------------------------------------------
# Component storage helpers: a dict {index tuple: Expr} holding non-zeros.
# ---------------------------------------------------------------------------


def _from_nested(comps: Any, rank: int) -> dict[tuple[int, ...], Any]:
    out: dict[tuple[int, ...], Any] = {}

    def walk(x: Any, key: tuple[int, ...]) -> None:
        if len(key) == rank:
            if not _is_zero(x):
                out[key] = x
            return
        for i, y in enumerate(x):
            walk(y, (*key, i))

    walk(comps, ())
    return out


def _to_nested(data: dict[tuple[int, ...], Any], rank: int, dim: int, zero: Any) -> Any:
    def build(key: tuple[int, ...]) -> Any:
        if len(key) == rank:
            return data.get(key, zero)
        return [build((*key, i)) for i in range(dim)]

    return build(())


def _variance(spec: str | Sequence[Any]) -> str:
    if isinstance(spec, str):
        s = spec.replace("^", "u").replace("_", "d").lower()
    else:
        s = "".join("u" if (v is True or v == "u" or v == 1) else "d" for v in spec)
    if any(ch not in "ud" for ch in s):
        raise _err("E-TENSOR-001", f"variance {spec!r} must be a string of 'u' and 'd'")
    return s


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """Components of a tensor with a fixed variance per slot.

    Parameters
    ----------
    components : nested lists
        ``components[i][j]...`` (rank-many levels; a bare ``Expr`` for rank 0).
    variance : str
        One character per slot: ``"u"`` (contravariant) or ``"d"`` (covariant),
        e.g. ``"uddd"`` for ``R^a_{bcd}``.
    metric : Metric, optional
        Used to raise/lower slots when indexed in the other variance, and to
        simplify components.

    ``T[0, 1]`` (integers) is a component; ``T[-a, -b]`` (indices) is an
    :class:`IndexedTensor`.
    """

    def __init__(
        self, components: Any, variance: str | Sequence[Any], metric: Metric | None = None
    ):
        self.variance = _variance(variance)
        self.rank = len(self.variance)
        self.metric = metric
        self._data: dict[tuple[int, ...], Any] = {}
        if self.rank == 0:
            self.dim = metric.dim if metric is not None else 0
            if not _is_zero(components):
                self._data[()] = components
        else:
            self.dim = len(components)
            self._data = _from_nested(components, self.rank)
            for key in self._data:
                if len(key) != self.rank or any(k >= self.dim for k in key):
                    raise _err("E-TENSOR-001", "components are not an n x n x ... array")
        self._zero = 0

    @property
    def components(self) -> Any:
        """Nested lists of components (zeros as ``0``)."""
        if self.rank == 0:
            return self._data.get((), 0)
        return _to_nested(self._data, self.rank, self.dim, self._zero)

    def component(self, *idx: int) -> Any:
        """The component at integer indices ``idx``."""
        if len(idx) != self.rank:
            raise _err("E-TENSOR-001", f"expected {self.rank} integer indices, got {len(idx)}")
        return self._data.get(tuple(idx), 0)

    def nonzero(self) -> dict[tuple[int, ...], Any]:
        """``{index tuple: component}`` for every non-zero component."""
        return dict(self._data)

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.rank:
            raise _err(
                "E-TENSOR-009",
                f"rank-{self.rank} tensor indexed with {len(key)} slots",
                _INDEX_REMEDIATION,
            )
        if all(isinstance(k, int) for k in key):
            return self.component(*key)
        return IndexedTensor._from_tensor(self, key)

    def __repr__(self) -> str:
        return (
            f"Tensor(rank={self.rank}, variance={self.variance!r}, dim={self.dim}, "
            f"nonzero={len(self._data)})"
        )


# ---------------------------------------------------------------------------
# IndexedTensor: a tensor expression with named free indices
# ---------------------------------------------------------------------------


class IndexedTensor:
    """A tensor with named free indices, the result of indexing a :class:`Tensor`.

    Supports ``+``, ``-``, ``*`` (outer product with Einstein summation over a
    repeated index, once up once down), scalar ``*``/``/``,
    :meth:`symmetrize`, :meth:`antisymmetrize`, and :meth:`tensor` /
    :attr:`components` to read the result.  Components are kept in the
    metric's canonical form, so a combination that vanishes is ``0``.
    """

    def __init__(
        self,
        data: dict[tuple[int, ...], Any],
        free: Sequence[TensorIndex],
        dim: int,
        metric: Metric | None,
    ):
        self._data: dict[tuple[int, ...], Any] = data
        self.free = tuple(free)
        self.dim = dim
        self.metric = metric
        names = [i.name for i in self.free]
        if len(set(names)) != len(names):
            raise _err(
                "E-TENSOR-009", f"free index repeated: {list(self.free)}", _INDEX_REMEDIATION
            )

    # -- construction ---------------------------------------------------------

    @classmethod
    def _from_tensor(cls, t: Tensor, key: tuple[Any, ...]) -> IndexedTensor:
        data = dict(t._data)
        slots: list[TensorIndex] = []
        fixed: list[tuple[int, int]] = []
        for pos, k in enumerate(key):
            if isinstance(k, int):
                if not 0 <= k < t.dim:
                    raise _err(
                        "E-TENSOR-009", f"component index {k} out of range", _INDEX_REMEDIATION
                    )
                fixed.append((pos, k))
            elif isinstance(k, TensorIndex):
                slots.append(k)
            else:
                raise TypeError("index a Tensor with TensorIndex objects (+a / -a) or integers")
        # Raise/lower slots whose requested variance differs from the stored one.
        free_positions = [p for p, k in enumerate(key) if isinstance(k, TensorIndex)]
        for p in free_positions:
            want_up = key[p].up
            have_up = t.variance[p] == "u"
            if want_up != have_up:
                if t.metric is None:
                    raise _err(
                        "E-TENSOR-009",
                        f"slot {p} is {'up' if have_up else 'down'}; changing it needs a metric",
                        "build the tensor with metric=... or index it in its stored variance",
                    )
                mat = t.metric._ginv_data() if want_up else t.metric._g_data()
                data = _apply_matrix(data, p, mat, t.dim)
                data = t.metric._canon_dict(data)
        # Fix integer slots.
        if fixed:
            fixed_map = dict(fixed)
            new: dict[tuple[int, ...], Any] = {}
            for k, v in data.items():
                if all(k[p] == val for p, val in fixed_map.items()):
                    new[tuple(k[p] for p in free_positions)] = v
            data = new
        out = cls.__new__(cls)
        out._data = data
        out.dim = t.dim
        out.metric = t.metric
        out.free = tuple(slots)
        return out._contract_self()

    def _contract_self(self) -> IndexedTensor:
        """Sum over any index that occurs twice (once up, once down)."""
        while True:
            names = [i.name for i in self.free]
            dup = next((n for n in names if names.count(n) > 1), None)
            if dup is None:
                return self
            pos = [p for p, i in enumerate(self.free) if i.name == dup]
            if len(pos) != 2 or self.free[pos[0]].up == self.free[pos[1]].up:
                raise _err(
                    "E-TENSOR-009",
                    f"index {dup!r} must appear exactly twice, once up and once down",
                    _INDEX_REMEDIATION,
                )
            i, j = pos
            new: dict[tuple[int, ...], Any] = {}
            for k, v in self._data.items():
                if k[i] != k[j]:
                    continue
                nk = tuple(x for q, x in enumerate(k) if q not in (i, j))
                new[nk] = v if nk not in new else new[nk] + v
            self._data = self._canon(new)
            self.free = tuple(x for q, x in enumerate(self.free) if q not in (i, j))

    # -- helpers -------------------------------------------------------------

    def _canon(self, data: dict[tuple[int, ...], Any]) -> dict[tuple[int, ...], Any]:
        if self.metric is None:
            return {k: v for k, v in data.items() if not _is_zero(v)}
        return self.metric._canon_dict(data)

    def _merge_metric(self, other: IndexedTensor) -> Metric | None:
        if self.metric is not None and other.metric is not None and self.metric is not other.metric:
            raise _err(
                "E-TENSOR-009", "the operands belong to different metrics", _INDEX_REMEDIATION
            )
        if self.dim != other.dim:
            raise _err("E-TENSOR-009", f"dimension {self.dim} vs {other.dim}", _INDEX_REMEDIATION)
        return self.metric if self.metric is not None else other.metric

    def _aligned(self, other: IndexedTensor) -> dict[tuple[int, ...], Any]:
        """``other``'s data with its slots permuted into ``self.free`` order."""
        if sorted(map(repr, self.free)) != sorted(map(repr, other.free)):
            raise _err(
                "E-TENSOR-009",
                f"cannot add tensors with free indices {list(self.free)} and {list(other.free)}",
                _INDEX_REMEDIATION,
            )
        perm = [other.free.index(i) for i in self.free]
        return {tuple(k[p] for p in perm): v for k, v in other._data.items()}

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> IndexedTensor:
        if not isinstance(other, IndexedTensor):
            if self.free == () and _scalar_ok(other):
                other = IndexedTensor(
                    {} if _is_zero(other) else {(): other}, (), self.dim, self.metric
                )
            else:
                return NotImplemented
        metric = self._merge_metric(other)
        od = self._aligned(other)
        data = dict(self._data)
        for k, v in od.items():
            data[k] = data[k] + v if k in data else v
        out = IndexedTensor({}, self.free, self.dim, metric)
        out._data = out._canon(data)
        return out

    def __radd__(self, other: Any) -> IndexedTensor:
        return self.__add__(other)

    def __neg__(self) -> IndexedTensor:
        out = IndexedTensor({}, self.free, self.dim, self.metric)
        out._data = {k: -v for k, v in self._data.items()}
        return out

    def __sub__(self, other: Any) -> IndexedTensor:
        if isinstance(other, IndexedTensor):
            return self.__add__(-other)
        if self.free == () and _scalar_ok(other):
            return self.__add__(-other)
        return NotImplemented

    def __rsub__(self, other: Any) -> IndexedTensor:
        return (-self).__add__(other)

    def _scale(self, c: Any) -> IndexedTensor:
        if isinstance(c, Rational) and not isinstance(c, int):
            c = Fraction(c)
        out = IndexedTensor({}, self.free, self.dim, self.metric)
        out._data = out._canon({k: v * c for k, v in self._data.items()})
        return out

    def __mul__(self, other: Any) -> IndexedTensor:
        if _scalar_ok(other):
            return self._scale(other)
        if not isinstance(other, IndexedTensor):
            return NotImplemented
        metric = self._merge_metric(other)
        names_a = {i.name: p for p, i in enumerate(self.free)}
        pairs = []  # (pos in self, pos in other)
        for q, j in enumerate(other.free):
            if j.name in names_a:
                p = names_a[j.name]
                if self.free[p].up == j.up:
                    raise _err(
                        "E-TENSOR-009",
                        f"index {j.name!r} is {'up' if j.up else 'down'} in both factors",
                        _INDEX_REMEDIATION,
                    )
                pairs.append((p, q))
        ca = {p for p, _ in pairs}
        cb = {q for _, q in pairs}
        free = tuple(i for p, i in enumerate(self.free) if p not in ca) + tuple(
            j for q, j in enumerate(other.free) if q not in cb
        )
        # Group the second factor by its contracted index values.
        by_key: dict[tuple[int, ...], list[tuple[tuple[int, ...], Any]]] = {}
        for kb, vb in other._data.items():
            ck = tuple(kb[q] for _, q in pairs)
            rest = tuple(x for q, x in enumerate(kb) if q not in cb)
            by_key.setdefault(ck, []).append((rest, vb))
        data: dict[tuple[int, ...], Any] = {}
        for ka, va in self._data.items():
            ck = tuple(ka[p] for p, _ in pairs)
            rest_a = tuple(x for p, x in enumerate(ka) if p not in ca)
            for rest_b, vb in by_key.get(ck, ()):
                k = rest_a + rest_b
                term = va * vb
                data[k] = data[k] + term if k in data else term
        out = IndexedTensor({}, free, self.dim, metric)
        out._data = out._canon(data)
        return out._contract_self()

    def __rmul__(self, other: Any) -> IndexedTensor:
        if _scalar_ok(other):
            return self._scale(other)
        return NotImplemented

    def __truediv__(self, other: Any) -> IndexedTensor:
        if isinstance(other, (int, Rational)) and not isinstance(other, bool):
            if other == 0:
                raise ZeroDivisionError("division of a tensor by zero")
            return self._scale(Fraction(1) / Fraction(other))
        if isinstance(other, Expr):
            return self._scale(other**-1)
        return NotImplemented

    # -- symmetries ------------------------------------------------------------

    def _slots(self, idx: Sequence[Any]) -> list[int]:
        pos = []
        for i in idx:
            name = i.name if isinstance(i, TensorIndex) else str(i)
            hits = [p for p, j in enumerate(self.free) if j.name == name]
            if not hits:
                raise _err(
                    "E-TENSOR-009",
                    f"{name!r} is not a free index of {list(self.free)}",
                    _INDEX_REMEDIATION,
                )
            pos.append(hits[0])
        if len({self.free[p].up for p in pos}) > 1:
            raise _err(
                "E-TENSOR-009",
                "symmetrised indices must all be up or all be down",
                _INDEX_REMEDIATION,
            )
        return pos

    def _sym(self, idx: Sequence[Any], anti: bool) -> IndexedTensor:
        pos = self._slots(idx)
        data: dict[tuple[int, ...], Any] = {}
        perms = list(itertools.permutations(range(len(pos))))
        for perm in perms:
            sign = _perm_sign(perm) if anti else 1
            for k, v in self._data.items():
                nk = list(k)
                for a, b in zip(pos, perm):
                    nk[a] = k[pos[b]]
                nk = tuple(nk)
                term = v if sign == 1 else -v
                data[nk] = data[nk] + term if nk in data else term
        out = IndexedTensor({}, self.free, self.dim, self.metric)
        out._data = out._canon(
            {k: v * Fraction(1, math.factorial(len(pos))) for k, v in data.items()}
        )
        return out

    def symmetrize(self, *idx: Any) -> IndexedTensor:
        """``T_{(ab)} = ½(T_{ab} + T_{ba})`` over the named free indices."""
        return self._sym(idx, anti=False)

    def antisymmetrize(self, *idx: Any) -> IndexedTensor:
        """``T_{[ab]} = ½(T_{ab} − T_{ba})`` over the named free indices."""
        return self._sym(idx, anti=True)

    # -- reading ---------------------------------------------------------------

    def tensor(self) -> Tensor:
        """A :class:`Tensor` with slots in :attr:`free` order."""
        var = "".join("u" if i.up else "d" for i in self.free)
        t = Tensor.__new__(Tensor)
        t.variance = var
        t.rank = len(var)
        t.metric = self.metric
        t.dim = self.dim
        t._data = dict(self._data)
        t._zero = 0
        return t

    @property
    def components(self) -> Any:
        """Nested lists of components, slots in :attr:`free` order."""
        return self.tensor().components

    def component(self, *idx: int) -> Any:
        """The component at integer indices (slots in :attr:`free` order)."""
        return self.tensor().component(*idx)

    def is_zero(self) -> bool:
        """``True`` when every component is exactly zero (canonical form)."""
        return not self._data

    def __repr__(self) -> str:
        return f"IndexedTensor(free={list(self.free)}, dim={self.dim}, nonzero={len(self._data)})"


def _perm_sign(perm: Sequence[int]) -> int:
    sign = 1
    seen = [False] * len(perm)
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def _apply_matrix(
    data: dict[tuple[int, ...], Any], slot: int, mat: dict[tuple[int, int], Any], dim: int
) -> dict:
    """``new[..a..] = Σ_b mat[a][b] old[..b..]`` on one slot."""
    out: dict[tuple[int, ...], Any] = {}
    for k, v in data.items():
        b = k[slot]
        for a in range(dim):
            m = mat.get((a, b))
            if m is None:
                continue
            nk = (*k[:slot], a, *k[slot + 1 :])
            term = m * v
            out[nk] = out[nk] + term if nk in out else term
    return out


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class Metric:
    """A (pseudo-)Riemannian metric ``g_{ab}`` on the chart ``coords``.

    Parameters
    ----------
    g : list of lists or Matrix
        The symmetric components ``g[a][b]`` (``Expr``, ``int`` or ``Fraction``).
    coords : sequence of Expr
        Distinct coordinate symbols, one per row.

    Raises
    ------
    TensorError
        ``E-TENSOR-001`` (shape / coordinates), ``E-TENSOR-002`` (not symmetric),
        ``E-TENSOR-003`` (determinant identically zero), ``E-TENSOR-004`` (a float
        entry), ``E-TENSOR-005`` (``Piecewise`` or another node the exact field
        cannot hold).

    Every curvature method is computed once and cached; :meth:`timings` reports
    the wall time of each stage.
    """

    def __init__(self, g: Any, coords: Sequence[Expr]):
        if hasattr(g, "to_list"):
            g = g.to_list()
        rows = [list(r) for r in g]
        self.coords = tuple(coords)
        self._native = _TensorMetric(rows, list(self.coords))
        self.dim = self._native.dim()
        self._cache: dict[str, Any] = {}

    # -- internal ---------------------------------------------------------------

    def _cached(self, key: str, fn: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _g_data(self) -> dict[tuple[int, int], Any]:
        return self._cached("g_data", lambda: _from_nested(self._native.components(), 2))

    def _ginv_data(self) -> dict[tuple[int, int], Any]:
        return self._cached("ginv_data", lambda: _from_nested(self._native.inverse(), 2))

    def _canon_dict(self, data: dict[tuple[int, ...], Any]) -> dict[tuple[int, ...], Any]:
        keys = [k for k, v in data.items() if not _is_zero(v)]
        if not keys:
            return {}
        vals = [data[k] for k in keys]
        exprs = [v for v in vals if isinstance(v, Expr)]
        if exprs:
            canon = iter(_tensor_canonical(exprs, list(self.coords)))
            vals = [next(canon) if isinstance(v, Expr) else v for v in vals]
        return {k: v for k, v in zip(keys, vals) if not _is_zero(v)}

    # -- components ------------------------------------------------------------

    def as_tensor(self) -> Tensor:
        """``g_{ab}`` as a :class:`Tensor` (variance ``"dd"``)."""
        return self._cached("g", lambda: Tensor(self._native.components(), "dd", self))

    def inverse(self) -> Tensor:
        """``g^{ab}`` (variance ``"uu"``)."""
        return self._cached("ginv", lambda: Tensor(self._native.inverse(), "uu", self))

    def determinant(self) -> Expr:
        """``det g``."""
        return self._cached("det", self._native.determinant)

    def christoffel(self) -> list:
        """Christoffel symbols of the second kind: ``out[a][b][c] = Γ^a_{bc}``."""
        return self._cached("gamma2", self._native.christoffel)

    def christoffel_first_kind(self) -> list:
        """Christoffel symbols of the first kind: ``out[a][b][c] = Γ_{abc} = g_{ad}Γ^d_{bc}``."""
        return self._cached("gamma1", self._native.christoffel_first_kind)

    def riemann(self) -> Tensor:
        """``R^a_{bcd}`` (variance ``"uddd"``)."""
        return self._cached("riemann", lambda: Tensor(self._native.riemann(), "uddd", self))

    def riemann_lower(self) -> Tensor:
        """``R_{abcd}`` (variance ``"dddd"``)."""
        return self._cached(
            "riemann_lower", lambda: Tensor(self._native.riemann_lower(), "dddd", self)
        )

    def ricci(self) -> Tensor:
        """``R_{ab} = R^c_{acb}`` (variance ``"dd"``)."""
        return self._cached("ricci", lambda: Tensor(self._native.ricci(), "dd", self))

    def ricci_scalar(self) -> Expr:
        """``R = g^{ab} R_{ab}``."""
        return self._cached("scalar", self._native.ricci_scalar)

    def einstein(self) -> Tensor:
        """``G_{ab} = R_{ab} − ½ R g_{ab}`` (variance ``"dd"``)."""
        return self._cached("einstein", lambda: Tensor(self._native.einstein(), "dd", self))

    def weyl(self) -> Tensor:
        """The Weyl tensor ``C_{abcd}`` (variance ``"dddd"``); ``E-TENSOR-008`` for ``n < 3``."""
        return self._cached("weyl", lambda: Tensor(self._native.weyl(), "dddd", self))

    def kretschmann(self) -> Expr:
        """``K = R_{abcd} R^{abcd}``."""
        return self._cached("kretschmann", self._native.kretschmann)

    def is_ricci_flat(self) -> bool:
        """``True`` iff every ``R_{ab}`` is exactly zero — a proof, not a sample."""
        return self._native.is_ricci_flat()

    def geodesic_equations(self, velocities: Sequence[Any] | None = None) -> list:
        """Coordinate accelerations ``d²x^a/dτ² = −Γ^a_{bc} ẋ^b ẋ^c``.

        ``velocities`` defaults to fresh symbols ``<coord>_dot``; any expressions
        may be passed (e.g. ``ak.diff(x(τ), τ)``).  Returns one ``Expr`` per
        coordinate.
        """
        if velocities is None:
            pool = self.coords[0]._pool
            velocities = [pool.symbol(f"{c}_dot") for c in self.coords]
        return self._native.geodesic_accelerations(list(velocities))

    def tensor(self, components: Any, variance: str | Sequence[Any]) -> Tensor:
        """A :class:`Tensor` bound to this metric (for raising/lowering)."""
        return Tensor(components, variance, self)

    # -- the exact field ---------------------------------------------------------

    def simplify(self, expr: Any) -> Any:
        """Canonical form of ``expr`` in the exact component field (``0`` iff zero)."""
        if not isinstance(expr, Expr):
            return expr
        return _tensor_canonical([expr], list(self.coords))[0]

    def is_zero(self, expr: Any) -> bool:
        """Exact zero test in the component field."""
        return _is_zero(self.simplify(expr))

    def diff(self, expr: Any, coord: Any) -> Any:
        """``∂expr/∂coord`` in canonical form (``coord`` an index or a coordinate)."""
        if not isinstance(expr, Expr):
            return 0
        k = coord if isinstance(coord, int) else self.coords.index(coord)
        return _tensor_derivative([expr], list(self.coords), k)[0]

    def timings(self) -> dict[str, float]:
        """Wall time in seconds of every stage computed so far."""
        out: dict[str, float] = {}
        for name, t in self._native.timings():
            out[name] = out.get(name, 0.0) + t
        return out

    def __repr__(self) -> str:
        return f"Metric(dim={self.dim}, coords={list(self.coords)})"
