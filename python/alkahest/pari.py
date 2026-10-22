"""Optional PARI/GP interop: exact objects from PARI, **checked by Alkahest**.

Alkahest's number-field layer has arithmetic only: no ring of integers, class
group, units, prime-ideal factorisation or Galois group; it has no elliptic
curve invariants (conductor, ``a_p``, rank, ``L``-values) and no modular
forms.  PARI/GP has all of these.  This module asks PARI for them and
returns the answers as *exact Alkahest objects* (Python ``int`` /
``Fraction``, :class:`alkahest.experimental.NumberFieldElement`,
:class:`alkahest.ArbBall`), then **re-checks every cheap consequence of the
answer with Alkahest's own exact arithmetic** and records what was and was
not checked in a ``verification`` mapping.  A PARI answer that contradicts a
check raises :class:`PariError` (``E-PARI-003``) instead of being returned.

Licence boundary
----------------
PARI/GP is GPL-licensed.  Alkahest is not, and never links, bundles, imports
at ``import alkahest`` time or redistributes PARI.  This module is **pure
Python**, loads a PARI binding lazily on first use, and works only if *you*
have installed one yourself:

* ``cypari2`` (needs a system libpari; ``pip install cypari2`` builds against
  it), or
* ``cypari`` (the SnapPy project's binding; its wheels carry their own
  libpari — ``pip install cypari``).

Nothing in Alkahest's wheels, Rust crates or default dependencies depends on
either.  Without a binding, :func:`available` is ``False`` and every
function raises :class:`PariUnavailable` (``E-PARI-001``).  A ``gp``
subprocess is *not* supported.  If you redistribute an application that
installs a PARI binding next to Alkahest, the GPL applies to *that*
combination; Alkahest itself is unaffected.

Trust levels
------------
Each result states how far it is trusted, in plain strings:

``certified``
    PARI's answer carries a proof (``bnfcertify``) or Alkahest verified it
    outright.
``grh_conditional``
    PARI's class group / units are correct only under the Generalised Riemann
    Hypothesis (``bnfinit`` without ``bnfcertify``).
``proven`` / ``numerical`` / ``bounds_only``
    For elliptic-curve ranks and ``L``-values; see
    :func:`elliptic_curve_invariants`.

Error codes
-----------
``E-PARI-001`` no PARI binding installed; ``E-PARI-002`` unsupported input;
``E-PARI-003`` a PARI answer failed Alkahest's verification; ``E-PARI-004``
PARI itself raised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EigenformOrbit",
    "EllipticCurveData",
    "ModularFormsData",
    "NumberFieldData",
    "PariError",
    "PariUnavailable",
    "PrimeIdeal",
    "available",
    "backend",
    "elliptic_curve_invariants",
    "modular_forms",
    "number_field_invariants",
    "pari_version",
    "ramanujan_tau",
]


# ---------------------------------------------------------------------------
# errors and backend detection
# ---------------------------------------------------------------------------


class PariError(RuntimeError):
    """Base error of :mod:`alkahest.pari`; ``.code`` is the stable ``E-PARI-NNN``."""

    code = "E-PARI-004"

    def __init__(self, message: str, code: str | None = None):
        if code is not None:
            self.code = code
        super().__init__(f"[{self.code}] {message}")


class PariUnavailable(PariError, ImportError):
    """No PARI binding (``cypari2`` / ``cypari``) is installed (``E-PARI-001``)."""

    code = "E-PARI-001"


def _unsupported(msg: str) -> PariError:
    return PariError(msg, "E-PARI-002")


def _mismatch(msg: str) -> PariError:
    return PariError("PARI result failed verification: " + msg, "E-PARI-003")


@lru_cache(maxsize=1)
def _load() -> tuple[str, Any] | None:
    try:  # pragma: no cover - depends on the environment
        import cypari2  # type: ignore

        return "cypari2", cypari2.Pari()
    except Exception:
        pass
    try:  # pragma: no cover - depends on the environment
        import cypari  # type: ignore

        return "cypari", cypari.pari
    except Exception:
        return None


def available() -> bool:
    """``True`` when a PARI binding (``cypari2`` or ``cypari``) can be loaded."""
    return _load() is not None


def backend() -> str | None:
    """Name of the binding in use (``"cypari2"`` / ``"cypari"``) or ``None``."""
    b = _load()
    return None if b is None else b[0]


def _pari() -> Any:
    b = _load()
    if b is None:
        raise PariUnavailable(
            "no PARI binding found; install `cypari` or `cypari2` separately "
            "(PARI is GPL and is never bundled with Alkahest)"
        )
    return b[1]


def pari_version() -> str | None:
    """The PARI library version string, or ``None`` when unavailable."""
    return str(_pari().version()) if available() else None


def _gp(expr: str, *args: Any) -> Any:
    """Evaluate a GP expression, optionally calling it as a closure on ``args``."""
    p = _pari()
    try:
        f = p(expr)
        return f(*args) if args else f
    except PariError:
        raise
    except Exception as exc:
        raise PariError(f"PARI failed on `{expr}`: {exc}") from exc


# ---------------------------------------------------------------------------
# PARI -> Python conversion (exact only)
# ---------------------------------------------------------------------------


def _ptype(g: Any) -> str:
    return str(g.type())


def _py(g: Any) -> Any:
    """Convert a PARI object to ints/Fractions/lists/strings.  Never floats."""
    t = _ptype(g)
    if t == "t_INT":
        return int(str(g))
    if t == "t_FRAC":
        return Fraction(str(g))
    if t in ("t_VEC", "t_COL"):
        return [_py(g[i]) for i in range(len(g))]
    if t == "t_POL":
        return _poly_coeffs(g)
    if t == "t_POLMOD":
        return _poly_coeffs(g.lift())
    if t == "t_STR":
        return str(g)
    if t == "t_MAT":
        rows = int(str(_gp("matsize")(g)[0]))
        return [_py(_gp("(M,i)->M[i,]")(g, i + 1)) for i in range(rows)]
    return str(g)


def _poly_coeffs(g: Any) -> list[Any]:
    """Ascending coefficient list of a PARI polynomial (or scalar)."""
    t = _ptype(g)
    if t in ("t_INT", "t_FRAC"):
        return [_py(g)]
    if t == "t_POLMOD":
        return _poly_coeffs(g.lift())
    if t != "t_POL":
        raise PariError(f"expected a polynomial, got {t}")
    v = g.Vecrev()
    return [_py(v[i]) for i in range(len(v))]


def _poly_str(coeffs: Sequence[int], var: str = "x") -> str:
    terms = [f"({int(c)})*{var}^{i}" for i, c in enumerate(coeffs) if c != 0]
    return "+".join(terms) if terms else "0"


def _rat(x: Any) -> Fraction:
    return Fraction(x)


def _rat_str(x: Fraction) -> str:
    return f"({x.numerator})" if x.denominator == 1 else f"({x.numerator}/{x.denominator})"


def _squarefree_part_is_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


# ---------------------------------------------------------------------------
# number fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimeIdeal:
    """A prime ideal ``P = (p, alpha)`` above a rational prime ``p``.

    ``e`` is the ramification index, ``f`` the residue degree, ``norm = p**f``
    and ``generators = (p, alpha)`` a two-element presentation with ``alpha`` a
    :class:`NumberFieldElement` of the field passed in.
    """

    p: int
    e: int
    f: int
    norm: int
    generators: tuple[Any, Any]


@dataclass(frozen=True)
class NumberFieldData:
    """What :func:`number_field_invariants` returns.

    Attributes
    ----------
    field
        The :class:`NumberField` the data is expressed in.
    discriminant
        ``d_K``, an ``int``.
    signature
        ``(r1, r2)``.
    integral_basis
        A ``Z``-basis of the ring of integers as ``NumberFieldElement`` s.
    class_number, class_group
        ``h`` and the cyclic invariants ``[d1, d2, ...]`` (``d1 | d2 | ...``)
        or ``None`` when ``class_group=False``.
    torsion_units
        ``(w, zeta)``: the order of the roots of unity in ``K`` and a generator.
    fundamental_units
        ``r1 + r2 - 1`` multiplicatively independent units (``NumberFieldElement``).
    regulator
        PARI's decimal string for the regulator (not an exact object).
    primes
        ``{p: [PrimeIdeal, ...]}`` for the primes requested.
    galois
        ``{"order", "parity", "name", "transitive_index"}`` or ``None``.
    certification
        ``"certified"`` (``bnfcertify`` succeeded: ``h``, units, regulator are
        proven), ``"grh_conditional"`` (not run, too expensive, or failed), or
        ``"not_computed"`` when ``class_group=False``.
    verification
        ``{check name: True}`` for every check Alkahest ran, and a
        ``"not_checked"`` list naming what rests on PARI alone.
    """

    field: Any
    discriminant: int
    signature: tuple[int, int]
    integral_basis: list[Any]
    class_number: int | None
    class_group: list[int] | None
    torsion_units: tuple[int, Any] | None
    fundamental_units: list[Any] | None
    regulator: str | None
    primes: dict[int, list[PrimeIdeal]]
    galois: dict[str, Any] | None
    certification: str
    verification: dict[str, Any] = field(default_factory=dict)


def _nf_class():
    from .experimental import NumberField

    return NumberField


def _elt(K: Any, g: Any) -> Any:
    """A PARI element (polmod / poly / scalar in ``x`` or ``y``) as an element of ``K``."""
    c = _poly_coeffs(g)
    if len(c) > K.degree:
        raise PariError("element has more coordinates than the field degree")
    return K.element(c)


def _det(rows: list[list[Fraction]]) -> Fraction:
    m = [r[:] for r in rows]
    n = len(m)
    det = Fraction(1)
    for i in range(n):
        piv = next((r for r in range(i, n) if m[r][i] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != i:
            m[i], m[piv] = m[piv], m[i]
            det = -det
        det *= m[i][i]
        for r in range(i + 1, n):
            f = m[r][i] / m[i][i]
            if f:
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    return det


def _inverse(rows: list[list[Fraction]]) -> list[list[Fraction]]:
    n = len(rows)
    m = [r[:] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(rows)]
    for i in range(n):
        piv = next((r for r in range(i, n) if m[r][i] != 0), None)
        if piv is None:
            raise _mismatch("integral basis is not linearly independent")
        m[i], m[piv] = m[piv], m[i]
        pv = m[i][i]
        m[i] = [a / pv for a in m[i]]
        for r in range(n):
            if r != i and m[r][i] != 0:
                f = m[r][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    return [r[n:] for r in m]


def _lattice_index(vectors: list[list[int]], n: int) -> int:
    """Index in ``Z^n`` of the lattice spanned by integer ``vectors`` (0 if rank < n)."""
    rows = [v[:] for v in vectors if any(v)]
    det = 1
    for col in range(n):
        while True:
            nz = [r for r in rows if r[col] != 0]
            if not nz:
                return 0
            piv = min(nz, key=lambda r: abs(r[col]))
            done = True
            for r in nz:
                if r is not piv:
                    q = r[col] // piv[col]
                    for j in range(n):
                        r[j] -= q * piv[j]
                    if r[col] != 0:
                        done = False
            if done:
                break
        det *= abs(piv[col])
        rows.remove(piv)
    return det


def _check_divisor_chain(cyc: Sequence[int]) -> None:
    """Reject a class-group answer whose invariants are not a divisor chain.

    PARI lists the invariants in *descending* order (``Q(sqrt(-65))`` has
    ``[4, 2]``), so each invariant must be divisible by the next one.  Raises
    ``E-PARI-003`` (via :func:`_mismatch`) otherwise.
    """
    for a_, b_ in zip(cyc, cyc[1:]):
        if a_ % b_:
            raise _mismatch("class group invariants do not form a divisor chain")


def _imag_quadratic_class_number(D: int) -> int:
    """Number of reduced primitive forms of discriminant ``D < 0`` (exact)."""
    h = 0
    a = 1
    while 3 * a * a <= -D:
        for b in range(-a + 1, a + 1):
            if (b * b - D) % (4 * a):
                continue
            c = (b * b - D) // (4 * a)
            if c < a or (c == a and b < 0):
                continue
            if math.gcd(math.gcd(a, abs(b)), c) == 1:
                h += 1
        a += 1
    return h


def number_field_invariants(
    field_or_poly: Any,
    *,
    primes: Sequence[int] = (),
    class_group: bool = True,
    galois: bool = True,
    certify: bool | str = "auto",
) -> NumberFieldData:
    """Ring of integers, class group, units, prime splitting and Galois group of a number field.

    Parameters
    ----------
    field_or_poly
        An :class:`alkahest.experimental.NumberField`, or an ascending integer
        coefficient list of a **monic** irreducible polynomial.
    primes
        Rational primes whose prime-ideal factorisation to return.
    class_group
        Run ``bnfinit`` (class group, units, regulator).  Disable for large
        discriminants.
    galois
        Compute the Galois group with ``polgalois`` (degree <= 7 in a stock
        PARI; refused with ``E-PARI-002`` above that unless the ``galdata``
        package is installed, in which case PARI answers up to degree 11).
    certify
        ``True`` always runs ``bnfcertify`` (can be very slow), ``False``
        never, ``"auto"`` (default) runs it when degree <= 6 and
        ``|d_K| < 10**15``.  The outcome decides ``certification``.

    Verification
    ------------
    All of the following are recomputed in Alkahest's exact arithmetic and a
    failure raises ``E-PARI-003``: the integral basis spans an order of
    discriminant ``d_K`` (Gram determinant of ``Tr(w_i w_j)``), is closed under
    multiplication, consists of algebraic integers, ``disc(f) = index^2 d_K``,
    ``d_K`` has the right sign and residue mod 4; every unit has norm ``±1``
    and the torsion generator has the stated order; each prime ideal's
    two-element presentation spans a lattice of index exactly ``p**f``,
    ``sum e f = n``, and ``p | d_K`` exactly when some ``e > 1``; the Galois
    parity matches whether ``disc(f)`` is a square; for imaginary quadratic
    fields ``h`` is recounted from reduced binary quadratic forms.

    Maximality of the order, fundamentality of the units and the value of
    the class group are *not* re-derived (they are listed under
    ``verification["not_checked"]``); they hold unconditionally only when
    ``certification == "certified"``.

    Raises
    ------
    PariUnavailable
        No PARI binding (``E-PARI-001``).
    PariError
        ``E-PARI-002`` for non-monic or reducible input, ``E-PARI-003`` for a
        failed check, ``E-PARI-004`` when PARI itself fails.

    Example
    -------
    >>> from alkahest import pari
    >>> d = pari.number_field_invariants([5, 0, 1])      # Q(sqrt(-5))   # doctest: +SKIP
    >>> d.discriminant, d.class_number                                   # doctest: +SKIP
    (-20, 2)
    """
    NumberField = _nf_class()
    if isinstance(field_or_poly, NumberField):
        K = field_or_poly
    else:
        K = NumberField(list(field_or_poly))
    coeffs = [int(c) for c in K.defining_polynomial]
    if coeffs[-1] != 1:
        raise _unsupported(
            "defining polynomial must be monic; substitute x -> x/lc and rescale first"
        )
    n = len(coeffs) - 1
    P = _poly_str(coeffs)
    ver: dict[str, Any] = {}
    not_checked = ["maximality of the order (PARI nfbasis is unconditional)"]

    gp_p = _gp(P)
    use_bnf = class_group
    if use_bnf:
        bnf = _gp("bnfinit", gp_p, 1)
        nf = _gp("(b)->b.nf", bnf)
    else:
        bnf = None
        nf = _gp("nfinit", gp_p)

    dK = _py(_gp("(n)->n.disc", nf))
    r1, r2 = _py(_gp("(n)->n.sign", nf))
    index = _py(_gp("(n)->n.index", nf))
    zk = _gp("(n)->n.zk", nf)
    basis = [_elt(K, zk[i]) for i in range(n)]

    # --- integral basis checks (all exact, in alkahest) -------------------
    if r1 + 2 * r2 != n:
        raise _mismatch("signature does not sum to the degree")
    if (dK < 0) != (r2 % 2 == 1):
        raise _mismatch("sign of d_K disagrees with the number of complex places")
    if dK % 4 not in (0, 1):
        raise _mismatch("d_K violates Stickelberger's congruence")
    gram = [[(basis[i] * basis[j]).trace() for j in range(n)] for i in range(n)]
    if _det([[Fraction(v) for v in row] for row in gram]) != dK:
        raise _mismatch("Gram determinant of the integral basis is not d_K")
    for w in basis:
        if any(Fraction(c).denominator != 1 for c in w.minimal_polynomial()):
            raise _mismatch("an integral-basis element is not an algebraic integer")
    M = [[Fraction(c) for c in w.coefficients()] for w in basis]
    Minv = _inverse(M)
    for i in range(n):
        for j in range(i, n):
            c = (basis[i] * basis[j]).coefficients()
            co = [sum(Fraction(c[k]) * Minv[k][m] for k in range(n)) for m in range(n)]
            if any(v.denominator != 1 for v in co):
                raise _mismatch("integral basis is not closed under multiplication")
    if K.polynomial_discriminant != index * index * dK:
        raise _mismatch("disc(f) != index^2 * d_K")
    if abs(_det(M)) != Fraction(1, index):
        raise _mismatch("index of Z[a] in the basis lattice is not nf.index")
    ver["integral_basis"] = True
    ver["discriminant"] = True

    def coords(e: Any) -> list[int]:
        c = e.coefficients()
        co = [sum(Fraction(c[k]) * Minv[k][m] for k in range(n)) for m in range(n)]
        if any(v.denominator != 1 for v in co):
            raise _mismatch("element not in the integral lattice")
        return [int(v) for v in co]

    # --- class group, units ------------------------------------------------
    h = cyc = tu = fus = reg = None
    certification = "not_computed"
    if bnf is not None:
        cyc = [int(v) for v in _py(_gp("(b)->b.cyc", bnf))]
        h = int(_py(_gp("(b)->b.no", bnf)))
        if math.prod(cyc) != h:
            raise _mismatch("class number is not the product of the invariants")
        _check_divisor_chain(cyc)
        reg = str(_gp("(b)->b.reg", bnf))
        fu_g = _gp("(b)->b.fu", bnf)
        fus = [_elt(K, fu_g[i]) for i in range(len(fu_g))]
        if len(fus) != r1 + r2 - 1:
            raise _mismatch("wrong number of fundamental units")
        for u in fus:
            if abs(u.norm()) != 1:
                raise _mismatch("a fundamental unit does not have norm ±1")
        w_g = _gp("(b)->b.tu", bnf)
        w = int(_py(w_g[0]))
        zeta = _elt(K, w_g[1])
        if not (zeta**w).is_one() or abs(zeta.norm()) != 1:
            raise _mismatch("torsion unit generator fails zeta^w = 1")
        if w % 2:
            raise _mismatch("number of roots of unity must be even")
        tu = (w, zeta)
        ver["unit_norms"] = True
        ver["torsion_unit_order_divides"] = True
        if n == 2 and dK < 0:
            if _imag_quadratic_class_number(dK) != h:
                raise _mismatch("class number differs from the reduced-form count")
            ver["class_number_by_reduced_forms"] = True
        else:
            not_checked.append("class group structure")
        not_checked.append("fundamentality and independence of units")

        do_cert = certify is True or (certify == "auto" and n <= 6 and abs(dK) < 10**15)
        certification = "grh_conditional"
        if do_cert:
            try:
                ok = int(str(_gp("bnfcertify", bnf)))
                certification = "certified" if ok == 1 else "grh_conditional"
            except PariError:
                certification = "grh_conditional"
    ver["certification"] = certification

    # --- prime ideals ---------------------------------------------------
    pdata: dict[int, list[PrimeIdeal]] = {}
    for p in primes:
        p = int(p)
        dec = _gp("idealprimedec", nf, p)
        out: list[PrimeIdeal] = []
        total = 0
        for i in range(len(dec)):
            pr = dec[i]
            e = int(_py(pr[2]))
            f = int(_py(pr[3]))
            col = _py(pr[1])
            alpha = K.zero()
            for c, wb in zip(col, basis):
                alpha = alpha + wb * K.element([Fraction(c)])
            # lattice p*O + alpha*O must have index p^f in O
            vecs = [[p if i == j else 0 for j in range(n)] for i in range(n)]
            vecs += [coords(alpha * wb) for wb in basis]
            if _lattice_index(vecs, n) != p**f:
                raise _mismatch(f"two-element presentation of a prime above {p} has wrong norm")
            total += e * f
            out.append(PrimeIdeal(p, e, f, p**f, (K.element([p]), alpha)))
        if total != n:
            raise _mismatch(f"sum e*f = {total} != {n} above p = {p}")
        if (dK % p == 0) != any(P_.e > 1 for P_ in out):
            raise _mismatch(f"ramification above {p} disagrees with d_K")
        pdata[p] = out
    if pdata:
        ver["prime_ideals"] = True

    # --- Galois group ---------------------------------------------------
    gal = None
    if galois:
        try:
            g = _gp("polgalois", gp_p)
        except PariError as exc:
            raise _unsupported(
                f"polgalois unavailable for degree {n} (install PARI's galdata): {exc}"
            ) from exc
        order, sign, tindex, name = int(_py(g[0])), int(_py(g[1])), int(_py(g[2])), str(g[3])
        even = _squarefree_part_is_square(K.polynomial_discriminant)
        if (sign == 1) != even:
            raise _mismatch("Galois-group parity disagrees with disc(f) being a square")
        if order % n:
            raise _mismatch("Galois group order is not divisible by the degree")
        gal = {
            "order": order,
            "parity": "even" if sign == 1 else "odd",
            "transitive_index": tindex,
            "name": name.strip('"'),
        }
        ver["galois_parity"] = True

    ver["not_checked"] = not_checked
    return NumberFieldData(
        field=K,
        discriminant=dK,
        signature=(r1, r2),
        integral_basis=basis,
        class_number=h,
        class_group=cyc,
        torsion_units=tu,
        fundamental_units=fus,
        regulator=reg,
        primes=pdata,
        galois=gal,
        certification=certification,
        verification=ver,
    )


# ---------------------------------------------------------------------------
# elliptic curves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EllipticCurveData:
    """What :func:`elliptic_curve_invariants` returns.

    ``ainvs``
        ``(a1, a2, a3, a4, a6)`` as ``Fraction`` s.
    ``discriminant``, ``conductor``
        ``int`` s (``discriminant`` is of the given model).
    ``ap``
        ``{p: a_p}`` for primes ``p <= ap_bound`` (``a_p`` for bad ``p`` is the
        PARI convention ``0``/``±1``).
    ``torsion``
        ``{"order", "structure", "generators"}``; generators are exact
        ``(x, y)`` ``Fraction`` pairs on the given model.
    ``root_number``
        ``±1``.
    ``analytic_rank`` / ``analytic_rank_status``
        PARI's numerical analytic rank; status ``"proven"`` when Alkahest's own
        series evaluation shows the leading ``L``-value is non-zero for rank 0
        or 1, otherwise ``"numerical"``.
    ``l_value``
        ``L(E, 1)`` (rank 0) or ``L'(E, 1)`` (rank 1; ``L(E,1) = 0`` exactly
        by the functional equation) as an :class:`alkahest.ArbBall`
        computed from ``a_n`` with a rigorous truncation bound, or ``None``
        when the analytic rank is >= 2 (value not enclosed).
    ``mw_rank_bounds`` / ``mw_rank_status``
        ``(lower, upper)`` from ``ellrank`` (2-descent); status ``"proven"``
        when they agree, else ``"bounds_only"`` (the true rank may be
        anywhere in the range; only conditional statements such as BSD
        give more).
    ``mw_points``
        Independent points of infinite order found by ``ellrank``.  They
        need *not* generate the Mordell–Weil group (no saturation).
    """

    ainvs: tuple[Fraction, ...]
    discriminant: int
    conductor: int
    ap: dict[int, int]
    torsion: dict[str, Any]
    root_number: int
    analytic_rank: int | None
    analytic_rank_status: str
    l_value: Any
    mw_rank_bounds: tuple[int, int] | None
    mw_rank_status: str
    mw_points: list[tuple[Fraction, Fraction]]
    verification: dict[str, Any] = field(default_factory=dict)


def _primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(n + 1) if sieve[i]]


def _b_invariants(a1, a2, a3, a4, a6):
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    c6 = -(b2**3) + 36 * b2 * b4 - 216 * b6
    disc = -b2 * b2 * b8 - 8 * b4**3 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    return b2, b4, b6, b8, c4, c6, disc


def _count_points(a, p: int) -> int:
    """``#E(F_p)`` for the (reduced) integral model, by brute force (p small)."""
    a1, a2, a3, a4, a6 = (
        int(Fraction(v).numerator * pow(Fraction(v).denominator, -1, p)) % p for v in a
    )
    cnt = 1
    for x in range(p):
        rhs = (x**3 + a2 * x * x + a4 * x + a6) % p
        # solutions y of y^2 + (a1 x + a3) y = rhs
        for y in range(p):
            if (y * y + (a1 * x + a3) * y - rhs) % p == 0:
                cnt += 1
    return cnt


def _count_points_short(c4: int, c6: int, p: int) -> int:
    """``#E(F_p)`` for ``p >= 5`` via ``y^2 = x^3 - 27 c4 x - 54 c6`` (O(p))."""
    sq = [0] * p
    for y in range(p):
        sq[y * y % p] += 1
    A, B = (-27 * c4) % p, (-54 * c6) % p
    return 1 + sum(sq[(x**3 + A * x + B) % p] for x in range(p))


def _short_model(ainvs, pt):
    """Map a point to ``y'^2 = x'^3 - 27 c4 x' - 54 c6`` (an isomorphism over Q)."""
    a1, _a2, a3, _a4, _a6 = ainvs
    b2, _, _, _, c4, c6, _ = _b_invariants(*ainvs)
    x, y = pt
    return 36 * x + 3 * b2, 108 * (2 * y + a1 * x + a3), -27 * c4, -54 * c6


def _on_curve(ainvs, pt) -> bool:
    a1, a2, a3, a4, a6 = ainvs
    x, y = pt
    return y * y + a1 * x * y + a3 * y == x**3 + a2 * x * x + a4 * x + a6


def _pic0_order(ainvs, pt) -> int | None:
    """Order of ``pt`` via Alkahest's ``Pic^0``; ``None`` for infinite order."""
    from . import experimental as ex

    xp, yp, A, B = _short_model(ainvs, pt)
    fld = ex.FunctionField.hyperelliptic([B, A, 0, 1])
    d = ex.Divisor(fld, [(ex.Place.finite(xp, yp), 1), (ex.Place.infinity(), -1)])
    cls = ex.DivisorClass.of(d)
    try:
        return int(cls.order())
    except Exception as exc:  # FunctionFieldError E-FFLD-007: infinite order
        if getattr(exc, "code", None) == "E-FFLD-007":
            return None
        raise


def _l_series_ball(an: list[int], N: int, root: int):
    """Rigorous enclosure of ``L(E,1)`` (root=+1) or ``L'(E,1)`` (root=-1).

    Uses ``L(E,1) = 2 sum a_n/n exp(-2 pi n/sqrt N)`` and
    ``L'(E,1) = 2 sum a_n/n E_1(2 pi n/sqrt N)``; ``|a_n| <= n`` bounds the
    tail by a geometric series.  Returns ``(mid_mpf, radius_mpf)``.
    """
    import mpmath as mp

    with mp.workdps(60):
        c = 2 * mp.pi / mp.sqrt(N)
        M = len(an) - 1
        s = mp.mpf(0)
        for k in range(1, M + 1):
            t = mp.mpf(an[k]) / k
            if root == 1:
                s += t * mp.exp(-c * k)
            else:
                s += t * mp.e1(c * k)
        tail = 2 * mp.exp(-c * (M + 1)) / (1 - mp.exp(-c))
        return 2 * s, tail + mp.mpf(10) ** -45


def _to_ball(mid, rad):
    import mpmath as mp

    from . import ArbBall

    m = float(mid)
    r = float(rad + abs(mid - mp.mpf(m))) * (1 + 1e-12) + 1e-300
    return ArbBall(m, r)


def elliptic_curve_invariants(
    ainvs: Sequence[Any],
    *,
    ap_bound: int = 100,
    rank: bool = True,
    l_value: bool = True,
) -> EllipticCurveData:
    """Conductor, ``a_p``, torsion, rank and ``L``-value of an elliptic curve over Q.

    The curve is ``y^2 + a1 xy + a3 y = x^3 + a2 x^2 + a4 x + a6``.

    Parameters
    ----------
    ainvs
        ``[a1, a2, a3, a4, a6]`` or ``[a4, a6]`` (integers or ``Fraction``s;
        the curve must be non-singular).  Non-integral coefficients are
        refused with ``E-PARI-002``.
    ap_bound
        Return ``a_p`` for all primes ``p <= ap_bound``.
    rank
        Run ``ellrank`` (2-descent) and ``ellanalyticrank``.
    l_value
        Evaluate the leading ``L``-value as a ball.

    Verification
    ------------
    ``discriminant`` is recomputed from the ``b``-invariants; the conductor's
    primes must divide it; ``a_p`` is recounted by brute-force point counting
    for good primes ``p <= 1000`` and checked against Hasse; each torsion
    generator is checked to lie on the curve and its **order is recomputed in
    Alkahest's own Pic^0** (``FunctionField`` / ``DivisorClass``) on the
    isomorphic short model — it must equal the order PARI claims (this
    certifies the listed points, not that no further torsion exists); the
    analytic rank must match the parity of the root number; ``L(E,1)`` (or
    ``L'(E,1)``) is *recomputed* from the ``a_n`` with a rigorous truncation
    bound and compared with PARI's ``lfun``; each Mordell–Weil point lies on
    the curve and is shown to have infinite order by ``Pic^0``.

    Trust
    -----
    ``analytic_rank_status`` / ``mw_rank_status`` as described on
    :class:`EllipticCurveData`; modularity (the theorem that ``L(E,s)`` has
    the functional equation used above) is assumed.

    Example
    -------
    >>> from alkahest import pari
    >>> e = pari.elliptic_curve_invariants([0, -1, 1, -10, -20])   # 11a1  # doctest: +SKIP
    >>> e.conductor, e.torsion["order"], e.analytic_rank           # doctest: +SKIP
    (11, 5, 0)
    """
    a = [Fraction(v) for v in ainvs]
    if len(a) == 2:
        a = [Fraction(0), Fraction(0), Fraction(0), a[0], a[1]]
    if len(a) != 5:
        raise _unsupported("ainvs must have 2 or 5 entries")
    if any(v.denominator != 1 for v in a):
        raise _unsupported("non-integral a-invariants are not supported")
    a = [Fraction(int(v)) for v in a]
    *_, c4, c6, disc = _b_invariants(*a)
    if disc == 0:
        raise _unsupported("singular curve (discriminant 0)")
    ver: dict[str, Any] = {}
    not_checked: list[str] = []

    E = _gp("ellinit", _gp("[" + ",".join(str(int(v)) for v in a) + "]"))
    if len(E) == 0:
        raise _unsupported("PARI rejected the curve")
    if int(_py(_gp("(e)->e.disc", E))) != int(disc):
        raise _mismatch("discriminant differs from the b-invariant formula")
    ver["discriminant"] = True
    N = int(_py(_gp("(e)->ellglobalred(e)[1]", E)))
    # primes of the conductor must divide the discriminant
    nn = N
    for p in _primes_upto(math.isqrt(N) + 1):
        if nn % p == 0:
            if int(disc) % p:
                raise _mismatch(f"conductor divisible by {p} which does not divide Delta")
            while nn % p == 0:
                nn //= p
    if nn > 1 and int(disc) % nn:
        raise _mismatch("conductor has a prime factor not dividing Delta")
    ver["conductor_primes_divide_discriminant"] = True

    # a_p
    ap: dict[int, int] = {}
    for p in _primes_upto(ap_bound):
        v = int(_py(_gp("ellap", E, p)))
        if v * v > 4 * p and N % p:
            raise _mismatch(f"a_{p} = {v} violates the Hasse bound")
        ap[p] = v
        if N % p and p <= 1000:
            cnt = _count_points(a, p) if p < 5 else _count_points_short(int(c4), int(c6), p)
            if cnt != p + 1 - v:
                raise _mismatch(f"a_{p} = {v} disagrees with a direct point count")
    ver["ap_point_count_up_to"] = min(ap_bound, 1000)
    not_checked.append("a_p for bad primes and p > 1000")

    # torsion
    t = _gp("elltors", E)
    t_order = int(_py(t[0]))
    t_struct = [int(v) for v in _py(t[1])]
    gens = [tuple(Fraction(c) for c in _py(t[2][i])) for i in range(len(t[2]))]
    if t_order not in (*range(1, 11), 12, 16):
        raise _mismatch("torsion order violates Mazur's theorem")
    if math.prod(t_struct) != t_order:
        raise _mismatch("torsion structure/order mismatch")
    for g_, n_ in zip(gens, t_struct):
        if not _on_curve(a, g_):
            raise _mismatch("torsion generator is not on the curve")
        o = _pic0_order(a, g_)
        if o != n_:
            raise _mismatch(f"torsion generator order {o} (Pic^0) != {n_} (PARI)")
    ver["torsion_generators_pic0"] = True
    not_checked.append("completeness of the torsion subgroup")

    root = int(_py(_gp("ellrootno", E)))
    a_rank = None
    a_status = "not_computed"
    lval = None
    mw_bounds = None
    mw_status = "not_computed"
    mw_pts: list[tuple[Fraction, Fraction]] = []

    if rank:
        ar = _gp("ellanalyticrank", E)
        a_rank = int(_py(ar[0]))
        if a_rank % 2 != (0 if root == 1 else 1):
            raise _mismatch("analytic rank parity disagrees with the root number")
        ver["rank_parity_root_number"] = True
        a_status = "numerical"
        if l_value and a_rank <= 1:
            import mpmath as mp

            c = 2 * math.pi / math.sqrt(N)
            M = min(max(60, int(70 / c) + 1), 200_000)
            an = [0] + [int(v) for v in _py(_gp("ellan", E, M))]
            mid, rad = _l_series_ball(an, N, root)
            ball = _to_ball(mid, rad)
            with mp.workdps(30):
                ref = mp.mpf(str(_gp("(e,d)->lfun(lfuncreate(e),1,d)", E, a_rank)))
            if abs(ref - mid) > mp.mpf(10) ** -10 * max(1, abs(mid)):
                raise _mismatch("L-value series disagrees with PARI lfun")
            ver["l_value_series_vs_lfun"] = True
            lval = ball
            if float(rad) < abs(float(mid)):
                a_status = "proven"
        elif l_value:
            not_checked.append("L-value for analytic rank >= 2")

        rk = _gp("ellrank", E)
        lo, hi = int(_py(rk[0])), int(_py(rk[1]))
        mw_bounds = (lo, hi)
        pts_g = rk[3]
        for i in range(len(pts_g)):
            pt = tuple(Fraction(c) for c in _py(pts_g[i]))
            if not _on_curve(a, pt):
                raise _mismatch("Mordell-Weil point is not on the curve")
            try:
                o = _pic0_order(a, pt)
            except Exception:
                o = -1  # undecided; do not claim
            if o is not None and o != -1:
                raise _mismatch("a Mordell-Weil point has finite order")
            mw_pts.append((pt[0], pt[1]))
        ver["mw_points_on_curve"] = True
        ver["mw_points_infinite_order_pic0"] = True
        mw_status = "proven" if lo == hi else "bounds_only"
        if a_status == "proven" and a_rank <= 1 and not (lo <= a_rank <= hi):
            raise _mismatch("analytic rank outside the 2-descent rank bounds")
        if mw_status == "proven" and a_status == "proven" and lo != a_rank:
            raise _mismatch("proven Mordell-Weil rank differs from proven analytic rank")
        if mw_status == "proven" and a_status == "proven":
            ver["mw_rank_equals_analytic_rank"] = True
        not_checked.append("Mordell-Weil points are independent but not saturated")

    ver["not_checked"] = not_checked
    return EllipticCurveData(
        ainvs=tuple(a),
        discriminant=int(disc),
        conductor=N,
        ap=ap,
        torsion={"order": t_order, "structure": t_struct, "generators": gens},
        root_number=root,
        analytic_rank=a_rank,
        analytic_rank_status=a_status,
        l_value=lval,
        mw_rank_bounds=mw_bounds,
        mw_rank_status=mw_status,
        mw_points=mw_pts,
        verification=ver,
    )


# ---------------------------------------------------------------------------
# modular forms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EigenformOrbit:
    """A Galois orbit of normalised Hecke eigenforms of the new subspace.

    ``field`` is ``None`` for rational eigenvalues, otherwise the
    :class:`NumberField` ``Q[y]/(polynomial)`` containing them.  ``coefficients``
    is ``[a_1, ..., a_nmax]`` with ``a_n`` an ``int`` / ``Fraction`` (rational
    orbit) or a ``NumberFieldElement``.
    """

    degree: int
    polynomial: list[int]
    field: Any
    coefficients: list[Any]


@dataclass(frozen=True)
class ModularFormsData:
    """Cusp forms ``S_k(Gamma_0(N))`` with trivial character.

    ``dimension``
        ``dim S_k(Gamma_0(N))`` (full space, old forms included).
    ``basis``
        ``dim`` lists ``[a_0, ..., a_nmax]`` of exact ``Fraction`` / ``int``
        q-expansion coefficients (PARI's basis; *not* an eigenbasis).
    ``new_dimension`` / ``orbits``
        New-subspace dimension and its Hecke eigenform orbits.
    """

    level: int
    weight: int
    nmax: int
    dimension: int
    basis: list[list[Any]]
    new_dimension: int
    orbits: list[EigenformOrbit]
    verification: dict[str, Any] = field(default_factory=dict)


def ramanujan_tau(n: int) -> int:
    """Ramanujan's ``tau(n)`` from the product ``q prod (1 - q^m)^24``, exactly.

    Pure Python (no PARI); the independent oracle used to check ``Delta``.
    """
    return _delta_coeffs(n)[n]


def _delta_coeffs(nmax: int) -> list[int]:
    # prod (1-q^m)^24 truncated to degree nmax-1, via 24 passes of the Euler product
    deg = max(nmax, 1)
    ser = [0] * deg
    ser[0] = 1
    for m in range(1, deg):
        for _ in range(24):
            for i in range(deg - 1, m - 1, -1):
                ser[i] -= ser[i - m]
    return [0, *ser[:nmax]]


def _verify_hecke(coeffs: list[Any], N: int, k: int, nmax: int) -> None:
    """Hecke relations for ``coeffs[n] = a_n`` (index 0 unused), exact in the orbit field."""
    a = coeffs
    if not a[1] == a[1] * a[1]:  # a_1 = 1 <=> a_1^2 = a_1 and nonzero
        raise _mismatch("a_1 is not 1")
    for p in _primes_upto(nmax):
        pk = p ** (k - 1)
        r = 1
        pr = p
        # a_{p^{r+1}} = a_p a_{p^r} - [p not | N] p^{k-1} a_{p^{r-1}}
        while pr * p <= nmax:
            prev = a[pr // p]
            lhs = a[pr * p]
            rhs = a[p] * a[pr]
            if N % p:
                rhs = rhs - prev * _scalar_like(a[1], pk)
            if lhs != rhs:
                raise _mismatch(f"Hecke relation fails at p={p}, r={r + 1}")
            pr *= p
            r += 1
    for m in range(2, nmax + 1):
        for n_ in range(2, nmax // m + 1):
            if math.gcd(m, n_) == 1 and a[m * n_] != a[m] * a[n_]:
                raise _mismatch(f"multiplicativity fails at ({m}, {n_})")


def _scalar_like(one: Any, c: int) -> Any:
    if hasattr(one, "field"):
        return one.field.element([c])
    return c


def modular_forms(N: int, k: int, *, nmax: int = 20) -> ModularFormsData:
    """q-expansions and Hecke eigenvalues of ``S_k(Gamma_0(N))`` via PARI's ``mf*`` suite.

    Parameters
    ----------
    N, k
        Level ``N >= 1`` and **integer** weight ``k >= 1`` (trivial character).
    nmax
        Number of coefficients to return (``a_0..a_nmax`` / ``a_1..a_nmax``).

    Returns exact coefficients: ``int`` / ``Fraction`` for the basis and for
    rational eigenforms, ``NumberFieldElement`` in the orbit's coefficient
    field otherwise.

    Verification
    ------------
    Every eigenform is re-checked in exact arithmetic: ``a_1 = 1``, full
    multiplicativity and the prime-power Hecke recursion
    ``a_{p^{r+1}} = a_p a_{p^r} - [p ∤ N] p^{k-1} a_{p^{r-1}}`` for all indices
    ``<= nmax``; each ``a_n`` is an algebraic integer (integral minimal
    polynomial); the orbit degrees sum to ``dim S_k^new``.  For ``(N, k) =
    (1, 12)`` the form is compared with ``q prod (1 - q^n)^24``
    (Ramanujan's ``tau``).  A failure raises ``E-PARI-003``.  The *list of
    orbits being complete* is PARI's claim (not re-derived).

    Example
    -------
    >>> from alkahest import pari
    >>> pari.modular_forms(1, 12).basis[0][:4]      # tau(n)  # doctest: +SKIP
    [0, 1, -24, 252]
    """
    N, k, nmax = int(N), int(k), int(nmax)
    if N < 1 or k < 1 or nmax < 1:
        raise _unsupported("need N >= 1, k >= 1, nmax >= 1")
    ver: dict[str, Any] = {}
    mf = _gp("(N,k)->mfinit([N,k],1)", N, k)
    dim = int(_py(_gp("mfdim", mf)))
    basis: list[list[Any]] = []
    if dim:
        mat = _gp("mfcoefs", mf, nmax)  # (nmax+1) x dim
        for j in range(dim):
            col = _gp("(M,j)->M[,j]", mat, j + 1)
            basis.append([_py(col[i]) for i in range(len(col))])
    mfn = _gp("(N,k)->mfinit([N,k],0)", N, k)
    ndim = int(_py(_gp("mfdim", mfn)))
    orbits: list[EigenformOrbit] = []
    if ndim:
        fields = _gp("mffields", mfn)
        eig = _gp("mfeigenbasis", mfn)
        NumberField = _nf_class()
        for i in range(len(eig)):
            poly = [int(c) for c in _poly_coeffs(fields[i])]
            deg = len(poly) - 1
            raw = _gp("mfcoefs", eig[i], nmax)
            if deg == 1:
                K = None
                co = [Fraction(_py_scalar(raw[j])) for j in range(len(raw))]
                co = [int(c) if c.denominator == 1 else c for c in co]
                a_list = co
            else:
                K = NumberField(poly).with_variable("y")
                a_list = [_elt(K, raw[j]) for j in range(len(raw))]
                for j, v in enumerate(a_list):
                    if any(Fraction(c).denominator != 1 for c in v.minimal_polynomial()):
                        raise _mismatch(f"a_{j} is not an algebraic integer")
            one = a_list[1].is_one() if K is not None else a_list[1] == 1  # ty: ignore[unresolved-attribute]
            if not one:
                raise _mismatch("eigenform is not normalised (a_1 != 1)")
            if nmax >= 2:
                _verify_hecke(a_list, N, k, nmax)
            orbits.append(EigenformOrbit(deg, poly, K, a_list[1:]))
        if sum(o.degree for o in orbits) != ndim:
            raise _mismatch("orbit degrees do not sum to dim S_k^new")
        ver["hecke_relations"] = True
        ver["orbit_degrees_sum_to_new_dimension"] = True
    if (N, k) == (1, 12) and dim == 1:
        tau = _delta_coeffs(nmax)
        if [Fraction(v) for v in basis[0]] != [Fraction(v) for v in tau]:
            raise _mismatch("Delta differs from q prod (1-q^n)^24")
        ver["delta_matches_eta_product"] = True
    ver["not_checked"] = ["completeness of the eigenform list; dimension"]
    return ModularFormsData(N, k, nmax, dim, basis, ndim, orbits, ver)


def _py_scalar(g: Any) -> Any:
    t = _ptype(g)
    if t == "t_POLMOD":
        g = g.lift()
        t = _ptype(g)
    if t == "t_POL":
        c = _poly_coeffs(g)
        return c[0] if c else 0
    return _py(g)
