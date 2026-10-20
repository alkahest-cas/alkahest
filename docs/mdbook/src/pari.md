# PARI/GP interop (optional)

Alkahest's number fields have arithmetic only; it has no ring of integers, class group,
units, prime-ideal factorisation, Galois group, elliptic-curve invariants or modular
forms. PARI/GP has all of these. `alkahest.pari` asks PARI for them and returns **exact
Alkahest objects** (`int`, `Fraction`, `NumberFieldElement`, `ArbBall`), then **re-checks
every cheap consequence** in Alkahest's own arithmetic.

## Licence boundary

PARI/GP is GPL. Alkahest never links, bundles or redistributes it, and `import alkahest`
does not import it. `alkahest.pari` is pure Python and loads, on first use, a binding **you
installed yourself**: `cypari2` (against a system libpari) or `cypari` (its wheels carry
their own libpari). With neither, `pari.available()` is `False` and every call raises
`PariUnavailable` (`E-PARI-001`). A `gp` subprocess is not supported. The GPL applies to
whatever *you* distribute that includes PARI; Alkahest itself is unaffected.

## Number fields

```python
from alkahest import pari
d = pari.number_field_invariants([5, 0, 1], primes=[2, 3, 5])   # Q(sqrt -5)
d.discriminant, d.class_number, d.class_group      # -20, 2, [2]
d.integral_basis, d.fundamental_units, d.torsion_units, d.galois
d.primes[3]          # [PrimeIdeal(p=3, e=1, f=1, ...), ...]  two-element (p, alpha) form
d.certification      # 'certified' | 'grh_conditional'
```

`bnfinit` is conditional on GRH; `bnfcertify` is run when degree ≤ 6 and `|d_K| < 10^15`
(`certify=True` forces it, `False` skips it) and the result is labelled accordingly.
Verified here: Gram determinant `= d_K`, integral basis closed under multiplication and
consisting of integers, `disc(f) = index² d_K`, unit norms `±1`, each prime ideal's
lattice index `= p^f`, `Σ e f = n`, ramification `⇔ p | d_K`, Galois parity vs
`disc(f)`, imaginary-quadratic class numbers by counting reduced forms. Listed under
`verification["not_checked"]`: maximality of the order, fundamentality of units, the class
group structure (beyond quadratic imaginary).

## Elliptic curves

```python
e = pari.elliptic_curve_invariants([0, -1, 1, -10, -20])   # 11a1
e.conductor, e.torsion, e.ap[7], e.analytic_rank, e.l_value
```

`a_p` is recounted by point counting (p ≤ 1000); torsion generators' orders are recomputed
in Alkahest's `Pic⁰`; `L(E,1)` (root number +1) or `L'(E,1)` (−1) is recomputed from the
`a_n` with a rigorous tail bound and returned as a ball. `analytic_rank_status` is
`proven` when that ball excludes 0, else `numerical`. `mw_rank_status` is `proven` when
the 2-descent bounds agree, else `bounds_only`; `mw_points` are independent points of
infinite order (checked in `Pic⁰`), **not** a saturated basis.

## Modular forms

```python
m = pari.modular_forms(23, 2, nmax=20)
m.dimension, m.basis, m.orbits[0].field, m.orbits[0].coefficients
```

Every eigenform is re-checked for `a_1 = 1`, multiplicativity and the prime-power Hecke
recursion in the eigenvalue field; `(1, 12)` is compared with `q∏(1−qⁿ)²⁴`
(`pari.ramanujan_tau`).

## Error codes

`E-PARI-001` no binding, `E-PARI-002` unsupported input (non-monic polynomial, singular or
non-integral curve, `polgalois` unavailable), `E-PARI-003` a PARI answer failed
verification, `E-PARI-004` PARI itself raised.
