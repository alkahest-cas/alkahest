# Orthogonal polynomials and angular momentum

Alkahest has the classical orthogonal polynomials as registered primitives,
the complex spherical harmonics, and exact Wigner 3j / 6j symbols and
Clebsch–Gordan coefficients (`alkahest_cas::orthopoly` in Rust).

## Polynomials

| function | DLMF | convention |
|---|---|---|
| `legendre_p(n, x)` | 18.3 | `P_n(1) = 1` |
| `assoc_legendre_p(n, m, x)` | 14.6.1 | **Condon–Shortley phase included**: `P_n^m = (-1)^m (1-x²)^{m/2} dᵐP_n/dxᵐ`, `P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m` (Ferrers function, real on `[-1, 1]`) |
| `hermite_h(n, x)` | 18.3 | physicists', weight `e^{-x²}` |
| `hermite_he(n, x)` | 18.3 | probabilists', monic, weight `e^{-x²/2}` |
| `laguerre_l(n, x)` | 18.3 | `L_n = L_n^{(0)}` |
| `assoc_laguerre_l(n, α, x)` | 18.3 | generalised Laguerre `L_n^{(α)}` |
| `chebyshev_t(n, x)`, `chebyshev_u(n, x)` | 18.3 | first / second kind |
| `gegenbauer_c(n, a, x)` | 18.3 | `C_n^{(a)}`; `C_n^{(0)} = 0` for `n ≥ 1` |
| `jacobi_p(n, a, b, x)` | 18.3 | `P_n^{(a,b)}` |

These match SymPy's `legendre`, `assoc_legendre`, `hermite`, `hermite_prob`,
`laguerre`, `assoc_laguerre`, `chebyshevt/u`, `gegenbauer` and `jacobi`.

**Integer degree → explicit polynomial.** With an integer degree (and an
integer Legendre order) the call returns the polynomial with exact rational
coefficients, generated from the closed-form sums of DLMF §18.5:

```python
>>> import alkahest as ak
>>> p = ak.ExprPool(); x = p.symbol("x"); a = p.symbol("a")
>>> ak.legendre_p(3, x)          # 5/2·x³ − 3/2·x
>>> ak.assoc_laguerre_l(2, a, x) # coefficients polynomial in a
>>> ak.assoc_legendre_p(2, 1, x) # (1 − x²)^(1/2) · (−3x)
```

Family parameters may be symbolic. Negative degrees use the reflections that
keep a polynomial (`P_{-n-1} = P_n`, `T_{-n} = T_n`, `U_{-n} = −U_{n-2}`).
Expansion is capped at degree 1000 (64 with a symbolic parameter); past it the
call refuses with `E-SPECFN-004` — pass `expand=False`.

**Symbolic degree → node.** A symbolic or non-integer degree, or
`expand=False`, gives the node `legendre_p(n, x)`. It

* differentiates in `x` (`d/dx P_n = C_{n−1}^{(3/2)}`, `d/dx H_n = 2n H_{n−1}`,
  `d/dx L_n^{(α)} = −L_{n−1}^{(α+1)}`, …). The derivative in the degree or a
  family parameter has no closed form: `diff` refuses (`E-DIFF-001`) rather
  than drop the term;
* evaluates numerically — `f64` by the three-term recurrence for an integer
  degree, and rigorous balls (`interval_eval`) plus non-integer degrees through
  Arb's `arb_hypgeom_*` functions;
* simplifies at `P_ν(1) = 1`, `T_ν(1) = 1`, `U_ν(1) = ν + 1`, `L_ν(0) = 1`,
  at rational points for an integer degree, and by parity
  (`P_n(−x) = (−1)ⁿ P_n(x)`).

`expand_orthopoly(expr)` expands the nodes whose degree has since become an
integer, e.g. after `subs`.

Capability flags (`capabilities()["primitives"]`): `numeric_f64`,
`diff_forward`, and `numeric_ball` when the build carries FLINT's Arb layer.
There is no `diff_reverse` slot (the degree has no cotangent; reverse mode
still works through `diff_forward`) and no Lean theorem. For an integer
literal degree (and Legendre order) every family has a `taylor_model` rule —
the polynomial is expanded exactly and evaluated in Taylor-model arithmetic —
so `bound_on_box` accepts them; a symbolic degree is refused with
`E-VALIDATED-005`.

## Spherical harmonics

`spherical_harmonic_y(l, m, theta, phi)` returns

```text
Y_l^m = √((2l+1)/(4π) · (l−m)!/(l+m)!) · P_l^m(cos θ) · e^{imφ}
```

with the Condon–Shortley phase inside `P_l^m` (SymPy's `Ynm`, Mathematica's
`SphericalHarmonicY`); orthonormal on the sphere, `Y_l^{−m} = (−1)^m conj(Y_l^m)`.
`(1 − cos²θ)^{|m|/2}` is written `sin(θ)^{|m|}`, exact on the polar range
`0 ≤ θ ≤ π`. The imaginary unit is the pool's `I`. Integer `l`, `m` only
(`E-SPECFN-003`).

`real_spherical_harmonic_y(l, m, theta, phi)` is the real (tesseral) basis in
Wikipedia's convention: `√2·N·P_l^{|m|}(cos θ)·cos(mφ)` for `m > 0`,
`√2·N·P_l^{|m|}(cos θ)·sin(|m|φ)` for `m < 0`, `Y_l^0` for `m = 0`, with
`P_l^{|m|}` *without* the Condon–Shortley phase (it cancels the `(−1)^m` in
`Y_{l,m} = √2(−1)^m Re/Im Y_l^{|m|}`); `Y_{1,1} = √(3/4π)·x/r`. Real and
orthonormal on the sphere.

## Coupling coefficients

`wigner_3j(j1, j2, j3, m1, m2, m3)`, `wigner_6j(j1, …, j6)` and
`clebsch_gordan(j1, j2, j3, m1, m2, m3)` (SymPy's argument order:
`⟨j1 m1; j2 m2 | j3 m3⟩`) are exact: Racah's formulas over big integers, the
result an `Expr` `q` or `q·r^(1/2)` with `q` rational and `r` square-free.

```python
>>> from fractions import Fraction as F
>>> ak.wigner_3j(1, 1, 0, 0, 0, 0)                          # −1/3·3^(1/2)
>>> ak.clebsch_gordan(F(1, 2), F(1, 2), 1, F(1, 2), F(-1, 2), 0)   # 1/2·2^(1/2)
>>> ak.wigner_6j(1, 1, 1, 1, 1, 1)                          # 1/6
```

Arguments are ints, `Fraction`s, floats that are exactly `k/2`, or numeric
`Expr`s. Selection-rule violations give `0`; a non-(half-)integer refuses with
`E-SPECFN-001`, a negative `j` with `E-SPECFN-002`, `2j > 20000` with
`E-SPECFN-004`. The result lives in the pool of an `Expr` argument, else
`pool=`, else a module-wide default pool.

## Errors

| code | meaning |
|---|---|
| `E-SPECFN-001` | an angular-momentum argument that is not an integer or half-integer |
| `E-SPECFN-002` | a negative `j` or `l` |
| `E-SPECFN-003` | a non-integer `l` or `m` for `spherical_harmonic_y` |
| `E-SPECFN-004` | an expansion / coefficient past the size cap |
| `E-SPECFN-005` | the wrong number of family parameters (Rust API) |
