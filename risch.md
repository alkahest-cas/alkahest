# V1-2: Algebraic-Function Risch — Design

**Status:** Deferred from v1.0; target v1.1. 
**Owner:** unassigned
**Prerequisites shipped:** V5-4 (exp/log Risch tower), V5-11 (parallel F4
Gröbner basis), V1-4 (polynomial system solver), V2-2 planned (resultants).

---

## 1. Goal

Extend `alkahest.integrate(...)` from the **exp/log transcendental tower**
shipped in V5-4 to the **algebraic** case.  Concretely, after V1-2 the engine
must produce an elementary antiderivative (or prove none exists) for integrands
containing:

- `sqrt(P(x))`, `cbrt(P(x))`, `(P(x))^(p/q)` for `P ∈ ℚ[x]`;
- symbolic `RootOf(y, f(x, y))` — the root of a polynomial relation over `ℚ(x)`;
- rational combinations of the above with existing exp/log transcendentals.

**Acceptance target (from `plan.md` §V1-2):**
- Oracle match rate ≥ 90 % against SymPy on 5 000 randomly-generated
  algebraic integrands.
- `examples/risch_integration.py` gains 10 worked algebraic examples.
- `tests/test_oracle.py` drops `@pytest.mark.skip` on the 45 currently-skipped
  algebraic tests.

The v1.0 scope explicitly **excludes** nested algebraic–transcendental towers
(e.g. `sqrt(exp(x) + 1)`) — those are tracked separately.

---

## 2. Mathematical background

The algorithm we follow is **Trager's** (1984 MIT PhD thesis, refined by
Bronstein, *Symbolic Integration I*, chs. 10–11).  Pipeline:

1. **Field setup.** The integrand lives in an algebraic extension
   `K = ℚ(x)[y] / (m(x, y))` for some minimal polynomial `m`.  Elements of
   `K` are represented as polynomials of degree `< deg_y(m)` in `y` with
   `ℚ(x)` coefficients.

2. **Integral basis.**  Find a `ℚ[x]`-basis `{w_0, …, w_{n-1}}` of the
   *integral closure* of `ℚ[x]` in `K`.  "Integral" means: each `w_i` is a
   root of a *monic* polynomial over `ℚ[x]`.  This is the Trager–Coates
   algorithm (Bronstein §10.4).

3. **Hermite reduction** (Bronstein §11.1).  Given `f ∈ K`, decompose
   `∫ f dx = g + ∫ h dx` with `g ∈ K` and `h` having only **simple poles**
   on the projective curve of `m`.  Same shape as the rational-function
   Hermite reduction, lifted through the integral basis.

4. **Logarithmic part** (Bronstein §11.2).  The remaining `∫ h dx` is an
   elementary function iff a certain **divisor** `div(h)` is torsion in the
   Jacobian of the curve defined by `m`.  Detection reduces to
   ideal-membership in `K[x]` — this is where V5-11 Gröbner bases + V2-2
   resultants become load-bearing.  When the divisor is torsion of order `N`,
   the antiderivative is `(1/N) · log(α)` for a computable `α ∈ K`.

5. **Failure case.**  If no torsion witness is found within the
   Mignotte/Hironaka bound, emit `NonElementary` — the user-facing semantics
   already exist in V5-4.

---

## 3. Module layout

```
alkahest-core/src/integrate/
├── mod.rs                     # existing façade; add `algebraic` route
├── rational.rs                # existing (V0.2)
├── transcendental.rs          # existing (V5-4 exp/log tower)
└── algebraic/                 # NEW
    ├── mod.rs                 # public entry: integrate_algebraic(f, x) -> Result
    ├── field.rs               # AlgebraicField, AlgebraicElement
    ├── minpoly.rs             # minimal-polynomial extraction
    ├── integral_basis.rs      # Trager–Coates
    ├── hermite.rs             # reduction to simple-pole form
    ├── divisor.rs             # Divisor, Place, linear equivalence
    ├── logarithmic.rs         # torsion detection → log(α)/N
    └── tests/                 # unit + oracle corpora
```

Python façade: `alkahest.integrate(...)` keeps its single entry point; the
dispatcher in `integrate/mod.rs` adds an "algebraic subterm detected" branch
before falling back to `transcendental::integrate`.

---

## 4. Data structures

### `AlgebraicField`

```rust
pub struct AlgebraicField {
    /// Base ring ℚ(x); `x` is var-index 0.
    pub base_var: Symbol,
    /// Minimal polynomial m(x, y) ∈ ℚ(x)[y], primitive and squarefree in y.
    pub minpoly: UniPolyOverRational,   // coefficients are RationalFunction<ℚ[x]>
    /// Cached discriminant, for ramified-place detection.
    disc_cache: OnceCell<UniPoly<Rational>>,
}
```

### `AlgebraicElement`

```rust
pub struct AlgebraicElement {
    field: Arc<AlgebraicField>,
    /// Coefficients `c_0(x) + c_1(x)*y + … + c_{n-1}(x)*y^{n-1}` in ℚ(x).
    coeffs: Vec<RationalFunction<UniPoly<Rational>>>,
}
```

Implements `Add`/`Sub`/`Mul`/`Div` by reduction modulo `minpoly` — standard
univariate polynomial arithmetic over `ℚ(x)`, with each coeff operation
routed through the existing `RationalFunction` machinery.

### `Divisor`

```rust
pub struct Divisor {
    /// Sparse: (place, multiplicity) pairs.  Multiplicity ∈ ℤ.
    terms: BTreeMap<Place, i64>,
}

pub struct Place {
    /// Point on the projective curve: either (α, β) with m(α,β)=0
    /// or "∞" (with valuation info).  Stored symbolically.
    coords: PlaceCoords,
    ramification_index: u32,
}
```

Divisor arithmetic is additive.  Linear equivalence `D ~ D'` ⇔ `D - D'` is
principal — tested via `logarithmic::is_principal(...)` using resultants +
Gröbner bases.

---

## 5. Algorithm components

### 5.1 Minimal polynomial extraction (`minpoly.rs`)

Input: an `ExprId` tree containing algebraic subterms like `sqrt(P)` or
user-declared `RootOf(...)`.
Output: `AlgebraicField` containing the minimal polynomial over `ℚ(x)` of
a chosen primitive element.

- Each `sqrt(P)` contributes `y² - P`; each `(Q)^(p/q)` contributes `y^q - Q^p`.
- Multiple algebraic generators are combined via **resultants** (V2-2, if
  available; else a fallback using Buchberger elimination from V5-11).
- Result is stored squarefree, primitive, and with `gcd(leading_coeff, disc) = 1`.

**Edge cases:**
- Integer radicands must be kept symbolic — no numerical rounding.
- Nested radicals (`sqrt(x + sqrt(x))`) require iterated resultants and a
  degree-bound check (`≤ 8` for v1.1; larger degrees reject as
  `UnsupportedExtensionDegree`).

### 5.2 Integral basis (`integral_basis.rs`)

Implements Trager–Coates (Bronstein Algorithm 10.4.1).

- For each prime `p(x)` dividing the discriminant of `minpoly`:
  - Localise at `p`, compute a `p`-integral basis via Puiseux expansion
    or the "round 2" algorithm.
  - Glue the local bases into a global `ℚ[x]`-basis.
- Puiseux expansions are computed to a degree bound `2 · deg_y(m) · v_p(disc)`.
- Output: `Vec<AlgebraicElement>` of length `deg_y(minpoly)`.

This is the **hardest** sub-component.  Dependencies:
- Newton polygon / Puiseux code (new; uses `MultiPoly` for the slope
  computations).
- Local rings / discrete valuations (new; `LocalRing<p>` wrapping
  `UniPoly<ℚ>`).

Effort: ~1.5 weeks of focused work.

### 5.3 Hermite reduction (`hermite.rs`)

Given `f ∈ K` and the integral basis, split
`f = df'/dx + simple_pole_part(g)` so `∫ f = f' + ∫ g`.

- Iterates the rational-function Hermite trick on each coordinate of `f` in
  the integral basis.
- Convergence: bounded by `Σ v_P(f) - 1` over all poles `P`.
- Output: `(closed_form_part: AlgebraicElement, simple_pole_integrand: AlgebraicElement)`.

### 5.4 Divisor calculus (`divisor.rs`)

- `div(f)`: compute the divisor of zeros and poles of `f ∈ K` on the curve.
  Uses resultants to locate places, valuations via Puiseux.
- Principal-divisor test: `is_principal(D) -> Option<AlgebraicElement>`.
  Implementation: reduce to ideal-membership in `ℚ[x, y] / (m)` and query
  a Gröbner basis (V5-11) for the structure constants.

### 5.5 Logarithmic part (`logarithmic.rs`)

Given `h` with only simple poles:
1. Compute `D = div(h) - poles_of_h` (effective at poles).
2. For each candidate torsion order `N` up to the **Hironaka bound**
   `N ≤ (2g + 1)²` where `g = genus(curve)`:
   - Test `N · D ~ 0` via `is_principal`.
   - If yes, recover `α ∈ K*` with `div(α) = N · D`; return `(1/N) · log(α)`.
3. If no `N ≤ bound` works, return `NonElementary { reason: "divisor non-torsion" }`.

**Performance note:** The Hironaka bound is theoretical worst-case.  In
practice most integrands resolve at `N ≤ 6`; we short-circuit there and
only escalate on failure.

### 5.6 Dispatch in `integrate/mod.rs`

```rust
pub fn integrate(expr: ExprId, var: Symbol, pool: &ExprPool) -> IntegrationResult {
    // existing routes...
    if contains_algebraic_subterm(expr, pool) {
        if let Ok(result) = algebraic::integrate_algebraic(expr, var, pool) {
            return result;
        }
    }
    transcendental::integrate(expr, var, pool)  // V5-4 fallback
}
```

Rust-side the new variant `IntegrationError::UnsupportedExtensionDegree(u32)`
needs a diagnostic code — suggest `E-INT-005` following V1-3 conventions.
Python mirrors via `alkahest.exceptions.IntegrationError`.

---

## 6. Test plan

### 6.1 Unit (`algebraic/tests/`)

| Component | Test |
|---|---|
| `minpoly` | `sqrt(x² + 1)` → `y² - (x² + 1)` |
| `minpoly` | `sqrt(x) + cbrt(x)` → degree-6 primitive element |
| `integral_basis` | `y² = x³ - x` → `{1, y}` (elliptic, smooth) |
| `integral_basis` | `y² = x³` (cuspidal) — local ring at 0 is not regular, basis is `{1, y/x}` |
| `hermite` | `1/(y*(x-1))` where `y² = x`: single Hermite step suffices |
| `divisor` | `div(x) = 2·(x=0) - 2·∞` on `y² = x³ - x` |
| `is_principal` | `N=2` torsion on elliptic curve `y² = x³ - 1` — recovers `α = (y + x^(3/2)) / …` |
| `logarithmic` | `∫ (dx)/(sqrt(x² - 1))` → `log(x + sqrt(x² - 1))` |

### 6.2 Oracle (`tests/test_oracle.py`)

- Corpus: `tests/oracle_algebraic.py` — 5 000 integrands generated from
  the grammar `integrand = rational * algebraic`, bounded by:
  - `deg_x ≤ 6`, `deg_y ≤ 4`, coefficient bit-width `≤ 32`.
  - Weighted toward integrands known to have elementary antiderivatives
    (oracle expected success ≥ 70 %).
- SymPy 1.12 is the oracle.  For each sample:
  - Compute `sympy_result = sympy.integrate(f, x)` (60 s timeout).
  - Compute `alk_result = alkahest.integrate(f, x)` (60 s timeout).
  - Verify by `sympy.simplify(alk_result.diff(x) - f) == 0`.
- **Agreement target:** ≥ 90 % on samples where SymPy returns a
  closed form.
- **Non-elementary detection:** on a hand-curated 100-integrand corpus
  of known non-elementary algebraic integrals (Abelian integrals), Alkahest
  must return `NonElementary` on ≥ 95 % — not silently succeed with an
  incorrect answer.

### 6.3 Criterion benchmark

New row `algebraic_integrate_corpus` in `benchmarks/tasks.py`:
- 20-integrand hand-picked corpus, mix of elliptic and hyperelliptic.
- Compare against SymPy on the same corpus.
- Target: ≤ 3× slower than SymPy on the 90 % that both solve (V1-2 is
  correctness-first; V2.x performance work tracked separately).

### 6.4 Lean certificates

For each `(integrand, antiderivative)` pair the algorithm emits, also emit
the Lean derivative-check tactic:

```lean
example : deriv (fun x => {antideriv_lean}) = fun x => {integrand_lean} := by
  ring_nf; field_simp; ring
```

Lean CI (V5-8) runs this on the full 5 000-integrand oracle output nightly.

---

## 7. Risks and open questions

### High-risk

1. **Integral basis for singular curves.**  The Trager–Coates algorithm
   degenerates when the curve has non-ordinary singularities.  Bronstein
   §10.4 uses the "round 2" algorithm as a fallback; implementing it
   correctly is fiddly.  *Mitigation:* explicit singular-curve test corpus
   in `integral_basis/tests/`.

2. **Hironaka bound is pessimistic.**  On low-genus curves the torsion
   search short-circuits, but for genus ≥ 3 the bound scales quadratically
   and the corpus will include integrands where we time out despite having
   an elementary answer.  *Mitigation:* adaptive bound with user-facing
   configuration `IntegrationConfig::torsion_bound: Option<u32>`.

3. **Puiseux expansion precision.**  Over-truncation → wrong valuations →
   silently wrong integrals.  *Mitigation:* cross-check each Puiseux
   expansion by substitution back into the minimal polynomial before use.

### Medium-risk

4. **Nested radicals.** `sqrt(x + sqrt(x))` is degree 4 but numerically
   close to a degree-2 extension; the resultant-based minpoly must produce
   the true degree-4 polynomial, not a factor.  *Mitigation:* squarefree
   factorisation of every computed minpoly before use.

5. **Rational-function coefficient blow-up.**  Integral-basis elements
   often have `ℚ(x)` coefficients with denominators divisible by the
   discriminant.  *Mitigation:* content/primitive-part normalisation at
   every multiplication.

### Open questions

- Should `RootOf(poly, root_index)` be a first-class `ExprData` variant, or
  remain synthetic (reconstructed from `Pow` + `rug::Rational`)?  Trager's
  output cleanly produces a single primitive element but users often want
  answers factored through the original radicals.  *Proposal:* introduce
  `ExprData::AlgebraicRoot { minpoly: ExprId, branch: u32 }` in v1.1 but
  keep it experimental until v2.0.

- Lean certificate granularity: one tactic per integrand, or one per
  Hermite-reduction step?  The step-by-step version is easier to debug
  but 10× slower to typecheck.  *Proposal:* emit the aggregated tactic by
  default, step-by-step only when `derivations.explain=True`.

---

## 8. Delivery plan

Phased; each phase lands behind the experimental feature flag
`integrate-algebraic`, removed when the acceptance gates clear.

| Phase | Scope | Eng-weeks |
|---|---|---|
| P1 | `AlgebraicField`, `AlgebraicElement`, field arithmetic, minpoly from single `sqrt(P)` | 1.0 |
| P2 | Integral basis (smooth curves only) + Hermite reduction | 1.5 |
| P3 | Divisor calculus + `is_principal` via Gröbner | 1.0 |
| P4 | Logarithmic part (torsion search) — handles elliptic + hyperelliptic of genus ≤ 3 | 1.0 |
| P5 | Singular curves (round-2 integral basis); nested radicals | 1.5 |
| P6 | Oracle corpus + Lean certificates + dropout of 45 skipped tests | 1.0 |

Total: ≈ 7 engineer-weeks.  Gate each phase on internal correctness tests
(§6.1) before the next one starts.

---

## 9. References

- Trager, B. (1984). *Integration of algebraic functions*. MIT PhD thesis.
- Bronstein, M. (2005). *Symbolic Integration I: Transcendental Functions*
  (2nd ed.), chs. 10–11. Springer.
- Davenport, J. H. (1981). *On the Integration of Algebraic Functions*.
  LNCS 102.
- van Hoeij, M. (1994). *An algorithm for computing an integral basis in
  an algebraic function field*. J. Symb. Comp. 18.
- Risch, R. H. (1970). *The solution of the problem of integration in
  finite terms*. Bull. AMS 76 — historical grounding.

**In-tree cross-references:**
- V5-4 exp/log Risch: `alkahest-core/src/integrate/transcendental.rs`
- V5-11 Gröbner bases: `alkahest-core/src/poly/groebner/`
- V1-3 structured errors: `alkahest-core/src/errors/mod.rs` (add `E-INT-005`)
- V1-4 polynomial solver: `alkahest-core/src/solver/` (reuse for
  ideal-membership witnesses)
