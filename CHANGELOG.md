# Changelog

## Unreleased

- **Every printer emitted `(-1)^n` as `-1^n`, which re-reads as `-(1^n)`.** A
  negative power base was rendered without parentheses by all three exported
  forms — `str`/`repr`, `latex` and `unicode_str` — so `str((-1)**n)` was
  `'-1^n'`, which sympy, Python's own `eval`, LaTeX **and `alkahest.parse`**
  all correctly read as `-(1^n) = -1`, not `(-1)^n`. Nothing internal was
  wrong: `(-1)^4` evaluated to `1` all along; only the exported text lied, and
  it lied in exactly the place it does the most damage — an `M1` boundary
  result handed out for external checking. `b(n) = -16·(-2)^n` printed as
  `-16 * -2^n`, worth `-64` at `n = 2` inside alkahest and `+16` once
  re-parsed, which is enough to make an audit harness report a correct engine
  as unsound. The printers now parenthesise any base that does not bind
  tighter than `^`: a negative literal (unary minus binds looser than `^`,
  matching `BP_UNARY` in the parser) and, in the LaTeX and Unicode renderers,
  a fraction — `\left(\frac{1}{2}\right)^n`, `(3/7)^(n)`. Bases that were
  already unambiguous are untouched (`2^n`, `x^n`, `½^(n)`), as are negative
  *exponents* (`x^-1`), where a leading `-` cannot be misread because `^` is
  right-associative. `alkahest.parse` itself was **not** changed: it was
  applying standard precedence correctly to the bad string it was given.

- **`telescope2d` generalizes from two bound indices to an arbitrary `m ≥ 1`:
  `experimental.telescope_md`** (M4 extension). `telescope2d(term, n, j, k)`
  only ever reached exactly two bound indices; the underlying ansatz search
  and boundary/face analysis are now implemented for general `m`, with
  `telescope2d` itself unchanged in behavior (it is now a thin `m = 2`
  wrapper over the general engine, not a separate implementation) and a new
  `telescope_md(term, n, [x_1, …, x_m])` for `m ≠ 2` — including `m = 1`,
  which degenerates cleanly to a single-sum-shaped search, and `m ≥ 3`,
  genuinely new. Same scope as before, generalized: proper hypergeometric
  summands only (no broader rational-summand class, no sum of several
  hypergeometric terms), a fixed (non-minimal) certificate denominator built
  from `F`'s own shift-ratio denominators rather than a minimal multivariate
  Gosper normal form, and constant-box-only boundary analysis. **Not
  attempted**: a genuine minimal Gosper-style certificate denominator (the
  roadmap's stated remaining-gap item 3) — real algorithm-design work, not an
  engineering extension of what already exists here.

  **The boundary is `2m` face sums, not `2^m` corner evaluations** — the
  `m = 2` module's "four strip sums, not four corners" result, generalized:
  summing the telescoping identity over an `m`-dimensional box gives `2m`
  sums, each over an `(m − 1)`-dimensional face where one bound index is
  fixed to a boundary value, not `2^m` point evaluations at the box's
  corners. The same sufficient (not necessary) "face vanishes pointwise"
  criterion the `m = 2` module used — a dominant `1/Γ` zero among `F`'s own
  gamma factors, or the certificate's own numerator vanishing there — carries
  over unchanged in kind: fix one axis to a constant and check that a gamma
  factor's argument no longer depends on `n` or any *other* bound index.

  **Resource ceilings, added after this generalization surfaced a real
  scaling cliff.** The linear system a probe builds has one equation per
  distinct monomial and one unknown per certificate-numerator box
  coefficient; both dimensions grow with `m` and the certificate degree
  bound far faster than the degree numbers suggest, and `rational_nullspace`
  is a plain dense `O(rows · cols²)` exact-rational Gaussian elimination.
  Profiling a genuinely coupled `m = 3` example
  (`C(n,x)·C(x,y)·C(y,z)`) at degree bounds that work fine at `m = 2` found a
  ≈10 000-row, 245-unknown system whose elimination step alone took ≈47
  seconds *per probe*, and the next certificate-degree step up (770 unknowns)
  was still running after several minutes — genuine `O(rows·cols²)`
  arithmetic cost on a correctly-posed system, not a bug or an infinite loop,
  but exactly the kind of resource cliff a caller needs protecting from.
  `holonomic::telescoping2d::search::MAX_ANSATZ_UNKNOWNS` now refuses any
  single probe above 400 unknowns outright, and
  `MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS` caps the total work spent on probes
  at or above 150 unknowns to 300 across one whole search call — so a search
  with no certificate in reach cannot be made to pay the same expensive
  elimination over and over across every `(order, a_degree)` combination
  tried. Neither ceiling touches the `m = 2` search, whose default probes
  never exceed ≈140 unknowns. A search that hits a ceiling reports
  `SearchExhausted` naming the ceiling explicitly, never a false certificate
  and never a silent hang.

  Verified on the 4-category multinomial coefficient
  `F(n,x,y,z) = n!/(x!y!z!(n−x−y−z)!)` (via `factorial`, not a
  product-of-binomials encoding, to avoid the redundant cancelling-gamma-
  factor pairs a naive `C(n,x)·C(n−x,y)·C(n−x−y,z)` encoding would carry) — a
  genuinely non-separable `m = 3` sum (all three bound indices interact
  through the shared `n − x − y − z` term) whose closed form,
  `Σ_{x,y,z} F = 4ⁿ` (the multinomial theorem), is checked by direct exact
  summation (`rug::Rational`, never floats) against the returned recurrence,
  plus the `4ⁿ`-decoupled fixed-support variant where the boundary genuinely
  is `n`-independent and `Vanishes` is provided rather than refused. A
  regression test pins the original scaling-cliff example
  (`C(n,x)·C(x,y)·C(y,z)` at `order ≤ 2`, `cert_degree ≤ 3`) to now return a
  bounded, ceiling-cited refusal rather than run unboundedly.

  New: `alkahest.experimental.telescope_md`,
  `alkahest.experimental.TelescopingMdCertificate`. Experimental, same
  refusal codes as `telescope2d` (`E-HOLO-040`/`041`/`042`).

## 3.9.0 — 2026-08-14

Everything in this section landed **after `v3.8.0` was tagged and published**,
so none of it is in the released 3.8.0 wheel. It is the result of acting on a
trial autoresearch run: fifteen logged issues, of which this release fixes
fourteen (issue #6 was expectation-setting, not a defect, and issue #10 is
reduced but not closed — see below).

**Upgrading from 3.8.0 is not transparent.** Two changes need a read before you
upgrade: `relation_confidence` now returns a *tri-state* verdict, and seventeen
zero-argument accessors became properties, which is a hard break with no alias.
Both are detailed under *Behaviour changes to plan for*.

### Behaviour changes to plan for

- **`relation_confidence` returns `credible: None` — *unknown* — for inputs
  whose precision it cannot establish, where it used to return `True`.** It
  judged only `float` inputs, on the premise that "decimal strings and ints are
  exactly the values they spell, so a relation among them holds exactly". That
  premise is false for the one way PSLQ is actually driven: a high-precision
  decimal string standing for a **truncated** numerical value. So on the input
  an experimental-mathematics loop produces, the gate passed everything —
  including three relations found during the 2026-08-13 autoresearch run that
  re-evaluation at 60 digits refutes. A gate that cannot fail is worse than no
  gate, because loop authors wire it into promotion logic. Now: `float` is 53
  bits as before, `mpmath.mpf` reports its working precision, `int` and
  `Fraction` are exact, and everything else — decimal strings included — is
  `None` until the caller declares `digits=` or `precision_bits=` (a cap, not an
  override: declaring 200 digits for a `float` still yields ~16). A relation must
  also clear the available precision by `margin_digits` (default 10) rather than
  merely fitting inside it, because PSLQ returns the *smallest* relation the
  precision can buy, so a purchased one lands just under the old bound rather
  than over it; all three of the run's spurious relations were 5–8 digits under
  it. The verdict dict gains `margin_digits` and `precision_source`, and
  `available_digits` / `spare_digits` are `None` when the verdict is. **If you
  branch on `credible`, treat `None` as "not checked" — `if verdict["credible"]`
  is the correct polarity, `is not False` is not.** `guess_relation` is
  unchanged for decimal strings: an unjudgeable input is returned unjudged, never
  refused, and `E-PSLQ-004` still fires on float inputs. Unknown precision does
  not disable the gate, either: available precision is a `min` over the inputs,
  so one input whose precision *is* known bounds the whole set, and a relation
  already too expensive for that bound comes back `False` even while
  `available_digits` stays `None` — a single `float` among decimal strings caps
  the search at ~16 digits however many digits the strings carry.
- **`pool.symbol("x")` now takes its domain from the ambient
  `alkahest.context(domain=...)`**, falling back to `Domain.Real` only outside a
  context block; an explicit `domain=` argument still wins. It previously
  ignored the context entirely, while the documented `alkahest.symbol("x")` —
  which is a thin wrapper that resolves the context and forwards it — did not.
  The two constructors sit side by side and only one consulted the domain, so
  building an integer problem through the pool silently emitted
  `(set-logic QF_NRA)` and `Real` sorts, and `solve()` answered the *real
  relaxation*: an Erdős–Straus instance came back `sat` with `z = 252/13` while
  `status`, `verification.status` (`exactly_verified`) and `supported().reason`
  (`ok`) all stayed green, because the model does satisfy the formula as
  emitted. Nothing was unsound; the question being answered had changed. Code
  that relied on pool symbols being `Real` inside an integer context should pass
  `domain="real"` explicitly.
- **Seventeen zero-argument scalar accessors became properties — drop the
  `()`.** There was no rule a caller could predict: `Enclosure.width` and
  `RegularChain.n_vars` were properties while `DAE.n_equations()` and
  `MultiPoly.total_degree()` were methods, so the same shape of question was
  asked two different ways depending on the class. The convention now is: **a
  zero-argument, O(1), non-allocating accessor returning a scalar or a flag is a
  property; anything that returns a collection, allocates, or does real work is
  a method.** No compatibility alias is provided, deliberately — an accessor
  that answers to both forms leaves the ambiguity in place. Migration is
  mechanical:

  | Before | After |
  |---|---|
  | `UniPoly.degree()` | `UniPoly.degree` |
  | `UniPoly.is_zero()` | `UniPoly.is_zero` |
  | `MultiPoly.is_zero()` | `MultiPoly.is_zero` |
  | `MultiPoly.total_degree()` | `MultiPoly.total_degree` |
  | `MultiPolyFp.is_zero()` | `MultiPolyFp.is_zero` |
  | `MultiPolyFp.total_degree()` | `MultiPolyFp.total_degree` |
  | `RationalFunction.is_zero()` | `RationalFunction.is_zero` |
  | `GbPoly.is_zero()` | `GbPoly.is_zero` |
  | `GbPoly.n_vars()` | `GbPoly.n_vars` |
  | `ODE.order()` | `ODE.order` |
  | `DAE.n_equations()` | `DAE.n_equations` |
  | `DAE.n_variables()` | `DAE.n_variables` |
  | `HybridODE.n_events()` | `HybridODE.n_events` |
  | `Component.n_equations()` | `Component.n_equations` |
  | `Component.n_ports()` | `Component.n_ports` |
  | `OdeTrajectory.t_final()` | `OdeTrajectory.t_final` |
  | `ArbBall.is_exact()` | `ArbBall.is_exact` |

  **Calling the old form now raises `TypeError: 'int' object is not callable`
  (or `'bool'`, `'float'`), which is loud. The reverse mistake is not.**
  Writing `if dae.n_equations:` against a *pre*-3.8.0 build silently reads a
  bound method, which is always truthy, and `f"{dae.n_equations}"` formats as
  `<built-in method ...>`; so grep for these names rather than waiting for a
  traceback. Accessors that were *already* properties (`Enclosure.lower`,
  `.upper`, `.width`, `.subdivisions`, `Matrix.rows`, `.cols`,
  `RegularChain.n_vars`, `RosenfeldGroebnerResult.consistent`, `.truncated`,
  `ArbBall.mid`, `.rad`, `.lo`, `.hi`, …) are unchanged; they were already
  correct under the rule, as were the collection-returning methods that sit
  beside them (`RegularChain.polys()`, `RosenfeldGroebnerResult.final_basis()`).
  Three zero-argument scalar calls stay methods because they do real work rather
  than read a field: `Matrix.rank()` (Gaussian elimination), `ODE.is_autonomous()`
  (walks every RHS expression) and `PositivityCertificate.verify()` (re-runs the
  exact SOS identity check). `tests/test_accessor_convention.py` pins the
  converted set and scans `alkahest-py/src/lib.rs` for new offenders, so the
  inconsistency cannot creep back.

### Fixed

- **A verified Zeilberger certificate could imply a *false* recurrence for the
  sum, and nothing in the result distinguished that case from a correct one.**
  This is the only defect logged in the 2026-08-13 autoresearch run that could
  make a loop report a theorem that is not one; everything else was a refusal, a
  slowdown or a discoverability gap. Zeilberger's algorithm proves
  `Σ_i a_i(n)·F(n+i,k) = ΔG`, an identity about the **summand**; a recurrence
  for `S(n) = Σ_k F(n,k)` needs a boundary term to vanish, which is a separate
  statement. On OEIS A279013, `a(n) = Σ_{k=0}^{n} C(2k,k)/(k+1)·C(2n−1,n−k)`,
  `zeilberger` returned a verified order-2 certificate in 0.1 s whose
  homogeneous recurrence fails against the real sequence (2, 8, 35, 161, 768,
  3773) at the very first term. It was caught only by a harness that carried an
  independent numeric check against OEIS data — every signal reachable through
  the API said "proved".

  The information was not missing: `side_conditions` already named the exact
  condition and even gave a counterexample. It was **invariant** — the same
  string for a correct and an incorrect case — so nothing a loop reads could
  tell them apart. `zeilberger` now computes a verdict.
  `ZeilbergerCertificate.boundary` is `"vanishes"` (proved; the homogeneous
  recurrence holds for the sum), `"nonzero"` (proved; the **inhomogeneous**
  `Σ_i a_i(n)·S(n+i) = b(n)` holds, with `b(n)` in `boundary_rhs`) or
  `"unknown"` (neither; nothing may be claimed about the sum).
  `boundary_reason`, `implies_sum_recurrence`, `limits` and `boundary_at(k_lo,
  k_hi)` come with it, and `side_conditions` now varies with the verdict.
  A279013 comes back `"nonzero"` with a `b(n)` that reproduces the sequence
  exactly; Franel, Dixon, Apéry and the binomial row sum come back `"vanishes"`.

  **`"vanishes"` is a proof, not a numeric check.** Each endpoint of `G` is
  evaluated by exact order counting in `Q(n)` — the multiplicity of the endpoint
  as a root of the certificate's numerator and denominator, deflated exactly,
  plus `−e` for every `Γ(a·n+b·k+c)^e` factor whose argument lands on a
  non-positive integer there, with the `(−1)^m/(m!·b)` residue carried exactly so
  that a pole cancelling a zero still gives the right finite value. A strictly
  positive total order *is* an exact zero; a negative one means `G` is unbounded
  at the endpoint and is reported `"unknown"`. Terms are then collected into
  hypergeometric similarity classes (`Γ(x+1) = x·Γ(x)` worked off until every
  argument is `a·n + c` with `c ∈ [0,1)`) and `"vanishes"` requires every
  collected coefficient to be the zero element of `Q(n)`. No floating point,
  interval or sampled value can produce it. Symmetrically `"nonzero"` requires a
  *witness* — an integer `n₀` with `b(n₀) ≠ 0` in exact rational arithmetic —
  so sampling that finds only zeros yields `"unknown"`, never `"vanishes"`.

  **The summation range is now part of the call**, because getting it wrong
  silently is the whole bug. `zeilberger(..., limits=(k_lo, k_hi))` takes it,
  each endpoint an `Expr` or an `int`; it defaults to `(0, n)` — the `Σ_{k=0}^n`
  convention the classical identities and the OEIS formula field use — and the
  range actually used is echoed back on `cert.limits`, so the assumption is on
  the record rather than inferred from the summand. The verdict is about that
  range and changes with it: A361712's certificate is `"vanishes"` over
  `k = 0..n` and `"nonzero"` over the `k = 0..n-1` that OEIS truncates to.
  A range the analysis cannot place — endpoints that are not integer-affine in
  `n` — is `"unknown"`, never `"vanishes"`. `boundary_at(k_lo, k_hi)` re-decides
  for another range without re-running the search.

  Getting this right needed one piece the issue report did not mention: when the
  limits move with `n`, `Σ_{k=0}^{n} F(n+i,k)` is **not** `S(n+i)`. For
  `Σ_{k=0}^{n} C(n,k) = 2ⁿ` the telescoped difference alone is `−1`, and it is
  the missing term `C(n+1,n+1) = 1` that cancels it — a verdict computed from the
  endpoints alone reports `"nonzero"` on a textbook identity. The full
  `b(n) = G(n,k_hi+1) − G(n,k_lo) + Σ_i a_i(n)·D_i(n)`, with `D_i` the finitely
  many values of `F` between the range at `n` and the range at `n+i`, is what is
  decided. New core module `holonomic::boundary` (`BoundaryStatus`,
  `boundary_status`, `natural_limits`); `zeilberger()`'s own signature is
  unchanged and the core function returns `Unknown` rather than guessing when no
  limits are supplied.

- **Interval evaluation refused `bessel_j0` / `bessel_j1`, and its Bessel ball
  kernel was not an enclosure.** `evaluate(bessel_j0(x), {x: ArbBall(...)},
  mode="interval")` came back `status="unsupported"` with `E-EVAL-010` even
  though both functions have rigorous ball kernels and `capabilities()` reported
  `numeric_ball: True` for both. The evaluator carried its own hand-written
  match over function names — the third such list in this area — and these two
  had simply never been added to it. It no longer has one: every `Func` node is
  now dispatched through the primitive registry, so the set of functions
  interval evaluation accepts **is** the set the registry advertises a
  `numeric_ball` kernel for, by construction rather than by agreement. Auditing
  the two sets against each other also turned up the gap in the other direction:
  `atanh` had a ball kernel and did *not* advertise it, because the capability
  probe tested ball kernels at `1.0` only and `atanh`'s domain is the open
  interval `(-1, 1)` — it declined the sole probe point and lost a bit it had
  earned, while keeping `numeric_f64` because that probe already tried `0.5`.
  Ball kernels are now probed at the same points as `f64` ones. Wrong-arity
  calls are declined rather than silently bounding the first argument
  (`sin(x, y)` was `sin(x)`), which the generic dispatch made reachable.

  **`ArbBall::bessel_jn` was separately unsound and is rewritten.** It hulled
  `Jₙ(lo)` and `Jₙ(hi)`, which is only an enclosure for a *monotone* function —
  `J₀` on `[-1, 1]` has equal endpoints (`≈ 0.7652`), so the hull collapsed to a
  point that excluded `J₀(0) = 1`, the function's own maximum. It is now a
  midpoint evaluation plus a mean-value bound with `L = 1`, which is rigorous at
  every order: `|Jₙ| ≤ 1` for all real `x`, and `J₀′ = −J₁`,
  `Jₙ′ = (Jₙ₋₁ − Jₙ₊₁)/2`, so `|Jₙ′| ≤ 1`. A randomised sweep samples the true
  function inside 200 random intervals and checks nothing escapes the enclosure.
- **`verified_no_roots` could not prove a root *exists* past an even root
  count.** The `"false"` direction fired only when a sign change was visible at
  the box's own endpoints, so any even number of roots defeated it however
  obvious they were: `x²−2` on `[-2, 0]` was `"false"`, but the same expression
  on `[-2, 2]` — which contains *both* roots — was `"undecided"`, as were
  `[-10, 10]` and `(x²−2)(x²+1)` on `[-2, 2]`. The machinery to settle it was
  already running; one bisection of `[-2, 2]` makes each half answer
  immediately. The `"false"` direction now searches for its two witness points
  by subdividing the box, and the intermediate-value argument is stated over the
  box rather than over an interval: a box is convex, so two points at which the
  expression is *rigorously proven* to have opposite signs certify a root on the
  segment joining them, which stays inside the box. That also lifts the test to
  several variables — `x − y` on `[-1, 1]²` is now `"false"` where it used to be
  `"undecided"`. Continuity, which the argument needs, is exactly what the
  full-box enclosure succeeding already certifies. **Nothing was weakened to buy
  this**: a root that never produces a sign change and never lands on a point the
  search can *prove* is a root — a double root like `(x−1/3)²` on `[0, 2]`, or
  `(x²−1)²` on `[-2, 2]` — still answers `"undecided"`, because no witness pair
  exists and none is invented.
- **`verified_no_roots` could not see a root sitting exactly on a box
  endpoint.** `x` on `[0, 1]` and `x−1` on `[0, 1]` both came back
  `"undecided"`, as did `sin(x)` on `[0, 1]` and `log(x)` on `[1, 2]`. The
  sign-change search above provably cannot settle them — `x` is non-negative
  everywhere on `[0, 1]`, so the negative witness it looks for does not exist to
  be found, however long it runs. But no search is needed: the box is **closed**,
  so a point of it at which the expression is *proven* to be zero is a root in
  the box, full stop. `"false"` is now returned when either of two independent
  proofs is in hand — the existing sign change, or a point whose value is pinned
  to zero. That also settles a root of even multiplicity that lands on a point
  the search visits (`(x−1)²` on `[0, 1]` and on `[0, 2]`, and the multivariate
  `(x−½)² + (y−½)²` on `[0, 1]²`), which a sign change can never reach.
  **The certificate is not weakened**: a value is pinned to zero only by a
  degenerate `[0, 0]` enclosure — an enclosure is a superset of the value, so
  `[0, 0]` forces it — or by substituting the point's exact rational coordinates
  and simplifying to the literal `0`, cross-checked against the enclosure exactly
  as the removable-singularity path already is. An enclosure that merely
  *contains* zero proves nothing and is not used: `exp(x) − 1 + 10⁻⁴⁰` on
  `[0, 1]` has no root at all, but its value at `x = 0` is far below the width of
  any enclosure computable there, and it stays `"undecided"` rather than being
  claimed either way.
- **`verified_sign` could hang on a rational constant with enough digits.**
  `sin(x)·D − N·x ≥ 0` on `[0, 3/2]` took 0.05 s at `N/D = 636/1000`, 0.08 s at
  nine digits, and **over 300 s at twelve** — three extra digits turned
  milliseconds into a hang. The cause was neither exact rational arithmetic nor
  repeated conversion but a **non-terminating loop** in the branch-and-bound:
  a sub-box bisected down to the width floor was pushed back onto the active
  list, immediately re-selected as the smallest key with nothing changed, and
  the loop spun *without ever consuming its subdivision budget* — which is why
  capping `max_subdivisions` at 64 did not help. It only triggered once `tol`
  became unreachable, and `tol` is an **absolute** width: at `D = 10¹²` the
  function is of order `10¹¹`, so the default `1e-9` asks for twenty digits and
  the floor arrives first. Nine digits happened to converge just above the
  floor, twelve just below. Boxes that reach the floor are now retired out of
  the active list, their keys still folded into the final bound, so the search
  always makes progress. Cost is now flat in the size of the constant: 0.33 s at
  three digits through 0.47 s at sixteen.
- **Inequalities that are tight at an endpoint no longer stay `"undecided"`.**
  Cusa–Huygens, Mitrinović–Adamović, Wilker and Huygens were `"true"` on
  `[0.1, 1.5]` but `"undecided"` on `[0.01, 1.5]` and at `x = 0` — precisely the
  point that makes them worth stating. Two independent things were in the way.
  First, `tol` was also the wrong *stopping rule* for a sign question: on
  `[0.01, 1.5]` the true minimum of the Cusa–Huygens form is `1.7·10⁻¹³`, so the
  search met the `1e-9` tolerance and stopped with an enclosure that still
  straddled zero. `verified_sign` now re-runs the search with the sign itself as
  the goal, refining while the running bound straddles zero instead of to an
  absolute width. Second, where the margin genuinely *vanishes* no subdivision
  can ever help, so the box is split: a collar `[a, a+δ]` at the endpoint is
  handled by a truncated Taylor expansion there, and the rest by the usual
  branch-and-bound. The two pieces are closed and share the join point `δ`, so
  their union is the original box with no gap. All four are now `"true"` on
  `[0, 1.5]`, as is Jordan's inequality stated exactly as
  `10¹²·sin x − 636619772368·x ≥ 0`.
  **The remainder is proven, not assumed.** Coefficients `c_k = g⁽ᵏ⁾(a)/k!` are
  accepted as *zero* only when substitution followed by `simplify` lands on the
  literal integer `0` — no numeric enclosure can prove a value is zero, and none
  is asked to — cross-checked against ball arithmetic, and the tail is a
  Lagrange remainder `|R(t)| ≤ t^m·sup|g⁽ᵐ⁾|/m!` whose sup is a rigorous
  enclosure over the whole collar. Analyticity, which Taylor's theorem needs, is
  certified by requiring every derivative `g … g⁽ᵐ⁾` to enclose successfully
  there. With `c_0 … c_{j−1}` proven zero, `g(a+t) ≥ t^j·[c_j − T(δ)]` and
  `t^j ≥ 0` finishes it. **Nothing was traded for the extra reach**: a margin
  that vanishes in the *interior* — `(x − 7/10)²(x + 1)` on `[0, 3/2]` — is
  still `"undecided"`, because the expansion does not apply there. A leading
  coefficient proven *negative* now returns `"false"` rather than `"undecided"`,
  which settles cases no sampling could see: `x³ − x²/1000` is negative only on
  `(0, 1/1000)`, and each of the four inequalities reversed is refuted at the
  same endpoint where the original is certified. A strict `"positive"` query is
  `"false"` where the expression is proven to vanish exactly, so `x² > 0` on
  `[0, 1]` is `"false"` while `x² ≥ 0` is `"true"`.
- **`verified_integral` refused removable singularities.** Taylor-model
  quadrature raised `E-VALIDATED-003` on any sub-interval where the reciprocal's
  enclosure contained zero, which put `∫₀¹ ln(1+x)/x dx = π²/12` out of reach
  even though nothing about that integral is singular — only the expression as
  written is, and the integrand extends continuously to 1 at `x = 0`. An
  integrand that splits as `N(x)/D(x)` with `N(p) = D(p) = 0` at a point `p` of
  the offending sub-interval is now enclosed through **Cauchy's mean value
  theorem** instead: `N(x)/D(x) = N′(ξ)/D′(ξ)` for some `ξ` in the sub-interval,
  so the piece is bounded by an enclosure of `N′/D′`, which is regular. The
  value returned is the integral of the continuous extension.
  `∫₀¹ ln(1+x)/x`, `∫_{-1}^{1} sin(x)/x`, `∫₀¹ (eˣ−1)/x` and
  `∫₀¹ (1−cos x)/x` now come back as enclosures that bracket `π²/12`,
  `2·Si(1)`, `Σ 1/(n·n!)` and `Cin(1)` to better than `10⁻⁹` wide.
  Three guards keep a genuine pole out: the two zeros are established
  **symbolically** (substitute the exact rational `p`, simplify, require a
  literal zero — no numeric enclosure can prove a value is exactly zero, so none
  is asked to), `D′` must be certified non-vanishing on the sub-interval, and
  `N` and `D` must each enclose successfully over it, which is what certifies
  they are analytic and hence that the symbolic derivatives are the real ones.
  `1/x`, `sin(x)/x²` and `(x−p)²/(x−p)³` on boxes containing the pole are all
  still refused.
- **A `verified_integral` refusal now says whether the *integral* fails to exist
  or only the enclosure of the *integrand* does.** An integrable singularity
  that is not removable — `∫₀¹ −ln x dx = 1`, `∫₀¹ (ln x)² dx = 2`,
  `∫₀¹ dx/√(1−x²) = π/2`, `∫₀¹ xˣ dx`, `∫₀¹ ln(x)·ln(1−x) dx` — is still
  refused, because no rigorous bound on the singular tail can be derived from
  the expression alone today. But the `E-VALIDATED-003` message now names the
  location (left endpoint, right endpoint or interior, with the approximate
  coordinate), reports the underlying cause, and states explicitly that an
  integrable singularity still has a finite integral that this routine cannot
  certify. Widening an enclosure to make those cases "pass" would have broken
  the contract that makes the module worth using, so they refuse honestly
  instead.
- **The ODE/DAE guide documented an API that does not exist.**
  [`docs/mdbook/src/ode-dae.md`](docs/mdbook/src/ode-dae.md) showed keyword
  constructors — `ODE(state=…, derivatives=…, independent=…)`,
  `DAE(equations=…, variables=…, independent=…)` — and a one-argument
  `lower_to_first_order(higher_order_ode)`, none of which were ever real: the
  actual calls are `DAE.new(equations, variables, derivatives, time_var)` with
  derivatives as *separate symbols* like `pool.symbol("dx/dt")`, and
  `lower_to_first_order(var, rhs, order, time_var)`. It also printed
  `reduced.differentiated` on the `pantelides` result, which the returned `DAE`
  does not have. Following the page failed on its first line, and it was the
  only documentation there was, because `DAE.new`, `ODE.new`,
  `rosenfeld_groebner` and the `RosenfeldGroebnerResult` accessors had no
  docstrings at all. The page is rewritten around calls that run — every block
  is executed by `tests/test_docs_ode_dae.py` — those docstrings now exist,
  `reduced.index` works as documented (`DAE.index` is exposed), and
  `.differentiated` is gone rather than faked.
- **`zeilberger` no longer refuses a constant base just because it is not
  already a literal.** `(-one)**(n+k)`, with `one = pool.integer(1)`, builds the
  node `Mul(1, -1)` — the pool does no arithmetic at construction — and the
  proper-hypergeometric parser demanded a *literal* rational base, so it raised
  `E-HOLO-001` *"not a proper hypergeometric term: power with symbolic exponent
  needs a rational base, got (1 * -1)"*. `1 * -1` **is** the rational −1, and
  the same summand written `pool.integer(-1)**(n+k)` was decided in 0.4 s. That
  made a spelling look like a capability limit: an autoresearch run recorded the
  OEIS targets A357558 and A357559 (`Σ (−1)^(n+k)·k·C(n,k)·C(n+k,k)²` and its
  `k³` sibling) as outside Alkahest's reach when both in fact yield an order-2
  recurrence. The base is now constant-folded first — `Mul`/`Add`/integer-`Pow`
  towers over integer and rational literals, e.g. `1 * -1`, `2/4`, `(-2)^3`,
  `3 - 4` — under the parser's existing depth bound and a new bit-width budget
  on folded values, so a nest like `((2^32)^32)^32` is refused rather than
  evaluated. What counts as a proper hypergeometric term is otherwise unchanged:
  a genuinely symbolic base is still refused, and a base that folds to `0` still
  refuses as `0` raised to a symbolic power.
- **`SmtResult.verification` now carries the emitted sorts alongside the
  status.** The logic and the sorts decide *which question was solved* — `Int`
  versus `Real` is `QF_NIA` versus `QF_NRA` and hence an integer problem versus
  its real relaxation — but they were reachable only via `SmtResult.logic` and a
  separately-called `supported()`, neither of which a loop recording `status`
  will look at. `verification` gains `sorts` (`{"x": "Int", ...}`) next to
  `logic` and `status` on all three statuses, with a new `SmtResult.sorts`
  property to read it back, so the sorts survive being recorded into a claim
  graph rather than having to be re-derived.
- **`E-SMT-002` now says how to fix a quantified formula.** `solve()` correctly
  refuses explicitly quantified input, but "does there exist x, y, z such that…"
  is the natural way to write a satisfiability question and `Exists` is exported
  at top level, so the refusal was landing on the obvious spelling without
  saying that free variables in a sat query are already implicitly existential.
  When the formula is a prefix of `Exists` over a quantifier-free body, the
  message now leads with dropping the quantifiers and passing the body, and
  names the bound variables; under a `Forall`, where that rewrite is invalid, it
  says so instead. `supported()` gives the same guidance. The quantifiers are
  **not** stripped automatically: `solve()` back-substitutes its model against
  the formula it was given, and rewriting it silently would answer about a
  different expression.
- **A source build without FLINT now fails immediately, with an install hint,
  instead of at link time.** `alkahest-core/build.rs` probes for a linkable
  FLINT (library file in any search directory, `cc -print-file-name`, headers /
  pkg-config, `ldconfig`) and, finding none, stops with the package name for
  every common platform rather than letting the build run to completion and die
  in `rust-lld: error: unable to find library -lflint`. **FLINT remains a hard
  requirement and `cargo:rustc-link-lib=flint` remains unconditional** — that is
  now documented at the emit site, in `flint/mod.rs` and in the `flint3` feature
  comment, because it looks gateable and is not: `src/flint/` is compiled
  unconditionally, `UniPoly` *is* a `FlintPoly`, and factorization, resultants,
  normal forms and `number_theory` call FLINT with no pure-Rust fallback.
  Gating the link behind `cfg(flint3)` was measured: it produces a `cdylib` with
  68 undefined FLINT symbols that links cleanly and then fails at
  `import alkahest` with `undefined symbol: nmod_poly_init` — a worse failure,
  later.
- **New `FLINT_LIB_DIR` / `FLINT_INCLUDE_DIR` build-time overrides.** They add a
  link search path and feed FLINT version detection, so a FLINT built into a
  user-local prefix — no root, no system package — is both linkable and
  correctly identified as FLINT 3. Verified end to end on a host with no system
  FLINT: `FLINT_LIB_DIR=$PREFIX/lib FLINT_INCLUDE_DIR=$PREFIX/include cargo
  build -p alkahest-py --release --features "parallel egraph cranelift groebner"`
  succeeds against a locally built FLINT 3.2.2. `ALKAHEST_SKIP_FLINT_CHECK=1`
  bypasses the probe; `DOCS_RS` skips it automatically, since rustdoc never
  links.
- **`fmpz_mat_struct` layout detection read the wrong header.** The probe looked
  for a `stride` field in `flint/fmpz_mat.h`, but FLINT 3 declares the struct in
  `flint/fmpz_types.h` — so *every* FLINT 3 was reported as the older
  `rows: **fmpz` layout, whatever it actually used. Both fields are
  pointer-sized, so this is not a size mismatch: on a FLINT that uses `stride` it
  would have made `FlintMat` dereference an integer as a pointer. The probe now
  extracts the `fmpz_mat_struct` typedef body from either header and skips a
  header that does not contain the declaration, rather than reading its absence
  as "no stride field". The version fallback (used only when no header is found)
  moved from "3.1+ is stride" to "3.5+ is stride"; FLINT 3.2.2 is confirmed to
  still use `rows`.
- **`E-SOS-002` now says "undecided, not a refutation" in the message itself.**
  The text was accurate and specific but could be read as "not SOS", which for a
  search loop is a wrongly closed branch — and a wrongly closed branch is
  invisible, since nothing downstream ever contradicts it. The message and the
  registered remediation now say to record it as `unknown`, and state that the
  diagonally dominant cone is a *strict subset* of the SOS cone, so refusal is
  compatible with `p` being SOS. All the original specifics (the basis degree,
  `raise basis_degree`, the Motzkin caveat, `decide` as the fallback) are kept.
  The corresponding sections of `positivity.md`, `errors.md` and the agent skill
  spell out the three worlds that produce `E-SOS-002` and note that `E-SOS-003`,
  which carries a witness point, is the only SOS *refutation*.
- **`parallel` now ships in the default PyPI wheel, on every platform.** It was
  the one feature whose absence was *silent*: `numpy_eval_par` and
  `simplify_par` exist in every build and quietly fall back to their sequential
  counterparts, so benchmarking `numpy_eval_par` against `numpy_eval` on a PyPI
  wheel timed one code path twice — and the only build that had threads was a
  Linux-only `+full` wheel from GitHub Releases, so no macOS or Windows user
  could obtain them from any wheel at all. `release-build.yml` now builds the
  default wheel with `egraph groebner cranelift parallel` on Linux
  (manylinux_2_28), macOS arm64 and Windows (MinGW); `+jit` gains `parallel`
  too, so no opt-in wheel is a silent downgrade from the default one. rayon and
  dashmap are pure Rust with no system dependency, and `ci-cross.yml` was
  already building `parallel` on both macos-14 and windows-2022, so this adds
  no toolchain requirement. `ci.yml`'s "PyPI-parity" `maturin develop` and the
  `wheel-smoke` job build it too, so the binary PR CI tests is once again the
  binary users install.

  `capabilities()["features"]["parallel"]` remains the way to check, because
  `parallel` is still not a Cargo *default*: a source build that does not ask
  for it gets the silent single-threaded aliases. `README.md`,
  `getting-started.md`, `codegen.md`, `interop.md`, `features.md` and the agent
  skill say so at each `*_par` entry point.

  **`+full` gains `cranelift`, making it the only wheel with both JIT
  backends.** Moving `parallel` into the default wheel briefly left `+full` with
  a feature set identical to `+jit`'s — `parallel` had been the entire
  distinction, and `groebner`/`egraph` are Cargo defaults, so naming them added
  nothing — which would have shipped two byte-identical wheels under different
  names. `+full` is now a strict superset of the default wheel, which is what
  its name promises, and its smoke test asserts both backends so the variants
  cannot silently converge again.

- **ThreadSanitizer was never given the parallel code to sanitize.** The nightly
  `tsan` shard ran `cargo +nightly test --workspace` with *default* features, so
  rayon and dashmap were not compiled in at all: `ExprPool`'s index was a plain
  `Mutex<HashMap>`, `simplify_par` / `simplify_redex` / `simplify_auto` and
  `CompiledFn::call_batch_par` did not exist, and F4 reduced sequentially. The
  shard was reporting a clean bill for code it had never seen — and per
  `CONTRIBUTING.md` no sanitizer runs `pytest`, so nothing covered the PyO3
  boundary either. It now builds with `--features parallel`.

  **The nightly `asan` shard had the same blind spot and is fixed the same
  way.** AddressSanitizer was checking memory safety on a build with no
  concurrent code compiled into it — precisely the code whose memory safety is
  hardest to get right. It now runs `--features parallel` too. Only the nightly
  shard: Tier 1a's ASan job stays as it is, so the PR critical path does not
  grow.

  New `alkahest-core/tests/parallel_stress.rs` gives that shard something to
  sanitize, from real OS threads rather than Rayon's own pool: concurrent
  interning checked against a single-threaded node-count baseline (a lost
  `DashMap::entry` race would show up as duplicate nodes, i.e. structural
  equality quietly ceasing to imply id equality), lock-free `ExprPool::with`
  reads against a growing `boxcar::Vec`, concurrent `simplify_par` /
  `simplify_redex` on one shared pool, interning *while* a GIL-free
  `simplify_par` walks the same pool, and nested `call_batch_par`. New
  `tests/test_parallel_threadsafety.py` covers the same shapes above the FFI
  boundary, where `ExprPool` is a plain sendable `#[pyclass]` and
  `py_simplify_par` holds a `PyRef` borrow across `Python::allow_threads`.

  Both suites are clean, under TSan and without. Two TSan findings, neither a
  defect in the parallel paths: a reported data race whose two stacks are
  entirely inside `crossbeam_epoch`/`crossbeam_deque` (epoch-based reclamation
  uses an asymmetric `membarrier` barrier that TSan cannot model) — suppressed
  by name in a new `tsan.supp`, deliberately narrow so a genuine race in one of
  our Rayon closures still fails the shard; and a SIGSEGV that is a stack
  overflow, not memory corruption, because `with_stack_segment`'s governor
  refills at 512 KiB against Rayon's default 2 MiB worker stack and TSan's
  instrumented frames do not fit that margin. The shard sets
  `RUST_MIN_STACK=33554432`; it does not reproduce in an uninstrumented build.

- **Documented that a `CompiledFn` is pinned to the thread that compiled it.**
  Surfaced while writing the tests above: `PyCompiledFn` is
  `#[pyclass(unsendable)]`, so using a `compile_expr` result from another
  `threading.Thread` raises
  `pyo3_runtime.PanicException: alkahest::PyCompiledFn is unsendable, but sent
  to another thread`. Behaviour is unchanged and the check is a safety net — it
  fires before anything unsound happens — but it becomes much easier to hit now
  that `parallel` ships by default and the obvious misuse is to compile once and
  fan the handle out over a thread pool. Two details make it sharper than an
  ordinary error, and both are now in `codegen.md` and the agent skill: it has
  nothing to do with `parallel` (plain `numpy_eval` is refused identically), and
  `PanicException` derives from `BaseException`, not `Exception`, so a worker
  wrapped in a bare `except Exception:` will not catch it. Compile per thread;
  `ExprPool` and `Expr` are shareable.

- **`unsafe impl Send for ExprPool` / `unsafe impl Sync for ExprPool` removed.**
  They were unnecessary in both builds — every field already derives the traits
  (`boxcar::Vec<Node>`, `DashMap` under `parallel`, `Mutex<PoolIndex>` without
  it) — and an unconditional `unsafe impl` on a type that qualifies anyway is
  worse than nothing, because it also silences the check *for the future*: add
  an `Rc`, a `Cell` or a raw pointer to `ExprPool`, `Node` or `ExprData` and the
  compiler would have gone on certifying the pool as shareable across Rayon
  workers and across `Python::allow_threads`. Replaced with a `const _` static
  assertion that the three types are `Send + Sync`, which costs nothing at run
  time and now fails the build instead.
- **`numpy_eval` now explains its calling convention instead of describing the
  symptom.** Passing the `Expr` rather than the `CompiledFn` raised
  `AttributeError: 'builtins.Expr' object has no attribute 'n_inputs'`, which
  names an implementation detail; it is now a `TypeError` saying to compile the
  expression first. Passing the arrays packed in one list — `numpy_eval(f, [a,
  b])` — raised `ValueError: expected 2 input array(s), got 1`, true but not
  actionable; the `ValueError` now says that arrays are separate positional
  arguments and to unpack with `numpy_eval(f, *arrays)`, and recognises a 2-D
  array whose first axis matches the arity as the same mistake. `numpy_eval_par`
  validates identically, and against its own name rather than the name of the
  function it falls back to. `CompiledFn.__call__` — which goes the *other* way,
  taking one point as a single sequence — answers `f(1.0, 2.0)` and `f(1.0)`
  with that convention and a pointer to `numpy_eval` for batches. Exception
  types are unchanged.
- **The Cranelift backend was never linted, and had stopped being clean.**
  `ci.yml` ran `cargo clippy -- -D warnings` for default / `egraph` /
  `parallel` / `jit` / `groebner-cuda` but not for `cranelift`, so three
  warnings accumulated in code that ships in the **default PyPI wheel** — and
  `CONTRIBUTING.md` tells contributors to run `--all-features`, which therefore
  failed on a clean checkout. A `cargo clippy (cranelift feature)` step now runs
  alongside the others; it needs no system LLVM, because the backend is pure
  Rust. The three lints are fixed rather than suppressed: `emit_eval_body` took
  nine positional arguments and now takes an `EvalTarget` (root node, inputs,
  pool) and an `InputLayout` enum (`Scalar { ptr }` / `Batch { ptr, point_idx,
  n_points }`) — which also makes the half-specified batch layout a point index
  with no point count, previously two independent `Option`s — unrepresentable;
  and the two `return`s in `compile_jit_only`'s cranelift arm collapse to the
  one-line `return` the other backends' arms already use.
- **`docs/mdbook/src/representations.md` documented a method that does not
  exist, and five outputs that are not what the code prints.** The page showed
  `p.leading_coeff()` on `UniPoly`, which had no leading-coefficient accessor at
  all (it now has one — see *Added*); `sparse_interp_univariate(..., T=3)` and
  `sparse_interp(..., T=2, D=5)`, whose parameters are `term_bound` and
  `degree_bound`; and `p.to_symbolic(pool)`, under the claim that "all
  specialized types can be converted back to a generic `Expr`" — no polynomial
  type exposes a symbolic conversion to Python, so that whole section was
  fiction. Every code block on the page is now executed against the built
  extension, and the `# output` comments match what `print` actually emits
  (`MultiPoly` prints in ascending exponent-vector order, an `ArbBall` prints as
  `midpoint ± radius` rather than as an interval, and
  `sparse_interp_univariate` returns a list of `(coefficient, exponent)` pairs
  rather than a `MultiPolyFp`). The round-trip section now shows the conversion
  that does exist, `GbPoly.to_expr()`, and says plainly that `UniPoly`,
  `MultiPoly` and `RationalFunction` have none.
- **Four `examples/` scripts had rotted against the 3.9.0 API**, two of them
  *silently* — they exited 0 while printing the wrong thing.
  `phase3_polynomials.py` died at line 30 on `ExprPool.pow`, which no longer
  exists on the Python side (use `a ** b`), and then on the trailing `pool`
  argument that `UniPoly` / `MultiPoly` / `RationalFunction.from_symbolic` no
  longer take, and on `UniPoly.pow(3)`. `agent_workflow.py` called `round()` on
  the exact symbolic solutions `solve` now returns, and passes `numeric=True`.
  The two silent ones: `risch_integration.py` claimed a non-elementary refusal
  for `∫√(x³+1)dx`, which is genus 1 and now returns `EllipticF`, so the guard
  moved to a genus-2 integrand that still raises `E-INT-004`; and
  `lean_certificates.py` printed an empty Lean export, because `to_lean` withholds
  a vacuous `e = e := rfl` when the simplifier found no rewrite. All 17 example
  scripts now run end to end. `examples/` is not covered by CI's ruff or pytest,
  which is why this went unnoticed.

### Added

- **Modular / `p`-adic evaluation of holonomic sequences: `ModularRecurrence`,
  `binomial_mod`, `supercongruence_sweep`** (M6). A supercongruence claim such
  as `A(p−1) ≡ 1 (mod p³)` is checked by evaluating a P-recursive sequence at
  one index per prime, over a range of primes. Alkahest had the *proving* half
  of that workload (`zeilberger`, `guess_holonomic`, the boundary verdict) and
  none of the evidence half: there was no way to evaluate a holonomic sequence
  modulo `p^k`, so every sweep ran outside the library in Python big-integer
  arithmetic, where `A(p−1)` is an integer with `Θ(p)` digits that gets touched
  `Θ(p)` times. Now:

  - `ModularRecurrence(coeffs, initial, *, rhs=None, start=0)` runs
    `Σ_i a_i(n)·S(n+i) = b(n)` forward in `Z/p^K` — `O(N)` machine-word
    multiplications and `O(1)` memory, whatever the size of `S(N)` over `Z`.
    `coeffs[i]` is lowest-degree-first, the same convention
    `GuessedRecurrence.coeffs` already uses, so a fitted recurrence is handed
    straight over. `value_mod(n, p, k)`, `values_mod(indices, p, k)` (one
    forward pass for a scattered index set) and `evaluate(...)`, which returns
    a `ModularEvaluation` carrying the precision accounting. Coefficients and
    initial values are arbitrary-precision, and initial values may be
    `fractions.Fraction`, so harmonic-number-style sequences work.
  - `binomial_mod(a, b, p, k)` — the Andrew Granville / Davis–Webb
    factorisation of `n!` into its `p`-free part, which at `k = 1` *is* Lucas'
    theorem. `O(p·k³ + log_p(a)·p·k)`, so `a` far larger than `p` is the
    ordinary case rather than the hard one; the `p`-free factorial is taken by
    a product tree over blocks of `p` consecutive integers rather than term by
    term, which is what keeps `p^k` out of the cost.
  - `supercongruence_sweep(recurrence, primes, k, *, index=…, expect=…)` →
    `CongruenceSweep`, with `holds`, `counterexamples()`, the histogram of
    `v_p(LHS − RHS)` in `valuations()`, and `sharp` — `True` when some prime
    achieves exactly `v_p = k`, i.e. when the modulus in the conjecture is best
    possible and `p^(k+1)` is false.

  **Singular indices are the part that had to be got right.** Stepping forward
  means dividing by the leading coefficient `a_J(n)`, which need not be a unit
  mod `p`: for the Apéry recurrence `a_2(n) = (n+2)³`, the index `n = p−2` is
  exactly the one a sweep crosses when it asks for `A(p)` instead of `A(p−1)`.
  A first pass computes `v_p(a_J(n))` at every step — falling back to exact
  integer arithmetic at the rare index a residue cannot decide — so the total
  precision loss `L` is known *before* the first sequence value exists, and the
  forward pass runs at working precision `k + L`. The division by `p^v` is
  checked, not assumed. Three new codes, and no path that returns a residue it
  cannot justify: `E-HOLO-006` (the modulus is not a prime power the
  machine-word backend supports), `E-HOLO-007` (a step does not determine its
  next term as a `p`-adic integer — the leading coefficient vanishes
  identically there, or the sequence leaves `Z_p`, as `H_p = H_{p−1} + 1/p`
  does), `E-HOLO-008` (`k + L` needs a modulus past `2^62`). The last is a
  real limit rather than a formality: reaching `A(199)` at `p = 5` crosses 40
  singular steps costing three digits each, and 120 digits of `5` is past any
  64-bit modulus, so that call refuses instead of answering.

  Measured on Apéry `A(p−1) mod p⁴` for the 237 primes below 1500: 95 ms,
  against 632 ms for iterating the recurrence exactly and reducing, and 3.47 s
  for the incremental-binomial sum the research harness uses today. The gap
  widens with the range, both exact routes being quadratic in `p` — at the 428
  primes below 3000 it is 270 ms against 4.7 s and 50 s.

- **`q`-analogue creative telescoping: `experimental.q_zeilberger`** (M4b).
  `q`-hypergeometric sums — Gaussian binomials `[n;k]_q`, `q`-Pochhammer
  symbols — are not proper hypergeometric terms in `(n,k)`, so `zeilberger`
  refused every one of them with `E-HOLO-001` and the whole `q`-literature was
  out of reach. `q_zeilberger` is the same algorithm read in the `q`-shift: with
  `x = qⁿ`, `y = q^k`, `k ↦ k+1` is `y ↦ q·y`, which is an automorphism of
  `Q(q)(x)(y)` exactly as `k ↦ k+1` is one of `Q(n)(k)`, so the Gosper normal
  form, the key equation `A(y)·X(q·y) − B(y/q)·X(y) = C(y)·N(y)` and the
  coefficient-comparison linear system all carry over. The bottom two levels of
  the coefficient tower are the *existing* `qfield` code re-read (`Q(v)` and
  `Q(v)(w)` naming their variables `q` and `x`), so only the third level, `y`,
  is new arithmetic.

  The discipline is unchanged: the pair `(a_i, R)` is substituted back and
  checked as an exact identity in `Q(q)(qⁿ)(q^k)` before it is returned, and a
  candidate that fails is discarded rather than returned with a caveat.
  Verified end to end on `Σ_k [n;k]_q²·q^{k²} = [2n;n]_q` (the `q`-Vandermonde
  convolution at `m = r = n`, i.e. the `q`-analogue of `Σ_k C(n,k)² = C(2n,n)`),
  which comes out as the order-1 relation
  `(1 − q^{n+1})·S(n+1) = (1 + q^{n+1})(1 − q^{2n+1})·S(n)` in ~0.2 s; also on
  the Galois numbers `Σ_k [n;k]_q` (order 2) and the alternating
  `Σ_k (−1)^k q^{k(k−1)/2}[n;k]_q = 0`.

  `QZeilbergerCertificate.sum_term(n0)` returns `S(n0)` as an exact polynomial
  in `q`, computed from the *definition* of the `q`-Pochhammer symbol and never
  through the shift quotients the search used. That is what the tests check the
  returned recurrence against, and it is the check that matters: the A279013
  failure this release's boundary work came from was a certificate that
  re-verified perfectly while implying a false recurrence for the sum, and only
  an independent look at the actual terms catches that.

  **The boundary verdict is two-valued here — `"vanishes"` or `"unknown"`, with
  no `"nonzero"` arm — and the sum it is about is `S(n) = Σ_{k ∈ Z} F(n,k)`.**
  Fixing the range at all of `Z` is what makes the proof short: the range does
  not move with `n`, so the `D_i` correction terms the classical
  `boundary_status` needs do not arise, and `"vanishes"` follows from two
  structural facts about the summand alone — that it vanishes outside an affine
  window in `k`, and that it is finite at every integer `k`. Both are decided by
  reading off when a `q`-Pochhammer factor is `1 − q⁰` (exactly zero) or has one
  inside the reciprocal product a negative length denotes (exactly infinite),
  which is a linear condition on `(n,k)` plus a divisibility, settled by
  Fourier–Motzkin over the rationals; rational-empty implies integer-empty, so a
  region proved empty is a proof and not a sample. What it does *not* do is
  evaluate the certificate at an endpoint, and that is the load-bearing part:
  `R` really does have poles at integer `k` — on the `q`-Vandermonde summand a
  double pole exactly where the summand has a double zero, making `G(n,n+1)` a
  finite *non-zero* limit of `0·∞` — so a proof that multiplied the two values
  there would be wrong. Instead it takes one `k` past both the window and the
  finitely many poles, where `G = R·0 = 0` unambiguously, and inducts downwards
  through `G(n,k) = G(n,k+1) − Σ_i a_i(qⁿ)·F(n+i,k)`, whose right-hand side the
  support analysis has already shown finite everywhere. The window is reported on
  `.support` so the caller knows which finite sum the verdict is about
  (`("0", "n")` for the `q`-Vandermonde summand). An inhomogeneous `b(n)` is
  *not* computed: it needs endpoint values of `G` that are not rational in `qⁿ`,
  so a summand whose support cannot be bounded gets `"unknown"` and no claim
  about its sum at all — `1/(q;q)_{n−k}` telescopes perfectly well and has no
  `Z`-sum, and the verdict says exactly that.

  **`q` is treated as transcendental**, and every verdict says so in
  `side_conditions`. These are identities in `Q(q)`; specialising `q` to a root
  of unity — which is what `q`-supercongruence work does — is a separate step
  with its own hypotheses that this engine does not take.

  Refusals are coded and disjoint from the classical engine's, so a caller can
  tell which one declined: `E-HOLO-020` outside the class (a bare `n` or `k`, a
  `gamma`, a `sin`), `E-HOLO-021` bounds exhausted, `E-HOLO-022` a candidate
  that failed verification, `E-HOLO-023` a malformed call, and `E-HOLO-024` for
  an input in the *shape* of the class whose shift quotient is not rational —
  the canonical case being `(q^k; q²)_n` under `k ↦ k+1`, where the first
  argument moves by `1` and the base `q²` does not divide it, making the
  quotient an infinite product. Like `E-HOLO-020` and unlike `E-HOLO-021`, that
  is a permanent answer about the input.

  New: `alkahest.experimental.q_zeilberger`,
  `alkahest.experimental.QZeilbergerCertificate`, and the two term builders
  `qbinomial(pool, N, K)` and `qpochhammer(pool, u, d, v)`. Experimental, so
  the surface may change; the mathematics it refuses to guess at will not.

- **Root-of-unity specialisation for `q_zeilberger`:
  `QZeilbergerCertificate.specialize_at_root_of_unity`** (M4). `q_zeilberger`
  proves identities in `Q(q)` with `q` transcendental and said, in every
  verdict's `side_conditions`, that specialising `q` to a root of unity — the
  step the `q`-supercongruence literature actually needs — was a separate step
  this engine did not take. It now does, and takes it as a three-valued
  **decision** rather than an assumption, because doing it unconditionally is
  the `q`-analogue of the A279013 failure mode transplanted to a new
  subsystem: a certificate that is perfectly valid in `Q(q)` and would
  silently produce a false statement if evaluated at a point where a
  coefficient or a sum value has a pole.

  The arithmetic underneath (`alkahest_core::holonomic::qzeil::cyclotomic`) is
  exact throughout: `Φ_d(q)`, the `d`-th cyclotomic polynomial, is built from
  `q^d − 1 = ∏_{e∣d} Φ_e(q)` by exact polynomial division over `Q`; `Q(ζ_d) =
  Q[q]/(Φ_d(q))` is the residue field, with inversion by the extended
  Euclidean algorithm (total on non-zero elements because `Φ_d` is irreducible
  over `Q`); and "does `p` vanish at `ζ_d`" is decided as "does `Φ_d` divide
  `p`" — a divisibility question over `Q`, never a floating-point evaluation.
  The same machinery gives the exact `Φ_d`-adic valuation of any element of
  `Q(q)`, which is the `q`-supercongruence statement in its precise form:
  `v ≥ r` is exactly `Φ_d(q)^r ∣ S(n)`.

  `spec = cert.specialize_at_root_of_unity(d, n)` returns a
  `QRootOfUnitySpecialization` whose `status` is `"specializes"` (every
  coefficient and every sum value proved to have non-negative `Φ_d`-adic
  valuation, so the specialisation map is defined on all of them, and the
  specialised identity was re-checked as an exact statement in `Q(ζ_d)` before
  being returned), `"obstructed"` (a pole at `ζ_d` was **exhibited** — no
  specialised value is offered, and this is a proof the route is blocked, not
  that the specialised identity is false), or `"unknown"` (the generic
  boundary verdict was already `"unknown"`). Three further conditions are
  reported on a `"specializes"` verdict rather than folded into it:
  `is_vacuous` (every coefficient died — always true at `d = 1`, the `q → 1`
  limit — so the recurrence is `0 = 0`, still a theorem but an empty one),
  `leading_coefficient_survives` (`False` means the specialised recurrence no
  longer determines the last value from the earlier ones), and
  `support_shrinks` / `effective_support` (the `q`-Lucas phenomenon —
  `[2;1]_q = 1 + q` is non-zero in `Q(q)` and zero at `ζ_2`, so the surviving
  window at a root of unity can be a strict subset of the generic one, though
  it can never grow, since a ring homomorphism sends the generic zero
  summands to zero).

  Verified end to end on `Σ_k [n;k]_q²·q^{k²} = [2n;n]_q` specialised at every
  `ζ_d` for `d = 1..6`: the returned value checked against a Gaussian binomial
  built independently by the Pascal recurrence directly in `Q(ζ_d)`, against
  the closed form `[2n;n]_{ζ_d}` predicted by the `q`-Lucas theorem from the
  base-`d` digits of `2n` and `n`, and — at the Python surface — against the
  same sum recomputed in floating-point `complex` arithmetic at the actual
  numeric root of unity `e^{2πi/d}`, evaluated by walking the returned
  expression's node tree by hand rather than through anything under test. Two
  refusals are exercised concretely rather than merely asserted: a genuine
  pole at `ζ_3` (obstructed at every `n` tested, never silently specialised)
  and the support-shrinks case above.

  New: `QZeilbergerCertificate.specialize_at_root_of_unity`,
  `alkahest.experimental.QRootOfUnitySpecialization`,
  `alkahest.experimental.cyclotomic_polynomial`. Experimental.

- **Double-sum creative telescoping: `experimental.telescope2d`** (M4).
  Everything `zeilberger`/`q_zeilberger` reach is a single sum over one
  index; `telescope2d` is the Apagodu–Zeilberger generalization to **two**
  bound indices — `F(n,j,k)` proper hypergeometric in each of `n`, `j`, `k`
  — finding `a_0(n), …, a_J(n)` (not all zero) and *two* rational
  certificates `c_1, c_2 ∈ Q(n,j,k)` with
  `Σ_i a_i(n)·F(n+i,j,k) = Δ_j(c_1·F) + Δ_k(c_2·F)`, re-checked as an exact
  identity in `Q(n,j,k)` — never trusted from the search — before being
  returned. This is a genuinely scoped-down piece of a much larger problem
  (full Wegschaider reduction is arbitrary rational summands over arbitrarily
  many indices), not a claim to have closed it: see the honest-limitations
  list in `alkahest_cas::holonomic::telescoping2d`'s module docs.

  **Method.** There is no standard 2-D analogue of Gosper's normal form for a
  general proper hypergeometric `F(n,j,k)`, so unlike the single-sum engine
  this is undetermined coefficients, not a normal-form reduction: the
  certificate ansatz `c_1 = P_1(n,j,k)/E_1`, `c_2 = P_2(n,j,k)/E_2` uses a
  *fixed*, search-independent denominator built from `F`'s own shift-ratio
  denominators — and, critically, **not just the raw denominator of the
  ratio being telescoped in that direction**. A certificate built from a
  product of two single-sum WZ pairs needs a factor from the *other*
  direction's `n`-shift ratio too (`c_1 ∝ R_A(n,j)·B(n+1,k)/B(n,k)` for
  `F = A(n,j)·B(n,k)`), which is why `E_1 := D_j·∏_i D_{n,i}` and
  symmetrically for `E_2`. Clearing that denominator turns the identity into
  one linear system over `Q`, solved by plain Gaussian elimination (nullspace,
  not a fixed right-hand side, since the leading `a_J(n)` is not normalized to
  a constant up front) — reusing nothing from `qfield`'s `Q(n)[k]` tower,
  because a fixed-denominator linear ansatz never needs that tower's gcd
  machinery; a new, deliberately simpler sparse `Q[n,j,k]` polynomial ring
  covers everything the search needs (`holonomic::telescoping2d::poly`).

  **The boundary is four strip sums, not four corner evaluations** — the
  single most important thing to get right at two dimensions, and the
  subject of its own module (`holonomic::telescoping2d::boundary`) kept
  strictly apart from the search so a returned certificate is checkable
  without reference to how it was found. Telescoping
  `Σ_j Σ_k [Δ_j G_1 + Δ_k G_2]` over a rectangle gives
  `Σ_k[G_1(j_hi+1,k) − G_1(j_lo,k)] + Σ_j[G_2(j,k_hi+1) − G_2(j,k_lo)]` —
  four *one-dimensional sums* along the rectangle's edges. Summing a strip in
  closed form is in general its own creative-telescoping problem, so this
  version proves the **sufficient** (not necessary) condition that each strip
  is identically the zero function of its remaining free variable — via the
  same `1/Γ` non-positive-integer-argument identity the single-sum engine's
  order counting uses, checked either on `F`'s own gamma factors (the natural
  boundary) or on the certificate's own numerator (the classical-WZ-style
  case, e.g. a certificate `∝ k` that is zero at `k = 0` even though `F`
  itself is not). It is honestly conservative: only **constant** (not
  `n`-dependent) rectangles are supported — the `n`-dependent-range case
  needs the same `D_i(n)` correction-term bookkeeping
  `holonomic::boundary`'s `b(n)` formula has, doubled for two independently
  moving bounds, and this version does not implement it — and an unresolved
  strip is reported `"unknown"`, never guessed. `BoundaryStatus2d::Nonzero`
  exists in the type for parity with the single-sum engine's three-valued
  discipline but is not yet produced by any code path: an inhomogeneous
  boundary term needs closed-form strip summation this version does not
  attempt.

  Verified on `S(n) = Σ_j Σ_k C(n,j)·C(j,k)`, a genuinely **non-separable**
  double sum (`C(j,k)` couples to the outer sum's own index `j`, so it is not
  a product of two independent single-index sums) with a known closed form,
  `S(n) = 3ⁿ` by the binomial theorem — checked by direct exact summation
  (`rug::Rational`, not floats) against the returned order-1 recurrence, and
  again on `2ⁿ·C(10,j)·C(j,k)` (the same non-separable coupling with the `n`
  dependence factored out, so the boundary genuinely is `n`-independent and
  `Vanishes` is provided rather than refused) — plus the separable-product
  fallback case (`C(n,j)·C(n,k)`, `S(n) = 4ⁿ`) the task's own scoping note
  flags as a weaker test, kept in the suite explicitly labeled as such since
  it does not exercise the corner-term logic.

  New: `alkahest.experimental.telescope2d`,
  `alkahest.experimental.Telescoping2dCertificate`. Experimental — the ansatz
  degree budgets, the fixed-denominator choice and the constant-rectangle
  boundary restriction are all real, stated limitations of this first cut,
  not polish left for later. Refusals: `E-HOLO-040` outside the proper
  hypergeometric class, `E-HOLO-041` when the bounded search is exhausted,
  `E-HOLO-042` for a malformed call.

- **Validated bounds reach five more functions: `asinh`, `acosh`, `atanh`,
  `erf`, `erfc`.** `bound_on_box` — and everything built on it,
  `verified_sign`, `verified_no_roots`, `verified_integral` — covered exactly
  13 elementary primitives and refused these five with `E-VALIDATED-001` even
  though `numeric_ball` reported `True` for all of them. The gap was not an
  oversight about ball arithmetic; a Taylor model needs a per-function rule
  with a *rigorous* remainder, and there was none. There is now, and each one
  is a proof rather than an estimate:

  - `asinh` uses `aₖ = b_{k-1}/k` from the exact point recurrence for
    `(1+x²)^{-1/2}` and the remainder bound
    `1/[(p+1)(1+ξ²)^{(p+1)/2}]`, which follows from
    `Σⱼ C(n,j)AⱼA_{n-j} = n!` for `Aⱼ = (2j-1)!!/2ʲ` — the same bound `atan`
    already used, arrived at the same way.
  - `acosh` uses the matching recurrence for `(x²-1)^{-1/2}`. Every term of
    its Leibniz expansion carries the sign `(-1)ⁿ`, so no cancellation is
    available to exploit and `|v⁽ⁿ⁾(x)| ≤ n!(x-1)^{-(n+1)}`, decreasing in
    `x`; the supremum therefore sits at the low end of the enclosure.
  - `atanh` expands `(1-x²)^{-1} = ½[(1-x)^{-1} + (1+x)^{-1}]` in closed
    form and bounds the two poles' contributions separately, each at its own
    maximising endpoint.
  - `erf` expands the Gaussian by its Hermite recurrence and bounds
    `|(e^{-z²})⁽ⁿ⁾|/n!` by a Cauchy estimate on a circle of radius `r`,
    using the *exact* minimum of `Re(z²)` on that circle rather than a
    triangle-inequality proxy. The bound holds for **every** `r > 0`, so the
    `r` the rule picks is a tightness choice that cannot affect soundness.
    `erfc` is `1 - erf`.

  All three inverse hyperbolics carry their branch's domain and refuse
  outside it with `E-VALIDATED-003` rather than returning a number: `acosh`
  needs the whole enclosure strictly above `1`, `atanh` needs it strictly
  inside `(-1, 1)` — **both** ends, so `[0.5, 2]` is refused rather than
  bounded off-domain — and a box that merely *touches* a boundary is refused
  too, because the derivative is unbounded there. `asinh`, `erf` and `erfc`
  are entire and never refuse on domain grounds; `asinh` in particular is
  expanded directly rather than through `log(x + √(1+x²))`, which loses the
  argument to cancellation for `x ≪ 0`.

  Each rule also computes a second, independent bound on the same remainder —
  a geometric bound on the series tail inside the disc of convergence — and
  keeps whichever is smaller. The Lagrange form takes the supremum of
  `|f⁽ᵖ⁺¹⁾|` over the *whole* argument enclosure, which near a singularity
  dwarfs the coefficients at the centre: on `atanh` over `[-0.5, 0.5]` at
  order 10 it gives `9.1e-2` where the tail bound gives `8.9e-5`.

  `capabilities()["primitives"][i]["taylor_model"]` and `bounds_supported`
  pick all five up with no edit — both are derived by running the evaluator,
  which is what that design was for.

- **…and five more after them: `bessel_j0`, `bessel_j1`, `digamma`, `gamma`,
  `lambert_w`.** That completes the M7 list and takes validated bounds from 13
  primitives to **23**. The two Bessel functions are the ones worth reading
  about: they *oscillate*, which is the property that made 3.8's ball kernel
  for them unsound (it hulled the two endpoint values, so on `[-1, 1]` it
  excluded `J₀(0) = 1`, the function's own maximum). Nothing in the rules
  below assumes monotonicity of anything.

  - `bessel_j0` / `bessel_j1` get both halves from one identity. Iterating
    `2Jν′ = J_{ν−1} − J_{ν+1}` by Pascal's rule gives
    `Jν⁽ⁿ⁾ = 2⁻ⁿ Σⱼ (−1)ʲ C(n,j) J_{ν−n+2j}`, which is the coefficient
    formula; feeding `|J_m(x)| ≤ 1` (from `J_m(x) = (1/π)∫₀^π cos(mθ − x sin θ)
    dθ`) into the same line collapses the binomials against the `2⁻ⁿ` and
    yields `|Jν⁽ⁿ⁾| ≤ 1` for **every** order, which is exactly the remainder
    `sin` and `cos` already use. Entire, so no box refuses on domain grounds.
    A Cauchy estimate against the entire-function growth
    (`|Jν(z)| ≤ |z/2|^ν e^{|Im z|}/ν!`) is available and was tried; minimised
    over the circle radius it is a factor `√(2πn)` *worse*, so it is not used.
  - `digamma` expands with `aₖ = (−1)^{k+1} ζ(k+1, m₀)`, the Hurwitz zeta
    being what `ψ`'s Taylor coefficients literally are. The remainder is
    `ζ(p+2, ξ)`, which every term of shows is decreasing in `ξ`, so its
    supremum is at the low end of the enclosure, where
    `ζ(s, L) ≤ L^{-s} + L^{1-s}/(s−1)` by comparing the sum to its integral.
  - `gamma` had **no ball arithmetic at all** before this release, let alone a
    Taylor rule. Coefficients come from `Γ′ = ψΓ` as the convolution
    `c_{n+1} = (1/(n+1)) Σⱼ dⱼ c_{n−j}` over the same `ψ` coefficients. The
    remainder is Cauchy's estimate, closed by two facts from the Euler
    integral: `|Γ(u+iv)| ≤ Γ(u)` for `u > 0`, and `Γ″ > 0` on `(0, ∞)` so `Γ`
    is convex and its maximum over an interval is at an endpoint. Every circle
    radius gives a valid bound, so the candidates the rule tries are a
    tightness choice only.
  - `lambert_w` uses the classical closed form
    `W₀⁽ⁿ⁾ = e^{−nw} pₙ(w)/(1+w)^{2n−1}` with `pₙ` carried as exact integers,
    and bounds each of the three factors by its own proved monotonicity in
    `w`, over a panelled split of the `w` range so the three maxima are not
    all taken at the same end.

  `Γ` and `ψ` refuse anything reaching `0` with `E-VALIDATED-003` — the strips
  between the negative poles are analytic but are not covered, since both the
  coefficients and the remainder are written for the positive axis. `W₀`
  refuses at and left of `−1/e`, where it has a square-root branch point and
  every derivative is unbounded; that guard is not a comparison against a
  rounded `−1/e` but the failure of the certified bracket itself.

  The Hurwitz zeta underneath `ψ` and `Γ` is Euler–Maclaurin after 100 exact
  terms, with exact rational Bernoulli numbers from their own recurrence and a
  remainder of twice the first omitted term — the factor 2 covering both
  standard forms of the Euler–Maclaurin remainder so the bound does not depend
  on which convention is quoted. It is cross-checked in tests against MPFR's
  Riemann `ζ(s)` at `a = 1` and against `ζ(s,a) − ζ(s,a+1) = a^{-s}` at
  arbitrary `a`.

  The reported set is now 23 names. `floor` and `ceil` are the only two left
  with ball arithmetic and no Taylor rule, and that is deliberate rather than
  pending: they are not differentiable, so on a box containing an integer no
  Taylor model exists, and on one that does not they are a constant. The four
  elliptic integrals still have neither ball arithmetic nor a Taylor rule.

- **`ArbBall::lambert_w0` was unsound; `ArbBall::gamma` is new.** The Lambert
  kernel hulled two `f64` evaluations — ~10⁻¹⁶ of error — inside a ball whose
  radius it set to `|mid|·2⁻ᵖʳᵉᶜ`, 5·10⁻⁴⁰ at the default 128 bits. On the
  degenerate ball `[1, 1]` that is an enclosure of width 10⁻³⁹ centred
  3·10⁻¹⁷ from `W₀(1)`: an interval that does not contain the value it
  encloses. It is now a Newton *guess* whose bracket is certified afterwards by
  evaluating `g(w) = w·eʷ` in ball arithmetic at both candidate endpoints —
  `g` is strictly increasing on `w ≥ −1`, so `g(v) ≤ x` proves `W₀(x) ≥ v` and
  `g(u) ≥ x` proves `W₀(x) ≤ u`, and how the candidates were produced never
  enters the argument. `ArbBall::gamma` uses convexity for both ends: the
  maximum of a convex function on `[a, b]` is at an endpoint, and each of its
  two tangent lines is a lower bound everywhere on the interval.

- **`guess_holonomic(terms, max_order, max_degree)` — the guessing half of
  *guess then prove*, with the guard that makes a fit mean something.** Alkahest
  shipped the proving half only (`zeilberger`), so the 2026-08-13 autoresearch
  run hand-rolled a fitter out of `Matrix.nullspace` and exact rationals in
  ~30 lines; it recovered Motzkin's recurrence from 21 terms, and every loop
  since has rewritten it. The rewrite is not the problem — the guard is. A
  recurrence of order `J` with degree-`D` coefficients has `U = (J+1)(D+1)`
  unknowns, and a homogeneous system in `U` unknowns has a nonzero solution the
  moment it has fewer than `U` independent equations, *whatever the numbers
  are*. An unguarded fitter therefore returns a recurrence for every input it
  is ever shown, and a hand-rolled one has nothing to say about which of those
  answers is evidence.

  So a candidate is fitted only where the terms **over-determine** it: at least
  `U + min_surplus` equations, `min_surplus` defaulting to `U` itself. The
  result carries `surplus_terms` (equations that were not needed and agreed
  anyway), `equations_used`, `dimension`, `untested_candidates` and
  `confirmed`, plus `evidence()` for logging the lot — `untested_candidates` is
  the minimality caveat, `0` when the returned order really is the smallest that
  fits anywhere in the bounds. Motzkin from 21 terms comes back order 2, degree 1,
  confirmed by 14 surplus equations; the same recurrence from the 7 terms that
  merely *determine* it is refused (`E-HOLO-005`), because that fit exists for
  any seven numbers.

  **`None` means the grid was swept and nothing fit.** If some `(order, degree)`
  candidate had to be skipped for lack of terms, the call refuses with
  `E-HOLO-005` and says how many terms the cheapest skipped candidate needs,
  rather than returning a negative for a question it never asked — 60 primes
  give an honest `None`, 40 primes give a refusal. This is the
  `relation_confidence` discipline for sequences, and it is deliberately the
  same shape: `check_evidence=False` is the escape hatch, in the role
  `check_precision=False` plays on `guess_relation`.

  Exact throughout: Python `int` of any size and `fractions.Fraction`;
  a `float` term is refused, not rounded. `holds_for(more_terms)` re-checks
  exactly against data the fit never saw and `to_exprs(pool, n)` hands the
  coefficient polynomials back to the expression layer, so a guess can be
  compared with a `ZeilbergerCertificate` coefficient for coefficient. Pure
  Python (`alkahest.guess_holonomic`, `alkahest.GuessedRecurrence`), because the
  one mathematical step is `Matrix.nullspace`, which was already in the kernel
  and already exact — everything on top is composition, validation and evidence
  bookkeeping, which is the Python column of `CONTRIBUTING.md` § *Rust vs
  Python* line by line, and the part where being easy to audit matters more than
  being fast.

- **`zeilberger` now says whether its order is minimal, and `minimal=True`
  makes it so.** The certified order is usually the publishable part — the one
  novel result of the 2026-08-13 run (OEIS A359643) was interesting *precisely*
  because the certified order was 4 where OEIS recorded a guessed 5. But nothing
  in 3.9.0 established minimality, and the natural assumption that an ascending
  search gives it for free is **false for this search**: since the iterative
  deepening added earlier in this cycle, the `(order, degree)` grid is visited
  cheapest-estimated-cost-first (weight `3·(J−1) + d`), which is exactly what
  took Dixon, Franel and Apéry from timeout to sub-second. Cheapest-first is not
  order-ascending, so the plan can reach a cheap order-2 probe long before an
  expensive order-1 one, and a returned order 2 does **not** rule out order 1.

  `ZeilbergerCertificate.order_is_minimal` reports it, computed from the probes
  that actually happened rather than from the mode, so it cannot drift away from
  what the search did. `False` means **not established**, never "a lower order
  exists" — a lower-order relation that had been found would have been the one
  returned. It is `True` for free at order 1, and `True` whenever the
  cost-ordered plan happened to spend every lower order first, which it does at
  narrow `max_degree`.

  `zeilberger(..., minimal=True)` walks the grid order-major instead — every
  degree `0..max_degree` at order `J` refused before order `J+1` is tried — so a
  returned order really is the least one reachable within `max_degree`. Same
  bounds, same exact `Q(n)(k)` verification, same certificate; only what was
  ruled out differs. **The default is unchanged and stays fast**, deliberately:
  the price of minimality is the whole hopeless low-order sweep, and it is
  charged against `max_degree`, which is the bound minimality is claimed
  relative to. Measured, `max_order=4`, default → `minimal=True`: Franel
  0.23 s → 0.23 s at `max_degree=6` (the default was already minimal there),
  0.56 s at 8, **9.7 s at 16**; Apéry 0.08 s → 0.11 s at 6, 0.29 s at 8,
  **13.1 s at 16**. The default column is flat in `max_degree` and the
  `minimal=True` column is not, which is the whole trade in one line. The sweep
  grows like `3^d`, so the useful move
  is to claim minimality against the smallest `max_degree` you are willing to
  state rather than against the default. In Rust the same thing is
  `holonomic::zeilberger_search` with `OrderSearch::MinimalOrder`, returning a
  `ZeilbergerSearchReport`; `zeilberger()` keeps its signature and its
  cost-ordered behaviour exactly.

- **A certified recurrence now answers the next question: how fast does the
  sequence grow?** `alkahest.experimental.asymptotics_from_recurrence(rec, n,
  terms=…)` takes what `zeilberger` or `guess_holonomic` just produced — or a
  bare list of coefficient polynomials — and returns
  `RecurrenceAsymptotics`. Until now the asymptotics family
  (`asymptotic_expand`, `euler_maclaurin`, `coefficient_asymptotics`) and the
  holonomic subsystem did not compose at all, so every loop that certified a
  recurrence stopped one step short of the growth law the recurrence already
  determines.

  **The derived half and the fitted half are separate fields, because the
  constant is the part that is usually hard and is exactly the part a loop is
  tempted to overclaim.** Poincaré–Perron gives the growth rate `ρ` (a root of
  the characteristic polynomial `χ(t) = Σᵢ [n^D]pᵢ · tⁱ`) and the polynomial
  exponent `α = −χ₁(ρ)/(ρ·χ'(ρ))` in `u(n) ~ C·ρⁿ·n^α`; both are functions of
  the coefficient polynomials and nothing else, and both come out **exact** as
  `growth_rate_exact` / `polynomial_exponent_exact` when the root is rational.
  The connection constant `C` does *not* follow from the recurrence — it is
  determined by the initial conditions — so it is extrapolated numerically from
  the exact terms, exposed only as `connection_constant`, and carries
  `connection_constant_converged` and `connection_constant_drift` from a second,
  independent extrapolation over a smaller index range. `evidence()` returns the
  two halves under separate `derived` / `fitted` keys, and `report()` is the
  family's usual `AsymptoticReport`, whose `rigor` here is always
  `numerically_consistent` and whose hypotheses name the fitted constant as
  `assumed`. This is the discipline `euler_maclaurin` uses for the `γ` in
  `H_n ~ log n + γ + …`, applied to the same kind of quantity.

  Measured against sequences whose asymptotics are known independently: the
  fitted constant reproduces `1/√5` for Fibonacci to `1.8e-14` (the control —
  it is the one case where `C` is derivable), `1/√π` for the central binomial
  coefficients to `3.6e-11` and for Catalan to `8.4e-9`, `3√3/(2√π)` for
  Motzkin to `7.9e-8`, and `(1+√2)²/(2^{9/4}π^{3/2})` for Apéry to `5.5e-11`.
  For **OEIS A359643** — the one novel result of the 2026-08-13 run — the
  order-4 recurrence gives `ρ = 283/27` and `α = −1/2` exactly and fits
  `C = √(283/3)/(2^{7/2}√π)` to `1.7e-10`, i.e. the whole of the entry's
  `a(n) ~ 283^(n+1/2)/(2^(7/2)·√(πn)·3^(3n+1/2))`.

  **Poincaré–Perron's hypotheses are stated and checked, not assumed.** Each way
  they fail gets its own `verdict` and no growth rate at all, rather than one of
  the roots reported as though it had won: `equal_modulus_roots` (`u(n+2) =
  4u(n)` has roots `±2` and its solutions oscillate), `repeated_dominant_root`
  (`χ'(ρ) = 0`, so the exponent formula does not apply), and
  `degenerate_leading_coefficient` (`deg χ < J`, a root at infinity, outside the
  theorem). Root multiplicity is **exact** — it comes from the squarefree
  decomposition of `χ` over `ℚ`, not from clustering numeric roots, which
  matters because A359643's `χ = (t−1)³·(27t−283)` has a triple root that is not
  the dominant one and a tolerance-based test would refuse the case. A leading
  coefficient vanishing at finitely many `n` is a reported side condition
  (`singular_indices`), not a refusal. And because Poincaré's conclusion is only
  that `u(n+1)/u(n)` tends to *some* root, a sequence whose dominant component
  is zero — the constant solution of `u(n+2) = 3u(n+1) − 2u(n)`, say — is caught
  and reported as `follows_dominant_root == False` instead of being handed the
  generic solution's growth rate. The sequence is run forward in exact rational
  arithmetic for that reason: `f64` iteration is attracted to the dominant
  solution and would manufacture the component the check exists to look for.

  In Rust: `holonomic::asymptotics_from_recurrence`, with
  `CharacteristicAnalysis`, `ConnectionConstant` and `PerronVerdict`.

- **Validated-bounds coverage is queryable: `bounds_supported(expr)` and a
  `taylor_model` bit in `capabilities()["primitives"]`.** The only
  per-function coverage flag the agent contract exposed was `numeric_ball`,
  and it is not the flag that governs `bound_on_box` / `verified_integral` /
  `verified_no_roots` / `verified_sign`. Ball arithmetic is *pointwise*; a
  Taylor model needs a rule with a rigorous Lagrange remainder, and eleven
  primitives have the first without the second — `erf`, `erfc`, `bessel_j0`,
  `bessel_j1`, `digamma`, `lambert_w`, `acosh`, `asinh`, `atanh`, `floor`,
  `ceil`. So
  `numeric_ball` said `True` for `bessel_j0` and every bound over a box died
  on `E-VALIDATED-001`. The boundary was enforced correctly and could not be
  found ahead of time, which is how a planning loop loses a whole designed
  workload (Turán-type inequalities for Bessel functions, in the 2026-08-13
  autoresearch run) to a route it could have ruled out for free.

  `taylor_model` reports it per primitive. When the flag was added that was
  `True` for the elementary fragment (`exp`, `log`, `sqrt`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `abs`) and `False`
  for every special function; by the time 3.9.0 shipped the two rounds of
  Taylor-model rules above had moved ten of those names across, leaving only
  `floor` and `ceil` with ball arithmetic and no rule. The flag needed no edit
  for either round, which is the point of deriving it.
  `ak.bounds_supported(expr)` asks for a whole expression, without
  running the bound: it is truthy when nothing in the expression will be
  refused as unsupported, and carries `.blocker` (the evaluator's own
  description of the first construct it has no rule for) and `.functions`
  (every blocking function, so a substitution can be planned in one round
  rather than found one at a time).

  **Neither is a maintained list.** Both are derived by running the real
  Taylor evaluator on a probe expression and asking whether it refuses with
  `E-VALIDATED-001` — a second hand-written table is how `numeric_ball` came
  to be read as coverage in the first place, and would have been a worse
  outcome than no flag at all. `tests/test_taylor_model_coverage.py`
  re-derives the bit the only other way there is, by calling `bound_on_box`
  on every registered primitive, and fails if the two ever disagree.

  `numeric_ball` itself is *accurate* and stays as it is: those eleven
  primitives really did have ball arithmetic. It answers a different
  question, and now says so next to a flag that answers this one. A `True`
  from either means "not `E-VALIDATED-001`" — a covered function can still be
  refused on a particular box for a domain violation (`E-VALIDATED-003`) or a
  non-finite enclosure (`-004`), which no box-free predicate can rule out.

  This is deliberately *not* folded into `certifiable`, which asks whether an
  operation emits a **Lean** certificate and answers from the certificate
  ledger. A rigorous enclosure is not a Lean proof term and the validated
  subsystem has no ledger rows; one predicate returning `True` for two kinds
  of evidence would be a worse contract than two predicates.

  New in Rust: `alkahest_cas::{taylor_model_refusal, taylor_model_blockers,
  taylor_model_supports, taylor_model_supports_call}` and
  `Capabilities::TAYLOR_MODEL` (also a `taylor_model` column in
  `CoverageReport::to_markdown`). `capabilities()["contract_version"]` stays
  `3`: the row gained a key and lost none, which is the same additive rule
  the `__all__` freeze check applies.

- **Gröbner results can be read back — `GbPoly.to_expr`, iteration over a
  `GroebnerBasis`, and `expr_to_gbpoly`.** Everything that returned a basis
  returned a handle nobody could open. `GbPoly` exposed only `is_zero` and
  `n_vars`; `GroebnerBasis` exposed only its own constructors plus `reduce` and
  `contains`, and `reduce` took a `GbPoly` that no exported function could
  build — `expr_to_gbpoly`, named in `compute_raw`'s own docstring, was never
  registered on the module. So `rosenfeld_groebner(...).final_basis()`,
  `triangularize(...)`, `primary_decomposition(...)` and a parametric `solve`
  all handed back objects whose only readable property was how many generators
  they had. Differential elimination was write-only: the input–output equations
  it computes could not be looked at, which is the whole of structural
  identifiability.

  Now: `alkahest.expr_to_gbpoly(expr, vars)` converts in, `GbPoly.to_expr()`
  converts back out, and `GbPoly.terms()` gives `(exponent tuple, exact
  int/Fraction)` pairs. A `GroebnerBasis` is a sequence — `len()`, indexing,
  iteration — with `polynomials()`, `to_exprs()`, `variables()` and an `order`
  property. `reduce()` now accepts an `Expr` as well as a `GbPoly`, so the
  membership and reduction API is reachable from expressions alone.
  `GroebnerBasis.eliminate(vars)` is bound too — the mdbook and Sphinx pages
  had documented it for releases, but it existed only in Rust, so the
  implicitization example on the solving page could not run.

  The part that needed a real fix rather than an accessor is the **variable
  ordering**. A `GbPoly` stores exponent vectors, not names, so a basis without
  its variable list cannot be read at all — and `rosenfeld_groebner` discovered
  its jet variables internally (`t`, `x`, `dx/dt`, `ddx/dt/dt`, …) and threw
  them away. Every object that hands out a `GbPoly` now carries that list:
  `RosenfeldGroebnerResult.variables()`, `GroebnerBasis.variables()`,
  `RegularChain.variables()`, and for a parametric `solve` the solve variables
  *followed by the free parameters*, which is the order the exponent vectors
  were actually built in. Asking for an `Expr` with too few variables named
  raises `ValueError` rather than silently misreading exponent slots.

  New in Rust: `alkahest_cas::gbpoly_to_expr`, `GroebnerBasis::order()`,
  `MonomialOrder::as_str()`, `solver::collect_parameters`, and
  `rosenfeld_groebner_ranked` / `dae_index_reduce_ranked`, which return the
  `DifferentialRanking` alongside the result (the existing entry points are
  unchanged wrappers, so no struct gained a field).

- **`DAE` can be read: `equations()`, `variables()`, `derivatives()`,
  `time_var`, `index`.** It previously exposed `n_equations` and `n_variables`
  and nothing else, so a prolonged system that reported six variables for a
  two-variable input gave no way to find out what any of them were.
  `pantelides` sets `index` on the DAE it returns — the number of
  differentiation rounds — and the equations it appended, plus the higher jets
  (`ddx/dt/dt`, …) they introduced, are now visible in `equations()` and
  `derivatives()`.
- **`UniPoly.leading_coeff`** — the leading (highest-degree) coefficient, `0`
  for the zero polynomial so it pairs with `degree == -1`. A **property**, per
  the accessor convention: it is a single FLINT coefficient read. It is also
  *exact*, returned as a Python `int` of any size, which `coefficients()` is
  not — that one is `i64` and truncates silently, so `3x² + 1` scaled by `2¹⁰⁰`
  reports a leading coefficient of `1` through the list and the true value
  through this accessor. Documented in `representations.md`, which had been
  showing a `leading_coeff()` *method* that never existed.

- **Gröbner bases over the coefficient field `Q(params)`: `GroebnerBasis.compute(polys, vars,
  params=[...])`, `experimental.ParametricGroebnerBasis` / `ParametricGbPoly`**
  (M9). Eliminating states from an ODE model's jet equations needs the rate
  constants to *not* enter the monomial order — computed over `Q[states, Y,
  params]` they are ordinary ring variables and generate S-pairs like any
  other, which is exactly the growth the elimination does not need. With
  `params=[...]` the same Buchberger engine (identical Gebauer–Möller pair
  management from `poly::groebner::pairs`, shared with the ℚ engine so a
  specialisation keeps every leading monomial and the whole pair schedule)
  runs over `Q(params)[vars]` instead: the parameters live in the
  coefficients as elements of `QParam`, a canonical, reduced fraction of
  sparse `ParamPoly`s (FLINT's `fmpz_mpoly_gcd` — a Hensel/Zippel hybrid, not
  a hand-rolled Euclidean gcd, which is exactly the swelling `holonomic::qfield`
  exists to avoid one variable further in) rather than ring monomials.

  Measured on a catenary compartmental model (linear ODE chain, output the
  first compartment, `n+1` derivatives eliminating `n` states down to the
  input–output relation): at `n = 4` states / 7 rate constants, the
  parametric route computes in 0.27s against 4.2s direct (both `Lex`,
  `--release`) — roughly 15× — and returns 5 total basis generators against
  25; at `n = 5` states / 9 rate constants the parametric route finishes in
  6.9s while the direct `Q[states, Y, params]` computation had not finished
  after 240s (>34× and counting). Numbers are wall-clock on one machine, not
  a guaranteed ratio — the point is the qualitative shape (S-pairs among the
  parameters are the cost the direct route pays and the parametric route
  never generates), not the specific multiplier.

  **The result is generic, and says so.** A leading coefficient in
  `Q(params)` can be a non-zero rational function of the parameters and still
  vanish at a particular parameter point, and there the basis the algorithm
  built is not the basis the same algorithm would have built over ℚ at that
  point. Every such assumption is logged as it happens — an inversion
  contributes its numerator and its denominator, an input coefficient
  contributes its denominator — and `conditions()` reports the union,
  factored into irreducible, primitive hypersurfaces so "wrong somewhere on
  this degree-12 surface" reads as a list of conditions rather than one
  opaque polynomial. `specialize(values)` substitutes and refuses with
  `ParamGroebnerError` (`E-PARAMGB-004`) on the locus rather than returning
  something that is not a basis; `is_regular_at` / `vanishing_conditions`
  check first. The set is sufficient, not necessary by construction — on the
  worked linear system `{a·x − y, x + y − 1}`, `a = -1` is a real
  disagreement (the direct basis over ℚ there is the unit ideal, not the
  2-generator triangular basis the generic formula predicts), while `a = 0`
  is flagged (the algorithm inverts `a` to make `a·x − y` monic) but the
  direct computation at `a = 0` agrees with the `a → 0` limit of the generic
  answer exactly — a refusal, not a wrong answer, on the conservative side of
  the report.

  Reads back the same way `GbPoly` does: `ParametricGbPoly.to_expr` /
  `.terms()`, `ParametricGroebnerBasis.to_exprs()`, `.conditions()` as
  `Expr`, and `.specialize(...)` returns an ordinary `GroebnerBasis` whose
  generators are `GbPoly` — the same read path issue #11 was about, so
  nothing here is write-only. `eliminate` has the same `Lex`-with-eliminated-
  variables-first contract as `GroebnerBasis.eliminate`, and refuses to
  eliminate a coefficient-field parameter (there is nothing to eliminate — it
  was never a ring variable) rather than silently ignoring the request.
  Errors: `E-PARAMGB-001` no generators, `E-PARAMGB-002` generators disagree
  on the variable/parameter shape, `E-PARAMGB-003` wrong specialisation
  arity, `E-PARAMGB-004` degenerate point (a result, not a malfunction — see
  above). New: `alkahest.experimental.ParametricGbPoly`,
  `ParametricGroebnerBasis`, `ParamGroebnerError`. Experimental; requires
  `--features groebner`. Tests: Rust `poly::groebner::parametric` and
  `poly::groebner::paramfield` (19 unit tests), Python
  `tests/test_parametric_groebner.py`, including a
  structural-identifiability worked example (two-compartment linear ODE
  model, states eliminated with rate constants in `Q(params)`, the recovered
  input–output relation checked against the model's characteristic
  polynomial `y'' − tr(M)·y' + det(M)·y = 0` at several parameter points).

- **Novelty filtering against OEIS: `experimental.novelty`** (M11). A search
  loop over this library can rediscover a known identity within the hour, and
  nothing before this told the difference between "produced 400 certified
  recurrences" and "produced three that nobody had". `RecurrenceClaim` puts a
  P-recursive relation into normal form — rescaling, sign flips, index shifts,
  a different clearing of denominators and a common polynomial factor are all
  quotiented out, so `(n+1)·u(n+1) − (4n+2)·u(n) = 0` and the same relation
  scaled by −2 and stated about `u(n+7)` hash equal via `claim_hash`, while
  genuinely different recurrences do not collide. `check_novelty(claim,
  sources, terms=…)` checks a claim against one or more sources — `OeisCache`
  offline (file-backed, the committed test fixture format) or `OeisWeb` when
  explicitly opted into (never constructed by default, serves from cache
  first, rate-limited, degrades to `unavailable` rather than raising when the
  network is unreachable) — and returns a `NoveltyVerdict`.

  **`NoveltyVerdict.found` is three-valued**, in the manner of
  `relation_confidence`'s tri-state `credible` and `GuessedRecurrence.confirmed`:
  `True` a source states the claim, `False` the sources searched do not state
  it (not "novel" — "not found in the one place looked"), `None` no source
  could answer. There is no `novel` attribute anywhere in the module, and
  `bool(verdict)` raises rather than reading `True`, because `if
  check_novelty(...):` is the exact sentence this module exists to prevent.
  `NoveltyVerdict.report()` carries the scope of the search — entries
  examined, statements compared, statements a parser could not use — so a
  negative's coverage is visible next to it. `RecurrenceClaim.from_text`
  parses OEIS's own prose formula lines by recursive descent over
  `+ - * / ^ ( )`, `n` and `a(n±k)`, refusing (returning `None`, never
  guessing) anything it does not model — a sum, a generating function, a
  reference to another sequence, an inhomogeneous relation — and every parsed
  line is re-checked against the entry's own data before it can produce a
  match, since a formula line is prose from a third party and the parser is
  the weakest link in the chain. No test in the repository requires the
  network: the offline path runs against `tests/data/oeis_novelty_fixture.json`,
  a cache recorded once from oeis.org (© The OEIS Foundation Inc., CC
  BY-NC-SA 4.0) and committed, exercising the four sequences this project
  already certifies recurrences for (Apéry, Motzkin, Catalan, central
  binomial) plus the session's own novel result (A359643, which OEIS records
  only as an unproved conjecture — the distinction `hedged` exists to keep
  separate from a proof).

- **`sos_decompose` searches the general PSD Gram cone and Reznick
  multipliers, not just diagonal dominance** (M10). `sos_decompose` already
  covered the diagonally dominant subcone (`gram::dsos_search`) — fast, but a
  strict subset of SOS, so it refused things that genuinely are SOS just not
  DSOS. It now falls back, when DSOS fails, to searching the *entire* PSD
  Gram cone (`real::sos::psd::psd_search`) over the same monomial basis, and
  — when even that fails on `p` itself — to a Positivstellensatz-lite
  multiplier search: trying `(x_1²+…+x_n²)^N·p` for `N = 1..4`
  (`MAX_MULTIPLIER_POWER`) and searching the PSD Gram cone of the product, up
  to a monomial-count budget (`MAX_MULTIPLIER_BASIS_LEN`). This is the
  standard route past DSOS's and even plain-SOS's incompleteness: some
  positive-definite forms are not SOS at all (Hilbert 1888), but Reznick's
  theorem guarantees `(Σxᵢ²)^N·p` is SOS for *some* `N` — the search just
  does not know `N` in advance and is honest when it runs out of budget
  before finding it (`multiplier_search_reports_undecided_not_not_sos_when_out_of_budget`):
  `SosError::NoCertificate` (`E-SOS-002`), never a claim that no certificate
  exists or that `p` is not SOS.

  The PSD Gram search underneath (`real::sos::psd`, `real::sos::sdp`,
  `real::sos::linalg` — an exact rational affine-system solver that reports
  free parameters on underdetermined systems, a Jacobi eigendecomposition,
  and a PSD-cone projection) is a floating-point *suggestion* mechanism only:
  it proposes a Gram matrix numerically, rounds it to nearby rationals, and
  the rounded result is re-expanded and compared against the target with
  exact rational arithmetic before anything is returned — a `Some` here is
  always sound regardless of what the numeric search actually converged to,
  the same discipline the rest of this module already applies. The search
  itself anneals a shrinking sequence of eigenvalue floors with several
  random restarts (`psd::FLOOR_SCHEDULE`, `psd::multistart_anneal`) rather
  than a single fixed-floor pass, specifically because the certificates this
  feature exists for are frequently *tight* — the witnessing Gram matrix is
  PSD but singular, sitting exactly on the boundary of the PSD cone rather
  than its interior, which a plain fixed-floor search reliably stalls short
  of.

  **Motzkin and Robinson now both certify — the gap that motivated this
  feature is closed, though not unconditionally.** The first version of this
  search (annealed alternating projection with several random restarts)
  reliably fell short of both classical textbook examples: a diagnostic
  trajectory (`psd::diag::diag_step1_step2_trajectory_and_family_sanity`)
  showed it converging *monotonically* toward Motzkin's boundary certificate
  (minimum eigenvalue running from roughly `−1.6` to roughly `−0.0018` as the
  floor annealed to `0`) without ever closing the last, asymptotically slow
  stretch to exactly `0` — the textbook signature of alternating projection
  stalling at a *tangential* (non-transversal) set intersection, which is
  exactly what a *singular* witnessing Gram matrix (sitting on the PSD cone's
  boundary rather than its interior) produces. The search now also tries
  Douglas–Rachford splitting with over-relaxation
  (`sdp::Family::douglas_rachford_from`) and a facial-reduction step
  (`psd::facial_reduction_search`) — both are standard escapes for exactly
  this stall — and with them, both `(x²+y²)·Motzkin(x,y)` (the affine,
  2-variable case, found via the full `sos_decompose` multiplier search, not
  just a hand-fed pre-multiplied target) and `(x²+y²+z²)·Robinson(x,y,z)`
  (via `psd_search` directly) are found and exactly re-verified —
  `real::sos::tests::motzkin_certifies_via_a_reznick_multiplier` and
  `psd::tests::psd_search_certifies_robinsons_form_with_a_reznick_multiplier`
  check the identities by hand, independent of the search that proposed them.

  **What's still open, now attempted to closure and precisely quantified
  (2026-08-17, round 3):** the homogeneous 3-variable form of Motzkin,
  `(x²+y²+z²)²·(x⁴y²+x²y⁴−3x²y²z²+z⁶)` — multiplier power `N = 2`, not `N = 1`
  (round 2 had already narrowed the target to this specific case and power;
  `N = 1` is not classically expected to work for this *homogeneous ternary*
  form at all, unlike the affine 2-variable case) — still is not found
  (`psd::tests::psd_search_does_not_yet_reach_homogeneous_motzkin_times_sum_of_squares`),
  despite a real attempt rather than a budget skip. Three approaches were
  tried:

  1. **Deep Douglas–Rachford on the raw 165-free-parameter family:** escalating
     from 15,000 to 615,000 cumulative iterations moved the minimum eigenvalue
     from `≈ −1.9·10⁻⁶` to only `≈ −1.7·10⁻⁷` — real progress, but clearly
     *sublinear*, not the finite-step convergence a transversal intersection
     would show.
  2. **Symmetry reduction** (new: `psd::symmetry_reduced_search`,
     `psd::detect_polynomial_symmetry_group`) — added specifically because of
     this case. `q`'s own signed-permutation symmetry (every exponent is even,
     so all eight sign patterns fix it; swapping `x, y` also fixes it — order
     16) restricts the 165-parameter family to a **26**-parameter symmetric
     slice, by construction still reproducing `q` exactly (a `G`-average of the
     original family). Deep DR on the smaller family converges *faster in wall
     time* (roughly 30× per iteration) but at essentially the **same iteration
     count** it does not reach a meaningfully different minimum eigenvalue
     (`≈ −2.4·10⁻⁸` at 2,000,000 iterations) — the bottleneck is the
     intersection's geometry, not raw parameter count.
  3. **Exact algebraic zero-vector restriction**, going further than either
     prior round: Motzkin's zero at `(1,1,1)` forces `Q·z(1,1,1) = 0` for any
     PSD witnessing `Q` (`z(1,1,1)` is literally the all-ones vector on this
     degree-5 basis — no rounding, no numerics). Imposing that exactly on the
     symmetric family collapses it again to **16** parameters. An additional
     tangent-direction candidate (the exact gradient of the basis vector at
     `(1,1,1)`) was also tried and found *inconsistent* with the family —
     confirming the corank contributed by this zero really is 1, not a missed
     higher-corank guess. Deep DR on this 16-parameter family still does not
     close it: 6,000,000 iterations reach only `≈ −1.4·10⁻⁸`, and rational
     rounding fails even with denominators tested up to roughly `10⁹`.

  All three lines of evidence agree: this is now a genuine, quantified
  numerical-hardness finding (an unusually slowly-converging tangential
  intersection), not an under-tuned budget or an unexplored structural avenue.
  `symmetry_reduced_search` ships anyway as a real, general capability — it is
  wired into `psd_search` as a further fallback (only once the direct search
  and facial reduction have both already failed, so it adds no cost to any
  case those already close) and will help *other* targets whose Gram family
  happens to have exploitable symmetry, even though it did not close this one.
  So a boundary-only certificate is still not guaranteed to be found in
  general; `E-SOS-002` still means "not found within this search", never "not
  SOS" or "not non-negative".

  Everything reachable today is exact end to end: `verify()` re-expands
  every returned certificate with exact rational arithmetic, `to_lean()`
  emits a sorry-free Lean sketch, and `PositivityCertificate.multiplier()` is
  populated exactly when the certificate needed one (`None` for a direct SOS
  decomposition — a method rather than a field, since adding a field to this
  already fully-public, exhaustively-constructible struct is a semver break
  regardless of the field's own visibility; see the method's own doc comment).
  No new public API surface — `sos_decompose` and `PositivityCertificate` are
  unchanged in shape; this is entirely a strengthening of what the existing
  search covers before it refuses.

### Performance

- **`zeilberger`'s exact `Q(n)(k)` post-processing no longer swells its own
  coefficients.** With the search fixed (below), what was left was entirely
  after it: on `Σ_k C(n,k)³` the search reached `(order 2, degree 3)` in 0.22 s
  and the run then spent ~29 s normalising the certificate and re-verifying it.
  The cause was `PolyK::gcd` — a textbook Euclidean remainder sequence over the
  *field* `Q(n)`, whose coefficients are rational functions in `n`: every
  division step adds numerator and denominator degrees and no step ever removes
  content, the classic intermediate-expression-swell blowup, and
  `RatK::normalize` ran it on every normalisation. The gcd now leaves the field
  and runs **Brown's subresultant PRS in `Z[n][k]`** (Collins 1967, Brown 1971;
  Knuth TAOCP 2 § 4.6.1), with both cofactors divided out in the same integral
  domain, and `Q[n]` gcds (`rn_mul` / `rn_add` / `rn_inv`, which cancel
  crosswise now rather than reducing the full cross-multiplied product) go
  through the same subresultant sequence over `Z[n]`. At the shipped defaults,
  measured before and after on one machine: `Σ (−1)^k C(n,k)³` **1.6 s →
  0.11 s** (15×), `Σ_k C(n,k)³` **56 s → 0.07 s** (800×),
  `Σ_k C(n,k)²C(n+k,k)²` **16.5 s → 0.05 s** (330×, and still Apéry's
  recurrence coefficient for coefficient). Two OEIS targets that timed out past 300 s at certificate
  degree ≥ 3 are now decided: **A357510** `Σ k·C(n,k)²·C(n+k,k)²` and
  **A357512** `Σ k⁵·C(n,k)²·C(n+k,k)²` both yield a verified order-3 recurrence
  in under a second. This is a change of algorithm, not of contract: a monic
  gcd is unique, so every certificate is the same one as before, and every
  certificate is still checked as an exact `Q(n)(k)` identity before it is
  returned — nothing here is probabilistic and no verification was weakened.
- **`zeilberger`'s `max_order` / `max_degree` are now upper bounds instead of
  starting points.** The search used to sweep certificate degrees
  `d = 0..=max_degree` at order 1 before ever trying order 2, and a single
  degree probe gets ~3× more expensive per degree step (measured on
  `Σ (−1)^k C(n,k)³`: 0.7 ms at `d = 0`, 0.6 s at `d = 7`, 84 s at `d = 12`).
  Every order ≥ 2 identity — Dixon, Franel, Apéry — therefore ran for minutes
  or never at the shipped defaults while being seconds away at `max_degree=4`,
  i.e. **raising the bound made easy inputs slower rather than admitting harder
  ones**. The `(order, degree)` grid is now visited by iterative deepening,
  cheapest estimated probe first (one extra order is priced at three extra
  degrees, which is what the measurements say), and the first *verified*
  relation is returned. `Σ (−1)^k C(n,k)³` at the defaults goes from >400 s
  (killed) to **0.67 s**; `Σ_k C(n,k)³` from >400 s (killed) to ~31 s. Nothing
  is skipped — the plan still visits every pair inside the bounds, so an
  exhausted search costs what it always did — and verification is unchanged: a
  candidate that fails the exact `Q(n)(k)` check is still discarded, never
  returned.

### Testing

- The deterministic silent-error gate grew from **213 to 241 scored cases**
  (`tests/silent_errors/`), still at **0 silent errors**. The new cases cover the
  PSLQ precision verdict as a *word* rather than a truthy value, so an `unknown`
  verdict cannot silently collapse into a pass.
- **`test_run_with_wall_fallback_bounds_a_cooperative_callee` no longer fails
  because the box is busy.** Its bound read `elapsed_ms < 20 * 300` against a
  call that measurably costs 2.7–5.3 s of work when it is behaving — a 13%
  margin — so it went red repeatedly during saturated parallel runs and always
  passed in isolation, which is the pattern that teaches people to ignore red.
  The bound is now on **process CPU time**, which tracks the work the callee
  actually did rather than the time the harness spent waiting for it. Wall time
  here is `wall_ms` — a real-time timer, which does not stretch — plus the join
  of a still-running worker, and only that second term was being measured
  against load. Idle vs. 24 spinners on 12 cores: wall 5.3 s → 24.9 s (4× over
  the old bound, a guaranteed failure), CPU 5.3 s → 8.8 s against a 60 s
  ceiling. The
  property is unchanged and still enforced from both ends — a callee that stops
  seeing the budget burns CPU without limit and trips the assertion, and one
  that stops coming back at all trips the `timeout(120)` marker — plus the test
  now also pins that the call ended on the fallback's own join rather than on
  some other check that raises the same code.

## 3.8.0 — 2026-08-12

### Silent errors fixed — do results you already computed need rechecking?

A *silent error* is a confident, plausible, mathematically wrong answer with no
exception, no `NaN` and no verification flag. Eleven were found and fixed this
release. **Eight of them shipped in 3.7 or earlier**, so if you have results
from an affected call, re-run them. The other three were in code added during
this release cycle and never reached a published wheel.

| Affected call | Wrong answer it gave | First shipped in | Recheck? |
|---|---|---|---|
| `decide(Forall(x, φ))` where the counterexample is a rational root whose denominator is **not a power of two** | `True` for a **false** universal theorem, e.g. `∀x. (3x+2)² > 0` (false at `x = −2/3`) | ≤ 3.7 | **Yes** — any `decide` verdict |
| `decide(Exists(x, φ))` with an `=` atom | `(True, witness)` where the witness does **not** satisfy the sentence, e.g. `∃x. 3x−2 = 0 → x = 1/2` | ≤ 3.7 | **Yes** — any cited witness |
| `Matrix.nullspace()` on a 2×2 with a symbolic determinant | A confident wrong kernel basis; `[[x,0],[0,1]]` returned `(0, x)`, for which `M·v ≠ 0` | 3.7 | **Yes** — verify `M·v = 0` numerically |
| `simplify` / `simplify_egraph` on a product containing `0⁻¹` | `1`, or `0`, depending on the engine — for an expression with no value at all. Reachable from `diff(2/(x − x), x)` | ≤ 3.7 | Yes, if any input could reduce to `0⁻¹` |
| `decide` on a two-variable sentence true only at an irrational point | `False` for a satisfiable `∃x∃y`, and `True` for its false `∀x∀y` dual | this cycle (2-var `decide` is new) | No published release affected |
| `batch_map(..., parallel=True)` under `context(budget=…)` | Ran **unbudgeted**, so candidates a sequential sweep reported as `E-BUDGET-001` came back as `E-INT-001` — a *mathematical* verdict | this cycle (batch APIs are new) | No published release affected |
| `product_definite` on a term with any non-integer coefficient | Off by `c^(hi−lo+1)`: `Π_{k=1}^{5} ½` returned `1` instead of `1/32`, `Π (2k−1)/(2k)` at `n = 6` returned `14.4375` instead of `0.2255859375` | ≤ 3.7 | **Yes** — any `product_definite` / `product_indefinite` result |
| `sum_definite` where the summand has a pole strictly *between* the bounds | A clean finite number for a sum with an undefined term: `Σ_{k=1}^{10} 1/((k−3)(k−2))` returned `−5/8` | ≤ 3.7 | **Yes** — any `sum_definite` over a range containing a denominator root |
| `euler_maclaurin` when `corrections` is too small for the summand | A fabricated additive constant — the missing term frozen at the fitting point. `Σ k⁹` at the default `corrections = 2` acquired `34359738368 = 512⁴/2` in a Faulhaber polynomial whose constant term is `0` | this cycle (Euler–Maclaurin is new) | No published release affected |
| `rsolve` on a **forward-shift** spelling with a non-zero right-hand side | The solution of a *different* equation: `f(n+1) − f(n) = n²` with `f(0) = 0` returned `Σ_{j=1}^{n} j²` instead of `Σ_{j=0}^{n−1} j²` | ≤ 3.7 | **Yes** — any `rsolve` written with `f(n+i)`, `i > 0`, and an inhomogeneous term |
| `rsolve` / `solve_linear_recurrence_homogeneous` on an order-2 recurrence with a **repeated** characteristic root | `C₀·rⁿ + C₁·rⁿ` — a one-parameter family presented as the general solution of a second-order equation, losing the `n·rⁿ` branch | ≤ 3.7 | **Yes** — check the discriminant of `r² + b r + c` |

Also fixed, and not a silent error but worse for an unattended loop: a Rust
panic escaped `interval_eval` as `pyo3_runtime.PanicException`, which inherits
from `BaseException` and therefore slips past `except Exception`. Shipped in
3.7 — a loop that survived everything else died on `x^(3/2)` over a negative
ball.

The deterministic silent-error gate (`tests/silent_errors/`, Tier-1 CI) now
scores **0 silent errors out of 213 scored cases** across evaluation,
integration, limits, linear algebra, number theory, real QE, series,
simplification, solving, and sums/products. Every trap added this cycle was
re-run against a build with the fix reverted and confirmed to score
`silent_error` there, and every trap is paired with a **control** — its nearest
convergent neighbour — so a subsystem cannot pass the gate by refusing
everything. That is a statement about the corpus, not a guarantee about the
library.

### Behaviour changes to plan for

Fixing a silent error means some calls that used to return now refuse. Every one
of these is a call whose previous answer was not justified:

- **`decide` raises `CadError` (`E-CAD-001`) where it used to answer**, whenever
  the formula has a non-strict atom (`=`, `≠`, `≤`, `≥`) and a boundary root has
  not been shown rational. This includes mixed-alternation sentences that route
  through De Morgan — `∀x∃y. p > 0` becomes `¬∃x∀y. p ≤ 0`, and the negation
  makes a strict body non-strict. `decide` is **not** a complete decision
  procedure in this implementation; treat `E-CAD-001` as *undecided*, never as
  *false*.
- **`rank`, `rref`, `nullspace`, `eigenvects`, `jordan_form` raise
  `E-LINALG-010`, and `inverse` raises the new `E-MAT-004`**, when an entry's or
  the determinant's vanishing can be decided neither way. Previously "could not
  prove non-zero" was silently read as "zero".
- **`simplify` leaves `0 · 0⁻¹` unevaluated** instead of returning `1` (or `0`).
  A result containing `(0 * 0^-1)` is Alkahest declining to give an
  indeterminate form a value, not a simplifier failure.
- **`sum_definite` raises `SumError` (`E-SUM-003`) when the summand is undefined
  at an integer inside `[lo, hi]`**, not only when the pole lands on `lo` or
  `hi+1`. The refusal names the offending index. Sums whose poles lie outside
  the range are unaffected: `Σ_{k=4}^{10} 1/((k−3)(k−2))` still returns `7/8`.
- **`euler_maclaurin` may return a shorter expansion, with no additive
  constant.** The constant is now fitted at a point outside the gate's check
  points and re-fitted at a second one; if the two disagree it is not a constant
  and none is claimed. The report says which way that went in `derivation`, and
  the `"fitted numerically"` hypothesis is only listed when a fitted constant is
  actually part of the answer. Genuine constants (`γ`, `ζ(2)`, `½log 2π`, …) are
  unaffected — they agree across fitting points to 13+ digits.
- **`product_definite(term, k, lo, hi)` with `lo > hi` returns `1` even for a
  zero term.** The empty product takes no factors; it previously returned `0`
  for `Π_{k=1}^{0} 0` while returning `1` for `Π_{k=1}^{0} k`.
- **`capabilities()["contract_version"]` is `3`, and `features` lost two keys:
  `groebner_cuda` and `numpy`.** Indexing either now raises `KeyError`; use
  `features.get(name, False)` if you need to span versions. Both were removed
  rather than wired up because neither was *falsifiable* — no observation a
  Python caller could make distinguished `True` from `False`:
  - `groebner_cuda` reported that the CUDA Macaulay-matrix kernel had been
    compiled in. The string `groebner_cuda` occurred exactly once anywhere in
    `alkahest-py` — the capability line itself. There was no binding, no
    `*gpu*` name in the public or the private module, and `GroebnerBasis`
    exposes only CPU methods. The kernel is unchanged and still reachable from
    Rust as `alkahest_cas::poly::groebner::compute_groebner_basis_gpu`; if
    dispatch ever prefers it, the binding lands first and a bit follows it.
  - `numpy` mapped to a Cargo feature gating the `numpy` crate, which
    `alkahest-py` never used an item from. The feature and the dependency are
    both gone. `ak.numpy_eval` and `ak.numpy_eval_par` are unaffected — they go
    through the buffer protocol and always worked with the bit `False`, which
    is its value on every wheel ever published.

  An unreachable `True` makes a caller trust something it should not, which is
  the same class of defect as a silent wrong answer; a bit that correlates with
  nothing is better removed than left to be misread.
  `tests/test_agent_contract.py::test_every_advertised_feature_has_an_entry_point`
  now walks `features` and fails on any key without a named, reachable entry
  point, so the next one cannot ship.
- **Rust, `--features groebner-cuda`: `compute_groebner_basis_gpu` and
  `reduce_batch` return `(polys, GpuBackendReport)` instead of `polys`.** Both
  fall back to CPU row reduction — when `device_id` is `None`, and when the
  driver fails — and the basis is identical either way, so a caller previously
  had no way to tell a GPU run from a CPU one. `GpuBackendReport::ran_on_gpu()`
  is true only when at least one mod-p reduction ran on a device and none fell
  back; `reductions_on_gpu`, `reductions_on_cpu` and `first_gpu_error` carry
  the detail. A compile error on upgrade is the intended failure mode for code
  that was recording these results as GPU results. Nothing at the Python
  surface changes: the feature has no binding.
- **`residue(f, z, point)` refuses a non-constant `point` with
  `AlkahestError` / `E-RESIDUE-005`** instead of leaking
  `AttributeError: 'Expr' object has no attribute 'numerator'` from the
  argument parser. `AttributeError` is not an `AlkahestError`, so
  `except ak.AlkahestError` missed it entirely. The existing `E-RESIDUE-001..4`
  refusals are now `AlkahestError`s carrying `.code` and `.remediation` too,
  rather than bare `ValueError`s with the code glued into the message;
  `AlkahestError` subclasses `ValueError`, so `except ValueError` still works.
- **`series` refuses instead of running forever, with `SeriesError` /
  `E-SERIES-003`.** `series(sqrt(t**-2 + t**-1), t, 0, 32)` never returned:
  coefficients are formed by repeated differentiation without re-simplifying, so
  a nested radical's derivatives grow by a constant factor per coefficient and
  the cost doubles per order. It now honours an active `Budget` (raising
  `BudgetExceededError`) and, with none, an internal work ceiling. It never
  returns a *shorter* series: `O(h^order)` on fewer coefficients than were asked
  for is a false statement about the remainder, which is worse than the refusal.
  Ordinary expansions are unaffected — the heaviest in the suites intern a few
  thousand nodes against a ceiling of 50 000.
- **`simplify_expanded` records a derivation step when its expansion bound stops
  it** (`expand_pow_limit_reached`, a no-op step naming the power it declined),
  and the bound itself is now a budget on the number of distributed products
  rather than a flat exponent cap. `(x+y)**6` and `(x+y+1)**7` now expand where
  the exponent-only cap refused them while permitting a twenty-term sum to the
  fourth power; anything above the budget comes back unexpanded *and says so*
  instead of looking like an expression that was already expanded.

### Known limits — documented, not fixed

These are properties of the design as it stands. They are called out here
because 3.8 is aimed at long unattended loops, and each of them is a way such a
loop fails.

- **`ExprPool` never reclaims.** The arena is append-only: no `clear`, no
  refcount, no GC, and the storage cannot shrink. The only way to free interned
  nodes is to **drop the whole pool** — and every `Expr`, `Matrix`, `Series` and
  `DerivedResult` holds a *strong* reference to its pool, so retaining one
  interesting result retains everything. Growth on a shared pool is linear and
  unbounded (~200 bytes/node; measured ~2 KB per `integrate` call over 20 000
  calls, 0 B/call with a fresh pool per iteration) while per-call latency stays
  **flat**, so the failure mode is a clean OOM with no slowdown to warn you
  first. `ExprPool` also exposes no `__len__` or `stats()`, so the growth is not
  observable from Python. The supported pattern is **one pool per problem**,
  documented in [`budgets.md`](docs/mdbook/src/budgets.md#exprpool-never-reclaims).
- **`run_with_wall_fallback` does not bound wall time for an uncooperative
  callee.** It joins its worker before the exception propagates, so it returns
  when the callee returns: `run_with_wall_fallback(time.sleep, 3.0,
  budget=Budget(wall_ms=50))` raises `E-BUDGET-001` after 3000 ms, and the
  message reports the real elapsed time so this shows up in a log rather than
  being inferred later. Python cannot kill a thread, and abandoning one would
  leak a live thread that still allocates into the pool and can only be stopped
  through the process-wide cancel flag. Only an **OS-level bound** (subprocess,
  process watchdog) is a hard deadline.
- **`wall_ms` granularity is one primitive operation, and FLINT calls cannot be
  interrupted.** After the checkpoint work above the overshoot is a small
  additive term (1.0–1.2×), but past a certain degree a single operation is a
  FLINT factorisation or resultant — one foreign-function call, ~2 s on a
  degree-62 integrand, which no cooperative mechanism can stop part-way.
- **`Matrix.eigenvals()` grows the pool on identical input** (~1.9 KB/call,
  measured over 20 000 calls on the same 2×2 integer matrix): it interns a fresh
  `__eigen_lambda_N` gensym per call. Every other Python-facing entry point
  measured is flat on repeated input. Cache eigenvalue results.
- **`Matrix.eigenvals()` can emit casus-irreducibilis cube roots** — correct
  under Alkahest's real cube-root convention (and honestly refused by
  `eval_expr` with `E-EVAL-009`, with `interval_eval` returning an unbounded
  ball) but evaluated on the **principal** branch by SymPy, NumPy and most other
  tools, which return a confident number that is not an eigenvalue. 14 of 720
  random integer matrices produced one. An honest refusal here becomes somebody
  else's silent error the moment the expression crosses the boundary, so
  evaluate inside Alkahest before exporting, or export a verified numeric
  enclosure instead. See [`interop.md`](docs/mdbook/src/interop.md).
- **The LLVM JIT leaks an LLVM `Context` per compile** (`Box::leak`, on the
  error paths as well as the success path). Feature-gated behind `jit`, so
  default PyPI wheels (Cranelift) are unaffected; do not compile in a loop under
  a `+jit` / `+full` wheel.
- **No sanitizer covers any Python-facing path.** The PR-gating ASan job runs
  with `detect_leaks=0`, the nightly LSan shard cannot reach a `cdylib` with no
  `#[test]` functions, and `pytest` is never run under a sanitizer. The
  behavioural substitute is the fresh-pool sweep described in
  [`TESTING.md`](TESTING.md#3-memory-safety--sanitizers).
- **There is no `cuda_device_count()`.** `CudaCompiledFn.call_batch_on(ordinal,
  …)` selects a device, but the valid range can only be discovered by trying an
  ordinal and catching `CudaError` (`E-CUDA-003`); the loop that does it is in
  [`gpu.md`](docs/mdbook/src/gpu.md#discovering-the-valid-device-ordinals). Not
  added yet on purpose: `cuda` implies LLVM 15 with NVPTX, so such a binding
  cannot be compiled on an ordinary dev box, no CI job builds the Python
  extension with either CUDA feature, and exercising it needs a device — it
  would ship with no verification of any kind, which is the provenance of the
  capability overclaims fixed above. It belongs in the same change as the
  missing `maturin develop --features cuda` + `pytest tests/test_cuda.py`
  nightly step.

### Fixed

- **`cargo test --features groebner-cuda` could not pass on a machine with no
  NVIDIA driver**, contradicting the header comment of
  `alkahest-core/tests/groebner_cuda.rs`. `cudarc` *panics* rather than
  returning `Err` when `libcuda.so` cannot be `dlopen`ed, so `gpu_available()`
  — whose entire job is to decide whether the GPU tier can run — aborted three
  tests instead of skipping them. A missing library and a missing device now
  both mean *not available*, while `ALKAHEST_GPU_TESTS=1` asserting a device
  that is not usable still fails hard. The GPU tier additionally asserts
  `GpuBackendReport::ran_on_gpu()`, so a "GPU test" whose reductions all landed
  on the CPU fails rather than passing on identical results.
- **`product_definite` dropped the scale it used to clear denominators.**
  `ratuni_poly_to_univ` multiplies a `ℚ[k]` polynomial through by the LCM of its
  coefficient denominators and never returned that factor, so every index
  contributed one spurious copy of it and the answer was off by
  `c^(hi−lo+1)`. It is called separately on numerator and denominator, so the
  two cancelled only when they happened to be equal — which is why integer-
  coefficient products were always right and `Π ½` was not. The scale is now
  returned and re-applied (`product_indefinite` gets `c^k`, the same factor in
  antidifference form). A 1936-case sweep over `(a₁k+b₁)/(a₂k+b₂)` against exact
  `Fraction` arithmetic finds 0 mismatches, down from 26 of 160.
- **`sum_definite` could not see a pole strictly inside the summation range.**
  The only check was `contains_zero_to_negative_power` applied to the telescoped
  difference `G(hi+1) − G(lo)`, which never mentions the interior indices, so
  only poles landing exactly on an endpoint were caught. The summand itself is
  now scanned, the same way the definite integrator's interior-pole guards look
  at the integrand rather than at `F(b) − F(a)`: the integer roots of the
  summand's own denominators are read off its ℤ-factorisation (so the cost does
  not grow with the range), each candidate is substituted, and refusal requires
  seeing an actual `0^{negative}` survive simplification — positive evidence,
  never a guess.
- **`euler_maclaurin` fitted its additive constant at the point its own gate
  scored.** The residual there was then zero by construction, so the `o()`-gate's
  decay test was satisfied whatever the number was, and any term the expansion
  was missing came back as a "constant" — `Σ k⁹` acquired `512⁴/2`. The constant
  is now fitted outside the gate's check points and only emitted if a second fit
  reproduces it; across the clean battery a genuine constant drifts by ≤ 3.2e-3
  of itself, a fabricated one by ≥ 0.93.
- **`rsolve` solved a shifted equation for forward-shift spellings.**
  `extract_recurrence` re-indexes the sequence terms into lag form (`f(n+o) ↦
  f(n−(max_o−o))`), which is the original equation with `n ↦ n − max_o`, but left
  the right-hand side at `n`. The right-hand side is now shifted with them, so
  `f(n+1) − f(n) = n²` and `f(n) − f(n−1) = (n−1)²` mean the same thing again.
  The answer is checked by substituting it back into the equation as supplied.
- **Order-2 recurrences with a repeated characteristic root lost a branch.**
  `rsolve` returned `C₀·rⁿ + C₁·rⁿ` and `solve_linear_recurrence_homogeneous`
  divided by `r₁ − r₂ = 0`, producing a closed form containing `0^{-1}` that
  evaluated nowhere. Both now use the basis `{rⁿ, n·rⁿ}`; order ≥ 3 already
  handled multiplicity correctly.
- **A Zeilberger certificate claimed a recurrence for the sum without its
  boundary hypothesis.** The verified statement is the telescoping identity in
  `k`; summing it over `k = k_lo..k_hi` leaves the boundary difference
  `G(n, k_hi+1) − G(n, k_lo)`, and `Σ_i a_i(n)·S(n+i) = 0` holds only when that
  vanishes. Both the core docs and the Python docstring asserted it
  unconditionally. It is false for `F = C(n,k)/(k+1)`, where `G(n,0) = −1` and
  `(n+2)·S(n+1) − (2n+2)·S(n) = 1`. The certificates themselves were and are
  correct; what was missing was the hypothesis. `ZeilbergerCertificate` now
  carries `side_conditions` (the hypothesis, in the same spirit as
  `DerivedResult.verification["side_conditions"]`) and `boundary_term`
  (`G(n,k) = R(n,k)·F(n,k)`), so a caller can discharge or refute it for their
  own range.
- **Claim graphs: a merge could close a dependency cycle, making the graph
  unreadable.** Claim IDs are content-addressed over the *normalised*
  statement, so two textually different statements (`"a"` and `" a"`) share an
  ID. Re-adding one took `ClaimGraph.add`'s merge path, which unions in its
  dependency edges — and those can point at claims recorded later, including
  ones that already depend on it. The resulting graph served fine in memory and
  serialised fine, then could never be read back: `from_json` topologically
  sorts and raised `CycleError`. `add` now refuses an edge that would close a
  cycle, naming both claims, so "a `ClaimGraph` is acyclic" is a real invariant
  and a JSON round-trip is total. Legitimate acyclic merging is unaffected.
  Found by the `test_json_round_trip_is_lossless` property test.

- **Sparse interpolation: Zippel oracle cost is now polynomial, not
  multiplicative.** `sparse_interp` was formulated recursively — interpolate the
  coefficients of `x₁` as polynomials in the remaining variables, recursively —
  which makes each level's oracle a batched Vandermonde lift calling the level
  below `t` times, so black-box evaluations grew as the **product** `∏ tᵢ` down
  the recursion instead of the sum. Measured on the V2-3 roadmap corpus:
  70 calls at 2 variables, 1,771 at 3, 139,552 at 4, 15,019,900 at 5 (75 s) —
  a factor of 25 → 79 → 108 per added variable, extrapolating to ~1e17 at ten,
  i.e. it never returned. Replaced with Zippel's actual iterative algorithm,
  which introduces one variable at a time and recovers the coefficients of the
  *known* skeleton from a transposed Vandermonde system: `O(n·d·T)` calls.
  The same corpus now takes **34 / 62 / 97 / 139 calls at 2–5 variables and
  601 for the 10-variable, 15-term roadmap case**, which completes in
  milliseconds. That acceptance criterion (≥ 95% success over 20 seeds) passes
  for the first time and its test is un-skipped; a new Rust test asserts the
  oracle *call count* grows linearly in the variable count, since a functional
  test alone cannot tell a correct implementation from one that never finishes.
  `sparse_interp` additionally verifies each candidate against the black box at
  random points and re-draws its anchors on mismatch, so an unlucky anchor
  (Zippel's skeleton hypothesis is probabilistic) now produces a refusal rather
  than a confidently wrong polynomial.
- **`solve` states the hypotheses a parametric answer rests on.**
  `solve([a*x - b], [x])` returns `b/a`, which is the solution *for `a ≠ 0`*: at
  `a = 0` the equation reads `-b = 0`, so there is no solution when `b ≠ 0` and
  every `x` when `b = 0`, and `b/a` is not even a number there. The
  generic-parameter reading is deliberate, but a parametric tuple is not a number
  and is therefore returned **unverified** — nothing substitutes it back — so the
  hypothesis was the only auditable signal and it was not being given. New
  `alkahest.solve_side_conditions() -> list[str]` reports the non-vanishing
  hypotheses the most recent `solve` assumed, in the shape
  `DerivedResult.verification["side_conditions"]` and
  `ZeilbergerCertificate.side_conditions` already use. An empty list means the
  solver *proved* every divisor non-zero: `solve([2*x - b], [x])` reports none.
### Added

- **`alkahest.ansatz` — parametric families and coefficient fitting** (P2
  autoresearch item 1). "Guess the shape, let the CAS pin the constants" is the
  most common move in experimental mathematics and everybody re-improvises the
  plumbing for it. `ansatz.polynomial`, `.rational`, `.exponential_polynomial`,
  `.linear_combination` and `.quadratic_form` build an `Ansatz` — an object
  rather than a bare `Expr`, because a bare expression loses the distinction
  between an *unknown coefficient* and an *independent variable*, and every
  downstream step needs it. `ansatz.fit(A, residual)` solves for the
  coefficients and returns an `AnsatzSolution` carrying `expr`, `assignment`,
  `rank`, `free`, `residual`, `points` and a `status` — `fit` reports
  `exactly_verified` only when the residual is symbolically zero, never on the
  strength of the collocation points alone (`certify="residual" | "exact" |
  "none"`). `enumerate_family` walks a coefficient grid for conjecture
  generation; `certify_nonneg` hands a fitted candidate to `sos_decompose`.
  Pure Python over primitives that are already fast in Rust (`Matrix.rref`,
  `simplify`, `subs`), so it works without the `groebner` feature; a residual
  genuinely nonlinear in the unknowns refuses with `E-ANSATZ-004` rather than
  degrading silently, and *no member of this family fits* is `E-ANSATZ-003` —
  a closed branch for that family, deliberately not phrased as a proof that no
  such object exists. See
  [`docs/mdbook/src/ansatz.md`](docs/mdbook/src/ansatz.md).

- **`alkahest.crosscheck` — cross-CAS differential testing** (P2 autoresearch
  item 2). A loop that only checks itself finds the bugs it already knows
  about. `crosscheck.check(op, …)` runs one comparison against an external
  oracle (SymPy today; `register_oracle` takes others) through a ladder of
  four rungs — syntactic, symbolic, rigorous-numeric, invariant — and reports
  `agree` / `diverge` / `incomparable` / `unavailable`. The rungs exist because
  most apparent disagreements are not disagreements: two antiderivatives differ
  by a constant, two simplifiers pick different normal forms. Only the
  invariant rung (differentiate the antiderivative, substitute the solution
  back, telescope the antidifference) settles those, and an operation that has
  no invariant stops at rung 3 rather than pretending. **A missing oracle is
  `unavailable`, never `agree`** (`E-XCHECK-002`) — the one failure mode that
  would quietly turn the whole module into a no-op. `sweep(cases=…, seed=…)`
  generates a seeded corpus and prints its seed in `summary()` always, because
  a sweep is only useful as a bug report if the run that found something can be
  reproduced; the seed defaults to `budget_seed()`, so a nightly job and a
  local reproduction share one knob. `run_frozen_corpus()` replays 9 pinned
  cases whose expected outcome is recorded with the reason. See
  [`docs/mdbook/src/crosscheck.md`](docs/mdbook/src/crosscheck.md).

- **`alkahest.smt` — SMT-LIB 2 export and a z3/cvc5 bridge** (P2 autoresearch
  item 3). Discrete and mixed integer/real/boolean subproblems are not
  Alkahest's problem class, and the fastest way to make it worse would be to
  pretend otherwise. `to_smtlib` emits a complete runnable script (the emitter
  lives in Rust next to `Formula`, with no `_ =>` arm anywhere in it, so a
  kernel node added later fails to compile rather than silently emitting
  plausible-but-wrong SMT-LIB); `smt.solve` runs an installed solver and reads
  the answer back. The trust asymmetry is the design: a **`sat` model is lifted
  to exact rationals and substituted back and checked in-process**
  (`exactly_verified`; a model that fails raises `E-SMT-004`), while **`unsat`
  is reported as `externally_asserted`** and is deliberately excluded from
  `research.MACHINE_CHECKED_STATUSES`, because consuming an unsat proof is a
  different project. Decimal literals are parsed from the *string*, so `0.1`
  becomes `Fraction(1, 10)` and never the nearest binary double; an algebraic
  witness (`root-obj`) is refused with `E-SMT-003` rather than evaluated to a
  float, since a float witness recorded as an exact one is precisely the silent
  error the bridge exists to prevent. `smt.supported(f)` answers "would this
  route work, and should I take it" *before* any solver runs, and recommends
  `prefer_in_tree` for real arithmetic with no integer variables — the in-tree
  routes produce artifacts, `nlsat` produces only an answer. `solve` takes
  quantifier-free formulas; `to_smtlib` exports quantified ones. See
  [`docs/mdbook/src/smt.md`](docs/mdbook/src/smt.md).

- **Asymptotics of sums — Euler–Maclaurin** (P1 mathematics item 10):
  `alkahest.experimental.euler_maclaurin(f, k, a, n, corrections=…)` expands
  `Σ_{k=a}^{n} f(k)` as `n → ∞`, recovering
  `H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …` for `f = 1/k`. `series` and Gruntz
  `limit` expand a *function*; this is the sum side, which is how conjectures
  about growth rates get settled. Returns an `AsymptoticReport` that records
  not just the terms but **how much is proved**: `rigor`, a per-hypothesis
  checked/assumed ledger, and the numeric evidence from the `o()`-gate. The
  additive constant (γ above) is *not* determined by Euler–Maclaurin from the
  `n`-side terms — it is fitted numerically, and the report says so rather than
  presenting it as derived. Terms are ordered by magnitude at the check points,
  so the constant lands below every growing term and above every decaying one.
  Refuses (`AsymptoticError`) when the summand has no symbolic antiderivative,
  is not evaluable at the check points, or no term survives the gate. See
  [`docs/mdbook/src/asymptotics.md`](docs/mdbook/src/asymptotics.md).

- **Rigorous global bounds — Taylor models and validated numerics** (P1
  mathematics item 9): `alkahest.bound_on_box(expr, box)` returns a rigorous
  enclosure of the *range* of an expression over an axis-aligned box;
  `alkahest.verified_integral(expr, var, a, b)` a rigorous enclosure of a
  definite integral; `alkahest.verified_no_roots` and
  `alkahest.verified_sign` three-valued (`"true"` / `"false"` /
  `"undecided"`) predicates. Ball arithmetic already gave rigorous *pointwise*
  evaluation — this closes the gap to rigorous statements quantified over a
  region, which is what turns a numeric observation into a theorem. New
  `alkahest_core::validated` module: Taylor model arithmetic in normalised box
  coordinates (polynomial part plus rigorously enclosing remainder, so `x - x`
  cancels symbolically instead of widening to `[-2, 2]`), and Moore–Skelboe
  branch-and-bound that prunes sub-boxes proven not to contain the extremum.
  Soundness over tightness throughout: outward rounding everywhere, and
  exhausting the work budget returns a wide-but-true enclosure with
  `budget_exhausted=True` rather than an error. Genuine failures refuse with
  `ValidatedError` (`E-VALIDATED-001` unsupported primitive, `-002` unbound
  symbol, `-003` singularity in the box, `-004` overflow, `-005` malformed
  request). See
  [`docs/mdbook/src/validated-bounds.md`](docs/mdbook/src/validated-bounds.md).
- **Positivity certificates — SOS and Positivstellensatz-lite** (P1 mathematics
  item 8): `alkahest.sos_decompose(p, vars)` returns an exact rational
  sum-of-squares decomposition `p = Σ σ_j q_j²`, and
  `alkahest.prove_nonneg(p, vars, constraints=[...], level=...)` returns a
  Handelman certificate `p = Σ_α c_α Π g_i^{α_i}` (`c_α ≥ 0`) on a basic
  semialgebraic set. This is the fast, certificate-producing complement to the
  complete-but-doubly-exponential `decide`: the output is a short algebraic
  identity anyone can re-expand, exportable to Lean via
  `PositivityCertificate.to_lean()`. New `alkahest_core::real::sos` module with
  its own exact rational simplex (Bland's rule — no floating point anywhere
  near a certificate), an ℚ multivariate polynomial layer, and a
  generator-cone Gram search. Every certificate is re-expanded and compared
  against the target identically before it is returned. The three outcomes are
  kept distinct on purpose: certified, `E-SOS-003` definitely negative (with a
  witness point), and `E-SOS-002` no certificate of this shape at this degree
  — which is a statement about the search, not a proof that none exists (the
  Motzkin polynomial refuses here rather than being misreported). See
  [`docs/mdbook/src/positivity.md`](docs/mdbook/src/positivity.md).
- **Creative telescoping — Zeilberger's algorithm** (P1 mathematics item 7):
  `alkahest.zeilberger(term, n, k, max_order=…, max_degree=…)` takes a proper
  hypergeometric term `F(n, k)` and returns a `ZeilbergerCertificate` carrying
  a P-recursive recurrence `Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)` together
  with the rational certificate `R` (`G = R·F`) — so `S(n) = Σ_k F(n,k)`
  satisfies `Σ_i a_i(n)·S(n+i) = 0`. This is the first operation in the CAS
  that is both a *decision procedure* over its class and a *certificate
  emitter*, which is what makes discovery→proof automatic in an agent loop
  rather than heuristic. New `alkahest_core::holonomic` module: exact `Q(n)`
  and `Q(n)(k)` arithmetic towers, proper-hypergeometric recognition
  (`gamma`, `factorial`, `binomial`, `pochhammer` heads), and the Gosper-style
  reduction over `Q(n)`. Every certificate is re-checked as an exact
  `Q(n)(k)` identity before return; a candidate that fails is discarded, never
  returned with a caveat. Refuses via `HolonomicError` with stable codes —
  `E-HOLO-001` (outside the proper hypergeometric class), `E-HOLO-002` (search
  bounds exhausted), `E-HOLO-003` (candidate failed exact verification),
  `E-HOLO-004` (malformed call). See
  [`docs/mdbook/src/telescoping.md`](docs/mdbook/src/telescoping.md).

- **Docs: autoresearch / search-plumbing guide.** New mdBook chapter
  [`search-plumbing.md`](docs/mdbook/src/search-plumbing.md) ties budgets,
  batch APIs, compact `DerivedResult` envelopes, claim graphs, and certificate
  coverage together; Sphinx gains [`api/workload.rst`](docs/sphinx/api/workload.rst)
  plus `DerivedResult.to_dict` / `BudgetExceededError` entries. Cross-links from
  getting-started, intro, README, claim-graphs, batch, and budgets.

- **Budgets, cooperative cancellation, and a determinism seed** (P1 search
  plumbing item 4): `alkahest.Budget(wall_ms=..., max_steps=..., seed=...)`
  and `alkahest.context(budget=...)` push a wall-clock/step budget into a
  new Rust-side cooperative checkpoint (`alkahest_core::budget`), so a
  fan-out loop trying many candidate integrals/rewrites can bound one
  candidate's cost instead of hanging on it. `alkahest.integrate` checks it
  at its top-level entry and its recursion boundary and raises
  `BudgetExceededError` (`E-BUDGET-001` wall clock, `E-BUDGET-002` step
  limit, `E-BUDGET-003` cancelled) rather than running unbounded;
  `alkahest.simplify` checks it once per rewrite pass and, since it has no
  error channel, stops early instead of raising (`run_with_wall_fallback`
  supplements this with a hard deadline via a worker thread when needed).
  `alkahest.request_cancel()` / `clear_cancel()` / `is_cancelled()` expose a
  process-wide cancellation flag so an orchestrator thread can stop a heavy
  call running elsewhere; `alkahest.budget_seed()` exposes the active
  budget's seed to RNG-consuming samplers for reproducible runs. See
  [`docs/mdbook/src/budgets.md`](docs/mdbook/src/budgets.md).

- **Batch and streaming evaluation** (`alkahest._batch`, Python-only): `batch_map` /
  `batch_map_iter` call a function once per item and **never raise** for a single bad
  element — the exception is captured into a `BatchItem(index, ok, value, error,
  elapsed_ms)`, with `error["code"]` set to the failing exception's own `E-*` code
  (`E-BATCH-001` as a fallback for exceptions with none). `batch_map` always returns
  results in input order, whether or not `parallel=True` fans the calls out over a
  `ThreadPoolExecutor`; `batch_map_iter` streams in input order when sequential and in
  completion order when parallel. `integrate_many`, `simplify_many`, and `diff_many` are
  thin `batch_map` wrappers over the three most common derivation entry points. See
  [`docs/mdbook/src/batch.md`](docs/mdbook/src/batch.md).

- **`DerivedResult.to_dict` / `.to_json`: versioned, token-efficient result
  envelopes** (P1 search plumbing item 6). Combines `.value`, `.verification`,
  `.certificate_status`, and `.steps` into one dict/JSON string with a stable
  `"kind": "alkahest.derived_result"` discriminator and independent
  `RESULT_SCHEMA_VERSION` / `STEPS_SCHEMA_VERSION` constants (also exported at
  module level and as `DerivedResult.SCHEMA_VERSION` /
  `.STEPS_SCHEMA_VERSION`). `mode="compact"` drops `before`/`after` step text
  and uses short step keys (`r`/`s`), but never renames, hides, or drops
  `verification["status"]` and never includes Lean certificate source in
  either mode. See `docs/mdbook/src/derivations.md`.

- **Python bindings for the parallel simplifiers**: `simplify_redex`,
  `simplify_auto` and `simplify_strategy` join the existing `simplify_par`.
  All take a single expression and return the same result as `simplify`; only
  the schedule differs. Each falls back to sequential `simplify` when the
  extension is built without `--features parallel` (as the PyPI wheel is), so
  the calls are always available and `simplify_strategy` then reports
  `"sequential"`. The three parallel entry points now release the GIL for the
  duration of the native call, so other Python threads run alongside them.

- **`simplify_redex`: level-scheduled parallel simplification** (Rust,
  `--features parallel`, exported through `experimental`). Buckets the
  expression DAG by height and simplifies each level with one `par_iter`, using
  a flat `Vec<AtomicU32>` indexed by `ExprId` as the memo instead of a hashed
  side table. Borrowed from HVM2's redex-bag scheduling; the interaction-net
  runtime itself does not transfer, since alkahest's hot paths are FLINT/Arb
  arithmetic. Does **not** replace `simplify_par` — best time over 1–32 threads
  on 32 cores: deep chain 23.1 ms → 5.5 ms, but a wide sum of independent terms
  5.1 ms → 10.3 ms. Fork-join keeps a chain on one worker and wins on width;
  level scheduling wins on depth, and on every shape at one thread. The
  traversal is iterative (no stack-overflow risk on deep inputs) and each node
  is visited once, so the derivation log is deterministic across thread counts.
  A barrier-free variant using per-node counters of unreduced children — HVM2's
  actual discipline — was measured and was never reliably faster, so it is not
  included.

### Fixes

- **`decide` proved false universal theorems** (silent error; shipped in 3.7).
  `∀x. (3x+2)² > 0` returned `(True, None)`. It is false at `x = −2/3`, exactly:
  `9·(4/9) + 12·(−2/3) + 4 = 0`, and `0 > 0` is false. No approximation appears
  anywhere in that argument, and `decide` is the engine behind every stability
  proof and bound check, so a false `True` here is a machine-checked-looking
  proof of a false theorem. Sweeping `∀x. (a·x − b)² > 0` over `a ∈ 1..9`,
  `b ∈ −6..6` gave a clean rule: the verdict was wrong **exactly when the double
  root `b/a` in lowest terms has a denominator that is not a power of two** —
  which is why `x² > 0` and `(x−1)² > 0`, the two cases already in the corpus,
  passed. The bug lived one denominator to the right of every existing test.
  Two layers, and the deeper one was a broken documented contract:
  `RootInterval` promises `lo == hi == r` for an exact rational root `r`, but
  the VAS isolator only recorded an exact root when the transformed polynomial
  vanished at a Möbius endpoint, which happens for dyadic roots and not in
  general (`real_roots(3x − 2, x)` returned the open bracket `(0, 1)`). CAD then
  built its sample set from rational bracket endpoints and midpoints and
  concluded `false` when none satisfied the formula — sound for a *strict* atom,
  whose solution set is open, but not for a non-strict one, whose solution set
  can be the single untested root; `∀x. φ` goes through `¬∃x. ¬φ`, so the missed
  witness became a `True` universal. Fixed exactly, not heuristically: by the
  rational-root theorem every rational root of an integer polynomial has
  denominator dividing the leading coefficient, so once a bracket is bisected
  below width `1/lc` it contains at most one such rational and exact rational
  evaluation settles it — `None` means "no rational root here", never "probably
  not". (Bisection requires a strict sign change and refuses to collapse onto a
  vanishing *endpoint*: neighbouring brackets share endpoints, and collapsing
  onto one deletes the root the bracket was isolating.) Where the boundary root
  is genuinely irrational the sample set is incomplete and nothing can fix that
  by sampling, so `decide` now refuses with `E-CAD-001` rather than fabricating
  a `false`. A randomised differential test against a `sympy.real_roots`
  multiplicity analysis found **18 wrong verdicts in the first 150 random
  polynomials** before the fix and **0 in 1 000** after.
- **`decide` returned existential witnesses that do not satisfy the sentence**
  (silent error; shipped in 3.7). `∃x. 3x − 2 = 0` returned
  `(True, {'x': '1/2'})`, and `3·(1/2) − 2 = −1/2 ≠ 0`. The verdict was right;
  the certificate was false — and a witness is the one part of an answer that
  looks like it needs no trust, so it is exactly the artefact a loop cites
  downstream. The `Eq`-interval fallback proved satisfiability on an isolating
  interval and then reported the interval **midpoint**. It now runs the same
  check any caller would (`eval_qf_formula` at the reported point) and reports
  `witness=None` rather than a point that fails. With the exact-rational-root
  recovery above in place the true witness is usually reported outright:
  `∃x. 3x − 2 = 0` → `(True, {'x': '2/3'})`, while `∃x. x² = 2` → `(True, None)`
  because no rational witness exists. Two existing tests that asserted the bogus
  witness are corrected with the reason spelled out.
- **A Rust panic escaped `interval_eval` as a `BaseException`** (shipped in 3.7).
  `interval_eval(x**Rational(3,2), {x: ArbBall(-3.3, 0.0)})` panicked at
  `ball/mod.rs` and surfaced as `pyo3_runtime.PanicException`, which inherits
  from `BaseException` — so a loop's `except Exception` handler did not catch it
  and the run died on an input it was supposed to survive. Not a silent error,
  but for multi-day unattended operation arguably worse than one. `ArbBall::pow_f`
  guarded a negative base with `!exp.is_exact()`, but `x^(3/2)` arrives as an
  *exact* point ball at 1.5, `(−3.3)^1.5` is `NaN`, and the corner-ordering
  `partial_cmp(...).unwrap()` then panicked; the same shape existed in
  `ArbBall::Div` via `∞/∞`, reachable from `(x^(3/2))^-2`. A negative base now
  requires an exact **integer** exponent, and both `pow_f` and `Div` check the
  corner set for `NaN`, returning the existing "no enclosure" answers. 306
  panicking expressions in the first fuzz run; **0** after, across 7 200
  expressions × 14 points.
- **`run_with_wall_fallback` poisoned the whole process on timeout.**
  `request_cancel()` sets a process-wide, sticky flag, and
  `run_with_wall_fallback` never cleared it — so one expired candidate, the exact
  event the API exists to handle, made every subsequent cooperative call in the
  process fail with `E-BUDGET-003` forever. A multi-day loop would have died at
  its first slow integral and then reported a cancellation storm that was really
  one timeout. The executor is now wrapped in `try/finally` and the flag restored
  *after* `ThreadPoolExecutor.__exit__` has joined the worker (so the cancelled
  call has already observed it), and only when this call was the one that raised
  it — an orchestrator with its own outstanding `request_cancel()` keeps its
  request. Survives 20 of 20 timeout+work cycles. Two regression tests in
  `tests/test_budget.py`. *(Introduced during this release cycle; no published
  release is affected.)*
- **`batch_map(parallel=True)` ran completely unbudgeted.** `BudgetGuard` is
  `!Send` and the budget frame stack is thread-local, so `context(budget=…)` had
  no effect on work fanned out over a `ThreadPoolExecutor`: measured, the main
  thread saw the budget and all four workers saw `False`. For unattended
  operation that is the safety mechanism silently not applying — and worse than
  simply "slower", because the candidates a sequential sweep reported as
  `E-BUDGET-001` came back from a parallel one as `E-INT-001`, the integrator's
  verdict that *no elementary antiderivative exists*. A loop records that as a
  permanently closed branch when nothing was decided. `batch_map` now snapshots
  the active budget on the calling thread and re-enters it inside every worker
  task, and `run_with_wall_fallback` likewise enters its `budget` argument on the
  worker thread it spawns. The semantics are documented rather than fudged:
  `wall_ms` stays a single sweep-wide deadline (captured at the `batch_map` call,
  since Python cannot read the frame's start instant), while `max_steps` becomes
  **per item**, because the Rust step counter lives in the frame and is not
  readable from Python. One item tripping its budget never cancels its siblings;
  `request_cancel()` still reaches every worker, because the flag is process-wide.
  *(Introduced during this release cycle; no published release is affected.)*
- **`simplify` gave `0 · 0⁻¹` a value** (silent error; shipped in 3.7). `0⁻¹` is division by
  zero, so `0 · 0⁻¹` is the indeterminate form `0·∞` and has no value under any
  convention — but `simplify` returned `1`, `simplify_egraph` returned `0`, and
  `simplify(5 · 0⁻¹ · 0)` returned `0`, so the three answers were their own
  proof that at least two of them were wrong. The rest of the library was
  already right: `eval_expr(0⁻¹)` raises `E-EVAL-009` and `simplify(0⁻¹)` leaves
  the power unevaluated. Four rules were each collapsing the surrounding
  product on their own: `collect_mul_factors` summed the exponents of a common
  base (`0¹ · 0⁻¹ → 0⁰ → 1`), which is `b^k·b^m = b^(k+m)` — an identity that
  needs `b ≠ 0` the moment one exponent is negative; `const_fold` absorbed the
  product to `0` because one factor was the literal zero; `collect_add_terms`
  dropped a summand whose integer coefficient was `0` without checking that the
  surviving factor was a *number*; and the e-graph's shrink ruleset contains
  both `(Mul ?x (Num 0)) → (Num 0)` and `(Mul ?x (Pow ?x (Num -1))) → (Num 1)`,
  so on this input it unioned `0` and `1` into one e-class. All four now decline.
  Reachable without writing `0⁻¹` by hand: `diff(2/(x - x), x)` returned `1` for
  a function whose domain is empty; it now returns an expression that
  `eval_expr` refuses. Scope, stated plainly: the guards test for a **literal**
  zero base, which — because the rule engine normalises strictly bottom-up —
  also covers every base the simplifier can reduce to zero, `x - x` included. A
  base that is zero but not provably so keeps the documented `b · b⁻¹ → 1`
  convention: a three-valued `zero_status` on the `Mul` rewrite path costs
  several 128-bit ball evaluations per node, which this path cannot afford.
  `simplify_egraph` is the exception — it hands the whole call to the rule
  engine when it finds a provably-zero denominator, and uses the full
  `zero_status` to decide that, because building and saturating an egglog
  program dwarfs the test. No measurable cost on
  `bench_codspeed.py::test_log_exp_simplify_depth4` (paired A/B over 20
  interleaved runs: median −0.8%, inside the ±13% noise of the machine).
  Nine cases added to the silent-error corpus, four of them controls that
  `x · x⁻¹ → 1`, `0 · x → 0`, `2x − 2x → 0` and the e-graph engine itself still
  work.
- **`decide` could deny a two-variable statement that is true only at an
  irrational point** (silent error; two-variable `decide` is new in this cycle,
  so no published release is affected). The univariate completeness guard shipped
  earlier in this release refuses rather than report an unsatisfiability it
  never checked at a boundary root; the two-variable path had the same guard but
  keyed on `=` / `≠` atoms only, so `≤` and `≥` still fell through.
  `∃x∃y. (x²−2)² + y² ≤ 0` — true at `(±√2, 0)`, where both squares vanish, and
  false everywhere else — came back `False`, and its dual
  `∀x∀y. (x²−2)² + y² > 0` came back `True`, a machine-checked-looking proof of
  a false theorem. `project_and_sample_x` already flagged the untested
  irrational projection root; the flag now escalates for every non-strict atom,
  matching `body_has_boundary_atom` one dimension down. Strict atoms are
  unaffected: their solution sets are open, so the open-cell midpoints are
  complete for them. Both sentences now refuse with `E-CAD-001`. The cost is
  more refusals in the mixed-alternation cases, which route through De Morgan
  and so present a negated (hence non-strict) body: `∀x∃y. p > 0` becomes
  `¬∃x∀y. p ≤ 0` and refuses where it used to answer. Five corpus cases and
  four Rust unit tests, including the controls that a *rational* boundary point
  is still found (`∃x∃y. (3x−2)² + y² ≤ 0` → `True` at `(2/3, 0)`) and that a
  genuinely unsatisfiable `≤` still decides `False`.
- **`Matrix.nullspace()` returned a confident wrong basis for any 2×2 with a
  symbolic determinant** (silent error; shipped in 3.7). The 2×2 fast path
  returns the perpendicular of a non-vanishing row, which is the kernel *only*
  when `det = 0`, and its full-rank gate recognised only a **literal** non-zero
  constant. Every non-literal determinant fell through into the rank-1 answer —
  "could not prove `det ≠ 0`" read as "`det = 0`", the exact mirror of the `rref`
  defect that motivated the three-valued zero test. `[[x, 0], [0, 1]]` returned
  the basis `(0, x)`, for which `M·v = (0, x) ≠ 0`, while `rank()` on the same
  matrix said 2 — two public calls making 2 + 1 = 3 for a 2-column matrix. No
  exotic function was needed to trigger it. The gate now uses the three-valued
  `zero_status`: proven non-zero → trivial kernel, proven zero → the
  perpendicular, undecidable → refuse with `E-LINALG-010`, matching what `rank`
  already did. The eigen paths (`eigenvects`, `jordan_form`, `matrix_exp`) are
  untouched: `det(A − λI) = 0` holds there *by construction* — λ is a root of the
  characteristic polynomial — so the caller states it via the new
  `KnownSingular` parameter rather than asking the simplifier to rediscover it
  from nested radicals, which it often cannot. Four cases added to the
  silent-error corpus, including a control that a genuinely rank-1 symbolic
  matrix still returns its kernel and one that checks `M·v = 0` numerically
  rather than just the dimension.
- **`nullspace`, `eigenvects` and `jordan_form` reported an undecidable entry
  with a vague code.** All three share one elimination routine, whose error
  type carried no payload (`Result<_, ()>`), so the specific refusal —
  "one entry's vanishing could be proven neither way, substitute concrete
  parameters and it works" (`E-LINALG-010`) — died at that boundary and came
  back as the generic `E-LINALG-002` / `E-EIGEN-006` "could not compute
  nullspace basis". The routine is `pub(crate)`, so widening its error type
  costs nothing on the public API (`cargo semver-checks` agrees). A *genuine*
  kernel failure still reports `E-LINALG-002` and can never inherit a previous
  refusal's code — `KernelFailed` is deliberately not an out-of-band carrier.
- **`Budget(wall_ms=…)` overshot `integrate` by an unbounded factor.** The
  checkpoints existed; the seconds were being spent between them. A 300 ms
  budget on `∫ cos x·sinⁿx/(sin⁹x + sin x + 1) dx` returned after 2–4 s, and the
  same family at degree 40 never returned at all — it had to be killed from
  outside the process. Measured rather than guessed: 98.7% of one such call was
  a single number-field Euclidean GCD (`alg_log_argument` → `kpoly_gcd`), and
  the residue after fixing that was a single ℚ[x] GCD normalising `A/D` to
  lowest terms (480 ms of a 482 ms call). Both Euclidean loops now check the
  budget per step, `integrate_raw` checks on entry (so a *sum* is bounded
  between summands), and the rational route checks at each stage boundary.
  The same ladder now overshoots by 1.0–1.2×, and the degree-40 case returns in
  317 ms. Because a GCD has no error channel and stopping one early returns a
  *wrong* GCD, the budgeted variants return `None` rather than a truncated
  answer, and the public `poly_gcd` / `NumberField::kpoly_gcd` signatures are
  unchanged. What remains is documented in `docs/mdbook/src/budgets.md`: the
  granularity is one primitive polynomial operation, and past a certain degree
  that is a FLINT call, which no cooperative mechanism can interrupt.
- **`request_cancel()` could not reach a running `integrate` or `limit`.**
  Two independent causes. The bindings held the GIL for the whole call, so a
  watchdog thread could not execute a single bytecode until the operation it
  wanted to cancel had already finished — only a flag set *before* the call was
  ever observed, which is the opposite of what a fan-out search loop needs.
  Both now release the GIL around the core call, using the idiom `simplify_par`
  already established. And `integrate`'s u-substitution search discarded every
  error from its recursive call, budget trips included, so it moved on to the
  next of up to twelve candidates instead of stopping; a budget error now
  propagates, and the search checks the budget once per candidate — the
  granularity where the seconds actually go.
- **Docs: `simplify_par` was documented with a signature it never had.** Both
  the Sphinx API page and the mdbook chapter showed it taking a list of
  expressions and returning a list; it takes one expression and returns one
  `DerivedResult`, and the documented call raises `TypeError`.
- **Docs: the documented local Valgrind command checked nothing.** `TESTING.md`
  globbed `target/.../deps/alkahest_core-*` and `CONTRIBUTING.md` /`TESTING.md`
  said `cargo test -p alkahest-core`, but the package is named **`alkahest-cas`**
  (`alkahest-core/` is only the directory). The glob matched zero binaries, the
  `[ -x "$bin" ] || continue` guard skipped the empty expansion, and the loop
  **exited 0 having run Valgrind on nothing**. Both are corrected, with a note
  on the naming so it does not come back. Also corrected in the same pass:
  `TESTING.md` claimed UndefinedBehaviorSanitizer coverage
  (`-Zsanitizer=undefined` appears nowhere in the repo), and `CONTRIBUTING.md`
  claimed Tier-1 CI runs "ASan on FFI tests" when that job is scoped to the crate
  *below* the FFI boundary and runs with `detect_leaks=0`.
- **Docs: `Matrix.inv()` and `M[i, j]` do not exist.** The Sphinx matrix page
  documented both; the methods are `inverse()` and `get(i, j)`, and `Matrix` is
  not subscriptable. The same page attributed the singular-matrix refusal to
  `E-MAT-001` (shape mismatch) rather than `E-MAT-003`.
- **Docs: the Rust crate path was wrong throughout.** Guide pages wrote
  `alkahest_core::…`, which is the *workspace-local alias* `alkahest-py` gives
  the dependency. A downstream crate writes `alkahest_cas::…`; corrected in the
  mdBook chapters, `ARCHITECTURE.md` and `CONTRIBUTING.md`.
- **Withhold Lean certificates for Basel-family infinite sums.** The
  `basel_zeta_even` derivation step had no Mathlib proof and fell through to
  the default `by ring_nf; simp` tactic, emitting false equalities (e.g.
  `3/k² = π²/2`) into the textbook-gate Lean pool. Treated like Gosper:
  withhold the whole certificate rather than emit a broken one.
- **Rustdoc:** drop a private-item intra-doc link from `sum_definite` that
  broke `cargo doc -D warnings`.
- **Ruff:** quiet unused `wit` bindings and a compound assert in
  `tests/test_cad_decide.py`.

### Performance

- **`numpy_eval` / `numpy_eval_par` no longer round-trip through a Python
  list of floats.** The previous implementation converted every NumPy array
  to a flat Python list via `.tolist()` before crossing into Rust
  (`compiled_fn.call_batch_raw(inputs_flat, ...)`), which boxes/unboxes one
  `PyFloat` object per element on both the input and output side — the root
  cause of `numpy_eval` measuring ~25× slower than `sympy.lambdify(...,
  "numpy")` for large batches. `CompiledFn` gains
  `call_batch_buffer`/`call_batch_buffer_par`, which read NumPy (or any
  buffer-protocol) `float64` arrays via a single bulk copy per array, run
  the native `call_batch`/`call_batch_par` with the GIL released, and write
  results directly into a caller-supplied output array. `numpy_eval` and
  `numpy_eval_par` (and the JAX primitive's concrete-eval path) use this
  fast path automatically, with a transparent fallback to the legacy
  `call_batch_raw`/`call_batch_raw_par` for older extension builds that
  lack it. `call_batch_raw`/`call_batch_raw_par` are unchanged and kept for
  backward compatibility. Non-contiguous or non-float64 inputs are still
  converted once via `np.ascontiguousarray(..., dtype=np.float64)`, never
  via `.tolist()`.

### Additions — earlier in this cycle

- **`decide` now handles two real variables with a quantifier prefix of
  length ≤ 2** (`alkahest_core::real::cad`), not just the single-variable
  fragment from V2-9: `∃x∃y`, `∀x∀y` (same-flavor blocks), and mixed
  alternation `∃x∀y` / `∀x∃y`, all for purely polynomial bodies over ℚ. The
  approach eliminates one variable via the existing [`cad_project`] (Brown
  projection), then re-decides the resulting univariate-in-the-other-variable
  sentence at every rational CAD-cell sample with the existing univariate
  engine — e.g. `∃x∃y. x²+y²=0` → `true` (witness `x=0, y=0`), `∃x∃y.
  x²+y²+1=0` → `false`, `∀x∀y. x²+y²≥0` → `true`, `∀x∀y. x·y>0` → `false`.
  If a projection root is irrational *and* the body contains an
  equality/inequation atom, that CAD cell can't be tested exactly with
  rational sampling alone (it would need full algebraic-number CAD lifting),
  so `decide` raises `Unsupported` (`E-CAD-001`) rather than risk an unsound
  `true`/`false` — same for quantifier prefixes longer than two variables.
  The univariate fragment's behavior is unchanged; `decide_exists_univariate`
  internals were also generalized (via `UniPoly::from_symbolic_clear_denoms`)
  to accept rational (not just integer) coefficient polynomials, which the
  two-variable path needs after substituting a rational sample for the
  eliminated variable.
- **`Assumptions` is now first-class for agent workflows.** Previously
  `Assumptions` had to be threaded through `simplify()` / `simplify_log_exp()`
  by hand on every call. Now:
  - `alkahest.context(pool=p, assumptions=my_assumptions)` sets a thread-local
    default; `simplify()`, `simplify_log_exp()`, and `solve()` pick it up
    automatically whenever the caller omits their own `assumptions=`/explicit
    argument (an explicit argument always overrides the context). See
    `alkahest._context.active_assumptions` and the updated `context()`
    docstring.
  - `solve(equations, vars, assumptions=...)` (or the context default) now
    drops any returned solution that assigns a non-positive value to a
    variable the assumptions declare `> 0` — e.g.
    `solve([x**2 - 4], [x], domain="real", assumptions=positive_x)` returns
    only `x = 2`. This composes with `domain="real"` as a final filter rather
    than replacing its complex/real logic, and is a no-op (returns the
    `GroebnerBasis`/list unfiltered) when there's nothing to check.
    `Assumptions.is_positive(expr)` is the new agent-facing predicate this is
    built on (true for an explicit `refine(x > 0)` fact or a `Domain.Positive`
    symbol).
  - **New sound rewrite: `abs(x) → x` under `x > 0`.** Joins the existing
    `sqrt(x**2) → x` / `exp(log(x)) → x` family gated on
    `Assumptions`/`Domain.Positive`. `abs(x) → -x` under `x < 0` is *not*
    added (a distinct, currently untracked, side condition) — only the sound
    direction ships. Emits a Lean certificate (`abs_of_pos`) for the
    bare-symbol case, withheld (no `sorry`) otherwise, matching the existing
    `exp_of_log`/`sqrt_of_square` certificate discipline.
  - `tests/test_assumptions.py` now imports `Assumptions` from the stable
    `alkahest` top level instead of `alkahest.experimental` (the experimental
    module still re-exports it unchanged, so old imports keep working).
- **Basel-family infinite sums:** `sum_definite(expr, k, lo, hi)` now recognizes
  `hi = pool.pos_infinity()` p-series with an even power, e.g.
  `sum_definite(1/k**2, k, 1, pool.pos_infinity())` → `π²/6` (the Basel
  problem) and `Σ 1/k⁴ = π⁴/90`, via a Bernoulli-number/even-zeta table
  (`alkahest_core::sum::special`) rather than Gosper, which cannot sum `1/k^p`
  in closed form. Odd powers (`ζ(3)`, …) and any other unrecognized infinite
  bound still honestly raise `E-SUM-002` instead of guessing. `sum_definite` /
  `sum_indefinite` docstrings also no longer claim Faulhaber/geometric sums
  are unsupported — they've worked via general Gosper summation all along.

### Lean certificates

- **Definite-integral certificates now cover finite sums and constant
  multiples**, not just a single `sin`/`cos`/`exp`/`xⁿ` term: `∫ (sin x + cos
  x)`, `∫ 3·cos x`, `∫ -exp x`, and mixed multi-term combinations like
  `∫ (x² + sin x + 3·cos x)` now emit a type-checking interval-FTC proof
  (`HasDerivAt.add`/`.const_mul`/`.mul_const` and the `IntervalIntegrable`
  analogues composed over the existing base fragment). A numeric-literal
  coefficient (`Integer` or `Rational`) is required — a symbolic factor (e.g.
  `y · cos x`) and any addend outside the certifiable base fragment still
  withhold the *entire* certificate, never a partial one.

### Fixes — earlier in this cycle

- **`alkahest.SumError` now actually catches native summation errors.**
  `sum_definite` / `sum_indefinite` raise the native `E-SUM-*` exception, but
  `SumError` was missing from the native-exception overlay list, so
  `except alkahest.SumError` (or `pytest.raises(alkahest.SumError)`) silently
  failed to catch it — the only error class with this gap.
- **`eval_expr` no longer returns `nan`/`inf` as if it were a value.**
  Substituting into an expression whose denominator is zero at that point
  (e.g. `(x²-1)/(x-1)` evaluated *as written* at `x = 1`) reduces to `0 ·
  (1/0)` under plain IEEE-754 arithmetic — previously `eval_expr` handed that
  `nan` straight back as a normal-looking float. It now raises `DomainError`
  (`E-EVAL-009`) instead. `cancel()` first is still the correct way to get
  the limit: `eval_expr(cancel((x**2-1)/(x-1)), {x: 1})` legitimately
  returns `2`. The structured `evaluate(..., mode="f64")` API already
  reported this case as `status="unsupported"`; `eval_expr` (the raw
  `float`-returning entry point, and the tree-walking interpreter
  `eval_interp` it's built on internally) is now consistent with it. New
  `alkahest_core::jit::eval_interp_checked` for Rust callers that want the
  same check without a panic-on-`None` `.unwrap()`.
- **`solve(..., domain="real")` filters out complex roots.** `solve([x**2 +
  1], [x])` always returns the complex roots `±i` (`x² = -1` has no real
  solutions) — previously there was no way to ask for real solutions only
  short of manually inspecting each returned expression for an imaginary
  part. `domain="real"` (default `None`, unchanged existing behavior) now
  filters the solver's output: `solve([x**2 + 1], [x], domain="real")` →
  `[]`, `solve([x**2 - 1], [x], domain="real")` → `±1`. Composes with
  `numeric=True` and the `numeric=True` degree-limit fallback to homotopy
  continuation (which already returns real roots only).
- **Definite integrals no longer integrate through poles.** `integrate(f, x, a, b)`
  computed the antiderivative and returned `F(b) − F(a)` without checking for
  singularities of `f` inside `[a, b]`, so divergent integrals came back as clean,
  plausible, wrong values with no error raised — `∫_{-1}^{1} x⁻² dx` returned
  `-2`, `∫_{-1}^{1} x⁻¹ dx` returned `-log(-1)`, and `∫_0^2 dx/(x²-1)` returned a
  residual containing `log(-1)`. (`verification.status` was `"unverified"` for
  these, but it is also `"unverified"` for correct results, so it did not
  distinguish them.) Rational integrands are now checked for real poles on the
  closed interval — with factors shared with the numerator divided out, so
  removable singularities such as `(x²-1)/(x-1)` at `x = 1` are still accepted —
  and an improper integral raises `E-INT-001` instead of returning a value.
  Non-polynomial denominators (`1/sin(x)`) are not analysed and are unaffected,
  as are symbolic bounds, which cannot be compared against root locations.

## 3.7.0 — 2026-07-25

### Matrix / linear algebra

- **SymPy-style matrix multiply:** `A * B` is the matrix product (same as
  `A @ B`); `A * k` / `k * A` scalar-multiply; `A ** n` for non-negative
  integer powers; named `multiply` / `scalar_mul` / `hadamard` methods.
- **Symbolic 3×3 eigenvalues:** closed-form eigenvalues for parametric 3×3
  matrices whose characteristic polynomial is an irreducible cubic over the
  parameter field (Cardano / trigonometric path), not only 2×2.

### Lean certificates

- **Definite-integral certificates:** definite `integrate` emits a
  type-checking Lean proof via Mathlib's FTC / interval-integral lemmas
  (previously only indefinite integrals had certificates).
- Broader Lean coverage for differentiation (chain rule, log/sqrt/tan,
  quotient) and assumption-gated exp/log identities; certificates that do
  not typecheck are withheld rather than emitting broken proofs.

### Fixes

- **Laplace hyperbolic inverse:** irreducible quadratics with `ω² < 0`
  (e.g. `1/(s²−2)`) now invert to sinh/cosh instead of `sin(√(−κ²))` (which
  evaluated to NaN / declined). Forward sinh/cosh folds `(√c)²→c` in the
  denominator and the inverse peels s-free amplitudes so `L⁻¹{L{sinh(√2 t)}}`
  round-trips. Literal negative Heaviside/Dirac shifts `θ(t+a)`, `δ(t+a)`
  with `a > 0` are refused (`E-TRANSFORM-001`) rather than emitting the wrong
  unilateral formula.

- **Transform round-trips:** Inverse Laplace now inverts repeated irreducible
  quadratic poles of order 2 (needed for `L⁻¹{L{t sin}}` / `t cos`). Inverse Z
  matches the forward sin/cos table forms directly so transcendental
  coefficients (`sin(ω)`, `cos(ω)`) do not block `Z⁻¹{Z{sin(ωn)}}` via `apart`.
  Locked in by Rust unit tests and `tests/test_transform_roundtrips.py`.

- **`log(exp(z))` over ℂ:** `simplify_log_exp` only folds `log(exp(x))→x` when
  every free symbol in `x` is real-valued; `Domain.Complex` (and `I`) refuse
  the rewrite. Egglog no longer loads `Log∘Exp` (no domain check). Prevents
  silent wrong answers when `Im(z) ∉ (−π, π]`. The real-valued check now also
  accounts for branch-cut sub-terms: a non-integer power of a negative real
  (`(−20)^(1/2) = √20·i`) and `sqrt`/`log`/inverse-trig of out-of-range real
  arguments are no longer misclassified as real, so `log(exp(√(−20)))` and
  `log(exp(log(−5)))` no longer fold to a wrong principal value.

- **Complex branch-cut evaluation:** `evaluate(..., mode="complex")` now
  auto-binds the canonical imaginary unit `I → 1j`, accepts real scalar
  bindings, and evaluates non-integer powers on the principal branch via
  `exp(w·Log z)` (so e.g. `(-1)**(1/2) → i`). Complex `sqrt` uses the same
  Log path to avoid cancellation near the negative-real cut. Locked in by an
  mpmath fuzz oracle (`tests/test_complex_branchcut_oracle.py`). Exact `Arg`
  on the cut still declines (`E-EVAL-011`). `ExprPool.imaginary_unit()` is
  exposed in Python; `Expr ** float` builds a float-exponent node.

- **Assumption-gated log/exp rewrites:** `simplify_log_exp` and egglog no longer
  apply branch-cut identities (`exp(log(x))→x`, `log(x)+log(y)→log(xy)`,
  `log(a^n)→n·log(a)`, `log(a/b)→log(a)−log(b)`) without positivity facts.
  Pass an `Assumptions` context or use `Domain.Positive` symbols; safe rules
  `log(exp(x))→x` and `exp(x)·exp(y)→exp(x+y)` still apply unconditionally.
  Static symbol domains are now collected into the colored e-graph pass for
  all `simplify_with` callers.

- **E-graph constant folding:** `simplify_egraph((x+x)/2)` now returns `x`
  instead of leaving `((x * 2) * 1/2)`. The post-extraction const-fold pass
  flattens nested `Add`/`Mul` so coefficients from linear canonization and
  reciprocal folds meet in one n-ary product.

- **Accurate `erf`/`erfc` in f64 eval:** use libm rather than a coarse
  approximation on the numeric evaluation path.

### Features

- **Parametric `solve`:** free symbols omitted from `vars` are treated as
  parameters, so e.g. `solve([x**2 - y], [x])` returns `±sqrt(y)` instead of
  raising `SolverError`.

### Output hygiene

- Parenthesize nested powers in `str` / LaTeX / Unicode so `x^(1/2)^3` is unambiguous.
- `MultiPoly.to_expr` omits unit coefficients (`cancel((x²−1)/(x−1))` → `x + 1`).
- `simplify(gamma(1))` → `1` via a new `PrimitiveFold` rule.
- Literal division by zero raises `ZeroDivisionError` instead of building `0^-1`.

### API

- Hide import-machinery leaks (`contextlib`, `exceptions`, `alkahest`) from
  ``dir(alkahest)`` / autocomplete; submodules remain explicitly importable.
- `UniPoly.from_coefficients` accepts Python ``int`` coefficients (not only ``Expr``).
- `cancel` / `together` / `MultiPoly.from_symbolic` / `radical` infer free symbols when
  *vars* is omitted.
- Structured error messages now include the stable code prefix, e.g. ``[E-INT-004] …``.

### Docs / release

- Document `parse` in the README quickstart; clarify that `limit` / `series` are not `DerivedResult`.
- Expand `sum_definite` / `sum_indefinite` / `diophantine` / `solve` docs (Faulhaber gap, binary Diophantine patterns, parametric solve).
- Release wheel smoke runs the README quickstart + fresh-interpreter `parse` against the built wheel; Windows runners force UTF-8 so Lean certificates containing `∫` do not abort the smoke step.

## 3.6.0 — 2026-07-17

### Release / packaging

- **Cranelift JIT in default PyPI wheels:** default Linux/macOS/Windows wheels ship `egraph` + `groebner` + Cranelift JIT (`cranelift_jit`); LLVM `+jit` / `+full` remain GitHub Release–only local versions.

### Complex / numeric evaluation

- **Complex numeric evaluation and rational residues:** complex-mode numeric evaluation with rational residue support.
- **Principal Arg and complex symbolics:** branch-safe `Arg` folds and conservative symbolic complex primitives.
- **Unified experimental evaluation API.**

### Special functions / solver

- **Special-function foundation:** Lambert W, digamma, Bessel J₀/J₁ primitives.
- **Lambert W / trig transcendental solve:** `solve` recognises `α·u·e^u = c` (affine `u`) via principal `W₀`, and `sin`/`cos`/`tan` of an affine argument equal to a constant (principal inverse only — no `2πk` family). Thin experimental constructor: `alkahest.experimental.lambert_w`.
- **Transcendental solve** for exp/log equations.

### Simplification

- **Trig normal form (`simplify_trig_normal_form`):** opt-in fixed-point simplifier for sin/cos polynomials (DCM `Rᵀ·R − I` → `0` in one call).
- **Sound assumptions:** conditional rewrites require explicit assumptions.

### Integration / Risch

- Genus-0 √quadratic (including arcsin / negative leading coeff), Weierstrass `t=tan(x/2)`, trig powers & products, inverse-trig / reciprocal-trig / inverse-hyperbolic antiderivatives, Coates genus≥2 hyperelliptic logs, and exact vs numeric verification status.

### Linear algebra / ODE / real

- **Matrix:** symbolic eigenvalues / `matrix_exp`; `Matrix.rref` on the agent surface.
- **ODE:** numeric RK4/RK45 integrator and `dsolve` Python binding.
- **Parametric Routh–Hurwitz (`routh_hurwitz`).**

### API / agents

- **`capabilities()` / feature parity reporting** and agent capability / verification contract metadata.

## 3.5.1 — 2026-06-15

### Integration / Risch

- **Exact elliptic-integral constants:** genus-1 elliptic antiderivatives now print their reduction constants as exact algebraic numbers (`√3`, `3^(-1/4)`, `(2+√3)/4`, `12^(-1/4)`, `2√3-2√2`, …) instead of `2^53`-denominator float reconstructions. `∫dx/√(x³+1)` → `3^(-1/4)·EllipticF(acos((√3-(x+1))/(x+1+√3)), 1/2+√3/4)`.
- **No-real-root quartic normalization:** the `atan` substitution's Möbius coefficients are normalized so they reduce to simple `a+b√n` forms (e.g. `∫dx/√(x⁴+1)`).
- **Region-aware soundness gate:** the elliptic verification gate samples each `P > 0` interval (derived from `P`'s real roots), so correct reductions whose valid region is narrow or shifted no longer spuriously decline (e.g. `∫dx/√(x³-7x-6)`, region `x ≥ 3`).

## 3.5.0 — 2026-06-12

### Kernel

- **Imaginary unit:** canonical `I = √(−1)` as a kernel-blessed `Complex` symbol (`ExprPool::imaginary_unit()`); `i^n` power cycling and `Mul` collapse via `i² = −1`.

### Transforms

- **Fourier / Laplace / Z-transform:** symbolic forward and inverse transforms.
- **Fourier:** shifted Gaussian `F{e^{−a(x−b)²}}` with explicit phase factor via completing the square.
- **Z-transform inverse:** irreducible quadratic denominators (complex-conjugate poles) → real damped sinusoids.

### Calculus

- **Formal power series:** lazy FPS ring over ℚ with analytic operations.
- **Multivariate limits:** path-certificate non-existence.
- **Asymptotic expansions** at infinity.

### ODE

- **Classical `dsolve`:** first-order classes, linear constant-coefficient, and Euler–Cauchy.
- **Series solutions:** power-series and Frobenius methods for linear ODEs.

### Python

- **Experimental surface** (`alkahest.experimental`) for calculus, ODE, and transform APIs.

### Integration / Risch

- **Elementary products:** `x·exp(a·x)` (and related cases).
- **K-rational Hermite** reduction in `k_rational_integrate`.

### Poly

- **Puiseux tower continuation** with additive API (semver-safe re-land).

### Lean certificates

- **Differentiation:** `to_lean` / `DerivedResult.certificate` on `diff` results now emit `deriv (fun x => …) x = …` goals with Mathlib derivative lemmas instead of incorrect rewrite equalities (e.g. `x³ = 3x²`).

### Demo playground

- **Outputs:** render cell results as markdown; copy cell with output.
- **Lean certificate** cell in the default notebook.
- **Server kernel:** isolated `alkahest-playground` kernelspec in the server venv; matplotlib inline + figure flush; route matplotlib/numpy/playground_helpers cells to the server.
- **Lean verify:** legacy diff certificate shim in `playground_helpers` for older wheels; `start.sh` builds local alkahest via `maturin develop` when developing in-repo.

### Fixes

- **JIT:** cover all numeric primitives in `eval_interp` (+ registry sync test).
- **simplify:** fold elementary constants, trivial powers, and rational canonicalization.
- **lean:** emit `deriv` goals for diff certificates.

## 3.4.0 — 2026-06-10

### Calculus / integration (Risch roadmap)

- **M4 algebraic tower:** `AlgExtension` as a `DifferentialField`; algebraic top-generator dispatch via radical-over-exp substitution; coupled `coupled_radical_rde` over exp/log tower bases; K-rational integration with K-log emission; certify `NonElementary` for entangled K-log coefficients.
- **Non-diagonal f Risch DE:** generalize coupled algebraic Risch DE to f ∈ ℚ(x)(α); ∫R·exp(β) with β algebraic; non-diagonal f for `RadicalExt` over ℚ(x); polymorphic RDE degree bounds (Bronstein §6.5).
- **Algebraic singular places:** van Hoeij enlargement; Newton–Puiseux expansion at algebraic base points.
- **Genus-1 elliptic:** diagnose and decline-stability for remaining genus-1 elliptic configs; M3 capstone tests.
- **Integration utilities:** partial fractions (`apart`) and definite integration via FTC; non-linear u-substitution (derivative-divides heuristic).

### Demo playground

- Clear notebook control and calculus starter demo.

### Fixes

- **simplify:** correct e-graph integer `Pow` constant folding.
- **poly:** accept integer-valued `Rational` nodes in `RationalFunction::from_symbolic`.

## 3.3.0 — 2026-06-08

### Calculus / integration (Risch roadmap)

- **M4 tower recursion:** `DifferentialField` trait with ℚ(x)/exp/log implementations; multi-generator recursive integrator (exp × radical-over-tower); radical extension as a generic `DifferentialField` with tower-recursive `rational_rde`.
- **Elliptic integral output:** `EllipticF` / `EllipticE` / `EllipticPi` / `EllipticK` primitives; first-, second-, and third-kind elliptic output for genus-1 ∫dx/√(cubic|quartic) and ∫R/√(cubic|quartic); all-complex-root genus-1 quartics (∫dx/√(x⁴+1)); cosφ-config third-kind output.
- **Genus-1 capstone:** wire quartic y²=quartic and cubic cases into the public engine; genus-1 quartic without a rational root (Nagell); genus-0 Euler substitution for ∫R(x,√quadratic)dx; Miller log-argument construction; Abel–Jacobi in FIND-ORDER.
- **Algebraic extensions:** tower algebraic base, conjugate reduction, non-Galois quartic, general quadratic; algebraic residues and ramified places; lazy Hermite; Trager Q-basis and algebraic places; FIND-ORDER for non-branch and algebraic places; genus-2 compositum and end-to-end path.

### Reinforcement learning

- Hub package import fixes and CI metadata for symbolic integration; Environments Hub install path updated to `alkahest` org.

## 3.2.0 — 2026-06-05

### Reinforcement learning

- **`alkahest.rl`:** framework-agnostic core (`BaseGenerator`, `BaseVerifier`, `Rubric`, `CurriculumScheduler`) and a symbolic integration environment (`alkahest.rl.envs.integration`) with Risch-tier task grammar, layered `IntegrationVerifier`, and Prime Intellect `verifiers` entry point (`load_environment`).
- Optional pip extra: `pip install "alkahest[rl]"` (Python ≥ 3.10; pulls `verifiers` + `datasets`).
- veRL recipe: `recipes/verl_integration_reward.py`.
- Environments Hub manifest: `python/alkahest/rl/envs/integration/`.

### Calculus / integration (Risch roadmap)

- Algebraic Risch extensions: tower field integration, simple radicals, coupled algebraic RDE, genus-0 reduction and parametrization.
- Genus-1 stack (in progress): integral basis (van Hoeij), Hermite on curve, residue divisor, FIND-ORDER, elliptic engine.
- Newton–Puiseux fractional-power expansions; algebraic-coefficient Puiseux.

### Linear algebra

- Expanded matrix coverage (`alkahest-core/src/matrix/linear_algebra.rs`); Python bindings and tests.

## Unreleased (historical notes)

### Breaking / default-feature change

- **`groebner` is now a default Cargo feature in `alkahest-cas`**, matching the Python wheel defaults. `alkahest-cas = "2"` now includes Gröbner-backed APIs (`solve`, `diophantine`, homotopy) without explicitly listing the feature. To opt out: `alkahest-cas = { version = "2", default-features = false }`.

## Unreleased (2.2.x)

### Calculus

- **Transcendental Risch integration (issue #4):** Implements the complete Risch decision procedure for elementary antiderivatives over the transcendental differential field tower K = ℚ(x)(t₁,…,tₙ) with tᵢ = exp(ηᵢ) or log(hᵢ). Modules: `risch/poly_rde.rs` (polynomial Risch DE solver over ℚ[x]), `risch/tower.rs` (generator detection and tower decomposition), `risch/exp_case.rs` (hyperexponential case via RDE), `risch/log_case.rs` (hyperlogarithmic case via IBP recursion), `risch/mod.rs` (router and detection predicate). The engine checks `contains_risch_form` before the rule-based fallback. **Non-elementary certification:** when the polynomial RDE y' + k·Dη·y = h has no polynomial solution, the integrand is certified non-elementary (`IntegrationError::NonElementary`, error code `E-INT-004`). **Elementary cases covered:** p(x)·exp(g(x)) for any polynomial p and any degree, log(x)ⁿ for any n, p(x)·log(x)ⁿ via IBP recursion. Derivation log records `risch_exp_rde` and `risch_exp` / `risch_log` steps. 24 Python tests in `tests/test_risch_integration.py` (4 non-elementary, 13 exp-tower, 7 log-tower). References: Risch (1969), *Trans. AMS* 139; Bronstein (2005), *Symbolic Integration I*, Ch. 5–7.

### Infrastructure (JIT and evaluation)

- **Cranelift Tier-1 JIT** (`--features cranelift`): pure-Rust backend in `jit/cranelift_backend.rs`; usage-based tier selection via `CompileConfig` (interp → Cranelift → LLVM).
- **`CompileCache`**: memoize `ExprId + inputs → Arc<CompiledFn>`; Python `CompileCache` class with hit/miss stats.
- **Bulk JIT evaluation**: native `alkahest_eval_bulk` in Cranelift/LLVM backends; `CompiledFn::call_bulk` / `call_batch` column-major batch path.
- **Parallel batch evaluation**: `CompiledFn::call_batch_par`, `numpy_eval_par` (Rayon, `--features parallel`, GIL released).
- **DAG traversal memo tables**: per-call `HashMap<ExprId, T>` on simplify, diff, forward diff, integrate `is_free_of`, and JIT interpreter paths.
- **SIMD Horner f64 eval**: `eval_horner_f64` / `eval_horner_f64_batch` (4-wide `wide::f64x4`) on the interpreter numeric path.

### Infrastructure (simplification and FFI)

- **Colored e-graphs**: native layered union-find (`simplify/colored_egraph.rs`); `SimplifyConfig::assumptions` wired through `simplify_with`.
- **Match-disjoint egglog schedule**: shrink/explore rules split by LHS root symbol; `EgraphConfig::disjoint_schedule` (default `true`).
- **Discrimination-net pattern indexing**: `DiscriminationIndex` / `PatternRuleSet` for user `PatternRule` sets (`simplify_with_pattern_rules`; Rust API).
- **FLINT drop-safe wrappers**: RAII `Drop` on all FLINT factor types; `FlintMPolyCtx` ref-counted via `Arc`.
- **Vendored egglog v0.4.0** (`vendor/egglog`): default PyPI wheels now ship with `egraph` feature.

### Tooling and CI

- **CodSpeed** continuous benchmarking (Rust + Python).
- **uv / ruff / ty** integrated for Python dev workflow (`pyproject.toml` dependency groups).

## 2.0.4 — 2026-05-22

### Polynomial algorithms

- **V2-3 — Sparse multivariate interpolation (Ben-Or/Tiwari, Zippel):** Rust `alkahest_core::poly::interp` — `sparse_interpolate_univariate(eval, T, p)` recovers a sparse univariate `f ∈ Fₚ[x]` from exactly `2T` evaluations via Berlekamp–Massey + Cantor–Zassenhaus root-finding + BSGS discrete-log + Vandermonde solve; `sparse_interpolate(eval, vars, T, D, p, seed)` recovers a sparse multivariate polynomial via Zippel's variable-by-variable algorithm with batched Vandermonde lifting. Supporting infrastructure: `MultiPolyFp` (sparse polynomial over `Fₚ`), `reduce_mod`, `lift_crt`, `rational_reconstruction`, `mignotte_bound`, `select_lucky_prime`. Python: `sparse_interp_univariate`, `sparse_interp`, `SparseInterpError`, `MultiPolyFp`, `modular` submodule. ROADMAP acceptance criteria: 10-variable 15-term polynomial recovered at ≥ 90% success over 20 random seeds (`test_roadmap_10var_15term`). Tests: Rust `poly::interp`, Python `tests/test_sparse_interp.py` (18 fast + 1 slow).

- **Sparse modular GCD (`gcd_sparse_modular` / `gcd_sparse`) — substrate for faster modular algorithms:** Rust `alkahest_core::poly::interp::gcd_sparse_modular` — Zippel evaluation–interpolation GCD over ℤ[x₁,…,xₙ]; for each lucky prime `p`: probes the GCD degree in `x₁` via one specialization, then for each `x₁^k` degree runs `sparse_interpolate` to recover the coefficient polynomial `c_k(x₂,…,xₙ)`, assembles the modular GCD image, and repeats until the CRT product exceeds the Mignotte bound; CRT lifting via `lift_crt`; result normalised to primitive part with positive leading coefficient. `SparseGcdError` (`E-INTERP-010…012`). Python: `gcd_sparse`, `SparseGcdError`. Rust unit tests: `gcd_sparse_univariate_linear_factor`, `gcd_sparse_univariate_coprime`, `gcd_sparse_bivariate_common_factor`. Python integration tests in `tests/test_sparse_interp.py::TestSparseGcd` (activated after wheel rebuild).

## 2.0.3 — 2026-05-21

### Calculus

- **Full Gruntz limits:** Rust `alkahest_core::calculus::gruntz` — Gruntz (1996) MRV comparability-graph algorithm for limits of exp-log combinations as var → +∞. Steps: collect diverging `exp(h)` subexpressions, build comparability ordering via limit ratios, extract the maximally-ranked (MRV) set, pick ω → 0⁺, rewrite as Laurent series in ω, and read off the limit from the leading power. Thread-local depth counter (max 8) prevents unbounded re-entry. Gruntz is invoked from `limit_inner` before the 1/t substitution so exp structure is visible; existing L'Hôpital and series fallback paths are preserved. 6 new tests in `tests/test_gruntz_v217.py`; Rust unit tests in `gruntz.rs`.

### Advanced polynomial solvers

- **Polyhedral / mixed-volume homotopy:** Rust `alkahest_core::solver::polyhedral` — Newton polytopes, Graham-scan convex hull, Shoelace mixed-volume for n=2; binomial start system per mixed cell via complex log branch enumeration; `polyhedral_cell_iter` yields `(GbPoly start system, start points)` per cell. `solve_numerical` auto-selects polyhedral start when MV < Bézout bound; new Euler–Newton tracker `track_path_sys`. `PolyhedralError` (`E-POLYHEDRAL-*`). Python tests in `tests/test_polyhedral_v217.py`.

- **F5 signature-based Gröbner basis:** Updated `alkahest-core/src/poly/groebner/f5.rs` — corrected signature comparison, S-polynomial formation, and reduction bookkeeping; new Criterion benchmark group `groebner_f5` in `benches/alkahest_bench.rs`.

### Lean 4

- **`Filter.Tendsto` certificate export:** `alkahest_core::lean::emit_tendsto_cert(expr, var, lim, pool)` generates a Lean 4 snippet with the appropriate `Filter.Tendsto` statement; pattern-dispatches to Mathlib theorems (`tendsto_exp_neg_atTop_nhds_zero`, `tendsto_exp_atTop`, etc.) and falls back to `by sorry` for unsupported cases. Codomain filter is `nhds L` for finite limits and `Filter.atTop` for +∞. `emit_limit_header()` emits the required Mathlib imports.

### Demo playground

- **Lean certificate panel:** `LeanCertificate.tsx` renders `Filter.Tendsto` proofs inline in notebook output cells with syntax highlighting and a copy button.
- **F5 verification in notebook:** `demo-playground/server/lean_verify.py` — server-side Lean 4 subprocess verification; `output_parse.py` and `playground_helpers.py` added for structured kernel output; agent chat gains awareness of Lean verification results.

### Packaging

- **Crate renamed to `alkahest-cas`:** The published Rust crate is now `alkahest-cas` on crates.io (was `alkahest-core`). All internal references updated; README badge updated.

## 2.0.2 — 2026-05-17

### Packaging / releases

- Version **2.0.2** (workspace + `pyproject.toml`). Git tag **`v2.0.2`** for release CI (PyPI default wheels + **`+jit` / `+full`** on GitHub Releases). (`v2.01.0` / `2.01.0` is not a valid Cargo semver — leading zeros in numeric components.)

## 2.0.1 — 2026-05-16

### Packaging / releases

- Version **2.0.1** (workspace + `pyproject.toml`).
- **Release CI (`+full` wheels):** Linux `linux_x86_64` wheels with PEP 440 local version **`X.Y.Z+full`**, built with Cargo features `jit groebner parallel egraph`, attached to **GitHub Releases** next to existing **`+jit`** wheels. **`+jit`** and **`+full`** wheels are **not** uploaded to the main PyPI simple API (same policy as before for `+jit`) so `pip install alkahest` stays on the small default wheels.

## 2.0.0 — 2026-05-06

### Calculus and series

- **V2-15 — `series()` / Laurent expansions:** Rust `alkahest_core::calculus::series` — `series(expr, var, point, order)`, `Series`, `SeriesError` (`E-SERIES-*`); truncated Taylor expansions via differentiation and Laurent tails for univariate rationals with poles; kernel `ExprData::BigO` (`ExprPool::big_o`); pool file format **v3** (node tag 12). Python: `series`, `Series`, `SeriesError`, `ExprPool.big_o`; `_pretty` recognizes `big_o` nodes for Unicode/LaTeX-style printing of $\mathcal{O}(\cdots)$. Tests: Rust `calculus::series`, Python `tests/test_series_v215.py`.

- **V2-16 — `limit()` (prototype rules):** Rust `calculus::limits` — `limit`, `LimitDirection`, `LimitError` (`E-LIMIT-*`); finite points via 0/0 L’Hôpital, local Laurent/Taylor expansions (`local_expansion`), specials, and guarded direct substitution (`0/0`, `0·pole` rejection); limits at `±∞` via `x ↦ ±1/t` with nested rational power flattening and polynomial quotient normalization before `t → 0⁺`; `ExprPool::pos_infinity()` (`∞` symbol). Python: `limit`, `LimitError`, `ExprPool.pos_infinity`. Limitations: not full Gruntz; oscillatory or unconstrained transcendental tails may return `Unsupported`. Tests: Rust `calculus::limits::tests`, Python `tests/test_limits_v216.py`.

- **Algebraic-function Risch integration (Trager):** `alkahest-core/src/integrate/algebraic/` — genus-0 integrals involving `sqrt(P(x))` over ℚ(x) for P of degree 0/1/2 (J₀ formula + substitution); `NonElementary` guard for deg P ≥ 3; mixed integrands `A(x) + B(x)·sqrt(P(x))` via field decomposition. 14 tests in `tests/test_algebraic_integration.py`; 10 worked examples in `examples/risch_integration.py`.

### Discrete mathematics

- **V2-10 — Symbolic summation (Gosper / Zeilberger):** Rust `alkahest_core::sum` — `sum_indefinite(term, k)`, `sum_definite(term, k, lo, hi)` for terms with rational shift ratio (polynomials × `gamma` of a linear expression in `k`); `solve_linear_recurrence_homogeneous` for constant-coefficient homogeneous recurrences; `verify_wz_pair(F, G, n, k)` for checking discrete telescoping certificates. `SumError` (`E-SUM-*`). Python: `sum_indefinite`, `sum_definite`, `solve_linear_recurrence_homogeneous`, `verify_wz_pair`, `SumError`. Tests: Rust `sum::tests`, Python `tests/test_sum_v210.py`.

- **V2-18 — Difference equations (`rsolve`):** Rust `alkahest_core::sum::rsolve` — linear recurrences with constant coefficients and polynomial right-hand side in the recurrence index; `rsolve(eq, n, fn_name, initials)` returns a closed-form `DerivedResult`; `RsolveError` (`E-RSOLVE-*`). Python: `rsolve`, `RsolveError`. Limitations: non-homogeneous order > 2 and polynomial-coefficient recurrences not implemented. Tests: `tests/test_rsolve.py`, Rust `sum::rsolve`.

- **V2-22 — Symbolic discrete products (`∏`):** Rust `alkahest_core::sum::product` — `product_definite` / `product_indefinite` for terms that are rational in the index variable with numerator and denominator polynomials that factor into ℤ-linear terms (Γ-ratio telescoping + leading powers); `ProductError` (`E-PROD-*`). Stable re-exports in `alkahest_core::stable`. Python: `product_definite`, `product_indefinite`, `Product` (SymPy-shaped `Product(term, (k, lo, hi))`), `ProductError`; `examples/products.py`; tests Rust `sum::product`, Python `tests/test_product_v222.py`.

### Algebra and number theory

- **V2-17 — Matrix eigenvalues / eigenvectors / diagonalize:** Rust `alkahest_core::matrix::eigen` — `characteristic_polynomial_lambda_minus_m`, `eigenvalues`, `eigenvectors`, `diagonalize`, `EigenError` (`E-EIGEN-*`); splits `det(λI−M)` via FLINT ℤ factorization after clearing rational denominators in the coefficients of χ; linear and quadratic characteristic factors; rotation `[[0,-1],[1,0]]` diagonalizes over ℚ(i). Python: `Matrix.characteristic_polynomial_lambda_minus_m`, `eigenvals`, `eigenvects`, `diagonalize`, `EigenError`. Limitations: defective matrices return `NonDiagonalizable`; irreducible χ factors of degree &gt; 2 are rejected. Tests: Rust `matrix::eigen`, Python `tests/test_eigen_v217.py`.

- **V3-1 — Integer number theory:** Rust `alkahest_core::number_theory` — FLINT-backed `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`, `nthroot_mod` (prime modulus), `discrete_log` (moderate primes), `QuadraticDirichlet`; `NumberTheoryError` (`E-NT-*`); stable re-exports. Python: module `alkahest.number_theory` plus `DirichletChi` / `NumberTheoryError` from the native extension. Tests: Rust `number_theory::tests`, Python `tests/test_number_theory_v31.py`.

- **V2-19 — Diophantine equations (`diophantine`):** Rust `alkahest-core::solver::diophantine` — two integer unknowns; linear parametric families (extended gcd); `x² + y² = n` (enumeration); unit Pell `x² - D y² = 1` (fundamental `(x₀,y₀)` via continued-fraction convergents); `DiophantineError` (`E-DIOPH-*`). Python (`groebner`): `diophantine`, `DiophantineSolution`, `DiophantineError`. CI builds the wheel with `--features groebner`; `pytest.ini` sets `pythonpath = python`. Tests: Rust `solver::diophantine`, Python `tests/test_diophantine_v219.py`.

- **V3-2 — Non-commutative algebra:** `ExprData::Symbol` carries `commutative: bool` (default `true`). `ExprPool::mul` and `canonical_order` skip sorting when any factor subtree contains `commutative: false`; `collect_mul_factors` merges powers **globally** only for fully commutative products and **adjacent** identical bases otherwise. E-graph simplification falls back to the rule engine when a non-commutative symbol appears. `alkahest_core::algebra::noncommutative` — Pauli table (`sx`/`sy`/`sz`) and orthogonal Clifford snippet (`cliff_e1`/`cliff_e2`); `NoncommutativeCost` (e-graph tie-break). Pool file format **v4** adds `commutative` on symbol nodes. Python: `ExprPool.symbol(..., commutative=False)`, `simplify_pauli`, `simplify_clifford_orthogonal`; `examples/noncommutative.py`; `tests/test_noncommutative_v32.py`.

### Advanced polynomial solvers

- **V2-11 — Regular chains / triangular decomposition:** Rust `triangularize`, `RegularChain`, `extract_regular_chain_from_basis`, `main_variable_recursive` (`alkahest_core::solver::regular_chains`); optional bottom-univariate factor splitting via V2-7; `solve_polynomial_system` fallback backsolve from an extracted chain after a lex-basis stall. Python: `triangularize`, `RegularChain`; benchmark task `solve_6r_ik` (planar IK proxy). Tests: `tests/test_regular_chains_v211.py`, Rust `solver::regular_chains`.

- **V2-12 — Primary decomposition:** Rust `primary_decomposition`, `radical`, `PrimaryComponent`, `PrimaryDecompositionError` (`alkahest_core::ideal::primary`); partial GTZ-style splitting (saturations + Lex univariate factorization). Python: `primary_decomposition`, `radical`, `PrimaryComponent`; tests: `tests/test_primary_decomposition_v212.py`, Rust `ideal::primary`.

- **V2-13 — Differential algebra / Rosenfeld–Gröbner:** Rust `rosenfeld_groebner`, `rosenfeld_groebner_with_options`, `dae_index_reduce`, `DifferentialRing` / `DifferentialIdeal` / `RegularDifferentialChain`, `DiffAlgError` (`alkahest_core::diffalg`); Python (`groebner`): `rosenfeld_groebner`, `dae_index_reduce`, `RosenfeldGroebnerResult`, `DaeIndexReduction`. Tests: `tests/test_diffalg_v213.py`, Rust `diffalg::tests`.

- **V2-14 — Numerical algebraic geometry:** Total-degree homotopy continuation in `ℂⁿ` with predictor–corrector tracking, Newton polish, conservative Smale heuristic, and `ArbBall` enclosures (`alkahest_core::solver::homotopy`); `solve_numerical`, `HomotopyOpts`, `CertifiedPoint`, `HomotopyError` (`E-HOMOTOPY-*`). Python (groebner): `solve(..., method="homotopy")`, `solve_numerical`, `CertifiedSolution`, benchmark task `numerical_homotopy`. Limitation: deficient systems (fewer roots than the Bézout bound) need a polyhedral start — not included. Tests: `tests/test_homotopy_v214.py`, Rust `solver::homotopy`.

### Developer experience

- **LaTeX / Unicode pretty-printing:** Pure-Python tree walk; `latex(expr)` emits `\sin\!\left(x\right)`, `\frac`, `\sqrt`, `\mathcal{O}` etc.; `unicode_str(expr)` emits `sin(x)² + cos(x)²` style. `Expr.node()` kernel hook for tree introspection. Exported from `alkahest.__all__`. 74 tests.

- **String expression parsing (`parse`):** Pratt recursive-descent parser in `python/alkahest/_parse.py`; `parse(source, pool, symbols=None) -> Expr`; supports integer/float literals, all 23 registered primitives, `^` / `**`, unary `-`, parentheses; `ParseError` (`E-PARSE-001`) with byte-level `.span`. 54 tests in `tests/test_parse.py`.

- **E-graph default rule completeness:** `simplify_egraph` now loads trig (`sin²+cos²→1`) and log/exp (`exp(log x)→x`) rules by default; opt-out via `EgraphConfig(include_trig_rules=False, include_log_exp_rules=False)`; `simplify_egraph_with(expr, config)` Python API.

- **Python API completeness:** `ExprPool.save_to(path)` / `load_from(path)` PyO3 bindings; `GroebnerBasis.compute(polys, vars)` static method; `solve()` returns `dict[Expr, Expr]` by default (`numeric=True` for float output); `IoError` exported from `alkahest`.

- **Windows + macOS CI parity:** `ci-cross.yml` matrix — `macos-14` (parallel + egraph + jit, FLINT via Homebrew) and `windows-2022` GNU (parallel + egraph, FLINT via MSYS2). `build.rs` Windows link-search branch added. Known limitation: `jit` excluded on Windows (inkwell pins LLVM 15; MSYS2 ships 17+).

## 1.0.0

### Features

- Integer Hermite / Smith normal forms (`IntegerMatrix`, FLINT HNF + pure-Rust SNF) and polynomial-matrix HNF/Smith over ℚ\[x\] (`RatUniPoly`, `PolyMatrixQ`); stable re-exports in `alkahest_core::stable`
- Exact LLL lattice reduction over ℤ (`alkahest_core::lattice::lattice_reduce_rows`; optional Lovász `δ`), plus an augmented-lattice + LLL heuristic for approximate integer relations (`guess_integer_relation` / Python `guess_relation` — **not** the Ferguson–Bailey PSLQ iteration); exposes `LatticeError` (`E-LAT-*`) and `PslqError` (`E-PSLQ-*`)
- Production NVPTX codegen for `sm_86` (Ampere): full inkwell-driven lowering, `libdevice.10.bc` linking, PTX emission via LLVM target machine, `cudarc 0.19` runtime — 16.2× speedup over CPU JIT on RTX 3090
- Gröbner-based polynomial system solver (`alkahest.solve`): Lex basis → triangular back-substitution → exact symbolic solutions including irrational roots (`sqrt(2)/2`)
- **V2-7 — Polynomial factorization:** FLINT-backed `fmpz_poly_factor` for ℤ\[x\] (Zassenhaus + van Hoeij), `fmpz_mpoly_factor` for multivariate ℤ, and `nmod_poly_factor` for 𝔽_p\[x\]; Rust `factor_univariate_z`, `factor_multivariate_z`, `factor_univariate_mod_p` + Python `UniPoly.factor_z`, `MultiPoly.factor_z`, `factor_univariate_mod_p`; `FactorError` (`E-POLY-008…010`)
- Custom `alkahest` MLIR dialect: `Sym`, `Const`, `Add`, `Mul`, `Pow`, `Call`, `Horner`, `PolyEval`, `SeriesTaylor`, `IntervalEval`, `RationalFn` ops; three lowering targets (ArithMath, StableHLO, LLVM); 1000-case round-trip proptest
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`): PTX elimination kernel, multi-prime CRT rational reconstruction, CPU fallback when no CUDA device present
- Semver-stable 1.0 API: `alkahest_core::stable` / `alkahest_core::experimental` split; `alkahest.__all__` freeze; `cargo semver-checks` + `scripts/check_api_freeze.py` in CI
- Primitive registry expanded to 23 primitives: added `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `erf`, `erfc`, `abs`, `sign`, `floor`, `ceil`, `round`, `atan2`, `gamma`, `min`, `max`
- Cross-CAS benchmark driver: Mathematica WolframEngine 14.3 and SymEngine 0.14 adapters; all six benchmark tasks implemented; nightly `--competitors` CI; per-competitor ratio columns in HTML report
- Persistent `ExprPool`: `save_to`, `load_from`, `open_persistent`, `checkpoint`; versioned binary format (`ALKP` magic); atomic temp-rename + fsync crash safety; all `ExprData` variants including `Piecewise` and `Predicate`

### Internal

- Structured errors across all subsystems: `.code`, `.remediation`, `.span` on every `AlkahestError` variant; `CudaError` (`E-CUDA-001…004`) and `SolverError` (`E-SOLVE-001…003`) added; PyO3 exception classes with typed attributes

## 0.5.0

### Features

- Lean 4 certificate exporter: pure-Rust, no FFI; 20+ rule→tactic mappings (`norm_num`, `simp`, `ring`, `rw`); `emit_lean_expr`, `emit_step`, `emit_goal`
- StableHLO / XLA bridge: pure-text MLIR emitter for `Add`, `Mul`, `Pow`, `sin`, `cos`, `exp`, `log`, `sqrt` → `stablehlo.*` ops via `to_stablehlo`
- Expanded Risch integration: exp/log tower + linear substitution; `∫ log(x) dx`, `∫ exp(a·x+b) dx`, `∫ c·x·exp(x) dx`, `∫ 1/(a·x+b) dx`; `is_linear_in` helper
- Branch-cut-aware log/exp simplification: `LogOfProduct` records `SideCondition::Positive` per factor; `SimplifyConfig::allow_branch_cut_rewrites`; `log_exp_rules_safe()` excludes `LogOfProduct`
- JAX primitive source integration: `to_jax` registers a symbolic expression with `def_impl`, `def_abstract_eval`, JVP rule (via symbolic grad), and vmap batching; graceful no-JAX fallback
- Parallel F4 Gröbner basis: Buchberger + product-criterion pruning; Rayon parallel S-poly reduction; `interreduce`; `Lex`/`GrLex`/`GRevLex` orders (`--features groebner`)

### Internal

- Structured errors MVP: `remediation()` and `span()` on `ConversionError` and `IntegrationError`
- Lean CI: GitHub Actions workflow generates 8 proof files and verifies via `lean` compiler; Mathlib build cached
- CUDA compute-sanitizer nightly: `memcheck` + `racecheck` on self-hosted `gpu-3090` runner; sanitizer logs uploaded as artifacts
- GPU benchmark suite: `GPUPolynomialEval` (1M-pt, 5-var), `GPUJacobian` (65k-pt), `DLPackZeroCopy`; `--gpu` flag added to `cas_comparison.py`

## 0.4.0

### Features

- Horner-form code emission: `horner(expr, var)`, `emit_c(expr, var, var_name, fn_name)`
- NumPy / JAX batch evaluation: `CompiledFn.call_batch_raw`, `numpy_eval` accepting NumPy, PyTorch, and JAX arrays via DLPack
- `collect_like_terms`: `2*x + 3*x → 5*x`
- `poly_normal`: polynomial normal form over given variables
- FLINT 3.x feature gate (`--features flint3`)
- Sharded `ExprPool`: concurrent insertion via `DashMap` (`--features parallel`)

### Internal

- GitHub Actions CI: tier-1 PR checks (< 10 min) + nightly integration (4–8 h) with AFL++ fuzzing, deep proptest, Valgrind, and SymPy oracle

## 0.3.0

### Features

- Reverse-mode automatic differentiation (`symbolic_grad`)
- Symbolic matrices and Jacobian (`Matrix`, `jacobian`)
- ODE representation and first-order lowering (`ODE`, `lower_to_first_order`)
- DAE structural analysis and Pantelides index reduction (`DAE`, `pantelides`)
- Acausal component modeling (`AcausalSystem`, `Port`, `resistor`)
- Sensitivity analysis: forward (`sensitivity_system`) and adjoint (`adjoint_system`)
- Hybrid system event handling (`HybridODE`, `Event`)
- LLVM JIT compiled evaluation (`compile_expr`, `CompiledFn`, `eval_expr`; `--features jit`, LLVM 15)
- Ball arithmetic (`ArbBall`, `AcbBall`, `interval_eval`) backed by Arb/FLINT
- Parallel simplification (`simplify_par`; `--features parallel`)
- Multivariate polynomial GCD via FLINT (`MultiPoly::gcd`, `RationalFunction::new`)

### Internal

- SymPy oracle cross-validation test suite for `integrate`
- E-graph vs rule-based Criterion benchmark (`bench_simplifier_comparison`)
- Rule engine hardening: trig/log rule sets, pattern rules, substitution, CI, AFL++ fuzzing

## 0.2.0

### Features

- E-graph equality saturation via egglog (`simplify_egraph`, `--features egraph`)
- Associative-commutative pattern matcher
- Forward-mode automatic differentiation (`diff_forward`)
- Rule-based integration: Risch subset (power, trig, exp/log table entries)
- `RationalFunction` arithmetic with multivariate GCD normalization

### Internal

- Pluggable e-graph cost functions: `SizeCost`, `DepthCost`, `OpCost`, `StabilityCost`; phased saturation via `node_limit` / `iter_limit`
- `PrimitiveRegistry` with `Capabilities` bitflags and `coverage_report()`; sin/cos/exp/log/sqrt registered
- `TracedFn`, `trace`, `grad`, `jit`, `trace_fn` Python transformation façade
- DLPack + `__array__` protocol on compiled functions
- `Piecewise` / `Predicate` expression nodes; diff/simplify/pattern/poly updated
- JAX-style pytree support (`flatten_exprs`, `unflatten_exprs`, `map_exprs`, `TreeDef`)
- `alkahest.context(pool=..., domain=..., simplify=True)` context manager
- Flat n-ary egglog: binary output flattened back to n-ary `Add`/`Mul` on extraction
- `canonicalize_linear` post-extraction pass
- Cross-CAS benchmark driver: HTML/JSONL report, Criterion dashboard

## 0.1.0

### Features

- Hash-consed expression DAG (`ExprPool`, `ExprId`): structural equality as pointer comparison, automatic subexpression sharing
- N-ary `Add` / `Mul` with AC normalization at construction
- Arbitrary-precision integers and rationals (FLINT/GMP)
- Symbol domains: `real`, `positive`, `nonnegative`, `integer`, `complex`
- Rule-based simplification with fixpoint iteration: identity elements, constant folding, polynomial normalization
- Symbolic differentiation with chain/product/quotient rules (`diff`)
- `UniPoly`: dense univariate polynomial backed by FLINT; GCD, degree, coefficients, arithmetic
- `MultiPoly`: sparse multivariate polynomial over ℤ
- `RationalFunction`: quotient with automatic GCD normalization
- PyO3 bindings for the full core API
- Derivation logs: ordered `RewriteStep` list on every `DerivedResult`
