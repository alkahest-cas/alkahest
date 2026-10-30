# Controls: LTI systems, Lyapunov and Riccati

`alkahest.controls` is a controls layer built on the exact kernel — `cancel`,
`solve` (with its `RootOf` roots), `Matrix` (`inverse`, `det`, `matrix_exp`) and the
inverse Laplace transform — so a plant may carry symbolic parameters (`m`, `b`, `k`,
a gain `K`) and every exact answer is exact. The main names are re-exported at the
top level: `TransferFunction`, `StateSpace`, `kronecker`, `lyapunov`, `care`, `lqr`,
`bode`, `margins`, `dlyap`, `dare`, `dlqr`, `controls_side_conditions`, `ControlsError`.

```python
import alkahest as ak
from alkahest import TransferFunction, StateSpace, controls_side_conditions

p = ak.ExprPool()
s, t, m, b, k = (p.symbol(n) for n in "s t m b k".split())

G = TransferFunction(1, m*s**2 + b*s + k, s)
G.poles()                    # two roots of m s² + b s + k (radicals)
controls_side_conditions()   # ['m ≠ 0']
sys = G.to_ss()              # controllable canonical form (scipy.signal.tf2ss layout)
sys.to_tf(s).equals(G)       # True
G.step_response(t)           # L⁻¹{G(s)/s}
```

## Transfer functions

`TransferFunction(num, den, var, dt=None)` is a SISO rational function of `var`.
Numerator and denominator are kept **as given**: common factors are not cancelled
behind the caller's back, because a pole–zero cancellation can hide an unstable mode.
(A single rational expression passed without `den` is brought to lowest terms.)
`from_coeffs(num, den, var)` takes coefficient lists highest power first, as scipy does.

| Method | Result |
|---|---|
| `poles()`, `zeros()` | Exact roots. Rational coefficients: square-free decomposition over ℚ, each factor solved by `solve` (radicals or `RootOf`), with multiplicity. Parametric coefficients: the distinct generic roots, with the solver's side conditions forwarded. |
| `dc_gain()` | `G(0)` (`G(1)` in discrete time) after cancelling common factors; `E-CTRL-004` for a remaining pole there |
| `series`, `parallel`, `feedback(H=1, sign=-1)`, `*`, `+`, `-`, `/` | Interconnection without cancellation: `feedback` is `n₁d₂ / (d₁d₂ − sign·n₁n₂)` |
| `minreal()` | Cancels common factors over ℚ(parameters); the cancelled factor is stated (`"(s + 1) ≠ 0"`) |
| `to_ss()` | Controllable canonical form, entry-for-entry `scipy.signal.tf2ss`; `E-CTRL-002` if improper |
| `step_response(t)`, `impulse_response(t)` | Inverse Laplace transform. A denominator with an irreducible factor of degree > 2 (rational coefficients) is expanded in partial fractions over ℚ and inverted by residues: an exact finite sum `Σₖ e^{rₖt}·Σᵢ cᵢ(rₖ)tⁱ` over `rₖ = RootOf(f, k)` (repeated factors included), evaluable numerically. Parametric high-order denominators still refuse (`E-TRANSFORM-002`) |
| `discretize(T, "tustin" \| "zoh")` | Tustin substitutes `s = (2/T)(z−1)/(z+1)`; ZOH goes through `StateSpace.discretize` |
| `d2c("tustin" \| "zoh")` | Inverse of `discretize`: Tustin is exact (`discretize(...).d2c(...)` equals the original); ZOH is exact when `Ad − I` is nilpotent (integrator chains) and numeric (float coefficients) otherwise |
| `subs({K: 2})`, `equals(other)`, `freqresp(w)` | Parameter substitution; proven equality of rational functions; numeric `G(jω)` |

## State space

`StateSpace(A, B, C, D=None, dt=None)` is MIMO; entries may be symbolic.

- `to_tf(s)` is `C (sI − A)⁻¹ B + D` over the common denominator `det(sI − A)`, not
  cancelled (as `scipy.signal.ss2tf`); a MIMO system returns rows of transfer functions.
- `poles()` are the roots of `det(sI − A)`; `dc_gain()` is `D − C A⁻¹ B` with `det A ≠ 0`
  stated when symbolic.
- `controllability_matrix()`, `observability_matrix()`, `controllability_rank()`,
  `observability_rank()`, `is_controllable()`, `is_observable()`. The ranks are
  **generic**: `controls_side_conditions()` lists the pivot hypotheses (`"a ≠ 0"` for an
  input gain `a`). The rank is exact where they hold and an upper bound everywhere.
- `series`, `parallel`, `feedback(K=None, sign=-1)` (a `StateSpace` or a static gain;
  `I − sign·D₁D₂` must be invertible).
- `discretize(T, "zoh")` is exact: `Ad = e^{AT}` and `Bd = ∫₀ᵀ e^{Aτ}dτ B` are read off
  `exp([[A, B], [0, 0]]·T)`, so no integral is needed. `"tustin"`/`"bilinear"` follows
  `scipy.signal.cont2discrete`'s bilinear formulas. `d2c("zoh" | "tustin")` inverts it.
- `minreal()` is the controllable-and-observable realisation. SISO goes through the transfer
  function (canonical form); MIMO (`method="kalman"`) is the exact Kalman decomposition:
  `range[B, AB, …]` is split off by an exact change of basis, then `ker[C; CA; …]` of what
  is left. Symbolic data eliminate over ℚ(parameters) with every pivot stated as a side
  condition. The result is verified — `D` and the Markov parameters `C Aᵏ B` agree with
  the original — so the transfer matrix is unchanged.

The same channel as `Matrix.inverse` applies to ranks generally:
`ak.Matrix([[a]]).rank()` is `1` and `ak.matrix_rank_side_conditions()` is `['a ≠ 0']`.

## Lyapunov, Kronecker, Riccati and LQR

- `kronecker(A, B)` — the Kronecker product, symbolic entries allowed.
- `lyapunov(A, Q)` solves `A X + X Aᵀ + Q = 0` (python-control's `lyap` convention;
  `scipy.linalg.solve_continuous_lyapunov(A, -Q)` is the same `X`). Exact: the system
  rational data of any size use a structured solve (Cayley–Hamilton turns the Sylvester
  operator into a polynomial in `A` and `Aᵀ`: `O(n)` matrix products over ℚ, no `n²×n²`
  system); symbolic data vectorise `(I ⊗ A + A ⊗ I) vec X = −vec Q`; either way the
  residual is **verified** to cancel. The operator is singular iff two eigenvalues satisfy
  `λᵢ + λⱼ = 0` — a symbolic determinant is a side condition (`"(4·k·b²) ≠ 0"` for
  `A = [[0, 1], [−k, −b]]`), a proven singular one is `E-CTRL-004`. Symbolic data are exact
  up to 3×3; numeric (numpy) with `method="numeric"` or float data.
- `dlyap(A, Q)` solves the Stein equation `A X Aᵀ − X + Q = 0` (python-control's `dlyap`):
  rational data go through the Cayley transform `F = (A − I)(A + I)⁻¹` to the continuous
  equation, residual verified; `E-CTRL-004` when two eigenvalues multiply to 1.
- `care(A, B, Q, R)` returns the stabilising solution of
  `Aᵀ X + X A − X B R⁻¹ Bᵀ X + Q = 0`; `lqr` returns `(K, X)` with `K = R⁻¹ Bᵀ X`.
  For `n ≤ 2` with exact data the polynomial system is solved with `solve`, each
  candidate's closed loop `A − B K` gets its Routh–Hurwitz conditions, and the branch
  whose conditions hold at a sample point of the parameters (`sample={r: 2}`; default
  every parameter in `(1, 2)`) is returned with those conditions **stated** — the result
  is the stabilising solution exactly where they hold. Otherwise the Hamiltonian's stable
  invariant subspace is computed numerically (ordered Schur through scipy when installed).
  For rational data of **any** size the subspace is computed exactly when the Hamiltonian's
  characteristic polynomial factors over ℚ into pieces that each lie on one side of the
  imaginary axis (an exact Routh test per irreducible factor): the kernels of `f(H)^m` for
  the stable factors give `[U₁; U₂]` and `X = U₂U₁⁻¹` is rational and verified. A factor
  with roots on both sides (the generic case, eigenvalues `±√c`) has no rational stable
  subspace: `method="exact"` refuses (`E-CTRL-005`) and `"auto"` returns the numeric Schur
  solution, stating the fallback in `controls_side_conditions()`.
- `dare(A, B, Q, R)` / `dlqr` solve the discrete Riccati equation numerically (symplectic
  stable subspace; scipy for a singular `A`).

```python
r, N = p.symbol("r"), p.integer                   # double integrator, R = r
A = ak.Matrix([[N(0), N(1)], [N(0), N(0)]])
B = ak.Matrix([[N(0)], [N(1)]])
Q = ak.Matrix([[N(1), N(0)], [N(0), N(1)]])
K, X = ak.lqr(A, B, Q, [[r]])
# X12 = √r, X22 = √(r(1 + 2√r)), K = [1/√r, √((1 + 2√r)/r)]
ak.controls_side_conditions()                     # ['r ≠ 0', '… r … > 0', …]
```

## Frequency response

`bode(sys, w)` returns `(magnitude_dB, phase_deg)` with the phase unwrapped, as
`scipy.signal.bode`. `margins(sys)` returns `Margins(gain_margin, phase_margin,
phase_crossover, gain_crossover)` with `control.margin`'s conventions (absolute gain
margin, `inf`/`nan` when there is no crossing). The crossovers are the real roots of
`Im[N(jω) D(−jω)]` and `|N(jω)|² − |D(jω)|²`; for rational coefficients they are isolated
exactly (Sturm sequences, exact bisection) — `K/(s(s+1)(s+5))` has its phase crossover at
`√5` and gain margin `30/K`. Both need numeric coefficients (`E-CTRL-003`).

Discrete-time systems (`dt` set) are evaluated on the unit circle `z = e^{jωT}`, `ω ≤ π/T`:
with `c = cos θ` both crossover conditions are real polynomials in `c` (Chebyshev `U`/`T`
expansions of `Im[N(z)D(z̄)]` and `|N|² − |D|²`) isolated exactly on `[−1, 1]`. `z = ±1` are
always on the real axis and count as phase crossovers, which python-control ignores.

## Refusals

| Code | Meaning |
|---|---|
| `E-CTRL-001` | Malformed input: not rational in the variable, shapes that do not conform, MIMO where SISO is needed |
| `E-CTRL-002` | An improper transfer function where a proper one is needed (realisation, impulse response) |
| `E-CTRL-003` | A numeric routine on data with free parameters — substitute values first |
| `E-CTRL-004` | The object does not exist: singular Lyapunov operator, no stabilising ARE solution, infinite DC gain, ill-posed feedback |
| `E-CTRL-005` | Beyond the exact engine (symbolic size, unsolvable system) — use `method="numeric"` |
| `E-CTRL-006` | numpy is required for the numeric route and is not installed |
