# Alkahest Agent Skill

Use this skill whenever you are writing Python code that uses the `alkahest` library, or Rust code using the `alkahest-cas` crate.

## Official links

- **Repository:** [github.com/alkahest-cas/alkahest](https://github.com/alkahest-cas/alkahest)
- **Website:** [alkahest-cas.github.io/](https://alkahest-cas.github.io/)
- **Documentation:** [alkahest-cas.github.io/alkahest/](https://alkahest-cas.github.io/alkahest/)
- **API reference:** [alkahest-cas.github.io/alkahest/api/](https://alkahest-cas.github.io/alkahest/api/)
- **Playground:** [alkahest-cas.github.io/playground/](https://alkahest-cas.github.io/playground/)
- **RL environment:** [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) (`alkahest/alkahest-symbolic-integration`)
- **Further reading:** [`ARCHITECTURE.md`](https://github.com/alkahest-cas/alkahest/blob/main/ARCHITECTURE.md), [`CONTRIBUTING.md`](https://github.com/alkahest-cas/alkahest/blob/main/CONTRIBUTING.md), [`TESTING.md`](https://github.com/alkahest-cas/alkahest/blob/main/TESTING.md), [`examples/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/)

## Install

**Requirements:** Python **3.9–3.13** ([PyPI](https://pypi.org/project/alkahest/) `requires-python`).

```bash
pip install alkahest
```

**RL environments** (symbolic integration tasks for Prime Intellect / veRL): Python **≥ 3.10** required.

```bash
pip install "alkahest[rl]"
```

For an isolated environment (recommended when juggling versions or building from source):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install alkahest
```

Default PyPI wheels (Linux, macOS, Windows) ship the **vendored egglog** e-graph backend (`egraph`), the **Gröbner solver** (`groebner` — so `alkahest.solve`, Diophantine, homotopy and related APIs work out of the box), the **Cranelift JIT** (`cranelift`, since 3.6.0 — so `jit_is_available()` is `True` on a plain `pip install alkahest`), and **`parallel`** (since 3.8.0 — the sharded `ExprPool`, parallel F4, and the `*_par` entry points are genuinely multi-core). They do **not** include the LLVM JIT (`jit`); for that use a **PyTorch-style** opt-in wheel (separate artifact / index), not the default PyPI resolver path.

Note that the Cargo `default` feature set (`egraph`, `groebner`) is narrower than what release CI builds into the published wheels (`egraph`, `groebner`, `cranelift`, `parallel`). A plain `cargo build` or `maturin develop` therefore has **no** JIT and **no** threads unless you pass `--features "cranelift parallel"`. Never infer the active feature set from the version number — probe it:

```python
import alkahest as ak
caps = ak.capabilities()
caps["features"]["cranelift_jit"]   # True on default wheels, False in a bare source build
caps["features"]["parallel"]        # likewise
ak.jit_is_available()
```

### Opt-in Linux wheels: `+jit` and `+full` (PyTorch-style)

Since 3.6.0 the default wheel already has a JIT (Cranelift), and since 3.8.0 it also has `parallel`, so `+jit` / `+full` now buy you only the **LLVM** backend — not "JIT vs no JIT" and no longer "threads vs no threads". Most agent code does not need them.

**Why a separate index or direct wheel URL:** feature-heavy wheels use a PEP 440 **local version** (for example `3.8.0+jit` or `3.8.0+full`). Those builds **must not** be mixed into the main PyPI project’s simple API for the same reason PyTorch publishes CUDA wheels on `download.pytorch.org`: otherwise `pip install alkahest` could resolve a `+jit` / `+full` build as “newer” than `3.8.0` and pull LLVM (or a much larger binary) when you wanted the default wheel.

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries for the same wheel slot.

**Until a dedicated PEP 503 simple index is published**, tagged releases attach Linux **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`, not the manylinux image used for default wheels). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---------------|----------------|-------------|
| `+jit` | `egraph groebner jit parallel` | LLVM CPU JIT in place of Cranelift; everything else matches the default wheel. |
| `+full` | `egraph groebner jit parallel` | Identical to `+jit` since `parallel` moved into the default wheel. |

Direct-install examples (adjust tag and filename after checking the release assets):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.8.0/alkahest-3.8.0+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.8.0/alkahest-3.8.0+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM (for JIT) and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages).

If your client chokes on `+` in the URL, use percent-encoding (`3.8.0%2Bfull` in the filename segment).

After installing `+jit` or `+full`, `capabilities()["features"]["llvm_jit"]` should be `True` (`jit_is_available()` is already `True` on the default wheel via Cranelift, so it does not distinguish the builds — check `llvm_jit`, which is the only remaining difference; `parallel` is `True` in all three). Gröbner-backed APIs such as `alkahest.solve` are available in **all** wheels (including the default PyPI wheel) since `groebner` became a default feature.

*macOS and Windows `+jit` / `+full` wheels are not produced in CI yet (LLVM / MSYS2 constraints); use [building from source](#from-source) there.*

**Target layout (roadmap):** a small **extra index** URL (PEP 503) hosting only `+jit` / `+full` wheels, mirroring PyTorch’s `--extra-index-url` workflow:

```bash
pip install 'alkahest==3.8.0+full' --extra-index-url https://EXAMPLE/alkahest-extras/simple
```

### From source

Required to enable optional features (`jit`, `cuda`, `parallel`) or for development. The `groebner` and `egraph` features are already built into default wheels; a source build inherits them automatically. Prerequisites:

- **Rust** stable ≥ 1.76 and nightly:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- **uv** (recommended Python tool manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **LLVM 15**: `apt install llvm-15 libllvm15 llvm-15-dev` / `brew install llvm@15`
- **FLINT ≥ 2.9** (includes GMP and MPFR): `apt install libflint-dev` / `brew install flint`

```bash
# Install dev tools (maturin, pytest, ruff, ty, …) without building the Rust extension:
uv sync --no-install-project --group dev
# Build and install the extension into the project venv:
uv run maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

Without `uv`, install maturin directly and run the same develop command:

```bash
pip install maturin
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

Optional Cargo features: `parallel` (sharded pool + parallel F4 + `numpy_eval_par`), `egraph` (vendored egglog backend; **default** in PyPI wheels), `groebner` (Gröbner solver + Diophantine + homotopy; **default** in both the Rust crate and PyPI wheels), `cranelift` (pure-Rust Tier-1 JIT; **shipped in PyPI wheels** but *not* in the Cargo `default` set — pass it explicitly in a source build), `jit` (LLVM JIT), `cuda` (NVPTX codegen — needs LLVM 15 with the NVPTX target), `groebner-cuda` (CUDA Macaulay-matrix kernel — needs only `cudarc`).

**GPU:** neither CUDA feature is in any published wheel, so `pip install alkahest` has
no GPU support. On a `--features cuda` source build, `ak.compile_cuda(expr, [x, y])`
returns a `CudaCompiledFn` with `.ptx` / `.n_inputs` / `.call_batch([xs, ys])`; the
name does not exist otherwise, so branch on
`ak.capabilities()["features"]["cuda"]` rather than calling it and catching
`AttributeError`. There is **no** `features["groebner_cuda"]` key: the CUDA
Macaulay-matrix kernel has no Python binding at all, so the bit could neither
be confirmed nor refuted from Python and was removed in contract v3. Nothing
about `solve` changes on a `--features groebner-cuda` build.

### Rust crate

`alkahest-cas` is also published on [crates.io](https://crates.io/crates/alkahest-cas) ([docs.rs](https://docs.rs/alkahest-cas)) for use directly from Rust without a Python runtime:

```toml
[dependencies]
alkahest-cas = "3"

# groebner is included by default; add other optional features as needed:
# alkahest-cas = { version = "3", features = ["parallel", "egraph"] }
```

**System prerequisites** (same libraries as the Python build — must be present before `cargo build`):

```bash
# Debian / Ubuntu
sudo apt-get install -y libflint-dev libgmp-dev libmpfr-dev

# macOS
brew install flint
```

The `jit` feature additionally requires LLVM 15 dev headers (`apt install llvm-15-dev` / `brew install llvm@15`). A self-contained runnable example is in [`examples/rust_quickstart/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/rust_quickstart/).

---

## Core mental model

Every expression lives in an **`ExprPool`** (a hash-consed DAG). You must create a pool before making any symbolic expression. **Python `int` and `float` literals work in arithmetic** (`x + 1`, `x * 2.5`, `x**2`); use `pool.rational(p, q)` for exact rationals and `pool.integer(n)` when you need an explicit `Expr` constant (e.g. for APIs that only accept `Expr`).

```python
import alkahest as ak
from alkahest import sin, cos, exp, log, sqrt, diff, integrate, simplify, simplify_trig

caps = ak.capabilities()  # probe the installed native build once per session
features = caps["features"]  # includes llvm_jit, cranelift_jit, cuda, and parallel

pool = ak.ExprPool()
x = pool.symbol("x", ak.Domain.Real)   # or domain="real"
y = pool.symbol("y")   # ambient context(domain=…) if one is open, else Domain.Real

expr = x**2 + 1          # int literals in +, -, *, **, / are fine
half = pool.rational(1, 2)  # exact rationals need pool.rational
```

Arithmetic operators (`+`, `-`, `*`, `**`, `/`) are all overloaded on `Expr` — use them freely.

### Expression representations

| Type | Description |
|---|---|
| `Expr` | Generic hash-consed symbolic expression |
| `UniPoly` | Dense univariate polynomial (FLINT-backed) |
| `MultiPoly` | Sparse multivariate polynomial over ℤ |
| `MultiPolyFp` | Sparse multivariate polynomial over 𝔽ₚ (modular arithmetic) |
| `RationalFunction` | Quotient of polynomials with GCD normalization |
| `ArbBall` | Real interval with rigorous error bounds (Arb) |

Representation types are explicit — no silent performance cliffs. Conversion between them is always an opt-in call (`UniPoly.from_symbolic(...)`, etc.).

---

## Return type: `DerivedResult`

**Most** top-level operations return a `DerivedResult` — but not all. Check before
you reach for `.value`:

| Operation | Returns |
|---|---|
| `diff`, `integrate`, `simplify*`, `sum_*`, `product_*`, `resultant`, … | `DerivedResult` |
| `limit` | plain `Expr` — **no `.value`** |
| `series` | `Series` object — **no `.value`** |
| `solve` | `list[dict[Expr, Expr]]` (or `GroebnerBasis`) |
| `evaluate` | `EvaluationResult` |
| `real_roots` | `list[RootInterval]` |
| `symbolic_grad` | `list[Expr]` |

`DerivedResult` fields:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.value` | `Expr` | The result expression |
| `.steps` | `list[dict]` | Rewrite log; each step has `"rule"`, `"before"`, `"after"` keys |
| `.verification` | `dict` | Evidence status, emitted artifact format, external-check status, and side conditions |
| `.certificate` | `str \| None` | Generated Lean 4 `.lean` source; generation is not Lean proof checking |
| `to_lean(result)` | `str` | Same as `.certificate`; also accepts `Expr` (runs `simplify` first) |
| `.to_dict(mode=…)` / `.to_json(mode=…)` | `dict` / `str` | Versioned envelope with a `"kind": "alkahest.derived_result"` discriminator. `mode="compact"` drops step `before`/`after` text and shortens keys, but **never** hides `verification["status"]` and never includes Lean source. Use this to carry a result out of a pool's lifetime, and in agent context windows. |

```python
caps = ak.capabilities()
if not caps["groebner"]:
    # Select a non-Gröbner strategy before calling solve().
    pass

result = diff(sin(x**2), x)
print(result.value)   # 2*x*cos(x^2)
print(result.steps)   # list of rewrite-rule dicts
evidence = result.verification
if evidence["status"] == "certificate_available":
    print(result.certificate)  # Lean source; check it with Lean separately
```

For indefinite integrals, distinguish `"exactly_verified"` (the symbolic
derivative residual is zero) from `"numerically_checked"` (the integration
gate passed floating-point samples only). Never treat the latter as an exact
or Lean-checked proof.

### Lean certificate coverage

`.certificate` is Lean 4 source against Mathlib; alkahest **generates** it but does
not run Lean, so "certificate available" ≠ "proved". Run `lake build` on it
yourself if you need the proof checked.

| Operation | Certificate |
|---|---|
| `diff` | Yes — chain rule, log/sqrt/tan, quotient |
| `integrate` (indefinite) | Yes |
| `integrate` (definite) | Yes, **since 3.8.0** (Mathlib FTC / interval-integral lemmas) |
| exp/log identities | Yes, assumption-gated |

Certificates that do not typecheck are **withheld** rather than emitted broken, so
`.certificate is None` means "no proof available", not "proof failed silently".
Because coverage is version-dependent, always branch on the value rather than
assuming it is present:

```python
r = integrate(x**2, x, pool.integer(0), pool.integer(1))
if r.certificate is not None:
    Path("cert.lean").write_text(r.certificate)
```

---

## Simplification

```python
from alkahest import (
    simplify,            # general algebraic simplification
    simplify_trig,       # sin²+cos²=1, sin(-x)=-sin(x), …
    simplify_log_exp,    # log(exp(x))=x, exp(log(x))=x, …
    simplify_expanded,   # expand and collect
    simplify_with,       # simplify with a custom RewriteRule list
    simplify_par,        # parallel simplification (thread-pool)
    simplify_egraph,     # e-graph (egglog) simplification
    simplify_egraph_with,# e-graph with custom EgraphConfig
    collect_like_terms,  # x+x+2x → 4x
    poly_normal,         # normalize to polynomial form (raises ConversionError if not poly)
)

r = simplify(x + 0)                        # → x (algebraic rules)
r = simplify_trig(sin(x)**2 + cos(x)**2)  # → 1  (trig identities — use this, not simplify)
r = simplify_log_exp(log(exp(x)))          # → x
r = collect_like_terms(x + x + 2*x + y)   # → 4*x + y
```

**Simplifier choice:** `simplify` is a general algebraic rewriter; it does **not** apply trig identities. For `sin²+cos²` and similar, use `simplify_trig` (or `simplify_egraph`). For `log`/`exp` laws, use `simplify_log_exp`.

### Assumptions gate the conditional rewrites

`simplify_log_exp(log(exp(x)))` → `x` is unconditionally valid, so it always fires.
The **reverse** direction is not: `exp(log(x)) = x` only holds for `x > 0`, so it
stays unevaluated unless positivity is known. This is a deliberate refusal, not a
missing rule — do not work around it by string-rewriting the result.

Two ways to supply the fact:

```python
from alkahest import Assumptions, Domain

# 1. Declare it on the symbol
xp = pool.symbol("xp", Domain.Positive)
simplify_log_exp(exp(log(xp))).value      # → xp

# 2. Pool-scoped assumption context
asm = Assumptions(pool)                    # takes the pool
asm.refine(pool.gt(x, pool.integer(0)))    # add a predicate
simplify_log_exp(exp(log(x)), assumptions=asm).value   # → x
asm.predicates                             # attribute, not a method
```

Only positive and non-zero facts authorize conditional rewrites; other predicates
are recorded for contradiction detection.

`simplify_egraph` / `simplify_egraph_with` run egglog e-graph saturation — use when algebraic rewriting is insufficient.

```python
from alkahest import EgraphConfig, HAS_EGRAPH

if HAS_EGRAPH:
    cfg = EgraphConfig(node_limit=50_000, iter_limit=20)
    r = simplify_egraph_with(expr, cfg)
```

---

## Differentiation

### `diff` vs `symbolic_grad` vs `grad` (keep these names)

| API | Input | Output | When to use |
|-----|--------|--------|-------------|
| **`diff`** | one `Expr`, one variable | `DerivedResult` (`.value`, `.steps`) | Single derivative; need derivation log |
| **`symbolic_grad`** | one `Expr`, `list[Expr]` vars | `list[Expr]` | All partials of one expression at once |
| **`grad`** | `TracedFn` from `@trace` | `GradTracedFn` → floats at a point | JAX-style pipeline; compose with `jit` |

Do **not** pass an `Expr` to `grad` — use `symbolic_grad` or `diff`. Do **not** pass a `TracedFn` to `symbolic_grad`.

```python
from alkahest import diff, diff_forward, symbolic_grad, jacobian

# Single variable + step log
d = diff(sin(x**2), x)          # DerivedResult; d.value = 2*x*cos(x^2)

# Forward-mode AD (cross-check)
d_fwd = diff_forward(x**2, x)

# All partials of one Expr (no TracedFn)
grads = symbolic_grad(x**2 + y**2, [x, y])  # list[Expr]: [2*x, 2*y]

# Vector-valued → matrix
J = jacobian([x**2 + y, sin(x)*y], [x, y])
entry = J.get(row, col)  # Expr
```

---

## Integration

```python
from alkahest import integrate, IntegrationError

# Basic rules
r = integrate(x**2, x)    # DerivedResult; r.value = x^3/3
r = integrate(sin(x), x)  # → -cos(x)
r = integrate(exp(x), x)  # → exp(x)
r = integrate(x**-1, x)   # → log(x)

# Rational functions — full Risch: Hermite + Rothstein–Trager + arctan + RootSum
r = integrate(pool.integer(1) / (x**2 - pool.integer(1)), x)
# → ½·(log(x−1) − log(x+1))

r = integrate(pool.integer(1) / (x**2 + pool.integer(1)), x)
# → arctan(x)

r = integrate(pool.integer(1) / (x + pool.integer(1))**2, x)
# → −1/(x+1)   (Hermite reduction for repeated factor)

# Degree-≥3 denominator → RootSum (Lazard–Rioboo–Trager)
r = integrate(pool.integer(1) / (x**3 - pool.integer(3)*x + pool.integer(1)), x)
# r.value is a RootSum node: Σ_{P(c)=0} c·log(gcd_x(numer − c·denom', denom))

# Rational coefficient × exp (rational Risch DE)
r = integrate((x - pool.integer(1)) / x**2 * exp(x), x)
# → exp(x)/x

# Non-elementary integrals raise IntegrationError with code E-INT-004
try:
    integrate(exp(x) / x, x)          # Ei function — non-elementary
except IntegrationError as e:
    print(e.code)         # E-INT-004
    print(e.remediation)  # "no elementary antiderivative (NonElementary)"

try:
    integrate(exp(x**2), x)            # Gaussian — non-elementary
except IntegrationError as e:
    print(e.code)         # E-INT-004
```

---

## Limits and series

Both take `point` as an **`Expr`** (a Python `0` raises `TypeError`), and **neither
returns a `DerivedResult`** — there is no `.value` to read.

```python
from alkahest import limit, series

limit(sin(x) / x, x, pool.integer(0))              # → Expr: 1
limit(pool.integer(1) / x, x, pool.integer(0), dir="+")   # one-sided

s = series(exp(x), x, pool.integer(0), 4)          # → Series
s.expr    # ((1 * 1) + (x * 1) + (1/2 * x^2) + (1/6 * x^3) + O(x^4))
```

`Series` exposes a single attribute, `.expr`, which retains the `O(...)` term —
strip or truncate it before feeding the result into numeric evaluation.

For asymptotics and multivariate limits, see `experimental.asymptotic_expand` and
`experimental.multilimit`.

---

## Logic and real quantifier elimination

```python
from alkahest import And, Or, Not, Exists, Forall, decide, satisfiable, CadError

# Predicates come from the pool, not Python comparison operators
pos = pool.gt(x, pool.integer(0))
lt1 = pool.lt(x, pool.integer(1))
# pool.ge, pool.le, pool.pred_eq, pool.pred_ne, pool.pred_and, pool.pred_or,
# pool.pred_not, pool.pred_true, pool.pred_false

satisfiable(And(pos, lt1))     # {'x': '1/2'} — witness as a rational string
                               # False if unsat, True if sat without a witness,
                               # None if the fragment is unsupported

decide(Forall(x, pool.ge(x**2, pool.integer(0))))   # (True, None)
# decide takes ONE bound symbol (not a list) and returns (truth, witness_or_none)
# ...OR RAISES CadError. See below — this is not an optional detail.

# Cylindrical algebraic decomposition primitives
from alkahest import cad_project, cad_lift
```

### `decide` is NOT complete — it refuses (`E-CAD-001`)

Always wrap `decide` in `try/except ak.CadError`. It covers polynomial bodies over ℚ in
**at most two real variables** with a quantifier prefix of **at most two**; anything
outside that raises `E-CAD-001`. Inside the fragment there is a second refusal that
matters more:

The CAD sample set is made of rational points. For a **strict** atom (`<`, `>`) that is
complete, because strict solution sets are open. For a **non-strict** atom (`=`, `≠`,
`≤`, `≥`) the solution set can be a single boundary point, and if that point is
irrational it is never sampled. Rather than report an unsatisfiability it never checked
there — which via `∀x. φ ≡ ¬∃x. ¬φ` would become a proof of a *false universal theorem* —
`decide` refuses.

```python
# rational double root at x = -2/3: found exactly, real verdict
body = pool.gt((pool.integer(3)*x + pool.integer(2))**pool.integer(2), pool.integer(0))
decide(Forall(x, body))                      # (False, None)

# irrational double root at ±sqrt(2): refuses
irr = pool.gt((x**pool.integer(2) - pool.integer(2))**pool.integer(2), pool.integer(0))
try:
    decide(Forall(x, irr))
except CadError as e:
    print(e.code)                            # E-CAD-001
```

Rules for agents:

- **`E-CAD-001` means "undecided", never "false".** Do not report it to the user as a
  disproof, and do not record it as a closed branch in a search.
- **Witnesses are verified.** `(True, {...})` means the point was substituted back and
  checked. `∃x. 3x−2=0` → `(True, {'x': '2/3'})`; `∃x. x²=2` → `(True, None)`, because no
  *rational* witness exists. A `None` witness with a `True` verdict is normal, not a bug.
- **Mixed alternation refuses more often.** `∀x∃y. p > 0` is decided via `¬∃x∀y. p ≤ 0`,
  and De Morgan turns a strict body non-strict.
- **Both 3.7 bugs here were silent errors.** Through 3.7, `∀x. (3x+2)² > 0` returned
  `True` (it is false at `x = −2/3`), and existential witnesses were interval midpoints
  that did not satisfy the sentence. If you have `decide` results from 3.7, re-run them.

Escalation when `decide` refuses: `sos_decompose` / `prove_nonneg` for a positivity
certificate, `alkahest.smt` (z3's `nlsat` is complete over the reals), or
`bound_on_box` / `verified_sign` if a rigorous statement over a box is enough.

**Check that route before you build the workload: `ak.bounds_supported(expr)`.** The
validated-bounds entry points (`bound_on_box`, `verified_integral`,
`verified_no_roots`, `verified_sign`) reach the elementary fragment only — `sin`, `cos`,
`tan`, `exp`, `log`, `sqrt`, `abs`, the inverse-trig and hyperbolic functions
(including `asinh`/`acosh`/`atanh`), plus `erf` and `erfc` — and refuse everything else
with `E-VALIDATED-001`: `bessel_j0/j1`, `digamma`, `lambert_w`, `gamma`, the elliptic
integrals, `floor`/`ceil`, and the two-argument `atan2`.
**`capabilities()["primitives"][i]` carries this as `taylor_model`; do not read
`numeric_ball` as the coverage flag** — it is pointwise ball arithmetic, it is `True`
for `bessel_j0` and `digamma`, and it says nothing about whether a bound can be
certified. `bounds_supported` answers for a whole
expression without running anything, and names the blocking functions:

```python
answer = ak.bounds_supported(ak.bessel_j0(x) * x)
bool(answer), answer.functions   # (False, ['bessel_j0'])
```

A `True` means "not `E-VALIDATED-001`"; a bad box can still refuse with
`E-VALIDATED-003` (domain violation) or `-004` (non-finite enclosure).

**`E-SOS-002` from that escalation is `unknown`, not "not SOS".** The SOS search covers
an LP-representable subcone of the PSD cone at one `basis_degree` (one `level` for
Handelman), so a refusal is compatible with `p` being SOS outside that subcone, SOS at a
higher degree, *or* non-negative without being SOS (Motzkin, Choi–Lam, Robinson all
refuse this way). Retry with a higher `basis_degree` / `level`, or fall back to `decide`
or `smt`; do not report it as a disproof and do not close the branch. `E-SOS-003` — which
carries a witness point where `p < 0` — is the only SOS refutation.

---

## Substitution and pattern matching

```python
from alkahest import subs, match_pattern, make_rule

# Substitute: values may be Expr, DerivedResult, or Python int/float (coerced to Expr)
result = subs(expr, {x: 2, y: cos(x)})

# Pattern matching
# Pattern *arguments are `Expr`, not strings.* Any symbol in the LHS acts as a
# wildcard that binds to whatever subterm appears in that position.
a = pool.symbol("a")
rule = make_rule(sin(a) ** 2 + cos(a) ** 2, pool.integer(1))
simplified = simplify_with(expr, [rule])
```

---

## Polynomial types (FLINT-backed)

All polynomial types are explicit opt-in — no silent performance cliffs.

```python
from alkahest import UniPoly, MultiPoly, RationalFunction

# Dense univariate polynomial
p = UniPoly.from_symbolic(x**3 + pool.integer(-2)*x + pool.integer(1), x)
p.degree           # 3
p.coefficients()   # [1, -2, 0, 1]  (constant first)
# `UniPoly` has no numeric-eval method.  Its full surface is:
#   from_symbolic, from_coefficients, coefficients, degree, is_zero, gcd, factor_z
# To evaluate numerically, evaluate the original `Expr` instead:
eval_expr(x**3 + pool.integer(-2) * x + pool.integer(1), {x: 2.0})   # 5.0

# GCD
a = UniPoly.from_symbolic(x**2 + pool.integer(-1), x)
b = UniPoly.from_symbolic(x + pool.integer(-1), x)
a.gcd(b)           # x - 1

# Sparse multivariate polynomial (over ℤ)
mp = MultiPoly.from_symbolic(x**2 * y + x * y**2, [x, y])
mp.total_degree    # 3

# Rational function (GCD-normalized automatically)
rf = RationalFunction.from_symbolic(x**2 + pool.integer(-1), x + pool.integer(-1), [x])
# rf displays as x + 1
```

---

## Polynomial system solver / Gröbner basis

The `groebner` Cargo feature is **included in all PyPI wheels** since 2.3.1 (it is a default feature). No special build flag or `ImportError` guard needed — `solve`, `GroebnerBasis`, and related APIs are available after a plain `pip install alkahest`.

```python
from alkahest import solve, solve_numerical, GroebnerBasis, GbPoly

# solve(equations, vars, *, numeric=False, method="groebner")
# - method="groebner" (default): Lex/triangular path. Each finite solution is a dict
#   mapping variable Expr → Expr (symbolic) or float if numeric=True.
# - Free symbols omitted from vars are parameters (e.g. solve([x**2 - y], [x]) → ±√y).
# - method="homotopy": numerical continuation in ℂⁿ; dict values are float.
# - If the ideal is underdetermined / not zero-dimensional finite, Groebner mode may
#   return a GroebnerBasis instead of a list of dicts.
solutions = solve(
    [x**2 + y**2 + pool.integer(-1), y + pool.integer(-1)*x],
    [x, y],
)
for s in solutions:
    xv, yv = s[x], s[y]

# Certified enclosures / residuals: solve_numerical(eqs, vars, ...)
```

### Coefficient fields for elimination: `Q(params)` (M9, experimental)

`GroebnerBasis.compute(polys, vars, params=[...])` puts the listed symbols in the **coefficient field** instead of the ring — they never enter the monomial order or generate S-pairs. This is the difference between eliminating states from `Q[states, Y, params]` and `Q(params)[states, Y]`: for a differential-elimination / structural-identifiability problem (states eliminated from an ODE model's jet equations, rate constants left symbolic), the parametric route can be an order of magnitude faster and produce far fewer basis generators than treating the parameters as ring variables — see `docs/mdbook/src/solving.md` for measured numbers on a worked example.

```python
gb = alkahest.GroebnerBasis.compute([a*x - y, x + y - one], [x, y], params=[a])
type(gb)                       # ParametricGroebnerBasis (alkahest.experimental)
[g.to_expr() for g in gb]      # coefficients are rational functions of a

gb.conditions()                # [a + 1] — hypersurfaces the basis assumed non-zero
gb.is_regular_at([-1])         # False
gb.specialize([3])             # ordinary GroebnerBasis over Q
gb.specialize([-1])            # raises ParamGroebnerError, code "E-PARAMGB-004"
```

**The basis is generic, not universal.** A leading coefficient in `Q(params)` can be non-zero as a rational function and still vanish at a specific parameter point; `conditions()` reports that locus (sufficient, not necessary — it can flag a point that turns out fine, never miss one that is genuinely wrong), and `specialize` refuses on it rather than returning something that is not a basis. Reads back the same way `GroebnerBasis` does — `ParametricGbPoly.to_expr()` / `.terms()`, `ParametricGroebnerBasis.to_exprs()` — so nothing here is write-only. `eliminate(vars)` has the same `Lex`-eliminated-variables-first contract and refuses to eliminate a coefficient-field parameter. `alkahest.experimental.ParametricGroebnerBasis.compute(polys, vars, params, order=None)` is the equivalent direct constructor. Requires `--features groebner` (default in all PyPI wheels); the class itself is experimental.

---

## JIT compilation and numeric evaluation

```python
from alkahest import compile_expr, eval_expr, CompiledFn, CompileCache, jit_is_available

jit_is_available()   # True on current default PyPI wheels (Cranelift ships by
                     # default since 3.6.0); False only for a source build
                     # without `cranelift`/`jit`.  Always probe, never assume.

# Compile (Cranelift on default wheels; LLVM with --features jit; tree-walking
# interpreter if the build has neither)
f = compile_expr(x**2 + pool.integer(1), [x])   # CompiledFn
f([3.0])          # → 10.0   (list in, *scalar float* out — not a list)
f.n_inputs        # 1

# Memoize repeated compilations within a session
cache = CompileCache()
f = cache.compile(x**2, [x])   # signature is (expr, inputs) — no pool argument
print(cache.stats())   # hits, compiles, hit_rate

# Interpreter (no JIT)
val = eval_expr(x**2 + y, {x: 3.0, y: 1.0})  # float

# Unified evaluation API — returns an EvaluationResult, not a bare float.
# Prefer this when you need to know *how* the number was produced.
r = ak.evaluate(x**2 + pool.integer(1), {x: 3.0})
r.value                     # 10.0
r.status                    # "ok"
r.backend                   # "interpreter_f64" | JIT / ball backends
r.is_enclosure              # False for f64; True when a rigorous ball was used
r.enclosure                 # ArbBall when is_enclosure
r.achieved_precision_bits
r.reason                    # why a backend was declined / downgraded

# Complex mode follows principal branch cuts
ak.evaluate(sqrt(x), {x: -1.0}, mode="complex")

# Vectorised evaluation (DLPack): NumPy, JAX, PyTorch CPU tensors, etc.
import numpy as np
from alkahest import numpy_eval, numpy_eval_par

xs = np.linspace(0, 1, 1_000_000)
ys = numpy_eval(f, xs)        # ndarray; much faster than a Python loop
ys = numpy_eval_par(f, xs)    # multi-core; requires the parallel feature (on in every wheel)
```

**`parallel` is `True` in the default PyPI wheel** (Linux, macOS and Windows), so `*_par`
really is multi-core there. But it is not a Cargo *default*: on a source build that did
not pass `--features parallel`, every `*_par` entry point (`numpy_eval_par`,
`simplify_par`) still exists and silently falls back to its single-threaded counterpart —
correct results, **no speedup**, no warning. Never claim a parallel speedup, and never
benchmark `*_par` against its sequential twin, without checking first:

```python
ak.capabilities()["features"]["parallel"]   # True on PyPI wheels
```

If it is `False`, you are on a source build; rebuild with `--features parallel`.

**A `CompiledFn` cannot cross threads.** `compile_expr` returns an object pinned to
its creating thread (it owns JIT code pages), so calling `numpy_eval` *or*
`numpy_eval_par` on it from another `threading.Thread` raises
`pyo3_runtime.PanicException: ... is unsendable, but sent to another thread`. This is
not about `parallel` — plain `numpy_eval` is refused the same way — and
`PanicException` derives from `BaseException`, not `Exception`, so a worker wrapped in
`except Exception:` will not catch it. Call `compile_expr` inside each thread. The
`ExprPool` and the expressions themselves are shareable; only the compiled function is
not. `numpy_eval_par` already fans out internally, so in most cases you do not want
threads of your own at all.

---

## trace / grad / jit (JAX-style transforms)

Here **`grad`** means “gradient of a **traced** function”, not `symbolic_grad`. For partials of a bare `Expr`, use **`symbolic_grad(expr, [x, y])`** (see [Differentiation](#differentiation)).

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

# Expr-level partials (no @trace):
partials = ak.symbolic_grad(x**2 + ak.sin(y), [x, y])  # list[Expr]

@ak.trace(pool)
def energy(x, y):
    return x**2 + ak.sin(y) * ak.exp(x)

# energy is a TracedFn
print(energy.expr)          # symbolic expression
print(energy(1.0, 0.0))     # numeric float
print(energy.symbols)       # [x, y]

# grad = gradient of TracedFn (GradTracedFn), not symbolic_grad
grad_energy = ak.grad(energy)
gs = grad_energy(1.0, 0.0)     # [∂/∂x, ∂/∂y] as floats

fast = ak.jit(energy)          # CompiledTracedFn
fast_grad = ak.jit(ak.grad(energy))  # compiled GradTracedFn
```

Non-decorator variant: `ak.trace_fn(fn, pool)`.

---

## Code emission

```python
from alkahest import horner, emit_c, to_stablehlo

poly = pool.integer(1) + pool.integer(2)*x + pool.integer(3)*x**2
print(horner(poly, x))                        # Horner-form Expr
c_code = emit_c(poly, x, "x_var", "f")       # C function string
stablehlo = to_stablehlo(sin(x)+exp(y), [x,y], fn_name="my_fn")  # StableHLO text
```

---

## Interval / ball arithmetic (Arb)

```python
from alkahest import ArbBall, interval_eval

ball = ArbBall(1.0, 1e-10)         # centre ± radius
result = interval_eval(sin(x), {x: ball})  # rigorous enclosure
```

The mapping values must be `ArbBall` objects — a `(centre, radius)` tuple raises
`TypeError`.

### Real roots, resultants, and stability

```python
from alkahest import real_roots, refine_root, resultant, subresultant_prs, routh_hurwitz

ivs = real_roots(x**2 - pool.integer(2), x)   # [RootInterval(-2, -1), RootInterval(1, 2)]
refine_root(x**2 - pool.integer(2), ivs[1], x)  # ArbBall(1.414214 ± 1.11e-16)
# signature is (poly, interval, var) — the interval comes second

resultant(x**2 - pool.integer(1), x - pool.integer(1), x)   # DerivedResult
subresultant_prs(...)                                       # subresultant chain

# Parametric Routh–Hurwitz: symbolic stability conditions
a = pool.symbol("a")
routh_hurwitz(x**3 + a*x**2 + pool.integer(3)*x + pool.integer(1), x)
# {'degree': 3, 'first_column': [1, a, (-1 + (a * 3)), 1],
#  'condition': ((a > 0) ∧ ((-1 + (a * 3)) > 0))}
```

`real_roots` returns isolating intervals, **not** floats — refine them to the
precision you need instead of assuming a numeric answer.

### Sparse polynomial algorithms

`gcd_sparse`, `sparse_interp`, `sparse_interp_univariate`, `factor_univariate_mod_p`,
`primary_decomposition`, `radical`, `triangularize`, `rosenfeld_groebner`, and
`diophantine` are all top-level exports for large / structured polynomial work.

---

## Symbolic matrices

```python
from alkahest import Matrix, jacobian

R = Matrix.from_rows([
    [cos(x), pool.integer(-1)*sin(x)],
    [sin(x), cos(x)],
])
R.rows            # 2
R.cols            # 2
R.shape()         # (2, 2) — a method, not a property
R.get(0, 1)       # Expr
R.to_list()       # list[list[Expr]]
```

### Arithmetic

`*` is the **matrix product** (SymPy convention), not elementwise — use `hadamard`
for elementwise. Since 3.8.0:

```python
A * B             # matrix product (same as A.multiply(B))
A * k, k * A      # scalar multiply when the operand is a scalar Expr
A ** n            # matrix power, non-negative integer n
A.multiply(B)     # explicit matrix product
A.scalar_mul(k)   # explicit scalar multiply
A.hadamard(B)     # elementwise product
```

### Decompositions and spectral data

```python
R.det()                   # symbolic determinant
R.trace()                 # Expr
R.rank()                  # int  (may raise E-LINALG-010 — see below)
R.transpose()             # Matrix
R.inverse()               # Matrix (E-MAT-003 if proven singular, E-MAT-004 if undecidable)
R.rref()                  # list[list[Expr]] — reduced row echelon form
R.nullspace()             # basis of the kernel (may raise E-LINALG-010)
R.column_space(), R.row_space()

R.eigenvals()             # dict: eigenvalue Expr → algebraic multiplicity
R.eigenvects()            # list of (eigenvalue, multiplicity, eigenvectors)
R.jordan_form()
R.characteristic_polynomial_lambda_minus_m()   # (poly, lambda_symbol)

R.lu(), R.qr(), R.cholesky()
R.matrix_exp()            # symbolic matrix exponential
R.simplify()              # simplify every entry
```

Some methods have narrower domains than the rest and raise rather than guess —
handle the error instead of assuming they apply:

| Method | Raises when | Code |
|---|---|---|
| `diagonalize()` | matrix is defective (fewer independent eigenvectors than the multiplicity) | `E-EIGEN-005` |
| `minimal_polynomial()` | entries contain free symbols | `E-LINALG-004` |
| `rational_canonical_form()` | any entry is not a rational constant | `E-LINALG-009` |
| `rank()`, `rref()`, `nullspace()`, `eigenvects()`, `jordan_form()` | an entry's vanishing can be proven **neither** zero nor non-zero | `E-LINALG-010` |
| `inverse()` | the determinant's vanishing cannot be decided | `E-MAT-004` |

So `minimal_polynomial` and `rational_canonical_form` are **numeric-matrix only**;
for symbolic matrices use `characteristic_polynomial_lambda_minus_m` or
`jordan_form`.

### The three-valued zero test (new in 3.8)

Elimination needs to know whether a pivot is zero. Alkahest's answer is three-valued —
*proven zero*, *proven non-zero*, *undecidable* — and the third case **refuses**:

```python
import alkahest as ak

pool = ak.ExprPool()
a = pool.symbol("a")
zero, one = pool.integer(0), pool.integer(1)
opaque = pool.func("mystery", [a])           # no eval rule → vanishing undecidable

try:
    ak.Matrix([[opaque, zero], [zero, zero]]).nullspace()
except ak.LinearAlgebraError as e:
    print(e.code)          # E-LINALG-010
    print(e.remediation)   # substitute concrete values for the parameters

try:
    ak.Matrix([[opaque, zero], [zero, one]]).inverse()
except ak.MatrixError as e:
    print(e.code)          # E-MAT-004
```

Before 3.8, "could not prove `det ≠ 0`" was read as "`det = 0`", and `nullspace()`
returned a **confident wrong basis** for any 2×2 with a symbolic determinant. If you
have results computed with 3.7 that came from `nullspace` on symbolic entries, recheck
them: verify `M @ v == 0` numerically rather than trusting the dimension.

`LinearAlgebraError` and `EigenError` are subclasses of `MatrixError`, so
`except ak.MatrixError` catches all three; `eigenvects()` raises `EigenError` carrying
code `E-LINALG-010` (the code names what could not be decided, not the wrapper).

### `eigenvals()` — two traps

1. **Casus irreducibilis.** For a 3×3 with an irreducible cubic characteristic
   polynomial and three real roots, `eigenvals()` returns the Cardano form, in which one
   cube root has a **negative radicand**. That expression is correct under the *real*
   cube-root convention; Alkahest is consistent about it and refuses to evaluate it
   (`eval_expr` → `E-EVAL-009`, `interval_eval` → an unbounded ball). Hand the same
   expression to SymPy/NumPy and the **principal** branch is taken instead: you get a
   confident number that is not an eigenvalue. Never export a radical eigenvalue to
   another tool without evaluating it in Alkahest first; prefer exporting a verified
   numeric enclosure (`refine_root`, `interval_eval`).
2. **It is not idempotent in memory.** `eigenvals()` interns a fresh gensym per call, so
   calling it repeatedly on the *same* matrix grows the pool by ~1.9 KB each time. Cache
   the result.

Symbolic eigenvalues are closed-form for 2×2 and, since 3.8.0, for parametric 3×3
matrices whose characteristic polynomial is an irreducible cubic (Cardano /
trigonometric path).

Note that products are built structurally, so entries come back unsimplified
(`((x * x) + (1 * 0))`). Call `.simplify()` on the result matrix, or `simplify`
on individual entries, before comparing or printing.

---

## ODE / DAE modeling

```python
import alkahest as ak
from alkahest import ODE, DAE, lower_to_first_order, pantelides

pool = ak.ExprPool()
t = pool.symbol("t")
y = pool.symbol("y")
k = pool.symbol("k")

ode = ODE.new([y], [pool.integer(-1)*k*y], t)
ode.order
ode.is_autonomous()
ode.state_vars()
ode.rhs()

ode_with_ic = ode.with_ic(y, pool.integer(1))

# Second-order → first-order
ode_1st = lower_to_first_order(x, pool.integer(-1)*x, 2, t)

# DAE Pantelides index reduction
# dae = DAE.new(...)
# reduced = pantelides(dae)
```

Numeric integration and closed-form `dsolve` live in `alkahest.experimental`
(see below).

---

## Integral transforms and `dsolve` (experimental)

These live in `alkahest.experimental`, **not** at top level. Import the submodule
explicitly — `alkahest.experimental` is not an attribute of the top-level module
until imported:

```python
from alkahest import experimental as ex

t = pool.symbol("t")
s = pool.symbol("s")

ex.laplace_transform(sin(t), t, s)                        # (1 + s^2)^-1
ex.inverse_laplace_transform(pool.integer(1) / (s**2 + pool.integer(1)), s, t)   # sin(t)

ex.fourier_transform(...), ex.inverse_fourier_transform(...)
ex.z_transform(...), ex.inverse_z_transform(...)

# ODEs
ex.dsolve(...)                    # closed-form solution
ex.ode_integrate_rk4(...)         # numeric, fixed step
ex.ode_integrate_rk45(...)        # numeric, adaptive → OdeTrajectory
```

Other experimental exports worth knowing: `asymptotic_expand`, `multilimit`,
`series_solve`, `residue`, `heaviside`, `dirac_delta`, `Fps`, `to_jax`.

Transform round-trips are supported but not total — inverse Laplace covers
repeated irreducible quadratic poles and sinh/cosh forms as of 3.8.0. Literal
negative Heaviside/Dirac shifts (`θ(t+a)`, `δ(t+a)` with `a > 0`) are **refused**
with `E-TRANSFORM-001` rather than silently applying the wrong unilateral formula.

---

## Sensitivity and adjoint systems

```python
from alkahest import sensitivity_system, adjoint_system, SensitivitySystem

ss = sensitivity_system(ode, [k])   # SensitivitySystem
ss.original_dim
ss.n_params
ss.extended_ode    # augmented ODE with sensitivity variables

adj = adjoint_system(ode, obj_grad_exprs)  # ODE run backward
```

---

## Context manager (thread-local defaults)

```python
import alkahest as ak

pool = ak.ExprPool()
with ak.context(pool=pool, domain="real", simplify=True):
    x = ak.symbol("x")   # pool and domain inferred
    y = ak.symbol("y")
    d = ak.diff(x**2 + x**2, x)  # .value is algebraically simplified

# simplify=True applies the general :func:`simplify` rewriter to results of
# diff / integrate / sum_* / product_* only — not to solve or simplify_trig.

# Inspect active context
ak.active_pool()
ak.active_domain()
ak.simplify_enabled()
ak.get_context_value("any_key")
```

---

## Budgets, cancellation, and determinism

Use these whenever you write a loop that calls Alkahest many times. A `Budget` is an
immutable `(wall_ms, max_steps, seed, max_bytes)` tuple pushed by `context(budget=…)`.

`max_bytes` is the memory analogue of `wall_ms`, and it is the one you cannot skip in an
unattended loop: without it, an exact-rational computation that outgrows the machine is
**not catchable at all** — GMP prints `GNU MP: Cannot allocate memory` and calls
`abort()`, so the whole interpreter dies and every result it was holding is lost, not
just the offending call. Alkahest additionally refuses (`E-BUDGET-005`) when the process
is about to exhaust a finite `RLIMIT_AS` (`ulimit -v`, a container limit), with or
without a budget.

```python
import alkahest as ak

with ak.context(pool=pool, budget=ak.Budget(wall_ms=300, max_steps=50_000, seed=7)):
    try:
        r = ak.integrate(hard_expr, x)
    except ak.BudgetExceededError as e:
        e.code      # E-BUDGET-001 wall clock | -002 max_steps | -003 cancelled
                    #             -004 max_bytes | -005 process address-space limit
        # deprioritise this candidate; DO NOT record it as "no antiderivative"

ak.request_cancel()      # process-wide flag, e.g. from a watchdog thread
ak.is_cancelled()        # read it
ak.clear_cancel()        # always clear it in a finally:
ak.budget_seed()         # the active budget's seed, for reproducible sampling
ak.active_budget(), ak.is_budget_active()
```

What actually honours a budget today: **`integrate` and `limit`** (they raise
`BudgetExceededError`) and **`simplify`** (no error channel — it stops early and returns
the best value so far, silently). Gröbner bases, homotopy continuation and the other
heavy primitives do **not** check it yet.

`integrate` and `limit` also **release the GIL** around their core call, so
`request_cancel()` from another thread reaches one that is already running. Nothing else
does, so a running Gröbner basis cannot be cancelled.

Three limits to state plainly, because they change what you should write:

1. **`wall_ms` is cooperative.** The call stops at the first checkpoint *after* the
   deadline. Typical overshoot is a small additive term (a `wall_ms=300` budget trips at
   ~320 ms), but the granularity is one primitive polynomial operation, and on a
   high-degree integrand that operation is a **FLINT** call which nothing can interrupt —
   there a 300 ms budget can return after ~2 s.
2. **`run_with_wall_fallback` does not bound wall time for an uncooperative callee.** It
   joins its worker before raising, so it returns when the callee returns.
   `ak.run_with_wall_fallback(time.sleep, 3.0, budget=ak.Budget(wall_ms=50))` raises
   `E-BUDGET-001` after **3000 ms**. Use it to turn `simplify`'s silent truncation into a
   coded error, not to contain an unknown callee. The only hard bound is an **OS-level
   timeout** (subprocess / process watchdog).
3. **Budget frames are thread-local; the cancel flag is process-wide.** A
   `ThreadPoolExecutor` you create yourself runs unbudgeted unless you re-enter the budget
   inside the worker. `batch_map` does that for you.

## Batch fan-out (`batch_map`, `*_many`)

```python
from alkahest import batch_map, batch_map_iter, integrate_many, simplify_many, diff_many

outs = ak.integrate_many([x**2, ak.log(ak.log(x)), ak.sin(x)], x, parallel=True)
for item in outs:                 # BatchItem(index, ok, value, error, elapsed_ms)
    if item.ok:
        use(item.value)           # a DerivedResult
    elif item.error["code"].startswith("E-BUDGET-"):
        requeue(item.index)       # resource limit — undecided
    else:
        close(item.index, item.error)   # a verdict about the mathematics
```

- **A batch never raises for one bad element** and never drops a slot; the exception is
  captured into `item.error` with the failing exception's own `E-*` code
  (`E-BATCH-001` when it has none).
- `batch_map` returns in **input order** either way. `batch_map_iter` streams in input
  order when sequential, completion order when `parallel=True`.
- Under `parallel=True` the active budget is snapshotted and re-entered in each worker.
  `wall_ms` stays one sweep-wide deadline; `max_steps` becomes **per item**.
- One item tripping its budget never cancels its siblings. `request_cancel()` does cancel
  everything in the process — that is the point of it being process-wide.

## Autoresearch modules: `ansatz`, `crosscheck`, `smt`

All three are `alkahest.<name>`; they resolve on attribute access, no separate import
needed.

```python
# --- alkahest.ansatz: guess a shape, let the CAS pin the constants ---
from alkahest.ansatz import polynomial, rational, exponential_polynomial, \
    linear_combination, quadratic_form, fit, enumerate_family, certify_nonneg

A = polynomial(pool, [x], degree=2)          # c_0 + c_1*x + c_2*x^2
sol = fit(A, A.expr - (x**2 - pool.integer(3)*x + pool.integer(2)))
sol.expr          # (2 + x^2 + (x * -3))
sol.status        # 'exactly_verified'
sol.rank, sol.free, sol.assignment, sol.residual, sol.certificate
# No member of the family fits -> AnsatzError E-ANSATZ-003.  That is a CLOSED BRANCH
# for this family, not a proof that no such object exists.

# --- alkahest.crosscheck: differential-test against another CAS (SymPy) ---
c = ak.crosscheck.check("integrate", x**2, x)
c.outcome        # 'agree' | 'diverge' | 'incomparable' | 'unavailable'
report = ak.crosscheck.sweep(cases=50, seed=7)   # seeded, reproducible
report.summary()
ak.crosscheck.oracles()          # which oracles are installed
ak.crosscheck.to_sympy(expr)     # one-way translation
# 'unavailable' = no oracle installed. It is NEVER reported as agreement.
# SWEEP_OPERATIONS is ('diff', 'integrate', 'simplify') — narrower than OPERATIONS.

# --- alkahest.smt: hand a discrete / mixed int-real subproblem to z3 or cvc5 ---
n = pool.symbol("n", "integer")
f = ak.And(pool.gt(x, n), pool.lt(x * x, pool.integer(10)))
ak.smt.supported(f).recommendation    # 'smt' | 'prefer_in_tree' — ask BEFORE solving
print(ak.to_smtlib(f))                # SMT-LIB 2 text; works with no solver installed
res = ak.smt.solve(f, budget=ak.Budget(wall_ms=5000))
res.status        # 'sat' | 'unsat' | 'unknown'
res.model         # exact Fractions — substituted back and checked in-process
```

Trust rules for `smt`, which an agent must not blur:

- **`sat` is checked** (`verification["status"] == "exactly_verified"`): the model was
  substituted back and evaluated exactly. A model that fails raises `E-SMT-004`.
- **`unsat` is `externally_asserted`** — nothing in Alkahest verified it, and it is
  deliberately excluded from `research.MACHINE_CHECKED_STATUSES`. Report it as "z3 says
  unsat", not as proved.
- **Algebraic-number witnesses are refused** (`E-SMT-003`) rather than converted to
  floats. Do not work around this by evaluating the `root-obj` yourself.
- **Read `res.sorts` before `res.status`.** `status` says the answer was checked; `sorts`
  (`{'n': 'Int', 'x': 'Real'}`, mirrored into `res.verification`) says *which question*
  was answered. A symbol built without an integer domain is declared `Real`, which turns
  an integer feasibility question into its real relaxation — and that relaxation's model
  is a genuine model of the formula as emitted, so it back-substitutes cleanly and reports
  `sat` / `exactly_verified` all the same. No status field distinguishes the two.
  Set the domain per symbol (`pool.symbol("n", "integer")`) or ambiently
  (`with ak.context(pool=pool, domain=ak.Domain.Integer):` — since 3.8 `pool.symbol` picks
  that up too, so it agrees with `ak.symbol`).
- `smt.solve` takes **quantifier-free** formulas; `to_smtlib` exports quantified ones.
  If you wrote the question as `Exists(x, Exists(y, body))`, **pass `body`**: a sat query
  already asks whether some assignment satisfies the formula, so the free variables of
  `body` are implicitly existentially quantified and the model is the witness. The two ask
  `solve` the same question; the `E-SMT-002` refusal is only about the wrapper. This does
  not extend to `Forall` (or an `Exists` beneath one) — export it or use `decide`.

## Memory: `ExprPool` never reclaims

There is no `clear`, no refcount and no GC. **The only way to free interned nodes is to
drop the whole pool**, and every `Expr` / `Matrix` / `Series` / `DerivedResult` holds a
strong reference to its pool — so keeping one result keeps every node ever interned.

Growth on a shared pool is linear and unbounded (~200 bytes/node; ~2–3.5 KB per
`integrate` call) while per-call **latency stays flat**, so a long loop OOMs with no
slowdown to warn you. There is no `len()` on `ExprPool`, so you cannot watch it either.

Write loops like this:

```python
for problem in problems:
    pool = ak.ExprPool()                      # fresh pool per problem
    x = pool.symbol("x")
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=500)):
        r = ak.integrate(build(pool, problem), x)
    record(r.to_dict(mode="compact"))         # a plain dict outlives the pool
    del pool, x, r                            # dropping the pool reclaims everything
```

Never carry a live `Expr` between iterations — `to_dict()` / `to_json()` / `str()` exist
partly for this. One operation grows even on identical input: `Matrix.eigenvals()`
(fresh gensym per call, ~1.9 KB); cache it. And the LLVM (`+jit` / `+full`) JIT leaks an
LLVM context per compile, so do not compile in a loop under those wheels.

---

## Error handling

All errors inherit `AlkahestError` and carry `.code`, `.remediation`, `.span`.

| Exception | Code prefix | Trigger |
|-----------|-------------|---------|
| `ConversionError` | `E-POLY-*` | Expression is not polynomial |
| `DomainError` | `E-DOMAIN-*`, `E-EVAL-*` | Side condition violated; `E-EVAL-009` = undefined at this point |
| `DiffError` | `E-DIFF-*` | Differentiation failed |
| `IntegrationError` | `E-INT-*` | No integration rule (`E-INT-001`); proven non-elementary (`E-INT-004`) |
| `LimitError` | `E-LIMIT-*` | Limit could not be established |
| `SeriesError` | `E-SERIES-*` | Series expansion failed |
| `MatrixError` | `E-MAT-*` | Shape mismatch, proven singular (`E-MAT-003`), **undecidable determinant (`E-MAT-004`)** |
| `LinearAlgebraError` | `E-LINALG-*` | *Subclass of `MatrixError`.* Elimination / decompositions; **undecidable entry (`E-LINALG-010`)** |
| `EigenError` | `E-EIGEN-*` | *Subclass of `MatrixError`.* Eigen/Jordan; defective matrix (`E-EIGEN-005`) |
| `CadError` | `E-CAD-*` | **`decide` refused** — outside the fragment, or an untestable irrational boundary point |
| `SosError` | `E-SOS-*` | No positivity certificate of this shape/degree (`E-SOS-002` — **a refusal: record `unknown`, not "not SOS"**); proved negative with a witness point (`E-SOS-003` — the only SOS verdict) |
| `HolonomicError` | `E-HOLO-*` | `zeilberger` outside the proper-hypergeometric class; `q_zeilberger` outside the `q`-hypergeometric one (`E-HOLO-020`) or with a non-rational shift quotient (`E-HOLO-024` — **permanent, not a bounds problem**); `telescope2d`/`telescope_md` outside the proper-hypergeometric-in-the-bound-indices class (`E-HOLO-040`), search exhausted — including a resource ceiling refusal, see item 31 — (`E-HOLO-041`), or a malformed call (`E-HOLO-042` — indices not pairwise distinct, or empty); `guess_holonomic` given too few terms to confirm a fit (`E-HOLO-005` — **a refusal: record `unknown`, not "no recurrence"**); `ModularRecurrence` / `binomial_mod` given an unsupported prime-power modulus (`E-HOLO-006`), a step with no `p`-adic integer answer (`E-HOLO-007` — **permanent**) or a working precision past `2**62` (`E-HOLO-008` — **resource: record `unknown`**) |
| `ValidatedError` | `E-VALIDATED-*` | Rigorous-bounds request unsupported / singular / malformed |
| `OdeError` | `E-ODE-*` | ODE construction failed |
| `DaeError` | `E-DAE-*` | DAE index reduction failed |
| `JitError` | `E-JIT-*` | JIT compilation failed |
| `SolverError` | `E-SOLVE-*` | Polynomial solver failed |
| `SumError` / `ProductError` | `E-SUM-*` / `E-PROD-*` | Summation / product failed |
| `PslqError` | `E-PSLQ-*` | Integer relation not justified by the input precision (`E-PSLQ-004`), or false for the exact rationals supplied (`E-PSLQ-005`) |
| `IoError` | `E-IO-*` | Pool checkpoint I/O |
| `PoolError` | `E-POOL-*` | Cross-pool or closed-pool misuse |
| `NumberTheoryError` | `E-NT-*` | Invalid input to number-theory helpers |
| `ParseError` | `E-PARSE-*` | String parse failures |
| `RsolveError` | `E-RSOLVE-*` | Recurrence / `rsolve` failures |
| `BudgetExceededError` | `E-BUDGET-*` | `001` wall clock, `002` `max_steps`, `003` cancelled |
| `AnsatzError` | `E-ANSATZ-*` | Family construction or fitting; `003` = no member fits |
| `CrossCheckError` | `E-XCHECK-*` | Check could not be posed; `002` = no oracle installed |
| `SmtError` | `E-SMT-*` | Export/solver/model-lift; `003` = algebraic witness, `004` = model failed the check |
| `CertificateUnavailableError` | `E-CERT-*` | A Lean certificate was required but withheld |

```python
from alkahest import ConversionError, IntegrationError

try:
    poly_normal(sin(x), [x])
except ConversionError as e:
    print(e.code)          # "E-POLY-006" (non-polynomial function in input)
                           # "E-POLY-001" is a different case: unexpected symbol
    print(e.remediation)   # human-readable fix hint
```

### Refusal vs verdict — the distinction that matters most

Some codes are **refusals**: "I could not establish this, and the alternative to saying
so is a confident wrong answer." Others are **verdicts** about the mathematics. Never
report a refusal to the user as a negative result, and never record one as a closed
branch in a search.

| Refusals (⇒ *undecided*) | Verdicts (⇒ a real answer) |
|---|---|
| `E-CAD-001`, `E-LINALG-010`, `E-MAT-004`, `E-SOS-002`, `E-ANSATZ-003`, `E-SMT-003`, `E-INT-001`, `E-LIMIT-003/005`, `E-BUDGET-001..003` | `E-INT-004` (proven non-elementary), `E-MAT-003` (proven singular), `E-EVAL-009` (undefined at this point), an `unsat` from `smt` (but only as *externally asserted*) |

When Alkahest refuses, say so precisely: *"Alkahest declined to decide this (E-CAD-001);
it is not a disproof."* Then offer an escalation route rather than substituting an
unverified answer from elsewhere.

---

## Available math functions

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`,
`asinh`, `acosh`, `atanh`,
`exp`, `log`, `sqrt`, `erf`, `erfc`, `gamma`, `digamma`,
`lambert_w`, `bessel_j0`, `bessel_j1`, `elliptic_e`, `elliptic_f`, `elliptic_k`, `elliptic_pi`,
`re`, `im`, `arg`, `conjugate`,
`abs`, `sign`, `floor`, `ceil`, `round`,
`min`, `max`, `piecewise`

All return `Expr`. They shadow Python builtins inside `alkahest` — use `alkahest.abs(expr)` to avoid ambiguity.

`atan2`, `gamma`, `min`, and `max` are reachable as attributes (`alkahest.gamma(...)`)
but are **not** in `__all__`, so `from alkahest import *` will not bring them into
scope — import them by name.

For `piecewise`, branch conditions must be symbolic predicates from the pool (not Python `>`):

```python
cond = pool.gt(x, pool.integer(0))
pw = alkahest.piecewise([(cond, x)], pool.integer(-1) * x)
```

---

## Pytree utilities (JAX-style)

```python
from alkahest import flatten_exprs, unflatten_exprs, map_exprs, TreeDef

leaves, treedef = flatten_exprs({"x": x_expr, "y": [y1, y2]})
# Note the argument order: leaves FIRST, treedef second (JAX puts treedef first).
reconstructed = unflatten_exprs(leaves, treedef)
mapped = map_exprs(lambda e: diff(e, x).value, {"f": f_expr})
```

---

## Parsing and pretty-printing

```python
from alkahest import parse, latex, unicode_str, ParseError

e = parse("x^2 + 2*x + 1", pool, {"x": x})
latex(e)
unicode_str(e)
```

---

## Summation, products, number theory

- Discrete summation: `sum_indefinite`, `sum_definite`, `verify_wz_pair`; linear recurrences: `solve_linear_recurrence_homogeneous`, `rsolve`.
- Symbolic products: `Product`, `product_indefinite`, `product_definite`.
- Integer number theory (FLINT-backed): `alkahest.number_theory` (`isprime`, `factorint`, `discrete_log`, …).

---

## Plotting

Alkahest never bundles a plotting library. All plot functions detect what is installed and call into it. The default backend is **Matplotlib**; **Plotly** is the interactive alternative (`backend="plotly"`).

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# 1-D curve — uses matplotlib by default, or plotly if specified
ax  = ak.plot(ak.sin(x), x, (-6.28, 6.28))
fig = ak.plot(ak.sin(x), x, (-6.28, 6.28), backend="plotly")

# 2-D surface
ak.plot3d(ak.sin(x) * ak.cos(y), x, y, (-5, 5), (-5, 5))

# Parametric curve
t = pool.symbol("t")
ak.plot_parametric(ak.cos(t), ak.sin(t), t, (0, 6.28318))

# Implicit curve f(x, y) = 0
ak.plot_implicit(x**2 + y**2 - pool.integer(1), x, y, (-2, 2), (-2, 2))

# Root markers on x-axis
p = ak.UniPoly.from_symbolic(x**3 - x, x)
ak.plot_roots(p, x)

# Series truncation vs exact
s = ak.series(ak.cos(x), x, pool.integer(0), 6)
ak.plot_series(s, ak.cos(x), x, (-4, 4))

# Expression DAG (graphviz package → rendered figure; else DOT string)
dot_or_src = ak.plot_dag(ak.sin(x**2))

# Dependency-free SVG — no matplotlib/plotly needed
svg_str = ak.plot_svg(ak.sin(x), x, (-6, 6))
# Use in Jupyter: from IPython.display import SVG; SVG(svg_str)
```

All `plot*` functions accept `**kw` forwarded to the backend. `plot` / `plot_parametric` also accept `ax=` (matplotlib) or `fig=` (plotly) to draw onto an existing figure.

GPU-accelerated plotting for dense grids (experimental; requires `pip install fastplotlib`):

```python
from alkahest.experimental._fastplotlib import fplot, fplot3d
fplot(ak.sin(x), x, (-10, 10), n=100_000)   # 100k-point GPU line
```

---

## Reinforcement learning

`alkahest.rl` exposes **verifiable RL environments** backed by the CAS. The core layer (`alkahest.rl.core`) is trainer-agnostic; domain environments live under `alkahest.rl.envs.*` and optionally integrate with [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

```bash
pip install "alkahest[rl]"   # Python ≥ 3.10; adds verifiers + datasets
```

```python
from alkahest.rl.envs.integration import IntegrationVerifier, load_environment

verifier = IntegrationVerifier()
# reward = verifier.verify(model_output, {"f_expr": f, "is_elementary": True, "pool": pool})

env = load_environment(difficulty_tier=0, n_train=1000, n_eval=100, adaptive=True)
```

| Component | Description |
|-----------|-------------|
| `IntegrationVerifier` | Layered check: symbolic diff → e-graph → interval spot checks; rewards honest refusal on NonElementary integrands |
| `load_environment()` | Returns a `verifiers.SingleTurnEnv` with Risch-tier curriculum |
| `recipes/verl_integration_reward.py` | Drop-in reward for [veRL](https://github.com/volcengine/verl) |

**Environments Hub:** [`alkahest/alkahest-symbolic-integration`](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) — install with `prime env install alkahest/alkahest-symbolic-integration`. Full checklist in the [RL guide](https://alkahest-cas.github.io/alkahest/rl.html).

---

## Stable vs experimental API

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_cas::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_cas::experimental` and `alkahest.experimental` and may change in minor releases—prefer top-level exports for agent-written code unless the user asks for experimental features.

Symbolic complex constructors (`conjugate`, `re`, `im`, `arg`) are experimental.
Principal `arg` only folds domain-safe literals (`arg(Positive)`, `arg(I)`);
`arg(0)`, negative reals, and branch-cut expressions stay unevaluated.

Constructing `arg(...)` does **not** fold on its own — it builds an unevaluated node.
The domain-safe folds are `simplify` rewrite rules, so you must call `simplify`:

```python
ak.arg(I)                      # arg(I)          — unevaluated node
ak.simplify(ak.arg(I)).value   # (pi * 1/2)      — folded
```

---

## Primitive registry

The registry is **read-only introspection**. There is no public API for registering
custom primitives from Python — do not attempt `reg.register(...)`, it does not exist.

```python
from alkahest import PrimitiveRegistry

reg = PrimitiveRegistry.default_registry()
reg.is_registered("sin")        # bool
reg.capabilities("sin")         # which rules (diff, integrate, …) are implemented
reg.coverage_report()           # structured coverage across all primitives
reg.coverage_report_markdown()  # same, rendered as a Markdown table
```

---

## Key rules for agents

1. **Always create a pool first.** `ExprPool()` before any symbol or expression. Optional: `with ak.context(pool=pool): x = ak.symbol("x")` to avoid repeating `pool=`.
2. **Pool first; literals in arithmetic are OK.** Use `x + 1`, not only `x + pool.integer(1)`. Use `pool.rational(p, q)` for exact rationals; `subs` accepts Python `int`/`float` in the mapping.
3. **Read `.value` for the expression** — for the operations that return `DerivedResult`. `limit` returns a bare `Expr`, `series` returns a `Series`, `solve` returns a list of dicts, and `evaluate` returns an `EvaluationResult`; none of those have `.value`. See the table in [Return type](#return-type-derivedresult).
4. **Use specific simplifiers.** Prefer `simplify_trig`, `simplify_log_exp`, `collect_like_terms` over the catch-all `simplify` when the structure is known — it is faster.
5. **Polynomial conversions raise.** `UniPoly.from_symbolic` and `poly_normal` raise `ConversionError` for non-polynomial input — catch it.
6. **`solve` / Gröbner-side APIs are available in all PyPI wheels.** The `groebner` Cargo feature is a default since 2.3.1 — no feature flag or `ImportError` guard needed. Default PyPI wheels also include egglog (`HAS_EGRAPH` is typically `True`); use `simplify_egraph` when rule-based simplification is insufficient.
7. **`trace` requires a pool argument.** Use `@alkahest.trace(pool)` (or `trace_fn(fn, pool)`). `@alkahest.trace` alone is invalid.
8. **`grad` ≠ `symbolic_grad`.** `symbolic_grad(expr, [x, y])` → `list[Expr]`. `grad(traced_fn)` → `GradTracedFn` (needs `@trace(pool)` first). `jit` accepts `TracedFn` or `GradTracedFn`.
9. **`numpy_eval` expects a `CompiledFn`** (from `compile_expr`), not an `Expr` and not a `TracedFn` — and the arrays are **separate positional arguments**, one per input variable: `numpy_eval(f, xs, ys)`, never `numpy_eval(f, [xs, ys])`. `CompiledFn.__call__` is the opposite: one point as one sequence, `f([1.0, 2.0])`.
10. **Symbols from different pools are incompatible.** Keep one pool per computation graph.
11. **`plot*` functions detect the backend automatically.** Never import matplotlib/plotly in user code just to call `ak.plot` — let alkahest dispatch. Use `backend="plotly"` or `backend="matplotlib"` to force one. Use `plot_svg` when no plotting library is available.
12. **`plot_dag` returns a `graphviz.Source` if the `graphviz` package is installed, otherwise a raw DOT string.** Call `.render()` or `.view()` on the returned object, or pipe the string to `dot -Tpng`.
13. **A refusal is not a negative result.** `E-CAD-001`, `E-LINALG-010`, `E-MAT-004`, `E-SOS-002`, `E-ANSATZ-003`, `E-SMT-003` and every `E-BUDGET-*` mean *undecided by this route*. Say so explicitly; do not paraphrase them as "false", "no solution exists", or "not possible". See [Refusal vs verdict](#refusal-vs-verdict--the-distinction-that-matters-most).
14. **`decide` can raise.** Always `try/except ak.CadError`. It is not complete: ≤ 2 variables, ≤ 2 quantifiers, and it refuses at irrational boundary points.
15. **One pool per problem in any loop.** `ExprPool` never reclaims and holding any `Expr` pins the whole pool. Carry `to_dict(mode="compact")` between iterations, not live expressions.
16. **Bound long calls with `context(budget=…)`, not `run_with_wall_fallback`.** The latter joins its worker and so does not bound wall time for an uncooperative callee. Only `integrate` and `limit` honour the cooperative budget and release the GIL.
17. **Do not export radical results to another CAS without evaluating them here first.** Casus-irreducibilis cube roots from `eigenvals()` are correct in Alkahest and wrong under a principal-branch evaluator. An `E-EVAL-009` or an infinite ball is the signal not to export as-is.
18. **`0 · 0⁻¹` is left unevaluated on purpose** (since 3.8). If you see `(0 * 0^-1)` in a result, that is Alkahest declining to give an indeterminate form a value — not a simplifier failure to work around.
19. **`relation_confidence` answers `None` when it cannot see the input precision** (since 3.8), which is the normal case: a decimal string may be an exact rational or a truncated constant, and nothing in it says which. `None` means *not checked*, never *passed* — branch on `if verdict["credible"]:`, not `is not False`. To get a real verdict on a `guess_relation` result computed from truncated decimal strings, pass the digits you trust: `relation_confidence(constants, coeffs, digits=60)` — or, since 3.9.1, declare them on the search itself with `guess_relation(constants, digits=60)`, which is the only way to make it raise `E-PSLQ-004` on input whose precision it cannot see (`precision_bits` is the *search* width, not a claim about the data). Without a declaration only `float` inputs are judged: an `mpmath.mpf` reports the ambient `mp.dps` at the moment it is asked rather than anything about itself, so it is *unknown* like a decimal string. `int` and `Fraction` inputs are judged by **evaluating** the relation in exact arithmetic — `credible` is `False` and `E-PSLQ-005` is raised when `Σ aᵢ·cᵢ ≠ 0`, which is what catches `Fraction(str(x))`-style conversions of a rounded value.
20. **`zeilberger` does not claim its order is minimal** (since 3.9). The search visits `(order, degree)` cheapest-first, so it can reach a cheap order-2 probe before an expensive order-1 one; `cert.order_is_minimal` is `False` to say *not established*, never "a lower order exists". Pass `minimal=True` for an order-ascending search that does establish it — *at certificate degree `<= max_degree`*, which is part of the claim and not a detail, since a lower-order relation needing a higher degree is never probed — and it costs the low-order sweep the default plan skips (Franel at `max_degree=16`: 0.23 s → 9.7 s), so claim minimality against the smallest `max_degree` you are willing to state.
21. **`guess_holonomic` returns `None` only for a swept grid, and its verdict is three-valued** (since 3.9). It fits a P-recursive recurrence to exact `int`/`Fraction` terms, but only where the terms *over-determine* the ansatz — twice the unknowns by default — and reports `surplus_terms`, the equations that confirmed the fit without being needed. Too few terms to test the whole grid is `E-HOLO-005`, a refusal, not `None`; recording it as "not holonomic" closes a branch that was never explored. `float` terms are refused outright. **Branch on `guess.status`, and treat `confirmed` as `True`/`False`/`None` rather than a boolean** — `"confirmed"` is the only result. `"singular"` means `guess.singular_indices` is non-empty: the leading coefficient vanishes at those indices *inside the data*, so the fit was unconstrained there, and the overwhelmingly likely cause is a corrupted term the fit absorbed into a root (one typo in an order-2/degree-1 sequence comes back at degree 4 with three roots, `dimension` 1 and 55 surplus equations — every other number perfect). The relation does hold on the terms, and on the clean sequence too, so no re-check finds it; recompute the terms at those indices. `"underdetermined"` means `dimension > 1` — read `guess.basis`, which is every relation the terms admit, rather than `coeffs`, which is an arbitrary member of it.

22. **A `zeilberger` certificate is about the *summand*; `cert.boundary` is what makes it about the *sum*** (since 3.9). `"vanishes"` licenses the homogeneous `Σ_i a_i(n)·S(n+i) = 0`; `"nonzero"` licenses the inhomogeneous `Σ_i a_i(n)·S(n+i) = b(n)` with `b(n)` in `cert.boundary_rhs` — a result, not a refusal; `"unknown"` licenses **nothing** about the sum, and recording the recurrence anyway is how a verified certificate becomes a false theorem (it did, on OEIS A279013). The verdict is about the range in `cert.limits`, which defaults to `k = 0..n` and is echoed back rather than inferred — pass `limits=(k_lo, k_hi)` when you are summing over anything else, because truncating a sum by one term generally flips `"vanishes"` to `"nonzero"`. `cert.boundary_at(k_lo, k_hi)` asks about another range without re-running the search. A verdict is not a claim about every `n`: `cert.boundary_valid_from` is the smallest `n` it holds from, because a declared range can be *empty* there (`k = 3..n−3` runs backwards at `n = 3, 4`, and the returned `b(n)` is false at exactly those `n`; `k = 5..3`, backwards everywhere, is `"unknown"`), and `cert.certificate_poles` lists integer points inside the range where `G = R·F` or the summand is not finite — an interior pole breaks the telescoping in the middle of the sum, so a non-empty list means `"unknown"` no matter how clean the certificate looks.

23. **`asymptotics_from_recurrence` separates what the recurrence *proves* from what the terms *fitted*** (since 3.9). Hand it a `ZeilbergerCertificate`, a `GuessedRecurrence`, or a bare list of coefficient polynomials and it returns `growth_rate` / `polynomial_exponent` — derived by Poincaré–Perron, and **exact** as `growth_rate_exact` / `polynomial_exponent_exact` when the root is rational — plus `connection_constant`, which is **fitted** from the terms and is not implied by the recurrence at all. Quote the constant only with `connection_constant_converged`; `evidence()` returns the two halves under separate `derived` / `fitted` keys for exactly this reason. `verdict != "single_dominant_root"` means the hypotheses failed (`equal_modulus_roots`, `repeated_dominant_root`, `degenerate_leading_coefficient`, `eventually_zero`) and `growth_rate` is `None` — no root is reported as if it had won. `follows_dominant_root is False` is a real answer, not an error: the sequence's dominant component vanishes and it grows more slowly than the recurrence's generic solution.

24. **`q`-sums need `experimental.q_zeilberger`, not `zeilberger`** (since 3.9). Gaussian binomials and `q`-Pochhammer symbols are *not* proper hypergeometric terms in `(n,k)`, so `zeilberger` refuses them with `E-HOLO-001` — correctly, and that refusal is not a statement about the sum. Build the summand with `qbinomial(pool, N, K)` / `qpochhammer(pool, u, d, v)` and call `q_zeilberger(term, q, n, k)`. Three things differ from the classical engine and all three matter: `cert.boundary` is **two-valued** (`"vanishes"` or `"unknown"` — there is no inhomogeneous arm, so an unbounded summand yields no claim at all); the sum it is about is `S(n) = Σ_{k ∈ Z} F(n,k)`, a finite sum over the proved window in `cert.support`; and **`q` is transcendental**, so a verdict is an identity in `Q(q)` and does *not* license specialising `q` to a root of unity — the step `q`-supercongruence work depends on, taken separately by `cert.specialize_at_root_of_unity` (item 29). `cert.sum_term(n0)` gives the exact `q`-series value from the definition of the `q`-Pochhammer symbol, so check a returned recurrence against it rather than trusting the certificate alone. `E-HOLO-024` is a permanent refusal, not a budget one: the input is in the shape of the class but its shift quotient is an infinite product (e.g. `(q; q**2)_k` shifted in `k`).

25. **Evaluate a holonomic sequence mod `p^k` with `ModularRecurrence`, not big integers** (since 3.9). `ModularRecurrence(coeffs, initial, *, rhs=None, start=0).value_mod(n, p, k)` runs `Σ_i a_i(n)·S(n+i) = b(n)` forward in `Z/p^K` — machine words, `O(1)` memory — instead of building an `S(n)` with `Θ(n)` digits and reducing it. `coeffs[i]` is lowest-degree-first, the convention `GuessedRecurrence.coeffs` already returns, so *guess → certify → sweep* composes with no reshaping. `supercongruence_sweep(rec, primes, k, index=…, expect=…)` is the loop; its `sharp` is the only thing a sweep can actually settle (some prime hits `v_p` exactly `k`, so `p^(k+1)` is **false**), and `holds` is falsification failing, not a proof. Measured on Apéry `A(p−1) mod p⁴` for the 237 primes below 1500: 95 ms against 3.47 s for the incremental-binomial route, and the gap widens quadratically.

26. **A singular index is the failure mode to plan for, and it is reported, not hidden** (since 3.9). Stepping forward divides by `a_J(n)`, which need not be a unit mod `p` — for Apéry `a_2(n) = (n+2)³` vanishes at every `n ≡ −2 (mod p)`, exactly the index a sweep crosses to reach `A(p)`. Alkahest measures the total `p`-adic precision loss before computing anything and runs the forward pass at `p^(k+loss)`; `ModularEvaluation.singular_indices()` and `.working_precision` say what it cost. Three refusals, none of which ever return a residue instead: `E-HOLO-006` (modulus not a supported prime power), `E-HOLO-007` (**permanent** — the step has no `p`-adic integer answer: `a_J(n) = 0` there, or the sequence leaves `Z_p` as `H_p = H_{p−1} + 1/p` does), `E-HOLO-008` (**resource** — `k + loss` needs a modulus past `2**62`; record `unknown`, and note that `supercongruence_sweep` puts these in `skipped()` and carries on rather than counting them as successes).

27. **`binomial_mod(a, b, p, k)` is Lucas at `k = 1` and Granville above it** (since 3.9). Cost is `O(p·k³ + log_p(a)·p·k)`, so `a` far larger than `p` is the ordinary case, not the hard one; `b > a` and `b < 0` return `0` rather than raising. Refuses with `E-HOLO-006` for a composite base or `p**k >= 2**62`, and `E-HOLO-008` when the one pass over `1 … p−1` is unaffordable.

28. **Check a fitted recurrence against OEIS with `experimental.novelty.check_novelty` before calling it new** (since 3.9). Build `RecurrenceClaim.from_recurrence(cert_or_guess, var=n)` from a `ZeilbergerCertificate` or `GuessedRecurrence`; it normalises away rescaling, sign flips, index shifts and a common polynomial factor, so `claim_hash` is equal for two presentations of the same relation and different for genuinely different ones. `check_novelty(claim, sources, terms=…)` returns a `NoveltyVerdict` whose `found` is **three-valued**, exactly like `relation_confidence`'s `credible`: `True` a source states the claim, `False` means *not found in the sources actually searched* — not "novel" — and `None` means no source could answer. There is no `novel` attribute anywhere on the type and `bool(verdict)` raises, so `if check_novelty(...):` cannot compile into the overclaim this API exists to prevent; branch on `verdict.status` (`"recorded"`, `"recorded_conjecturally"`, `"not_found"`, `"unavailable"`) or `verdict.found`. `verdict.hedged` is the difference between OEIS stating a recurrence as a theorem and marking it `Conjecture`/`Empirical` — restating the latter is not a result, proving it is. Sources are explicit and there is no default: `OeisCache` (file-backed, offline, what every test in this repository uses) or `OeisWeb` (opt-in, rate-limited, serves its cache first, degrades to `unavailable` rather than raising when there is no network) — pass `[cache]` or `[cache, web]` yourself. `RecurrenceClaim.from_text` parses OEIS's own formula lines and returns `None`, never a guess, for anything outside a homogeneous linear recurrence with polynomial coefficients (a sum, a generating function, a relation between two sequences, an inhomogeneous relation). Four things it also does, each added because the filter came back `not_found` for the Fibonacci recurrence against A000045: (a) it reads the **name** of an entry, which is where OEIS puts the recurrence for the entries defined by one (`Fibonacci numbers: F(n) = F(n-1) + F(n-2)`), reads any single letter as the sequence and juxtaposition as multiplication, and still holds every parsed line to reproducing the entry's own terms; (b) an `OeisWeb` `terms=` search is **paged** — `fmt=json` returns at most ten results and no total count, so one full page gives `exhaustive=False` and hence `unavailable`, never `not_found`, while an `ids=` lookup is exhaustive after one request; (c) `QRecurrenceClaim` is the same normal form and hash for a `q`-recurrence `Σ_i c_i(q, q^n)·u(n+i) = 0` (tagged `q-recurrence/1`, so it cannot collide with the ordinary kind), and since no source here can *state* one, `check_novelty` reports OEIS sources as `unavailable` for it rather than manufacturing a negative; (d) `terms=` is checked against the claim as well as used to search — `verdict.terms_check` is `"holds"`/`"fails"`/`"not_checked"`, and a `"fails"` means the lookup was about a different sequence from the claim (pass `check_novelty(..., start=…)` if `terms[0]` is not `u(0)`).

29. **`cert.specialize_at_root_of_unity(d, n)` is the decision that carries a `q_zeilberger` verdict to `q = ζ_d`, and it is three-valued** (since 3.9). A proved `Q(q)` recurrence does not by itself license setting `q` to a primitive `d`-th root of unity — a coefficient or a sum value can have a pole there, and specialising anyway is the `q`-analogue of the A279013 failure (item 22): a certificate that re-checks perfectly while the specialised claim is false. The hypotheses (no pole in any `a_i(qⁿ)` or `S(n+i)` at `ζ_d`) are decided **exactly**, by polynomial divisibility by `Φ_d(q)` over `Q` in the cyclotomic field `Q(ζ_d) = Q[q]/(Φ_d(q))` — never numerically — and `cyclotomic_polynomial(pool, d)` exposes `Φ_d(q)` itself so a caller can redo the check by hand. `status` is `"specializes"` (proved, and re-checked as an exact identity in `Q(ζ_d)` before being returned), `"obstructed"` (a pole was **exhibited** — `sum_value`/`coefficient` raise, but `sum_valuation(i)` is still available since the negative valuation *is* the obstruction — and this is not a claim the specialised identity is false, only that this route is blocked), or `"unknown"` (the generic boundary verdict was already `"unknown"`, so there is nothing to specialise). Three things a `"specializes"` verdict does **not** by itself mean, each with its own accessor: `is_vacuous` (every coefficient died — always true at `d = 1`, the `q → 1` limit — so the recurrence is `0 = 0`, still true, but empty), `leading_coefficient_survives` (`False` means the specialised recurrence no longer determines the last value from the earlier ones), and `support_shrinks` (`q`-Lucas killing terms — `[2;1]_q = 1 + q` is non-zero in `Q(q)` and zero at `ζ_2` — reported via `effective_support`, which can shrink but never grow). `sum_valuation(i)` is the `q`-supercongruence content itself: the exact integer `v` with `Φ_d(q)^v ∥ S(n+i)`, so `v ≥ r` is precisely `Φ_d(q)^r | S(n)`.

30. **`sos_decompose` tries the full PSD Gram cone and a Reznick multiplier search before refusing, and now certifies Motzkin and Robinson's form too** (since 3.9). Past diagonal dominance (`E-SOS-002` from DSOS alone) it searches the general PSD Gram cone, and past that — when `p` itself is not SOS — tries `(x_1²+…+x_n²)^N·p` for `N = 1..4` and searches *that* cone; a witness for `p < 0` still refuses separately with `E-SOS-003`, unaffected. Every certificate this returns is exact end to end: the numeric search only ever proposes a Gram matrix, which is rounded to nearby rationals and re-expanded to check it equals the target exactly before anything is returned — a `Some`/returned certificate is always sound regardless of what the float search converged to. Budget exhaustion is still `E-SOS-002`, undecided, never "not SOS" — say so, don't paraphrase it as a disproof. **The textbook PSD-not-SOS examples whose multiplier certificates are *singular* Gram matrices sitting exactly on the boundary of the PSD cone** — Motzkin's polynomial and Robinson's form — used to be out of reach for the original annealed alternating-projection search (a diagnosed convergence limitation at tangential PSD-cone intersections, not a soundness bug); the search now also tries Douglas–Rachford splitting with over-relaxation and a facial-reduction step, and with them both examples are found and exactly re-verified. **Correction (2026-08-20):** earlier revisions of this entry said the homogeneous 3-variable Motzkin form "is not classically expected to be SOS at `N = 1` at all" and that reaching it required `N = 2`. That was **false**. `(x²+y²+z²)·(x⁴y²+x²y⁴−3x²y²z²+z⁶)` *is* a sum of squares — that identity is precisely why Motzkin is the standard example of a PSD non-SOS form made SOS by one factor of `Σxᵢ²` — and it now certifies at `N = 1`, together with Choi–Lam. What was missing was not iterations but a **half-Newton-polytope reduction**: `psd_search` now restricts the Gram basis to the lattice points of `½·Newton(p)` (Reznick: every square in every SOS decomposition already has its support there, so nothing is lost), which takes `σ·Motzkin_hom` from a 75-parameter family to an 18-parameter one — the difference between landing `0.96` away from the certificate in parameter space and landing on it exactly. **What's still open:** the Horn/C₅ and C₇ copositivity forms, whose Newton polytopes are already full and whose `N = 1` families (420 and 2646 free parameters) are above `psd_search`'s numeric-search ceiling of 200 — so for those *no multiplier power is searched at all*. `E-SOS-002` now carries a trace of what the search actually did, with `NOT SEARCHED` marking budgets that fired; read it before recording a refusal as exhaustive, because "we did not look" and "we looked and found nothing" share the error code. `E-SOS-002` still means "not found within this search", never "not SOS". `basis_degree` now does reach the multiplier path (it used to be ignored there), or fall back to `alkahest.decide`.
31. **Multi-sums need `experimental.telescope2d` (two bound indices) or `experimental.telescope_md` (any number `m >= 1`), not `zeilberger`** (since 3.9; `telescope_md` since 3.10). `zeilberger`/`q_zeilberger` reach a sum over *one* index; `telescope2d(term, n, j, k)` is the Apagodu–Zeilberger generalization to a proper hypergeometric `F(n,j,k)` with **two** bound indices `j`, `k`, returning `a_0(n), …, a_J(n)` and *two* certificates `cert1`, `cert2` with `Σ_i a_i(n)·F(n+i,j,k) = Δ_j(cert1·F) + Δ_k(cert2·F)`, re-checked exactly in `Q(n,j,k)`. `telescope_md(term, n, [x_1, ..., x_m])` is the same engine generalized to arbitrary `m` — `m = 1` degenerates to a single-sum-shaped search, `m = 2` behaves identically to `telescope2d` (which is now a thin wrapper over it), `m >= 3` is genuinely new — returning `cert.certs()` (a list of `m` certificates, a method not a property since it's a collection) instead of `cert1`/`cert2`. Four real, stated scope limits, not unfinished polish: (1) the certificate ansatz uses a *fixed* denominator built from `F`'s own shift-ratio denominators rather than a minimal Gosper normal form, so a search that finds nothing raises `E-HOLO-041` and does not prove no certificate exists; (2) `cert.boundary_status(j_lo, j_hi, k_lo, k_hi)` / `cert.boundary_status([(lo_1, hi_1), ..., (lo_m, hi_m)])` only accept **constant** (not `n`-dependent) boxes — for a natural range like `j = 0..n`, pick a fixed bound safely larger than any `n` you check and let `F`'s own combinatorial vanishing do the rest, exactly as the module's own worked examples do; (3) the boundary of a box is **`2m` `(m-1)`-dimensional face sums, not `2^m` corner-point evaluations** — a naive corner-evaluation formula is simply wrong — and this version only proves the sufficient (not necessary) condition that each face vanishes identically, so `boundary_status` can return `"unknown"` for a boundary that is genuinely `0` but not by that pointwise route; it never guesses `"vanishes"`. There is no inhomogeneous `"nonzero"` verdict yet — an unresolved face is always `"unknown"`; (4) `telescope_md`'s underlying exact linear solve is `O(rows · cols²)` and both grow fast with `m` and the certificate degree bound (measured: `m = 3` at certificate degree 2 already means a ~10,000-row, 245-unknown system taking ~47s to solve *per probe*), so two resource ceilings apply — a single probe above 400 unknowns is refused outright, and total work across every probe at or above 150 unknowns in one search call is capped to 300 — meaning `E-HOLO-041` can also mean "refused by a resource ceiling, not searched and found nothing," which the error message states explicitly; raising `m` or `max_cert_degree` further will not help once a ceiling is the reason. `E-HOLO-040` is the class refusal (not proper hypergeometric in the bound indices), `E-HOLO-042` a malformed call (indices not pairwise distinct, or `indices` empty for `telescope_md`).
