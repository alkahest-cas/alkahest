# Getting started

## Install

### PyPI (default)

Alkahest is on the [Python Package Index](https://pypi.org/project/alkahest/). Supported interpreters are **Python 3.9 through 3.13** (`requires-python` on PyPI).

```bash
python -m pip install -U pip
pip install alkahest
```

Use a virtual environment when you also build from source or test multiple Python versions:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install alkahest
```

Default PyPI wheels are built with `egraph`, `groebner`, `cranelift` and `parallel`: the
**vendored egglog** e-graph backend, the **Gröbner solver** (so `alkahest.solve`,
Diophantine and homotopy work out of the box), the pure-Rust **Cranelift JIT** (so
`alkahest.jit_is_available()` is `True` without a system LLVM), and the Rayon-backed
**multi-core paths**. They do **not** include the LLVM JIT (`llvm_jit`) or `cuda`.

> ### `parallel` is on in the default wheel — but still check it
>
> `capabilities()["features"]["parallel"]` is `True` on every wheel published to PyPI, on
> Linux, macOS and Windows alike. The `parallel` Cargo feature is what powers the sharded
> `ExprPool` and every `*_par` entry point (`numpy_eval_par`, `simplify_par`), so those are
> genuinely multi-core out of the box. (It does not parallelise Gröbner bases: those are
> computed by sequential Buchberger over ℚ in every build.)
>
> It is still the one feature worth probing rather than assuming, because its absence is
> **silent**. `parallel` is not a Cargo *default*: a source build that does not pass
> `--features parallel` still gets working `numpy_eval_par` and `simplify_par` — they
> transparently fall back to their single-threaded counterparts, with no error, no
> warning, and no speedup. Benchmarking `numpy_eval_par` against `numpy_eval` on such a
> build measures the same number twice.
>
> Never infer parallelism from the function existing; ask:
>
> ```python
> import alkahest as ak
> if not ak.capabilities()["features"]["parallel"]:
>     ...  # a source build without --features parallel; *_par is a no-op alias here
> ```

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries.

For native LLVM CPU JIT use an opt-in **`+jit`** or **`+full`** Linux wheel from GitHub Releases (below), or [build from source](#from-source) with `--features jit`. See the repository [`README.md`](https://github.com/alkahest-cas/alkahest/blob/main/README.md) for the same policy in short form.

### Optional Linux wheels (`+jit` / `+full`)

Tagged releases attach **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`; these are **not** the manylinux wheels published as the default PyPI binaries). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---|---|---|
| `+jit` | `egraph groebner jit parallel` | LLVM CPU JIT in place of Cranelift; everything else matches the default PyPI wheel. |
| `+full` | `egraph groebner jit cranelift parallel` | The only wheel with **both** JIT backends, and a strict superset of the default wheel. Use it when you want LLVM without giving up Cranelift. |

Example direct installs (replace `<version>` and the wheel name using the release asset list):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v<version>/alkahest-<version>+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v<version>/alkahest-<version>+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libLLVM-*.so` or `libffi-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages).

If your downloader rejects `+` in the URL, percent-encode it in the filename segment (e.g. `2.0.2%2Bfull`).

After `+jit` or `+full`, `alkahest.jit_is_available()` should be `True`. Gröbner-backed APIs such as `alkahest.solve` are available in **all** wheels (including the default PyPI wheel) since `groebner` became a default Cargo feature in 2.3.1.

macOS and Windows `+jit` / `+full` wheels are **not** produced in CI yet; use [building from source](#from-source) there.

**Roadmap:** a small PEP 503 **extras index** URL hosting only `+jit` / `+full` wheels (PyTorch-style `--extra-index-url`). Until then, use PyPI for the default wheel or direct URLs / asset downloads from Releases.

### Build-profile verification

Every published wheel runs an import-and-capability smoke test in release CI.
After installing, inspect the exact native build rather than inferring features
from available Python functions:

```python
import alkahest as ak

features = ak.capabilities()["features"]
print(features)
```

| Distribution | Tested platforms | Native feature profile | `parallel` |
|---|---|---|---|
| Default PyPI wheel | Linux x86_64, macOS arm64, Windows x86_64 | `egraph`, `groebner`, `cranelift_jit`, `parallel` | `True` |
| Release `+jit` | Linux x86_64 | `egraph`, `groebner`, `llvm_jit` (no Cranelift), `parallel` | `True` |
| Release `+full` | Linux x86_64 | same as `+jit` | `True` |
| Source build | any | whatever you pass to `--features` | `True` **only** with `--features parallel` |

`parallel` is called out separately because it is the one feature whose absence is
silent: `numpy_eval_par` and `simplify_par` exist and work in every build, and simply
stop being parallel when it is off. Every published wheel now enables it, so the row
that can surprise you is the last one — `parallel` is not a Cargo default, so a source
build has to ask for it.

`jit` and `cranelift` remain compatibility names in this mapping. Prefer
`llvm_jit` and `cranelift_jit` when selecting a backend explicitly. `cuda`
indicates that the extension was compiled with NVPTX codegen — it guarantees
that `ak.compile_cuda` and `ak.CudaCompiledFn` exist, but not that a usable GPU
is present at runtime, and it is in no published wheel. There is **no
`groebner_cuda` bit**: the GPU Gröbner kernel has no Python binding, so the bit
was unfalsifiable from Python and was removed in contract v3. Read
[GPU support](./gpu.md) before branching on `cuda`.

### Optional: RL environments (`alkahest[rl]`)

Reinforcement-learning environments (symbolic integration, Prime Intellect Hub) are an
optional extra. Requires **Python ≥ 3.10** (`verifiers` does not support 3.9).

```bash
pip install "alkahest[rl]"
```

This adds `verifiers` and `datasets`. Environment code ships in the main wheel under
`alkahest.rl`. See the [RL guide](./rl.md) for API details, veRL integration, and
[Environments Hub publishing](./rl.md#hub-checklist).

### From source

For optional Cargo features (`jit`, `parallel`, `cuda`, …), GPU/NVPTX, or development, build the PyO3 extension with [maturin](https://github.com/PyO3/maturin). The `groebner` and `egraph` features are default and included automatically.

Prerequisites (typical): **Rust** stable (≥ 1.76) and nightly, **LLVM 21** (only for `--features jit`), **FLINT** (≥ 3.3, ≥ 3.4 for Riemann theta; pulls in GMP/MPFR — many distribution packages are older, in which case build FLINT from its release tarball into a prefix as shown below). The PyPI wheels are built with the opt-in `system-gmp` feature, which makes rug share FLINT's libgmp instead of compiling its own; to match them add `system-gmp` to `--features`, which needs **GMP ≥ 6.3 / MPFR ≥ 4.2** (not Ubuntu 22.04 / Debian 12; on Homebrew, export `CPATH`/`LIBRARY_PATH` for the `gmp` and `mpfr` kegs). See the repository `README` for distro-specific package names.

> **FLINT is a hard requirement of every source build.** There is no FLINT-free
> configuration: `UniPoly` is a FLINT polynomial, and factorization, resultants,
> normal forms and `number_theory` call FLINT directly with no pure-Rust fallback.
> The `flint3` Cargo feature selects which FLINT *version's* API to use — it does
> **not** make FLINT optional. Without it the build stops early with an install
> hint (`sudo apt-get install libflint-dev`, `dnf install flint-devel`,
> `brew install flint`, `conda install -c conda-forge libflint`, …).
>
> **Without root**, build FLINT into a user-local prefix and point the build at it:
>
> ```bash
> FLINT_LIB_DIR=$PREFIX/lib FLINT_INCLUDE_DIR=$PREFIX/include \
>   maturin develop --manifest-path alkahest-py/Cargo.toml --release
> export LD_LIBRARY_PATH=$PREFIX/lib      # DYLD_LIBRARY_PATH on macOS
> ```
>
> Both variables also feed FLINT version detection, so a locally built FLINT 3 is
> recognised as FLINT 3. `ALKAHEST_SKIP_FLINT_CHECK=1` bypasses the presence
> probe. If you only need a working install, the PyPI wheels already bundle FLINT.

```bash
pip install maturin
git clone https://github.com/alkahest-cas/alkahest.git
cd alkahest
maturin develop --manifest-path alkahest-py/Cargo.toml --release
```

The default build already includes `egraph` and `groebner`. Additional optional features:

```bash
# LLVM JIT for native compiled evaluation
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features jit

# Pure-Rust Cranelift JIT (fast compile, no system LLVM required)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features cranelift

# Parallel simplification (sharded ExprPool + simplify_par + numpy_eval_par)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features parallel

# CUDA / NVPTX codegen (requires CUDA toolkit and LLVM with NVPTX target)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features cuda

# Full native build (JIT + parallel; egraph and groebner are already default)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel cranelift jit"
```

### Rust crate

`alkahest-cas` is also published on [crates.io](https://crates.io/crates/alkahest-cas) ([docs.rs](https://docs.rs/alkahest-cas)) for use directly from Rust:

```toml
[dependencies]
alkahest-cas = "3"

# groebner is included by default; add other optional features as needed:
# alkahest-cas = { version = "3", features = ["parallel", "egraph"] }
```

**System prerequisites** (same libraries as the Python build — must be installed before `cargo build`):

```bash
# Debian / Ubuntu
sudo apt-get install -y libflint-dev libgmp-dev libmpfr-dev

# macOS (Homebrew)
brew install flint
```

The `jit` feature additionally requires **LLVM 21 dev headers** (`llvm-21-dev` / `brew install llvm@21`).

A self-contained runnable example is in [`examples/rust_quickstart/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/rust_quickstart).

## First steps

Every computation starts with an `ExprPool`. It owns all expressions; you create symbols and integers from it.

```python
import alkahest
from alkahest import ExprPool, diff, simplify, integrate, sin, exp, cos

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")
```

### Building expressions

Python operators build expression trees. Python numbers work on either side
of an operator and are coerced exactly — `int` to an integer, `fractions.Fraction`
to a rational; a `float` stays a float:

```python
from fractions import Fraction

expr = x**2 + 2 * x + 1
print(expr)  # x^2 + 2*x + 1
alt = (-1)**x * 2**x + Fraction(1, 3)
```

Math functions accept expressions and Python numbers:

```python
f = sin(x**2) + exp(x * y)
alkahest.sin(2), alkahest.sqrt(2), alkahest.exp(1)   # exact arguments, no TypeError
```

### Constants, factoring and expanding

The SymPy spellings exist at the top level:

```python
from alkahest import pi, E, I, oo, factor, expand, binomial, factorial

alkahest.simplify(sin(pi)).value          # 0
factor(x**2 - y**2)                       # (x - y)*(x + y), an Expr product over ℤ
expand((x + 1)**3)                        # x^3 + 3*x^2 + 3*x + 1
binomial(10, 3), factorial(5)             # 120, 120 (exact); symbolic n gives gamma ratios
alkahest.Matrix([[1, 2], [3, 4]]).det()   # -2
```

`pi`, `E` (Euler's number, `exp(1)`), `I` and `oo` are expressions in the pool of
the active `alkahest.context(pool=...)`, or else in the process-wide
`alkahest.default_pool()` — the same pool `sin(2)`, `parse("x + 1")` and
`alkahest.symbol("x")` use when no `Expr` argument names one. Because they have no
free symbols they mix into any pool: `x + pi` works for `x` from your own pool.

`pool.symbol("pi")` is an ordinary **variable** that prints as `pi`; it is not π
(`simplify(sin(pool.symbol("pi")))` stays as it is). The constant is `pi`,
`pool.pi()` or `parse("pi", pool)`.

Results pickle — `DerivedResult`, `Expr` and `Matrix` — so they can be sent through
`multiprocessing.Pool.map` and come back into the pool they were sent from.

### Parsing expressions from strings

Use `parse` when the expression comes from user input or a config file:

```python
from alkahest import parse

e = parse("x^2 + 2*x + 1", pool, {"x": x})
print(e)   # x^2 + 2*x + 1
```

Identifiers not in the `symbols` dict are auto-created as symbols in `pool`.
Both `^` and `**` denote exponentiation. See [Parsing from strings](./parsing.md)
for the full syntax reference.

### Simplification

```python
r = simplify(x + pool.integer(0))
print(r.value)  # x
print(r.steps)  # [RewriteStep(rule='add_zero', ...)]
```

### Differentiation

```python
dr = diff(sin(x**2), x)
print(dr.value)  # 2*x*cos(x^2)
```

### Integration

```python
r = integrate(exp(x), x)
print(r.value)   # exp(x)

r = integrate(sin(x), x)
print(r.value)   # -cos(x)
```

### Polynomial arithmetic

```python
from alkahest import UniPoly, RationalFunction

# Convert to FLINT-backed univariate polynomial
p = UniPoly.from_symbolic(x**3 + pool.integer(-1), x)
q = UniPoly.from_symbolic(x + pool.integer(-1), x)
print(p.gcd(q))          # x - 1
print(p // q)            # x^2 + x + 1
```

### Compiled evaluation

```python
from alkahest import compile_expr, eval_expr

# Scalar evaluation via a dict binding
result = eval_expr(x**2 + y, {x: 3.0, y: 1.0})
print(result)  # 10.0

# JIT-compiled callable
f = compile_expr(x**2 + pool.integer(1), [x])
print(f([3.0]))  # 10.0
```

### Vectorized evaluation over NumPy arrays

```python
import numpy as np
from alkahest import compile_expr, numpy_eval

f = compile_expr(sin(x) * exp(pool.integer(-1) * x), [x])
xs = np.linspace(0, 10, 1_000_000)
ys = numpy_eval(f, xs)  # vectorised; much faster than a Python loop
```

### Context manager

```python
with alkahest.context(pool=pool, simplify=True):
    z = alkahest.symbol("z")  # uses the active pool
    expr = z**2 + alkahest.sin(z)
```

### Agent / autoresearch loops

For a fan-out of candidates under a wall-clock or step budget, with results that
survive context compaction:

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

with ak.context(pool=pool, budget=ak.Budget(wall_ms=100, seed=1)):
    outs = ak.integrate_many([x**2, ak.sin(x)], x)
    for item in outs:
        if item.ok:
            print(item.value.to_dict(mode="compact")["verification"]["status"])
        else:
            print(item.error["code"])  # e.g. E-INT-001 or E-BUDGET-001
```

Full picture: [Autoresearch / agent loops](./search-plumbing.md),
[Budgets](./budgets.md), [Batch](./batch.md), [Claim graphs](./claim-graphs.md).

## Run the Math playground locally

The [playground](https://alkahest-cas.github.io/playground/)'s **Math** tab reads plain-language maths (`integrate sin(x)^2 dx`, `solve x^2 - 5x + 6 = 0`) and computes it with Alkahest. The site itself is static, so the calculations run on your machine:

```bash
pip install -U alkahest
python -m alkahest.playground    # serves http://127.0.0.1:8765 to this computer only
```

Then open the playground; it finds the server by itself (your browser may ask whether the site can connect to apps on your device: choose Allow). Nothing you type is sent anywhere else. It works in Chrome, Edge and Firefox; Safari blocks HTTPS pages from reaching `http://127.0.0.1`, so there run the playground itself locally (`demo-playground/README.md`). The server only answers the playground's origin, never runs Python from a request, and runs each calculation in a resource-limited child process; `--port`, `--allow-origin` and `--quiet` are the options. A WebAssembly build that needs no install is future work.

## Running the examples

The `examples/` directory in the Git repository has runnable end-to-end scripts. With `alkahest` installed (`pip install alkahest` or `maturin develop` as above), from the repository root run:

```bash
python examples/calculus.py
python examples/polynomials.py
python examples/jit_eval.py
python examples/ball_arithmetic.py
python examples/ode_modeling.py
```

If you are developing without installing the extension into the active environment, set `PYTHONPATH=python` so the pure-Python package is importable alongside your build.
