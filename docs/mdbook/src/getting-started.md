# Getting started

## Install

### Rust crate

`alkahest-core` is published on [crates.io](https://crates.io/crates/alkahest-core).
Add it to your `Cargo.toml`:

```toml
[dependencies]
alkahest-core = "2"

# Optional features (combine as needed):
# alkahest-core = { version = "2", features = ["groebner", "parallel", "egraph"] }
```

**System prerequisites:** `alkahest-core` links against GMP, MPFR, and FLINT.
Install them before running `cargo build`:

```bash
# Debian / Ubuntu
sudo apt-get install -y libflint-dev libgmp-dev libmpfr-dev

# macOS (Homebrew)
brew install flint
```

The `jit` feature additionally requires **LLVM 15 dev headers** (`llvm-15-dev` / `brew install llvm@15`).
See the main [README](https://github.com/alkahest-cas/alkahest/blob/main/README.md) for the full
feature table.

A runnable quickstart lives in [`examples/rust_quickstart/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/rust_quickstart).

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

Wheels on PyPI are built **without** LLVM JIT and **without** the optional Rust features `groebner`, `egraph`, and `parallel`, so installs stay small and avoid an LLVM runtime dependency. Numeric APIs still work through the interpreter fallback.

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries.

For native LLVM CPU JIT—or JIT plus Gröbner / parallel F4 / egglog—use an opt-in **`+jit`** or **`+full`** Linux wheel from GitHub Releases (below), or [build from source](#from-source). See the repository [`README.md`](https://github.com/alkahest-cas/alkahest/blob/main/README.md) for the same policy in short form.

### Optional Linux wheels (`+jit` / `+full`)

Tagged releases attach **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`; these are **not** the manylinux wheels published as the default PyPI binaries). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---|---|---|
| `+jit` | `jit` | Native LLVM CPU JIT only (smaller than `+full`). |
| `+full` | `jit groebner parallel egraph` | JIT plus Gröbner-backed solvers, parallel F4, egglog e-graph backend (typical maximal from-source dev stack). |

Example direct installs (replace **version**, tag, and wheel name using the release asset list):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v2.0.2/alkahest-2.0.2+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v2.0.2/alkahest-2.0.2+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libLLVM-*.so` or `libffi-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages).

If your downloader rejects `+` in the URL, percent-encode it in the filename segment (e.g. `2.0.2%2Bfull`).

After `+jit`, `alkahest.jit_is_available()` should be `True`. After `+full`, expect that **and** Gröbner-backed APIs such as `alkahest.solve`.

macOS and Windows `+jit` / `+full` wheels are **not** produced in CI yet; use [building from source](#from-source) there.

**Roadmap:** a small PEP 503 **extras index** URL hosting only `+jit` / `+full` wheels (PyTorch-style `--extra-index-url`). Until then, use PyPI for the default wheel or direct URLs / asset downloads from Releases.

### From source

For optional Cargo features (`jit`, `groebner`, `cuda`, …), GPU/NVPTX, or development, build the PyO3 extension with [maturin](https://github.com/PyO3/maturin).

Prerequisites (typical): **Rust** stable (≥ 1.76) and nightly, **LLVM 15**, **FLINT** (≥ 2.9, pulls in GMP/MPFR). See the repository `README` for distro-specific package names.

```bash
pip install maturin
git clone https://github.com/alkahest-cas/alkahest.git
cd alkahest
maturin develop --manifest-path alkahest-py/Cargo.toml --release
```

Optional features (combine as needed):

```bash
# LLVM JIT for native compiled evaluation
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features jit

# E-graph simplification (egglog)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features egraph

# Parallel simplification (sharded ExprPool)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features parallel

# Gröbner basis solver (+ Diophantine / homotopy-related paths)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features groebner

# CUDA / NVPTX codegen (requires CUDA toolkit and LLVM with NVPTX target)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features cuda

# Full native build (all optional features above; add cuda separately if needed)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

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

Python operators build expression trees:

```python
expr = x**2 + pool.integer(2) * x + pool.integer(1)
print(expr)  # x^2 + 2*x + 1
```

Math functions accept expressions:

```python
f = sin(x**2) + exp(x * y)
```

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
