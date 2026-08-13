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
> `ExprPool`, the parallel F4 reduction, and every `*_par` entry point (`numpy_eval_par`,
> `simplify_par`), so those are genuinely multi-core out of the box.
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

Prerequisites (typical): **Rust** stable (≥ 1.76) and nightly, **LLVM 15** (only for `--features jit`), **FLINT** (≥ 2.9, 3.x recommended; pulls in GMP/MPFR). See the repository `README` for distro-specific package names.

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

# Parallel simplification and parallel F4 (sharded ExprPool + numpy_eval_par)
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
alkahest-cas = "2"

# groebner is included by default; add other optional features as needed:
# alkahest-cas = { version = "2", features = ["parallel", "egraph"] }
```

**System prerequisites** (same libraries as the Python build — must be installed before `cargo build`):

```bash
# Debian / Ubuntu
sudo apt-get install -y libflint-dev libgmp-dev libmpfr-dev

# macOS (Homebrew)
brew install flint
```

The `jit` feature additionally requires **LLVM 15 dev headers** (`llvm-15-dev` / `brew install llvm@15`).

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
