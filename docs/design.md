# Design and Architecture

## Overview

Alkahest is a general-purpose computer algebra system designed for the era of accelerated computing and AI agents. It provides symbolic mathematics with first-class support for hardware acceleration, proof-checkable results, and ergonomic Python APIs. It is built as a Rust core with Python bindings, targeting MLIR for code generation and Lean 4 for proof export.

The goal is to occupy a gap in the current ecosystem: SymPy is readable but slow, SageMath is powerful but sprawling, Symbolics.jl is fast but Julia-locked and unverified, and Mathematica is closed. None of these were designed with AI agents or modern accelerators as first-class users. Alkahest is.

## Design Goals

**Performance.** Symbolic operations run orders of magnitude faster than SymPy for common workloads, approaching specialized libraries (FLINT, Singular) for operations they cover. Compilation of symbolic expressions to native or GPU code is a routine operation, not a research project.

**Correctness.** Every simplification produces a derivation log. A meaningful subset of operations can export Lean 4 proofs that external tools can verify independently. Agents and humans both get ground truth they can rely on.

**Ergonomics.** The Python API feels natural for interactive use, numerical integration via NumPy/PyTorch/JAX is seamless, and error messages are structured and location-tagged. Agents can use the library effectively through documentation alone, without bespoke tooling.

**Generality.** While digital twin simulation is a motivating use case, the library is general-purpose. It serves number theorists, physicists, engineers, ML researchers, and AI agents with equal competence on their respective problem sets.

**Composability.** Transformations (simplify, differentiate, compile, certify, lower to hardware) operate on a shared traced representation and compose freely. Power comes from combining a small set of primitives, not from accumulating features.

## Architecture

### Layer Stack

The system is organized as a stack of layers, each with a single responsibility and a stable interface to the layers above and below.

| Layer | Name | Responsibility |
|---|---|---|
| 0 | Foundation bindings | Rust wrappers around FLINT, Arb, GMP, MPFR |
| 1 | Expression kernel | Hash-consed DAG, `ExprPool`, `ExprId`, `ExprData` |
| 2 | Simplification | E-graph equality saturation (egglog) + rule-based fixpoint |
| 3 | Mathematical algorithms | Calculus, solving, linear algebra, ODE/DAE, series |
| 4 | Code generation | MLIR dialect → standard dialects → LLVM IR / PTX / StableHLO |
| 5 | Proof export | Derivation logs → Lean 4 proof terms via Mathlib theorem mapping |
| 6 | Python bindings | PyO3 API: expressions, transformations, interop, error handling |

### Expression Kernel

The kernel uses a hash-consed directed acyclic graph representation. Every expression is interned in a shared pool (`ExprPool`), making structural equality a pointer comparison and making subexpression sharing automatic. This is critical for symbolic computation, where the same subterms recur constantly and naive tree representations explode in memory.

Multiple representations are exposed as distinct types with explicit conversions between them:

- `Expr` — generic expression tree, used when no more specialized form fits
- `UniPoly` — dense univariate polynomials backed by FLINT
- `MultiPoly` — sparse multivariate polynomials keyed by exponent vectors
- `RationalFunction` — quotients of polynomials with GCD-based normalization
- `ArbBall` — midpoint/radius ball arithmetic backed by Arb for rigorous bounds

The representation type is part of the Python type system, not hidden behind a uniform facade. Users and agents can see what representation they're in and what conversions are happening. This avoids the silent performance cliffs that plague SymPy.

All kernel types are `Send + Sync`. Thread-safety is a property of the kernel from day one.

### Primitive System

Following the JAX architectural pattern, the kernel defines a small set of primitive operations. Each primitive registers a full bundle:

- Numerical evaluator (`f64` and `ArbBall`)
- Forward- and reverse-mode differentiation rules
- MLIR lowering
- Lean 4 / Mathlib theorem name for certificate export
- Pattern-matching behavior in the e-graph

New operations are added by registering a primitive with its full rule bundle. Transformations are defined by how they handle each primitive, not by per-operation code paths scattered across the system. This keeps the core small, makes extension mechanical, and ensures every transformation interacts correctly with every operation.

### Tracing and Transformations

The API is built around pure functions and traced computations in addition to direct expression manipulation. Users write Python functions over the library's types and apply transformations to those functions:

```python
@alkahest.trace
def f(x):
    return alkahest.sin(x**2)

df = alkahest.grad(f)
df_fast = alkahest.jit(df)
```

Transformations compose: `jit(grad(f))` produces a compiled derivative. All transformations operate on the traced representation, not on values directly.

Multiple tracer types layer on top of each other:

- `SymbolicValue` — base symbolic tracer
- `CertifiedValue` — carries a derivation log
- `IntervalValue` — wraps computations in Arb ball arithmetic
- `DualValue` — forward-mode automatic differentiation

Stacking tracers stacks capabilities.

### Code Generation Pipeline

Expressions are lowered through multiple IR levels:

1. Expression DAG (hash-consed, Rust)
2. Canonical form after e-graph extraction
3. Custom `alkahest` MLIR dialect (math-aware optimizations: Horner's rule, stability-preserving rearrangements, vectorization-friendly layouts)
4. Standard MLIR dialects (`arith`, `math`, `scf`, `vector`, `gpu`, `linalg`)
5. LLVM IR, PTX, or StableHLO depending on target

The custom dialect is where math-aware optimizations happen. Standard MLIR passes handle the remainder. Multiple hardware targets are supported through MLIR's existing target infrastructure: CPU (default, with autovectorization), NVIDIA GPU (via PTX), and XLA/TPU targets (via StableHLO).

CPU evaluation uses three tiers selected by `CompileConfig`: tree-walking interpreter (small DAGs / few evals), Cranelift JIT (fast compile, pure Rust), and LLVM JIT (best batch throughput, requires LLVM 15). `CompileCache` memoizes `(ExprId, inputs) → CompiledFn` within a session. Native JIT backends expose a bulk column-major entry point for batch sweeps. When no JIT feature is enabled, the interpreter is used automatically.

### Proof Certificates

The library produces proof artifacts at three levels of rigor:

**Derivation logs.** Every simplification and transformation records the sequence of rewrite rules applied, with arguments and side conditions. Always on, cheap, and inspectable.

**Lean certificate export.** Each rewrite rule is tagged with a corresponding Lean 4 / Mathlib theorem name. For operations expressible as sequences of rewrites, the library emits a `.lean` file containing a proof term that Lean can verify independently. Side conditions (non-zero denominators, branch cuts, domain constraints) are tracked and propagated into the Lean output.

**Algorithmic evidence.** For operations where rewrite sequences do not work,
the library can expose an operation-specific witness or exact checker. Integer
factorization results carry an in-kernel reconstruction check: the unit and
powered factors are multiplied in the exact coefficient ring and compared with
the input. This establishes the represented factor product, not irreducibility
or a Lean-checked certificate. Risch integration verification remains a
separate, narrower effort.

### Statelessness

The library is stateless by design. No global assumption contexts, no implicit configuration, no hidden caches that change behavior. All context (domains, simplification policies, precision) is passed explicitly as arguments or bundled into expression structure.

Symbols carry their domain as part of their structural identity: `pool.symbol("x", domain="real")` and `pool.symbol("x", domain="complex")` are distinct expressions and will not be unified.

### PyTrees

Transformations work uniformly over nested data structures: lists, dicts, tuples, dataclasses, and user-registered container types. `diff(system, parameters)` works when `system` is a list of equations and `parameters` is a dict of symbols, without the user needing to flatten anything manually.

This is implemented in the Python layer via flatten/unflatten registration, following the JAX pytree pattern. The Rust kernel sees only flat sequences of expressions; structure is a Python-layer concern.

## Python API Design

The API targets two audiences: human users writing interactive or library code, and AI agents constructing and manipulating expressions programmatically.

Operator overloading makes common expression construction natural: `x + y`, `x**2`, `sin(x)`. An explicit functional API is also available for code that benefits from predictability.

Representation types are explicit: converting a symbolic expression to a polynomial is `UniPoly.from_symbolic(expr, var=x)`, not a hidden step. Users who care about performance opt into specific representations; users who don't get sensible defaults.

Results of top-level operations are rich objects with `.value`, `.steps`, `.assumptions`, `.warnings`, and `.certificate` attributes. Errors are a structured exception hierarchy with machine-readable codes, location information, and suggested remediations. See [Error handling](mdbook/src/errors.md).

Integration with the numerical ecosystem happens at the boundary:

- Compiled functions accept and return NumPy arrays via the buffer protocol (zero-copy)
- PyTorch and JAX integration via DLPack (`__dlpack__`) and `__jax_array__`
- The library can act as a source of JAX primitives for users who want to embed symbolic computation inside JAX programs

## Use Cases

**Digital twin simulation.** Symbolic modeling of physical systems, automatic index reduction for DAEs, acausal component-based modeling, compilation of simulation kernels to CPU and GPU, sensitivity analysis and parameter estimation via automatic differentiation.

**AI agent tooling.** Deterministic, stateless behavior; rich structured results; Lean certificates for verifiable reasoning; clear error messages with location information; comprehensive documentation. Agents can use the library reliably across long chains of computation without losing track of what has been proven versus conjectured.

**Scientific computing.** Symbolic preprocessing of equations before numerical solution, symbolic generation of Jacobians and Hessians, code generation for PDE solvers and optimization routines.

**Research mathematics.** Polynomial algebra, number theory (via FLINT), linear algebra over symbolic rings, and enough extensibility that researchers can add their own primitives and rewrite rules.

**Machine learning research.** Symbolic manipulation of loss landscapes, exact gradients for theoretical analysis, compilation to JAX/PyTorch for integration with existing training pipelines.

## Technical Stack

| Component | Technology |
|---|---|
| Core language | Rust (edition 2021) |
| Number libraries | FLINT (polynomials, Arb ball arithmetic, number theory), GMP, MPFR |
| Simplification | Vendored `egglog` (e-graph equality saturation, match-disjoint scheduling); native colored e-graphs for conditional rewrites; discrimination-net indexing for user `PatternRule` sets; rule-based fixpoint |
| Code generation | MLIR via `melior`; Cranelift JIT (`--features cranelift`); LLVM JIT (`--features jit`); NVPTX/CUDA |
| Continuous benchmarking | CodSpeed (Rust + Python) |
| GPU codegen | NVPTX via inkwell + `libdevice`; `cudarc` runtime |
| Proof export | Lean 4 with Mathlib |
| Python bindings | PyO3 + maturin |
| Numerical interop | NumPy buffer protocol, DLPack, `__jax_array__` |
| Testing | `cargo test`, `proptest`, `hypothesis`, SymPy oracle cross-validation |

## Non-Goals

- A new programming language or notebook environment. Python is the frontend; Jupyter/Marimo are the interactive environment.
- Full verification of the library's own implementation. Users trust the Lean certificates the library emits, not the Rust code that emits them.
- A replacement for specialized systems in their domains. GAP remains the right choice for serious group theory; PARI/GP for serious number theory. Alkahest is a strong general-purpose foundation that can call out to specialists when needed.
- Bespoke integrations for specific AI agent frameworks. Good documentation and a clean API serve agents better than framework-specific tooling.
- A GUI. The surface is a Python library.
