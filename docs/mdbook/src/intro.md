# Alkahest

Alkahest is a high-performance computer algebra system with a Rust core and Python API.

## What it is

A general-purpose symbolic math library designed around three axes:

**Performance.** The Rust kernel uses hash-consed directed acyclic graphs so structural equality is a pointer comparison and subexpression sharing is automatic. FLINT backs polynomial arithmetic. An LLVM JIT compiles symbolic expressions to native or GPU code at runtime. Common operations run orders of magnitude faster than SymPy.

**Correctness.** Every simplification and transformation produces a derivation log — the exact sequence of rewrite rules applied, with arguments and side conditions. A subset of operations can export Lean 4 proof terms verifiable by an independent checker.

**Ergonomics.** The Python API uses operator overloading for natural expression construction. Results are rich objects with `.value`, `.steps`, and `.certificate` attributes. Error messages carry structured codes, location information, and suggested remediations.

## Design principles

**Explicit representations.** The type system distinguishes `UniPoly` (FLINT-backed univariate polynomial), `MultiPoly` (sparse multivariate), `RationalFunction`, and the generic `Expr` tree. Converting between them is an explicit call. There are no silent representation changes hiding performance cliffs.

**Stateless by design.** No global assumption contexts. No hidden caches that change behavior. All context (domains, simplification policy, precision) is passed explicitly or bundled into expression structure. This makes results deterministic and parallelism safe.

**Composable transformations.** `trace`, `grad`, `jit`, and `certify` operate on a shared traced representation and stack freely: `jit(grad(f))` compiles a derivative, `jit(grad(grad(f)))` compiles a second derivative.

**A small primitive set.** Each primitive (sin, exp, add, mul, ...) registers a full bundle: simplification rule, forward- and reverse-mode differentiation, MLIR lowering, Lean theorem tag, numerical evaluation. New operations are added by registering a primitive, not by adding code paths across the system.

## Compared to alternatives

| | SymPy | SageMath | Symbolics.jl | Alkahest |
|---|---|---|---|---|
| Performance | Slow | Moderate | Fast | Fast |
| GPU codegen | No | No | No | Yes |
| Lean proofs | No | No | No | Yes |
| Python API | Yes | Yes | No (Julia) | Yes |
| Open source | Yes | Yes | Yes | Yes |

## This guide

The guide covers the Rust-level design concepts. For the generated **Python API reference** (Sphinx), see the [API documentation](https://alkahest-cas.github.io/alkahest/api/).
