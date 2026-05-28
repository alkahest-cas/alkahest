# Alkahest Documentation

Published guides (including **install**: PyPI default wheel vs optional ``+jit`` / ``+full`` Linux
releases, and building from source) live at https://alkahest-cas.github.io/alkahest/ — start at
*Getting started*.

## User guide (mdBook)

The conceptual guide covers kernel design, rule engine, e-graph saturation,
transformations, code generation, Lean certificates, and more.

Located at `docs/mdbook/`. Build with:

```bash
mdbook build docs/mdbook/
# HTML output: docs/mdbook/book/
```

Chapters:
- Introduction and getting started
- Kernel design (ExprPool, hash-consing, ExprData)
- Expression representations (UniPoly, MultiPoly, RationalFunction, ArbBall)
- Simplification: rule engine and e-graph saturation
- Calculus: differentiation and integration
- Composable transformations (trace, grad, jit)
- Code generation (MLIR, LLVM JIT, GPU / NVPTX)
- Ball arithmetic (rigorous interval bounds)
- ODE and DAE modeling
- Polynomial system solving
- Interoperability (NumPy, JAX, PyTorch, StableHLO)
- Plotting (backend dispatch, Matplotlib/Plotly/fastplotlib, SVG renderer)
- Derivation logs
- Lean certificates
- Error handling
- Stability policy

## Python API reference (Sphinx)

Located at `docs/sphinx/`. Build with:

```bash
sphinx-build -b html docs/sphinx/ docs/sphinx/_build/
```

Covers: core types, simplification, calculus, polynomials, numerics,
transformations, matrices, ODE/DAE, solver, code generation, plotting, errors.

## Other reference

- `design.md` — architecture, design goals, layer stack, and technical stack.
- `features.md` — current feature surface at a glance.
