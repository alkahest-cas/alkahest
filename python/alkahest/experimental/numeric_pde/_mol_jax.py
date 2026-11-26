"""JAX export of a :class:`MolSystem`: the same kernels and gathers as the
native program, as pure ``jax.numpy`` (so ``jax.jit``, ``jax.grad``,
``jax.vmap`` and StableHLO lowering all work)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ...exceptions import NumericPdeError

_FUNCS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    "sign": "sign",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
    "atan": "arctan",
    "asin": "arcsin",
    "acos": "arccos",
    "erf": None,
}


def _lambdify(expr, syms, jnp):
    """Translate ``expr`` (in ``syms``) to a function of arrays."""
    index = {str(s): i for i, s in enumerate(syms)}
    memo = {}

    def build(e):
        key = e
        if key in memo:
            return memo[key]
        n = e.node()
        tag = n[0]
        if tag == "symbol":
            name = n[1]
            if name in index:
                i = index[name]
                fn = lambda a, i=i: a[i]  # noqa: E731
            elif name == "pi":
                fn = lambda a: jnp.pi  # noqa: E731
            elif name in ("E", "e"):
                fn = lambda a: jnp.e  # noqa: E731
            else:
                raise NumericPdeError(f"unbound symbol {name} in a kernel", code="E-PDE-370")
        elif tag == "integer":
            v = float(int(n[1]))
            fn = lambda a, v=v: v  # noqa: E731
        elif tag == "rational":
            v = float(Fraction(int(n[1]), int(n[2])))
            fn = lambda a, v=v: v  # noqa: E731
        elif tag == "float":
            v = float(n[1])
            fn = lambda a, v=v: v  # noqa: E731
        elif tag == "add":
            parts = [build(c) for c in n[1]]

            def fn(a, parts=parts):
                acc = parts[0](a)
                for p in parts[1:]:
                    acc = acc + p(a)
                return acc

        elif tag == "mul":
            parts = [build(c) for c in n[1]]

            def fn(a, parts=parts):
                acc = parts[0](a)
                for p in parts[1:]:
                    acc = acc * p(a)
                return acc

        elif tag == "pow":
            base = build(n[1])
            en = n[2].node()
            if en[0] == "integer":
                k = int(en[1])
                if k == 2:
                    fn = lambda a, base=base: base(a) * base(a)  # noqa: E731
                elif k == -1:
                    fn = lambda a, base=base: 1.0 / base(a)  # noqa: E731
                else:
                    fn = lambda a, base=base, k=k: base(a) ** k  # noqa: E731
            else:
                ex = build(n[2])
                fn = lambda a, base=base, ex=ex: base(a) ** ex(a)  # noqa: E731
        elif tag == "func":
            name = n[1]
            target = _FUNCS.get(name)
            args = [build(c) for c in n[2] if hasattr(c, "node")]
            if name == "erf":
                from jax.scipy.special import erf as jerf

                fn = lambda a, args=args: jerf(args[0](a))  # noqa: E731
            elif target is None:
                raise NumericPdeError(
                    f"function {name} has no jax.numpy translation", code="E-PDE-370"
                )
            else:
                g = getattr(jnp, target)
                fn = lambda a, g=g, args=args: g(*[x(a) for x in args])  # noqa: E731
        else:
            raise NumericPdeError(f"cannot translate {tag} nodes to JAX", code="E-PDE-370")
        memo[key] = fn
        return fn

    f = build(expr)
    return lambda *arrays: f(arrays)


def _points(region, strides):
    import numpy as np

    axes = [np.arange(a, b) for a, b in region]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = sum(g * s for g, s in zip(grids, strides))
    return np.asarray(flat, dtype=np.int64).ravel()


def build_jax_rhs(ms):
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise NumericPdeError("to_jax needs JAX (pip install jax)", code="E-PDE-370") from exc
    import numpy as np

    sys = ms._sys
    nf = len(sys.fields)
    size = sys.size
    strides = sys.strides
    sf = np.asarray(sys.state_field, dtype=np.int64)
    sp = np.asarray(sys.state_flat, dtype=np.int64)
    plan: list[Any] = []
    for op in sys.ops:
        tidx = _points(op["region"], strides)
        if op["kind"] == "dense":
            k = op["axis"]
            st = strides[k]
            M = np.asarray(op["matrix"], dtype=float).reshape(-1, op["src_len"])
            rlo = op["region"][k][0]
            i = (tidx // st) % sys.shape[k] - rlo
            base = tidx - (rlo + i) * st
            src = base[:, None] + (op["src_lo"] + np.arange(op["src_len"]))[None, :] * st
            plan.append(("dense", op["src"], op["dst"], tidx, src, jnp.asarray(M[i])))
            continue
        fn = _lambdify(op["expr"], op["syms"], jnp)
        gathers: list[Any] = []
        for inp in op["inputs"]:
            if inp[0] == "field":
                sh = sum(o * s for o, s in zip(inp[2], strides))
                gathers.append(("field", inp[1], tidx + sh))
            elif inp[0] == "coef":
                gathers.append(("const", jnp.asarray(sys.coefs[inp[1]][tidx])))
            else:
                gathers.append(("time",))
        plan.append(("kernel", op["target"], tidx, gathers, fn))

    def rhs(t, y):
        full = jnp.zeros((nf, size), dtype=jnp.result_type(y))
        full = full.at[sf, sp].set(y)
        dfull = jnp.zeros_like(full)
        for item in plan:
            if item[0] == "dense":
                _, src, dst, tidx, sidx, M = item
                vals = jnp.sum(M * full[src][sidx], axis=1)
                full = full.at[dst, tidx].set(vals)
                continue
            _, (kind, f), tidx, gathers, fn = item
            args = []
            for g in gathers:
                if g[0] == "field":
                    args.append(full[g[1]][g[2]])
                elif g[0] == "const":
                    args.append(g[1])
                else:
                    args.append(t)
            val = jnp.broadcast_to(fn(*args), tidx.shape)
            if kind == "fill":
                full = full.at[f, tidx].set(val)
            else:
                dfull = dfull.at[f, tidx].set(val)
        return dfull[sf, sp]

    return rhs
