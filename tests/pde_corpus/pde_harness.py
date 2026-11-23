"""Shared harness for the first-order PDE corpus (milestone M1).

A corpus entry is a dict:

``id``       stable identifier
``source``   where the problem comes from
``vars``     independent variables, e.g. ``"x y"`` (the unknown is ``u(vars)``)
``eq``       the equation as a Python expression in the namespace below,
             read as ``= 0`` (or ``lhs == rhs`` written as ``(lhs, rhs)``)
``ics``      optional list of ``(lhs, rhs)`` strings, e.g. ``("U(x, 0)", "sin(x)")``
``params``   optional extra symbol names (``"a b c"``)

Namespace: the variables and parameters as symbols, ``u`` the applied unknown,
``U`` the unknown as a function (``U(x, 0)``), ``ux``/``uy``/``ut``/``uz`` the
first partials, ``D``, ``f``/``g`` undefined functions, and ``exp``, ``log``,
``sin``, ``cos``, ``tan``, ``sqrt``, ``sinh``, ``cosh``, ``tanh``, ``atan`` (and the
other elementary functions); any other name is a symbol (a parameter).
"""

from __future__ import annotations

import re
import time

import alkahest as ak
import alkahest.experimental as akx


class _Namespace(dict):
    """Unknown names become symbols of the problem's pool."""

    def __init__(self, pool):
        super().__init__()
        self._pool = pool

    def __missing__(self, key):
        v = self._pool.symbol(key)
        self[key] = v
        return v


def build(entry: dict):
    pool = ak.ExprPool()
    names = entry["vars"].split()
    vs = [pool.symbol(n) for n in names]
    U = akx.Function("u")
    u = U(*vs)
    ns = _Namespace(pool)
    ns.update(dict(zip(names, vs)))
    for pn in entry.get("params", "").split():
        ns[pn] = pool.symbol(pn)
    ns.update(
        u=u,
        U=U,
        D=akx.D,
        f=akx.Function("f"),
        g=akx.Function("g"),
        h=akx.Function("h"),
        Derivative=akx.D,
        cot=lambda a: 1 / ak.tan(a),
        coth=lambda a: 1 / ak.tanh(a),
        sec=lambda a: 1 / ak.cos(a),
        csc=lambda a: 1 / ak.sin(a),
        acot=lambda a: ak.atan(1 / a),
        asin=ak.asin,
        acos=ak.acos,
        asinh=ak.asinh,
        acosh=ak.acosh,
        atanh=ak.atanh,
        E=ak.exp(pool.integer(1)),
        Rational=lambda a, b: pool.rational(a, b),
        exp=ak.exp,
        log=ak.log,
        sin=ak.sin,
        cos=ak.cos,
        tan=ak.tan,
        sqrt=ak.sqrt,
        sinh=ak.sinh,
        cosh=ak.cosh,
        tanh=ak.tanh,
        atan=ak.atan,
        pool=pool,
        R=lambda a, b: pool.rational(a, b),
    )
    for n, v in zip(names, vs):
        ns["u" + n] = akx.D(u, v)
    sources = [entry["eq"]] + [t for ic in entry.get("ics", ()) for t in ic]
    for src in sources:
        for name in re.findall(r"\b([A-Za-z_]\w*)\(", src):
            if name not in ns:
                ns[name] = akx.Function(name)
    eq = eval(entry["eq"], {"__builtins__": {}}, ns)
    if isinstance(eq, tuple):
        eq = eq[0] - eq[1]
    if not isinstance(eq, ak.Expr):
        eq = pool.integer(eq)
    pde = akx.PDE(eq, u)
    for lhs, rhs in entry.get("ics", ()):
        lv = eval(lhs, {"__builtins__": {}}, ns)
        rv = eval(rhs, {"__builtins__": {}}, ns)
        if not isinstance(rv, ak.Expr):
            rv = pool.integer(rv)
        pde = pde.ic(lv, rv)
    return pde, u


def run_entry(entry: dict) -> dict:
    """Solve one entry; the outcome dict never contains an unverified answer
    without saying so (``status == "unverified"`` is a corpus failure)."""
    t0 = time.perf_counter()
    try:
        pde, _ = build(entry)
    except Exception as e:  # pragma: no cover - corpus authoring error
        return {"id": entry["id"], "status": "build_error", "error": repr(e)}
    try:
        sol = akx.pdsolve(pde)
    except akx.PdeError as e:
        return {
            "id": entry["id"],
            "status": "refused",
            "code": getattr(e, "code", None),
            "error": str(e)[:300],
            "seconds": time.perf_counter() - t0,
        }
    out = {
        "id": entry["id"],
        "kind": sol.kind,
        "method": sol.method,
        "implicit": sol.implicit,
        "value": str(sol.value),
        "integral": "Integral(" in str(sol.value),
        "side_conditions": [s[2] for s in sol.side_conditions],
        "seconds": time.perf_counter() - t0,
    }
    if not sol.verified:
        out["status"] = "unverified"
        return out
    # Independent re-check of explicit answers through the M0 verifier.
    if not sol.implicit:
        v = akx.verify_pde_solution(pde, sol.value)
        if v["status"] != "exactly_verified":
            out["status"] = "unverified"
            out["recheck"] = v["status"]
            return out
    out["status"] = "verified"
    return out
