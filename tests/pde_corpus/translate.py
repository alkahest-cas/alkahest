"""Translate corpus entries (``second_order.py``) to alkahest objects and to
Wolfram Language.  The equation syntax is a small subset of Python, parsed
with :mod:`ast` (never ``eval``-ed as Python)."""

from __future__ import annotations

import ast
import re

_FUNCS = {"sin": "Sin", "cos": "Cos", "exp": "Exp", "log": "Log", "sqrt": "Sqrt"}


def _deriv_name(name: str, vars_: str):
    """``u_xt`` → orders per variable, or ``None``."""
    m = re.fullmatch(r"u_([a-z]+)", name)
    if not m:
        return None
    orders = [0, 0]
    for ch in m.group(1):
        if ch not in vars_:
            return None
        orders[vars_.index(ch)] += 1
    return orders


# ---------------------------------------------------------------------------
# Wolfram Language
# ---------------------------------------------------------------------------


def _wl(node, vars_: str) -> str:
    a, b = vars_[0], vars_[1]
    if isinstance(node, ast.Expression):
        return _wl(node.body, vars_)
    if isinstance(node, ast.BinOp):
        lhs, rhs = _wl(node.left, vars_), _wl(node.right, vars_)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}[
            type(node.op)
        ]
        return f"({lhs}){op}({rhs})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-({_wl(node.operand, vars_)}))"
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        if node.id == "u":
            return f"u[{a},{b}]"
        orders = _deriv_name(node.id, vars_)
        if orders is not None:
            return f"Derivative[{orders[0]},{orders[1]}][u][{a},{b}]"
        if node.id == "pi":
            return "Pi"
        return node.id
    if isinstance(node, ast.Call):
        name = node.func.id
        args = ",".join(_wl(x, vars_) for x in node.args)
        return f"{_FUNCS.get(name, name)}[{args}]"
    raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def _lhs_parts(lhs: str):
    m = re.fullmatch(r"u(?:_([a-z]))?\(([a-z]),(-?[0-9]+)\)", lhs.replace(" ", ""))
    if not m:
        raise ValueError(f"unsupported condition {lhs!r}")
    return m.group(1), m.group(2), m.group(3)


def to_mathematica(entry) -> tuple[str, str, str]:
    """``(pde, ics, assumptions)`` in Wolfram Language."""
    v = entry["vars"]
    pde = _wl(ast.parse(entry["eq"], mode="eval"), v) + " == 0"
    ics = []
    for lhs, rhs in entry.get("ics", []):
        d, _x, at = _lhs_parts(lhs)
        r = _wl(ast.parse(rhs, mode="eval"), v)
        if d is None:
            ics.append(f"u[{v[0]},{at}] == {r}")
        else:
            ics.append(f"Derivative[0,1][u][{v[0]},{at}] == {r}")
    assum = " && ".join(f"{k} > 0" for k, d in entry.get("params", {}).items() if d == "positive")
    return pde, "{" + ", ".join(ics) + "}", assum or "True"


# ---------------------------------------------------------------------------
# alkahest
# ---------------------------------------------------------------------------


def to_alkahest(entry, pool):
    """``(equation, unknown, ics)`` as alkahest expressions; ``ics`` is a list
    of ``(lhs, rhs)`` pairs."""
    import alkahest as ak
    import alkahest.experimental as akx

    v = entry["vars"]
    params = entry.get("params", {})

    def sym(name):
        dom = params.get(name)
        return pool.symbol(name, dom) if dom else pool.symbol(name)

    X, Y = sym(v[0]), sym(v[1])
    u = akx.Function("u")(X, Y)
    funcs = {"sin": ak.sin, "cos": ak.cos, "exp": ak.exp, "log": ak.log, "sqrt": ak.sqrt}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp):
            lhs, rhs = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                if isinstance(lhs, int) and isinstance(rhs, int):
                    return pool.rational(lhs, rhs)
                return lhs / rhs
            if isinstance(node.op, ast.Pow):
                return lhs**rhs
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "u":
                return u
            orders = _deriv_name(node.id, v)
            if orders is not None:
                return akx.D(u, (X, orders[0]), (Y, orders[1]))
            if node.id == v[0]:
                return X
            if node.id == v[1]:
                return Y
            return sym(node.id)
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [ev(a) for a in node.args]
            if name in funcs:
                return funcs[name](*args)
            return akx.Function(name)(*args)
        raise ValueError(f"unsupported syntax: {ast.dump(node)}")

    def expr(s):
        r = ev(ast.parse(s, mode="eval"))
        return pool.integer(r) if isinstance(r, int) else r

    eq = expr(entry["eq"])
    ics = []
    for lhs, rhs in entry.get("ics", []):
        d, _x, at = _lhs_parts(lhs)
        at_e = pool.integer(int(at))
        if d is None:
            ics.append((akx.Function("u")(X, at_e), expr(rhs)))
        else:
            ics.append((ak.subs(akx.D(u, Y), {Y: at_e}), expr(rhs)))
    return eq, u, ics
