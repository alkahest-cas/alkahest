"""Execute a parsed Math-tab command with Alkahest.

Only :mod:`mathlang` syntax trees reach this module, and every tree node is
mapped onto an explicit whitelist of Alkahest calls (``_AK_FUNCS`` and the
``_run_*`` methods below).  There is no ``eval``/``exec`` and no attribute
lookup driven by user text.  Each request runs inside
``ak.context(budget=ak.Budget(wall_ms=..., max_bytes=...))``; the caller
(:mod:`alkahest.playground.service`) additionally runs this in a separate, resource-limited
process with a hard timeout, because a single FLINT call is not interruptible
by the cooperative budget.
"""

from __future__ import annotations

import math
import re
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import alkahest as ak

from . import mathlang as ml
from .mathlang import Command, MathInputError, Node, Session

DEFAULT_WALL_MS = 8_000
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
MAX_RESULT_TEXT = 20_000
MAX_RESULT_LATEX = 40_000

# Name in the input language → Alkahest callable.  Built from a literal list so
# no user-supplied string is ever used as an attribute name.
_AK_FUNCS: dict[str, Any] = {
    name: getattr(ak, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "gamma",
        "erf",
        "erfc",
        "zeta",
        "atan2",
        "min",
        "max",
    )
    if hasattr(ak, name)
}

#: Symbols Alkahest treats as named constants rather than free variables.
_CONSTANT_SYMBOLS = {"pi", "I", "∞", "-∞", "E", "EulerGamma", "Catalan", "Glaisher", "GoldenRatio"}

_VERIFIED_STATUSES = {"exactly_verified", "lean_checked"}


def _err(code: str, message: str, hint: str | None = None) -> MathInputError:
    return MathInputError(code, message, hint)


class Builder:
    """Turn a :class:`mathlang.Node` into an Alkahest ``Expr``."""

    def __init__(self, pool, session: Session):
        self.pool = pool
        self.session = session
        self.cache: dict[str, Any] = {}
        self.resolving: set[str] = set()
        self._pi = ak.parse("pi", pool)

    def sym(self, name: str):
        return self.pool.symbol(name)

    def build(self, node: Node, bound: frozenset = frozenset()):
        k = node.kind
        p = self.pool
        if k == "num":
            if "." in node.value:
                q = Fraction(Decimal(node.value))
                return p.rational(q.numerator, q.denominator)
            return p.integer(int(node.value))
        if k == "const":
            if node.value == "pi":
                return self._pi
            if node.value == "e":
                return ak.exp(p.integer(1))
            if node.value == "i":
                return p.imaginary_unit()
            return p.pos_infinity()
        if k == "sym":
            return self.resolve(node.value, bound)
        if k == "neg":
            c = node.children[0]
            if c.kind == "const" and c.value == "oo":
                return p.neg_infinity()
            return -self.build(c, bound)
        if k in ("add", "sub", "mul", "div"):
            a_node, b_node = node.children
            if (
                k == "div"
                and a_node.kind == "num"
                and b_node.kind == "num"
                and a_node.value.isdigit()
                and b_node.value.isdigit()
            ):
                if int(b_node.value) == 0:
                    raise _err(ml.E_UNSUPPORTED, "Division by zero.")
                return p.rational(int(a_node.value), int(b_node.value))
            a = self.build(a_node, bound)
            b = self.build(b_node, bound)
            if k == "add":
                return a + b
            if k == "sub":
                return a - b
            if k == "mul":
                return a * b
            return a / b
        if k == "pow":
            base, ex = node.children
            if base.kind == "const" and base.value == "e":
                return ak.exp(self.build(ex, bound))
            return self.build(base, bound) ** self.build(ex, bound)
        if k == "fact":
            inner = node.children[0]
            if inner.kind == "num" and inner.value.isdigit():
                return p.integer(math.factorial(int(inner.value)))
            return ak.gamma(self.build(inner, bound) + 1)
        if k == "call":
            return self.call(node.value, [self.build(c, bound) for c in node.children])
        if k == "ucall":
            params, body = self.session.functions[node.value]
            body_expr = self.build(body, frozenset(params))
            args = [self.build(c, bound) for c in node.children]
            return ak.subs(body_expr, {self.sym(n): a for n, a in zip(params, args)})
        raise _err(ml.E_INTERNAL, f"unknown node {k}")

    def resolve(self, name: str, bound: frozenset):
        if name in bound:
            return self.sym(name)
        if name in self.session.names or name in self.session.functions:
            if name in self.cache:
                return self.cache[name]
            if name in self.resolving:
                raise _err(ml.E_UNSUPPORTED, f"{name} is defined in terms of itself.")
            self.resolving.add(name)
            try:
                if name in self.session.names:
                    value = self.build(self.session.names[name])
                else:
                    params, body = self.session.functions[name]
                    value = self.build(body, frozenset(params))
            finally:
                self.resolving.discard(name)
            self.cache[name] = value
            return value
        return self.sym(name)

    def call(self, name: str, args: list):
        p = self.pool
        if name == "sec":
            return p.integer(1) / ak.cos(args[0])
        if name == "csc":
            return p.integer(1) / ak.sin(args[0])
        if name == "cot":
            return ak.cos(args[0]) / ak.sin(args[0])
        if name == "cbrt":
            return args[0] ** p.rational(1, 3)
        if name == "log" and len(args) == 2:
            return ak.log(args[0]) / ak.log(args[1])
        if name == "log10":
            return ak.log(args[0]) / ak.log(p.integer(10))
        if name == "log2":
            return ak.log(args[0]) / ak.log(p.integer(2))
        if name == "factorial":
            return ak.gamma(args[0] + 1)
        fn = _AK_FUNCS.get(name)
        if fn is None:
            raise _err(ml.E_UNKNOWN_NAME, f"{name} is not available in this Alkahest build.")
        return fn(*args)


# ── Expression inspection ────────────────────────────────────────────────────


def free_symbols(expr) -> list[str]:
    """Names of the free symbols in an Alkahest expression (constants excluded)."""
    out: set[str] = set()
    stack = [expr]
    seen = 0
    while stack:
        e = stack.pop()
        seen += 1
        if seen > 200_000:
            break
        node = e.node()
        tag = node[0]
        if tag == "symbol":
            if node[1] not in _CONSTANT_SYMBOLS:
                out.add(node[1])
            continue
        for part in node[1:]:
            if isinstance(part, list):
                stack.extend(x for x in part if isinstance(x, ak.Expr))
            elif isinstance(part, ak.Expr):
                stack.append(part)
    return sorted(out)


def pick_var(expr, what: str) -> str:
    names = free_symbols(expr)
    if "x" in names:
        return "x"
    if len(names) == 1:
        return names[0]
    if not names:
        raise _err(
            ml.E_NEED_VARIABLE,
            f"There is no variable to {what} with respect to.",
        )
    raise _err(
        ml.E_NEED_VARIABLE,
        f"Which variable? The expression has {', '.join(names)}.",
        "Say it explicitly, e.g. with dx, d/dy, 'with respect to t' or 'as t -> 0'.",
    )


_E1 = re.compile(r"e\^1(?![0-9])")
_IMAG = re.compile(r"(?<![A-Za-z\\])I(?![A-Za-z])")


def latex_of(expr) -> str:
    s = ak.latex(expr)
    s = _E1.sub("e", s)
    s = s.replace("-∞", r"-\infty").replace("∞", r"\infty")
    s = _IMAG.sub("i", s)
    return s


def text_of(expr) -> str:
    return str(expr)


# ── Results ──────────────────────────────────────────────────────────────────


def _derived_meta(r) -> dict:
    """Certificate / verification fields of a DerivedResult (or nothing)."""
    out: dict[str, Any] = {}
    ver = getattr(r, "verification", None)
    if isinstance(ver, dict):
        status = ver.get("status")
        if isinstance(status, str):
            out["verification"] = status
            out["verified"] = status in _VERIFIED_STATUSES
        side = ver.get("side_conditions")
        if side:
            out["side_conditions"] = [str(s) for s in side][:10]
    cert = getattr(r, "certificate", None)
    if isinstance(cert, str) and cert.strip():
        out["certificate"] = cert
    steps = getattr(r, "steps", None)
    if isinstance(steps, list):
        out["steps"] = len(steps)
    return out


def _value(r):
    return r.value if hasattr(r, "value") and not isinstance(r, ak.Expr) else r


class Executor:
    def __init__(self, definitions: list[dict], wall_ms: int, max_bytes: int):
        self.pool = ak.ExprPool()
        self.session = Session()
        self.wall_ms = wall_ms
        self.max_bytes = max_bytes
        self._load_definitions(definitions)
        self.builder = Builder(self.pool, self.session)

    def _load_definitions(self, definitions: list[dict]) -> None:
        self.session = ml.build_session(definitions)

    # entry
    def run(self, text: str) -> dict:
        t0 = time.monotonic()
        cmd, interp = ml.interpret(text, self.session)
        out: dict[str, Any] = {"interpreted_latex": interp, "kind": cmd.kind}
        with ak.context(
            pool=self.pool,
            budget=ak.Budget(wall_ms=self.wall_ms, max_bytes=self.max_bytes),
        ):
            if cmd.kind == "define":
                out.update(self.define(cmd))
            else:
                out.update(self.dispatch(cmd))
        out["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return out

    def dispatch(self, cmd: Command) -> dict:
        k = cmd.kind
        fn = {
            "expr": self.run_simplify,
            "simplify": self.run_simplify,
            "integrate": self.run_integrate,
            "diff": self.run_diff,
            "solve": self.run_solve,
            "factor": self.run_factor,
            "expand": self.run_expand,
            "limit": self.run_limit,
            "series": self.run_series,
            "sum": self.run_sum,
            "product": self.run_sum,
            "numeric": self.run_numeric,
        }.get(k)
        if fn is None:
            raise _err(ml.E_UNSUPPORTED, f"Unsupported command {k}.")
        return fn(cmd)

    def define(self, cmd: Command) -> dict:
        assert cmd.body is not None
        assert cmd.name is not None
        body = cmd.body
        params = cmd.params
        if body.kind == "expr" and body.expr is not None:
            # Definitions keep the expression as written (no simplification), so
            # f = x^2 sin(x) stays exactly that.
            value = self.builder.build(body.expr, frozenset(params))
            res: dict[str, Any] = {"latex": latex_of(value), "text": text_of(value)}
        else:
            res = self.dispatch(body)
            value = res.pop("_value", None)
            if value is None:
                raise _err(
                    ml.E_UNSUPPORTED,
                    "This result is not a single expression, so it cannot be given a name.",
                )
        lhs = ml.sym_latex(cmd.name)
        if params:
            lhs += r"\left(" + ", ".join(ml.sym_latex(q) for q in params) + r"\right)"
        res["latex"] = lhs + " = " + res["latex"]
        res["definition"] = {"name": cmd.name, "params": params, "value": text_of(value)}
        res.pop("_value", None)
        return res

    def expr(self, node: Node | None, bound=frozenset()):
        assert node is not None
        return self.builder.build(node, frozenset(bound))

    def _result(self, value, meta: dict | None = None, **extra) -> dict:
        out = {"latex": latex_of(value), "text": text_of(value), "_value": value}
        if meta:
            out.update(meta)
        out.update(extra)
        return out

    # simplify chain
    def run_simplify(self, cmd: Command) -> dict:
        e = self.expr(cmd.expr)
        start = text_of(e)
        current = e
        changed: list[tuple[str, Any]] = []
        for name, fn in (
            ("simplify", ak.simplify),
            ("simplify_trig", ak.simplify_trig),
            ("simplify_log_exp", ak.simplify_log_exp),
        ):
            if name != "simplify" and not _mentions(current, name):
                continue
            r = fn(current)
            if text_of(r.value) != text_of(current):
                changed.append((name, r))
                current = r.value
        # cancel common polynomial factors ((x^2-1)/(x-1) → x+1)
        if "^-1" in text_of(current) or "/" in text_of(current):
            try:
                c = ak.cancel(current)
                c_val = _value(c)
                if text_of(c_val) != text_of(current) and len(text_of(c_val)) <= len(
                    text_of(current)
                ):
                    changed.append(("cancel", c))
                    current = c_val
            except ak.AlkahestError:
                pass
        meta: dict[str, Any] = {}
        if len(changed) == 1:
            meta = _derived_meta(changed[0][1])
        notes = []
        if changed:
            notes.append("Alkahest steps: " + " → ".join(n for n, _ in changed))
        elif text_of(current) == start:
            notes.append("Already in simplest form for Alkahest's simplifier.")
        res = self._result(current, meta, notes=notes)
        approx = self._approx(current)
        if approx:
            res["approx"] = approx
        return res

    def _approx(self, value) -> str | None:
        if free_symbols(value):
            return None
        node = value.node()
        if node[0] == "integer":
            return None
        try:
            r = ak.evaluate(value, {}, precision_bits=64)
        except (ak.AlkahestError, TypeError, ValueError):
            return None
        if getattr(r, "status", None) != "ok" or not getattr(r, "is_enclosure", False):
            return None
        return _ball_digits(r.value, 15)

    def run_integrate(self, cmd: Command) -> dict:
        f = self.expr(cmd.expr)
        var = cmd.var or pick_var(f, "integrate")
        if not cmd.var_explicit:
            f = self.expr(cmd.expr, {var})
        v = self.builder.sym(var)
        if cmd.lower is not None and cmd.upper is not None:
            r = ak.integrate(f, v, self.expr(cmd.lower), self.expr(cmd.upper))
        else:
            r = ak.integrate(f, v)
        notes = []
        if cmd.lower is None:
            notes.append("Antiderivative; add your own constant of integration.")
        return self._result(r.value, _derived_meta(r), notes=notes)

    def run_diff(self, cmd: Command) -> dict:
        f = self.expr(cmd.expr)
        var = cmd.var or pick_var(f, "differentiate")
        f = self.expr(cmd.expr, {var})
        v = self.builder.sym(var)
        r = None
        cur = f
        for _ in range(cmd.order):
            r = ak.diff(cur, v)
            cur = r.value
        meta = _derived_meta(r) if cmd.order == 1 else {}
        if cmd.order > 1 and r is not None:
            ver = getattr(r, "verification", None)
            if isinstance(ver, dict) and isinstance(ver.get("status"), str):
                meta["verification"] = ver["status"]
        return self._result(cur, meta)

    def run_solve(self, cmd: Command) -> dict:
        eqs = []
        for lhs, rhs in cmd.equations:
            left = self.expr(lhs)
            if rhs.kind == "num" and rhs.value == "0":
                eqs.append(left)
            else:
                eqs.append(left - self.expr(rhs))
        names = list(cmd.vars)
        if not names:
            found: set[str] = set()
            for e in eqs:
                found.update(free_symbols(e))
            names = sorted(found)
            if not names:
                raise _err(ml.E_NEED_VARIABLE, "There is no unknown to solve for.")
            if len(names) > len(eqs):
                raise _err(
                    ml.E_NEED_VARIABLE,
                    f"{len(eqs)} equation(s) in {len(names)} unknowns ({', '.join(names)}).",
                    "Say which to solve for, e.g. 'solve a x + b = 0 for x'.",
                )
        vars_ = [self.builder.sym(n) for n in names]
        sols = ak.solve(eqs, vars_)
        if not isinstance(sols, list):
            raise _err(
                ml.E_UNSUPPORTED,
                "The solution set is not a finite list of points "
                "(Alkahest returned a Gröbner basis instead).",
                "Add equations or list fewer unknowns.",
            )
        rows = []
        texts = []
        for s in sols:
            if not isinstance(s, dict):
                continue
            pairs = []
            for n, v in zip(names, vars_):
                val = s.get(v)
                if val is None:
                    continue
                if isinstance(val, ak.Expr):
                    pairs.append((n, latex_of(val), text_of(val)))
                else:
                    pairs.append((n, str(val), str(val)))
            rows.append(pairs)
            texts.append(", ".join(f"{n} = {t}" for n, _, t in pairs))
        if not rows:
            return {"latex": r"\text{No solutions}", "text": "no solutions", "notes": []}
        order = sorted(range(len(rows)), key=lambda i: texts[i])
        rows = [rows[i] for i in order]
        texts = [texts[i] for i in order]
        if len(names) == 1:
            parts = [rf"{ml.sym_latex(n)} = {lx}" for row in rows for n, lx, _ in row]
            latex = r"\quad\text{or}\quad ".join(parts)
        else:
            lines = [r",\ ".join(rf"{ml.sym_latex(n)} = {lx}" for n, lx, _ in row) for row in rows]
            latex = r"\begin{aligned}" + r"\\ ".join(lines) + r"\end{aligned}"
        n = len(rows)
        return {
            "latex": latex,
            "text": "; ".join(texts),
            "notes": [f"{n} solution{'s' if n != 1 else ''}."],
        }

    def run_factor(self, cmd: Command) -> dict:
        e = self.expr(cmd.expr)
        names = free_symbols(e)
        p = self.pool
        if not names:
            node = e.node()
            if node[0] == "integer":
                from alkahest.number_theory import factorint

                n = int(node[1])
                if abs(n) < 2:
                    return self._result(e)
                fac = factorint(abs(n))
                parts = [(str(q) if m == 1 else f"{q}^{{{m}}}") for q, m in sorted(fac.items())]
                sign = "-" if n < 0 else ""
                text = (sign and "-") + " * ".join(
                    str(q) if m == 1 else f"{q}^{m}" for q, m in sorted(fac.items())
                )
                return {"latex": sign + r" \cdot ".join(parts), "text": text}
            raise _err(ml.E_UNSUPPORTED, "factor works on polynomials and whole numbers.")
        vars_ = [self.builder.sym(n) for n in names]
        if len(vars_) == 1:
            fz = ak.UniPoly.from_symbolic(e, vars_[0]).factor_z()
            factors = [(_unipoly_expr(f, vars_[0], p), m) for f, m in fz.factor_list()]
        else:
            fz = ak.MultiPoly.from_symbolic(e, vars_).factor_z()
            factors = [(self._multipoly_expr(f, names), m) for f, m in fz.factor_list()]
        unit = fz.unit
        latex_parts = []
        text_parts = []
        unit_int = int(unit) if not isinstance(unit, ak.Expr) else None
        if unit_int is not None and unit_int not in (1, -1):
            latex_parts.append(str(unit_int))
            text_parts.append(str(unit_int))
        for f, m in factors:
            fl = latex_of(f)
            ft = text_of(f)
            if (len(factors) > 1 or m > 1 or latex_parts) and (" + " in fl or " - " in fl):
                fl = r"\left(" + fl + r"\right)"
                if not ft.startswith("("):
                    ft = "(" + ft + ")"
            if m > 1:
                fl = fl + "^{" + str(m) + "}"
                ft = ft + "^" + str(m)
            latex_parts.append(fl)
            text_parts.append(ft)
        latex = " ".join(latex_parts)
        text = " * ".join(text_parts)
        if unit_int == -1:
            latex, text = "-" + latex, "-(" + text + ")"
        meta = {}
        ver = getattr(fz, "verification", None)
        if isinstance(ver, dict) and isinstance(ver.get("status"), str):
            meta["verification"] = ver["status"]
            meta["verified"] = ver["status"] in _VERIFIED_STATUSES
        # A factored product, as one expression, for definitions.
        value = p.integer(unit_int if unit_int is not None else 1)
        for f, m in factors:
            value = value * (f**m if m > 1 else f)
        return {"latex": latex, "text": text, "_value": value, **meta}

    def _multipoly_expr(self, poly, names: list[str]):
        # MultiPoly prints over generic generators x0, x1, …; read it back with
        # the restricted parser, mapping generator i to the i-th variable.
        s = str(poly)
        m = re.fullmatch(r"MultiPoly\((.*)\)", s, re.S)
        body = m.group(1) if m else s
        sess = Session()
        cp = ml.CommandParser(sess)
        gens = {f"x{i}" for i in range(len(names))}
        node = cp.expr(ml.tokenize(body)[:-1], bound=gens)
        expr = Builder(self.pool, sess).build(node, frozenset(gens))
        mapping = {self.pool.symbol(f"x{i}"): self.builder.sym(n) for i, n in enumerate(names)}
        return ak.subs(expr, mapping)

    def run_expand(self, cmd: Command) -> dict:
        r = ak.simplify_expanded(self.expr(cmd.expr))
        return self._result(r.value, _derived_meta(r))

    def run_limit(self, cmd: Command) -> dict:
        assert cmd.var is not None
        f = self.expr(cmd.expr, {cmd.var})
        v = self.builder.sym(cmd.var)
        point = self.expr(cmd.point)
        r = ak.limit(f, v, point, dir=cmd.direction) if cmd.direction else ak.limit(f, v, point)
        return self._result(_value(r), _derived_meta(r) if r is not _value(r) else None)

    def run_series(self, cmd: Command) -> dict:
        f = self.expr(cmd.expr)
        var = cmd.var or pick_var(f, "expand")
        f = self.expr(cmd.expr, {var})
        v = self.builder.sym(var)
        s = ak.series(f, v, self.expr(cmd.point), cmd.order)
        res = self._result(s.expr)
        res["_value"] = s.truncated()
        return res

    def run_sum(self, cmd: Command) -> dict:
        assert cmd.var is not None
        f = self.expr(cmd.expr, {cmd.var})
        k = self.builder.sym(cmd.var)
        fn = ak.sum_definite if cmd.kind == "sum" else ak.product_definite
        r = fn(f, k, self.expr(cmd.lower), self.expr(cmd.upper))
        meta = _derived_meta(r)
        if not meta.get("side_conditions"):
            steps = getattr(r, "steps", None) or []
            side = [
                str(c)
                for st in steps
                if isinstance(st, dict)
                for c in st.get("side_conditions", []) or []
            ]
            if side:
                meta["side_conditions"] = side[:10]
        return self._result(r.value, meta)

    def run_numeric(self, cmd: Command) -> dict:
        e = self.expr(cmd.expr)
        names = free_symbols(e)
        if names:
            raise _err(
                ml.E_UNSUPPORTED,
                f"Numeric evaluation needs numbers, but {', '.join(names)} has no value.",
                "Define it first (e.g. x = 2), or use a number.",
            )
        digits = cmd.digits
        bits = math.ceil(digits * math.log2(10)) + 32
        r = ak.evaluate(e, {}, precision_bits=bits)
        if getattr(r, "status", None) != "ok" or not getattr(r, "is_enclosure", False):
            r2 = ak.evaluate(e, {}, mode="complex", precision_bits=bits)
            if getattr(r2, "status", None) == "ok":
                val = r2.value
                text = str(val)
                return {"latex": r"\approx " + _escape_tex(text), "text": text, "notes": []}
            reason = getattr(r, "reason", None) or getattr(r, "status", "unsupported")
            raise _err(
                ml.E_UNSUPPORTED,
                f"Alkahest could not evaluate this numerically ({reason}).",
            )
        ball = r.value
        text = _ball_digits(ball, digits)
        try:
            rad = f"{float(ball.rad_exact()):.1e}"
        except (AttributeError, OverflowError, TypeError, ValueError):
            rad = ""
        notes = [f"Certified: rigorous ball enclosure at {bits} bits, radius {rad}."] if rad else []
        latex = r"\approx " + _group_digits(text)
        q = Fraction(Decimal(text)) if "e" not in text.lower() else None
        value = self.pool.rational(q.numerator, q.denominator) if q is not None else None
        return {
            "latex": latex,
            "text": text,
            "notes": notes,
            "verification": "rigorous_enclosure",
            "verified": True,
            "_value": value,
        }


def _mentions(expr, which: str) -> bool:
    s = text_of(expr)
    if which == "simplify_trig":
        return any(f in s for f in ("sin(", "cos(", "tan("))
    return "log(" in s or "exp(" in s


def _unipoly_expr(poly, var, pool):
    """Expr for a UniPoly from its (ascending) integer coefficients."""
    expr = None
    for i, c in enumerate(poly.coefficients()):
        c = int(c)
        if c == 0:
            continue
        mono = var if i == 1 else var**i
        if i == 0:
            term = pool.integer(c)
        elif c == 1:
            term = mono
        elif c == -1:
            term = -mono
        else:
            term = pool.integer(c) * mono
        expr = term if expr is None else expr + term
    return expr if expr is not None else pool.integer(0)


def _ball_digits(ball, digits: int) -> str:
    """Round an ArbBall's midpoint to ``digits`` significant digits."""
    mid = ball.mid_exact()
    if not isinstance(mid, Fraction):
        mid = Fraction(mid)
    with localcontext() as ctx:
        ctx.prec = digits
        d = Decimal(mid.numerator) / Decimal(mid.denominator)
        d = +d
    s = format(d, "f") if abs(d.adjusted()) < 30 else format(d, "e")
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".") if digits <= 15 else s
    return s


def _group_digits(text: str) -> str:
    """Thin spaces every 5 digits after the point, so 50 digits stay readable."""
    if "." not in text or "e" in text:
        return text
    head, tail = text.split(".", 1)
    groups = [tail[i : i + 5] for i in range(0, len(tail), 5)]
    return head + "." + r"\,".join(groups)


def _escape_tex(s: str) -> str:
    return re.sub(r"([#$%&_{}])", r"\\\1", s)


# ── Error mapping ────────────────────────────────────────────────────────────

_CODE_RE = re.compile(r"^\s*\[(E-[A-Z0-9]+-\d+)\]\s*")


def describe_exception(exc: BaseException) -> dict:
    """Map an exception to the response's error fields."""
    if isinstance(exc, MathInputError):
        return {
            "error_code": exc.code,
            "error_message": exc.message,
            "error_hint": exc.hint,
            "error_source": "input" if exc.code != ml.E_INTERNAL else "playground",
        }
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.startswith("E-"):
        raw = str(exc)
        raw = _CODE_RE.sub("", raw)
        message, _, rest = raw.partition("\nRemediation:")
        hint = getattr(exc, "remediation", None) or rest.strip() or None
        message = message.strip()
        # "integrate: not implemented: …" reads better without the op prefix
        message = re.sub(r"^[a-z_]+: ", "", message, count=1)
        first, _, detail = message.partition("\n")
        return {
            "error_code": code,
            "error_message": first.strip()[:1000],
            "error_detail": detail.strip()[:2000] or None,
            "error_hint": (hint or "")[:2000] or None,
            "error_source": "alkahest",
        }
    if isinstance(exc, MemoryError):
        return {
            "error_code": ml.E_MEMORY,
            "error_message": "The computation ran out of its memory allowance.",
            "error_hint": None,
            "error_source": "playground",
        }
    if isinstance(exc, RecursionError):
        return {
            "error_code": ml.E_TOO_LARGE_INPUT,
            "error_message": "The expression is nested too deeply.",
            "error_hint": None,
            "error_source": "input",
        }
    return {
        "error_code": ml.E_INTERNAL,
        "error_message": f"Unexpected {type(exc).__name__}: {str(exc)[:500]}",
        "error_hint": "This is a playground bug, not a verdict about the maths.",
        "error_source": "playground",
    }


def execute(request: dict) -> dict:
    """Run one Math-tab request; never raises."""
    text = request.get("input", "")
    defs = request.get("definitions") or []
    wall_ms = int(request.get("wall_ms") or DEFAULT_WALL_MS)
    max_bytes = int(request.get("max_bytes") or DEFAULT_MAX_BYTES)
    interp: str | None = None
    try:
        if not isinstance(text, str):
            raise _err(ml.E_SYNTAX, "The input must be text.")
        if not isinstance(defs, list):
            raise _err(ml.E_SYNTAX, "Malformed definitions.")
        executor = Executor(defs, wall_ms, max_bytes)
        # interpretation first, so a failing computation still shows it
        interp = ml.interpret(text, executor.session)[1]
        out = executor.run(text)
        out.pop("_value", None)
        if (
            len(out.get("text", "")) > MAX_RESULT_TEXT
            or len(out.get("latex", "")) > MAX_RESULT_LATEX
        ):
            raise _err(
                ml.E_TOO_LARGE_INPUT,
                "The result is too large to display in the playground.",
            )
        out["ok"] = True
        return out
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        res = {"ok": False, "interpreted_latex": interp}
        res.update(describe_exception(exc))
        return res
