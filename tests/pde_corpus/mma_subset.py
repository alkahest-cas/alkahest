"""A small Wolfram Language subset → alkahest translator for the separation
corpus (``separation_corpus.json``) and the bounded-domain items of Abbasi's
PDE benchmark (``abbasi_bounded.json``), whose problems are stored as the
benchmark's own Mathematica input.

Supported: numbers, symbols, strings, ``f[...]`` calls (also chained, as in
``Derivative[1, 0][u][0, t]``), lists, ``+ - * / ^``, comparisons,
``&&`` / ``||``, rules, assignments ``a = …;`` and definitions
``f[x_] := …;``.  The input is tokenised and parsed, never evaluated as
Python or Wolfram code.

:func:`build` returns the alkahest :class:`PDE` with an inferred domain:
the time variable (the one with initial conditions) on a half-line, every
other Cartesian variable on the interval between the two faces where
conditions are given, and polar / cylindrical / spherical coordinates
(``r, theta, z, phi``) on a :class:`Curvilinear` region.  A problem on an
unbounded domain raises :class:`Unsupported`.
"""

from __future__ import annotations

import re
from typing import ClassVar

import alkahest as ak
import alkahest.experimental as akx


class Unsupported(Exception):
    """The input is outside the translated subset (or the domain is
    unbounded)."""


# ---------------------------------------------------------------------------
# Tokeniser and parser
# ---------------------------------------------------------------------------

_TOK = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*|\d+)|(?P<str>\"[^\"]*\")|(?P<name>[A-Za-z$][A-Za-z0-9$]*)"
    r"|(?P<op>:=|==|!=|->|<=|>=|&&|\|\||[-+*/^=<>\[\]{}(),;_.!']))"
)


def _tokens(src: str):
    pos = 0
    out = []
    while pos < len(src):
        m = _TOK.match(src, pos)
        if not m or m.end() == pos:
            if src[pos:].strip() == "":
                break
            raise Unsupported(f"cannot tokenise at {src[pos : pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
    out.append(("end", ""))
    return out


class _Parser:
    PREC: ClassVar[dict] = {
        "||": 1,
        "&&": 2,
        "==": 3,
        "!=": 3,
        "<": 3,
        ">": 3,
        "<=": 3,
        ">=": 3,
        "+": 4,
        "-": 4,
        "*": 5,
        "/": 5,
        "^": 7,
    }

    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i]

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, v):
        _k, s = self.next()
        if s != v:
            raise Unsupported(f"expected {v!r}, got {s!r}")

    def statements(self):
        out = []
        while self.peek()[0] != "end":
            if self.peek()[1] == ";":
                self.next()
                continue
            start = self.i
            try:
                out.append(self.statement())
            except Unsupported:
                # Skip to the next top-level ';' (post-processing such as
                # `sol = sol /. K[1] -> n`).
                self.i = start
                depth = 0
                while self.peek()[0] != "end":
                    _k, s = self.next()
                    if s in ("[", "{", "("):
                        depth += 1
                    elif s in ("]", "}", ")"):
                        depth -= 1
                    elif s == ";" and depth <= 0:
                        break
                continue
            if self.peek()[1] == ";":
                self.next()
        return out

    def statement(self):
        lhs = self.expr(0)
        _k, s = self.peek()
        if s in ("=", ":="):
            self.next()
            rhs = self.expr(0)
            return ("assign", s, lhs, rhs)
        return ("expr", lhs)

    def expr(self, minp):
        left = self.unary()
        while True:
            k, s = self.peek()
            if s == "->":
                if minp > 0:
                    return left
                self.next()
                left = ("rule", left, self.expr(0))
                continue
            p = self.PREC.get(s) if k == "op" else None
            if p is None:
                # Implicit multiplication: a primary right after a primary.
                if k in ("num", "name") or s in ("(",):
                    p = 5
                    if p < minp:
                        return left
                    right = self.expr(p + 1)
                    left = ("bin", "*", left, right)
                    continue
                return left
            if p < minp:
                return left
            self.next()
            right = self.expr(p if s == "^" else p + 1)
            left = ("bin", s, left, right)

    def unary(self):
        _k, s = self.peek()
        if s == "-":
            self.next()
            return ("neg", self.expr(6))
        if s == "+":
            self.next()
            return self.expr(6)
        if s == "!":
            self.next()
            return ("not", self.expr(6))
        return self.postfix(self.primary())

    def primary(self):
        k, s = self.next()
        if k == "num":
            return ("num", s)
        if k == "str":
            return ("str", s[1:-1])
        if k == "name":
            if self.peek()[1] == "_":
                self.next()
                return ("pattern", s)
            return ("sym", s)
        if s == "(":
            e = self.expr(0)
            self.expect(")")
            return e
        if s == "{":
            items = []
            if self.peek()[1] != "}":
                items.append(self.expr(0))
                while self.peek()[1] == ",":
                    self.next()
                    items.append(self.expr(0))
            self.expect("}")
            return ("list", items)
        raise Unsupported(f"unexpected token {s!r}")

    def postfix(self, e):
        while self.peek()[1] == "[":
            self.next()
            args = []
            if self.peek()[1] != "]":
                args.append(self.expr(0))
                while self.peek()[1] == ",":
                    self.next()
                    args.append(self.expr(0))
            self.expect("]")
            e = ("call", e, args)
        return e


def parse_statements(src: str):
    return _Parser(_tokens(src)).statements()


# ---------------------------------------------------------------------------
# Problem extraction
# ---------------------------------------------------------------------------

_SKIP_HEADS = {"ClearAll", "AbsoluteTiming", "TimeConstrained", "Print", "Failed"}


def _dsolve_call(stmts):
    """Find ``DSolve[{…}, u[x, t], {x, t}, opts…]`` / ``DSolveValue`` in the
    statements (possibly wrapped in timing calls)."""

    def find(e):
        if isinstance(e, tuple):
            if (e[0] == "call" and e[1] == ("sym", "DSolve")) or (
                e[0] == "call" and e[1] == ("sym", "DSolveValue")
            ):
                return e
            for x in e[1:]:
                if isinstance(x, list):
                    for y in x:
                        r = find(y)
                        if r:
                            return r
                else:
                    r = find(x)
                    if r:
                        return r
        return None

    for s in stmts:
        r = find(s)
        if r:
            return r
    return None


class Problem:
    """Statements of one corpus entry, with the parts ``separation_solve``
    needs."""

    def __init__(self, code: str):
        self.stmts = parse_statements(code)
        call = _dsolve_call(self.stmts)
        if call is None:
            raise Unsupported("no DSolve call")
        args = call[2]
        self.eqs_ast = args[0]
        self.unknown_ast = args[1]
        defs = {st[2][1]: st[3] for st in self.stmts if st[0] == "assign" and st[2][0] == "sym"}
        if self.unknown_ast[0] == "sym" and self.unknown_ast[1] in defs:
            self.unknown_ast = defs[self.unknown_ast[1]]
        rest = [a for a in args[2:] if a[0] != "rule"]
        self.vars_ast = rest[0] if len(rest) == 1 else ("list", rest)
        self.assumptions = [a for a in args[2:] if a[0] == "rule"]
        if self.unknown_ast[0] != "call" or self.unknown_ast[1][0] != "sym":
            raise Unsupported("the unknown is not u[x, …]")
        self.uname = self.unknown_ast[1][1]
        vs = self.vars_ast[1] if self.vars_ast[0] == "list" else [self.vars_ast]
        self.var_names = [v[1] for v in vs]


# ---------------------------------------------------------------------------
# Evaluation to alkahest
# ---------------------------------------------------------------------------

_UNARY = {
    "Sin": ak.sin,
    "Cos": ak.cos,
    "Tan": ak.tan,
    "Exp": ak.exp,
    "Log": ak.log,
    "Sqrt": ak.sqrt,
    "Sinh": ak.sinh,
    "Cosh": ak.cosh,
    "Tanh": ak.tanh,
    "ArcTan": ak.atan,
}


_UNSUPPORTED = {
    "DiracDelta",
    "KroneckerDelta",
    "Integrate",
    "Sum",
    "NIntegrate",
    "Abs",
    "Max",
    "Min",
    "Floor",
    "Ceiling",
    "Mod",
    "SquareWave",
    "TriangleWave",
    "Boole",
    "If",
    "Which",
    "Erf",
    "LegendreP",
    "SphericalHarmonicY",
    "BesselJZero",
    "Hold",
    "Inactive",
}


class _Env:
    def __init__(self, pool, var_names, uname, positive):
        self.pool = pool
        self.vars = {}
        for v in var_names:
            dom = "positive" if v == "r" else None
            self.vars[v] = pool.symbol(v, dom) if dom else pool.symbol(v)
        self.var_list = [self.vars[v] for v in var_names]
        self.uname = uname
        self.u = akx.Function(uname)(*self.var_list)
        self.positive = positive
        self.params = {}
        self.defs = {}  # name -> value AST
        self.fdefs = {}  # name -> (param names, body AST)
        self.local = {}

    def sym(self, name):
        if name in self.local:
            return self.local[name]
        if name in self.vars:
            return self.vars[name]
        if name in self.defs:
            return self.eval(self.defs[name])
        if name == "Pi":
            return ak.parse("pi", self.pool)
        if name == "E":
            return ak.exp(self.pool.integer(1))
        if name == "I":
            return ak.parse("I", self.pool)
        if name not in self.params:
            dom = "positive" if name in self.positive else None
            self.params[name] = self.pool.symbol(name, dom) if dom else self.pool.symbol(name)
        return self.params[name]

    def num(self, v):
        if isinstance(v, int):
            return self.pool.integer(v)
        return v

    def eval(self, e):
        p = self.pool
        tag = e[0]
        if tag == "num":
            s = e[1]
            if "." in s:
                return p.float(float(s))
            return p.integer(int(s))
        if tag == "sym":
            return self.sym(e[1])
        if tag == "neg":
            return -self.eval(e[1])
        if tag == "bin":
            op = e[1]
            a, b = self.eval(e[2]), self.eval(e[3])
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "^":
                return a**b
            preds = {"<": p.lt, ">": p.gt, "<=": p.le, ">=": p.ge}
            if op in preds:
                inner = e[2]
                if inner[0] == "bin" and inner[1] in preds:
                    # a < b < c  →  (a < b) && (b < c)
                    mid = self.eval(inner[3])
                    return ak.And(a, preds[op](mid, b))
                return preds[op](a, b)
            if op == "==":
                return ("eq", a, b)
            if op == "&&":
                return ak.And(a, b)
            if op == "||":
                return ak.Or(a, b)
            raise Unsupported(f"operator {op}")
        if tag == "list":
            return [self.eval(x) for x in e[1]]
        if tag == "call":
            return self.call(e[1], e[2])
        if tag == "str":
            return e[1]
        raise Unsupported(f"node {tag}")

    def call(self, head, args):
        p = self.pool
        if head[0] == "call":
            # Derivative[i, j][u][a, b]
            inner = head
            if (
                inner[1][0] == "call"
                and inner[1][1] == ("sym", "Derivative")
                and inner[2] == [("sym", self.uname)]
            ):
                orders = [int(o[1]) for o in inner[1][2]]
                f = self.u
                for v, k in zip(self.var_list, orders):
                    for _ in range(k):
                        f = ak.diff(f, v).value
                return self._at(f, args)
            raise Unsupported("chained call")
        if head[0] != "sym":
            raise Unsupported("call of a non-symbol")
        name = head[1]
        if name == self.uname:
            return self._at(self.u, args)
        if name in self.fdefs:
            params, body = self.fdefs[name]
            saved = dict(self.local)
            for pn, a in zip(params, args):
                self.local[pn] = self.eval(a)
            try:
                return self.eval(body)
            finally:
                self.local = saved
        if name in _UNARY and len(args) == 1:
            return _UNARY[name](self.eval(args[0]))
        if name == "D":
            f = self.eval(args[0])
            for spec in args[1:]:
                if spec[0] == "list":
                    v = self.eval(spec[1][0])
                    k = int(spec[1][1][1])
                else:
                    v, k = self.eval(spec), 1
                for _ in range(k):
                    f = ak.diff(f, v).value
            return f
        if name == "Laplacian":
            f = self.eval(args[0])
            vs = [self.eval(a) for a in args[1][1]]
            if len(args) > 2:
                system = args[2][1].lower()
                coords = vs + [None] * ({"polar": 2}.get(system, 3) - len(vs))
                return akx.curvilinear_laplacian(f, system, coords)
            total = None
            for v in vs:
                d2 = ak.diff(ak.diff(f, v).value, v).value
                total = d2 if total is None else total + d2
            return total
        if name == "Inequality":
            # Inequality[a, Less, b, LessEqual, c, …]
            ops = {"Less": p.lt, "LessEqual": p.le, "Greater": p.gt, "GreaterEqual": p.ge}
            vals = [self.eval(a) for a in args[0::2]]
            conds = []
            for i, o in enumerate(args[1::2]):
                if o[0] != "sym" or o[1] not in ops:
                    raise Unsupported("Inequality operator")
                conds.append(ops[o[1]](self.num(vals[i]), self.num(vals[i + 1])))
            out = conds[0]
            for c in conds[1:]:
                out = ak.And(out, c)
            return out
        if name == "Piecewise":
            branches = []
            default_ast = args[1] if len(args) > 1 else None
            for br in args[0][1]:
                if br[1][1] == ("sym", "True"):
                    default_ast = br[1][0]
                    break
                val = self.eval(br[1][0])
                cond = self.eval(br[1][1])
                if isinstance(cond, tuple):
                    raise Unsupported("an equality condition in Piecewise")
                branches.append((cond, self.num(val)))
            default = self.num(self.eval(default_ast)) if default_ast is not None else p.integer(0)
            return ak.piecewise(branches, default)
        if name in ("UnitStep", "HeavisideTheta"):
            a = self.eval(args[0])
            return ak.piecewise([(p.lt(a, p.integer(0)), p.integer(0))], p.integer(1))
        if name == "BesselJ":
            return ak.parse(
                "bessel_j(n_, z_)", p, {"n_": self.eval(args[0]), "z_": self.eval(args[1])}
            )
        if name in _UNSUPPORTED:
            raise Unsupported(name)
        # an undefined data function f[x], g[t], Q[x], A[t]
        return akx.Function(name)(*[self.num(self.eval(a)) for a in args])

    def _at(self, f, args):
        vals = [self.num(self.eval(a)) for a in args]
        m = {}
        for v, a in zip(self.var_list, vals):
            if a is not v and str(a) != str(v):
                m[v] = a
        return ak.subs(f, m) if m else f


def _positive_from(problem):
    pos = set()
    for _r, _a, b in [(None, *x[1:]) for x in problem.assumptions]:
        stack = [b]
        while stack:
            e = stack.pop()
            if e[0] == "bin" and e[1] in (">", ">=") and e[2][0] == "sym":
                pos.add(e[2][1])
            if e[0] == "bin" and e[1] in ("<", "<=") and e[3][0] == "sym":
                pos.add(e[3][1])
            if e[0] == "bin":
                stack.extend([e[2], e[3]])
            if e[0] == "list":
                stack.extend(e[1])
    return pos


def _has_radial_term(prob):
    """A curvilinear Laplacian in the equation (``u_r / r`` or a named
    coordinate system), as opposed to a rod whose variable is called ``r``."""
    src = " ".join(
        w
        for w in (to_wl(st[3]) for st in prob.stmts if st[0] == "assign")
        if "D[" in w or "Laplacian[" in w
    ).replace(" ", "")
    keys = ("/r)", "/r]", "/r^", '"Polar"', '"Cylindrical"', '"Spherical"', "(r^2*")
    return any(k in src for k in keys)


def _looks_spherical(prob):
    """A spherical Laplacian: ``"Spherical"``, ``2 u_r / r``, ``(r² u_r)_r`` or
    ``sin θ`` in the equation."""
    src = " ".join(
        w
        for w in (to_wl(st[3]) for st in prob.stmts if st[0] == "assign")
        if "D[" in w or "Laplacian[" in w
    )
    src = src.replace(" ", "")
    return (
        "Spherical" in src
        or re.search(r"\(2\*D\[u\[r[^]]*\],r\]\)", src) is not None
        or "D[(r^2*" in src
        or "Sin[theta]" in src
    )


def _assumes_outside(prob):
    """An assumption ``r > a`` (the exterior of a disk)."""
    stack = [a[2] for a in prob.assumptions]
    while stack:
        e = stack.pop()
        if not isinstance(e, tuple):
            continue
        if e[0] == "bin" and e[1] in (">", ">=") and e[2] == ("sym", "r") and e[3][0] == "sym":
            return True
        if e[0] == "bin" and e[1] in ("<", "<=") and e[3] == ("sym", "r") and e[2][0] == "sym":
            return True
        for x in e[1:]:
            if isinstance(x, list):
                stack.extend(x)
            else:
                stack.append(x)
    return False


def _sample(e, env_params):
    """Numeric value of a bound with every parameter at 1.3 (π stays π)."""
    try:
        vals = dict.fromkeys(env_params, 1.3)
        return ak.eval_expr(e, vals)
    except Exception:
        return None


def build(code: str, pool=None):
    """``(pde, env)``: the alkahest problem for one entry."""
    prob = Problem(code)
    pool = pool or ak.ExprPool()
    positive = _positive_from(prob)
    env = _Env(pool, prob.var_names, prob.uname, positive)
    eqs = []
    for st in prob.stmts:
        if st[0] != "assign":
            continue
        _, _kind, lhs, rhs = st
        if lhs[0] == "sym":
            env.defs[lhs[1]] = rhs
        elif lhs[0] == "call" and lhs[1][0] == "sym":
            params = [a[1] for a in lhs[2] if a[0] == "pattern"]
            env.fdefs[lhs[1][1]] = (params, rhs)
    items = prob.eqs_ast[1] if prob.eqs_ast[0] == "list" else [prob.eqs_ast]
    flat = []
    for it in items:
        v = env.eval(it)
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    for v in flat:
        if isinstance(v, tuple) and v[0] == "eq":
            eqs.append(v)
        elif isinstance(v, ak.Expr):
            eqs.append(("eq", v, pool.integer(0)))
        else:
            raise Unsupported("an equation that is not lhs == rhs")
    pde_eqs = [e for e in eqs if _is_pde(e, env)]
    conds = [e for e in eqs if not _is_pde(e, env)]
    if len(pde_eqs) != 1:
        raise Unsupported(f"{len(pde_eqs)} differential equations")
    lhs, rhs = env.num(pde_eqs[0][1]), env.num(pde_eqs[0][2])
    pde = akx.PDE(lhs - rhs, env.u)
    dom = _domain(conds, env, prob)
    pde = pde.on(dom)
    for _, lhs_c, rhs_c in conds:
        pde = pde.bc(env.num(lhs_c), env.num(rhs_c))
    return pde, env


def _is_pde(e, env):
    """An equation that mentions the unknown at its own arguments."""
    _, a, b = e
    s = f"{a} {b}"
    own = f"{env.uname}({', '.join(str(v) for v in env.var_list)})"
    return own in s


def _cond_asts(prob, env):
    """The equation ASTs of the DSolve list, with named definitions
    resolved."""
    out = []

    def walk(e):
        if e[0] == "sym" and e[1] in env.defs:
            walk(env.defs[e[1]])
        elif e[0] == "list":
            for x in e[1]:
                walk(x)
        else:
            out.append(e)

    walk(prob.eqs_ast)
    return out


def _faces(prob, env):
    """Per variable name, the points (expressions) where a condition fixes
    it: from every ``u[a, b]`` / ``Derivative[…][u][a, b]`` whose argument is
    not the variable itself."""
    pts = {v: [] for v in env.vars}
    names = list(env.vars)

    def walk(e):
        if not isinstance(e, tuple):
            return
        if e[0] == "call":
            head, args = e[1], e[2]
            is_u = head == ("sym", env.uname) or (
                head[0] == "call"
                and head[2] == [("sym", env.uname)]
                and head[1][0] == "call"
                and head[1][1] == ("sym", "Derivative")
            )
            if is_u and len(args) == len(names):
                for v, a in zip(names, args):
                    if a != ("sym", v):
                        pts[v].append(env.num(env.eval(a)))
                return
            for x in args:
                walk(x)
            walk(head)
            return
        for x in e[1:]:
            if isinstance(x, list):
                for y in x:
                    walk(y)
            else:
                walk(x)

    for c in _cond_asts(prob, env):
        if c[0] == "bin" and c[1] == "==":
            # the PDE itself has u at its own arguments only
            walk(c)
    return pts


def _domain(conds, env, prob):
    pool = env.pool
    names = list(env.vars)
    pts = _faces(prob, env)
    time = None
    for v in names:
        if v == "t":
            time = v
    curvi = "r" in names and ("theta" in names or "phi" in names or _has_radial_term(prob))
    parts = []
    if time:
        if not pts[time]:
            raise Unsupported("no initial condition")
        parts.append(akx.HalfLine(env.vars[time], 0))
    space = [v for v in names if v != time]

    def bounds(v):
        vals = []
        for e in pts[v]:
            x = _sample(e, list(env.params.values()))
            if x is None:
                raise Unsupported(f"face of {v} not numeric: {e}")
            if not any(abs(x - y[0]) < 1e-12 for y in vals):
                vals.append((x, e))
        vals.sort(key=lambda x: x[0])
        return vals

    if curvi and "r" in names:
        rv = bounds("r")
        if len(rv) == 1:
            if rv[0][0] <= 0:
                raise Unsupported("a face at r = 0")
            r_rng = (0, rv[0][1])
            if _assumes_outside(prob):
                raise Unsupported("the exterior of a disk is unbounded")
        elif len(rv) == 2:
            r_rng = (rv[0][1], rv[1][1])
        else:
            raise Unsupported("radius faces")
        has_z = "z" in names
        has_phi = "phi" in names
        pi = ak.parse("pi", pool)
        if has_phi or _looks_spherical(prob):
            th = env.vars.get("theta")
            ph = env.vars.get("phi")
            parts.append(
                akx.Curvilinear("spherical", [env.vars["r"], th, ph], [r_rng, (0, pi), (-pi, pi)])
            )
        elif has_z:
            zb = bounds("z")
            if len(zb) != 2:
                raise Unsupported("z faces")
            th = env.vars.get("theta")
            th_rng = (-pi, pi)
            if th is not None and pts["theta"]:
                tb = bounds("theta")
                th_rng = (tb[0][1], tb[1][1]) if len(tb) == 2 else th_rng
            parts.append(
                akx.Curvilinear(
                    "cylindrical",
                    [env.vars["r"], th, env.vars["z"]],
                    [r_rng, th_rng, (zb[0][1], zb[1][1])],
                )
            )
        else:
            th = env.vars.get("theta")
            th_rng = (-pi, pi)
            if th is not None and pts["theta"]:
                tb = bounds("theta")
                if len(tb) != 2:
                    raise Unsupported("theta faces")
                th_rng = (tb[0][1], tb[1][1])
            parts.append(akx.Curvilinear("polar", [env.vars["r"], th], [r_rng, th_rng]))
        rest = [v for v in space if v not in ("r", "theta", "z", "phi")]
    else:
        rest = space
    for v in rest:
        b = bounds(v)
        if len(b) != 2:
            raise Unsupported(f"{v} is not on a bounded interval")
        parts.append(akx.Interval(env.vars[v], b[0][1], b[1][1]))
    dom = parts[0]
    for q in parts[1:]:
        dom = dom * q
    return dom


# ---------------------------------------------------------------------------
# Back to Wolfram Language (for running the reference locally)
# ---------------------------------------------------------------------------


def to_wl(e) -> str:
    """Print an AST back as Wolfram Language."""
    tag = e[0]
    if tag == "num":
        return e[1]
    if tag == "sym":
        return e[1]
    if tag == "str":
        return '"' + e[1] + '"'
    if tag == "pattern":
        return e[1] + "_"
    if tag == "neg":
        return f"(-{to_wl(e[1])})"
    if tag == "not":
        return f"(!{to_wl(e[1])})"
    if tag == "bin":
        return f"({to_wl(e[2])} {e[1]} {to_wl(e[3])})"
    if tag == "rule":
        return f"({to_wl(e[1])} -> {to_wl(e[2])})"
    if tag == "list":
        return "{" + ", ".join(to_wl(x) for x in e[1]) + "}"
    if tag == "call":
        return to_wl(e[1]) + "[" + ", ".join(to_wl(x) for x in e[2]) + "]"
    raise Unsupported(f"cannot print {tag}")


def wl_parts(code: str):
    """``(definitions, equations, unknown, vars, options)`` as Wolfram
    Language strings: the entry's assignments (other than the one holding the
    DSolve call), and the arguments of its DSolve call."""
    prob = Problem(code)
    defs = []
    for st in prob.stmts:
        if st[0] != "assign":
            continue
        if _dsolve_call([st]) is not None:
            continue
        _, kind, lhs, rhs = st
        defs.append(f"{to_wl(lhs)} {kind} {to_wl(rhs)};")
    opts = ", ".join(to_wl(a) for a in prob.assumptions)
    return (
        "\n".join(defs),
        to_wl(prob.eqs_ast),
        to_wl(prob.unknown_ast),
        to_wl(prob.vars_ast),
        opts,
    )
