"""V2-21: Pratt recursive-descent expression parser."""

from __future__ import annotations

import re

from .exceptions import ParseError

# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------

_TK_NUM = "num"
_TK_IDENT = "ident"
_TK_EOF = "eof"

# Operator tokens use their literal text as the kind: "+" "-" "*" "/" "^" "**" "(" ")" ","

# ---------------------------------------------------------------------------
# Binding powers (Pratt precedence)
# ---------------------------------------------------------------------------

_BP_ADD = 10  # +  -  (infix, left-associative)
_BP_MUL = 20  # *  /  (infix, left-associative)
_BP_POW = 30  # ^  ** (infix, right-associative — led calls expr(BP_POW - 1))
_BP_UNARY = 25  # unary -/+  (between MUL and POW so that -x^2 = -(x^2))

_INFIX_BP: dict[str, int] = {
    "+": _BP_ADD,
    "-": _BP_ADD,
    "*": _BP_MUL,
    "/": _BP_MUL,
    "^": _BP_POW,
    "**": _BP_POW,
}


# ---------------------------------------------------------------------------
# Unary minus on a literal
# ---------------------------------------------------------------------------


def _negate_literal(pool, operand):
    """The negation of `operand` if it is an exact numeric literal, else None.

    Prefix ``-`` is otherwise ``Expr.__neg__``, i.e. ``(-1) * operand``, which
    for a literal operand leaves an unevaluated product in the pool: ``x^(-1)``
    used to build ``x^(1 * -1)`` while ``1/x`` built ``x^(-1)``.  The two are
    the same function, but every structural detector that reads an exponent by
    matching on an integer node saw only the second, so the *spelling* of an
    integrand decided its route through the integrator.

    Scope is deliberately just integers and rationals, and no arithmetic is
    evaluated — ``-(2+3)`` keeps its tree.  Mirrors ``negate_literal`` in
    ``alkahest-core/src/parse.rs``; keep the two in step.

    ``(-1) * literal`` stays reachable through the builder API (``-expr``,
    ``pool.mul([pool.integer(-1), pool.integer(1)])``), so the detectors keep
    their own normalising view of an integer exponent.  This is the first of
    two layers, not a replacement for the second.
    """
    node = operand.node()
    if node[0] == "integer":
        return pool.integer(-int(node[1]))
    if node[0] == "rational":
        return pool.rational(-int(node[1]), int(node[2]))
    return None


# ---------------------------------------------------------------------------
# Known math functions (one or two arguments)
# ---------------------------------------------------------------------------

_FUNC_NAMES = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "atan2",
        "exp",
        "log",
        "sqrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "round",
        "erf",
        "erfc",
        "gamma",
        "lambert_w",
        "digamma",
        "bessel_j0",
        "bessel_j1",
        "EllipticK",
        "EllipticE",
        "EllipticF",
        "EllipticPi",
        # The non-elementary output basis (3.10.0).  Without these the
        # integrator can emit an antiderivative that neither parser can read
        # back, so ``parse(str(integrate(f)))`` is not a round trip.  Mirrors
        # ``KNOWN_FUNCS`` in ``alkahest-core/src/parse.rs``.
        "Ei",
        "li",
        "Si",
        "Ci",
        "Shi",
        "Chi",
        "fresnels",
        "fresnelc",
        "dilog",
        "trigamma",
        # Reciprocal trig / hyperbolic functions (desugared to base(x) ** -1).
        "sec",
        "csc",
        "cot",
        "sech",
        "csch",
        "coth",
        # Desugared to ``u ** Fraction(1, 3)``; see ``_apply_func``.
        "cbrt",
    }
)

# Reciprocal trig / hyperbolic functions map to the elementary primitive they
# are the reciprocal of; a call ``f(x)`` is desugared to ``base(x) ** -1`` so no
# dedicated ``sec``/``csc``/… node is ever built and everything downstream
# (diff, eval, integrate, simplify) runs on the existing cos/sin/tan/cosh/sinh/
# tanh primitives.  Mirrors the desugar in ``alkahest-core/src/parse.rs``.
_RECIPROCAL_BASE: dict[str, str] = {
    "sec": "cos",
    "csc": "sin",
    "cot": "tan",
    "sech": "cosh",
    "csch": "sinh",
    "coth": "tanh",
}

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<num>   \d+(?:\.\d*)?(?:[eE][+\-]?\d+)?   # int or float: 3  3.14  1e5  1.5e-3
               | \.\d+(?:[eE][+\-]?\d+)?            # .5  .5e3
      )
    | (?P<ident> [A-Za-z_][A-Za-z0-9_]*)            # identifier
    | (?P<pow2>  \*\*)                               # ** (must come before *)
    | (?P<op>    [+\-*/^(),])                        # single-char ops and delimiters
    | (?P<ws>    [ \t\r\n]+)                         # whitespace (skip)
    """,
    re.VERBOSE,
)

_Token = tuple[str, str, int]  # (kind, text, offset)


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    n = len(source)
    while pos < n:
        m = _TOKEN_RE.match(source, pos)
        if m is None:
            raise ParseError(
                f"unexpected character {source[pos]!r} at offset {pos}",
                span=(pos, pos + 1),
                remediation="only ASCII arithmetic expressions are supported",
            )
        kind = m.lastgroup
        text = m.group()
        offset = m.start()
        if kind == "ws":
            pass
        elif kind == "num":
            tokens.append((_TK_NUM, text, offset))
        elif kind == "ident":
            tokens.append((_TK_IDENT, text, offset))
        elif kind == "pow2":
            tokens.append(("**", text, offset))
        else:  # op
            tokens.append((text, text, offset))
        pos = m.end()
    tokens.append((_TK_EOF, "", n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    __slots__ = ("_pool", "_pos", "_source", "_symbols", "_tokens")

    def __init__(self, source: str, pool, symbols: dict[str, object]) -> None:
        self._source = source
        self._pool = pool
        self._symbols = symbols
        self._tokens = _tokenize(source)
        self._pos = 0

    # -- token helpers --

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        if tok[0] != _TK_EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: str) -> _Token:
        tok = self._advance()
        if tok[0] != kind:
            if tok[0] == _TK_EOF:
                raise ParseError(
                    f"expected {kind!r} but reached end of input",
                    span=(tok[2], tok[2]),
                )
            raise ParseError(
                f"expected {kind!r}, got {tok[1]!r}",
                span=(tok[2], tok[2] + len(tok[1])),
            )
        return tok

    # -- entry point --

    def parse(self):
        if self._peek()[0] == _TK_EOF:
            raise ParseError("empty expression", span=(0, 0))
        expr = self._expr(0)
        tok = self._peek()
        if tok[0] != _TK_EOF:
            raise ParseError(
                f"unexpected token {tok[1]!r}",
                span=(tok[2], tok[2] + len(tok[1])),
            )
        return expr

    # -- Pratt expression --

    def _expr(self, rbp: int):
        tok = self._advance()
        left = self._nud(tok)
        while True:
            tok = self._peek()
            lbp = _INFIX_BP.get(tok[0], 0)
            if lbp <= rbp:
                break
            self._advance()
            left = self._led(tok, left)
        return left

    # -- null denotation (prefix / atom) --

    def _nud(self, tok):
        kind, text, offset = tok
        pool = self._pool

        if kind == _TK_NUM:
            if "." in text or "e" in text.lower():
                return pool.float(float(text), 53)
            return pool.integer(int(text))

        if kind == _TK_IDENT:
            # Function call?
            if self._peek()[0] == "(":
                return self._funcall(text, offset)
            # Symbol: look up in the caller-supplied map first, then intern.
            sym = self._symbols.get(text)
            if sym is None:
                sym = pool.symbol(text)
                self._symbols[text] = sym
            return sym

        if kind == "-":
            operand = self._expr(_BP_UNARY)
            folded = _negate_literal(pool, operand)
            return -operand if folded is None else folded

        if kind == "+":
            return self._expr(_BP_UNARY)

        if kind == "(":
            if self._peek()[0] == ")":
                raise ParseError(
                    "empty parentheses",
                    span=(offset, offset + 1),
                    remediation="parentheses must contain an expression",
                )
            inner = self._expr(0)
            self._expect(")")
            return inner

        raise ParseError(
            f"unexpected token {text!r}",
            span=(offset, offset + len(text)),
        )

    # -- left denotation (infix) --

    def _led(self, tok, left):
        kind, text, offset = tok

        if kind == "+":
            return left + self._expr(_BP_ADD)

        if kind == "-":
            return left - self._expr(_BP_ADD)

        if kind == "*":
            return left * self._expr(_BP_MUL)

        if kind == "/":
            return left / self._expr(_BP_MUL)

        if kind in ("^", "**"):
            # Right-associative: use BP_POW - 1 as the right-binding-power.
            right = self._expr(_BP_POW - 1)
            # `pow_expr` takes the exponent node as-is.  Going through `**`
            # instead would round-trip it via `PyExpr.__pow__`, which falls back
            # to an f64 exponent when the integer does not fit in an i64 — a
            # silent precision loss.  For an exponent that *does* fit the two
            # build the identical node, so there is nothing to gain from the
            # detour.  (Only reachable for negative exponents since the parser
            # started folding `-<literal>`; it was always live for huge positive
            # ones.)
            return left.pow_expr(right)

        raise ParseError(
            f"unexpected token {text!r} in infix position",
            span=(offset, offset + len(text)),
        )

    # -- function call --

    def _funcall(self, name: str, offset: int):
        self._advance()  # consume "("
        args = []
        if self._peek()[0] != ")":
            args.append(self._expr(0))
            while self._peek()[0] == ",":
                self._advance()  # consume ","
                args.append(self._expr(0))
        self._expect(")")
        return _apply_func(name, args, offset, self._pool)


def _apply_func(name: str, args: list, offset: int, pool=None):
    # Local import to avoid circular dependency at module load time.
    import alkahest as _ak

    _funcs = {
        "sin": _ak.sin,
        "cos": _ak.cos,
        "tan": _ak.tan,
        "sinh": _ak.sinh,
        "cosh": _ak.cosh,
        "tanh": _ak.tanh,
        "asin": _ak.asin,
        "acos": _ak.acos,
        "atan": _ak.atan,
        "asinh": _ak.asinh,
        "acosh": _ak.acosh,
        "atanh": _ak.atanh,
        "atan2": _ak.atan2,
        "exp": _ak.exp,
        "log": _ak.log,
        "sqrt": _ak.sqrt,
        "abs": _ak.abs,
        "sign": _ak.sign,
        "floor": _ak.floor,
        "ceil": _ak.ceil,
        "round": _ak.round,
        "erf": _ak.erf,
        "erfc": _ak.erfc,
        "gamma": _ak.gamma,
        "lambert_w": _ak.lambert_w,
        "digamma": _ak.digamma,
        "bessel_j0": _ak.bessel_j0,
        "bessel_j1": _ak.bessel_j1,
        # Elliptic special functions (parameter convention m = k^2).
        "EllipticK": _ak.elliptic_k,
        "EllipticE": _ak.elliptic_e,  # 1 arg (complete) or 2 args (incomplete)
        "EllipticF": _ak.elliptic_f,
        "EllipticPi": _ak.elliptic_pi,
        # The non-elementary output basis (3.10.0).  The integrator emits
        # these, so the parser has to be able to read them back; mirrors
        # ``KNOWN_FUNCS`` in ``alkahest-core/src/parse.rs``.  The spelling is
        # the *node* name, not the Python constructor's name — ``Si(x)`` is
        # what ``str(expr)`` prints, and a round trip is the whole point.
        "Ei": _ak.exp_integral_ei,
        "li": _ak.log_integral,
        "Si": _ak.sin_integral,
        "Ci": _ak.cos_integral,
        "Shi": _ak.sinh_integral,
        "Chi": _ak.cosh_integral,
        "fresnels": _ak.fresnels,
        "fresnelc": _ak.fresnelc,
        "dilog": _ak.dilog,
        "trigamma": _ak.trigamma,
    }
    # Desugar ``cbrt(u)`` to ``u ** (1/3)``.  It is not a registered primitive
    # and is not being made one: the power node already differentiates,
    # evaluates, simplifies and integrates.  The one thing this does not
    # reproduce is ``math.cbrt``'s real branch on negatives — ``cbrt(-8)`` is
    # ``(-8) ** (1/3)``, which the numeric interpreter reports as no-value
    # rather than as ``-2``.  That is a refusal, not a wrong answer, and it is
    # the principal-branch convention the pool already uses for fractional
    # powers.  Mirrors the desugar in ``alkahest-core/src/parse.rs``.
    if name == "cbrt":
        if len(args) != 1:
            raise ParseError(
                f"cbrt takes exactly 1 argument, got {len(args)}",
                span=(offset, offset + len(name)),
            )
        if pool is None:
            raise ParseError(
                "cbrt requires a pool to build its exponent",
                span=(offset, offset + len(name)),
            )
        return args[0].pow_expr(pool.rational(1, 3))
    # Desugar reciprocal trig/hyperbolic calls to ``base(x) ** -1``.  Only the
    # single-argument form is meaningful; any other arity is a parse error.
    base = _RECIPROCAL_BASE.get(name)
    if base is not None:
        if len(args) != 1:
            raise ParseError(
                f"{name} takes exactly 1 argument, got {len(args)}",
                span=(offset, offset + len(name)),
            )
        return _funcs[base](*args) ** -1

    fn = _funcs.get(name)
    if fn is None:
        raise ParseError(
            f"unknown function {name!r}",
            span=(offset, offset + len(name)),
            remediation=(
                f"known functions: {', '.join(sorted(set(_funcs) | set(_RECIPROCAL_BASE)))}"
            ),
        )
    return fn(*args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(source: str, pool, symbols: dict[str, object] | None = None):
    """Parse a mathematical expression string into an :class:`~alkahest.Expr`.

    Uses a Pratt (top-down operator precedence) recursive-descent parser.

    Parameters
    ----------
    source : str
        Expression string, e.g. ``"sin(x)^2 + cos(x)^2"``.
    pool : ExprPool
        Expression pool used to intern new symbols and constants.
    symbols : dict[str, Expr] | None
        Optional pre-bound symbol map.  Identifiers not in the map are
        created via ``pool.symbol(name)`` and added to the map so they are
        reused within the same call.

    Returns
    -------
    Expr
        The parsed expression.

    Raises
    ------
    ParseError
        On a lexical or syntax error, with ``.span`` set to the byte range
        of the offending token.

    Examples
    --------
    >>> import alkahest
    >>> pool = alkahest.ExprPool()
    >>> x = pool.symbol("x")
    >>> e = alkahest.parse("x^2 + 2*x + 1", pool, {"x": x})
    >>> alkahest.parse("sin(x)^2 + cos(x)^2", pool, {"x": x})
    sin(x)^2 + cos(x)^2
    """
    if symbols is None:
        symbols = {}
    return _Parser(source, pool, symbols).parse()
