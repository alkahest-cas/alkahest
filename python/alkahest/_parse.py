"""V2-21: Pratt recursive-descent expression parser."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

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

_BP_ADD = 10    # +  -  (infix, left-associative)
_BP_MUL = 20    # *  /  (infix, left-associative)
_BP_POW = 30    # ^  ** (infix, right-associative — led calls expr(BP_POW - 1))
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
# Known math functions (one or two arguments)
# ---------------------------------------------------------------------------

_FUNC_NAMES = frozenset({
    "sin", "cos", "tan",
    "sinh", "cosh", "tanh",
    "asin", "acos", "atan", "atan2",
    "exp", "log", "sqrt",
    "abs", "sign", "floor", "ceil", "round",
    "erf", "erfc", "gamma",
})

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

_Token = Tuple[str, str, int]  # (kind, text, offset)


def _tokenize(source: str) -> List[_Token]:
    tokens: List[_Token] = []
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
    __slots__ = ("_source", "_pool", "_symbols", "_tokens", "_pos")

    def __init__(self, source: str, pool, symbols: Dict[str, object]) -> None:
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
            return -operand

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
            # Rust __pow__ only accepts Python int; use pow_expr for Expr exponents.
            rn = right.node()
            if rn[0] == "integer":
                return left ** int(rn[1])
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
        return _apply_func(name, args, offset)


def _apply_func(name: str, args: list, offset: int):
    # Local import to avoid circular dependency at module load time.
    import alkahest as _ak

    _funcs = {
        "sin":   _ak.sin,
        "cos":   _ak.cos,
        "tan":   _ak.tan,
        "sinh":  _ak.sinh,
        "cosh":  _ak.cosh,
        "tanh":  _ak.tanh,
        "asin":  _ak.asin,
        "acos":  _ak.acos,
        "atan":  _ak.atan,
        "atan2": _ak.atan2,
        "exp":   _ak.exp,
        "log":   _ak.log,
        "sqrt":  _ak.sqrt,
        "abs":   _ak.abs,
        "sign":  _ak.sign,
        "floor": _ak.floor,
        "ceil":  _ak.ceil,
        "round": _ak.round,
        "erf":   _ak.erf,
        "erfc":  _ak.erfc,
        "gamma": _ak.gamma,
    }
    fn = _funcs.get(name)
    if fn is None:
        raise ParseError(
            f"unknown function {name!r}",
            span=(offset, offset + len(name)),
            remediation=f"known functions: {', '.join(sorted(_funcs))}",
        )
    return fn(*args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(source: str, pool, symbols: Optional[Dict[str, object]] = None):
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
