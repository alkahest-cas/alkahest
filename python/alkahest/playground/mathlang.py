"""The Math tab's input language: tokenizer, parser and LaTeX printer.

This module is deliberately free of any Alkahest import and of ``eval``: it turns
one line of ordinary maths ("integrate sin(x)^2 dx", "d/dx x^3 sin(x)",
"solve x^2 - 5x + 6 = 0") into a small, fully-inspected syntax tree.  The tree
is then executed by :mod:`alkahest.playground.math_exec`, which maps every
node onto an explicit whitelist of Alkahest calls.  Nothing a user types is
ever handed to a Python interpreter, so there is no ``__import__``, attribute
access, string, or subscript to smuggle through: anything outside the grammar
below is a :class:`MathInputError` before any Alkahest function is called.

Grammar (informal)
------------------

command   := definition | integrate | derivative | solve | simplify | factor
           | expand | limit | series | sum | product | numeric | equation | expr
definition:= NAME ["(" NAME {"," NAME} ")"] ("=" | ":=") (command | expr)
integrate := ("integrate" | "integral" ["of"] | "∫") expr [dVAR] ["from" expr "to" expr]
derivative:= "d/dVAR" expr | "d^n/dVAR^n" expr | ("derivative" | "differentiate") ["of"] expr
             [("with respect to" | "wrt") VAR] | "diff(" expr "," VAR ["," n] ")"
solve     := "solve" equation {("," | "and") equation} ["for" VAR {"," VAR}]
limit     := "limit" ["of"] expr "as" VAR "->" expr ["+" | "-"] | "lim" VAR "->" expr expr
series    := ("series" | "taylor") ["of"] expr [("at" | "around" | "about") [VAR "="] expr]
             [["to" | "up to"] "order" n]
sum       := "sum" ["of"] expr ("for" | "from") VAR "=" expr "to" expr     (also "product")
numeric   := ("numeric" | "approx" | "numerically") expr ["to" n "digits"] | "N[" expr ["," n] "]"
expr      := the usual infix maths with implicit multiplication (2x, x sin(x),
             (x+1)(x-1)), ^ or ** for powers, ! for factorial, |x| for abs, and
             unicode (π, ∞, √, ², ·, ×, ÷, −, Greek letters).

Every command also has a call form, e.g. ``integrate(f, x, 0, 1)``,
``limit(f, x, 0)``, ``series(f, x, 0, 6)``, ``sum(f, k, 1, oo)``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

# ── Limits ───────────────────────────────────────────────────────────────────
# These bound the work done *before* Alkahest sees anything; the executor adds
# a wall-clock Budget, and the server adds a hard process timeout and memory cap.

MAX_INPUT_CHARS = 500
MAX_DEFINITION_CHARS = 20_000
MAX_TOKENS = 4_000
MAX_NODES = 2_000
MAX_DEPTH = 60
MAX_NUMBER_DIGITS = 100
MAX_INT_BITS = 100_000  # largest integer power we let the input denote
MAX_SYMBOLIC_EXPONENT = 1_000
MAX_FACTORIAL = 1_000
MAX_DIFF_ORDER = 20
MAX_SERIES_ORDER = 30
MAX_DIGITS = 1_000
MAX_SESSION_DEFINITIONS = 32

# ── Error codes (front-end refusals, distinct from Alkahest's own E-* codes) ─

E_SYNTAX = "E-PLAYGROUND-001"
E_TOO_LARGE_INPUT = "E-PLAYGROUND-002"
E_UNKNOWN_NAME = "E-PLAYGROUND-003"
E_NUMBER_TOO_LARGE = "E-PLAYGROUND-004"
E_NEED_VARIABLE = "E-PLAYGROUND-005"
E_TIMEOUT = "E-PLAYGROUND-006"
E_MEMORY = "E-PLAYGROUND-007"
E_UNSUPPORTED = "E-PLAYGROUND-008"
E_INTERNAL = "E-PLAYGROUND-009"
E_BUSY = "E-PLAYGROUND-010"


class MathInputError(Exception):
    """The input could not be read; carries a stable ``E-PLAYGROUND-*`` code."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


# ── Vocabulary ───────────────────────────────────────────────────────────────

GREEK = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]
GREEK_UNICODE = dict(
    zip(
        "αβγδεζηθικλμνξορστυφχψω",
        GREEK,
    )
)
GREEK_UNICODE.update({"ϵ": "epsilon", "ϑ": "theta", "ϕ": "phi", "ς": "sigma"})
# Capital Greek letters that are not Latin look-alikes.
GREEK_UNICODE.update(
    {"Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Φ": "Phi"}
)
GREEK_UNICODE.update({"Ψ": "Psi", "Ω": "Omega"})
GREEK_NAMES = set(GREEK) | {"Gamma", "Delta", "Theta", "Lambda", "Xi", "Phi", "Psi", "Omega"}

#: Functions the language knows, mapped to their arity.  ``gamma`` and ``zeta``
#: shadow the Greek letters when called with parentheses.
FUNCTIONS: dict[str, int] = {
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "sec": 1,
    "csc": 1,
    "cot": 1,
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "exp": 1,
    "log": 1,  # natural log; log(x, b) is log base b
    "ln": 1,
    "log10": 1,
    "log2": 1,
    "sqrt": 1,
    "cbrt": 1,
    "abs": 1,
    "sign": 1,
    "floor": 1,
    "ceil": 1,
    "gamma": 1,
    "erf": 1,
    "erfc": 1,
    "zeta": 1,
    "atan2": 2,
    "factorial": 1,
    "min": 2,
    "max": 2,
}
FUNCTION_ALIASES = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arsinh": "asinh",
    "arcsinh": "asinh",
    "arcosh": "acosh",
    "arccosh": "acosh",
    "artanh": "atanh",
    "arctanh": "atanh",
    "ln": "log",
    "Sqrt": "sqrt",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Exp": "exp",
    "Log": "log",
}
INVERSE_TRIG = {
    "sin": "asin",
    "cos": "acos",
    "tan": "atan",
    "sinh": "asinh",
    "cosh": "acosh",
    "tanh": "atanh",
}
CONSTANTS = {
    "pi": "pi",
    "Pi": "pi",
    "e": "e",
    "E": "e",
    "i": "i",
    "I": "i",
    "oo": "oo",
    "inf": "oo",
    "infinity": "oo",
    "Infinity": "oo",
}
#: Words that separate the parts of a command; they can never be variables.
KEYWORDS = {
    "from",
    "to",
    "for",
    "as",
    "at",
    "around",
    "about",
    "order",
    "with",
    "respect",
    "wrt",
    "of",
    "and",
    "digits",
    "up",
    "where",
}
COMMAND_WORDS = {
    "integrate",
    "integral",
    "int",
    "antiderivative",
    "derivative",
    "differentiate",
    "diff",
    "solve",
    "simplify",
    "factor",
    "expand",
    "limit",
    "lim",
    "series",
    "taylor",
    "sum",
    "product",
    "numeric",
    "numerically",
    "approx",
    "evaluate",
    "N",
}

# ── Unicode normalisation ────────────────────────────────────────────────────

_SUPERSCRIPTS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
    "ⁿ": "n",
}
_REPLACEMENTS = [
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
    ("×", "*"),
    ("·", "*"),
    ("⋅", "*"),
    ("∗", "*"),
    ("÷", "/"),
    ("∕", "/"),
    ("π", " pi "),
    ("∞", " oo "),
    ("√", " sqrt "),
    ("∛", " cbrt "),
    ("∫", " integrate "),
    ("∑", " sum "),
    ("Σ", " sum "),
    ("∏", " product "),
    ("∂", "d"),
    ("→", "->"),
    ("⟶", "->"),
    ("≔", ":="),
    ("ℯ", " e "),
    ("ⅇ", " e "),
    ("ⅈ", " i "),
    ("ⅆ", "d"),
    (" ", " "),
]


def normalize(text: str) -> str:
    """Map the unicode a palette or a phone keyboard produces onto ASCII."""
    # 1. superscripts (x², x⁻¹) — before NFKC, which would turn ² into a bare 2
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _SUPERSCRIPTS:
            j = i
            run = []
            while j < len(text) and text[j] in _SUPERSCRIPTS:
                run.append(_SUPERSCRIPTS[text[j]])
                j += 1
            joined = "".join(run)
            out.append("^" + (joined if len(joined) == 1 else "(" + joined + ")"))
            i = j
            continue
        out.append(ch)
        i += 1
    # 2. fullwidth / compatibility forms (ｘ, µ, ⅈ) to their plain letters
    s = unicodedata.normalize("NFKC", "".join(out))
    # 3. Greek letters to names; "dθ" stays one differential token
    out = []
    for k, ch in enumerate(s):
        if ch in GREEK_UNICODE:
            name = GREEK_UNICODE[ch]
            prev = s[k - 1] if k > 0 else ""
            before = s[k - 2] if k > 1 else ""
            glued = prev in ("d", "∂") and not before.isalpha()
            out.append((name if glued else " " + name) + " ")
        else:
            out.append(ch)
    s = "".join(out)
    for a, b in _REPLACEMENTS:
        s = s.replace(a, b)
    # lim_{x->0} f  →  lim {x->0} f
    s = re.sub(r"\blim_", "lim ", s)
    return s


# ── Tokens ───────────────────────────────────────────────────────────────────


@dataclass
class Tok:
    kind: str  # num, ident, op, end
    text: str
    pos: int


_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<num>\d+(?:\.\d*)?|\.\d+)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>->|\*\*|:=|\.\.|[-+*/^()\[\]{},=!|'<>])"
    r")"
)


def tokenize(text: str) -> list[Tok]:
    toks: list[Tok] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            ch = text[pos]
            if ch in '"`':
                msg = "Quotes are not part of the maths language."
            elif ch == ".":
                msg = "A '.' is only allowed inside a number (like 0.5)."
            elif ch in ";@#$%&\\~?":
                msg = f"The character {ch!r} is not part of the maths language."
            else:
                msg = f"Unexpected character {ch!r}."
            raise MathInputError(E_SYNTAX, msg, _hint_at(text, pos))
        kind = m.lastgroup or "op"
        value = m.group(kind)
        start = m.start(kind)
        if kind == "num" and len(value.replace(".", "")) > MAX_NUMBER_DIGITS:
            raise MathInputError(
                E_NUMBER_TOO_LARGE,
                f"Numbers are limited to {MAX_NUMBER_DIGITS} digits in the playground.",
            )
        if kind == "ident" and value.startswith("_"):
            raise MathInputError(
                E_UNKNOWN_NAME,
                f"Names cannot start with an underscore ({value!r}).",
            )
        if kind == "op" and value == "**":
            value = "^"
        toks.append(Tok(kind, value, start))
        if len(toks) > MAX_TOKENS:
            raise MathInputError(E_TOO_LARGE_INPUT, "The input is too long.")
        pos = m.end()
    toks.append(Tok("end", "", n))
    return toks


def _hint_at(text: str, pos: int) -> str:
    return f"at position {pos + 1}: {text[max(0, pos - 12) : pos]}⟨{text[pos : pos + 1]}⟩"


# ── Syntax tree ──────────────────────────────────────────────────────────────


@dataclass
class Node:
    """An expression node.

    kinds: num(value=str), sym(value=name), const(value=pi|e|i|oo),
    add/sub/mul/div/pow (children=[a, b]), neg (children=[a]),
    call(value=function, children=args), fact (children=[a]),
    ucall(value=user function name, children=args), implicit mul is ``mul`` with
    ``implicit=True``.
    """

    kind: str
    value: str = ""
    children: list[Node] = field(default_factory=list)
    implicit: bool = False


@dataclass
class Command:
    kind: str  # expr, integrate, diff, solve, simplify, factor, expand, limit,
    # series, sum, product, numeric, define
    expr: Node | None = None
    var: str | None = None
    lower: Node | None = None
    upper: Node | None = None
    order: int = 1
    point: Node | None = None
    direction: str | None = None  # "+", "-" for one-sided limits
    equations: list[tuple[Node, Node]] = field(default_factory=list)
    vars: list[str] = field(default_factory=list)
    digits: int = 15
    # definitions
    name: str | None = None
    params: list[str] = field(default_factory=list)
    body: Command | None = None
    var_explicit: bool = False


# ── Expression parser ────────────────────────────────────────────────────────


class Parser:
    """Pratt-style parser over a token slice.

    ``known`` holds names the session has defined (plain names → value,
    functions → parameter count) so that ``f`` reads as one symbol and ``f(2)``
    as a call rather than ``f·2``.
    """

    def __init__(
        self,
        toks: list[Tok],
        known_names: set[str] | None = None,
        known_functions: dict[str, int] | None = None,
        bound: set[str] | None = None,
    ):
        self.toks = toks
        self.i = 0
        self.depth = 0
        self.nodes = 0
        self.known_names = known_names or set()
        self.known_functions = known_functions or {}
        self.bound = bound or set()

    # helpers
    def peek(self, k: int = 0) -> Tok:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Tok:
        t = self.toks[self.i]
        if self.i < len(self.toks) - 1:
            self.i += 1
        return t

    def at_op(self, *ops: str) -> bool:
        t = self.peek()
        return t.kind == "op" and t.text in ops

    def expect_op(self, op: str) -> None:
        t = self.next()
        if t.kind != "op" or t.text != op:
            raise MathInputError(
                E_SYNTAX,
                f"Expected {op!r} but found {t.text or 'the end of the input'!r}.",
            )

    def mk(self, kind: str, value: str = "", children=None, implicit=False) -> Node:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise MathInputError(E_TOO_LARGE_INPUT, "The expression is too large.")
        return Node(kind, value, list(children or []), implicit)

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MathInputError(E_TOO_LARGE_INPUT, "The expression is nested too deeply.")

    def leave(self) -> None:
        self.depth -= 1

    def at_end(self) -> bool:
        return self.peek().kind == "end"

    # grammar
    def parse_full(self) -> Node:
        if self.at_end():
            raise MathInputError(E_SYNTAX, "Expected an expression.")
        node = self.parse_expr()
        if not self.at_end():
            t = self.peek()
            raise MathInputError(E_SYNTAX, f"Unexpected {t.text!r} after the expression.")
        return node

    def parse_expr(self) -> Node:
        self.enter()
        node = self.parse_term()
        while self.at_op("+", "-"):
            op = self.next().text
            rhs = self.parse_term()
            node = self.mk("add" if op == "+" else "sub", children=[node, rhs])
        self.leave()
        return node

    def starts_factor(self) -> bool:
        t = self.peek()
        if t.kind == "num":
            return True
        if t.kind == "ident":
            return t.text not in KEYWORDS
        return t.kind == "op" and t.text in ("(", "[", "{")

    def parse_term(self) -> Node:
        self.enter()
        node = self.parse_unary()
        while True:
            if self.at_op("*", "/"):
                op = self.next().text
                rhs = self.parse_unary()
                node = self.mk("mul" if op == "*" else "div", children=[node, rhs])
            elif self.starts_factor():
                if self.peek().kind == "num" and _ends_in_number(node):
                    raise MathInputError(
                        E_SYNTAX,
                        "Two numbers next to each other; did you mean to multiply (use *)?",
                    )
                rhs = self.parse_power()
                node = self.mk("mul", children=[node, rhs], implicit=True)
            else:
                break
        self.leave()
        return node

    def parse_unary(self) -> Node:
        self.enter()
        if self.at_op("-"):
            self.next()
            node = self.mk("neg", children=[self.parse_unary()])
        elif self.at_op("+"):
            self.next()
            node = self.parse_unary()
        else:
            node = self.parse_power()
        self.leave()
        return node

    def parse_power(self) -> Node:
        self.enter()
        base = self.parse_postfix()
        if self.at_op("^"):
            self.next()
            exponent = self.parse_unary_power()
            base = self.mk("pow", children=[base, exponent])
            base = self.parse_postfix_ops(base)
        self.leave()
        return base

    def parse_unary_power(self) -> Node:
        """Exponent: a signed power, so ``x^-1`` and ``2^2^3`` (right-assoc) work."""
        self.enter()
        if self.at_op("-"):
            self.next()
            node = self.mk("neg", children=[self.parse_unary_power()])
        elif self.at_op("+"):
            self.next()
            node = self.parse_unary_power()
        else:
            node = self.parse_power()
        self.leave()
        return node

    def parse_postfix(self) -> Node:
        return self.parse_postfix_ops(self.parse_primary())

    def parse_postfix_ops(self, node: Node) -> Node:
        while self.at_op("!"):
            self.next()
            node = self.mk("fact", children=[node])
        return node

    def parse_group(self, open_: str, close: str) -> Node:
        self.expect_op(open_)
        if self.at_op(close):
            raise MathInputError(E_SYNTAX, f"Empty {open_}{close}.")
        node = self.parse_expr()
        self.expect_op(close)
        return node

    def parse_args(self) -> list[Node]:
        self.expect_op("(")
        args = [self.parse_expr()]
        while self.at_op(","):
            self.next()
            args.append(self.parse_expr())
        self.expect_op(")")
        return args

    def parse_primary(self) -> Node:
        self.enter()
        t = self.peek()
        if t.kind == "num":
            self.next()
            node = self.mk("num", t.text)
        elif t.kind == "op" and t.text == "(":
            node = self.parse_group("(", ")")
        elif t.kind == "op" and t.text == "[":
            node = self.parse_group("[", "]")
        elif t.kind == "op" and t.text == "{":
            node = self.parse_group("{", "}")
        elif t.kind == "op" and t.text == "|":
            self.next()
            inner = self.parse_expr()
            self.expect_op("|")
            node = self.mk("call", "abs", [inner])
        elif t.kind == "ident":
            node = self.parse_ident()
        elif t.kind == "end":
            raise MathInputError(E_SYNTAX, "The expression ends too early.")
        else:
            raise MathInputError(E_SYNTAX, f"Unexpected {t.text!r}.")
        self.leave()
        return node

    def parse_ident(self) -> Node:
        t = self.next()
        name = t.text
        if name in KEYWORDS:
            raise MathInputError(E_SYNTAX, f"The word {name!r} cannot appear here.")
        if name in self.bound:
            return self.mk("sym", name)
        if name in self.known_functions and self.at_op("("):
            args = self.parse_args()
            if len(args) != self.known_functions[name]:
                raise MathInputError(
                    E_SYNTAX,
                    f"{name} takes {self.known_functions[name]} argument(s), got {len(args)}.",
                )
            return self.mk("ucall", name, args)
        if name in self.known_names or name in self.known_functions:
            return self.mk("sym", name)
        fname = FUNCTION_ALIASES.get(name, name)
        is_func = fname in FUNCTIONS
        # gamma/zeta are Greek letters unless called
        if fname in ("gamma", "zeta") and not self.at_op("(", "^"):
            is_func = False
        if is_func:
            return self.parse_function(fname)
        if name in CONSTANTS:
            return self.mk("const", CONSTANTS[name])
        if _is_single_symbol(name):
            return self.mk("sym", name)
        if name in COMMAND_WORDS:
            raise MathInputError(
                E_SYNTAX,
                f"{name!r} is a command; put it at the start of the line.",
            )
        # A short unknown run of letters reads as a product of one-letter
        # symbols (xy → x·y, like most calculators); the preview shows that
        # reading.  Longer words are almost always typos or unsupported
        # functions, so they are refused rather than silently multiplied out.
        if not name.isalpha():
            raise MathInputError(E_UNKNOWN_NAME, f"Unknown name {name!r}.")
        if len(name) > 3:
            raise MathInputError(
                E_UNKNOWN_NAME,
                f"Unknown name {name!r}.",
                "Write products with a space or *, e.g. a*b, or define the name first.",
            )
        nodes = [self._letter(ch) for ch in name]
        node = nodes[0]
        for nxt in nodes[1:]:
            node = self.mk("mul", children=[node, nxt], implicit=True)
        return node

    def _letter(self, ch: str) -> Node:
        if ch in self.bound or ch in self.known_names:
            return self.mk("sym", ch)
        if ch in ("e", "i"):
            return self.mk("const", ch)
        return self.mk("sym", ch)

    def parse_function(self, fname: str) -> Node:
        power: Node | None = None
        inverse = False
        if self.at_op("^"):
            # sin^2(x) = sin(x)^2 ; sin^-1(x) = asin(x)
            self.next()
            if self.at_op("-") and self.peek(1).kind == "num" and self.peek(1).text == "1":
                self.next()
                self.next()
                if fname not in INVERSE_TRIG:
                    raise MathInputError(
                        E_SYNTAX, f"{fname}^-1 has no inverse function in the playground."
                    )
                inverse = True
            else:
                power = self.parse_primary()
        if inverse:
            fname = INVERSE_TRIG[fname]
        if self.at_op("("):
            args = self.parse_args()
        elif self.at_op("["):
            args = [self.parse_group("[", "]")]
        else:
            if not self.starts_factor():
                raise MathInputError(E_SYNTAX, f"{fname} needs an argument, like {fname}(x).")
            # sin x, sin 2x, sin x^2: the argument is a run of juxtaposed
            # powers that stops at the next function name or operator.
            arg = self.parse_power()
            while self.starts_factor() and not self._next_is_function():
                arg = self.mk("mul", children=[arg, self.parse_power()], implicit=True)
            args = [arg]
        arity = FUNCTIONS[fname]
        if fname == "log" and len(args) == 2:
            pass
        elif len(args) != arity:
            raise MathInputError(E_SYNTAX, f"{fname} takes {arity} argument(s), got {len(args)}.")
        node = self.mk("call", fname, args)
        if power is not None:
            node = self.mk("pow", children=[node, power])
        return node

    def _next_is_function(self) -> bool:
        t = self.peek()
        if t.kind != "ident":
            return False
        name = FUNCTION_ALIASES.get(t.text, t.text)
        return name in FUNCTIONS and t.text not in self.known_names


def _is_single_symbol(name: str) -> bool:
    if len(name) == 1:
        return True
    if name in GREEK_NAMES:
        return True
    # x1, x_1, x_max, v0 → one subscripted symbol
    return bool(re.fullmatch(r"[A-Za-z][0-9]+|[A-Za-z][A-Za-z]*_[A-Za-z0-9]+", name))


def _ends_in_number(node: Node) -> bool:
    while node.kind in ("mul",) and node.children:
        node = node.children[-1]
    return node.kind == "num"


# ── Static size analysis ─────────────────────────────────────────────────────

_TOO_BIG = object()


def check_sizes(node: Node, defs: dict[str, Node] | None = None) -> None:
    """Refuse integer powers / factorials whose value would be astronomically large.

    ``2^2^2^2^2^2`` is a perfectly well-formed expression whose value has
    2^65536 bits; no budget can make building it safe, so it is refused here,
    before Alkahest sees it.  ``defs`` lets a name defined earlier in the
    session be looked through.
    """
    _static_int(node, defs or {}, set())


def _static_int(node: Node, defs: dict[str, Node], seen: set[str]):
    """Integer value of a constant subtree, ``None`` if not an integer constant."""
    k = node.kind
    if k == "num":
        return int(node.value) if node.value.isdigit() else None
    if k == "sym":
        if node.value in defs and node.value not in seen:
            return _static_int(defs[node.value], defs, seen | {node.value})
        return None
    vals = [_static_int(c, defs, seen) for c in node.children]
    if k == "neg":
        v = vals[0]
        return -v if isinstance(v, int) else v
    if k in ("add", "sub", "mul"):
        a, b = vals
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        if k == "mul" and a.bit_length() + b.bit_length() > MAX_INT_BITS:
            raise MathInputError(E_NUMBER_TOO_LARGE, "A number in this input is too large.")
        return a + b if k == "add" else a - b if k == "sub" else a * b
    if k == "pow":
        base, exp = vals
        if isinstance(exp, int):
            if isinstance(base, int):
                if base in (0, 1, -1):
                    return base ** (exp % 2 + 2) if base == -1 else (base if exp > 0 else None)
                # |base| >= 2, so the value has at least |exp| bits; test that
                # first so a huge exponent never reaches float arithmetic.
                bits = float("inf") if abs(exp) > MAX_INT_BITS else abs(exp) * math.log2(abs(base))
                if bits > MAX_INT_BITS:
                    raise MathInputError(
                        E_NUMBER_TOO_LARGE,
                        "This power is too large to compute: its value would have "
                        f"{_fmt_bits(bits)} binary digits.",
                        "The playground caps integer powers at "
                        f"{MAX_INT_BITS:,} bits so one request cannot exhaust the server.",
                    )
                return base**exp if exp >= 0 else None
            if abs(exp) > MAX_SYMBOLIC_EXPONENT and not _is_unit_like(node.children[0]):
                raise MathInputError(
                    E_NUMBER_TOO_LARGE,
                    f"Exponents are capped at {MAX_SYMBOLIC_EXPONENT} for symbolic bases.",
                )
        return None
    if k == "fact":
        v = vals[0]
        if isinstance(v, int):
            if v > MAX_FACTORIAL:
                raise MathInputError(
                    E_NUMBER_TOO_LARGE, f"Factorials are capped at {MAX_FACTORIAL}!."
                )
            return math.factorial(v) if v >= 0 else None
        return None
    return None


def _is_unit_like(node: Node) -> bool:
    return node.kind == "num" and node.value in ("0", "1")


def _fmt_bits(bits: float) -> str:
    if bits == float("inf"):
        return "more than 10^30000"
    if bits < 1e6:
        return f"about {int(bits):,}"
    exp = int(math.log10(bits))
    return f"about 10^{exp}"


# ── Free variables ───────────────────────────────────────────────────────────


def free_names(node: Node, out: set[str] | None = None) -> set[str]:
    if out is None:
        out = set()
    if node.kind == "sym":
        out.add(node.value)
    for c in node.children:
        free_names(c, out)
    return out


# ── Command parser ───────────────────────────────────────────────────────────


@dataclass
class Session:
    """What earlier cells defined, as seen by the parser."""

    names: dict[str, Node] = field(default_factory=dict)  # name → value tree
    functions: dict[str, tuple[list[str], Node]] = field(default_factory=dict)


def _split_top(toks: list[Tok], pred) -> list[list[Tok]]:
    """Split a token list at depth-0 tokens matching ``pred``."""
    parts: list[list[Tok]] = [[]]
    depth = 0
    for t in toks:
        if t.kind == "op" and t.text in "([{":
            depth += 1
        elif t.kind == "op" and t.text in ")]}":
            depth -= 1
        if depth == 0 and pred(t):
            parts.append([])
            continue
        parts[-1].append(t)
    return parts


def _find_top(toks: list[Tok], pred, last: bool = False) -> int:
    depth = 0
    found = -1
    for idx, t in enumerate(toks):
        if t.kind == "op" and t.text in "([{":
            depth += 1
        elif t.kind == "op" and t.text in ")]}":
            depth -= 1
        elif depth == 0 and pred(t):
            if not last:
                return idx
            found = idx
    return found


def _is_word(*words: str):
    return lambda t: t.kind == "ident" and t.text in words


def _is_op(*ops: str):
    return lambda t: t.kind == "op" and t.text in ops


class CommandParser:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()
        self.extra_bound: set[str] = set()

    def expr(self, toks: list[Tok], bound: set[str] | None = None) -> Node:
        if not toks:
            raise MathInputError(E_SYNTAX, "Expected an expression.")
        known_fns = {k: len(v[0]) for k, v in self.session.functions.items()}
        p = Parser(
            [*toks, Tok("end", "", toks[-1].pos + 1)],
            known_names=set(self.session.names),
            known_functions=known_fns,
            bound=(bound or set()) | self.extra_bound,
        )
        node = p.parse_full()
        check_sizes(node, self.session.names)
        return node

    def var_name(self, toks: list[Tok], what: str = "variable") -> str:
        if len(toks) != 1 or toks[0].kind != "ident" or not _is_single_symbol(toks[0].text):
            text = " ".join(t.text for t in toks) or "nothing"
            raise MathInputError(E_SYNTAX, f"Expected a {what} name, found {text!r}.")
        name = toks[0].text
        if name in CONSTANTS and name not in ("e", "i"):
            raise MathInputError(E_SYNTAX, f"{name!r} is a constant, not a variable.")
        return name

    def int_arg(self, toks: list[Tok], what: str, cap: int) -> int:
        if len(toks) != 1 or toks[0].kind != "num" or not toks[0].text.isdigit():
            raise MathInputError(E_SYNTAX, f"The {what} must be a whole number.")
        n = int(toks[0].text)
        if n > cap:
            raise MathInputError(
                E_NUMBER_TOO_LARGE, f"The {what} is capped at {cap} in the playground."
            )
        return n

    # entry point
    def parse(self, text: str, *, allow_definition: bool = True) -> Command:
        if len(text) > MAX_INPUT_CHARS:
            raise MathInputError(
                E_TOO_LARGE_INPUT,
                f"The input is {len(text)} characters long; the limit is {MAX_INPUT_CHARS}.",
            )
        toks = tokenize(normalize(text))[:-1]
        if not toks:
            raise MathInputError(E_SYNTAX, "Type some maths, e.g. integrate x^2 dx.")
        return self.parse_tokens(toks, allow_definition=allow_definition)

    def parse_tokens(self, toks: list[Tok], *, allow_definition: bool = True) -> Command:
        if allow_definition:
            d = self.try_definition(toks)
            if d is not None:
                return d
        head = toks[0]
        word = head.text if head.kind == "ident" else ""
        lw = word.lower()

        # call form: cmd(args...) spanning the whole input
        if word in COMMAND_WORDS or lw in COMMAND_WORDS:
            call = self.try_call_form(toks)
            if call is not None:
                return call

        if word == "N" and len(toks) > 1 and toks[1].kind == "op" and toks[1].text == "[":
            return self.numeric_bracket(toks)
        if lw in ("integrate", "integral", "int", "antiderivative"):
            return self.integrate(self._drop_of(toks[1:]))
        if word == "d" and len(toks) > 1 and toks[1].kind == "op" and toks[1].text in "/^":
            return self.leibniz(toks)
        if lw in ("derivative", "differentiate", "diff"):
            return self.derivative(self._drop_of(toks[1:]), order=1)
        if lw in ("second", "third") and len(toks) > 1 and toks[1].text == "derivative":
            return self.derivative(self._drop_of(toks[2:]), order=2 if lw == "second" else 3)
        if lw == "solve":
            return self.solve(toks[1:])
        if lw in ("simplify", "factor", "expand"):
            rest = self._drop_of(toks[1:])
            return Command(kind=lw, expr=self.expr(rest))
        if lw in ("limit", "lim"):
            return self.limit(self._drop_of(toks[1:]))
        if lw in ("series", "taylor"):
            return self.series(self._drop_of(toks[1:]))
        if lw in ("sum", "product"):
            return self.sum(lw, self._drop_of(toks[1:]))
        if lw in ("numeric", "numerically", "approx"):
            return self.numeric(toks[1:])
        if lw == "evaluate":
            return Command(kind="expr", expr=self.expr(self._drop_of(toks[1:])))
        # bare equation → solve
        if _find_top(toks, _is_op("=")) >= 0:
            return self.solve(toks)
        return Command(kind="expr", expr=self.expr(toks))

    @staticmethod
    def _drop_of(toks: list[Tok]) -> list[Tok]:
        if toks and toks[0].kind == "ident" and toks[0].text == "of":
            return toks[1:]
        return toks

    # definitions
    def try_definition(self, toks: list[Tok]) -> Command | None:
        eq = _find_top(toks, _is_op("=", ":="))
        if eq <= 0:
            return None
        lhs, rhs = toks[:eq], toks[eq + 1 :]
        if lhs[0].kind != "ident":
            return None
        name = lhs[0].text
        if (
            name in COMMAND_WORDS
            or name.lower() in COMMAND_WORDS
            or name in KEYWORDS
            or name in CONSTANTS
            or FUNCTION_ALIASES.get(name, name) in FUNCTIONS
            or name == "d"
        ):
            return None
        params: list[str] = []
        if len(lhs) == 1:
            pass
        elif (
            len(lhs) >= 4
            and lhs[1].kind == "op"
            and lhs[1].text == "("
            and lhs[-1].kind == "op"
            and lhs[-1].text == ")"
        ):
            for part in _split_top(lhs[2:-1], _is_op(",")):
                params.append(self.var_name(part, "parameter"))
            if len(set(params)) != len(params):
                raise MathInputError(E_SYNTAX, "Parameter names must be different.")
        else:
            return None
        if not rhs:
            raise MathInputError(E_SYNTAX, f"Nothing after '=' in the definition of {name}.")
        starts_solve = rhs[0].kind == "ident" and rhs[0].text.lower() == "solve"
        if _find_top(rhs, _is_op("=")) >= 0 and not starts_solve:
            raise MathInputError(E_SYNTAX, "A definition can only have one '='.")
        if not (name.isidentifier() and (_is_single_symbol(name) or name.isalnum())):
            raise MathInputError(E_SYNTAX, f"{name!r} is not a usable name.")
        if len(name) > 24:
            raise MathInputError(E_SYNTAX, "Names are limited to 24 characters.")
        self.extra_bound = set(params)
        try:
            body = self.parse_tokens(rhs, allow_definition=False)
        finally:
            self.extra_bound = set()
        if body.kind == "solve":
            raise MathInputError(
                E_UNSUPPORTED,
                "A definition needs a single value; solve can return several.",
                "Run solve on its own line, then define the solution you want.",
            )
        return Command(kind="define", name=name, params=params, body=body)

    def try_call_form(self, toks: list[Tok]) -> Command | None:
        if len(toks) < 3 or toks[1].kind != "op" or toks[1].text != "(":
            return None
        if toks[-1].kind != "op" or toks[-1].text != ")":
            return None
        # the opening paren must close at the very end
        depth = 0
        for idx, t in enumerate(toks[1:], start=1):
            if t.kind == "op" and t.text in "([{":
                depth += 1
            elif t.kind == "op" and t.text in ")]}":
                depth -= 1
                if depth == 0 and idx != len(toks) - 1:
                    return None
        args = _split_top(toks[2:-1], _is_op(","))
        name = toks[0].text.lower()
        n = len(args)
        if name in ("simplify", "factor", "expand") and n == 1:
            return Command(kind=name, expr=self.expr(args[0]))
        if name in ("integrate", "integral", "int") and n in (1, 2, 4):
            if n == 1:
                return self.integrate(args[0])
            var = self.var_name(args[1])
            body = self.expr(args[0])
            if n == 4:
                return Command(
                    kind="integrate",
                    expr=body,
                    var=var,
                    lower=self.expr(args[2]),
                    upper=self.expr(args[3]),
                    var_explicit=True,
                )
            return Command(kind="integrate", expr=body, var=var, var_explicit=True)
        if name in ("diff", "derivative", "differentiate") and n in (1, 2, 3):
            body = self.expr(args[0])
            if n == 1:
                return Command(kind="diff", expr=body, var=None, order=1)
            var = self.var_name(args[1])
            order = self.int_arg(args[2], "derivative order", MAX_DIFF_ORDER) if n == 3 else 1
            return Command(kind="diff", expr=body, var=var, order=order, var_explicit=True)
        if name in ("limit", "lim") and n in (3, 4):
            cmd = Command(
                kind="limit",
                expr=self.expr(args[0]),
                var=self.var_name(args[1]),
                point=self.expr(args[2]),
                var_explicit=True,
            )
            if n == 4:
                d = "".join(t.text for t in args[3])
                if d not in ("+", "-"):
                    raise MathInputError(E_SYNTAX, "The direction must be + or -.")
                cmd.direction = d
            return cmd
        if name in ("series", "taylor") and n in (1, 2, 3, 4):
            cmd = Command(kind="series", expr=self.expr(args[0]), order=6)
            if n >= 2:
                cmd.var = self.var_name(args[1])
                cmd.var_explicit = True
            cmd.point = self.expr(args[2]) if n >= 3 else Node("num", "0")
            if n == 4:
                cmd.order = self.int_arg(args[3], "series order", MAX_SERIES_ORDER)
            return cmd
        if name in ("sum", "product") and n == 4:
            var = self.var_name(args[1])
            return Command(
                kind=name,
                expr=self.expr(args[0], bound={var}),
                var=var,
                lower=self.expr(args[2]),
                upper=self.expr(args[3]),
                var_explicit=True,
            )
        if name == "solve" and n >= 1:
            # solve(eq1, eq2, x, y): trailing bare names are the unknowns
            eqs, vars_ = [], []
            for a in args:
                if (
                    len(a) == 1
                    and a[0].kind == "ident"
                    and _is_single_symbol(a[0].text)
                    and a[0].text not in self.session.names
                    and eqs
                ):
                    vars_.append(a[0].text)
                else:
                    if vars_:
                        raise MathInputError(E_SYNTAX, "Put the unknowns after the equations.")
                    eqs.append(a)
            return self._solve_parts(eqs, vars_)
        if name in ("numeric", "numerically", "approx", "n") and n in (1, 2):
            digits = self.int_arg(args[1], "number of digits", MAX_DIGITS) if n == 2 else 15
            return Command(kind="numeric", expr=self.expr(args[0]), digits=max(digits, 1))
        return None

    # integrate
    def integrate(self, toks: list[Tok]) -> Command:
        if not toks:
            raise MathInputError(E_SYNTAX, "What should be integrated? e.g. integrate x^2 dx")
        lower = upper = None
        var: str | None = None
        f = _find_top(toks, _is_word("from"), last=True)
        body = toks
        if f >= 0:
            body, bounds = toks[:f], toks[f + 1 :]
            # a trailing "dx" after the bounds
            if bounds and _dvar(bounds[-1]):
                var = _dvar(bounds[-1])
                bounds = bounds[:-1]
            t = _find_top(bounds, _is_word("to"))
            if t < 0:
                raise MathInputError(E_SYNTAX, "Write the limits as: from a to b.")
            # "from x = 0 to 1"
            lo_toks = bounds[:t]
            eqi = _find_top(lo_toks, _is_op("="))
            if eqi >= 0:
                v = self.var_name(lo_toks[:eqi])
                if var and var != v:
                    raise MathInputError(E_SYNTAX, f"d{var} does not match {v} = …")
                var = v
                lo_toks = lo_toks[eqi + 1 :]
            lower = self.expr(lo_toks)
            upper = self.expr(bounds[t + 1 :])
        # with respect to x / wrt x
        body, wrt = self._strip_wrt(body)
        if wrt:
            var = wrt
        if body and _dvar(body[-1]):
            v = _dvar(body[-1])
            if var and var != v:
                raise MathInputError(E_SYNTAX, f"d{v} does not match the variable {var}.")
            var = v
            body = body[:-1]
            if body and body[-1].kind == "op" and body[-1].text == "*":
                body = body[:-1]
        explicit = var is not None
        bound = {var} if var else set()
        expr = self.expr(body, bound=bound)
        return Command(
            kind="integrate",
            expr=expr,
            var=var,
            lower=lower,
            upper=upper,
            var_explicit=explicit,
        )

    def _strip_wrt(self, toks: list[Tok]) -> tuple[list[Tok], str | None]:
        w = _find_top(toks, _is_word("wrt", "with"), last=True)
        if w < 0:
            return toks, None
        rest = toks[w + 1 :]
        if toks[w].text == "with":
            if len(rest) >= 2 and rest[0].text == "respect" and rest[1].text == "to":
                rest = rest[2:]
            else:
                raise MathInputError(E_SYNTAX, "Write: with respect to x.")
        return toks[:w], self.var_name(rest)

    # derivatives
    def leibniz(self, toks: list[Tok]) -> Command:
        # d/dx …, d/d x …, d^2/dx^2 …, d2/dx2
        i = 1
        order = 1
        if toks[i].text == "^":
            if i + 1 >= len(toks) or toks[i + 1].kind != "num":
                raise MathInputError(E_SYNTAX, "Write the order as d^2/dx^2.")
            order = self.int_arg([toks[i + 1]], "derivative order", MAX_DIFF_ORDER)
            i += 2
        if i >= len(toks) or toks[i].text != "/":
            raise MathInputError(E_SYNTAX, "Write a derivative as d/dx followed by the expression.")
        i += 1
        if i >= len(toks):
            raise MathInputError(E_SYNTAX, "d/d… needs a variable, like d/dx.")
        var: str | None = None
        t = toks[i]
        if _dvar(t):
            var = _dvar(t)
            i += 1
        elif t.kind == "ident" and t.text == "d" and i + 1 < len(toks):
            var = self.var_name([toks[i + 1]])
            i += 2
        else:
            raise MathInputError(E_SYNTAX, "d/d… needs a variable, like d/dx.")
        if i < len(toks) and toks[i].text == "^":
            if i + 1 >= len(toks) or toks[i + 1].kind != "num":
                raise MathInputError(E_SYNTAX, "Write the order as d^2/dx^2.")
            o2 = int(toks[i + 1].text)
            if o2 != order:
                raise MathInputError(
                    E_SYNTAX, "The orders on top and bottom of d^n/dx^n must match."
                )
            i += 2
        body = toks[i:]
        return Command(kind="diff", expr=self.expr(body), var=var, order=order, var_explicit=True)

    def derivative(self, toks: list[Tok], order: int) -> Command:
        body, wrt = self._strip_wrt(toks)
        return Command(
            kind="diff", expr=self.expr(body), var=wrt, order=order, var_explicit=wrt is not None
        )

    # solve
    def solve(self, toks: list[Tok]) -> Command:
        f = _find_top(toks, _is_word("for"), last=True)
        vars_: list[str] = []
        if f >= 0:
            for part in _split_top(toks[f + 1 :], lambda t: t.text in (",", "and")):
                vars_.append(self.var_name(part, "unknown"))
            toks = toks[:f]
        eqs = _split_top(toks, lambda t: (t.kind == "op" and t.text == ",") or t.text == "and")
        return self._solve_parts(eqs, vars_)

    def _solve_parts(self, eqs: list[list[Tok]], vars_: list[str]) -> Command:
        if not eqs or not any(eqs):
            raise MathInputError(E_SYNTAX, "Give an equation to solve, e.g. solve x^2 = 4.")
        if len(eqs) > 8:
            raise MathInputError(E_TOO_LARGE_INPUT, "At most 8 equations at once.")
        pairs: list[tuple[Node, Node]] = []
        for e in eqs:
            k = _find_top(e, _is_op("="))
            if k >= 0:
                lhs, rhs = e[:k], e[k + 1 :]
                if _find_top(rhs, _is_op("=")) >= 0:
                    raise MathInputError(E_SYNTAX, "An equation can only have one '='.")
                pairs.append((self.expr(lhs), self.expr(rhs)))
            else:
                pairs.append((self.expr(e), Node("num", "0")))
        if len(set(vars_)) != len(vars_):
            raise MathInputError(E_SYNTAX, "Each unknown should be listed once.")
        return Command(kind="solve", equations=pairs, vars=vars_, var_explicit=bool(vars_))

    # limit
    def limit(self, toks: list[Tok]) -> Command:
        a = _find_top(toks, _is_word("as"), last=True)
        if a >= 0:
            body, spec = toks[:a], toks[a + 1 :]
            var, point, direction = self._approach(spec)
            return Command(
                kind="limit",
                expr=self.expr(body),
                var=var,
                point=point,
                direction=direction,
                var_explicit=True,
            )
        # lim x -> 0 f(x)    (also lim_(x->0) / lim_{x->0})
        if toks and toks[0].kind == "op" and toks[0].text in "([{":
            close = {"(": ")", "[": "]", "{": "}"}[toks[0].text]
            depth = 0
            for idx, t in enumerate(toks):
                if t.kind == "op" and t.text in "([{":
                    depth += 1
                elif t.kind == "op" and t.text in ")]}":
                    depth -= 1
                    if depth == 0:
                        if t.text != close:
                            break
                        var, point, direction = self._approach(toks[1:idx])
                        return Command(
                            kind="limit",
                            expr=self.expr(toks[idx + 1 :]),
                            var=var,
                            point=point,
                            direction=direction,
                            var_explicit=True,
                        )
        if len(toks) >= 3 and toks[1].kind == "op" and toks[1].text == "->":
            var = self.var_name([toks[0]])
            # the point is one primary (number, name, oo, or a group), optional ±
            p = Parser([*toks[2:], Tok("end", "", 0)], known_names=set(self.session.names))
            if p.at_op("-"):
                p.next()
                point = Node("neg", children=[p.parse_primary()])
            else:
                point = p.parse_primary()
            direction = None
            # a one-sided approach is written x -> 0^+ in this form
            if p.at_op("^") and p.peek(1).kind == "op" and p.peek(1).text in "+-":
                p.next()
                direction = p.next().text
            rest = toks[2 + p.i :]
            return Command(
                kind="limit",
                expr=self.expr(rest),
                var=var,
                point=point,
                direction=direction,
                var_explicit=True,
            )
        raise MathInputError(
            E_SYNTAX, "Write a limit as: limit sin(x)/x as x -> 0", "or: lim x->0 sin(x)/x"
        )

    def _approach(self, spec: list[Tok]) -> tuple[str, Node, str | None]:
        k = _find_top(spec, _is_op("->"))
        if k < 0:
            k = _find_top(spec, _is_word("to"))
        if k < 0:
            raise MathInputError(E_SYNTAX, "Write the approach as x -> 0.")
        var = self.var_name(spec[:k])
        rest = spec[k + 1 :]
        direction = None
        # trailing direction: 0+, 0-, 0^+, 0^-, "from the left/right/above/below"
        f = _find_top(rest, _is_word("from"))
        if f >= 0:
            words = [t.text for t in rest[f + 1 :] if t.text != "the"]
            if words in (["left"], ["below"]):
                direction = "-"
            elif words in (["right"], ["above"]):
                direction = "+"
            else:
                raise MathInputError(E_SYNTAX, "Write: from the left / from the right.")
            rest = rest[:f]
        if rest and rest[-1].kind == "op" and rest[-1].text in "+-":
            direction = rest[-1].text
            rest = rest[:-1]
            if rest and rest[-1].kind == "op" and rest[-1].text == "^":
                rest = rest[:-1]
        return var, self.expr(rest), direction

    # series
    def series(self, toks: list[Tok]) -> Command:
        order = 6
        o = _find_top(toks, _is_word("order"), last=True)
        if o >= 0:
            order = self.int_arg(toks[o + 1 :], "series order", MAX_SERIES_ORDER)
            toks = toks[:o]
            while toks and toks[-1].kind == "ident" and toks[-1].text in ("to", "up"):
                toks = toks[:-1]
            if toks and toks[-1].kind == "op" and toks[-1].text == ",":
                toks = toks[:-1]
        var: str | None = None
        point: Node = Node("num", "0")
        a = _find_top(toks, _is_word("at", "around", "about"), last=True)
        if a >= 0:
            spec = toks[a + 1 :]
            toks = toks[:a]
            k = _find_top(spec, _is_op("="))
            if k >= 0:
                var = self.var_name(spec[:k])
                spec = spec[k + 1 :]
            point = self.expr(spec)
        body, wrt = self._strip_wrt(toks)
        if wrt:
            var = wrt
        return Command(
            kind="series",
            expr=self.expr(body),
            var=var,
            point=point,
            order=order,
            var_explicit=var is not None,
        )

    # sum / product
    def sum(self, kind: str, toks: list[Tok]) -> Command:
        f = _find_top(toks, _is_word("for", "from"), last=True)
        if f < 0:
            raise MathInputError(E_SYNTAX, f"Write the range as: {kind} 1/k^2 for k = 1 to oo")
        body, spec = toks[:f], toks[f + 1 :]
        k = _find_top(spec, _is_op("="))
        if k < 0:
            raise MathInputError(E_SYNTAX, "Write the range as: for k = 1 to oo")
        var = self.var_name(spec[:k], "index")
        rng = spec[k + 1 :]
        t = _find_top(rng, lambda tk: tk.text in ("to", ".."))
        if t < 0:
            raise MathInputError(E_SYNTAX, "Write the range as: for k = 1 to oo")
        return Command(
            kind=kind,
            expr=self.expr(body, bound={var}),
            var=var,
            lower=self.expr(rng[:t]),
            upper=self.expr(rng[t + 1 :]),
            var_explicit=True,
        )

    # numeric
    def numeric(self, toks: list[Tok]) -> Command:
        digits = 15
        d = _find_top(toks, _is_word("digits"), last=True)
        if d >= 0:
            if d != len(toks) - 1:
                raise MathInputError(E_SYNTAX, "Write: numeric pi to 50 digits")
            spec = toks[:d]
            k = _find_top(spec, _is_word("to", "with"), last=True)
            if k < 0:
                raise MathInputError(E_SYNTAX, "Write: numeric pi to 50 digits")
            digits = self.int_arg(spec[k + 1 :], "number of digits", MAX_DIGITS)
            toks = spec[:k]
        return Command(kind="numeric", expr=self.expr(toks), digits=max(digits, 1))

    def numeric_bracket(self, toks: list[Tok]) -> Command:
        if toks[-1].kind != "op" or toks[-1].text != "]":
            raise MathInputError(E_SYNTAX, "N[...] needs a closing ].")
        args = _split_top(toks[2:-1], _is_op(","))
        if len(args) not in (1, 2):
            raise MathInputError(E_SYNTAX, "Write N[expression] or N[expression, digits].")
        digits = self.int_arg(args[1], "number of digits", MAX_DIGITS) if len(args) == 2 else 15
        return Command(kind="numeric", expr=self.expr(args[0]), digits=max(digits, 1))


def _dvar(t: Tok) -> str | None:
    """``dx`` → ``x``, ``dtheta`` → ``theta`` (an integration/differential token)."""
    if t.kind != "ident" or len(t.text) < 2 or t.text[0] != "d":
        return None
    v = t.text[1:]
    if len(v) == 1 and v.isalpha():
        return v
    if v in GREEK_NAMES:
        return v
    return None


# ── LaTeX of the *interpretation* ────────────────────────────────────────────
# This prints the tree exactly as parsed (no simplification), so the preview
# shows how the input was read: 1/2x as ½·x, sin x^2 as sin(x²), xy as x·y.

_LATEX_FUNCS = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sec": r"\sec",
    "csc": r"\csc",
    "cot": r"\cot",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "log": r"\ln",
    "gamma": r"\Gamma",
    "zeta": r"\zeta",
}


def sym_latex(name: str) -> str:
    if name in GREEK_NAMES:
        return "\\" + name
    m = re.fullmatch(r"([A-Za-z]+)_?([0-9]+)", name) or re.fullmatch(
        r"([A-Za-z]+)_([A-Za-z0-9]+)", name
    )
    if m:
        base, sub = m.group(1), m.group(2)
        base_l = "\\" + base if base in GREEK_NAMES else base
        sub_l = "\\" + sub if sub in GREEK_NAMES else sub
        return f"{base_l}_{{{sub_l}}}"
    if len(name) > 1:
        return r"\mathrm{" + name + "}"
    return name


_PREC = {"add": 1, "sub": 1, "neg": 2, "mul": 3, "div": 3, "pow": 5, "fact": 6}


def to_latex(node: Node) -> str:
    return _tex(node, 0)


def _paren(s: str) -> str:
    return r"\left(" + s + r"\right)"


def _tex(n: Node, parent: int) -> str:
    k = n.kind
    if k == "num":
        return n.value
    if k == "sym":
        return sym_latex(n.value)
    if k == "const":
        return {"pi": r"\pi", "e": "e", "i": "i", "oo": r"\infty"}[n.value]
    if k in ("add", "sub"):
        a = _tex(n.children[0], 1)
        b_node = n.children[1]
        b = _tex(b_node, 2 if k == "sub" else 1)
        if k == "add" and b_node.kind == "neg":
            s = f"{a} - {_tex(b_node.children[0], 2)}"
        else:
            s = f"{a} {'+' if k == 'add' else '-'} {b}"
        return _paren(s) if parent > 1 else s
    if k == "neg":
        s = "-" + _tex(n.children[0], 2)
        return _paren(s) if parent > 2 else s
    if k == "mul":
        a = _tex(n.children[0], 3)
        b = _tex(n.children[1], 3)
        right = n.children[1]
        left_last = _rightmost(n.children[0])
        needs_dot = (
            right.kind == "num"
            or (right.kind == "pow" and _leftmost(right).kind == "num")
            or (left_last.kind == "num" and _leftmost(right).kind == "num")
            or b.startswith("-")
        )
        s = f"{a} \\cdot {b}" if needs_dot else f"{a} {b}"
        return _paren(s) if parent > 3 else s
    if k == "div":
        s = r"\frac{" + _tex(n.children[0], 0) + "}{" + _tex(n.children[1], 0) + "}"
        return _paren(s) if parent >= 5 else s
    if k == "pow":
        base, ex = n.children
        if base.kind == "const" and base.value == "e":
            return "e^{" + _tex(ex, 0) + "}"
        if base.kind == "call" and base.value in _LATEX_FUNCS and len(base.children) == 1:
            # sin^2(x) reads better than (sin(x))^2
            return (
                _LATEX_FUNCS[base.value]
                + "^{"
                + _tex(ex, 0)
                + "}"
                + _paren(_tex(base.children[0], 0))
            )
        b = _tex(base, 5)
        if base.kind in ("pow", "neg", "div") or (base.kind == "num" and "." in base.value):
            b = _paren(_tex(base, 0))
        s = b + "^{" + _tex(ex, 0) + "}"
        return s
    if k == "fact":
        inner = n.children[0]
        b = _tex(inner, 6)
        if inner.kind not in ("num", "sym", "const"):
            b = _paren(_tex(inner, 0))
        return b + "!"
    if k == "call":
        f = n.value
        args = n.children
        if f == "sqrt":
            return r"\sqrt{" + _tex(args[0], 0) + "}"
        if f == "cbrt":
            return r"\sqrt[3]{" + _tex(args[0], 0) + "}"
        if f == "abs":
            return r"\left|" + _tex(args[0], 0) + r"\right|"
        if f == "exp":
            return "e^{" + _tex(args[0], 0) + "}"
        if f == "floor":
            return r"\left\lfloor " + _tex(args[0], 0) + r"\right\rfloor"
        if f == "ceil":
            return r"\left\lceil " + _tex(args[0], 0) + r"\right\rceil"
        if f == "factorial":
            return _paren(_tex(args[0], 0)) + "!"
        if f == "log" and len(args) == 2:
            return r"\log_{" + _tex(args[1], 0) + "}" + _paren(_tex(args[0], 0))
        if f == "log10":
            return r"\log_{10}" + _paren(_tex(args[0], 0))
        if f == "log2":
            return r"\log_{2}" + _paren(_tex(args[0], 0))
        head = _LATEX_FUNCS.get(f, r"\operatorname{" + f + "}")
        return head + _paren(", ".join(_tex(a, 0) for a in args))
    if k == "ucall":
        return sym_latex(n.value) + _paren(", ".join(_tex(a, 0) for a in n.children))
    raise MathInputError(E_INTERNAL, f"cannot print node {k}")


def _rightmost(n: Node) -> Node:
    while n.kind in ("mul",):
        n = n.children[-1]
    return n


def _leftmost(n: Node) -> Node:
    while n.kind in ("mul", "pow", "fact") and n.children:
        n = n.children[0]
    return n


def _bracket(n: Node) -> str:
    s = to_latex(n)
    if n.kind in ("add", "sub", "mul", "neg"):
        return r"\left[" + s + r"\right]"
    return s


def command_latex(cmd: Command, var_hint: str | None = None) -> str:
    """LaTeX for how the whole line was understood."""
    k = cmd.kind
    var = cmd.var or var_hint or "x"
    v = sym_latex(var)
    if k == "define":
        assert cmd.body is not None
        assert cmd.name is not None
        lhs = sym_latex(cmd.name)
        if cmd.params:
            lhs += _paren(", ".join(sym_latex(p) for p in cmd.params))
        return lhs + r" := " + command_latex(cmd.body)
    if k == "expr":
        assert cmd.expr is not None
        return to_latex(cmd.expr)
    if k == "integrate":
        assert cmd.expr is not None
        lim = ""
        if cmd.lower is not None and cmd.upper is not None:
            lim = "_{" + to_latex(cmd.lower) + "}^{" + to_latex(cmd.upper) + "}"
        body = to_latex(cmd.expr)
        if cmd.expr.kind in ("add", "sub"):
            body = _paren(body)
        return r"\int" + lim + " " + body + r"\,d" + v
    if k == "diff":
        assert cmd.expr is not None
        if cmd.order == 1:
            op = r"\frac{d}{d" + v + "}"
        else:
            op = r"\frac{d^{" + str(cmd.order) + "}}{d" + v + "^{" + str(cmd.order) + "}}"
        return op + _bracket(cmd.expr)
    if k == "solve":
        eqs = r",\quad ".join(to_latex(a) + " = " + to_latex(b) for a, b in cmd.equations)
        s = r"\text{solve}\quad " + eqs
        if cmd.vars:
            s += r"\quad\text{for }" + ", ".join(sym_latex(x) for x in cmd.vars)
        return s
    if k in ("simplify", "factor", "expand"):
        assert cmd.expr is not None
        return r"\operatorname{" + k + "}" + _paren(to_latex(cmd.expr))
    if k == "limit":
        assert cmd.expr is not None
        assert cmd.point is not None
        pt = to_latex(cmd.point)
        if cmd.direction:
            pt += "^{" + cmd.direction + "}"
        return r"\lim_{" + v + r" \to " + pt + "} " + _bracket_if_sum(cmd.expr)
    if k == "series":
        assert cmd.expr is not None
        assert cmd.point is not None
        return (
            to_latex(cmd.expr)
            + r"\quad\text{as a series at }"
            + v
            + " = "
            + to_latex(cmd.point)
            + r"\text{, up to }O\!\left("
            + _series_o(v, cmd.point)
            + "^{"
            + str(cmd.order)
            + r"}\right)"
        )
    if k in ("sum", "product"):
        assert cmd.expr is not None
        assert cmd.lower is not None
        assert cmd.upper is not None
        op = r"\sum" if k == "sum" else r"\prod"
        return (
            op
            + "_{"
            + v
            + "="
            + to_latex(cmd.lower)
            + "}^{"
            + to_latex(cmd.upper)
            + "} "
            + _bracket_if_sum(cmd.expr)
        )
    if k == "numeric":
        assert cmd.expr is not None
        return to_latex(cmd.expr) + r"\quad\text{to " + str(cmd.digits) + r" significant digits}"
    raise MathInputError(E_INTERNAL, f"cannot print command {k}")


def _bracket_if_sum(n: Node) -> str:
    s = to_latex(n)
    return _paren(s) if n.kind in ("add", "sub") else s


def _series_o(v: str, point: Node) -> str:
    if point.kind == "num" and point.value == "0":
        return v
    if point.kind == "const" and point.value == "oo":
        return "1/" + v
    return _paren(v + " - " + to_latex(point))


def interpret(text: str, session: Session | None = None) -> tuple[Command, str]:
    """Parse ``text`` and return the command and its interpretation LaTeX."""
    cmd = CommandParser(session).parse(text)
    return cmd, command_latex(cmd, infer_var_hint(cmd))


def infer_var_hint(cmd: Command) -> str | None:
    """Best guess at the variable a command acts on, for the preview only."""
    if cmd.var:
        return cmd.var
    node = cmd.expr if cmd.kind != "define" else (cmd.body.expr if cmd.body else None)
    if node is None:
        return None
    names = sorted(free_names(node))
    if "x" in names:
        return "x"
    return names[0] if len(names) == 1 else None


def build_session(definitions: list) -> Session:
    """Rebuild what earlier cells defined from the client's ``definitions`` list.

    Each entry is ``{"name": str, "params": [str], "value": str}`` where
    ``value`` is the text the server returned for that cell.  Values are
    re-parsed with the same restricted grammar as user input (larger length
    limit), so a hostile client gains nothing by editing them.
    """
    if not isinstance(definitions, list):
        raise MathInputError(E_SYNTAX, "Malformed definitions.")
    if len(definitions) > MAX_SESSION_DEFINITIONS:
        raise MathInputError(
            E_TOO_LARGE_INPUT, f"At most {MAX_SESSION_DEFINITIONS} definitions per request."
        )
    session = Session()
    for d in definitions:
        if not isinstance(d, dict):
            raise MathInputError(E_SYNTAX, "Malformed definition.")
        name = d.get("name")
        value = d.get("value")
        params = d.get("params") or []
        if not isinstance(name, str) or not isinstance(value, str):
            raise MathInputError(E_SYNTAX, "Malformed definition.")
        if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
            raise MathInputError(E_SYNTAX, "Malformed definition parameters.")
        if len(value) > MAX_DEFINITION_CHARS:
            raise MathInputError(
                E_TOO_LARGE_INPUT, f"The definition of {name} is too large to reuse."
            )
        # Re-validate the name and parameters exactly as the parser would.
        lhs = f"{name}({', '.join(params)}) = 0" if params else f"{name} = 0"
        if len(lhs) > MAX_INPUT_CHARS:
            raise MathInputError(E_SYNTAX, "Malformed definition.")
        probe = CommandParser(session).try_definition(tokenize(normalize(lhs))[:-1])
        if probe is None or probe.name != name or probe.params != params:
            raise MathInputError(E_SYNTAX, f"{name!r} is not a usable name.")
        cp = CommandParser(session)
        try:
            node = cp.expr(tokenize(normalize(value))[:-1], bound=set(params))
        except MathInputError as e:
            raise MathInputError(
                e.code, f"The earlier definition of {name} cannot be reused here: {e.message}"
            ) from e
        if params:
            session.functions[name] = (list(params), node)
            session.names.pop(name, None)
        else:
            session.names[name] = node
            session.functions.pop(name, None)
    return session
