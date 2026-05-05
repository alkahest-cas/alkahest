"""V2-20: LaTeX and Unicode pretty-printing for Alkahest expressions.

Pure-Python tree walk over the structure exposed by ``Expr.node()``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Precedence constants
# ---------------------------------------------------------------------------
_PREC_ADD = 10
_PREC_MUL = 20
_PREC_POW = 30
_PREC_ATOM = 100

# ---------------------------------------------------------------------------
# Name/operator tables
# ---------------------------------------------------------------------------

_LATEX_FUNCS: dict[str, str] = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "log": r"\ln",
    "sign": r"\operatorname{sign}",
    "round": r"\operatorname{round}",
    "erf": r"\operatorname{erf}",
    "erfc": r"\operatorname{erfc}",
    "gamma": r"\Gamma",
}

_UNICODE_FUNCS: dict[str, str] = {
    "gamma": "Γ",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "log": "ln",
}

_GREEK: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lamda": "λ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Alpha": "Α",
    "Beta": "Β",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Epsilon": "Ε",
    "Zeta": "Ζ",
    "Eta": "Η",
    "Theta": "Θ",
    "Iota": "Ι",
    "Kappa": "Κ",
    "Lambda": "Λ",
    "Mu": "Μ",
    "Nu": "Ν",
    "Xi": "Ξ",
    "Pi": "Π",
    "Rho": "Ρ",
    "Sigma": "Σ",
    "Tau": "Τ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Chi": "Χ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "inf": "∞",
    "oo": "∞",
}

_PREDICATE_LATEX: dict[str, str] = {
    "lt": "<",
    "le": r"\le",
    "gt": ">",
    "ge": r"\ge",
    "eq": "=",
    "ne": r"\ne",
    "and": r"\land",
    "or": r"\lor",
    "not": r"\lnot",
    "true": r"\top",
    "false": r"\bot",
}

_PREDICATE_UNICODE: dict[str, str] = {
    "lt": "<",
    "le": "≤",
    "gt": ">",
    "ge": "≥",
    "eq": "=",
    "ne": "≠",
    "and": "∧",
    "or": "∨",
    "not": "¬",
    "true": "⊤",
    "false": "⊥",
}

_UNICODE_FRACS: dict[tuple[int, int], str] = {
    (1, 2): "½",
    (1, 3): "⅓",
    (2, 3): "⅔",
    (1, 4): "¼",
    (3, 4): "¾",
    (1, 5): "⅕",
    (2, 5): "⅖",
    (3, 5): "⅗",
    (4, 5): "⅘",
    (1, 6): "⅙",
    (5, 6): "⅚",
    (1, 7): "⅐",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (5, 8): "⅝",
    (7, 8): "⅞",
    (1, 9): "⅑",
    (1, 10): "⅒",
}

_SUP_TABLE = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _to_superscript(s: str) -> str | None:
    """Convert an integer string to Unicode superscript, or None if unsupported."""
    translated = s.translate(_SUP_TABLE)
    return (
        translated
        if translated != s or s in ("0",)
        else (translated if all(c in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻" for c in translated) else None)
    )


def _to_superscript(s: str) -> str | None:
    out = []
    for ch in s:
        mapped = ch.translate(_SUP_TABLE)
        if mapped == ch:
            # No translation happened — unsupported character
            return None
        out.append(mapped)
    return "".join(out)


def _unicode_frac(num: int, den: int) -> str:
    common = _UNICODE_FRACS.get((num, den))
    return common if common else f"{num}/{den}"


# ---------------------------------------------------------------------------
# LaTeX renderer
# ---------------------------------------------------------------------------


class _LatexRenderer:
    def render(self, expr) -> str:
        s, _ = self._r(expr)
        return s

    # ---- core dispatch ----

    def _r(self, expr) -> tuple[str, int]:
        """Returns (latex, precedence)."""
        n = expr.node()
        t = n[0]

        if t == "symbol":
            return self._symbol(n[1]), _PREC_ATOM
        if t == "integer":
            v = int(n[1])
            return (str(-v) if v < 0 else str(v)), _PREC_ATOM
        if t == "rational":
            num, den = int(n[1]), int(n[2])
            sign = -1 if num < 0 else 1
            s = self._frac(str(abs(num)), str(den))
            # Signed rational as atom — sign handled by parent
            return (("-" + s) if sign < 0 else s), _PREC_ATOM
        if t == "float":
            return n[1], _PREC_ATOM
        if t == "add":
            return self._add(n[1]), _PREC_ADD
        if t == "mul":
            return self._mul(n[1]), _PREC_MUL
        if t == "pow":
            return self._pow(n[1], n[2]), _PREC_POW
        if t == "func":
            return self._func(n[1], n[2]), _PREC_ATOM
        if t == "piecewise":
            return self._piecewise(n[1], n[2]), _PREC_ATOM
        if t == "predicate":
            return self._predicate(n[1], n[2]), _PREC_ADD
        if t == "big_o":
            inner, _ = self._r(n[1])
            return rf"\mathcal{{O}}\!\left({inner}\right)", _PREC_ATOM
        return str(expr), _PREC_ATOM

    # ---- atoms ----

    def _symbol(self, name: str) -> str:
        if name in _GREEK:
            # Multi-char Greek names → \alpha, \beta, etc.
            return "\\" + name if len(name) > 1 else name
        if "_" in name:
            base, sub = name.split("_", 1)
            return f"{{{base}}}_{{{sub}}}"
        return name

    @staticmethod
    def _frac(num: str, den: str) -> str:
        return rf"\frac{{{num}}}{{{den}}}"

    # ---- paren wrapping ----

    def _wrap(self, expr, req_prec: int) -> str:
        s, prec = self._r(expr)
        if prec < req_prec:
            return r"\left(" + s + r"\right)"
        return s

    # ---- add ----

    def _add(self, args: list) -> str:
        parts: list[str] = []
        for child in args:
            sign, tex = self._signed(child)
            if not parts:
                parts.append(("-" + tex) if sign < 0 else tex)
            else:
                parts.append((" - " + tex) if sign < 0 else (" + " + tex))
        return "".join(parts)

    def _signed(self, expr) -> tuple[int, str]:
        """Return (sign, abs_latex) for a term in a sum."""
        n = expr.node()
        t = n[0]
        if t == "integer":
            v = int(n[1])
            return (-1, str(-v)) if v < 0 else (1, str(v))
        if t == "rational":
            num, den = int(n[1]), int(n[2])
            s = self._frac(str(abs(num)), str(den))
            return (-1, s) if num < 0 else (1, s)
        if t == "mul":
            return self._signed_mul(n[1])
        tex, _ = self._r(expr)
        return (1, tex)

    # ---- mul ----

    def _mul(self, args: list) -> str:
        sign, tex = self._signed_mul(args)
        return ("-" + tex) if sign < 0 else tex

    def _signed_mul(self, args: list) -> tuple[int, str]:
        """Returns (sign, abs_latex_product)."""
        numer_int = 1
        denom_int = 1
        others: list = []

        for child in args:
            n = child.node()
            t = n[0]
            if t == "integer":
                numer_int *= int(n[1])
            elif t == "rational":
                numer_int *= int(n[1])
                denom_int *= int(n[2])
            else:
                others.append(child)

        sign = 1
        if numer_int < 0:
            sign = -1
            numer_int = -numer_int
        if denom_int < 0:
            sign *= -1
            denom_int = -denom_int

        num_parts: list[str] = []
        den_parts: list[str] = []

        for child in others:
            n = child.node()
            if n[0] == "pow":
                base_expr = n[1]
                exp_expr = n[2]
                en = exp_expr.node()
                if en[0] == "integer" and int(en[1]) < 0:
                    exp_abs = -int(en[1])
                    base_tex = self._wrap(base_expr, _PREC_POW + 1)
                    if exp_abs == 1:
                        den_parts.append(base_tex)
                    else:
                        den_parts.append(f"{{{base_tex}}}^{{{exp_abs}}}")
                    continue
            tex = self._wrap(child, _PREC_MUL)
            num_parts.append(tex)

        # Prepend coefficient
        if numer_int != 1 or denom_int != 1:
            coeff = self._frac(str(numer_int), str(denom_int)) if denom_int != 1 else str(numer_int)
            num_parts.insert(0, coeff)

        if not num_parts and not den_parts:
            return sign, "1"

        if den_parts:
            num_str = " ".join(num_parts) if num_parts else "1"
            den_str = " ".join(den_parts)
            return sign, self._frac(num_str, den_str)

        return sign, " ".join(num_parts)

    # ---- pow ----

    def _pow(self, base_expr, exp_expr) -> str:
        en = exp_expr.node()

        # x^(1/n) → nth-root
        if en[0] == "rational":
            num, den = int(en[1]), int(en[2])
            if num == 1 and den >= 2:
                base_tex = self._wrap(base_expr, _PREC_POW + 1)
                if den == 2:
                    return rf"\sqrt{{{base_tex}}}"
                return rf"\sqrt[{den}]{{{base_tex}}}"

        # x^(-1) → 1/x
        if en[0] == "integer" and int(en[1]) == -1:
            base_tex = self._wrap(base_expr, _PREC_POW + 1)
            return self._frac("1", base_tex)

        base_tex = self._wrap(base_expr, _PREC_POW)
        exp_tex, _ = self._r(exp_expr)
        exp_braced = exp_tex if len(exp_tex) == 1 else f"{{{exp_tex}}}"
        return f"{base_tex}^{exp_braced}"

    # ---- func ----

    def _func(self, name: str, args: list) -> str:
        if name == "abs":
            inner, _ = self._r(args[0])
            return rf"\left|{inner}\right|"
        if name == "floor":
            inner, _ = self._r(args[0])
            return rf"\lfloor {inner} \rfloor"
        if name == "ceil":
            inner, _ = self._r(args[0])
            return rf"\lceil {inner} \rceil"
        if name == "sqrt":
            inner, _ = self._r(args[0])
            return rf"\sqrt{{{inner}}}"
        if name == "exp":
            inner, _ = self._r(args[0])
            exp_braced = inner if len(inner) == 1 else f"{{{inner}}}"
            return rf"e^{exp_braced}"

        fn_latex = _LATEX_FUNCS.get(name, rf"\operatorname{{{name}}}")
        rendered_args = ", ".join(self._r(a)[0] for a in args)
        return rf"{fn_latex}\!\left({rendered_args}\right)"

    # ---- piecewise ----

    def _piecewise(self, branches: list, default_expr) -> str:
        rows: list[str] = []
        for cond_expr, val_expr in branches:
            val_tex, _ = self._r(val_expr)
            cond_tex, _ = self._r(cond_expr)
            rows.append(rf"{val_tex} & \text{{if }} {cond_tex}")
        default_tex, _ = self._r(default_expr)
        rows.append(rf"{default_tex} & \text{{otherwise}}")
        body = r" \\ ".join(rows)
        return rf"\begin{{cases}} {body} \end{{cases}}"

    # ---- predicate ----

    def _predicate(self, kind: str, args: list) -> str:
        op = _PREDICATE_LATEX.get(kind, kind)
        if kind in ("true", "false"):
            return op
        if kind == "not":
            inner, _ = self._r(args[0])
            return rf"{op} {inner}"
        rendered = [self._r(a)[0] for a in args]
        return rf" {op} ".join(rendered)


# ---------------------------------------------------------------------------
# Unicode renderer
# ---------------------------------------------------------------------------


class _UnicodeRenderer:
    def render(self, expr) -> str:
        s, _ = self._r(expr)
        return s

    def _r(self, expr) -> tuple[str, int]:
        n = expr.node()
        t = n[0]

        if t == "symbol":
            return self._symbol(n[1]), _PREC_ATOM
        if t == "integer":
            v = int(n[1])
            return (str(-v) if v < 0 else str(v)), _PREC_ATOM
        if t == "rational":
            num, den = int(n[1]), int(n[2])
            s = _unicode_frac(abs(num), den)
            return (("-" + s) if num < 0 else s), _PREC_ATOM
        if t == "float":
            return n[1], _PREC_ATOM
        if t == "add":
            return self._add(n[1]), _PREC_ADD
        if t == "mul":
            return self._mul(n[1]), _PREC_MUL
        if t == "pow":
            return self._pow(n[1], n[2]), _PREC_POW
        if t == "func":
            return self._func(n[1], n[2]), _PREC_ATOM
        if t == "piecewise":
            return self._piecewise(n[1], n[2]), _PREC_ATOM
        if t == "predicate":
            return self._predicate(n[1], n[2]), _PREC_ADD
        if t == "big_o":
            inner, _ = self._r(n[1])
            return f"O({inner})", _PREC_ATOM
        return str(expr), _PREC_ATOM

    def _symbol(self, name: str) -> str:
        return _GREEK.get(name, name)

    def _wrap(self, expr, req_prec: int) -> str:
        s, prec = self._r(expr)
        if prec < req_prec:
            return "(" + s + ")"
        return s

    def _add(self, args: list) -> str:
        parts: list[str] = []
        for child in args:
            sign, tex = self._signed(child)
            if not parts:
                parts.append(("-" + tex) if sign < 0 else tex)
            else:
                parts.append((" - " + tex) if sign < 0 else (" + " + tex))
        return "".join(parts)

    def _signed(self, expr) -> tuple[int, str]:
        n = expr.node()
        t = n[0]
        if t == "integer":
            v = int(n[1])
            return (-1, str(-v)) if v < 0 else (1, str(v))
        if t == "rational":
            num, den = int(n[1]), int(n[2])
            return (-1, _unicode_frac(-num, den)) if num < 0 else (1, _unicode_frac(num, den))
        if t == "mul":
            return self._signed_mul(n[1])
        tex, _ = self._r(expr)
        return (1, tex)

    def _mul(self, args: list) -> str:
        sign, tex = self._signed_mul(args)
        return ("-" + tex) if sign < 0 else tex

    def _signed_mul(self, args: list) -> tuple[int, str]:
        numer_int = 1
        denom_int = 1
        others: list = []

        for child in args:
            n = child.node()
            t = n[0]
            if t == "integer":
                numer_int *= int(n[1])
            elif t == "rational":
                numer_int *= int(n[1])
                denom_int *= int(n[2])
            else:
                others.append(child)

        sign = 1
        if numer_int < 0:
            sign = -1
            numer_int = -numer_int
        if denom_int < 0:
            sign *= -1
            denom_int = -denom_int

        num_parts: list[str] = []
        den_parts: list[str] = []

        for child in others:
            n = child.node()
            if n[0] == "pow":
                base_expr = n[1]
                exp_expr = n[2]
                en = exp_expr.node()
                if en[0] == "integer" and int(en[1]) < 0:
                    exp_abs = -int(en[1])
                    base_tex = self._wrap(base_expr, _PREC_POW + 1)
                    sup = _to_superscript(str(exp_abs))
                    den_parts.append(base_tex + (sup if sup else f"^{exp_abs}"))
                    continue
            tex = self._wrap(child, _PREC_MUL)
            num_parts.append(tex)

        if numer_int != 1 or denom_int != 1:
            coeff = _unicode_frac(numer_int, denom_int) if denom_int != 1 else str(numer_int)
            num_parts.insert(0, coeff)

        if not num_parts and not den_parts:
            return sign, "1"

        if den_parts:
            num_str = "·".join(num_parts) if num_parts else "1"
            den_str = "·".join(den_parts)
            if len(den_parts) > 1:
                return sign, f"({num_str})/({den_str})"
            return sign, f"{num_str}/{den_str}"

        return sign, "·".join(num_parts)

    def _pow(self, base_expr, exp_expr) -> str:
        en = exp_expr.node()

        if en[0] == "rational":
            num, den = int(en[1]), int(en[2])
            if num == 1:
                base_tex = self._wrap(base_expr, _PREC_POW + 1)
                if den == 2:
                    return "√" + base_tex
                if den == 3:
                    return "∛" + base_tex
                if den == 4:
                    return "∜" + base_tex

        base_tex = self._wrap(base_expr, _PREC_POW)

        if en[0] == "integer":
            sup = _to_superscript(str(int(en[1])))
            if sup is not None:
                return base_tex + sup

        exp_tex, _ = self._r(exp_expr)
        return f"{base_tex}^({exp_tex})"

    def _func(self, name: str, args: list) -> str:
        if name == "sqrt":
            inner = self._wrap(args[0], _PREC_POW + 1)
            return "√" + inner
        if name == "abs":
            inner, _ = self._r(args[0])
            return f"|{inner}|"
        if name == "floor":
            inner, _ = self._r(args[0])
            return f"⌊{inner}⌋"
        if name == "ceil":
            inner, _ = self._r(args[0])
            return f"⌈{inner}⌉"
        if name == "exp":
            inner, _ = self._r(args[0])
            sup = _to_superscript(inner)
            if sup is not None:
                return "e" + sup
            return f"e^({inner})"
        fn = _UNICODE_FUNCS.get(name, name)
        rendered_args = ", ".join(self._r(a)[0] for a in args)
        return f"{fn}({rendered_args})"

    def _piecewise(self, branches: list, default_expr) -> str:
        rows: list[str] = []
        for cond_expr, val_expr in branches:
            val_tex, _ = self._r(val_expr)
            cond_tex, _ = self._r(cond_expr)
            rows.append(f"{val_tex}  if {cond_tex}")
        default_tex, _ = self._r(default_expr)
        rows.append(f"{default_tex}  otherwise")
        return "{ " + "\n  ".join(rows)

    def _predicate(self, kind: str, args: list) -> str:
        op = _PREDICATE_UNICODE.get(kind, kind)
        if kind in ("true", "false"):
            return op
        if kind == "not":
            inner, _ = self._r(args[0])
            return f"{op}{inner}"
        rendered = [self._r(a)[0] for a in args]
        return f" {op} ".join(rendered)


# ---------------------------------------------------------------------------
# Module-level singletons and public API
# ---------------------------------------------------------------------------

_LATEX_RENDERER = _LatexRenderer()
_UNICODE_RENDERER = _UnicodeRenderer()


def latex(expr) -> str:
    """Return a LaTeX string for the given :class:`~alkahest.Expr`.

    The result is suitable for ``$...$`` in a LaTeX document or
    ``IPython.display.Math(alkahest.latex(expr))`` in a Jupyter notebook.

    Example::

        >>> import alkahest
        >>> p = alkahest.ExprPool()
        >>> x = p.symbol("x")
        >>> alkahest.latex(alkahest.sin(x)**2 + alkahest.cos(x)**2)
        '\\\\sin\\\\!\\\\left(x\\\\right)^{2} + \\\\cos\\\\!\\\\left(x\\\\right)^{2}'
    """
    return _LATEX_RENDERER.render(expr)


def unicode_str(expr) -> str:
    """Return a Unicode pretty-printed string for the given :class:`~alkahest.Expr`.

    Uses Unicode superscripts, Greek letters, fraction characters, and root
    symbols for a human-readable representation without LaTeX markup.

    Example::

        >>> import alkahest
        >>> p = alkahest.ExprPool()
        >>> x = p.symbol("x")
        >>> alkahest.unicode_str(x**2 + p.integer(1))
        'x² + 1'
    """
    return _UNICODE_RENDERER.render(expr)
