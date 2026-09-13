"""Silent-error cases for printing.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import re
from typing import Callable

import alkahest as ak
from contracts import Case, Returns

from ._shared import POOL, N, X, Y, _int

_PRINT_SYMS = {"x": X, "y": Y, "n": N}
_PRINT_A = POOL.lt(X, _int(0))
_PRINT_B = POOL.gt(X, _int(1))
_PRINT_C = POOL.pred_eq(X, _int(2))
_PRINT_OR_IN_AND = ak.latex(POOL.pred_and([POOL.pred_or([_PRINT_A, _PRINT_B]), _PRINT_C]))
_PRINT_AND_IN_OR = ak.latex(POOL.pred_or([_PRINT_A, POOL.pred_and([_PRINT_B, _PRINT_C])]))


def _latex_units(tex: str) -> list[str]:
    """Split emitted LaTeX into ASCII tokens, one per source token.

    Only the subset these cases emit is handled — `\\cdot`, `\\times`,
    `\\left(`, `\\right)`, `^`, and literals.  Anything else raises, so a case
    whose rendering grows a new construct is reported rather than quietly
    passing.
    """
    units: list[str] = []
    i, n = 0, len(tex)
    while i < n:
        c = tex[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            j = i + 1
            while j < n and tex[j].isalpha():
                j += 1
            name, k = (tex[i + 1 : j], j) if j > i + 1 else (tex[i + 1 : i + 2], i + 2)
            if name in {"!", ",", ";", " "}:
                pass
            elif name in {"cdot", "times"}:
                units.append("*")
            elif name == "left" and tex[k : k + 1] == "(":
                units.append("(")
                k += 1
            elif name == "right" and tex[k : k + 1] == ")":
                units.append(")")
                k += 1
            else:
                raise ValueError(f"unhandled LaTeX macro {name!r} in {tex!r}")
            i = k
            continue
        if c == "^":
            # `^` scopes to exactly one token: `x^c y` is `(x^c)·y`.
            if tex[i + 1 : i + 2] == "{":
                depth, j = 1, i + 2
                while j < n and depth:
                    depth += (tex[j] == "{") - (tex[j] == "}")
                    j += 1
                if depth:
                    raise ValueError(f"unbalanced braces in {tex!r}")
                body, i = tex[i + 2 : j - 1], j
            else:
                body, i = tex[i + 1 : i + 2], i + 2
            units.append("^(" + "".join(_latex_units(body)) + ")")
            continue
        if c.isdigit():
            j = i
            while j < n and (tex[j].isdigit() or tex[j] == "."):
                j += 1
            units.append(tex[i:j])
            i = j
            continue
        units.append(c)
        i += 1
    return units


def _latex_as_read(tex: str) -> str:
    """Emitted LaTeX, as the page reads it.

    Math mode discards white space (Knuth, *The TeXbook*, ch. 8), so juxtaposed
    tokens fuse: `2 3` sets as `23`, `x 2` as `x2`.  Every other juxtaposition
    is a product.  Modelling the space as a multiplication instead would assume
    away exactly the defect this is looking for.
    """
    out = ""
    for unit in _latex_units(tex):
        if out:
            prev, here = out[-1], unit[0]
            kind = lambda ch: (  # noqa: E731 - three-line local classifier
                "d" if ch.isdigit() or ch == "." else "l" if ch.isalpha() else ch
            )
            a, b = kind(prev), kind(here)
            fuses = (a, b) in {("d", "d"), ("l", "d")}
            juxtaposed = a in {"d", "l", ")"} and b in {"d", "l", "("}
            if juxtaposed and not fuses:
                out += "*"
        out += unit
    return out


def latex_value(expr: ak.Expr, env: dict) -> Callable[[], float]:
    """Answer = the value of alkahest's own LaTeX for *expr*, read back."""

    def op() -> float:
        source = _latex_as_read(ak.latex(expr))
        return float(ak.eval_expr(ak.parse(source, POOL, dict(_PRINT_SYMS)), env))

    return op


def printed_value(expr: ak.Expr, env: dict) -> Callable[[], float]:
    """Answer = the value of ``parse(str(expr))`` — the plain-text round trip."""

    def op() -> float:
        return float(ak.eval_expr(ak.parse(str(expr), POOL, dict(_PRINT_SYMS)), env))

    return op


def _has_double_superscript(tex: str) -> bool:
    """`x^a^b`: ambiguous to a reader, and the TeX error "Double superscript"."""
    i, n = 0, len(tex)
    while i < n:
        if tex[i] != "^":
            i += 1
            continue
        if tex[i + 1 : i + 2] == "{":
            depth, j = 1, i + 2
            while j < n and depth:
                depth += (tex[j] == "{") - (tex[j] == "}")
                j += 1
            i = j
        else:
            i += 2
        while tex[i : i + 1] == " ":
            i += 1
        if tex[i : i + 1] == "^":
            return True
    return False


def _superscript_count(rendered: str) -> int:
    """How many Unicode superscript runs the rendering carries."""
    runs, previous = 0, False
    for ch in rendered:
        here = ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
        runs += here and not previous
        previous = here
    return runs


def _bar_nesting_depth(rendered: str) -> int:
    """Nesting depth of ``|…|`` — or ``-1`` when the bars cannot be paired.

    ``|`` is its own mirror, so ``||2||`` has no unique reading; the printer has
    to parenthesise the inner one for the outer pair to be recoverable.
    """

    def parse(s: str) -> tuple[int, str]:
        if not s.startswith("|"):
            raise ValueError(f"not an absolute value: {s!r}")
        body = s[1:]
        if body.startswith("("):
            depth, j = 1, 1
            while j < len(body) and depth:
                depth += (body[j] == "(") - (body[j] == ")")
                j += 1
            if depth or body[j : j + 1] != "|":
                raise ValueError(f"unpaired bars in {s!r}")
            return 1 + parse(body[1 : j - 1])[0], body[j + 1 :]
        end = body.find("|")
        if end < 0:
            raise ValueError(f"unpaired bars in {s!r}")
        inner = body[:end]
        if "|" in inner:
            raise ValueError(f"ambiguous nested bars in {s!r}")
        return 1, body[end + 1 :]

    try:
        depth, rest = parse(rendered)
    except ValueError:
        return -1
    return depth if not rest else -1


def _operatorname_underscores_are_escaped(tex: str) -> bool:
    """Every ``_`` inside an ``\\operatorname{…}`` must be escaped."""
    for match in re.finditer(r"\\operatorname\{([^}]*)\}", tex):
        body = match.group(1)
        if any(body[k] == "_" and (k == 0 or body[k - 1] != "\\") for k in range(len(body))):
            return False
    return True


def _grouping_depth(tex: str, needle: str) -> int:
    """Parenthesis depth at which *needle* appears in *tex*."""
    index = tex.index(needle)
    prefix = tex[:index]
    return prefix.count(r"\left") - prefix.count(r"\right")


CASES: list[Case] = [
    Case(
        id="printing_latex_repeated_factor_keeps_its_power",
        subsystem="printing",
        statement="latex(x·x), read back, is x² — at x = 3 it is 9",
        op=latex_value(POOL.mul([X, X]), {X: 3.0}),
        contract=Returns(9.0),
        verified_by="x·x = x² by the definition of a power; at x = 3 both are 9. LaTeX math "
        "mode discards white space (Knuth, The TeXbook, ch. 8), so a product printed "
        "as `x x` sets as `xx` — one two-letter identifier, not a power.",
    ),
    Case(
        id="printing_control_latex_distinct_factors_are_a_product",
        subsystem="printing",
        statement="latex(x·y), read back, is a product — at (3, 5) it is 15",
        op=latex_value(POOL.mul([X, Y]), {X: 3.0, Y: 5.0}),
        contract=Returns(15.0),
        verified_by="3·5 = 15. The control for the case above: single italic letters "
        "juxtaposed are the conventional spelling of a product (ISO 80000-2), so "
        "collapsing repeated factors must not turn every product into a power.",
    ),
    Case(
        id="printing_latex_numeral_before_a_numeral_needs_cdot",
        subsystem="printing",
        statement="latex(2·3ⁿ), read back at n = 3, is 54 — not 23³",
        op=latex_value(POOL.mul([POOL.integer(2), POOL.integer(3) ** N]), {N: 3.0}),
        contract=Returns(54.0),
        verified_by="2·3³ = 54, while 23³ = 12167. Typeset without a separator the two "
        "numerals fuse (The TeXbook, ch. 8); SymPy's LaTeX printer inserts `\\cdot` "
        "between adjacent numerals for exactly this reason "
        "(sympy/printing/latex.py, `_between_two_numbers_p`).",
    ),
    Case(
        id="printing_control_latex_coefficient_before_a_symbol",
        subsystem="printing",
        statement="latex(2·x), read back at x = 3, is 6",
        op=latex_value(POOL.mul([POOL.integer(2), X]), {X: 3.0}),
        contract=Returns(6.0),
        verified_by="2·3 = 6. The control for the case above: `2x` is the standard "
        "spelling of a coefficient times a symbol (ISO 80000-2), and it has to go "
        "on reading as a product — a fix that made *this* juxtaposition fuse "
        "would be worse than the defect it was fixing.",
    ),
    Case(
        id="printing_latex_negative_float_factor_is_not_a_subtraction",
        subsystem="printing",
        statement="latex(x·(−1.5)), read back at x = 2, is −3 — not x − 1.5",
        op=latex_value(POOL.mul([X, POOL.float(-1.5)]), {X: 2.0}),
        contract=Returns(-3.0),
        verified_by="2·(−1.5) = −3, while 2 − 1.5 = 0.5. A factor emitted with its leading "
        "minus sign exposed is read as a binary minus: juxtaposition and "
        "subtraction are not distinguishable once the sign is bare.",
    ),
    Case(
        id="printing_control_latex_positive_float_factor",
        subsystem="printing",
        statement="latex(x·1.5), read back at x = 2, is 3",
        op=latex_value(POOL.mul([X, POOL.float(1.5)]), {X: 2.0}),
        contract=Returns(3.0),
        verified_by="2·1.5 = 3. The control for the case above: a float factor with no "
        "sign of its own still has to read as a product.",
    ),
    Case(
        id="printing_latex_float_exponent_is_a_power_of_ten",
        subsystem="printing",
        statement="latex(0.25·x), read back at x = 4, is 1",
        op=latex_value(POOL.mul([POOL.float(0.25), X]), {X: 4.0}),
        contract=Returns(1.0),
        verified_by="0.25·4 = 1, and 0.25 = 2.5×10⁻¹. `rug::Float` prints 0.25 as `2.5e-1`; "
        "copied verbatim into math mode that reads as 2.5·e − 1, a subtraction "
        "against Euler's number, so the exponent has to be typeset as a power of "
        "ten (ISO 80000-1 §7.3.2).",
    ),
    Case(
        id="printing_latex_exp_base_is_not_a_double_superscript",
        subsystem="printing",
        statement="latex((eˣ)²) carries no `^…^` — TeX rejects a doubled superscript",
        op=lambda: _has_double_superscript(ak.latex(ak.exp(X) ** 2)),
        contract=Returns(False),
        verified_by="`e^{x}^2` is the TeX error 'Double superscript' (Knuth, The TeXbook, "
        "ch. 7: an atom may carry only one superscript), so the document does not "
        "compile at all; and read as written it would be e^(x²), which at x = 3 is "
        "e⁹ rather than e⁶.",
    ),
    Case(
        id="printing_control_latex_nested_exponent_is_braced",
        subsystem="printing",
        statement="latex(x^(y^n)) is a *nested* superscript, which is legal",
        op=lambda: _has_double_superscript(ak.latex(X ** (Y**N))),
        contract=Returns(False),
        verified_by="`x^{y^n}` puts the second superscript inside the first one's group, "
        "which TeX accepts (The TeXbook, ch. 7). The control: the check above must "
        "not fire on every expression that has two superscripts in it.",
    ),
    Case(
        id="printing_unicode_reciprocal_has_no_leftover_exponent",
        subsystem="printing",
        statement="unicode_str(1/(x+1)) carries no superscript",
        op=lambda: _superscript_count(ak.unicode_str(_int(1) / (X + _int(1)))),
        contract=Returns(0),
        verified_by="x⁻¹ is 1/x: the exponent is spent by writing the quotient, so nothing "
        "may remain. SymPy's pretty printer renders this as `1/(x + 1)`.",
    ),
    Case(
        id="printing_control_unicode_squared_denominator_keeps_its_exponent",
        subsystem="printing",
        statement="unicode_str(1/(x+1)²) does carry one superscript",
        op=lambda: _superscript_count(ak.unicode_str(_int(1) / (X + _int(1)) ** 2)),
        contract=Returns(1),
        verified_by="1/(x+1)² has an exponent that the quotient does *not* absorb — only one "
        "of the two powers is spent. The control for the case above: suppressing "
        "every exponent in a denominator would be a worse error than printing a "
        "redundant one.",
    ),
    Case(
        id="printing_str_negative_base_power_round_trips",
        subsystem="printing",
        statement="parse(str((−1)ⁿ)) at n = 2 is +1 — `-1^n` would give −1",
        op=printed_value(POOL.integer(-1) ** N, {N: 2.0}),
        contract=Returns(1.0),
        verified_by="(−1)² = +1 while −(1²) = −1. Unary minus binds looser than `^` in "
        "Python (docs.python.org, 6.5: '-1**2 is -1'), in sympy and in every "
        "parser that reads these strings, so only `(-1)^n` round-trips.",
    ),
    Case(
        id="printing_control_str_positive_base_power_round_trips",
        subsystem="printing",
        statement="parse(str(2ⁿ)) at n = 3 is 8, with no parentheses added",
        op=printed_value(POOL.integer(2) ** N, {N: 3.0}),
        contract=Returns(8.0),
        verified_by="2³ = 8. The control for the case above: a non-negative base has to "
        "keep round-tripping, so the fix cannot be 'parenthesise nothing' and "
        "cannot break the ordinary shape either.",
    ),
    Case(
        id="printing_latex_disjunction_inside_conjunction_is_grouped",
        subsystem="printing",
        statement="latex((a ∨ b) ∧ c) puts the ∨ one group deeper than the ∧",
        op=lambda: (
            _grouping_depth(_PRINT_OR_IN_AND, r"\lor") - _grouping_depth(_PRINT_OR_IN_AND, r"\land")
        ),
        contract=Returns(1),
        verified_by="∧ binds tighter than ∨ (Enderton, A Mathematical Introduction to Logic, "
        "§1.2), so `a ∨ b ∧ c` *is* a ∨ (b ∧ c). Printing (a ∨ b) ∧ c without the "
        "group states a different proposition: at a = ⊤, b = ⊥, c = ⊥ the first is "
        "⊥ and the second is ⊤.",
    ),
    Case(
        id="printing_control_latex_conjunction_inside_disjunction_is_not_grouped",
        subsystem="printing",
        statement="latex(a ∨ (b ∧ c)) needs no group, and gets none",
        op=lambda: (
            _grouping_depth(_PRINT_AND_IN_OR, r"\lor") - _grouping_depth(_PRINT_AND_IN_OR, r"\land")
        ),
        contract=Returns(0),
        verified_by="a ∨ (b ∧ c) is exactly what `a ∨ b ∧ c` means under the standard "
        "precedence (Enderton §1.2). The control: parenthesising every connective "
        "would satisfy the case above while making the output unreadable.",
    ),
    Case(
        id="printing_latex_operatorname_escapes_its_underscore",
        subsystem="printing",
        statement="latex of an unregistered head `my_func` escapes the underscore",
        op=lambda: _operatorname_underscores_are_escaped(ak.latex(POOL.func("my_func", [X]))),
        contract=Returns(True),
        verified_by="A bare `_` in math mode is a subscript, so `\\operatorname{my_func}` sets "
        "as my_func with a subscripted f, and a second underscore is the hard error "
        "'Double subscript' (Knuth, The TeXbook, ch. 7).",
    ),
    Case(
        id="printing_control_latex_symbol_subscript_keeps_its_underscore",
        subsystem="printing",
        statement="latex(u_0) still uses a real subscript, unescaped",
        op=lambda: (
            "_" in ak.latex(POOL.symbol("u_0")) and "\\_" not in ak.latex(POOL.symbol("u_0"))
        ),
        contract=Returns(True),
        verified_by="`u_0` names a subscripted variable, and `{u}_{0}` is how that is "
        "written (The TeXbook, ch. 7). The control for the case above: escaping "
        "every underscore would turn subscripted symbols into literal text.",
    ),
    Case(
        id="printing_unicode_nested_absolute_value_is_pairable",
        subsystem="printing",
        statement="unicode_str(||x||) is printed so the two bar pairs can be told apart",
        op=lambda: _bar_nesting_depth(ak.unicode_str(ak.abs(ak.abs(X)))),
        contract=Returns(2),
        verified_by="U+007C is its own mirror — it is not a bracket pair (Unicode 15.0, "
        "table 4-2, Bidi_Paired_Bracket_Type=None) — so `||x||` has no unique "
        "reading; it is equally the norm ‖x‖. The inner pair has to be grouped.",
    ),
    Case(
        id="printing_control_unicode_absolute_value_stays_plain",
        subsystem="printing",
        statement="unicode_str(|x|) is one bar pair, ungrouped",
        op=lambda: _bar_nesting_depth(ak.unicode_str(ak.abs(X))),
        contract=Returns(1),
        verified_by="|x| is unambiguous on its own. The control for the case above: adding "
        "the group unconditionally would print `|(x)|` for every absolute value.",
    ),
]
