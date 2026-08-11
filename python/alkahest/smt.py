"""SMT/SAT bridge — hand a discrete or mixed subproblem to an external solver.

P2 item 3 — see ``docs/mdbook/src/smt.md``.

Discrete and mixed integer/real/boolean problems are not Alkahest's problem
class and should not become one.  What a search loop needs is not an in-tree
solver but a way to *hand the subproblem off* — and to bring the answer back in
a form the rest of the toolchain can trust.  This module is that hand-off:

``to_smtlib``
    Re-exported from the Rust emitter (``alkahest_core::logic::smtlib``).  Turns
    a predicate :class:`~alkahest.Expr` into a complete SMT-LIB 2 script.  The
    emitter is exhaustive over ``Formula`` and ``PredicateKind`` by construction
    — `rustc` refuses to compile a missing case — so a node added later cannot
    silently emit wrong SMT-LIB.

:func:`supported`
    The plan-ahead predicate, in the spirit of :func:`alkahest.certifiable`: can
    this formula be sent at all, is a solver installed, and *should* it be sent
    rather than kept in-tree.

:func:`solve`
    Runs the solver in a subprocess and returns an :class:`SmtResult`.

Two asymmetries drive the whole design and are worth stating up front.

**`sat` and `unsat` are not equally trustworthy.**  A `sat` model is checkable
inside Alkahest for free: substitute it back and evaluate the formula exactly.
:func:`solve` always does this — it is not optional and there is no flag to skip
it — and a model that fails raises :class:`~alkahest.SmtError` ``E-SMT-004``
rather than warning, because a failure there means the bridge or the solver is
broken.  `unsat`, by contrast, is not machine-checked here at all: consuming an
unsat proof is a large project with unstable formats.  So an `unsat` result gets
the status :data:`EXTERNALLY_ASSERTED`, which is deliberately **not** in
``alkahest.research.MACHINE_CHECKED_STATUSES``.  "z3 said so" is a fact about
z3, not a proof, and the claim graph must keep being able to tell the two apart.

**Exactness is where a model reader breaks.**  Rationals lift cleanly; algebraic
numbers (z3's ``root-obj``) do not, and the tempting move — round to a float —
would record an approximation as an exact witness.  That is the precise silent
error this subsystem exists to prevent, so ``root-obj`` is refused with
``E-SMT-003`` until it can be lifted into the real-algebraic machinery
(``RootInterval`` / ``refine_root``) that already exists for it.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from .exceptions import SmtError

__all__ = [
    "EXTERNALLY_ASSERTED",
    "SOLVERS",
    "STATUS_BADGES",
    "SmtResult",
    "SmtSupport",
    "solve",
    "solvers",
    "supported",
    "to_smtlib",
]

# ---------------------------------------------------------------------------
# Verification vocabulary
# ---------------------------------------------------------------------------

#: Status for a result an external tool asserted and nothing in-process checked.
#:
#: Introduced by this module for `unsat`.  It is **not** a member of
#: ``alkahest.research.MACHINE_CHECKED_STATUSES`` — that set means a checker
#: actually ran, and widening it to include an unverified external assertion
#: would erode the one guarantee ``research.py`` makes.  ``tests/test_smt.py``
#: pins that exclusion.
EXTERNALLY_ASSERTED = "externally_asserted"

#: Badges for the statuses this module produces, worded as unflatteringly as the
#: ones in ``alkahest.research.STATUS_BADGES``.
STATUS_BADGES: dict[str, str] = {
    "exactly_verified": (
        "the solver's model was substituted back and the kernel evaluated the formula "
        "to true exactly"
    ),
    EXTERNALLY_ASSERTED: (
        "an external solver asserted this; NO proof was checked and nothing in Alkahest verified it"
    ),
    "unverified": "the solver returned no verdict; nothing was established either way",
}

# ---------------------------------------------------------------------------
# Solver registry and discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SolverSpec:
    """How to invoke one SMT-LIB 2 solver."""

    name: str
    binary: str
    version_args: tuple[str, ...]
    script_args: tuple[str, ...]
    #: ``str.format``-style template taking ``ms``; ``None`` if the solver has
    #: no wall-clock flag (the parent-side deadline still applies).
    timeout_flag: str | None


_SPECS: dict[str, _SolverSpec] = {
    "z3": _SolverSpec(
        name="z3",
        binary="z3",
        version_args=("--version",),
        script_args=("-smt2",),
        timeout_flag="-t:{ms}",
    ),
    "cvc5": _SolverSpec(
        name="cvc5",
        binary="cvc5",
        version_args=("--version",),
        script_args=("--lang=smt2",),
        timeout_flag="--tlimit={ms}",
    ),
}

#: Solver names this module knows how to drive, in preference order.
SOLVERS: tuple[str, ...] = tuple(_SPECS)

# Version strings are cached per resolved path (a given binary's version does
# not change under us).  Discovery itself is *not* cached, so a solver that
# appears on PATH mid-session is seen.
_VERSION_CACHE: dict[str, str] = {}


def _search_path() -> str:
    """``PATH`` plus the running interpreter's script directory.

    A ``pip install z3-solver`` drops the ``z3`` binary in the environment's
    script directory, which is on ``PATH`` only when the environment has been
    *activated*.  A loop that runs ``.venv/bin/python`` directly would otherwise
    report "no solver installed" while one sits right next to the interpreter.
    """
    parts = [os.environ.get("PATH", os.defpath)]
    for extra in (sysconfig.get_path("scripts"), os.path.dirname(sys.executable)):
        if extra and extra not in parts:
            parts.append(extra)
    return os.pathsep.join(p for p in parts if p)


def _find_binary(binary: str) -> str | None:
    return shutil.which(binary, path=_search_path())


def _version_of(path: str, spec: _SolverSpec) -> str:
    cached = _VERSION_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [path, *spec.version_args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        line = (proc.stdout or proc.stderr or "").strip().splitlines()
        version = line[0].strip() if line else "unknown"
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    _VERSION_CACHE[path] = version
    return version


def solvers() -> dict[str, str | None]:
    """Which SMT solvers are installed, and at what version.

    Returns a mapping over every name in :data:`SOLVERS`; the value is the
    solver's self-reported version string, or ``None`` when the binary was not
    found.  Absence is reported **negatively and explicitly** so an agent can
    tell before it plans that the hand-off is unavailable, rather than
    discovering it as a refusal mid-loop.

    Examples
    --------
    >>> import alkahest as ak
    >>> ak.smt.solvers()                      # doctest: +SKIP
    {'z3': 'Z3 version 4.13.0 - 64 bit', 'cvc5': None}
    """
    out: dict[str, str | None] = {}
    for name, spec in _SPECS.items():
        path = _find_binary(spec.binary)
        out[name] = _version_of(path, spec) if path else None
    return out


def _resolve_solver(solver: str) -> tuple[_SolverSpec, str]:
    """Return ``(spec, path)`` or raise ``E-SMT-001``."""
    if solver == "auto":
        wanted = list(_SPECS.values())
    else:
        spec = _SPECS.get(solver)
        if spec is None:
            raise SmtError(
                f"[E-SMT-001] unknown solver {solver!r}; known solvers are {list(SOLVERS)}",
                code="E-SMT-001",
                remediation=(
                    "pass solver='auto' to use the first installed solver, or one of "
                    f"{list(SOLVERS)}"
                ),
            )
        wanted = [spec]
    for spec in wanted:
        path = _find_binary(spec.binary)
        if path:
            return spec, path
    names = [s.binary for s in wanted]
    raise SmtError(
        f"[E-SMT-001] no SMT solver binary found (looked for {names} on PATH and "
        f"{os.path.dirname(sys.executable)!r})",
        code="E-SMT-001",
        remediation=(
            "install z3 (`pip install z3-solver`) or cvc5 and make it visible on PATH. "
            "This is a refusal, not a fallback: alkahest.satisfiable is an interval "
            "heuristic that answers Unknown for almost everything a solver would settle, "
            "so silently routing to it would look like the solver had run and found "
            "nothing"
        ),
    )


# ---------------------------------------------------------------------------
# to_smtlib — the native emitter, with the error type callers expect
# ---------------------------------------------------------------------------


def _native() -> Any:
    from . import alkahest as _alkahest_native

    return _alkahest_native


def _as_smt_error(exc: BaseException) -> SmtError:
    """Re-raise a native ``E-SMT-*`` failure as :class:`~alkahest.SmtError`.

    The native emitter raises the base ``AlkahestError`` class rather than a
    dedicated PyO3 ``SmtError``: ``SmtError`` is a pure-Python class and is not
    in ``alkahest.__init__._NATIVE_EXCEPTION_OVERLAY``, so a native one would be
    a *different* class from the ``ak.SmtError`` callers write ``except`` for.
    """
    return SmtError(
        str(exc),
        code=str(getattr(exc, "code", "E-SMT-002")),
        remediation=getattr(exc, "remediation", None),
    )


def to_smtlib(
    formula: Any,
    logic: str = "auto",
    *,
    check_sat: bool = True,
    get_model: bool = True,
) -> str:
    """Export a predicate :class:`~alkahest.Expr` as an SMT-LIB 2 script.

    The SMT counterpart of :func:`alkahest.to_lean`: Alkahest emits a standard
    artifact and an independently maintained external tool consumes it.  The
    emitter itself lives in Rust (``alkahest_core::logic::smtlib``) so that
    match exhaustiveness over ``Formula`` / ``PredicateKind`` enforces total
    coverage; this wrapper only maps the error type.

    Parameters
    ----------
    formula : Expr
        A predicate or quantified expression — built with ``pool.gt``/``lt``/…
        and :func:`alkahest.And` / :func:`alkahest.Or` / :func:`alkahest.Not`,
        or :func:`alkahest.Forall` / :func:`alkahest.Exists`.
    logic : str
        ``"auto"`` (default) infers the weakest logic that fits: ``QF_LIA``,
        ``QF_NIA``, ``QF_LRA``, ``QF_NRA``, ``QF_LIRA``, ``QF_NIRA``, their
        quantified counterparts, or ``ALL``.  Naming a logic too weak for the
        formula is an error (``E-SMT-002``), not a silent downgrade.
    check_sat, get_model : bool
        Append ``(check-sat)`` / ``(get-model)`` and set ``:produce-models``.

    Raises
    ------
    SmtError
        ``E-SMT-002`` when the formula is outside the exportable fragment.  Use
        :func:`supported` to ask the same question without raising.

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> print(ak.to_smtlib(ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(3)))))
    ... # doctest: +SKIP
    ; alkahest SMT-LIB 2 export
    (set-logic QF_LRA)
    ...
    """
    try:
        return _native().to_smtlib(formula, logic, check_sat=check_sat, get_model=get_model)
    except Exception as exc:
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code.startswith("E-SMT-"):
            raise _as_smt_error(exc) from None
        raise


# ---------------------------------------------------------------------------
# supported() — the plan-ahead predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmtSupport:
    """The answer to "can I hand this formula to a solver, and should I?".

    Truthy exactly when :func:`solve` would run, so it drops into
    ``if supported(f):`` — but it carries the *reason*, which is what a
    planning loop actually needs.  This mirrors
    :class:`alkahest._certificates.Certifiability`.

    Attributes
    ----------
    supported : bool
        The verdict: exportable, quantifier-free, and a solver is installed.
        ``bool(self)`` is the same value.
    exportable : bool
        :func:`to_smtlib` would succeed.  Independent of solver installation,
        so an emitter-only workflow can rely on it.
    quantified : bool
        The formula contains ``Forall``/``Exists``.  :func:`to_smtlib` handles
        these; :func:`solve` refuses them, because it guarantees exact
        in-process model checking and a quantified model has nothing to check.
    solver : str or None
        The solver :func:`solve` would use, or ``None`` if none is installed.
    logic : str or None
        The SMT-LIB logic the emitter chose, e.g. ``"QF_NRA"``.
    reason : str
        Stable reason code: ``"ok"``, ``"outside_fragment"``, ``"quantified"``,
        or ``"no_solver"``.
    detail : str
        One-sentence human explanation.
    recommendation : str
        ``"smt"`` or ``"prefer_in_tree"`` — see :func:`supported`.
    script : str or None
        The emitted script, handed back so a caller that goes on to use it does
        not pay for the export twice.
    error : SmtError or None
        The refusal, when ``exportable`` is ``False``.
    """

    supported: bool
    exportable: bool
    quantified: bool
    solver: str | None
    logic: str | None
    reason: str
    detail: str
    recommendation: str
    script: str | None = None
    error: SmtError | None = None

    def __bool__(self) -> bool:
        return self.supported


_LOGIC_RE = re.compile(r"^\(set-logic\s+(\S+?)\)\s*$", re.MULTILINE)

#: Logics for which the in-tree certified route is preferred over SMT.
_IN_TREE_LOGICS = frozenset({"QF_LRA", "QF_NRA", "LRA", "NRA"})

_PREFER_IN_TREE_DETAIL = (
    "this is real arithmetic with no integer variables, where the in-tree route is "
    "strictly better evidence: sos_decompose / prove_nonneg return a PositivityCertificate "
    "that composes with to_lean, and decide is complete. z3's nlsat returns an answer and "
    "no artifact. Use SMT here as a fallback when the in-tree route refuses or exceeds "
    "its budget"
)


def _walk_exprs(obj: Any):
    """Every :class:`~alkahest.Expr` reachable from ``obj``, including itself.

    Structural rather than tag-driven on purpose: it stays total when a node
    kind is added to ``Expr.node()``.
    """
    if hasattr(obj, "node") and hasattr(obj, "node_tag"):
        yield obj
        for child in obj.node()[1:]:
            yield from _walk_exprs(child)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_exprs(item)


def _formula_symbols(formula: Any) -> dict[str, Any]:
    """Symbol name → the :class:`~alkahest.Expr` for it, for back-substitution."""
    out: dict[str, Any] = {}
    for expr in _walk_exprs(formula):
        node = expr.node()
        if node[0] == "symbol":
            out.setdefault(node[1], expr)
    return out


def _is_quantified(formula: Any) -> bool:
    return any(expr.node()[0] in ("forall", "exists") for expr in _walk_exprs(formula))


#: Probe assignments used to decide whether a formula can be checked exactly.
#: A fixed list rather than a sample, so the verdict is deterministic; several
#: points because a single one can land on a pole (``1/x`` at ``0``) and say
#: "unsupported" about a formula that is perfectly evaluable elsewhere.
_PROBE_POINTS: tuple[Fraction, ...] = (
    Fraction(1),
    Fraction(2),
    Fraction(-1),
    Fraction(1, 2),
    Fraction(0),
    Fraction(7, 3),
)


def _exactly_checkable(formula: Any) -> tuple[bool, str | None]:
    """Can the kernel evaluate this formula exactly at a rational point?

    :func:`solve` guarantees that every ``sat`` model is checked exactly, so a
    formula it cannot evaluate is refused **before** the solver runs rather than
    after — a refusal that arrives only once an answer is in hand costs the
    caller the whole solver run and reads like a bug in the solver.

    Derived from the kernel by probing rather than from a hand-maintained list
    of evaluable heads, so it cannot drift away from what ``evaluate`` actually
    supports.
    """
    from . import evaluate as _evaluate

    symbols = _formula_symbols(formula)
    reason: str | None = None
    for point in _PROBE_POINTS:
        result = _evaluate(formula, dict.fromkeys(symbols.values(), point), mode="exact")
        if result.status == "ok":
            return True, None
        reason = result.reason or reason
    return False, reason


def _refuse_uncheckable(reason: str | None) -> SmtError:
    return SmtError(
        "[E-SMT-002] solve() cannot check a model for this formula: the kernel's exact "
        f"evaluator refused it at every probe point ({reason})",
        code="E-SMT-002",
        remediation=(
            "solve() only answers about formulas whose sat models it can verify exactly "
            "in-process, and this one contains a head the exact evaluator does not "
            "implement. Restrict the formula to polynomial (in)equalities (Piecewise is "
            "fine), or use alkahest.to_smtlib to export it and take the solver's word for "
            "the answer yourself"
        ),
    )


def supported(formula: Any, *, solver: str = "auto") -> SmtSupport:
    """Can this formula be handed to an SMT solver — and should it be?

    The plan-ahead predicate, for the same reason
    :func:`alkahest.certifiable` exists: a loop must be able to choose a route
    *before* it commits, not discover a refusal after paying for the setup.

    The "should it be" half matters and cuts the opposite way to the usual
    instinct.  For real arithmetic with no integer variables (``QF_LRA`` /
    ``QF_NRA``), ``recommendation`` is ``"prefer_in_tree"``:
    :func:`alkahest.prove_nonneg` and :func:`alkahest.sos_decompose` yield a
    ``PositivityCertificate`` that composes with :func:`alkahest.to_lean`,
    whereas an SMT solver yields an answer and no artifact.  The genuinely new
    capability the bridge buys is **mixed integer/real/boolean** problems, which
    neither CAD nor :func:`alkahest.diophantine` handles — there
    ``recommendation`` is ``"smt"``.

    Returns
    -------
    SmtSupport
        Truthy when :func:`solve` would run.

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> n = pool.symbol("n", "integer")
    >>> s = ak.smt.supported(pool.gt(n * n, pool.integer(10)))
    >>> s.recommendation                                      # doctest: +SKIP
    'smt'
    """
    quantified = _is_quantified(formula)
    script: str | None = None
    logic: str | None = None
    error: SmtError | None = None
    try:
        script = to_smtlib(formula)
    except SmtError as exc:
        error = exc
    if script is not None:
        match = _LOGIC_RE.search(script)
        logic = match.group(1) if match else None

    solver_name: str | None = None
    try:
        spec, _path = _resolve_solver(solver)
        solver_name = spec.name
    except SmtError:
        solver_name = None

    recommendation = "prefer_in_tree" if logic in _IN_TREE_LOGICS else "smt"

    if error is not None:
        return SmtSupport(
            supported=False,
            exportable=False,
            quantified=quantified,
            solver=solver_name,
            logic=None,
            reason="outside_fragment",
            detail=str(error),
            recommendation=recommendation,
            error=error,
        )
    if quantified:
        return SmtSupport(
            supported=False,
            exportable=True,
            quantified=True,
            solver=solver_name,
            logic=logic,
            reason="quantified",
            detail=(
                "to_smtlib exports this, but solve() takes quantifier-free formulas only: "
                "it guarantees every sat model is back-substituted and checked exactly "
                "in-process, and a model for a quantified formula binds nothing that can be "
                "checked. Export it and drive the solver yourself, or use alkahest.decide "
                "for real quantifier elimination"
            ),
            recommendation=recommendation,
            script=script,
        )
    checkable, why = _exactly_checkable(formula)
    if not checkable:
        refusal = _refuse_uncheckable(why)
        return SmtSupport(
            supported=False,
            exportable=True,
            quantified=False,
            solver=solver_name,
            logic=logic,
            reason="not_exactly_checkable",
            detail=str(refusal),
            recommendation=recommendation,
            script=script,
            error=refusal,
        )
    if solver_name is None:
        return SmtSupport(
            supported=False,
            exportable=True,
            quantified=False,
            solver=None,
            logic=logic,
            reason="no_solver",
            detail=(
                f"the formula exports cleanly as {logic}, but no solver binary was found; "
                "see alkahest.smt.solvers()"
            ),
            recommendation=recommendation,
            script=script,
        )
    detail = (
        f"{logic}; {_PREFER_IN_TREE_DETAIL}"
        if recommendation == "prefer_in_tree"
        else f"{logic}; mixed/integer arithmetic is what the bridge is for"
    )
    return SmtSupport(
        supported=True,
        exportable=True,
        quantified=False,
        solver=solver_name,
        logic=logic,
        reason="ok",
        detail=detail,
        recommendation=recommendation,
        script=script,
    )


# ---------------------------------------------------------------------------
# S-expression reading — the hard half
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      ;[^\n]*                      # comment
    | \|[^|]*\|                    # quoted symbol
    | "(?:[^"]|"")*"               # string literal
    | [()]                         # parens
    | [^\s()|";]+                  # simple symbol / numeral
    """,
    re.VERBOSE,
)


def _parse_sexprs(text: str) -> list[Any]:
    """Parse SMT-LIB 2 solver output into nested lists of atom strings."""
    stack: list[list[Any]] = []
    out: list[Any] = []
    for token in _TOKEN_RE.findall(text):
        if token.startswith(";"):
            continue
        if token == "(":
            new: list[Any] = []
            stack.append(new)
            continue
        if token == ")":
            if not stack:
                # Unbalanced output; drop the stray close rather than crash —
                # the caller decides what a missing model means.
                continue
            done = stack.pop()
            (stack[-1] if stack else out).append(done)
            continue
        (stack[-1] if stack else out).append(token)
    # An unterminated form at EOF is dropped for the same reason.
    return out


_INT_RE = re.compile(r"^-?\d+$")
_DEC_RE = re.compile(r"^-?\d+\.\d+$")


def _inexact(name: str, rendered: str, why: str) -> SmtError:
    return SmtError(
        f"[E-SMT-003] model value for {name!r} cannot be lifted exactly: {rendered} ({why})",
        code="E-SMT-003",
        remediation=(
            "this is refused, not rounded. A float witness recorded as an exact one is the "
            "silent error this bridge exists to prevent: the loop would build on a value "
            "that does not actually satisfy the constraints. Constrain the variable to a "
            "rational range, or use alkahest.real_roots / refine_root for the algebraic case"
        ),
    )


def _render(sexp: Any) -> str:
    if isinstance(sexp, str):
        return sexp
    return "(" + " ".join(_render(item) for item in sexp) + ")"


def _lift_value(sexp: Any, name: str) -> Fraction:
    """Lift one SMT-LIB model value to an exact :class:`~fractions.Fraction`.

    Refuses anything that is not exactly a rational — most importantly z3's
    ``root-obj`` algebraic numbers.
    """
    if isinstance(sexp, str):
        if _INT_RE.match(sexp):
            return Fraction(int(sexp))
        if _DEC_RE.match(sexp):
            # SMT-LIB decimals are exact decimal rationals, and `Fraction` parses
            # the *string*, so `1.1` becomes 11/10 and never a binary float.
            return Fraction(sexp)
        raise _inexact(name, sexp, "not a numeral")
    if not sexp:
        raise _inexact(name, "()", "empty term")
    head = sexp[0]
    if head == "-" and len(sexp) == 2:
        return -_lift_value(sexp[1], name)
    if head == "/" and len(sexp) == 3:
        numer = _lift_value(sexp[1], name)
        denom = _lift_value(sexp[2], name)
        if denom == 0:
            raise _inexact(name, _render(sexp), "zero denominator")
        return numer / denom
    if head == "root-obj":
        raise _inexact(
            name,
            _render(sexp),
            "an algebraic number of degree > 1; lifting these into RootInterval / "
            "refine_root is not implemented yet",
        )
    raise _inexact(name, _render(sexp), f"unhandled term head {head!r}")


def _parse_model(sexprs: list[Any]) -> dict[str, Any]:
    """Collect ``(define-fun name () Sort value)`` bindings from solver output.

    Handles both the bare ``( (define-fun …) … )`` shape modern z3 prints and
    the older ``(model (define-fun …) …)`` wrapper.
    """
    out: dict[str, Any] = {}

    def visit(form: Any) -> None:
        if not isinstance(form, list):
            return
        if form and form[0] == "define-fun" and len(form) == 5 and form[2] == []:
            name = form[1]
            if name.startswith("|") and name.endswith("|") and len(name) >= 2:
                name = name[1:-1]
            out[name] = form[4]
            return
        for item in form:
            visit(item)

    for form in sexprs:
        visit(form)
    return out


# ---------------------------------------------------------------------------
# Exact evaluation of the emitted script against a model
# ---------------------------------------------------------------------------

_DECL_RE = re.compile(r"^\(declare-fun\s+(\|[^|]*\||\S+)\s+\(\)\s+(Int|Real)\)\s*$", re.MULTILINE)


def _declared_sorts(script: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, sort in _DECL_RE.findall(script):
        if name.startswith("|") and name.endswith("|"):
            name = name[1:-1]
        out[name] = sort
    return out


class _ScriptCheckFailure(Exception):
    """An assertion in the emitted script is false (or unevaluable) under the model."""


def _num(value: Fraction | bool) -> Fraction:
    """Require an arithmetic operand to be an exact rational.

    The emitter never produces arithmetic over booleans, so this cannot fire on
    a script alkahest wrote. It is *enforced* rather than assumed because the
    alternative is Python's own coercion: ``-True`` is ``-1`` and ``True / True``
    is ``1.0`` — a float. A float reaching this checker would silently weaken
    the one guarantee it exists to provide, so an impossible operand is a hard
    failure rather than a quiet numeric downgrade.
    """
    if isinstance(value, bool) or not isinstance(value, Fraction):
        raise _ScriptCheckFailure(f"arithmetic on a non-rational operand: {value!r}")
    return value


def _eval_term(sexp: Any, env: Mapping[str, Fraction]) -> Fraction | bool:
    """Evaluate an emitted SMT-LIB term/formula exactly over ``Fraction``.

    Deliberately narrow: it understands exactly the operators
    ``alkahest_core::logic::smtlib`` emits, and raises on anything else rather
    than guessing.  This is the second, independent check on a `sat` model — the
    first evaluates the *original* Alkahest formula through the kernel, this one
    evaluates the *script that was actually sent*, so a mistranslation in the
    emitter cannot pass both.
    """
    if isinstance(sexp, str):
        if sexp == "true":
            return True
        if sexp == "false":
            return False
        if _INT_RE.match(sexp):
            return Fraction(int(sexp))
        if _DEC_RE.match(sexp):
            return Fraction(sexp)
        key = sexp[1:-1] if sexp.startswith("|") and sexp.endswith("|") else sexp
        if key in env:
            return env[key]
        raise _ScriptCheckFailure(f"unbound symbol {key!r}")
    if not sexp:
        raise _ScriptCheckFailure("empty term")
    head, args = sexp[0], sexp[1:]
    vals = [_eval_term(a, env) for a in args]

    if head == "-" and len(vals) == 1:
        return -_num(vals[0])
    if head in ("+", "-", "*", "/"):
        acc = _num(vals[0])
        for raw in vals[1:]:
            v = _num(raw)
            if head == "+":
                acc = acc + v
            elif head == "-":
                acc = acc - v
            elif head == "*":
                acc = acc * v
            else:
                if v == 0:
                    raise _ScriptCheckFailure("division by zero")
                acc = acc / v
        return acc
    if head == "to_real" and len(vals) == 1:
        return vals[0]
    if head == "abs" and len(vals) == 1:
        return abs(_num(vals[0]))
    if head == "ite" and len(vals) == 3:
        return vals[1] if vals[0] else vals[2]
    if head == "not" and len(vals) == 1:
        return not vals[0]
    if head == "and":
        return all(bool(v) for v in vals)
    if head == "or":
        return any(bool(v) for v in vals)
    if head == "=>" and len(vals) == 2:
        return (not vals[0]) or bool(vals[1])
    if head in ("=", "<", "<=", ">", ">=") and len(vals) == 2:
        lhs, rhs = vals
        if head == "=":
            return lhs == rhs
        if head == "<":
            return lhs < rhs
        if head == "<=":
            return lhs <= rhs
        if head == ">":
            return lhs > rhs
        return lhs >= rhs
    raise _ScriptCheckFailure(f"unhandled operator {head!r}/{len(vals)}")


def _check_script(script: str, model: Mapping[str, Fraction]) -> list[str]:
    """Return the emitted assertions that the model does **not** satisfy."""
    failures: list[str] = []
    for form in _parse_sexprs(script):
        if not (isinstance(form, list) and len(form) == 2 and form[0] == "assert"):
            continue
        try:
            value = _eval_term(form[1], model)
        except _ScriptCheckFailure as exc:
            failures.append(f"{_render(form[1])} — {exc}")
            continue
        if value is not True:
            failures.append(_render(form[1]))
    return failures


# ---------------------------------------------------------------------------
# SmtResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmtResult:
    """What an external solver said, and how much of it was checked here.

    Quacks like a :class:`~alkahest.DerivedResult` (``value`` / ``steps`` /
    ``verification`` / ``certificate``) so ``ResearchSession.record`` accepts it
    unchanged — and so the honest status travels with it.

    Attributes
    ----------
    status : {"sat", "unsat", "unknown"}
        The solver's verdict.
    model : dict[str, Fraction]
        Exact rational witness, on ``sat``.  Always the values that were
        actually verified.
    model_exprs : dict[str, Expr]
        The same values interned into an :class:`~alkahest.ExprPool`, when one
        was passed or was active via ``alkahest.context(pool=...)``.  Empty
        otherwise — an ``Expr`` carries no reference to its pool, so this is not
        always constructible, and :attr:`model` never depends on it.
    engine : str
        Which solver answered, and at what version.  A result that does not say
        what produced it is not something a loop can weigh.
    logic : str
        The SMT-LIB logic that was sent.
    smtlib : str
        The exact script that was sent.  Also exposed as ``certificate``.
    verification : dict
        ``DerivedResult``-shaped: ``status`` is ``exactly_verified`` for a
        checked ``sat`` model, :data:`EXTERNALLY_ASSERTED` for ``unsat``, and
        ``unverified`` for ``unknown``.
    badge : str
        The unflattering one-line rendering of ``verification["status"]``.
    reason_unknown : str or None
        The solver's own explanation, on ``unknown``.
    elapsed_ms : float
        Wall-clock time spent in the solver process.
    """

    status: str
    model: dict[str, Fraction]
    model_exprs: dict[str, Any]
    engine: str
    logic: str
    smtlib: str
    verification: dict[str, Any]
    reason_unknown: str | None
    elapsed_ms: float
    raw_output: str
    steps: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def value(self) -> Any:
        """The formula that was asked about — ``DerivedResult``-compatible."""
        return self._formula

    @property
    def certificate(self) -> str:
        """The SMT-LIB script.  An artifact, explicitly *not* a checked proof."""
        return self.smtlib

    @property
    def badge(self) -> str:
        """Honest one-line description of what was actually established."""
        return STATUS_BADGES.get(self.verification.get("status", ""), "unrecognised status")

    @property
    def machine_checked(self) -> bool:
        """``True`` only when a checker ran **in this process**.

        ``unsat`` is never machine-checked here, however confident the solver
        was.
        """
        return self.verification.get("status") == "exactly_verified"

    _formula: Any = None

    # Deliberately no `__bool__`: `unsat` and `unknown` would both be falsy, and
    # a loop that wrote `if not result:` would treat "proved impossible" and
    # "gave up" as the same outcome. Branch on `.status`.


# ---------------------------------------------------------------------------
# solve()
# ---------------------------------------------------------------------------

_STATUS_TOKENS = ("sat", "unsat", "unknown")
_REASON_RE = re.compile(r"\(\s*:reason-unknown\s+(.*?)\s*\)\s*$", re.MULTILINE | re.DOTALL)
_TIMEOUT_REASONS = ("timeout", "canceled", "cancelled", "resourceout", "resource")


def _budget_exceeded_error(message: str) -> BaseException:
    from ._budget import _budget_exceeded

    return _budget_exceeded(
        message,
        remediation=(
            "raise Budget(wall_ms=...), or record this candidate as hard-not-hung and move "
            "on; a solver timeout is a resource verdict, not a mathematical one"
        ),
    )


def _run_solver(
    spec: _SolverSpec, path: str, script: str, wall_ms: float | None
) -> tuple[str, float]:
    args = [path, *spec.script_args]
    if wall_ms is not None and spec.timeout_flag is not None:
        args.append(spec.timeout_flag.format(ms=max(1, int(wall_ms))))
    # A grace window on top of the solver's own limit: the parent deadline is a
    # backstop for a solver that ignores or overshoots its flag, not the primary
    # mechanism.
    deadline = None if wall_ms is None else max(wall_ms / 1000.0 * 1.5, wall_ms / 1000.0 + 0.5)

    handle, filename = tempfile.mkstemp(suffix=".smt2", prefix="alkahest-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(script)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [*args, filename],
                capture_output=True,
                text=True,
                timeout=deadline,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _budget_exceeded_error(
                f"[E-BUDGET-001] budget exceeded: {spec.name} did not answer within {wall_ms} ms"
            ) from exc
        elapsed_ms = (time.monotonic() - started) * 1000.0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(filename)
    return (proc.stdout or "") + (proc.stderr or ""), elapsed_ms


def _extract_status(output: str) -> str | None:
    for line in output.splitlines():
        token = line.strip()
        if token in _STATUS_TOKENS:
            return token
    return None


def solve(
    formula: Any,
    *,
    solver: str = "auto",
    logic: str = "auto",
    budget: Any = None,
    pool: Any = None,
) -> SmtResult:
    """Ask an external SMT solver about ``formula``.

    Parameters
    ----------
    formula : Expr
        A **quantifier-free** predicate expression.  Quantified formulas export
        fine with :func:`to_smtlib` but are refused here (``E-SMT-002``): this
        function guarantees that every ``sat`` model is checked exactly
        in-process, and there is nothing to check for a quantified model.
    solver : str
        ``"auto"`` (first installed, in :data:`SOLVERS` order) or a name.
    logic : str
        Forwarded to :func:`to_smtlib`.
    budget : Budget, optional
        ``budget.wall_ms`` is passed to the solver's own timeout flag *and*
        enforced as a parent-side deadline.  A timeout raises
        :class:`~alkahest.BudgetExceededError` (``E-BUDGET-001``), so a loop
        gets the structured "hard, not hung" distinction rather than a bare
        ``unknown``.
    pool : ExprPool, optional
        Pool for :attr:`SmtResult.model_exprs`; defaults to
        :func:`alkahest.active_pool`.  :attr:`SmtResult.model` is exact and
        present either way.

    Returns
    -------
    SmtResult

    Raises
    ------
    SmtError
        ``E-SMT-001`` no solver installed — a refusal, never a silent fallback
        to the weak interval :func:`alkahest.satisfiable`.
        ``E-SMT-002`` formula outside the exportable/solvable fragment.
        ``E-SMT-003`` a model value (``root-obj``) cannot be lifted exactly.
        ``E-SMT-004`` the model failed back-substitution.  This is raised, never
        warned: it means the bridge or the solver is broken.
    BudgetExceededError
        The solver hit ``budget.wall_ms``.

    Examples
    --------
    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> n = pool.symbol("n", "integer")
    >>> x = pool.symbol("x")
    >>> f = ak.And(pool.gt(x, n), pool.lt(x * x, pool.integer(10)))
    >>> r = ak.smt.solve(f)                     # doctest: +SKIP
    >>> r.status, r.badge                       # doctest: +SKIP
    ('sat', 'the solver's model was substituted back and ...')
    """
    if _is_quantified(formula):
        raise SmtError(
            "[E-SMT-002] solve() takes quantifier-free formulas only; this one is quantified",
            code="E-SMT-002",
            remediation=(
                "solve() guarantees that every sat model is back-substituted and checked "
                "exactly in-process, and a model for a quantified formula binds nothing "
                "that can be checked. Use alkahest.to_smtlib to export it and drive the "
                "solver yourself, or alkahest.decide for real quantifier elimination"
            ),
        )
    spec, path = _resolve_solver(solver)
    script = to_smtlib(formula, logic)
    checkable, why = _exactly_checkable(formula)
    if not checkable:
        raise _refuse_uncheckable(why)
    match = _LOGIC_RE.search(script)
    logic_used = match.group(1) if match else logic
    # `(get-info :reason-unknown)` is appended rather than emitted, so the
    # golden-file artifact stays exactly what `to_smtlib` produced.
    driver_script = script + "(get-info :reason-unknown)\n(exit)\n"

    wall_ms = getattr(budget, "wall_ms", None) if budget is not None else None
    output, elapsed_ms = _run_solver(spec, path, driver_script, wall_ms)
    engine = f"{spec.name} {_version_of(path, spec)}"

    status = _extract_status(output)
    if status is None:
        raise SmtError(
            f"[E-SMT-002] {spec.name} returned no sat/unsat/unknown verdict; output was:\n"
            f"{output.strip()[:2000]}",
            code="E-SMT-002",
            remediation=(
                "the emitted script was rejected by the solver. Report this with the script "
                "from alkahest.to_smtlib(formula) — a script alkahest emits and a solver "
                "cannot read is an emitter bug, not a user error"
            ),
        )

    reason_match = _REASON_RE.search(output)
    reason_unknown = reason_match.group(1).strip('"') if reason_match else None
    if status == "unknown" and reason_unknown:
        lowered = reason_unknown.lower()
        if wall_ms is not None and any(token in lowered for token in _TIMEOUT_REASONS):
            raise _budget_exceeded_error(
                f"[E-BUDGET-001] budget exceeded: {spec.name} stopped after {wall_ms} ms "
                f"(reason-unknown: {reason_unknown})"
            )

    model: dict[str, Fraction] = {}
    model_exprs: dict[str, Any] = {}
    if status == "sat":
        model = _lift_model(output, script, formula)
        _verify_model(formula, script, model, engine)
        model_exprs = _intern_model(model, pool)

    verification = _verification_for(status, engine, reason_unknown, logic_used)
    return SmtResult(
        status=status,
        model=model,
        model_exprs=model_exprs,
        engine=engine,
        logic=logic_used,
        smtlib=script,
        verification=verification,
        reason_unknown=reason_unknown,
        elapsed_ms=elapsed_ms,
        raw_output=output,
        steps=(
            {
                "rule": "smt_export",
                "detail": f"emitted {logic_used} SMT-LIB 2 and ran {engine}",
                "result": status,
            },
        ),
        _formula=formula,
    )


def _lift_model(output: str, script: str, formula: Any) -> dict[str, Fraction]:
    """Parse, lift, and complete the solver's model."""
    raw = _parse_model(_parse_sexprs(output))
    model: dict[str, Fraction] = {}
    for name, sexp in raw.items():
        model[name] = _lift_value(sexp, name)

    sorts = _declared_sorts(script)
    for name, sort in sorts.items():
        if name not in model:
            # A solver may omit a "don't care" variable.  Completing it here (and
            # then checking the completed model) keeps the witness total: a model
            # a caller cannot substitute is not a witness.
            model[name] = Fraction(0)
        if sort == "Int" and model[name].denominator != 1:
            raise SmtError(
                f"[E-SMT-004] model assigns the Int-sorted symbol {name!r} the "
                f"non-integer value {model[name]}",
                code="E-SMT-004",
                remediation=(
                    "the solver returned a model that violates its own declaration; report "
                    "this with the script from alkahest.to_smtlib(formula)"
                ),
            )
    # Symbols the formula mentions but the script never declared would mean the
    # emitter dropped something; the back-substitution check below catches it.
    for name in _formula_symbols(formula):
        model.setdefault(name, Fraction(0))
    return model


def _verify_model(formula: Any, script: str, model: Mapping[str, Fraction], engine: str) -> None:
    """Two independent exact checks on a ``sat`` model.  Never optional.

    1. Substitute the model into the **original Alkahest formula** and evaluate
       it exactly through the kernel.  This is the invariant the whole bridge
       rests on.
    2. Evaluate every assertion in the **script that was actually sent**.  A
       mistranslation in the emitter would have to fool both to slip through,
       and this one also re-checks the refined-domain side conditions
       (``Positive``/``NonNegative``/``NonZero``) that are asserted separately.
    """
    from . import evaluate as _evaluate

    symbols = _formula_symbols(formula)
    bindings = {expr: model[name] for name, expr in symbols.items() if name in model}
    missing = sorted(name for name in symbols if name not in model)
    if missing:
        raise SmtError(
            f"[E-SMT-004] {engine} returned a model with no value for {missing}",
            code="E-SMT-004",
            remediation=_BROKEN_BRIDGE,
        )

    result = _evaluate(formula, bindings, mode="exact")
    if result.status != "ok":
        raise SmtError(
            f"[E-SMT-004] the model from {engine} could not be checked: exact evaluation "
            f"of the formula reported {result.status} ({result.reason})",
            code="E-SMT-004",
            remediation=(
                "an unchecked sat model is not a result this bridge will hand back. Reduce "
                "the formula to the exactly-evaluable fragment, or use alkahest.to_smtlib "
                "and drive the solver yourself if you are willing to take the model on "
                "trust"
            ),
        )
    if result.value != 1:
        raise SmtError(
            f"[E-SMT-004] the model from {engine} does not satisfy the formula: "
            f"back-substitution evaluated it to false. Model: "
            f"{ {k: str(v) for k, v in sorted(model.items())} }",
            code="E-SMT-004",
            remediation=_BROKEN_BRIDGE,
        )

    failures = _check_script(script, model)
    if failures:
        raise SmtError(
            f"[E-SMT-004] the model from {engine} does not satisfy the emitted script; "
            f"unsatisfied assertions: {failures}",
            code="E-SMT-004",
            remediation=_BROKEN_BRIDGE,
        )


_BROKEN_BRIDGE = (
    "this is raised, never warned: either the SMT-LIB emitter mistranslated the formula or "
    "the solver returned an unsound model, and both are bugs that must not be absorbed as a "
    "log line. Report it with the script from alkahest.to_smtlib(formula)"
)


def _intern_model(model: Mapping[str, Fraction], pool: Any) -> dict[str, Any]:
    if pool is None:
        from . import active_pool

        pool = active_pool()
    if pool is None:
        return {}
    return {
        name: pool.rational(value.numerator, value.denominator) for name, value in model.items()
    }


def _verification_for(
    status: str, engine: str, reason_unknown: str | None, logic: str
) -> dict[str, Any]:
    if status == "sat":
        return {
            "status": "exactly_verified",
            "evidence": "model_back_substitution",
            "externally_verified": False,
            "method": "smt_model_check",
            "engine": engine,
            "logic": logic,
            "artifact_format": "smtlib2",
            "side_conditions": [],
        }
    if status == "unsat":
        return {
            "status": EXTERNALLY_ASSERTED,
            "evidence": "external_solver_assertion",
            "externally_verified": False,
            "method": "smt_solver",
            "engine": engine,
            "logic": logic,
            "artifact_format": "smtlib2",
            # Stated in the record itself, so a reader who never looks up the
            # badge still cannot mistake this for a proof.
            "note": (
                "no unsat proof was consumed or checked; this status is not in "
                "research.MACHINE_CHECKED_STATUSES"
            ),
            "side_conditions": [],
        }
    return {
        "status": "unverified",
        "evidence": "none",
        "externally_verified": False,
        "method": "smt_solver",
        "engine": engine,
        "logic": logic,
        "artifact_format": "smtlib2",
        "reason_unknown": reason_unknown,
        "side_conditions": [],
    }
