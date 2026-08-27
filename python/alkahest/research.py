"""Session-level provenance for autoresearch loops — the Alkahest claim graph.

:class:`~alkahest.DerivedResult` is a *per-call* object: it carries a value, a
derivation log, an optional Lean certificate, and verification metadata for one
computation.  A research loop that runs for days produces thousands of them and
has nowhere to put them, so its output degrades into a chat transcript — and
transcripts do not survive context compaction.

This module supplies the missing session-level artifact: a **directed acyclic
graph of claims**.  Each :class:`Claim` carries

* a **stable, content-addressed identifier** derived from the normalised
  statement, the hypotheses, and the method — so the same claim derived twice,
  in different processes or on different days, gets the same ID and citing it
  across runs is meaningful;
* the statement (an :class:`~alkahest.Expr`, an equation, or free text);
* the **hypotheses** it holds under, taken from :class:`~alkahest.Assumptions`
  (hypotheses travel with the claim rather than living in an agent's context
  window, where they are dropped at the next compaction);
* the derivation (``DerivedResult.steps``);
* certificate text and verification status, **copied verbatim**;
* provenance metadata (operation, arguments, Alkahest version, build features);
* **dependency edges** to the claim IDs it was derived from.

The graph serialises to a versioned, deterministic, diffable JSON document
(:meth:`ClaimGraph.to_json`), renders to Markdown and LaTeX
(:meth:`ClaimGraph.to_markdown`, :meth:`ClaimGraph.to_latex`), and can be
**re-verified** against a newer library version (:meth:`ClaimGraph.verify`).

Honesty invariant
-----------------
The recording layer **never upgrades a claim's status**.  A claim's status is
whatever ``DerivedResult.verification["status"]`` said; claims created by
:meth:`ResearchSession.conjecture` are always ``"unverified"``, and there is no
parameter to say otherwise.  :meth:`ClaimGraph.verify` may only *lower*
confidence — a failed re-check marks a claim ``"refuted"``; a successful
re-check is recorded as an audit entry and does not promote anything.

Quick start
-----------
>>> import alkahest as ak
>>> pool = ak.ExprPool()
>>> x = pool.symbol("x")
>>> with ak.research.session(title="Demo", pool=pool, capture=True) as s:
...     result = ak.integrate(x / (x**2 + pool.integer(1)), x)   # captured
>>> graph = s.graph
>>> len(graph) >= 1
True
>>> doc = graph.to_markdown()
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import threading
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import os
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "MACHINE_CHECKED_STATUSES",
    "SCHEMA_VERSION",
    "STATUS_BADGES",
    "Claim",
    "ClaimGraph",
    "ClaimGraphError",
    "CycleError",
    "MissingClaimError",
    "RecheckOutcome",
    "ResearchSession",
    "VerificationReport",
    "captured_operations",
    "claim_id",
    "session",
]

#: Version of the on-disk claim-graph schema.  Bumped whenever the serialised
#: shape changes in a way that older readers cannot interpret.
SCHEMA_VERSION = 1

#: ``"kind"`` discriminator written into every serialised document.
DOCUMENT_KIND = "alkahest.claim_graph"

#: Statuses that mean an in-kernel or external checker actually verified the
#: claim, as opposed to merely producing an artifact that *could* be checked.
MACHINE_CHECKED_STATUSES = frozenset({"exactly_verified", "lean_checked"})

#: Human-readable, deliberately non-flattering renderings of each status.  A
#: reviewer must be able to see at a glance which subset of a document is
#: machine-checkable, so an emitted-but-unchecked certificate is never worded
#: like a proof.
STATUS_BADGES: dict[str, str] = {
    "lean_checked": "an external Lean 4 run checked this claim",
    "exactly_verified": "the kernel proved the symbolic residual is zero",
    "certificate_available": ("Lean 4 source was generated but has NOT been machine-checked"),
    "numerically_checked": "floating-point samples only; evidence, not a proof",
    "externally_asserted": (
        "an external solver asserted this; no proof was checked and none was produced"
    ),
    "asserted": (
        "the caller wrote this statement; the result it was recorded from was checked, "
        "but nothing checked that the statement is what was checked"
    ),
    "unverified": "recorded without verification evidence",
    "refuted": "re-verification contradicted this claim",
}

_STATUS_MARK: dict[str, str] = {
    "lean_checked": "[VERIFIED]",
    "exactly_verified": "[VERIFIED]",
    "certificate_available": "[CERT ONLY, UNCHECKED]",
    "numerically_checked": "[NUMERIC ONLY]",
    "externally_asserted": "[EXTERNAL, UNCHECKED]",
    "asserted": "[ASSERTED, UNCHECKED]",
    "unverified": "[UNVERIFIED]",
    "refuted": "[REFUTED]",
}

# Operations known to return a :class:`DerivedResult`.  Extracted from the PyO3
# bindings plus the Python-layer wrappers; the capture hook additionally checks
# ``isinstance`` at run time, so a stale entry here is harmless.
_KNOWN_PRODUCERS: tuple[str, ...] = (
    "apart",
    "cancel",
    "collect_like_terms",
    "diff",
    "diff_forward",
    "expand",
    "factor",
    "integrate",
    "limit",
    "product_definite",
    "product_indefinite",
    "resultant",
    "rsolve",
    "series",
    "simplify",
    "simplify_auto",
    "simplify_clifford_orthogonal",
    "simplify_egraph",
    "simplify_egraph_with",
    "simplify_expanded",
    "simplify_log_exp",
    "simplify_par",
    "simplify_pauli",
    "simplify_redex",
    "simplify_trig",
    "simplify_trig_normal_form",
    "simplify_with",
    "subs",
    "sum_definite",
    "sum_indefinite",
    "symbolic_grad",
)

# Single-input, semantics-preserving rewrites: ``op(e)`` must equal ``e``, so a
# re-verification pass can check ``simplify(input - output) == 0``.
_IDENTITY_OPS = frozenset(
    {
        "apart",
        "cancel",
        "collect_like_terms",
        "expand",
        "factor",
        "simplify",
        "simplify_auto",
        "simplify_egraph",
        "simplify_expanded",
        "simplify_log_exp",
        "simplify_par",
        "simplify_redex",
        "simplify_trig",
        "simplify_trig_normal_form",
    }
)

# Names the expression parser turns into plain symbols but which denote
# constants; bound explicitly when a re-check falls back to numeric sampling.
_NUMERIC_CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}

#: Gap between the values consecutive free symbols are bound to in the numeric
#: residual fallback.  Anything nonzero takes the evaluation off the diagonal;
#: this is small enough to stay inside the usual domains and not a round binary
#: fraction, so it is unlikely to sit on a zero of the residual by accident.
_SYMBOL_SPACING = 0.11


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClaimGraphError(ValueError):
    """Base class for claim-graph errors (``E-RESEARCH-*``)."""

    code = "E-RESEARCH-000"


class MissingClaimError(ClaimGraphError, KeyError):
    """A dependency edge referenced a claim that is not in the graph."""

    code = "E-RESEARCH-001"

    def __str__(self) -> str:  # KeyError repr()s its argument otherwise
        return ClaimGraphError.__str__(self)


class CycleError(ClaimGraphError):
    """The deserialised graph contains a dependency cycle."""

    code = "E-RESEARCH-002"


# ---------------------------------------------------------------------------
# Lazy access to the parent package (research.py is imported from __init__)
# ---------------------------------------------------------------------------


def _ak() -> Any:
    import alkahest

    return alkahest


def _is_expr(obj: Any) -> bool:
    """True for :class:`~alkahest.Expr` without importing the native class."""
    return hasattr(obj, "display_latex") and hasattr(obj, "display_unicode")


def _is_derived(obj: Any) -> bool:
    """True for :class:`~alkahest.DerivedResult`."""
    return hasattr(obj, "verification") and hasattr(obj, "steps") and hasattr(obj, "value")


def _as_expr(obj: Any) -> Any:
    """Coerce a :class:`DerivedResult` to its ``.value``; pass an ``Expr`` through."""
    if _is_derived(obj):
        return obj.value
    if _is_expr(obj):
        return obj
    return None


def _subexpressions(expr: Any, limit: int = 4096) -> Iterator[Any]:
    """Yield *expr* and every distinct subexpression, breadth first.

    Traversal is bounded by *limit* nodes so dependency inference cannot become
    the dominant cost of a hot loop.
    """
    if not _is_expr(expr):
        return
    seen: set[int] = set()
    frontier = [expr]
    while frontier and len(seen) < limit:
        current = frontier.pop(0)
        try:
            key = hash(current)
        except TypeError:  # pragma: no cover - Expr is hashable
            continue
        if key in seen:
            continue
        seen.add(key)
        yield current
        try:
            node = current.node()
        except Exception:
            continue
        stack = [node]
        while stack:
            item = stack.pop()
            if _is_expr(item):
                frontier.append(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)


def _expr_key(expr: Any) -> tuple | None:
    """A hashable identity for an interned expression.

    ``Expr.__hash__`` returns the raw interned ``ExprId``, which is only unique
    within one :class:`~alkahest.ExprPool`; the rendered form is folded in so
    that same-``ExprId`` expressions from two different pools do not collide
    into a spurious dependency edge.
    """
    if not _is_expr(expr):
        return None
    try:
        return ("expr", hash(expr), str(expr))
    except TypeError:  # pragma: no cover - Expr is hashable
        return None


# ---------------------------------------------------------------------------
# Statement normalisation and content addressing
# ---------------------------------------------------------------------------


def _canonical_text(text: str) -> str:
    return " ".join(str(text).split())


def _normalize_statement(statement: Any, *, normalize: bool = True) -> dict[str, Any]:
    """Return ``{"kind", "statement", "latex"}`` for *statement*.

    An :class:`~alkahest.Expr` (or :class:`~alkahest.DerivedResult`) is put into
    the kernel's normal form via :func:`alkahest.simplify` when *normalize* is
    true, so that two structurally different but semantically identical
    constructions hash to the same claim ID.  Normalisation is best-effort: if
    the simplifier raises, the expression is recorded as written.

    A mapping with a ``"statement"`` key is passed through unchanged, which is
    how a pre-rendered relation (see :func:`_infer_assertion`) is supplied.
    """
    if isinstance(statement, dict) and "statement" in statement:
        return {
            "kind": str(statement.get("kind", "text")),
            "statement": _canonical_text(statement["statement"]),
            "latex": statement.get("latex"),
        }
    expr = _as_expr(statement)
    if expr is None:
        return {
            "kind": "text",
            "statement": _canonical_text(statement),
            "latex": None,
        }
    with _suppress_capture():
        if normalize:
            with suppress(Exception):
                expr = _ak().simplify(expr).value
        latex: str | None
        try:
            latex = _ak().latex(expr)
        except Exception:
            latex = None
        return {"kind": "expr", "statement": str(expr), "latex": latex}


def _tex(expr: Any) -> str:
    try:
        return _ak().latex(expr)
    except Exception:
        return _tex_escape(str(expr))


def _infer_assertion(
    name: str, sources: Sequence[Any], result: Any, *, normalize: bool
) -> dict[str, Any] | None:
    """Render what an operation actually asserts, as an equation.

    ``integrate(f, x)`` does not claim ``F``; it claims ``∫ f dx = F``.  Recording
    only the value would make the document unreadable, so the operations whose
    assertion shape is known are rendered as relations.

    The right-hand side (the operation's output) is normalised exactly as a
    plain expression statement would be; the left-hand side is recorded **as
    written**, because "this integrand has that antiderivative" is a claim about
    the integrand the caller supplied.

    Returns ``None`` for operations whose assertion shape is not known, in which
    case the caller falls back to recording the value alone.
    """
    exprs = [e for e in (_as_expr(s) for s in sources) if e is not None]
    value = _as_expr(result)
    if value is None:
        return None
    with _suppress_capture():
        if normalize:
            with suppress(Exception):
                value = _ak().simplify(value).value
        if name == "integrate" and len(exprs) == 2:
            text = f"integral({exprs[0]}, d{exprs[1]}) = {value}"
            latex = rf"\int {_tex(exprs[0])} \, \mathrm{{d}}{_tex(exprs[1])} = {_tex(value)}"
        elif name == "integrate" and len(exprs) == 4:
            text = f"integral({exprs[0]}, d{exprs[1]}, {exprs[2]}, {exprs[3]}) = {value}"
            latex = (
                rf"\int_{{{_tex(exprs[2])}}}^{{{_tex(exprs[3])}}} {_tex(exprs[0])} "
                rf"\, \mathrm{{d}}{_tex(exprs[1])} = {_tex(value)}"
            )
        elif name in {"diff", "diff_forward"} and len(exprs) == 2:
            text = f"d/d{exprs[1]}({exprs[0]}) = {value}"
            latex = (
                rf"\frac{{\mathrm{{d}}}}{{\mathrm{{d}}{_tex(exprs[1])}}}"
                rf"\left({_tex(exprs[0])}\right) = {_tex(value)}"
            )
        elif name in _IDENTITY_OPS and len(exprs) == 1:
            text = f"{exprs[0]} = {value}"
            latex = rf"{_tex(exprs[0])} = {_tex(value)}"
        else:
            return None
    # ``inferred`` marks this as the *engine's* rendering of what the operation
    # asserts, not caller prose.  :meth:`ResearchSession.record` reads it to
    # decide whether a machine-checked status may be carried over; it is dropped
    # by :func:`_normalize_statement` and never reaches the stored claim.
    return {
        "kind": "relation",
        "statement": _canonical_text(text),
        "latex": latex,
        "inferred": True,
    }


def claim_id(statement: str, hypotheses: Sequence[str] = (), method: str = "") -> str:
    """Return the content-addressed identifier for a claim.

    The identifier is ``"clm_" + sha256(payload)[:16]`` where *payload* is a
    canonical JSON encoding of the normalised statement, the **sorted** set of
    hypotheses, and the method.  It deliberately does *not* include timestamps,
    dependency edges, or the library version, so the same claim derived twice
    in different sessions receives the same ID.

    Parameters
    ----------
    statement : str
        The already-normalised statement text (see
        :meth:`ResearchSession.record`, which normalises for you).
    hypotheses : sequence of str
        Hypothesis predicates.  Order is irrelevant — they are sorted.
    method : str
        The operation or method that produced the claim, e.g. ``"integrate"``.

    Returns
    -------
    str
        A stable identifier such as ``"clm_9f1c0b2a7d4e5f60"``.

    Examples
    --------
    >>> from alkahest.research import claim_id
    >>> claim_id("(1/2 * log(2))", ["x > 0"], "integrate") == claim_id(
    ...     "(1/2 * log(2))", ["x > 0"], "integrate"
    ... )
    True
    """
    payload = json.dumps(
        {
            "statement": _canonical_text(statement),
            "hypotheses": sorted(_canonical_text(h) for h in hypotheses),
            "method": _canonical_text(method),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "clm_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One recorded mathematical claim and everything needed to audit it.

    Instances are immutable; :class:`ClaimGraph` produces updated copies via
    :func:`dataclasses.replace` rather than mutating in place.

    Attributes
    ----------
    id : str
        Content-addressed identifier (see :func:`claim_id`).
    statement : str
        Normalised statement text.
    kind : str
        ``"expr"`` when the statement is an expression/equation, ``"text"`` for
        a prose claim or conjecture.
    latex : str or None
        LaTeX rendering of the statement, when it is an expression.
    hypotheses : tuple of str
        Predicates from the governing :class:`~alkahest.Assumptions`.
    method : str
        Operation that produced the claim, e.g. ``"integrate"``.
    status : str
        Copied verbatim from ``DerivedResult.verification["status"]``, or
        ``"unverified"`` for a conjecture.  Never upgraded by this layer.
    evidence : str
        Copied verbatim from ``verification["evidence"]`` (``"none"`` when
        absent), or the free-text evidence supplied for a conjecture.
    verification : dict
        The full verification mapping, verbatim.
    derivation : tuple of dict
        ``DerivedResult.steps``.
    certificate : str or None
        Certificate source text, when one was emitted.
    certificate_format : str or None
        Format of *certificate*, e.g. ``"lean4"``.
    depends_on : tuple of str
        IDs of the claims this one was derived from.
    check : dict or None
        Machine-readable re-verification recipe (see :meth:`ClaimGraph.verify`).
    provenance : dict
        Identity-neutral but reproducible metadata: operation, argument
        renderings, Alkahest version, build feature flags.
    recorded_at : str or None
        ISO-8601 UTC timestamp.  **Volatile** — excluded from
        ``to_json(stable=True)`` and from the graph digest.
    audit : tuple of dict
        Re-verification outcomes appended by :meth:`ClaimGraph.verify`.
    label : str or None
        Short human-readable name used in rendered documents.
    tags : tuple of str
        Free-form labels for querying.
    notes : str or None
        Free-form prose attached by the loop author.
    """

    id: str
    statement: str
    kind: str = "expr"
    latex: str | None = None
    hypotheses: tuple[str, ...] = ()
    method: str = ""
    status: str = "unverified"
    evidence: str = "none"
    verification: dict[str, Any] = field(default_factory=dict)
    derivation: tuple[dict[str, Any], ...] = ()
    certificate: str | None = None
    certificate_format: str | None = None
    depends_on: tuple[str, ...] = ()
    check: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    recorded_at: str | None = None
    audit: tuple[dict[str, Any], ...] = ()
    label: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None

    # -- identity ---------------------------------------------------------

    @property
    def machine_checked(self) -> bool:
        """True when a checker actually verified this claim.

        An emitted-but-unchecked Lean certificate is **not** machine-checked.
        """
        return self.status in MACHINE_CHECKED_STATUSES

    @property
    def badge(self) -> str:
        """Honest one-line description of this claim's verification status."""
        return STATUS_BADGES.get(self.status, "unrecognised status")

    @property
    def mark(self) -> str:
        """Short all-caps marker, e.g. ``"[VERIFIED]"`` or ``"[UNVERIFIED]"``."""
        return _STATUS_MARK.get(self.status, "[UNKNOWN STATUS]")

    def content_id(self) -> str:
        """Recompute the content-addressed ID from this claim's own fields."""
        return claim_id(self.statement, self.hypotheses, self.method)

    # -- serialisation ----------------------------------------------------

    def to_dict(self, *, stable: bool = False) -> dict[str, Any]:
        """Return a JSON-ready mapping.

        Parameters
        ----------
        stable : bool
            When true, omit volatile fields (``recorded_at`` and any audit
            timestamps) so that two runs of the same computation produce
            byte-identical output.
        """
        data: dict[str, Any] = {
            "id": self.id,
            "statement": self.statement,
            "kind": self.kind,
            "latex": self.latex,
            "hypotheses": list(self.hypotheses),
            "method": self.method,
            "status": self.status,
            "evidence": self.evidence,
            "verification": _jsonable(self.verification),
            "derivation": [_jsonable(step) for step in self.derivation],
            "certificate": self.certificate,
            "certificate_format": self.certificate_format,
            "depends_on": list(self.depends_on),
            "check": _jsonable(self.check) if self.check is not None else None,
            "provenance": _jsonable(self.provenance),
            "audit": [_jsonable(entry) for entry in self.audit],
            "label": self.label,
            "tags": list(self.tags),
            "notes": self.notes,
        }
        if stable:
            data["audit"] = [
                {k: v for k, v in entry.items() if k != "at"} for entry in data["audit"]
            ]
        else:
            data["recorded_at"] = self.recorded_at
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Claim:
        """Rebuild a :class:`Claim` from :meth:`to_dict` output."""
        return cls(
            id=str(data["id"]),
            statement=str(data["statement"]),
            kind=str(data.get("kind", "expr")),
            latex=data.get("latex"),
            hypotheses=tuple(data.get("hypotheses", ()) or ()),
            method=str(data.get("method", "")),
            status=str(data.get("status", "unverified")),
            evidence=str(data.get("evidence", "none")),
            verification=dict(data.get("verification", {}) or {}),
            derivation=tuple(dict(step) for step in data.get("derivation", ()) or ()),
            certificate=data.get("certificate"),
            certificate_format=data.get("certificate_format"),
            depends_on=tuple(data.get("depends_on", ()) or ()),
            check=dict(data["check"]) if data.get("check") else None,
            provenance=dict(data.get("provenance", {}) or {}),
            recorded_at=data.get("recorded_at"),
            audit=tuple(dict(entry) for entry in data.get("audit", ()) or ()),
            label=data.get("label"),
            tags=tuple(data.get("tags", ()) or ()),
            notes=data.get("notes"),
        )


def _crosscheck_record(cc: Any) -> dict[str, Any] | None:
    """Flatten a :class:`alkahest.crosscheck.CrossCheck` for storage in a claim.

    Returns ``None`` for ``None`` so callers can pass the value through
    unconditionally.  Everything stored is a plain JSON-serialisable scalar, and
    the ``outcome`` key is always present — including for an *unavailable*
    oracle, which must stay distinguishable from agreement.
    """
    if cc is None:
        return None
    outcome = str(getattr(cc, "outcome", "incomparable"))
    record: dict[str, Any] = {
        "outcome": outcome,
        "conclusive": bool(getattr(cc, "conclusive", False)),
        "checked": bool(getattr(cc, "checked", False)),
    }
    for attr in ("rung", "rung_name", "reason", "oracle", "oracle_version", "witness"):
        value = getattr(cc, attr, None)
        if value is not None:
            record[attr] = value if isinstance(value, (int, float, bool)) else str(value)
    divergence = getattr(cc, "divergence", None)
    if divergence is not None:
        record["divergence"] = str(divergence)
    return record


def _run_crosscheck(name: str, args: tuple, kwargs: dict) -> Any:
    """Best-effort cross-check for a captured operation; never raises.

    A cross-check is an optional extra signal, so a failure to *pose* one must
    not take down the recording of an otherwise good claim.  Any failure is
    turned into an ``incomparable`` record carrying the reason, which is the
    honest reading: we learned nothing, as opposed to we agree.

    The whole call runs under :func:`_suppress_capture`.  A cross-check drives
    ``integrate``/``diff``/``simplify`` to evaluate its comparison rungs, and
    those are captured operations: without the guard each recorded claim would
    record the claims generated while checking it, which recurses without
    terminating.
    """
    try:
        from . import crosscheck as _crosscheck
    except Exception:  # pragma: no cover - crosscheck is always importable
        return None
    if name not in getattr(_crosscheck, "OPERATIONS", ()):
        return None

    # `check()` has keyword-only parameters of its own (`oracle`, `points`,
    # `pool`, ...). Forwarding the captured operation's kwargs blindly would let
    # a same-named argument be swallowed as a cross-check *setting*, so the
    # comparison would silently run against a different call than the one being
    # recorded — a wrong "agree" rather than an honest "we could not check
    # this". Refuse instead. Introspected rather than hard-coded so the guard
    # cannot drift out of step with `check`'s signature.
    reserved = {
        p.name
        for p in inspect.signature(_crosscheck.check).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    clashing = sorted(reserved & set(kwargs))
    if clashing:
        return _StubCrossCheck(
            "operation arguments collide with cross-check parameters: " + ", ".join(clashing)
        )

    with _suppress_capture():
        try:
            return _crosscheck.check(name, *args, **kwargs)
        except Exception as exc:
            return _StubCrossCheck(f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class _StubCrossCheck:
    """Stand-in recorded when a cross-check could not even be posed."""

    reason: str
    outcome: str = "incomparable"
    conclusive: bool = False
    checked: bool = False


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary metadata to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Re-verification report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecheckOutcome:
    """Result of re-checking one claim.

    ``outcome`` is one of

    ``"ok"``
        The claim was re-established exactly (symbolic residual is zero).
    ``"numeric_ok"``
        The exact check was inconclusive but a numeric residual check passed.
        This is evidence, not a proof.
    ``"failed"``
        The re-check contradicts the claim.  :meth:`ClaimGraph.verify` marks
        such claims ``"refuted"``.
    ``"inconclusive"``
        The recipe ran but decided nothing (e.g. the simplifier could not
        reduce the residual and no numeric sample was possible).
    ``"skipped"``
        No re-verification recipe is attached to the claim.
    """

    claim_id: str
    outcome: str
    kind: str = "none"
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True when the re-check did not contradict the claim."""
        return self.outcome != "failed"


@dataclass(frozen=True)
class VerificationReport:
    """Aggregate outcome of :meth:`ClaimGraph.verify`."""

    outcomes: tuple[RecheckOutcome, ...]
    alkahest_version: str = "unknown"
    checked_at: str | None = None

    def __iter__(self) -> Iterator[RecheckOutcome]:
        return iter(self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def ok(self) -> bool:
        """True when no claim was refuted."""
        return all(o.ok for o in self.outcomes)

    @property
    def failed(self) -> tuple[RecheckOutcome, ...]:
        """Outcomes that contradicted their claim."""
        return tuple(o for o in self.outcomes if o.outcome == "failed")

    def summary(self) -> dict[str, int]:
        """Count of outcomes by kind, e.g. ``{"ok": 3, "skipped": 2}``."""
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        return dict(sorted(counts.items()))

    def to_markdown(self) -> str:
        """Render the report as a Markdown table."""
        lines = [
            f"### Re-verification (Alkahest {self.alkahest_version})",
            "",
            "| Claim | Outcome | Check | Detail |",
            "| --- | --- | --- | --- |",
        ]
        for outcome in self.outcomes:
            lines.append(
                f"| `{outcome.claim_id}` | `{outcome.outcome}` | `{outcome.kind}` | "
                f"{_md_escape(outcome.detail)} |"
            )
        lines.append("")
        lines.append(f"Summary: {self.summary()}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ClaimGraph
# ---------------------------------------------------------------------------


class ClaimGraph:
    """A cycle-free, content-addressed set of :class:`Claim` records.

    A graph owns its claims and their dependency edges.  Edges always point
    from a claim to claims that already exist in the graph, which makes cycles
    impossible by construction for graphs built incrementally; graphs loaded
    from JSON are topologically checked at load time.

    Parameters
    ----------
    title : str, optional
        Document title used by the renderers.
    metadata : mapping, optional
        Free-form session metadata (build features, seeds, notes).

    Examples
    --------
    >>> from alkahest.research import ClaimGraph
    >>> g = ClaimGraph(title="Notes")
    >>> len(g)
    0
    """

    __slots__ = ("_claims", "_dependents", "_metadata", "_order", "_title")

    def __init__(
        self,
        *,
        title: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._claims: dict[str, Claim] = {}
        self._dependents: dict[str, set[str]] = {}
        self._order: list[str] = []
        self._title = title
        self._metadata: dict[str, Any] = dict(metadata or {})

    # -- basic container protocol ----------------------------------------

    def __len__(self) -> int:
        return len(self._claims)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._claims

    def __iter__(self) -> Iterator[Claim]:
        return iter(self.claims)

    def __getitem__(self, key: str) -> Claim:
        try:
            return self._claims[key]
        except KeyError:
            raise MissingClaimError(f"no claim {key!r} in this graph") from None

    def __repr__(self) -> str:
        return f"ClaimGraph(title={self._title!r}, claims={len(self._claims)})"

    def get(self, key: str, default: Claim | None = None) -> Claim | None:
        """Return the claim with this ID, or *default*."""
        return self._claims.get(key, default)

    @property
    def title(self) -> str | None:
        """Document title used by :meth:`to_markdown` / :meth:`to_latex`."""
        return self._title

    @property
    def metadata(self) -> dict[str, Any]:
        """Mutable free-form session metadata."""
        return self._metadata

    @property
    def claims(self) -> tuple[Claim, ...]:
        """All claims, in insertion order (which is a topological order)."""
        return tuple(self._claims[i] for i in self._order)

    @property
    def ids(self) -> tuple[str, ...]:
        """All claim IDs, in insertion order."""
        return tuple(self._order)

    # -- mutation ---------------------------------------------------------

    def add(self, claim: Claim) -> Claim:
        """Insert *claim*, or merge it into an identically-identified claim.

        Because IDs are content-addressed, re-deriving the same claim is not an
        error: the stored claim keeps its original status and derivation and
        gains any new dependency edges, tags, and audit entries.  Self-edges
        (a claim that cites an identically-addressed claim, which happens when
        an operation is a no-op) are dropped.

        A **re-verification recipe on the later claim is adopted** when the
        stored claim carries none.  Attaching a ``check`` is the one supported
        way to link a statement to evidence, so recording a statement bare and
        then recording it again with a recipe has to work; dropping the recipe
        made :meth:`verify` report ``skipped`` for a claim that was in fact
        checkable.  An existing recipe is never overwritten — the first
        recorded evidence wins, as the status does.

        Raises
        ------
        MissingClaimError
            If any dependency is absent from the graph.
        CycleError
            If the edge would make the graph cyclic.
        """
        deps = tuple(dict.fromkeys(d for d in claim.depends_on if d != claim.id))
        for dep in deps:
            if dep not in self._claims:
                raise MissingClaimError(
                    f"claim {claim.id!r} depends on {dep!r}, which is not in this graph"
                )
        self._reject_cycles(claim.id, deps)
        existing = self._claims.get(claim.id)
        if existing is None:
            stored = replace(claim, depends_on=deps)
            self._claims[claim.id] = stored
            self._order.append(claim.id)
            self._dependents.setdefault(claim.id, set())
            for dep in deps:
                self._dependents.setdefault(dep, set()).add(claim.id)
            return stored

        merged_deps = tuple(dict.fromkeys((*existing.depends_on, *deps)))
        merged_tags = tuple(dict.fromkeys((*existing.tags, *claim.tags)))
        merged_check = existing.check if existing.check else claim.check
        stored = replace(
            existing,
            depends_on=merged_deps,
            tags=merged_tags,
            check=dict(merged_check) if merged_check else None,
            audit=(*existing.audit, *claim.audit),
        )
        self._claims[claim.id] = stored
        for dep in merged_deps:
            self._dependents.setdefault(dep, set()).add(claim.id)
        return stored

    def _reject_cycles(self, claim_id_: str, deps: Sequence[str]) -> None:
        """Refuse an edge set that would make the graph cyclic.

        A first insertion cannot create a cycle: every dependency must already
        be present, and nothing depends on the new claim yet. The *merge* path
        can, though. IDs are content-addressed over the **normalised**
        statement, so two textually different statements (``"a"`` and ``" a"``)
        share an ID; re-adding one of them merges its edges into the stored
        claim, and those edges may point at claims recorded later — including
        ones that already depend on it.

        The result used to be a graph that served fine in memory, serialised
        fine, and then could never be read back: :meth:`from_json` topologically
        sorts and raised :class:`CycleError`. Rejecting the edge here keeps
        "a ClaimGraph is acyclic" a real invariant, so a round-trip is total.
        """
        if claim_id_ not in self._claims:
            # Fresh insertion: dependencies pre-exist and nothing points here yet.
            return
        for dep in deps:
            if dep == claim_id_:
                continue
            # Does `dep` already (transitively) depend on this claim?
            if claim_id_ in self._walk(dep, lambda i: self._claims[i].depends_on):
                raise CycleError(
                    f"claim {claim_id_!r} cannot depend on {dep!r}: "
                    f"{dep!r} already depends on it, so the edge would close a cycle"
                )

    def _replace_claim(self, claim: Claim) -> None:
        """Overwrite an existing claim in place (edges unchanged)."""
        self._claims[claim.id] = claim

    def merge(self, other: ClaimGraph) -> ClaimGraph:
        """Return a new graph containing the claims of both graphs.

        *other*'s claims are inserted in its own topological order, so cross-
        graph citations resolve as long as the union is acyclic.
        """
        combined = ClaimGraph(title=self._title, metadata={**self._metadata, **other._metadata})
        for claim in self.claims:
            combined.add(claim)
        for claim in other.claims:
            combined.add(claim)
        return combined

    # -- queries ----------------------------------------------------------

    def dependencies(self, key: str) -> tuple[str, ...]:
        """Direct dependencies of the claim with this ID."""
        return self[key].depends_on

    def dependents(self, key: str) -> tuple[str, ...]:
        """Claims that cite the claim with this ID directly."""
        if key not in self._claims:
            raise MissingClaimError(f"no claim {key!r} in this graph")
        return tuple(i for i in self._order if i in self._dependents.get(key, ()))

    def ancestors(self, key: str) -> tuple[str, ...]:
        """All claims this one transitively rests on, nearest first."""
        return self._walk(key, lambda i: self[i].depends_on)

    def impact(self, key: str) -> tuple[str, ...]:
        """Claims that would be invalidated if this claim turned out false.

        The transitive closure of :meth:`dependents` — the answer to "what
        depends on this claim if it turns out to be wrong?".
        """
        return self._walk(key, self.dependents)

    def _walk(self, key: str, step: Callable[[str], Iterable[str]]) -> tuple[str, ...]:
        if key not in self._claims:
            raise MissingClaimError(f"no claim {key!r} in this graph")
        seen: dict[str, None] = {}
        frontier = list(step(key))
        while frontier:
            current = frontier.pop(0)
            if current in seen or current == key:
                continue
            seen[current] = None
            frontier.extend(step(current))
        return tuple(i for i in self._order if i in seen)

    def by_status(self, status: str) -> tuple[Claim, ...]:
        """All claims whose status equals *status*."""
        return tuple(c for c in self.claims if c.status == status)

    def by_tag(self, tag: str) -> tuple[Claim, ...]:
        """All claims carrying *tag*."""
        return tuple(c for c in self.claims if tag in c.tags)

    def by_method(self, method: str) -> tuple[Claim, ...]:
        """All claims produced by *method*."""
        return tuple(c for c in self.claims if c.method == method)

    def machine_checkable(self) -> tuple[Claim, ...]:
        """Claims a checker actually verified (see :attr:`Claim.machine_checked`)."""
        return tuple(c for c in self.claims if c.machine_checked)

    def unverified(self) -> tuple[Claim, ...]:
        """Claims carrying no verification evidence at all."""
        return tuple(c for c in self.claims if c.status == "unverified")

    def roots(self) -> tuple[Claim, ...]:
        """Claims with no dependencies."""
        return tuple(c for c in self.claims if not c.depends_on)

    def leaves(self) -> tuple[Claim, ...]:
        """Claims nothing else depends on."""
        return tuple(c for c in self.claims if not self._dependents.get(c.id))

    def summary(self) -> dict[str, int]:
        """Claim counts by status, sorted by status name."""
        counts: dict[str, int] = {}
        for claim in self.claims:
            counts[claim.status] = counts.get(claim.status, 0) + 1
        return dict(sorted(counts.items()))

    def topological_order(self) -> tuple[str, ...]:
        """A dependency-respecting ordering of the claim IDs."""
        return self.ids

    # -- serialisation ----------------------------------------------------

    def to_dict(self, *, stable: bool = False) -> dict[str, Any]:
        """Return the graph as a JSON-ready mapping.

        Claims are emitted in insertion (topological) order; every mapping is
        written with sorted keys by :meth:`to_json`.
        """
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": DOCUMENT_KIND,
            "title": self._title,
            "claims": [c.to_dict(stable=stable) for c in self.claims],
        }
        metadata = _jsonable(self._metadata)
        if stable:
            metadata = {k: v for k, v in metadata.items() if k not in _VOLATILE_METADATA}
        data["metadata"] = metadata
        return data

    def to_json(self, *, indent: int | None = 2, stable: bool = False) -> str:
        """Serialise to JSON with deterministic key order.

        Parameters
        ----------
        indent : int or None
            Passed to :func:`json.dumps`.  ``None`` emits a compact document.
        stable : bool
            When true, drop volatile fields (timestamps, working directory) so
            two runs of the same computation produce byte-identical output.
            Identity — the claim IDs — never depends on volatile data, so a
            ``stable=False`` document still diffs cleanly outside those fields.

        Examples
        --------
        >>> from alkahest.research import ClaimGraph
        >>> ClaimGraph().to_json(indent=None, stable=True).startswith('{"claims": []')
        True
        """
        return json.dumps(
            self.to_dict(stable=stable),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            separators=(",", ": ") if indent is not None else (", ", ": "),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClaimGraph:
        """Rebuild a graph from :meth:`to_dict` output.

        Raises
        ------
        ClaimGraphError
            On an unknown document kind or a schema version this build cannot
            read.
        CycleError
            If the dependency edges contain a cycle.
        """
        kind = data.get("kind")
        if kind is not None and kind != DOCUMENT_KIND:
            raise ClaimGraphError(f"not an Alkahest claim graph (kind={kind!r})")
        version = int(data.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ClaimGraphError(
                f"claim-graph schema v{version} is newer than this build supports "
                f"(v{SCHEMA_VERSION}); upgrade alkahest to read it"
            )
        graph = cls(title=data.get("title"), metadata=data.get("metadata") or {})
        claims = [Claim.from_dict(entry) for entry in data.get("claims", ())]
        for claim in _topological(claims):
            graph.add(claim)
        return graph

    @classmethod
    def from_json(cls, text: str) -> ClaimGraph:
        """Rebuild a graph from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

    def save(self, path: str | os.PathLike[str], *, stable: bool = False) -> None:
        """Write the graph to *path* as JSON (UTF-8, trailing newline)."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(stable=stable))
            handle.write("\n")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ClaimGraph:
        """Read a graph previously written by :meth:`save`."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_json(handle.read())

    def digest(self) -> str:
        """SHA-256 of the stable serialisation — a fingerprint of the content.

        Two runs of the same computation produce the same digest even though
        their timestamps differ.
        """
        return hashlib.sha256(self.to_json(indent=None, stable=True).encode("utf-8")).hexdigest()

    # -- re-verification --------------------------------------------------

    def verify(
        self,
        *,
        pool: Any = None,
        tolerance: float = 1e-8,
        samples: Sequence[float] = (0.37, 1.23, 2.71),
        mark_refuted: bool = True,
    ) -> VerificationReport:
        """Re-check every claim that carries a re-verification recipe.

        Claims are re-derived from their serialised text — the recipe is parsed
        into a fresh :class:`~alkahest.ExprPool` — so a graph loaded from disk
        can be revalidated against a newer library build rather than trusted
        blindly.

        This pass can only *lower* confidence.  A failed re-check marks the
        claim ``"refuted"`` (when *mark_refuted*); a successful one appends an
        audit entry and leaves the status alone.

        Parameters
        ----------
        pool : ExprPool, optional
            Pool to parse into.  A fresh pool is created when omitted.
        tolerance : float
            Absolute tolerance for the numeric residual fallback.  It does not
            apply to a ``numeric_relation`` recipe whose constants are supplied
            at a precision a float cannot hold: those are evaluated exactly and
            judged against the precision the caller actually gave (see below).
        samples : sequence of float
            Sample points used for the numeric fallback.  Each sample gives a
            *point*, not a single value: free symbols are bound to the sample
            offset by their rank in sorted name order, so they take distinct
            values and the evaluation is off the diagonal ``x = y = z``.  A
            ``numeric_ok`` outcome is still finitely many points — evidence,
            never a proof — and a symbol whose offset leaves an operation's
            domain makes that point unevaluable, which the detail string
            reports and which can leave the outcome ``inconclusive``.
        mark_refuted : bool
            When true (default), failed claims have their status set to
            ``"refuted"`` in place.

        Returns
        -------
        VerificationReport
        """
        ak = _ak()
        pool = pool if pool is not None else ak.ExprPool()
        checked_at = _utcnow()
        version = getattr(ak, "__version__", "unknown")
        outcomes: list[RecheckOutcome] = []
        for claim in self.claims:
            with _suppress_capture():
                outcome = _recheck(claim, pool, tolerance, samples)
            outcomes.append(outcome)
            if outcome.outcome == "skipped":
                continue
            entry = {
                "at": checked_at,
                "alkahest_version": version,
                "outcome": outcome.outcome,
                "kind": outcome.kind,
                "detail": outcome.detail,
            }
            updated = replace(claim, audit=(*claim.audit, entry))
            if outcome.outcome == "failed" and mark_refuted:
                updated = replace(updated, status="refuted")
            self._replace_claim(updated)
        return VerificationReport(
            outcomes=tuple(outcomes), alkahest_version=version, checked_at=checked_at
        )

    # -- rendering --------------------------------------------------------

    def to_markdown(
        self,
        *,
        include_derivations: bool = True,
        include_certificates: bool = False,
        max_steps: int = 12,
    ) -> str:
        """Render the graph as a Markdown research document.

        Every claim gets its statement, hypotheses, method, dependency links,
        derivation, and an honest verification badge.  A summary table at the
        top states exactly which subset of the document a machine actually
        checked.

        Parameters
        ----------
        include_derivations : bool
            Emit the per-claim rewrite-step table.
        include_certificates : bool
            Inline the certificate source (can be long).
        max_steps : int
            Truncate derivation tables to this many steps.
        """
        return _render_markdown(
            self,
            include_derivations=include_derivations,
            include_certificates=include_certificates,
            max_steps=max_steps,
        )

    def to_latex(self, *, standalone: bool = True, include_derivations: bool = True) -> str:
        """Render the graph as a LaTeX research document.

        Each claim becomes a numbered subsection carrying ``\\label{clm:<id>}``;
        dependencies are emitted as ``\\hyperref`` links, so a loop that ran for
        a week emits a writeup with every claim linked to its derivation and
        its certificate status.

        Parameters
        ----------
        standalone : bool
            Emit a full ``article`` document (preamble + ``\\begin{document}``)
            rather than only the body.
        include_derivations : bool
            Emit the per-claim derivation list.
        """
        return _render_latex(self, standalone=standalone, include_derivations=include_derivations)


_VOLATILE_METADATA = frozenset({"started_at", "finished_at", "cwd", "hostname", "duration_s"})


def _topological(claims: Sequence[Claim]) -> list[Claim]:
    """Order *claims* so every claim follows its dependencies."""
    remaining = {c.id: c for c in claims}
    known = set(remaining)
    ordered: list[Claim] = []
    placed: set[str] = set()
    progress = True
    while remaining and progress:
        progress = False
        for cid in list(remaining):
            claim = remaining[cid]
            pending = [d for d in claim.depends_on if d in known and d not in placed and d != cid]
            if pending:
                continue
            ordered.append(claim)
            placed.add(cid)
            del remaining[cid]
            progress = True
    if remaining:
        raise CycleError(
            "claim graph contains a dependency cycle involving: " + ", ".join(sorted(remaining))
        )
    return ordered


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Re-verification recipes
# ---------------------------------------------------------------------------


def _parse_into(pool: Any, text: str, symbols: dict[str, Any]) -> Any:
    return _ak().parse(text, pool, symbols)


def _residual_is_zero(residual: Any) -> bool:
    return str(residual).strip() == "0"


def _numeric_residual(
    expr: Any, symbols: Mapping[str, Any], samples: Sequence[float]
) -> tuple[float | None, int]:
    """Largest ``|expr|`` over the sample points, and how many were evaluable.

    Every free symbol used to be bound to the *same* value, which put the
    evaluation on the diagonal ``x = y = z``.  A residual that vanishes only
    there — ``sin(x)cos(y) - sin(y)cos(x)``, ``x - y``, any symmetry error
    between two variables, the commonest bug class this fallback exists to
    catch — came back indistinguishable from an identity.  Each symbol is
    therefore offset by :data:`_SYMBOL_SPACING` times its rank in the sorted
    symbol names, so the point is off the diagonal in every coordinate while
    the single-symbol case (rank 0) evaluates exactly where it always did.

    An offset can push a sample out of an operation's domain; ``eval_expr``
    then raises and the point is skipped.  The count of points that *did*
    evaluate is returned alongside the worst value so the caller can say
    ``inconclusive`` rather than read a verdict off a residual it could not
    sample.
    """
    ak = _ak()
    worst: float | None = None
    evaluated = 0
    ranks = {name: rank for rank, name in enumerate(sorted(symbols))}
    for sample in samples:
        bindings = {}
        for name, sym in symbols.items():
            offset = ranks[name] * _SYMBOL_SPACING
            bindings[sym] = _NUMERIC_CONSTANTS.get(name, float(sample) + offset)
        try:
            value = ak.eval_expr(expr, bindings)
        except Exception:
            continue
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):  # pragma: no cover - complex results
            continue
        evaluated += 1
        worst = magnitude if worst is None else max(worst, magnitude)
    return worst, evaluated


def _decide(residual: Any, symbols: dict[str, Any], tolerance: float, samples) -> tuple[str, str]:
    """Classify a residual that ought to be identically zero."""
    if _residual_is_zero(residual):
        return "ok", "symbolic residual simplified to 0"
    worst, evaluated = _numeric_residual(residual, symbols, samples)
    if worst is None:
        return "inconclusive", f"residual did not simplify to 0 (got {residual}); no numeric sample"
    where = f"{evaluated} of {len(samples)} sample point(s)"
    if worst <= tolerance:
        return "numeric_ok", (
            f"|residual| <= {worst:.3g} at {where}, free symbols at distinct "
            f"values (numeric evidence, not a proof)"
        )
    return "failed", f"|residual| = {worst:.6g} at {where} exceeds tolerance {tolerance:g}"


def _exact_and_uncertainty(value: Any) -> tuple[Fraction, Fraction]:
    """*value* as an exact rational, with the half-ulp its notation implies.

    A ``numeric_relation`` recipe carries its constants as text — the form
    :func:`alkahest.guess_relation`'s own docstring tells callers to use — and
    that text carries its precision with it.  ``"1"`` is the integer one and is
    exact; ``"1.15572734962273134279187535795567192711"`` names a value known
    to half a unit in its last decimal place, which is *far* more than a
    ``float`` holds.  Narrowing either to 53 bits throws that away, so an exact
    relation with coefficients around ``5e9`` picks up a ``9.5e-7`` rounding
    residual and gets refuted.

    :raises ValueError: when *value* is not a recognisable exact numeral.
    """
    if isinstance(value, bool):  # bool is an int; refuse it explicitly
        raise ValueError(f"not a numeric constant: {value!r}")
    if isinstance(value, Fraction):
        return value, Fraction(0)
    if isinstance(value, int):
        return Fraction(value), Fraction(0)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"not a finite constant: {value!r}")
        # A float names itself exactly, but only to its own resolution.
        return Fraction(value), Fraction(math.ulp(value)) / 2
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, str):
        try:
            decimal_value = Decimal(value.strip())
        except InvalidOperation:
            raise ValueError(f"not a decimal numeral: {value!r}") from None
    elif hasattr(value, "__float__"):
        # An in-process numeric of some other type (``mpmath.mpf``, ``numpy``).
        # Narrowing is what the old code did to everything; here it is the last
        # resort, and the ulp it reports says the digits were lost.
        try:
            narrowed = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"not a numeric constant: {value!r}") from None
        if not math.isfinite(narrowed):
            raise ValueError(f"not a finite constant: {value!r}")
        return Fraction(narrowed), Fraction(math.ulp(narrowed)) / 2
    else:
        raise ValueError(f"unsupported constant type {type(value).__name__}")
    if not decimal_value.is_finite():
        raise ValueError(f"not a finite constant: {value!r}")
    exponent = decimal_value.as_tuple().exponent
    text = str(value).strip() if isinstance(value, str) else str(decimal_value)
    if "." not in text and "e" not in text.lower():
        # An integer written as an integer is exact, not "±0.5".
        return Fraction(decimal_value), Fraction(0)
    ulp = Fraction(10) ** int(exponent)
    return Fraction(decimal_value), ulp / 2


def _relation_residual(
    constants: Sequence[Any], coefficients: Sequence[Any]
) -> tuple[Fraction, Fraction]:
    """``(|sum a_i c_i|, uncertainty)`` computed exactly from the given numerals.

    The uncertainty is the first-order propagation of each input's own half-ulp,
    so it is zero when every constant and coefficient is exact.  The true
    residual lies in ``[|R| - U, |R| + U]``, which is what lets
    :func:`_recheck` distinguish "this relation is false" from "you did not give
    me the digits to tell".
    """
    residual = Fraction(0)
    uncertainty = Fraction(0)
    for raw_constant, raw_coefficient in zip(constants, coefficients):
        constant, constant_ulp = _exact_and_uncertainty(raw_constant)
        coefficient, coefficient_ulp = _exact_and_uncertainty(raw_coefficient)
        residual += coefficient * constant
        uncertainty += abs(coefficient) * constant_ulp + abs(constant) * coefficient_ulp
    return abs(residual), uncertainty


def _recheck(claim: Claim, pool: Any, tolerance: float, samples: Sequence[float]) -> RecheckOutcome:
    check = claim.check
    if not check:
        return RecheckOutcome(claim.id, "skipped", "none", "no re-verification recipe recorded")
    kind = str(check.get("kind", ""))
    ak = _ak()
    symbols: dict[str, Any] = {}
    try:
        if kind == "antiderivative":
            integrand = _parse_into(pool, check["integrand"], symbols)
            antiderivative = _parse_into(pool, check["antiderivative"], symbols)
            var = _parse_into(pool, check["var"], symbols)
            derivative = ak.diff(antiderivative, var).value
            residual = ak.simplify(derivative - integrand).value
            outcome, detail = _decide(residual, symbols, tolerance, samples)
            return RecheckOutcome(claim.id, outcome, kind, detail)
        if kind == "definite_integral":
            integrand = _parse_into(pool, check["integrand"], symbols)
            var = _parse_into(pool, check["var"], symbols)
            lower = _parse_into(pool, check["lower"], symbols)
            upper = _parse_into(pool, check["upper"], symbols)
            value = _parse_into(pool, check["value"], symbols)
            antiderivative = ak.integrate(integrand, var).value
            residual = ak.simplify(
                ak.subs(antiderivative, {var: upper})
                - ak.subs(antiderivative, {var: lower})
                - value
            ).value
            outcome, detail = _decide(residual, symbols, tolerance, samples)
            return RecheckOutcome(claim.id, outcome, kind, detail)
        if kind == "derivative":
            expr = _parse_into(pool, check["expr"], symbols)
            value = _parse_into(pool, check["value"], symbols)
            var = _parse_into(pool, check["var"], symbols)
            residual = ak.simplify(ak.diff(expr, var).value - value).value
            outcome, detail = _decide(residual, symbols, tolerance, samples)
            return RecheckOutcome(claim.id, outcome, kind, detail)
        if kind == "identity":
            lhs = _parse_into(pool, check["lhs"], symbols)
            rhs = _parse_into(pool, check["rhs"], symbols)
            residual = ak.simplify(lhs - rhs).value
            outcome, detail = _decide(residual, symbols, tolerance, samples)
            return RecheckOutcome(claim.id, outcome, kind, detail)
        if kind == "zero":
            expr = _parse_into(pool, check["expr"], symbols)
            residual = ak.simplify(expr).value
            outcome, detail = _decide(residual, symbols, tolerance, samples)
            return RecheckOutcome(claim.id, outcome, kind, detail)
        if kind == "numeric_relation":
            constants = list(check["constants"])
            coefficients = list(check["coefficients"])
            if len(constants) != len(coefficients):
                return RecheckOutcome(
                    claim.id, "inconclusive", kind, "constant/coefficient length mismatch"
                )
            try:
                residual, uncertainty = _relation_residual(constants, coefficients)
            except ValueError as exc:
                return RecheckOutcome(claim.id, "inconclusive", kind, str(exc))
            bound = float(check.get("tolerance", tolerance))
            # The residual is exact; the *inputs* are only as precise as their
            # own notation, so the true value lies in [residual +- uncertainty].
            upper = float(residual + uncertainty)
            lower = float(max(Fraction(0), residual - uncertainty))
            at = f"at the supplied precision (+-{float(uncertainty):.3g})"
            if upper <= bound:
                return RecheckOutcome(
                    claim.id,
                    "numeric_ok",
                    kind,
                    f"|sum a_i c_i| <= {upper:.3g} <= {bound:g} {at} (numeric evidence only)",
                )
            if lower > bound:
                return RecheckOutcome(
                    claim.id, "failed", kind, f"|sum a_i c_i| >= {lower:.6g} > {bound:g} {at}"
                )
            return RecheckOutcome(
                claim.id,
                "inconclusive",
                kind,
                f"|sum a_i c_i| = {float(residual):.6g} {at}, which straddles the "
                f"tolerance {bound:g}: the constants were not supplied to enough "
                f"digits to decide this relation",
            )
    except Exception as exc:
        return RecheckOutcome(claim.id, "inconclusive", kind, f"{type(exc).__name__}: {exc}")
    return RecheckOutcome(claim.id, "skipped", kind, f"unknown check kind {kind!r}")


# ---------------------------------------------------------------------------
# Automatic capture hooks
# ---------------------------------------------------------------------------

_HOOK_LOCK = threading.Lock()
_HOOKED: dict[str, Any] = {}
_LOCAL = threading.local()


@contextmanager
def _suppress_capture() -> Iterator[None]:
    """Disable automatic capture for the duration of the block.

    The recorder itself calls :func:`alkahest.simplify` (to normalise
    statements) and :func:`alkahest.diff` (to re-check claims); without this
    guard those internal calls would be captured as claims of their own.
    """
    previous = getattr(_LOCAL, "suppressed", False)
    _LOCAL.suppressed = True
    try:
        yield
    finally:
        _LOCAL.suppressed = previous


def _install_hooks() -> tuple[str, ...]:
    """Install capture wrappers on the ``alkahest`` module namespace.

    Wrappers are installed **once and never removed**: uninstalling them when a
    session exits would silently disable capture for a session still running on
    another thread, and a provenance object that is quietly incomplete is worse
    than one that is obviously manual.  An installed wrapper is a no-op — one
    thread-local attribute lookup — whenever no session is recording.
    """
    with _HOOK_LOCK:
        ak = _ak()
        for name in _KNOWN_PRODUCERS:
            if name in _HOOKED:
                continue
            original = getattr(ak, name, None)
            if original is None or not callable(original) or isinstance(original, type):
                continue
            wrapper = _make_hook(name, original)
            setattr(ak, name, wrapper)
            _HOOKED[name] = original
        return tuple(sorted(_HOOKED))


def _make_hook(name: str, original: Any) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if getattr(_LOCAL, "suppressed", False):
            return result
        sessions = getattr(_LOCAL, "sessions", None)
        if sessions and _is_derived(result):
            sessions[-1]._auto_record(name, result, args, kwargs)
        return result

    # Marks the wrapper so an installed hook is identifiable by inspection.
    setattr(wrapper, "__alkahest_capture__", name)  # noqa: B010
    return wrapper


def captured_operations() -> tuple[str, ...]:
    """Operations currently wrapped for automatic capture.

    Empty until the first :func:`session` with ``capture=True`` is entered.
    Automatic capture sees calls made **through the module namespace**
    (``alkahest.integrate(...)``).  It does not see calls through a name bound
    before the hooks were installed (``from alkahest import integrate``) or
    methods on objects (``Assumptions.simplify``); use
    :meth:`ResearchSession.record` for those.
    """
    return tuple(sorted(_HOOKED))


# ---------------------------------------------------------------------------
# ResearchSession
# ---------------------------------------------------------------------------


class ResearchSession:
    """Records the results produced inside a block as a :class:`ClaimGraph`.

    Enter the session with :func:`session`.  Inside the block:

    * with ``capture=True``, every :class:`~alkahest.DerivedResult` returned by
      a module-level Alkahest operation is recorded automatically, and its
      dependency edges are inferred from the expressions it was computed from;
    * :meth:`record` registers a result explicitly (one line, always complete);
    * :meth:`conjecture` registers a claim that is *not* proved — its status is
      hard-wired to ``"unverified"``.

    Parameters
    ----------
    title : str, optional
        Document title.
    pool : ExprPool, optional
        Entered as the active :func:`alkahest.context` pool for the block.
    assumptions : Assumptions, optional
        Hypotheses attached to every claim recorded in the block, and entered
        as the active assumption context.  Defaults to whatever
        :func:`alkahest.active_assumptions` reports at record time.
    capture : bool
        Enable automatic capture (see :func:`captured_operations` for its exact
        scope).
    normalize : bool
        Put expression statements into the kernel's normal form before hashing.
        Disable for very hot loops; IDs then depend on how the expression was
        built.
    graph : ClaimGraph, optional
        Continue an existing graph — this is how iteration *N+1* cites
        iteration *N*.
    metadata : mapping, optional
        Extra session metadata stored on the graph.
    """

    def __init__(
        self,
        *,
        title: str | None = None,
        pool: Any = None,
        assumptions: Any = None,
        capture: bool = False,
        normalize: bool = True,
        graph: ClaimGraph | None = None,
        metadata: Mapping[str, Any] | None = None,
        crosscheck: bool = False,
    ) -> None:
        self.graph = graph if graph is not None else ClaimGraph(title=title)
        self.pool = pool
        self.assumptions = assumptions
        self.capture = bool(capture)
        self.normalize = bool(normalize)
        #: Run an independent-implementation cross-check as each captured
        #: operation is recorded (P2-2 design decision D8).  Deliberately a
        #: *recording-time* policy rather than an ``ak.context`` flag: an oracle
        #: round-trip costs orders of magnitude more than the kernel call, so it
        #: belongs at stage-5 frequency (hundreds of claims) and not at stage-2
        #: frequency (millions of candidates).  It never changes a claim's
        #: status — see :meth:`record`.
        self.crosscheck = bool(crosscheck)
        #: Exceptions raised *inside* the capture hook, recorded rather than
        #: swallowed so an incomplete graph is never silently incomplete.
        self.capture_errors: list[str] = []
        self._origins: dict[tuple, str] = {}
        self._pending: list[str] = []
        self._stack: ExitStack | None = None
        self._entered = False
        self._metadata = dict(metadata or {})

    # -- context manager --------------------------------------------------

    def __enter__(self) -> ResearchSession:
        ak = _ak()
        self._entered = True
        self._stack = ExitStack()
        context_kwargs: dict[str, Any] = {}
        if self.pool is not None:
            context_kwargs["pool"] = self.pool
        if self.assumptions is not None:
            context_kwargs["assumptions"] = self.assumptions
        if context_kwargs:
            self._stack.enter_context(ak.context(**context_kwargs))
        if self.capture:
            _install_hooks()
            sessions = getattr(_LOCAL, "sessions", None)
            if sessions is None:
                sessions = []
                _LOCAL.sessions = sessions
            sessions.append(self)
        self.graph.metadata.setdefault("started_at", _utcnow())
        self.graph.metadata.setdefault("alkahest_version", getattr(ak, "__version__", "unknown"))
        self.graph.metadata.setdefault("features", _build_features())
        self.graph.metadata.setdefault("schema_version", SCHEMA_VERSION)
        self.graph.metadata.update(self._metadata)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sessions = getattr(_LOCAL, "sessions", None)
        if sessions:
            with suppress(ValueError):  # pragma: no cover - defensive
                sessions.remove(self)
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._entered = False
        self.graph.metadata["finished_at"] = _utcnow()
        self.graph.metadata["capture"] = "auto" if self.capture else "explicit"
        self.graph.metadata["captured_operations"] = list(captured_operations())
        if self.capture_errors:
            self.graph.metadata["capture_errors"] = list(self.capture_errors)
        return False

    # -- recording --------------------------------------------------------

    def record(
        self,
        result: Any,
        *,
        statement: Any = None,
        method: str | None = None,
        label: str | None = None,
        hypotheses: Sequence[str] | None = None,
        depends_on: Sequence[str] = (),
        sources: Sequence[Any] = (),
        tags: Sequence[str] = (),
        notes: str | None = None,
        check: Mapping[str, Any] | None = None,
        arguments: Sequence[str] | None = None,
        crosscheck: Any = None,
    ) -> Claim:
        """Record a :class:`~alkahest.DerivedResult` (or bare ``Expr``) as a claim.

        The claim's status, evidence, and certificate are copied **verbatim**
        from ``result.verification``; this method never upgrades them.

        Parameters
        ----------
        result : DerivedResult or Expr
            The result to record.
        statement : Expr, DerivedResult, str or mapping, optional
            What the claim asserts, if different from ``result.value``.  A
            mapping of the form ``{"kind": ..., "statement": ..., "latex": ...}``
            is used verbatim, which is how relations such as
            ``∫ f dx = F`` are supplied.

            It is **free text, and nothing checks that it describes**
            ``result``.  So a machine-checked status is not carried over onto
            it: when *statement* is supplied without a *check* recipe and the
            result's status is in :data:`MACHINE_CHECKED_STATUSES`, the claim
            is stored as ``"asserted"`` instead, with the result's own status
            preserved under ``verification["result_status"]``.  Supply *check*
            — the recipe :meth:`ClaimGraph.verify` re-runs — to keep the
            machine-checked status, or record the result without *statement*.
        method : str, optional
            Operation name.  Defaults to ``"record"``.
        label : str, optional
            Short human-readable title for rendered documents.
        hypotheses : sequence of str, optional
            Overrides the hypotheses taken from the active
            :class:`~alkahest.Assumptions`.
        depends_on : sequence of str
            Explicit dependency claim IDs.
        sources : sequence
            Expressions or results this was computed from; their originating
            claims (if recorded in this session) become dependency edges.
        tags, notes : optional
            Free-form metadata.
        check : mapping, optional
            A re-verification recipe (see :meth:`ClaimGraph.verify`).
        arguments : sequence of str, optional
            Rendered arguments stored in the claim's provenance.
        crosscheck : CrossCheck, optional
            An already-computed :class:`alkahest.crosscheck.CrossCheck`,
            attached to the claim under ``verification["crosscheck"]`` and
            surfaced as a ``crosscheck:<outcome>`` tag.

            It is **evidence, never a verdict**.  Agreement with an independent
            implementation does not upgrade the status, and a divergence does
            not set ``"refuted"`` either: a divergence names two suspects, not
            one (P2-2 design decision D5), so which side is wrong is for a human
            or a later check to adjudicate.  The recording layer's honesty
            invariant — it may never raise confidence — is preserved exactly.

        Returns
        -------
        Claim
            The stored claim (merged, if an identical claim already existed).
        """
        verification = {}
        certificate = None
        derivation: tuple[dict[str, Any], ...] = ()
        if _is_derived(result):
            verification = dict(result.verification or {})
            certificate = result.certificate
            derivation = tuple(dict(step) for step in (result.steps or ()))
        cc_record = _crosscheck_record(crosscheck)
        extra_tags: tuple[str, ...] = ()
        if cc_record is not None:
            verification = dict(verification)
            verification["crosscheck"] = cc_record
            extra_tags = (f"crosscheck:{cc_record['outcome']}",)

        status = str(verification.get("status", "unverified"))
        evidence = str(verification.get("evidence", "none"))
        # A caller-supplied *statement* is free text: nothing relates it to the
        # result whose status is being copied, so `record(integrate(...),
        # statement="0 = 1")` must not inherit `exactly_verified`.  A `check`
        # recipe re-establishes the link — it is the recipe `verify()` runs
        # against the statement — so it, and an assertion the engine rendered
        # itself (`_infer_assertion`), keep the status.  Everything else is
        # badged `"asserted"` until a recipe is attached.
        if (
            statement is not None
            and not (isinstance(statement, dict) and statement.get("inferred"))
            and not check
            and status in MACHINE_CHECKED_STATUSES
        ):
            verification = dict(verification)
            verification["result_status"] = status
            verification["statement_source"] = "caller"
            status = "asserted"
        certificate_format = verification.get("artifact_format")
        if certificate is not None and certificate_format is None:
            certificate_format = "lean4"

        normalized = _normalize_statement(
            statement if statement is not None else result, normalize=self.normalize
        )
        resolved_hypotheses = (
            tuple(sorted(_canonical_text(h) for h in hypotheses))
            if hypotheses is not None
            else self._hypotheses()
        )
        resolved_method = method or "record"
        cid = claim_id(normalized["statement"], resolved_hypotheses, resolved_method)

        edges = list(depends_on) + list(self._pending)
        for source in sources:
            edges.extend(self._origins_in(source))
        self._pending.clear()

        claim = Claim(
            id=cid,
            statement=normalized["statement"],
            kind=normalized["kind"],
            latex=normalized["latex"],
            hypotheses=resolved_hypotheses,
            method=resolved_method,
            status=status,
            evidence=evidence,
            verification=verification,
            derivation=derivation,
            certificate=certificate,
            certificate_format=certificate_format if certificate is not None else None,
            depends_on=tuple(dict.fromkeys(edges)),
            check=dict(check) if check else None,
            provenance=self._provenance(resolved_method, arguments),
            recorded_at=_utcnow(),
            label=label,
            tags=tuple(dict.fromkeys((*tags, *extra_tags))),
            notes=notes,
        )
        stored = self.graph.add(claim)
        self._register_origin(result, stored.id)
        if statement is not None:
            self._register_origin(statement, stored.id)
        return stored

    def conjecture(
        self,
        statement: Any,
        *,
        evidence: str,
        method: str = "conjecture",
        label: str | None = None,
        hypotheses: Sequence[str] | None = None,
        depends_on: Sequence[str] = (),
        sources: Sequence[Any] = (),
        tags: Sequence[str] = (),
        notes: str | None = None,
        check: Mapping[str, Any] | None = None,
    ) -> Claim:
        """Record a claim that is **not** proved.

        The status is hard-wired to ``"unverified"`` — there is deliberately no
        parameter to say otherwise.  *evidence* is free text describing why the
        conjecture is plausible (e.g. ``"PSLQ at 60 digits"``); it is rendered
        as supporting evidence, never as a proof.

        Returns
        -------
        Claim
        """
        normalized = _normalize_statement(statement, normalize=self.normalize)
        resolved_hypotheses = (
            tuple(sorted(_canonical_text(h) for h in hypotheses))
            if hypotheses is not None
            else self._hypotheses()
        )
        cid = claim_id(normalized["statement"], resolved_hypotheses, method)
        edges = list(depends_on) + list(self._pending)
        for source in sources:
            edges.extend(self._origins_in(source))
        self._pending.clear()
        claim = Claim(
            id=cid,
            statement=normalized["statement"],
            kind=normalized["kind"],
            latex=normalized["latex"],
            hypotheses=resolved_hypotheses,
            method=method,
            status="unverified",
            evidence=_canonical_text(evidence),
            verification={"status": "unverified", "evidence": _canonical_text(evidence)},
            depends_on=tuple(dict.fromkeys(edges)),
            check=dict(check) if check else None,
            provenance=self._provenance(method, None),
            recorded_at=_utcnow(),
            label=label,
            tags=tuple(tags),
            notes=notes,
        )
        stored = self.graph.add(claim)
        self._register_origin(statement, stored.id)
        return stored

    def cite(self, claim: str | Claim) -> Claim:
        """Mark a claim as a dependency of the next result recorded.

        Lets iteration *N+1* cite iteration *N* by ID even when the two are not
        linked by a shared expression object.

        Raises
        ------
        MissingClaimError
            If the ID is not in this session's graph.
        """
        cid = claim.id if isinstance(claim, Claim) else str(claim)
        stored = self.graph[cid]
        self._pending.append(cid)
        return stored

    def run(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call *operation* and record its result, returning the result.

        The explicit, always-complete alternative to automatic capture:
        dependency edges are inferred from *args* exactly as they would be by
        the capture hook.

        >>> import alkahest as ak                      # doctest: +SKIP
        >>> value = s.run(ak.integrate, f, x)          # doctest: +SKIP
        """
        result = operation(*args, **kwargs)
        if _is_derived(result):
            name = getattr(operation, "__name__", "run")
            self._record_operation(name, result, args, kwargs)
        return result

    def capture_report(self) -> dict[str, Any]:
        """What automatic capture is doing, and anything it could not do.

        Returns a mapping with ``mode``, the ``operations`` currently hooked,
        the number of ``claims`` recorded, and any ``errors`` raised inside the
        hook.
        """
        return {
            "mode": "auto" if self.capture else "explicit",
            "operations": list(captured_operations()),
            "claims": len(self.graph),
            "errors": list(self.capture_errors),
        }

    # -- internals --------------------------------------------------------

    def _auto_record(self, name: str, result: Any, args: tuple, kwargs: dict) -> None:
        try:
            self._record_operation(name, result, args, kwargs)
        except Exception as exc:
            self.capture_errors.append(f"{name}: {type(exc).__name__}: {exc}")

    def _record_operation(self, name: str, result: Any, args: tuple, kwargs: dict) -> Claim:
        sources = [a for a in args if _as_expr(a) is not None]
        sources += [v for v in kwargs.values() if _as_expr(v) is not None]
        rendered = [str(_as_expr(s)) for s in sources]
        return self.record(
            result,
            statement=_infer_assertion(name, sources, result, normalize=self.normalize),
            method=name,
            sources=sources,
            arguments=rendered,
            check=_infer_check(name, sources, result),
            crosscheck=_run_crosscheck(name, args, kwargs) if self.crosscheck else None,
        )

    def _hypotheses(self) -> tuple[str, ...]:
        context = self.assumptions
        if context is None:
            try:
                context = _ak().active_assumptions()
            except Exception:
                context = None
        if context is None:
            return ()
        try:
            predicates = list(context.predicates)
        except Exception:
            return ()
        return tuple(sorted(_canonical_text(str(p)) for p in predicates))

    def _provenance(self, method: str, arguments: Sequence[str] | None) -> dict[str, Any]:
        ak = _ak()
        provenance: dict[str, Any] = {
            "operation": method,
            "alkahest_version": getattr(ak, "__version__", "unknown"),
            "features": _build_features(),
        }
        if arguments:
            provenance["arguments"] = list(arguments)
        return provenance

    def _register_origin(self, obj: Any, cid: str) -> None:
        expr = _as_expr(obj)
        key = _expr_key(expr)
        if key is not None:
            self._origins[key] = cid

    def _origins_in(self, obj: Any) -> list[str]:
        """Claim IDs produced by *obj* or by any of its subexpressions.

        This is how dependency edges are inferred: a result computed from an
        expression that a previous claim produced cites that claim, even when
        the earlier value is buried inside a larger expression.
        """
        expr = _as_expr(obj)
        if expr is None or not self._origins:
            return []
        found: dict[str, None] = {}
        for sub in _subexpressions(expr):
            key = _expr_key(sub)
            if key is None:
                continue
            origin = self._origins.get(key)
            if origin is not None:
                found[origin] = None
        return list(found)


@functools.lru_cache(maxsize=1)
def _build_features_cached() -> tuple[tuple[str, bool], ...]:
    try:
        features = _ak().capabilities().get("features", {})
    except Exception:
        return ()
    return tuple((str(k), bool(v)) for k, v in sorted(features.items()))


def _build_features() -> dict[str, Any]:
    """Sorted build-feature map from :func:`alkahest.capabilities` (fresh copy)."""
    return dict(_build_features_cached())


def _infer_check(name: str, sources: Sequence[Any], result: Any) -> dict[str, Any] | None:
    """Derive a re-verification recipe from the operation and its arguments."""
    exprs = [_as_expr(s) for s in sources]
    exprs = [e for e in exprs if e is not None]
    value = _as_expr(result)
    if value is None:
        return None
    if name == "integrate" and len(exprs) == 2:
        return {
            "kind": "antiderivative",
            "integrand": str(exprs[0]),
            "var": str(exprs[1]),
            "antiderivative": str(value),
        }
    if name == "integrate" and len(exprs) == 4:
        return {
            "kind": "definite_integral",
            "integrand": str(exprs[0]),
            "var": str(exprs[1]),
            "lower": str(exprs[2]),
            "upper": str(exprs[3]),
            "value": str(value),
        }
    if name in {"diff", "diff_forward"} and len(exprs) == 2:
        return {
            "kind": "derivative",
            "expr": str(exprs[0]),
            "var": str(exprs[1]),
            "value": str(value),
        }
    if name in _IDENTITY_OPS and len(exprs) >= 1:
        return {"kind": "identity", "lhs": str(exprs[0]), "rhs": str(value)}
    return None


def session(
    *,
    title: str | None = None,
    pool: Any = None,
    assumptions: Any = None,
    capture: bool = False,
    normalize: bool = True,
    graph: ClaimGraph | None = None,
    metadata: Mapping[str, Any] | None = None,
    crosscheck: bool = False,
) -> ResearchSession:
    """Open a research session (see :class:`ResearchSession`).

    >>> import alkahest as ak
    >>> pool = ak.ExprPool()
    >>> x = pool.symbol("x")
    >>> with ak.research.session(title="Demo", pool=pool, capture=True) as s:
    ...     _ = ak.integrate(x, x)
    >>> len(s.graph)
    1
    """
    return ResearchSession(
        title=title,
        pool=pool,
        assumptions=assumptions,
        capture=capture,
        normalize=normalize,
        graph=graph,
        metadata=metadata,
        crosscheck=crosscheck,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _render_markdown(
    graph: ClaimGraph,
    *,
    include_derivations: bool,
    include_certificates: bool,
    max_steps: int,
) -> str:
    metadata = graph.metadata
    title = graph.title or "Alkahest research record"
    total = len(graph)
    checked = len(graph.machine_checkable())
    lines: list[str] = [f"# {title}", ""]
    version = metadata.get("alkahest_version", "unknown")
    lines.append(
        f"*Alkahest {version} · claim-graph schema v{SCHEMA_VERSION} · "
        f"{total} claim{'s' if total != 1 else ''} · digest `{graph.digest()[:16]}`*"
    )
    lines.append("")

    lines.append("## Verification summary")
    lines.append("")
    lines.append("| Status | Claims | What it means |")
    lines.append("| --- | ---: | --- |")
    for status, count in graph.summary().items():
        lines.append(f"| `{status}` | {count} | {_md_escape(STATUS_BADGES.get(status, status))} |")
    lines.append("")
    if total:
        percent = 100.0 * checked / total
        lines.append(
            f"> **Machine-checkable subset: {checked} of {total} claims ({percent:.0f}%).** "
            "Only claims marked *verified* were checked by a checker. Everything else is "
            "recorded evidence and must not be read as proved."
        )
    else:
        lines.append("> No claims recorded.")
    lines.append("")

    errors = metadata.get("capture_errors")
    if errors:
        lines.append(
            "> **Capture warnings** — the following results could not be recorded, "
            "so this document is incomplete:"
        )
        lines.append(">")
        for err in errors:
            lines.append(f"> - `{_md_escape(err)}`")
        lines.append("")

    lines.append("## Claims")
    lines.append("")
    for index, claim in enumerate(graph.claims, start=1):
        heading = claim.label or claim.method or f"Claim {index}"
        lines.append(f'### {index}. {heading} <a id="{claim.id}"></a>')
        lines.append("")
        lines.append(f"**Status:** {claim.mark} `{claim.status}` — {claim.badge}")
        lines.append("")
        if claim.latex:
            lines.append("$$")
            lines.append(claim.latex)
            lines.append("$$")
            lines.append("")
        lines.append(f"- **Statement:** `{_md_escape(claim.statement)}`")
        lines.append(f"- **Claim ID:** `{claim.id}`")
        lines.append(f"- **Method:** `{claim.method}`")
        if claim.hypotheses:
            rendered = ", ".join(f"`{_md_escape(h)}`" for h in claim.hypotheses)
            lines.append(f"- **Hypotheses:** {rendered}")
        else:
            lines.append("- **Hypotheses:** none recorded (claim asserted unconditionally)")
        if claim.evidence and claim.evidence != "none":
            lines.append(f"- **Evidence:** {_md_escape(claim.evidence)}")
        if claim.depends_on:
            links = ", ".join(f"[`{d}`](#{d})" for d in claim.depends_on)
            lines.append(f"- **Depends on:** {links}")
        dependents = graph.dependents(claim.id)
        if dependents:
            links = ", ".join(f"[`{d}`](#{d})" for d in dependents)
            lines.append(f"- **Cited by:** {links}")
        if claim.certificate:
            fmt = claim.certificate_format or "unknown"
            lines.append(
                f"- **Certificate:** {len(claim.certificate)} bytes of `{fmt}` source emitted "
                "(generated, not machine-checked)"
            )
        else:
            lines.append("- **Certificate:** none emitted")
        if claim.notes:
            lines.append(f"- **Notes:** {_md_escape(claim.notes)}")
        if claim.audit:
            last = claim.audit[-1]
            lines.append(
                f"- **Last re-check:** `{last.get('outcome')}` via `{last.get('kind')}` "
                f"on Alkahest {last.get('alkahest_version')} — {_md_escape(last.get('detail', ''))}"
            )
        lines.append("")
        if include_derivations and claim.derivation:
            steps = claim.derivation[:max_steps]
            lines.append(
                "<details><summary>Derivation "
                f"({len(claim.derivation)} step"
                f"{'s' if len(claim.derivation) != 1 else ''})</summary>"
            )
            lines.append("")
            lines.append("| # | Rule | Before | After |")
            lines.append("| ---: | --- | --- | --- |")
            for number, step in enumerate(steps, start=1):
                lines.append(
                    f"| {number} | `{_md_escape(step.get('rule', ''))}` "
                    f"| `{_md_escape(step.get('before', ''))}` "
                    f"| `{_md_escape(step.get('after', ''))}` |"
                )
            if len(claim.derivation) > max_steps:
                lines.append(f"| … | *{len(claim.derivation) - max_steps} more steps* | | |")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if include_certificates and claim.certificate:
            lines.append(f"```{claim.certificate_format or ''}")
            lines.append(claim.certificate.rstrip())
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def _render_latex(graph: ClaimGraph, *, standalone: bool, include_derivations: bool) -> str:
    metadata = graph.metadata
    title = graph.title or "Alkahest research record"
    total = len(graph)
    checked = len(graph.machine_checkable())
    lines: list[str] = []
    if standalone:
        lines += [
            r"\documentclass[11pt]{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage[colorlinks=true,linkcolor=blue]{hyperref}",
            rf"\title{{{_tex_escape(title)}}}",
            rf"\author{{Alkahest {_tex_escape(metadata.get('alkahest_version', 'unknown'))}}}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            "",
        ]
    else:
        lines.append(rf"\section*{{{_tex_escape(title)}}}")
        lines.append("")

    lines.append(r"\section*{Verification summary}")
    lines.append(r"\begin{tabular}{lrl}")
    lines.append(r"\textbf{Status} & \textbf{Claims} & \textbf{Meaning} \\ \hline")
    for status, count in graph.summary().items():
        meaning = _tex_escape(STATUS_BADGES.get(status, status))
        lines.append(rf"\texttt{{{_tex_escape(status)}}} & {count} & {meaning} \\")
    lines.append(r"\end{tabular}")
    lines.append("")
    if total:
        percent = 100.0 * checked / total
        lines.append(
            rf"\noindent\textbf{{Machine-checkable subset: {checked} of {total} claims "
            rf"({percent:.0f}\%).}} Only claims marked \emph{{verified}} were checked by a "
            r"checker; everything else is recorded evidence and must not be read as proved."
        )
        lines.append("")

    lines.append(r"\section*{Claims}")
    for index, claim in enumerate(graph.claims, start=1):
        heading = _tex_escape(claim.label or claim.method or f"Claim {index}")
        lines.append(rf"\subsection*{{{index}. {heading}}}")
        lines.append(rf"\label{{clm:{claim.id}}}")
        if claim.latex:
            lines.append(r"\[" + claim.latex + r"\]")
        else:
            lines.append(rf"\texttt{{{_tex_escape(claim.statement)}}}")
        lines.append(r"\begin{itemize}")
        lines.append(
            rf"\item \textbf{{Status:}} {_tex_escape(claim.mark)} "
            rf"\texttt{{{_tex_escape(claim.status)}}} -- {_tex_escape(claim.badge)}"
        )
        lines.append(rf"\item \textbf{{Claim ID:}} \texttt{{{_tex_escape(claim.id)}}}")
        lines.append(rf"\item \textbf{{Method:}} \texttt{{{_tex_escape(claim.method)}}}")
        if claim.hypotheses:
            rendered = ", ".join(rf"\texttt{{{_tex_escape(h)}}}" for h in claim.hypotheses)
            lines.append(rf"\item \textbf{{Hypotheses:}} {rendered}")
        else:
            lines.append(r"\item \textbf{Hypotheses:} none recorded")
        if claim.evidence and claim.evidence != "none":
            lines.append(rf"\item \textbf{{Evidence:}} {_tex_escape(claim.evidence)}")
        if claim.depends_on:
            links = ", ".join(
                rf"\hyperref[clm:{d}]{{\texttt{{{_tex_escape(d)}}}}}" for d in claim.depends_on
            )
            lines.append(rf"\item \textbf{{Depends on:}} {links}")
        if claim.certificate:
            fmt = _tex_escape(claim.certificate_format or "unknown")
            lines.append(
                rf"\item \textbf{{Certificate:}} {len(claim.certificate)} bytes of "
                rf"\texttt{{{fmt}}} source emitted (generated, not machine-checked)"
            )
        else:
            lines.append(r"\item \textbf{Certificate:} none emitted")
        if include_derivations and claim.derivation:
            steps = ", ".join(
                rf"\texttt{{{_tex_escape(step.get('rule', ''))}}}" for step in claim.derivation
            )
            lines.append(rf"\item \textbf{{Derivation:}} {len(claim.derivation)} step(s): {steps}")
        lines.append(r"\end{itemize}")
        lines.append("")
    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines).rstrip() + "\n"
