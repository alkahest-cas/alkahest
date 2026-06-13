"""
Helpers for emitting Lean certificates in the demo playground notebook.

Import in cells::

    from playground_helpers import display_lean_cert

Or rely on ``result.certificate`` / ``alkahest.to_lean(result)`` and print the marker.
"""

from __future__ import annotations

import json
import re
from typing import Any

AK_LEAN_MARKER = "__AK_LEAN_CERT__"
AK_LEAN_MIME = "application/x-alkahest-lean+json"

_DIFF_HEADER = (
    "import Mathlib.Tactic\n"
    "import Mathlib.Analysis.Calculus.Deriv.Basic\n"
    "import Mathlib.Analysis.Calculus.Deriv.Pow\n"
    "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv\n"
    "import Mathlib.Analysis.SpecialFunctions.ExpDeriv\n"
    "import Mathlib.Analysis.SpecialFunctions.Log.Deriv\n"
    "\n"
    "open Real\n\n"
)

# Close goals when Alkahest's canonical mul order differs from Mathlib (e.g. x^2 * 3 vs 3 * x^2).
_DIFF_RING = "; ring"

_DIFF_RULE_TACTICS: dict[str, str] = {
    "diff_identity": "by simp [deriv_id]",
    "diff_const": "by simp [deriv_const]",
    "diff_univariate_poly": f"by simp [deriv_pow, deriv_add, deriv_mul, deriv_const]{_DIFF_RING}",
    "sum_rule": f"by simp [deriv_add]{_DIFF_RING}",
    "product_rule": f"by simp [deriv_mul]{_DIFF_RING}",
    "power_rule": f"by simp [deriv_pow, deriv_mul]{_DIFF_RING}",
    "power_rule_n1": f"by simp [deriv_pow, deriv_mul]{_DIFF_RING}",
    "power_rule_n0": "by simp [deriv_const]",
}

_DIFF_STEP_RULES = frozenset(_DIFF_RULE_TACTICS) | frozenset(
    {
        "diff_sin",
        "diff_cos",
        "diff_exp",
        "diff_log",
        "diff_sqrt",
        "diff_forward",
        "diff_primitive_registry",
        "diff_piecewise",
        "diff_root_sum",
    }
)

_STEP_BLOCK_RE = re.compile(
    r"(-- Step (?P<num>\d+): (?P<rule>\S+)\n)"
    r"example : (?P<lhs>.+?) = (?P<rhs>.+?) :=\n"
    r"  (?P<tactic>.+?)(?=\n\n|-- Step |\Z)",
    re.DOTALL,
)


def fix_legacy_diff_lean(source: str) -> str:
    """
    Upgrade pre-wrt Lean certificates from older alkahest wheels.

    Older releases emitted ``x^3 = 3*x^2`` for ``diff(x**3, x)`` instead of a
    ``deriv`` goal.  The playground server rewrites those certificates so
    ``lake env lean`` can typecheck them with Mathlib's derivative lemmas.
    """
    if "deriv (fun" in source:
        return source
    if not any(
        "-- Step" in line and rule in line for line in source.splitlines() for rule in _DIFF_STEP_RULES
    ):
        return source

    var = "x"
    var_match = re.search(r"\(([A-Za-z_]\w*) : ℝ\)", source)
    if var_match:
        var = var_match.group(1)

    blocks: list[str] = []
    for match in _STEP_BLOCK_RE.finditer(source):
        rule = match.group("rule")
        if rule not in _DIFF_STEP_RULES:
            continue
        lhs = match.group("lhs").strip()
        rhs = match.group("rhs").strip()
        tactic = _DIFF_RULE_TACTICS.get(rule, "by sorry")
        blocks.append(
            f"-- Step {match.group('num')}: {rule}\n"
            f"example : deriv (fun ({var} : ℝ) => {lhs}) {var} = {rhs} :=\n"
            f"  {tactic}\n"
        )

    if not blocks:
        return source

    return _DIFF_HEADER + "\n".join(blocks) + "\n"


def lean_cert_payload(
    source: str,
    *,
    operation: str | None = None,
    steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "lean_certificate",
        "source": source,
    }
    if operation is not None:
        payload["operation"] = operation
    if steps is not None:
        payload["steps"] = steps
    if extra:
        payload.update(extra)
    return payload


def emit_lean_marker(
    source: str,
    *,
    operation: str | None = None,
    steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print a line the playground server parses into a Lean certificate output."""
    payload = lean_cert_payload(source, operation=operation, steps=steps, extra=extra)
    print(AK_LEAN_MARKER + json.dumps(payload))


def display_lean_cert(result, operation: str | None = None) -> None:
    """
    Emit a Lean certificate for a :class:`alkahest.DerivedResult`.

    Uses IPython rich display when available, else the stdout marker.
    """
    import alkahest

    cert = getattr(result, "certificate", None)
    if cert is None:
        cert = alkahest.to_lean(result)
    op = operation or "compute"
    if op == "diff" or "diff_" in cert:
        cert = fix_legacy_diff_lean(cert)
    steps = len(getattr(result, "steps", []) or [])
    payload = lean_cert_payload(cert, operation=op, steps=steps)

    try:
        from IPython.display import display

        display({AK_LEAN_MIME: json.dumps(payload)}, raw=True)
    except ImportError:
        emit_lean_marker(cert, operation=op, steps=steps)
