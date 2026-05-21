"""
Helpers for emitting Lean certificates in the demo playground notebook.

Import in cells::

    from playground_helpers import display_lean_cert

Or rely on ``result.certificate`` / ``alkahest.to_lean(result)`` and print the marker.
"""

from __future__ import annotations

import json
from typing import Any

AK_LEAN_MARKER = "__AK_LEAN_CERT__"
AK_LEAN_MIME = "application/x-alkahest-lean+json"


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
    steps = len(getattr(result, "steps", []) or [])
    op = operation or "compute"
    payload = lean_cert_payload(cert, operation=op, steps=steps)

    try:
        from IPython.display import display

        display({AK_LEAN_MIME: json.dumps(payload)}, raw=True)
    except ImportError:
        emit_lean_marker(cert, operation=op, steps=steps)
