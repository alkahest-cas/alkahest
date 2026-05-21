"""Post-process Jupyter kernel outputs for Lean certificate extraction."""

from __future__ import annotations

import json
from typing import Any

from playground_helpers import AK_LEAN_MARKER, AK_LEAN_MIME

AK_LEAN_MIME_ALT = "application/vnd.alkahest.lean+json"


def _lean_output_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("kind") != "lean_certificate":
        return None
    source = payload.get("source")
    if not source or not isinstance(source, str):
        return None
    out: dict[str, Any] = {
        "type": "lean",
        "source": source,
    }
    if payload.get("operation"):
        out["operation"] = payload["operation"]
    if payload.get("steps") is not None:
        out["steps"] = payload["steps"]
    return out


def _split_stdout_lean(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract ``__AK_LEAN_CERT__`` lines from stdout; return cleaned text + lean items."""
    lean_items: list[dict[str, Any]] = []
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(AK_LEAN_MARKER):
            raw = stripped[len(AK_LEAN_MARKER) :]
            try:
                payload = json.loads(raw)
                item = _lean_output_from_payload(payload)
                if item:
                    lean_items.append(item)
                continue
            except json.JSONDecodeError:
                pass
        kept.append(line)
    cleaned = "".join(kept)
    if cleaned.endswith("\n\n"):
        cleaned = cleaned.rstrip("\n") + "\n"
    return cleaned, lean_items


def classify_rich(data: dict[str, str]) -> dict | None:
    """Pick the best MIME type from a rich display data dict."""
    for mime in (AK_LEAN_MIME, AK_LEAN_MIME_ALT):
        if mime in data:
            raw = data[mime]
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
                item = _lean_output_from_payload(payload)
                if item:
                    return item
            except (json.JSONDecodeError, TypeError):
                pass

    if "text/latex" in data:
        return {"type": "latex", "latex": data["text/latex"]}
    if "image/png" in data:
        return {"type": "image", "format": "png", "data": data["image/png"]}
    if "image/svg+xml" in data:
        return {"type": "image", "format": "svg", "data": data["image/svg+xml"]}
    if "text/html" in data:
        return {"type": "html", "html": data["text/html"]}
    if "application/json" in data:
        raw = data["application/json"]
        return {"type": "json", "data": json.loads(raw) if isinstance(raw, str) else raw}
    if "text/plain" in data:
        return {"type": "text", "stream": "stdout", "text": data["text/plain"]}
    return None


def postprocess_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split lean markers from text streams and normalize lean MIME items."""
    result: list[dict[str, Any]] = []
    for out in outputs:
        if out.get("type") == "text" and out.get("stream") in ("stdout", "stderr"):
            text = out.get("text") or ""
            cleaned, lean_items = _split_stdout_lean(text)
            if lean_items:
                result.extend(lean_items)
            if cleaned.strip():
                result.append({**out, "text": cleaned})
            elif not lean_items:
                result.append(out)
        else:
            result.append(out)
    return result
