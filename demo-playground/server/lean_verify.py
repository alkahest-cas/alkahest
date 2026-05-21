"""Lean 4 certificate verification for the demo playground server."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent.parent
LEAN_PROJECT = REPO_ROOT / "lean"
LEAN_TOOLCHAIN = LEAN_PROJECT / "lean-toolchain"

# Allow override for deployments where the repo lives elsewhere
LEAN_PROJECT_OVERRIDE = os.environ.get("ALKAHEST_LEAN_PROJECT", "").strip()


def lean_project_dir() -> Path:
    if LEAN_PROJECT_OVERRIDE:
        return Path(LEAN_PROJECT_OVERRIDE)
    return LEAN_PROJECT


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def lean_status() -> dict:
    """Report whether Lean verification is available on this host."""
    project = lean_project_dir()
    lake = _which("lake")
    lean = _which("lean")
    elan = _which("elan")
    project_ok = project.is_dir() and (project / "lakefile.lean").is_file()
    return {
        "available": bool(lake and lean and project_ok),
        "lake": lake,
        "lean": lean,
        "elan": elan,
        "project_dir": str(project),
        "project_exists": project_ok,
        "toolchain_file": str(LEAN_TOOLCHAIN) if LEAN_TOOLCHAIN.is_file() else None,
    }


def verify_lean_source(source: str, timeout_sec: int = 120) -> dict:
    """
    Typecheck a generated .lean file with ``lake env lean``.

    Returns ``{ok, stdout, stderr, duration_ms, proof_file}``.
    """
    status = lean_status()
    if not status["available"]:
        return {
            "ok": False,
            "stdout": "",
            "stderr": (
                "Lean verification unavailable. Install elan + lake, ensure Mathlib is set up "
                f"in {status['project_dir']} (see repo lean/ and CONTRIBUTING.md)."
            ),
            "duration_ms": 0,
            "proof_file": None,
        }

    project = lean_project_dir()
    t0 = time.monotonic()
    proof_path: str | None = None

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".lean",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(source)
        proof_path = f.name

    try:
        proc = subprocess.run(
            ["lake", "env", "lean", proof_path],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": duration_ms,
            "proof_file": os.path.basename(proof_path),
        }
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        return {
            "ok": False,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\nTimed out after {timeout_sec}s",
            "duration_ms": duration_ms,
            "proof_file": os.path.basename(proof_path) if proof_path else None,
        }
    except FileNotFoundError as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(e),
            "duration_ms": 0,
            "proof_file": None,
        }
    finally:
        if proof_path:
            Path(proof_path).unlink(missing_ok=True)
