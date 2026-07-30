"""Per-skill isolated execution environments.

The single most important property of this benchmark is that an agent given the
*sympy* skill cannot score points by importing alkahest, and vice versa.  The
old harness executed generated code with ``sys.executable`` in an environment
where every CAS was importable, which made every cell of the results table
uninterpretable.

This module gives each skill its own virtualenv containing **only** that skill's
library.  Cross-library use then fails at import time rather than being silently
rewarded.  A static AST check runs first so that the failure is reported as a
clean ``wrong_library`` outcome instead of an opaque ``ImportError``.

Environments are created under ``agent-benchmark/.envs/<skill>/`` and reused
across runs.  Creating them needs network access, so it is an explicit step
(``run.py --setup-envs``) rather than something that happens implicitly mid-run.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent

ENV_ROOT = _HERE / ".envs"

# Modules every arm is allowed to use regardless of skill: the standard library
# is not enumerated here (it is always allowed), only third-party helpers that
# are not themselves computer algebra systems.
_COMMON_ALLOWED = frozenset({"numpy", "mpmath"})

# Any import of these names from the *wrong* arm is a contamination attempt.
_CAS_MODULES = frozenset(
    {
        "alkahest",
        "sympy",
        "wolframclient",
        "sage",
        "sagemath",
        "giacpy",
        "symengine",
        "maxima",
    }
)


@dataclass(frozen=True)
class SkillSpec:
    """Everything needed to build, probe, and police one benchmark arm."""

    name: str
    skill_file: Path
    # pip requirement strings installed into this arm's venv.
    packages: tuple[str, ...]
    # Modules this arm is permitted to import (beyond stdlib + _COMMON_ALLOWED).
    allowed: frozenset[str]
    # Short description used in reports.
    description: str
    # Python snippet printing a JSON dict of provenance for this arm.
    probe: str = "import json; print(json.dumps({}))"

    @property
    def forbidden(self) -> frozenset[str]:
        return _CAS_MODULES - self.allowed


_ALKAHEST_PROBE = """
import json, sys
info = {"python": sys.version.split()[0]}
try:
    import alkahest as ak
    info["alkahest_version"] = getattr(ak, "__version__", "?")
    try:
        info["capabilities"] = ak.capabilities()
    except Exception as exc:
        info["capabilities_error"] = repr(exc)
except Exception as exc:
    info["import_error"] = repr(exc)
print(json.dumps(info, default=str))
"""

_SYMPY_PROBE = """
import json, sys
info = {"python": sys.version.split()[0]}
try:
    import sympy
    info["sympy_version"] = sympy.__version__
except Exception as exc:
    info["import_error"] = repr(exc)
print(json.dumps(info, default=str))
"""

_MATHEMATICA_PROBE = """
import json, sys, shutil
info = {"python": sys.version.split()[0]}
try:
    import wolframclient
    info["wolframclient_version"] = getattr(wolframclient, "__version__", "?")
except Exception as exc:
    info["import_error"] = repr(exc)
info["wolframscript"] = shutil.which("wolframscript")
info["kernel"] = shutil.which("WolframKernel")
try:
    from wolframclient.evaluation import WolframLanguageSession
    s = WolframLanguageSession()
    s.start()
    try:
        from wolframclient.language import wlexpr
        info["kernel_ok"] = s.evaluate(wlexpr("1+1")) == 2
    finally:
        s.terminate()
except Exception as exc:
    info["kernel_ok"] = False
    info["kernel_error"] = repr(exc)
print(json.dumps(info, default=str))
"""

_NONE_PROBE = """
import json, sys
info = {"python": sys.version.split()[0]}
try:
    import numpy
    info["numpy_version"] = numpy.__version__
except Exception as exc:
    info["import_error"] = repr(exc)
print(json.dumps(info, default=str))
"""


def build_registry(alkahest_spec: str = "alkahest") -> dict[str, SkillSpec]:
    """Return the skill registry.

    *alkahest_spec* is the pip requirement used for the alkahest arm.  It
    defaults to the published PyPI wheel — which is what a prospective user
    would actually install — but can be pointed at a local wheel or a pinned
    version for reproducibility.
    """
    return {
        "alkahest": SkillSpec(
            name="alkahest",
            skill_file=_REPO_ROOT / "alkahest-skill" / "alkahest.md",
            packages=(alkahest_spec, "numpy"),
            allowed=frozenset({"alkahest"}),
            description="Alkahest (this library)",
            probe=_ALKAHEST_PROBE,
        ),
        "sympy": SkillSpec(
            name="sympy",
            skill_file=_HERE / "skills" / "sympy.md",
            packages=("sympy", "numpy"),
            allowed=frozenset({"sympy"}),
            description="SymPy",
            probe=_SYMPY_PROBE,
        ),
        "mathematica": SkillSpec(
            name="mathematica",
            skill_file=_HERE / "skills" / "mathematica.md",
            packages=("wolframclient", "numpy"),
            allowed=frozenset({"wolframclient"}),
            description="Wolfram Engine via wolframclient",
            probe=_MATHEMATICA_PROBE,
        ),
        "none": SkillSpec(
            name="none",
            skill_file=_HERE / "skills" / "none.md",
            packages=("numpy",),
            allowed=frozenset(),
            description="Control arm: no CAS, plain Python + NumPy",
            probe=_NONE_PROBE,
        ),
    }


# ---------------------------------------------------------------------------
# Static import checking
# ---------------------------------------------------------------------------


def imported_modules(code: str) -> set[str]:
    """Return the set of *top-level* module names imported by *code*.

    Returns an empty set when the code does not parse; a syntax error is caught
    later by execution and reported as its own outcome.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        # `from . import x` has module None; relative imports cannot reach
        # a third-party CAS, so they are ignored.
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def check_imports(code: str, spec: SkillSpec) -> str | None:
    """Return a violation message if *code* imports a CAS this arm may not use."""
    used = imported_modules(code)
    violations = sorted(used & spec.forbidden)
    if violations:
        return f"used forbidden library: {', '.join(violations)}"
    return None


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def env_dir(skill: str, root: Path = ENV_ROOT) -> Path:
    return root / skill


def env_python(skill: str, root: Path = ENV_ROOT) -> Path:
    d = env_dir(skill, root)
    if sys.platform == "win32":
        return d / "Scripts" / "python.exe"
    return d / "bin" / "python"


def env_exists(skill: str, root: Path = ENV_ROOT) -> bool:
    return env_python(skill, root).exists()


def create_env(
    spec: SkillSpec,
    root: Path = ENV_ROOT,
    *,
    recreate: bool = False,
    quiet: bool = False,
) -> Path:
    """Create (or reuse) the venv for *spec* and return its interpreter path."""
    d = env_dir(spec.name, root)
    py = env_python(spec.name, root)

    if recreate and d.exists():
        shutil.rmtree(d)

    if not py.exists():
        if not quiet:
            print(f"  creating venv for {spec.name} …", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(d)],
            check=True,
            capture_output=True,
        )

    if not quiet:
        print(f"  installing {', '.join(spec.packages) or '(nothing)'} …", flush=True)
    if spec.packages:
        proc = subprocess.run(
            [str(py), "-m", "pip", "install", "-q", "--upgrade", *spec.packages],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pip install failed for skill '{spec.name}':\n{proc.stderr.strip()}"
            )
    return py


def probe_env(spec: SkillSpec, root: Path = ENV_ROOT) -> dict:
    """Run the arm's provenance probe and return its parsed output."""
    py = env_python(spec.name, root)
    if not py.exists():
        return {"error": "environment not created"}
    proc = subprocess.run(
        [str(py), "-c", spec.probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-500:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"unparseable probe output: {proc.stdout[:200]}"}


def arm_available(spec: SkillSpec, provenance: dict) -> tuple[bool, str]:
    """Decide whether an arm can produce meaningful results.

    The Wolfram arm in particular will be unavailable for most readers.  It must
    be reported as *unavailable* rather than scored as a wall of failures, which
    would silently flatter the other arms.
    """
    if "error" in provenance:
        return False, str(provenance["error"])[:200]
    if "import_error" in provenance:
        return False, f"library import failed: {provenance['import_error']}"
    if spec.name == "mathematica" and not provenance.get("kernel_ok"):
        return False, (
            f"Wolfram kernel not usable: {provenance.get('kernel_error', 'no kernel found')}"[:200]
        )
    return True, ""
