"""Agent benchmark harness.

For each (skill, task, repeat) triple the harness:

  1. Sends the skill guide + task prompt to a model via LiteLLM.
  2. Extracts the Python script from the response.
  3. Statically rejects it if it imports a CAS this arm may not use.
  4. Executes it **in that arm's isolated venv**, under memory and CPU limits.
  5. Classifies the captured ANSWER line into an :class:`Outcome`.

Design notes that differ from the original harness, and why:

* Execution happens in a per-skill venv (see :mod:`envs`), not in the caller's
  interpreter.  Otherwise every arm can import every CAS and the comparison is
  meaningless.
* LLM latency and code execution are timed separately.  A single combined
  ``wall_ms`` cannot support any claim about library speed.
* Prompt and completion tokens are reported separately.  Skill guides differ in
  length by 3–4x, so a combined total mostly measures documentation size and
  would misrepresent the arm with the most thorough guide as the most expensive.
* Generated code is untrusted.  It runs with an address-space cap, a CPU-time
  cap, and a scratch working directory.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import envs
from envs import SkillSpec
from tasks.base import AgentTask, Outcome

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert mathematical computing assistant.\n"
    "You have been given a skill guide for a specific computer algebra library.\n"
    "When asked to solve a math problem, you MUST write a complete, self-contained "
    "Python script using exactly that library and no other computer algebra "
    "system.\n"
    "Enclose your script in a single ```python ... ``` code block.\n"
    "The script must print exactly one line `ANSWER: <value>` as its final output.\n"
    "Prefer a correct refusal over a confident wrong answer."
)

CONTROL_SYSTEM_PROMPT = (
    "You are an expert mathematical computing assistant.\n"
    "You have NO computer algebra system available — only the Python standard "
    "library and NumPy.  Solve the problem numerically or by hand-derived "
    "formula.\n"
    "Enclose your script in a single ```python ... ``` code block.\n"
    "The script must print exactly one line `ANSWER: <value>` as its final output.\n"
    "Prefer a correct refusal over a confident wrong answer."
)

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> str | None:
    """Return the Python script from a model response.

    Takes the **last** fenced block rather than the first: models commonly show
    an illustrative fragment before the final script, and the original harness
    graded that fragment instead of the answer.
    """
    blocks = _CODE_BLOCK_RE.findall(response_text or "")
    if not blocks:
        return None
    # Prefer the last block that actually prints an answer; fall back to the last.
    for block in reversed(blocks):
        if "ANSWER" in block:
            return block
    return blocks[-1]


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------


def _limit_resources(memory_mb: int, cpu_seconds: int):
    """Return a preexec_fn applying rlimits to the child (POSIX only)."""

    def apply() -> None:
        soft_mem = memory_mb * 1024 * 1024
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (soft_mem, soft_mem))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    exec_ms: float = 0.0


def execute_code(
    code: str,
    python: Path,
    *,
    timeout: int = 45,
    memory_mb: int = 4096,
) -> ExecResult:
    """Run *code* with *python*, isolated and resource-capped."""
    workdir = Path(tempfile.mkdtemp(prefix="agentbench-"))
    script = workdir / "solution.py"
    script.write_text(code, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONIOENCODING": "utf-8",
        # Keep the parent's site-packages off the child's path.
        "PYTHONNOUSERSITE": "1",
    }
    # The Wolfram arm needs to find its kernel.
    for passthrough in ("WOLFRAM_KERNEL", "WOLFRAMSCRIPT_ENTITLEMENTID", "LD_LIBRARY_PATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]

    preexec = _limit_resources(memory_mb, timeout) if platform.system() != "Windows" else None

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(python), str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=env,
            preexec_fn=preexec,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        err = None
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()
            err = f"exit {proc.returncode}: {tail[-1] if tail else '(no stderr)'}"
        return ExecResult(
            stdout=proc.stdout,
            stderr=proc.stderr[-2000:],
            error=err,
            exec_ms=round(elapsed, 2),
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            error=f"timeout after {timeout}s",
            timed_out=True,
            exec_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as exc:
        return ExecResult(
            error=f"subprocess error: {exc}",
            exec_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------


def _is_anthropic(model: str) -> bool:
    m = model.lower()
    return m.startswith("claude") or m.startswith("anthropic/")


@dataclass
class LLMResponse:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    llm_ms: float = 0.0
    error: str | None = None


def call_agent(
    skill_text: str,
    task_prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    is_control: bool = False,
    retries: int = 3,
) -> LLMResponse:
    """Call a LiteLLM-supported model, retrying transient failures."""
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "litellm not found — run: pip install -r agent-benchmark/requirements.txt"
        ) from exc

    system_prompt = CONTROL_SYSTEM_PROMPT if is_control else SYSTEM_PROMPT
    if is_control:
        system_content: Any = system_prompt
    elif _is_anthropic(model):
        system_content = [
            {"type": "text", "text": system_prompt},
            {
                "type": "text",
                "text": f"## Skill guide\n\n{skill_text}",
                "cache_control": {"type": "ephemeral"},
            },
        ]
    else:
        system_content = f"{system_prompt}\n\n## Skill guide\n\n{skill_text}"

    last_error = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            response = litellm.completion(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": task_prompt},
                ],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            usage = response.usage
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            return LLMResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_tokens=cached,
                llm_ms=round(elapsed, 2),
            )
        except Exception as exc:
            last_error = traceback.format_exception_only(type(exc), exc)[0].strip()
            if attempt < retries - 1:
                time.sleep(2**attempt)

    return LLMResponse(error=f"llm_error: {last_error}")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    skill: str
    task: str
    kind: str
    category: str
    difficulty: int
    model: str
    repeat: int
    #: One of the Outcome values, or a harness-level failure:
    #: no_code / wrong_library / exec_error / timeout / llm_error.
    status: str = "llm_error"
    success: bool = False
    #: True only for a confidently stated but mathematically wrong answer.
    silent_error: bool = False
    answer: str = ""
    error: str | None = None
    llm_ms: float = 0.0
    exec_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    code: str = ""
    stdout: str = ""
    stderr: str = ""


def run_one(
    spec: SkillSpec,
    skill_text: str,
    task: AgentTask,
    model: str,
    python: Path,
    *,
    repeat: int = 0,
    temperature: float = 0.0,
    memory_mb: int = 4096,
    keep_code: bool = False,
) -> RunResult:
    """Execute one (skill, task, repeat) triple."""
    result = RunResult(
        skill=spec.name,
        task=task.name,
        kind=task.kind.value,
        category=task.category,
        difficulty=task.difficulty,
        model=model,
        repeat=repeat,
    )

    response = call_agent(
        skill_text,
        task.prompt,
        model,
        temperature=temperature,
        is_control=(spec.name == "none"),
    )
    result.llm_ms = response.llm_ms
    result.prompt_tokens = response.prompt_tokens
    result.completion_tokens = response.completion_tokens
    result.cached_tokens = response.cached_tokens

    if response.error:
        result.status = "llm_error"
        result.error = response.error
        return result

    code = extract_code(response.text)
    if code is None:
        result.status = "no_code"
        result.error = "no python code block in response"
        return result
    if keep_code:
        result.code = code

    violation = envs.check_imports(code, spec)
    if violation:
        # The arm's venv would raise ImportError anyway; catching it statically
        # keeps contamination attempts visible instead of hiding them among
        # ordinary execution failures.
        result.status = "wrong_library"
        result.error = violation
        return result

    execution = execute_code(code, python, timeout=task.timeout_s, memory_mb=memory_mb)
    result.exec_ms = execution.exec_ms
    result.stdout = execution.stdout.strip()[-2000:]
    result.stderr = execution.stderr.strip()[-1000:]

    if execution.timed_out:
        result.status = "timeout"
        result.error = execution.error
        return result
    if execution.error:
        result.status = "exec_error"
        result.error = execution.error
        return result

    outcome = task.verify(execution.stdout)
    result.status = outcome.value
    result.success = task.score(outcome)
    result.silent_error = outcome is Outcome.WRONG_ANSWER
    from tasks.base import answer_line

    result.answer = (answer_line(execution.stdout) or "")[:200]
    return result


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_HERE,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def skill_digest(path: Path) -> dict[str, Any]:
    import hashlib

    if not path.exists():
        return {"path": str(path), "missing": True}
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
        "lines": data.decode("utf-8", "replace").count("\n") + 1,
        "bytes": len(data),
    }


def build_provenance(
    specs: dict[str, SkillSpec],
    model: str,
    temperature: float,
    repeats: int,
    env_root: Path,
) -> dict[str, Any]:
    """Capture everything needed to reproduce or discount a result set.

    Without this the results file records numbers produced by a build nobody can
    reconstruct — which is fatal for a benchmark meant to be shown to people
    evaluating the library.
    """
    arms: dict[str, Any] = {}
    for name, spec in specs.items():
        info = envs.probe_env(spec, env_root)
        available, reason = envs.arm_available(spec, info)
        arms[name] = {
            "packages": list(spec.packages),
            "skill_file": skill_digest(spec.skill_file),
            "environment": info,
            "available": available,
            "unavailable_reason": reason,
        }
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "model": model,
        "temperature": temperature,
        "repeats": repeats,
        "harness_python": sys.version.split()[0],
        "platform": platform.platform(),
        "arms": arms,
    }
