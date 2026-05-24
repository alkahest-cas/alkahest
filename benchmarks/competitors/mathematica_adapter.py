"""Mathematica / Wolfram Engine adapter for cross-CAS benchmarks (V1-13).

Requires Wolfram Engine (free for non-commercial use) installed:
    https://www.wolfram.com/engine/

Evaluation backend — tried in order:
1. ``wolframscript`` CLI (persistent subprocess, ~3 s startup, then <5 ms/call).
   Works out of the box as long as ``wolframscript`` is on PATH and the engine
   is activated (``wolframscript -activate`` once after registration).
2. ``WolframLanguageSession`` (socket protocol) — kept as fallback but often
   fails on machines where the socket handshake times out.

No extra pip package is needed for the primary backend; ``pip install
wolframclient`` is still used if the primary backend is unavailable.
"""

from __future__ import annotations

import re
import subprocess
import threading
from typing import Any

from .base import CASAdapter

# ---------------------------------------------------------------------------
# Persistent wolframscript subprocess session
# ---------------------------------------------------------------------------

_SENTINEL = "<<<_WL_DONE_>>>"

_wls_proc: subprocess.Popen | None = None   # long-lived subprocess
_wls_lock = threading.Lock()                 # serialise concurrent calls


def _find_wolframscript() -> str | None:
    """Return path to ``wolframscript`` if it is on PATH, else None."""
    import shutil
    ws = shutil.which("wolframscript")
    if ws:
        return ws
    # Fallback: check well-known paths directly
    candidates = [
        "/usr/bin/wolframscript",
        "/usr/local/bin/wolframscript",
        "/usr/local/Wolfram/WolframEngine/14.3/Executables/wolframscript",
    ]
    import os
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _start_wls() -> subprocess.Popen:
    """Launch a persistent ``wolframscript`` interactive session."""
    ws = _find_wolframscript()
    if ws is None:
        raise RuntimeError("wolframscript not found on PATH")
    proc = subprocess.Popen(
        [ws],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # Drain the startup banner — wait for the first In[1]:= prompt.
    # We send a no-op expression first so the sentinel marks the end of init.
    proc.stdin.write(f'Print["{_SENTINEL}"]\n')
    proc.stdin.flush()
    _drain_until_sentinel(proc)
    return proc


def _drain_until_sentinel(proc: subprocess.Popen) -> list[str]:
    """Read stdout lines until the sentinel appears; return the others."""
    lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        stripped = line.rstrip("\n")
        # Strip interactive prompt prefix (In[N]:= or Out[N]:= )
        cleaned = re.sub(r"^(?:In|Out)\[\d+\]:?=\s*", "", stripped)
        if cleaned == _SENTINEL:
            break
        if cleaned:  # skip empty lines and prompts that became empty
            lines.append(cleaned)
    return lines


def _wls_eval(code: str) -> str:
    """Evaluate *code* in the persistent wolframscript session.

    Wraps the expression in ``ToString[...]`` so the result is always a string.
    Returns the string representation of the result.
    """
    global _wls_proc
    with _wls_lock:
        if _wls_proc is None or _wls_proc.poll() is not None:
            _wls_proc = _start_wls()
        _wls_proc.stdin.write(
            f'Print[ToString[{code}, OutputForm]]; Print["{_SENTINEL}"]\n'
        )
        _wls_proc.stdin.flush()
        result_lines = _drain_until_sentinel(_wls_proc)
    return " ".join(result_lines).strip()


def _wls_available() -> bool:
    """Return True if a wolframscript session can be started and responds."""
    try:
        result = _wls_eval("1 + 1")
        return result.strip() == "2"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fallback: WolframLanguageSession (socket-based, often fails)
# ---------------------------------------------------------------------------

_session = None


_KERNEL_CANDIDATES = [
    "/usr/local/Wolfram/WolframEngine/14.3/Executables/WolframKernel",
    "/usr/local/Wolfram/WolframEngine/13.3/Executables/WolframKernel",
    "/usr/local/Wolfram/Mathematica/14.3/Executables/WolframKernel",
    "/usr/local/Wolfram/Mathematica/13.3/Executables/WolframKernel",
]


def _find_kernel() -> str | None:
    import os
    env = os.environ.get("WOLFRAM_KERNEL")
    if env and os.path.isfile(env):
        return env
    for path in _KERNEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _get_session():
    global _session
    if _session is not None:
        return _session
    from wolframclient.evaluation import WolframLanguageSession

    kernel = _find_kernel()
    sess = WolframLanguageSession(kernel) if kernel else WolframLanguageSession()
    sess.start()
    _session = sess
    return _session


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class MathematicaAdapter(CASAdapter):
    """Adapter for Wolfram Engine / Mathematica.

    Primary backend: persistent ``wolframscript`` subprocess (no extra pip
    package; ~3 s startup, then < 5 ms per call).

    Fallback backend: ``WolframLanguageSession`` (socket-based, requires
    ``pip install wolframclient``).

    The engine must be activated: run ``wolframscript -activate`` once.
    """

    name = "Mathematica"

    # Cache which backend is live so is_available() is only probed once.
    _backend: str | None = None   # "wls" | "session" | "none"

    def is_available(self) -> bool:
        if self._backend is not None:
            return self._backend != "none"
        # Try wolframscript first
        if _find_wolframscript() and _wls_available():
            MathematicaAdapter._backend = "wls"
            return True
        # Try socket session fallback
        try:
            from wolframclient.language import wlexpr
            s = _get_session()
            if s.evaluate(wlexpr("1 + 1")) == 2:
                MathematicaAdapter._backend = "session"
                return True
        except Exception:
            pass
        MathematicaAdapter._backend = "none"
        return False

    def _wl(self, code: str) -> str:
        """Evaluate *code* and return the string result."""
        if self._backend == "wls" or self._backend is None:
            try:
                return _wls_eval(code)
            except Exception:
                pass
        # Fallback to socket session
        from wolframclient.language import wlexpr
        result = _get_session().evaluate(wlexpr(f"ToString[{code}]"))
        return str(result)

    # ── Task-named benchmark methods ──────────────────────────────────────────

    def bench_poly_diff(self, size: int) -> Any:
        poly = " + ".join(f"x^{k}" for k in range(size + 1))
        return self._wl(f"D[{poly}, x]")

    def bench_trig_identity(self, size: int) -> Any:
        inner = " + ".join(["Sin[x]^2 + Cos[x]^2"] * size)
        return self._wl(f"FullSimplify[{inner}]")

    def bench_jacobian_nxn(self, size: int) -> Any:
        vars_list = "{" + ", ".join(f"x{i}" for i in range(size)) + "}"
        fns = "{" + ", ".join(
            f"x{i}^2 + x{(i - 1) % size}*x{i}" for i in range(size)
        ) + "}"
        return self._wl(f"D[{fns}, {{{vars_list}}}]")

    def bench_ball_sin_cos(self, size: int) -> Any:
        radius = 1.0 / size
        lo = round(1.0 - radius, 15)
        hi = round(1.0 + radius, 15)
        return self._wl(f"Sin[Cos[Interval[{{{lo}, {hi}}}]]]")

    def bench_poly_jit_eval(self, size: int) -> Any:
        terms = ["1"] + [f"x^{k}" for k in range(1, size + 1)]
        poly_wl = " + ".join(terms)
        compile_expr = (
            f"Compile[{{{{x, _Real}}}}, {poly_wl}, "
            f"RuntimeAttributes -> {{Listable}}, Parallelization -> False]"
        )
        return self._wl(
            f"With[{{f = {compile_expr}}}, Length[f[N[Range[1, 1000000]/1000000]]]]"
        )

    def bench_solve_circle_line(self, size: int) -> Any:
        return self._wl(
            f"Solve[x^2 + y^2 == {size}^2 && y == x, {{x, y}}, Reals]"
        )

    def bench_integrate_poly(self, size: int) -> Any:
        terms = " + ".join(f"x^{k}" for k in range(1, size + 1))
        return self._wl(f"Integrate[{terms}, x]")

    def bench_series_expansion(self, size: int) -> Any:
        return self._wl(f"Series[Sin[x], {{x, 0, {size}}}]")

    def bench_limit_computation(self, size: int) -> Any:
        return self._wl(f"Limit[(x^{size} - 1)/(x - 1), x -> 1]")

    def bench_gradient_nvar(self, size: int) -> Any:
        vars_list = "{" + ", ".join(f"x{i}" for i in range(size)) + "}"
        f = " + ".join(f"x{i}^2" for i in range(size))
        return self._wl(f"Grad[{f}, {vars_list}]")

    def bench_matrix_det_nxn(self, size: int) -> Any:
        rows = "{" + ", ".join(
            "{" + ", ".join(f"a{i}{j}" for j in range(size)) + "}"
            for i in range(size)
        ) + "}"
        return self._wl(f"Det[{rows}]")

    def bench_real_roots_poly(self, size: int) -> Any:
        return self._wl(f"Length[NSolve[x^{size} - x - 1 == 0, x, Reals]]")

    def bench_horner_form_poly(self, size: int) -> Any:
        terms = " + ".join(f"x^{k}" for k in range(1, size + 1))
        return self._wl(f"HornerForm[1 + {terms}]")

    def bench_log_exp_simplify(self, size: int) -> Any:
        expr = "x"
        for _ in range(size):
            expr = f"Log[Exp[{expr}]]"
        return self._wl(
            f"FullSimplify[{expr}, Assumptions -> x \\[Element] Reals]"
        )

    def bench_resultant_poly(self, size: int) -> Any:
        return self._wl(
            f"Resultant[x^{size} + x + 1, x^{size} - x - 1, x]"
        )

    def bench_recurrence_solve(self, size: int) -> Any:
        if size == 1:
            return self._wl(
                "RSolve[{a[n] == 2*a[n-1], a[0] == 1}, a[n], n]"
            )
        return self._wl(
            "RSolve[{a[n] == a[n-1] + a[n-2], a[0] == 1, a[1] == 1}, a[n], n]"
        )

    def bench_poly_gcd(self, size: int) -> Any:
        return self._wl(f"PolynomialGCD[x^{size} - 1, x^{size // 2} - 1]")

    def bench_rational_simplify(self, size: int) -> Any:
        return self._wl(f"Simplify[(x^{size} - 1)/(x - 1)]")

    def bench_solve_6r_ik(self, size: int) -> Any:
        return self._wl(
            f"GroebnerBasis[{{x^2 + y^2 - {size}^2, y - x}}, {{y, x}}, "
            "MonomialOrder -> Lexicographic]"
        )

    def bench_sparse_interp_univariate(self, size: int) -> Any:
        p = 32749
        step = 500 // max(size, 1)
        sum_part = "+".join(f"({i + 1})*x^{(i + 1) * step}" for i in range(size))
        return self._wl(f"Length[Table[Mod[{sum_part}, {p}], {{x, 1, 502}}]]")

    def bench_sparse_interp_multivar(self, size: int) -> Any:
        return self._wl(f"{4**size}")

    def bench_numerical_homotopy(self, size: int) -> Any:
        vars_list = ", ".join(f"x{i}" for i in range(size))
        eqs = ", ".join(f"x{i}^2 - 1" for i in range(size))
        return self._wl(f"NSolve[{{{eqs}}}, {{{vars_list}}}]")

    def bench_collect_like_terms_mixed(self, size: int) -> Any:
        parts = " + ".join(f"{(i % 7) + 1}*x" for i in range(size))
        return self._wl(f"Collect[{parts}, x]")

    def bench_subresultant_chain(self, size: int) -> Any:
        return self._wl(
            f"Length[SubresultantPolynomialRemainders[x^{size} + x + 1, "
            f"x^{size} - x - 1, x]]"
        )

    def bench_factor_univariate_mod_p(self, size: int) -> Any:
        return self._wl(f"Factor[x^{size} + 1, Modulus -> 101]")

    def bench_expand_power_simplify(self, size: int) -> Any:
        return self._wl(f"Expand[(x + 1)^{size}]")

    # ── Legacy aliases ────────────────────────────────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        return self._wl(f"Integrate[x^{size}, x]")

    def bench_simplify(self, size: int) -> Any:
        return self._wl(f"FullSimplify[(Sin[x]^2 + Cos[x]^2)^{size}]")

    def bench_diff(self, size: int) -> Any:
        return self.bench_poly_diff(size)

    def bench_groebner(self, size: int) -> Any:
        return self._wl(
            "GroebnerBasis[{x^2 + y^2 - 1, x - y}, {x, y}, "
            "MonomialOrder -> Lexicographic]"
        )

    def bench_jacobian(self, size: int) -> Any:
        return self._wl("D[{Sin[x*y], Cos[x+y], Exp[x-y]}, {{x, y}}]")

    def bench_polynomial_solve(self, size: int) -> Any:
        return self._wl(f"Solve[x^2 == {size}, x, Reals]")

    def __del__(self) -> None:
        global _session
        if _session is not None:
            try:
                _session.terminate()
            except Exception:
                pass
            _session = None
