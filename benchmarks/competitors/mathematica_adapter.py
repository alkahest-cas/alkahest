"""Mathematica / Wolfram Engine adapter for cross-CAS benchmarks (V1-13).

Requires:
1. Wolfram Engine (free for non-commercial use) installed and activated:
   https://www.wolfram.com/engine/

2. ``wolframclient`` Python package:
   pip install wolframclient

Usage:
    from benchmarks.competitors import MathematicaAdapter
    adapter = MathematicaAdapter()
    if adapter.is_available():
        print(adapter.bench_integrate(10))
"""

from __future__ import annotations

from typing import Any

from .base import CASAdapter

_session = None  # module-level cached WolframLanguageSession


_KERNEL_CANDIDATES = [
    "/usr/local/Wolfram/WolframEngine/14.3/Executables/WolframKernel",
    "/usr/local/Wolfram/WolframEngine/13.3/Executables/WolframKernel",
    "/usr/local/Wolfram/Mathematica/14.3/Executables/WolframKernel",
    "/usr/local/Wolfram/Mathematica/13.3/Executables/WolframKernel",
]


def _find_kernel() -> str | None:
    """Return the first existing WolframKernel executable path, or None."""
    import os
    # Honour explicit override
    env = os.environ.get("WOLFRAM_KERNEL")
    if env and os.path.isfile(env):
        return env
    for path in _KERNEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _get_session():
    """Return (or create) a cached Wolfram Language session.

    Tries an explicit kernel path first (auto-discovery) so the session works
    even when WolframClient cannot locate the kernel on its own.
    """
    global _session
    if _session is not None:
        return _session
    from wolframclient.evaluation import WolframLanguageSession

    kernel = _find_kernel()
    sess = WolframLanguageSession(kernel) if kernel else WolframLanguageSession()
    try:
        sess.start()
    except Exception:
        _session = None
        raise
    _session = sess
    return _session


class MathematicaAdapter(CASAdapter):
    """Adapter wrapping the Wolfram Engine via ``wolframclient``.

    Requires Wolfram Engine for Developers (free, https://www.wolfram.com/engine/)
    to be installed **and activated** (run ``wolframscript -activate`` once
    after registration).  Degrades gracefully when unavailable.
    """

    name = "Mathematica"

    def is_available(self) -> bool:
        try:
            import wolframclient  # noqa: F401
            from wolframclient.language import wlexpr
            s = _get_session()
            # Probe: evaluate a trivial expression to confirm the kernel is live.
            result = s.evaluate(wlexpr("1 + 1"))
            return result == 2
        except Exception:
            return False

    def _wl(self, code: str) -> str:
        """Evaluate a Wolfram Language expression and return its string form."""
        from wolframclient.language import wlexpr

        session = _get_session()
        result = session.evaluate(wlexpr(f"ToString[{code}]"))
        return str(result)

    # ── Task-named benchmark methods (match tasks.py task names) ─────────────

    def bench_poly_diff(self, size: int) -> Any:
        poly = " + ".join(f"x^{k}" for k in range(size + 1))
        return self._wl(f"D[{poly}, x]")

    def bench_trig_identity(self, size: int) -> Any:
        # FullSimplify[N * (Sin[x]^2 + Cos[x]^2)] → N
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
        # Build polynomial without x^0 to avoid 0.^0 in Compile.
        terms = ["1"] + [f"x^{k}" for k in range(1, size + 1)]
        poly_wl = " + ".join(terms)
        # RuntimeAttributes -> {Listable} lets Mathematica vectorise over a
        # packed Real array — matching the 1 000 000-point alkahest task.
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

    # ── New comprehensive task methods ───────────────────────────────────────

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
        return self._wl(
            f"Length[NSolve[x^{size} - x - 1 == 0, x, Reals]]"
        )

    def bench_horner_form_poly(self, size: int) -> Any:
        terms = " + ".join(f"x^{k}" for k in range(1, size + 1))
        return self._wl(f"HornerForm[1 + {terms}]")

    def bench_log_exp_simplify(self, size: int) -> Any:
        expr = "x"
        for _ in range(size):
            expr = f"Log[Exp[{expr}]]"
        return self._wl(f"FullSimplify[{expr}, Assumptions -> x \\[Element] Reals]")

    def bench_resultant_poly(self, size: int) -> Any:
        return self._wl(
            f"Resultant[x^{size} + x + 1, x^{size} - x - 1, x]"
        )

    def bench_recurrence_solve(self, size: int) -> Any:
        if size == 1:
            return self._wl(
                "RSolve[{a[n] == 2*a[n-1], a[0] == 1}, a[n], n]"
            )
        else:
            return self._wl(
                "RSolve[{a[n] == a[n-1] + a[n-2], a[0] == 1, a[1] == 1}, a[n], n]"
            )

    # ── Legacy names ─────────────────────────────────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        return self._wl(f"Integrate[x^{size}, x]")

    def bench_simplify(self, size: int) -> Any:
        return self._wl(f"FullSimplify[(Sin[x]^2 + Cos[x]^2)^{size}]")

    def bench_diff(self, size: int) -> Any:
        return self.bench_poly_diff(size)

    def bench_groebner(self, size: int) -> Any:
        return self._wl(
            "GroebnerBasis[{x^2 + y^2 - 1, x - y}, {x, y}, MonomialOrder -> Lexicographic]"
        )

    def bench_poly_gcd(self, size: int) -> Any:
        return self._wl(f"PolynomialGCD[x^{size} - 1, x^{size // 2} - 1]")

    def bench_jacobian(self, size: int) -> Any:
        return self._wl("D[{Sin[x*y], Cos[x+y], Exp[x-y]}, {{x, y}}]")

    def bench_polynomial_solve(self, size: int) -> Any:
        return self._wl(f"Solve[x^2 == {size}, x, Reals]")

    def __del__(self):
        global _session
        if _session is not None:
            try:
                _session.terminate()
            except Exception:
                pass
            _session = None
