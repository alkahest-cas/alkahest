"""Budget conformance of ``integrate``: a wall budget is a deadline, not a hint.

The landing page promises that a ``Budget`` makes a call raise
``E-BUDGET-*`` instead of hanging an agent loop. ``integrate`` broke that on
``∫ d/dx[atan x + atan x·log x·√(x+1)] dx``: under ``wall_ms=4000`` it was
still running when killed at 60 s, inside one ``cancel`` (a chain of
multivariate GCDs over ``atan x``, ``log x``, ``√(x+1)``) that the by-parts
route called on an intermediate antiderivative, with no budget checkpoint
anywhere in the rational-function arithmetic.

That hole had siblings, so this file fuzzes for the whole class rather than
pinning the one repro: random elementary integrands (``atan``, ``log``,
``sqrt``, ``exp``, rational functions; products, quotients and compositions,
depth 3–4), each integrated — half of them as ``F`` itself, half as ``F′``,
the shape of the original report — under a wall budget of
:data:`WALL_MS` in a worker subprocess with a hard timeout. Every call must
return, refuse with a coded error, or raise ``BudgetExceededError``
(``E-BUDGET-*``) within ``3·wall_ms + 1 s``; a call still running at that
point is an overrun, and the worker is killed rather than waited for.

A small deterministic subset runs in the default suite; the larger sweep is
``@pytest.mark.slow``. The same harness runs from the command line for
local sweeps::

    python tests/test_integrate_budget_fuzz.py --cases 400 --seed 1 --wall 300
"""

from __future__ import annotations

import json
import os
import random
import select
import subprocess
import sys
import time

import pytest

#: Wall budget for each fuzz case, in milliseconds.
WALL_MS = 300


def deadline_s(wall_ms: int) -> float:
    """The hard limit for one call: ``3·wall_ms + 1 s``."""
    return 3 * wall_ms / 1000.0 + 1.0


# ---------------------------------------------------------------------------
# Generator — produces Python source over `ak` and `x`, so the worker process
# rebuilds the integrand in its own pool and a failing case can be pasted.
# ---------------------------------------------------------------------------


def _atom(rng: random.Random) -> str:
    k = rng.randint(1, 3)
    return rng.choice(["x", f"(x + {k})", f"({k}*x)", f"(x**2 + {k})", "x"])


_UNARY = [
    "ak.atan({})",
    "ak.log({})",
    "ak.sqrt({})",
    "ak.exp({})",
    "(1/({}))",
    "({})**2",
]

_BINARY = ["({} + {})", "({} * {})", "({} / {})", "({} - {})"]


def _tree(rng: random.Random, depth: int) -> str:
    """A random expression tree of the given depth (compositions)."""
    if depth <= 0 or rng.random() < 0.05:
        return _atom(rng)
    if rng.random() < 0.45:
        return rng.choice(_UNARY).format(_tree(rng, depth - 1))
    left = _tree(rng, depth - 1)
    right = _tree(rng, rng.randint(0, depth - 1))
    return rng.choice(_BINARY).format(left, right)


_PIECES = ["ak.atan({})", "ak.log({})", "ak.sqrt({})", "ak.exp({})"]


def _piece(rng: random.Random) -> str:
    """A transcendental or algebraic factor, possibly composed once."""
    inner = _atom(rng) if rng.random() < 0.7 else rng.choice(_PIECES).format(_atom(rng))
    if rng.random() < 0.15:
        return f"(1/({inner}))"
    return rng.choice(_PIECES).format(inner)


def _sum_of_products(rng: random.Random) -> str:
    """``Σ ∏ pieces`` — the shape of the report's ``atan x + atan x·log x·√(x+1)``."""
    terms = []
    for _ in range(rng.randint(1, 3)):
        factors = [_piece(rng) for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.3:
            factors.append(_atom(rng))
        terms.append("*".join(factors))
    return " + ".join(terms)


def generate_cases(seed: int, n: int) -> list[dict]:
    """``n`` deterministic cases: ``{"src": …, "deriv": bool}``.

    Half are random trees of depth 3–4, half sums of products of composed
    pieces; half of each are integrated as ``F′`` rather than ``F``.
    """
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        src = _tree(rng, rng.choice([3, 4])) if rng.random() < 0.5 else _sum_of_products(rng)
        cases.append({"src": src, "deriv": rng.random() < 0.5})
    return cases


#: The report's repro, kept in every run.
REPRO = {"src": "ak.atan(x) + ak.atan(x)*ak.log(x)*ak.sqrt(x + 1)", "deriv": True}

#: Overruns the sweep found before the fix, under ``wall_ms=300`` (seconds
#: elapsed in brackets): all three were the same unpolled ``cancel`` in the
#: by-parts route.
REGRESSIONS = [
    # 17.6 s (seed 3)
    {
        "src": "ak.atan((x**2 + 1))*ak.atan((x**2 + 3))*ak.sqrt(ak.exp((3*x))) + "
        "ak.atan(ak.log((x + 1)))*ak.sqrt(ak.atan((x + 2)))*ak.atan((x + 1))",
        "deriv": True,
    },
    # 3.7 s (seed 2)
    {
        "src": "(1/(ak.sqrt((x + 2))))*ak.log(ak.sqrt(x))*ak.atan(ak.exp((2*x))) + "
        "ak.exp(ak.atan((x**2 + 1)))*(x**2 + 1) + ak.log(x)*ak.atan(x)*ak.exp(x)",
        "deriv": True,
    },
    # 3.0 s (seed 2)
    {
        "src": "ak.atan(ak.sqrt(x))*(x**2 + 3) + ak.sqrt(ak.log(x))*x + "
        "ak.atan(ak.exp((x + 2)))*ak.atan((x**2 + 3))*ak.log((x + 3))",
        "deriv": True,
    },
]


# ---------------------------------------------------------------------------
# Worker — one long-lived process; one JSON request and reply per line.
# ---------------------------------------------------------------------------


def _worker() -> None:  # pragma: no cover - runs in the subprocess
    import alkahest as ak

    for line in sys.stdin:
        req = json.loads(line)
        out: dict = {"status": "setup"}
        try:
            p = ak.ExprPool()
            x = p.symbol("x")
            f = eval(req["src"], {"ak": ak, "x": x})
            if req["deriv"]:
                f = ak.diff(f, x).value
            out["status"] = "running"
            t0 = time.perf_counter()
            try:
                with ak.context(budget=ak.Budget(wall_ms=req["wall_ms"])):
                    ak.integrate(f, x)
                out["status"] = "ok"
            except ak.BudgetExceededError as e:
                out["status"] = "budget"
                out["code"] = getattr(e, "code", None)
            except Exception as e:
                code = getattr(e, "code", None)
                out["status"] = "coded" if code else "uncoded"
                out["code"] = code
                out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out["elapsed"] = time.perf_counter() - t0
        except Exception as e:
            out["status"] = "setup_error"
            out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


class _Pool:
    """Owns the worker subprocess; restarts it after an overrun kill."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

    def _start(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        # Warm the import outside any case's clock.
        self._ask({"src": "x", "deriv": False, "wall_ms": 1000}, timeout=120.0)

    def _ask(self, req: dict, timeout: float) -> dict | None:
        assert self.proc is not None
        assert self.proc.stdin
        assert self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line) if line else {"status": "crashed"}

    def run(self, case: dict, wall_ms: int) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        # The hard timeout bounds the whole round trip (integrand construction
        # included); the verdict below uses the worker's own clock around the
        # `integrate` call when it has one.
        limit = deadline_s(wall_ms)
        t0 = time.perf_counter()
        out = self._ask({**case, "wall_ms": wall_ms}, timeout=limit + 30.0)
        if out is None:
            self.close()
            return {"status": "overrun", "elapsed": time.perf_counter() - t0}
        if out.get("status") == "crashed":
            self.close()
        return out

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


def verdict(out: dict, wall_ms: int) -> str | None:
    """``None`` when the call conformed, else a one-word reason."""
    st = out["status"]
    if st == "setup_error":
        return None  # the generator built something `ak` refuses to construct
    if st in ("overrun", "crashed", "uncoded"):
        return st
    if out["elapsed"] > deadline_s(wall_ms):
        return "late"
    if st == "budget" and not str(out.get("code", "")).startswith("E-BUDGET-"):
        return "budget-without-code"
    return None


def sweep(cases: list[dict], wall_ms: int, log=None) -> tuple[list, list]:
    """Run ``cases``; return ``(results, failures)``."""
    pool = _Pool()
    results, failures = [], []
    try:
        for case in cases:
            out = pool.run(case, wall_ms)
            bad = verdict(out, wall_ms)
            results.append((case, out))
            if bad:
                failures.append((bad, case, out))
            if log:
                log(case, out, bad)
    finally:
        pool.close()
    return results, failures


def _fmt(failures) -> str:
    return "\n".join(
        f"{bad}: {c['src']} (deriv={c['deriv']}) elapsed={o.get('elapsed', 0):.2f}s "
        f"{o.get('code') or ''} {o.get('error', '')[:120]}"
        for bad, c, o in failures
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_repro_stops_within_budget():
    """The report's integrand under its own 4 s budget: E-BUDGET-001 or an
    answer, well inside 1.2·wall_ms (it ran past 60 s before the fix)."""
    wall = 4000
    results, failures = sweep([REPRO], wall)
    assert not failures, _fmt(failures)
    out = results[0][1]
    assert out["status"] in ("ok", "budget", "coded"), out
    assert out["elapsed"] < 1.2 * wall / 1000.0 + 0.5, out


def test_cancel_honours_a_wall_budget():
    """``cancel`` itself polls: a budget trip inside its rational-function
    arithmetic is ``BudgetExceededError``, not the ``E-POLY-004`` it travels
    as inside the Rust enum."""
    import alkahest as ak

    p = ak.ExprPool()
    x = p.symbol("x")
    gens = [ak.atan(x), ak.log(x), ak.sqrt(x + 1), ak.exp(x), ak.atan(x + 1)]
    expr = p.integer(0)
    for k in range(1, 40):
        g, h = gens[k % 5], gens[(k * 2 + 1) % 5]
        expr = expr + g * h**2 / (x + k) + h / (g + k)
    t0 = time.perf_counter()
    with pytest.raises(ak.BudgetExceededError) as ei, ak.context(budget=ak.Budget(wall_ms=50)):
        ak.cancel(expr, [x])
    assert ei.value.code == "E-BUDGET-001"
    assert time.perf_counter() - t0 < 5.0
    # A small cancel under a generous budget is unaffected.
    with ak.context(budget=ak.Budget(wall_ms=5000)):
        out = ak.cancel((x**2 - 1) / (x - 1), [x])
    assert str(out).strip("()") in ("x + 1", "1 + x")


def test_fuzz_budget_conformance_fast():
    cases = [REPRO, *REGRESSIONS, *generate_cases(seed=20261108, n=40)]
    _, failures = sweep(cases, WALL_MS)
    assert not failures, _fmt(failures)


@pytest.mark.slow
def test_fuzz_budget_conformance_slow():
    # Seed 3 is the sweep that found the 17.6 s overrun above.
    cases = generate_cases(seed=3, n=400) + generate_cases(seed=4, n=400)
    _, failures = sweep(cases, WALL_MS)
    assert not failures, _fmt(failures)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:  # pragma: no cover - manual sweeps
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wall", type=int, default=WALL_MS)
    ap.add_argument("--jsonl", default=None, help="append every result here")
    args = ap.parse_args(argv)
    if args.worker:
        _worker()
        return 0

    def log(case, out, bad):
        if args.jsonl:
            with open(args.jsonl, "a") as sink:
                sink.write(json.dumps({"case": case, "out": out, "bad": bad}) + "\n")
        if bad:
            print(f"!! {bad}: {case} {out}", flush=True)

    cases = generate_cases(args.seed, args.cases)
    results, failures = sweep(cases, args.wall, log)
    ratios = [o["elapsed"] / (args.wall / 1000.0) for _, o in results if "elapsed" in o]
    counts: dict = {}
    for _, o in results:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    print(f"cases={len(results)} statuses={counts} failures={len(failures)}")
    if ratios:
        print(f"max elapsed/wall = {max(ratios):.2f}")
    print(_fmt(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
