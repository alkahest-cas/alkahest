"""Run ``separation_solve`` on the separation corpora and write one JSON line
per problem: status, kind, method, error code, time, and the partial sum
(indices ≤ N) at sample points for the comparison with Mathematica.

    PYTHONPATH=<site> python3 tests/pde_corpus/run_separation.py out.jsonl [abbasi|textbook|all]
    python3 tests/pde_corpus/run_separation.py --one abbasi_bounded.json abbasi-151
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import alkahest as ak  # noqa: E402
import alkahest.experimental as akx  # noqa: E402
import mma_subset  # noqa: E402

# Parameter values for the numeric comparison (both systems).
PARAMS = {
    "L": 1,
    "a": 1,
    "b": 2,
    "H": 2,
    "c": 1,
    "k": 0.5,
    "h": 1,
    "R": 1,
    "A": 1,
    "B": 2,
    "T0": 1,
    "T1": 2,
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.5,
    "lambda": 0.5,
    "omega": 1.5,
    "Q0": 1,
    "u0": 1,
}
NTERMS = 6


def load(which: str):
    out = []
    if which in ("abbasi", "all"):
        with open(os.path.join(HERE, "abbasi_bounded.json")) as fh:
            out += json.load(fh)
    if which in ("textbook", "all"):
        with open(os.path.join(HERE, "separation_corpus.json")) as fh:
            out += json.load(fh)
    return out


def sample_points(pde, env):
    """Two points inside the domain (fractions of each range), with the
    parameters at ``PARAMS``."""
    vals = {s: PARAMS.get(n, 1) for n, s in env.params.items()}
    rngs = []
    for v in env.var_list:
        lo, hi = None, None
        for part in getattr(pde.domain, "parts", (pde.domain,)):
            spec = part.spec()
            if spec[0] == "interval" and spec[1] is v:
                lo, hi = spec[2], spec[3]
            if spec[0] == "half_line" and spec[1] is v:
                lo, hi = spec[2], spec[2] + 1
            if spec[0] == "curvilinear":
                for c, b in zip(spec[2], spec[3]):
                    if c is v:
                        lo, hi = b
        if lo is None:
            return None
        lo = ak.eval_expr(lo, vals)
        hi = ak.eval_expr(hi, vals)
        rngs.append((lo, hi))
    pts = []
    for fr in (0.37, 0.61):
        pts.append([lo + fr * (hi - lo) for lo, hi in rngs])
    # time-like (half-line) variable: a small positive time
    for i, v in enumerate(env.var_list):
        if str(v) == "t":
            pts[0][i] = 0.2
            pts[1][i] = 0.45
    return pts


def run_one(entry):
    rec = {"id": entry["id"]}
    t0 = time.time()
    try:
        pde, env = mma_subset.build(entry["code"])
    except mma_subset.Unsupported as e:
        rec.update(status="untranslated", error=str(e)[:200])
        return rec
    except Exception as e:  # translation failure
        rec.update(status="untranslated", error=f"{type(e).__name__}: {str(e)[:200]}")
        return rec
    try:
        sol = akx.separation_solve(pde)
    except Exception as e:
        rec.update(
            status="refused",
            code=getattr(e, "code", type(e).__name__),
            error=str(e)[:300],
            time=time.time() - t0,
        )
        return rec
    rec.update(
        status=sol.status,
        kind=sol.kind,
        method=sol.method,
        closed=all(g.closed for g in sol.modes),
        convergence=sol.convergence,
        value=str(sol.value)[:2000],
        time=time.time() - t0,
        vars=[str(v) for v in env.var_list],
        params={n: PARAMS.get(n, 1) for n in env.params},
    )
    # Numeric partial sums (when the answer has no arbitrary function).
    if "(" in str(sol.value) and any(
        f"{fn}(" in str(sol.value) for fn in ("f", "g", "Q", "F", "G", "h1", "f1")
    ):
        return rec
    pts = sample_points(pde, env)
    if pts is None:
        return rec
    try:
        params = {s: PARAMS.get(n, 1) for n, s in env.params.items()}
        expr = sol.numeric(NTERMS, params)
        f = ak.compile_expr(expr, env.var_list)
        rec["points"] = pts
        rec["values"] = [f(p) for p in pts]
    except Exception as e:
        rec["numeric_error"] = str(e)[:200]
    return rec


def main():
    if sys.argv[1] == "--one":
        # ``--one <corpus file> <id>``: one problem, its record on stdout (the
        # test suite runs each problem in its own process).
        with open(os.path.join(HERE, sys.argv[2])) as fh:
            entry = next(e for e in json.load(fh) if e["id"] == sys.argv[3])
        print(json.dumps(run_one(entry)), flush=True)
        return
    out = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    with open(out, "w") as fh:
        for e in load(which):
            if only and e["id"] not in only:
                continue
            rec = run_one(e)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                rec["id"],
                rec["status"],
                rec.get("code", ""),
                rec.get("error", "")[:100],
                flush=True,
            )


if __name__ == "__main__":
    main()
