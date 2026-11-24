"""Benchmarks for ``alkahest.fex`` on the FEX paper's problems.

Reproduces the Poisson (d = 10, 20, 50) and linear conservation law
(d = 5, 20) problems of Liang & Yang (arXiv 2206.10121), with the reference
implementation's settings (depth2_sub tree, the paper's operator tables,
boundary weight 100, 5000 interior + 1000 boundary points for Poisson,
20000 + 4000 for the conservation law), and reports:

* ``eval``: residual + full parameter Jacobian per candidate (ms), against
  compiling each candidate's residual with ``compile_expr`` (Cranelift) after
  a symbolic Laplacian — the "compile every candidate" alternative;
* ``score``: candidates scored per second (coarse fit: Levenberg–Marquardt,
  and the paper's 20 Adam + 20 BFGS);
* ``solve``: an end-to-end :func:`alkahest.fex.search` — wall time, best
  loss, relative L² error in float64 on 100 000 fresh points, and whether the
  exact structure was recovered (canonical key of the best candidate equals
  that of the true solution's skeleton).

Usage::

    python benchmarks/fex/bench_fex.py [eval|score|solve|all] [--quick] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import alkahest as ak
import numpy as np
from alkahest import fex

C = math.pi / 4


def poisson(d: int) -> fex.Problem:
    return fex.Problem.poisson(
        d,
        rhs=-d,
        bc=lambda x: 0.5 * sum(xi**2 for xi in x),
        exact=lambda p: 0.5 * (p**2).sum(axis=1),
    )


def conservation(d: int) -> fex.Problem:
    # (d−1)·π/4 · u_t − Σ_{i≥1} u_{x_i} = 0,  u(0, x) = sin(π/4 · Σ x_i),
    # exact u = sin(t + π/4 · Σ x_i); x_0 is t, domain [0, 1]^d.
    coef = [(d - 1) * C] + [-1.0] * (d - 1)
    return fex.Problem.transport(
        coef,
        bc=lambda x: ak.sin(C * sum(x[1:])),
        time=0,
        exact=lambda p: np.sin(p[:, 0] + C * p[:, 1:].sum(axis=1)),
    )


# True skeletons (opcode vectors over PAPER_UNARY / PAPER_BINARY, slots
# [leaf, binary, leaf, root unary]).
TRUE = {
    "poisson": [3, 0, 0, 2],  # id(L[sq] + L[0])
    "conservation": [2, 0, 0, 7],  # sin(L[id] + L[0])
}

# Their exact parameters (π/4 does not snap to a small rational, so for the
# conservation law the snapped check is expected to be undecided).
TRUE_THETA = {
    "poisson": lambda d: [0.5] * d + [0.0] + [0.0] * (d + 1) + [1.0, 0.0],
    "conservation": lambda d: [1.0] + [C] * (d - 1) + [0.0] + [0.0] * (d + 1) + [1.0, 0.0],
}


def setup(kind: str, d: int, rng, n_int=None, n_bd=None):
    tpl = fex.Template("depth2_sub", dim=d)
    if kind == "poisson":
        prob = poisson(d)
        xi = fex.sample_box(n_int or 5000, d, -1.0, 1.0, rng)
        xb = fex.sample_box_boundary(n_bd or 1000, d, -1.0, 1.0, rng)
    else:
        prob = conservation(d)
        xi = fex.sample_box(n_int or 20000, d, 0.0, 1.0, rng)
        xb = fex.sample_initial_slice(n_bd or 4000, d, 0.0, 1.0, 0, 0.0, rng)
    return tpl, prob, xi, xb


def timed(f, repeat=3):
    best = math.inf
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def bench_eval(kind: str, d: int, rng, quick: bool) -> dict:
    tpl, prob, xi, xb = setup(kind, d, rng)
    ev = fex.Evaluator(tpl, prob, xi, xb)
    B = 32 if quick else 128
    E = tpl.random_skeletons(B, rng)
    Th = tpl.init_params(B, rng)
    t_rj, _ = timed(lambda: ev.residual_and_jacobian(E, Th))
    t_loss, _ = timed(lambda: ev.loss(E, Th))
    t_lg, _ = timed(lambda: ev.loss_and_grad(E, Th))
    # Single-candidate latency (one call, parallel over points).
    t_one, _ = timed(lambda: ev.residual_and_jacobian(E[0], Th[0]), repeat=5)
    # The alternative: per candidate, instantiate, take the Laplacian (or the
    # transport combination) symbolically, compile with compile_expr.
    xs = prob.x
    n_comp = 3 if quick else 6
    t_sym = t_comp = 0.0
    for c in range(n_comp):
        t = time.perf_counter()
        u = tpl.instantiate(E[c], Th[c], xs)
        if kind == "poisson":
            res = -sum(ak.diff(ak.diff(u, x).value, x).value for x in xs) + d
        else:
            coef = [(d - 1) * C] + [-1.0] * (d - 1)
            res = sum(ci * ak.diff(u, x).value for ci, x in zip(coef, xs))
        t_sym += time.perf_counter() - t
        t = time.perf_counter()
        f = ak.compile_expr(res, xs)
        t_comp += time.perf_counter() - t
        del f
    return {
        "kind": kind,
        "d": d,
        "n_int": len(xi),
        "n_bd": len(xb),
        "n_params": tpl.n_params,
        "batch": B,
        "residual_jacobian_ms_per_candidate": 1e3 * t_rj / B,
        "loss_ms_per_candidate": 1e3 * t_loss / B,
        "loss_grad_ms_per_candidate": 1e3 * t_lg / B,
        "single_candidate_residual_jacobian_ms": 1e3 * t_one,
        "per_candidate_symbolic_ms": 1e3 * t_sym / n_comp,
        "per_candidate_compile_expr_ms": 1e3 * t_comp / n_comp,
        "compile_tier": "cranelift" if "cranelift" in str(ak.capabilities()) else "default",
    }


def bench_score(kind: str, d: int, rng, quick: bool) -> dict:
    tpl, prob, xi, xb = setup(kind, d, rng)
    ev = fex.Evaluator(tpl, prob, xi, xb)
    B = 32 if quick else 160
    E = tpl.random_skeletons(B, rng)
    Th = tpl.init_params(B, rng)
    t_lm, _ = timed(lambda: ev.fit(E, Th, method="lm", max_iter=20), repeat=1)
    t_paper, _ = timed(
        lambda: ev.fit(E, Th, method="adam_bfgs", adam_iters=20, bfgs_iters=20, lr=1e-3),
        repeat=1,
    )
    return {
        "kind": kind,
        "d": d,
        "batch": B,
        "lm20_candidates_per_s": B / t_lm,
        "adam20_bfgs20_candidates_per_s": B / t_paper,
    }


def bench_solve(kind: str, d: int, rng, quick: bool, epochs: int) -> dict:
    tpl, prob, xi, xb = setup(kind, d, rng)
    t0 = time.perf_counter()
    res = fex.search(
        tpl,
        prob,
        xi,
        xb,
        epochs=epochs,
        batch=10,
        target_loss=1e-20,
        rng=rng,
    )
    wall = time.perf_counter() - t0
    ev = fex.Evaluator(tpl, prob, xi[:10], xb[:10])
    lo, hi = (-1.0, 1.0) if kind == "poisson" else (0.0, 1.0)
    test = fex.sample_box(100_000, d, lo, hi, rng)
    rel = ev.relative_l2(res.e, res.theta, points=test)
    want = tpl.canonical_key(TRUE[kind])
    # Exact recovery: snap the fitted parameters to rationals and prove the
    # candidate equal to the true solution with the zero test.
    true_theta = TRUE_THETA[kind](d)
    snapped = tpl.same_function(res.e, res.theta, TRUE[kind], true_theta, prob.x)
    return {
        "kind": kind,
        "d": d,
        "wall_s": wall,
        "epochs_run": len(res.history),
        "candidates_scored": res.candidates_scored,
        "unique_skeletons": res.unique_skeletons,
        "best_loss": res.loss,
        "relative_l2": rel,
        "best_key": res.key,
        "true_key": want,
        "exact_structure": res.key == want,
        "snapped_equal_to_truth": snapped,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["eval", "score", "solve", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    configs = [
        ("poisson", 10),
        ("poisson", 20),
        ("poisson", 50),
        ("conservation", 5),
        ("conservation", 20),
    ]
    if args.quick:
        configs = [("poisson", 10), ("conservation", 5)]
    out = {
        "threads": os.environ.get("RAYON_NUM_THREADS", str(os.cpu_count())),
        "alkahest": ak.__version__,
    }
    for what, fn in [("eval", bench_eval), ("score", bench_score), ("solve", bench_solve)]:
        if args.what not in (what, "all"):
            continue
        rows = []
        for kind, d in configs:
            kw = {"epochs": args.epochs} if what == "solve" else {}
            row = fn(kind, d, rng, args.quick, **kw)
            print(json.dumps(row), flush=True)
            rows.append(row)
        out[what] = rows
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
