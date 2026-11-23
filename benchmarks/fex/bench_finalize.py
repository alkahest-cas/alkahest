"""Benchmarks for ``alkahest.fex.finalize`` (milestone F3) on the FEX paper's
problems: from a search result to an exact closed form, verified at the
fitted dimension and — after lifting — for every dimension ``d``.

Problems (Liang & Yang, arXiv 2206.10121 / JMLR 26, 2025):

* Poisson ``−Δu = −d`` on ``[−1, 1]^d``, ``u = ½Σxᵢ²`` on the boundary
  (d = 10, 20, 50);
* linear conservation law ``(d−1)π/4 · u_t − Σ u_{xᵢ} = 0``,
  ``u(0, x) = sin(π/4 · Σxᵢ)`` on ``[0, 1]^d`` (d = 5, 20; ``d`` counts ``t``);
* cubic nonlinear Schrödinger ``−Δu + u³ + V u = 0`` on ``[−1, 1]^d`` with
  ``V = −⅑·exp(2/d·Σcos xᵢ) + Σ(sin² xᵢ / d² − cos xᵢ / d)``, exact solution
  ``⅓·exp(Σcos xᵢ / d)`` (normalised loss, no boundary condition);
* eigenvalue problem ``−Δu + V u = λ u`` on ``[−3, 3]^d`` (a truncation of
  ℝ^d) with ``V = |x|²`` (ground state ``exp(−|x|²/2)``, ``λ = d``) and with
  ``V = |x|² + 0.1·Σxᵢ⁴`` (no closed form).

Per problem: search time, finalize time, the exact closed form, the D9
status at the fitted ``d`` and for symbolic ``d``, the family it was proved
for, the Lean certificate status, and the recovery metric against the known
solution.

Usage::

    python benchmarks/fex/bench_finalize.py [poisson|conservation|nls|eigen|all]
        [--quick] [--json out.json] [--lean-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import alkahest as ak
import numpy as np
from alkahest import fex

C = math.pi / 4


def poisson(d):
    return fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(v**2 for v in x))


def conservation(d):
    coef = [(d - 1) * C] + [-1.0] * (d - 1)
    return fex.Problem.transport(coef, bc=lambda x: ak.sin(C * sum(x[1:])), time=0)


def nls(d):
    def V(x):
        S = sum(ak.cos(v) for v in x)
        return -ak.exp(2.0 / d * S) / 9 + sum(ak.sin(v) ** 2 for v in x) / d**2 - S / d

    return fex.Problem.cubic_schrodinger(d, potential=V)


def eigen(d, delta=0.0):
    def V(x):
        out = sum(v**2 for v in x)
        if delta:
            out = out + delta * sum(v**4 for v in x)
        return out

    return fex.Problem.eigen(d, potential=V)


def truth(kind, prob):
    x = prob.x
    if kind == "poisson":
        return sum(v**2 for v in x) / 2
    if kind == "conservation":
        return ak.sin(x[0] + sum(x[1:]) * prob.jet.pool.pi() / 4)
    if kind == "nls":
        return ak.exp(sum(ak.cos(v) for v in x) / len(x)) / 3
    if kind == "eigen":
        return ak.exp(-sum(v**2 for v in x) / 2)
    return None


def paper_family(kind, prob):
    """The family the paper means, written in symbolic d."""
    fs = fex.family_symbols(prob)
    pi = fs.pool.pi()
    if kind == "poisson":
        return {"rhs": -fs.d, "bc": fs.sum(lambda xi: xi**2 / 2)}
    if kind == "conservation":
        return {
            "time_coeff": pi * fs.d / 4,
            "space_coeff": -1,
            "bc": ak.sin(pi / 4 * fs.sum(lambda xi: xi)),
        }
    if kind == "nls":
        S = fs.sum(lambda xi: ak.cos(xi))
        return {
            "potential": -ak.exp(2 * S / fs.d) / 9
            + fs.sum(lambda xi: ak.sin(xi) ** 2) / fs.d**2
            - S / fs.d
        }
    return None


def run(kind, d, rng, quick, epochs, delta=0.0, lean_dir=None):
    tpl = fex.Template("depth2_sub", dim=d)
    if kind == "poisson":
        prob = poisson(d)
        xi = fex.sample_box(2000 if quick else 5000, d, -1.0, 1.0, rng)
        xb = fex.sample_box_boundary(500 if quick else 1000, d, -1.0, 1.0, rng)
    elif kind == "conservation":
        prob = conservation(d)
        xi = fex.sample_box(5000 if quick else 20000, d, 0.0, 1.0, rng)
        xb = fex.sample_initial_slice(1000 if quick else 4000, d, 0.0, 1.0, 0, 0.0, rng)
    elif kind == "nls":
        prob = nls(d)
        xi = fex.sample_box(2000 if quick else 5000, d, -1.0, 1.0, rng)
        xb = None
    else:
        prob = eigen(d, delta)
        xi = fex.sample_box(2000 if quick else 5000, d, -3.0, 3.0, rng)
        xb = fex.sample_box_boundary(500 if quick else 1000, d, -3.0, 3.0, rng)
    t0 = time.perf_counter()
    res = fex.search(
        tpl,
        prob,
        xi,
        xb,
        epochs=epochs,
        target_loss=1e-20 if kind != "eigen" else None,
        rng=rng,
    )
    t_search = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = fex.finalize(res, prob)
    t_fin = time.perf_counter() - t0
    v = out["verification"]
    sym = v["symbolic_d"] or {}
    conc = v["concrete"] or {}
    row = {
        "kind": kind if not delta else f"{kind}(delta={delta})",
        "d": d,
        "search_s": round(t_search, 2),
        "finalize_s": round(t_fin, 2),
        "skeleton": res.key,
        "search_loss": res.loss,
        "exact": out["exact"],
        "expr": str(out["expr"]),
        "concrete": conc.get("status"),
        "concrete_interior": conc.get("interior_verified"),
        "symbolic_d": sym.get("status"),
        "symbolic_d_interior": sym.get("interior_verified"),
        "expr_symbolic_d": str(out["expr_symbolic_d"]),
        "family": sym.get("family"),
        "other_families": [o["family"] for o in sym.get("other_families", [])],
        "lift_reason": sym.get("reason"),
        "certificate": (out["certificate"] or {}).get("status"),
        "certificate_reason": (out["certificate"] or {}).get("reason"),
        "eigenvalue": out.get("eigenvalue"),
        "interval": (v["interval"] or {}).get("max_abs_residual_upper"),
    }
    cert = out["certificate"] or {}
    if lean_dir and cert.get("status") == "emitted":
        path = f"{lean_dir}/fex_{kind}_{d}.lean"
        with open(path, "w") as f:
            f.write(cert["lean"])
        row["certificate_file"] = path
    tr = truth(kind, prob)
    if tr is not None and not delta:
        rm = fex.recovery_metric(out["expr"], tr)
        row["recovered_skeleton"] = rm["skeleton_equivalent"]
        row["recovered_exact"] = rm["exact"]
        # The negative of a solution of a homogeneous problem is a solution.
        if rm["exact"] is not True and kind in ("nls", "eigen"):
            row["recovered_exact_up_to_sign"] = ak.is_zero(out["expr"] + tr)
    fam = paper_family(kind, prob) if not delta else None
    if fam is not None and out["exact"]:
        t0 = time.perf_counter()
        out2 = fex.finalize(res, prob, family=fam)
        s2 = out2["verification"]["symbolic_d"] or {}
        row["paper_family"] = s2.get("status")
        row["paper_family_expr"] = str(out2["expr_symbolic_d"])
        row["paper_family_s"] = round(time.perf_counter() - t0, 2)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lean-dir", help="write emitted Lean certificates here")
    args = ap.parse_args(argv)
    plan = []
    if args.which in ("poisson", "all"):
        plan += [("poisson", d, 200, 0.0) for d in ((10,) if args.quick else (10, 20, 50))]
    if args.which in ("conservation", "all"):
        plan += [("conservation", d, 200, 0.0) for d in ((5,) if args.quick else (5, 20))]
    if args.which in ("nls", "all"):
        plan += [("nls", d, 200, 0.0) for d in ((6,) if args.quick else (6, 12, 24))]
    if args.which in ("eigen", "all"):
        plan += [("eigen", d, 100, 0.0) for d in ((4,) if args.quick else (4, 10))]
        plan += [("eigen", 10, 100, 0.1)] if not args.quick else []
    rows = []
    for kind, d, epochs, delta in plan:
        rng = np.random.default_rng(args.seed)
        row = run(kind, d, rng, args.quick, epochs, delta, args.lean_dir)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
