"""Benchmarks for milestone F4 of the FEX backend: symbolic pruning, the
controller options, and the GPU evaluator.

Problems (Liang & Yang, arXiv 2206.10121 / JMLR 26, 2025, and Song,
Cameron & Yang, arXiv 2306.12268):

* Poisson ``−Δu = −d`` on ``[−1, 1]^d``, ``u = ½Σxᵢ²`` on the boundary;
* linear conservation law ``(d−1)π/4 · u_t − Σ u_{xᵢ} = 0``,
  ``u(0, x) = sin(π/4 · Σxᵢ)`` on ``[0, 1]^d`` (``d`` counts ``t``);
* cubic NLS ``−Δu + u³ + V u = 0`` on ``[−1, 1]^d``, exact
  ``⅓·exp(Σcos xᵢ / d)``;
* eigenproblem ``−Δu + V u = λu`` on ``[−3, 3]^d`` with ``V = |x|²``
  (``λ = d``, ``u = exp(−|x|²/2)``) and ``V = |x|² + 0.1·Σxᵢ⁴`` (no closed
  form; reference from a converged 1-D finite-difference ground state, the
  problem being separable);
* committor of the double well ``V = (x₁² − 1)² + 0.3·Σ_{i≥2} xᵢ²``,
  ``β⁻¹ = 0.2``, ``d = 10``: ``β⁻¹Δq − ∇V·∇q = 0`` with ``q = 0`` at
  ``x₁ = −1`` and ``q = 1`` at ``x₁ = 1`` (reference: the exact 1-D
  committor ``∫e^{β(y²−1)²}`` by quadrature). Operator table: the paper's
  plus ``tanh``, as in Song–Cameron–Yang;
* the high-dimensional PIDEs of Hardwick, Liang & Yang (FEX-PG,
  arXiv 2410.00835, §4.3), ``d`` spatial coordinates plus ``t ∈ [0, 1]``,
  ``x ∈ [−1, 1]^d``, terminal condition ``u(1, x) = |x|²/d``, jumps
  ``z ~ N(μ, σ²)`` i.i.d. per coordinate at rate ``λ``:

  - ``pide4`` (their eq. 32): ``u_t + ½θ²Δu + λ∫(u(x+z) − u − z·∇u)φ(z)dz =
    λ(μ² + σ²) + θ²`` with ``θ = 0.3``, ``λ = 0.3``, ``μ = 1``, ``σ² = 10⁻⁴``
    (``ε = 0``, so no drift);
  - ``pide5`` (their eq. 33): drift ``(ε/2)|x|·x``, ``σ = θ(I + S)`` lower
    bidiagonal, ``ε = 0.05``, ``θ = 0.2``, ``λ = 0.3``, jump standard
    deviation ``10⁻⁴`` and ``μ = 1`` (the paper does not state ``μ``). The
    right-hand side is the one the stated solution satisfies,
    ``λ(μ² + σ²) + (2d−1)/d·θ² + (ε/d)|x|³``: with ``σ`` as printed,
    ``½Tr(σσᵀ·(2/d)I) = (2d−1)/d·θ²``, not the paper's ``(2d−2)/d·θ²``
    (see ``tests/test_fex_g6.py``).

Modes:

* ``suite`` — per problem, dimension and seed: the unpruned search
  (:func:`alkahest.fex.search`, the paper's operator tables and the
  reference controller) against :func:`alkahest.fex.solve` (analysis +
  pruned search); time to solution, relative L² on 20 000 fresh points,
  and :func:`alkahest.fex.finalize`'s verdict;
* ``controller`` — the controller options on the unpruned search space;
* ``gpu`` — candidates scored per second (batched Adam, 40 steps) on the
  CPU evaluator against :class:`alkahest.fex.TorchEvaluator` on CUDA.

Usage::

    python benchmarks/fex/bench_f4.py suite [--problems poisson,nls] [--seeds 3]
        [--dims poisson=10,20] [--epochs 200] [--no-finalize] [--json out.json]
    python benchmarks/fex/bench_f4.py controller [--seeds 5] [--json out.json]
    python benchmarks/fex/bench_f4.py gpu [--json out.json]
    python benchmarks/fex/bench_f4.py table --json suite.json
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

PAPER_DIMS = {
    "poisson": [10, 20, 30, 40, 50],
    "conservation": [5, 11, 17, 23, 29, 35, 41, 47, 53],
    "nls": [6, 12, 18, 24, 30, 36, 42, 48],
    "eigen": [10, 12, 14, 16, 18],
    "eigen_quartic": [10, 12, 14, 16, 18],
    "committor": [10],
    "pide4": [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "pide5": [25, 50, 75, 100],
}

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _ground_state_1d(delta: float, lo: float = -3.0, hi: float = 3.0, n: int = 6001):
    """1-D ground state of −φ'' + (x² + δx⁴)φ = μφ, φ(±3) = 0 (finite
    differences, n points): (μ, x grid, φ normalised to max 1)."""
    x = np.linspace(lo, hi, n)
    h = x[1] - x[0]
    inner = x[1:-1]
    diag = 2.0 / h**2 + inner**2 + delta * inner**4
    off = -np.ones(len(inner) - 1) / h**2
    try:
        from scipy.linalg import eigh_tridiagonal

        w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        mu, phi = float(w[0]), v[:, 0]
    except ImportError:  # pragma: no cover
        A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        w, v = np.linalg.eigh(A)
        mu, phi = float(w[0]), v[:, 0]
    phi = np.concatenate([[0.0], phi, [0.0]])
    phi = phi / phi[np.argmax(np.abs(phi))]
    return mu, x, phi


def _committor_1d(beta: float, n: int = 200001):
    y = np.linspace(-1.0, 1.0, n)
    f = np.exp(beta * (y**2 - 1.0) ** 2)
    cum = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) * 0.5 * (y[1] - y[0]))])
    return y, cum / cum[-1]


def make(kind: str, d: int, rng: np.random.Generator, n_test: int = 20000):
    """(problem, interior, boundary, test points, exact(pts) or None,
    unary table, success rule)."""
    unary = list(fex.PAPER_UNARY)
    if kind == "poisson":
        prob = fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(v**2 for v in x))
        xi = fex.sample_box(5000, d, -1.0, 1.0, rng)
        xb = fex.sample_box_boundary(1000, d, -1.0, 1.0, rng)
        xt = fex.sample_box(n_test, d, -1.0, 1.0, rng)
        exact = lambda p: 0.5 * (p**2).sum(1)  # noqa: E731
        mode = "plain"
    elif kind == "conservation":
        coef = [(d - 1) * C] + [-1.0] * (d - 1)
        prob = fex.Problem.transport(coef, bc=lambda x: ak.sin(C * sum(x[1:])), time=0)
        big = d >= 36
        xi = fex.sample_box(20000 if big else 5000, d, 0.0, 1.0, rng)
        xb = fex.sample_initial_slice(4000 if big else 1000, d, 0.0, 1.0, 0, 0.0, rng)
        xt = fex.sample_box(n_test, d, 0.0, 1.0, rng)
        exact = lambda p: np.sin(p[:, 0] + C * p[:, 1:].sum(1))  # noqa: E731
        mode = "plain"
    elif kind == "nls":

        def V(x):
            S = sum(ak.cos(v) for v in x)
            return -ak.exp(2.0 / d * S) / 9 + sum(ak.sin(v) ** 2 for v in x) / d**2 - S / d

        prob = fex.Problem.cubic_schrodinger(d, potential=V)
        xi = fex.sample_box(5000, d, -1.0, 1.0, rng)
        xb = None
        xt = fex.sample_box(n_test, d, -1.0, 1.0, rng)
        exact = lambda p: np.exp(np.cos(p).sum(1) / d) / 3  # noqa: E731
        mode = "sign"
    elif kind in ("eigen", "eigen_quartic"):
        delta = 0.1 if kind == "eigen_quartic" else 0.0

        def V(x):
            out = sum(v**2 for v in x)
            return out + delta * sum(v**4 for v in x) if delta else out

        prob = fex.Problem.eigen(d, potential=V)
        xi = fex.sample_box(5000, d, -3.0, 3.0, rng)
        xb = fex.sample_box_boundary(1000, d, -3.0, 3.0, rng)
        # Test where the eigenfunction has mass (Gaussian-weighted points).
        xt = np.clip(rng.normal(0.0, 1.0 / math.sqrt(2), size=(n_test, d)), -3, 3)
        if delta:
            mu, grid, phi = _ground_state_1d(delta)
            prob.reference_eigenvalue = d * mu
            exact = lambda p: np.prod(np.interp(p, grid, phi), axis=1)  # noqa: E731
        else:
            prob.reference_eigenvalue = float(d)
            exact = lambda p: np.exp(-0.5 * (p**2).sum(1))  # noqa: E731
        mode = "scale"
    elif kind == "committor":
        beta = 5.0
        jet = fex.Jet(d)
        x = jet.x
        dV = [4 * x[0] * (x[0] ** 2 - 1)] + [0.6 * v for v in x[1:]]
        op = jet.lap / beta - sum(a * g for a, g in zip(dV, jet.grad))
        prob = fex.Problem.from_operator(jet, op, rhs=0.0, bc=(1 + x[0]) / 2, bc_weight=100.0)
        xi = rng.uniform(-1.0, 1.0, size=(5000, d))
        xi[:, 1:] = rng.normal(0.0, math.sqrt(1 / (0.6 * beta)), size=(5000, d - 1))
        xb = rng.uniform(-1.0, 1.0, size=(1000, d))
        xb[:, 1:] = rng.normal(0.0, math.sqrt(1 / (0.6 * beta)), size=(1000, d - 1))
        xb[:500, 0] = -1.0
        xb[500:, 0] = 1.0
        xt = rng.uniform(-1.0, 1.0, size=(n_test, d))
        xt[:, 1:] = rng.normal(0.0, math.sqrt(1 / (0.6 * beta)), size=(n_test, d - 1))
        y, q = _committor_1d(beta)
        exact = lambda p: np.interp(p[:, 0], y, q)  # noqa: E731
        unary = [*unary, "tanh"]
        mode = "plain"
    elif kind in ("pide4", "pide5"):
        prob = make_pide(kind, d)
        dim = d + 1
        xi = fex.sample_box(5000, dim, -1.0, 1.0, rng)
        xi[:, 0] = rng.uniform(0.0, 1.0, 5000)
        xb = fex.sample_initial_slice(1000, dim, -1.0, 1.0, 0, 1.0, rng)
        xt = fex.sample_box(n_test, dim, -1.0, 1.0, rng)
        xt[:, 0] = rng.uniform(0.0, 1.0, n_test)
        exact = lambda p: (p[:, 1:] ** 2).sum(1) / d  # noqa: E731
        mode = "plain"
    else:
        raise ValueError(kind)
    return prob, xi, xb, xt, exact, unary, mode


def make_pide(kind: str, d: int):
    """The FEX-PG benchmark PIDEs (``d`` spatial coordinates; coordinate 0
    is ``t``), with exact constants."""
    from fractions import Fraction as Q

    jet = fex.Jet(d + 1, time=0)
    pool = jet.pool
    x = jet.x[1:]
    lam, mu = Q(3, 10), Q(1)
    r2 = sum(v**2 for v in x)
    if kind == "pide4":
        theta, var = Q(3, 10), Q(1, 10000)
        rhs = lam * (mu**2 + var) + theta**2
        return fex.Problem.pide(
            d + 1,
            time=0,
            diffusion=theta,
            jump=fex.GaussianJump(lam, mu, var),
            rhs=pool.rational(rhs.numerator, rhs.denominator),
            terminal=r2 / d,
            T=1,
            jet=jet,
        )
    eps, theta, var = Q(1, 20), Q(1, 5), Q(1, 10**8)
    c0 = lam * (mu**2 + var) + Q(2 * d - 1, d) * theta**2
    c1 = eps / d
    rhs = pool.rational(c0.numerator, c0.denominator) + pool.rational(
        c1.numerator, c1.denominator
    ) * r2 ** pool.rational(3, 2)
    half_eps = eps / 2
    return fex.Problem.pide(
        d + 1,
        time=0,
        drift=fex.RadialDrift(
            lambda r: pool.rational(half_eps.numerator, half_eps.denominator) * r
        ),
        diffusion=fex.BidiagonalDiffusion(theta),
        jump=fex.GaussianJump(lam, mu, var),
        rhs=rhs,
        terminal=r2 / d,
        T=1,
        jet=jet,
    )


def pide_family(kind: str, prob):
    """The FEX-PG problem family in symbolic ``d`` (for ``finalize(family=)``):
    ``pide5``'s right-hand side has a ``(2d − 1)/d`` that one dimension does
    not determine."""
    if kind != "pide5":
        return None
    import alkahest as ak

    fs = fex.family_symbols(prob)
    pool = fs.pool
    lam, mu, var, theta, eps = (
        pool.rational(3, 10),
        pool.integer(1),
        pool.rational(1, 10**8),
        pool.rational(1, 5),
        pool.rational(1, 20),
    )
    r2 = fs.sum(lambda xi: xi**2)
    rhs = (
        lam * (mu**2 + var)
        + (2 * fs.d - 1) / fs.d * theta**2
        + eps / fs.d * r2 ** pool.rational(3, 2)
    )
    _ = ak
    return {"rhs": rhs, "bc": r2 / fs.d}


def rel_l2(res, prob, xt, exact, mode) -> float:
    tpl = res.template
    ev = fex.Evaluator(tpl, prob, xt[:10], None)
    u, _, _ = ev.jets(res.e, res.theta, xt)
    ref = exact(xt)
    if not np.all(np.isfinite(u)):
        return math.inf
    if mode == "scale":
        c = float(u @ ref) / max(float(u @ u), 1e-300)
        u = c * u
    elif mode == "sign":
        u = u if np.linalg.norm(u - ref) <= np.linalg.norm(u + ref) else -u
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def eigenvalue_of(res, prob, xt, delta) -> float | None:
    """Eigenvalue estimate of the found u: the u²-weighted mean of the local
    energy (−Δu + V u)/u at the test points (exact for an eigenfunction,
    unlike a Monte Carlo Rayleigh quotient in high dimension)."""
    if prob.loss != "rayleigh":
        return None
    ev = fex.Evaluator(res.template, prob, xt[:10], None)
    u, _, lap = ev.jets(res.e, res.theta, xt)
    V = (xt**2).sum(1) + delta * (xt**4).sum(1)
    w = u * u
    if not np.all(np.isfinite(lap)) or w.sum() <= 0:
        return None
    return float(np.sum(w * (-lap + V * u) / np.where(u == 0, 1, u)) / w.sum())


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def run_one(kind, d, seed, method, epochs, do_finalize, controller=None):
    rng = np.random.default_rng(seed)
    prob, xi, xb, xt, exact, unary, mode = make(kind, d, rng)
    t0 = time.perf_counter()
    target = None if prob.loss == "rayleigh" else 1e-20
    if method == "unpruned":
        tpl = fex.Template("depth2_sub", unary=unary, dim=d)
        res = fex.search(
            tpl, prob, xi, xb, epochs=epochs, target_loss=target, rng=rng, **(controller or {})
        )
        tts = res.seconds_to_target
        prop = None
    else:
        res = fex.solve(
            prob,
            xi,
            xb,
            unary=unary,
            epochs=epochs,
            rng=rng,
            stall_epochs=20,
            restrict="soft" if method == "soft" else "hard",
            **(controller or {}),
        )
        tts = res.seconds_to_target
        prop = res.proposal.name if res.proposal is not None else None
    t_search = time.perf_counter() - t0
    err = rel_l2(res, prob, xt, exact, mode)
    row = {
        "problem": kind,
        "d": d,
        "seed": seed,
        "method": method,
        "seconds": round(t_search, 2),
        "seconds_to_target": None if tts is None else round(tts, 2),
        "loss": res.loss,
        "key": res.key,
        "rel_l2": err,
        "candidates_scored": res.candidates_scored,
        "proposal": prop,
        "attempts": None
        if method == "unpruned"
        else [
            {k: a.get(k) for k in ("proposal", "search_loss", "original_loss", "key", "seconds")}
            for a in res.attempts
        ],
    }
    lam = eigenvalue_of(res, prob, xt, 0.1 if kind == "eigen_quartic" else 0.0)
    if lam is not None:
        row["eigenvalue"] = lam
        row["eigenvalue_rel_err"] = abs(lam - prob.reference_eigenvalue) / prob.reference_eigenvalue
    if do_finalize and err <= 1e-4 and kind not in ("committor", "eigen_quartic"):
        t1 = time.perf_counter()
        try:
            out = fex.finalize(res, prob, family=pide_family(kind, prob))
            v = out["verification"]
            row["finalize_s"] = round(time.perf_counter() - t1, 2)
            row["finalize_exact"] = out["exact"]
            row["verified_concrete"] = (v["concrete"] or {}).get("status")
            row["verified_concrete_interior"] = (v["concrete"] or {}).get("interior_verified")
            row["verified_symbolic_d"] = (v["symbolic_d"] or {}).get("status")
            row["verified_symbolic_d_interior"] = (v["symbolic_d"] or {}).get("interior_verified")
            row["eigenvalue_exact"] = out.get("eigenvalue")
            row["expr"] = str(out["expr"])[:240]
            cert = out.get("certificate") or {}
            row["certificate"] = cert.get("status")
            row["certificate_path"] = cert.get("path")
            row["certificate_reason"] = cert.get("reason")
            if cert.get("lean") and getattr(run_one, "lean_dir", None):
                import hashlib

                h = hashlib.sha1(cert["lean"].encode()).hexdigest()[:10]
                fn = f"{run_one.lean_dir}/{kind}_{d}_{seed}_{method}_{h}.lean"
                with open(fn, "w") as f:
                    f.write(cert["lean"])
                row["certificate_file"] = fn
            conc = v["concrete"] or {}
            if "taylor_exact_on_solution" in conc:
                row["taylor_exact_on_solution"] = conc["taylor_exact_on_solution"]
        except Exception as exc:  # report, never hide
            row["finalize_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def suite(args) -> list[dict]:
    problems = args.problems.split(",") if args.problems else list(PAPER_DIMS)
    dims = dict(PAPER_DIMS)
    for spec in args.dims or []:
        k, v = spec.split("=")
        dims[k] = [int(t) for t in v.split(",")]
    methods = args.methods.split(",")
    rows = []
    for kind in problems:
        for d in dims[kind]:
            for seed in range(args.seeds):
                for m in methods:
                    row = run_one(kind, d, seed, m, args.epochs, not args.no_finalize)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------

CONTROLLERS = {
    "reference": {},
    "entropy": {"entropy_weight": 0.005},
    "pqt": {"pqt_size": 10, "pqt_weight": 0.2},
    "population": {"population": 4, "batch": 3, "restart_patience": 25},
    "curriculum": {"curriculum": 15},
    "ewma": {"baseline": "ewma"},
    "dso": {
        "entropy_weight": 0.005,
        "pqt_size": 10,
        "pqt_weight": 0.2,
        "baseline": "ewma",
        "restart_patience": 40,
        "curriculum": 15,
    },
}

CONTROLLER_GRID = [("nls", 12), ("nls", 18), ("conservation", 11), ("eigen", 10)]


def controller(args) -> list[dict]:
    rows = []
    names = args.controllers.split(",") if args.controllers else list(CONTROLLERS)
    for kind, d in CONTROLLER_GRID:
        for name in names:
            for seed in range(args.seeds):
                row = run_one(kind, d, seed, "unpruned", args.epochs, False, CONTROLLERS[name])
                row["controller"] = name
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------


def gpu(args) -> list[dict]:
    import torch

    rows = []
    for kind, d in [
        ("poisson", 10),
        ("poisson", 50),
        ("conservation", 20),
        ("nls", 12),
        ("eigen", 10),
    ]:
        rng = np.random.default_rng(0)
        prob, xi, xb, _xt, _ex, unary, _m = make(kind, d, rng, n_test=10)
        tpl = fex.Template("depth2_sub", unary=unary, dim=d)
        n = args.batch
        es = tpl.random_skeletons(n, rng)
        th = tpl.init_params(n, rng)
        ev = fex.Evaluator(tpl, prob, xi, xb)
        t = time.perf_counter()
        ev.fit(es, th, method="adam", max_iter=40, lr=1e-3)
        cpu = n / (time.perf_counter() - t)
        row = {
            "problem": kind,
            "d": d,
            "batch": n,
            "cpu_threads": args.threads,
            "cpu_cand_s": round(cpu, 1),
        }
        for dt in ("float64", "float32"):
            tev = fex.TorchEvaluator(tpl, prob, xi, xb, device="cuda", dtype=dt, max_batch=n)
            tev.fit(es[:16], th[:16], max_iter=2)
            torch.cuda.synchronize()
            t = time.perf_counter()
            tev.fit(es, th, max_iter=40, lr=1e-3)
            torch.cuda.synchronize()
            row[f"gpu_{dt}_cand_s"] = round(n / (time.perf_counter() - t), 1)
        # Agreement of the two evaluators on these candidates (float64).
        tev = fex.TorchEvaluator(tpl, prob, xi, xb, device="cuda", dtype="float64", max_batch=n)
        l1 = ev.loss(es[:256], th[:256])
        l2 = tev.loss(es[:256], th[:256])
        fin = np.isfinite(l1) & np.isfinite(l2)
        row["max_rel_diff"] = float(
            np.max(np.abs(l1 - l2)[fin] / np.maximum(np.abs(l1[fin]), 1e-300))
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[len(xs) // 2]


def _fmt(v, spec="{:.2f}"):
    return "—" if v is None else spec.format(v)


def table(args) -> list[dict]:
    """Markdown summary of a ``suite`` JSON (``--json`` names the input)."""
    with open(args.json) as f:
        rows = json.load(f)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["problem"], r["d"]), {}).setdefault(r["method"], []).append(r)
    print(
        "| problem | d | method | solved | time to solution (median s) | total (median s) "
        "| candidates | rel. L² (median) | finalize (median s) "
        "| finalize proves (fitted d / all d) | Lean certificate emitted |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (kind, d), by in groups.items():
        for m in ("unpruned", "pruned", "soft"):
            rs = by.get(m, [])
            if not rs:
                continue
            ok = [r for r in rs if r["rel_l2"] <= 1e-6]
            fin = sum(1 for r in rs if r.get("verified_concrete_interior"))
            sym = sum(1 for r in rs if r.get("verified_symbolic_d_interior"))
            tts = _median([r["seconds_to_target"] for r in rs])
            cert = sum(1 for r in rs if r.get("certificate") == "emitted")
            print(
                f"| {kind} | {d} | {m} | {len(ok)}/{len(rs)} | {_fmt(tts)} "
                f"| {_fmt(_median([r['seconds'] for r in rs]))} "
                f"| {_median([r['candidates_scored'] for r in rs])} "
                f"| {_fmt(_median([r['rel_l2'] for r in rs]), '{:.1e}')} "
                f"| {_fmt(_median([r.get('finalize_s') for r in rs]))} "
                f"| {fin}/{sym} of {len(rs)} | {cert}/{len(rs)} |"
            )
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["suite", "controller", "gpu", "table"])
    ap.add_argument("--problems")
    ap.add_argument("--dims", action="append")
    ap.add_argument("--methods", default="unpruned,pruned")
    ap.add_argument("--controllers")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--lean-dir", help="write each emitted Lean certificate here")
    args = ap.parse_args(argv)
    run_one.lean_dir = args.lean_dir
    rows = {"suite": suite, "controller": controller, "gpu": gpu, "table": table}[args.mode](args)
    if args.json and args.mode != "table":
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
