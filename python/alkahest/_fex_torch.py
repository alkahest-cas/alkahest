"""A PyTorch (CUDA) template interpreter for FEX scoring (milestone F4, R10).

:class:`TorchEvaluator` evaluates a batch of candidates of one template on
one point set as dense tensors — ``(batch, points, d)`` for the gradients —
with the same fused Forward-Laplacian recursion as the Rust interpreter:

* a leaf ``Σᵢ αᵢ φ(xᵢ) + β`` gathers the point tables ``φ(x), φ'(x), φ''(x)``
  of its operator (computed once per point set) and contracts them with
  ``α``;
* an inner slot ``α·g(s) + β`` maps ``(s, ∇s, Δs)`` to
  ``(g(s), g'(s)∇s, g''(s)|∇s|² + g'(s)Δs)``; binary slots use the sum and
  product rules (``Δ(ab) = aΔb + bΔa + 2∇a·∇b``).

Each slot's operator is chosen per candidate by its opcode, so one compiled
graph scores any mix of skeletons, and the parameter gradient of the loss
comes from autograd over the batch (candidates are independent). It runs
on whatever device ``torch`` gives it (``"cuda"`` on the 3090s, ``"cpu"``
otherwise), in float64 by default or float32.

It covers the built-in problems — Poisson (and the eigenproblem's Rayleigh
quotient), the linear conservation law, the cubic Schrödinger equation —
with their least-squares, normalised and Rayleigh losses, and operators
given as expressions (``Problem.from_operator``, the problems
:func:`alkahest.fex.transform_problem` and :func:`~alkahest.fex.reduce_problem`
derive, and PIDEs from ``Problem.pide`` with their nonlocal terms) with the
least-squares, centred and normalised losses, matching
:class:`alkahest.fex.Evaluator`'s loss to rounding.

An expression operator is split, symbolically and once, as
``F = F₀(u, Δu, x) + Σᵢ aᵢ(x) ∂ᵢu + b(u, Δu, x)·Σ_{i∈S} (∂ᵢu)²`` (``S`` the
spatial coordinates); the split is checked to be exact (``F`` minus the
three parts simplifies to ``0``), and an operator of any other form — a
gradient inside a non-polynomial function, a gradient coefficient that
depends on ``u`` — is refused with ``E-FEX-014``. ``F₀``, ``aᵢ`` and ``b`` are
translated to ``torch`` operations; the gradient terms use the same leaf
Gram contractions as the built-in operators, so no ``(batch, points, d)``
tensor is formed. Problems with a non-diagonal second-order form
(``Problem.pide`` with a matrix diffusion) stay on the CPU evaluator.
:meth:`TorchEvaluator.fit` is batched Adam, the GPU counterpart of the
coarse scoring step; pass the evaluator as ``scorer=`` to
:func:`alkahest.fex.search`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["TorchEvaluator"]

_REMEDIATION = (
    "install PyTorch (with CUDA for device='cuda'); the GPU evaluator covers the built-in "
    "problems and expression operators of the form F0(u, lap, x) + sum a_i(x) u_i + "
    "b(u, lap, x) |grad u|^2 (least-squares, centred, normalised and Rayleigh losses, without "
    "a non-diagonal second-order form); anything else runs on alkahest.fex.Evaluator"
)


def _err(message: str) -> Exception:
    from .alkahest import FexError

    exc = FexError(f"[E-FEX-014] {message}\nRemediation: {_REMEDIATION}")
    exc.code = "E-FEX-014"
    exc.remediation = _REMEDIATION
    exc.span = None
    return exc


def _source_values(v: Any, coords: list[Any], pts: np.ndarray) -> np.ndarray:
    """A number or an expression in ``coords``, evaluated at the points."""
    import alkahest as ak

    n = pts.shape[0]
    if v is None:
        return np.zeros(n)
    if isinstance(v, (int, float)):
        return np.full(n, float(v))
    f = ak.compile_expr(v, list(coords))
    return np.asarray(ak.numpy_eval(f, *[pts[:, i].copy() for i in range(pts.shape[1])]))


class TorchEvaluator:
    """Batched candidate scoring on a ``torch`` device.

    Parameters
    ----------
    template, problem : alkahest.fex.Template, alkahest.fex.Problem
    points, boundary : arrays ``(n, d)`` / ``(m, d)``
    device : str
        ``"cuda"``, ``"cuda:1"``, ``"cpu"``, …
    dtype : str
        ``"float64"`` (default) or ``"float32"``.
    max_batch : int
        Candidates evaluated together; larger batches are split.

    Raises :class:`FexError` ``E-FEX-014`` when torch (or the device) is
    missing or the problem is not one of the built-in kinds.
    """

    def __init__(
        self,
        template: Any,
        problem: Any,
        points: Any,
        boundary: Any = None,
        *,
        device: str = "cuda",
        dtype: str = "float64",
        max_batch: int = 1024,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise _err(f"PyTorch is not installed ({exc})") from None
        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise _err("CUDA is not available to PyTorch")
        spec = problem.spec or {}
        kind = spec.get("kind")
        if kind not in ("poisson", "transport", "cubic_schrodinger", "expr"):
            raise _err(f"problem kind {kind!r} is not supported on the torch evaluator")
        if problem.loss not in ("least_squares", "normalized", "rayleigh", "centered"):
            raise _err(f"loss {problem.loss!r} is not supported on the torch evaluator")
        if spec.get("second_order") is not None:
            raise _err("a non-diagonal second-order form is not supported on the torch evaluator")
        self.template = template
        self.problem = problem
        self.device = torch.device(device)
        self.dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
        self.max_batch = int(max_batch)
        self.kind = kind
        self.loss_kind = problem.loss
        self.power = float(spec.get("power", 3.0))
        self.bc_weight = float(spec.get("bc_weight", 100.0))
        d = template.dim
        pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, d)
        bd = (
            None
            if boundary is None
            else np.ascontiguousarray(boundary, dtype=np.float64).reshape(-1, d)
        )
        coords = list(problem.x)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)  # noqa: E731
        self.x = T(pts)
        self.xb = None if bd is None or len(bd) == 0 else T(bd)
        self._blocks: dict[str, Any] = {"int": self.x, "bd": self.xb}
        self.nonlocal_terms: list[tuple[str, float, float]] = []
        for k, (scale, shift, cu, clap) in enumerate(spec.get("nonlocal_terms") or []):
            y = pts * (np.asarray(scale, dtype=np.float64) if len(scale) else 1.0)
            y = y + np.asarray(shift, dtype=np.float64)
            self._blocks[f"nl{k}"] = T(y)
            self.nonlocal_terms.append((f"nl{k}", float(cu), float(clap)))
        self.rhs = T(_source_values(spec.get("rhs"), coords, pts))
        pot = spec.get("potential")
        self.pot = None if pot is None else T(_source_values(pot, coords, pts))
        self.target = (
            None if self.xb is None or bd is None else T(_source_values(spec.get("bc"), coords, bd))
        )
        mask = np.ones(d)
        if problem.time is not None:
            mask[problem.time] = 0.0
        self.mask = T(mask)
        gc = spec.get("grad_coeffs")
        self.grad_coeffs = None if gc is None else T(gc)
        self.expr_op = None
        if kind == "expr":
            self.expr_op = _split_operator(problem, pts, T, mask)
        from .fex import _shape_tree

        self.tree = _shape_tree(template.shape)
        self.root = next(i for i, n in enumerate(self.tree) if n["depth"] == 0)
        post: list[int] = []

        def visit(i: int) -> None:
            for c in self.tree[i]["children"]:
                visit(c)
            post.append(i)

        visit(self.root)
        self.post = post
        self.offs = list(template.param_offsets)
        self.unary = list(template.unary)
        self.binary = list(template.binary)
        for name in self.unary:
            if name not in _UNARY:
                raise _err(f"unary operator {name!r} has no torch kernel")
        self._tables: dict[tuple, Any] = {}
        self._group_cache: dict[tuple, Any] = {}

    # -- point tables -------------------------------------------------------

    def _table(self, k: int, boundary: Any) -> tuple[Any, Any, Any]:
        block = _block_name(boundary)
        key = (k, block)
        if key not in self._tables:
            self._tables[key] = _UNARY[self.unary[k]](self._blocks[block])
        return self._tables[key]

    def _coeff_table(self, k: int) -> Any:
        """``aᵢ(x_p) · φ'_k(x_pi)`` for an expression operator's gradient
        coefficients (interior points)."""
        key = ("coeff", k)
        if key not in self._tables:
            self._tables[key] = self.expr_op["a"] * self._table(k, "int")[1]
        return self._tables[key]

    # -- forward ------------------------------------------------------------

    def _groups(self, ops: Any) -> tuple[list, Any]:
        """``[(k, idx)]`` and the inverse permutation for a column of opcodes,
        cached per batch (one host sync per distinct column, not per call)."""
        key = id(ops), ops.data_ptr() if hasattr(ops, "data_ptr") else 0
        hit = self._group_cache.get(key)
        if hit is not None and hit[0] is ops:
            return hit[1], hit[2]
        torch = self.torch
        ks = torch.unique(ops).tolist()
        groups = [(int(k), (ops == k).nonzero(as_tuple=True)[0]) for k in ks]
        perm = torch.argsort(torch.cat([g[1] for g in groups]))
        self._group_cache[key] = (ops, groups, perm)
        return groups, perm

    def _grouped_mm(self, ops: Any, w: Any, tab: Any) -> Any:
        """``out[b] = tab[ops[b]] @ w[b]`` for per-candidate tables
        (``tab(k)`` is ``(N, d)``), as one matrix product per distinct op."""
        torch = self.torch
        groups, perm = self._groups(ops)
        if len(groups) == 1:
            return w @ tab(groups[0][0]).T
        out = torch.cat([w[idx] @ tab(k).T for k, idx in groups])
        return out[perm]

    def _column(self, e: Any, s: int) -> Any:
        ck = ("col", id(e), s)
        hit = self._group_cache.get(ck)
        if hit is None or hit[0] is not e:
            hit = (e, e[:, s])
            self._group_cache[ck] = hit
        return hit[1]

    def _pair_column(self, e: Any, key: tuple[int, int]) -> Any:
        ck = ("paircol", id(e), key)
        hit = self._group_cache.get(ck)
        if hit is None or hit[0] is not e:
            col = e[:, key[0]] * len(self.unary) + e[:, key[1]]
            hit = (e, col)
            self._group_cache[ck] = hit
        return hit[1]

    def _pair_table(self, k: int, k2: int, boundary: Any) -> Any:
        key = ("pair", min(k, k2), max(k, k2), _block_name(boundary))
        if key not in self._tables:
            t1 = self._table(k, boundary)[1]
            t2 = self._table(k2, boundary)[1]
            self._tables[key] = t1 * t2 * self.mask
        return self._tables[key]

    def _forward(self, e: Any, th: Any, boundary: Any, order: int):
        """Root ``(u, lap, grads)`` for the batch.

        The gradient of a node is never formed as a ``(batch, points, d)``
        tensor: it is ``Σ_ℓ c_ℓ · (w_ℓ ⊙ φ'_ℓ(x))`` over the leaves ``ℓ`` below
        it, kept as the per-point coefficients ``c_ℓ``. Dot products of
        gradients then need only the leaf Gram terms
        ``G_ℓm = Σᵢ maskᵢ w_ℓᵢ w_mᵢ φ'_ℓ(xᵢ) φ'_m(xᵢ)``, each a matrix product
        of a point table with the candidates' weights."""
        torch = self.torch
        x: Any = self._blocks[_block_name(boundary)]
        d = x.shape[1]
        vals: dict[int, Any] = {}
        W: dict[int, Any] = {}
        gram: dict[tuple[int, int], Any] = {}

        def G(l1: int, l2: int) -> Any:
            key = (min(l1, l2), max(l1, l2))
            if key not in gram:
                ww = W[key[0]] * W[key[1]]
                pair = self._pair_column(e, key)
                gram[key] = self._grouped_mm(
                    pair,
                    ww,
                    lambda p: self._pair_table(p // len(self.unary), p % len(self.unary), boundary),
                )
            return gram[key]

        def dot(ca: dict, cb: dict) -> Any:
            out = 0.0
            for l1, c1 in ca.items():
                for l2, c2 in cb.items():
                    out = out + c1 * c2 * G(l1, l2)
            return out

        def scale(c: dict, f: Any) -> dict:
            return {k: v * f for k, v in c.items()}

        def add(ca: dict, cb: dict, sb: float = 1.0) -> dict:
            out = dict(ca)
            for k, v in cb.items():
                out[k] = out[k] + sb * v if k in out else sb * v
            return out

        for s in self.post:
            node = self.tree[s]
            o = self.offs[s]
            ops = self._column(e, s)
            if node["kind"] == "leaf":
                w = th[:, o : o + d]
                W[s] = w
                u = self._grouped_mm(ops, w, lambda k: self._table(k, boundary)[0])
                u = u + th[:, o + d][:, None]
                lap = (
                    self._grouped_mm(ops, w * self.mask, lambda k: self._table(k, boundary)[2])
                    if order >= 2
                    else None
                )
                vals[s] = (u, lap, {s: torch.ones_like(u)} if order else {})
            elif node["kind"] == "unary":
                su, sl, sc = vals[node["children"][0]]
                a = th[:, o][:, None]
                b = th[:, o + 1][:, None]
                fv = torch.zeros_like(su)
                d1 = torch.zeros_like(su)
                d2 = torch.zeros_like(su)
                for k, _idx in self._groups(ops)[0]:
                    sel = (ops == k)[:, None]
                    v0, v1, v2 = _UNARY[self.unary[int(k)]](su)
                    fv = torch.where(sel, v0, fv)
                    d1 = torch.where(sel, v1, d1)
                    d2 = torch.where(sel, v2, d2)
                u = a * fv + b
                c = scale(sc, a * d1)
                lap = a * (d2 * dot(sc, sc) + d1 * sl) if order >= 2 else None
                vals[s] = (u, lap, c)
            else:
                lu, ll, lc = vals[node["children"][0]]
                ru, rl, rc = vals[node["children"][1]]
                u = torch.zeros_like(lu)
                lap: Any = torch.zeros_like(lu) if order >= 2 else None
                c: dict = {k: torch.zeros_like(lu) for k in set(lc) | set(rc)}
                for k, _idx in self._groups(ops)[0]:
                    sel = (ops == k)[:, None]
                    op = self.binary[int(k)]
                    if op in ("+", "-"):
                        sg = 1.0 if op == "+" else -1.0
                        uk = lu + sg * ru
                        ck: dict = add(lc, rc, sg)
                        lk: Any = ll + sg * rl if order >= 2 else None
                    elif op == "*":
                        uk = lu * ru
                        ck = add(scale(lc, ru), scale(rc, lu))
                        lk = lu * rl + ru * ll + 2 * dot(lc, rc) if order >= 2 else None
                    else:  # "/" = a · (1/b)
                        inv = 1.0 / ru
                        uk = lu * inv
                        ic = scale(rc, -(inv * inv))
                        ck = add(scale(lc, inv), scale(ic, lu))
                        lk = None
                        if order >= 2:
                            il = -(inv * inv) * rl + 2 * inv**3 * dot(rc, rc)
                            lk = lu * il + inv * ll + 2 * dot(lc, ic)
                    u = torch.where(sel, uk, u)
                    for key in c:
                        if key in ck:
                            c[key] = torch.where(sel, ck[key], c[key])
                    if lap is not None and lk is not None:
                        lap = torch.where(sel, lk, lap)
                vals[s] = (u, lap, c)
        u, lap, c = vals[self.root]
        return u, lap, c, dot, W

    def _loss(self, e: Any, th: Any) -> Any:
        if self.expr_op is not None:
            order = 2 if self.expr_op["uses_lap"] else 1
        else:
            order = 1 if (self.loss_kind == "rayleigh" or self.kind == "transport") else 2
        self._th_cur = th
        u, lap, c, dot, W = self._forward(e, th, False, order)
        if self.loss_kind == "rayleigh":
            pot = self.pot if self.pot is not None else 0.0
            num = (dot(c, c) + pot * u * u).mean(-1)
            den = (u * u).mean(-1)
            loss = num / den
        else:
            if self.kind == "poisson":
                r = -lap
                if self.pot is not None:
                    r = r + self.pot * u
            elif self.kind == "transport":
                # Σᵢ cᵢ ∂ᵢu = Σ_ℓ c_ℓ · ((w_ℓ ⊙ coeffs) · φ'_ℓ(x))
                r = 0.0
                for leaf, cl in c.items():
                    r = r + cl * self._grouped_mm(
                        self._column(e, leaf),
                        W[leaf] * self.grad_coeffs,
                        lambda k: self._table(k, False)[1],
                    )
            elif self.kind == "cubic_schrodinger":
                r = -lap + u**3
                if self.pot is not None:
                    r = r + self.pot * u
            else:
                r = self._expr_residual(e, u, lap, c, dot, W)
            r = r - self.rhs
            if self.loss_kind == "centered":
                r = r - r.mean(-1, keepdim=True)
            loss = (r * r).mean(-1)
            if self.loss_kind == "normalized":
                loss = loss / (u.abs() ** self.power).mean(-1)
        if self.xb is not None and self.bc_weight:
            ub = self._forward(e, th, True, 0)[0]
            loss = loss + self.bc_weight * ((ub - self.target) ** 2).mean(-1)
        return loss

    def _expr_residual(self, e: Any, u: Any, lap: Any, c: dict, dot: Any, W: dict) -> Any:
        """``F₀(u, Δu, x) + Σᵢ aᵢ(x) ∂ᵢu + b·|∇u|² (+ nonlocal terms)``."""
        op = self.expr_op
        env = {"u": u, "lap": lap}
        r = op["F0"](env)
        if op["a"] is not None:
            for leaf, cl in c.items():
                r = r + cl * self._grouped_mm(self._column(e, leaf), W[leaf], self._coeff_table)
        if op["b"] is not None:
            r = r + op["b"](env) * dot(c, c)
        for block, cu, clap in self.nonlocal_terms:
            uy, ly, _cy, _d, _w = self._forward(e, self._th_cur, block, 2 if clap else 0)
            r = r + cu * uy
            if clap:
                r = r + clap * ly
        return r

    # -- public -------------------------------------------------------------

    def _tensors(self, e: Any, theta: Any):
        torch = self.torch
        e = torch.as_tensor(np.asarray(e, dtype=np.int64), device=self.device)
        th = torch.as_tensor(np.asarray(theta, dtype=np.float64), device=self.device)
        return e.reshape(-1, self.template.n_slots), th.reshape(-1, self.template.n_params).to(
            self.dtype
        )

    def loss(self, e: Any, theta: Any) -> np.ndarray:
        """Losses of a batch (``+inf`` where not finite)."""
        torch = self.torch
        self._group_cache.clear()
        E, TH = self._tensors(e, theta)
        out = []
        with torch.no_grad():
            for i in range(0, E.shape[0], self.max_batch):
                out.append(self._loss(E[i : i + self.max_batch], TH[i : i + self.max_batch]))
        v = torch.cat(out).double().cpu().numpy()
        return np.where(np.isfinite(v), v, np.inf)

    def loss_and_grad(self, e: Any, theta: Any) -> tuple[np.ndarray, np.ndarray]:
        """Losses and their gradients in ``theta``."""
        torch = self.torch
        self._group_cache.clear()
        E, TH = self._tensors(e, theta)
        ls, gs = [], []
        for i in range(0, E.shape[0], self.max_batch):
            th = TH[i : i + self.max_batch].clone().requires_grad_(True)
            loss = self._loss(E[i : i + self.max_batch], th)
            (g,) = torch.autograd.grad(torch.nan_to_num(loss, nan=0.0, posinf=0.0).sum(), th)
            ls.append(loss.detach())
            gs.append(g)
        v = torch.cat(ls).double().cpu().numpy()
        g = torch.cat(gs).double().cpu().numpy()
        bad = ~np.isfinite(v)
        g[bad] = 0.0
        return np.where(bad, np.inf, v), g

    def fit(
        self,
        e: Any,
        theta0: Any = None,
        *,
        method: str = "adam",
        max_iter: int = 40,
        lr: float = 1e-3,
        **_: Any,
    ) -> list:
        """Batched Adam (``max_iter`` steps, learning rate ``lr``) from
        ``theta0``; returns one :class:`~alkahest.fex.FitResult` per
        candidate with the best parameters seen. ``method`` is accepted for
        interface compatibility with :meth:`alkahest.fex.Evaluator.fit`;
        every method runs Adam here."""
        from .fex import FitResult

        torch = self.torch
        self._group_cache.clear()
        E, TH = self._tensors(e, theta0 if theta0 is not None else self.template.init_params(1))
        results = []
        for i in range(0, E.shape[0], self.max_batch):
            ee = E[i : i + self.max_batch]
            th = TH[i : i + self.max_batch].clone().requires_grad_(True)
            opt = torch.optim.Adam([th], lr=lr)
            best = torch.full((ee.shape[0],), math.inf, dtype=torch.float64, device=self.device)
            best_th = th.detach().clone()
            for _it in range(max_iter + 1):
                opt.zero_grad()
                loss = self._loss(ee, th)
                with torch.no_grad():
                    lv = torch.nan_to_num(loss.double(), nan=math.inf)
                    better = lv < best
                    best = torch.where(better, lv, best)
                    best_th[better] = th.detach()[better]
                if _it == max_iter:
                    break
                torch.nan_to_num(loss, nan=0.0, posinf=0.0).sum().backward()
                with torch.no_grad():
                    th.grad = torch.nan_to_num(th.grad, nan=0.0, posinf=0.0, neginf=0.0)
                opt.step()
            bt = best_th.double().cpu().numpy()
            bl = best.cpu().numpy()
            en = ee.cpu().numpy()
            for j in range(ee.shape[0]):
                results.append(
                    FitResult([int(v) for v in en[j]], bt[j], float(bl[j]), max_iter, max_iter + 1)
                )
        return results


def _block_name(b: Any) -> str:
    if isinstance(b, str):
        return b
    return "bd" if b else "int"


def _torch_fn(expr: Any, names: dict) -> Any:
    """Translate an Expr in the symbols of ``names`` (symbol → key of the
    environment dict, or a fixed tensor) into a function of that dict."""
    from ._fex_finalize import _children, _number_value

    memo: dict = {}

    def go(e: Any):
        if e in memo:
            return memo[e]
        nv = _number_value(e)
        if nv is not None:
            c = float(nv)
            f = lambda env, c=c: c  # noqa: E731
        else:
            nd = e.node()
            k = nd[0]
            if k == "symbol":
                if e not in names:
                    raise _err(f"symbol {e} has no torch value")
                v = names[e]
                f = (lambda env, v=v: env[v]) if isinstance(v, str) else (lambda env, v=v: v)
            elif k == "add":
                fs = [go(c) for c in _children(e)]

                def f(env, fs=fs):
                    out = fs[0](env)
                    for g in fs[1:]:
                        out = out + g(env)
                    return out
            elif k == "mul":
                fs = [go(c) for c in _children(e)]

                def f(env, fs=fs):
                    out = fs[0](env)
                    for g in fs[1:]:
                        out = out * g(env)
                    return out
            elif k == "pow":
                b, x = _children(e)
                fb, fx = go(b), go(x)
                xv = _number_value(x)
                if xv is not None and float(xv) == int(xv):
                    n = int(xv)
                    f = lambda env, fb=fb, n=n: fb(env) ** n  # noqa: E731
                else:
                    f = lambda env, fb=fb, fx=fx: fb(env) ** fx(env)  # noqa: E731
            elif k == "func" and nd[1] in _TORCH_FUNCS:
                (a,) = _children(e)
                fa = go(a)
                name = nd[1]
                f = lambda env, fa=fa, name=name: _TORCH_FUNCS[name](fa(env))  # noqa: E731
            else:
                raise _err(f"no torch kernel for {k} {nd[1] if len(nd) > 1 else ''}")
        memo[e] = f
        return f

    return go(expr)


def _tf(name: str):
    def f(x):
        import torch

        if not isinstance(x, torch.Tensor):
            return getattr(math, name)(x)
        return getattr(torch, name)(x)

    return f


_TORCH_FUNCS = {n: _tf(n) for n in ("exp", "log", "sin", "cos", "tanh", "sqrt", "abs")}


def _split_operator(problem: Any, pts: np.ndarray, T: Any, mask: np.ndarray) -> dict:
    """``F = F₀(u, Δu, x) + Σᵢ aᵢ(x) gᵢ + b(u, Δu, x)·Σ_{i∈S} gᵢ²``, checked."""
    import alkahest as ak

    spec = problem.spec
    jet = problem.jet
    op = spec["operator"]
    if jet.ut is not None and problem.time is not None:
        op = ak.subs(op, {jet.ut: jet.grad[problem.time]})
    grads = list(jet.grad)
    zero = jet.pool.integer(0)
    g0 = dict.fromkeys(grads, zero)
    F0 = ak.subs(op, g0)
    a = [ak.simplify(ak.subs(ak.diff(op, g).value, g0)).value for g in grads]
    lin = None
    for ai, g in zip(a, grads):
        if ak.is_zero(ai) is True:
            continue
        lin = ai * g if lin is None else lin + ai * g
    R = op - F0 - (lin if lin is not None else zero)
    from ._fex_finalize import _symbols_in

    spatial = [i for i in range(len(grads)) if mask[i] != 0.0]
    b = None
    q = zero
    if spatial:
        g = grads[spatial[0]]
        b = ak.subs(ak.diff(ak.diff(R, g).value, g).value / 2, g0)
        for i in spatial:
            q = q + grads[i] ** 2
    # The split is checked numerically at random jets and points (it only
    # decides how the GPU evaluates F; the result is compared with the CPU
    # evaluator, and nothing is certified from it).
    syms = [jet.u, jet.lap, *grads, *jet.x]
    full = ak.compile_expr(op, syms)
    parts = [ak.compile_expr(F0, syms)] + [ak.compile_expr(ai, syms) for ai in a]
    bq = None if b is None else ak.compile_expr(b, syms)
    rs = np.random.default_rng(12345)
    vals = rs.uniform(0.3, 1.2, size=(len(syms), 8))
    cols = [vals[k].copy() for k in range(len(syms))]
    want = np.asarray(ak.numpy_eval(full, *cols), dtype=np.float64)
    got = np.asarray(ak.numpy_eval(parts[0], *cols), dtype=np.float64)
    for k, f in enumerate(parts[1:]):
        got = got + np.asarray(ak.numpy_eval(f, *cols), dtype=np.float64) * vals[2 + k]
    if bq is not None:
        bv = np.asarray(ak.numpy_eval(bq, *cols), dtype=np.float64)
        if np.all(np.abs(bv) == 0.0):
            b = None
        else:
            got = got + bv * sum(vals[2 + i] ** 2 for i in spatial)
    if not np.allclose(got, want, rtol=1e-10, atol=1e-12):
        raise _err("the operator's gradient terms are not linear plus b·|∇u|²")
    if b is not None and _symbols_in(b) & set(grads):
        raise _err("the operator is not quadratic in the gradient in the supported way")
    coords = list(jet.x)
    for ai in a:
        if _symbols_in(ai) & ({jet.u, jet.lap} | set(grads)):
            raise _err("a gradient coefficient depends on u or Δu")
    names: dict = {jet.u: "u", jet.lap: "lap"}
    for k, xk in enumerate(coords):
        names[xk] = T(pts[:, k])[None, :]
    if _symbols_in(F0) & set(grads):
        raise _err("a gradient sits inside a non-polynomial function of the operator")
    uses_lap = jet.lap in _symbols_in(op)
    A = None
    if any(ak.is_zero(ai) is not True for ai in a):
        xenv = {xk: T(pts[:, k]) for k, xk in enumerate(coords)}
        cols = []
        for ai in a:
            v = 0.0 if ak.is_zero(ai) is True else _torch_fn(ai, xenv)({})
            if not hasattr(v, "shape") or v.dim() == 0:
                v = T(np.full(len(pts), float(v)))
            cols.append(v)
        import torch

        A = torch.stack(cols, dim=1)
    return {
        "F0": _torch_fn(F0, names),
        "a": A,
        "b": None if b is None else _torch_fn(b, names),
        "uses_lap": uses_lap,
    }


# Point tables φ, φ', φ'' of leaf operators, and the same for inner slots.


def _u_tables():
    def const(c):
        def f(x):
            import torch

            z = torch.zeros_like(x)
            return (z + c, z, z)

        return f

    def id_(x):
        import torch

        return (x, torch.ones_like(x), torch.zeros_like(x))

    def sq(x):
        import torch

        return (x * x, 2 * x, 2 * torch.ones_like(x))

    def cube(x):
        return (x**3, 3 * x * x, 6 * x)

    def quartic(x):
        return (x**4, 4 * x**3, 12 * x * x)

    def exp(x):
        import torch

        v = torch.exp(x)
        return (v, v, v)

    def sin(x):
        import torch

        s, c = torch.sin(x), torch.cos(x)
        return (s, c, -s)

    def cos(x):
        import torch

        s, c = torch.sin(x), torch.cos(x)
        return (c, -s, -c)

    def log(x):
        import torch

        bad = x <= 0
        safe = torch.where(bad, torch.ones_like(x), x)
        nan = torch.full_like(x, math.nan)
        r = 1.0 / safe
        return (
            torch.where(bad, nan, torch.log(safe)),
            torch.where(bad, nan, r),
            torch.where(bad, nan, -r * r),
        )

    def abs_(x):
        import torch

        return (x.abs(), torch.where(x >= 0, 1.0, -1.0).to(x.dtype), torch.zeros_like(x))

    def tanh(x):
        import torch

        t = torch.tanh(x)
        q = 1 - t * t
        return (t, q, -2 * t * q)

    return {
        "0": const(0.0),
        "1": const(1.0),
        "id": id_,
        "sq": sq,
        "cube": cube,
        "quartic": quartic,
        "exp": exp,
        "sin": sin,
        "cos": cos,
        "log": log,
        "abs": abs_,
        "tanh": tanh,
    }


_UNARY = _u_tables()
