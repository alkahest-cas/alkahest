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
with their least-squares, normalised and Rayleigh losses, matching
:class:`alkahest.fex.Evaluator`'s loss to rounding. Operators given as
expressions (``Problem.from_operator``) stay on the CPU evaluator.
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
    "install PyTorch (with CUDA for device='cuda'), and use the GPU evaluator for the built-in "
    "problems (poisson, transport, cubic_schrodinger, eigen); operators written as expressions "
    "and the centred loss run on alkahest.fex.Evaluator"
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
        if kind not in ("poisson", "transport", "cubic_schrodinger"):
            raise _err(f"problem kind {kind!r} is not supported on the torch evaluator")
        if problem.loss not in ("least_squares", "normalized", "rayleigh"):
            raise _err(f"loss {problem.loss!r} is not supported on the torch evaluator")
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

    def _table(self, k: int, boundary: bool) -> tuple[Any, Any, Any]:
        key = (k, boundary)
        if key not in self._tables:
            x = self.xb if boundary else self.x
            self._tables[key] = _UNARY[self.unary[k]](x)
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

    def _pair_table(self, k: int, k2: int, boundary: bool) -> Any:
        key = ("pair", min(k, k2), max(k, k2), boundary)
        if key not in self._tables:
            t1 = self._table(k, boundary)[1]
            t2 = self._table(k2, boundary)[1]
            self._tables[key] = t1 * t2 * self.mask
        return self._tables[key]

    def _forward(self, e: Any, th: Any, boundary: bool, order: int):
        """Root ``(u, lap, grads)`` for the batch.

        The gradient of a node is never formed as a ``(batch, points, d)``
        tensor: it is ``Σ_ℓ c_ℓ · (w_ℓ ⊙ φ'_ℓ(x))`` over the leaves ``ℓ`` below
        it, kept as the per-point coefficients ``c_ℓ``. Dot products of
        gradients then need only the leaf Gram terms
        ``G_ℓm = Σᵢ maskᵢ w_ℓᵢ w_mᵢ φ'_ℓ(xᵢ) φ'_m(xᵢ)``, each a matrix product
        of a point table with the candidates' weights."""
        torch = self.torch
        x: Any = self.xb if boundary else self.x
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
        order = 1 if (self.loss_kind == "rayleigh" or self.kind == "transport") else 2
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
            else:
                r = -lap + u**3
                if self.pot is not None:
                    r = r + self.pot * u
            r = r - self.rhs
            loss = (r * r).mean(-1)
            if self.loss_kind == "normalized":
                loss = loss / (u.abs() ** self.power).mean(-1)
        if self.xb is not None and self.bc_weight:
            ub = self._forward(e, th, True, 0)[0]
            loss = loss + self.bc_weight * ((ub - self.target) ** 2).mean(-1)
        return loss

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
