"""GPU-accelerated plotting via fastplotlib (experimental).

fastplotlib (https://github.com/fastplotlib/fastplotlib) is built on WGPU /
pygfx and handles millions of points at interactive frame rates.  It is the
recommended backend for users evaluating expressions on dense grids with the
``+full`` JIT wheel (JIT + parallel evaluation).

Install
-------
    pip install fastplotlib

Usage
-----
    import alkahest as ak
    from alkahest.experimental._fastplotlib import fplot, fplot3d

    pool = ak.ExprPool()
    x = pool.symbol("x")
    fplot(ak.sin(x), x, (-10, 10))

Stability
---------
This module is **experimental**: the fastplotlib API is still evolving, and
this adapter may change in minor Alkahest releases.
"""

from __future__ import annotations

from typing import Any


def _require_fpl():
    try:
        import fastplotlib as fpl

        return fpl
    except ImportError:
        raise ImportError(
            "fastplotlib is not installed.\n"
            "Install it with:  pip install fastplotlib\n"
            "Note: fastplotlib requires a GPU context (WGPU / WebGPU)."
        ) from None


def _eval_1d(expr, var, lo: float, hi: float, n: int):
    import numpy as np

    from alkahest._dlpack import numpy_eval_dlpack
    from alkahest.alkahest import compile_expr  # type: ignore[import]

    xs = np.linspace(lo, hi, n, dtype=np.float32)
    fn = compile_expr(expr, [var])
    ys = numpy_eval_dlpack(fn, xs.astype(np.float64)).astype(np.float32)
    return xs, ys


def _eval_2d(expr, var_x, var_y, x_range, y_range, n: int):
    import numpy as np

    from alkahest._dlpack import numpy_eval_dlpack
    from alkahest.alkahest import compile_expr  # type: ignore[import]

    xs = np.linspace(x_range[0], x_range[1], n, dtype=np.float64)
    ys = np.linspace(y_range[0], y_range[1], n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    fn = compile_expr(expr, [var_x, var_y])
    Z = numpy_eval_dlpack(fn, X, Y).astype(np.float32)
    return X.astype(np.float32), Y.astype(np.float32), Z


def fplot(
    expr,
    var,
    range_: tuple[float, float] = (-5.0, 5.0),
    *,
    n: int = 100_000,
    title: str | None = None,
    **kw: Any,
):
    """GPU-accelerated 1-D curve plot via fastplotlib.

    Evaluates ``expr`` at ``n`` points (default 100 000) and renders a line
    graphic in a fastplotlib Figure.

    Parameters
    ----------
    expr   : Expr  — symbolic expression in *var*.
    var    : Expr  — the free variable.
    range_ : (lo, hi) — plotting interval.
    n      : number of sample points (default: ``100_000``).
    **kw   : forwarded to ``fpl.Figure()[0, 0].add_line()``.

    Returns
    -------
    fastplotlib.Figure
    """
    import numpy as np

    fpl = _require_fpl()
    lo, hi = float(range_[0]), float(range_[1])
    xs, ys = _eval_1d(expr, var, lo, hi, n)
    data = np.column_stack([xs, ys])

    fig = fpl.Figure(size=(800, 400))
    fig[0, 0].add_line(data=data, **kw)
    if title:
        fig[0, 0].set_title(title)
    fig.show()
    return fig


def fplot3d(
    expr,
    var_x,
    var_y,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    *,
    n: int = 512,
    title: str | None = None,
    **kw: Any,
):
    """GPU-accelerated 2-D surface plot via fastplotlib ImageGraphic.

    Renders ``expr(var_x, var_y)`` as a heatmap/image on an ``n×n`` grid.
    Use ``n=512`` or higher for dense evaluation — the JIT + parallel wheel
    handles this efficiently with :func:`alkahest.numpy_eval_par`.

    Returns
    -------
    fastplotlib.Figure
    """
    fpl = _require_fpl()
    _, _, Z = _eval_2d(expr, var_x, var_y, x_range, y_range, n)

    fig = fpl.Figure(size=(600, 600))
    fig[0, 0].add_image(data=Z, **kw)
    if title:
        fig[0, 0].set_title(title)
    fig.show()
    return fig


__all__ = ["fplot", "fplot3d"]
