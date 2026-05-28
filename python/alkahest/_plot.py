"""Plotting helpers for Alkahest symbolic expressions.

Backend dispatch
----------------
alkahest never lists a plotting library in its own dependencies.  Instead,
``plot()`` and friends detect what the user has installed and call into it.
The active backend can be forced with the ``backend`` keyword argument:

    ak.plot(expr, x, (-3, 3), backend="plotly")

Supported backends:
- ``"matplotlib"`` — default when importable; static PNG/SVG, Jupyter inline.
- ``"plotly"``     — interactive browser figures; also renders in the
                     demo-playground notebook (HTML output blob).

``plot_dag`` additionally accepts ``"graphviz"`` (renders via the Graphviz
Python package) and falls back to a text DOT string when it is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_KNOWN_BACKENDS = {"matplotlib", "plotly"}


def _default_backend() -> str:
    try:
        import matplotlib  # noqa: F401
        return "matplotlib"
    except ImportError:
        pass
    try:
        import plotly  # noqa: F401
        return "plotly"
    except ImportError:
        pass
    raise ImportError(
        "No plotting backend found.\n"
        "Install one with:  pip install matplotlib\n"
        "Or:               pip install plotly"
    )


def _require_backend(name: str | None) -> str:
    if name is None:
        return _default_backend()
    if name not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Choose from: {sorted(_KNOWN_BACKENDS)}"
        )
    return name


# ---------------------------------------------------------------------------
# Internal: evaluate expr numerically over a 1-D grid
# ---------------------------------------------------------------------------

def _eval_1d(expr, var, lo: float, hi: float, n: int = 300):
    """Return (xs, ys) as float64 arrays, skipping non-finite samples."""
    from .alkahest import compile_expr  # type: ignore[import]
    from ._dlpack import numpy_eval_dlpack

    xs = np.linspace(lo, hi, n)
    fn = compile_expr(expr, [var])
    ys = numpy_eval_dlpack(fn, xs)
    finite = np.isfinite(ys)
    return xs[finite], ys[finite]


def _eval_2d(expr, var_x, var_y, x_range, y_range, n: int = 80):
    """Return (X, Y, Z) meshgrid arrays."""
    from .alkahest import compile_expr  # type: ignore[import]
    from ._dlpack import numpy_eval_dlpack

    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xs, ys)
    fn = compile_expr(expr, [var_x, var_y])
    Z = numpy_eval_dlpack(fn, X, Y)
    return X, Y, Z


# ---------------------------------------------------------------------------
# plot — 1-D curve
# ---------------------------------------------------------------------------

def plot(
    expr,
    var,
    range_: tuple[float, float] = (-5.0, 5.0),
    *,
    n: int = 300,
    label: str | None = None,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Plot a single-variable symbolic expression.

    Parameters
    ----------
    expr    : Expr  — symbolic expression in *var*.
    var     : Expr  — the free variable.
    range_  : (lo, hi) — plotting interval (default: ``(-5, 5)``).
    n       : number of sample points (default: ``300``).
    label   : legend label.
    title   : axis title.
    backend : ``"matplotlib"`` (default) or ``"plotly"``.
    **kw    : forwarded to the backend's line-plot call.

    Returns
    -------
    Axes (matplotlib) or Figure (plotly).
    """
    b = _require_backend(backend)
    lo, hi = float(range_[0]), float(range_[1])
    xs, ys = _eval_1d(expr, var, lo, hi, n)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        ax = kw.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(xs, ys, label=label, **kw)
        if label:
            ax.legend()
        if title:
            ax.set_title(title)
        ax.set_xlabel(str(var))
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label or str(expr)))
        if title:
            fig.update_layout(title=title)
        fig.update_layout(xaxis_title=str(var))
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot3d — 2-D surface
# ---------------------------------------------------------------------------

def plot3d(
    expr,
    var_x,
    var_y,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    *,
    n: int = 80,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Plot a two-variable symbolic expression as a 3-D surface.

    Parameters
    ----------
    expr    : Expr  — symbolic expression in *var_x* and *var_y*.
    var_x, var_y : Expr — the two free variables.
    x_range, y_range : (lo, hi) — plotting intervals.
    n       : grid resolution per axis (default: ``80``).
    title   : figure title.
    backend : ``"matplotlib"`` (default) or ``"plotly"``.
    **kw    : forwarded to the backend surface call.

    Returns
    -------
    Axes3D (matplotlib) or Figure (plotly).
    """
    b = _require_backend(backend)
    X, Y, Z = _eval_2d(expr, var_x, var_y, x_range, y_range, n)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = kw.pop("fig", None) or plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, **kw)
        ax.set_xlabel(str(var_x))
        ax.set_ylabel(str(var_y))
        if title:
            ax.set_title(title)
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, **kw))
        if title:
            fig.update_layout(title=title)
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot_parametric — parametric curve (x(t), y(t))
# ---------------------------------------------------------------------------

def plot_parametric(
    expr_x,
    expr_y,
    param,
    range_: tuple[float, float] = (0.0, 6.283185307179586),
    *,
    n: int = 500,
    label: str | None = None,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Plot a parametric curve (x(t), y(t)).

    Parameters
    ----------
    expr_x, expr_y : Expr — x- and y-components in *param*.
    param  : Expr  — the parameter variable.
    range_ : (lo, hi) — parameter interval (default: ``(0, 2π)``).
    n      : number of sample points.
    """
    b = _require_backend(backend)
    lo, hi = float(range_[0]), float(range_[1])
    _, xs_vals = _eval_1d(expr_x, param, lo, hi, n)
    _, ys_vals = _eval_1d(expr_y, param, lo, hi, n)
    # Re-evaluate on a common grid (no filtering) for a connected curve.
    from .alkahest import compile_expr  # type: ignore[import]
    from ._dlpack import numpy_eval_dlpack
    ts = np.linspace(lo, hi, n)
    fn_x = compile_expr(expr_x, [param])
    fn_y = compile_expr(expr_y, [param])
    xs_vals = numpy_eval_dlpack(fn_x, ts)
    ys_vals = numpy_eval_dlpack(fn_y, ts)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        ax = kw.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(xs_vals, ys_vals, label=label, **kw)
        if label:
            ax.legend()
        if title:
            ax.set_title(title)
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        fig.add_trace(go.Scatter(x=xs_vals, y=ys_vals, mode="lines", name=label or ""))
        if title:
            fig.update_layout(title=title)
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot_implicit — implicit curve f(x,y) = 0
# ---------------------------------------------------------------------------

def plot_implicit(
    expr,
    var_x,
    var_y,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    *,
    n: int = 200,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Plot the zero-set of a two-variable expression via matplotlib contour.

    Evaluates ``expr`` on an ``n×n`` grid and draws the zero contour.
    Only supports the ``"matplotlib"`` backend (Plotly contour lines are
    available as a ``"plotly"`` fallback using ``Contour``).
    """
    b = _require_backend(backend)
    X, Y, Z = _eval_2d(expr, var_x, var_y, x_range, y_range, n)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        ax = kw.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        ax.contour(X, Y, Z, levels=[0], **kw)
        ax.set_xlabel(str(var_x))
        ax.set_ylabel(str(var_y))
        if title:
            ax.set_title(title)
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        fig.add_trace(
            go.Contour(x=X[0], y=Y[:, 0], z=Z, contours=dict(value=0, type="constraint"), **kw)
        )
        if title:
            fig.update_layout(title=title)
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot_roots — real root markers
# ---------------------------------------------------------------------------

def plot_roots(
    unipoly,
    var,
    *,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Mark real roots of a ``UniPoly`` on the x-axis (rug plot).

    Uses :func:`alkahest.real_roots` to isolate roots, then places a vertical
    line at the midpoint of each isolating interval.

    Parameters
    ----------
    unipoly : UniPoly — polynomial whose real roots are to be marked.
    var     : Expr    — the variable (used for axis labelling only).
    """
    from .alkahest import real_roots  # type: ignore[import]

    b = _require_backend(backend)
    result = real_roots(unipoly, var)
    intervals = result.value if hasattr(result, "value") else result

    mids = []
    for interval in intervals:
        lo_r = float(str(interval.lo)) if hasattr(interval, "lo") else float(interval[0])
        hi_r = float(str(interval.hi)) if hasattr(interval, "hi") else float(interval[1])
        mids.append((lo_r + hi_r) / 2.0)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        ax = kw.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        for m in mids:
            ax.axvline(m, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel(str(var))
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        for m in mids:
            fig.add_vline(x=m, line_color="red", line_dash="dash")
        if title:
            fig.update_layout(title=title)
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot_series — series truncation vs original
# ---------------------------------------------------------------------------

def plot_series(
    series_result,
    original_expr,
    var,
    range_: tuple[float, float] = (-3.0, 3.0),
    *,
    n: int = 300,
    title: str | None = None,
    backend: str | None = None,
    **kw: Any,
):
    """Overlay a Taylor/Laurent series truncation against the original function.

    Parameters
    ----------
    series_result : Series or DerivedResult — output of :func:`alkahest.series`.
    original_expr : Expr — the exact expression (plotted as a reference).
    var           : Expr — the variable.
    range_        : (lo, hi) — plotting interval.
    """
    b = _require_backend(backend)
    lo, hi = float(range_[0]), float(range_[1])

    series_expr = (
        series_result.expr if hasattr(series_result, "expr") else series_result.value
    )
    xs_o, ys_o = _eval_1d(original_expr, var, lo, hi, n)
    xs_s, ys_s = _eval_1d(series_expr, var, lo, hi, n)

    if b == "matplotlib":
        import matplotlib.pyplot as plt
        ax = kw.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(xs_o, ys_o, label="exact", **kw)
        ax.plot(xs_s, ys_s, label="series", linestyle="--")
        ax.legend()
        if title:
            ax.set_title(title)
        ax.set_xlabel(str(var))
        return ax

    if b == "plotly":
        import plotly.graph_objects as go
        fig = kw.pop("fig", None) or go.Figure()
        fig.add_trace(go.Scatter(x=xs_o, y=ys_o, mode="lines", name="exact"))
        fig.add_trace(go.Scatter(x=xs_s, y=ys_s, mode="lines", name="series", line=dict(dash="dash")))
        if title:
            fig.update_layout(title=title)
        return fig

    raise RuntimeError(f"Unreachable backend: {b!r}")


# ---------------------------------------------------------------------------
# plot_dag — expression tree / DAG
# ---------------------------------------------------------------------------

def plot_dag(expr, *, title: str | None = None, **kw: Any):
    """Visualise the expression DAG.

    Tries the ``graphviz`` Python package first (renders a nice PNG/SVG via
    system Graphviz).  Falls back to returning the raw DOT string so the
    caller can pipe it to ``dot`` manually.

    Parameters
    ----------
    expr : Expr — the expression to visualise.

    Returns
    -------
    graphviz.Source  — if the ``graphviz`` package is installed.
    str              — raw DOT string otherwise.
    """
    from .alkahest import plot_dot as _plot_dot  # type: ignore[import]
    dot_src = _plot_dot(expr)

    try:
        import graphviz
        src = graphviz.Source(dot_src)
        if title:
            src.comment = title
        return src
    except ImportError:
        pass

    return dot_src


# ---------------------------------------------------------------------------
# plot_svg — Rust SVG renderer (no Python plotting dep required)
# ---------------------------------------------------------------------------

def plot_svg(
    expr,
    var,
    range_: tuple[float, float] = (-5.0, 5.0),
    *,
    width: int = 640,
    height: int = 400,
    n: int = 300,
) -> str:
    """Return a standalone SVG string for the expression (no Python dep needed).

    This calls the Rust ``render_svg`` function directly and requires no
    matplotlib / plotly installation.  The result can be:
    - Written to a ``.svg`` file.
    - Embedded in HTML as ``<img src="data:image/svg+xml;base64,...">``.
    - Displayed in Jupyter: ``IPython.display.SVG(alkahest.plot_svg(...))``.
    """
    from .alkahest import plot_svg as _rust_plot_svg  # type: ignore[import]
    lo, hi = float(range_[0]), float(range_[1])
    return _rust_plot_svg(expr, var, lo, hi, width, height, n)


__all__ = [
    "plot",
    "plot3d",
    "plot_dag",
    "plot_implicit",
    "plot_parametric",
    "plot_roots",
    "plot_series",
    "plot_svg",
]
