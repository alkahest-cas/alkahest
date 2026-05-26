"""LaTeX and Unicode pretty-printing for Alkahest expressions.

Rendering is performed in Rust (``alkahest_core::render_latex`` /
``alkahest_core::render_unicode``) via the ``Expr.display_latex()`` and
``Expr.display_unicode()`` methods, which dispatch on the typed ``ExprData``
enum rather than string-tagged Python lists.

The public ``latex()`` and ``unicode_str()`` functions below are thin Python
wrappers that call through to those Rust methods.
"""

from __future__ import annotations


def latex(expr) -> str:
    """Return a LaTeX string for the given :class:`~alkahest.Expr`.

    The result is suitable for ``$...$`` in a LaTeX document or
    ``IPython.display.Math(alkahest.latex(expr))`` in a Jupyter notebook.

    Example::

        >>> import alkahest
        >>> p = alkahest.ExprPool()
        >>> x = p.symbol("x")
        >>> alkahest.latex(alkahest.sin(x)**2 + alkahest.cos(x)**2)
        '\\\\sin\\\\!\\\\left(x\\\\right)^{2} + \\\\cos\\\\!\\\\left(x\\\\right)^{2}'
    """
    return expr.display_latex()


def unicode_str(expr) -> str:
    """Return a Unicode pretty-printed string for the given :class:`~alkahest.Expr`.

    Uses Unicode superscripts, Greek letters, fraction characters, and root
    symbols for a human-readable representation without LaTeX markup.

    Example::

        >>> import alkahest
        >>> p = alkahest.ExprPool()
        >>> x = p.symbol("x")
        >>> alkahest.unicode_str(x**2 + p.integer(1))
        'x² + 1'
    """
    return expr.display_unicode()
