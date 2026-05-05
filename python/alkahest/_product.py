"""Symbolic discrete products (∏) — V2-22."""

from __future__ import annotations


class Product:
    """Unevaluated discrete product — SymPy-shaped `Product(term, (k, lo, hi))`.

    Call :meth:`doit` for a closed form when `term(k)` reduces to ℚ(k)
    whose numerator and denominator factor into ℤ-linear polynomials in `k`.

    Bounds are **inclusive** on both ends, matching :func:`~alkahest.sum_definite`.

    Examples
    --------
    ::
        import alkahest as ah
        p = ah.ExprPool()
        k, n = p.symbol("k"), p.symbol("n")
        prod = ah.Product(k, (k, p.integer(1), n))
        out = ah.simplify(prod.doit().value)
    """

    __slots__ = ("term", "_k", "_lo", "_hi")

    def __init__(self, term, bounds: tuple) -> None:
        if not isinstance(bounds, tuple) or len(bounds) != 3:
            raise ValueError("Product(term, bounds) expects bounds=(k, lo, hi)")
        self.term = term
        self._k = bounds[0]
        self._lo = bounds[1]
        self._hi = bounds[2]

    def doit(self):
        """Closed form as :class:`~alkahest.DerivedResult`; use `.value` for the `Expr`."""
        import alkahest as ah

        return ah.product_definite(self.term, self._k, self._lo, self._hi)

    def __repr__(self) -> str:
        return f"Product({self.term!r}, ({self._k!r}, {self._lo!r}, {self._hi!r}))"
