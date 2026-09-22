"""alkahest.lattice — LLL lattice reduction (V2-6).

LLL reduces the rows of a matrix (basis given as integer rows, all having the
same length). Optional ``delta = delta_num/delta_den`` must lie strictly between
``1/4`` and ``1`` when both parts are supplied; otherwise the canonical
``δ = 3/4`` is used.

The reduction runs on FLINT's ``fmpz_lll`` and the result is then **verified**
against the exact rational LLL inequalities — ``|μ_ij| ≤ 1/2`` and the Lovász
condition at the requested ``δ`` — before it is returned; a floating-point LLL
promises only ``|μ_ij| ≤ η`` for some ``η > 1/2``. An LLL-reduced basis is not
unique, so assert those properties (and that the lattice is unchanged) rather
than an exact matrix.

For the rest of the lattice toolkit — determinants, duals, exact shortest and
closest vectors, theta series, kissing numbers and sphere-packing densities —
see :class:`alkahest.experimental.Lattice`.

Raises
------
LatticeError
    Empty basis (``E-LAT-001``), ragged rows (``E-LAT-002``),
    invalid ``δ`` (``E-LAT-003``), or iteration guard (``E-LAT-004``).
OverflowError (subclass PyOverflowError after reduction)
    A reduced coefficient no longer fits in a signed ``i64``.
"""

from __future__ import annotations

from .alkahest import lat_lll_reduce_rows

__all__ = ["lll_reduce_rows"]


def lll_reduce_rows(
    rows: list[list[int]],
    *,
    delta_num: int | None = None,
    delta_den: int | None = None,
) -> list[list[int]]:
    """LLL‑reduce *rows* with optional rational Lovász parameter ``delta_num/delta_den``."""
    return lat_lll_reduce_rows(rows, delta_num, delta_den)
