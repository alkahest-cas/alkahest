"""alkahest.lattice — LLL lattice reduction (V2-6).

Exact integer arithmetic LLL reduces the rows of a matrix (basis given as integer
rows, all having the same length). Optional ``delta = delta_num/delta_den`` must
lie strictly between ``1/4`` and ``1`` when both parts are supplied; otherwise the
canonical ``δ = 3/4`` is used.

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
