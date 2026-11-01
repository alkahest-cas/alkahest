"""Locate the Math playground's translator and executor (``alkahest.playground``).

They ship inside the Alkahest package so that ``python -m alkahest.playground``
can serve the Math tab from a visitor's own machine.  This server reuses that
code rather than keeping a copy.  When the installed Alkahest wheel predates
the subpackage (e.g. CI installs the latest release from PyPI), the copy in
this repository's ``python/alkahest/playground`` is used, on top of the
installed native extension.
"""

from __future__ import annotations

from pathlib import Path

import alkahest

_REPO_PKG = Path(__file__).resolve().parents[2] / "python" / "alkahest"

try:
    from alkahest import playground
except ImportError:
    if not (_REPO_PKG / "playground" / "__init__.py").is_file():
        raise
    alkahest.__path__.append(str(_REPO_PKG))
    from alkahest import playground

ml = playground.mathlang
math_service = playground.service

__all__ = ["math_service", "ml", "playground"]
