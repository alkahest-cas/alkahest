"""The Math playground's restricted maths language and its local server.

This subpackage backs the **Math** tab of the Alkahest playground
(https://alkahest-cas.github.io/playground/).  It is a tool, not part of the
library API, and is deliberately not re-exported from :mod:`alkahest`.

* :mod:`~alkahest.playground.mathlang` — parses one line of plain-language maths
  ("integrate sin(x)^2 dx") into an inspected syntax tree (no ``eval``);
* :mod:`~alkahest.playground.math_exec` — maps that tree onto a whitelist of
  Alkahest calls under a :class:`~alkahest.Budget`;
* :mod:`~alkahest.playground.service` — runs each calculation in a fresh,
  resource-limited child process;
* :mod:`~alkahest.playground.server` — a localhost-only HTTP server
  (standard library only), started with ``python -m alkahest.playground``.
"""

from .service import interpret_payload, run_math

__all__ = ["interpret_payload", "run_math"]
