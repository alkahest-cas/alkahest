"""Make this directory importable so ``contracts`` / ``corpus`` resolve.

Mirrors what ``tests/conftest.py`` does for ``_tg_helpers`` one level up: the
gate's helper modules live beside its test files rather than in ``tests/`` root,
so the package stays self-contained and easy to lift into another repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Echo the gate's summary into the terminal report.

    pytest captures stdout by default, so a ``print`` inside a passing test is
    swallowed and a CI log reader would never see the measured rate. Writing it
    from the terminal-summary hook puts it in the log unconditionally, pass or
    fail, without anyone having to remember ``-s``.
    """
    module = sys.modules.get("test_silent_error_gate")
    text = getattr(module, "SUMMARY_TEXT", "") if module is not None else ""
    if text:
        terminalreporter.write_line(text)
