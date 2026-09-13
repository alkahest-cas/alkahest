"""The declarative trap corpus.

Every case here is a *classic* silent-error shape — a place where a CAS has a
clean, plausible, wrong answer available and has to choose not to give it.  The
expected value of every case was derived by hand from the definition (and, for
the classical constants, cross-checked against ``math``), never read off
alkahest's own output; :attr:`Case.verified_by` records which.

Adding a case: see ``tests/silent_errors/README.md``.

Cost discipline: this corpus runs on every pull request, so each op must finish
in well under a second.  Known-slow inputs are documented in the README rather
than being added here — a gate that times out is a gate that gets disabled.

One module per :attr:`Case.subsystem` — ``corpus/linear_algebra.py`` and so on
— with ``corpus/_shared.py`` holding the pool and the few helpers more than one
subsystem needs.  The modules are discovered rather than listed, so adding a
subsystem is a single new file, and the usual edit (one agent, one subsystem)
touches one file that nobody else is touching.

:func:`_validate` re-checks the structure of every case at import: a bad merge
that truncates a helper still parses, still formats and still lints, and used to
surface only as every case in the subsystem raising the same uninformative
``'NoneType' object is not callable`` from inside the runner.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contracts import Case

#: Subsystem modules, discovered rather than listed.  Adding a subsystem is a
#: one-file change, and two agents adding two subsystems do not collide on a
#: shared registry the way they collided on the single ``corpus.py``.
SUBSYSTEMS: tuple[str, ...] = tuple(
    sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))
)


def validate(cases: Sequence[Case], *, module: str | None = None) -> None:
    """Structural checks, run at import before any case is executed.

    A corpus is only as good as its ability to fail loudly.  A bad merge that
    truncates a helper leaves ``op`` bound to ``None``: the file still parses,
    still formats and still lints, and then every case in the subsystem reports
    the same useless ``'NoneType' object is not callable`` from inside the
    runner, with nothing pointing at the merge.  These checks turn that into one
    import-time error naming the case.

    ``module`` is the subsystem module the cases were found in; each case must
    declare it, so a case cannot drift into the wrong file and quietly change
    which per-subsystem rate it is counted against.
    """
    seen: set[str] = set()
    for case in cases:
        where = f"corpus/{case.subsystem}.py::{case.id}"
        if module is not None and case.subsystem != module:
            raise RuntimeError(
                f"{case.id} lives in corpus/{module}.py but declares "
                f"subsystem={case.subsystem!r}; move the case or fix the field"
            )
        if case.id in seen:
            raise RuntimeError(f"duplicate case id in the silent-error corpus: {case.id!r}")
        seen.add(case.id)
        if not callable(case.op):
            raise TypeError(f"{where}: op is {case.op!r}, not callable")
        try:
            inspect.signature(case.op).bind()
        except TypeError as exc:
            # Nearly always the factory passed instead of called: the shape
            # helpers all *return* the op, so ``op=definite`` is callable and
            # wrong, and only shows up at run time as an uninformative
            # ``no_answer``.
            raise TypeError(
                f"{where}: op needs arguments ({exc}); a case op is called with none. "
                "Did you write op=definite where you meant op=definite(f, a, b)?"
            ) from exc
        except ValueError:  # pragma: no cover - builtin with no introspectable signature
            pass
        if not case.statement.strip():
            raise RuntimeError(f"{where}: empty statement")
        if not case.verified_by.strip():
            raise RuntimeError(f"{where}: empty verified_by - cite a non-alkahest source")
        if case.contract is None:
            raise RuntimeError(f"{where}: no contract")


CASES: list[Case] = []
for _name in SUBSYSTEMS:
    _mod = importlib.import_module(f"{__name__}.{_name}")
    _cases = getattr(_mod, "CASES", None)
    if _cases is None:  # pragma: no cover - corpus authoring guard
        raise RuntimeError(f"corpus/{_name}.py defines no CASES list")
    validate(_cases, module=_name)
    CASES.extend(_cases)

# Ids have to be unique across the whole corpus, not just within a module:
# they are the pytest parameter ids and the keys the benchmark catalogue
# ratchet matches on.
validate(CASES)

#: Fast lookup by id.
CASES_BY_ID: dict[str, Case] = {c.id: c for c in CASES}

del _cases, _mod, _name

__all__ = ["CASES", "CASES_BY_ID", "SUBSYSTEMS", "validate"]
