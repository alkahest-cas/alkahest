#!/usr/bin/env python3
"""Certificate corpus runner — the observation half of the certificate ledger.

The coverage table in ``python/alkahest/certificate_ledger.json`` is *generated*
by running real derivations and recording what the emitter actually did, never
by asserting what it ought to do. That distinction is the point: a hand-written
capability bit is how ``lean_theorem: true`` came to be advertised for ``log``,
``tan``, and ``gamma`` while the emitter could not prove any of them. A table
built only from observations cannot make that mistake, and the CI drift check
(``scripts/gen_certificate_ledger.py --check``) makes sure it stays honest.

The corpus is not a new list of cases. It reuses, unchanged:

* ``tests/lean_corpus.py`` — the strict, no-admission corpus whose every entry
  is expected to typecheck under pinned Lean/Mathlib in
  ``.github/workflows/lean.yml``. These populate the *certified* side.
* ``tests/textbook_gate/`` — the first-course calculus/algebra suite, run under
  pytest exactly as ``tests/lean_corpus_sample.py`` runs it. These reach far
  beyond the certifiable fragment and so populate the *withheld* side.

Both are driven with :func:`alkahest.diff` and friends instrumented, so each
call yields ``(operation, arguments, outcome)`` without either suite needing a
parallel case list that could drift out of sync.

Usage::

    python tests/certificate_corpus.py      # print a summary of observations
"""

from __future__ import annotations

import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest
from alkahest._certificates import OPERATIONS, canonical_expression, classify

#: Outcome of one observed call.
CERTIFIED = "certified"
WITHHELD = "withheld"
NO_DERIVATION = "no_derivation"


class Observation:
    """One recorded call: what was asked, and what the emitter did about it."""

    __slots__ = ("blocking_rules", "expression", "features", "operation", "outcome", "shape")

    def __init__(self, operation, shape, features, expression, outcome, blocking_rules):
        self.operation = operation
        self.shape = shape
        self.features = features
        self.expression = expression
        self.outcome = outcome
        self.blocking_rules = blocking_rules


class _Recorder:
    """Instruments the ledger-tracked ``alkahest`` entry points.

    Exceptions from the wrapped call propagate unmodified — the textbook gate
    marks known-broken cases ``xfail(strict=True)`` and must keep behaving
    exactly as pytest expects.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def _wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, alkahest.DerivedResult):
                self._record(name, args, kwargs, result)
            return result

        return wrapper

    def _record(self, name, args, kwargs, result) -> None:
        try:
            shape, features = classify(name, args, kwargs)
        except ValueError:
            # Argument shapes the ledger does not model (e.g. a matrix-valued
            # first argument). Nothing to say about them; say nothing.
            return
        status = result.certificate_status
        if status["certifiable"]:
            outcome = CERTIFIED
        elif status["reason"] == "withheld_no_derivation":
            outcome = NO_DERIVATION
        else:
            outcome = WITHHELD
        expression = args[0]
        if isinstance(expression, alkahest.DerivedResult):
            expression = expression.value
        self.observations.append(
            Observation(
                operation=name,
                shape=shape,
                features=features,
                # Canonical, not `str()`: see `canonical_expression`. The ledger
                # is compared byte-for-byte in CI, so nothing in it may depend
                # on a per-process hash seed.
                expression=canonical_expression(expression),
                outcome=outcome,
                blocking_rules=sorted({s["rule"] for s in status["blocking_steps"]}),
            )
        )

    def __enter__(self):
        self._originals = {n: getattr(alkahest, n) for n in OPERATIONS if hasattr(alkahest, n)}
        for name, fn in self._originals.items():
            setattr(alkahest, name, self._wrap(name, fn))
        return self

    def __exit__(self, *_exc):
        for name, fn in self._originals.items():
            setattr(alkahest, name, fn)
        return False


def _run_strict_corpus() -> None:
    """Drive every case in ``tests/lean_corpus.py``'s strict corpus."""
    import lean_corpus

    pool = alkahest.ExprPool()
    for name, _expected_rule, builder in lean_corpus.STRICT_CASES:
        try:
            builder(pool)
        except Exception as exc:  # pragma: no cover — a broken strict case
            print(f"WARNING: strict corpus case {name!r} raised: {exc}", file=sys.stderr)


def _run_textbook_gate(gate_dir: str) -> None:
    """Drive ``tests/textbook_gate/`` under pytest, quietly."""
    import pytest

    ret = pytest.main([gate_dir, "-q", "-p", "no:cacheprovider", "--no-header"])
    # 0 = all passed, 1 = some assertions failed. Calls made before a failing
    # assertion are still recorded and still worth tabulating; only a
    # pytest-internal error (bad usage, collection error) is fatal.
    if int(ret) not in (0, 1):
        raise RuntimeError(f"textbook gate pytest run errored: exit code {ret}")


def collect() -> list[Observation]:
    """Run the whole corpus and return every observation, deterministically."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    with _Recorder() as recorder:
        _run_strict_corpus()
        _run_textbook_gate(os.path.join(here, "textbook_gate"))
    recorder.observations.sort(key=lambda o: (o.operation, o.shape, o.expression))
    return recorder.observations


def main() -> int:
    observations = collect()
    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation.outcome] = counts.get(observation.outcome, 0) + 1
    shapes = {o.shape for o in observations}
    print(f"\n{len(observations)} observations over {len(shapes)} shape classes")
    for outcome in (CERTIFIED, WITHHELD, NO_DERIVATION):
        print(f"  {outcome:>14}: {counts.get(outcome, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
