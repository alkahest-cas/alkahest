#!/usr/bin/env python3
"""
Lean corpus sampler — V5-9.

Extends the fixed strict corpus (``tests/lean_corpus.py``, 14 curated cases)
with a deterministic, randomized SAMPLE of Lean certificates harvested live
from the textbook gate (``tests/textbook_gate/``) — the first-course
calculus/algebra identity suite. The point: every certificate the library
actually emits from a textbook identity must compile, not just the 14 fixed
showcase cases.

How it works: this script instruments ``alkahest.diff``, ``alkahest.simplify``,
``alkahest.simplify_trig``, ``alkahest.simplify_log_exp``,
``alkahest.sum_indefinite``, ``alkahest.sum_definite`` and
``alkahest.integrate`` to record every :class:`DerivedResult` they produce,
then runs the textbook gate suite under pytest so those functions get
exercised exactly as the gate already exercises them (no duplicated case
lists to drift out of sync — see ``tests/_tg_helpers.py`` and
``tests/textbook_gate/*.py``, which this reuses as-is).

``alkahest.to_lean()`` deliberately WITHHOLDS a certificate (returns ``""``)
for results it cannot certify soundly yet — most notably every
``integrate()`` result, since integration certificates are not currently
symbolically re-derivable (see the ``to_lean`` docstring in
``alkahest-py/src/lib.rs``). Those withheld/empty certificates are
correct-by-design and are skipped here, not treated as generation failures.
Only genuinely emitted (non-empty) certificates are sampled and written out.

Usage::

    python tests/lean_corpus_sample.py --output /tmp/lean_sample/ [--n 25] [--seed 1337]
"""

from __future__ import annotations

import argparse
import functools
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest
import pytest

FORBIDDEN_TOKENS = ("sorry", "admit", "axiom")

# Public alkahest functions that return a DerivedResult (a value plus a
# derivation log) and are therefore candidates for alkahest.to_lean().
# `integrate` is included for completeness even though to_lean() currently
# withholds all of its certificates (see module docstring) -- it is
# naturally filtered out below like any other empty cert, so including it
# costs nothing and keeps this list matching "every derivation-producing
# function the textbook gate exercises."
RECORDED_FUNCTIONS = (
    "diff",
    "simplify",
    "simplify_trig",
    "simplify_log_exp",
    "sum_indefinite",
    "sum_definite",
    "integrate",
)


class _Recorder:
    """Wraps ``alkahest.<name>`` to capture every DerivedResult it returns
    while the textbook gate suite runs, tagged with the pytest test id that
    produced it.

    Exceptions from the wrapped call propagate unmodified: known-broken
    cases in the gate are marked ``@pytest.mark.xfail(strict=True, ...)``
    and must keep behaving exactly as pytest expects.
    """

    def __init__(self) -> None:
        self.captured: list[tuple[str, str, int, object]] = []
        self._call_counts: dict[tuple[str, str], int] = {}
        self.current_test_id = "<unknown>"

    def wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            key = (self.current_test_id, name)
            idx = self._call_counts.get(key, 0)
            self._call_counts[key] = idx + 1
            self.captured.append((name, self.current_test_id, idx, result))
            return result

        return wrapper


class _TestIdPlugin:
    """pytest plugin that records the nodeid of the currently-executing test
    so the recorder above can tag captured results with it."""

    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    def pytest_runtest_setup(self, item):
        self.recorder.current_test_id = item.nodeid

    def pytest_runtest_teardown(self, item):
        self.recorder.current_test_id = "<unknown>"


def harvest(textbook_gate_dir: str) -> list[tuple[str, str, int, object]]:
    """Run the textbook gate suite with alkahest's derivation-producing
    functions instrumented, and return every DerivedResult produced."""
    recorder = _Recorder()
    originals = {name: getattr(alkahest, name) for name in RECORDED_FUNCTIONS}
    for name, fn in originals.items():
        setattr(alkahest, name, recorder.wrap(name, fn))
    try:
        ret = pytest.main(
            [textbook_gate_dir, "-q", "-p", "no:cacheprovider"],
            plugins=[_TestIdPlugin(recorder)],
        )
        # 0 = all passed, 1 = some assertions failed. The textbook gate marks
        # known-broken cases `xfail(strict=True)`, so ordinary assertion
        # failures are not expected in a healthy tree -- but even if a few
        # slip through, the derivation calls that happened before the
        # failing assertion are still captured and still worth sampling.
        # Only a pytest-internal error (bad usage, collection error, ...) is
        # fatal here.
        if int(ret) not in (0, 1):
            raise RuntimeError(f"textbook gate pytest run errored: exit code {ret}")
    finally:
        for name, fn in originals.items():
            setattr(alkahest, name, fn)
    return recorder.captured


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample Lean certificates harvested from the textbook gate"
    )
    parser.add_argument("--output", default=".", help="Output directory for .lean files")
    parser.add_argument("--n", type=int, default=25, help="Max number of certificates to sample")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic sample seed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    textbook_gate_dir = os.path.join(here, "textbook_gate")

    captured = harvest(textbook_gate_dir)
    print(f"Harvested {len(captured)} derivation calls from the textbook gate.")

    candidates: list[tuple[str, str, int, str]] = []
    empty = 0
    to_lean_errors = 0
    for name, test_id, idx, result in captured:
        try:
            lean_src = alkahest.to_lean(result)
        except Exception as e:
            print(f"ERROR: to_lean({name} @ {test_id}#{idx}) raised: {e}", file=sys.stderr)
            to_lean_errors += 1
            continue
        if not lean_src:
            empty += 1
            continue
        candidates.append((name, test_id, idx, lean_src))

    print(
        f"{len(candidates)} non-empty certificates available "
        f"({empty} withheld as empty, {to_lean_errors} raised errors)."
    )

    # Deterministic regardless of harvesting/collection order: sort first,
    # then draw a fixed-seed sample.
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    rng = random.Random(args.seed)
    sample = candidates if len(candidates) <= args.n else rng.sample(candidates, args.n)
    sample.sort(key=lambda c: (c[0], c[1], c[2]))

    success = 0
    admission_failures = 0
    for i, (name, test_id, idx, lean_src) in enumerate(sample):
        bad_tokens = [token for token in FORBIDDEN_TOKENS if token in lean_src]
        if bad_tokens:
            print(
                f"ERROR: sample {i} ({name} @ {test_id}#{idx}) contains "
                f"forbidden token(s) {bad_tokens!r}",
                file=sys.stderr,
            )
            admission_failures += 1
            continue
        fname = f"sample_{i:03d}_{sanitize(name)}_{sanitize(test_id)}_{idx}.lean"
        out_path = os.path.join(args.output, fname)
        with open(out_path, "w") as f:
            f.write(lean_src)
        print(f"Generated: {out_path}")
        success += 1

    print(f"\n{success}/{len(sample)} sampled certificates written to {args.output}")
    return 0 if success == len(sample) and admission_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
