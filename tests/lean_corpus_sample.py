#!/usr/bin/env python3
"""
Lean corpus harvester — V5-10.

Extends the fixed strict corpus (``tests/lean_corpus.py``, 14 curated cases)
with the FULL, DEDUPLICATED set of Lean certificates harvested live from the
textbook gate (``tests/textbook_gate/``) — the first-course calculus/algebra
identity suite. The point: every certificate the library actually emits from a
textbook identity must compile, not just the 14 fixed showcase cases.

This deliberately replaces the earlier fixed-seed RANDOM SAMPLE (n=25,
seed=1337). A random draw made CI green *by luck*: the candidate pool contained
certificates that did not typecheck, and whether the seed happened to draw one
was the difference between a passing and a failing run. Any emitter change
reshuffled the draw and could surface a latent non-compiling certificate.
Harvesting and checking the *entire* deduplicated pool removes that lottery —
if the library can emit a broken certificate from a textbook identity, CI sees
it every time.

How it works: this script instruments ``alkahest.diff``, ``alkahest.simplify``,
``alkahest.simplify_trig``, ``alkahest.simplify_log_exp``,
``alkahest.sum_indefinite``, ``alkahest.sum_definite`` and
``alkahest.integrate`` to record every :class:`DerivedResult` they produce,
then runs the textbook gate suite under pytest so those functions get
exercised exactly as the gate already exercises them (no duplicated case
lists to drift out of sync — see ``tests/_tg_helpers.py`` and
``tests/textbook_gate/*.py``, which this reuses as-is).

``alkahest.to_lean()`` deliberately WITHHOLDS a certificate (returns ``""``)
for results it cannot certify soundly yet. Those withheld/empty certificates
are correct-by-design and are skipped here, not treated as generation
failures. Only genuinely emitted (non-empty) certificates are sampled and
written out.

Indefinite ``integrate()`` results now emit certificates via the FTC
derivative relation ``deriv (fun x => F) x = f`` (see the ``to_lean`` docstring
in ``alkahest-py/src/lib.rs``), but they are intentionally excluded here (the
``diff`` instrumentation still captures the derivative-check steps the gate
runs) and covered instead by the deterministic strict corpus in
``tests/lean_corpus.py`` (the ``int_*`` cases).

Deduplication: many textbook-gate cases produce byte-identical certificates
(e.g. every ``mul_one`` cleanup step). The pool is deduplicated by certificate
text so each *distinct* certificate is written — and hence typechecked — exactly
once, keeping the Lean CI runtime bounded (~1.5 s of Mathlib load per file)
while still covering every distinct shape the library can emit.

Usage::

    python tests/lean_corpus_sample.py --output /tmp/lean_sample/
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
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
# NOTE: `integrate` is deliberately omitted. Its certificates now emit (Part A,
# via the FTC derivative relation) but are verified deterministically through
# the strict corpus (`tests/lean_corpus.py`), not this harvested pool — see
# the module docstring.
RECORDED_FUNCTIONS = (
    "diff",
    "simplify",
    "simplify_trig",
    "simplify_log_exp",
    "sum_indefinite",
    "sum_definite",
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
        description="Harvest the full deduplicated set of Lean certificates "
        "emitted by the textbook gate"
    )
    parser.add_argument("--output", default=".", help="Output directory for .lean files")
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

    # Deterministic and independent of harvesting order: sort by identity, then
    # deduplicate by certificate TEXT so each distinct certificate is written —
    # and typechecked in CI — exactly once. NO random sampling: a sample could
    # hide a non-compiling certificate (the "green by luck" seed lottery this
    # harvester exists to eliminate).
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    seen_hashes: set[str] = set()
    distinct: list[tuple[str, str, int, str]] = []
    for cand in candidates:
        digest = hashlib.sha256(cand[3].encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        distinct.append(cand)

    print(
        f"{len(distinct)} distinct certificates after text-deduplication "
        f"(of {len(candidates)} non-empty)."
    )

    success = 0
    admission_failures = 0
    for i, (name, test_id, idx, lean_src) in enumerate(distinct):
        bad_tokens = [token for token in FORBIDDEN_TOKENS if token in lean_src]
        if bad_tokens:
            print(
                f"ERROR: certificate {i} ({name} @ {test_id}#{idx}) contains "
                f"forbidden token(s) {bad_tokens!r}",
                file=sys.stderr,
            )
            admission_failures += 1
            continue
        fname = f"cert_{i:03d}_{sanitize(name)}_{sanitize(test_id)}_{idx}.lean"
        out_path = os.path.join(args.output, fname)
        with open(out_path, "w") as f:
            f.write(lean_src)
        print(f"Generated: {out_path}")
        success += 1

    print(f"\n{success}/{len(distinct)} distinct certificates written to {args.output}")
    return 0 if success == len(distinct) and admission_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
