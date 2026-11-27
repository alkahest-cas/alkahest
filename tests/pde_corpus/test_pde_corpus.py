"""Corpus regression for ``pdsolve`` (milestone M1, first order).

Invariant: **no unverified answer**.  Every answer ``pdsolve`` returns on the
corpus must be ``exactly_verified``, and explicit answers are re-verified
independently through :func:`verify_pde_solution`.  A refusal is always
allowed by the invariant; the solved sets recorded in ``pde_expected.py`` guard
against regressions in coverage.
"""

from __future__ import annotations

import pytest

pytest.importorskip("alkahest.experimental")

from corpus_abbasi import ABBASI
from corpus_sympy import SYMPY
from corpus_textbook import TEXTBOOK
from pde_expected import EXPECTED_CLOSED, EXPECTED_SOLVED
from pde_harness import run_entry

ALL = [("sympy", e) for e in SYMPY] + [("textbook", e) for e in TEXTBOOK]
# The first 155-problem sample runs by default; the other 245 of the
# 400-problem sample are `slow` (Tier 1b).
ABBASI_ENTRIES = [
    pytest.param(e, id=e["id"], marks=() if e["in155"] else pytest.mark.slow) for e in ABBASI
]
# Per-problem cooperative budget; the expected-solved set only lists problems
# that solve well inside it.
WALL_MS = 30_000


def _check(entry):
    r = run_entry(entry, wall_ms=WALL_MS)
    assert r["status"] != "unverified", f"{entry['id']}: unverified answer {r}"
    assert r["status"] != "build_error", r
    if entry.get("expect") == "refused":
        assert r["status"] == "refused", f"{entry['id']} must be refused: {r}"
    if entry["id"] in EXPECTED_SOLVED:
        assert r["status"] == "verified", f"{entry['id']} regressed: {r}"
    if entry["id"] in EXPECTED_CLOSED:
        assert r["status"] == "verified", f"{entry['id']} regressed: {r}"
        assert not r["integral"], f"{entry['id']}: closed form regressed to an Integral: {r}"
    return r


@pytest.mark.parametrize("entry", [e for _, e in ALL], ids=[e["id"] for _, e in ALL])
def test_corpus_entry(entry):
    _check(entry)


@pytest.mark.parametrize("entry", ABBASI_ENTRIES)
def test_abbasi_entry(entry):
    _check(entry)


def test_every_sympy_problem_solves():
    """The milestone target: all of SymPy's pdsolve tests solve and verify."""
    missing = [e["id"] for e in SYMPY if e["id"] not in EXPECTED_SOLVED]
    assert not missing, missing
