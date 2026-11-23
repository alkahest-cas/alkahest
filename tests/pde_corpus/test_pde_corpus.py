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

from corpus_abbasi import ABBASI  # noqa: E402
from corpus_sympy import SYMPY  # noqa: E402
from corpus_textbook import TEXTBOOK  # noqa: E402
from pde_expected import EXPECTED_SOLVED  # noqa: E402
from pde_harness import run_entry  # noqa: E402

ALL = [("sympy", e) for e in SYMPY] + [("textbook", e) for e in TEXTBOOK]
ABBASI_ENTRIES = [("abbasi", e) for e in ABBASI]


def _check(entry):
    r = run_entry(entry)
    assert r["status"] != "unverified", f"{entry['id']}: unverified answer {r}"
    assert r["status"] != "build_error", r
    if entry.get("expect") == "refused":
        assert r["status"] == "refused", f"{entry['id']} must be refused: {r}"
    if entry["id"] in EXPECTED_SOLVED:
        assert r["status"] == "verified", f"{entry['id']} regressed: {r}"
    return r


@pytest.mark.parametrize("entry", [e for _, e in ALL], ids=[e["id"] for _, e in ALL])
def test_corpus_entry(entry):
    _check(entry)


@pytest.mark.parametrize(
    "entry", [e for _, e in ABBASI_ENTRIES], ids=[e["id"] for _, e in ABBASI_ENTRIES]
)
def test_abbasi_entry(entry):
    _check(entry)


def test_every_sympy_problem_solves():
    """The milestone target: all of SymPy's pdsolve tests solve and verify."""
    missing = [e["id"] for e in SYMPY if e["id"] not in EXPECTED_SOLVED]
    assert not missing, missing
