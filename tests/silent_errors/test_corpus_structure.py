"""The corpus package's own import-time guard.

:func:`corpus.validate` is the check that fires before a single case is
executed. It exists because the ways this corpus has actually broken were all
invisible to the parser, the formatter and the linter: a merge that cut a helper
mid-body left ``op`` bound to ``None``, and the only symptom was every case in
the subsystem raising the same ``'NoneType' object is not callable`` from inside
the runner. A guard nobody tests is a guard that quietly stops guarding, so each
failure mode it claims to catch gets a case here.
"""

from __future__ import annotations

import dataclasses

import pytest
from contracts import Case, Returns
from corpus import CASES, CASES_BY_ID, SUBSYSTEMS, validate

_GOOD = Case(
    id="structure_probe",
    subsystem="evaluation",
    statement="1 + 1 = 2",
    op=lambda: 2,
    contract=Returns(2),
    verified_by="arithmetic",
)


def _broken(**overrides) -> Case:
    return dataclasses.replace(_GOOD, **overrides)


def test_a_well_formed_case_passes() -> None:
    """The control: without it, a guard that rejects everything would pass."""
    validate([_GOOD])
    validate([_GOOD], module="evaluation")


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        # The mangled-merge signature: a helper truncated past its ``return``
        # hands back ``None``, so ``op=helper(...)`` binds ``None``.
        ("op", None, TypeError, "not callable"),
        ("statement", "   ", RuntimeError, "empty statement"),
        ("verified_by", "", RuntimeError, "empty verified_by"),
        ("contract", None, RuntimeError, "no contract"),
    ],
)
def test_a_malformed_case_is_rejected_by_name(field, value, error, message) -> None:
    with pytest.raises(error, match=message) as excinfo:
        validate([_broken(**{field: value})])
    assert "structure_probe" in str(excinfo.value)


def test_an_op_that_needs_arguments_is_rejected() -> None:
    """``Case.op`` is called with no arguments, and the helpers are factories.

    ``op=definite`` instead of ``op=definite(f, a, b)`` is callable, passes a
    bare ``callable()`` check, and then surfaces only as a ``no_answer`` — the
    outcome that means "the corpus is broken", with no hint of where.
    """
    with pytest.raises(TypeError, match="needs arguments"):
        validate([_broken(op=lambda _x: 2)])


def test_a_case_in_the_wrong_module_is_rejected() -> None:
    """``subsystem`` decides which rate a case is counted against.

    A case that drifts into another module keeps reporting under its declared
    subsystem, so the per-subsystem breakdown silently stops matching the files.
    """
    with pytest.raises(RuntimeError, match="move the case or fix the field"):
        validate([_GOOD], module="linear_algebra")


def test_duplicate_ids_are_rejected() -> None:
    """Ids are the pytest parameter ids and the benchmark ratchet's keys."""
    with pytest.raises(RuntimeError, match="duplicate case id"):
        validate([_GOOD, _broken(statement="a different case, same id")])


def test_every_subsystem_has_a_module_and_every_module_has_cases() -> None:
    declared = {case.subsystem for case in CASES}
    assert declared == set(SUBSYSTEMS)


def test_the_real_corpus_is_indexed_completely() -> None:
    assert len(CASES_BY_ID) == len(CASES)
