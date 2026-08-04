"""P1 search plumbing item 6 — versioned, machine-parseable ``DerivedResult``.

Agents pay for every character. These tests pin the ``to_dict`` / ``to_json``
envelope schema (``RESULT_SCHEMA_VERSION`` / ``STEPS_SCHEMA_VERSION``) and,
critically, that ``mode="compact"`` never drops or obscures the honesty
signal (``verification["status"]``) even while it is strictly smaller than
``mode="full"``.
"""

from __future__ import annotations

import json

import alkahest as ak
import pytest
from alkahest._result_schema import (
    RESULT_SCHEMA_VERSION,
    STEP_FIELDS,
    STEP_FIELDS_COMPACT,
    STEPS_SCHEMA_VERSION,
)


@pytest.fixture
def pool():
    return ak.ExprPool()


def _multistep_derivation(pool):
    """A derivation with several rewrite steps and at least one side condition."""
    x = pool.symbol("x", domain=ak.Domain.Positive)
    return ak.diff(ak.sqrt(x**2) * ak.sin(x), x)


# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------


def test_schema_version_constants_are_one():
    # Pin the initial version; bump deliberately (with a docs update) if the
    # envelope or step shape ever changes.
    assert RESULT_SCHEMA_VERSION == 1
    assert STEPS_SCHEMA_VERSION == 1


def test_schema_version_constants_exported_from_top_level():
    assert ak.RESULT_SCHEMA_VERSION == RESULT_SCHEMA_VERSION
    assert ak.STEPS_SCHEMA_VERSION == STEPS_SCHEMA_VERSION
    assert "RESULT_SCHEMA_VERSION" in ak.__all__
    assert "STEPS_SCHEMA_VERSION" in ak.__all__


def test_schema_version_class_attrs_match_module_constants():
    assert ak.DerivedResult.SCHEMA_VERSION == RESULT_SCHEMA_VERSION
    assert ak.DerivedResult.STEPS_SCHEMA_VERSION == STEPS_SCHEMA_VERSION


def test_documented_step_fields_match_actual_dict_keys(pool):
    dr = _multistep_derivation(pool)
    assert dr.steps, "fixture derivation must have at least one step"
    assert set(dr.steps[0].keys()) == set(STEP_FIELDS)

    compact = dr.to_dict(mode="compact")
    compact_keys = set()
    for step in compact["steps"]:
        compact_keys.update(step.keys())
    # every key seen in a compact step is one of the two documented short keys
    assert compact_keys <= set(STEP_FIELDS_COMPACT)


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------


def test_full_mode_has_required_keys_and_versions(pool):
    dr = _multistep_derivation(pool)
    full = dr.to_dict()  # mode="full" is the default
    assert dr.to_dict(mode="full") == full

    required = {
        "kind",
        "schema_version",
        "steps_schema_version",
        "value",
        "verification",
        "certificate_status",
        "steps",
        "has_certificate",
    }
    assert required <= set(full.keys())
    assert full["schema_version"] == RESULT_SCHEMA_VERSION
    assert full["steps_schema_version"] == STEPS_SCHEMA_VERSION
    assert full["value"] == str(dr.value)
    assert full["has_certificate"] == (dr.certificate is not None)


def test_full_mode_steps_match_steps_getter(pool):
    dr = _multistep_derivation(pool)
    full = dr.to_dict()
    assert full["steps"] == dr.steps


def test_full_mode_verification_matches_getter(pool):
    dr = _multistep_derivation(pool)
    full = dr.to_dict()
    assert full["verification"] == dr.verification


def test_full_mode_certificate_status_matches_getter(pool):
    dr = _multistep_derivation(pool)
    full = dr.to_dict()
    assert full["certificate_status"] == dr.certificate_status


# ---------------------------------------------------------------------------
# Compact mode: smaller, but never dishonest
# ---------------------------------------------------------------------------


def test_compact_is_strictly_smaller_than_full_for_multistep_derivation(pool):
    dr = _multistep_derivation(pool)
    full_json = dr.to_json(mode="full")
    compact_json = dr.to_json(mode="compact")
    assert len(compact_json) < len(full_json)


def test_verification_status_present_and_equal_in_both_modes(pool):
    dr = _multistep_derivation(pool)
    full = dr.to_dict(mode="full")
    compact = dr.to_dict(mode="compact")

    assert "status" in full["verification"]
    assert "status" in compact["verification"]
    # the honesty signal itself is never renamed, abbreviated, or changed
    assert full["verification"]["status"] == compact["verification"]["status"]
    assert compact["verification"]["status"] == dr.verification["status"]


def test_compact_verification_keeps_externally_verified(pool):
    dr = _multistep_derivation(pool)
    compact = dr.to_dict(mode="compact")
    assert compact["verification"]["externally_verified"] == dr.verification["externally_verified"]


def test_compact_steps_use_short_keys_and_drop_before_after(pool):
    dr = _multistep_derivation(pool)
    compact = dr.to_dict(mode="compact")
    assert len(compact["steps"]) == len(dr.steps)
    for step in compact["steps"]:
        assert "before" not in step
        assert "after" not in step
        assert "rule" not in step
        assert "r" in step


def test_compact_steps_omit_empty_side_conditions_but_keep_nonempty(pool):
    dr = _multistep_derivation(pool)
    compact = dr.to_dict(mode="compact")
    saw_side_condition = False
    for full_step, compact_step in zip(dr.steps, compact["steps"]):
        if full_step["side_conditions"]:
            assert compact_step["s"] == full_step["side_conditions"]
            saw_side_condition = True
        else:
            assert "s" not in compact_step
    assert saw_side_condition, "fixture derivation must exercise a side condition"


def test_compact_certificate_status_omits_blocking_steps(pool):
    dr = _multistep_derivation(pool)
    compact = dr.to_dict(mode="compact")
    assert "certifiable" in compact["certificate_status"]
    assert "reason" in compact["certificate_status"]
    assert "blocking_steps" not in compact["certificate_status"]
    assert compact["certificate_status"]["reason"] == dr.certificate_status["reason"]


def test_compact_never_contains_lean_source_text(pool):
    dr = _multistep_derivation(pool)
    compact_json = dr.to_json(mode="compact")
    # Lean certificate source is theorem/proof syntax; make sure none of it
    # leaked into the compact envelope regardless of derivation shape.
    for marker in ("theorem ", "import Mathlib", ":= by"):
        assert marker not in compact_json


# ---------------------------------------------------------------------------
# JSON round-trip and discriminator
# ---------------------------------------------------------------------------


def test_to_json_round_trips_for_both_modes(pool):
    dr = _multistep_derivation(pool)
    for mode in ("full", "compact"):
        loaded = json.loads(dr.to_json(mode=mode))
        assert loaded == dr.to_dict(mode=mode)


def test_kind_discriminator_is_stable_across_modes(pool):
    dr = _multistep_derivation(pool)
    assert dr.to_dict(mode="full")["kind"] == "alkahest.derived_result"
    assert dr.to_dict(mode="compact")["kind"] == "alkahest.derived_result"


def test_invalid_mode_raises_value_error(pool):
    dr = _multistep_derivation(pool)
    with pytest.raises(ValueError):
        dr.to_dict(mode="bogus")
    with pytest.raises(ValueError):
        dr.to_json(mode="bogus")


def test_simple_zero_step_derivation_round_trips(pool):
    x = pool.symbol("x")
    dr = ak.diff(x, x)
    for mode in ("full", "compact"):
        loaded = json.loads(dr.to_json(mode=mode))
        assert loaded["kind"] == "alkahest.derived_result"
        assert "status" in loaded["verification"]
