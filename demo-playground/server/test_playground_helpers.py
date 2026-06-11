"""Tests for playground Lean certificate helpers."""

from playground_helpers import fix_legacy_diff_lean

_LEGACY_DIFF_X3 = """import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral

open Real MeasureTheory

-- Step 1: diff_univariate_poly
example : ((x : ℝ)) ^ (3 : ℕ) = ((3 : ℝ) * ((x : ℝ)) ^ (2 : ℕ)) :=
  by ring_nf; simp

"""


def test_fix_legacy_diff_rewrites_deriv_goal():
    fixed = fix_legacy_diff_lean(_LEGACY_DIFF_X3)
    assert "deriv (fun (x : ℝ)" in fixed
    assert "deriv_pow" in fixed
    assert "MeasureTheory" not in fixed
    assert "ring_nf" not in fixed
    assert "((x : ℝ)) ^ (3 : ℕ) = ((3 : ℝ)" not in fixed


def test_fix_legacy_diff_idempotent():
    once = fix_legacy_diff_lean(_LEGACY_DIFF_X3)
    twice = fix_legacy_diff_lean(once)
    assert once == twice
