"""Tests for V2-6 — LLL (``alkahest.lattice``) + ``guess_relation``."""

from __future__ import annotations

from decimal import Decimal, getcontext

import alkahest
import pytest
from alkahest import LatticeError, guess_relation, lattice


def test_lll_planar_fixture_same_as_core():
    """Planar fixture from Rust tests; reduced basis stays well-formed."""
    rows = [[2, 15], [1, 21]]
    out = lattice.lll_reduce_rows(rows)
    assert len(out) == 2
    assert all(len(r) == 2 for r in out)


def test_lll_idempotent_under_second_pass():
    rows = [[2, 15], [1, 21]]
    once = lattice.lll_reduce_rows(rows)
    twice = lattice.lll_reduce_rows(once)
    assert twice == once


def test_lll_empty_basis():
    with pytest.raises(LatticeError, match="basis"):
        lattice.lll_reduce_rows([])


def test_lll_ragged_rows():
    with pytest.raises(LatticeError):
        lattice.lll_reduce_rows([[1, 2], [3]])


def test_guess_relation_accepts_python_float_entries():
    rel = guess_relation([1.0, 2.0, 3.0], precision_bits=384)
    assert rel is not None
    xs = [Decimal(str(x)) for x in (1.0, 2.0, 3.0)]
    getcontext().prec = 140
    s = sum(Decimal(ci) * x for ci, x in zip(rel, xs))
    assert s.copy_abs() < Decimal(10) ** -100


def test_guess_relation_1_2_3_tuple():
    rel = guess_relation(["1", "2", "3"], precision_bits=256)
    assert rel is not None
    xs = [Decimal(x) for x in ("1", "2", "3")]
    getcontext().prec = 120
    s = sum(Decimal(ci) * x for ci, x in zip(rel, xs))
    assert s.copy_abs() < Decimal(10) ** -70


def test_guess_relation_rejects_bad_element_type():
    with pytest.raises(TypeError):
        guess_relation([1 + 2j])


def test_expose_errors_module_level():
    assert issubclass(LatticeError, alkahest.AlkahestError)
    assert issubclass(alkahest.PslqError, alkahest.AlkahestError)
