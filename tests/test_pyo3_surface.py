"""Python-visible behaviour that a PyO3 upgrade changes silently.

Each of these compiled cleanly on the far side of an upgrade while behaving
differently: a returned ``Vec<u8>`` became ``bytes``, a trailing ``Option``
argument became required, a simple enum lost ``==``, and a class with no
constructor could in principle have become constructible.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest.experimental import (
    PauliOperator,
    symplectic_complement,
    theta_characteristic_bits,
)


def _is_int_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(b, int) for b in v)


def test_bit_vectors_are_lists_of_ints_not_bytes():
    p = PauliOperator("XZ")
    for bits, expected in [
        (p.x_bits, [1, 0]),
        (p.z_bits, [0, 1]),
        (p.symplectic(), [1, 0, 0, 1]),
    ]:
        assert _is_int_list(bits)
        assert bits == expected


def test_nested_bit_vectors_are_lists_too():
    comp = symplectic_complement([[1, 0, 0, 0]], 2)
    assert isinstance(comp, list)
    assert comp
    assert all(_is_int_list(row) for row in comp)
    a, b = theta_characteristic_bits(8, 2)
    assert _is_int_list(a)
    assert _is_int_list(b)


def test_trailing_optional_arguments_stay_optional():
    pool = ak.ExprPool()
    assert pool.float(1.5) == pool.float(1.5, 53)
    assert "prec=None" in ak.interval_eval.__text_signature__


def test_domain_compares_to_itself_and_to_its_integer_value():
    assert ak.Domain.Real == ak.Domain.Real
    assert ak.Domain.Real != ak.Domain.Complex
    assert ak.Domain.Real == int(ak.Domain.Real)


def test_class_without_a_constructor_still_refuses_construction():
    with pytest.raises(TypeError):
        ak.CompiledFn()
