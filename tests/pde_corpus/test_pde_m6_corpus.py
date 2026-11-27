"""Corpus regression for the nonlinear layer (milestone M6).

Invariants: **no unverified answer** from ``pdsolve`` on the nonlinear corpus
(every answer re-verified independently through ``verify_pde_solution``); the
point-symmetry dimensions match the literature tables; every recovered
conservation law verifies exactly.  ``EXPECTED_SOLVED_M6`` is a lower bound on
coverage (an entry that stops solving is a regression).
"""

from __future__ import annotations

import pytest

akx = pytest.importorskip("alkahest.experimental")
if not hasattr(akx, "pde_symmetries"):  # pragma: no cover - old extension
    pytest.skip("extension built without the M6 layer", allow_module_level=True)

import alkahest as ak  # noqa: E402
from corpus_nonlinear import ABBASI_NONLINEAR, CONSERVATION, SOLITONS, SYMMETRIES  # noqa: E402
from pde_expected_m6 import EXPECTED_SOLVED_M6  # noqa: E402
from pde_harness import build, run_entry  # noqa: E402

WALL_MS = 60_000

ENTRIES = [("abbasi", e) for e in ABBASI_NONLINEAR] + [("soliton", e) for e in SOLITONS]


@pytest.mark.parametrize("entry", [e for _, e in ENTRIES], ids=[e["id"] for _, e in ENTRIES])
def test_nonlinear_entry(entry):
    r = run_entry(entry, wall_ms=WALL_MS)
    assert r["status"] != "unverified", f"{entry['id']}: unverified answer {r}"
    assert r["status"] != "build_error", r
    if entry["id"] in EXPECTED_SOLVED_M6:
        assert r["status"] == "verified", f"{entry['id']} regressed: {r}"


@pytest.mark.parametrize("entry", SYMMETRIES, ids=[e["id"] for e in SYMMETRIES])
def test_symmetry_dimension(entry):
    e = dict(entry, vars="x t")
    pde, u = build(e)
    with ak.context(budget=ak.Budget(wall_ms=WALL_MS)):
        S = akx.pde_symmetries(pde.equations[0], u)
    assert all(g.verified for g in S.generators)
    assert len(S.generators) == entry["dim"], (entry["id"], S)
    assert (S.infinite is not None) == entry["infinite"], (entry["id"], S)
    assert S.complete, (entry["id"], S.notes)


@pytest.mark.parametrize("entry", CONSERVATION, ids=[e["id"] for e in CONSERVATION])
def test_conservation_laws(entry):
    e = dict(entry, vars="x t")
    pde, u = build(e)
    eq = pde.equations[0]
    with ak.context(budget=ak.Budget(wall_ms=WALL_MS)):
        laws = akx.conservation_laws(eq, u, order=2)
        m = akx.conservation_multipliers(eq, u, order=2)
    for law in laws:
        assert law.verified, (entry["id"], law)
        assert akx.verify_conservation_law(eq, u, law.multiplier, law.components) == "proved_zero"
    if entry["laws"] is None:
        assert m["infinite"] is not None
    else:
        assert len(laws) == entry["laws"], (entry["id"], laws)
