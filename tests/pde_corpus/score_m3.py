"""Score the M3 corpus: Alkahest against the Mathematica references.

Run from the repository root with the extension importable::

    python3 tests/pde_corpus/score_m3.py

Prints one row per problem and a summary.  An Alkahest answer counts only when
it is exactly verified; a Mathematica one when ``DEigenvalues`` returned the
eigenvalues (eigenproblems) or ``DSolveValue`` returned a closed form or an
integral representation (transform problems).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import test_pde_m3 as t  # noqa: E402


def main() -> None:
    sl_ok = sl_math = 0
    print("eigenproblem          alkahest                         mathematica DEigenvalues")
    for e in t.SL_PROBLEMS:
        try:
            es = t._solve_sl(e)
            ok = all(f.verification["verified"] for f in es.families)
            kinds = ",".join("closed" if f.closed_form else "RootOfFamily" for f in es.families)
            mine = f"{'verified' if ok else 'UNVERIFIED'} ({kinds})"
        except Exception as exc:  # noqa: BLE001 - report the refusal
            ok, mine = False, f"refused {str(exc)[:30]}"
        math_ok = t.SL_REF[e["id"]][0] == "True"
        sl_ok += ok
        sl_math += math_ok
        print(f"{e['id']:<21} {mine:<32} {'solved' if math_ok else 'unevaluated'}")
    print(
        f"eigenproblems: alkahest {sl_ok}/{len(t.SL_PROBLEMS)}, mathematica {sl_math}/{len(t.SL_PROBLEMS)}\n"
    )
    tr_ok = tr_closed = m_closed = m_any = 0
    print(
        "transform problem           alkahest                              mathematica DSolveValue"
    )
    for e in t.TR_PROBLEMS:
        try:
            _, _, _, pde = t._build_transform_problem(e)
            sol = t.akx.transform_solve(pde)
            ok = sol.verified
            mine = f"{'verified' if ok else 'UNVERIFIED'} {sol.kind}"
            tr_closed += ok and sol.kind == "closed_form"
        except Exception as exc:  # noqa: BLE001
            ok, mine = False, f"refused {str(exc)[:30]}"
        ref = t.TR_REF[e["id"]][0]
        tr_ok += ok
        m_closed += ref == "closed"
        m_any += ref in ("closed", "integral")
        print(f"{e['id']:<27} {mine:<37} {ref}")
    n = len(t.TR_PROBLEMS)
    print(
        f"transform problems: alkahest {tr_ok}/{n} verified ({tr_closed} closed form); "
        f"mathematica {m_any}/{n} answered ({m_closed} closed form)"
    )


if __name__ == "__main__":
    main()
