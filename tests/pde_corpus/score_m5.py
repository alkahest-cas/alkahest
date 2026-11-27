"""Score the M5 corpus (Green's functions): Alkahest against Mathematica 14.3.

Run from the repository root with the extension importable::

    python3 tests/pde_corpus/score_m5.py

An Alkahest answer counts only when it is exactly verified (a refusal that the
corpus expects — no Green's function exists — counts as correct); a
Mathematica one when ``GreenFunction`` / ``DSolveValue`` returned a closed form
(``integral`` counts separately).  Where both are closed the values were
compared at the sample points by ``tests/test_pde_m5.py``.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import test_pde_m5 as t


def main() -> None:
    rows = []
    by_cat: dict = {}
    for e in t.PROBLEMS:
        try:
            if "expect" in e:
                try:
                    t.build(e)
                    mine, ok = "ANSWERED (expected refusal)", False
                except Exception as exc:
                    ok = e["expect"] in str(exc)
                    mine = f"refused {e['expect']}" if ok else f"refused {str(exc)[:30]}"
            else:
                res, _ = t.build(e)
                ok = res.verified
                kind = getattr(res, "kind", None) or getattr(res, "method", "")
                mine = f"{'verified' if ok else 'UNVERIFIED'} {kind}"
        except Exception as exc:
            ok, mine = False, f"refused {str(exc)[:40]}"
        ref = t.REF[e["id"]][0]
        if "reference_error" in e:
            ref += " (wrong)"
        cat = by_cat.setdefault(e["category"], [0, 0, 0])
        cat[0] += 1
        cat[1] += ok
        cat[2] += ref == "closed" and "reference_error" not in e
        rows.append((e["id"], mine, ref))
    print(f"{'problem':<32} {'alkahest':<44} mathematica")
    for r in rows:
        print(f"{r[0]:<32} {r[1]:<44} {r[2]}")
    print()
    tot = [0, 0, 0]
    for cat, (n, a, m) in by_cat.items():
        print(f"{cat:<16} alkahest {a}/{n}   mathematica closed {m}/{n}")
        tot = [tot[0] + n, tot[1] + a, tot[2] + m]
    print(f"{'total':<16} alkahest {tot[1]}/{tot[0]}   mathematica closed {tot[2]}/{tot[0]}")


if __name__ == "__main__":
    main()
