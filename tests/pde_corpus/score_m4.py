"""Score the separation corpora: alkahest (``run_separation.py`` output)
against Mathematica 14.3 ``DSolveValue`` (``run_mathematica_sep.py`` output)
and the Abbasi benchmark's published verdicts.

    python3 tests/pde_corpus/score_m4.py alk.jsonl [mma.jsonl]

Counts are of *exactly verified* alkahest answers.  "agree" compares the
partial sums with every index ≤ N at two interior points (parameters at fixed
values), where both systems returned a solution without arbitrary functions.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    out = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                out[r["id"]] = r
    return out


def close(a, b):
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b) or not a:
        return None
    for x, y in zip(a, b):
        if isinstance(y, list):  # complex from Mathematica: [re, im]
            return None
        if x is None or y is None or x != x:
            return None
        if abs(x - y) > 1e-6 * (1 + abs(y)):
            return False
    return True


def score(entries, alk, mma, label):
    n = len(entries)
    ver = closed = withint = refused = untr = unver = 0
    m_solved = m_closed = m_pub = mp_pub = 0
    agree = disagree = compared = 0
    both = only_a = only_m = 0
    bad = []
    for e in entries:
        a = alk.get(e["id"], {})
        m = mma.get(e["id"], {})
        st = a.get("status")
        if st == "exactly_verified":
            ver += 1
            if a.get("closed"):
                closed += 1
            else:
                withint += 1
        elif st == "refused":
            refused += 1
        elif st == "untranslated":
            untr += 1
        else:
            unver += 1
        if m.get("solved"):
            m_solved += 1
            if not m.get("integral"):
                m_closed += 1
        if e.get("mma_published"):
            m_pub += 1
        if e.get("maple_published"):
            mp_pub += 1
        if st == "exactly_verified" and m.get("solved"):
            both += 1
        elif st == "exactly_verified":
            only_a += 1
        elif m.get("solved"):
            only_m += 1
        c = close(a.get("values"), m.get("values"))
        if c is not None:
            compared += 1
            if c:
                agree += 1
            else:
                disagree += 1
                bad.append((e["id"], a.get("values"), m.get("values")))
    print(f"## {label}: {n} problems")
    print(
        f"alkahest exactly verified: {ver} ({100 * ver / max(n, 1):.1f} %) = "
        f"{closed} closed coefficients + {withint} with unevaluated integrals"
    )
    print(f"  refused {refused}, outside the translated subset {untr}, unverified {unver}")
    if mma:
        print(
            f"Mathematica 14.3 DSolveValue (60 s): {m_solved} returned "
            f"({m_closed} without unevaluated integrals)"
        )
        print(f"  both {both}, alkahest only {only_a}, Mathematica only {only_m}")
        print(f"partial sums compared: {compared}; agree {agree}, disagree {disagree}")
        for b in bad:
            print("   disagree:", b)
    if any("mma_published" in e for e in entries):
        print(f"published verdicts: Mathematica 15 {m_pub}, Maple 2026.1 {mp_pub}")
    print()


def main():
    alk = load_jsonl(sys.argv[1])
    mma = load_jsonl(sys.argv[2] if len(sys.argv) > 2 else None)
    with open(os.path.join(HERE, "separation_corpus.json")) as fh:
        tb = json.load(fh)
    with open(os.path.join(HERE, "abbasi_bounded.json")) as fh:
        ab = json.load(fh)
    score(tb, alk, mma, "Textbook (Haberman, Strauss, Mathematica docs)")
    score(ab, alk, mma, "Abbasi benchmark, bounded-domain items of chapters 3-5")
    for sec in ("3.", "4.", "5."):
        sub = [e for e in ab if e["section"].startswith(sec)]
        score(sub, alk, mma, f"Abbasi section {sec}x")


if __name__ == "__main__":
    main()
