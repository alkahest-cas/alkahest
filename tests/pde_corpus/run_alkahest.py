"""Run the second-order corpus through alkahest and print one JSON line per
entry: ``{"id", "status", "kind", "method", "verified", "solution", "seconds"}``.

``python tests/pde_corpus/run_alkahest.py [out.jsonl]``
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alkahest as ak  # noqa: E402
import alkahest.experimental as akx  # noqa: E402
from second_order import CORPUS  # noqa: E402
from translate import to_alkahest  # noqa: E402


def run_entry(entry) -> dict:
    pool = ak.ExprPool()
    eq, u, ics = to_alkahest(entry, pool)
    t0 = time.perf_counter()
    try:
        s = akx.solve_second_order(eq, u, ics=ics or None)
        res = {
            "status": "solved",
            "kind": s.kind,
            "method": s.method,
            "verified": s.verification["status"],
            "solution": str(s.solution),
        }
    except akx.PdeError as e:
        code = getattr(e, "code", None) or str(e)[1:11]
        res = {"status": code, "message": str(e).splitlines()[0][:300]}
    res["id"] = entry["id"]
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def main():
    out = open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout
    for entry in CORPUS:
        print(json.dumps(run_entry(entry), ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
