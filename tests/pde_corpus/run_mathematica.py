"""Run the second-order corpus through Mathematica (``DSolveValue``) and
re-verify each answer by substitution in Mathematica itself.

``python tests/pde_corpus/run_mathematica.py /path/to/WolframKernel [out.jsonl]``

For each entry it records whether ``DSolveValue`` returned a solution, whether
``FullSimplify`` proves the residuals of the PDE (and of the initial
conditions) zero, and the time.  Not run in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from second_order import CORPUS  # noqa: E402
from translate import to_mathematica  # noqa: E402

TEMPLATE = r"""
pde = {pde};
ics = {ics};
v = {{{a}, {b}}};
r = AbsoluteTiming[TimeConstrained[
  If[ics === {{}}, DSolveValue[pde, u[{a}, {b}], v, Assumptions -> {assum}],
     DSolveValue[Join[{{pde}}, ics], u[{a}, {b}], v, Assumptions -> {assum}]], 120, $Failed]];
sol = r[[2]] /. Piecewise[{{e_, _}}, Indeterminate] :> e;
ok = Head[sol] =!= DSolveValue && sol =!= $Failed && FreeQ[sol, DSolveValue];
check[e_] := TimeConstrained[
  Simplify[FullSimplify[(e /. Equal -> Subtract) /. u -> Function[Evaluate[v], Evaluate[sol]],
    Assumptions -> {assum} && Element[v, Reals]]] === 0, 120, False];
pdeok = If[ok, check[pde], False];
icok = If[ok, And @@ (check /@ ics), False];
Print[ExportString[<|"ok" -> ok, "pde_verified" -> pdeok, "ic_verified" -> icok,
  "seconds" -> r[[1]], "solution" -> ToString[sol, InputForm]|>, "RawJSON", "Compact" -> True]];
"""


def main():
    kernel = sys.argv[1]
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
    for entry in CORPUS:
        pde, ics, assum = to_mathematica(entry)
        a, b = entry["vars"]
        code = TEMPLATE.format(pde=pde, ics=ics, a=a, b=b, assum=assum)
        with tempfile.NamedTemporaryFile("w", suffix=".wl", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.run([kernel, "-script", path], capture_output=True, text=True, timeout=600)
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "{}"
            res = json.loads(line)
        except Exception as e:  # noqa: BLE001 - a harness, record and go on
            res = {"ok": False, "error": str(e)[:200]}
        finally:
            os.unlink(path)
        res["id"] = entry["id"]
        print(json.dumps(res, ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
