"""Run Mathematica's ``DSolveValue`` on the separation corpora (locally:
``WolframKernel -script``) and write one JSON line per problem: whether it
returned a solution, whether that contains an unevaluated sum / integral,
and its partial sum (indices ≤ N, the same convention as alkahest's
``truncate``) at the sample points recorded by ``run_separation.py``.

    python3 tests/pde_corpus/run_mathematica_sep.py alk.jsonl out.jsonl [batch] [timeout]
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mma_subset  # noqa: E402
from run_separation import NTERMS, load  # noqa: E402

KERNEL = os.environ.get(
    "WOLFRAM_KERNEL", "/usr/local/Wolfram/WolframEngine/14.3/Executables/WolframKernel"
)


def wl_number(v: float) -> str:
    return repr(float(v)).replace("e", "*^")


def script_for(entry, alk, timeout):
    defs, eqs, unknown, vars_, opts = mma_subset.wl_parts(entry["code"])
    params = alk.get("params") or {}
    points = alk.get("points") or []
    names = alk.get("vars") or []
    prules = "{" + ", ".join(f"{k} -> {wl_number(v)}" for k, v in params.items()) + "}"
    pts = (
        "{"
        + ", ".join(
            "{" + ", ".join(f"{n} -> {wl_number(x)}" for n, x in zip(names, p)) + "}"
            for p in points
        )
        + "}"
    )
    opt = (", " + opts) if opts else ""
    return f"""
Block[{{res, solved, unevalSum, unevalInt, ps, vals, t0}},
ClearAll["Global`*"];
{defs}
t0 = AbsoluteTime[];
res = TimeConstrained[
  Quiet[DSolveValue[Flatten[{eqs}], {unknown}, {vars_}{opt}]], {timeout}, $TimedOut];
solved = !MatchQ[res, $TimedOut | $Failed | _DSolveValue | Null] &&
  FreeQ[res, DSolveValue];
unevalSum = !FreeQ[res, Inactive[Sum]] || !FreeQ[res, Sum];
unevalInt = !FreeQ[res, Inactive[Integrate]] || !FreeQ[res, Integrate];
vals = If[solved && Length[{pts}] > 0,
  ps = res //. {{Inactive[Sum][e_, {{k_, a_, Infinity}}] :> Sum[e, {{k, a, {NTERMS}}}],
               Sum[e_, {{k_, a_, Infinity}}] :> Sum[e, {{k, a, {NTERMS}}}]}};
  Quiet[Table[Chop[N[Activate[ps /. {prules} /. pt], 20]], {{pt, {pts}}}]],
  {{}}];
vals = If[VectorQ[vals, NumericQ],
  Map[If[Im[#] == 0, Re[#], {{Re[#], Im[#]}}] &, vals], ToString[vals, InputForm]];
Print[ExportString[<|"id" -> "{entry["id"]}", "solved" -> solved, "sum" -> unevalSum,
  "integral" -> unevalInt, "time" -> AbsoluteTime[] - t0,
  "value" -> StringTake[ToString[res, InputForm], UpTo[1500]], "values" -> vals|>,
  "JSON", "Compact" -> True]];
];
"""


def main():
    alk_file, out = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    alk = {}
    with open(alk_file) as fh:
        for line in fh:
            r = json.loads(line)
            alk[r["id"]] = r
    entries = [e for e in load("all") if e["id"] in alk]
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for line in fh:
                with contextlib.suppress(Exception):
                    done.add(json.loads(line)["id"])
    todo = [e for e in entries if e["id"] not in done]
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        parts = []
        for e in chunk:
            try:
                parts.append(script_for(e, alk[e["id"]], timeout))
            except Exception as ex:
                with open(out, "a") as fh:
                    fh.write(
                        json.dumps({"id": e["id"], "solved": False, "error": str(ex)[:200]}) + "\n"
                    )
        if not parts:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".wl", delete=False) as fh:
            fh.write("\n".join(parts))
            path = fh.name
        try:
            r = subprocess.run(
                [KERNEL, "-script", path],
                capture_output=True,
                text=True,
                timeout=timeout * len(parts) + 120,
            )
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        except subprocess.TimeoutExpired:
            lines = []
        finally:
            os.unlink(path)
        got = set()
        with open(out, "a") as fh:
            for ln in lines:
                fh.write(ln + "\n")
                with contextlib.suppress(Exception):
                    got.add(json.loads(ln)["id"])
            for e in chunk:
                if e["id"] not in got:
                    fh.write(
                        json.dumps({"id": e["id"], "solved": False, "error": "no output"}) + "\n"
                    )
        print(f"{i + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
