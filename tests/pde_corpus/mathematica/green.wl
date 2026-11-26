(* Reference values for tests/pde_corpus/green_problems.json (PDE milestone M5).
   Run from tests/pde_corpus/mathematica:
       WolframKernel -script green.wl > green_ref.txt

   Each line: id <TAB> status <TAB> values <TAB> result
   * status: "closed" (GreenFunction / DSolveValue returned an expression free of
     GreenFunction, DSolveValue and unevaluated integrals), "integral" (an
     unevaluated / inactive integral in the result), "unevaluated" (returned
     unevaluated or timed out after 120 s), or "none" (no Mathematica query);
   * values: N[result, 20] at the problem's sample points with the parameters
     k = 3/2, kappa = 1, c = 1, m = 1, a = 1/2, L = 1, h = 2, R = 1, and
     DiracDelta[z] replaced by Exp[-z^2] for kernels marked delta_test. *)

SetOptions[$Output, PageWidth -> Infinity];
problems = Import[FileNameJoin[{DirectoryName[$InputFileName], "..", "green_problems.json"}], "RawJSON"];
params = {k -> 3/2, kappa -> 1, c -> 1, m -> 1, a -> 1/2, L -> 1, h -> 2, R -> 1};
status[r_] := Which[
   r === $Failed || !FreeQ[r, GreenFunction | DSolveValue], "unevaluated",
   !FreeQ[r, Integrate | Inactive | Inactivate | Integral], "integral",
   True, "closed"];
vals[r_, e_] := If[status[r] =!= "closed", Indeterminate,
   Table[
     Module[{rules = KeyValueMap[ToExpression[#1] -> #2 &, p], rr = r},
       If[TrueQ[Lookup[e, "delta_test", False]], rr = rr /. DiracDelta[z_] :> Exp[-z^2]];
       Quiet@N[rr /. rules /. params, 20]],
     {p, e["points"]}]];
Do[
  If[e["mma"] === "None",
    Print[e["id"], "\t", "none", "\t", "{}", "\t", "None"],
    Module[{r = TimeConstrained[Quiet@ToExpression[e["mma"]], 120, $Failed]},
      Print[e["id"], "\t", status[r], "\t", ToString[vals[r, e], InputForm], "\t", ToString[r, InputForm]]]],
  {e, problems}];
