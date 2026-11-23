(* Reference solutions for tests/pde_corpus/transform_problems.json.
   Run: WolframKernel -script transforms.wl > transforms_ref.txt

   Each line: id <TAB> status <TAB> value at the sample points <TAB> DSolveValue result
   * status: "closed" (no unevaluated integral or DSolveValue), "integral"
     (an Inactive / unevaluated Integrate in the result), or "failed";
   * values: N[u] at (x, t) = (3/10, 7/10) and (11/10, 1/5) when closed, with
     k = c = 1 for symbolic constants (Indeterminate otherwise). *)

SetOptions[$Output, PageWidth -> Infinity];
pts = {{3/10, 7/10}, {11/10, 1/5}};
status[r_] := Which[
   !FreeQ[r, DSolveValue] || r === $Failed, "failed",
   !FreeQ[r, Integrate | Inactive | Inactivate | Integral | InverseFourierTransform | InverseLaplaceTransform | FourierTransform | LaplaceTransform], "integral",
   True, "closed"];
vals[r_] := If[status[r] === "closed",
   Quiet@Table[N[r /. {x -> p[[1]], t -> p[[2]], y -> p[[2]], k -> 1, c -> 1}, 20], {p, pts}],
   Indeterminate];
row[id_, r_] := Print[id, "\t", status[r], "\t", ToString[vals[r], InputForm], "\t", ToString[r, InputForm]];
go[expr_] := TimeConstrained[Quiet@expr, 300, $Failed];

(* Fourier in x on the line (Haberman ch. 10, Strauss §2.4, §12.3) *)
row["heat_line_gauss", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == Exp[-x^2]}, u[x, t], {x, t}]];
row["heat_line_symbolic", go@DSolveValue[{D[u[x, t], t] == k D[u[x, t], {x, 2}], u[x, 0] == f[x]}, u[x, t], {x, t}, Assumptions -> k > 0]];
row["heat_line_reaction", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}] - u[x, t], u[x, 0] == Exp[-x^2]}, u[x, t], {x, t}]];
row["wave_line_dalembert", go@DSolveValue[{D[u[x, t], {t, 2}] == c^2 D[u[x, t], {x, 2}], u[x, 0] == f[x], Derivative[0, 1][u][x, 0] == g[x]}, u[x, t], {x, t}, Assumptions -> c > 0]];
row["wave_line_gauss", go@DSolveValue[{D[u[x, t], {t, 2}] == D[u[x, t], {x, 2}], u[x, 0] == Exp[-x^2], Derivative[0, 1][u][x, 0] == 0}, u[x, t], {x, t}]];
row["wave_line_velocity", go@DSolveValue[{D[u[x, t], {t, 2}] == 4 D[u[x, t], {x, 2}], u[x, 0] == 0, Derivative[0, 1][u][x, 0] == x Exp[-x^2]}, u[x, t], {x, t}]];
row["laplace_halfplane_lorentz", go@DSolveValue[{D[u[x, y], {x, 2}] + D[u[x, y], {y, 2}] == 0, u[x, 0] == 1/(1 + x^2)}, u[x, y], {x, y}, Assumptions -> y > 0]];
row["laplace_halfplane_symbolic", go@DSolveValue[{D[u[x, y], {x, 2}] + D[u[x, y], {y, 2}] == 0, u[x, 0] == f[x]}, u[x, y], {x, y}, Assumptions -> y > 0]];
row["schrodinger_gauss", go@DSolveValue[{I D[u[x, t], t] + D[u[x, t], {x, 2}] == 0, u[x, 0] == Exp[-x^2]}, u[x, t], {x, t}]];
row["schrodinger_symbolic", go@DSolveValue[{I D[u[x, t], t] + D[u[x, t], {x, 2}] == 0, u[x, 0] == f[x]}, u[x, t], {x, t}]];
row["klein_gordon", go@DSolveValue[{D[u[x, t], {t, 2}] - D[u[x, t], {x, 2}] + u[x, t] == 0, u[x, 0] == f[x], Derivative[0, 1][u][x, 0] == 0}, u[x, t], {x, t}]];
row["telegraph", go@DSolveValue[{D[u[x, t], {t, 2}] + 2 D[u[x, t], t] - D[u[x, t], {x, 2}] == 0, u[x, 0] == f[x], Derivative[0, 1][u][x, 0] == 0}, u[x, t], {x, t}]];
(* Laplace in t and sine / cosine transforms on the half-line (Haberman §10.5–10.6, Strauss §3.1–3.4) *)
row["heat_halfline_const", go@DSolveValue[{D[u[x, t], t] == k D[u[x, t], {x, 2}], u[x, 0] == 0, u[0, t] == 1}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0 && k > 0]];
row["heat_halfline_duhamel", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == 0, u[0, t] == g[t]}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0]];
row["heat_halfline_dirichlet", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == f[x], u[0, t] == 0}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0]];
row["heat_halfline_neumann", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == f[x], Derivative[1, 0][u][0, t] == 0}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0]];
row["wave_halfline_signal", go@DSolveValue[{D[u[x, t], {t, 2}] == c^2 D[u[x, t], {x, 2}], u[x, 0] == 0, Derivative[0, 1][u][x, 0] == 0, u[0, t] == g[t]}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0 && c > 0]];
row["heat_halfline_mixed", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == f[x], u[0, t] == 1}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0]];
row["heat_halfline_exp", go@DSolveValue[{D[u[x, t], t] == D[u[x, t], {x, 2}], u[x, 0] == Exp[-x], u[0, t] == 0}, u[x, t], {x, t}, Assumptions -> x > 0 && t > 0]];
