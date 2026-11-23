(* Reference eigenvalues for tests/pde_corpus/sturm_liouville.json.
   Run: WolframKernel -script sturm_liouville.wl > sturm_liouville_ref.txt

   Each line: id <TAB> symbolic <TAB> reference
   * symbolic: whether Mathematica's symbolic DEigenvalues returned the
     eigenvalues (True) or stayed unevaluated (False) - the head-to-head score;
   * reference: the first eigenvalues to 20 digits, from the symbolic result
     when there is one, otherwise from an independent route (FindRoot on the
     textbook eigenvalue condition, BesselJZero, NDEigenvalues). *)

SetOptions[$Output, PageWidth -> Infinity];
row[id_, sym_, ref_] := Print[id, "\t", sym, "\t", ToString[N[ref, 20], InputForm]];
symOK[e_] := ListQ[e] && AllTrue[e, NumericQ];
tryS[expr_] := TimeConstrained[expr, 120, $Failed];

(* the k-th positive root of f on (0, oo), bracketed by a fine sign scan *)
roots[f_, n_, step_: 1/50, top_: 200] := Module[{xs, s, out = {}},
  xs = Range[step, top, step];
  s = Sign[f /@ N[xs, 30]];
  Do[If[s[[i]] s[[i + 1]] < 0 && Length[out] < n,
     AppendTo[out, x /. FindRoot[f[x], {x, xs[[i]], xs[[i + 1]]}, WorkingPrecision -> 40, Method -> "Brent"]]],
   {i, Length[xs] - 1}];
  out];

ev["dd_L1"] = tryS@DEigenvalues[{-u''[x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, 1}, 5];
row["dd_L1", symOK[ev["dd_L1"]], Table[(k Pi)^2, {k, 1, 5}]];
ev["nn_L1"] = tryS@DEigenvalues[{-u''[x]}, u[x], {x, 0, 1}, 5];
row["nn_L1", symOK[ev["nn_L1"]], Table[(k Pi)^2, {k, 0, 4}]];
ev["dn_L1"] = tryS@DEigenvalues[{-u''[x], DirichletCondition[u[x] == 0, x == 0]}, u[x], {x, 0, 1}, 5];
row["dn_L1", symOK[ev["dn_L1"]], Table[((k - 1/2) Pi)^2, {k, 1, 5}]];
ev["nd_L2"] = tryS@DEigenvalues[{-u''[x], DirichletCondition[u[x] == 0, x == 2]}, u[x], {x, 0, 2}, 5];
row["nd_L2", symOK[ev["nd_L2"]], Table[((k - 1/2) Pi/2)^2, {k, 1, 5}]];
ev["periodic_pi"] = tryS@DEigenvalues[{-u''[x], PeriodicBoundaryCondition[u[x], x == Pi, Function[x, x - 2 Pi]]}, u[x], {x, -Pi, Pi}, 5];
row["periodic_pi", symOK[ev["periodic_pi"]], {0, 1, 1, 4, 4}];
(* Robin (Haberman 5.8, Strauss 4.3). With -u'', NeumannValue[g] at an end means du/dn = -g: u'(1) + u(1) = 0 is NeumannValue[u] at x = 1, u'(0) - u(0) = 0 is NeumannValue[u] at x = 0 *)
ev["dr_h1"] = tryS@DEigenvalues[{-u''[x] + NeumannValue[u[x], x == 1], DirichletCondition[u[x] == 0, x == 0]}, u[x], {x, 0, 1}, 4];
row["dr_h1", symOK[ev["dr_h1"]], roots[Function[v, Sin[v] + v Cos[v]], 4]^2];
ev["nr_h1"] = tryS@DEigenvalues[{-u''[x] + NeumannValue[u[x], x == 1]}, u[x], {x, 0, 1}, 4];
row["nr_h1", symOK[ev["nr_h1"]], roots[Function[v, v Sin[v] - Cos[v]], 4]^2];
(* u'(0) - u(0) = 0, u'(1) + u(1) = 0: D = (1 - nu^2) sin nu + 2 nu cos nu *)
ev["rr_11"] = tryS@DEigenvalues[{-u''[x] + NeumannValue[u[x], x == 1] + NeumannValue[u[x], x == 0]}, u[x], {x, 0, 1}, 4];
row["rr_11", symOK[ev["rr_11"]], roots[Function[v, (1 - v^2) Sin[v] + 2 v Cos[v]], 4]^2];
(* u(0) = 0, u'(1) - 2 u(1) = 0: one negative eigenvalue -kappa^2, tanh kappa = kappa/2 *)
ev["dr_hm2"] = tryS@DEigenvalues[{-u''[x] + NeumannValue[-2 u[x], x == 1], DirichletCondition[u[x] == 0, x == 0]}, u[x], {x, 0, 1}, 4];
row["dr_hm2", symOK[ev["dr_hm2"]], Join[-roots[Function[k, 2 Sinh[k] - k Cosh[k]], 1]^2, roots[Function[v, -2 Sin[v] + v Cos[v]], 3]^2]];
row["dr_hm2_numeric", "FEM", NDEigenvalues[{-u''[x] + NeumannValue[-2 u[x], x == 1], DirichletCondition[u[x] == 0, x == 0]}, u[x], {x, 0, 1}, 4]];
ev["dd_q2"] = tryS@DEigenvalues[{-u''[x] - 2 u[x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, 1}, 4];
row["dd_q2", symOK[ev["dd_q2"]], Table[(k Pi)^2 - 2, {k, 1, 4}]];
ev["dd_w4"] = tryS@DEigenvalues[{-u''[x]/4, DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, 1}, 4];
row["dd_w4", symOK[ev["dd_w4"]], Table[(k Pi)^2/4, {k, 1, 4}]];
(* Haberman 5.3: (x u')' + lambda u / x = 0 on [1, 2] *)
ev["euler_12"] = tryS@DEigenvalues[{-x D[x u'[x], x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 1, 2}, 4];
row["euler_12", symOK[ev["euler_12"]], Table[(k Pi/Log[2])^2, {k, 1, 4}]];
(* Bessel on the unit disk *)
ev["bessel0_dir"] = tryS@DEigenvalues[{-D[r u'[r], r]/r, DirichletCondition[u[r] == 0, r == 1]}, u[r], {r, 0, 1}, 4];
row["bessel0_dir", symOK[ev["bessel0_dir"]], Table[BesselJZero[0, k]^2, {k, 1, 4}]];
ev["bessel1_dir"] = tryS@DEigenvalues[{-D[r u'[r], r]/r + u[r]/r^2, DirichletCondition[u[r] == 0, r == 1]}, u[r], {r, 0, 1}, 4];
row["bessel1_dir", symOK[ev["bessel1_dir"]], Table[BesselJZero[1, k]^2, {k, 1, 4}]];
ev["bessel2_dir"] = tryS@DEigenvalues[{-D[r u'[r], r]/r + 4 u[r]/r^2, DirichletCondition[u[r] == 0, r == 1]}, u[r], {r, 0, 1}, 4];
row["bessel2_dir", symOK[ev["bessel2_dir"]], Table[BesselJZero[2, k]^2, {k, 1, 4}]];
ev["bessel0_neu"] = tryS@DEigenvalues[{-D[r u'[r], r]/r}, u[r], {r, 0, 1}, 4];
row["bessel0_neu", symOK[ev["bessel0_neu"]], Join[{0}, Table[BesselJZero[1, k]^2, {k, 1, 3}]]];
(* Bessel, radius 2, Robin R'(2) + R(2) = 0: mu J0'(2 mu) + J0(2 mu) = 0 *)
ev["bessel0_rob"] = tryS@DEigenvalues[{-D[r u'[r], r]/r + NeumannValue[u[r], r == 2]}, u[r], {r, 0, 2}, 3];
row["bessel0_rob", symOK[ev["bessel0_rob"]], roots[Function[m, BesselJ[0, 2 m] - m BesselJ[1, 2 m]], 3]^2];
ev["legendre"] = tryS@DEigenvalues[{-D[(1 - x^2) u'[x], x]}, u[x], {x, -1, 1}, 5];
row["legendre", symOK[ev["legendre"]], Table[n (n + 1), {n, 0, 4}]];
ev["chebyshev"] = tryS@DEigenvalues[{-Sqrt[1 - x^2] D[Sqrt[1 - x^2] u'[x], x]}, u[x], {x, -1, 1}, 5];
row["chebyshev", symOK[ev["chebyshev"]], Table[n^2, {n, 0, 4}]];
ev["hermite"] = tryS@DEigenvalues[{-Exp[x^2] D[Exp[-x^2] u'[x], x]}, u[x], {x, -Infinity, Infinity}, 5];
row["hermite", symOK[ev["hermite"]], Table[2 n, {n, 0, 4}]];
ev["laguerre"] = tryS@DEigenvalues[{-Exp[x] D[x Exp[-x] u'[x], x]}, u[x], {x, 0, Infinity}, 5];
row["laguerre", symOK[ev["laguerre"]], Table[n, {n, 0, 4}]];
