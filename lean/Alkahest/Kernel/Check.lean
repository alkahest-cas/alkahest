/-
Top-level checkers and their soundness theorems.

* `checkEq H lhs rhs w`: `lhs = rhs` holds wherever the hypotheses `H` hold,
  because `lhs - rhs = Σ cᵢ·(Lᵢ - Rᵢ)` as polynomials in the atoms of `w`,
  every relation `Lᵢ = Rᵢ` holds under its side conditions, and those side
  conditions follow from `H` (`checkEq_sound`).
* `checkIntegral H F f w`: `F' = f` on the domain `H` — `D 0 F` is computed
  by the kernel, its differentiability conditions are implied by `H`, and
  `D 0 F = f` is checked by `checkEq` (`integral_sound`).

All checkers return `Bool` and are evaluated by `decide +kernel` (default) or,
opt-in, by `native_decide`.
-/
import Alkahest.Kernel.Norm
import Alkahest.Kernel.Sign
import Alkahest.Kernel.Deriv
import Alkahest.Kernel.Rel

namespace Alkahest.Kernel

/-- A cofactor: terms `(monomial, numerator, denominator)`. -/
abbrev Cof := List (Mono × Int × Nat)

/-- The polynomial of a cofactor. -/
def Cof.toPoly (c : Cof) : Poly := Poly.ofTerms (c.map fun t => (t.1, mkRat t.2.1 t.2.2))

/-- Certificate data for an identity: atom table, relations, one cofactor per relation. -/
structure Witness where
  atoms : List Atom
  rels : List Rel
  cofs : List Cof

/-- The generator `lhs - rhs` of a relation. -/
def relPoly (NT : NTable) (r : Rel) : Option Poly :=
  match norm NT r.lhs, norm NT r.rhs with
  | some p, some q => some (Poly.sub p q)
  | _, _ => none

/-- `Σ cᵢ·gᵢ`. -/
def combo (NT : NTable) : List Rel → List Cof → Option Poly
  | [], _ => some []
  | _ :: _, [] => none
  | r :: rs, c :: cs =>
    match relPoly NT r, combo NT rs cs with
    | some g, some s => some (Poly.add (Poly.mul c.toPoly g) s)
    | _, _ => none

/-- Side conditions of all relations. -/
def relConds : List Rel → List Cond
  | [] => []
  | r :: rs => r.conds ++ relConds rs

/-- `lhs = rhs` under `H`, by the witness. -/
def checkEq (H : List Cond) (lhs rhs : AExpr) (w : Witness) : Bool :=
  condsOk H (relConds w.rels) &&
  match buildNT w.atoms with
  | none => false
  | some NT =>
    match norm NT lhs, norm NT rhs, combo NT w.rels w.cofs with
    | some p, some q, some s => (Poly.sub (Poly.sub p q) s).isEmpty
    | _, _, _ => false

theorem relConds_mem {r : Rel} {c : Cond} : ∀ {rs : List Rel}, r ∈ rs → c ∈ r.conds →
    c ∈ relConds rs
  | [], h, _ => by simp at h
  | s :: rs, h, hc => by
    rcases List.mem_cons.1 h with rfl | h
    · simp [relConds, hc]
    · simp [relConds, relConds_mem h hc]

theorem combo_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) :
    ∀ {rs : List Rel} {cs : List Cof} {s : Poly}, (∀ c ∈ relConds rs, c.holds ρ) →
      combo NT rs cs = some s → s.eval vals = 0
  | [], _, s, _, h => by simp [combo] at h; subst h; rfl
  | _ :: _, [], _, _, h => by simp [combo] at h
  | r :: rs, c :: cs, s, hc, h => by
    unfold combo at h
    split at h
    · rename_i g s' hg hs
      simp at h; subst h
      have hr : r.lhs.eval ρ = r.rhs.eval ρ :=
        Rel.sound ρ r fun d hd => hc d (relConds_mem (List.mem_cons_self) hd)
      have hg0 : g.eval vals = 0 := by
        unfold relPoly at hg
        split at hg
        · rename_i p q hp hq
          simp at hg; subst hg
          rw [Poly.eval_sub, norm_sound hG hp, norm_sound hG hq, hr, sub_self]
        · simp at hg
      have hs0 : s'.eval vals = 0 :=
        combo_sound hG (fun d hd => hc d (by simp [relConds, hd])) hs
      rw [Poly.eval_add, Poly.eval_mul, hg0, hs0]; ring
    · simp at h

theorem checkEq_sound {H : List Cond} {lhs rhs : AExpr} {w : Witness}
    (h : checkEq H lhs rhs w = true) (ρ : List ℝ) (hH : HoldsAll ρ H) :
    lhs.eval ρ = rhs.eval ρ := by
  unfold checkEq at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨hc, h⟩ := h
  have hrc := condsOk_sound hH hc
  split at h
  · simp at h
  · rename_i NT hNT
    have hG := buildNT_good (ρ := ρ) hNT
    split at h
    · rename_i p q s hp hq hs
      have h0 : Poly.sub (Poly.sub p q) s = [] := List.isEmpty_iff.1 h
      have := congrArg (Poly.eval (atomVals ρ w.atoms)) h0
      rw [Poly.eval_sub, Poly.eval_sub, norm_sound hG hp, norm_sound hG hq,
        combo_sound hG hrc hs] at this
      simp at this
      linarith
    · simp at h

/-- `F' = f` on the domain `H`. -/
def checkIntegral (H : List Cond) (F f : AExpr) (w : Witness) : Bool :=
  condsOk H (dconds F) && checkEq H (D 0 F) f w

/-- **Integration certificates.** If the checker accepts, `F` is an antiderivative
of `f` at every point `x` (parameters `ps`) where the hypotheses `H` hold. -/
theorem integral_sound {H : List Cond} {F f : AExpr} {w : Witness}
    (h : checkIntegral H F f w = true) (ps : List ℝ) (x : ℝ) (hH : HoldsAll (x :: ps) H) :
    HasDerivAt (fun y => F.eval (y :: ps)) (f.eval (x :: ps)) x := by
  unfold checkIntegral at h
  simp only [Bool.and_eq_true] at h
  have hd := D0_hasDerivAt ps x F (condsOk_sound hH h.1)
  rw [checkEq_sound h.2 (x :: ps) hH] at hd
  exact hd

/-- **Derivative certificates** (ODE residuals): `(d/dx) F = G` on the domain `H`. -/
theorem deriv_sound {H : List Cond} {F G : AExpr} {w : Witness}
    (h : checkIntegral H F G w = true) (ps : List ℝ) (x : ℝ) (hH : HoldsAll (x :: ps) H) :
    deriv (fun y => F.eval (y :: ps)) x = G.eval (x :: ps) :=
  (integral_sound h ps x hH).deriv

end Alkahest.Kernel
