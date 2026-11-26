/-
Symbolic partial derivative `D i : AExpr → AExpr` and its soundness:
on the domain `dconds e`, `y ↦ eval e (ρ.set i y)` has derivative
`eval (D i e)` (`D_hasDerivAt`).
-/
import Alkahest.Kernel.Expr
import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic.FieldSimp

namespace Alkahest.Kernel

/-- Derivative of a unary function, as an expression in its argument. -/
def dfn : Fn → AExpr → AExpr
  | .exp, a => .app .exp a
  | .log, a => .inv a
  | .logabs, a => .inv a
  | .sin, a => .app .cos a
  | .cos, a => .neg (.app .sin a)
  | .tan, a => .inv (.pow (.app .cos a) 2)
  | .sqrt, a => .inv (.mul (.lit 2) (.app .sqrt a))
  | .atan, a => .inv (.add .one (.pow a 2))
  | .asin, a => .inv (.app .sqrt (.sub .one (.pow a 2)))
  | .acos, a => .neg (.inv (.app .sqrt (.sub .one (.pow a 2))))
  | .sinh, a => .app .cosh a
  | .cosh, a => .app .sinh a
  | .tanh, a => .inv (.pow (.app .cosh a) 2)
  | .arsinh, a => .inv (.app .sqrt (.add .one (.pow a 2)))

/-- Partial derivative with respect to variable `i`. -/
def D (i : Nat) : AExpr → AExpr
  | .var j => if j = i then .one else .zero
  | .zero => .zero
  | .one => .zero
  | .lit _ => .zero
  | .pi => .zero
  | .add a b => .add (D i a) (D i b)
  | .sub a b => .sub (D i a) (D i b)
  | .mul a b => .add (.mul (D i a) b) (.mul a (D i b))
  | .div a b => .div (.sub (.mul (D i a) b) (.mul a (D i b))) (.pow b 2)
  | .neg a => .neg (D i a)
  | .inv a => .div (.neg (D i a)) (.pow a 2)
  | .pow a n => .mul (.mul (AExpr.ofNat n) (.pow a (n - 1))) (D i a)
  | .app f a => .mul (dfn f a) (D i a)

/-- Where a unary function is differentiable (with the derivative `dfn`). -/
def fconds : Fn → AExpr → List Cond
  | .log, a => [.ne a]
  | .logabs, a => [.ne a]
  | .tan, a => [.ne (.app .cos a)]
  | .sqrt, a => [.pos a]
  | .asin, a => [.lt (.neg .one) a, .lt a .one]
  | .acos, a => [.lt (.neg .one) a, .lt a .one]
  | _, _ => []

/-- The differentiability domain of an expression. -/
def dconds : AExpr → List Cond
  | .add a b => dconds a ++ dconds b
  | .sub a b => dconds a ++ dconds b
  | .mul a b => dconds a ++ dconds b
  | .div a b => dconds a ++ dconds b ++ [.ne b]
  | .neg a => dconds a
  | .pow a _ => dconds a
  | .inv a => dconds a ++ [.ne a]
  | .app f a => dconds a ++ fconds f a
  | _ => []

theorem tanh_hasDerivAt (t : ℝ) : HasDerivAt Real.tanh (Real.cosh t ^ 2)⁻¹ t := by
  have hc : Real.cosh t ≠ 0 := (Real.cosh_pos t).ne'
  have h := (Real.hasDerivAt_sinh t).div (Real.hasDerivAt_cosh t) hc
  have hf : (Real.sinh / Real.cosh) = Real.tanh := by
    funext y; simp [Real.tanh_eq_sinh_div_cosh]
  rw [hf] at h
  refine h.congr_deriv ?_
  have := Real.cosh_sq_sub_sinh_sq t
  field_simp
  linear_combination this

theorem fn_hasDerivAt (ρ : List ℝ) (f : Fn) (a : AExpr) (h : ∀ c ∈ fconds f a, c.holds ρ) :
    HasDerivAt f.eval ((dfn f a).eval ρ) (a.eval ρ) := by
  cases f with
  | exp => exact Real.hasDerivAt_exp _
  | log =>
    exact Real.hasDerivAt_log (h (.ne a) (by simp [fconds]))
  | logabs =>
    have hf : Fn.eval .logabs = Real.log := funext fun t => Real.log_abs t
    rw [hf]
    exact Real.hasDerivAt_log (h (.ne a) (by simp [fconds]))
  | sin => exact Real.hasDerivAt_sin _
  | cos => exact Real.hasDerivAt_cos _
  | tan =>
    have hc : Real.cos (a.eval ρ) ≠ 0 := h (.ne (.app .cos a)) (by simp [fconds])
    exact (Real.hasDerivAt_tan hc).congr_deriv (by simp [dfn, AExpr.eval, Fn.eval])
  | sqrt =>
    have hp : 0 < a.eval ρ := h (.pos a) (by simp [fconds])
    exact (Real.hasDerivAt_sqrt hp.ne').congr_deriv (by simp [dfn, AExpr.eval, Fn.eval])
  | atan =>
    exact (Real.hasDerivAt_arctan _).congr_deriv (by simp [dfn, AExpr.eval])
  | asin =>
    have h1 : -1 < a.eval ρ := by
      have := h (.lt (.neg .one) a) (by simp [fconds]); simpa [Cond.holds, AExpr.eval] using this
    have h2 : a.eval ρ < 1 := by
      have := h (.lt a .one) (by simp [fconds]); simpa [Cond.holds, AExpr.eval] using this
    exact (Real.hasDerivAt_arcsin h1.ne' h2.ne).congr_deriv (by simp [dfn, AExpr.eval, Fn.eval])
  | acos =>
    have h1 : -1 < a.eval ρ := by
      have := h (.lt (.neg .one) a) (by simp [fconds]); simpa [Cond.holds, AExpr.eval] using this
    have h2 : a.eval ρ < 1 := by
      have := h (.lt a .one) (by simp [fconds]); simpa [Cond.holds, AExpr.eval] using this
    exact (Real.hasDerivAt_arccos h1.ne' h2.ne).congr_deriv (by simp [dfn, AExpr.eval, Fn.eval])
  | sinh => exact Real.hasDerivAt_sinh _
  | cosh => exact Real.hasDerivAt_cosh _
  | tanh => exact tanh_hasDerivAt _
  | arsinh => exact Real.hasDerivAt_arsinh _

theorem getD_set (ρ : List ℝ) (i j : Nat) (hi : i < ρ.length) (y : ℝ) :
    (ρ.set i y).getD j 0 = if j = i then y else ρ.getD j 0 := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_set]
  split_ifs with h1 h2 h2 <;> subst_vars <;> simp_all

theorem D_hasDerivAt (i : Nat) (ρ : List ℝ) (hi : i < ρ.length) (x : ℝ) :
    ∀ e : AExpr, (∀ c ∈ dconds e, c.holds (ρ.set i x)) →
      HasDerivAt (fun y => e.eval (ρ.set i y)) ((D i e).eval (ρ.set i x)) x
  | .var j, _ => by
    simp only [AExpr.eval, getD_set ρ i j hi, D]
    split_ifs
    · exact hasDerivAt_id' x
    · exact hasDerivAt_const x _
  | .zero, _ => hasDerivAt_const x _
  | .one, _ => hasDerivAt_const x _
  | .lit _, _ => hasDerivAt_const x _
  | .pi, _ => hasDerivAt_const x _
  | .add a b, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hb := D_hasDerivAt i ρ hi x b fun c hc => h c (by simp [dconds, hc])
    exact ha.add hb
  | .sub a b, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hb := D_hasDerivAt i ρ hi x b fun c hc => h c (by simp [dconds, hc])
    exact ha.sub hb
  | .mul a b, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hb := D_hasDerivAt i ρ hi x b fun c hc => h c (by simp [dconds, hc])
    exact ha.mul hb
  | .div a b, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hb := D_hasDerivAt i ρ hi x b fun c hc => h c (by simp [dconds, hc])
    have hne : b.eval (ρ.set i x) ≠ 0 := h (.ne b) (by simp [dconds])
    exact ha.div hb hne
  | .neg a, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    exact ha.neg
  | .inv a, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hne : a.eval (ρ.set i x) ≠ 0 := h (.ne a) (by simp [dconds])
    exact (ha.inv hne).congr_deriv (by simp [D, AExpr.eval, neg_div])
  | .pow a n, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    exact (ha.pow n).congr_deriv (by simp [D, AExpr.eval])
  | .app f a, h => by
    have ha := D_hasDerivAt i ρ hi x a fun c hc => h c (by simp [dconds, hc])
    have hf := fn_hasDerivAt (ρ.set i x) f a fun c hc => h c (by simp [dconds, hc])
    exact hf.comp x ha

/-- The derivative in the first variable at `x :: ps` (the integration form). -/
theorem D0_hasDerivAt (ps : List ℝ) (x : ℝ) (e : AExpr)
    (h : ∀ c ∈ dconds e, c.holds (x :: ps)) :
    HasDerivAt (fun y => e.eval (y :: ps)) ((D 0 e).eval (x :: ps)) x :=
  D_hasDerivAt 0 (x :: ps) (by simp) x e h

end Alkahest.Kernel
