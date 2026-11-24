/-
Alkahest algebraic kernel: the certificate AST and its real semantics.

`AExpr` is deliberately small: just what integration / ODE certificates need.
Every constructor evaluates to exactly the Lean term a reader would write
(`.add a b ↦ a + b`, `.app .sin a ↦ Real.sin a`, `.lit n ↦ (n : ℝ)`), so a
certificate can state its theorem in ordinary notation and the kernel's
conclusion matches it by definitional unfolding.
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace Alkahest.Kernel

/-- Unary elementary functions known to the kernel. -/
inductive Fn where
  | exp | log | sin | cos | tan | sqrt | atan | asin | acos
  | sinh | cosh | tanh | arsinh
  /-- `Real.log |a|`; normalised to the same atom as `log a` (`Real.log_abs`). -/
  | logabs
  deriving DecidableEq, Repr

/-- The real function a `Fn` denotes. -/
noncomputable def Fn.eval : Fn → ℝ → ℝ
  | .exp => Real.exp
  | .log => Real.log
  | .sin => Real.sin
  | .cos => Real.cos
  | .tan => Real.tan
  | .sqrt => Real.sqrt
  | .atan => Real.arctan
  | .asin => Real.arcsin
  | .acos => Real.arccos
  | .sinh => Real.sinh
  | .cosh => Real.cosh
  | .tanh => Real.tanh
  | .arsinh => Real.arsinh
  | .logabs => fun t => Real.log |t|

/-- Certificate expressions over variables `var 0, var 1, …`. -/
inductive AExpr where
  | var (i : Nat)
  | zero
  | one
  /-- `(n : ℝ)`; the emitter uses it for `n ≥ 2`. -/
  | lit (n : Nat)
  | pi
  | add (a b : AExpr)
  | sub (a b : AExpr)
  | mul (a b : AExpr)
  | div (a b : AExpr)
  | neg (a : AExpr)
  | inv (a : AExpr)
  | pow (a : AExpr) (n : Nat)
  | app (f : Fn) (a : AExpr)
  deriving DecidableEq, Repr

/-- Real semantics; variable `i` is the `i`-th entry of `ρ` (0 past the end). -/
noncomputable def AExpr.eval (ρ : List ℝ) : AExpr → ℝ
  | .var i => ρ.getD i 0
  | .zero => 0
  | .one => 1
  | .lit n => (n : ℝ)
  | .pi => Real.pi
  | .add a b => a.eval ρ + b.eval ρ
  | .sub a b => a.eval ρ - b.eval ρ
  | .mul a b => a.eval ρ * b.eval ρ
  | .div a b => a.eval ρ / b.eval ρ
  | .neg a => -a.eval ρ
  | .inv a => (a.eval ρ)⁻¹
  | .pow a n => a.eval ρ ^ n
  | .app f a => f.eval (a.eval ρ)

/-- The numeral `n` with `0` and `1` spelled as `zero` / `one`. -/
def AExpr.ofNat : Nat → AExpr
  | 0 => .zero
  | 1 => .one
  | n => .lit n

@[simp] theorem AExpr.eval_ofNat (ρ : List ℝ) (n : Nat) : (AExpr.ofNat n).eval ρ = n := by
  unfold AExpr.ofNat
  split <;> simp_all [AExpr.eval]

/-- Side conditions a certificate may assume or derive. -/
inductive Cond where
  | ne (e : AExpr)
  | pos (e : AExpr)
  | nonneg (e : AExpr)
  | lt (a b : AExpr)
  deriving DecidableEq, Repr

/-- What a side condition means at the point `ρ`. -/
def Cond.holds (ρ : List ℝ) : Cond → Prop
  | .ne e => e.eval ρ ≠ 0
  | .pos e => 0 < e.eval ρ
  | .nonneg e => 0 ≤ e.eval ρ
  | .lt a b => a.eval ρ < b.eval ρ

/-- All conditions of a list hold (nested `∧`, closed by `True`). -/
@[reducible] def HoldsAll (ρ : List ℝ) : List Cond → Prop
  | [] => True
  | c :: cs => c.holds ρ ∧ HoldsAll ρ cs

theorem HoldsAll.mem {ρ : List ℝ} : ∀ {cs : List Cond}, HoldsAll ρ cs → ∀ c ∈ cs, c.holds ρ
  | [], _, c, hc => by simp at hc
  | d :: ds, h, c, hc => by
    rcases List.mem_cons.1 hc with rfl | hc
    · exact h.1
    · exact HoldsAll.mem h.2 c hc

end Alkahest.Kernel
