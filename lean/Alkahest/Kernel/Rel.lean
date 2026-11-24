/-
Atom relations: the identities a certificate may combine with cofactors.

Each `Rel` denotes an equation `lhs = rhs` between expressions that holds
under its side conditions `conds` (`Rel.sound`, proved once here from
Mathlib). The kernel normalises `lhs - rhs` to a polynomial generator; a
certificate's cofactor witness says the claimed identity is a combination of
the generators.
-/
import Alkahest.Kernel.Expr
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace Alkahest.Kernel

/-- Known identities between atoms. -/
inductive Rel where
  /-- `u * u⁻¹ = 1` for `u ≠ 0`. -/
  | inv (u : AExpr)
  /-- `√u ^ 2 = u` for `0 ≤ u`. -/
  | sqrt (u : AExpr)
  /-- `sin a ^ 2 + cos a ^ 2 = 1`. -/
  | pyth (a : AExpr)
  /-- `cosh a ^ 2 - sinh a ^ 2 = 1`. -/
  | hpyth (a : AExpr)
  /-- `tan a = sin a / cos a`. -/
  | tan (a : AExpr)
  /-- `tanh a = sinh a / cosh a`. -/
  | tanh (a : AExpr)
  /-- `sin (a + b) = sin a cos b + cos a sin b`. -/
  | sinAdd (a b : AExpr)
  /-- `cos (a + b) = cos a cos b - sin a sin b`. -/
  | cosAdd (a b : AExpr)
  /-- `sin (-a) = -sin a`. -/
  | sinNeg (a : AExpr)
  /-- `cos (-a) = cos a`. -/
  | cosNeg (a : AExpr)
  /-- `sinh (a + b) = sinh a cosh b + cosh a sinh b`. -/
  | sinhAdd (a b : AExpr)
  /-- `cosh (a + b) = cosh a cosh b + sinh a sinh b`. -/
  | coshAdd (a b : AExpr)
  /-- `exp (a + b) = exp a * exp b`. -/
  | expAdd (a b : AExpr)
  /-- `exp (-a) = (exp a)⁻¹`. -/
  | expNeg (a : AExpr)
  /-- `log (exp a) = a`. -/
  | logExp (a : AExpr)
  /-- `exp (log a) = a` for `0 < a`. -/
  | expLog (a : AExpr)
  /-- `log (a * b) = log a + log b` for `a ≠ 0`, `b ≠ 0`. -/
  | logMul (a b : AExpr)
  /-- `log a⁻¹ = -log a`. -/
  | logInv (a : AExpr)
  /-- `log (a ^ n) = n * log a`. -/
  | logPow (a : AExpr) (n : Nat)
  /-- `√(a * b) = √a * √b` for `0 ≤ a`. -/
  | sqrtMul (a b : AExpr)
  deriving DecidableEq, Repr

namespace Rel

/-- Left-hand side. -/
def lhs : Rel → AExpr
  | .inv u => .mul u (.inv u)
  | .sqrt u => .pow (.app .sqrt u) 2
  | .pyth a => .add (.pow (.app .sin a) 2) (.pow (.app .cos a) 2)
  | .hpyth a => .sub (.pow (.app .cosh a) 2) (.pow (.app .sinh a) 2)
  | .tan a => .app .tan a
  | .tanh a => .app .tanh a
  | .sinAdd a b => .app .sin (.add a b)
  | .cosAdd a b => .app .cos (.add a b)
  | .sinNeg a => .app .sin (.neg a)
  | .cosNeg a => .app .cos (.neg a)
  | .sinhAdd a b => .app .sinh (.add a b)
  | .coshAdd a b => .app .cosh (.add a b)
  | .expAdd a b => .app .exp (.add a b)
  | .expNeg a => .app .exp (.neg a)
  | .logExp a => .app .log (.app .exp a)
  | .expLog a => .app .exp (.app .log a)
  | .logMul a b => .app .log (.mul a b)
  | .logInv a => .app .log (.inv a)
  | .logPow a n => .app .log (.pow a n)
  | .sqrtMul a b => .app .sqrt (.mul a b)

/-- Right-hand side. -/
def rhs : Rel → AExpr
  | .inv _ => .one
  | .sqrt u => u
  | .pyth _ => .one
  | .hpyth _ => .one
  | .tan a => .div (.app .sin a) (.app .cos a)
  | .tanh a => .div (.app .sinh a) (.app .cosh a)
  | .sinAdd a b => .add (.mul (.app .sin a) (.app .cos b)) (.mul (.app .cos a) (.app .sin b))
  | .cosAdd a b => .sub (.mul (.app .cos a) (.app .cos b)) (.mul (.app .sin a) (.app .sin b))
  | .sinNeg a => .neg (.app .sin a)
  | .cosNeg a => .app .cos a
  | .sinhAdd a b =>
    .add (.mul (.app .sinh a) (.app .cosh b)) (.mul (.app .cosh a) (.app .sinh b))
  | .coshAdd a b =>
    .add (.mul (.app .cosh a) (.app .cosh b)) (.mul (.app .sinh a) (.app .sinh b))
  | .expAdd a b => .mul (.app .exp a) (.app .exp b)
  | .expNeg a => .inv (.app .exp a)
  | .logExp a => a
  | .expLog a => a
  | .logMul a b => .add (.app .log a) (.app .log b)
  | .logInv a => .neg (.app .log a)
  | .logPow a n => .mul (AExpr.ofNat n) (.app .log a)
  | .sqrtMul a b => .mul (.app .sqrt a) (.app .sqrt b)

/-- Side conditions. -/
def conds : Rel → List Cond
  | .inv u => [.ne u]
  | .sqrt u => [.nonneg u]
  | .expLog a => [.pos a]
  | .logMul a b => [.ne a, .ne b]
  | .sqrtMul a _ => [.nonneg a]
  | _ => []

theorem sound (ρ : List ℝ) (r : Rel) (h : ∀ c ∈ r.conds, c.holds ρ) :
    r.lhs.eval ρ = r.rhs.eval ρ := by
  cases r with
  | inv u =>
    have : u.eval ρ ≠ 0 := h (.ne u) (by simp [conds])
    simp [lhs, rhs, AExpr.eval, this]
  | sqrt u =>
    have : 0 ≤ u.eval ρ := h (.nonneg u) (by simp [conds])
    simp [lhs, rhs, AExpr.eval, Fn.eval, Real.sq_sqrt this]
  | pyth a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.sin_sq_add_cos_sq]
  | hpyth a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.cosh_sq_sub_sinh_sq]
  | tan a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.tan_eq_sin_div_cos]
  | tanh a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.tanh_eq_sinh_div_cosh]
  | sinAdd a b => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.sin_add]
  | cosAdd a b => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.cos_add]
  | sinNeg a => simp [lhs, rhs, AExpr.eval, Fn.eval]
  | cosNeg a => simp [lhs, rhs, AExpr.eval, Fn.eval]
  | sinhAdd a b => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.sinh_add]
  | coshAdd a b => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.cosh_add]
  | expAdd a b => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.exp_add]
  | expNeg a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.exp_neg]
  | logExp a => simp [lhs, rhs, AExpr.eval, Fn.eval]
  | expLog a =>
    have : 0 < a.eval ρ := h (.pos a) (by simp [conds])
    simp [lhs, rhs, AExpr.eval, Fn.eval, Real.exp_log this]
  | logMul a b =>
    have ha : a.eval ρ ≠ 0 := h (.ne a) (by simp [conds])
    have hb : b.eval ρ ≠ 0 := h (.ne b) (by simp [conds])
    simp [lhs, rhs, AExpr.eval, Fn.eval, Real.log_mul ha hb]
  | logInv a => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.log_inv]
  | logPow a n => simp [lhs, rhs, AExpr.eval, Fn.eval, Real.log_pow]
  | sqrtMul a b =>
    have ha : 0 ≤ a.eval ρ := h (.nonneg a) (by simp [conds])
    simp [lhs, rhs, AExpr.eval, Fn.eval, Real.sqrt_mul ha]

end Rel

end Alkahest.Kernel
