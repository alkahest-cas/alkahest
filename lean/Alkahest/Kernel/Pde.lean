/-
PDE residual certificates: an operator in the jet `(u, Δu, ∇u)` of an explicit
function `u` of `d` coordinates vanishes identically.

The operator is an `AExpr` `op` over variables `0 … d-1` (the coordinates),
`d` (`u`), `d+1` (`Δu`) and `d+2+i` (`∂u/∂xᵢ`). The checker computes the
jets symbolically with the kernel's own derivative `D`, substitutes them
(`AExpr.bind`), and decides that the result is zero with `checkEq`;
`pde_sound` turns acceptance into the statement about Mathlib's `deriv`:

  `op.eval (ρ ++ u(ρ) :: Σᵢ ∂ᵢ²u(ρ) :: [∂₀u(ρ), …]) = 0`

for every point `ρ ∈ ℝ^d`, where `∂ᵢ` is `deriv` of the restriction of `u`
to the line through `ρ` parallel to the `i`-th axis. Differentiability side
conditions must be decidable with no hypotheses (they hold on all of `ℝ^d`):
the certificate is for solutions defined everywhere, such as the polynomial,
exponential and trigonometric closed forms of FEX.
-/
import Alkahest.Kernel.Check

namespace Alkahest.Kernel

/-- Substitute `σ.getD j zero` for `var j`. -/
def AExpr.bind (σ : List AExpr) : AExpr → AExpr
  | .var j => σ.getD j .zero
  | .zero => .zero
  | .one => .one
  | .lit n => .lit n
  | .pi => .pi
  | .add a b => .add (a.bind σ) (b.bind σ)
  | .sub a b => .sub (a.bind σ) (b.bind σ)
  | .mul a b => .mul (a.bind σ) (b.bind σ)
  | .div a b => .div (a.bind σ) (b.bind σ)
  | .neg a => .neg (a.bind σ)
  | .inv a => .inv (a.bind σ)
  | .pow a n => .pow (a.bind σ) n
  | .app f a => .app f (a.bind σ)

theorem getD_map_eval (ρ : List ℝ) (σ : List AExpr) (j : Nat) :
    (σ.map (fun e => e.eval ρ)).getD j 0 = (σ.getD j .zero).eval ρ := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  cases σ[j]? <;> simp [AExpr.eval]

theorem AExpr.eval_bind (ρ : List ℝ) (σ : List AExpr) :
    ∀ e : AExpr, (e.bind σ).eval ρ = e.eval (σ.map (fun e => e.eval ρ))
  | .var j => by simp only [AExpr.bind, AExpr.eval]; rw [getD_map_eval]
  | .zero => rfl
  | .one => rfl
  | .lit _ => rfl
  | .pi => rfl
  | .add a b => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a, eval_bind ρ σ b]
  | .sub a b => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a, eval_bind ρ σ b]
  | .mul a b => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a, eval_bind ρ σ b]
  | .div a b => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a, eval_bind ρ σ b]
  | .neg a => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a]
  | .inv a => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a]
  | .pow a n => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a]
  | .app f a => by simp only [AExpr.bind, AExpr.eval, eval_bind ρ σ a]

/-- `Σ_{i<n} D i (D i u)`, nested to the left: `((0 + t₀) + t₁) + …`. -/
def lapE (u : AExpr) : Nat → AExpr
  | 0 => .zero
  | n + 1 => .add (lapE u n) (D n (D n u))

/-- `[D 0 u, …, D (n-1) u]`. -/
def gradE (u : AExpr) (n : Nat) : List AExpr := (List.range n).map fun i => D i u

/-- `[var 0, …, var (n-1)]`. -/
def varsE (n : Nat) : List AExpr := (List.range n).map AExpr.var

/-- The differentiability conditions of `u` and of every first partial. -/
def pdeConds (u : AExpr) (n : Nat) : List Cond :=
  dconds u ++ ((List.range n).map fun i => dconds (D i u)).flatten

/-- The Laplacian of `u` at `ρ`, as Mathlib second derivatives along the
coordinate lines, nested to the left like `lapE`. -/
noncomputable def lapR (u : AExpr) (ρ : List ℝ) : Nat → ℝ
  | 0 => 0
  | n + 1 => lapR u ρ n + deriv (deriv (fun y => u.eval (ρ.set n y))) (ρ.getD n 0)

/-- The gradient of `u` at `ρ`, one `deriv` per coordinate. -/
noncomputable def gradR (u : AExpr) (ρ : List ℝ) (n : Nat) : List ℝ :=
  (List.range n).map fun i => deriv (fun y => u.eval (ρ.set i y)) (ρ.getD i 0)

/-- The jet the operator is evaluated on: coordinates, `u`, `Δu`, `∇u`. -/
noncomputable def jetR (u : AExpr) (ρ : List ℝ) (n : Nat) : List ℝ :=
  ρ ++ u.eval ρ :: lapR u ρ n :: gradR u ρ n

/-- `op` vanishes on the jet of `u` in `n` coordinates. -/
def checkPde (u op : AExpr) (n : Nat) (w : Witness) : Bool :=
  condsOk [] (pdeConds u n) &&
    checkEq [] (op.bind (varsE n ++ u :: lapE u n :: gradE u n)) .zero w

theorem getD_of_lt (ρ : List ℝ) (i : Nat) (hi : i < ρ.length) : ρ.getD i 0 = ρ[i] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]

theorem set_getD_self (ρ : List ℝ) (i : Nat) (hi : i < ρ.length) : ρ.set i (ρ.getD i 0) = ρ := by
  rw [getD_of_lt ρ i hi]
  exact List.set_getElem_self hi

theorem pdeConds_all {u : AExpr} {n : Nat} (h : condsOk [] (pdeConds u n) = true) (ρ : List ℝ) :
    (∀ c ∈ dconds u, c.holds ρ) ∧ ∀ i < n, ∀ c ∈ dconds (D i u), c.holds ρ := by
  have hall := condsOk_sound (ρ := ρ) (H := []) trivial h
  refine ⟨fun c hc => hall c (by simp [pdeConds, hc]), fun i hi c hc => hall c ?_⟩
  simp only [pdeConds, List.mem_append, List.mem_flatten, List.mem_map, List.mem_range]
  exact Or.inr ⟨_, ⟨i, hi, rfl⟩, hc⟩

/-- First partials: `deriv` along coordinate `i` is `D i u`, at every point of
the line. -/
theorem deriv_line {u : AExpr} {n : Nat} (h : condsOk [] (pdeConds u n) = true) (ρ : List ℝ)
    (i : Nat) (hi : i < ρ.length) :
    deriv (fun y => u.eval (ρ.set i y)) = fun y => (D i u).eval (ρ.set i y) := by
  funext y
  exact (D_hasDerivAt i ρ hi y u ((pdeConds_all h (ρ.set i y)).1)).deriv

theorem deriv2_line {u : AExpr} {n : Nat} (h : condsOk [] (pdeConds u n) = true) (ρ : List ℝ)
    (i : Nat) (hi : i < ρ.length) (hin : i < n) :
    deriv (deriv (fun y => u.eval (ρ.set i y))) (ρ.getD i 0) = (D i (D i u)).eval ρ := by
  rw [deriv_line h ρ i hi]
  have := (D_hasDerivAt i ρ hi (ρ.getD i 0) (D i u)
    ((pdeConds_all h (ρ.set i (ρ.getD i 0))).2 i hin)).deriv
  rw [this, set_getD_self ρ i hi]

theorem lapR_eq {u : AExpr} {n : Nat} (h : condsOk [] (pdeConds u n) = true) (ρ : List ℝ)
    (hn : ρ.length = n) : ∀ m ≤ n, lapR u ρ m = (lapE u m).eval ρ
  | 0, _ => rfl
  | m + 1, hm => by
    simp only [lapR, lapE, AExpr.eval]
    rw [lapR_eq h ρ hn m (Nat.le_of_succ_le hm), deriv2_line h ρ m (by omega) (by omega)]

theorem gradR_eq {u : AExpr} {n : Nat} (h : condsOk [] (pdeConds u n) = true) (ρ : List ℝ)
    (hn : ρ.length = n) : gradR u ρ n = (gradE u n).map fun e => e.eval ρ := by
  simp only [gradR, gradE, List.map_map]
  apply List.map_congr_left
  intro i hi
  have hi' : i < n := List.mem_range.1 hi
  rw [deriv_line h ρ i (by omega)]
  beta_reduce
  rw [set_getD_self ρ i (by omega)]
  rfl

theorem varsE_eval (ρ : List ℝ) (n : Nat) (hn : ρ.length = n) :
    (varsE n).map (fun e => e.eval ρ) = ρ := by
  apply List.ext_getElem
  · simp [varsE, hn]
  · intro i h1 h2
    simp [varsE, AExpr.eval, List.getElem?_eq_getElem h2]

/-- **PDE residual certificates.** If the checker accepts, the operator
vanishes on the jet of `u` at every point `ρ` of `ℝⁿ`. -/
theorem pde_sound {u op : AExpr} {n : Nat} {w : Witness} (h : checkPde u op n w = true)
    (ρ : List ℝ) (hn : ρ.length = n) : op.eval (jetR u ρ n) = 0 := by
  unfold checkPde at h
  simp only [Bool.and_eq_true] at h
  have hz := checkEq_sound h.2 ρ trivial
  rw [AExpr.eval_bind] at hz
  have hj : (varsE n ++ u :: lapE u n :: gradE u n).map (fun e => e.eval ρ) = jetR u ρ n := by
    simp only [List.map_append, List.map_cons, jetR, varsE_eval ρ n hn,
      lapR_eq h.1 ρ hn n le_rfl, gradR_eq h.1 ρ hn]
  rw [hj] at hz
  simpa [AExpr.eval] using hz

end Alkahest.Kernel
