/-
Sparse polynomials over ℚ in numbered atoms, with kernel-reducible arithmetic.

A monomial is a list `(atom, exponent)` sorted by atom; a polynomial is a list
of `(monomial, coefficient)` sorted strictly decreasingly by `Mono.cmp` (the
lexicographic order on dense exponent vectors, which is a monomial order), with
no zero coefficients. Every operation is structural recursion on lists and
uses only `Nat` / `ℚ` arithmetic, so `decide +kernel` evaluates it with the
kernel's GMP-accelerated literals.

Soundness (`Poly.eval_add`, `Poly.eval_mul`, …) holds for *any* inputs: the
sortedness invariant is needed only for completeness (a zero polynomial is
recognised as `[]`), never for soundness.
-/
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.LinearCombination
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic.Ring

namespace Alkahest.Kernel

/-- `(atom, exponent)` pairs, sorted by atom. -/
abbrev Mono := List (Nat × Nat)
/-- `(monomial, coefficient)` pairs. -/
abbrev Poly := List (Mono × ℚ)

namespace Mono

/-- Value of a monomial under atom values `v`. -/
noncomputable def eval (v : List ℝ) : Mono → ℝ
  | [] => 1
  | (i, e) :: m => v.getD i 0 ^ e * eval v m

/-- Merge step of `Mono.mul` (outer recursion supplied as `rec`). Comparisons are
`Nat.blt` matched as `Bool`s: no `Decidable` instance to unfold in the kernel. -/
def mulAux (i e : Nat) (m : Mono) (rec : Mono → Mono) : Mono → Mono
  | [] => (i, e) :: m
  | (j, f) :: n =>
    match Nat.blt i j, Nat.blt j i with
    | true, _ => (i, e) :: rec ((j, f) :: n)
    | false, true => (j, f) :: mulAux i e m rec n
    | false, false => (i, e + f) :: rec n

/-- Product of monomials (merge of the sorted lists). -/
def mul : Mono → Mono → Mono
  | [], n => n
  | (i, e) :: m, n => mulAux i e m (mul m) n

/-- Lexicographic order on dense exponent vectors. -/
def cmp : Mono → Mono → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | (i, e) :: m, (j, f) :: n =>
    match Nat.blt i j, Nat.blt j i, Nat.blt e f, Nat.blt f e with
    | true, _, _, _ => .gt
    | false, true, _, _ => .lt
    | false, false, true, _ => .lt
    | false, false, false, true => .gt
    | false, false, false, false => cmp m n

theorem le_of_blt_false {a b : Nat} (h : Nat.blt a b = false) : b ≤ a := by
  by_contra hc
  have ht : Nat.blt a b = true := by rw [Nat.blt_eq]; omega
  rw [ht] at h
  exact Bool.noConfusion h

theorem eq_of_cmp_eq : ∀ {m n : Mono}, cmp m n = .eq → m = n
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [cmp] at h
  | _ :: _, [], h => by simp [cmp] at h
  | (i, e) :: m, (j, f) :: n, h => by
    unfold cmp at h
    generalize hb1 : Nat.blt i j = b1 at h
    generalize hb2 : Nat.blt j i = b2 at h
    generalize hb3 : Nat.blt e f = b3 at h
    generalize hb4 : Nat.blt f e = b4 at h
    cases b1 <;> cases b2 <;> cases b3 <;> cases b4 <;> dsimp only at h <;>
      try exact absurd h (by decide)
    have := le_of_blt_false hb1
    have := le_of_blt_false hb2
    have := le_of_blt_false hb3
    have := le_of_blt_false hb4
    rw [show i = j by omega, show e = f by omega, eq_of_cmp_eq h]

theorem eval_mulAux (v : List ℝ) (i e : Nat) (m : Mono) (rec : Mono → Mono)
    (hrec : ∀ n, eval v (rec n) = eval v m * eval v n) :
    ∀ n, eval v (mulAux i e m rec n) = eval v ((i, e) :: m) * eval v n
  | [] => by simp [mulAux, eval]
  | (j, f) :: n => by
    unfold mulAux
    cases h1 : Nat.blt i j <;> cases h2 : Nat.blt j i <;> dsimp only
    · have := le_of_blt_false h1
      have := le_of_blt_false h2
      have : j = i := by omega
      subst this
      simp only [eval, hrec, pow_add]; ring
    · simp only [eval, eval_mulAux v i e m rec hrec n]; ring
    · simp only [eval, hrec]; ring
    · simp only [eval, hrec]; ring

theorem eval_mul (v : List ℝ) : ∀ m n, eval v (mul m n) = eval v m * eval v n
  | [], n => by simp [mul, eval]
  | (i, e) :: m, n => eval_mulAux v i e m (mul m) (eval_mul v m) n

/-- Structural Boolean equality. -/
def beq : Mono → Mono → Bool
  | [], [] => true
  | (i, e) :: m, (j, f) :: n => Nat.beq i j && Nat.beq e f && beq m n
  | _, _ => false

theorem eq_of_beq : ∀ {m n : Mono}, beq m n = true → m = n
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [beq] at h
  | _ :: _, [], h => by simp [beq] at h
  | (i, e) :: m, (j, f) :: n, h => by
    simp only [beq, Bool.and_eq_true, Nat.beq_eq] at h
    rw [h.1.1, h.1.2, eq_of_beq h.2]

end Mono

namespace Poly

/-- Value of a polynomial under atom values `v`. -/
noncomputable def eval (v : List ℝ) : Poly → ℝ
  | [] => 0
  | (m, c) :: p => (c : ℝ) * Mono.eval v m + eval v p

/-- The constant `c`. -/
def const (c : ℚ) : Poly := if c = 0 then [] else [([], c)]

/-- The atom `i`. -/
def atom (i : Nat) : Poly := [([(i, 1)], 1)]

/-- Merge step of `Poly.add`. -/
def addAux (tm : Mono) (tc : ℚ) (p : Poly) (rec : Poly → Poly) : Poly → Poly
  | [] => (tm, tc) :: p
  | (sm, sc) :: q =>
    match Mono.cmp tm sm with
    | .gt => (tm, tc) :: rec ((sm, sc) :: q)
    | .lt => (sm, sc) :: addAux tm tc p rec q
    | .eq =>
      match (tc + sc).num with
      | 0 => rec q
      | _ => (tm, tc + sc) :: rec q

/-- Sum (merge of sorted term lists, cancelling zero coefficients). -/
def add : Poly → Poly → Poly
  | [], q => q
  | (tm, tc) :: p, q => addAux tm tc p (add p) q

/-- Scalar multiple. -/
def scale (c : ℚ) (p : Poly) : Poly :=
  if c = 0 then [] else p.map fun t => (t.1, c * t.2)

/-- Negation. -/
def neg (p : Poly) : Poly := p.map fun t => (t.1, -t.2)

/-- Difference. -/
def sub (p q : Poly) : Poly := add p (neg q)

/-- A term times a polynomial (order-preserving: `Mono.cmp` is a monomial order). -/
def mulTerm (tm : Mono) (tc : ℚ) : Poly → Poly
  | [] => []
  | (sm, sc) :: q => (Mono.mul tm sm, tc * sc) :: mulTerm tm tc q

/-- Product. -/
def mul : Poly → Poly → Poly
  | [], _ => []
  | (tm, tc) :: p, q => add (mulTerm tm tc q) (mul p q)

/-- Natural power. -/
def pow (p : Poly) : Nat → Poly
  | 0 => const 1
  | n + 1 => mul (pow p n) p

/-- A polynomial from an arbitrary list of terms (sums them, so any order works). -/
def ofTerms : List (Mono × ℚ) → Poly
  | [] => []
  | (m, c) :: ts => add (scale c [(m, 1)]) (ofTerms ts)

/-- Structural Boolean equality (coefficients compared by numerator and denominator). -/
def beq : Poly → Poly → Bool
  | [], [] => true
  | (m, c) :: p, (n, d) :: q =>
    Mono.beq m n && decide (c.num = d.num) && Nat.beq c.den d.den && beq p q
  | _, _ => false

theorem eq_of_beq : ∀ {p q : Poly}, beq p q = true → p = q
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [beq] at h
  | _ :: _, [], h => by simp [beq] at h
  | (m, c) :: p, (n, d) :: q, h => by
    simp only [beq, Bool.and_eq_true, decide_eq_true_eq, Nat.beq_eq] at h
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
    rw [Mono.eq_of_beq h1, eq_of_beq h4, Rat.ext h2 h3]

@[simp] theorem eval_nil (v : List ℝ) : eval v [] = 0 := rfl

theorem eval_cons (v : List ℝ) (m : Mono) (c : ℚ) (p : Poly) :
    eval v ((m, c) :: p) = (c : ℝ) * Mono.eval v m + eval v p := rfl

theorem eval_const (v : List ℝ) (c : ℚ) : eval v (const c) = c := by
  unfold const; split_ifs with h
  · simp [h]
  · simp [eval, Mono.eval]

theorem eval_atom (v : List ℝ) (i : Nat) : eval v (atom i) = v.getD i 0 := by
  simp [atom, eval, Mono.eval]

theorem eval_addAux (v : List ℝ) (tm : Mono) (tc : ℚ) (p : Poly) (rec : Poly → Poly)
    (hrec : ∀ q, eval v (rec q) = eval v p + eval v q) :
    ∀ q, eval v (addAux tm tc p rec q) = eval v ((tm, tc) :: p) + eval v q
  | [] => by simp [addAux]
  | (sm, sc) :: q => by
    unfold addAux
    split
    · simp only [eval_cons, hrec]; ring
    · simp only [eval_cons, eval_addAux v tm tc p rec hrec q]; ring
    · rename_i h
      have hm : tm = sm := Mono.eq_of_cmp_eq h
      subst hm
      split
      · rename_i hz
        have hz' : tc + sc = 0 := Rat.num_eq_zero.1 hz
        simp only [eval_cons, hrec]
        have : (tc : ℝ) + sc = 0 := by exact_mod_cast hz'
        linear_combination (-Mono.eval v tm) * this
      · simp only [eval_cons, hrec]; push_cast; ring

theorem eval_add (v : List ℝ) : ∀ p q, eval v (add p q) = eval v p + eval v q
  | [], q => by simp [add]
  | (tm, tc) :: p, q => eval_addAux v tm tc p (add p) (eval_add v p) q

theorem eval_map (v : List ℝ) (f : ℚ → ℚ) (k : ℝ) (hf : ∀ c, (f c : ℝ) = k * c) :
    ∀ p : Poly, eval v (p.map fun t => (t.1, f t.2)) = k * eval v p
  | [] => by simp
  | (m, c) :: p => by
    simp only [List.map_cons, eval_cons, hf, eval_map v f k hf p]; ring

theorem eval_scale (v : List ℝ) (c : ℚ) (p : Poly) : eval v (scale c p) = c * eval v p := by
  unfold scale; split_ifs with h
  · simp [h]
  · exact eval_map v (fun d => c * d) c (fun d => by push_cast; ring) p

theorem eval_neg (v : List ℝ) (p : Poly) : eval v (neg p) = -eval v p := by
  have := eval_map v (fun d => -d) (-1) (fun d => by push_cast; ring) p
  simpa [neg] using this

theorem eval_sub (v : List ℝ) (p q : Poly) : eval v (sub p q) = eval v p - eval v q := by
  simp [sub, eval_add, eval_neg]; ring

theorem eval_mulTerm (v : List ℝ) (m : Mono) (c : ℚ) :
    ∀ p, eval v (mulTerm m c p) = (c : ℝ) * Mono.eval v m * eval v p
  | [] => by simp [mulTerm]
  | (n, d) :: p => by
    simp only [mulTerm, eval_cons, Mono.eval_mul, eval_mulTerm v m c p]; push_cast; ring

theorem eval_mul (v : List ℝ) : ∀ p q, eval v (mul p q) = eval v p * eval v q
  | [], q => by simp [mul]
  | (m, c) :: p, q => by
    simp only [mul, eval_add, eval_mulTerm, eval_mul v p q, eval_cons]; ring

theorem eval_pow (v : List ℝ) (p : Poly) : ∀ n, eval v (pow p n) = eval v p ^ n
  | 0 => by simp [pow, eval_const]
  | n + 1 => by simp [pow, eval_mul, eval_pow v p n, pow_succ]

theorem eval_ofTerms (v : List ℝ) : ∀ ts : List (Mono × ℚ), eval v (ofTerms ts) = eval v ts
  | [] => rfl
  | (m, c) :: ts => by
    simp only [ofTerms, eval_add, eval_scale, eval_ofTerms v ts, eval_cons, eval_nil]
    ring

end Poly

end Alkahest.Kernel
