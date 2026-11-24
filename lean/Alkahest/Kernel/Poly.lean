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

/-- Merge step of `Mono.mul` (outer recursion supplied as `rec`). -/
def mulAux (a : Nat × Nat) (m : Mono) (rec : Mono → Mono) : Mono → Mono
  | [] => a :: m
  | b :: n =>
    if a.1 < b.1 then a :: rec (b :: n)
    else if b.1 < a.1 then b :: mulAux a m rec n
    else (a.1, a.2 + b.2) :: rec n

/-- Product of monomials (merge of the sorted lists). -/
def mul : Mono → Mono → Mono
  | [], n => n
  | a :: m, n => mulAux a m (mul m) n

/-- Lexicographic order on dense exponent vectors. -/
def cmp : Mono → Mono → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | (i, e) :: m, (j, f) :: n =>
    if i < j then .gt
    else if j < i then .lt
    else if e < f then .lt
    else if f < e then .gt
    else cmp m n

theorem eq_of_cmp_eq : ∀ {m n : Mono}, cmp m n = .eq → m = n
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [cmp] at h
  | _ :: _, [], h => by simp [cmp] at h
  | (i, e) :: m, (j, f) :: n, h => by
    simp only [cmp] at h
    split_ifs at h with h1 h2 h3 h4
    have hij : i = j := by omega
    have hef : e = f := by omega
    subst hij hef
    rw [eq_of_cmp_eq h]

theorem eval_mulAux (v : List ℝ) (a : Nat × Nat) (m : Mono) (rec : Mono → Mono)
    (hrec : ∀ n, eval v (rec n) = eval v m * eval v n) :
    ∀ n, eval v (mulAux a m rec n) = eval v (a :: m) * eval v n
  | [] => by simp [mulAux, eval]
  | b :: n => by
    unfold mulAux
    split_ifs with h1 h2
    · simp only [eval, hrec]; ring
    · simp only [eval, eval_mulAux v a m rec hrec n]; ring
    · have : b.1 = a.1 := by omega
      obtain ⟨b1, b2⟩ := b
      simp only at this
      subst this
      simp only [eval, hrec, pow_add]; ring

theorem eval_mul (v : List ℝ) : ∀ m n, eval v (mul m n) = eval v m * eval v n
  | [], n => by simp [mul, eval]
  | a :: m, n => eval_mulAux v a m (mul m) (eval_mul v m) n

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
def addAux (t : Mono × ℚ) (p : Poly) (rec : Poly → Poly) : Poly → Poly
  | [] => t :: p
  | s :: q =>
    match Mono.cmp t.1 s.1 with
    | .gt => t :: rec (s :: q)
    | .lt => s :: addAux t p rec q
    | .eq => if t.2 + s.2 = 0 then rec q else (t.1, t.2 + s.2) :: rec q

/-- Sum (merge of sorted term lists, cancelling zero coefficients). -/
def add : Poly → Poly → Poly
  | [], q => q
  | t :: p, q => addAux t p (add p) q

/-- Scalar multiple. -/
def scale (c : ℚ) (p : Poly) : Poly :=
  if c = 0 then [] else p.map fun t => (t.1, c * t.2)

/-- Negation. -/
def neg (p : Poly) : Poly := p.map fun t => (t.1, -t.2)

/-- Difference. -/
def sub (p q : Poly) : Poly := add p (neg q)

/-- A term times a polynomial (order-preserving: `Mono.cmp` is a monomial order). -/
def mulTerm (t : Mono × ℚ) (p : Poly) : Poly := p.map fun s => (Mono.mul t.1 s.1, t.2 * s.2)

/-- Product. -/
def mul : Poly → Poly → Poly
  | [], _ => []
  | t :: p, q => add (mulTerm t q) (mul p q)

/-- Natural power. -/
def pow (p : Poly) : Nat → Poly
  | 0 => const 1
  | n + 1 => mul (pow p n) p

/-- A polynomial from an arbitrary list of terms (sums them, so any order works). -/
def ofTerms : List (Mono × ℚ) → Poly
  | [] => []
  | (m, c) :: ts => add (scale c [(m, 1)]) (ofTerms ts)

@[simp] theorem eval_nil (v : List ℝ) : eval v [] = 0 := rfl

theorem eval_cons (v : List ℝ) (m : Mono) (c : ℚ) (p : Poly) :
    eval v ((m, c) :: p) = (c : ℝ) * Mono.eval v m + eval v p := rfl

theorem eval_const (v : List ℝ) (c : ℚ) : eval v (const c) = c := by
  unfold const; split_ifs with h
  · simp [h]
  · simp [eval, Mono.eval]

theorem eval_atom (v : List ℝ) (i : Nat) : eval v (atom i) = v.getD i 0 := by
  simp [atom, eval, Mono.eval]

theorem eval_addAux (v : List ℝ) (t : Mono × ℚ) (p : Poly) (rec : Poly → Poly)
    (hrec : ∀ q, eval v (rec q) = eval v p + eval v q) :
    ∀ q, eval v (addAux t p rec q) = eval v (t :: p) + eval v q
  | [] => by simp [addAux]
  | s :: q => by
    obtain ⟨tm, tc⟩ := t
    obtain ⟨sm, sc⟩ := s
    unfold addAux
    dsimp only
    split
    · simp only [eval_cons, hrec]; ring
    · simp only [eval_cons, eval_addAux v (tm, tc) p rec hrec q]; ring
    · rename_i h
      have hm : tm = sm := Mono.eq_of_cmp_eq h
      subst hm
      split_ifs with hz
      · simp only [eval_cons, hrec]
        have : (tc : ℝ) + sc = 0 := by exact_mod_cast hz
        linear_combination (-Mono.eval v tm) * this
      · simp only [eval_cons, hrec]; push_cast; ring

theorem eval_add (v : List ℝ) : ∀ p q, eval v (add p q) = eval v p + eval v q
  | [], q => by simp [add]
  | t :: p, q => eval_addAux v t p (add p) (eval_add v p) q

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
    ∀ p, eval v (mulTerm (m, c) p) = (c : ℝ) * Mono.eval v m * eval v p
  | [] => by simp [mulTerm]
  | (n, d) :: p => by
    have := eval_mulTerm v m c p
    simp only [mulTerm, List.map_cons, eval_cons, Mono.eval_mul] at this ⊢
    rw [this]; push_cast; ring

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
