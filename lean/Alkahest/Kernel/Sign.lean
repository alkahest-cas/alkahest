/-
Side-condition checker: decides `e ≠ 0`, `0 < e`, `0 ≤ e` from a list of
hypotheses by structural sign propagation (the reflective analogue of
`positivity`), with one soundness theorem.
-/
import Alkahest.Kernel.Expr
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

namespace Alkahest.Kernel

/-- Signs established for an expression. -/
structure Sign where
  pos : Bool
  nonneg : Bool
  ne : Bool

/-- Strengthen a sign with what the hypotheses say about `e` itself. -/
def Sign.close (H : List Cond) (e : AExpr) (s : Sign) : Sign :=
  let p := s.pos || H.contains (.pos e)
  { pos := p
    nonneg := p || s.nonneg || H.contains (.nonneg e)
    ne := p || s.ne || H.contains (.ne e) }

/-- Signs of an expression under hypotheses `H`. -/
def signOf (H : List Cond) : AExpr → Sign
  | .var i => Sign.close H (.var i) ⟨false, false, false⟩
  | .zero => ⟨false, true, false⟩
  | .one => ⟨true, true, true⟩
  | .lit n => ⟨0 < n, true, 0 < n⟩
  | .pi => ⟨true, true, true⟩
  | .add a b =>
    let sa := signOf H a
    let sb := signOf H b
    Sign.close H (.add a b)
      ⟨(sa.pos && sb.nonneg) || (sa.nonneg && sb.pos), sa.nonneg && sb.nonneg, false⟩
  | .sub a b =>
    let l := H.contains (.lt b a)
    Sign.close H (.sub a b) ⟨l, l, l⟩
  | .mul a b =>
    let sa := signOf H a
    let sb := signOf H b
    Sign.close H (.mul a b) ⟨sa.pos && sb.pos, sa.nonneg && sb.nonneg, sa.ne && sb.ne⟩
  | .div a b =>
    let sa := signOf H a
    let sb := signOf H b
    Sign.close H (.div a b) ⟨sa.pos && sb.pos, sa.nonneg && sb.nonneg, sa.ne && sb.ne⟩
  | .neg a =>
    let sa := signOf H a
    Sign.close H (.neg a) ⟨false, false, sa.ne⟩
  | .inv a =>
    let sa := signOf H a
    Sign.close H (.inv a) ⟨sa.pos, sa.nonneg, sa.ne⟩
  | .pow a n =>
    let sa := signOf H a
    let ev := n % 2 == 0
    Sign.close H (.pow a n)
      ⟨n == 0 || sa.pos || (ev && sa.ne), n == 0 || ev || sa.nonneg, n == 0 || sa.ne⟩
  | .app f a =>
    let sa := signOf H a
    match f with
    | .exp => ⟨true, true, true⟩
    | .cosh => ⟨true, true, true⟩
    | .sqrt => Sign.close H (.app .sqrt a) ⟨sa.pos, true, sa.pos⟩
    | g => Sign.close H (.app g a) ⟨false, false, false⟩

/-- Is a side condition implied by the hypotheses? -/
def implied (H : List Cond) : Cond → Bool
  | .ne e => (signOf H e).ne
  | .pos e => (signOf H e).pos
  | .nonneg e => (signOf H e).nonneg
  | .lt a b => H.contains (.lt a b) || (signOf H (.sub b a)).pos

/-- What a `Sign` claims about the value `t`. -/
def SignOk (t : ℝ) (s : Sign) : Prop :=
  (s.pos = true → 0 < t) ∧ (s.nonneg = true → 0 ≤ t) ∧ (s.ne = true → t ≠ 0)

theorem mem_of_contains {H : List Cond} {c : Cond} (h : H.contains c = true) : c ∈ H := by
  simpa using h

theorem close_ok {ρ : List ℝ} {H : List Cond} (hH : HoldsAll ρ H) {e : AExpr} {s : Sign}
    (hs : SignOk (e.eval ρ) s) : SignOk (e.eval ρ) (Sign.close H e s) := by
  obtain ⟨h1, h2, h3⟩ := hs
  have hp : (s.pos || H.contains (.pos e)) = true → 0 < e.eval ρ := by
    intro h
    rcases Bool.or_eq_true_iff.1 h with h | h
    · exact h1 h
    · exact (hH.mem _ (mem_of_contains h) : Cond.holds ρ (.pos e))
  refine ⟨hp, ?_, ?_⟩
  · intro h
    simp only [Sign.close, Bool.or_eq_true] at h
    rcases h with (h | h) | h
    · exact (hp (by simpa using h)).le
    · exact h2 h
    · exact (hH.mem _ (mem_of_contains h) : Cond.holds ρ (.nonneg e))
  · intro h
    simp only [Sign.close, Bool.or_eq_true] at h
    rcases h with (h | h) | h
    · exact (hp (by simpa using h)).ne'
    · exact h3 h
    · exact (hH.mem _ (mem_of_contains h) : Cond.holds ρ (.ne e))

theorem signOf_sound {ρ : List ℝ} {H : List Cond} (hH : HoldsAll ρ H) :
    ∀ e : AExpr, SignOk (e.eval ρ) (signOf H e)
  | .var i => close_ok hH ⟨by simp, by simp, by simp⟩
  | .zero => ⟨by simp [signOf], by simp [AExpr.eval], by simp [signOf]⟩
  | .one => ⟨by simp [AExpr.eval], by simp [AExpr.eval], by simp [AExpr.eval]⟩
  | .lit n => ⟨fun h => by simp [signOf] at h; simp [AExpr.eval]; exact_mod_cast h,
      fun _ => by simp [AExpr.eval],
      fun h => by simp [signOf] at h; simp [AExpr.eval]; omega⟩
  | .pi => ⟨fun _ => Real.pi_pos, fun _ => Real.pi_pos.le, fun _ => Real.pi_ne_zero⟩
  | .add a b => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    obtain ⟨b1, b2, b3⟩ := signOf_sound hH b
    refine close_ok hH ⟨?_, ?_, by simp⟩
    · intro h
      simp only [Bool.or_eq_true, Bool.and_eq_true] at h
      show 0 < a.eval ρ + b.eval ρ
      rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact add_pos_of_pos_of_nonneg (a1 h1) (b2 h2)
      · exact add_pos_of_nonneg_of_pos (a2 h1) (b1 h2)
    · intro h
      simp only [Bool.and_eq_true] at h
      exact add_nonneg (a2 h.1) (b2 h.2)
  | .sub a b => by
    refine close_ok hH ?_
    have key : H.contains (.lt b a) = true → 0 < (AExpr.sub a b).eval ρ := by
      intro h
      have := (hH.mem _ (mem_of_contains h) : Cond.holds ρ (.lt b a))
      show 0 < a.eval ρ - b.eval ρ
      simp only [Cond.holds] at this
      linarith
    exact ⟨key, fun h => (key h).le, fun h => (key h).ne'⟩
  | .mul a b => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    obtain ⟨b1, b2, b3⟩ := signOf_sound hH b
    refine close_ok hH ⟨?_, ?_, ?_⟩ <;> intro h <;> simp only [Bool.and_eq_true] at h
    · exact mul_pos (a1 h.1) (b1 h.2)
    · exact mul_nonneg (a2 h.1) (b2 h.2)
    · exact mul_ne_zero (a3 h.1) (b3 h.2)
  | .div a b => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    obtain ⟨b1, b2, b3⟩ := signOf_sound hH b
    refine close_ok hH ⟨?_, ?_, ?_⟩ <;> intro h <;> simp only [Bool.and_eq_true] at h
    · exact div_pos (a1 h.1) (b1 h.2)
    · exact div_nonneg (a2 h.1) (b2 h.2)
    · exact div_ne_zero (a3 h.1) (b3 h.2)
  | .neg a => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    refine close_ok hH ⟨by simp, by simp, fun h => ?_⟩
    exact neg_ne_zero.2 (a3 h)
  | .inv a => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    refine close_ok hH ⟨fun h => inv_pos.2 (a1 h), fun h => inv_nonneg.2 (a2 h),
      fun h => inv_ne_zero (a3 h)⟩
  | .pow a n => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    refine close_ok hH ⟨?_, ?_, ?_⟩ <;> intro h <;>
      simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at h
    · show 0 < a.eval ρ ^ n
      rcases h with (h | h) | ⟨h1, h2⟩
      · subst h; simp
      · exact pow_pos (a1 h) n
      · obtain ⟨k, hk⟩ : ∃ k, n = 2 * k := ⟨n / 2, by omega⟩
        subst hk; rw [pow_mul]
        exact pow_pos (lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (a3 h2)))) k
    · show 0 ≤ a.eval ρ ^ n
      rcases h with (h | h) | h
      · subst h; simp
      · obtain ⟨k, hk⟩ : ∃ k, n = 2 * k := ⟨n / 2, by omega⟩
        subst hk; rw [pow_mul]; positivity
      · exact pow_nonneg (a2 h) n
    · show a.eval ρ ^ n ≠ 0
      rcases h with h | h
      · subst h; simp
      · exact pow_ne_zero n (a3 h)
  | .app f a => by
    obtain ⟨a1, a2, a3⟩ := signOf_sound hH a
    cases f with
    | exp =>
      exact ⟨fun _ => Real.exp_pos _, fun _ => (Real.exp_pos _).le, fun _ => (Real.exp_pos _).ne'⟩
    | cosh =>
      exact ⟨fun _ => Real.cosh_pos _, fun _ => (Real.cosh_pos _).le,
        fun _ => (Real.cosh_pos _).ne'⟩
    | sqrt =>
      refine close_ok hH ⟨fun h => Real.sqrt_pos.2 (a1 h), fun _ => Real.sqrt_nonneg _,
        fun h => (Real.sqrt_pos.2 (a1 h)).ne'⟩
    | _ => exact close_ok hH ⟨by simp, by simp, by simp⟩

theorem implied_sound {ρ : List ℝ} {H : List Cond} (hH : HoldsAll ρ H) :
    ∀ {c : Cond}, implied H c = true → c.holds ρ
  | .ne e, h => (signOf_sound hH e).2.2 h
  | .pos e, h => (signOf_sound hH e).1 h
  | .nonneg e, h => (signOf_sound hH e).2.1 h
  | .lt a b, h => by
    simp only [implied, Bool.or_eq_true] at h
    rcases h with h | h
    · exact hH.mem _ (mem_of_contains h)
    · have := (signOf_sound hH (.sub b a)).1 h
      show a.eval ρ < b.eval ρ
      simp only [AExpr.eval] at this
      linarith

/-- All conditions of a list are implied by the hypotheses. -/
def condsOk (H : List Cond) (cs : List Cond) : Bool := cs.all (implied H)

theorem condsOk_sound {ρ : List ℝ} {H cs : List Cond} (hH : HoldsAll ρ H)
    (h : condsOk H cs = true) : ∀ c ∈ cs, c.holds ρ := by
  intro c hc
  simp only [condsOk, List.all_eq_true] at h
  exact implied_sound hH (h c hc)

end Alkahest.Kernel
