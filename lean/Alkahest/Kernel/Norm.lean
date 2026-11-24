/-
Polynomial normalisation of `AExpr` over ℚ in atoms.

The certificate supplies an *atom table*: a list of `Atom`s (a head — a
variable, `π`, a unary function, or an inverse — and an argument). The kernel
normalises each atom's argument against the earlier atoms (`buildNT`), and
then normalises expressions against the table: a function application
`f a` becomes the atom whose head is `f` and whose normalised argument equals
the normal form of `a`. So two spellings of one atom (`sin (x + 1)` and
`sin (1 + x)`) are the same atom, without any trusted matching.

Inverses distribute over monomials (`(c·a²)⁻¹ = c⁻¹·(a⁻¹)²`, `(a⁻¹)⁻¹ = a`) —
true in every field with `0⁻¹ = 0`, so unconditionally; the inverse of any
other polynomial is one atom. Facts such as `u · u⁻¹ = 1` (which need `u ≠ 0`)
enter only as relations (`Alkahest.Kernel.Rel`).
-/
import Alkahest.Kernel.Expr
import Alkahest.Kernel.Poly

namespace Alkahest.Kernel

/-- What an atom is: a variable, `π`, a unary function of its argument, or an inverse. -/
inductive Head where
  | var (i : Nat)
  | pi
  | fn (f : Fn)
  | inv
  deriving DecidableEq, Repr

/-- An atom-table entry. For `var` / `pi` heads the argument is ignored. -/
structure Atom where
  head : Head
  arg : AExpr
  deriving DecidableEq, Repr

/-- The value of a head applied to (the value of) its argument. -/
noncomputable def Head.apply (ρ : List ℝ) : Head → ℝ → ℝ
  | .var i => fun _ => ρ.getD i 0
  | .pi => fun _ => Real.pi
  | .fn f => f.eval
  | .inv => fun t => t⁻¹

/-- A normalised atom table: heads with normalised arguments. -/
abbrev NTable := List (Head × Poly)

/-- Index of the entry `(h, p)`, searching from position `k`. -/
def find (h : Head) (p : Poly) : NTable → Nat → Option Nat
  | [], _ => none
  | (h', p') :: t, k => if h' = h ∧ p' = p then some k else find h p t (k + 1)

/-- The atom `(h, p)` as a polynomial, if it is in the table. -/
def atomOf (NT : NTable) (h : Head) (p : Poly) : Option Poly :=
  (find h p NT 0).map Poly.atom

/-- `log |a|` and `log a` are one atom (`Real.log_abs`). -/
def canonFn : Fn → Fn
  | .logabs => .log
  | f => f

/-- `(atom i)⁻¹`: the inner polynomial for an inverse atom, else the inverse atom. -/
def invAtom (NT : NTable) (i : Nat) : Option Poly :=
  match NT[i]? with
  | some (.inv, q) => some q
  | _ => atomOf NT .inv (Poly.atom i)

/-- The inverse of a monomial. -/
def invMono (NT : NTable) : Mono → Option Poly
  | [] => some (Poly.const 1)
  | (i, e) :: m =>
    match invMono NT m, invAtom NT i with
    | some r, some w => some (Poly.mul (Poly.pow w e) r)
    | _, _ => none

/-- The inverse of a normalised polynomial. -/
def normInv (NT : NTable) (p : Poly) : Option Poly :=
  match p with
  | [] => some []
  | [(m, c)] => (invMono NT m).map (Poly.scale c⁻¹)
  | _ => atomOf NT .inv p

/-- Normal form of an expression against a normalised table (`none`: an atom is missing). -/
def norm (NT : NTable) : AExpr → Option Poly
  | .var i => atomOf NT (.var i) []
  | .zero => some []
  | .one => some (Poly.const 1)
  | .lit n => some (Poly.const n)
  | .pi => atomOf NT .pi []
  | .add a b =>
    match norm NT a, norm NT b with
    | some p, some q => some (Poly.add p q)
    | _, _ => none
  | .sub a b =>
    match norm NT a, norm NT b with
    | some p, some q => some (Poly.sub p q)
    | _, _ => none
  | .mul a b =>
    match norm NT a, norm NT b with
    | some p, some q => some (Poly.mul p q)
    | _, _ => none
  | .div a b =>
    match norm NT a, norm NT b with
    | some p, some q =>
      match normInv NT q with
      | some r => some (Poly.mul p r)
      | none => none
    | _, _ => none
  | .neg a =>
    match norm NT a with
    | some p => some (Poly.neg p)
    | none => none
  | .inv a =>
    match norm NT a with
    | some p => normInv NT p
    | none => none
  | .pow a n =>
    match norm NT a with
    | some p => some (Poly.pow p n)
    | none => none
  | .app f a =>
    match norm NT a with
    | some p => atomOf NT (.fn (canonFn f)) p
    | none => none

/-- Normalise the table: each argument against the entries before it. -/
def buildNT.go : List Atom → NTable → Option NTable
  | [], acc => some acc
  | a :: t, acc =>
    match a.head with
    | .var i => buildNT.go t (acc ++ [(.var i, [])])
    | .pi => buildNT.go t (acc ++ [(.pi, [])])
    | h =>
      match norm acc a.arg with
      | some p => buildNT.go t (acc ++ [(h, p)])
      | none => none

/-- The normalised table of an atom list. -/
def buildNT (tbl : List Atom) : Option NTable := buildNT.go tbl []

/-- The values of the atoms at the point `ρ`. -/
noncomputable def atomVals (ρ : List ℝ) (tbl : List Atom) : List ℝ :=
  tbl.map fun a => a.head.apply ρ (a.arg.eval ρ)

/-- A normalised table is *good* for atom values `vals` at `ρ`. -/
def Good (ρ vals : List ℝ) (NT : NTable) : Prop :=
  ∀ k h p, NT[k]? = some (h, p) → vals.getD k 0 = h.apply ρ (p.eval vals)

theorem find_spec {h : Head} {p : Poly} :
    ∀ {NT : NTable} {k0 k : Nat}, find h p NT k0 = some k → k0 ≤ k ∧ NT[k - k0]? = some (h, p)
  | [], _, _, hf => by simp [find] at hf
  | (h', p') :: t, k0, k, hf => by
    unfold find at hf
    split_ifs at hf with hc
    · obtain ⟨rfl, rfl⟩ := hc
      simp at hf; subst hf; simp
    · obtain ⟨h1, h2⟩ := find_spec hf
      refine ⟨by omega, ?_⟩
      have : k - k0 = (k - (k0 + 1)) + 1 := by omega
      rw [this]; simpa using h2

theorem atomOf_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) {h : Head} {p r : Poly}
    (hr : atomOf NT h p = some r) : r.eval vals = h.apply ρ (p.eval vals) := by
  unfold atomOf at hr
  cases hk : find h p NT 0 with
  | none => simp [hk] at hr
  | some k =>
    simp [hk] at hr; subst hr
    obtain ⟨_, h2⟩ := find_spec hk
    simp only [Nat.sub_zero] at h2
    rw [Poly.eval_atom]; exact hG k h p h2

theorem invAtom_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) {i : Nat} {r : Poly}
    (hr : invAtom NT i = some r) : r.eval vals = (vals.getD i 0)⁻¹ := by
  unfold invAtom at hr
  split at hr
  · rename_i q hq
    simp at hr; subst hr
    rw [hG i .inv _ hq]; simp [Head.apply]
  · rw [atomOf_sound hG hr]; simp [Head.apply, Poly.eval_atom]

theorem invMono_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) :
    ∀ {m : Mono} {r : Poly}, invMono NT m = some r → r.eval vals = (Mono.eval vals m)⁻¹
  | [], r, hr => by simp [invMono] at hr; subst hr; simp [Poly.eval_const, Mono.eval]
  | (i, e) :: m, r, hr => by
    unfold invMono at hr
    split at hr
    · rename_i r' w h1 h2
      simp at hr; subst hr
      rw [Poly.eval_mul, Poly.eval_pow, invAtom_sound hG h2, invMono_sound hG h1]
      simp [Mono.eval]; ring
    · simp at hr

theorem normInv_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) {p r : Poly}
    (hr : normInv NT p = some r) : r.eval vals = (p.eval vals)⁻¹ := by
  unfold normInv at hr
  split at hr
  · simp at hr; subst hr; simp
  · rename_i m c
    cases hm : invMono NT m with
    | none => simp [hm] at hr
    | some s =>
      simp [hm] at hr; subst hr
      rw [Poly.eval_scale, invMono_sound hG hm]
      simp [Poly.eval]; ring
  · rw [atomOf_sound hG hr]; rfl

theorem norm_sound {ρ vals : List ℝ} {NT : NTable} (hG : Good ρ vals NT) :
    ∀ {e : AExpr} {p : Poly}, norm NT e = some p → p.eval vals = e.eval ρ
  | .var i, p, h => by
    unfold norm at h; rw [atomOf_sound hG h]; rfl
  | .zero, p, h => by simp [norm] at h; subst h; simp [AExpr.eval]
  | .one, p, h => by simp [norm] at h; subst h; simp [AExpr.eval, Poly.eval_const]
  | .lit n, p, h => by simp [norm] at h; subst h; simp [AExpr.eval, Poly.eval_const]
  | .pi, p, h => by
    unfold norm at h; rw [atomOf_sound hG h]; rfl
  | .add a b, p, h => by
    unfold norm at h
    split at h
    · rename_i pa pb ha hb
      simp at h; subst h
      rw [Poly.eval_add, norm_sound hG ha, norm_sound hG hb]; rfl
    · simp at h
  | .sub a b, p, h => by
    unfold norm at h
    split at h
    · rename_i pa pb ha hb
      simp at h; subst h
      rw [Poly.eval_sub, norm_sound hG ha, norm_sound hG hb]; rfl
    · simp at h
  | .mul a b, p, h => by
    unfold norm at h
    split at h
    · rename_i pa pb ha hb
      simp at h; subst h
      rw [Poly.eval_mul, norm_sound hG ha, norm_sound hG hb]; rfl
    · simp at h
  | .div a b, p, h => by
    unfold norm at h
    split at h
    · rename_i pa pb ha hb
      split at h
      · rename_i r hr
        simp at h; subst h
        rw [Poly.eval_mul, norm_sound hG ha, normInv_sound hG hr, norm_sound hG hb]
        simp [AExpr.eval, div_eq_mul_inv]
      · simp at h
    · simp at h
  | .neg a, p, h => by
    unfold norm at h
    split at h
    · rename_i pa ha
      simp at h; subst h
      rw [Poly.eval_neg, norm_sound hG ha]; rfl
    · simp at h
  | .inv a, p, h => by
    unfold norm at h
    split at h
    · rename_i pa ha
      rw [normInv_sound hG h, norm_sound hG ha]; rfl
    · simp at h
  | .pow a n, p, h => by
    unfold norm at h
    split at h
    · rename_i pa ha
      simp at h; subst h
      rw [Poly.eval_pow, norm_sound hG ha]; rfl
    · simp at h
  | .app f a, p, h => by
    unfold norm at h
    split at h
    · rename_i pa ha
      rw [atomOf_sound hG h, norm_sound hG ha]
      cases f <;> simp [canonFn, Head.apply, AExpr.eval, Fn.eval, Real.log_abs]
    · simp at h

/-- Invariant of `buildNT.go`: entry `k` of `acc` denotes atom `k` of `tbl`. -/
def TableInv (ρ : List ℝ) (tbl : List Atom) (acc : NTable) : Prop :=
  ∀ (k : Nat) (h : Head) (p : Poly), acc[k]? = some (h, p) →
    ∃ a : Atom, tbl[k]? = some a ∧ h.apply ρ (p.eval (atomVals ρ tbl)) = a.head.apply ρ (a.arg.eval ρ)

theorem good_of_inv {ρ : List ℝ} {tbl : List Atom} {acc : NTable} (hI : TableInv ρ tbl acc) :
    Good ρ (atomVals ρ tbl) acc := by
  intro k h p hk
  obtain ⟨a, ha, he⟩ := hI k h p hk
  rw [he]
  simp [atomVals, List.getD_eq_getElem?_getD, ha]

theorem buildNT_go_good {ρ : List ℝ} {tbl : List Atom} :
    ∀ (rest : List Atom) (acc : NTable) {NT : NTable},
      (∃ done, tbl = done ++ rest ∧ done.length = acc.length) → TableInv ρ tbl acc →
      buildNT.go rest acc = some NT → Good ρ (atomVals ρ tbl) NT
  | [], acc, NT, _, hI, h => by
    simp [buildNT.go] at h; subst h; exact good_of_inv hI
  | a :: t, acc, NT, ⟨done, hd, hl⟩, hI, h => by
    have hk : tbl[acc.length]? = some a := by
      rw [hd, ← hl]; simp
    -- extending the invariant by one entry `(h', p)` that denotes `a`
    have ext : ∀ (h' : Head) (p : Poly),
        h'.apply ρ (p.eval (atomVals ρ tbl)) = a.head.apply ρ (a.arg.eval ρ) →
        TableInv ρ tbl (acc ++ [(h', p)]) := by
      intro h' p hp
      unfold TableInv
      intro k hh pp hkk
      rcases Nat.lt_or_ge k acc.length with hlt | hge
      · rw [List.getElem?_append_left hlt] at hkk
        exact hI k hh pp hkk
      · rw [List.getElem?_append_right hge] at hkk
        have : k - acc.length = 0 := by
          by_contra hne
          obtain ⟨j, hj⟩ : ∃ j, k - acc.length = j + 1 := ⟨k - acc.length - 1, by omega⟩
          rw [hj] at hkk
          simp at hkk
        rw [this] at hkk
        simp at hkk
        obtain ⟨rfl, rfl⟩ := hkk
        have hk' : k = acc.length := by omega
        subst hk'
        exact ⟨a, hk, hp⟩
    unfold buildNT.go at h
    split at h
    · rename_i i hh
      refine buildNT_go_good t _ ⟨done ++ [a], by simp [hd], by simp [hl]⟩ ?_ h
      exact ext _ _ (by simp [hh, Head.apply])
    · rename_i hh
      refine buildNT_go_good t _ ⟨done ++ [a], by simp [hd], by simp [hl]⟩ ?_ h
      exact ext _ _ (by simp [hh, Head.apply])
    · rename_i hv hp
      split at h
      · rename_i p hn
        refine buildNT_go_good t _ ⟨done ++ [a], by simp [hd], by simp [hl]⟩ ?_ h
        exact ext _ _ (by rw [norm_sound (good_of_inv hI) hn])
      · simp at h

theorem buildNT_good {ρ : List ℝ} {tbl : List Atom} {NT : NTable} (h : buildNT tbl = some NT) :
    Good ρ (atomVals ρ tbl) NT :=
  buildNT_go_good tbl [] ⟨[], by simp, rfl⟩ (by unfold TableInv; intro k h p hk; simp at hk) h

end Alkahest.Kernel
