-- Smoke tests for the reflective kernel (`lake env lean test/KernelSmoke.lean`, run in CI).
import Alkahest.Kernel

open Alkahest.Kernel

-- ∫ x² sin x = -x² cos x + 2x sin x + 2 cos x
theorem t1 (x : ℝ) :
    HasDerivAt (fun (x : ℝ) => -(x ^ 2 * Real.cos x) + (2:ℝ) * x * Real.sin x + (2:ℝ) * Real.cos x)
      (x ^ 2 * Real.sin x) x :=
  integral_sound (H := []) (w := ⟨[⟨.var 0, .zero⟩, ⟨.fn .sin, .var 0⟩, ⟨.fn .cos, .var 0⟩], [], []⟩)
    (F := .add (.add (.neg (.mul (.pow (.var 0) 2) (.app .cos (.var 0))))
      (.mul (.mul (.lit 2) (.var 0)) (.app .sin (.var 0)))) (.mul (.lit 2) (.app .cos (.var 0))))
    (f := .mul (.pow (.var 0) 2) (.app .sin (.var 0)))
    (by decide +kernel) [] x trivial

-- ∫ 1/x = log x  on x ≠ 0
theorem t2 (x : ℝ) (h1 : x ≠ 0) : HasDerivAt (fun (x : ℝ) => Real.log x) (x⁻¹) x :=
  integral_sound (H := [.ne (.var 0)])
    (w := ⟨[⟨.var 0, .zero⟩, ⟨.fn .log, .var 0⟩, ⟨.inv, .var 0⟩], [], []⟩)
    (F := .app .log (.var 0)) (f := .inv (.var 0)) (by decide +kernel) [] x ⟨h1, trivial⟩

-- ∫ sin x cos x = -cos(2x)/4, through the relation `sin (x + x) = …` with cofactor 1/2
theorem t3 (x : ℝ) :
    HasDerivAt (fun (x : ℝ) => -(Real.cos ((2:ℝ) * x)) / (4:ℝ)) (Real.sin x * Real.cos x) x :=
  integral_sound (H := [])
    (w := ⟨[⟨.var 0, .zero⟩, ⟨.fn .sin, .var 0⟩, ⟨.fn .cos, .var 0⟩,
      ⟨.fn .sin, .mul (.lit 2) (.var 0)⟩, ⟨.fn .cos, .mul (.lit 2) (.var 0)⟩],
      [.sinAdd (.var 0) (.var 0)], [[([], 1, 2)]]⟩)
    (F := .div (.neg (.app .cos (.mul (.lit 2) (.var 0)))) (.lit 4))
    (f := .mul (.app .sin (.var 0)) (.app .cos (.var 0))) (by decide +kernel) [] x trivial

-- ∫ 1/(x² + 1) = atan x: `1 + x²` (from `D`) and `x² + 1` are one atom
theorem t4 (x : ℝ) : HasDerivAt (fun (x : ℝ) => Real.arctan x) ((x ^ 2 + 1)⁻¹) x :=
  integral_sound (H := [])
    (w := ⟨[⟨.var 0, .zero⟩, ⟨.inv, .add (.pow (.var 0) 2) .one⟩], [], []⟩)
    (F := .app .atan (.var 0)) (f := .inv (.add (.pow (.var 0) 2) .one)) (by decide +kernel) [] x
    trivial

-- A wrong antiderivative is rejected by the checker (d/dx sin x ≠ sin x).
example : checkIntegral [] (.app .sin (.var 0)) (.app .sin (.var 0))
    ⟨[⟨.var 0, .zero⟩, ⟨.fn .sin, .var 0⟩, ⟨.fn .cos, .var 0⟩], [], []⟩ = false := by
  decide +kernel

-- A missing domain hypothesis is rejected (log needs x ≠ 0).
example : checkIntegral [] (.app .log (.var 0)) (.inv (.var 0))
    ⟨[⟨.var 0, .zero⟩, ⟨.fn .log, .var 0⟩, ⟨.inv, .var 0⟩], [], []⟩ = false := by
  decide +kernel
