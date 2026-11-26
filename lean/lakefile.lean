import Lake
open Lake DSL

-- Minimal lake project used by CI to verify generated Lean 4 proofs.
-- `lake update` resolves the Mathlib4 revision compatible with the
-- lean-toolchain pin; `lake exe cache get` downloads prebuilt oleans.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.36.0"

package alkahestProofs

-- The reflective certificate kernel (`Alkahest.Kernel`): kernel certificates
-- `import Alkahest.Kernel`, so CI runs `lake build Alkahest` before checking them.
lean_lib Alkahest
