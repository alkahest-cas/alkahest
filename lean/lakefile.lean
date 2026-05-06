import Lake
open Lake DSL

-- Minimal lake project used by CI to verify generated Lean 4 proofs.
-- `lake update` resolves the Mathlib4 revision compatible with the
-- lean-toolchain pin; `lake exe cache get` downloads prebuilt oleans.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.9.0"

package alkahestProofs
