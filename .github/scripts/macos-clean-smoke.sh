#!/usr/bin/env bash
# Smoke-test a macOS wheel with the libraries it was built against gone.
#
#   macos-clean-smoke.sh WHEEL [DEPS_PREFIX]
#
# 3.12.0 through 3.13.1 shipped wheels linked to Homebrew's FLINT, GMP and MPFR,
# with no copies of them, and the release smoke test passed because the runner
# had those Homebrew kegs. So: move away everything a build could have linked
# (`brew unlink` would not do: it leaves /opt/homebrew/opt/<keg> in place, and
# that is the path the broken wheels load from), insist that nothing was missed,
# then install into a fresh venv and run from outside the checkout.
#
# Destructive for the runner's Homebrew; the last thing a job should do.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: macos-clean-smoke.sh WHEEL [DEPS_PREFIX]" >&2
  exit 2
fi
wheel=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
deps_prefix=${2:-$HOME/macos-deps}
checkout=$(cd "$(dirname "$0")/../.." && pwd)

unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES
unset FLINT_LIB_DIR FLINT_INCLUDE_DIR CPATH LIBRARY_PATH

hidden=$(mktemp -d)
if [ -e "$deps_prefix" ]; then
  mv "$deps_prefix" "$hidden/build-prefix"
fi
for brew_root in /opt/homebrew /usr/local; do
  for keg in flint gmp mpfr; do
    for path in "$brew_root/opt/$keg" "$brew_root/Cellar/$keg"; do
      if [ -e "$path" ] || [ -L "$path" ]; then
        mkdir -p "$hidden/$(dirname "${path#/}")"
        mv "$path" "$hidden/${path#/}"
      fi
    done
  done
done

# A library still reachable would make everything below prove nothing.
# (`-type f`, so the kegs' now-dangling symlinks in lib/ do not count.)
leftovers=$(find /opt/homebrew /usr/local \
  \( -name 'libflint*.dylib' -o -name 'libgmp*.dylib' -o -name 'libmpfr*.dylib' \) \
  -type f 2>/dev/null || true)
if [ -n "$leftovers" ]; then
  echo "::error::these could still satisfy the wheel, so this test proves nothing:"
  echo "$leftovers"
  exit 1
fi

work=$(mktemp -d)
python -m venv "$work/venv"
"$work/venv/bin/pip" install --no-index --no-deps "$wheel"
cd "$work"

# Every FLINT/GMP/MPFR that dyld loads must be the wheel's own.
if ! DYLD_PRINT_LIBRARIES=1 "$work/venv/bin/python" -c 'import alkahest' 2> "$work/dyld.log"; then
  cat "$work/dyld.log"
  echo "::error::import alkahest failed with the build-time libraries unavailable"
  exit 1
fi
grep -E 'lib(flint|gmp|mpfr)' "$work/dyld.log" > "$work/dyld.loaded" || {
  echo "::error::dyld reported loading no FLINT/GMP/MPFR"
  exit 1
}
cat "$work/dyld.loaded"
if grep -v '/site-packages/alkahest/\.dylibs/' "$work/dyld.loaded"; then
  echo "::error::loaded a FLINT/GMP/MPFR from outside the wheel (listed above)"
  exit 1
fi

"$work/venv/bin/python" - <<'PY'
import alkahest as ak

p = ak.ExprPool()
# A bignum through the number-conversion path: #395's PyIndex_Check broke this.
assert str(p.integer(10**30)) == str(10**30)
x = p.symbol("x")
print(ak.simplify(x + 0).value)
print(ak.simplify_egraph(x + 0).value)
# Reaches FLINT's polynomial factorisation, not just its dynamic linkage.
factored = str(ak.factor(x**2 - 1))
assert "x + 1" in factored and "x + -1" in factored, factored
print("factor(x**2 - 1) =", factored)
assert hasattr(ak, "solve"), "expected groebner APIs in default wheel"
assert ak.alkahest.GMP_SHARED_WITH_FLINT, "wheel built without system-gmp"
f = ak.capabilities()["features"]
assert f == {
    "egraph": True, "groebner": True, "jit": False,
    "cranelift": True, "llvm_jit": False, "cranelift_jit": True,
    "parallel": True, "cuda": False,
    "arb_backend": True, "riemann_theta": True,
}, f
assert ak.jit_is_available(), "expected Cranelift JIT in default wheel"
print("clean-environment smoke ok")
PY

# See build-linux's step of the same name.
"$work/venv/bin/python" "$checkout/tests/smoke_readme.py"
