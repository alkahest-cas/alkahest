#!/usr/bin/env bash
# Build GMP, MPFR and FLINT for the macOS arm64 wheels, into PREFIX.
#
#   MACOSX_DEPLOYMENT_TARGET=11.0 build-macos-deps.sh PREFIX
#
# From source, not Homebrew, because of what the wheel can then promise. The
# bottles' minimum macOS is whatever the bottle was built on, delocate copies
# them into the wheel as they are, and so a Homebrew-based wheel can only be
# tagged for the runner's macOS and up. Built here with MACOSX_DEPLOYMENT_TARGET
# they cover every macOS an Apple Silicon Mac can run (Rust's
# aarch64-apple-darwin floor, too, is 11.0).
#
# The same pins and checksums as build-manylinux-deps.sh; keep them in step.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: build-macos-deps.sh PREFIX" >&2
  exit 2
fi
prefix=$1
: "${MACOSX_DEPLOYMENT_TARGET:?set MACOSX_DEPLOYMENT_TARGET}"

here=$(cd "$(dirname "$0")" && pwd)
fetch=$here/fetch.sh
jobs=$(sysctl -n hw.ncpu)

# Nothing from the environment may leak into what gets bundled: a Homebrew
# GMP's header or library picked up here would quietly undo the point.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH PKG_CONFIG_PATH

# GMP's and FLINT's config.guess ask the machine which Apple chip this is, and
# FLINT's configure then adds -march=armv8.6-a for an M2 and -march=armv8.7-a for
# an M4; the wheel would be built for whichever chip the runner happened to have.
# Stating the plain triple selects -march=armv8-a and the generic arm64 assembly,
# which every Apple Silicon Mac runs.
triple=aarch64-apple-darwin

work=$(mktemp -d)
cd "$work"

# The second URL is a mirror; the checksum makes it as trustworthy as the first.
"$fetch" gmp-6.3.0.tar.xz \
  a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898 \
  https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz \
  https://mirrors.kernel.org/gnu/gmp/gmp-6.3.0.tar.xz
"$fetch" mpfr-4.2.2.tar.xz \
  b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01 \
  https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.2.tar.xz \
  https://mirrors.kernel.org/gnu/mpfr/mpfr-4.2.2.tar.xz
"$fetch" flint-3.6.0.tar.gz \
  b95e2c7792f5eea4a1c8d2d42c4098434756832e57a094b295eb5dfdc9b4c36b \
  https://github.com/flintlib/flint/releases/download/v3.6.0/flint-3.6.0.tar.gz \
  https://flintlib.org/download/flint-3.6.0.tar.gz
tar xf gmp-6.3.0.tar.xz
tar xf mpfr-4.2.2.tar.xz
tar xzf flint-3.6.0.tar.gz

# `make check` for GMP and MPFR: the generic arm64 assembly on Darwin is not what
# Homebrew's bottle selects, and this is the only time it is built. FLINT's own
# suite takes hours and exercises nothing specific to this configuration.
( cd gmp-6.3.0 \
  && ./configure --prefix="$prefix" --enable-shared --disable-static \
       --host="$triple" --build="$triple" \
  && grep -E '^(CC|CFLAGS) *=' Makefile \
  && make -j"$jobs" && make -j"$jobs" check && make install )
( cd mpfr-4.2.2 \
  && ./configure --prefix="$prefix" --with-gmp="$prefix" --enable-shared --disable-static \
  && make -j"$jobs" && make -j"$jobs" check && make install )
( cd flint-3.6.0 \
  && ./configure --prefix="$prefix" --with-gmp="$prefix" --with-mpfr="$prefix" --disable-static \
       --host="$triple" --build="$triple" \
  && grep -E '^CFLAGS *=' Makefile > "$work/flint-cflags" \
  && cat "$work/flint-cflags" \
  && make -j"$jobs" && make install )

case "$(cat "$work/flint-cflags")" in
  *-march=armv8.[1-9]*)
    echo "FLINT was configured for a specific Apple chip; the triple did not take" >&2
    exit 1
    ;;
esac

# One GMP: FLINT must resolve the libgmp just built.
libflint=$(cd "$prefix/lib" && ls libflint.[0-9]*.dylib | head -n 1)
otool -L "$prefix/lib/$libflint" > "$work/libflint.deps"
cat "$work/libflint.deps"
grep -q "$prefix/lib/libgmp" "$work/libflint.deps" \
  || { echo "libflint does not resolve $prefix/lib/libgmp" >&2; exit 1; }

# Fail loudly if theta would be stubbed (build.rs probes for these).
nm -gU "$prefix/lib/$libflint" > "$work/libflint.syms"
grep -q ' _acb_theta_ql_exact$' "$work/libflint.syms" \
  || { echo 'FLINT lacks acb_theta_ql_exact; theta would ship stubbed' >&2; exit 1; }

# What the wheel's tag rests on: each library's minimum macOS, and any symbol
# linked weakly because it is newer than the deployment target (such a symbol is
# NULL on an older macOS, so the list should be empty or understood).
for lib in "$prefix"/lib/lib{gmp,mpfr,flint}.*dylib; do
  [ -L "$lib" ] && continue
  echo "== $lib"
  vtool -show-build "$lib"
  nm -m -u "$lib" | grep -i 'weak' || echo "no weak imports"
done
