# shellcheck shell=bash
# Build GMP, MPFR and FLINT from pinned sources inside a manylinux_2_28
# container. Sourced (not executed) by build-manylinux-wheels.sh, so its exports
# stay in force for the maturin command that follows, and its working directory
# is the checkout.
#
# FLINT is built from a pinned source release rather than taken from dnf. The
# distro package is whatever the manylinux base happens to carry, and the
# previous fallback pinned 2.9.0 — both predate or undershoot what this crate
# needs. FLINT absorbed Arb in 3.0 and rewrote the acb_theta API in 3.2, and
# `alkahest-core/build.rs` probes for those symbols and silently stubs the
# `theta` module when they are missing. A wheel built against an older FLINT
# would therefore ship with theta quietly absent rather than failing to build,
# so the version is pinned here and not negotiated.
#
# GMP and MPFR are source-built for the same reason, and *before* FLINT so FLINT
# links them: the `system-gmp` feature makes rug use the system libgmp/libmpfr
# (one GMP shared with FLINT), and gmp-mpfr-sys requires GMP >= 6.3 / MPFR >= 4.2,
# while AlmaLinux 8's gmp-devel / mpfr-devel are 6.1 / 3.1. The dnf -devel
# packages are deliberately NOT installed, so no older gmp.h or libgmp.so can
# shadow these. (The 6.1 *runtime* stays — the compiler itself needs it — hence
# the ld.so.conf entry below, which puts /usr/local/lib ahead of it for the
# loader and for maturin's auditwheel pass.)
set -euo pipefail

fetch=.github/scripts/fetch.sh

# Everything shipped must run on any CPU of the architecture, not just the one
# this runner happens to have.
case "$(uname -m)" in
  x86_64)
    # --enable-fat: without it GMP's configure tunes its assembly to the build
    # runner's CPU, and the wheel ships that libgmp (auditwheel bundles it) to
    # every user; on an older CPU that is SIGILL on the first bignum multiply. A
    # fat build carries every x86_64 kernel and picks one at load time via CPUID.
    gmp_portable=(--enable-fat)
    # FLINT's configure otherwise reads the runner's CPU and adds e.g.
    # -march=znver4, so the wheel's libflint needed AVX-512 and crashed (SIGILL)
    # in factor/gcd on any CPU without it (3.13.0). A generic x86_64 triple
    # selects no -march and no AVX code paths.
    flint_triple=x86_64-pc-linux-gnu
    ;;
  aarch64)
    # GMP's --enable-fat exists only for x86 (it is silently ignored here), and
    # GMP's config.guess reads the CPU part from /proc/cpuinfo: on a Cortex-A76
    # it would select -march=armv8.2-a. The plain triple selects the generic
    # arm64 assembly and -march=armv8-a.
    gmp_portable=(--host=aarch64-unknown-linux-gnu --build=aarch64-unknown-linux-gnu)
    # FLINT 3.6's configure.ac, same config.guess: `aarch64*` selects
    # -march=armv8-a, no -mtune, asm_path=arm64 and the NEON fft_small (NEON is
    # baseline on ARMv8-A), rather than a CPU-specific armv8.x / -mtune=cortex-*.
    flint_triple=aarch64-unknown-linux-gnu
    ;;
  *)
    echo "no portable build flags for $(uname -m)" >&2
    exit 1
    ;;
esac

dnf install -y make gcc m4 xz
export LIBRARY_PATH=/usr/local/lib

# The second URL is a mirror; the checksum makes it as trustworthy as the first.
"$fetch" /tmp/gmp-6.3.0.tar.xz \
  a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898 \
  https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz \
  https://mirrors.kernel.org/gnu/gmp/gmp-6.3.0.tar.xz
"$fetch" /tmp/mpfr-4.2.2.tar.xz \
  b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01 \
  https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.2.tar.xz \
  https://mirrors.kernel.org/gnu/mpfr/mpfr-4.2.2.tar.xz
tar xf /tmp/gmp-6.3.0.tar.xz -C /tmp
tar xf /tmp/mpfr-4.2.2.tar.xz -C /tmp
# No bare `cd`: the maturin command runs after this script in the same shell,
# from the checkout.
( cd /tmp/gmp-6.3.0 && ./configure --prefix=/usr/local --enable-shared --disable-static "${gmp_portable[@]}" && make -j"$(nproc)" && make install )
( cd /tmp/mpfr-4.2.2 && ./configure --prefix=/usr/local --with-gmp=/usr/local --enable-shared --disable-static && make -j"$(nproc)" && make install )
echo /usr/local/lib > /etc/ld.so.conf.d/00-usr-local.conf
ldconfig
export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# FLINT >= 3.3 is the floor build.rs enforces (>= 3.4 for theta); keep this pin
# in step with the default in .github/actions/setup-flint.
"$fetch" /tmp/flint-3.6.0.tar.gz \
  b95e2c7792f5eea4a1c8d2d42c4098434756832e57a094b295eb5dfdc9b4c36b \
  https://github.com/flintlib/flint/releases/download/v3.6.0/flint-3.6.0.tar.gz \
  https://flintlib.org/download/flint-3.6.0.tar.gz
tar xzf /tmp/flint-3.6.0.tar.gz -C /tmp
( cd /tmp/flint-3.6.0 \
  && ./configure --prefix=/usr/local --with-gmp=/usr/local --with-mpfr=/usr/local --disable-static \
       --host="$flint_triple" --build="$flint_triple" \
  && grep -E '^CFLAGS *=' Makefile \
  && make -j"$(nproc)" && make install )
ldconfig

# One GMP: FLINT must resolve the libgmp just built, not the 6.1 runtime.
ldd /usr/local/lib/libflint.so | grep -E 'libgmp|libmpfr'
# Never `producer | grep -q` under pipefail: grep -q exits at the first match,
# the producer dies of SIGPIPE, and the pipeline "fails" precisely when the
# pattern is found. Capture, then grep.
ldd /usr/local/lib/libflint.so > /tmp/libflint.ldd
grep -q '/usr/local/lib/libgmp.so' /tmp/libflint.ldd \
  || { echo 'libflint does not resolve /usr/local/lib/libgmp.so'; exit 1; }
# Fail the wheel build loudly if theta would be stubbed.
nm -D --defined-only /usr/local/lib/libflint.so > /tmp/libflint.syms
grep -q ' T acb_theta_ql_exact$' /tmp/libflint.syms \
  || { echo 'FLINT lacks acb_theta_ql_exact; theta would ship stubbed'; exit 1; }
