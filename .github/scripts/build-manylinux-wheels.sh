#!/usr/bin/env bash
# Build the default manylinux_2_28 wheels. Runs inside the manylinux container,
# from the checkout (mounted at the working directory), on x86_64 and aarch64.
#
# Run with plain `docker run` rather than PyO3/maturin-action because the action
# fetches maturin, rustup and uv inside the container with bare `curl`, no
# retries, before any step of ours runs, and offers no hook to change that.
# Here rustup comes through fetch.sh and maturin from PyPI (pip retries).
#
#   MATURIN_VERSION  exact maturin release, e.g. 1.16.0
#   INTERPRETERS     space-separated, e.g. "python3.10 python3.11". Explicit
#                    because --find-interpreter also matches PyPy.
#   FEATURES         comma-separated Cargo features
set -euo pipefail
: "${MATURIN_VERSION:?}" "${INTERPRETERS:?}" "${FEATURES:?}"

fetch=.github/scripts/fetch.sh

echo "::group::Install Rust"
rustup_base="https://static.rust-lang.org/rustup/dist/$(uname -m)-unknown-linux-gnu/rustup-init"
"$fetch" /tmp/rustup-init.sha256 - "$rustup_base.sha256"
"$fetch" /tmp/rustup-init "$(cut -d' ' -f1 /tmp/rustup-init.sha256)" "$rustup_base"
chmod +x /tmp/rustup-init
export RUSTUP_MAX_RETRIES=10 CARGO_NET_RETRY=10
/tmp/rustup-init -y --profile minimal --default-toolchain stable
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
echo "::endgroup::"

echo "::group::Install maturin"
python3.12 -m pip install --retries 10 "maturin==$MATURIN_VERSION"
python3.12 -m maturin --version
echo "::endgroup::"

echo "::group::Build GMP, MPFR and FLINT"
# shellcheck source=.github/scripts/build-manylinux-deps.sh
source .github/scripts/build-manylinux-deps.sh
echo "::endgroup::"

# Covers both the dnf-installed path (/usr/lib64) and the source-built path
# (/usr/local/lib) so rust-lld can always resolve -lflint.
export RUSTFLAGS="-L/usr/local/lib -L/usr/lib -L/usr/lib64"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib64/pkgconfig"
# gmp-mpfr-sys (the `system-gmp` feature every wheel is built with) probes
# GMP/MPFR with a bare `gcc probe.c -lgmp`; gcc searches /usr/local/include by
# default but not /usr/local/lib.
export LIBRARY_PATH=/usr/local/lib

# Word-splitting of INTERPRETERS is the point: one --interpreter, many values.
# shellcheck disable=SC2086
python3.12 -m maturin build --release --strip --out dist \
  --compatibility manylinux_2_28 \
  --features "$FEATURES" \
  --interpreter $INTERPRETERS
