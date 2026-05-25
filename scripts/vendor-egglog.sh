#!/usr/bin/env bash
# vendor-egglog.sh — re-vendor egglog at a new tag.
#
# Usage: scripts/vendor-egglog.sh v0.5.0
#
# This script replaces vendor/egglog/ with a fresh clone of the given tag,
# strips the git history and web-demo, removes the nested [workspace] block
# (which conflicts with alkahest's workspace resolver), and updates Cargo.lock.
#
# The [patch.crates-io] entry in the root Cargo.toml always points to
# vendor/egglog, so no Cargo.toml edits are needed when upgrading.
#
# After running this script:
#   1. Build smoke-test: cargo build -p alkahest-cas --features egraph
#   2. Test:             cargo test  -p alkahest-cas --features egraph -- simplify
#   3. Commit vendor/egglog and Cargo.lock together.

set -euo pipefail

TAG="${1:?Usage: $0 <egglog-tag>  e.g. v0.4.0}"
REPO="https://github.com/egraphs-good/egglog.git"
DEST="vendor/egglog"

cd "$(git rev-parse --show-toplevel)"

echo "==> Removing old vendor/egglog"
rm -rf "$DEST"

echo "==> Cloning egglog $TAG"
git clone --depth=1 --branch "$TAG" "$REPO" "$DEST"

echo "==> Stripping .git, nested Cargo.lock, web-demo"
rm -rf "$DEST/.git" "$DEST/Cargo.lock" "$DEST/web-demo"

echo "==> Removing [workspace] block from vendor/egglog/Cargo.toml"
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("vendor/egglog/Cargo.toml")
text = p.read_text()
# Remove the [workspace] stanza and its members line
text = re.sub(r'\[workspace\]\nmembers = \[.*?\]\n\n?', '', text, flags=re.DOTALL)
p.write_text(text)
print("  Cargo.toml cleaned")
PY

echo "==> Updating Cargo.lock"
cargo update -p egglog

echo ""
echo "Done. Now run:"
echo "  cargo build -p alkahest-cas --features egraph"
echo "  cargo test  -p alkahest-cas --features egraph -- simplify"
echo "Then commit: git add vendor/egglog Cargo.lock && git commit -m 'chore(vendor): update egglog to $TAG'"
