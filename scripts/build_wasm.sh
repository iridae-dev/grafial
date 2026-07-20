#!/usr/bin/env bash
# Builds the grafial-wasm browser package.
#
# Usage:
#   ./scripts/build_wasm.sh            # ES-module package for browsers (default)
#   ./scripts/build_wasm.sh nodejs     # CommonJS package for Node.js
#   ./scripts/build_wasm.sh bundler    # package for webpack/vite bundlers
#
# Output: crates/grafial-wasm/pkg/ (JS glue, TypeScript definitions, .wasm)
#
# Requires wasm-pack (cargo install wasm-pack) and the wasm32 target
# (rustup target add wasm32-unknown-unknown).

set -euo pipefail

TARGET="${1:-web}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/crates/grafial-wasm"
wasm-pack build --target "$TARGET" --out-dir pkg --release

echo
echo "✓ Package built at crates/grafial-wasm/pkg (target: $TARGET)"
