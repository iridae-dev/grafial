#!/usr/bin/env bash
# Builds a fully self-contained, static Grafial Composer bundle suitable for
# uploading to any static web host (S3, Netlify, GitHub Pages, nginx, a
# plain shared-hosting webroot, ...).
#
# Usage: ./scripts/build_composer_dist.sh [output-dir]   (default: dist/composer)
#
# The bundle uses only relative paths, so it works from any subdirectory of a
# site (e.g. https://example.com/composer/). Everything runs client-side —
# no server-side code, no CDN dependencies, no analytics.
#
# Requires wasm-pack and the wasm32-unknown-unknown target (see BUILDING.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO_ROOT/dist/composer}"

echo "Building wasm package..."
cd "$REPO_ROOT/crates/grafial-wasm"
wasm-pack build --target web --out-dir "$REPO_ROOT/webapp/pkg" --release

echo "Assembling bundle at $OUT ..."
rm -rf "$OUT"
mkdir -p "$OUT/pkg" "$OUT/examples"

cd "$REPO_ROOT"
cp webapp/index.html webapp/style.css "$OUT/"
cp -R webapp/js "$OUT/js"
# The wasm package: only the runtime files (skip wasm-pack's package.json,
# README, and .gitignore — this is a site bundle, not an npm package).
cp webapp/pkg/grafial_wasm.js webapp/pkg/grafial_wasm_bg.wasm "$OUT/pkg/"
cp webapp/pkg/grafial_wasm.d.ts webapp/pkg/grafial_wasm_bg.wasm.d.ts "$OUT/pkg/" 2>/dev/null || true
# Bundle the repository examples for the Examples dropdown.
cp crates/grafial-examples/*.grafial "$OUT/examples/"

SIZE=$(du -sh "$OUT" | cut -f1)
echo
echo "✓ Static bundle ready: $OUT ($SIZE)"
echo "  Upload the directory contents to any static host."
echo "  Local preview: python3 -m http.server -d $OUT 8080  →  http://localhost:8080/"
echo
echo "  Note: hosts should serve .wasm as application/wasm for streaming"
echo "  instantiation (all major hosts do); the loader falls back gracefully"
echo "  if not."
