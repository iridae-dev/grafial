#!/usr/bin/env bash
# Builds the wasm package for the composer and serves the repository root.
#
# Usage: ./scripts/serve_composer.sh [port]
# Then open: http://localhost:<port>/webapp/
#
# Serving the repo root (not webapp/) lets the app load repository examples
# from crates/grafial-examples/.

set -euo pipefail

PORT="${1:-8000}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/crates/grafial-wasm"
wasm-pack build --target web --out-dir "$REPO_ROOT/webapp/pkg" --release

echo
echo "✓ Composer ready — open http://localhost:${PORT}/webapp/"
cd "$REPO_ROOT"
exec python3 -m http.server "$PORT"
