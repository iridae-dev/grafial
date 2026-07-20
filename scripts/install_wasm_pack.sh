#!/usr/bin/env bash
# Install a pinned wasm-pack release with SHA-256 verification.
# Override with WASM_PACK_VERSION / WASM_PACK_SHA256 if needed.

set -euo pipefail

VERSION="${WASM_PACK_VERSION:-0.15.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${WASM_PACK_CACHE:-$ROOT/.cache/wasm-pack}"
mkdir -p "$CACHE_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Linux-x86_64)
    ASSET="wasm-pack-v${VERSION}-x86_64-unknown-linux-musl.tar.gz"
    SHA256="${WASM_PACK_SHA256:-c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a}"
    ;;
  Darwin-arm64|Darwin-aarch64)
    # Official release assets currently ship x86_64 darwin; Rosetta runs it on Apple Silicon CI.
    ASSET="wasm-pack-v${VERSION}-x86_64-apple-darwin.tar.gz"
    SHA256="${WASM_PACK_SHA256:-d3f1a4a33e95f8f0d7801b024e08624c479999ac96aa150908b2394015cd0363}"
    ;;
  Darwin-x86_64)
    ASSET="wasm-pack-v${VERSION}-x86_64-apple-darwin.tar.gz"
    SHA256="${WASM_PACK_SHA256:-d3f1a4a33e95f8f0d7801b024e08624c479999ac96aa150908b2394015cd0363}"
    ;;
  *)
    echo "install_wasm_pack: unsupported platform $OS-$ARCH" >&2
    exit 1
    ;;
esac

URL="https://github.com/wasm-bindgen/wasm-pack/releases/download/v${VERSION}/${ASSET}"
ARCHIVE="$CACHE_DIR/$ASSET"

if [[ ! -f "$ARCHIVE" ]]; then
  curl -fsSL "$URL" -o "$ARCHIVE"
fi

echo "$SHA256  $ARCHIVE" | shasum -a 256 -c -

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
tar -xzf "$ARCHIVE" -C "$tmpdir"
bin="$(find "$tmpdir" -type f -name wasm-pack | head -n1)"
install -m 755 "$bin" "$HOME/.cargo/bin/wasm-pack"
echo "Installed wasm-pack $(wasm-pack --version) from $ASSET"
