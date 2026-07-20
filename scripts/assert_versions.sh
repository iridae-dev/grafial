#!/usr/bin/env bash
# Assert that workspace package versions agree.
# When EXPECT_VERSION is set (release CI), also require manifests match that value.
# Optionally verifies `grafial --version` when the CLI binary is on PATH.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo_ver="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path("Cargo.toml").read_text()
m = re.search(r'(?m)^\[workspace\.package\]\s*\n(?:.*\n)*?^version\s*=\s*"([^"]+)"', text)
if not m:
    raise SystemExit("workspace.package version not found in Cargo.toml")
print(m.group(1))
PY
)"

py_ver="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path("crates/grafial-python/pyproject.toml").read_text()
m = re.search(r'(?m)^\[project\]\s*\n(?:.*\n)*?^version\s*=\s*"([^"]+)"', text)
if not m:
    raise SystemExit("project version not found in pyproject.toml")
print(m.group(1))
PY
)"

lsp_ver="$(python3 - <<'PY'
import re, pathlib
text = pathlib.Path("crates/grafial-lsp/Cargo.toml").read_text()
m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
if not m:
    raise SystemExit("version not found in grafial-lsp Cargo.toml")
print(m.group(1))
PY
)"

echo "Cargo workspace version: $cargo_ver"
echo "Python package version:  $py_ver"
echo "LSP crate version:       $lsp_ver"

if [[ "$cargo_ver" != "$py_ver" || "$cargo_ver" != "$lsp_ver" ]]; then
  echo "assert_versions: version mismatch among Cargo.toml / pyproject.toml / grafial-lsp" >&2
  exit 1
fi

if [[ -n "${EXPECT_VERSION:-}" ]]; then
  if [[ "$cargo_ver" != "$EXPECT_VERSION" ]]; then
    echo "assert_versions: expected $EXPECT_VERSION, found $cargo_ver" >&2
    exit 1
  fi
  echo "Matches EXPECT_VERSION=$EXPECT_VERSION"
fi

if command -v grafial >/dev/null 2>&1; then
  cli_out="$(grafial --version || true)"
  echo "CLI: $cli_out"
  if ! grep -Fq "$cargo_ver" <<<"$cli_out"; then
    echo "assert_versions: CLI version does not contain $cargo_ver" >&2
    exit 1
  fi
fi

echo "assert_versions: ok ($cargo_ver)"
