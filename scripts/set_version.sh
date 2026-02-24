#!/usr/bin/env bash
# Set version in Cargo.toml and pyproject.toml from VERSION env or git tag.
# Usage: VERSION=0.1.0 ./scripts/set_version.sh
#        ./scripts/set_version.sh  (reads from GITHUB_REF if set, else exits)

set -euo pipefail

if [[ -n "${VERSION:-}" ]]; then
  ver="$VERSION"
elif [[ -n "${GITHUB_REF:-}" ]] && [[ "$GITHUB_REF" =~ ^refs/tags/v(.+)$ ]]; then
  ver="${BASH_REMATCH[1]}"
else
  echo "set_version: set VERSION or run from CI with GITHUB_REF" >&2
  exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Update workspace Cargo.toml [workspace.package] version only
sed -i.bak '/^\[workspace\.package\]$/,/^\[/ s/^version = ".*"/version = "'"$ver"'"/' "$root/Cargo.toml"
rm -f "$root/Cargo.toml.bak"

# Update grafial-python pyproject.toml [project] version only
sed -i.bak '/^\[project\]$/,/^\[/ s/^version = ".*"/version = "'"$ver"'"/' "$root/crates/grafial-python/pyproject.toml"
rm -f "$root/crates/grafial-python/pyproject.toml.bak"

echo "Version set to $ver"
