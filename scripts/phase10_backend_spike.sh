#!/usr/bin/env bash
set -euo pipefail

echo "[phase10] running backend spike benchmark scaffold"
cargo bench -p grafial-benches --bench backend_spike "$@"

echo "[phase10] benchmark run finished"
echo "[phase10] record decision inputs in documentation/PHASE10_BACKEND_SPIKE.md"
