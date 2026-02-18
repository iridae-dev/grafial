#!/usr/bin/env bash
set -euo pipefail

echo "[phase10] running backend matrix (multi-workload, parity+timing report)"
cargo run -p grafial-benches --release --bin backend_matrix -- "$@"

echo "[phase10] report generated:"
echo "  documentation/PHASE10_BACKEND_RESULTS.md"
echo "  documentation/phase10_backend_results.json"
