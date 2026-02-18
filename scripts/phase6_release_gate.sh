#!/usr/bin/env bash

set -euo pipefail

echo "[phase6] checking formatting"
cargo fmt --all -- --check

echo "[phase6] running all-example parse+validate+flow gate"
cargo test -p grafial-tests --test phase6_release_gate

echo "[phase6] running Bayesian and determinism property tests"
cargo test -p grafial-tests --test property_tests
