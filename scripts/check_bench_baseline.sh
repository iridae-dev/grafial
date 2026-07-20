#!/usr/bin/env bash
# Compare baseline_probe results against benchmarks/baseline.json limits.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

BASELINE="$root/benchmarks/baseline.json"
if [[ ! -f "$BASELINE" ]]; then
  echo "check_bench_baseline: missing $BASELINE" >&2
  exit 1
fi

echo "[bench] running baseline_probe (release)"
PROBE_JSON="$(cargo run -p grafial-benches --bin baseline_probe --release --quiet)"

echo "[bench] running legacy analytical smoke timing"
START="$(python3 -c 'import time; print(time.time())')"
cargo test -p grafial-tests --test bayesian_updates_tests -- --quiet
END="$(python3 -c 'import time; print(time.time())')"

export PROBE_JSON START END BASELINE
python3 <<'PY'
import json, os, sys

data = json.load(open(os.environ["BASELINE"]))
probes = json.loads(os.environ["PROBE_JSON"])
elapsed_ms = (float(os.environ["END"]) - float(os.environ["START"])) * 1000.0

legacy = data["gates"]["bayesian_updates_tests_ms"]["max_ms"]
print(f"bayesian_updates_tests wall time: {elapsed_ms:.1f} ms (limit {legacy:.1f} ms)")
if elapsed_ms > legacy:
    raise SystemExit(
        f"benchmark regression: analytical tests {elapsed_ms:.1f} ms > {legacy:.1f} ms"
    )

expected = data["probes"]
seen = {p["name"] for p in probes}
missing = set(expected) - seen
if missing:
    raise SystemExit(f"missing probe results: {sorted(missing)}")

for p in probes:
    name = p["name"]
    gate = expected[name]
    wall = float(p["wall_ms"])
    limit = float(gate["max_wall_ms"])
    print(
        f"{name}: wall={wall:.3f} ms (limit {limit:.1f}), "
        f"n={gate['n_nodes']}/{gate['n_edges']}, density={gate['density']}, "
        f"transform={gate['transform']}, backend={gate['backend']}"
    )
    if wall > limit:
        raise SystemExit(f"benchmark regression: {name} wall {wall:.3f} > {limit:.1f} ms")
    if gate.get("require_converged"):
        if not p.get("bp_converged", False):
            raise SystemExit(f"benchmark regression: {name} did not converge")
        iters = int(p["bp_iterations"])
        max_iters = int(gate["max_bp_iterations"])
        print(f"  bp_iterations={iters} (limit {max_iters}), converged={p['bp_converged']}")
        if iters > max_iters:
            raise SystemExit(f"benchmark regression: {name} iterations {iters} > {max_iters}")
    if "score_ref" in gate:
        score = float(p["score"])
        ref = float(gate["score_ref"])
        tol = float(gate.get("score_rel_tol", 1e-6))
        rel = abs(score - ref) / max(abs(ref), 1e-12)
        print(f"  score={score:.6f} ref={ref:.6f} rel_err={rel:.3e}")
        if rel > tol:
            raise SystemExit(f"score golden mismatch for {name}")

print("check_bench_baseline: ok")
PY
