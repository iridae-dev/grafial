# Phase 10 Backend Spike

This document defines the backend comparison scaffold for selecting a default JIT codegen backend in Phase 10.

## Goal

Compare backend candidates on:

- cold compile+execute latency
- steady-state runtime latency
- implementation complexity
- maintenance burden
- deterministic parity with interpreter output

The target decision is the default backend between LLVM and Cranelift once both candidates are wired.

## Current Harness

Benchmark harness:

- `/Users/charleshinshaw/Desktop/content/baygraph/crates/grafial-benches/benches/backend_spike.rs`

Runner script:

- `/Users/charleshinshaw/Desktop/content/baygraph/scripts/phase10_backend_spike.sh`

Run commands:

```bash
# compile-check benchmark target
cargo bench -p grafial-benches --bench backend_spike --no-run

# run benchmark
./scripts/phase10_backend_spike.sh
```

The harness currently includes interpreter and prototype-JIT baselines so metric collection is already wired before LLVM/Cranelift integration.

## Decision Matrix

Use this matrix for backend selection once candidate backends are available.

| Candidate | Cold Run p50 (ms) | Warm Run p50 (ms) | Complexity (1-5, low is better) | Maintenance (1-5, low is better) | Parity vs Interpreter | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Interpreter (baseline) | TBD | TBD | 1 | 1 | Pass | Baseline |
| Prototype JIT (baseline) | TBD | TBD | 2 | 2 | Pass | Baseline |
| LLVM backend | TBD | TBD | TBD | TBD | TBD | Pending |
| Cranelift backend | TBD | TBD | TBD | TBD | TBD | Pending |

## Scoring Rubric

Apply weighted score to LLVM and Cranelift only:

- cold run latency weight: 0.30
- warm run latency weight: 0.40
- complexity weight: 0.15
- maintenance weight: 0.15

Normalization rule:

- for each metric, normalize to `[0, 1]` where `1` is best among candidates and `0` is worst.
- for complexity/maintenance, lower raw score is better.

Selection gate:

- parity must pass for all example and integration flows.
- selected backend must not regress warm run p50 by more than 10% relative to the other candidate on median workload.
- if weighted scores differ by < 0.05, prefer lower maintenance score.

## Required Artifacts For Final Selection

- benchmark output snapshot for both candidates (cold/warm runs)
- parity report for deterministic outputs
- implementation complexity notes (code size, unsafe blocks, debugability)
- maintenance assessment (dependency surface, update cadence, CI/tooling complexity)
