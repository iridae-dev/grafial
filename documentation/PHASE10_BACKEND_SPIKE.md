# Phase 10 Backend Spike

This document defines the backend comparison scaffold for selecting a default JIT codegen backend in Phase 10.

## Goal

Compare backend candidates on:

- cold compile+execute latency
- steady-state runtime latency
- implementation complexity
- maintenance burden
- deterministic parity with interpreter output

The target decision is the default backend between LLVM and Cranelift after benchmark and maintenance scoring.

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

The harness currently includes:

- interpreter baseline
- prototype hot-expression JIT baseline
- `llvm-candidate` backend entry
- `cranelift-candidate` backend entry

## Decision Matrix

Use this matrix for backend selection after collecting benchmark data.

| Candidate | Cold Run p50 (ms) | Warm Run p50 (ms) | Complexity (1-5, low is better) | Maintenance (1-5, low is better) | Parity vs Interpreter | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Interpreter (baseline) | 0.0163 | 0.0162 | 1 | 1 | Pass | Baseline |
| Prototype JIT (baseline) | 0.0170 | 0.0164 | 2 | 2 | Pass | Baseline |
| LLVM backend candidate | 0.0169 | 0.0164 | 2 | 2 | Pass | Pending |
| Cranelift backend candidate | 0.0168 | 0.0167 | 2 | 2 | Pass | Pending |

Measurement note:

- Values above come from one local spike run on February 18, 2026 (`cargo bench -p grafial-benches --bench backend_spike -- --sample-size 10`).
- LLVM and Cranelift entries currently route through candidate wrappers over the shared prototype hot-expression runtime path; native codegen differentiation is the next implementation step.

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
