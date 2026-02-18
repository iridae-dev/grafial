# Phase 10 Backend Matrix Results

- Generated: unix timestamp `1771411344`
- Workloads: `12`
- Warm-up runs per backend/workload: `4`
- Measured warm runs per backend/workload: `30`
- Metric compile threshold: `1`
- Prune compile threshold: `1`

## Aggregate (Mean Across Workloads)

| Backend | Mean Cold ms | Mean Warm Median ms | Mean Warm p95 ms |
| --- | ---: | ---: | ---: |
| interpreter | 0.0496 | 0.0412 | 0.0479 |
| prototype_jit | 0.0457 | 0.0405 | 0.0455 |
| llvm_candidate | 0.0495 | 0.0407 | 0.0511 |
| cranelift_candidate | 0.0452 | 0.0384 | 0.0486 |

## Per Workload

### ab_testing.grafial (`ab_testing.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0473 | 0.0331 | 0.0373 | pass |
| prototype_jit | 0.0340 | 0.0308 | 0.0318 | pass |
| llvm_candidate | 0.0516 | 0.0305 | 0.0415 | pass |
| cranelift_candidate | 0.0368 | 0.0313 | 0.0581 | pass |

### advanced_metrics.grafial (`advanced_metrics.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.1565 | 0.1340 | 0.1385 | pass |
| prototype_jit | 0.1357 | 0.1241 | 0.1754 | pass |
| llvm_candidate | 0.1518 | 0.1234 | 0.1355 | pass |
| cranelift_candidate | 0.1403 | 0.1114 | 0.1548 | pass |

### common_mistakes.grafial (`common_mistakes.grafial` flows=3)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0569 | 0.0525 | 0.0760 | pass |
| prototype_jit | 0.0613 | 0.0533 | 0.0538 | pass |
| llvm_candidate | 0.0685 | 0.0530 | 0.0584 | pass |
| cranelift_candidate | 0.0538 | 0.0487 | 0.0642 | pass |

### competing_choices.grafial (`competing_choices.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0366 | 0.0323 | 0.0325 | pass |
| prototype_jit | 0.0351 | 0.0325 | 0.0328 | pass |
| llvm_candidate | 0.0351 | 0.0327 | 0.0525 | pass |
| cranelift_candidate | 0.0361 | 0.0325 | 0.0476 | pass |

### minimal.grafial (`minimal.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0092 | 0.0077 | 0.0080 | pass |
| prototype_jit | 0.0097 | 0.0078 | 0.0080 | pass |
| llvm_candidate | 0.0089 | 0.0078 | 0.0078 | pass |
| cranelift_candidate | 0.0087 | 0.0078 | 0.0080 | pass |

### pipeline_composition.grafial (`pipeline_composition.grafial` flows=3)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.1248 | 0.0978 | 0.1283 | pass |
| prototype_jit | 0.1122 | 0.0989 | 0.0998 | pass |
| llvm_candidate | 0.1092 | 0.1015 | 0.1467 | pass |
| cranelift_candidate | 0.1139 | 0.0943 | 0.0948 | pass |

### prior_sensitivity.grafial (`prior_sensitivity.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0067 | 0.0057 | 0.0057 | pass |
| prototype_jit | 0.0069 | 0.0057 | 0.0059 | pass |
| llvm_candidate | 0.0065 | 0.0057 | 0.0058 | pass |
| cranelift_candidate | 0.0063 | 0.0057 | 0.0059 | pass |

### probabilistic_pattern_matching.grafial (`probabilistic_pattern_matching.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0467 | 0.0378 | 0.0390 | pass |
| prototype_jit | 0.0399 | 0.0379 | 0.0382 | pass |
| llvm_candidate | 0.0395 | 0.0390 | 0.0670 | pass |
| cranelift_candidate | 0.0427 | 0.0365 | 0.0371 | pass |

### social.grafial (`social.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0269 | 0.0205 | 0.0234 | pass |
| prototype_jit | 0.0283 | 0.0205 | 0.0206 | pass |
| llvm_candidate | 0.0265 | 0.0204 | 0.0205 | pass |
| cranelift_candidate | 0.0233 | 0.0201 | 0.0235 | pass |

### soft_vs_hard_updates.grafial (`soft_vs_hard_updates.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0277 | 0.0247 | 0.0251 | pass |
| prototype_jit | 0.0299 | 0.0250 | 0.0258 | pass |
| llvm_candidate | 0.0439 | 0.0255 | 0.0270 | pass |
| cranelift_candidate | 0.0274 | 0.0248 | 0.0259 | pass |

### transitive_closure.grafial (`transitive_closure.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0350 | 0.0305 | 0.0432 | pass |
| prototype_jit | 0.0358 | 0.0315 | 0.0356 | pass |
| llvm_candidate | 0.0340 | 0.0315 | 0.0323 | pass |
| cranelift_candidate | 0.0341 | 0.0308 | 0.0452 | pass |

### uncertainty_propagation.grafial (`uncertainty_propagation.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0203 | 0.0173 | 0.0177 | pass |
| prototype_jit | 0.0191 | 0.0175 | 0.0177 | pass |
| llvm_candidate | 0.0186 | 0.0173 | 0.0175 | pass |
| cranelift_candidate | 0.0187 | 0.0174 | 0.0176 | pass |

