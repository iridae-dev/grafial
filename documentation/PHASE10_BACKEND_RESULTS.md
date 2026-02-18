# Phase 10 Backend Matrix Results

- Generated: unix timestamp `1771410623`
- Workloads: `12`
- Warm-up runs per backend/workload: `4`
- Measured warm runs per backend/workload: `30`

## Aggregate (Mean Across Workloads)

| Backend | Mean Cold ms | Mean Warm Median ms | Mean Warm p95 ms |
| --- | ---: | ---: | ---: |
| interpreter | 0.0446 | 0.0345 | 0.0478 |
| prototype_jit | 0.0383 | 0.0349 | 0.0436 |
| llvm_candidate | 0.0415 | 0.0346 | 0.0416 |
| cranelift_candidate | 0.0350 | 0.0341 | 0.0364 |

## Per Workload

### ab_testing.grafial (`ab_testing.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0264 | 0.0208 | 0.0229 | pass |
| prototype_jit | 0.0227 | 0.0209 | 0.0231 | pass |
| llvm_candidate | 0.0219 | 0.0200 | 0.0327 | pass |
| cranelift_candidate | 0.0202 | 0.0199 | 0.0201 | pass |

### advanced_metrics.grafial (`advanced_metrics.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.1016 | 0.0880 | 0.0956 | pass |
| prototype_jit | 0.1078 | 0.0926 | 0.1363 | pass |
| llvm_candidate | 0.1028 | 0.0880 | 0.0985 | pass |
| cranelift_candidate | 0.0917 | 0.0875 | 0.0941 | pass |

### common_mistakes.grafial (`common_mistakes.grafial` flows=3)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.1060 | 0.0417 | 0.0450 | pass |
| prototype_jit | 0.0444 | 0.0405 | 0.0420 | pass |
| llvm_candidate | 0.0411 | 0.0403 | 0.0407 | pass |
| cranelift_candidate | 0.0410 | 0.0405 | 0.0426 | pass |

### competing_choices.grafial (`competing_choices.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0342 | 0.0272 | 0.0290 | pass |
| prototype_jit | 0.0284 | 0.0281 | 0.0365 | pass |
| llvm_candidate | 0.0320 | 0.0282 | 0.0295 | pass |
| cranelift_candidate | 0.0284 | 0.0280 | 0.0373 | pass |

### minimal.grafial (`minimal.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0077 | 0.0067 | 0.0069 | pass |
| prototype_jit | 0.0074 | 0.0067 | 0.0069 | pass |
| llvm_candidate | 0.0069 | 0.0067 | 0.0068 | pass |
| cranelift_candidate | 0.0068 | 0.0067 | 0.0068 | pass |

### pipeline_composition.grafial (`pipeline_composition.grafial` flows=3)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0993 | 0.0881 | 0.0917 | pass |
| prototype_jit | 0.0935 | 0.0919 | 0.1104 | pass |
| llvm_candidate | 0.0970 | 0.0920 | 0.0967 | pass |
| cranelift_candidate | 0.0937 | 0.0898 | 0.0943 | pass |

### prior_sensitivity.grafial (`prior_sensitivity.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0068 | 0.0056 | 0.0058 | pass |
| prototype_jit | 0.0062 | 0.0056 | 0.0059 | pass |
| llvm_candidate | 0.0059 | 0.0055 | 0.0056 | pass |
| cranelift_candidate | 0.0057 | 0.0055 | 0.0056 | pass |

### probabilistic_pattern_matching.grafial (`probabilistic_pattern_matching.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0456 | 0.0386 | 0.1061 | pass |
| prototype_jit | 0.0400 | 0.0362 | 0.0372 | pass |
| llvm_candidate | 0.0813 | 0.0388 | 0.0628 | pass |
| cranelift_candidate | 0.0367 | 0.0370 | 0.0388 | pass |

### social.grafial (`social.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0240 | 0.0207 | 0.0378 | pass |
| prototype_jit | 0.0275 | 0.0207 | 0.0336 | pass |
| llvm_candidate | 0.0328 | 0.0214 | 0.0392 | pass |
| cranelift_candidate | 0.0220 | 0.0205 | 0.0228 | pass |

### soft_vs_hard_updates.grafial (`soft_vs_hard_updates.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0279 | 0.0261 | 0.0628 | pass |
| prototype_jit | 0.0278 | 0.0254 | 0.0258 | pass |
| llvm_candidate | 0.0257 | 0.0252 | 0.0253 | pass |
| cranelift_candidate | 0.0254 | 0.0252 | 0.0255 | pass |

### transitive_closure.grafial (`transitive_closure.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0347 | 0.0323 | 0.0511 | pass |
| prototype_jit | 0.0345 | 0.0318 | 0.0471 | pass |
| llvm_candidate | 0.0327 | 0.0308 | 0.0427 | pass |
| cranelift_candidate | 0.0309 | 0.0305 | 0.0308 | pass |

### uncertainty_propagation.grafial (`uncertainty_propagation.grafial` flows=1)

| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |
| --- | ---: | ---: | ---: | --- |
| interpreter | 0.0204 | 0.0177 | 0.0185 | pass |
| prototype_jit | 0.0190 | 0.0180 | 0.0181 | pass |
| llvm_candidate | 0.0181 | 0.0180 | 0.0181 | pass |
| cranelift_candidate | 0.0181 | 0.0180 | 0.0181 | pass |

