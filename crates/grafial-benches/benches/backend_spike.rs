//! Phase 10 backend spike benchmark scaffold.
//!
//! Focus:
//! - cold execution latency (compile + execute path)
//! - warm execution latency (steady-state path)
//! - deterministic parity checks across backends
//!
//! This currently benchmarks:
//! - interpreter backend
//! - prototype hot-expression JIT backend
//!
//! Future LLVM/Cranelift backend candidates can be plugged into the same harness.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use grafial_core::{
    parse_validate_and_lower, run_flow_ir_with_backend, InterpreterExecutionBackend, ProgramIR,
    PrototypeJitConfig, PrototypeJitExecutionBackend,
};

const FLOW_NAME: &str = "Spike";

const SPIKE_SOURCE: &str = r#"
schema Spike {
  node Person {
    score: Real
  }
  edge REL { }
}

belief_model SpikeBeliefs on Spike {
  node Person {
    score ~ Gaussian(mean=0.0, precision=0.1)
  }
  edge REL {
    exist ~ Bernoulli(prior=0.5, weight=2.0)
  }
}

evidence SpikeEvidence on SpikeBeliefs {
  Person {
    "A" { score: 10.0 },
    "B" { score: 5.0 },
    "C" { score: 2.0 },
    "D" { score: 8.0 }
  }
  REL(Person -> Person) {
    "A" -> "B";
    "B" -> "C";
    "C" -> "D";
    "A" -> "D"
  }
}

flow Spike on SpikeBeliefs {
  graph base = from_evidence SpikeEvidence
  graph pruned = base |> prune_edges REL where prob(edge) < 0.5

  metric m_hot = 1.0 + 2.0
  metric edge_density = avg_degree(Person, REL, min_prob=0.1)

  export pruned as "spike"
  export_metric m_hot as "m_hot"
  export_metric edge_density as "edge_density"
}
"#;

fn load_program() -> ProgramIR {
    parse_validate_and_lower(SPIKE_SOURCE).expect("phase10 spike program must parse+validate")
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn assert_backend_parity(program: &ProgramIR) {
    let interpreter = InterpreterExecutionBackend;
    let expected = run_flow_ir_with_backend(program, FLOW_NAME, None, &interpreter)
        .expect("interpreter flow execution");

    let prototype = PrototypeJitExecutionBackend::new(PrototypeJitConfig {
        metric_compile_threshold: 1,
        prune_compile_threshold: 1,
    });
    let actual = run_flow_ir_with_backend(program, FLOW_NAME, None, &prototype)
        .expect("prototype flow execution");

    let expected_hot = expected
        .metric_exports
        .get("m_hot")
        .copied()
        .unwrap_or_default();
    let actual_hot = actual
        .metric_exports
        .get("m_hot")
        .copied()
        .unwrap_or_default();
    assert!(
        approx_eq(expected_hot, actual_hot),
        "m_hot mismatch: interpreter={} prototype={}",
        expected_hot,
        actual_hot
    );

    let expected_density = expected
        .metric_exports
        .get("edge_density")
        .copied()
        .unwrap_or_default();
    let actual_density = actual
        .metric_exports
        .get("edge_density")
        .copied()
        .unwrap_or_default();
    assert!(
        approx_eq(expected_density, actual_density),
        "edge_density mismatch: interpreter={} prototype={}",
        expected_density,
        actual_density
    );

    let expected_edges = expected
        .exports
        .get("spike")
        .map(|graph| graph.edges().len())
        .unwrap_or_default();
    let actual_edges = actual
        .exports
        .get("spike")
        .map(|graph| graph.edges().len())
        .unwrap_or_default();
    assert_eq!(
        expected_edges, actual_edges,
        "exported edge count mismatch: interpreter={} prototype={}",
        expected_edges, actual_edges
    );
}

fn bench_backend_spike(c: &mut Criterion) {
    let program = load_program();
    assert_backend_parity(&program);

    let mut group = c.benchmark_group("phase10_backend_spike");
    group.sample_size(20);
    group.throughput(Throughput::Elements(1));

    group.bench_function("interpreter_cold_run", |b| {
        b.iter(|| {
            let backend = InterpreterExecutionBackend;
            let result = run_flow_ir_with_backend(
                black_box(&program),
                black_box(FLOW_NAME),
                None,
                black_box(&backend),
            )
            .expect("interpreter cold run");
            black_box(result.metric_exports.get("edge_density").copied());
        });
    });

    group.bench_function("prototype_jit_cold_run", |b| {
        b.iter(|| {
            let backend = PrototypeJitExecutionBackend::new(PrototypeJitConfig {
                metric_compile_threshold: 1,
                prune_compile_threshold: 1,
            });
            let result = run_flow_ir_with_backend(
                black_box(&program),
                black_box(FLOW_NAME),
                None,
                black_box(&backend),
            )
            .expect("prototype cold run");
            black_box(result.metric_exports.get("edge_density").copied());
        });
    });

    let interpreter = InterpreterExecutionBackend;
    group.bench_function("interpreter_warm_run", |b| {
        b.iter(|| {
            let result = run_flow_ir_with_backend(
                black_box(&program),
                black_box(FLOW_NAME),
                None,
                black_box(&interpreter),
            )
            .expect("interpreter warm run");
            black_box(result.metric_exports.get("edge_density").copied());
        });
    });

    let prototype = PrototypeJitExecutionBackend::new(PrototypeJitConfig {
        metric_compile_threshold: 1,
        prune_compile_threshold: 1,
    });
    let _ = run_flow_ir_with_backend(&program, FLOW_NAME, None, &prototype)
        .expect("prototype warm-up run");
    group.bench_function("prototype_jit_warm_run", |b| {
        b.iter(|| {
            let result = run_flow_ir_with_backend(
                black_box(&program),
                black_box(FLOW_NAME),
                None,
                black_box(&prototype),
            )
            .expect("prototype warm run");
            black_box(result.metric_exports.get("edge_density").copied());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_backend_spike);
criterion_main!(benches);
