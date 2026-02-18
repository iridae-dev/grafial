//! Phase 10 backend matrix runner.
//!
//! Runs all selected example workloads across available backends, records parity checks,
//! and emits both Markdown and JSON reports for backend selection decisions.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use grafial_core::engine::flow_exec::FlowResult;
use grafial_core::{
    parse_validate_and_lower, run_flow_ir_with_backend, CraneliftCandidateExecutionBackend,
    InterpreterExecutionBackend, IrExecutionBackend, LlvmCandidateExecutionBackend, ProgramIR,
    PrototypeJitConfig, PrototypeJitExecutionBackend, PrototypeJitProfile,
};
use serde::Serialize;

const FLOAT_EPSILON: f64 = 1e-12;

#[derive(Debug)]
struct Config {
    repeats: usize,
    warmup_runs: usize,
    metric_compile_threshold: usize,
    prune_compile_threshold: usize,
    examples_dir: PathBuf,
    output_markdown: PathBuf,
    output_json: PathBuf,
    filters: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let mut examples_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        examples_dir.push("../grafial-examples");
        Self {
            repeats: 60,
            warmup_runs: 5,
            metric_compile_threshold: 1,
            prune_compile_threshold: 1,
            examples_dir,
            output_markdown: PathBuf::from("documentation/PHASE10_BACKEND_RESULTS.md"),
            output_json: PathBuf::from("documentation/phase10_backend_results.json"),
            filters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct Workload {
    name: String,
    file: String,
    program: ProgramIR,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum BackendKind {
    Interpreter,
    PrototypeJit,
    LlvmCandidate,
    CraneliftCandidate,
}

impl BackendKind {
    fn all() -> [Self; 4] {
        [
            Self::Interpreter,
            Self::PrototypeJit,
            Self::LlvmCandidate,
            Self::CraneliftCandidate,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::PrototypeJit => "prototype_jit",
            Self::LlvmCandidate => "llvm_candidate",
            Self::CraneliftCandidate => "cranelift_candidate",
        }
    }
}

enum BackendInstance {
    Interpreter(InterpreterExecutionBackend),
    Prototype(PrototypeJitExecutionBackend),
    Llvm(LlvmCandidateExecutionBackend),
    Cranelift(CraneliftCandidateExecutionBackend),
}

impl BackendInstance {
    fn new(kind: BackendKind, jit_config: PrototypeJitConfig) -> Self {
        match kind {
            BackendKind::Interpreter => Self::Interpreter(InterpreterExecutionBackend),
            BackendKind::PrototypeJit => {
                Self::Prototype(PrototypeJitExecutionBackend::new(jit_config))
            }
            BackendKind::LlvmCandidate => {
                Self::Llvm(LlvmCandidateExecutionBackend::new(jit_config))
            }
            BackendKind::CraneliftCandidate => {
                Self::Cranelift(CraneliftCandidateExecutionBackend::new(jit_config))
            }
        }
    }

    fn as_backend(&self) -> &dyn IrExecutionBackend {
        match self {
            Self::Interpreter(backend) => backend,
            Self::Prototype(backend) => backend,
            Self::Llvm(backend) => backend,
            Self::Cranelift(backend) => backend,
        }
    }

    fn clear_profile(&self) -> Result<(), String> {
        match self {
            Self::Interpreter(_) => Ok(()),
            Self::Prototype(backend) => backend.clear_profile().map_err(|e| e.to_string()),
            Self::Llvm(backend) => backend.clear_profile().map_err(|e| e.to_string()),
            Self::Cranelift(backend) => backend.clear_profile().map_err(|e| e.to_string()),
        }
    }

    fn profile_snapshot(&self) -> Option<PrototypeJitProfile> {
        match self {
            Self::Interpreter(_) => None,
            Self::Prototype(backend) => backend.profile_snapshot().ok(),
            Self::Llvm(backend) => backend.profile_snapshot().ok(),
            Self::Cranelift(backend) => backend.profile_snapshot().ok(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct FlowCapture {
    flow: String,
    metric_exports: BTreeMap<String, f64>,
    export_edge_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
struct DurationStats {
    runs: usize,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BackendResult {
    backend: BackendKind,
    cold_ms: f64,
    warm: DurationStats,
    parity_pass: bool,
    profile: Option<BackendProfileCounters>,
}

#[derive(Debug, Clone, Serialize)]
struct BackendProfileCounters {
    metric_eval_count: usize,
    metric_compile_count: usize,
    metric_cache_hits: usize,
    metric_fallback_count: usize,
    prune_eval_count: usize,
    prune_compile_count: usize,
    prune_cache_hits: usize,
    prune_fallback_count: usize,
}

impl From<PrototypeJitProfile> for BackendProfileCounters {
    fn from(value: PrototypeJitProfile) -> Self {
        Self {
            metric_eval_count: value.metric_eval_count,
            metric_compile_count: value.metric_compile_count,
            metric_cache_hits: value.metric_cache_hits,
            metric_fallback_count: value.metric_fallback_count,
            prune_eval_count: value.prune_eval_count,
            prune_compile_count: value.prune_compile_count,
            prune_cache_hits: value.prune_cache_hits,
            prune_fallback_count: value.prune_fallback_count,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct WorkloadResult {
    workload: String,
    file: String,
    flow_count: usize,
    results: Vec<BackendResult>,
}

#[derive(Debug, Clone, Serialize)]
struct BackendAggregate {
    backend: BackendKind,
    mean_cold_ms: f64,
    mean_warm_median_ms: f64,
    mean_warm_p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixReport {
    generated_unix_seconds: u64,
    repeats: usize,
    warmup_runs: usize,
    metric_compile_threshold: usize,
    prune_compile_threshold: usize,
    workload_count: usize,
    workloads: Vec<WorkloadResult>,
    aggregate: Vec<BackendAggregate>,
}

fn parse_config() -> Result<Config, String> {
    let mut cfg = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repeats" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --repeats".to_string())?;
                cfg.repeats = value
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --repeats '{}': {}", value, e))?;
            }
            "--warmup" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --warmup".to_string())?;
                cfg.warmup_runs = value
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --warmup '{}': {}", value, e))?;
            }
            "--examples-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --examples-dir".to_string())?;
                cfg.examples_dir = PathBuf::from(value);
            }
            "--metric-threshold" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --metric-threshold".to_string())?;
                cfg.metric_compile_threshold = value
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --metric-threshold '{}': {}", value, e))?;
            }
            "--prune-threshold" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --prune-threshold".to_string())?;
                cfg.prune_compile_threshold = value
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --prune-threshold '{}': {}", value, e))?;
            }
            "--output-md" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --output-md".to_string())?;
                cfg.output_markdown = PathBuf::from(value);
            }
            "--output-json" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --output-json".to_string())?;
                cfg.output_json = PathBuf::from(value);
            }
            "--filter" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --filter".to_string())?;
                cfg.filters.push(value);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument '{}'", other));
            }
        }
    }
    if cfg.repeats == 0 {
        return Err("--repeats must be >= 1".into());
    }
    if cfg.warmup_runs == 0 {
        return Err("--warmup must be >= 1".into());
    }
    if cfg.metric_compile_threshold == 0 {
        return Err("--metric-threshold must be >= 1".into());
    }
    if cfg.prune_compile_threshold == 0 {
        return Err("--prune-threshold must be >= 1".into());
    }
    Ok(cfg)
}

fn print_help() {
    println!("Phase 10 backend matrix runner");
    println!("Usage: cargo run -p grafial-benches --bin backend_matrix --release -- [OPTIONS]");
    println!("Options:");
    println!("  --repeats <N>        Warm measurement runs per workload/backend (default: 60)");
    println!("  --warmup <N>         Warm-up runs per workload/backend (default: 5)");
    println!("  --metric-threshold <N>  Metric compile threshold (default: 1)");
    println!("  --prune-threshold <N>   Prune compile threshold (default: 1)");
    println!("  --examples-dir <P>   Example directory (default: crates/grafial-examples)");
    println!("  --output-md <P>      Markdown report path (default: documentation/PHASE10_BACKEND_RESULTS.md)");
    println!("  --output-json <P>    JSON report path (default: documentation/phase10_backend_results.json)");
    println!(
        "  --filter <S>         Include only workload files containing substring S (repeatable)"
    );
}

fn load_workloads(cfg: &Config) -> Result<Vec<Workload>, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&cfg.examples_dir).map_err(|e| {
        format!(
            "failed to read examples dir {}: {}",
            cfg.examples_dir.display(),
            e
        )
    })? {
        let entry = entry.map_err(|e| format!("failed to read directory entry: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("grafial") {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or_default();
        if !cfg.filters.is_empty()
            && !cfg
                .filters
                .iter()
                .any(|filter| file_name.contains(filter) || path.to_string_lossy().contains(filter))
        {
            continue;
        }
        files.push(path);
    }
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "no .grafial workload files matched in {}",
            cfg.examples_dir.display()
        ));
    }

    let mut workloads = Vec::with_capacity(files.len());
    for path in files {
        let src = fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
        let program = parse_validate_and_lower(&src)
            .map_err(|e| format!("failed to parse/validate {}: {}", path.display(), e))?;
        if program.flows.is_empty() {
            continue;
        }
        let file = path
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or("unknown.grafial")
            .to_string();
        workloads.push(Workload {
            name: file.clone(),
            file,
            program,
        });
    }
    if workloads.is_empty() {
        return Err("no workloads with executable flows were found".to_string());
    }
    Ok(workloads)
}

fn execute_all_flows(
    program: &ProgramIR,
    backend: &dyn IrExecutionBackend,
) -> Result<Vec<FlowCapture>, String> {
    let mut prior: Option<FlowResult> = None;
    let mut captures = Vec::with_capacity(program.flows.len());

    for flow in &program.flows {
        let result = run_flow_ir_with_backend(program, &flow.name, prior.as_ref(), backend)
            .map_err(|e| {
                format!(
                    "flow '{}' failed on {}: {}",
                    flow.name,
                    backend.backend_name(),
                    e
                )
            })?;

        let metric_exports = result
            .metric_exports
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect::<BTreeMap<_, _>>();

        let export_edge_counts = result
            .exports
            .iter()
            .map(|(k, g)| (k.clone(), g.edges().len()))
            .collect::<BTreeMap<_, _>>();

        captures.push(FlowCapture {
            flow: flow.name.clone(),
            metric_exports,
            export_edge_counts,
        });

        prior = Some(result);
    }

    Ok(captures)
}

fn assert_parity(
    expected: &[FlowCapture],
    actual: &[FlowCapture],
    backend: BackendKind,
    workload: &str,
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "parity failed for {} on {}: flow count mismatch expected {} got {}",
            backend.label(),
            workload,
            expected.len(),
            actual.len()
        ));
    }

    for (idx, (lhs, rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        if lhs.flow != rhs.flow {
            return Err(format!(
                "parity failed for {} on {} at flow index {}: expected '{}' got '{}'",
                backend.label(),
                workload,
                idx,
                lhs.flow,
                rhs.flow
            ));
        }
        if lhs.export_edge_counts != rhs.export_edge_counts {
            return Err(format!(
                "parity failed for {} on {} flow '{}': export edge counts differ",
                backend.label(),
                workload,
                lhs.flow
            ));
        }
        if lhs.metric_exports.len() != rhs.metric_exports.len() {
            return Err(format!(
                "parity failed for {} on {} flow '{}': metric export count differs",
                backend.label(),
                workload,
                lhs.flow
            ));
        }
        for (k, expected_value) in &lhs.metric_exports {
            let Some(actual_value) = rhs.metric_exports.get(k) else {
                return Err(format!(
                    "parity failed for {} on {} flow '{}': missing metric export '{}'",
                    backend.label(),
                    workload,
                    lhs.flow,
                    k
                ));
            };
            if (expected_value - actual_value).abs() > FLOAT_EPSILON {
                return Err(format!(
                    "parity failed for {} on {} flow '{}' metric '{}': expected {} got {}",
                    backend.label(),
                    workload,
                    lhs.flow,
                    k,
                    expected_value,
                    actual_value
                ));
            }
        }
    }

    Ok(())
}

fn ms_from(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn summarize_ms(values: &[f64]) -> DurationStats {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let runs = sorted.len();
    let min_ms = *sorted.first().unwrap_or(&0.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let mean_ms = if runs == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / runs as f64
    };
    let median_ms = percentile(&sorted, 0.50);
    let p95_ms = percentile(&sorted, 0.95);

    DurationStats {
        runs,
        min_ms,
        max_ms,
        mean_ms,
        median_ms,
        p95_ms,
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let p = pct.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn run_backend_measurement(
    workload: &Workload,
    backend_kind: BackendKind,
    jit_config: PrototypeJitConfig,
    repeats: usize,
    warmup_runs: usize,
    expected_capture: &[FlowCapture],
) -> Result<BackendResult, String> {
    let cold_instance = BackendInstance::new(backend_kind, jit_config);
    let cold_start = Instant::now();
    let cold_capture = execute_all_flows(&workload.program, cold_instance.as_backend())?;
    let cold_ms = ms_from(cold_start);
    assert_parity(
        expected_capture,
        &cold_capture,
        backend_kind,
        workload.name.as_str(),
    )?;

    let warm_instance = BackendInstance::new(backend_kind, jit_config);
    for _ in 0..warmup_runs {
        let warm_capture = execute_all_flows(&workload.program, warm_instance.as_backend())?;
        assert_parity(
            expected_capture,
            &warm_capture,
            backend_kind,
            workload.name.as_str(),
        )?;
    }

    warm_instance.clear_profile()?;

    let mut warm_samples_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let run_start = Instant::now();
        let run_capture = execute_all_flows(&workload.program, warm_instance.as_backend())?;
        let elapsed_ms = ms_from(run_start);
        assert_parity(
            expected_capture,
            &run_capture,
            backend_kind,
            workload.name.as_str(),
        )?;
        warm_samples_ms.push(elapsed_ms);
    }

    let warm = summarize_ms(&warm_samples_ms);
    let profile = warm_instance
        .profile_snapshot()
        .map(BackendProfileCounters::from);

    Ok(BackendResult {
        backend: backend_kind,
        cold_ms,
        warm,
        parity_pass: true,
        profile,
    })
}

fn build_aggregate(workloads: &[WorkloadResult]) -> Vec<BackendAggregate> {
    let mut rows = Vec::new();
    for backend in BackendKind::all() {
        let mut cold = Vec::new();
        let mut warm_median = Vec::new();
        let mut warm_p95 = Vec::new();

        for workload in workloads {
            if let Some(result) = workload.results.iter().find(|r| r.backend == backend) {
                cold.push(result.cold_ms);
                warm_median.push(result.warm.median_ms);
                warm_p95.push(result.warm.p95_ms);
            }
        }

        if cold.is_empty() {
            continue;
        }

        rows.push(BackendAggregate {
            backend,
            mean_cold_ms: cold.iter().sum::<f64>() / cold.len() as f64,
            mean_warm_median_ms: warm_median.iter().sum::<f64>() / warm_median.len() as f64,
            mean_warm_p95_ms: warm_p95.iter().sum::<f64>() / warm_p95.len() as f64,
        });
    }
    rows
}

fn render_markdown(report: &MatrixReport) -> String {
    let mut out = String::new();
    let _ = writeln!(&mut out, "# Phase 10 Backend Matrix Results");
    let _ = writeln!(&mut out, "");
    let _ = writeln!(
        &mut out,
        "- Generated: unix timestamp `{}`",
        report.generated_unix_seconds
    );
    let _ = writeln!(&mut out, "- Workloads: `{}`", report.workload_count);
    let _ = writeln!(
        &mut out,
        "- Warm-up runs per backend/workload: `{}`",
        report.warmup_runs
    );
    let _ = writeln!(
        &mut out,
        "- Measured warm runs per backend/workload: `{}`",
        report.repeats
    );
    let _ = writeln!(
        &mut out,
        "- Metric compile threshold: `{}`",
        report.metric_compile_threshold
    );
    let _ = writeln!(
        &mut out,
        "- Prune compile threshold: `{}`",
        report.prune_compile_threshold
    );
    let _ = writeln!(&mut out, "");

    let _ = writeln!(&mut out, "## Aggregate (Mean Across Workloads)");
    let _ = writeln!(&mut out, "");
    let _ = writeln!(
        &mut out,
        "| Backend | Mean Cold ms | Mean Warm Median ms | Mean Warm p95 ms |"
    );
    let _ = writeln!(&mut out, "| --- | ---: | ---: | ---: |");
    for row in &report.aggregate {
        let _ = writeln!(
            &mut out,
            "| {} | {:.4} | {:.4} | {:.4} |",
            row.backend.label(),
            row.mean_cold_ms,
            row.mean_warm_median_ms,
            row.mean_warm_p95_ms
        );
    }
    let _ = writeln!(&mut out, "");

    let _ = writeln!(&mut out, "## Per Workload");
    let _ = writeln!(&mut out, "");
    for workload in &report.workloads {
        let _ = writeln!(
            &mut out,
            "### {} (`{}` flows={})",
            workload.workload, workload.file, workload.flow_count
        );
        let _ = writeln!(&mut out, "");
        let _ = writeln!(
            &mut out,
            "| Backend | Cold ms | Warm Median ms | Warm p95 ms | Parity |"
        );
        let _ = writeln!(&mut out, "| --- | ---: | ---: | ---: | --- |");
        for result in &workload.results {
            let _ = writeln!(
                &mut out,
                "| {} | {:.4} | {:.4} | {:.4} | {} |",
                result.backend.label(),
                result.cold_ms,
                result.warm.median_ms,
                result.warm.p95_ms,
                if result.parity_pass { "pass" } else { "fail" }
            );
        }
        let _ = writeln!(&mut out, "");
    }

    out
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create parent directory {}: {}",
                parent.display(),
                e
            )
        })?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let cfg = parse_config()?;
    let workloads = load_workloads(&cfg)?;

    let jit_config = PrototypeJitConfig {
        metric_compile_threshold: cfg.metric_compile_threshold,
        prune_compile_threshold: cfg.prune_compile_threshold,
    };

    let mut workload_results = Vec::with_capacity(workloads.len());
    for workload in &workloads {
        let baseline_backend = BackendInstance::new(BackendKind::Interpreter, jit_config);
        let expected_capture = execute_all_flows(&workload.program, baseline_backend.as_backend())?;

        let mut backend_results = Vec::new();
        for backend in BackendKind::all() {
            let result = run_backend_measurement(
                workload,
                backend,
                jit_config,
                cfg.repeats,
                cfg.warmup_runs,
                &expected_capture,
            )?;
            backend_results.push(result);
        }

        workload_results.push(WorkloadResult {
            workload: workload.name.clone(),
            file: workload.file.clone(),
            flow_count: workload.program.flows.len(),
            results: backend_results,
        });
    }

    let report = MatrixReport {
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("system clock error: {}", e))?
            .as_secs(),
        repeats: cfg.repeats,
        warmup_runs: cfg.warmup_runs,
        metric_compile_threshold: cfg.metric_compile_threshold,
        prune_compile_threshold: cfg.prune_compile_threshold,
        workload_count: workload_results.len(),
        aggregate: build_aggregate(&workload_results),
        workloads: workload_results,
    };

    ensure_parent_dir(&cfg.output_json)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize JSON report: {}", e))?;
    fs::write(&cfg.output_json, json)
        .map_err(|e| format!("failed to write {}: {}", cfg.output_json.display(), e))?;

    ensure_parent_dir(&cfg.output_markdown)?;
    let markdown = render_markdown(&report);
    fs::write(&cfg.output_markdown, markdown)
        .map_err(|e| format!("failed to write {}: {}", cfg.output_markdown.display(), e))?;

    println!(
        "Wrote backend matrix reports:\n- {}\n- {}",
        cfg.output_markdown.display(),
        cfg.output_json.display()
    );
    Ok(())
}
