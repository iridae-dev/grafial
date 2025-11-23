//! # Grafial Python Bindings
//!
//! Phase 1 scaffold: expose `compile()` and `Program` wrapper.

use std::sync::Arc;

use grafial_core::ExecError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::{Bound, PyObject};
use std::collections::HashMap;

/// FORCE_PRECISION constant from grafial-core (1e6)
/// Used to detect forced present/absent edges by checking Beta parameters
const FORCE_PRECISION: f64 = 1_000_000.0;

/// Python-visible Program wrapper holding an immutable AST.
///
/// Methods provide discovery of flow names, schema names, and belief models.
#[pyclass(name = "Program")]
pub struct PyProgram {
    inner: Arc<grafial_frontend::ProgramAst>,
}

#[pymethods]
impl PyProgram {
    /// List all flow names defined in the program
    pub fn get_flow_names(&self) -> Vec<String> {
        self.inner.flows.iter().map(|f| f.name.clone()).collect()
    }

    /// List all schema names defined in the program
    pub fn get_schema_names(&self) -> Vec<String> {
        self.inner.schemas.iter().map(|s| s.name.clone()).collect()
    }

    /// List all belief model names defined in the program
    pub fn get_belief_model_names(&self) -> Vec<String> {
        self.inner
            .belief_models
            .iter()
            .map(|b| b.name.clone())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(schemas={}, models={}, evidences={}, rules={}, flows={})",
            self.inner.schemas.len(),
            self.inner.belief_models.len(),
            self.inner.evidences.len(),
            self.inner.rules.len(),
            self.inner.flows.len()
        )
    }
}

/// Compile a Grafial source string into a Program.
///
/// Releases the GIL while parsing and validating.
#[pyfunction]
pub fn compile(py: Python<'_>, source: &str) -> PyResult<PyProgram> {
    // Release the GIL for potentially heavy parsing/validation
    let ast_result = py.allow_threads(|| grafial_core::parse_and_validate(source));
    match ast_result {
        Ok(ast) => Ok(PyProgram {
            inner: Arc::new(ast),
        }),
        Err(err) => Err(map_exec_error(err)),
    }
}

/// Map core ExecError to rich Python exceptions
fn map_exec_error(err: ExecError) -> PyErr {
    match err {
        ExecError::ParseError(msg) => PyValueError::new_err(format!("Parse error: {}", msg)),
        ExecError::ValidationError(msg) => PyValueError::new_err(format!("Validation error: {}", msg)),
        ExecError::Execution(msg) => PyRuntimeError::new_err(format!("Execution error: {}", msg)),
        ExecError::Numerical(msg) => PyRuntimeError::new_err(format!("Numerical error: {}", msg)),
        ExecError::Internal(msg) => PyRuntimeError::new_err(format!("Internal error: {}", msg)),
        _ => PyRuntimeError::new_err(format!("Unexpected error: {:?}", err)),
    }
}

/// Flow execution context with exported graphs and metrics.
#[pyclass(name = "Context")]
pub struct PyContext {
    graphs: HashMap<String, Py<PyBeliefGraph>>, // exported graphs
    metrics: HashMap<String, f64>,              // exported metrics
}

#[pymethods]
impl PyContext {
    #[getter]
    pub fn graphs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.graphs {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    #[getter]
    pub fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.metrics {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Get a single exported graph by name
    pub fn get_graph<'py>(&self, py: Python<'py>, name: &str) -> Option<Py<PyBeliefGraph>> {
        self.graphs.get(name).map(|g| g.clone_ref(py))
    }

    /// Get a single exported metric by name
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    fn __repr__(&self) -> String {
        format!(
            "Context(graphs={}, metrics={})",
            self.graphs.len(),
            self.metrics.len()
        )
    }
}

impl PyContext {
    fn from_flow_result(py: Python<'_>, fr: grafial_core::engine::flow_exec::FlowResult) -> PyResult<Self> {
        let mut graphs: HashMap<String, Py<PyBeliefGraph>> = HashMap::new();
        for (alias, g) in fr.exports.into_iter() {
            // Ensure no pending deltas remain for iteration/inspection
            let mut g2 = g;
            g2.ensure_owned(); // Apply any pending deltas before exposing to Python
            let handle = Py::new(py, PyBeliefGraph { inner: Arc::new(g2) })?;
            graphs.insert(alias, handle);
        }
        let metrics = fr.metric_exports;
        Ok(Self { graphs, metrics })
    }
}

/// BeliefGraph wrapper for inspection and export.
#[pyclass(name = "BeliefGraph")]
pub struct PyBeliefGraph {
    inner: Arc<grafial_core::BeliefGraph>,
}

#[pymethods]
impl PyBeliefGraph {
    /// Iterate nodes, optionally filtered by label.
    pub fn nodes(&self, label: Option<&str>) -> PyResult<Vec<PyNodeView>> {
        let mut out = Vec::new();
        for n in self.inner.nodes() {
            if let Some(l) = label {
                if n.label.as_ref() != l {
                    continue;
                }
            }
            out.push(PyNodeView {
                graph: self.inner.clone(),
                node_id: n.id.0,
                label: n.label.to_string(),
            });
        }
        Ok(out)
    }

    /// Iterate edges, optionally filtered by edge type.
    pub fn edges(&self, edge_type: Option<&str>) -> PyResult<Vec<PyEdgeView>> {
        let mut out = Vec::new();
        for e in self.inner.edges() {
            if let Some(t) = edge_type {
                if e.ty.as_ref() != t {
                    continue;
                }
            }
            out.push(PyEdgeView {
                graph: self.inner.clone(),
                edge_id: e.id.0,
                src: e.src.0,
                dst: e.dst.0,
                ty: e.ty.to_string(),
            });
        }
        Ok(out)
    }

    /// Iterate competing edge groups, optionally filtered by edge type.
    pub fn competing_groups(&self, edge_type: Option<&str>) -> PyResult<Vec<PyCompetingGroup>> {
        let mut out = Vec::new();
        for (_gid, group) in self.inner.competing_groups() {
            if let Some(t) = edge_type {
                if group.edge_type.as_str() != t {
                    continue;
                }
            }
            let probs = group.posterior.mean_probabilities();
            let categories: Vec<String> = group.categories.iter().map(|n| n.0.to_string()).collect();
            let entropy = group.posterior.entropy();
            out.push(PyCompetingGroup {
                graph: self.inner.clone(),
                source_node: group.source.0.to_string(),
                edge_type: group.edge_type.clone(),
                categories,
                probabilities: probs,
                entropy,
            });
        }
        Ok(out)
    }

    /// Export to pandas DataFrames: (nodes_df, edges_df)
    pub fn to_pandas(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        // Try import pandas
        let pandas = match PyModule::import_bound(py, "pandas") {
            Ok(m) => m,
            Err(_) => {
                return Err(PyRuntimeError::new_err(
                    "pandas is not installed. Install with: uv pip install pandas",
                ))
            }
        };

        // Collect union of attribute names
        let mut attr_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for n in self.inner.nodes() {
            for k in n.attrs.keys() {
                attr_keys.insert(k.clone());
            }
        }

        // Build nodes records
        let nodes_list = PyList::empty_bound(py);
        for n in self.inner.nodes() {
            let rec = PyDict::new_bound(py);
            rec.set_item("id", n.id.0.to_string())?;
            rec.set_item("label", n.label.as_ref())?;
            for k in &attr_keys {
                if let Some(g) = n.attrs.get(k) {
                    rec.set_item(format!("{}_mean", k), g.mean)?;
                    rec.set_item(format!("{}_var", k), 1.0 / g.precision)?;
                } else {
                    rec.set_item(format!("{}_mean", k), py.None())?;
                    rec.set_item(format!("{}_var", k), py.None())?;
                }
            }
            nodes_list.append(&rec)?;
        }

        // Build edges records
        let edges_list = PyList::empty_bound(py);
        for e in self.inner.edges() {
            let rec = PyDict::new_bound(py);
            rec.set_item("src", e.src.0.to_string())?;
            rec.set_item("dst", e.dst.0.to_string())?;
            rec.set_item("type", e.ty.as_ref())?;
            let prob = self
                .inner
                .prob_mean(e.id)
                .map_err(|err| map_exec_error(err))?;
            rec.set_item("prob", prob)?;
            // forced_state for independent edges
            let forced = match &e.exist {
                grafial_core::engine::graph::EdgePosterior::Independent(beta) => {
                    if (beta.alpha - FORCE_PRECISION).abs() < f64::EPSILON && (beta.beta - 1.0).abs() < f64::EPSILON {
                        Some("present")
                    } else if (beta.alpha - 1.0).abs() < f64::EPSILON
                        && (beta.beta - FORCE_PRECISION).abs() < f64::EPSILON
                    {
                        Some("absent")
                    } else {
                        None
                    }
                }
                _ => None,
            };
            rec.set_item("forced_state", forced)?;
            edges_list.append(&rec)?;
        }

        // DataFrames
        let nodes_df = pandas.getattr("DataFrame")?.call1((&nodes_list,))?;
        let edges_df = pandas.getattr("DataFrame")?.call1((&edges_list,))?;
        Ok((nodes_df.into_py(py), edges_df.into_py(py)))
    }

    /// Export to NetworkX Graph, filtering edges by probability threshold
    pub fn to_networkx(&self, py: Python<'_>, threshold: Option<f64>) -> PyResult<PyObject> {
        let threshold = threshold.unwrap_or(0.0);
        // Try import networkx
        let nx = match PyModule::import_bound(py, "networkx") {
            Ok(m) => m,
            Err(_) => {
                return Err(PyRuntimeError::new_err(
                    "networkx is not installed. Install with: uv pip install networkx",
                ))
            }
        };

        let g = nx.getattr("Graph")?.call0()?;

        // Add nodes with attributes
        for n in self.inner.nodes() {
            let attrs = PyDict::new_bound(py);
            attrs.set_item("label", n.label.as_ref())?;
            for (k, v) in &n.attrs {
                attrs.set_item(format!("{}_mean", k), v.mean)?;
                attrs.set_item(format!("{}_var", k), 1.0 / v.precision)?;
            }
            g.call_method1("add_node", (n.id.0.to_string(), &attrs))?;
        }

        // Add edges with attributes if prob >= threshold
        for e in self.inner.edges() {
            let prob = self
                .inner
                .prob_mean(e.id)
                .map_err(|err| map_exec_error(err))?;
            if prob < threshold {
                continue;
            }
            let attrs = PyDict::new_bound(py);
            attrs.set_item("type", e.ty.as_ref())?;
            attrs.set_item("prob", prob)?;
            let forced = match &e.exist {
                grafial_core::engine::graph::EdgePosterior::Independent(beta) => {
                    if (beta.alpha - FORCE_PRECISION).abs() < f64::EPSILON && (beta.beta - 1.0).abs() < f64::EPSILON {
                        Some("present")
                    } else if (beta.alpha - 1.0).abs() < f64::EPSILON
                        && (beta.beta - FORCE_PRECISION).abs() < f64::EPSILON
                    {
                        Some("absent")
                    } else {
                        None
                    }
                }
                _ => None,
            };
            attrs.set_item("forced_state", forced)?;
            g.call_method1(
                "add_edge",
                (e.src.0.to_string(), e.dst.0.to_string(), &attrs),
            )?;
        }

        Ok(g.into_py(py))
    }

    fn __repr__(&self) -> String {
        "BeliefGraph(<opaque>)".to_string()
    }
}

/// Read-only node view.
#[pyclass(name = "NodeView")]
pub struct PyNodeView {
    graph: Arc<grafial_core::BeliefGraph>,
    node_id: u32,
    #[pyo3(get)]
    pub label: String,
}

#[pymethods]
impl PyNodeView {
    #[getter]
    pub fn id(&self) -> String {
        self.node_id.to_string()
    }

    /// Expected value of attribute.
    #[allow(non_snake_case)]
    pub fn E(&self, attr: &str) -> PyResult<f64> {
        let nid = grafial_core::engine::graph::NodeId(self.node_id);
        self.graph
            .node(nid)
            .and_then(|n| n.attrs.get(attr).map(|g| g.mean))
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing attr '{}'", attr)))
    }

    /// Variance (1/precision) of attribute.
    #[allow(non_snake_case)]
    pub fn Var(&self, attr: &str) -> PyResult<f64> {
        let nid = grafial_core::engine::graph::NodeId(self.node_id);
        self.graph
            .node(nid)
            .and_then(|n| n.attrs.get(attr).map(|g| 1.0 / g.precision))
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing attr '{}'", attr)))
    }

    /// Whether attribute exists.
    pub fn has_attr(&self, attr: &str) -> bool {
        let nid = grafial_core::engine::graph::NodeId(self.node_id);
        self.graph
            .node(nid)
            .map(|n| n.attrs.contains_key(attr))
            .unwrap_or(false)
    }

    fn __repr__(&self) -> String {
        format!("NodeView(id={}, label={})", self.node_id, self.label)
    }
}

/// Read-only edge view.
#[pyclass(name = "EdgeView")]
pub struct PyEdgeView {
    graph: Arc<grafial_core::BeliefGraph>,
    edge_id: u32,
    src: u32,
    dst: u32,
    ty: String,
}

#[pymethods]
impl PyEdgeView {
    #[getter]
    pub fn src(&self) -> String { self.src.to_string() }

    #[getter]
    pub fn dst(&self) -> String { self.dst.to_string() }

    #[getter]
    pub fn prob(&self) -> PyResult<f64> {
        let eid = grafial_core::engine::graph::EdgeId(self.edge_id);
        self.graph
            .prob_mean(eid)
            .map_err(|e| map_exec_error(e))
    }

    #[getter]
    pub fn forced_state(&self) -> Option<String> {
        let eid = grafial_core::engine::graph::EdgeId(self.edge_id);
        if let Some(e) = self.graph.edge(eid) {
            match &e.exist {
                grafial_core::engine::graph::EdgePosterior::Independent(beta) => {
                    // Heuristic: detect forced params
                    if (beta.alpha - FORCE_PRECISION).abs() < f64::EPSILON && (beta.beta - 1.0).abs() < f64::EPSILON {
                        Some("present".to_string())
                    } else if (beta.alpha - 1.0).abs() < f64::EPSILON
                        && (beta.beta - FORCE_PRECISION).abs() < f64::EPSILON
                    {
                        Some("absent".to_string())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn is_competing(&self) -> bool {
        let eid = grafial_core::engine::graph::EdgeId(self.edge_id);
        self.graph.edge(eid).map(|e| matches!(
            e.exist,
            grafial_core::engine::graph::EdgePosterior::Competing { .. }
        )).unwrap_or(false)
    }

    pub fn is_independent(&self) -> bool {
        let eid = grafial_core::engine::graph::EdgeId(self.edge_id);
        self.graph.edge(eid).map(|e| matches!(
            e.exist,
            grafial_core::engine::graph::EdgePosterior::Independent(_)
        )).unwrap_or(false)
    }

    #[getter]
    pub fn r#type(&self) -> String { self.ty.clone() }

    fn __repr__(&self) -> String {
        format!("EdgeView(id={}, src={}, dst={}, type={})", self.edge_id, self.src, self.dst, self.ty)
    }
}

/// Competing edges group view.
#[pyclass(name = "CompetingGroup")]
pub struct PyCompetingGroup {
    graph: Arc<grafial_core::BeliefGraph>,
    #[pyo3(get)]
    pub source_node: String,
    #[pyo3(get)]
    pub edge_type: String,
    #[pyo3(get)]
    pub categories: Vec<String>,
    #[pyo3(get)]
    pub probabilities: Vec<f64>,
    #[pyo3(get)]
    pub entropy: f64,
}

#[pymethods]
impl PyCompetingGroup {
    /// Winner category by max probability; None if tied within epsilon
    pub fn winner(&self, epsilon: Option<f64>) -> Option<String> {
        let epsilon = epsilon.unwrap_or(0.01);
        // Parse source_node string back to NodeId
        let node_id = self.source_node.parse::<u32>().ok()?;
        let node = grafial_core::engine::graph::NodeId(node_id);
        self.graph
            .winner(node, &self.edge_type, epsilon)
            .map(|nid| nid.0.to_string())
    }

    pub fn prob_vector(&self) -> Vec<f64> {
        self.probabilities.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompetingGroup(source={}, type={}, categories={})",
            self.source_node,
            self.edge_type,
            self.categories.len()
        )
    }
}

/// Runtime Evidence builder from Python
#[pyclass(name = "Evidence")]
pub struct PyEvidence {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub model: String,
    observations: Vec<grafial_frontend::ast::ObserveStmt>,
}

#[pymethods]
impl PyEvidence {
    #[new]
    pub fn new(name: String, model: String) -> Self {
        Self {
            name,
            model,
            observations: Vec::new(),
        }
    }

    /// Observe independent edge presence/absence
    pub fn observe_edge(
        &mut self,
        node_type: &str,
        src_id: &str,
        edge_type: &str,
        dst_type: &str,
        dst_id: &str,
        present: bool,
    ) {
        let mode = if present {
            grafial_frontend::ast::EvidenceMode::Present
        } else {
            grafial_frontend::ast::EvidenceMode::Absent
        };
        self.observations.push(grafial_frontend::ast::ObserveStmt::Edge {
            edge_type: edge_type.to_string(),
            src: (node_type.to_string(), src_id.to_string()),
            dst: (dst_type.to_string(), dst_id.to_string()),
            mode,
        });
    }

    /// Observe a chosen competing edge category
    pub fn observe_edge_chosen(
        &mut self,
        node_type: &str,
        src_id: &str,
        edge_type: &str,
        dst_type: &str,
        dst_id: &str,
    ) {
        self.observations.push(grafial_frontend::ast::ObserveStmt::Edge {
            edge_type: edge_type.to_string(),
            src: (node_type.to_string(), src_id.to_string()),
            dst: (dst_type.to_string(), dst_id.to_string()),
            mode: grafial_frontend::ast::EvidenceMode::Chosen,
        });
    }

    /// Observe an unchosen competing edge category
    pub fn observe_edge_unchosen(
        &mut self,
        node_type: &str,
        src_id: &str,
        edge_type: &str,
        dst_type: &str,
        dst_id: &str,
    ) {
        self.observations.push(grafial_frontend::ast::ObserveStmt::Edge {
            edge_type: edge_type.to_string(),
            src: (node_type.to_string(), src_id.to_string()),
            dst: (dst_type.to_string(), dst_id.to_string()),
            mode: grafial_frontend::ast::EvidenceMode::Unchosen,
        });
    }

    /// Force a deterministic competing edge choice
    pub fn observe_edge_forced_choice(
        &mut self,
        node_type: &str,
        src_id: &str,
        edge_type: &str,
        dst_type: &str,
        dst_id: &str,
    ) {
        self.observations.push(grafial_frontend::ast::ObserveStmt::Edge {
            edge_type: edge_type.to_string(),
            src: (node_type.to_string(), src_id.to_string()),
            dst: (dst_type.to_string(), dst_id.to_string()),
            mode: grafial_frontend::ast::EvidenceMode::ForcedChoice,
        });
    }

    /// Observe a numeric node attribute value
    pub fn observe_numeric(&mut self, node_type: &str, node_id: &str, attr: &str, value: f64) {
        self.observations
            .push(grafial_frontend::ast::ObserveStmt::Attribute {
                node: (node_type.to_string(), node_id.to_string()),
                attr: attr.to_string(),
                value,
                precision: None,
            });
    }

    /// Clear all observations
    pub fn clear(&mut self) {
        self.observations.clear();
    }
}

/// Run a flow using only program-defined evidence.
///
/// Releases the GIL during execution.
#[pyfunction]
pub fn run_flow(py: Python<'_>, program: &PyProgram, flow_name: &str) -> PyResult<PyContext> {
    let result = py.allow_threads(|| grafial_core::run_flow(&program.inner, flow_name, None));
    match result {
        Ok(fr) => PyContext::from_flow_result(py, fr),
        Err(err) => Err(map_exec_error(err)),
    }
}

/// Run a flow with a prior Context (chaining flows).
///
/// Preserves exported graphs and metrics from the prior context.
/// Releases the GIL during execution.
#[pyfunction]
pub fn run_flow_with_context(
    py: Python<'_>,
    program: &PyProgram,
    flow_name: &str,
    ctx: &PyContext,
) -> PyResult<PyContext> {
    // Reconstruct a FlowResult with metric exports and exported graphs to pass as prior
    let mut prior = grafial_core::engine::flow_exec::FlowResult::default();
    // Restore metric exports
    prior.metric_exports = ctx.metrics.clone();
    // Restore exported graphs
    for (alias, pygraph) in &ctx.graphs {
        // Extract Arc<BeliefGraph> from wrapper (clone for sharing)
        let g = pygraph.borrow(py).inner.as_ref().clone();
        prior.exports.insert(alias.clone(), g);
    }

    let result = py.allow_threads(|| grafial_core::run_flow(&program.inner, flow_name, Some(&prior)));
    match result {
        Ok(fr) => PyContext::from_flow_result(py, fr),
        Err(err) => Err(map_exec_error(err)),
    }
}

/// Run a flow with runtime Evidence merged into program evidence for the same model.
///
/// Optionally accepts a prior Context for chaining. Releases the GIL during execution.
#[pyfunction]
pub fn run_flow_with_evidence(
    py: Python<'_>,
    program: &PyProgram,
    flow_name: &str,
    evidence: &PyEvidence,
    ctx: Option<&PyContext>,
) -> PyResult<PyContext> {
    // Build optional prior from Context
    let prior = if let Some(ctx) = ctx {
        let mut p = grafial_core::engine::flow_exec::FlowResult::default();
        p.metric_exports = ctx.metrics.clone();
        for (alias, pygraph) in &ctx.graphs {
            let g = pygraph.borrow(py).inner.as_ref().clone();
            p.exports.insert(alias.clone(), g);
        }
        Some(p)
    } else {
        None
    };

    // Capture runtime evidence for merging inside builder
    let ev_model = evidence.model.clone();
    let ev_obs = evidence.observations.clone();

    let builder = move |base_ev: &grafial_frontend::ast::EvidenceDef| {
        // Merge if models match; otherwise use base evidence as-is
        if base_ev.on_model == ev_model {
            let merged = grafial_frontend::ast::EvidenceDef {
                name: format!("{}_merged_runtime", base_ev.name),
                on_model: base_ev.on_model.clone(),
                observations: {
                    let mut v = base_ev.observations.clone();
                    v.extend(ev_obs.clone());
                    v
                },
                body_src: String::new(),
            };
            grafial_core::engine::evidence::build_graph_from_evidence(&merged, &program.inner)
        } else {
            grafial_core::engine::evidence::build_graph_from_evidence(base_ev, &program.inner)
        }
    };

    let result = py.allow_threads(|| match &prior {
        Some(p) => grafial_core::engine::flow_exec::run_flow_with_builder(
            &program.inner,
            flow_name,
            &builder,
            Some(p),
        ),
        None => grafial_core::engine::flow_exec::run_flow_with_builder(
            &program.inner,
            flow_name,
            &builder,
            None,
        ),
    });

    match result {
        Ok(fr) => PyContext::from_flow_result(py, fr),
        Err(err) => Err(map_exec_error(err)),
    }
}

/// grafial Python module definition
#[pymodule]
fn grafial(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(run_flow, m)?)?;
    m.add_function(wrap_pyfunction!(run_flow_with_context, m)?)?;
    m.add_function(wrap_pyfunction!(run_flow_with_evidence, m)?)?;
    m.add_class::<PyProgram>()?;
    m.add_class::<PyContext>()?;
    m.add_class::<PyBeliefGraph>()?;
    m.add_class::<PyEvidence>()?;
    m.add_class::<PyNodeView>()?;
    m.add_class::<PyEdgeView>()?;
    m.add_class::<PyCompetingGroup>()?;
    Ok(())
}
