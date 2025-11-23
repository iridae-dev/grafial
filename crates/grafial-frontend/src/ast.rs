//! # Abstract Syntax Tree
//!
//! This module defines the Abstract Syntax Tree (AST) data structures for the Grafial DSL.
//!
//! ## Structure
//!
//! A Grafial program consists of:
//! - **Schemas**: Define node and edge types with attributes
//! - **Belief models**: Associate inference models with schemas
//! - **Evidence**: Specify observations and ground truth
//! - **Rules**: Define pattern-action transformations
//! - **Flows**: Specify graph transformation pipelines
//!
//! ## Expression AST
//!
//! Expressions support:
//! - Literals: numbers (f64), booleans
//! - Variables and field access
//! - Binary operations: arithmetic, comparison, logical
//! - Unary operations: negation, logical not
//! - Function calls with named and positional arguments
//!
//! Numbers are stored as parsed `f64` values (not strings) for performance.

/// The root of a parsed Grafial program.
///
/// A complete program consists of schemas, belief models, evidence,
/// rules for inference, and flows for graph transformations.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramAst {
    /// Schema definitions for graph structure
    pub schemas: Vec<Schema>,
    /// Belief models associating inference parameters with schemas
    pub belief_models: Vec<BeliefModel>,
    /// Evidence definitions specifying observations
    pub evidences: Vec<EvidenceDef>,
    /// Inference rules for pattern-based transformations
    pub rules: Vec<RuleDef>,
    /// Flow definitions for graph processing pipelines
    pub flows: Vec<FlowDef>,
}

/// A schema defines the structure of nodes and edges in a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// The schema name
    pub name: String,
    /// Node type definitions
    pub nodes: Vec<NodeDef>,
    /// Edge type definitions
    pub edges: Vec<EdgeDef>,
}

/// A node type definition with typed attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDef {
    /// The node type name
    pub name: String,
    /// Attribute definitions for this node type
    pub attrs: Vec<AttrDef>,
}

/// An attribute definition with name and type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrDef {
    /// The attribute name
    pub name: String,
    /// The attribute type (e.g., "Real")
    pub ty: String,
}

/// An edge type definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeDef {
    /// The edge type name
    pub name: String,
}

/// Posterior type for edge existence or node attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum PosteriorType {
    /// Gaussian posterior for continuous attributes
    Gaussian {
        /// Named parameters (e.g., "prior_mean" = 0.0)
        params: Vec<(String, f64)>,
    },
    /// Bernoulli posterior (Beta-Bernoulli) for independent edges
    Bernoulli {
        /// Named parameters (e.g., "prior" = 0.3, "pseudo_count" = 2.0)
        params: Vec<(String, f64)>,
    },
    /// Categorical posterior (Dirichlet-Categorical) for competing edges
    Categorical {
        /// Grouping direction: "source" or "destination"
        group_by: String,
        /// Prior specification: either uniform (with pseudo_count) or explicit array
        prior: CategoricalPrior,
        /// Optional category names for validation
        categories: Option<Vec<String>>,
    },
}

/// Prior specification for CategoricalPosterior.
#[derive(Debug, Clone, PartialEq)]
pub enum CategoricalPrior {
    /// Uniform prior: all α_k = pseudo_count / K
    Uniform { pseudo_count: f64 },
    /// Explicit prior: α = [α_1, α_2, ..., α_K]
    Explicit { concentrations: Vec<f64> },
}

/// Node belief declaration in a belief model.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeBeliefDecl {
    /// Node type name
    pub node_type: String,
    /// Attribute posterior declarations
    pub attrs: Vec<(String, PosteriorType)>,
}

/// Edge belief declaration in a belief model.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeBeliefDecl {
    /// Edge type name
    pub edge_type: String,
    /// Posterior type for edge existence
    pub exist: PosteriorType,
}

/// A belief model associates inference parameters with a schema.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefModel {
    /// The belief model name
    pub name: String,
    /// The schema this model operates on
    pub on_schema: String,
    /// Node belief declarations
    pub nodes: Vec<NodeBeliefDecl>,
    /// Edge belief declarations
    pub edges: Vec<EdgeBeliefDecl>,
    /// Raw source text of the model body (preserved for backward compatibility)
    pub body_src: String,
}

/// Evidence mode for edge observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceMode {
    /// Independent edge: edge exists
    Present,
    /// Independent edge: edge does not exist
    Absent,
    /// Competing edge: this category was chosen
    Chosen,
    /// Competing edge: this category was not chosen (rare)
    Unchosen,
    /// Competing edge: deterministic choice (force)
    ForcedChoice,
}

/// An evidence observation statement.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserveStmt {
    /// Observe an edge with a specific mode
    Edge {
        /// Edge type name
        edge_type: String,
        /// Source node reference (type and label)
        src: (String, String),
        /// Destination node reference (type and label)
        dst: (String, String),
        /// Evidence mode
        mode: EvidenceMode,
    },
    /// Observe a node attribute value
    Attribute {
        /// Node type and label
        node: (String, String),
        /// Attribute name
        attr: String,
        /// Observed value
        value: f64,
        /// Optional per-observation precision (τ_obs)
        precision: Option<f64>,
    },
}

/// An evidence definition specifies observations for a belief model.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceDef {
    /// The evidence name
    pub name: String,
    /// The belief model this evidence applies to
    pub on_model: String,
    /// Parsed observation statements
    pub observations: Vec<ObserveStmt>,
    /// Raw source text of the evidence body (preserved for backward compatibility)
    pub body_src: String,
}

/// A rule defines pattern-based transformations on belief graphs.
///
/// Rules match patterns in the graph, evaluate a where clause, and
/// execute actions to update the graph state.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleDef {
    /// The rule name
    pub name: String,
    /// The belief model this rule operates on
    pub on_model: String,
    /// Graph patterns to match
    pub patterns: Vec<PatternItem>,
    /// Optional where clause for filtering matches
    pub where_expr: Option<ExprAst>,
    /// Actions to execute for each match
    pub actions: Vec<ActionStmt>,
    /// Execution mode (e.g., "for_each")
    pub mode: Option<String>,
}

/// A single pattern item representing a directed edge with typed nodes.
///
/// Patterns have the form: `(src:Label)-[edge:Type]->(dst:Label)`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternItem {
    /// Source node pattern
    pub src: NodePattern,
    /// Edge pattern
    pub edge: EdgePattern,
    /// Destination node pattern
    pub dst: NodePattern,
}

/// A node pattern binds a variable to a node with a specific label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodePattern {
    /// The variable name for this node
    pub var: String,
    /// The node type label
    pub label: String,
}

/// An edge pattern binds a variable to an edge with a specific type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgePattern {
    /// The variable name for this edge
    pub var: String,
    /// The edge type
    pub ty: String,
}

/// A flow defines a graph transformation pipeline with metrics and exports.
///
/// Flows transform graphs through sequences of operations (rules, pruning)
/// and compute metrics on the results.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowDef {
    /// The flow name
    pub name: String,
    /// The belief model this flow operates on
    pub on_model: String,
    /// Graph definitions and transformations
    pub graphs: Vec<GraphDef>,
    /// Metric computations
    pub metrics: Vec<MetricDef>,
    /// Export declarations
    pub exports: Vec<ExportDef>,
    /// Metric export declarations
    pub metric_exports: Vec<MetricExportDef>,
    /// Metric import declarations
    pub metric_imports: Vec<MetricImportDef>,
}

/// A graph definition binds a name to a graph expression.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDef {
    /// The graph variable name
    pub name: String,
    /// The graph expression (from evidence or pipeline)
    pub expr: GraphExpr,
}

/// A graph expression creates or transforms a graph.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphExpr {
    /// Load a graph from evidence
    FromEvidence {
        /// The evidence name to load from
        evidence: String,
    },
    /// Import a graph exported from another flow
    FromGraph {
        /// The export alias or snapshot name to import
        alias: String,
    },
    /// Apply a pipeline of transformations to a graph
    Pipeline {
        /// The starting graph variable
        start: String,
        /// The sequence of transformations to apply
        transforms: Vec<Transform>,
    },
}

/// A graph transformation operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    /// Apply a rule to transform the graph
    ApplyRule {
        /// The rule name to apply
        rule: String,
    },
    /// Apply multiple rules sequentially
    ApplyRuleset {
        /// The rule names to apply in order
        rules: Vec<String>,
    },
    /// Save a snapshot of the current graph state
    Snapshot {
        /// The snapshot name (for later retrieval)
        name: String,
    },
    /// Remove edges matching a predicate
    PruneEdges {
        /// The edge type to prune
        edge_type: String,
        /// The predicate expression to evaluate
        predicate: ExprAst,
    },
}

/// A metric definition computes a scalar value from a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDef {
    /// The metric variable name
    pub name: String,
    /// The metric expression to evaluate
    pub expr: ExprAst,
}

/// An export definition specifies a graph to export with an alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDef {
    /// The graph variable to export
    pub graph: String,
    /// The export alias
    pub alias: String,
}

/// A metric export definition (export a metric by alias).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricExportDef {
    /// The metric variable to export
    pub metric: String,
    /// The export alias (external name)
    pub alias: String,
}

/// A metric import definition (bring prior exported metric into local name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricImportDef {
    /// The source alias to import from prior context
    pub source_alias: String,
    /// The local metric variable name
    pub local_name: String,
}

/// An action statement that modifies the belief graph.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionStmt {
    /// Bind a local variable to an expression result
    Let {
        /// The variable name
        name: String,
        /// The expression to evaluate
        expr: ExprAst,
    },
    /// Set the expected value of a node attribute
    SetExpectation {
        /// The node variable
        node_var: String,
        /// The attribute name
        attr: String,
        /// The new expectation value
        expr: ExprAst,
    },
    /// Force an edge to be absent (near-zero probability)
    ForceAbsent {
        /// The edge variable
        edge_var: String,
    },
    /// Non-Bayesian nudge: set mean to expr with variance strategy
    NonBayesianNudge {
        node_var: String,
        attr: String,
        expr: ExprAst,
        variance: Option<VarianceSpec>,
    },
    /// Bayesian soft update: ~= value with precision and optional count
    SoftUpdate {
        node_var: String,
        attr: String,
        expr: ExprAst,
        precision: Option<f64>,
        count: Option<f64>,
    },
    /// Delete edge with optional confidence preset
    DeleteEdge {
        edge_var: String,
        confidence: Option<String>,
    },
    /// Suppress edge with optional weight
    SuppressEdge {
        edge_var: String,
        weight: Option<f64>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum VarianceSpec {
    Preserve,
    Increase { factor: Option<f64> },
    Decrease { factor: Option<f64> },
}

/// An expression in the Grafial expression language.
///
/// Supports arithmetic, comparisons, logical operations, and special
/// functions for probabilistic reasoning (prob, degree, E).
#[derive(Debug, Clone, PartialEq)]
pub enum ExprAst {
    /// Numeric literal (f64)
    Number(f64),
    /// Boolean literal
    Bool(bool),
    /// Variable reference
    Var(String),
    /// Field access (target.field)
    Field {
        /// The target expression
        target: Box<ExprAst>,
        /// The field name
        field: String,
    },
    /// Function call with arguments
    Call {
        /// The function name
        name: String,
        /// The function arguments
        args: Vec<CallArg>,
    },
    /// Unary operation (negation, logical not)
    Unary {
        /// The unary operator
        op: UnaryOp,
        /// The operand expression
        expr: Box<ExprAst>,
    },
    /// Binary operation (arithmetic, comparison, logical)
    Binary {
        /// The binary operator
        op: BinaryOp,
        /// The left operand
        left: Box<ExprAst>,
        /// The right operand
        right: Box<ExprAst>,
    },
    /// Exists subquery: checks if a pattern exists in the graph
    Exists {
        /// The pattern to search for
        pattern: PatternItem,
        /// Optional where clause to filter matches
        where_expr: Option<Box<ExprAst>>,
        /// Whether this is negated (not exists)
        negated: bool,
    },
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation (-)
    Neg,
    /// Logical negation (not)
    Not,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Equality (==)
    Eq,
    /// Inequality (!=)
    Ne,
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
    /// Logical and
    And,
    /// Logical or
    Or,
}

/// A function call argument (positional or named).
#[derive(Debug, Clone, PartialEq)]
pub enum CallArg {
    /// A positional argument
    Positional(ExprAst),
    /// A named argument (name=value)
    Named {
        /// The parameter name
        name: String,
        /// The argument value
        value: ExprAst,
    },
}
