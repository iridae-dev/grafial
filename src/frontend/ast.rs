//! # Abstract Syntax Tree
//!
//! This module defines the Abstract Syntax Tree (AST) data structures for the Baygraph DSL.
//!
//! ## Structure
//!
//! A Baygraph program consists of:
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

/// The root of a parsed Baygraph program.
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

/// A belief model associates inference parameters with a schema.
///
/// The body source is preserved as text for future processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefModel {
    /// The belief model name
    pub name: String,
    /// The schema this model operates on
    pub on_schema: String,
    /// Raw source text of the model body
    pub body_src: String,
}

/// An evidence definition specifies observations for a belief model.
///
/// The body source is preserved as text for future processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceDef {
    /// The evidence name
    pub name: String,
    /// The belief model this evidence applies to
    pub on_model: String,
    /// Raw source text of the evidence body
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
        evidence: String
    },
    /// Apply a pipeline of transformations to a graph
    Pipeline {
        /// The starting graph variable
        start: String,
        /// The sequence of transformations to apply
        transforms: Vec<Transform>
    },
}

/// A graph transformation operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    /// Apply a rule to transform the graph
    ApplyRule {
        /// The rule name to apply
        rule: String
    },
    /// Remove edges matching a predicate
    PruneEdges {
        /// The edge type to prune
        edge_type: String,
        /// The predicate expression to evaluate
        predicate: ExprAst
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
        expr: ExprAst
    },
    /// Set the expected value of a node attribute
    SetExpectation {
        /// The node variable
        node_var: String,
        /// The attribute name
        attr: String,
        /// The new expectation value
        expr: ExprAst
    },
    /// Force an edge to be absent (near-zero probability)
    ForceAbsent {
        /// The edge variable
        edge_var: String
    },
}

/// An expression in the Baygraph expression language.
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
        field: String
    },
    /// Function call with arguments
    Call {
        /// The function name
        name: String,
        /// The function arguments
        args: Vec<CallArg>
    },
    /// Unary operation (negation, logical not)
    Unary {
        /// The unary operator
        op: UnaryOp,
        /// The operand expression
        expr: Box<ExprAst>
    },
    /// Binary operation (arithmetic, comparison, logical)
    Binary {
        /// The binary operator
        op: BinaryOp,
        /// The left operand
        left: Box<ExprAst>,
        /// The right operand
        right: Box<ExprAst>
    },
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation (-)
    Neg,
    /// Logical negation (not)
    Not
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
        value: ExprAst
    },
}
