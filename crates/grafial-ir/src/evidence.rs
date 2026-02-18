//! Evidence IR for Grafial.
//!
//! Provides a typed observation representation decoupled from parser AST nodes.

use grafial_frontend::ast::{EvidenceDef, EvidenceMode, ObserveStmt};

/// Node reference used in evidence observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRefIR {
    pub node_type: String,
    pub label: String,
}

impl NodeRefIR {
    fn to_ast_tuple(&self) -> (String, String) {
        (self.node_type.clone(), self.label.clone())
    }
}

impl From<&(String, String)> for NodeRefIR {
    fn from(value: &(String, String)) -> Self {
        Self {
            node_type: value.0.clone(),
            label: value.1.clone(),
        }
    }
}

/// Evidence mode in IR form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceModeIR {
    Present,
    Absent,
    Chosen,
    Unchosen,
    ForcedChoice,
}

impl From<EvidenceMode> for EvidenceModeIR {
    fn from(value: EvidenceMode) -> Self {
        match value {
            EvidenceMode::Present => Self::Present,
            EvidenceMode::Absent => Self::Absent,
            EvidenceMode::Chosen => Self::Chosen,
            EvidenceMode::Unchosen => Self::Unchosen,
            EvidenceMode::ForcedChoice => Self::ForcedChoice,
        }
    }
}

impl EvidenceModeIR {
    fn to_ast(self) -> EvidenceMode {
        match self {
            Self::Present => EvidenceMode::Present,
            Self::Absent => EvidenceMode::Absent,
            Self::Chosen => EvidenceMode::Chosen,
            Self::Unchosen => EvidenceMode::Unchosen,
            Self::ForcedChoice => EvidenceMode::ForcedChoice,
        }
    }
}

/// Observation statement in IR form.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserveStmtIR {
    Edge {
        edge_type: String,
        src: NodeRefIR,
        dst: NodeRefIR,
        mode: EvidenceModeIR,
    },
    Attribute {
        node: NodeRefIR,
        attr: String,
        value: f64,
        precision: Option<f64>,
    },
}

impl ObserveStmtIR {
    /// Convert this IR observation back to frontend AST.
    pub fn to_ast(&self) -> ObserveStmt {
        match self {
            Self::Edge {
                edge_type,
                src,
                dst,
                mode,
            } => ObserveStmt::Edge {
                edge_type: edge_type.clone(),
                src: src.to_ast_tuple(),
                dst: dst.to_ast_tuple(),
                mode: mode.to_ast(),
            },
            Self::Attribute {
                node,
                attr,
                value,
                precision,
            } => ObserveStmt::Attribute {
                node: node.to_ast_tuple(),
                attr: attr.clone(),
                value: *value,
                precision: *precision,
            },
        }
    }
}

impl From<&ObserveStmt> for ObserveStmtIR {
    fn from(value: &ObserveStmt) -> Self {
        match value {
            ObserveStmt::Edge {
                edge_type,
                src,
                dst,
                mode,
            } => Self::Edge {
                edge_type: edge_type.clone(),
                src: NodeRefIR::from(src),
                dst: NodeRefIR::from(dst),
                mode: mode.clone().into(),
            },
            ObserveStmt::Attribute {
                node,
                attr,
                value,
                precision,
            } => Self::Attribute {
                node: NodeRefIR::from(node),
                attr: attr.clone(),
                value: *value,
                precision: *precision,
            },
        }
    }
}

/// Evidence definition in IR form.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceIR {
    pub name: String,
    pub on_model: String,
    pub observations: Vec<ObserveStmtIR>,
    pub body_src: String,
}

impl EvidenceIR {
    /// Convert this IR evidence back to frontend AST.
    pub fn to_ast(&self) -> EvidenceDef {
        EvidenceDef {
            name: self.name.clone(),
            on_model: self.on_model.clone(),
            observations: self
                .observations
                .iter()
                .map(ObserveStmtIR::to_ast)
                .collect(),
            body_src: self.body_src.clone(),
        }
    }
}

impl From<&EvidenceDef> for EvidenceIR {
    fn from(value: &EvidenceDef) -> Self {
        Self {
            name: value.name.clone(),
            on_model: value.on_model.clone(),
            observations: value.observations.iter().map(ObserveStmtIR::from).collect(),
            body_src: value.body_src.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_ir_roundtrip() {
        let ast = EvidenceDef {
            name: "Ev1".into(),
            on_model: "M".into(),
            observations: vec![
                ObserveStmt::Attribute {
                    node: ("Person".into(), "Alice".into()),
                    attr: "score".into(),
                    value: 1.0,
                    precision: Some(2.0),
                },
                ObserveStmt::Edge {
                    edge_type: "KNOWS".into(),
                    src: ("Person".into(), "Alice".into()),
                    dst: ("Person".into(), "Bob".into()),
                    mode: EvidenceMode::Present,
                },
            ],
            body_src: "raw".into(),
        };

        let ir = EvidenceIR::from(&ast);
        let back = ir.to_ast();
        assert_eq!(ast, back);
    }
}
