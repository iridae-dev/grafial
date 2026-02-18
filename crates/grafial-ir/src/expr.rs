//! Expression IR for Grafial.
//!
//! This module defines the canonical expression representation used by
//! the IR pipeline and provides lossless conversion to/from frontend AST.

use grafial_frontend::ast::{BinaryOp, CallArg, ExprAst, PatternItem, UnaryOp};

/// Function call argument in IR form.
#[derive(Debug, Clone, PartialEq)]
pub enum CallArgIR {
    /// Positional argument.
    Positional(ExprIR),
    /// Named argument (`name=value`).
    Named { name: String, value: ExprIR },
}

impl CallArgIR {
    /// Convert this IR call argument back to frontend AST.
    pub fn to_ast(&self) -> CallArg {
        match self {
            Self::Positional(expr) => CallArg::Positional(expr.to_ast()),
            Self::Named { name, value } => CallArg::Named {
                name: name.clone(),
                value: value.to_ast(),
            },
        }
    }
}

impl From<&CallArg> for CallArgIR {
    fn from(value: &CallArg) -> Self {
        match value {
            CallArg::Positional(expr) => Self::Positional(ExprIR::from(expr)),
            CallArg::Named { name, value } => Self::Named {
                name: name.clone(),
                value: ExprIR::from(value),
            },
        }
    }
}

/// Unary operators in IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpIR {
    Neg,
    Not,
}

impl From<UnaryOp> for UnaryOpIR {
    fn from(value: UnaryOp) -> Self {
        match value {
            UnaryOp::Neg => Self::Neg,
            UnaryOp::Not => Self::Not,
        }
    }
}

impl UnaryOpIR {
    /// Convert this IR unary operator back to frontend AST.
    pub fn to_ast(self) -> UnaryOp {
        match self {
            Self::Neg => UnaryOp::Neg,
            Self::Not => UnaryOp::Not,
        }
    }
}

/// Binary operators in IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpIR {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl From<BinaryOp> for BinaryOpIR {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
            BinaryOp::Eq => Self::Eq,
            BinaryOp::Ne => Self::Ne,
            BinaryOp::Lt => Self::Lt,
            BinaryOp::Le => Self::Le,
            BinaryOp::Gt => Self::Gt,
            BinaryOp::Ge => Self::Ge,
            BinaryOp::And => Self::And,
            BinaryOp::Or => Self::Or,
        }
    }
}

impl BinaryOpIR {
    /// Convert this IR binary operator back to frontend AST.
    pub fn to_ast(self) -> BinaryOp {
        match self {
            Self::Add => BinaryOp::Add,
            Self::Sub => BinaryOp::Sub,
            Self::Mul => BinaryOp::Mul,
            Self::Div => BinaryOp::Div,
            Self::Eq => BinaryOp::Eq,
            Self::Ne => BinaryOp::Ne,
            Self::Lt => BinaryOp::Lt,
            Self::Le => BinaryOp::Le,
            Self::Gt => BinaryOp::Gt,
            Self::Ge => BinaryOp::Ge,
            Self::And => BinaryOp::And,
            Self::Or => BinaryOp::Or,
        }
    }
}

/// Expression IR.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprIR {
    Number(f64),
    Bool(bool),
    Var(String),
    Field {
        target: Box<ExprIR>,
        field: String,
    },
    Call {
        name: String,
        args: Vec<CallArgIR>,
    },
    Unary {
        op: UnaryOpIR,
        expr: Box<ExprIR>,
    },
    Binary {
        op: BinaryOpIR,
        left: Box<ExprIR>,
        right: Box<ExprIR>,
    },
    Exists {
        pattern: PatternItem,
        where_expr: Option<Box<ExprIR>>,
        negated: bool,
    },
}

impl ExprIR {
    /// Convert this IR expression back to frontend AST.
    pub fn to_ast(&self) -> ExprAst {
        match self {
            Self::Number(v) => ExprAst::Number(*v),
            Self::Bool(v) => ExprAst::Bool(*v),
            Self::Var(v) => ExprAst::Var(v.clone()),
            Self::Field { target, field } => ExprAst::Field {
                target: Box::new(target.to_ast()),
                field: field.clone(),
            },
            Self::Call { name, args } => ExprAst::Call {
                name: name.clone(),
                args: args.iter().map(CallArgIR::to_ast).collect(),
            },
            Self::Unary { op, expr } => ExprAst::Unary {
                op: op.to_ast(),
                expr: Box::new(expr.to_ast()),
            },
            Self::Binary { op, left, right } => ExprAst::Binary {
                op: op.to_ast(),
                left: Box::new(left.to_ast()),
                right: Box::new(right.to_ast()),
            },
            Self::Exists {
                pattern,
                where_expr,
                negated,
            } => ExprAst::Exists {
                pattern: pattern.clone(),
                where_expr: where_expr.as_ref().map(|expr| Box::new(expr.to_ast())),
                negated: *negated,
            },
        }
    }
}

impl From<&ExprAst> for ExprIR {
    fn from(expr: &ExprAst) -> Self {
        match expr {
            ExprAst::Number(v) => Self::Number(*v),
            ExprAst::Bool(v) => Self::Bool(*v),
            ExprAst::Var(v) => Self::Var(v.clone()),
            ExprAst::Field { target, field } => Self::Field {
                target: Box::new(ExprIR::from(target.as_ref())),
                field: field.clone(),
            },
            ExprAst::Call { name, args } => Self::Call {
                name: name.clone(),
                args: args.iter().map(CallArgIR::from).collect(),
            },
            ExprAst::Unary { op, expr } => Self::Unary {
                op: (*op).into(),
                expr: Box::new(ExprIR::from(expr.as_ref())),
            },
            ExprAst::Binary { op, left, right } => Self::Binary {
                op: (*op).into(),
                left: Box::new(ExprIR::from(left.as_ref())),
                right: Box::new(ExprIR::from(right.as_ref())),
            },
            ExprAst::Exists {
                pattern,
                where_expr,
                negated,
            } => Self::Exists {
                pattern: pattern.clone(),
                where_expr: where_expr
                    .as_ref()
                    .map(|expr| Box::new(ExprIR::from(expr.as_ref()))),
                negated: *negated,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafial_frontend::ast::{EdgePattern, NodePattern};

    #[test]
    fn expr_ir_roundtrip_binary() {
        let ast = ExprAst::Binary {
            op: BinaryOp::Add,
            left: Box::new(ExprAst::Number(1.0)),
            right: Box::new(ExprAst::Var("x".into())),
        };

        let ir = ExprIR::from(&ast);
        let back = ir.to_ast();
        assert_eq!(ast, back);
    }

    #[test]
    fn expr_ir_roundtrip_exists() {
        let ast = ExprAst::Exists {
            pattern: PatternItem {
                src: NodePattern {
                    var: "a".into(),
                    label: "Person".into(),
                },
                edge: EdgePattern {
                    var: "e".into(),
                    ty: "KNOWS".into(),
                },
                dst: NodePattern {
                    var: "b".into(),
                    label: "Person".into(),
                },
            },
            where_expr: Some(Box::new(ExprAst::Bool(true))),
            negated: false,
        };

        let ir = ExprIR::from(&ast);
        let back = ir.to_ast();
        assert_eq!(ast, back);
    }
}
