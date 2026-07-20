//! AST → canonical source printing for expressions, actions, and patterns.
//!
//! The AST carries no source spans, so structured editors (LSP refactors, the
//! browser composer) need a way to render AST fragments back to text. Output
//! is canonical-style and re-parseable; extra parentheses are inserted
//! conservatively rather than reproducing the original text exactly.

use crate::ast::{ActionStmt, BinaryOp, CallArg, ExprAst, PatternItem, UnaryOp, VarianceSpec};

/// Formats an f64 as a Grafial Real literal that round-trips.
pub fn number_to_source(v: f64) -> String {
    let s = format!("{}", v);
    if s.contains('.')
        || s.contains('e')
        || s.contains('E')
        || s.contains("inf")
        || s.contains("NaN")
    {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Operator precedence for parenthesization (higher binds tighter).
fn prec(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Or => 1,
        BinaryOp::And => 2,
        BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
            3
        }
        BinaryOp::Add | BinaryOp::Sub => 4,
        BinaryOp::Mul | BinaryOp::Div => 5,
    }
}

fn op_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Eq => "==",
        BinaryOp::Ne => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
    }
}

/// Renders an expression as canonical source.
pub fn expr_to_source(expr: &ExprAst) -> String {
    print_expr(expr, 0)
}

fn print_expr(expr: &ExprAst, parent_prec: u8) -> String {
    match expr {
        ExprAst::Number(v) => number_to_source(*v),
        ExprAst::Bool(b) => b.to_string(),
        ExprAst::Var(name) => name.clone(),
        ExprAst::Field { target, field } => format!("{}.{}", print_expr(target, 6), field),
        ExprAst::Call { name, args } if name == "E" && args.len() == 1 => {
            // Expectation sugar: E[A.attr]
            match &args[0] {
                CallArg::Positional(inner) => format!("E[{}]", print_expr(inner, 0)),
                CallArg::Named { .. } => print_call(name, args),
            }
        }
        ExprAst::Call { name, args } => print_call(name, args),
        ExprAst::Unary { op, expr } => {
            let inner = print_expr(expr, 6);
            match op {
                UnaryOp::Neg => format!("-{}", inner),
                UnaryOp::Not => format!("not {}", inner),
            }
        }
        ExprAst::Binary { op, left, right } => {
            let p = prec(*op);
            // Conservative: parenthesize children at equal precedence too,
            // except the left side of an associative chain.
            let l = print_expr(left, p - if is_associative(*op) { 1 } else { 0 });
            let r = print_expr(right, p);
            let s = format!("{} {} {}", l, op_str(*op), r);
            if p <= parent_prec {
                format!("({})", s)
            } else {
                s
            }
        }
        ExprAst::Exists {
            pattern,
            where_expr,
            negated,
        } => {
            let mut s = String::new();
            if *negated {
                s.push_str("not ");
            }
            s.push_str("exists ");
            s.push_str(&pattern_to_source(pattern));
            if let Some(w) = where_expr {
                s.push_str(" where ");
                s.push_str(&print_expr(w, 0));
            }
            s
        }
    }
}

fn is_associative(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Mul | BinaryOp::And | BinaryOp::Or
    )
}

fn print_call(name: &str, args: &[CallArg]) -> String {
    let rendered: Vec<String> = args
        .iter()
        .map(|a| match a {
            CallArg::Positional(e) => print_expr(e, 0),
            CallArg::Named { name, value } => format!("{}={}", name, print_expr(value, 0)),
        })
        .collect();
    format!("{}({})", name, rendered.join(", "))
}

/// Renders a single pattern item: `(A:Person)-[ab:REL]->(B:Person)`.
pub fn pattern_to_source(pattern: &PatternItem) -> String {
    format!(
        "({}:{})-[{}:{}]->({}:{})",
        pattern.src.var,
        pattern.src.label,
        pattern.edge.var,
        pattern.edge.ty,
        pattern.dst.var,
        pattern.dst.label
    )
}

/// Renders an action statement in canonical style (one line, no indent).
pub fn action_to_source(action: &ActionStmt) -> String {
    match action {
        ActionStmt::Let { name, expr } => format!("let {} = {}", name, expr_to_source(expr)),
        ActionStmt::NonBayesianNudge {
            node_var,
            attr,
            expr,
            variance,
        } => {
            let variance_str = match variance {
                None => String::new(),
                Some(VarianceSpec::Preserve) => " variance=preserve".into(),
                Some(VarianceSpec::Increase { factor: None }) => " variance=increase".into(),
                Some(VarianceSpec::Increase { factor: Some(f) }) => {
                    format!(" variance=increase(factor={})", number_to_source(*f))
                }
                Some(VarianceSpec::Decrease { factor: None }) => " variance=decrease".into(),
                Some(VarianceSpec::Decrease { factor: Some(f) }) => {
                    format!(" variance=decrease(factor={})", number_to_source(*f))
                }
            };
            format!(
                "non_bayesian_nudge {}.{} to {}{}",
                node_var,
                attr,
                expr_to_source(expr),
                variance_str
            )
        }
        ActionStmt::SoftUpdate {
            node_var,
            attr,
            expr,
            precision,
            count,
        } => {
            let mut s = format!("{}.{} ~= {}", node_var, attr, expr_to_source(expr));
            if let Some(p) = precision {
                s.push_str(&format!(" precision={}", number_to_source(*p)));
            }
            if let Some(c) = count {
                s.push_str(&format!(" count={}", number_to_source(*c)));
            }
            s
        }
        ActionStmt::DeleteEdge {
            edge_var,
            confidence,
        } => match confidence {
            Some(c) => format!("delete {} confidence={}", edge_var, c),
            None => format!("delete {}", edge_var),
        },
        ActionStmt::SuppressEdge { edge_var, weight } => match weight {
            Some(w) => format!("suppress {} weight={}", edge_var, number_to_source(*w)),
            None => format!("suppress {}", edge_var),
        },
        // Legacy statements: canonical equivalents.
        ActionStmt::SetExpectation {
            node_var,
            attr,
            expr,
        } => format!(
            "non_bayesian_nudge {}.{} to {} variance=preserve",
            node_var,
            attr,
            expr_to_source(expr)
        ),
        ActionStmt::ForceAbsent { edge_var } => format!("delete {} confidence=high", edge_var),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    /// Round-trip: parse a rule containing the expression, print it, reparse,
    /// and check the ASTs match.
    fn roundtrip_where(expr_src: &str) {
        let program = format!(
            r#"
schema S {{ node N {{ x: Real }} edge R {{ }} }}
belief_model M on S {{
  node N {{ x ~ Gaussian(mean=0.0, precision=1.0) }}
  edge R {{ exist ~ Bernoulli(prior=0.5, weight=2.0) }}
}}
rule Rl on M {{
  pattern (A:N)-[ab:R]->(B:N)
  where {expr_src}
  action {{ let y = 1.0 }}
  mode: for_each
}}
"#
        );
        let ast = parse_program(&program)
            .unwrap_or_else(|e| panic!("input parse failed for '{}': {}", expr_src, e));
        let expr = ast.rules[0].where_expr.clone().expect("where expr");
        let printed = expr_to_source(&expr);

        let reprogram = program.replace(expr_src, &printed);
        let reast = parse_program(&reprogram)
            .unwrap_or_else(|e| panic!("printed form failed to parse: '{}': {}", printed, e));
        assert_eq!(
            reast.rules[0].where_expr.as_ref().unwrap(),
            &expr,
            "AST changed after printing: '{}' -> '{}'",
            expr_src,
            printed
        );
    }

    #[test]
    fn expr_roundtrips() {
        roundtrip_where("prob(ab) >= 0.5");
        roundtrip_where("E[A.x] > E[B.x] + 0.05");
        roundtrip_where("(E[A.x] + 1.0) * 2.0 / 3.0 - -1.0 == 0.5");
        roundtrip_where("prob(ab) >= 0.5 and (E[A.x] < 1.0 or not (E[B.x] > 2.0))");
        roundtrip_where("E[A.x] - (1.0 - 2.0) > 0.0");
        roundtrip_where("degree(A, min_prob=0.5) > 1.0");
        roundtrip_where("exists (A:N)-[ax:R]->(X:N) where prob(ax) >= 0.7 and E[X.x] > 0.0");
        roundtrip_where("not exists (A:N)-[ax:R]->(X:N) where prob(ax) >= 0.5");
        roundtrip_where("winner(A, R) == B and entropy(A, R) > 1.0");
    }

    #[test]
    fn actions_roundtrip_via_reparse() {
        let program = r#"
schema S { node N { x: Real } edge R { } }
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=1.0) }
  edge R { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
rule Rl on M {
  pattern (A:N)-[ab:R]->(B:N)
  where prob(ab) >= 0.0
  action {
    let y = E[A.x] * 0.5
    non_bayesian_nudge B.x to y + 1.0 variance=preserve
    non_bayesian_nudge A.x to 0.0 variance=increase(factor=0.5)
    B.x ~= 1.0 precision=0.2 count=2.0
    delete ab confidence=high
    suppress ab weight=10.0
  }
  mode: for_each
}
"#;
        let ast = parse_program(program).expect("parse");
        let actions = &ast.rules[0].actions;

        // Print each action into a fresh rule body and reparse.
        let printed: Vec<String> = actions.iter().map(action_to_source).collect();
        let body = printed.join("\n    ");
        let reprogram = format!(
            r#"
schema S {{ node N {{ x: Real }} edge R {{ }} }}
belief_model M on S {{
  node N {{ x ~ Gaussian(mean=0.0, precision=1.0) }}
  edge R {{ exist ~ Bernoulli(prior=0.5, weight=2.0) }}
}}
rule Rl on M {{
  pattern (A:N)-[ab:R]->(B:N)
  where prob(ab) >= 0.0
  action {{
    {body}
  }}
  mode: for_each
}}
"#
        );
        let reast = parse_program(&reprogram)
            .unwrap_or_else(|e| panic!("printed actions failed to parse:\n{}\n{}", body, e));
        assert_eq!(&reast.rules[0].actions, actions);
    }

    #[test]
    fn pattern_prints_canonically() {
        let program = r#"
schema S { node N { x: Real } edge R { } }
belief_model M on S {
  node N { x ~ Gaussian(mean=0.0, precision=1.0) }
  edge R { exist ~ Bernoulli(prior=0.5, weight=2.0) }
}
rule Rl on M {
  pattern (A:N)-[ab:R]->(B:N)
  where prob(ab) >= 0.0
  action { let y = 1.0 }
  mode: for_each
}
"#;
        let ast = parse_program(program).expect("parse");
        assert_eq!(
            pattern_to_source(&ast.rules[0].patterns[0]),
            "(A:N)-[ab:R]->(B:N)"
        );
    }
}
