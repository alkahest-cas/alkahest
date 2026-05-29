/// Graphviz DOT emitter for symbolic expression DAGs.
///
/// Walks the expression tree rooted at `expr` and emits a `digraph` in DOT
/// format.  Shared sub-expressions (same `ExprId`) are rendered as a single
/// node with multiple incoming edges, faithfully representing the DAG
/// structure.
///
/// Pipe the output through `dot -Tpng -o graph.png` or `dot -Tsvg` to render.
use crate::kernel::expr::ExprData;
use crate::kernel::{ExprId, ExprPool};
use std::collections::HashSet;

/// Emit a Graphviz DOT string for the expression DAG rooted at `expr`.
pub fn render_dot(pool: &ExprPool, expr: ExprId) -> String {
    let mut out =
        String::from("digraph expr {\n  node [shape=box fontname=\"Courier\" fontsize=10];\n");
    let mut visited = HashSet::new();
    emit_node(pool, expr, &mut out, &mut visited);
    out.push_str("}\n");
    out
}

fn node_id(id: ExprId) -> String {
    format!("n{}", id.0)
}

fn node_label(pool: &ExprPool, id: ExprId) -> String {
    match pool.get(id) {
        ExprData::Symbol { name, .. } => format!("sym\\n{}", escape_dot(&name)),
        ExprData::Integer(n) => format!("int\\n{}", n),
        ExprData::Rational(r) => format!("rat\\n{}", r),
        ExprData::Float(f) => format!("float\\n{}", f),
        ExprData::Add(_) => "Add".to_string(),
        ExprData::Mul(_) => "Mul".to_string(),
        ExprData::Pow { .. } => "Pow".to_string(),
        ExprData::Func { name, .. } => format!("fn\\n{}", escape_dot(&name)),
        ExprData::Piecewise { .. } => "Piecewise".to_string(),
        ExprData::Predicate { kind, .. } => format!("pred\\n{}", kind),
        ExprData::Forall { .. } => "Forall".to_string(),
        ExprData::Exists { .. } => "Exists".to_string(),
        ExprData::BigO(_) => "BigO".to_string(),
        ExprData::RootSum { .. } => "RootSum".to_string(),
    }
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn emit_node(pool: &ExprPool, id: ExprId, out: &mut String, visited: &mut HashSet<u32>) {
    if !visited.insert(id.0) {
        return;
    }
    let label = node_label(pool, id);
    out.push_str(&format!("  {} [label=\"{}\"];\n", node_id(id), label));

    let children: Vec<ExprId> = match pool.get(id) {
        ExprData::Add(kids) | ExprData::Mul(kids) => kids,
        ExprData::Pow { base, exp } => vec![base, exp],
        ExprData::Func { args, .. } => args,
        ExprData::Piecewise { branches, default } => {
            let mut v: Vec<ExprId> = branches.iter().flat_map(|(c, v)| [*c, *v]).collect();
            v.push(default);
            v
        }
        ExprData::Predicate { args, .. } => args,
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => vec![var, body],
        ExprData::BigO(inner) => vec![inner],
        _ => vec![],
    };

    for child in &children {
        emit_node(pool, *child, out, visited);
        out.push_str(&format!("  {} -> {};\n", node_id(id), node_id(*child)));
    }
}
