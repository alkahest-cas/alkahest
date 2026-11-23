//! Template IR (R1): a fixed binary tree of operator slots.
//!
//! A [`Template`] fixes three things: the tree **shape** (which slots are
//! leaves, inner unary nodes and binary nodes), the operator **tables** each
//! slot chooses from, and the dimension `d`. A *candidate* is then an opcode
//! vector `e` (one index per slot into that slot's table) plus a parameter
//! vector `θ`.
//!
//! * A **leaf** slot computes `Σᵢ αᵢ φ(xᵢ) + β` over all `d` coordinates,
//!   with `φ` from the unary table. It is evaluated as one contracted node,
//!   never expanded into `d` DAG nodes. Its parameters are `α₀ … α_{d−1}, β`.
//! * An **inner unary** slot computes `α·op(s) + β` (parameters `α, β`).
//! * A **binary** slot computes `a op b` and has no parameters.
//!
//! Slots are numbered in **in-order** (left subtree, node, right subtree),
//! which is the order the FEX reference implementation [Liang & Yang] uses
//! for its action vectors, so an action vector from that code is an opcode
//! vector here when the tables are [`super::ops::PAPER_UNARY`] /
//! [`super::ops::PAPER_BINARY`]. Parameters are laid out slot by slot in the
//! same order.

use crate::kernel::{ExprId, ExprPool};

use super::error::FexError;
use super::ops::{BinaryOp, UnaryOp};

/// The largest tree accepted, in nodes. Depth-4 binary trees have 31 nodes.
pub const MAX_TEMPLATE_NODES: usize = 127;

/// What a slot is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SlotKind {
    /// A leaf `Σᵢ αᵢ φ(xᵢ) + β`.
    Leaf,
    /// An inner unary node `α·op(s) + β`.
    Unary,
    /// A binary node `a op b`.
    Binary,
}

/// One slot of a template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotInfo {
    /// Leaf, inner unary or binary.
    pub kind: SlotKind,
    /// Offset of this slot's parameters in `θ`.
    pub param_offset: usize,
    /// Number of parameters (`d + 1`, `2` or `0`).
    pub n_params: usize,
}

/// A node of the evaluation plan (post-order: children before parents).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanNode {
    pub(crate) kind: SlotKind,
    /// Slot index (into the opcode vector).
    pub(crate) slot: usize,
    /// Left / only child (plan index), when not a leaf.
    pub(crate) a: usize,
    /// Right child (plan index), for binary nodes.
    pub(crate) b: usize,
    /// Offset of the slot's parameters.
    pub(crate) poff: usize,
}

#[derive(Clone, Debug)]
enum ShapeNode {
    Leaf,
    Unary(Box<ShapeNode>),
    Binary(Box<ShapeNode>, Box<ShapeNode>),
}

impl ShapeNode {
    fn count(&self) -> usize {
        match self {
            ShapeNode::Leaf => 1,
            ShapeNode::Unary(c) => 1 + c.count(),
            ShapeNode::Binary(l, r) => 1 + l.count() + r.count(),
        }
    }

    fn spec(&self, out: &mut String) {
        match self {
            ShapeNode::Leaf => out.push('L'),
            ShapeNode::Unary(c) => {
                out.push_str("U(");
                c.spec(out);
                out.push(')');
            }
            ShapeNode::Binary(l, r) => {
                out.push_str("B(");
                l.spec(out);
                out.push(',');
                r.spec(out);
                out.push(')');
            }
        }
    }
}

struct ShapeParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl ShapeParser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), FexError> {
        self.skip_ws();
        if self.i < self.s.len() && self.s[self.i] == c {
            self.i += 1;
            Ok(())
        } else {
            Err(FexError::Shape {
                detail: format!("expected '{}' at position {}", c as char, self.i),
            })
        }
    }

    fn node(&mut self, depth: usize) -> Result<ShapeNode, FexError> {
        if depth > MAX_TEMPLATE_NODES {
            return Err(FexError::Shape {
                detail: "nesting too deep".into(),
            });
        }
        self.skip_ws();
        let c = self.s.get(self.i).copied().ok_or_else(|| FexError::Shape {
            detail: "unexpected end of specification".into(),
        })?;
        self.i += 1;
        match c.to_ascii_uppercase() {
            b'L' => Ok(ShapeNode::Leaf),
            b'U' => {
                self.expect(b'(')?;
                let child = self.node(depth + 1)?;
                self.expect(b')')?;
                Ok(ShapeNode::Unary(Box::new(child)))
            }
            b'B' => {
                self.expect(b'(')?;
                let l = self.node(depth + 1)?;
                self.expect(b',')?;
                let r = self.node(depth + 1)?;
                self.expect(b')')?;
                Ok(ShapeNode::Binary(Box::new(l), Box::new(r)))
            }
            other => Err(FexError::Shape {
                detail: format!(
                    "unexpected '{}' at position {}; use L, U(..) or B(..,..)",
                    other as char,
                    self.i - 1
                ),
            }),
        }
    }
}

/// Expand a preset name into a shape specification, or return the input.
///
/// Presets mirror the reference implementation's `--tree` choices:
/// `depth1 = B(L,L)`, `depth2_sub = U(B(L,L))`, `depth2_rml = B(U(L),U(L))`,
/// `depth2 = B(U(B(L,L)),U(B(L,L)))`, and `depth3` (two more levels).
pub fn shape_preset(name: &str) -> Option<&'static str> {
    Some(match name.trim() {
        "depth1" => "B(L,L)",
        "depth2_sub" => "U(B(L,L))",
        "depth2_rml" => "B(U(L),U(L))",
        "depth2_rmu" => "B(U(B(L,L)),B(L,L))",
        "depth2_rmu2" => "B(U(B(L,L)),U(L))",
        "depth2" => "B(U(B(L,L)),U(B(L,L)))",
        "depth3" => "B(U(B(U(B(L,L)),U(B(L,L)))),U(B(U(B(L,L)),U(B(L,L)))))",
        _ => return None,
    })
}

/// A FEX template: tree shape, operator tables and dimension.
#[derive(Clone, Debug)]
pub struct Template {
    spec: String,
    dim: usize,
    unary: Vec<UnaryOp>,
    binary: Vec<BinaryOp>,
    slots: Vec<SlotInfo>,
    pub(crate) plan: Vec<PlanNode>,
    n_params: usize,
}

impl Template {
    /// Build a template from a shape specification (or preset name), the
    /// unary and binary tables, and the dimension `d ≥ 1`.
    pub fn new(
        shape: &str,
        unary: &[UnaryOp],
        binary: &[BinaryOp],
        dim: usize,
    ) -> Result<Template, FexError> {
        let text = shape_preset(shape).unwrap_or(shape);
        let mut p = ShapeParser {
            s: text.as_bytes(),
            i: 0,
        };
        let root = p.node(0)?;
        p.skip_ws();
        if p.i != p.s.len() {
            return Err(FexError::Shape {
                detail: format!("trailing input at position {}", p.i),
            });
        }
        if root.count() > MAX_TEMPLATE_NODES {
            return Err(FexError::Shape {
                detail: format!(
                    "{} nodes is more than the {MAX_TEMPLATE_NODES} supported",
                    root.count()
                ),
            });
        }
        if dim == 0 {
            return Err(FexError::Dimension {
                detail: "dim must be at least 1".into(),
            });
        }
        if unary.is_empty() {
            return Err(FexError::Shape {
                detail: "the unary table is empty".into(),
            });
        }
        let mut has_binary = false;
        fn scan(n: &ShapeNode, b: &mut bool) {
            match n {
                ShapeNode::Leaf => {}
                ShapeNode::Unary(c) => scan(c, b),
                ShapeNode::Binary(l, r) => {
                    *b = true;
                    scan(l, b);
                    scan(r, b);
                }
            }
        }
        scan(&root, &mut has_binary);
        if has_binary && binary.is_empty() {
            return Err(FexError::Shape {
                detail: "the shape has binary slots but the binary table is empty".into(),
            });
        }

        // Slot numbering: in-order. Parameters follow slot order.
        let mut kinds: Vec<SlotKind> = Vec::new();
        fn inorder(n: &ShapeNode, kinds: &mut Vec<SlotKind>) {
            match n {
                ShapeNode::Leaf => kinds.push(SlotKind::Leaf),
                ShapeNode::Unary(c) => {
                    inorder(c, kinds);
                    kinds.push(SlotKind::Unary);
                }
                ShapeNode::Binary(l, r) => {
                    inorder(l, kinds);
                    kinds.push(SlotKind::Binary);
                    inorder(r, kinds);
                }
            }
        }
        inorder(&root, &mut kinds);
        let mut slots = Vec::with_capacity(kinds.len());
        let mut off = 0;
        for &k in &kinds {
            let n = match k {
                SlotKind::Leaf => dim + 1,
                SlotKind::Unary => 2,
                SlotKind::Binary => 0,
            };
            slots.push(SlotInfo {
                kind: k,
                param_offset: off,
                n_params: n,
            });
            off += n;
        }

        // Plan: post-order, recording each node's in-order slot number.
        let mut plan: Vec<PlanNode> = Vec::with_capacity(kinds.len());
        let mut next_slot = 0usize;
        fn build(
            n: &ShapeNode,
            plan: &mut Vec<PlanNode>,
            next_slot: &mut usize,
            slots: &[SlotInfo],
        ) -> usize {
            match n {
                ShapeNode::Leaf => {
                    let slot = *next_slot;
                    *next_slot += 1;
                    plan.push(PlanNode {
                        kind: SlotKind::Leaf,
                        slot,
                        a: usize::MAX,
                        b: usize::MAX,
                        poff: slots[slot].param_offset,
                    });
                    plan.len() - 1
                }
                ShapeNode::Unary(c) => {
                    let a = build(c, plan, next_slot, slots);
                    let slot = *next_slot;
                    *next_slot += 1;
                    plan.push(PlanNode {
                        kind: SlotKind::Unary,
                        slot,
                        a,
                        b: usize::MAX,
                        poff: slots[slot].param_offset,
                    });
                    plan.len() - 1
                }
                ShapeNode::Binary(l, r) => {
                    let a = build(l, plan, next_slot, slots);
                    let slot = *next_slot;
                    *next_slot += 1;
                    let b = build(r, plan, next_slot, slots);
                    plan.push(PlanNode {
                        kind: SlotKind::Binary,
                        slot,
                        a,
                        b,
                        poff: slots[slot].param_offset,
                    });
                    plan.len() - 1
                }
            }
        }
        build(&root, &mut plan, &mut next_slot, &slots);

        let mut spec = String::new();
        root.spec(&mut spec);
        Ok(Template {
            spec,
            dim,
            unary: unary.to_vec(),
            binary: binary.to_vec(),
            slots,
            plan,
            n_params: off,
        })
    }

    /// [`Template::new`] with operator names (`"sin"`, `"+"`, …).
    pub fn from_names(
        shape: &str,
        unary: &[&str],
        binary: &[&str],
        dim: usize,
    ) -> Result<Template, FexError> {
        let u = unary
            .iter()
            .map(|s| UnaryOp::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        let b = binary
            .iter()
            .map(|s| BinaryOp::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        Template::new(shape, &u, &b, dim)
    }

    /// The canonical shape specification, e.g. `"U(B(L,L))"`.
    pub fn shape(&self) -> &str {
        &self.spec
    }

    /// The dimension `d` (number of coordinates each leaf sums over).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The unary table.
    pub fn unary_table(&self) -> &[UnaryOp] {
        &self.unary
    }

    /// The binary table.
    pub fn binary_table(&self) -> &[BinaryOp] {
        &self.binary
    }

    /// The slots, in opcode-vector order.
    pub fn slots(&self) -> &[SlotInfo] {
        &self.slots
    }

    /// Number of slots (length of an opcode vector).
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    /// Number of parameters (length of `θ`).
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Size of each slot's table, in slot order (what a controller samples
    /// from).
    pub fn slot_arities(&self) -> Vec<usize> {
        self.slots
            .iter()
            .map(|s| match s.kind {
                SlotKind::Binary => self.binary.len(),
                _ => self.unary.len(),
            })
            .collect()
    }

    /// Number of distinct opcode vectors.
    pub fn n_skeletons(&self) -> u128 {
        self.slot_arities()
            .iter()
            .fold(1u128, |acc, &k| acc.saturating_mul(k as u128))
    }

    /// Check an opcode vector against the slot tables.
    pub fn check_opcodes(&self, e: &[usize]) -> Result<(), FexError> {
        if e.len() != self.slots.len() {
            return Err(FexError::Opcode {
                detail: format!(
                    "{} opcodes for a template with {} slots",
                    e.len(),
                    self.slots.len()
                ),
            });
        }
        for (i, (&op, s)) in e.iter().zip(&self.slots).enumerate() {
            let n = match s.kind {
                SlotKind::Binary => self.binary.len(),
                _ => self.unary.len(),
            };
            if op >= n {
                return Err(FexError::Opcode {
                    detail: format!("slot {i} has opcode {op}, its table has {n} entries"),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn check_params(&self, theta: &[f64]) -> Result<(), FexError> {
        if theta.len() != self.n_params {
            return Err(FexError::Params {
                expected: self.n_params,
                got: theta.len(),
            });
        }
        Ok(())
    }

    /// The unary operator a unary or leaf slot holds under `e`.
    pub(crate) fn unary_at(&self, e: &[usize], slot: usize) -> UnaryOp {
        self.unary[e[slot]]
    }

    /// The binary operator a binary slot holds under `e`.
    pub(crate) fn binary_at(&self, e: &[usize], slot: usize) -> BinaryOp {
        self.binary[e[slot]]
    }

    /// A human-readable rendering of the skeleton, e.g.
    /// `"id(L[sq] + L[0])"`.
    pub fn describe(&self, e: &[usize]) -> Result<String, FexError> {
        self.check_opcodes(e)?;
        fn go(t: &Template, e: &[usize], k: usize, out: &mut String) {
            let n = t.plan[k];
            match n.kind {
                SlotKind::Leaf => {
                    out.push_str("L[");
                    out.push_str(t.unary_at(e, n.slot).name());
                    out.push(']');
                }
                SlotKind::Unary => {
                    out.push_str(t.unary_at(e, n.slot).name());
                    out.push('(');
                    go(t, e, n.a, out);
                    out.push(')');
                }
                SlotKind::Binary => {
                    out.push('(');
                    go(t, e, n.a, out);
                    out.push(' ');
                    out.push_str(t.binary_at(e, n.slot).name());
                    out.push(' ');
                    go(t, e, n.b, out);
                    out.push(')');
                }
            }
        }
        let mut s = String::new();
        go(self, e, self.plan.len() - 1, &mut s);
        Ok(s)
    }

    /// The candidate `(e, θ)` as an Alkahest expression in `coords`
    /// (`coords.len() == d`), with each parameter a float.
    ///
    /// Leaf weights that are exactly `0.0` are left out (they contribute
    /// nothing); everything else is kept, so the expression evaluates to the
    /// same value as the fused evaluator up to rounding.
    pub fn instantiate(
        &self,
        e: &[usize],
        theta: &[f64],
        coords: &[ExprId],
        pool: &ExprPool,
    ) -> Result<ExprId, FexError> {
        self.check_params(theta)?;
        let params: Vec<Option<ExprId>> = theta
            .iter()
            .map(|&v| {
                if v == 0.0 {
                    None
                } else {
                    Some(pool.float(v, 53))
                }
            })
            .collect();
        self.instantiate_impl(e, &params, coords, pool)
    }

    /// The skeleton `e` as an expression whose parameters are the given
    /// expressions (typically fresh symbols), one per entry of `θ`.
    pub fn instantiate_with(
        &self,
        e: &[usize],
        params: &[ExprId],
        coords: &[ExprId],
        pool: &ExprPool,
    ) -> Result<ExprId, FexError> {
        if params.len() != self.n_params {
            return Err(FexError::Params {
                expected: self.n_params,
                got: params.len(),
            });
        }
        let p: Vec<Option<ExprId>> = params.iter().map(|&x| Some(x)).collect();
        self.instantiate_impl(e, &p, coords, pool)
    }

    fn instantiate_impl(
        &self,
        e: &[usize],
        params: &[Option<ExprId>],
        coords: &[ExprId],
        pool: &ExprPool,
    ) -> Result<ExprId, FexError> {
        self.check_opcodes(e)?;
        if coords.len() != self.dim {
            return Err(FexError::Dimension {
                detail: format!(
                    "{} coordinates for a template of dimension {}",
                    coords.len(),
                    self.dim
                ),
            });
        }
        let zero = pool.integer(0);
        let p = |k: usize| params[k].unwrap_or(zero);
        let mut vals: Vec<ExprId> = Vec::with_capacity(self.plan.len());
        for n in &self.plan {
            let v = match n.kind {
                SlotKind::Leaf => {
                    let phi = self.unary_at(e, n.slot);
                    let mut terms = Vec::with_capacity(self.dim + 1);
                    for (i, &x) in coords.iter().enumerate() {
                        if let Some(w) = params[n.poff + i] {
                            terms.push(pool.mul(vec![w, phi.apply_expr(x, pool)]));
                        }
                    }
                    if let Some(b) = params[n.poff + self.dim] {
                        terms.push(b);
                    }
                    match terms.len() {
                        0 => zero,
                        1 => terms[0],
                        _ => pool.add(terms),
                    }
                }
                SlotKind::Unary => {
                    let op = self.unary_at(e, n.slot);
                    let inner = op.apply_expr(vals[n.a], pool);
                    pool.add(vec![pool.mul(vec![p(n.poff), inner]), p(n.poff + 1)])
                }
                SlotKind::Binary => {
                    let op = self.binary_at(e, n.slot);
                    op.apply_expr(vals[n.a], vals[n.b], pool)
                }
            };
            vals.push(v);
        }
        Ok(*vals.last().expect("a template has at least one node"))
    }
}
