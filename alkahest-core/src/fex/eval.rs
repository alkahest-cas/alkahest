//! The template interpreter (R2–R4): fused exact jets, PDE residuals and the
//! full parameter Jacobian for batches of candidates.
//!
//! # What is computed
//!
//! For each point the interpreter walks the template's nodes once forward,
//! carrying the **jet** `(v, Δv, ∇v)` of every node (Forward-Laplacian
//! structure):
//!
//! * leaf `v = Σᵢ αᵢ φ(xᵢ) + β`: `∂ᵢv = αᵢ φ′(xᵢ)`, `Δv = Σ_{i spatial} αᵢ φ″(xᵢ)`;
//! * unary `h = α g(s) + β`: `∇h = α g′(s) ∇s`,
//!   `Δh = α (g″(s) |∇s|² + g′(s) Δs)`;
//! * binary nodes by the product / quotient rules for jets.
//!
//! That is O(d) per node — never `d` separate derivative passes. The residual
//! `F(u, ∇u, Δu, x) − f(x)` is formed at the root, and one **reverse** sweep
//! over the same nodes gives the whole row `∂r/∂θ` of the Jacobian (mixed
//! `∂θ∂x` derivatives included, which is why unary nodes need `g‴`).
//!
//! # Why an interpreter
//!
//! The kernel is ordinary Rust compiled ahead of time; a candidate is data
//! (an opcode per slot), so scoring a new candidate costs no JIT compile. The
//! per-point values `φ(xᵢ), φ′(xᵢ), φ″(xᵢ)` of each leaf operator depend only
//! on the points, so they are tabulated once per operator and point set and
//! shared by every candidate: a leaf is then a dot product.
//!
//! # Guards
//!
//! A candidate whose value or Jacobian is not finite anywhere (an `exp`
//! overflow, a zero denominator, `log` of a non-positive number) is marked
//! invalid: its residuals are `+∞`, its Jacobian rows zero and its loss
//! `+∞`. Nothing returns NaN.

#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::error::FexError;
use super::ops::{BinaryOp, UnaryOp, ALL_UNARY};
use super::problem::{LossKind, PdeOperator, Problem};
use super::template::{SlotKind, Template};

/// Rows per parallel work item.
const CHUNK: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) enum NodeOp {
    Leaf(UnaryOp),
    Unary(UnaryOp),
    Bin(BinaryOp),
}

/// `φ, φ′, φ″` (the last masked to spatial coordinates) at every point and
/// coordinate, row-major `n × d`.
struct LeafTable {
    f0: Vec<f64>,
    f1: Vec<f64>,
    f2: Vec<f64>,
}

struct Block {
    n: usize,
    x: Vec<f64>,
    /// `f(x)` at interior points, `g(x)` at boundary points.
    target: Vec<f64>,
    /// `V(x)` (empty when the potential is zero or unused).
    pot: Vec<f64>,
    /// Row weight: `1/√n` (interior), `√(λ/n)` (boundary).
    weight: f64,
    tables: Vec<OnceLock<LeafTable>>,
}

impl Block {
    fn new(x: Vec<f64>, n: usize, target: Vec<f64>, pot: Vec<f64>, weight: f64) -> Block {
        Block {
            n,
            x,
            target,
            pot,
            weight,
            tables: (0..ALL_UNARY.len()).map(|_| OnceLock::new()).collect(),
        }
    }

    fn table(&self, op: UnaryOp, dim: usize, mask: &[f64]) -> &LeafTable {
        self.tables[op.index()].get_or_init(|| {
            let len = self.n * dim;
            let mut t = LeafTable {
                f0: Vec::with_capacity(len),
                f1: Vec::with_capacity(len),
                f2: Vec::with_capacity(len),
            };
            for p in 0..self.n {
                for (i, &m) in mask.iter().enumerate() {
                    let f = op.eval4(self.x[p * dim + i]);
                    t.f0.push(f[0]);
                    t.f1.push(f[1]);
                    t.f2.push(f[2] * m);
                }
            }
            t
        })
    }
}

struct Scratch {
    jets: Vec<f64>,
    adj: Vec<f64>,
    aux: Vec<f64>,
    tmp: Vec<f64>,
    fc: Vec<[f64; 4]>,
    opin: Vec<f64>,
    opout: Vec<f64>,
    partials: Vec<f64>,
}

/// Jets of a candidate at a set of points (see [`Evaluator::jets_at`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Jets {
    /// `u` at each point.
    pub u: Vec<f64>,
    /// `∇u`, row-major `n × d`.
    pub grad: Vec<f64>,
    /// `Δu` over the spatial coordinates.
    pub lap: Vec<f64>,
}

#[cfg(feature = "parallel")]
fn map_tasks<T: Send, R: Send>(tasks: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    tasks.into_par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_tasks<T: Send, R: Send>(tasks: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    tasks.into_iter().map(f).collect()
}

/// A compiled scoring kernel for one template, one problem and fixed point
/// sets. Build once; score any number of candidates.
pub struct Evaluator {
    template: Template,
    problem: Problem,
    int: Block,
    bd: Block,
    order: u8,
    mask: Vec<f64>,
    stride: usize,
}

#[inline]
fn mdot(a: &[f64], b: &[f64], m: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..m.len() {
        s += a[i] * b[i] * m[i];
    }
    s
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

impl Evaluator {
    /// Build an evaluator. `interior` and `boundary` are row-major point
    /// arrays with `dim` columns (`boundary` may be empty).
    pub fn new(
        template: &Template,
        problem: &Problem,
        interior: &[f64],
        boundary: &[f64],
    ) -> Result<Evaluator, FexError> {
        let dim = template.dim();
        if problem.dim() != dim {
            return Err(FexError::Dimension {
                detail: format!(
                    "template dimension {dim}, problem dimension {}",
                    problem.dim()
                ),
            });
        }
        if interior.len() % dim != 0 || boundary.len() % dim != 0 {
            return Err(FexError::Points {
                detail: format!("point arrays must have {dim} columns"),
            });
        }
        let ni = interior.len() / dim;
        let nb = boundary.len() / dim;
        if ni == 0 {
            return Err(FexError::Points {
                detail: "no interior points".into(),
            });
        }
        let mut mask = vec![1.0; dim];
        if let Some(t) = problem.time() {
            mask[t] = 0.0;
        }
        let rows = |x: &[f64], n: usize, f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
            (0..n).map(|p| f(&x[p * dim..(p + 1) * dim])).collect()
        };
        let rhs = rows(interior, ni, &|x| problem.rhs.eval(x));
        let uses_pot = !problem.potential.is_zero()
            && (matches!(problem.loss, LossKind::Rayleigh)
                || matches!(&problem.operator, PdeOperator::Linear(l) if l.potential));
        let pot = if uses_pot {
            rows(interior, ni, &|x| problem.potential.eval(x))
        } else {
            Vec::new()
        };
        let target = rows(boundary, nb, &|x| problem.bc.eval(x));
        let wi = 1.0 / (ni as f64).sqrt();
        let wb = if nb > 0 {
            (problem.bc_weight / nb as f64).sqrt()
        } else {
            0.0
        };
        Ok(Evaluator {
            template: template.clone(),
            problem: problem.clone(),
            int: Block::new(interior.to_vec(), ni, rhs, pot, wi),
            bd: Block::new(boundary.to_vec(), nb, target, Vec::new(), wb),
            order: problem.interior_order(),
            mask,
            stride: dim + 2,
        })
    }

    /// The template.
    pub fn template(&self) -> &Template {
        &self.template
    }

    /// The problem.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Number of interior points.
    pub fn n_interior(&self) -> usize {
        self.int.n
    }

    /// Number of boundary points.
    pub fn n_boundary(&self) -> usize {
        self.bd.n
    }

    /// Length of the residual vector: interior rows, then boundary rows.
    /// Rows are weighted so that `‖r‖² = mean(r_int²) + λ·mean(r_bd²)`.
    pub fn n_residuals(&self) -> usize {
        self.int.n + self.bd.n
    }

    fn scratch(&self) -> Scratch {
        let n = self.template.plan.len();
        let st = self.stride;
        let dim = self.template.dim();
        Scratch {
            jets: vec![0.0; n * st],
            adj: vec![0.0; n * st],
            aux: vec![0.0; n * st],
            tmp: vec![0.0; st],
            fc: vec![[0.0; 4]; n],
            opin: vec![0.0; 2 + 2 * dim],
            opout: vec![0.0; 3 + dim],
            partials: vec![0.0; st],
        }
    }

    pub(crate) fn resolve(&self, e: &[usize]) -> Result<Vec<NodeOp>, FexError> {
        self.template.check_opcodes(e)?;
        Ok(self
            .template
            .plan
            .iter()
            .map(|n| match n.kind {
                SlotKind::Leaf => NodeOp::Leaf(self.template.unary_at(e, n.slot)),
                SlotKind::Unary => NodeOp::Unary(self.template.unary_at(e, n.slot)),
                SlotKind::Binary => NodeOp::Bin(self.template.binary_at(e, n.slot)),
            })
            .collect())
    }

    fn tables<'a>(&'a self, blk: &'a Block, ops: &[NodeOp]) -> Vec<Option<&'a LeafTable>> {
        let dim = self.template.dim();
        ops.iter()
            .map(|op| match op {
                NodeOp::Leaf(phi) if !phi.is_constant() => Some(blk.table(*phi, dim, &self.mask)),
                _ => None,
            })
            .collect()
    }

    /// Forward sweep at point `p` (of the block `tabs` was resolved for); the root jet ends up in the last
    /// stride of `s.jets`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        ops: &[NodeOp],
        tabs: &[Option<&LeafTable>],
        th: &[f64],
        p: usize,
        order: u8,
        s: &mut Scratch,
    ) {
        let dim = self.template.dim();
        let st = self.stride;
        let m = &self.mask;
        let Scratch { jets, aux, fc, .. } = s;
        for (k, node) in self.template.plan.iter().enumerate() {
            let (lo, hi) = jets.split_at_mut(k * st);
            let out = &mut hi[..st];
            match ops[k] {
                NodeOp::Leaf(phi) => {
                    let w = &th[node.poff..node.poff + dim];
                    let beta = th[node.poff + dim];
                    match phi {
                        UnaryOp::Zero | UnaryOp::One => {
                            out[0] = if phi == UnaryOp::One {
                                beta + w.iter().sum::<f64>()
                            } else {
                                beta
                            };
                            if order >= 1 {
                                out[1..].fill(0.0);
                            }
                        }
                        _ => {
                            let t = tabs[k].expect("table resolved for a non-constant leaf");
                            let r = p * dim..(p + 1) * dim;
                            out[0] = beta + dot(w, &t.f0[r.clone()]);
                            if order >= 1 {
                                let f1 = &t.f1[r.clone()];
                                for i in 0..dim {
                                    out[2 + i] = w[i] * f1[i];
                                }
                                out[1] = if order >= 2 { dot(w, &t.f2[r]) } else { 0.0 };
                            }
                        }
                    }
                }
                NodeOp::Unary(op) => {
                    let c = &lo[node.a * st..node.a * st + st];
                    let f = op.eval4(c[0]);
                    fc[k] = f;
                    let al = th[node.poff];
                    let be = th[node.poff + 1];
                    out[0] = al * f[0] + be;
                    if order >= 1 {
                        let a1 = al * f[1];
                        for i in 0..dim {
                            out[2 + i] = a1 * c[2 + i];
                        }
                        out[1] = if order >= 2 {
                            let g2 = mdot(&c[2..], &c[2..], m);
                            al * (f[2] * g2 + f[1] * c[1])
                        } else {
                            0.0
                        };
                    }
                }
                NodeOp::Bin(op) => {
                    let a = &lo[node.a * st..node.a * st + st];
                    let b = &lo[node.b * st..node.b * st + st];
                    let len = if order == 0 { 1 } else { st };
                    match op {
                        BinaryOp::Add => {
                            for i in 0..len {
                                out[i] = a[i] + b[i];
                            }
                        }
                        BinaryOp::Sub => {
                            for i in 0..len {
                                out[i] = a[i] - b[i];
                            }
                        }
                        BinaryOp::Mul => mul_jet(a, b, out, order, m),
                        BinaryOp::Div => {
                            // q = a · r with r = 1/b (r's jet kept for the
                            // reverse sweep).
                            let rv = 1.0 / b[0];
                            let f = [rv, -rv * rv, 2.0 * rv * rv * rv, -6.0 * rv * rv * rv * rv];
                            fc[k] = f;
                            let r = &mut aux[k * st..k * st + st];
                            r[0] = rv;
                            if order >= 1 {
                                for i in 0..dim {
                                    r[2 + i] = f[1] * b[2 + i];
                                }
                                r[1] = if order >= 2 {
                                    f[2] * mdot(&b[2..], &b[2..], m) + f[1] * b[1]
                                } else {
                                    0.0
                                };
                            }
                            mul_jet(a, &aux[k * st..k * st + st], out, order, m);
                        }
                    }
                }
            }
        }
    }

    /// Reverse sweep: `s.adj` holds the seeded root adjoint (everything else
    /// zero); accumulates `∂(seed · jet)/∂θ` into `out`.
    #[allow(clippy::too_many_arguments)]
    fn backward(
        &self,
        ops: &[NodeOp],
        tabs: &[Option<&LeafTable>],
        th: &[f64],
        p: usize,
        order: u8,
        s: &mut Scratch,
        out: &mut [f64],
    ) {
        let dim = self.template.dim();
        let st = self.stride;
        let m = &self.mask;
        let Scratch {
            jets,
            adj,
            aux,
            tmp,
            fc,
            ..
        } = s;
        for (k, node) in self.template.plan.iter().enumerate().rev() {
            let (lo, hi) = adj.split_at_mut(k * st);
            let ad = &hi[..st];
            let vb = ad[0];
            let lb = ad[1];
            match ops[k] {
                NodeOp::Leaf(phi) => {
                    let po = node.poff;
                    out[po + dim] += vb;
                    match phi {
                        UnaryOp::Zero => {}
                        UnaryOp::One => {
                            for o in &mut out[po..po + dim] {
                                *o += vb;
                            }
                        }
                        _ => {
                            let t = tabs[k].expect("table resolved for a non-constant leaf");
                            let r = p * dim..(p + 1) * dim;
                            let f0 = &t.f0[r.clone()];
                            let o = &mut out[po..po + dim];
                            match order {
                                0 => {
                                    for i in 0..dim {
                                        o[i] += vb * f0[i];
                                    }
                                }
                                1 => {
                                    let f1 = &t.f1[r];
                                    for i in 0..dim {
                                        o[i] += vb * f0[i] + ad[2 + i] * f1[i];
                                    }
                                }
                                _ => {
                                    let f1 = &t.f1[r.clone()];
                                    let f2 = &t.f2[r];
                                    for i in 0..dim {
                                        o[i] += vb * f0[i] + ad[2 + i] * f1[i] + lb * f2[i];
                                    }
                                }
                            }
                        }
                    }
                }
                NodeOp::Unary(_) => {
                    let c = &jets[node.a * st..node.a * st + st];
                    let f = fc[k];
                    let al = th[node.poff];
                    let (g2, gdot) = if order >= 1 {
                        let gd = dot(&ad[2..], &c[2..]);
                        let g2 = if order >= 2 {
                            mdot(&c[2..], &c[2..], m)
                        } else {
                            0.0
                        };
                        (g2, gd)
                    } else {
                        (0.0, 0.0)
                    };
                    let lc = if order >= 2 { c[1] } else { 0.0 };
                    let lb = if order >= 2 { lb } else { 0.0 };
                    out[node.poff] += vb * f[0] + f[1] * gdot + lb * (f[2] * g2 + f[1] * lc);
                    out[node.poff + 1] += vb;
                    let ca = node.a * st;
                    lo[ca] += al * (vb * f[1] + f[2] * gdot + lb * (f[3] * g2 + f[2] * lc));
                    if order >= 1 {
                        let a1 = al * f[1];
                        let a2 = 2.0 * al * f[2] * lb;
                        for i in 0..dim {
                            lo[ca + 2 + i] += a1 * ad[2 + i] + a2 * m[i] * c[2 + i];
                        }
                        lo[ca + 1] += a1 * lb;
                    }
                }
                NodeOp::Bin(op) => {
                    let len = if order == 0 { 1 } else { st };
                    let ca = node.a * st;
                    let cb = node.b * st;
                    match op {
                        BinaryOp::Add => {
                            for i in 0..len {
                                lo[ca + i] += ad[i];
                                lo[cb + i] += ad[i];
                            }
                        }
                        BinaryOp::Sub => {
                            for i in 0..len {
                                lo[ca + i] += ad[i];
                                lo[cb + i] -= ad[i];
                            }
                        }
                        BinaryOp::Mul => {
                            let a = &jets[ca..ca + st];
                            let b = &jets[cb..cb + st];
                            tmp.fill(0.0);
                            // a's adjoint into lo[ca..], b's into tmp, then add.
                            mul_jet_adj(ad, a, b, &mut lo[ca..ca + st], tmp, order, m);
                            for i in 0..len {
                                lo[cb + i] += tmp[i];
                            }
                        }
                        BinaryOp::Div => {
                            let a = &jets[ca..ca + st];
                            let r = &aux[k * st..k * st + st];
                            let b = &jets[cb..cb + st];
                            tmp.fill(0.0);
                            mul_jet_adj(ad, a, r, &mut lo[ca..ca + st], tmp, order, m);
                            // tmp is r's adjoint; r = 1/b with α = 1, β = 0.
                            let f = fc[k];
                            let rb = tmp[0];
                            let rl = if order >= 2 { tmp[1] } else { 0.0 };
                            let lbb = if order >= 2 { b[1] } else { 0.0 };
                            let (g2, gdot) = if order >= 1 {
                                let gd = dot(&tmp[2..], &b[2..]);
                                let g2 = if order >= 2 {
                                    mdot(&b[2..], &b[2..], m)
                                } else {
                                    0.0
                                };
                                (g2, gd)
                            } else {
                                (0.0, 0.0)
                            };
                            lo[cb] += rb * f[1] + f[2] * gdot + rl * (f[3] * g2 + f[2] * lbb);
                            if order >= 1 {
                                let a2 = 2.0 * f[2] * rl;
                                for i in 0..dim {
                                    lo[cb + 2 + i] += f[1] * tmp[2 + i] + a2 * m[i] * b[2 + i];
                                }
                                lo[cb + 1] += f[1] * rl;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `F(jet, x)` at interior point `p` (without the source term) and its
    /// jet partials `[F_u, F_Δ, F_g…]` into `s.partials`.
    fn operator(&self, p: usize, s: &mut Scratch) -> f64 {
        let dim = self.template.dim();
        let st = self.stride;
        let root = (self.template.plan.len() - 1) * st;
        let Scratch {
            jets,
            partials,
            opin,
            opout,
            ..
        } = s;
        let j = &jets[root..root + st];
        let u = j[0];
        match &self.problem.operator {
            PdeOperator::Linear(l) => {
                let v = if self.int.pot.is_empty() || !l.potential {
                    0.0
                } else {
                    self.int.pot[p]
                };
                let mut f = l.lap * j[1] + l.u * u + l.cubic * u * u * u + v * u;
                partials[0] = l.u + 3.0 * l.cubic * u * u + v;
                partials[1] = l.lap;
                if l.grad.is_empty() {
                    partials[2..].fill(0.0);
                } else {
                    for i in 0..dim {
                        partials[2 + i] = l.grad[i];
                        f += l.grad[i] * j[2 + i];
                    }
                }
                f
            }
            PdeOperator::Compiled(h) => {
                let c = &h.0;
                opin[0] = u;
                opin[1] = j[1];
                opin[2..2 + dim].copy_from_slice(&j[2..]);
                opin[2 + dim..].copy_from_slice(&self.int.x[p * dim..(p + 1) * dim]);
                let nout = 3 + c.active_grad.len();
                c.f.call(opin, &mut opout[..nout]);
                partials[0] = opout[1];
                partials[1] = opout[2];
                partials[2..].fill(0.0);
                for (k, &i) in c.active_grad.iter().enumerate() {
                    partials[2 + i] = opout[3 + k];
                }
                opout[0]
            }
        }
    }

    fn seed_root(&self, s: &mut Scratch, scale: f64, partials_too: bool) {
        let st = self.stride;
        let n = self.template.plan.len();
        s.adj[..n * st].fill(0.0);
        let root = (n - 1) * st;
        if partials_too {
            for i in 0..st {
                s.adj[root + i] = scale * s.partials[i];
            }
        } else {
            s.adj[root] = scale;
        }
    }

    /// Residual rows `row0 .. row0 + r.len()` (and their Jacobian rows).
    /// Returns `false` if anything non-finite appeared.
    #[allow(clippy::too_many_arguments)]
    fn rows_chunk(
        &self,
        ops: &[NodeOp],
        ti: &[Option<&LeafTable>],
        tb: &[Option<&LeafTable>],
        th: &[f64],
        row0: usize,
        r: &mut [f64],
        mut jac: Option<&mut [f64]>,
        s: &mut Scratch,
    ) -> bool {
        let np = self.template.n_params();
        let root = (self.template.plan.len() - 1) * self.stride;
        for (ri, row) in (row0..row0 + r.len()).enumerate() {
            if row < self.int.n {
                let p = row;
                self.forward(ops, ti, th, p, self.order, s);
                let w = self.int.weight;
                let f = self.operator(p, s);
                let rv = w * (f - self.int.target[p]);
                if !rv.is_finite() {
                    return false;
                }
                r[ri] = rv;
                if let Some(j) = jac.as_deref_mut() {
                    let jr = &mut j[ri * np..(ri + 1) * np];
                    jr.fill(0.0);
                    self.seed_root(s, w, true);
                    self.backward(ops, ti, th, p, self.order, s, jr);
                    if !jr.iter().all(|v| v.is_finite()) {
                        return false;
                    }
                }
            } else {
                let p = row - self.int.n;
                self.forward(ops, tb, th, p, 0, s);
                let w = self.bd.weight;
                let rv = w * (s.jets[root] - self.bd.target[p]);
                if !rv.is_finite() {
                    return false;
                }
                r[ri] = rv;
                if let Some(j) = jac.as_deref_mut() {
                    let jr = &mut j[ri * np..(ri + 1) * np];
                    jr.fill(0.0);
                    self.seed_root(s, w, false);
                    self.backward(ops, tb, th, p, 0, s, jr);
                    if !jr.iter().all(|v| v.is_finite()) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn require_ls(&self, what: &str) -> Result<(), FexError> {
        if self.problem.loss != LossKind::LeastSquares {
            return Err(FexError::LossKind {
                detail: format!(
                    "{what} needs a least-squares problem, this one is {:?}",
                    self.problem.loss
                ),
            });
        }
        Ok(())
    }

    /// Residuals and Jacobian of one candidate, parallel over row chunks.
    /// Returns `false` (residuals `+∞`, Jacobian zero) for an invalid
    /// candidate.
    pub fn residual_and_jacobian(
        &self,
        e: &[usize],
        theta: &[f64],
        r: &mut [f64],
        jac: Option<&mut [f64]>,
    ) -> Result<bool, FexError> {
        self.require_ls("a residual vector")?;
        let ops = self.resolve(e)?;
        self.template.check_params(theta)?;
        let m = self.n_residuals();
        let np = self.template.n_params();
        if r.len() != m {
            return Err(FexError::Points {
                detail: format!("residual buffer has {} entries, need {m}", r.len()),
            });
        }
        if let Some(j) = &jac {
            if j.len() != m * np {
                return Err(FexError::Points {
                    detail: format!("Jacobian buffer has {} entries, need {}", j.len(), m * np),
                });
            }
        }
        let ok = self.cand_rows(&ops, theta, r, jac);
        Ok(ok)
    }

    pub(crate) fn cand_rows(
        &self,
        ops: &[NodeOp],
        th: &[f64],
        r: &mut [f64],
        jac: Option<&mut [f64]>,
    ) -> bool {
        let np = self.template.n_params();
        let ti = self.tables(&self.int, ops);
        let tb = self.tables(&self.bd, ops);
        let ok = match jac {
            Some(j) => {
                let tasks: Vec<(usize, &mut [f64], &mut [f64])> = r
                    .chunks_mut(CHUNK)
                    .zip(j.chunks_mut(CHUNK * np))
                    .enumerate()
                    .map(|(c, (rc, jc))| (c * CHUNK, rc, jc))
                    .collect();
                let oks = map_tasks(tasks, |(row0, rc, jc)| {
                    let mut s = self.scratch();
                    self.rows_chunk(ops, &ti, &tb, th, row0, rc, Some(jc), &mut s)
                });
                let ok = oks.iter().all(|&b| b);
                if !ok {
                    j.fill(0.0);
                }
                ok
            }
            None => {
                let tasks: Vec<(usize, &mut [f64])> = r
                    .chunks_mut(CHUNK)
                    .enumerate()
                    .map(|(c, rc)| (c * CHUNK, rc))
                    .collect();
                let oks = map_tasks(tasks, |(row0, rc)| {
                    let mut s = self.scratch();
                    self.rows_chunk(ops, &ti, &tb, th, row0, rc, None, &mut s)
                });
                oks.iter().all(|&b| b)
            }
        };
        if !ok {
            r.fill(f64::INFINITY);
        }
        ok
    }

    fn check_batch(&self, es: &[usize], thetas: &[f64]) -> Result<usize, FexError> {
        let ns = self.template.n_slots();
        let np = self.template.n_params();
        if ns == 0 || es.len() % ns != 0 {
            return Err(FexError::Opcode {
                detail: format!("opcode batch length {} is not a multiple of {ns}", es.len()),
            });
        }
        let b = es.len() / ns;
        if thetas.len() != b * np {
            return Err(FexError::Params {
                expected: b * np,
                got: thetas.len(),
            });
        }
        for c in 0..b {
            self.template.check_opcodes(&es[c * ns..(c + 1) * ns])?;
        }
        Ok(b)
    }

    /// Residuals `(B × M)` and optionally Jacobians `(B × M × P)` of a batch
    /// of candidates (`es` is `B × n_slots`, `thetas` is `B × n_params`, all
    /// row-major). Returns one validity flag per candidate.
    pub fn residual_and_jacobian_batch(
        &self,
        es: &[usize],
        thetas: &[f64],
        r: &mut [f64],
        jac: Option<&mut [f64]>,
    ) -> Result<Vec<bool>, FexError> {
        self.require_ls("a residual vector")?;
        let b = self.check_batch(es, thetas)?;
        let ns = self.template.n_slots();
        let np = self.template.n_params();
        let m = self.n_residuals();
        if r.len() != b * m {
            return Err(FexError::Points {
                detail: format!("residual buffer has {} entries, need {}", r.len(), b * m),
            });
        }
        if let Some(j) = &jac {
            if j.len() != b * m * np {
                return Err(FexError::Points {
                    detail: format!(
                        "Jacobian buffer has {} entries, need {}",
                        j.len(),
                        b * m * np
                    ),
                });
            }
        }
        let ops: Vec<Vec<NodeOp>> = (0..b)
            .map(|c| self.resolve(&es[c * ns..(c + 1) * ns]))
            .collect::<Result<_, _>>()?;
        let oks = match jac {
            Some(j) => {
                let tasks: Vec<_> = r
                    .chunks_mut(m)
                    .zip(j.chunks_mut(m * np))
                    .enumerate()
                    .collect();
                map_tasks(tasks, |(c, (rc, jc))| {
                    self.cand_rows(&ops[c], &thetas[c * np..(c + 1) * np], rc, Some(jc))
                })
            }
            None => {
                let tasks: Vec<_> = r.chunks_mut(m).enumerate().collect();
                map_tasks(tasks, |(c, rc)| {
                    self.cand_rows(&ops[c], &thetas[c * np..(c + 1) * np], rc, None)
                })
            }
        };
        Ok(oks)
    }

    /// The loss of one candidate (`+∞` if invalid).
    pub fn loss(&self, e: &[usize], theta: &[f64]) -> Result<f64, FexError> {
        let ops = self.resolve(e)?;
        self.template.check_params(theta)?;
        Ok(self.loss_impl(&ops, theta, None))
    }

    /// The loss of one candidate and its gradient in `θ` (written to
    /// `grad`). An invalid candidate gives `+∞` and a zero gradient.
    pub fn loss_and_grad(
        &self,
        e: &[usize],
        theta: &[f64],
        grad: &mut [f64],
    ) -> Result<f64, FexError> {
        let ops = self.resolve(e)?;
        self.template.check_params(theta)?;
        if grad.len() != theta.len() {
            return Err(FexError::Params {
                expected: theta.len(),
                got: grad.len(),
            });
        }
        Ok(self.loss_impl(&ops, theta, Some(grad)))
    }

    /// Losses of a batch (`+∞` for invalid candidates), parallel over
    /// candidates.
    pub fn loss_batch(&self, es: &[usize], thetas: &[f64]) -> Result<Vec<f64>, FexError> {
        let b = self.check_batch(es, thetas)?;
        let ns = self.template.n_slots();
        let np = self.template.n_params();
        let ops: Vec<Vec<NodeOp>> = (0..b)
            .map(|c| self.resolve(&es[c * ns..(c + 1) * ns]))
            .collect::<Result<_, _>>()?;
        Ok(map_tasks((0..b).collect(), |c| {
            self.loss_impl(&ops[c], &thetas[c * np..(c + 1) * np], None)
        }))
    }

    /// Losses and gradients (`B × n_params`) of a batch.
    pub fn loss_and_grad_batch(
        &self,
        es: &[usize],
        thetas: &[f64],
        grads: &mut [f64],
    ) -> Result<Vec<f64>, FexError> {
        let b = self.check_batch(es, thetas)?;
        let ns = self.template.n_slots();
        let np = self.template.n_params();
        if grads.len() != b * np {
            return Err(FexError::Params {
                expected: b * np,
                got: grads.len(),
            });
        }
        let ops: Vec<Vec<NodeOp>> = (0..b)
            .map(|c| self.resolve(&es[c * ns..(c + 1) * ns]))
            .collect::<Result<_, _>>()?;
        let tasks: Vec<_> = grads.chunks_mut(np).enumerate().collect();
        Ok(map_tasks(tasks, |(c, g)| {
            self.loss_impl(&ops[c], &thetas[c * np..(c + 1) * np], Some(g))
        }))
    }

    pub(crate) fn loss_impl(&self, ops: &[NodeOp], th: &[f64], grad: Option<&mut [f64]>) -> f64 {
        let np = self.template.n_params();
        let out = match self.problem.loss {
            LossKind::LeastSquares => self.ls_loss(ops, th, grad.is_some()),
            LossKind::Normalized { power } => {
                self.scalar_loss(ops, th, grad.is_some(), Some(power))
            }
            LossKind::Rayleigh => self.scalar_loss(ops, th, grad.is_some(), None),
        };
        match out {
            Some((l, g)) if l.is_finite() && g.iter().all(|v| v.is_finite()) => {
                if let Some(gr) = grad {
                    if g.len() == np {
                        gr.copy_from_slice(&g);
                    } else {
                        gr.fill(0.0);
                    }
                }
                l
            }
            _ => {
                if let Some(gr) = grad {
                    gr.fill(0.0);
                }
                f64::INFINITY
            }
        }
    }

    /// Point chunks over all rows (interior then boundary).
    fn row_chunks(&self) -> Vec<(usize, usize)> {
        let m = self.n_residuals();
        (0..m)
            .step_by(CHUNK)
            .map(|r0| (r0, (r0 + CHUNK).min(m)))
            .collect()
    }

    /// `Σ r²` and (optionally) `Σ 2 r ∂r/∂θ`, without storing the Jacobian.
    fn ls_loss(&self, ops: &[NodeOp], th: &[f64], want_grad: bool) -> Option<(f64, Vec<f64>)> {
        let np = self.template.n_params();
        let ti = self.tables(&self.int, ops);
        let tb = self.tables(&self.bd, ops);
        let root = (self.template.plan.len() - 1) * self.stride;
        let parts = map_tasks(self.row_chunks(), |(r0, r1)| {
            let mut s = self.scratch();
            let mut g = if want_grad { vec![0.0; np] } else { Vec::new() };
            let mut l = 0.0;
            for row in r0..r1 {
                if row < self.int.n {
                    let p = row;
                    self.forward(ops, &ti, th, p, self.order, &mut s);
                    let w = self.int.weight;
                    let rv = w * (self.operator(p, &mut s) - self.int.target[p]);
                    if !rv.is_finite() {
                        return None;
                    }
                    l += rv * rv;
                    if want_grad {
                        self.seed_root(&mut s, 2.0 * rv * w, true);
                        self.backward(ops, &ti, th, p, self.order, &mut s, &mut g);
                    }
                } else {
                    let p = row - self.int.n;
                    self.forward(ops, &tb, th, p, 0, &mut s);
                    let w = self.bd.weight;
                    let rv = w * (s.jets[root] - self.bd.target[p]);
                    if !rv.is_finite() {
                        return None;
                    }
                    l += rv * rv;
                    if want_grad {
                        self.seed_root(&mut s, 2.0 * rv * w, false);
                        self.backward(ops, &tb, th, p, 0, &mut s, &mut g);
                    }
                }
            }
            Some((l, g))
        });
        let mut loss = 0.0;
        let mut grad = if want_grad { vec![0.0; np] } else { Vec::new() };
        for part in parts {
            let (l, g) = part?;
            loss += l;
            for (a, b) in grad.iter_mut().zip(&g) {
                *a += b;
            }
        }
        Some((loss, grad))
    }

    /// Normalised (`power = Some(p)`) and Rayleigh (`None`) losses: one
    /// forward pass for the global sums, then a forward/reverse pass seeded
    /// with the derivative of the quotient.
    fn scalar_loss(
        &self,
        ops: &[NodeOp],
        th: &[f64],
        want_grad: bool,
        power: Option<f64>,
    ) -> Option<(f64, Vec<f64>)> {
        let np = self.template.n_params();
        let ti = self.tables(&self.int, ops);
        let tb = self.tables(&self.bd, ops);
        let st = self.stride;
        let root = (self.template.plan.len() - 1) * st;
        let ni = self.int.n as f64;
        let rayleigh = power.is_none();
        let order = self.order;
        let pot = |p: usize| {
            if self.int.pot.is_empty() {
                0.0
            } else {
                self.int.pot[p]
            }
        };
        // Pass 1: numerator / denominator sums and the boundary term.
        let sums = map_tasks(self.row_chunks(), |(r0, r1)| {
            let mut s = self.scratch();
            let (mut num, mut den, mut bd) = (0.0, 0.0, 0.0);
            for row in r0..r1 {
                if row < self.int.n {
                    let p = row;
                    self.forward(ops, &ti, th, p, order, &mut s);
                    let u = s.jets[root];
                    if rayleigh {
                        let g = &s.jets[root + 2..root + st];
                        num += mdot(g, g, &self.mask) + pot(p) * u * u;
                        den += u * u;
                    } else {
                        let rv = self.operator(p, &mut s) - self.int.target[p];
                        num += rv * rv;
                        den += u.abs().powf(power.unwrap_or(1.0));
                    }
                } else {
                    let p = row - self.int.n;
                    self.forward(ops, &tb, th, p, 0, &mut s);
                    let rv = self.bd.weight * (s.jets[root] - self.bd.target[p]);
                    bd += rv * rv;
                }
            }
            (num, den, bd)
        });
        let (mut num, mut den, mut bd) = (0.0, 0.0, 0.0);
        for (a, b, c) in sums {
            num += a;
            den += b;
            bd += c;
        }
        let (num, den) = (num / ni, den / ni);
        if !(num.is_finite() && den.is_finite() && bd.is_finite()) || den <= 0.0 {
            return None;
        }
        let q = num / den;
        let loss = q + bd;
        if !want_grad {
            return Some((loss, Vec::new()));
        }
        // Pass 2: gradient.
        let parts = map_tasks(self.row_chunks(), |(r0, r1)| {
            let mut s = self.scratch();
            let mut g = vec![0.0; np];
            for row in r0..r1 {
                if row < self.int.n {
                    let p = row;
                    self.forward(ops, &ti, th, p, order, &mut s);
                    let u = s.jets[root];
                    let n_adj = s.adj.len();
                    s.adj[..n_adj].fill(0.0);
                    if rayleigh {
                        // ∂/∂jet of (a − q b)/(N·den), a = |∇u|² + V u², b = u².
                        let c = 1.0 / (ni * den);
                        s.adj[root] = c * (2.0 * pot(p) * u - 2.0 * q * u);
                        for i in 0..st - 2 {
                            s.adj[root + 2 + i] = c * 2.0 * self.mask[i] * s.jets[root + 2 + i];
                        }
                    } else {
                        let pw = power.unwrap_or(1.0);
                        let rv = self.operator(p, &mut s) - self.int.target[p];
                        // d(num/den): (2 r ∂r)/(N den) − num/(den²) · p|u|^{p−1} sgn(u)/N
                        let c = 2.0 * rv / (ni * den);
                        for i in 0..st {
                            s.adj[root + i] = c * s.partials[i];
                        }
                        let du = if u == 0.0 {
                            0.0
                        } else {
                            pw * u.abs().powf(pw - 1.0) * u.signum()
                        };
                        s.adj[root] -= num / (den * den) * du / ni;
                    }
                    self.backward(ops, &ti, th, p, order, &mut s, &mut g);
                } else {
                    let p = row - self.int.n;
                    self.forward(ops, &tb, th, p, 0, &mut s);
                    let w = self.bd.weight;
                    let rv = w * (s.jets[root] - self.bd.target[p]);
                    self.seed_root(&mut s, 2.0 * rv * w, false);
                    self.backward(ops, &tb, th, p, 0, &mut s, &mut g);
                }
            }
            g
        });
        let mut grad = vec![0.0; np];
        for g in parts {
            for (a, b) in grad.iter_mut().zip(&g) {
                *a += b;
            }
        }
        Some((loss, grad))
    }

    /// `u`, `∇u` and `Δu` of a candidate at arbitrary points (row-major,
    /// `dim` columns). Non-finite values are returned as they are.
    pub fn jets_at(&self, e: &[usize], theta: &[f64], points: &[f64]) -> Result<Jets, FexError> {
        let ops = self.resolve(e)?;
        self.template.check_params(theta)?;
        let dim = self.template.dim();
        if points.len() % dim != 0 {
            return Err(FexError::Points {
                detail: format!("point array must have {dim} columns"),
            });
        }
        let n = points.len() / dim;
        let blk = Block::new(points.to_vec(), n, Vec::new(), Vec::new(), 1.0);
        let tabs = self.tables(&blk, &ops);
        let mut s = self.scratch();
        let st = self.stride;
        let root = (self.template.plan.len() - 1) * st;
        let mut out = Jets {
            u: Vec::with_capacity(n),
            grad: Vec::with_capacity(n * dim),
            lap: Vec::with_capacity(n),
        };
        for p in 0..n {
            self.forward(&ops, &tabs, theta, p, 2, &mut s);
            out.u.push(s.jets[root]);
            out.lap.push(s.jets[root + 1]);
            out.grad.extend_from_slice(&s.jets[root + 2..root + st]);
        }
        Ok(out)
    }

    /// Jacobian of the jet `(u, Δu, ∇u)` in `θ` at one point: a
    /// `(dim + 2) × n_params` row-major matrix (rows `u`, `Δu`, `∂₀u`, …).
    /// For tests and diagnostics.
    pub fn jet_jacobian_at(
        &self,
        e: &[usize],
        theta: &[f64],
        point: &[f64],
    ) -> Result<Vec<f64>, FexError> {
        let ops = self.resolve(e)?;
        self.template.check_params(theta)?;
        let dim = self.template.dim();
        if point.len() != dim {
            return Err(FexError::Points {
                detail: format!("a point has {dim} coordinates"),
            });
        }
        let blk = Block::new(point.to_vec(), 1, Vec::new(), Vec::new(), 1.0);
        let tabs = self.tables(&blk, &ops);
        let mut s = self.scratch();
        let st = self.stride;
        let np = self.template.n_params();
        let n = self.template.plan.len();
        let root = (n - 1) * st;
        let mut out = vec![0.0; st * np];
        for comp in 0..st {
            self.forward(&ops, &tabs, theta, 0, 2, &mut s);
            s.adj[..n * st].fill(0.0);
            s.adj[root + comp] = 1.0;
            self.backward(
                &ops,
                &tabs,
                theta,
                0,
                2,
                &mut s,
                &mut out[comp * np..(comp + 1) * np],
            );
        }
        Ok(out)
    }
}

/// Jet product `out = a·b`.
#[inline]
fn mul_jet(a: &[f64], b: &[f64], out: &mut [f64], order: u8, m: &[f64]) {
    out[0] = a[0] * b[0];
    if order >= 1 {
        let d = m.len();
        for i in 0..d {
            out[2 + i] = a[0] * b[2 + i] + b[0] * a[2 + i];
        }
        out[1] = if order >= 2 {
            a[0] * b[1] + b[0] * a[1] + 2.0 * mdot(&a[2..], &b[2..], m)
        } else {
            0.0
        };
    }
}

/// Reverse of [`mul_jet`]: adds the adjoints of `a` to `ab` and of `b` to
/// `bb` given the output adjoint `ad`.
#[inline]
fn mul_jet_adj(
    ad: &[f64],
    a: &[f64],
    b: &[f64],
    ab: &mut [f64],
    bb: &mut [f64],
    order: u8,
    m: &[f64],
) {
    let vb = ad[0];
    ab[0] += vb * b[0];
    bb[0] += vb * a[0];
    if order >= 1 {
        let d = m.len();
        let lb = if order >= 2 { ad[1] } else { 0.0 };
        let gb = &ad[2..2 + d];
        ab[0] += dot(gb, &b[2..]) + lb * b[1];
        bb[0] += dot(gb, &a[2..]) + lb * a[1];
        for i in 0..d {
            ab[2 + i] += gb[i] * b[0] + 2.0 * lb * m[i] * b[2 + i];
            bb[2 + i] += gb[i] * a[0] + 2.0 * lb * m[i] * a[2 + i];
        }
        ab[1] += lb * b[0];
        bb[1] += lb * a[0];
    }
}
