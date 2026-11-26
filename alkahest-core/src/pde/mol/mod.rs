//! Method-of-lines runtime: semi-discrete PDE systems evaluated by **stencil
//! kernels over index ranges** (PDE milestone M7b).
//!
//! **Experimental surface.** The symbolic discretisation (which stencil, which
//! boundary closure, which kernel expression) is done in Python
//! (`alkahest.experimental.method_of_lines`); this module is the numeric
//! engine it targets. It never builds one expression per grid point: a
//! kernel is one compiled function of the values at a fixed set of *offsets*
//! (plus coefficient arrays and `t`), and the program applies it over a box
//! of grid indices, row by row, reading neighbours through shifted slices of
//! the field arrays. A 10⁶-point grid costs the same compile time as a
//! 10-point one.
//!
//! # Layout
//!
//! Every field (unknown, auxiliary derivative array) is stored as a padded
//! row-major array of shape [`MolProgram::shape`]. Padding holds ghost layers
//! (periodic halos) and boundary nodes. The ODE/DAE state is a list of
//! `(field, flat index)` nodes; everything else is *derived* — filled each
//! evaluation by [`MolOp`]s from the state, in program order:
//!
//! 1. scatter the state into the field arrays;
//! 2. run the ops: a **fill** kernel writes a derived node of a field (a
//!    Dirichlet value, an eliminated Neumann/Robin closure, a periodic copy),
//!    a [`MolDenseOp`] applies a dense differentiation matrix along one axis
//!    (Chebyshev / Fourier collocation) into an auxiliary field, and an
//!    **rhs** kernel writes the time derivative of state nodes;
//! 3. gather the right-hand side of the state nodes.
//!
//! # Exact sparse Jacobian
//!
//! Each kernel may carry a second compiled kernel holding its symbolic
//! partial derivatives with respect to its field inputs. [`MolProgram::jacobian`]
//! evaluates them over the same boxes and composes them through the derived
//! nodes (chain rule through fills and dense ops), giving the exact Jacobian
//! of the discrete right-hand side as a CSR matrix whose pattern is fixed
//! after the first call.

mod banded;
mod integrate;

use std::sync::{Arc, Mutex};

use crate::jit::CompiledVecFn;

pub use banded::{rcm_order, BandedLu};
pub use integrate::{
    integrate, MolIntegrateOptions, MolMethod, MolSolution, MolStats, StepFailure,
};

/// Errors from building or running a [`MolProgram`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MolError {
    /// The program description is inconsistent (out-of-range index, region
    /// outside the padded shape, state nodes not covered exactly once, ...).
    InvalidProgram(String),
    /// An integrator gave up (step size underflow, non-finite values, Newton
    /// failure at the minimum step, singular iteration matrix).
    Integration(String),
    /// An argument does not match the program (wrong state length, ...).
    BadArgument(String),
}

impl std::fmt::Display for MolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MolError::InvalidProgram(m) => write!(f, "invalid method-of-lines program: {m}"),
            MolError::Integration(m) => write!(f, "integration failed: {m}"),
            MolError::BadArgument(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for MolError {}

/// One input column of a [`MolKernelOp`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MolInput {
    /// The value of `field` at the target point shifted by `offset` (one
    /// signed shift per dimension).
    Field { field: usize, offset: Vec<isize> },
    /// The coefficient array `coef` at the target point (grid coordinates,
    /// non-uniform stencil weights, spatially varying data).
    Coef(usize),
    /// The current time.
    Time,
}

/// Where a kernel writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolTarget {
    /// The time derivative of the state nodes of `field` in the region.
    Rhs(usize),
    /// The (derived, non-state) nodes of `field` in the region.
    Fill(usize),
}

/// A compiled pointwise kernel applied over a box of grid indices.
#[derive(Clone)]
pub struct MolKernelOp {
    /// One output: the value written at each target point.
    pub kernel: Arc<CompiledVecFn>,
    /// The kernel's inputs, in the order it was compiled with.
    pub inputs: Vec<MolInput>,
    pub target: MolTarget,
    /// Half-open index box `[lo, hi)` per dimension, in padded coordinates.
    pub region: Vec<(usize, usize)>,
    /// Partial derivatives of the output: `jac_kernel` has one output per
    /// entry of `jac_inputs`, the index (into `inputs`) of the `Field` input
    /// it differentiates by. Inputs left out have zero partials.
    pub jac_kernel: Option<Arc<CompiledVecFn>>,
    pub jac_inputs: Vec<usize>,
}

/// `dst[p] = Σ_j matrix[i][j] · src[line_j]` along `axis`, for every point
/// `p` of `region` (whose index along `axis` is `region[axis].0 + i`), where
/// `line_j` is `p` with its `axis` index replaced by `src_lo + j`.
#[derive(Clone, Debug)]
pub struct MolDenseOp {
    pub src_field: usize,
    pub dst_field: usize,
    pub axis: usize,
    pub region: Vec<(usize, usize)>,
    pub src_lo: usize,
    pub src_len: usize,
    /// Row-major, `(region[axis].1 - region[axis].0) × src_len`.
    pub matrix: Vec<f64>,
}

/// One step of a [`MolProgram`] evaluation.
#[derive(Clone)]
pub enum MolOp {
    Kernel(MolKernelOp),
    Dense(MolDenseOp),
}

/// A CSR sparse matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CsrMatrix {
    /// `y = A x`.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (r, yr) in y.iter_mut().enumerate().take(self.n_rows) {
            let mut s = 0.0;
            for k in self.indptr[r]..self.indptr[r + 1] {
                s += self.data[k] * x[self.indices[k]];
            }
            *yr = s;
        }
    }
}

struct Scratch {
    full: Vec<Vec<f64>>,
    dfull: Vec<Vec<f64>>,
    tbuf: Vec<f64>,
    obuf: Vec<f64>,
    jbuf: Vec<Vec<f64>>,
}

/// The cached Jacobian structure: the CSR pattern and, for every entry the
/// assembly emits (in its deterministic order), the CSR slot it adds into.
struct JacCache {
    pattern: CsrMatrix,
    slots: Vec<usize>,
}

/// A semi-discrete PDE system `M y' = f(t, y)` evaluated by stencil kernels.
/// See the [module docs](self).
pub struct MolProgram {
    shape: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
    n_fields: usize,
    coefs: Vec<Vec<f64>>,
    /// `(field, flat)` per state component.
    states: Vec<(usize, usize)>,
    algebraic: Vec<bool>,
    /// `field * size + flat` → state index, or `usize::MAX`.
    state_of: Vec<usize>,
    /// Maximal runs of states that are consecutive in both the state vector
    /// and one field array: `(field, flat start, state start, length)`.
    runs: Vec<(usize, usize, usize, usize)>,
    ops: Vec<MolOp>,
    scratch: Mutex<Scratch>,
    jac: Mutex<Option<JacCache>>,
}

const NONE: usize = usize::MAX;

impl MolProgram {
    /// Build and validate a program.
    ///
    /// * `shape` — the padded grid shape (1–3 dimensions);
    /// * `n_fields` — number of field arrays;
    /// * `coefs` — coefficient arrays, each of the padded size;
    /// * `states` — `(field, flat index)` per state component;
    /// * `algebraic` — per state component, whether it is algebraic (mass 0);
    /// * `ops` — evaluated in order.
    ///
    /// Every state node must be written by exactly one `Rhs` kernel point,
    /// and no `Fill` or dense op may write a state node.
    pub fn new(
        shape: Vec<usize>,
        n_fields: usize,
        coefs: Vec<Vec<f64>>,
        states: Vec<(usize, usize)>,
        algebraic: Vec<bool>,
        ops: Vec<MolOp>,
    ) -> Result<Self, MolError> {
        let bad = |m: String| Err(MolError::InvalidProgram(m));
        if shape.is_empty() || shape.len() > 3 || shape.contains(&0) {
            return bad(format!("shape {shape:?} must have 1-3 positive extents"));
        }
        let d = shape.len();
        let mut strides = vec![1usize; d];
        for k in (0..d.saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * shape[k + 1];
        }
        let size: usize = shape.iter().product();
        if algebraic.len() != states.len() {
            return bad("algebraic mask length differs from the number of states".into());
        }
        for (i, c) in coefs.iter().enumerate() {
            if c.len() != size {
                return bad(format!(
                    "coefficient array {i} has {} values, not {size}",
                    c.len()
                ));
            }
        }
        let mut state_of = vec![NONE; n_fields * size];
        for (s, &(f, p)) in states.iter().enumerate() {
            if f >= n_fields || p >= size {
                return bad(format!(
                    "state {s} = (field {f}, index {p}) is out of range"
                ));
            }
            if state_of[f * size + p] != NONE {
                return bad(format!("state node (field {f}, index {p}) is listed twice"));
            }
            state_of[f * size + p] = s;
        }
        let mut covered = vec![0u32; states.len()];
        for (k, op) in ops.iter().enumerate() {
            let region = match op {
                MolOp::Kernel(o) => &o.region,
                MolOp::Dense(o) => &o.region,
            };
            if region.len() != d
                || region
                    .iter()
                    .zip(&shape)
                    .any(|(&(lo, hi), &n)| lo >= hi || hi > n)
            {
                return bad(format!(
                    "op {k}: region {region:?} is empty or outside {shape:?}"
                ));
            }
            match op {
                MolOp::Kernel(o) => {
                    if o.kernel.n_outputs() != 1 || o.kernel.n_inputs() != o.inputs.len() {
                        return bad(format!("op {k}: kernel arity does not match its inputs"));
                    }
                    for inp in &o.inputs {
                        match inp {
                            MolInput::Field { field, offset } => {
                                if *field >= n_fields || offset.len() != d {
                                    return bad(format!("op {k}: bad field input {inp:?}"));
                                }
                                for (a, &(lo, hi)) in region.iter().enumerate() {
                                    let l = lo as isize + offset[a];
                                    let h = hi as isize - 1 + offset[a];
                                    if l < 0 || h >= shape[a] as isize {
                                        return bad(format!(
                                            "op {k}: input {inp:?} reads outside the padded grid"
                                        ));
                                    }
                                }
                            }
                            MolInput::Coef(c) => {
                                if *c >= coefs.len() {
                                    return bad(format!("op {k}: coefficient {c} does not exist"));
                                }
                            }
                            MolInput::Time => {}
                        }
                    }
                    if let Some(j) = &o.jac_kernel {
                        if j.n_outputs() != o.jac_inputs.len() || j.n_inputs() != o.inputs.len() {
                            return bad(format!("op {k}: Jacobian kernel arity mismatch"));
                        }
                        if o.jac_inputs
                            .iter()
                            .any(|&i| !matches!(o.inputs.get(i), Some(MolInput::Field { .. })))
                        {
                            return bad(format!("op {k}: Jacobian input is not a field input"));
                        }
                    }
                    let (field, is_rhs) = match o.target {
                        MolTarget::Rhs(f) => (f, true),
                        MolTarget::Fill(f) => (f, false),
                    };
                    if field >= n_fields {
                        return bad(format!("op {k}: target field {field} out of range"));
                    }
                    let mut err = None;
                    for_each_point(&strides, region, |p| {
                        let s = state_of[field * size + p];
                        if is_rhs {
                            if s == NONE {
                                err = Some(format!("op {k}: rhs kernel writes a non-state node"));
                            } else {
                                covered[s] += 1;
                            }
                        } else if s != NONE {
                            err = Some(format!("op {k}: fill kernel overwrites a state node"));
                        }
                    });
                    if let Some(e) = err {
                        return bad(e);
                    }
                }
                MolOp::Dense(o) => {
                    if o.src_field >= n_fields || o.dst_field >= n_fields || o.axis >= d {
                        return bad(format!("op {k}: dense op field or axis out of range"));
                    }
                    let rows = region[o.axis].1 - region[o.axis].0;
                    if o.matrix.len() != rows * o.src_len || o.src_lo + o.src_len > shape[o.axis] {
                        return bad(format!("op {k}: dense matrix shape mismatch"));
                    }
                    let mut err = None;
                    for_each_point(&strides, region, |p| {
                        if state_of[o.dst_field * size + p] != NONE {
                            err = Some(format!("op {k}: dense op overwrites a state node"));
                        }
                    });
                    if let Some(e) = err {
                        return bad(e);
                    }
                }
            }
        }
        if let Some(s) = covered.iter().position(|&c| c != 1) {
            return bad(format!(
                "state {s} = {:?} is written by {} rhs kernel points (must be exactly one)",
                states[s], covered[s]
            ));
        }
        let max_row = shape[d - 1];
        let max_jac = ops
            .iter()
            .filter_map(|o| match o {
                MolOp::Kernel(k) => Some(k.jac_inputs.len()),
                MolOp::Dense(_) => None,
            })
            .max()
            .unwrap_or(0);
        let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (s, &(f, p)) in states.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if r.0 == f && r.1 + r.3 == p && r.2 + r.3 == s => r.3 += 1,
                _ => runs.push((f, p, s, 1)),
            }
        }
        let scratch = Scratch {
            full: vec![vec![0.0; size]; n_fields],
            dfull: vec![vec![0.0; size]; n_fields],
            tbuf: vec![0.0; size],
            obuf: vec![0.0; size],
            jbuf: vec![vec![0.0; max_row]; max_jac],
        };
        Ok(MolProgram {
            shape,
            strides,
            size,
            n_fields,
            coefs,
            states,
            algebraic,
            state_of,
            runs,
            ops,
            scratch: Mutex::new(scratch),
            jac: Mutex::new(None),
        })
    }

    /// The padded grid shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of field arrays.
    pub fn n_fields(&self) -> usize {
        self.n_fields
    }

    /// Number of state components.
    pub fn n_states(&self) -> usize {
        self.states.len()
    }

    /// Per state component, whether it is algebraic (a zero row of the mass
    /// matrix).
    pub fn algebraic(&self) -> &[bool] {
        &self.algebraic
    }

    /// `(field, flat index)` per state component.
    pub fn states(&self) -> &[(usize, usize)] {
        &self.states
    }

    /// Number of ops.
    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }

    fn check_y(&self, y: &[f64]) -> Result<(), MolError> {
        if y.len() != self.states.len() {
            return Err(MolError::BadArgument(format!(
                "state vector has {} values, the program has {} states",
                y.len(),
                self.states.len()
            )));
        }
        Ok(())
    }

    /// Scatter the state, run every fill/dense op, then (when `rhs` is set)
    /// every rhs kernel; return the right-hand side in `out`.
    fn run(&self, sc: &mut Scratch, t: f64, y: &[f64], out: Option<&mut [f64]>) {
        // Only state nodes change between calls: derived nodes are rewritten
        // by their fill / dense ops before anything reads them, and nodes no
        // op writes keep their initial value.
        for &(f, p, s, len) in &self.runs {
            sc.full[f][p..p + len].copy_from_slice(&y[s..s + len]);
        }
        let want_rhs = out.is_some();
        for op in &self.ops {
            match op {
                MolOp::Dense(o) => self.apply_dense(sc, o),
                MolOp::Kernel(o) => {
                    if matches!(o.target, MolTarget::Rhs(_)) && !want_rhs {
                        continue;
                    }
                    self.apply_kernel(sc, o, t);
                }
            }
        }
        if let Some(out) = out {
            for &(f, p, s, len) in &self.runs {
                out[s..s + len].copy_from_slice(&sc.dfull[f][p..p + len]);
            }
        }
    }

    fn apply_kernel(&self, sc: &mut Scratch, o: &MolKernelOp, t: f64) {
        let d = self.shape.len();
        let last = d - 1;
        let (lo_l, hi_l) = o.region[last];
        let len = hi_l - lo_l;
        // Several short rows: evaluate the kernel once over the contiguous
        // flat span from the first to the last point of the box (the points
        // in between that are outside the box read in-bounds neighbours and
        // their results are discarded), then copy the rows out. One kernel
        // call instead of one per row; the extra work is the box's share of
        // padding, bounded below by the 2x test.
        let points: usize = o.region.iter().map(|&(a, b)| b - a).product();
        let first: usize = o
            .region
            .iter()
            .zip(&self.strides)
            .map(|(&(a, _), &s)| a * s)
            .sum();
        let last_pt: usize = o
            .region
            .iter()
            .zip(&self.strides)
            .map(|(&(_, b), &s)| (b - 1) * s)
            .sum();
        let span = last_pt - first + 1;
        if points > len && span <= 2 * points {
            sc.tbuf[..span].fill(t);
            let Scratch {
                full,
                dfull,
                tbuf,
                obuf,
                ..
            } = sc;
            {
                let cols: Vec<&[f64]> = o
                    .inputs
                    .iter()
                    .map(|inp| match inp {
                        MolInput::Field { field, offset } => {
                            let s = (first as isize + self.shift(offset)) as usize;
                            &full[*field][s..s + span]
                        }
                        MolInput::Coef(c) => &self.coefs[*c][first..first + span],
                        MolInput::Time => &tbuf[..span],
                    })
                    .collect();
                o.kernel.call_batch(&cols, &mut [&mut obuf[..span]]);
            }
            let target = match o.target {
                MolTarget::Rhs(f) => &mut dfull[f],
                MolTarget::Fill(f) => &mut full[f],
            };
            for_each_row(&self.strides, &o.region, |start| {
                let r = start - first;
                target[start..start + len].copy_from_slice(&obuf[r..r + len]);
            });
            return;
        }
        sc.tbuf[..len].fill(t);
        let Scratch {
            full,
            dfull,
            tbuf,
            obuf,
            ..
        } = sc;
        for_each_row(&self.strides, &o.region, |start| {
            let cols: Vec<&[f64]> = o
                .inputs
                .iter()
                .map(|inp| match inp {
                    MolInput::Field { field, offset } => {
                        let shift = self.shift(offset);
                        let s = (start as isize + shift) as usize;
                        &full[*field][s..s + len]
                    }
                    MolInput::Coef(c) => &self.coefs[*c][start..start + len],
                    MolInput::Time => &tbuf[..len],
                })
                .collect();
            match o.target {
                MolTarget::Rhs(f) => {
                    let out = &mut dfull[f][start..start + len];
                    o.kernel.call_batch(&cols, &mut [out]);
                }
                MolTarget::Fill(_) => {
                    let out = &mut obuf[..len];
                    o.kernel.call_batch(&cols, &mut [out]);
                }
            }
            drop(cols);
            if let MolTarget::Fill(f) = o.target {
                full[f][start..start + len].copy_from_slice(&obuf[..len]);
            }
        });
    }

    fn apply_dense(&self, sc: &mut Scratch, o: &MolDenseOp) {
        let stride = self.strides[o.axis];
        let (rlo, rhi) = o.region[o.axis];
        // Iterate over the lines: the region with the axis collapsed to its
        // first index.
        let mut line_region = o.region.clone();
        line_region[o.axis] = (rlo, rlo + 1);
        let mut tmp = vec![0.0; rhi - rlo];
        for_each_point(&self.strides, &line_region, |p0| {
            // p0 is the point with axis index rlo; the source line starts at
            // index src_lo along the axis.
            let base = p0 - rlo * stride;
            let src = &sc.full[o.src_field];
            for (i, ti) in tmp.iter_mut().enumerate() {
                let row = &o.matrix[i * o.src_len..(i + 1) * o.src_len];
                let mut s = 0.0;
                for (j, &m) in row.iter().enumerate() {
                    s += m * src[base + (o.src_lo + j) * stride];
                }
                *ti = s;
            }
            let dst = &mut sc.full[o.dst_field];
            for (i, &v) in tmp.iter().enumerate() {
                dst[base + (rlo + i) * stride] = v;
            }
        });
    }

    fn shift(&self, offset: &[isize]) -> isize {
        offset
            .iter()
            .zip(&self.strides)
            .map(|(&o, &s)| o * s as isize)
            .sum()
    }

    /// The right-hand side `f(t, y)`.
    pub fn rhs(&self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), MolError> {
        self.check_y(y)?;
        if out.len() != self.states.len() {
            return Err(MolError::BadArgument(
                "output length differs from n_states".into(),
            ));
        }
        let mut sc = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        self.run(&mut sc, t, y, Some(out));
        Ok(())
    }

    /// Every field array (padded, row-major) after the fills for state `y`
    /// at time `t` — boundary and ghost values included.
    pub fn fields(&self, t: f64, y: &[f64]) -> Result<Vec<Vec<f64>>, MolError> {
        self.check_y(y)?;
        let mut sc = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        self.run(&mut sc, t, y, None);
        Ok(sc.full.clone())
    }

    /// The exact Jacobian `∂f/∂y` of the right-hand side at `(t, y)`, as CSR.
    ///
    /// Kernels without a Jacobian kernel contribute nothing, so a program
    /// built without them returns a matrix of explicit zeros (the pattern is
    /// still that of the kernels that have one).
    pub fn jacobian(&self, t: f64, y: &[f64]) -> Result<CsrMatrix, MolError> {
        self.check_y(y)?;
        let mut sc = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        self.run(&mut sc, t, y, None);
        let entries = self.jac_entries(&mut sc, t);
        drop(sc);
        let mut cache = self.jac.lock().unwrap_or_else(|e| e.into_inner());
        if cache.is_none() {
            *cache = Some(self.build_pattern(&entries));
        }
        let c = cache.as_ref().expect("pattern just built");
        let mut m = c.pattern.clone();
        m.data.iter_mut().for_each(|v| *v = 0.0);
        for (&(_, _, v), &slot) in entries.iter().zip(&c.slots) {
            m.data[slot] += v;
        }
        Ok(m)
    }

    /// The Jacobian's sparsity pattern (values zero).
    pub fn jacobian_pattern(&self, t: f64, y: &[f64]) -> Result<CsrMatrix, MolError> {
        let mut m = self.jacobian(t, y)?;
        m.data.iter_mut().for_each(|v| *v = 0.0);
        Ok(m)
    }

    fn build_pattern(&self, entries: &[(usize, usize, f64)]) -> JacCache {
        let n = self.states.len();
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c, _) in entries {
            rows[r].push(c);
        }
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        indptr.push(0);
        for row in rows.iter_mut() {
            row.sort_unstable();
            row.dedup();
            indices.extend_from_slice(row);
            indptr.push(indices.len());
        }
        let slots = entries
            .iter()
            .map(|&(r, c, _)| {
                let seg = &indices[indptr[r]..indptr[r + 1]];
                indptr[r] + seg.binary_search(&c).expect("entry in pattern")
            })
            .collect();
        let nnz = indices.len();
        JacCache {
            pattern: CsrMatrix {
                n_rows: n,
                n_cols: n,
                indptr,
                indices,
                data: vec![0.0; nnz],
            },
            slots,
        }
    }

    /// Emit `(row, col, value)` contributions in a deterministic order.
    /// Requires the field arrays filled for the current state (`run`).
    #[allow(clippy::needless_range_loop)]
    fn jac_entries(&self, sc: &mut Scratch, t: f64) -> Vec<(usize, usize, f64)> {
        let size = self.size;
        let n_nodes = self.n_fields * size;
        // Derived-node rows: node -> range into `ents`.
        let mut row_of: Vec<(usize, usize)> = vec![(NONE, 0); n_nodes];
        let mut ents: Vec<(usize, f64)> = Vec::new();
        let mut out: Vec<(usize, usize, f64)> = Vec::new();
        let mut contrib: Vec<(usize, f64)> = Vec::new();
        let d = self.shape.len();
        let last = d - 1;

        // Push the dependence of `node` (scaled by `w`) on the state.
        fn push_dep(
            node: usize,
            w: f64,
            state_of: &[usize],
            row_of: &[(usize, usize)],
            ents: &[(usize, f64)],
            contrib: &mut Vec<(usize, f64)>,
        ) {
            let s = state_of[node];
            if s != NONE {
                contrib.push((s, w));
                return;
            }
            let (start, len) = row_of[node];
            if start != NONE {
                for &(s, v) in &ents[start..start + len] {
                    contrib.push((s, w * v));
                }
            }
        }
        fn merge(contrib: &mut Vec<(usize, f64)>) {
            // Stable sort keeps the emission order deterministic.
            contrib.sort_by_key(|e| e.0);
            let mut w = 0;
            for r in 0..contrib.len() {
                if w > 0 && contrib[w - 1].0 == contrib[r].0 {
                    contrib[w - 1].1 += contrib[r].1;
                } else {
                    contrib[w] = contrib[r];
                    w += 1;
                }
            }
            contrib.truncate(w);
        }

        for op in &self.ops {
            match op {
                MolOp::Dense(o) => {
                    let stride = self.strides[o.axis];
                    let (rlo, _) = o.region[o.axis];
                    for_each_point(&self.strides, &o.region, |p| {
                        let i = (p / stride) % self.shape[o.axis] - rlo;
                        let base = p - (rlo + i) * stride;
                        contrib.clear();
                        let row = &o.matrix[i * o.src_len..(i + 1) * o.src_len];
                        for (j, &m) in row.iter().enumerate() {
                            if m != 0.0 {
                                let q = o.src_field * size + base + (o.src_lo + j) * stride;
                                push_dep(q, m, &self.state_of, &row_of, &ents, &mut contrib);
                            }
                        }
                        merge(&mut contrib);
                        row_of[o.dst_field * size + p] = (ents.len(), contrib.len());
                        ents.extend_from_slice(&contrib);
                    });
                }
                MolOp::Kernel(o) => {
                    let Some(jk) = &o.jac_kernel else {
                        if let MolTarget::Fill(f) = o.target {
                            // A fill without partials: its nodes depend on no state.
                            for_each_point(&self.strides, &o.region, |p| {
                                row_of[f * size + p] = (NONE, 0);
                            });
                        }
                        continue;
                    };
                    let (lo_l, hi_l) = o.region[last];
                    let len = hi_l - lo_l;
                    sc.tbuf[..len].fill(t);
                    let shifts: Vec<isize> = o
                        .jac_inputs
                        .iter()
                        .map(|&i| match &o.inputs[i] {
                            MolInput::Field { offset, .. } => self.shift(offset),
                            _ => 0,
                        })
                        .collect();
                    let fields: Vec<usize> = o
                        .jac_inputs
                        .iter()
                        .map(|&i| match &o.inputs[i] {
                            MolInput::Field { field, .. } => *field,
                            _ => 0,
                        })
                        .collect();
                    let Scratch {
                        full, tbuf, jbuf, ..
                    } = &mut *sc;
                    for_each_row(&self.strides, &o.region, |start| {
                        {
                            let cols: Vec<&[f64]> = o
                                .inputs
                                .iter()
                                .map(|inp| match inp {
                                    MolInput::Field { field, offset } => {
                                        let s = (start as isize + self.shift(offset)) as usize;
                                        &full[*field][s..s + len]
                                    }
                                    MolInput::Coef(c) => &self.coefs[*c][start..start + len],
                                    MolInput::Time => &tbuf[..len],
                                })
                                .collect();
                            let mut outs: Vec<&mut [f64]> =
                                jbuf.iter_mut().map(|b| &mut b[..len]).collect();
                            outs.truncate(o.jac_inputs.len());
                            jk.call_batch(&cols, &mut outs);
                        }
                        for j in 0..len {
                            let p = start + j;
                            contrib.clear();
                            for (k, (&sh, &fld)) in shifts.iter().zip(&fields).enumerate() {
                                let w = jbuf[k][j];
                                let q = fld * size + (p as isize + sh) as usize;
                                push_dep(q, w, &self.state_of, &row_of, &ents, &mut contrib);
                            }
                            merge(&mut contrib);
                            match o.target {
                                MolTarget::Rhs(f) => {
                                    let r = self.state_of[f * size + p];
                                    out.extend(contrib.iter().map(|&(c, v)| (r, c, v)));
                                }
                                MolTarget::Fill(f) => {
                                    row_of[f * size + p] = (ents.len(), contrib.len());
                                    ents.extend_from_slice(&contrib);
                                }
                            }
                        }
                    });
                }
            }
        }
        out
    }
}

/// Call `f(flat index of the row start)` for every row (last-axis run) of
/// `region`.
fn for_each_row(strides: &[usize], region: &[(usize, usize)], mut f: impl FnMut(usize)) {
    let d = region.len();
    let lo_last = region[d - 1].0;
    match d {
        1 => f(lo_last),
        2 => {
            for i in region[0].0..region[0].1 {
                f(i * strides[0] + lo_last);
            }
        }
        _ => {
            for i in region[0].0..region[0].1 {
                for j in region[1].0..region[1].1 {
                    f(i * strides[0] + j * strides[1] + lo_last);
                }
            }
        }
    }
}

/// Call `f(flat index)` for every point of `region`, row-major.
fn for_each_point(strides: &[usize], region: &[(usize, usize)], mut f: impl FnMut(usize)) {
    let d = region.len();
    let (lo, hi) = region[d - 1];
    for_each_row(strides, region, |start| {
        for p in start..start + (hi - lo) {
            f(p);
        }
    });
}

#[cfg(test)]
mod tests;
