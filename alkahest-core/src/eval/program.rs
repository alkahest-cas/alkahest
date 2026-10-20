//! A flat, reusable `f64` program for evaluating one expression many times.
//!
//! [`crate::jit::eval_interp`] walks the pool on every call: it looks every
//! node up, builds a fresh memo table, and resolves each function by name in
//! the primitive registry. That is the right trade for a single evaluation, and
//! the wrong one for a sampler or a batch that evaluates the *same* expression
//! at thousands of points. [`NumericProgram`] does that work once: the DAG is
//! flattened into a post-order list of operations over numbered slots (each
//! shared node gets exactly one slot, so sharing is preserved without a memo),
//! constants are folded to their `f64` value, and every function head is
//! resolved to its primitive.
//!
//! # Agreement with the interpreter
//!
//! A program computes, bit for bit, what [`crate::jit::eval_interp`] computes
//! for the same bindings — same operations, same order, same numeric kernels:
//!
//! * `Add` folds left from `0.0` and `Mul` from `1.0`, exactly as the
//!   interpreter does (`0.0 + -0.0` is `+0.0`, so the seed is not cosmetic);
//! * powers go through [`pow_f64`] with the exponent's exact node, so a wide
//!   integer exponent keeps its parity;
//! * functions call the same [`Primitive::numeric_f64`] kernel the registry
//!   dispatches to;
//! * an unbound symbol (other than `π`) or an unknown function head makes every
//!   evaluation `None`, as it does in the interpreter, where such a node is
//!   always reached.
//!
//! Only forms whose every node is reached on every evaluation are compiled.
//! `Piecewise`, `Predicate` and `RootSum` — where the interpreter evaluates a
//! branch lazily or does something other than arithmetic — are declined
//! (`compile` returns `None`) and the caller keeps its tree walk for them.
//!
//! # Blocks of points
//!
//! A batch is evaluated a *block* of points at a time
//! ([`eval_slices`](NumericProgram::eval_slices),
//! [`eval_columns`](NumericProgram::eval_columns)): every slot holds one lane
//! per point of the block, and each operation runs over all lanes before the
//! next one starts — one dispatch per operation per block rather than per
//! point, and loops simple enough for the compiler to vectorise (`Add`, `Mul`,
//! powers with a constant exponent). Each lane computes exactly the
//! operations, in exactly the order, a single-point evaluation does, so a
//! block result is bit for bit the single-point result. A point at which an
//! operation has no value (a kernel's `None`) is tracked by a per-lane failure
//! flag and comes out `NaN`, as a single-point failure does.
//!
//! # Piecewise
//!
//! [`compile_lazy`](NumericProgram::compile_lazy) also takes `Piecewise` and
//! predicates (comparisons, `And` / `Or` / `Not`), with the semantics of
//! [`crate::jit::eval_interp`]: branches are tried in order, `And` / `Or`
//! short-circuit, a condition or a selected value that has no value at a
//! point makes the point fail, and a branch that is *not* selected never
//! does; a predicate used as a value is `1.0` or `0.0`. Every branch is
//! computed on every lane, and the per-lane failure flags carry the
//! laziness: a lane's result takes the value and the failure flag of the
//! branch it selects — a masked select.

use crate::kernel::{integer_to_f64, pow_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use crate::primitive::Primitive;

use super::IdMap;

/// Index of a value slot.
type Slot = u32;

#[derive(Clone, Copy)]
enum OpKind {
    /// `0.0 + s[a] + s[b]`.
    Add2(Slot, Slot),
    /// `0.0 + Σ s[operands[start..start+len]]`, left to right.
    Add { start: u32, len: u32 },
    /// `1.0 · s[a] · s[b]`.
    Mul2(Slot, Slot),
    /// `1.0 · Π s[operands[start..start+len]]`, left to right.
    Mul { start: u32, len: u32 },
    /// `pow_f64(s[base], s[exp], exp_node)`; `wide` indexes
    /// [`NumericProgram::exp_nodes`] when the exponent node is an exact
    /// number (`u32::MAX` otherwise). `konst`: the exponent is a numeric
    /// constant, the same in every lane of a block.
    Pow {
        base: Slot,
        exp: Slot,
        wide: u32,
        konst: bool,
    },
    /// A one-argument primitive.
    Func1 {
        f: &'static dyn Primitive,
        arg: Slot,
    },
    /// A primitive of any other arity.
    FuncN {
        f: &'static dyn Primitive,
        start: u32,
        len: u32,
    },
    /// A function head with no numeric kernel: no value anywhere.
    Fail,
    /// `1.0` when `s[a] <op> s[b]`, else `0.0` (lazy programs only).
    Cmp(CmpKind, Slot, Slot),
    /// `1.0` when `s[a] == 0.0`, else `0.0` (lazy programs only).
    Not(Slot),
    /// Short-circuit conjunction of truth values (lazy programs only): an
    /// argument after the first false one is never consulted.
    And { start: u32, len: u32 },
    /// Short-circuit disjunction (lazy programs only).
    Or { start: u32, len: u32 },
    /// `operands[start..start+len]` = `c1, v1, …, ck, vk, default`: the value
    /// of the first branch whose condition holds (lazy programs only).
    Piecewise { start: u32, len: u32 },
}

#[derive(Clone, Copy)]
enum CmpKind {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpKind {
    #[inline(always)]
    fn holds(self, a: f64, b: f64) -> bool {
        match self {
            CmpKind::Lt => a < b,
            CmpKind::Le => a <= b,
            CmpKind::Gt => a > b,
            CmpKind::Ge => a >= b,
            CmpKind::Eq => a == b,
            CmpKind::Ne => a != b,
        }
    }
}

#[derive(Clone, Copy)]
struct Op {
    kind: OpKind,
    dst: Slot,
}

/// A compiled, reusable evaluation plan. See the [module docs](self).
#[derive(Clone)]
pub(crate) struct NumericProgram {
    n_inputs: usize,
    /// Initial slot contents: `n_inputs` input slots, then constants; slots
    /// written by operations follow. Constants are never overwritten, so one
    /// scratch buffer can be reused across any number of evaluations.
    template: Vec<f64>,
    ops: Vec<Op>,
    operands: Vec<Slot>,
    /// Exponent nodes a wide power needs to see exactly (cold path only).
    exp_nodes: Vec<ExprData>,
    result: Slot,
    /// The expression reaches a node no binding can make evaluable.
    always_fails: bool,
    /// Constant slots with no value (an unbound symbol, a non-real root):
    /// every lane of them starts out failed in a block.
    failing: Vec<Slot>,
    /// Built by [`compile_lazy`](Self::compile_lazy): has `Piecewise`
    /// branches, so single points are evaluated through the block runner,
    /// which tracks failures per slot.
    lazy: bool,
}

/// Whether [`NumericProgram::compile_lazy`] models every node of `expr`.
///
/// Values are numbers, symbols, arithmetic, functions (`RootOf` folded to its
/// value), `Piecewise`, and predicates (`1.0` / `0.0`, as [`eval_interp`]
/// gives them). A `Piecewise` condition and an argument of `And` / `Or` /
/// `Not` must be a predicate node — anything else has no truth value in any
/// evaluator here, and is left to the tree walk, as are `RootSum`, `BigO`,
/// quantifiers and a predicate of the wrong arity.
///
/// [`eval_interp`]: crate::jit::eval_interp
fn lazy_supported(expr: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::expr::PredicateKind as K;
    // `(node, must be a predicate)`.
    let mut seen: std::collections::HashSet<(ExprId, bool)> = Default::default();
    let mut stack = vec![(expr, false)];
    while let Some((id, cond)) = stack.pop() {
        if !seen.insert((id, cond)) {
            continue;
        }
        let ok = pool.with(id, |data| match data {
            ExprData::Predicate { kind, args } => match kind {
                K::True | K::False => true,
                K::Not if args.len() == 1 => {
                    stack.push((args[0], true));
                    true
                }
                K::And | K::Or => {
                    stack.extend(args.iter().map(|&a| (a, true)));
                    true
                }
                K::Lt | K::Le | K::Gt | K::Ge | K::Eq | K::Ne if args.len() == 2 => {
                    stack.push((args[0], false));
                    stack.push((args[1], false));
                    true
                }
                _ => false,
            },
            _ if cond => false,
            ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_)
            | ExprData::Symbol { .. } => true,
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                stack.extend(args.iter().map(|&a| (a, false)));
                true
            }
            ExprData::Pow { base, exp } => {
                stack.push((*base, false));
                stack.push((*exp, false));
                true
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push((c, true));
                    stack.push((v, false));
                }
                stack.push((*default, false));
                true
            }
            _ => false,
        });
        if !ok {
            return false;
        }
    }
    true
}

/// Which primitive a `Func` head resolves to — the same table
/// [`crate::jit::eval_interp`] consults by name.
fn primitive(name: &str) -> Option<&'static dyn Primitive> {
    crate::jit::registry().get(name)
}

impl NumericProgram {
    /// Compile `expr` with `inputs[i]` bound to input slot `i`.
    ///
    /// A symbol listed twice takes its value from the *last* position, as a
    /// `HashMap` environment built by inserting `inputs` in order would.
    /// Returns `None` for a form the program does not model (see the module
    /// docs); the caller should fall back to the tree-walking interpreter.
    pub(crate) fn compile(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Option<Self> {
        Self::compile_mode(expr, inputs, pool, false)
    }

    /// Like [`compile`](Self::compile), and also accepts `Piecewise` and
    /// predicates (see the [module docs](self#piecewise)): each point gets
    /// the value `eval_interp` gives it, bit for bit. Returns `None` for a
    /// form outside that (see [`lazy_supported`]).
    pub(crate) fn compile_lazy(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Option<Self> {
        if !lazy_supported(expr, pool) {
            return None;
        }
        Self::compile_mode(expr, inputs, pool, true)
    }

    fn compile_mode(expr: ExprId, inputs: &[ExprId], pool: &ExprPool, lazy: bool) -> Option<Self> {
        let n_inputs = inputs.len();
        let mut template = vec![0.0f64; n_inputs];
        let mut slot_of: IdMap<Slot> = IdMap::default();
        for (i, &var) in inputs.iter().enumerate() {
            // Environments are consulted only at `Symbol` nodes, so an input
            // that is not a symbol binds nothing — and must not here either.
            if pool.with(var, |d| matches!(d, ExprData::Symbol { .. })) {
                slot_of.insert(var, slot(i)?);
            }
        }
        let mut ops: Vec<Op> = Vec::new();
        let mut operands: Vec<Slot> = Vec::new();
        let mut exp_nodes: Vec<ExprData> = Vec::new();
        let mut always_fails = false;
        let mut failing: Vec<Slot> = Vec::new();

        // Iterative post-order: a deep DAG must not overflow the stack here.
        let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];
        while let Some((id, expanded)) = stack.pop() {
            if slot_of.contains_key(&id) {
                continue;
            }
            if !expanded {
                enum Visit {
                    Const(f64),
                    Fail,
                    Children,
                    Unsupported,
                }
                // A predicate the kernel decides exactly (`2 > 1`) is a
                // constant, as `eval_interp` takes it.
                let folded = if lazy && pool.with(id, |d| matches!(d, ExprData::Predicate { .. })) {
                    crate::kernel::eval_const::try_predicate_bool_from_expr(id, pool)
                } else {
                    None
                };
                let visit = pool.with(id, |data| match data {
                    ExprData::Integer(n) => Visit::Const(integer_to_f64(&n.0)),
                    ExprData::Rational(r) => Visit::Const(rational_to_f64(&r.0)),
                    ExprData::Float(f) => Visit::Const(f.inner.to_f64()),
                    // Not an input (inputs were seeded above): `π` has its
                    // own value, anything else is unbound.
                    ExprData::Symbol { name, .. } => {
                        if let Some(v) = super::symbols::constant_value_f64(name) {
                            Visit::Const(v)
                        } else {
                            Visit::Fail
                        }
                    }
                    // An algebraic number: a constant slot holding its value
                    // (a non-real root has no `f64` value, like an unbound
                    // symbol).
                    ExprData::Func { name, .. } if name == crate::poly::root_of::ROOT_OF => {
                        match crate::poly::root_of::root_of_real_f64(id, pool) {
                            Some(v) => Visit::Const(v),
                            None => Visit::Fail,
                        }
                    }
                    ExprData::Add(_)
                    | ExprData::Mul(_)
                    | ExprData::Pow { .. }
                    | ExprData::Func { .. } => Visit::Children,
                    ExprData::Predicate { kind, .. } if lazy => match (folded, kind) {
                        (Some(b), _) => Visit::Const(if b { 1.0 } else { 0.0 }),
                        (None, crate::kernel::expr::PredicateKind::True) => Visit::Const(1.0),
                        (None, crate::kernel::expr::PredicateKind::False) => Visit::Const(0.0),
                        _ => Visit::Children,
                    },
                    ExprData::Piecewise { .. } if lazy => Visit::Children,
                    _ => Visit::Unsupported,
                });
                match visit {
                    Visit::Const(v) => {
                        slot_of.insert(id, slot(template.len())?);
                        template.push(v);
                    }
                    Visit::Fail => {
                        // In a lazy program the node may sit in a branch no
                        // point selects; its lanes fail instead.
                        always_fails |= !lazy;
                        let s = slot(template.len())?;
                        failing.push(s);
                        slot_of.insert(id, s);
                        template.push(f64::NAN);
                    }
                    Visit::Unsupported => return None,
                    Visit::Children => {
                        stack.push((id, true));
                        pool.with(id, |data| match data {
                            ExprData::Add(args)
                            | ExprData::Mul(args)
                            | ExprData::Func { args, .. }
                            | ExprData::Predicate { args, .. } => {
                                // Reversed so children are *finished* in
                                // argument order (only matters for layout).
                                for &a in args.iter().rev() {
                                    stack.push((a, false));
                                }
                            }
                            ExprData::Piecewise { branches, default } => {
                                stack.push((*default, false));
                                for &(c, v) in branches.iter().rev() {
                                    stack.push((v, false));
                                    stack.push((c, false));
                                }
                            }
                            ExprData::Pow { base, exp } => {
                                stack.push((*exp, false));
                                stack.push((*base, false));
                            }
                            _ => unreachable!(),
                        });
                    }
                }
                continue;
            }

            // Every child has a slot now; emit this node's operation.
            let s = |c: &ExprId| slot_of[c];
            let kind = pool.with(id, |data| -> Option<OpKind> {
                Some(match data {
                    ExprData::Add(args) if args.len() == 2 => {
                        OpKind::Add2(s(&args[0]), s(&args[1]))
                    }
                    ExprData::Mul(args) if args.len() == 2 => {
                        OpKind::Mul2(s(&args[0]), s(&args[1]))
                    }
                    ExprData::Add(args) | ExprData::Mul(args) => {
                        let start = u32::try_from(operands.len()).ok()?;
                        operands.extend(args.iter().map(s));
                        let len = u32::try_from(args.len()).ok()?;
                        if matches!(data, ExprData::Add(_)) {
                            OpKind::Add { start, len }
                        } else {
                            OpKind::Mul { start, len }
                        }
                    }
                    ExprData::Pow { base, exp } => {
                        let exact = pool.with(*exp, |e| {
                            matches!(e, ExprData::Integer(_) | ExprData::Rational(_))
                                .then(|| e.clone())
                        });
                        let konst = pool.with(*exp, |e| {
                            matches!(
                                e,
                                ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
                            )
                        });
                        let wide = match exact {
                            Some(node) => {
                                exp_nodes.push(node);
                                u32::try_from(exp_nodes.len() - 1).ok()?
                            }
                            None => u32::MAX,
                        };
                        OpKind::Pow {
                            base: s(base),
                            exp: s(exp),
                            wide,
                            konst,
                        }
                    }
                    ExprData::Func { name, args } => match primitive(name) {
                        None => {
                            // A strict program never runs it: `run` returns
                            // before the first op when `always_fails`.
                            always_fails |= !lazy;
                            OpKind::Fail
                        }
                        Some(f) if args.len() == 1 => OpKind::Func1 {
                            f,
                            arg: s(&args[0]),
                        },
                        Some(f) => {
                            let start = u32::try_from(operands.len()).ok()?;
                            operands.extend(args.iter().map(s));
                            OpKind::FuncN {
                                f,
                                start,
                                len: u32::try_from(args.len()).ok()?,
                            }
                        }
                    },
                    ExprData::Predicate { kind, args } => {
                        use crate::kernel::expr::PredicateKind as K;
                        let cmp = |k| OpKind::Cmp(k, s(&args[0]), s(&args[1]));
                        match kind {
                            K::Lt => cmp(CmpKind::Lt),
                            K::Le => cmp(CmpKind::Le),
                            K::Gt => cmp(CmpKind::Gt),
                            K::Ge => cmp(CmpKind::Ge),
                            K::Eq => cmp(CmpKind::Eq),
                            K::Ne => cmp(CmpKind::Ne),
                            K::Not => OpKind::Not(s(&args[0])),
                            K::And | K::Or => {
                                let start = u32::try_from(operands.len()).ok()?;
                                operands.extend(args.iter().map(s));
                                let len = u32::try_from(args.len()).ok()?;
                                if matches!(kind, K::And) {
                                    OpKind::And { start, len }
                                } else {
                                    OpKind::Or { start, len }
                                }
                            }
                            // Folded to constants when visited.
                            K::True | K::False => unreachable!(),
                        }
                    }
                    ExprData::Piecewise { branches, default } => {
                        let start = u32::try_from(operands.len()).ok()?;
                        for (c, v) in branches {
                            operands.push(s(c));
                            operands.push(s(v));
                        }
                        operands.push(s(default));
                        OpKind::Piecewise {
                            start,
                            len: u32::try_from(2 * branches.len() + 1).ok()?,
                        }
                    }
                    _ => unreachable!(),
                })
            })?;
            let dst = slot(template.len())?;
            template.push(0.0);
            slot_of.insert(id, dst);
            ops.push(Op { kind, dst });
        }

        let prog = Self {
            n_inputs,
            result: slot_of[&expr],
            template,
            ops,
            operands,
            exp_nodes,
            always_fails,
            failing,
            lazy,
        };
        debug_assert!(prog.is_well_formed(), "compiled an ill-formed program");
        Some(prog)
    }

    /// A scratch buffer for [`eval_in`](Self::eval_in); reusable across calls.
    pub(crate) fn scratch(&self) -> Vec<f64> {
        self.template.clone()
    }

    /// Length of a buffer from [`scratch`](Self::scratch).
    pub(crate) fn scratch_len(&self) -> usize {
        self.template.len()
    }

    /// Re-initialise a buffer of [`scratch_len`](Self::scratch_len) slots —
    /// one that may have been used by a *different* program — for this one.
    pub(crate) fn reset_scratch(&self, scratch: &mut [f64]) {
        scratch.copy_from_slice(&self.template);
    }

    /// Evaluate at `inputs` (one value per compiled input), allocating
    /// a scratch buffer. Same value as `eval_interp` with the same bindings.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn eval(&self, inputs: &[f64]) -> Option<f64> {
        let mut scratch = self.scratch();
        self.eval_in(inputs, &mut scratch)
    }

    /// Evaluate at `inputs` using a buffer from [`scratch`](Self::scratch).
    #[inline]
    pub(crate) fn eval_in(&self, inputs: &[f64], scratch: &mut [f64]) -> Option<f64> {
        debug_assert_eq!(inputs.len(), self.n_inputs);
        debug_assert_eq!(scratch.len(), self.template.len());
        if self.lazy {
            return self.eval_lazy_point(inputs);
        }
        scratch[..self.n_inputs].copy_from_slice(inputs);
        self.run(scratch)
    }

    /// One point of a lazy program: a block of one lane, so the failure
    /// tracking that makes an unselected branch harmless is the block's.
    fn eval_lazy_point(&self, inputs: &[f64]) -> Option<f64> {
        let mut b = self.block_scratch(1);
        b.vals[..self.n_inputs].copy_from_slice(inputs);
        self.run_block(&mut b, 1);
        let r = self.result as usize;
        (!(b.any[r] && b.fail[r] != 0)).then_some(b.vals[r])
    }

    /// Evaluate over `output.len()` points laid out column-major in
    /// `inputs_flat` (input `i` of point `j` at `i * n_points + j`), writing
    /// `NaN` where an evaluation fails — the interpreter tier's convention.
    /// Runs a block of points at a time; each value is bit for bit the
    /// single-point value.
    pub(crate) fn eval_columns(&self, inputs_flat: &[f64], output: &mut [f64]) {
        let n_points = output.len();
        debug_assert_eq!(inputs_flat.len(), self.n_inputs * n_points);
        let cols: Vec<&[f64]> = (0..self.n_inputs)
            .map(|i| &inputs_flat[i * n_points..(i + 1) * n_points])
            .collect();
        self.eval_slices(&cols, output, 0);
    }

    /// Like [`eval_columns`](Self::eval_columns) with one slice per input;
    /// point `j` of `output` reads element `offset + j` of every input.
    pub(crate) fn eval_slices(&self, inputs: &[&[f64]], output: &mut [f64], offset: usize) {
        debug_assert_eq!(inputs.len(), self.n_inputs);
        if self.always_fails {
            output.fill(f64::NAN);
            return;
        }
        if output.is_empty() {
            return;
        }
        let w = self.block_width(output.len());
        let mut b = self.block_scratch(w);
        for (k, out) in output.chunks_mut(w).enumerate() {
            let start = offset + k * w;
            let m = out.len();
            for (i, col) in inputs.iter().enumerate() {
                b.vals[i * w..i * w + m].copy_from_slice(&col[start..start + m]);
            }
            self.run_block(&mut b, m);
            let r = self.result as usize;
            out.copy_from_slice(&b.vals[r * w..r * w + m]);
            if b.any[r] {
                for (o, &f) in out.iter_mut().zip(&b.fail[r * w..r * w + m]) {
                    if f != 0 {
                        *o = f64::NAN;
                    }
                }
            }
        }
    }

    /// Points per block: up to 256, fewer for a program with many slots so
    /// that a block's values stay within 256 KiB (cache-resident).
    fn block_width(&self, n_points: usize) -> usize {
        let slots = self.template.len().max(1);
        let cap = ((1usize << 15) / slots).clamp(8, 256) & !7;
        cap.min(n_points.max(1))
    }

    /// Storage for [`run_block`](Self::run_block) with `width` lanes per
    /// slot: input lanes (filled per block), every constant broadcast to all
    /// its lanes, and every lane of a constant with no value marked failed.
    fn block_scratch(&self, width: usize) -> BlockScratch {
        let slots = self.template.len();
        let mut vals = vec![0.0f64; slots * width];
        for (k, &v) in self.template.iter().enumerate().skip(self.n_inputs) {
            vals[k * width..(k + 1) * width].fill(v);
        }
        let mut fail = vec![0u8; slots * width];
        let mut any = vec![false; slots];
        for &s in &self.failing {
            let s = s as usize;
            fail[s * width..(s + 1) * width].fill(1);
            any[s] = true;
        }
        BlockScratch {
            width,
            vals,
            fail,
            any,
        }
    }

    /// Run every operation over lanes `0..n` of `b` (`1 <= n <= b.width`),
    /// whose input lanes the caller has filled.
    ///
    /// Each lane computes what [`run`](Self::run) computes for its point —
    /// same operations, same order, same kernels. Failures are per lane and
    /// per slot: `b.any[s]` says whether any lane of slot `s` failed, and
    /// only then are its `b.fail` lanes meaningful. A strict operation's lane
    /// fails where an operand's lane failed (or its kernel has no value
    /// there); `And`, `Or` and `Piecewise` take failures only from the
    /// arguments a point actually consults.
    fn run_block(&self, b: &mut BlockScratch, n: usize) {
        let w = b.width;
        debug_assert!((1..=w).contains(&n));
        for op in &self.ops {
            let dst = op.dst as usize;
            // Every operand slot is below `dst` (`is_well_formed`), so the
            // operands are all in `lo` / `flo`.
            let (lo, hi) = b.vals.split_at_mut(dst * w);
            let d = &mut hi[..n];
            let (flo, fhi) = b.fail.split_at_mut(dst * w);
            let fd = &mut fhi[..n];
            let any = &mut b.any;
            let lane = |s: Slot| &lo[s as usize * w..s as usize * w + n];
            let flane = |s: Slot| &flo[s as usize * w..s as usize * w + n];
            // A strict operation's lane fails where an operand's lane did.
            let merge = |any: &mut Vec<bool>, fd: &mut [u8], srcs: &[Slot]| {
                any[dst] = false;
                for &src in srcs {
                    if !any[src as usize] {
                        continue;
                    }
                    if any[dst] {
                        for (o, &f) in fd.iter_mut().zip(flane(src)) {
                            *o |= f;
                        }
                    } else {
                        fd.copy_from_slice(flane(src));
                        any[dst] = true;
                    }
                }
            };
            // A kernel with no value at lane `p`.
            let fail_lane = |any: &mut Vec<bool>, fd: &mut [u8], p: usize| {
                if !any[dst] {
                    fd.fill(0);
                    any[dst] = true;
                }
                fd[p] = 1;
            };
            match op.kind {
                OpKind::Add2(a, c) => {
                    merge(any, fd, &[a, c]);
                    for ((o, &x), &y) in d.iter_mut().zip(lane(a)).zip(lane(c)) {
                        *o = 0.0 + x + y;
                    }
                }
                OpKind::Mul2(a, c) => {
                    merge(any, fd, &[a, c]);
                    for ((o, &x), &y) in d.iter_mut().zip(lane(a)).zip(lane(c)) {
                        *o = 1.0 * x * y;
                    }
                }
                OpKind::Add { start, len } => {
                    let args = &self.operands[operand_range(start, len)];
                    merge(any, fd, args);
                    d.fill(0.0);
                    for &a in args {
                        for (o, &x) in d.iter_mut().zip(lane(a)) {
                            *o += x;
                        }
                    }
                }
                OpKind::Mul { start, len } => {
                    let args = &self.operands[operand_range(start, len)];
                    merge(any, fd, args);
                    d.fill(1.0);
                    for &a in args {
                        for (o, &x) in d.iter_mut().zip(lane(a)) {
                            *o *= x;
                        }
                    }
                }
                OpKind::Pow {
                    base,
                    exp,
                    wide,
                    konst,
                } => {
                    merge(any, fd, &[base, exp]);
                    let node = self.exp_nodes.get(wide as usize);
                    let (xb, xe) = (lane(base), lane(exp));
                    if konst {
                        crate::kernel::eval_const::pow_f64_slice(xb, xe[0], node, d);
                    } else {
                        for ((o, &x), &e) in d.iter_mut().zip(xb).zip(xe) {
                            *o = pow_f64(x, e, || node.cloned());
                        }
                    }
                }
                OpKind::Func1 { f, arg } => {
                    merge(any, fd, &[arg]);
                    for (p, (o, &x)) in d.iter_mut().zip(lane(arg)).enumerate() {
                        match f.numeric_f64(&[x]) {
                            Some(v) => *o = v,
                            None => {
                                *o = f64::NAN;
                                fail_lane(any, fd, p);
                            }
                        }
                    }
                }
                OpKind::FuncN { f, start, len } => {
                    let args = &self.operands[operand_range(start, len)];
                    merge(any, fd, args);
                    let mut buf = vec![0.0f64; args.len()];
                    for (p, o) in d.iter_mut().enumerate() {
                        for (v, &a) in buf.iter_mut().zip(args) {
                            *v = lo[a as usize * w + p];
                        }
                        match f.numeric_f64(&buf) {
                            Some(v) => *o = v,
                            None => {
                                *o = f64::NAN;
                                fail_lane(any, fd, p);
                            }
                        }
                    }
                }
                OpKind::Fail => {
                    d.fill(f64::NAN);
                    fd.fill(1);
                    any[dst] = true;
                }
                OpKind::Cmp(k, a, c) => {
                    merge(any, fd, &[a, c]);
                    for ((o, &x), &y) in d.iter_mut().zip(lane(a)).zip(lane(c)) {
                        *o = if k.holds(x, y) { 1.0 } else { 0.0 };
                    }
                }
                OpKind::Not(a) => {
                    merge(any, fd, &[a]);
                    for (o, &x) in d.iter_mut().zip(lane(a)) {
                        *o = if x == 0.0 { 1.0 } else { 0.0 };
                    }
                }
                // Folded from the last argument back: lane by lane, an
                // argument that fails makes the result fail, one that decides
                // (false for `And`, true for `Or`) fixes the result with no
                // failure, and any other leaves what the later arguments gave
                // — which is the left-to-right short circuit.
                OpKind::And { start, len } | OpKind::Or { start, len } => {
                    let decides_on = !matches!(op.kind, OpKind::And { .. });
                    let args = &self.operands[operand_range(start, len)];
                    d.fill(if decides_on { 0.0 } else { 1.0 });
                    fd.fill(0);
                    let mut any_out = false;
                    for &a in args.iter().rev() {
                        let xs = lane(a);
                        let fs = any[a as usize].then(|| flane(a));
                        any_out |= fs.is_some();
                        for p in 0..n {
                            if fs.is_some_and(|fs| fs[p] != 0) {
                                fd[p] = 1;
                            } else if (xs[p] != 0.0) == decides_on {
                                d[p] = if decides_on { 1.0 } else { 0.0 };
                                fd[p] = 0;
                            }
                        }
                    }
                    any[dst] = any_out;
                }
                // Folded from the default back, as `And` / `Or` are: a lane
                // whose condition fails fails, one whose condition holds
                // takes that branch's value and failure, and any other keeps
                // what the later branches gave.
                OpKind::Piecewise { start, len } => {
                    let args = &self.operands[operand_range(start, len)];
                    let (pairs, default) = args.split_at(args.len() - 1);
                    let dflt = default[0];
                    d.copy_from_slice(lane(dflt));
                    let mut any_out = any[dflt as usize];
                    if any_out {
                        fd.copy_from_slice(flane(dflt));
                    } else {
                        fd.fill(0);
                    }
                    for pair in pairs.chunks_exact(2).rev() {
                        let (c, v) = (pair[0], pair[1]);
                        let (cs, vs) = (lane(c), lane(v));
                        let cf = any[c as usize].then(|| flane(c));
                        let vf = any[v as usize].then(|| flane(v));
                        any_out |= cf.is_some() || vf.is_some();
                        if cf.is_none() && vf.is_none() {
                            for ((o, f), (&cv, &x)) in
                                d.iter_mut().zip(fd.iter_mut()).zip(cs.iter().zip(vs))
                            {
                                if cv != 0.0 {
                                    *o = x;
                                    *f = 0;
                                }
                            }
                        } else {
                            for p in 0..n {
                                if cf.is_some_and(|cf| cf[p] != 0) {
                                    fd[p] = 1;
                                } else if cs[p] != 0.0 {
                                    d[p] = vs[p];
                                    fd[p] = vf.map_or(0, |vf| vf[p]);
                                }
                            }
                        }
                    }
                    any[dst] = any_out;
                }
            }
        }
    }

    #[inline]
    fn run(&self, s: &mut [f64]) -> Option<f64> {
        if self.always_fails {
            return None;
        }
        for op in &self.ops {
            let v = match op.kind {
                OpKind::Add2(a, b) => 0.0 + s[a as usize] + s[b as usize],
                OpKind::Mul2(a, b) => 1.0 * s[a as usize] * s[b as usize],
                OpKind::Add { start, len } => {
                    let mut acc = 0.0f64;
                    for &a in &self.operands[operand_range(start, len)] {
                        acc += s[a as usize];
                    }
                    acc
                }
                OpKind::Mul { start, len } => {
                    let mut acc = 1.0f64;
                    for &a in &self.operands[operand_range(start, len)] {
                        acc *= s[a as usize];
                    }
                    acc
                }
                OpKind::Pow {
                    base, exp, wide, ..
                } => {
                    let node = self.exp_nodes.get(wide as usize);
                    pow_f64(s[base as usize], s[exp as usize], || node.cloned())
                }
                OpKind::Func1 { f, arg } => f.numeric_f64(&[s[arg as usize]])?,
                OpKind::FuncN { f, start, len } => {
                    let args = &self.operands[operand_range(start, len)];
                    if args.len() <= 8 {
                        let mut buf = [0.0f64; 8];
                        for (k, &a) in args.iter().enumerate() {
                            buf[k] = s[a as usize];
                        }
                        f.numeric_f64(&buf[..args.len()])?
                    } else {
                        let vals: Vec<f64> = args.iter().map(|&a| s[a as usize]).collect();
                        f.numeric_f64(&vals)?
                    }
                }
                OpKind::Fail => return None,
                // Only lazy programs hold these, and `eval_in` sends those
                // to the block runner.
                OpKind::Cmp(..)
                | OpKind::Not(_)
                | OpKind::And { .. }
                | OpKind::Or { .. }
                | OpKind::Piecewise { .. } => {
                    debug_assert!(false, "lazy operation in a strict run");
                    return None;
                }
            };
            s[op.dst as usize] = v;
        }
        Some(s[self.result as usize])
    }
}

/// A block's working storage (see [`NumericProgram::run_block`]): slot `s`
/// holds lanes `s * width .. (s + 1) * width` of `vals` and `fail`.
struct BlockScratch {
    width: usize,
    vals: Vec<f64>,
    /// Per lane: `1` when the lane has no value.
    fail: Vec<u8>,
    /// Per slot: whether any lane failed (otherwise its `fail` lanes are
    /// stale and must not be read).
    any: Vec<bool>,
}

fn slot(n: usize) -> Option<Slot> {
    Slot::try_from(n).ok().filter(|&s| s != u32::MAX)
}

/// `operands[start..start + len]` as a `usize` range. Summed in `usize`, not
/// `u32`: `start + len` in `u32` would wrap once the operand list passed
/// `u32::MAX` entries, which `compile` does not rule out on its own.
#[inline]
fn operand_range(start: u32, len: u32) -> std::ops::Range<usize> {
    start as usize..start as usize + len as usize
}

impl NumericProgram {
    /// The invariant `run` relies on for every index it takes: each operation
    /// writes a fresh slot after the inputs, in increasing order, and reads
    /// only slots below its own (inputs, constants, or earlier results); every
    /// operand range lies inside `operands`; a wide power's exponent index is
    /// `u32::MAX` or inside `exp_nodes`; the result is a slot. `compile`
    /// `debug_assert`s it, and a Kani harness proves `run` in bounds on any
    /// program that satisfies it (TESTING.md § 7).
    fn is_well_formed(&self) -> bool {
        let len = self.template.len();
        if self.n_inputs > len || self.result as usize >= len {
            return false;
        }
        let range_ok = |start: u32, n: u32, dst: Slot| {
            let r = operand_range(start, n);
            r.end <= self.operands.len() && self.operands[r].iter().all(|&a| a < dst)
        };
        let mut prev: Option<Slot> = None;
        for op in &self.ops {
            let dst = op.dst;
            if (dst as usize) < self.n_inputs
                || dst as usize >= len
                || prev.is_some_and(|p| dst <= p)
            {
                return false;
            }
            prev = Some(dst);
            let ok = match op.kind {
                OpKind::Add2(a, b) | OpKind::Mul2(a, b) => a < dst && b < dst,
                OpKind::Add { start, len }
                | OpKind::Mul { start, len }
                | OpKind::And { start, len }
                | OpKind::Or { start, len } => range_ok(start, len, dst),
                OpKind::Piecewise { start, len } => len % 2 == 1 && range_ok(start, len, dst),
                OpKind::Cmp(_, a, b) => a < dst && b < dst,
                OpKind::Not(a) => a < dst,
                OpKind::Fail => true,
                OpKind::Pow {
                    base, exp, wide, ..
                } => {
                    base < dst
                        && exp < dst
                        && (wide == u32::MAX || (wide as usize) < self.exp_nodes.len())
                }
                OpKind::Func1 { arg, .. } => arg < dst,
                OpKind::FuncN { start, len, .. } => range_ok(start, len, dst),
            };
            if !ok {
                return false;
            }
        }
        true
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    const MAX_SLOTS: u32 = 5;

    fn any_arith_op() -> Op {
        // Full `u32` range: `is_well_formed` must reject what `run` cannot index.
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let dst: u32 = kani::any();
        let kind = match kani::any::<u8>() % 4 {
            0 => OpKind::Add2(a, b),
            1 => OpKind::Mul2(a, b),
            // `a`, `b` double as `start`, `len` of an n-ary range.
            2 => OpKind::Add { start: a, len: b },
            _ => OpKind::Mul { start: a, len: b },
        };
        Op { kind, dst }
    }

    /// Any program of two arithmetic operations over five slots and three
    /// operand entries — every `u32` slot index, operand range, destination, the
    /// input count and the result symbolic — that passes `is_well_formed`
    /// runs without an out-of-bounds index or a panic. The sizes are fixed
    /// (a symbolic-length `Vec` multiplies the solver's work) and constants
    /// are `1.0`: indexing, not arithmetic, is what is checked. `Pow`/`Func*`
    /// read slots through the same `s[x as usize]` and are bounded by the
    /// same clauses, but their kernels (`pow_f64`, `dyn Primitive`) are
    /// outside what Kani can run.
    #[kani::proof]
    #[kani::unwind(6)]
    fn run_in_bounds_when_well_formed_small() {
        let prog = NumericProgram {
            n_inputs: kani::any_where(|&n| n <= MAX_SLOTS as usize),
            template: vec![1.0; MAX_SLOTS as usize],
            ops: vec![any_arith_op(), any_arith_op()],
            operands: vec![kani::any(), kani::any(), kani::any()],
            exp_nodes: Vec::new(),
            result: kani::any(),
            always_fails: false,
            failing: Vec::new(),
            lazy: false,
        };
        kani::assume(prog.is_well_formed());
        let mut scratch = prog.scratch();
        assert!(prog.run(&mut scratch).is_some());
    }

    /// `operand_range` never wraps: for every `start` and `len` the range
    /// has exactly `len` entries (it once summed them in `u32`).
    #[kani::proof]
    fn operand_range_no_wrap_full_width() {
        let start: u32 = kani::any();
        let len: u32 = kani::any();
        let r = operand_range(start, len);
        assert_eq!(r.start, start as usize);
        assert_eq!(r.end - r.start, len as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use proptest::prelude::*;
    use rug::ops::Pow;
    use std::collections::HashMap;

    fn same(a: Option<f64>, b: Option<f64>) -> bool {
        match (a, b) {
            (Some(x), Some(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            (None, None) => true,
            _ => false,
        }
    }

    fn check(expr: ExprId, vars: &[ExprId], pool: &ExprPool, points: &[Vec<f64>]) {
        let prog = NumericProgram::compile(expr, vars, pool).expect("compilable");
        for pt in points {
            let env: HashMap<ExprId, f64> = vars.iter().copied().zip(pt.iter().copied()).collect();
            let want = eval_interp(expr, &env, pool);
            let got = prog.eval(pt);
            assert!(
                same(want, got),
                "{} at {pt:?}: interp {want:?} vs program {got:?}",
                pool.display(expr)
            );
        }
    }

    #[test]
    fn matches_interpreter_on_mixed_heads() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let pi = crate::eval::symbols::pi_symbol(&pool);
        let e = pool.add(vec![
            pool.mul(vec![pool.func("sin", vec![x]), pool.func("exp", vec![y])]),
            pool.pow(x, pool.integer(3_i32)),
            pool.pow(x, pool.rational(1, 2)),
            pool.pow(y, x),
            pool.mul(vec![pool.rational(-7, 3), pi]),
            pool.func("atan2", vec![y, x]),
            pool.func("tan", vec![pool.add(vec![x, y])]),
            pool.float(1.25, 53),
        ]);
        let pts: Vec<Vec<f64>> = [-2.5, -1.0, -0.0, 0.0, 0.3, 1.0, 7.5]
            .iter()
            .flat_map(|&a| [-1.5, 0.0, 2.0].iter().map(move |&b| vec![a, b]))
            .collect();
        check(e, &[x, y], &pool, &pts);
    }

    #[test]
    fn wide_integer_exponent_keeps_its_parity() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let odd = pool.integer(rug::Integer::from(10).pow(30u32) + 1u32);
        let e = pool.pow(x, odd);
        let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
        assert_eq!(prog.eval(&[-1.0]), Some(-1.0));
        check(
            e,
            &[x],
            &pool,
            &[vec![-1.0], vec![1.0], vec![-2.0], vec![0.5]],
        );
    }

    #[test]
    fn unbound_symbol_and_unknown_head_always_fail() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let e1 = pool.add(vec![x, z]);
        let e2 = pool.func("no_such_function", vec![x]);
        for e in [e1, e2] {
            let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
            assert_eq!(prog.eval(&[1.0]), None);
            check(e, &[x], &pool, &[vec![1.0]]);
            let mut out = [0.0; 3];
            prog.eval_columns(&[1.0, 2.0, 3.0], &mut out);
            assert!(out.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn duplicate_input_takes_the_last_position() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let prog = NumericProgram::compile(x, &[x, x], &pool).unwrap();
        assert_eq!(prog.eval(&[1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn lazy_forms_are_declined() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_ge(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        assert!(NumericProgram::compile(pw, &[x], &pool).is_none());
        let inside = pool.add(vec![pool.integer(1_i32), pw]);
        assert!(NumericProgram::compile(inside, &[x], &pool).is_none());
    }

    #[test]
    fn deep_chain_compiles_without_recursion() {
        // 200k nested heads on the default 2 MiB test-thread stack: a
        // recursive compile would overflow long before the end.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut e = x;
        for _ in 0..200_000 {
            e = pool.func("sin", vec![e]);
        }
        let prog = NumericProgram::compile(e, &[x], &pool).unwrap();
        let want = (0..200_000).fold(0.5f64, |v, _| v.sin());
        let got = prog.eval(&[0.5]).unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn chebyshev_dag_is_linear_in_distinct_nodes() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let (mut a, mut b) = (pool.integer(1_i32), x);
        for _ in 0..60 {
            let next = pool.add(vec![
                pool.mul(vec![two, x, b]),
                pool.mul(vec![pool.integer(-1_i32), a]),
            ]);
            a = b;
            b = next;
        }
        let prog = NumericProgram::compile(b, &[x], &pool).unwrap();
        assert!(prog.ops.len() < 200, "{} ops", prog.ops.len());
        // T_61(cos t) = cos(61 t).
        let t = 0.3f64;
        let v = prog.eval(&[t.cos()]).unwrap();
        assert!((v - (61.0 * t).cos()).abs() < 1e-9, "{v}");
        check(b, &[x], &pool, &[vec![t.cos()], vec![-0.7], vec![1.0]]);
    }

    // Random expressions over a small vocabulary: the program must agree with
    // the interpreter bit for bit (or both NaN, or both refuse).
    fn arb_expr(depth: u32) -> BoxedStrategy<Tree> {
        let leaf = prop_oneof![
            Just(Tree::X),
            Just(Tree::Y),
            Just(Tree::Pi),
            Just(Tree::Z),
            (-5i32..=5).prop_map(Tree::Int),
            ((-9i32..=9), (1i32..=7)).prop_map(|(n, d)| Tree::Rat(n, d)),
            (-3.0f64..3.0).prop_map(Tree::Float),
        ];
        leaf.prop_recursive(depth, 48, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 1..5).prop_map(Tree::Add),
                prop::collection::vec(inner.clone(), 1..5).prop_map(Tree::Mul),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Tree::Pow(Box::new(a), Box::new(b))),
                (
                    prop::sample::select(vec![
                        "sin", "cos", "exp", "log", "sqrt", "tan", "abs", "atan", "sinh", "erf",
                        "gamma", "floor", "nope"
                    ]),
                    inner.clone()
                )
                    .prop_map(|(f, a)| Tree::F1(f, Box::new(a))),
                (inner.clone(), inner).prop_map(|(a, b)| Tree::Atan2(Box::new(a), Box::new(b))),
            ]
        })
        .boxed()
    }

    #[derive(Debug, Clone)]
    enum Tree {
        X,
        Y,
        Z,
        Pi,
        Int(i32),
        Rat(i32, i32),
        Float(f64),
        Add(Vec<Tree>),
        Mul(Vec<Tree>),
        Pow(Box<Tree>, Box<Tree>),
        F1(&'static str, Box<Tree>),
        Atan2(Box<Tree>, Box<Tree>),
    }

    fn build(t: &Tree, pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        match t {
            Tree::X => x,
            Tree::Y => y,
            Tree::Z => pool.symbol("z", Domain::Real),
            Tree::Pi => crate::eval::symbols::pi_symbol(pool),
            Tree::Int(n) => pool.integer(*n),
            Tree::Rat(n, d) => pool.rational(*n, *d),
            Tree::Float(f) => pool.float(*f, 53),
            Tree::Add(v) => pool.add(v.iter().map(|c| build(c, pool, x, y)).collect()),
            Tree::Mul(v) => pool.mul(v.iter().map(|c| build(c, pool, x, y)).collect()),
            Tree::Pow(a, b) => pool.pow(build(a, pool, x, y), build(b, pool, x, y)),
            Tree::F1(f, a) => pool.func(*f, vec![build(a, pool, x, y)]),
            Tree::Atan2(a, b) => {
                pool.func("atan2", vec![build(a, pool, x, y), build(b, pool, x, y)])
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn program_agrees_with_eval_interp(
            t in arb_expr(4),
            pts in prop::collection::vec((-4.0f64..4.0, -4.0f64..4.0), 1..6),
        ) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let e = build(&t, &pool, x, y);
            let prog = NumericProgram::compile(e, &[x, y], &pool).expect("no lazy forms generated");
            let mut scratch = prog.scratch();
            for (a, b) in pts {
                let env = HashMap::from([(x, a), (y, b)]);
                let want = eval_interp(e, &env, &pool);
                let got = prog.eval_in(&[a, b], &mut scratch);
                prop_assert!(same(want, got), "{} at ({a}, {b}): {want:?} vs {got:?}", pool.display(e));
            }
        }
    }
}
