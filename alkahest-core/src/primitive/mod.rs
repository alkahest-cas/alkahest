//! Primitive registry — central table mapping function names to their
//! full capability bundles.
//!
//! # Motivation
//!
//! Before this registry existed, adding a new primitive (e.g. `erf`) required
//! editing match arms in `simplify/rules.rs`, `diff/forward.rs`,
//! `diff/reverse.rs`, `jit/mod.rs`, `ball/mod.rs`, and `horner.rs`
//! independently.  The registry collapses that to one call:
//! `registry.register(Box::new(ErfPrimitive))`.
//!
//! # Design
//!
//! Each primitive implements the [`Primitive`] trait.  Every method except
//! [`Primitive::name`] and [`Primitive::pretty`] is optional — returning
//! `None` means "not implemented yet".  Callers fall back gracefully
//! (e.g. `diff_forward` returns a `Derivative(...)` placeholder if the
//! registry returns `None`).
//!
//! The [`Capabilities`] bitfield lets tooling and agents ask
//! "can I JIT this expression?" without attempting the operation.
//!
//! # Example
//!
//! ```rust
//! use alkahest_cas::primitive::{PrimitiveRegistry, Capabilities};
//!
//! let reg = PrimitiveRegistry::default_registry();
//! let caps = reg.capabilities("sin");
//! assert!(caps.contains(Capabilities::NUMERIC_F64));
//! assert!(caps.contains(Capabilities::DIFF_FORWARD));
//!
//! let report = reg.coverage_report();
//! // Every built-in has at least NUMERIC_F64 coverage.
//! for row in &report.rows {
//!     assert!(row.caps.contains(Capabilities::NUMERIC_F64));
//! }
//! ```

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Capability flags
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Bit-field recording which capability bundle slots a primitive has filled.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Capabilities: u32 {
        const SIMPLIFY      = 1 << 0;
        const DIFF_FORWARD  = 1 << 1;
        const DIFF_REVERSE  = 1 << 2;
        const NUMERIC_F64   = 1 << 3;
        const NUMERIC_BALL  = 1 << 4;
        const LOWER_LLVM    = 1 << 5;
        const LEAN_THEOREM  = 1 << 6;
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = [
            (Capabilities::SIMPLIFY, "simplify"),
            (Capabilities::DIFF_FORWARD, "diff_fwd"),
            (Capabilities::DIFF_REVERSE, "diff_rev"),
            (Capabilities::NUMERIC_F64, "numeric_f64"),
            (Capabilities::NUMERIC_BALL, "numeric_ball"),
            (Capabilities::LOWER_LLVM, "lower_llvm"),
            (Capabilities::LEAN_THEOREM, "lean"),
        ];
        let present: Vec<&str> = names
            .iter()
            .filter(|(flag, _)| self.contains(*flag))
            .map(|(_, name)| *name)
            .collect();
        write!(f, "[{}]", present.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Primitive trait
// ---------------------------------------------------------------------------

/// A primitive function that can be registered in [`PrimitiveRegistry`].
///
/// Only [`name`](Primitive::name) and [`pretty`](Primitive::pretty) are
/// required; every other method is optional and defaults to returning `None`.
pub trait Primitive: 'static + Send + Sync {
    /// The canonical name used in `ExprData::Func { name, .. }`.
    fn name(&self) -> &'static str;

    /// Human-readable display.  Called by `ExprDisplay` for `Func` nodes.
    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String;

    // ── Optional capability bundle ──────────────────────────────────────────

    /// Algebraic simplification: try to reduce `self(args)` to a simpler form.
    fn simplify(&self, _args: &[ExprId], _pool: &ExprPool) -> Option<ExprId> {
        None
    }

    /// Forward-mode differentiation: `d/d_wrt (self(args))`.
    /// Returns `None` if not implemented (caller should return a placeholder).
    fn diff_forward(&self, _args: &[ExprId], _wrt: ExprId, _pool: &ExprPool) -> Option<ExprId> {
        None
    }

    /// Reverse-mode differentiation: cotangent propagation.
    /// Returns one cotangent per argument, or `None` if not implemented.
    fn diff_reverse(
        &self,
        _args: &[ExprId],
        _cotan: ExprId,
        _pool: &ExprPool,
    ) -> Option<Vec<ExprId>> {
        None
    }

    /// Numerical evaluation at `f64` precision.
    fn numeric_f64(&self, _args: &[f64]) -> Option<f64> {
        None
    }

    /// Rigorous ball-arithmetic evaluation.
    fn numeric_ball(&self, _args: &[ArbBall]) -> Option<ArbBall> {
        None
    }

    /// Name of the Lean 4 / Mathlib theorem that certifies this primitive.
    fn lean_theorem(&self) -> Option<&'static str> {
        None
    }
}

// ---------------------------------------------------------------------------
// Coverage report
// ---------------------------------------------------------------------------

/// One row in the coverage report table.
#[derive(Debug, Clone)]
pub struct CoverageRow {
    pub name: String,
    pub caps: Capabilities,
}

/// Human-readable and machine-readable coverage table for all registered
/// primitives.  Returned by [`PrimitiveRegistry::coverage_report`].
#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub rows: Vec<CoverageRow>,
}

impl CoverageReport {
    /// Render as a Markdown table (suitable for CI PR comments or docs).
    pub fn to_markdown(&self) -> String {
        let header = "| Primitive | simplify | diff_fwd | diff_rev | numeric_f64 | numeric_ball | lower_llvm | lean |\n\
                      |---|---|---|---|---|---|---|---|";
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let tick = |flag: Capabilities| {
                    if r.caps.contains(flag) {
                        "✓"
                    } else {
                        "✗"
                    }
                };
                format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |",
                    r.name,
                    tick(Capabilities::SIMPLIFY),
                    tick(Capabilities::DIFF_FORWARD),
                    tick(Capabilities::DIFF_REVERSE),
                    tick(Capabilities::NUMERIC_F64),
                    tick(Capabilities::NUMERIC_BALL),
                    tick(Capabilities::LOWER_LLVM),
                    tick(Capabilities::LEAN_THEOREM),
                )
            })
            .collect();
        format!("{}\n{}", header, rows.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Registry entry (stores the primitive + probed capabilities)
// ---------------------------------------------------------------------------

struct Entry {
    primitive: Box<dyn Primitive>,
    caps: Capabilities,
}

// ---------------------------------------------------------------------------
// PrimitiveRegistry
// ---------------------------------------------------------------------------

/// Central registry mapping function names to their [`Primitive`]
/// implementations.
///
/// Use [`default_registry`](PrimitiveRegistry::default_registry) to get a
/// registry pre-populated with Alkahest's built-in functions.
pub struct PrimitiveRegistry {
    map: HashMap<&'static str, Entry>,
}

impl PrimitiveRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        PrimitiveRegistry {
            map: HashMap::new(),
        }
    }

    /// Register a primitive.  Probes capabilities by calling each optional
    /// method with a canonical zero-element input and recording which ones
    /// return `Some`.
    pub fn register(&mut self, p: Box<dyn Primitive>) {
        let caps = probe_caps(&*p);
        let name = p.name();
        self.map.insert(name, Entry { primitive: p, caps });
    }

    /// Look up a primitive by name.
    pub fn get(&self, name: &str) -> Option<&dyn Primitive> {
        self.map.get(name).map(|e| &*e.primitive)
    }

    /// Return the [`Capabilities`] bitfield for a named primitive.
    /// Returns `Capabilities::empty()` if the primitive is not registered.
    pub fn capabilities(&self, name: &str) -> Capabilities {
        self.map
            .get(name)
            .map(|e| e.caps)
            .unwrap_or(Capabilities::empty())
    }

    /// Generate a coverage table for all registered primitives, sorted by
    /// name.
    pub fn coverage_report(&self) -> CoverageReport {
        let mut rows: Vec<CoverageRow> = self
            .map
            .iter()
            .map(|(name, e)| CoverageRow {
                name: name.to_string(),
                caps: e.caps,
            })
            .collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name));
        CoverageReport { rows }
    }

    /// Call `diff_forward` on a registered primitive.
    /// Returns `None` if the primitive is unknown or lacks `DIFF_FORWARD`.
    pub fn diff_forward(
        &self,
        name: &str,
        args: &[ExprId],
        wrt: ExprId,
        pool: &ExprPool,
    ) -> Option<ExprId> {
        let entry = self.map.get(name)?;
        entry.primitive.diff_forward(args, wrt, pool)
    }

    /// Call `diff_reverse` on a registered primitive.
    pub fn diff_reverse(
        &self,
        name: &str,
        args: &[ExprId],
        cotan: ExprId,
        pool: &ExprPool,
    ) -> Option<Vec<ExprId>> {
        let entry = self.map.get(name)?;
        entry.primitive.diff_reverse(args, cotan, pool)
    }

    /// Call `numeric_f64` on a registered primitive.
    pub fn numeric_f64(&self, name: &str, args: &[f64]) -> Option<f64> {
        let entry = self.map.get(name)?;
        entry.primitive.numeric_f64(args)
    }

    /// Call `numeric_ball` on a registered primitive.
    pub fn numeric_ball(&self, name: &str, args: &[ArbBall]) -> Option<ArbBall> {
        let entry = self.map.get(name)?;
        entry.primitive.numeric_ball(args)
    }

    /// Return a registry pre-populated with Alkahest's built-in primitives.
    pub fn default_registry() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(builtins::SinPrimitive));
        reg.register(Box::new(builtins::CosPrimitive));
        reg.register(Box::new(builtins::ExpPrimitive));
        reg.register(Box::new(builtins::LogPrimitive));
        reg.register(Box::new(builtins::SqrtPrimitive));
        // V1-12: expanded registry
        reg.register(Box::new(builtins::TanPrimitive));
        reg.register(Box::new(builtins::SinhPrimitive));
        reg.register(Box::new(builtins::CoshPrimitive));
        reg.register(Box::new(builtins::TanhPrimitive));
        reg.register(Box::new(builtins::AsinPrimitive));
        reg.register(Box::new(builtins::AcosPrimitive));
        reg.register(Box::new(builtins::AtanPrimitive));
        reg.register(Box::new(builtins::ErfPrimitive));
        reg.register(Box::new(builtins::ErfcPrimitive));
        reg.register(Box::new(builtins::AbsPrimitive));
        reg.register(Box::new(builtins::SignPrimitive));
        reg.register(Box::new(builtins::FloorPrimitive));
        reg.register(Box::new(builtins::CeilPrimitive));
        reg.register(Box::new(builtins::RoundPrimitive));
        reg.register(Box::new(builtins::Atan2Primitive));
        reg.register(Box::new(builtins::GammaPrimitive));
        reg.register(Box::new(builtins::MinPrimitive));
        reg.register(Box::new(builtins::MaxPrimitive));
        reg
    }

    /// Returns true if a primitive with this name is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Iterate over all registered (name, capabilities) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Capabilities)> {
        self.map.iter().map(|(k, e)| (*k, e.caps))
    }
}

impl Default for PrimitiveRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

// ---------------------------------------------------------------------------
// Capability probing
// ---------------------------------------------------------------------------

/// Probe which optional methods a primitive has implemented by calling them
/// with a single-element `f64 = 1.0` / `ExprId` argument.  We use an
/// independent `ExprPool` so that probing is side-effect free.
fn probe_caps(p: &dyn Primitive) -> Capabilities {
    let mut caps = Capabilities::empty();

    // Probe with both unary and binary argument lists so n-ary primitives
    // (e.g. atan2, min, max) register their capabilities.
    let probe_f64_sets: [&[f64]; 2] = [&[1.0], &[1.0, 2.0]];
    for args in probe_f64_sets {
        if p.numeric_f64(args).is_some() {
            caps |= Capabilities::NUMERIC_F64;
            break;
        }
    }

    let ball1 = [ArbBall::from_f64(1.0, 128)];
    let ball2 = [ArbBall::from_f64(1.0, 128), ArbBall::from_f64(2.0, 128)];
    if p.numeric_ball(&ball1).is_some() || p.numeric_ball(&ball2).is_some() {
        caps |= Capabilities::NUMERIC_BALL;
    }

    // diff_forward / diff_reverse / simplify: probe with a fresh pool
    let pool = ExprPool::new();
    let x = pool.symbol("_probe", crate::kernel::Domain::Real);
    let y = pool.symbol("_probe_y", crate::kernel::Domain::Real);
    let probe_id_sets: [Vec<ExprId>; 2] = [vec![x], vec![x, y]];

    for args in &probe_id_sets {
        if p.diff_forward(args, x, &pool).is_some() {
            caps |= Capabilities::DIFF_FORWARD;
            break;
        }
    }
    for args in &probe_id_sets {
        if p.diff_reverse(args, x, &pool).is_some() {
            caps |= Capabilities::DIFF_REVERSE;
            break;
        }
    }
    for args in &probe_id_sets {
        if p.simplify(args, &pool).is_some() {
            caps |= Capabilities::SIMPLIFY;
            break;
        }
    }
    if p.lean_theorem().is_some() {
        caps |= Capabilities::LEAN_THEOREM;
    }
    caps
}

// ---------------------------------------------------------------------------
// Built-in primitives
// ---------------------------------------------------------------------------

pub mod builtins {
    use super::Primitive;
    use crate::ball::ArbBall;
    use crate::kernel::{ExprId, ExprPool};

    // ── sin ──────────────────────────────────────────────────────────────────

    pub struct SinPrimitive;

    impl Primitive for SinPrimitive {
        fn name(&self) -> &'static str {
            "sin"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("sin({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let cos_x = pool.func("cos", vec![x]);
            Some(pool.mul(vec![cos_x, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let cos_x = pool.func("cos", vec![x]);
            Some(vec![pool.mul(vec![cotan, cos_x])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].sin())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].sin())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.sin_deriv")
        }
    }

    // ── cos ──────────────────────────────────────────────────────────────────

    pub struct CosPrimitive;

    impl Primitive for CosPrimitive {
        fn name(&self) -> &'static str {
            "cos"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("cos({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let neg_one = pool.integer(-1_i32);
            let sin_x = pool.func("sin", vec![x]);
            Some(pool.mul(vec![neg_one, sin_x, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let neg_one = pool.integer(-1_i32);
            let sin_x = pool.func("sin", vec![x]);
            Some(vec![pool.mul(vec![cotan, neg_one, sin_x])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].cos())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].cos())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.cos_deriv")
        }
    }

    // ── exp ──────────────────────────────────────────────────────────────────

    pub struct ExpPrimitive;

    impl Primitive for ExpPrimitive {
        fn name(&self) -> &'static str {
            "exp"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("exp({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let exp_x = pool.func("exp", vec![x]);
            Some(pool.mul(vec![exp_x, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let exp_x = pool.func("exp", vec![x]);
            Some(vec![pool.mul(vec![cotan, exp_x])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].exp())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].exp())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.exp_deriv")
        }
    }

    // ── log ──────────────────────────────────────────────────────────────────

    pub struct LogPrimitive;

    impl Primitive for LogPrimitive {
        fn name(&self) -> &'static str {
            "log"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("log({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            // d/dwrt log(x) = dx / x
            let x_inv = pool.pow(x, pool.integer(-1_i32));
            Some(pool.mul(vec![x_inv, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x_inv = pool.pow(x, pool.integer(-1_i32));
            Some(vec![pool.mul(vec![cotan, x_inv])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].ln())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].log()
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.log_deriv")
        }
    }

    // ── sqrt ─────────────────────────────────────────────────────────────────

    pub struct SqrtPrimitive;

    impl Primitive for SqrtPrimitive {
        fn name(&self) -> &'static str {
            "sqrt"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("sqrt({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            // d/dwrt sqrt(x) = dx / (2 * sqrt(x))
            let sqrt_x = pool.func("sqrt", vec![x]);
            let two = pool.integer(2_i32);
            let denom = pool.mul(vec![two, sqrt_x]);
            let denom_inv = pool.pow(denom, pool.integer(-1_i32));
            Some(pool.mul(vec![dx, denom_inv]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let sqrt_x = pool.func("sqrt", vec![x]);
            let two = pool.integer(2_i32);
            let denom = pool.mul(vec![two, sqrt_x]);
            let denom_inv = pool.pow(denom, pool.integer(-1_i32));
            Some(vec![pool.mul(vec![cotan, denom_inv])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].sqrt())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].sqrt()
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.sqrt_deriv")
        }
    }

    // ── tan ──────────────────────────────────────────────────────────────────

    pub struct TanPrimitive;

    impl Primitive for TanPrimitive {
        fn name(&self) -> &'static str {
            "tan"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("tan({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx tan(x) = dx / cos²(x) = dx * (1 + tan²(x))
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let tan_x = pool.func("tan", vec![x]);
            let tan2 = pool.pow(tan_x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let sec2 = pool.add(vec![one, tan2]);
            Some(pool.mul(vec![sec2, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let tan_x = pool.func("tan", vec![x]);
            let tan2 = pool.pow(tan_x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let sec2 = pool.add(vec![one, tan2]);
            Some(vec![pool.mul(vec![cotan, sec2])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].tan())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].tan()
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.tan_deriv")
        }
    }

    // ── sinh ─────────────────────────────────────────────────────────────────

    pub struct SinhPrimitive;

    impl Primitive for SinhPrimitive {
        fn name(&self) -> &'static str {
            "sinh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("sinh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let cosh_x = pool.func("cosh", vec![x]);
            Some(pool.mul(vec![cosh_x, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let cosh_x = pool.func("cosh", vec![x]);
            Some(vec![pool.mul(vec![cotan, cosh_x])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].sinh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].sinh())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.sinh_deriv")
        }
    }

    // ── cosh ─────────────────────────────────────────────────────────────────

    pub struct CoshPrimitive;

    impl Primitive for CoshPrimitive {
        fn name(&self) -> &'static str {
            "cosh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("cosh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let sinh_x = pool.func("sinh", vec![x]);
            Some(pool.mul(vec![sinh_x, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let sinh_x = pool.func("sinh", vec![x]);
            Some(vec![pool.mul(vec![cotan, sinh_x])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].cosh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].cosh())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.cosh_deriv")
        }
    }

    // ── tanh ─────────────────────────────────────────────────────────────────

    pub struct TanhPrimitive;

    impl Primitive for TanhPrimitive {
        fn name(&self) -> &'static str {
            "tanh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("tanh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx tanh(x) = dx * (1 - tanh²(x))
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let tanh_x = pool.func("tanh", vec![x]);
            let tanh2 = pool.pow(tanh_x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let sech2 = pool.add(vec![one, pool.mul(vec![neg_one, tanh2])]);
            Some(pool.mul(vec![sech2, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let tanh_x = pool.func("tanh", vec![x]);
            let tanh2 = pool.pow(tanh_x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let sech2 = pool.add(vec![one, pool.mul(vec![neg_one, tanh2])]);
            Some(vec![pool.mul(vec![cotan, sech2])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].tanh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].tanh())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.tanh_deriv")
        }
    }

    // ── asin ─────────────────────────────────────────────────────────────────

    pub struct AsinPrimitive;

    impl Primitive for AsinPrimitive {
        fn name(&self) -> &'static str {
            "asin"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("asin({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx asin(x) = dx / sqrt(1 - x²)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let one_minus_x2 = pool.add(vec![one, pool.mul(vec![neg_one, x2])]);
            let denom = pool.func("sqrt", vec![one_minus_x2]);
            Some(pool.mul(vec![dx, pool.pow(denom, pool.integer(-1_i32))]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let one_minus_x2 = pool.add(vec![one, pool.mul(vec![neg_one, x2])]);
            let denom = pool.func("sqrt", vec![one_minus_x2]);
            Some(vec![
                pool.mul(vec![cotan, pool.pow(denom, pool.integer(-1_i32))])
            ])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].asin())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].asin()
        }
    }

    // ── acos ─────────────────────────────────────────────────────────────────

    pub struct AcosPrimitive;

    impl Primitive for AcosPrimitive {
        fn name(&self) -> &'static str {
            "acos"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("acos({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx acos(x) = -dx / sqrt(1 - x²)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let one_minus_x2 = pool.add(vec![one, pool.mul(vec![neg_one, x2])]);
            let denom = pool.func("sqrt", vec![one_minus_x2]);
            Some(pool.mul(vec![neg_one, dx, pool.pow(denom, pool.integer(-1_i32))]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let neg_one = pool.integer(-1_i32);
            let one_minus_x2 = pool.add(vec![one, pool.mul(vec![neg_one, x2])]);
            let denom = pool.func("sqrt", vec![one_minus_x2]);
            Some(vec![pool.mul(vec![
                cotan,
                neg_one,
                pool.pow(denom, pool.integer(-1_i32)),
            ])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].acos())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].acos()
        }
    }

    // ── atan ─────────────────────────────────────────────────────────────────

    pub struct AtanPrimitive;

    impl Primitive for AtanPrimitive {
        fn name(&self) -> &'static str {
            "atan"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("atan({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx atan(x) = dx / (1 + x²)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let denom = pool.add(vec![one, x2]);
            Some(pool.mul(vec![dx, pool.pow(denom, pool.integer(-1_i32))]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let denom = pool.add(vec![one, x2]);
            Some(vec![
                pool.mul(vec![cotan, pool.pow(denom, pool.integer(-1_i32))])
            ])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].atan())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].atan())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.arctan_deriv")
        }
    }

    // ── erf ──────────────────────────────────────────────────────────────────

    pub struct ErfPrimitive;

    impl Primitive for ErfPrimitive {
        fn name(&self) -> &'static str {
            "erf"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("erf({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx erf(x) = (2/sqrt(π)) * exp(-x²) * dx
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let exp_neg_x2 = pool.func("exp", vec![neg_x2]);
            let coeff = pool.float(2.0 / std::f64::consts::PI.sqrt(), 53);
            Some(pool.mul(vec![coeff, exp_neg_x2, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let exp_neg_x2 = pool.func("exp", vec![neg_x2]);
            let coeff = pool.float(2.0 / std::f64::consts::PI.sqrt(), 53);
            Some(vec![pool.mul(vec![cotan, coeff, exp_neg_x2])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(libm_erf(args[0]))
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].erf())
        }
    }

    // ── erfc ─────────────────────────────────────────────────────────────────

    pub struct ErfcPrimitive;

    impl Primitive for ErfcPrimitive {
        fn name(&self) -> &'static str {
            "erfc"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("erfc({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let exp_neg_x2 = pool.func("exp", vec![neg_x2]);
            let coeff = pool.float(-2.0 / std::f64::consts::PI.sqrt(), 53);
            Some(pool.mul(vec![coeff, exp_neg_x2, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let exp_neg_x2 = pool.func("exp", vec![neg_x2]);
            let coeff = pool.float(-2.0 / std::f64::consts::PI.sqrt(), 53);
            Some(vec![pool.mul(vec![cotan, coeff, exp_neg_x2])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(1.0 - libm_erf(args[0]))
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].erfc())
        }
    }

    // ── abs ──────────────────────────────────────────────────────────────────

    pub struct AbsPrimitive;

    impl Primitive for AbsPrimitive {
        fn name(&self) -> &'static str {
            "abs"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("|{}|", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].abs())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].abs_ball())
        }
    }

    // ── sign ─────────────────────────────────────────────────────────────────

    pub struct SignPrimitive;

    impl Primitive for SignPrimitive {
        fn name(&self) -> &'static str {
            "sign"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("sign({})", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(if args[0] > 0.0 {
                1.0
            } else if args[0] < 0.0 {
                -1.0
            } else {
                0.0
            })
        }
    }

    // ── floor ────────────────────────────────────────────────────────────────

    pub struct FloorPrimitive;

    impl Primitive for FloorPrimitive {
        fn name(&self) -> &'static str {
            "floor"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("floor({})", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].floor())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].floor_ball())
        }
    }

    // ── ceil ─────────────────────────────────────────────────────────────────

    pub struct CeilPrimitive;

    impl Primitive for CeilPrimitive {
        fn name(&self) -> &'static str {
            "ceil"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("ceil({})", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].ceil())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].ceil_ball())
        }
    }

    // ── round ────────────────────────────────────────────────────────────────

    pub struct RoundPrimitive;

    impl Primitive for RoundPrimitive {
        fn name(&self) -> &'static str {
            "round"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("round({})", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].round())
        }
    }

    // ── atan2 ────────────────────────────────────────────────────────────────

    pub struct Atan2Primitive;

    impl Primitive for Atan2Primitive {
        fn name(&self) -> &'static str {
            "atan2"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!(
                "atan2({}, {})",
                pool.display(args[0]),
                pool.display(args[1])
            )
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 2 {
                Some(args[0].atan2(args[1]))
            } else {
                None
            }
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.arctan2")
        }
    }

    // ── gamma ────────────────────────────────────────────────────────────────

    pub struct GammaPrimitive;

    impl Primitive for GammaPrimitive {
        fn name(&self) -> &'static str {
            "gamma"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("Γ({})", pool.display(args[0]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(libm_gamma(args[0]))
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.Gamma")
        }
    }

    // ── min ──────────────────────────────────────────────────────────────────

    pub struct MinPrimitive;

    impl Primitive for MinPrimitive {
        fn name(&self) -> &'static str {
            "min"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("min({}, {})", pool.display(args[0]), pool.display(args[1]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 2 {
                Some(args[0].min(args[1]))
            } else {
                None
            }
        }
    }

    // ── max ──────────────────────────────────────────────────────────────────

    pub struct MaxPrimitive;

    impl Primitive for MaxPrimitive {
        fn name(&self) -> &'static str {
            "max"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("max({}, {})", pool.display(args[0]), pool.display(args[1]))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 2 {
                Some(args[0].max(args[1]))
            } else {
                None
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Helper: Lanczos approximation for Γ(x), x ∈ ℝ.  Accurate to ~15 digits.
    // Coefficients from Cephes (g = 7, n = 9).
    // ---------------------------------------------------------------------------

    fn libm_gamma(x: f64) -> f64 {
        const G: f64 = 7.0;
        const P: [f64; 9] = [
            0.999_999_999_999_809_9,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        if x < 0.5 {
            // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
            std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * libm_gamma(1.0 - x))
        } else {
            let xm = x - 1.0;
            let mut a = P[0];
            for (i, p) in P.iter().enumerate().skip(1) {
                a += p / (xm + i as f64);
            }
            let t = xm + G + 0.5;
            (2.0 * std::f64::consts::PI).sqrt() * t.powf(xm + 0.5) * (-t).exp() * a
        }
    }

    // ---------------------------------------------------------------------------
    // Helper: erf via polynomial approximation (Horner-form, max error ≤ 1.5e-7)
    // Abramowitz & Stegun 7.1.26
    // ---------------------------------------------------------------------------

    fn libm_erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (1.0 - poly * (-x * x).exp())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn default_registry_has_builtins() {
        let reg = PrimitiveRegistry::default_registry();
        for name in &["sin", "cos", "exp", "log", "sqrt"] {
            assert!(reg.is_registered(name), "{name} not registered");
            let caps = reg.capabilities(name);
            assert!(
                caps.contains(Capabilities::NUMERIC_F64),
                "{name} missing NUMERIC_F64"
            );
            assert!(
                caps.contains(Capabilities::DIFF_FORWARD),
                "{name} missing DIFF_FORWARD"
            );
            assert!(
                caps.contains(Capabilities::DIFF_REVERSE),
                "{name} missing DIFF_REVERSE"
            );
            assert!(
                caps.contains(Capabilities::NUMERIC_BALL),
                "{name} missing NUMERIC_BALL"
            );
        }
    }

    #[test]
    fn numeric_f64_correct() {
        let reg = PrimitiveRegistry::default_registry();
        let cases: &[(&str, f64, f64)] = &[
            ("sin", 0.0, 0.0),
            ("cos", 0.0, 1.0),
            ("exp", 0.0, 1.0),
            ("log", 1.0, 0.0),
            ("sqrt", 4.0, 2.0),
        ];
        for (name, input, expected) in cases {
            let got = reg.numeric_f64(name, &[*input]).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "{name}({input}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn diff_forward_sin() {
        let reg = PrimitiveRegistry::default_registry();
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let result = reg.diff_forward("sin", &[x], x, &pool);
        assert!(result.is_some(), "sin diff_forward returned None");
    }

    #[test]
    fn coverage_report_markdown() {
        let reg = PrimitiveRegistry::default_registry();
        let report = reg.coverage_report();
        let md = report.to_markdown();
        assert!(md.contains("sin"), "coverage report missing sin");
        assert!(md.contains("✓"), "coverage report has no ticks");
    }

    #[test]
    fn custom_primitive_registration() {
        struct TanhPrimitive;
        impl Primitive for TanhPrimitive {
            fn name(&self) -> &'static str {
                "tanh"
            }
            fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
                format!("tanh({})", pool.display(args[0]))
            }
            fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
                Some(args[0].tanh())
            }
        }

        let mut reg = PrimitiveRegistry::new();
        reg.register(Box::new(TanhPrimitive));
        assert!(reg.is_registered("tanh"));
        let caps = reg.capabilities("tanh");
        assert!(caps.contains(Capabilities::NUMERIC_F64));
        assert!(!caps.contains(Capabilities::DIFF_FORWARD));

        let got = reg.numeric_f64("tanh", &[0.0]).unwrap();
        assert!((got - 0.0).abs() < 1e-12);
    }
}
