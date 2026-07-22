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
//! // Every built-in *function* has at least NUMERIC_F64 coverage; the only
//! // exceptions are the Dirac delta `δ` and symbolic complex constructors,
//! // which deliberately have no pointwise real `f64` value.
//! for row in &report.rows {
//!     if matches!(
//!         row.name.as_str(),
//!         "diracdelta" | "conjugate" | "re" | "im" | "arg"
//!     ) {
//!         continue;
//!     }
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
        reg.register(Box::new(builtins::AsinhPrimitive));
        reg.register(Box::new(builtins::AcoshPrimitive));
        reg.register(Box::new(builtins::AtanhPrimitive));
        reg.register(Box::new(builtins::ErfPrimitive));
        reg.register(Box::new(builtins::ErfcPrimitive));
        // Elliptic special functions (parameter convention m = k²).
        reg.register(Box::new(builtins::EllipticKPrimitive));
        reg.register(Box::new(builtins::EllipticEPrimitive));
        reg.register(Box::new(builtins::EllipticFPrimitive));
        reg.register(Box::new(builtins::EllipticPiPrimitive));
        reg.register(Box::new(builtins::AbsPrimitive));
        reg.register(Box::new(builtins::SignPrimitive));
        reg.register(Box::new(builtins::HeavisidePrimitive));
        reg.register(Box::new(builtins::DiracDeltaPrimitive));
        reg.register(Box::new(builtins::FloorPrimitive));
        reg.register(Box::new(builtins::CeilPrimitive));
        reg.register(Box::new(builtins::RoundPrimitive));
        reg.register(Box::new(builtins::Atan2Primitive));
        reg.register(Box::new(builtins::GammaPrimitive));
        reg.register(Box::new(builtins::LambertWPrimitive));
        reg.register(Box::new(builtins::DigammaPrimitive));
        reg.register(Box::new(builtins::BesselJ0Primitive));
        reg.register(Box::new(builtins::BesselJ1Primitive));
        reg.register(Box::new(builtins::MinPrimitive));
        reg.register(Box::new(builtins::MaxPrimitive));
        reg.register(Box::new(builtins::ConjugatePrimitive));
        reg.register(Box::new(builtins::RePrimitive));
        reg.register(Box::new(builtins::ImPrimitive));
        reg.register(Box::new(builtins::ArgPrimitive));
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

    // Probe with unary, binary, and ternary argument lists so n-ary
    // primitives (e.g. atan2, min, max, EllipticPi) register their
    // capabilities.  We also try a set of small fractional values that lie
    // inside the convergence domain of the elliptic integrals (where larger
    // probe inputs would be rejected as out-of-domain).
    let probe_f64_sets: [&[f64]; 6] = [
        &[1.0],
        &[1.0, 2.0],
        &[1.0, 2.0, 3.0],
        &[0.5],
        &[0.5, 0.3],
        &[0.2, 0.3, 0.4],
    ];
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
    let z = pool.symbol("_probe_z", crate::kernel::Domain::Real);
    let probe_id_sets: [Vec<ExprId>; 3] = [vec![x], vec![x, y], vec![x, y, z]];

    // A primitive may only differentiate w.r.t. *some* of its arguments
    // (e.g. EllipticPi only has a clean ∂/∂φ), so probe differentiating w.r.t.
    // each argument position, not just the first.
    'fwd: for args in &probe_id_sets {
        for &wrt in args {
            if p.diff_forward(args, wrt, &pool).is_some() {
                caps |= Capabilities::DIFF_FORWARD;
                break 'fwd;
            }
        }
    }
    'rev: for args in &probe_id_sets {
        for &wrt in args {
            if p.diff_reverse(args, wrt, &pool).is_some() {
                caps |= Capabilities::DIFF_REVERSE;
                break 'rev;
            }
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
    use crate::kernel::expr::ExprData;
    use crate::kernel::{ExprId, ExprPool};

    macro_rules! symbolic_complex_primitive {
        ($type:ident, $name:literal) => {
            pub struct $type;
            impl Primitive for $type {
                fn name(&self) -> &'static str {
                    $name
                }
                fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
                    format!("{}({})", $name, pool.display(args[0]))
                }
            }
        };
    }
    symbolic_complex_primitive!(ConjugatePrimitive, "conjugate");
    symbolic_complex_primitive!(RePrimitive, "re");
    symbolic_complex_primitive!(ImPrimitive, "im");
    symbolic_complex_primitive!(ArgPrimitive, "arg");

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

    // ── asinh ────────────────────────────────────────────────────────────────

    pub struct AsinhPrimitive;

    impl Primitive for AsinhPrimitive {
        fn name(&self) -> &'static str {
            "asinh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("asinh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx asinh(x) = dx / sqrt(x² + 1)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let one = pool.integer(1_i32);
            let x2_plus_one = pool.add(vec![x2, one]);
            let denom = pool.func("sqrt", vec![x2_plus_one]);
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
            let x2_plus_one = pool.add(vec![x2, one]);
            let denom = pool.func("sqrt", vec![x2_plus_one]);
            Some(vec![
                pool.mul(vec![cotan, pool.pow(denom, pool.integer(-1_i32))])
            ])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].asinh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            Some(args[0].asinh())
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.arsinh")
        }
    }

    // ── acosh ────────────────────────────────────────────────────────────────

    pub struct AcoshPrimitive;

    impl Primitive for AcoshPrimitive {
        fn name(&self) -> &'static str {
            "acosh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("acosh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx acosh(x) = dx / sqrt(x² − 1)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_one = pool.integer(-1_i32);
            let x2_minus_one = pool.add(vec![x2, neg_one]);
            let denom = pool.func("sqrt", vec![x2_minus_one]);
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
            let neg_one = pool.integer(-1_i32);
            let x2_minus_one = pool.add(vec![x2, neg_one]);
            let denom = pool.func("sqrt", vec![x2_minus_one]);
            Some(vec![
                pool.mul(vec![cotan, pool.pow(denom, pool.integer(-1_i32))])
            ])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].acosh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].acosh()
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.arcosh")
        }
    }

    // ── atanh ────────────────────────────────────────────────────────────────

    pub struct AtanhPrimitive;

    impl Primitive for AtanhPrimitive {
        fn name(&self) -> &'static str {
            "atanh"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("atanh({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx atanh(x) = dx / (1 − x²)
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let x2 = pool.pow(x, pool.integer(2_i32));
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let denom = pool.add(vec![pool.integer(1_i32), neg_x2]);
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
            let neg_x2 = pool.mul(vec![pool.integer(-1_i32), x2]);
            let denom = pool.add(vec![pool.integer(1_i32), neg_x2]);
            Some(vec![
                pool.mul(vec![cotan, pool.pow(denom, pool.integer(-1_i32))])
            ])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(args[0].atanh())
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            args[0].atanh()
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.artanh")
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

    // ── Elliptic integrals ────────────────────────────────────────────────────
    //
    // All five elliptic special functions use the *parameter* convention
    // `m = k²` (matching Mathematica's `EllipticK[m]`, `EllipticF[phi, m]`,
    // …), NOT the *modulus* convention `k`.  So, e.g., the complete integral
    // of the first kind is
    //
    //     K(m) = ∫₀^{π/2} dθ / √(1 − m·sin²θ).
    //
    // Numeric evaluation uses the arithmetic–geometric mean (AGM) for the
    // complete integrals K(m), E(m) and adaptive Simpson quadrature of the
    // defining integrand over [0, φ] for the incomplete integrals
    // F(φ,m), E(φ,m), Π(n,φ,m).  Out-of-domain inputs (where the integrand
    // is singular / complex, i.e. `m·sin²θ ≥ 1`) return `None`.
    //
    // NOTE on AD coverage: the symbolic differentiator `crate::diff::diff`
    // (see `diff/diff_impl.rs`) routes multi-arg `Func` nodes through these
    // primitives' `diff_forward`, so the symbolic path is fully wired.  The
    // dual-number forward-AD (`diff/forward.rs`) and reverse-AD
    // (`diff/reverse.rs`) modes only special-case *single*-argument
    // functions; for multi-arg elliptic functions they fall back to their
    // existing safe behaviour (returning a `ForwardUnknownFunction` error /
    // treating the node as opaque), which is acceptable for this PR.

    /// Build the elliptic "Δ-factor" `√(1 − m·sin²φ)` as an expression.
    fn elliptic_delta(phi: ExprId, m: ExprId, pool: &ExprPool) -> ExprId {
        let sin_phi = pool.func("sin", vec![phi]);
        let sin2 = pool.pow(sin_phi, pool.integer(2_i32));
        let m_sin2 = pool.mul(vec![m, sin2]);
        let neg = pool.mul(vec![pool.integer(-1_i32), m_sin2]);
        let one = pool.integer(1_i32);
        let inside = pool.add(vec![one, neg]);
        pool.func("sqrt", vec![inside])
    }

    /// Numeric AGM-based complete elliptic integral of the first kind K(m).
    /// Returns `None` if `m ≥ 1` (out of the real convergent domain).
    fn agm_k(m: f64) -> Option<f64> {
        if m >= 1.0 {
            return None;
        }
        let mut a = 1.0_f64;
        let mut b = (1.0 - m).sqrt();
        for _ in 0..100 {
            if (a - b).abs() <= 1e-16 * a.abs() {
                break;
            }
            let an = 0.5 * (a + b);
            let bn = (a * b).sqrt();
            a = an;
            b = bn;
        }
        Some(std::f64::consts::PI / (2.0 * a))
    }

    /// Numeric AGM-based complete elliptic integral of the second kind E(m).
    fn agm_e(m: f64) -> Option<f64> {
        if m >= 1.0 {
            // E(1) = 1 exactly; reject anything beyond.
            if (m - 1.0).abs() < 1e-15 {
                return Some(1.0);
            }
            return None;
        }
        let k = agm_k(m)?;
        let mut a = 1.0_f64;
        let mut b = (1.0 - m).sqrt();
        // E/K = 1 − Σ_{n=0}^∞ 2^(n-1) c_n²  with c_0² = m (Abramowitz & Stegun
        // 17.6).  The n = 0 term contributes 2^(-1)·m = m/2; subsequent terms
        // use c_n = (a_{n-1} − b_{n-1})/2 with weight 2^(n-1) (1, 2, 4, …).
        let mut sum = 0.5 * m;
        let mut weight = 1.0_f64; // 2^(n-1) for n = 1, 2, 3, …
                                  // Iterate until the AGM converges.  We must stop once `c_n` reaches the
                                  // f64 noise floor: otherwise the geometrically growing weight `2^(n-1)`
                                  // amplifies that rounding residue and corrupts the sum.
        for _ in 0..40 {
            let cn = 0.5 * (a - b);
            if cn.abs() <= 1e-15 * a.abs() {
                break;
            }
            let an = 0.5 * (a + b);
            let bn = (a * b).sqrt();
            sum += weight * cn * cn;
            weight *= 2.0;
            a = an;
            b = bn;
        }
        Some(k * (1.0 - sum))
    }

    /// Adaptive Simpson integration of `f` over `[a, b]` to absolute
    /// tolerance `tol`.  Returns `None` if the integrand is non-finite.
    fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> Option<f64> {
        fn simpson<F: Fn(f64) -> f64>(_f: &F, a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
            (b - a) / 6.0 * (fa + 4.0 * fm + fb)
        }
        #[allow(clippy::too_many_arguments)]
        fn recur<F: Fn(f64) -> f64>(
            f: &F,
            a: f64,
            b: f64,
            fa: f64,
            fb: f64,
            fm: f64,
            whole: f64,
            tol: f64,
            depth: u32,
        ) -> Option<f64> {
            let m = 0.5 * (a + b);
            let lm = 0.5 * (a + m);
            let rm = 0.5 * (m + b);
            let flm = f(lm);
            let frm = f(rm);
            if !flm.is_finite() || !frm.is_finite() {
                return None;
            }
            let left = simpson(f, a, m, fa, fm, flm);
            let right = simpson(f, m, b, fm, fb, frm);
            if depth == 0 || (left + right - whole).abs() <= 15.0 * tol {
                return Some(left + right + (left + right - whole) / 15.0);
            }
            let l = recur(f, a, m, fa, fm, flm, left, tol * 0.5, depth - 1)?;
            let r = recur(f, m, b, fm, fb, frm, right, tol * 0.5, depth - 1)?;
            Some(l + r)
        }
        if a == b {
            return Some(0.0);
        }
        let fa = f(a);
        let fb = f(b);
        let m = 0.5 * (a + b);
        let fm = f(m);
        if !fa.is_finite() || !fb.is_finite() || !fm.is_finite() {
            return None;
        }
        let whole = simpson(f, a, b, fa, fb, fm);
        recur(f, a, b, fa, fb, fm, whole, tol, 50)
    }

    /// Check the incomplete-integral integrand is real over `[0, φ]`:
    /// requires `m·sin²θ < 1` for all θ in range.  Since `sin²θ ≤ 1`, the
    /// worst case is where `|sin|` is largest in `[0, φ]`.  We reject when
    /// `m·sin²θ_max ≥ 1`.
    fn incomplete_domain_ok(phi: f64, m: f64) -> bool {
        if m <= 0.0 {
            return true;
        }
        // Largest sin² over [0, φ] (φ may exceed π/2 in either direction).
        let lo = phi.min(0.0);
        let hi = phi.max(0.0);
        let mut max_sin2 = lo.sin().powi(2).max(hi.sin().powi(2));
        // π/2 + kπ inside the interval gives sin² = 1.
        let half_pi = std::f64::consts::FRAC_PI_2;
        let pi = std::f64::consts::PI;
        let kstart = ((lo - half_pi) / pi).ceil() as i64;
        for k in kstart..kstart + 4 {
            let x = half_pi + (k as f64) * pi;
            if x >= lo && x <= hi {
                max_sin2 = 1.0;
                break;
            }
        }
        m * max_sin2 < 1.0
    }

    // ── EllipticK (complete, first kind) ───────────────────────────────────────

    pub struct EllipticKPrimitive;

    impl Primitive for EllipticKPrimitive {
        fn name(&self) -> &'static str {
            "EllipticK"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("EllipticK({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dm K(m) = E(m)/(2m(1−m)) − K(m)/(2m)
            let m = args[0];
            let dm = crate::diff::diff(m, wrt, pool).ok()?.value;
            let e = pool.func("EllipticE", vec![m]);
            let k = pool.func("EllipticK", vec![m]);
            let one_minus_m = pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), m]),
            ]);
            let two_m = pool.mul(vec![pool.integer(2_i32), m]);
            let two_m_1mm = pool.mul(vec![pool.integer(2_i32), m, one_minus_m]);
            let term1 = pool.mul(vec![e, pool.pow(two_m_1mm, pool.integer(-1_i32))]);
            let term2 = pool.mul(vec![
                pool.integer(-1_i32),
                k,
                pool.pow(two_m, pool.integer(-1_i32)),
            ]);
            let dkdm = pool.add(vec![term1, term2]);
            Some(pool.mul(vec![dkdm, dm]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let m = args[0];
            let e = pool.func("EllipticE", vec![m]);
            let k = pool.func("EllipticK", vec![m]);
            let one_minus_m = pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), m]),
            ]);
            let two_m = pool.mul(vec![pool.integer(2_i32), m]);
            let two_m_1mm = pool.mul(vec![pool.integer(2_i32), m, one_minus_m]);
            let term1 = pool.mul(vec![e, pool.pow(two_m_1mm, pool.integer(-1_i32))]);
            let term2 = pool.mul(vec![
                pool.integer(-1_i32),
                k,
                pool.pow(two_m, pool.integer(-1_i32)),
            ]);
            let dkdm = pool.add(vec![term1, term2]);
            Some(vec![pool.mul(vec![cotan, dkdm])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() != 1 {
                return None;
            }
            agm_k(args[0])
        }
    }

    // ── EllipticE (complete & incomplete, second kind) ─────────────────────────

    pub struct EllipticEPrimitive;

    impl Primitive for EllipticEPrimitive {
        fn name(&self) -> &'static str {
            "EllipticE"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            if args.len() == 1 {
                format!("EllipticE({})", pool.display(args[0]))
            } else {
                format!(
                    "EllipticE({}, {})",
                    pool.display(args[0]),
                    pool.display(args[1])
                )
            }
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            if args.len() == 1 {
                // d/dm E(m) = (E(m) − K(m))/(2m)
                let m = args[0];
                let dm = crate::diff::diff(m, wrt, pool).ok()?.value;
                let e = pool.func("EllipticE", vec![m]);
                let k = pool.func("EllipticK", vec![m]);
                let two_m = pool.mul(vec![pool.integer(2_i32), m]);
                let num = pool.add(vec![e, pool.mul(vec![pool.integer(-1_i32), k])]);
                let dedm = pool.mul(vec![num, pool.pow(two_m, pool.integer(-1_i32))]);
                Some(pool.mul(vec![dedm, dm]))
            } else {
                // Incomplete E(φ, m).
                let phi = args[0];
                let m = args[1];
                let dphi = crate::diff::diff(phi, wrt, pool).ok()?.value;
                let dm = crate::diff::diff(m, wrt, pool).ok()?.value;
                // ∂/∂φ E(φ,m) = √(1 − m·sin²φ)
                let delta = elliptic_delta(phi, m, pool);
                let mut terms = vec![pool.mul(vec![delta, dphi])];
                // ∂/∂m E(φ,m) = (E(φ,m) − F(φ,m))/(2m)
                let e = pool.func("EllipticE", vec![phi, m]);
                let f = pool.func("EllipticF", vec![phi, m]);
                let two_m = pool.mul(vec![pool.integer(2_i32), m]);
                let num = pool.add(vec![e, pool.mul(vec![pool.integer(-1_i32), f])]);
                let dedm = pool.mul(vec![num, pool.pow(two_m, pool.integer(-1_i32))]);
                terms.push(pool.mul(vec![dedm, dm]));
                Some(pool.add(terms))
            }
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            if args.len() == 1 {
                let m = args[0];
                let e = pool.func("EllipticE", vec![m]);
                let k = pool.func("EllipticK", vec![m]);
                let two_m = pool.mul(vec![pool.integer(2_i32), m]);
                let num = pool.add(vec![e, pool.mul(vec![pool.integer(-1_i32), k])]);
                let dedm = pool.mul(vec![num, pool.pow(two_m, pool.integer(-1_i32))]);
                Some(vec![pool.mul(vec![cotan, dedm])])
            } else {
                let phi = args[0];
                let m = args[1];
                let delta = elliptic_delta(phi, m, pool);
                let dphi = pool.mul(vec![cotan, delta]);
                let e = pool.func("EllipticE", vec![phi, m]);
                let f = pool.func("EllipticF", vec![phi, m]);
                let two_m = pool.mul(vec![pool.integer(2_i32), m]);
                let num = pool.add(vec![e, pool.mul(vec![pool.integer(-1_i32), f])]);
                let dedm = pool.mul(vec![num, pool.pow(two_m, pool.integer(-1_i32))]);
                Some(vec![dphi, pool.mul(vec![cotan, dedm])])
            }
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            match args.len() {
                1 => agm_e(args[0]),
                2 => {
                    let (phi, m) = (args[0], args[1]);
                    if !incomplete_domain_ok(phi, m) {
                        return None;
                    }
                    adaptive_simpson(
                        &|t: f64| (1.0 - m * t.sin().powi(2)).sqrt(),
                        0.0,
                        phi,
                        1e-11,
                    )
                }
                _ => None,
            }
        }
    }

    // ── EllipticF (incomplete, first kind) ─────────────────────────────────────

    pub struct EllipticFPrimitive;

    impl Primitive for EllipticFPrimitive {
        fn name(&self) -> &'static str {
            "EllipticF"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!(
                "EllipticF({}, {})",
                pool.display(args[0]),
                pool.display(args[1])
            )
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            if args.len() != 2 {
                return None;
            }
            let phi = args[0];
            let m = args[1];
            let dphi = crate::diff::diff(phi, wrt, pool).ok()?.value;
            let dm = crate::diff::diff(m, wrt, pool).ok()?.value;
            // ∂/∂φ F(φ,m) = 1/√(1 − m·sin²φ)
            let delta = elliptic_delta(phi, m, pool);
            let inv_delta = pool.pow(delta, pool.integer(-1_i32));
            let mut terms = vec![pool.mul(vec![inv_delta, dphi])];
            // ∂/∂m F(φ,m) = E(φ,m)/(2m(1−m)) − F(φ,m)/(2m)
            //               − sin(2φ)/(4(1−m)·√(1−m·sin²φ))
            let e = pool.func("EllipticE", vec![phi, m]);
            let f = pool.func("EllipticF", vec![phi, m]);
            let one_minus_m = pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), m]),
            ]);
            let two_m = pool.mul(vec![pool.integer(2_i32), m]);
            let two_m_1mm = pool.mul(vec![pool.integer(2_i32), m, one_minus_m]);
            let t1 = pool.mul(vec![e, pool.pow(two_m_1mm, pool.integer(-1_i32))]);
            let t2 = pool.mul(vec![
                pool.integer(-1_i32),
                f,
                pool.pow(two_m, pool.integer(-1_i32)),
            ]);
            let two_phi = pool.mul(vec![pool.integer(2_i32), phi]);
            let sin_2phi = pool.func("sin", vec![two_phi]);
            let four_1mm = pool.mul(vec![pool.integer(4_i32), one_minus_m]);
            let denom = pool.mul(vec![four_1mm, delta]);
            let t3 = pool.mul(vec![
                pool.integer(-1_i32),
                sin_2phi,
                pool.pow(denom, pool.integer(-1_i32)),
            ]);
            let dfdm = pool.add(vec![t1, t2, t3]);
            terms.push(pool.mul(vec![dfdm, dm]));
            Some(pool.add(terms))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            if args.len() != 2 {
                return None;
            }
            let phi = args[0];
            let m = args[1];
            let delta = elliptic_delta(phi, m, pool);
            let inv_delta = pool.pow(delta, pool.integer(-1_i32));
            let dphi = pool.mul(vec![cotan, inv_delta]);
            let e = pool.func("EllipticE", vec![phi, m]);
            let f = pool.func("EllipticF", vec![phi, m]);
            let one_minus_m = pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), m]),
            ]);
            let two_m = pool.mul(vec![pool.integer(2_i32), m]);
            let two_m_1mm = pool.mul(vec![pool.integer(2_i32), m, one_minus_m]);
            let t1 = pool.mul(vec![e, pool.pow(two_m_1mm, pool.integer(-1_i32))]);
            let t2 = pool.mul(vec![
                pool.integer(-1_i32),
                f,
                pool.pow(two_m, pool.integer(-1_i32)),
            ]);
            let two_phi = pool.mul(vec![pool.integer(2_i32), phi]);
            let sin_2phi = pool.func("sin", vec![two_phi]);
            let four_1mm = pool.mul(vec![pool.integer(4_i32), one_minus_m]);
            let denom = pool.mul(vec![four_1mm, delta]);
            let t3 = pool.mul(vec![
                pool.integer(-1_i32),
                sin_2phi,
                pool.pow(denom, pool.integer(-1_i32)),
            ]);
            let dfdm = pool.add(vec![t1, t2, t3]);
            Some(vec![dphi, pool.mul(vec![cotan, dfdm])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() != 2 {
                return None;
            }
            let (phi, m) = (args[0], args[1]);
            if !incomplete_domain_ok(phi, m) {
                return None;
            }
            adaptive_simpson(
                &|t: f64| 1.0 / (1.0 - m * t.sin().powi(2)).sqrt(),
                0.0,
                phi,
                1e-11,
            )
        }
    }

    // ── EllipticPi (incomplete, third kind) ────────────────────────────────────

    pub struct EllipticPiPrimitive;

    impl Primitive for EllipticPiPrimitive {
        fn name(&self) -> &'static str {
            "EllipticPi"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!(
                "EllipticPi({}, {}, {})",
                pool.display(args[0]),
                pool.display(args[1]),
                pool.display(args[2])
            )
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            if args.len() != 3 {
                return None;
            }
            let n = args[0];
            let phi = args[1];
            let m = args[2];
            // Only the ∂/∂φ partial has a clean closed form; decline if n or m
            // depend on `wrt` (the ∂/∂n, ∂/∂m closed forms are messy — see
            // the module-level doc comment).
            let dn = crate::diff::diff(n, wrt, pool).ok()?.value;
            let dm = crate::diff::diff(m, wrt, pool).ok()?.value;
            let zero = pool.integer(0_i32);
            if dn != zero || dm != zero {
                return None;
            }
            let dphi = crate::diff::diff(phi, wrt, pool).ok()?.value;
            // ∂/∂φ Π(n,φ,m) = 1/((1 − n·sin²φ)·√(1 − m·sin²φ))
            let sin_phi = pool.func("sin", vec![phi]);
            let sin2 = pool.pow(sin_phi, pool.integer(2_i32));
            let n_sin2 = pool.mul(vec![n, sin2]);
            let one_minus_n_sin2 = pool.add(vec![
                pool.integer(1_i32),
                pool.mul(vec![pool.integer(-1_i32), n_sin2]),
            ]);
            let delta = elliptic_delta(phi, m, pool);
            let denom = pool.mul(vec![one_minus_n_sin2, delta]);
            let dpidphi = pool.pow(denom, pool.integer(-1_i32));
            Some(pool.mul(vec![dpidphi, dphi]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            // Only the φ-partial is implemented; ∂/∂n and ∂/∂m are declined.
            // Returning None keeps reverse-AD safe rather than reporting a
            // partial cotangent vector that omits arguments.
            let _ = (args, cotan, pool);
            None
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() != 3 {
                return None;
            }
            let (n, phi, m) = (args[0], args[1], args[2]);
            if !incomplete_domain_ok(phi, m) {
                return None;
            }
            adaptive_simpson(
                &|t: f64| {
                    let s2 = t.sin().powi(2);
                    let pole = 1.0 - n * s2;
                    if pole == 0.0 {
                        return f64::NAN;
                    }
                    1.0 / (pole * (1.0 - m * s2).sqrt())
                },
                0.0,
                phi,
                1e-11,
            )
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

    // ── Heaviside (unit step) ──────────────────────────────────────────────────

    /// The Heaviside unit-step function `θ(x)`: `0` for `x < 0`, `1` for `x > 0`.
    ///
    /// The value at `x = 0` is left unspecified (the half-maximum convention
    /// `θ(0) = 1/2` is used for numeric evaluation).  Its distributional
    /// derivative is the [`DiracDeltaPrimitive`].  Registered so the Laplace
    /// transform can recognise shifted steps `θ(t − a)`.
    pub struct HeavisidePrimitive;

    impl Primitive for HeavisidePrimitive {
        fn name(&self) -> &'static str {
            "heaviside"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("θ({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dx θ(g(x)) = δ(g(x))·g'(x)  (distributional derivative).
            let g = args[0];
            let dg = crate::diff::diff(g, wrt, pool).ok()?.value;
            let delta = pool.func("diracdelta", vec![g]);
            Some(pool.mul(vec![delta, dg]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let delta = pool.func("diracdelta", vec![args[0]]);
            Some(vec![pool.mul(vec![cotan, delta])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            Some(if args[0] > 0.0 {
                1.0
            } else if args[0] < 0.0 {
                0.0
            } else {
                0.5
            })
        }
    }

    // ── Dirac delta ─────────────────────────────────────────────────────────────

    /// The Dirac delta distribution `δ(x)`: the distributional derivative of the
    /// [`HeavisidePrimitive`], with `∫ δ = 1`.  Not a classical function; numeric
    /// evaluation is intentionally unimplemented.  Registered as a symbol so the
    /// Laplace transform can map `δ(t − a) ↦ e^{−as}`.
    pub struct DiracDeltaPrimitive;

    impl Primitive for DiracDeltaPrimitive {
        fn name(&self) -> &'static str {
            "diracdelta"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("δ({})", pool.display(args[0]))
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

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            // d/dt atan2(y, x) = (x·y' − y·x') / (x² + y²)
            if args.len() != 2 {
                return None;
            }
            let y = args[0];
            let x = args[1];
            let dy = crate::diff::diff(y, wrt, pool).ok()?.value;
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;

            let neg_one = pool.integer(-1_i32);
            let neg_dx = pool.mul(vec![neg_one, dx]);

            let x_dy = pool.mul(vec![x, dy]);
            let y_dx = pool.mul(vec![y, neg_dx]);
            let numerator = pool.add(vec![x_dy, y_dx]);

            let two = pool.integer(2_i32);
            let x2 = pool.pow(x, two);
            let y2 = pool.pow(y, two);
            let denominator = pool.add(vec![x2, y2]);

            Some(pool.mul(vec![numerator, pool.pow(denominator, neg_one)]))
        }

        fn lean_theorem(&self) -> Option<&'static str> {
            Some("Real.arctan2")
        }
    }

    // ── lambert_w ────────────────────────────────────────────────────────────

    pub struct LambertWPrimitive;

    impl Primitive for LambertWPrimitive {
        fn name(&self) -> &'static str {
            "lambert_w"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("W({})", pool.display(args[0]))
        }

        fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            if matches!(pool.get(x), ExprData::Integer(n) if n.0.is_zero()) {
                return Some(pool.integer(0_i32));
            }
            if is_neg_inv_e_literal(x, pool) {
                return Some(pool.integer(-1_i32));
            }
            None
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let w = pool.func("lambert_w", vec![x]);
            let denom = pool.mul(vec![x, pool.add(vec![pool.integer(1_i32), w])]);
            Some(pool.mul(vec![w, pool.pow(denom, pool.integer(-1_i32)), dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let w = pool.func("lambert_w", vec![x]);
            let denom = pool.mul(vec![x, pool.add(vec![pool.integer(1_i32), w])]);
            Some(vec![pool.mul(vec![
                cotan,
                w,
                pool.pow(denom, pool.integer(-1_i32)),
            ])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 1 {
                crate::special::lambert_w0(args[0])
            } else {
                None
            }
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            if args.len() == 1 {
                args[0].lambert_w0()
            } else {
                None
            }
        }
    }

    // ── digamma ──────────────────────────────────────────────────────────────

    pub struct DigammaPrimitive;

    impl Primitive for DigammaPrimitive {
        fn name(&self) -> &'static str {
            "digamma"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("ψ({})", pool.display(args[0]))
        }

        fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
            let n = positive_integer_literal(args[0], pool)?;
            if n > 20 {
                return None;
            }
            Some(harmonic_minus_gamma(n, pool))
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 1 {
                crate::special::digamma(args[0])
            } else {
                None
            }
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            if args.len() == 1 {
                args[0].digamma()
            } else {
                None
            }
        }
    }

    // ── bessel_j0 / bessel_j1 ────────────────────────────────────────────────

    pub struct BesselJ0Primitive;

    impl Primitive for BesselJ0Primitive {
        fn name(&self) -> &'static str {
            "bessel_j0"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("J₀({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let j1 = pool.func("bessel_j1", vec![x]);
            Some(pool.mul(vec![pool.integer(-1_i32), j1, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let j1 = pool.func("bessel_j1", vec![x]);
            Some(vec![pool.mul(vec![pool.integer(-1_i32), cotan, j1])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 1 {
                Some(crate::special::bessel_j0(args[0]))
            } else {
                None
            }
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            if args.len() == 1 {
                Some(args[0].bessel_jn(0))
            } else {
                None
            }
        }
    }

    pub struct BesselJ1Primitive;

    impl Primitive for BesselJ1Primitive {
        fn name(&self) -> &'static str {
            "bessel_j1"
        }

        fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
            format!("J₁({})", pool.display(args[0]))
        }

        fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
            let x = args[0];
            let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
            let j0 = pool.func("bessel_j0", vec![x]);
            let j1 = pool.func("bessel_j1", vec![x]);
            let quot = pool.mul(vec![j1, pool.pow(x, pool.integer(-1_i32))]);
            let deriv = pool.add(vec![j0, pool.mul(vec![pool.integer(-1_i32), quot])]);
            Some(pool.mul(vec![deriv, dx]))
        }

        fn diff_reverse(
            &self,
            args: &[ExprId],
            cotan: ExprId,
            pool: &ExprPool,
        ) -> Option<Vec<ExprId>> {
            let x = args[0];
            let j0 = pool.func("bessel_j0", vec![x]);
            let j1 = pool.func("bessel_j1", vec![x]);
            let quot = pool.mul(vec![j1, pool.pow(x, pool.integer(-1_i32))]);
            let local = pool.add(vec![j0, pool.mul(vec![pool.integer(-1_i32), quot])]);
            Some(vec![pool.mul(vec![cotan, local])])
        }

        fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
            if args.len() == 1 {
                Some(crate::special::bessel_j1(args[0]))
            } else {
                None
            }
        }

        fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
            if args.len() == 1 {
                Some(args[0].bessel_jn(1))
            } else {
                None
            }
        }
    }

    fn is_neg_inv_e_literal(arg: ExprId, pool: &ExprPool) -> bool {
        match pool.get(arg) {
            ExprData::Float(f) => (f.inner.to_f64() * std::f64::consts::E + 1.0).abs() < 1e-14,
            _ => false,
        }
    }

    fn positive_integer_literal(arg: ExprId, pool: &ExprPool) -> Option<i64> {
        if let ExprData::Integer(n) = pool.get(arg) {
            let v = n.0.to_i64()?;
            (v >= 1).then_some(v)
        } else {
            None
        }
    }

    fn harmonic_minus_gamma(n: i64, pool: &ExprPool) -> ExprId {
        let mut terms = Vec::new();
        for k in 1..n {
            terms.push(pool.rational(1, k as i32));
        }
        let harmonic = if terms.is_empty() {
            pool.integer(0_i32)
        } else {
            pool.add(terms)
        };
        pool.add(vec![harmonic, pool.float(-crate::special::EULER_GAMMA, 53)])
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

        fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
            // Γ(n) = (n-1)! for positive integers n.
            let n = positive_integer_literal(args[0], pool)?;
            if !(1..=21).contains(&n) {
                return None;
            }
            let mut acc: i64 = 1;
            for k in 2..n {
                acc = acc.checked_mul(k)?;
            }
            Some(pool.integer(acc))
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
    fn diff_atan2_full_chain_rule() {
        // d/dx atan2(y, x) = (x·y' - y·x') / (x² + y²), evaluated numerically
        // against a finite-difference approximation for y = x^2, var = x.
        use crate::jit::eval_interp;
        use std::collections::HashMap;

        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.pow(x, pool.integer(2_i32));
        let expr = pool.func("atan2", vec![y, x]);

        let derived = crate::diff::diff(expr, x, &pool).expect("atan2 should be differentiable");

        let h = 1e-6;
        for &xv in &[0.5_f64, 1.0, 2.0, -1.5] {
            let f = |xv: f64| xv.powi(2).atan2(xv);
            let numeric = (f(xv + h) - f(xv - h)) / (2.0 * h);

            let mut env = HashMap::new();
            env.insert(x, xv);
            let analytic = eval_interp(derived.value, &env, &pool)
                .expect("derivative should evaluate numerically");

            assert!(
                (numeric - analytic).abs() < 1e-4,
                "atan2 derivative mismatch at x={xv}: numeric={numeric}, analytic={analytic}"
            );
        }
    }

    #[test]
    fn diff_atan2_simple_case_x_over_constant() {
        // atan2(x, c) for constant c: d/dx atan2(x, c) = c / (c² + x²)
        use crate::jit::eval_interp;
        use std::collections::HashMap;

        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let c = pool.integer(2_i32);
        let expr = pool.func("atan2", vec![x, c]);

        let derived = crate::diff::diff(expr, x, &pool).expect("atan2 should be differentiable");

        for &xv in &[0.0_f64, 1.0, -3.0, 5.0] {
            let expected = 2.0 / (4.0 + xv * xv);
            let mut env = HashMap::new();
            env.insert(x, xv);
            let got = eval_interp(derived.value, &env, &pool)
                .expect("derivative should evaluate numerically");
            assert!(
                (got - expected).abs() < 1e-9,
                "atan2(x,2) derivative mismatch at x={xv}: got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn inverse_hyperbolic_registered() {
        let reg = PrimitiveRegistry::default_registry();
        for name in ["asinh", "acosh", "atanh"] {
            assert!(reg.is_registered(name), "{name} should be registered");
        }
    }

    #[test]
    fn inverse_hyperbolic_numeric_f64_matches_std() {
        let reg = PrimitiveRegistry::default_registry();
        // Sample points chosen inside each function's real domain.
        let cases: &[(&str, f64, f64)] = &[
            ("asinh", 0.7, 0.7_f64.asinh()),
            ("asinh", -2.3, (-2.3_f64).asinh()),
            ("acosh", 1.5, 1.5_f64.acosh()),
            ("acosh", 3.2, 3.2_f64.acosh()),
            ("atanh", 0.4, 0.4_f64.atanh()),
            ("atanh", -0.85, (-0.85_f64).atanh()),
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
    fn inverse_hyperbolic_diff_matches_finite_difference() {
        // d/dx asinh = 1/√(x²+1), acosh = 1/√(x²−1), atanh = 1/(1−x²),
        // checked against a central finite difference at in-domain points.
        use crate::jit::eval_interp;

        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let h = 1e-6;

        // Reference closed forms and in-domain sample points per function.
        let check = |name: &str, f: fn(f64) -> f64, pts: &[f64]| {
            let expr = pool.func(name, vec![x]);
            let derived = crate::diff::diff(expr, x, &pool)
                .unwrap_or_else(|_| panic!("{name} should be differentiable"));
            for &xv in pts {
                let numeric = (f(xv + h) - f(xv - h)) / (2.0 * h);
                let mut env = HashMap::new();
                env.insert(x, xv);
                let analytic = eval_interp(derived.value, &env, &pool)
                    .expect("derivative should evaluate numerically");
                assert!(
                    (numeric - analytic).abs() < 1e-4,
                    "{name}' mismatch at x={xv}: numeric={numeric}, analytic={analytic}"
                );
            }
        };

        check("asinh", f64::asinh, &[0.3, 0.7, 1.9, -1.2]);
        check("acosh", f64::acosh, &[1.4, 2.1, 3.5]);
        check("atanh", f64::atanh, &[0.2, 0.5, -0.7]);
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

    #[test]
    fn heaviside_and_dirac_registered() {
        use crate::kernel::{Domain, ExprData, ExprPool};
        let reg = PrimitiveRegistry::default_registry();
        assert!(reg.is_registered("heaviside"));
        assert!(reg.is_registered("diracdelta"));

        // Heaviside numeric: θ(−1)=0, θ(0)=1/2, θ(1)=1.
        assert_eq!(reg.numeric_f64("heaviside", &[-1.0]), Some(0.0));
        assert_eq!(reg.numeric_f64("heaviside", &[0.0]), Some(0.5));
        assert_eq!(reg.numeric_f64("heaviside", &[1.0]), Some(1.0));

        // Dirac delta has no pointwise f64 value (it is a distribution).
        assert_eq!(reg.numeric_f64("diracdelta", &[0.0]), None);

        // d/dx θ(x) = δ(x): the forward derivative is a δ-function node.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let dh = reg.diff_forward("heaviside", &[x], x, &pool).unwrap();
        let dh = crate::simplify::simplify(dh, &pool).value;
        match pool.get(dh) {
            ExprData::Func { name, args } => {
                assert_eq!(name, "diracdelta");
                assert_eq!(args, vec![x]);
            }
            other => panic!("expected diracdelta(x), got {other:?}"),
        }
    }

    #[test]
    fn lambert_w_registered_and_numeric() {
        let reg = PrimitiveRegistry::default_registry();
        assert!(reg.is_registered("lambert_w"));
        assert_eq!(reg.numeric_f64("lambert_w", &[0.0]), Some(0.0));
        let w1 = reg.numeric_f64("lambert_w", &[1.0]).unwrap();
        assert!((w1 * w1.exp() - 1.0).abs() < 1e-12);
        assert!(reg.numeric_f64("lambert_w", &[-1.0]).is_none()); // < -1/e
    }

    // ── Elliptic special functions ─────────────────────────────────────────────

    /// Recursively evaluate an expression to f64, substituting `var := val`,
    /// dispatching named functions through the default registry.
    fn eval_expr_f64(expr: ExprId, var: ExprId, val: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        let reg = PrimitiveRegistry::default_registry();
        fn go(
            expr: ExprId,
            var: ExprId,
            val: f64,
            pool: &ExprPool,
            reg: &PrimitiveRegistry,
        ) -> f64 {
            pool.with(expr, |data| match data {
                ExprData::Integer(n) => n.0.to_f64(),
                ExprData::Rational(q) => q.0.to_f64(),
                ExprData::Float(f) => f.inner.to_f64(),
                ExprData::Symbol { .. } => {
                    if expr == var {
                        val
                    } else {
                        f64::NAN
                    }
                }
                ExprData::Add(args) => args.iter().map(|&a| go(a, var, val, pool, reg)).sum(),
                ExprData::Mul(args) => args.iter().map(|&a| go(a, var, val, pool, reg)).product(),
                ExprData::Pow { base, exp } => {
                    let b = go(*base, var, val, pool, reg);
                    let e = go(*exp, var, val, pool, reg);
                    b.powf(e)
                }
                ExprData::Func { name, args } => {
                    let vals: Vec<f64> = args.iter().map(|&a| go(a, var, val, pool, reg)).collect();
                    reg.numeric_f64(name, &vals).unwrap_or(f64::NAN)
                }
                _ => f64::NAN,
            })
        }
        go(expr, var, val, pool, &reg)
    }

    #[test]
    fn elliptic_complete_numeric() {
        let reg = PrimitiveRegistry::default_registry();
        let half_pi = std::f64::consts::FRAC_PI_2;
        // K(0) = E(0) = π/2.
        assert!((reg.numeric_f64("EllipticK", &[0.0]).unwrap() - half_pi).abs() < 1e-9);
        assert!((reg.numeric_f64("EllipticE", &[0.0]).unwrap() - half_pi).abs() < 1e-9);
        // Reference values (parameter convention m).
        assert!((reg.numeric_f64("EllipticK", &[0.5]).unwrap() - 1.854_074_677_3).abs() < 1e-6);
        assert!((reg.numeric_f64("EllipticE", &[0.5]).unwrap() - 1.350_643_881_0).abs() < 1e-6);
    }

    #[test]
    fn elliptic_incomplete_numeric() {
        let reg = PrimitiveRegistry::default_registry();
        let qpi = std::f64::consts::FRAC_PI_4;
        // Reference (scipy/Mathematica, parameter convention m):
        //   EllipticF(π/4, 1/2) = 0.8260178762, EllipticE(π/4, 1/2) = 0.7481865042.
        assert!((reg.numeric_f64("EllipticF", &[qpi, 0.5]).unwrap() - 0.826_017_876).abs() < 1e-6);
        assert!((reg.numeric_f64("EllipticE", &[qpi, 0.5]).unwrap() - 0.748_186_504).abs() < 1e-6);
        // φ = 0 gives 0 for both incomplete integrals.
        assert!(reg.numeric_f64("EllipticF", &[0.0, 0.5]).unwrap().abs() < 1e-12);
        assert!(reg.numeric_f64("EllipticE", &[0.0, 0.5]).unwrap().abs() < 1e-12);
    }

    #[test]
    fn elliptic_pi_numeric() {
        let reg = PrimitiveRegistry::default_registry();
        // Π(0, φ, m) = F(φ, m).
        let qpi = std::f64::consts::FRAC_PI_4;
        let pi0 = reg.numeric_f64("EllipticPi", &[0.0, qpi, 0.5]).unwrap();
        let f = reg.numeric_f64("EllipticF", &[qpi, 0.5]).unwrap();
        assert!((pi0 - f).abs() < 1e-7, "Π(0,φ,m)={pi0} F(φ,m)={f}");
    }

    #[test]
    fn elliptic_out_of_domain() {
        let reg = PrimitiveRegistry::default_registry();
        // K(m) diverges for m ≥ 1.
        assert!(reg.numeric_f64("EllipticK", &[1.0]).is_none());
        // F(π/2, 1) integrand singular at the endpoint.
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!(reg.numeric_f64("EllipticF", &[half_pi, 1.0]).is_none());
    }

    #[test]
    fn elliptic_f_parse_roundtrip() {
        use crate::kernel::{Domain, ExprData};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        let e = crate::parse::parse("EllipticF(x, 1/2)", &pool, &mut syms).unwrap();
        pool.with(e, |data| match data {
            ExprData::Func { name, args } => {
                assert_eq!(name, "EllipticF");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected a 2-arg EllipticF Func node"),
        });
    }

    #[test]
    fn elliptic_f_diff_phi() {
        use crate::kernel::Domain;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let m = pool.rational(1_i32, 2_i32);
        let f = pool.func("EllipticF", vec![x, m]);
        let d = crate::diff::diff(f, x, &pool).unwrap().value;
        // d/dx F(x, 1/2) = 1/√(1 − (1/2)·sin²x); check numerically at x=0.7.
        let xv = 0.7_f64;
        let got = eval_expr_f64(d, x, xv, &pool);
        let expect = 1.0 / (1.0 - 0.5 * xv.sin().powi(2)).sqrt();
        assert!((got - expect).abs() < 1e-9, "got {got} expect {expect}");
    }

    #[test]
    fn elliptic_e_incomplete_diff_phi() {
        use crate::kernel::Domain;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let m = pool.rational(1_i32, 2_i32);
        let e = pool.func("EllipticE", vec![x, m]);
        let d = crate::diff::diff(e, x, &pool).unwrap().value;
        // d/dx E(x, 1/2) = √(1 − (1/2)·sin²x).
        let xv = 0.7_f64;
        let got = eval_expr_f64(d, x, xv, &pool);
        let expect = (1.0 - 0.5 * xv.sin().powi(2)).sqrt();
        assert!((got - expect).abs() < 1e-9, "got {got} expect {expect}");
    }

    #[test]
    fn elliptic_pi_diff_phi() {
        use crate::kernel::Domain;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let n = pool.rational(1_i32, 4_i32);
        let m = pool.rational(1_i32, 2_i32);
        let p = pool.func("EllipticPi", vec![n, x, m]);
        let d = crate::diff::diff(p, x, &pool).unwrap().value;
        // ∂/∂φ Π(n,φ,m) = 1/((1 − n·sin²φ)·√(1 − m·sin²φ)).
        let xv = 0.6_f64;
        let got = eval_expr_f64(d, x, xv, &pool);
        let s2 = xv.sin().powi(2);
        let expect = 1.0 / ((1.0 - 0.25 * s2) * (1.0 - 0.5 * s2).sqrt());
        assert!((got - expect).abs() < 1e-9, "got {got} expect {expect}");
    }

    #[test]
    fn elliptic_k_diff_m_finite_difference() {
        use crate::kernel::Domain;
        let pool = ExprPool::new();
        let mvar = pool.symbol("m", Domain::Real);
        let k = pool.func("EllipticK", vec![mvar]);
        let d = crate::diff::diff(k, mvar, &pool).unwrap().value;
        let reg = PrimitiveRegistry::default_registry();
        let m0 = 0.4_f64;
        let analytic = eval_expr_f64(d, mvar, m0, &pool);
        let h = 1e-6;
        let kp = reg.numeric_f64("EllipticK", &[m0 + h]).unwrap();
        let km = reg.numeric_f64("EllipticK", &[m0 - h]).unwrap();
        let fd = (kp - km) / (2.0 * h);
        assert!((analytic - fd).abs() < 1e-5, "analytic {analytic} fd {fd}");
    }

    #[test]
    fn elliptic_e_complete_diff_m_finite_difference() {
        use crate::kernel::Domain;
        let pool = ExprPool::new();
        let mvar = pool.symbol("m", Domain::Real);
        let e = pool.func("EllipticE", vec![mvar]);
        let d = crate::diff::diff(e, mvar, &pool).unwrap().value;
        let reg = PrimitiveRegistry::default_registry();
        let m0 = 0.4_f64;
        let analytic = eval_expr_f64(d, mvar, m0, &pool);
        let h = 1e-6;
        let ep = reg.numeric_f64("EllipticE", &[m0 + h]).unwrap();
        let em = reg.numeric_f64("EllipticE", &[m0 - h]).unwrap();
        let fd = (ep - em) / (2.0 * h);
        assert!((analytic - fd).abs() < 1e-5, "analytic {analytic} fd {fd}");
    }

    #[test]
    fn special_functions_numeric_and_folds() {
        use crate::kernel::{Domain, ExprData};
        use crate::special::EULER_GAMMA;

        let reg = PrimitiveRegistry::default_registry();
        let pool = ExprPool::new();

        assert_eq!(reg.numeric_f64("lambert_w", &[0.0]).unwrap(), 0.0);
        let em = crate::special::lambert_w0_domain_min();
        assert!((reg.numeric_f64("lambert_w", &[em]).unwrap() + 1.0).abs() < 1e-12);
        assert!(reg.numeric_f64("lambert_w", &[em - 0.1]).is_none());

        let psi1 = reg.numeric_f64("digamma", &[1.0]).unwrap();
        assert!((psi1 + EULER_GAMMA).abs() < 1e-12);

        assert!((reg.numeric_f64("bessel_j0", &[0.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!(reg.numeric_f64("bessel_j1", &[0.0]).unwrap().abs() < 1e-15);

        assert_eq!(
            reg.get("lambert_w")
                .unwrap()
                .simplify(&[pool.integer(0_i32)], &pool)
                .unwrap(),
            pool.integer(0_i32)
        );

        let neg_em = pool.float(em, 53);
        assert_eq!(
            reg.get("lambert_w")
                .unwrap()
                .simplify(&[neg_em], &pool)
                .unwrap(),
            pool.integer(-1_i32)
        );

        let psi3 = reg
            .get("digamma")
            .unwrap()
            .simplify(&[pool.integer(3_i32)], &pool)
            .unwrap();
        pool.with(psi3, |data| {
            if let ExprData::Add(args) = data {
                assert_eq!(args.len(), 2);
            } else {
                panic!("expected harmonic − γ fold");
            }
        });

        let x = pool.symbol("x", Domain::Real);
        let j0 = pool.func("bessel_j0", vec![x]);
        let dj0 = crate::diff::diff(j0, x, &pool).unwrap().value;
        let expected = pool.mul(vec![pool.integer(-1_i32), pool.func("bessel_j1", vec![x])]);
        assert_eq!(dj0, expected);
    }

    #[test]
    fn lambert_w_ball_capability() {
        use crate::ball::ArbBall;
        use crate::primitive::{Capabilities, PrimitiveRegistry};

        let reg = PrimitiveRegistry::default_registry();
        let b = ArbBall::from_f64(1.0, 128);
        assert!(
            reg.numeric_ball("lambert_w", std::slice::from_ref(&b))
                .is_some(),
            "lambert_w ball eval at 1.0 should succeed"
        );
        assert!(reg
            .capabilities("lambert_w")
            .contains(Capabilities::NUMERIC_BALL));
    }

    #[test]
    fn lambert_w_diff_finite_difference() {
        use crate::jit::eval_interp;
        use crate::kernel::Domain;
        use std::collections::HashMap;

        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let w = pool.func("lambert_w", vec![x]);
        let d = crate::diff::diff(w, x, &pool).unwrap().value;
        let reg = PrimitiveRegistry::default_registry();
        let x0 = 0.5_f64;
        let h = 1e-6;
        let wp = reg.numeric_f64("lambert_w", &[x0 + h]).unwrap();
        let wm = reg.numeric_f64("lambert_w", &[x0 - h]).unwrap();
        let fd = (wp - wm) / (2.0 * h);
        let mut env = HashMap::new();
        env.insert(x, x0);
        let analytic = eval_interp(d, &env, &pool).unwrap();
        assert!((analytic - fd).abs() < 1e-4, "analytic {analytic} fd {fd}");
    }
}
