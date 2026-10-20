//! Recognising named SI derived units (`kg*m/s^2` → `N`).

use super::{table, Scale, Unit};
use rug::Rational;

/// SI derived units with special names, in preference order. Their
/// dimensions are pairwise distinct, so at most one matches a given unit; the
/// order only fixes how [`Unit::to_derived`]'s `allowed` filter is read.
/// `Gy`/`Sv` (`m^2/s^2`) and `Bq` (`1/s`) are deliberately absent: they share
/// a dimension with a speed squared / `Hz`, so renaming would assert a
/// physical meaning the dimension does not carry.
pub const DERIVED_UNITS: [&str; 13] = [
    "N", "Pa", "J", "W", "C", "V", "ohm", "F", "H", "T", "Wb", "S", "Hz",
];

/// SI prefixes (multiples of 1000 only) tried when the scale is not 1.
const PREFIXES_3: [(&str, i32); 16] = [
    ("Y", 24),
    ("Z", 21),
    ("E", 18),
    ("P", 15),
    ("T", 12),
    ("G", 9),
    ("M", 6),
    ("k", 3),
    ("m", -3),
    ("µ", -6),
    ("n", -9),
    ("p", -12),
    ("f", -15),
    ("a", -18),
    ("z", -21),
    ("y", -24),
];

fn pow10(e: i32) -> Rational {
    super::table::dec(&format!("1e{e}"))
}

impl Unit {
    /// The named derived unit (optionally with an SI prefix that is a power of
    /// 1000) that is **exactly** this unit, if there is one.
    ///
    /// `kg*m/s^2 → N`, `g*m/s^2 → mN`, `A*s → C`, `kg*m^2/(A*s^3) → V`. Returns
    /// `None` when the unit already is a single named factor, is dimensionless,
    /// affine, mentions a dimensionless factor (`rad/s` is not `Hz`), has no
    /// derived unit of its dimension, or differs from the match by a scale
    /// that is not a power of 1000 (`kg*cm/s^2` stays). `allowed` restricts
    /// the candidates to the given names (a subset of [`DERIVED_UNITS`]).
    pub fn to_derived(&self, allowed: Option<&[&str]>) -> Option<Unit> {
        if self.is_affine() || self.dim.is_dimensionless() {
            return None;
        }
        if self.factors.len() == 1 && self.factors[0].1 == 1 {
            return None;
        }
        for (name, _) in &self.factors {
            let atom = super::parse::parse_atom(name).ok()?;
            if atom.dim.is_dimensionless() {
                return None;
            }
        }
        for name in DERIVED_UNITS {
            if let Some(a) = allowed {
                if !a.contains(&name) {
                    continue;
                }
            }
            let Some(base) = table::lookup(name) else {
                continue;
            };
            if base.dim != self.dim {
                continue;
            }
            let ratio = self.scale.div(&base.scale);
            if ratio.pi_power != 0 {
                return None;
            }
            if ratio.is_one() {
                return Some(base);
            }
            for (p, e) in PREFIXES_3 {
                if ratio.ratio == pow10(e) {
                    return table::lookup(&format!("{p}{name}"));
                }
            }
            return None;
        }
        None
    }

    /// [`Unit::simplify`] followed by [`Unit::to_derived`]: merge factors of
    /// equal dimension, then rename the result to a named derived unit when it
    /// is exactly one. The returned factor is the same as for `simplify`
    /// (renaming never changes the scale).
    pub fn simplify_derived(&self, allowed: Option<&[&str]>) -> (Scale, Unit) {
        let (f, u) = self.simplify();
        match u.to_derived(allowed) {
            Some(d) => (f, d),
            None => (f, u),
        }
    }
}
