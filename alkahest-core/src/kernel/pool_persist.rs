//! V1-14 — Persistent / incremental `ExprPool`.
//!
//! Opt-in serialization of the intern table to disk so long-running notebooks
//! and repeated simplifications don't rebuild the pool from scratch on every
//! process start.
//!
//! # Status
//!
//! This is the v1.0 scope: a **versioned binary file** (not a true mmap-backed
//! arena).  `checkpoint()` writes the full node vector atomically (temp file +
//! `rename`); `open_persistent(path)` reads it back if it exists.  Structural
//! hashes line up by construction — the re-interned `ExprData` values hash
//! identically, so a subsequent `pool.add([x, y])` lookup hits the rebuilt
//! index.
//!
//! A true mmap/CapnProto arena with `ExprData` stored inline is tracked as a
//! v2.0 follow-up; it requires a ground-up redesign of `ExprData` to avoid
//! heap allocations for `Vec<ExprId>` children.
//!
//! # File format (v1)
//!
//! ```text
//!   Magic     = "ALKP"             (4 bytes)
//!   Version   = u32 (**4** = symbol `commutative` flag on `(tag 0)`; **3** = BigO tag 12; **2** = quantifiers 10–11; **1** = original 0–9)
//!   Flags     = u32                 (reserved; always 0 in v1)
//!   NodeCount = u64
//!   Nodes     = NodeCount × TaggedNode
//! ```
//!
//! Each `TaggedNode`:
//! ```text
//!   tag : u8
//!     0 Symbol     -> domain:u8, [commutative:u8 if format≥4], len:u32, name
//!     1 Integer    -> len:u32, base-10 digits (ASCII, optionally '-' prefix)
//!     2 Rational   -> numer_len:u32, numer, denom_len:u32, denom
//!     3 Float      -> prec:u32, len:u32, base-16 mantissa (rug to_string_radix)
//!     4 Add        -> arity:u32, ExprId.0 (u32) × arity
//!     5 Mul        -> arity:u32, ExprId.0 × arity
//!     6 Pow        -> base:u32, exp:u32
//!     7 Func       -> len:u32, name, arity:u32, ExprId.0 × arity
//!     8 Piecewise  -> n_branches:u32, (cond:u32, val:u32) × n, default:u32
//!     9 Predicate  -> kind:u8, arity:u32, ExprId.0 × arity
//!     10 Forall   -> var:u32, body:u32
//!     11 Exists   -> var:u32, body:u32
//!     12 BigO    -> inner:u32
//! ```
//!
//! File version (`Version` u32 field): **1** is the original v1.0 layout (tags 0–9 only).
//! **2** adds tags 10–11 for quantifiers. **3** adds tag 12 for `BigO`. **4** adds
//! `commutative: u8` after `domain` on symbol nodes (V3-2).
//! Current writers emit version **4**; readers accept **1** … **4**.
//!
//! All integers are little-endian.

use crate::kernel::domain::Domain;
use crate::kernel::expr::{BigFloat, BigInt, BigRat, ExprData, ExprId, PredicateKind};
use crate::kernel::pool::ExprPool;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"ALKP";
/// Oldest readable format (predicate / piecewise only).
const POOL_FORMAT_V1: u32 = 1;
/// Adds `Forall` / `Exists` node tags 10–11.
const POOL_FORMAT_V2: u32 = 2;
/// Adds `BigO` tag 12 (V2-15 series API).
const POOL_FORMAT_V3: u32 = 3;
/// Symbol nodes carry `commutative: u8` after `domain` (V3-2).
const POOL_FORMAT_V4: u32 = 4;
const POOL_FORMAT_WRITE: u32 = POOL_FORMAT_V4;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// I/O errors from checkpoint and restore operations on `ExprPool`.
///
/// Codes: `E-IO-001` … `E-IO-009`.
#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    BadUtf8,
    BadDomain(u8),
    BadTag(u8),
    BadPredicateKind(u8),
    BadNumeric(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "io error: {e}"),
            IoError::BadMagic => write!(f, "not an alkahest pool file (bad magic)"),
            IoError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported pool file version {v}; run `alkahest migrate-pool`"
                )
            }
            IoError::Truncated => write!(f, "pool file truncated or incomplete"),
            IoError::BadUtf8 => write!(f, "pool file contains invalid UTF-8"),
            IoError::BadDomain(b) => write!(f, "pool file has unknown domain tag {b}"),
            IoError::BadTag(b) => write!(f, "pool file has unknown node tag {b}"),
            IoError::BadPredicateKind(b) => {
                write!(f, "pool file has unknown predicate kind {b}")
            }
            IoError::BadNumeric(s) => write!(f, "pool file has invalid numeric: {s}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(e: io::Error) -> Self {
        IoError::Io(e)
    }
}

impl crate::errors::AlkahestError for IoError {
    fn code(&self) -> &'static str {
        match self {
            IoError::Io(_) => "E-IO-001",
            IoError::BadMagic => "E-IO-002",
            IoError::UnsupportedVersion(_) => "E-IO-003",
            IoError::Truncated => "E-IO-004",
            IoError::BadUtf8 => "E-IO-005",
            IoError::BadDomain(_) => "E-IO-006",
            IoError::BadTag(_) => "E-IO-007",
            IoError::BadPredicateKind(_) => "E-IO-008",
            IoError::BadNumeric(_) => "E-IO-009",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            IoError::BadMagic => Some(
                "file is not an alkahest pool; check the path or regenerate with ExprPool::checkpoint()",
            ),
            IoError::UnsupportedVersion(_) => Some(
                "run the `alkahest migrate-pool` CLI to upgrade the file, or regenerate from source",
            ),
            IoError::Truncated => Some(
                "file was truncated (likely a crash during checkpoint); rerun from source and checkpoint again",
            ),
            _ => None,
        }
    }
}

/// Deprecated alias — use [`IoError`] instead.
#[deprecated(since = "2.0.0", note = "renamed to IoError with E-IO-* codes")]
pub type PoolPersistError = IoError;

// ---------------------------------------------------------------------------
// Low-level binary helpers
// ---------------------------------------------------------------------------

fn write_u8(w: &mut impl Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

fn write_ids(w: &mut impl Write, ids: &[ExprId]) -> io::Result<()> {
    write_u32(w, ids.len() as u32)?;
    for id in ids {
        write_u32(w, id.0)?;
    }
    Ok(())
}

fn read_u8(r: &mut impl Read) -> Result<u8, IoError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|_| IoError::Truncated)?;
    Ok(b[0])
}

fn read_u32(r: &mut impl Read) -> Result<u32, IoError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|_| IoError::Truncated)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64, IoError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|_| IoError::Truncated)?;
    Ok(u64::from_le_bytes(b))
}

fn read_str(r: &mut impl Read) -> Result<String, IoError> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|_| IoError::Truncated)?;
    String::from_utf8(buf).map_err(|_| IoError::BadUtf8)
}

fn read_ids(r: &mut impl Read) -> Result<Vec<ExprId>, IoError> {
    let arity = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(arity);
    for _ in 0..arity {
        out.push(ExprId(read_u32(r)?));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Domain <-> u8
// ---------------------------------------------------------------------------

fn domain_to_u8(d: &Domain) -> u8 {
    match d {
        Domain::Real => 0,
        Domain::Complex => 1,
        Domain::Integer => 2,
        Domain::Positive => 3,
        Domain::NonNegative => 4,
        Domain::NonZero => 5,
    }
}

fn u8_to_domain(b: u8) -> Result<Domain, IoError> {
    match b {
        0 => Ok(Domain::Real),
        1 => Ok(Domain::Complex),
        2 => Ok(Domain::Integer),
        3 => Ok(Domain::Positive),
        4 => Ok(Domain::NonNegative),
        5 => Ok(Domain::NonZero),
        b => Err(IoError::BadDomain(b)),
    }
}

fn pred_to_u8(k: &PredicateKind) -> u8 {
    // Enumerate all variants in a stable order.
    match k {
        PredicateKind::Eq => 0,
        PredicateKind::Ne => 1,
        PredicateKind::Lt => 2,
        PredicateKind::Le => 3,
        PredicateKind::Gt => 4,
        PredicateKind::Ge => 5,
        PredicateKind::And => 6,
        PredicateKind::Or => 7,
        PredicateKind::Not => 8,
        PredicateKind::True => 9,
        PredicateKind::False => 10,
    }
}

fn u8_to_pred(b: u8) -> Result<PredicateKind, IoError> {
    match b {
        0 => Ok(PredicateKind::Eq),
        1 => Ok(PredicateKind::Ne),
        2 => Ok(PredicateKind::Lt),
        3 => Ok(PredicateKind::Le),
        4 => Ok(PredicateKind::Gt),
        5 => Ok(PredicateKind::Ge),
        6 => Ok(PredicateKind::And),
        7 => Ok(PredicateKind::Or),
        8 => Ok(PredicateKind::Not),
        9 => Ok(PredicateKind::True),
        10 => Ok(PredicateKind::False),
        b => Err(IoError::BadPredicateKind(b)),
    }
}

// ---------------------------------------------------------------------------
// Node ↔ bytes
// ---------------------------------------------------------------------------

fn write_node(w: &mut impl Write, node: &ExprData) -> io::Result<()> {
    match node {
        ExprData::Symbol {
            name,
            domain,
            commutative,
        } => {
            write_u8(w, 0)?;
            write_u8(w, domain_to_u8(domain))?;
            write_u8(w, u8::from(*commutative))?;
            write_str(w, name)
        }
        ExprData::Integer(BigInt(n)) => {
            write_u8(w, 1)?;
            write_str(w, &n.to_string())
        }
        ExprData::Rational(BigRat(r)) => {
            write_u8(w, 2)?;
            write_str(w, &r.numer().to_string())?;
            write_str(w, &r.denom().to_string())
        }
        ExprData::Float(BigFloat { inner, prec }) => {
            write_u8(w, 3)?;
            write_u32(w, *prec)?;
            // rug::Float::to_string_radix(16, None) round-trips exactly.
            write_str(w, &inner.to_string_radix(16, None))
        }
        ExprData::Add(children) => {
            write_u8(w, 4)?;
            write_ids(w, children)
        }
        ExprData::Mul(children) => {
            write_u8(w, 5)?;
            write_ids(w, children)
        }
        ExprData::Pow { base, exp } => {
            write_u8(w, 6)?;
            write_u32(w, base.0)?;
            write_u32(w, exp.0)
        }
        ExprData::Func { name, args } => {
            write_u8(w, 7)?;
            write_str(w, name)?;
            write_ids(w, args)
        }
        ExprData::Piecewise { branches, default } => {
            write_u8(w, 8)?;
            write_u32(w, branches.len() as u32)?;
            for (c, v) in branches {
                write_u32(w, c.0)?;
                write_u32(w, v.0)?;
            }
            write_u32(w, default.0)
        }
        ExprData::Predicate { kind, args } => {
            write_u8(w, 9)?;
            write_u8(w, pred_to_u8(kind))?;
            write_ids(w, args)
        }
        ExprData::Forall { var, body } => {
            write_u8(w, 10)?;
            write_u32(w, var.0)?;
            write_u32(w, body.0)
        }
        ExprData::Exists { var, body } => {
            write_u8(w, 11)?;
            write_u32(w, var.0)?;
            write_u32(w, body.0)
        }
        ExprData::BigO(inner) => {
            write_u8(w, 12)?;
            write_u32(w, inner.0)
        }
    }
}

fn read_node(r: &mut impl Read, format_version: u32) -> Result<ExprData, IoError> {
    let tag = read_u8(r)?;
    match tag {
        0 => {
            let domain = u8_to_domain(read_u8(r)?)?;
            let commutative = if format_version >= POOL_FORMAT_V4 {
                read_u8(r)? != 0
            } else {
                true
            };
            let name = read_str(r)?;
            Ok(ExprData::Symbol {
                name,
                domain,
                commutative,
            })
        }
        1 => {
            let s = read_str(r)?;
            let n: rug::Integer = s
                .parse()
                .map_err(|_| IoError::BadNumeric(format!("integer: {s}")))?;
            Ok(ExprData::Integer(BigInt(n)))
        }
        2 => {
            let nstr = read_str(r)?;
            let dstr = read_str(r)?;
            let n: rug::Integer = nstr
                .parse()
                .map_err(|_| IoError::BadNumeric(format!("numer: {nstr}")))?;
            let d: rug::Integer = dstr
                .parse()
                .map_err(|_| IoError::BadNumeric(format!("denom: {dstr}")))?;
            Ok(ExprData::Rational(BigRat(rug::Rational::from((n, d)))))
        }
        3 => {
            let prec = read_u32(r)?;
            let s = read_str(r)?;
            let f = rug::Float::parse_radix(&s, 16)
                .map_err(|_| IoError::BadNumeric(format!("float: {s}")))?;
            let inner = rug::Float::with_val(prec, f);
            Ok(ExprData::Float(BigFloat { inner, prec }))
        }
        4 => Ok(ExprData::Add(read_ids(r)?)),
        5 => Ok(ExprData::Mul(read_ids(r)?)),
        6 => {
            let base = ExprId(read_u32(r)?);
            let exp = ExprId(read_u32(r)?);
            Ok(ExprData::Pow { base, exp })
        }
        7 => {
            let name = read_str(r)?;
            let args = read_ids(r)?;
            Ok(ExprData::Func { name, args })
        }
        8 => {
            let n = read_u32(r)? as usize;
            let mut branches = Vec::with_capacity(n);
            for _ in 0..n {
                let c = ExprId(read_u32(r)?);
                let v = ExprId(read_u32(r)?);
                branches.push((c, v));
            }
            let default = ExprId(read_u32(r)?);
            Ok(ExprData::Piecewise { branches, default })
        }
        9 => {
            let kind = u8_to_pred(read_u8(r)?)?;
            let args = read_ids(r)?;
            Ok(ExprData::Predicate { kind, args })
        }
        10 => {
            if format_version < POOL_FORMAT_V2 {
                return Err(IoError::BadTag(10));
            }
            let var = ExprId(read_u32(r)?);
            let body = ExprId(read_u32(r)?);
            Ok(ExprData::Forall { var, body })
        }
        11 => {
            if format_version < POOL_FORMAT_V2 {
                return Err(IoError::BadTag(11));
            }
            let var = ExprId(read_u32(r)?);
            let body = ExprId(read_u32(r)?);
            Ok(ExprData::Exists { var, body })
        }
        12 => {
            if format_version < POOL_FORMAT_V3 {
                return Err(IoError::BadTag(12));
            }
            let inner = ExprId(read_u32(r)?);
            Ok(ExprData::BigO(inner))
        }
        b => Err(IoError::BadTag(b)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write the pool's full node table to `path` atomically (temp + rename).
pub fn save_to(pool: &ExprPool, path: impl AsRef<Path>) -> Result<(), IoError> {
    let path = path.as_ref();
    let tmp: PathBuf = {
        let mut p = path.to_path_buf();
        let mut name = p
            .file_name()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| std::ffi::OsString::from("pool"));
        name.push(".tmp");
        p.set_file_name(name);
        p
    };

    {
        let f = File::create(&tmp)?;
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC)?;
        write_u32(&mut w, POOL_FORMAT_WRITE)?;
        write_u32(&mut w, 0u32)?; // flags

        let count = pool.len();
        write_u64(&mut w, count as u64)?;
        for i in 0..count {
            let data = pool.get(ExprId(i as u32));
            write_node(&mut w, &data)?;
        }

        w.flush()?;
        w.get_ref().sync_all()?;
    }

    fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a pool from `path`.  Returns `Ok(None)` if the file does not exist,
/// so callers can use `load_or_new` semantics.
pub fn load_from(path: impl AsRef<Path>) -> Result<Option<ExprPool>, IoError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }

    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|_| IoError::Truncated)?;
    if &magic != MAGIC {
        return Err(IoError::BadMagic);
    }

    let version = read_u32(&mut r)?;
    if version != POOL_FORMAT_V1
        && version != POOL_FORMAT_V2
        && version != POOL_FORMAT_V3
        && version != POOL_FORMAT_V4
    {
        return Err(IoError::UnsupportedVersion(version));
    }
    let _flags = read_u32(&mut r)?;

    let pool = ExprPool::new();
    let count = read_u64(&mut r)? as usize;
    for expected in 0..count {
        let data = read_node(&mut r, version)?;
        let got = pool.intern(data);
        debug_assert_eq!(got.0 as usize, expected, "pool id drift during load");
    }

    Ok(Some(pool))
}

/// Load if `path` exists, else return a fresh pool.
pub fn open_persistent(path: impl AsRef<Path>) -> Result<ExprPool, IoError> {
    match load_from(path)? {
        Some(p) => Ok(p),
        None => Ok(ExprPool::new()),
    }
}

// ---------------------------------------------------------------------------
// ExprPool convenience methods
// ---------------------------------------------------------------------------

impl ExprPool {
    /// V1-14 — write the current pool to `path` atomically.  Equivalent to
    /// [`save_to`].
    pub fn checkpoint(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        save_to(self, path)
    }

    /// V1-14 — load a persisted pool, or return a fresh one if the file does
    /// not exist.  Equivalent to [`open_persistent`].
    pub fn open_persistent(path: impl AsRef<Path>) -> Result<Self, IoError> {
        open_persistent(path)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprData};

    fn tempfile() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "alkahest_pool_{}_{}.akp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn round_trip_small_pool() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Positive);
        let two = p.integer(2_i32);
        let three_halves = p.rational(3, 2);
        let f = p.float(1.5_f64, 53);
        let xp = p.pow(x, two);
        let fn_node = p.func("sin", vec![xp]);
        let _sum = p.add(vec![fn_node, y, three_halves, f]);

        let path = tempfile();
        p.checkpoint(&path).unwrap();

        let q = ExprPool::open_persistent(&path).unwrap();
        assert_eq!(q.len(), p.len(), "node count must match");
        for i in 0..p.len() {
            let id = ExprId(i as u32);
            assert_eq!(p.get(id), q.get(id), "node {i} mismatch after round-trip");
        }

        // Re-interning the same structures under q must collide with the
        // restored IDs — this is the hash-cons stability guarantee.
        let q_x = q.symbol("x", Domain::Real);
        assert_eq!(q_x, x, "symbol id drifted across checkpoint");
        let q_two = q.integer(2_i32);
        assert_eq!(q_two, two);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_rejected() {
        let path = tempfile();
        std::fs::write(&path, b"nope1234").unwrap();
        match load_from(&path) {
            Err(IoError::BadMagic) => {}
            other => panic!("expected BadMagic, got {:?}", other.err()),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_file_returns_fresh() {
        let path = tempfile();
        assert!(!path.exists());
        let p = ExprPool::open_persistent(&path).unwrap();
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn predicate_and_piecewise_round_trip() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let zero = p.integer(0_i32);
        let one = p.integer(1_i32);
        let neg_one = p.integer(-1_i32);
        let cond = p.intern(ExprData::Predicate {
            kind: PredicateKind::Gt,
            args: vec![x, zero],
        });
        let pc = p.intern(ExprData::Piecewise {
            branches: vec![(cond, one)],
            default: neg_one,
        });

        let path = tempfile();
        p.checkpoint(&path).unwrap();
        let q = ExprPool::open_persistent(&path).unwrap();
        assert_eq!(p.get(pc), q.get(pc));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn big_o_round_trip() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let o = p.big_o(p.pow(x, p.integer(6)));
        let path = tempfile();
        p.checkpoint(&path).unwrap();
        let q = ExprPool::open_persistent(&path).unwrap();
        assert_eq!(q.get(o), p.get(o));
        let _ = fs::remove_file(&path);
    }
}
