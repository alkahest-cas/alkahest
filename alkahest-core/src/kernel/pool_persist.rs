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
/// Adds `RootSum` tag 13 (algebraic-residue logarithmic part).
const POOL_FORMAT_V5: u32 = 5;
const POOL_FORMAT_WRITE: u32 = POOL_FORMAT_V5;

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

/// Largest float precision, in bits, a pool file may declare.
///
/// `rug::Float::with_val` panics on a precision of 0 or past its own maximum,
/// and allocates the whole mantissa up front, so a crafted `prec = u32::MAX`
/// asks for half a gigabyte before a single digit is read.  This is the same
/// ceiling the Python bindings put on every user-supplied precision.
const MAX_FILE_FLOAT_PREC: u32 = 1 << 24;

/// The smallest encoding of any node (`tag` + one `u32`), used to bound the
/// declared node count by the bytes actually present.
const MIN_NODE_BYTES: u64 = 5;

/// A reader that knows how many bytes are left in the file.
///
/// Every length-prefixed field (a string, a child list, a piecewise branch
/// list) is checked against the bytes remaining *before* anything is
/// allocated for it.  Without that, a four-byte length of `0xFFFF_FFF0` in a
/// corrupt or hostile file is a 4 GiB `vec![0; len]` — an allocation failure
/// that aborts the process instead of returning an error.
struct Bounded<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> Bounded<R> {
    fn new(inner: R, len: u64) -> Self {
        Bounded {
            inner,
            remaining: len,
        }
    }

    /// Refuse a field that claims `count` items of `size` bytes each when the
    /// file does not hold that many bytes.
    fn ensure(&self, count: u64, size: u64) -> Result<(), IoError> {
        match count.checked_mul(size) {
            Some(n) if n <= self.remaining => Ok(()),
            _ => Err(IoError::Truncated),
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), IoError> {
        self.ensure(buf.len() as u64, 1)?;
        self.inner.read_exact(buf).map_err(|_| IoError::Truncated)?;
        self.remaining -= buf.len() as u64;
        Ok(())
    }
}

fn read_u8(r: &mut Bounded<impl Read>) -> Result<u8, IoError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32(r: &mut Bounded<impl Read>) -> Result<u32, IoError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut Bounded<impl Read>) -> Result<u64, IoError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_str(r: &mut Bounded<impl Read>) -> Result<String, IoError> {
    let len = u64::from(read_u32(r)?);
    r.ensure(len, 1)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| IoError::BadUtf8)
}

fn read_ids(r: &mut Bounded<impl Read>) -> Result<Vec<ExprId>, IoError> {
    let arity = u64::from(read_u32(r)?);
    r.ensure(arity, 4)?;
    let mut out = Vec::with_capacity(arity as usize);
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
        ExprData::RootSum { poly, var, body } => {
            write_u8(w, 13)?;
            write_u32(w, poly.0)?;
            write_u32(w, var.0)?;
            write_u32(w, body.0)
        }
    }
}

fn read_node(r: &mut Bounded<impl Read>, format_version: u32) -> Result<ExprData, IoError> {
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
            if d == 0 {
                // `Rational::from((n, 0))` panics.
                return Err(IoError::BadNumeric(format!("rational {nstr}/0")));
            }
            Ok(ExprData::Rational(BigRat(rug::Rational::from((n, d)))))
        }
        3 => {
            let prec = read_u32(r)?;
            if prec == 0 || prec > MAX_FILE_FLOAT_PREC {
                // `Float::with_val(0, _)` panics; a huge one allocates first.
                return Err(IoError::BadNumeric(format!(
                    "float precision {prec} bits is outside 1..={MAX_FILE_FLOAT_PREC}"
                )));
            }
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
            let n = u64::from(read_u32(r)?);
            r.ensure(n, 8)?;
            let mut branches = Vec::with_capacity(n as usize);
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
        13 => {
            if format_version < POOL_FORMAT_V5 {
                return Err(IoError::BadTag(13));
            }
            let poly = ExprId(read_u32(r)?);
            let var = ExprId(read_u32(r)?);
            let body = ExprId(read_u32(r)?);
            Ok(ExprData::RootSum { poly, var, body })
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
    let len = f.metadata()?.len();
    let mut r = Bounded::new(BufReader::new(f), len);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(IoError::BadMagic);
    }

    let version = read_u32(&mut r)?;
    if version != POOL_FORMAT_V1
        && version != POOL_FORMAT_V2
        && version != POOL_FORMAT_V3
        && version != POOL_FORMAT_V4
        && version != POOL_FORMAT_V5
    {
        return Err(IoError::UnsupportedVersion(version));
    }
    let _flags = read_u32(&mut r)?;

    let pool = ExprPool::new();
    let count = read_u64(&mut r)?;
    // Each node takes at least `MIN_NODE_BYTES`, so a count the rest of the
    // file cannot hold is a truncated (or corrupt) file, refused before the
    // loop rather than after it has run out of bytes.
    r.ensure(count, MIN_NODE_BYTES)?;
    if count > u64::from(u32::MAX) {
        return Err(IoError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("pool file declares {count} nodes, more than an ExprId can address"),
        )));
    }
    let count = count as usize;
    // File index → id in the rebuilt pool.  These are equal for a file this
    // build wrote, but not in general: `intern` canonicalises (a `Rational`
    // with denominator 1 becomes an `Integer`, a one-argument `Add` becomes
    // its argument, …), so a node an older build saved in a non-canonical
    // spelling lands on an id that already exists and every later node
    // shifts down.  Reading children through this table keeps each reference
    // pointing at the node the file meant, instead of silently at whichever
    // node now occupies that index.
    let mut ids: Vec<ExprId> = Vec::new();
    for index in 0..count {
        let data = read_node(&mut r, version)?;
        // `Add`/`Mul` go through their constructors, not bare `intern`, so a
        // nested or unsorted node an older build saved is flattened and put
        // in canonical order — otherwise it would sit in the pool beside the
        // canonical node for the same sum as a second id.
        let id = match remap_children(data, &ids, index)? {
            ExprData::Add(args) => pool.add(args),
            ExprData::Mul(args) => pool.mul(args),
            other => pool.intern(other),
        };
        ids.push(id);
    }

    Ok(Some(pool))
}

/// Rewrite every child reference in `data` (a node read at file position
/// `index`) through `ids`, the file-index → pool-id table built so far.
///
/// A child must be an *earlier* node: the writer emits the pool in id order
/// and a node is interned only after its children, so a reference at or past
/// `index` is corrupt data, and following it would read a node that does not
/// exist yet.
fn remap_children(data: ExprData, ids: &[ExprId], index: usize) -> Result<ExprData, IoError> {
    let map = |c: ExprId| -> Result<ExprId, IoError> {
        ids.get(c.0 as usize).copied().ok_or_else(|| {
            IoError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "pool file node {index} refers to node {}, which is not an earlier node",
                    c.0
                ),
            ))
        })
    };
    let map_all =
        |v: Vec<ExprId>| -> Result<Vec<ExprId>, IoError> { v.into_iter().map(map).collect() };
    Ok(match data {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => data,
        ExprData::Add(args) => ExprData::Add(map_all(args)?),
        ExprData::Mul(args) => ExprData::Mul(map_all(args)?),
        ExprData::Pow { base, exp } => ExprData::Pow {
            base: map(base)?,
            exp: map(exp)?,
        },
        ExprData::Func { name, args } => ExprData::Func {
            name,
            args: map_all(args)?,
        },
        ExprData::Piecewise { branches, default } => ExprData::Piecewise {
            branches: branches
                .into_iter()
                .map(|(c, v)| Ok((map(c)?, map(v)?)))
                .collect::<Result<_, IoError>>()?,
            default: map(default)?,
        },
        ExprData::Predicate { kind, args } => ExprData::Predicate {
            kind,
            args: map_all(args)?,
        },
        ExprData::Forall { var, body } => ExprData::Forall {
            var: map(var)?,
            body: map(body)?,
        },
        ExprData::Exists { var, body } => ExprData::Exists {
            var: map(var)?,
            body: map(body)?,
        },
        ExprData::BigO(inner) => ExprData::BigO(map(inner)?),
        ExprData::RootSum { poly, var, body } => ExprData::RootSum {
            poly: map(poly)?,
            var: map(var)?,
            body: map(body)?,
        },
    })
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

    /// A path no other test in this process will pick.
    ///
    /// The timestamp alone is not enough: on Windows the system clock ticks
    /// about every 15 ms, so two of these tests running in parallel read the
    /// same nanosecond count, build the same path, and then race — one sees
    /// the other's pool (`node count must match` off by two) or its
    /// `remove_file` (`NotFound`). The counter makes the name unique within
    /// the process and the pid keeps it unique across processes.
    fn tempfile() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "alkahest_pool_{}_{}_{}.akp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    fn round_trip_root_sum() {
        // RootSum nodes (format V5) must round-trip through checkpoint/restore.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let t = p.symbol("t", Domain::Complex);
        let poly = p.add(vec![p.pow(t, p.integer(2_i32)), p.integer(1_i32)]);
        let body = p.mul(vec![t, p.func("log", vec![p.add(vec![x, t])])]);
        let rs = p.root_sum(poly, t, body);
        assert!(matches!(p.get(rs), ExprData::RootSum { .. }));

        let path = tempfile();
        p.checkpoint(&path).unwrap();
        let q = ExprPool::open_persistent(&path).unwrap();
        assert_eq!(q.len(), p.len());
        for i in 0..p.len() {
            let id = ExprId(i as u32);
            assert_eq!(p.get(id), q.get(id), "node {i} mismatch after round-trip");
        }
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

    /// Write `nodes` as a pool file exactly as given — no canonicalisation —
    /// the way an older build could have.
    fn write_raw(path: &Path, nodes: &[ExprData]) {
        let mut w = BufWriter::new(File::create(path).unwrap());
        w.write_all(MAGIC).unwrap();
        write_u32(&mut w, POOL_FORMAT_WRITE).unwrap();
        write_u32(&mut w, 0).unwrap();
        write_u64(&mut w, nodes.len() as u64).unwrap();
        for n in nodes {
            write_node(&mut w, n).unwrap();
        }
        w.flush().unwrap();
    }

    /// A file holding non-canonical spellings (a `Rational` with denominator
    /// 1 beside the `Integer` it equals, a one-argument `Add`, an unsorted
    /// `Add`) loads onto the canonical nodes, and every later reference still
    /// points at the node the file meant — not at whatever now sits at that
    /// index once the duplicates have collapsed.
    #[test]
    fn non_canonical_nodes_collapse_without_drift() {
        let x = ExprData::Symbol {
            name: "x".into(),
            domain: Domain::Real,
            commutative: true,
        };
        let y = ExprData::Symbol {
            name: "y".into(),
            domain: Domain::Real,
            commutative: true,
        };
        let nodes = vec![
            x,                                                       // 0
            ExprData::Rational(BigRat(rug::Rational::from((4, 2)))), // 1: 2/1
            ExprData::Integer(BigInt(rug::Integer::from(2))),        // 2: 2
            ExprData::Add(vec![ExprId(0)]),                          // 3: (x)
            y,                                                       // 4
            ExprData::Add(vec![ExprId(4), ExprId(3)]),               // 5: y + (x)
            ExprData::Pow {
                base: ExprId(5),
                exp: ExprId(2),
            }, // 6: (y + x)^2
            ExprData::Mul(vec![]),                                   // 7: empty product
            ExprData::Pow {
                base: ExprId(4),
                exp: ExprId(7),
            }, // 8: y^1
        ];
        let path = tempfile();
        write_raw(&path, &nodes);
        let q = load_from(&path).unwrap().unwrap();
        let _ = fs::remove_file(&path);

        let x = q.symbol("x", Domain::Real);
        let y = q.symbol("y", Domain::Real);
        let two = q.integer(2_i32);
        let sum = q.add(vec![x, y]);
        let square = q.pow(sum, two);
        let y_one = q.pow(y, q.integer(1_i32));
        // x, 2, y, x + y, (x + y)^2, 1, y^1 — nothing else, nothing twice.
        assert_eq!(q.len(), 7, "a non-canonical node survived as a duplicate");
        for id in [x, y, two, sum, square, y_one] {
            assert!(
                (id.0 as usize) < 7,
                "{} was not in the loaded pool",
                q.display(id)
            );
        }
    }

    #[test]
    fn a_forward_child_reference_is_refused() {
        let nodes = vec![ExprData::Add(vec![ExprId(0), ExprId(1)])];
        let path = tempfile();
        write_raw(&path, &nodes);
        let r = load_from(&path);
        let _ = fs::remove_file(&path);
        assert!(
            r.is_err(),
            "a child reference past the node itself must be refused"
        );
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

    // -- Corrupt and hostile files (audit B8/B9) ---------------------------

    fn header(count: u64) -> Vec<u8> {
        let mut v = MAGIC.to_vec();
        v.extend_from_slice(&POOL_FORMAT_WRITE.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&count.to_le_bytes());
        v
    }

    fn load_bytes(bytes: &[u8]) -> Result<Option<ExprPool>, IoError> {
        let path = tempfile();
        std::fs::write(&path, bytes).unwrap();
        let r = load_from(&path);
        let _ = fs::remove_file(&path);
        r
    }

    fn le32(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }

    /// Each of these used to abort the process (a length-prefixed field was
    /// allocated at its declared size before the bytes were read) or panic
    /// inside `rug`.  All of them must now come back as an `Err`.
    #[test]
    fn hostile_headers_are_refused_not_allocated() {
        let mut cases: Vec<(&str, Vec<u8>, &str)> = Vec::new();
        // Symbol whose name claims ~4 GiB.
        let mut b = header(1);
        b.extend_from_slice(&[0, 0, 1]);
        b.extend_from_slice(&le32(0xFFFF_FFF0));
        cases.push(("huge string length", b, "E-IO-004"));
        // Add whose arity claims ~4 G children.
        let mut b = header(1);
        b.push(4);
        b.extend_from_slice(&le32(0xFFFF_FFF0));
        cases.push(("huge arity", b, "E-IO-004"));
        // Piecewise whose branch count claims ~4 G pairs.
        let mut b = header(1);
        b.push(8);
        b.extend_from_slice(&le32(0xFFFF_FFF0));
        cases.push(("huge branch count", b, "E-IO-004"));
        // Node count far past what the file holds.
        let mut b = header(1 << 62);
        b.extend_from_slice(&[0, 0, 1]);
        b.extend_from_slice(&le32(1));
        b.push(b'x');
        cases.push(("huge node count", b, "E-IO-004"));
        // Rational with a zero denominator.
        let mut b = header(1);
        b.push(2);
        b.extend_from_slice(&le32(1));
        b.push(b'1');
        b.extend_from_slice(&le32(1));
        b.push(b'0');
        cases.push(("zero denominator", b, "E-IO-009"));
        // Floats at precision 0 and u32::MAX.
        for prec in [0u32, u32::MAX, MAX_FILE_FLOAT_PREC + 1] {
            let mut b = header(1);
            b.push(3);
            b.extend_from_slice(&le32(prec));
            b.extend_from_slice(&le32(1));
            b.push(b'1');
            cases.push(("float precision", b, "E-IO-009"));
        }
        use crate::errors::AlkahestError;
        for (what, bytes, code) in cases {
            let r = std::panic::catch_unwind(|| load_bytes(&bytes))
                .unwrap_or_else(|_| panic!("{what}: loading panicked"));
            match r {
                Err(e) => assert_eq!(e.code(), code, "{what}: {e}"),
                Ok(_) => panic!("{what}: a corrupt file loaded"),
            }
        }
    }

    /// Two copies of the same node in a file must not shift every later
    /// reference by one: `[x, x, y, Pow(1, 2)]` is `x^y`, not `y^?`.
    #[test]
    fn duplicate_nodes_do_not_drift_later_references() {
        let sym = |n: &str| ExprData::Symbol {
            name: n.into(),
            domain: Domain::Real,
            commutative: true,
        };
        let nodes = vec![
            sym("x"),
            sym("x"),
            sym("y"),
            ExprData::Pow {
                base: ExprId(1),
                exp: ExprId(2),
            },
        ];
        let path = tempfile();
        write_raw(&path, &nodes);
        let q = load_from(&path).unwrap().unwrap();
        let _ = fs::remove_file(&path);
        let x = q.symbol("x", Domain::Real);
        let y = q.symbol("y", Domain::Real);
        let before = q.len();
        let _ = q.pow(x, y);
        assert_eq!(q.len(), before, "x^y was not the node the file meant");
        assert_eq!(q.len(), 3);
    }

    /// A pool using every node tag, as bytes.
    fn every_tag_file() -> Vec<u8> {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let t = p.symbol("t", Domain::Complex);
        let big = p.integer(rug::Integer::from(rug::Integer::u_pow_u(10, 30)));
        let r = p.rational(-3, 7);
        let f = p.float(2.5, 80);
        let s = p.add(vec![x, big, r, f]);
        let m = p.mul(vec![x, t]);
        let pw = p.pow(s, m);
        let sn = p.func("sin", vec![pw]);
        let g = p.func("f", vec![x, t, sn]);
        let c = p.pred_gt(x, p.integer(0));
        let pc = p.piecewise(vec![(c, g)], r);
        let fa = p.forall(x, c);
        let ex = p.exists(t, fa);
        let o = p.big_o(p.pow(x, p.integer(3)));
        let poly = p.add(vec![p.pow(t, p.integer(2)), p.integer(1)]);
        let rs = p.root_sum(poly, t, p.func("log", vec![p.add(vec![x, t])]));
        let _ = p.add(vec![pc, ex, o, rs]);
        let path = tempfile();
        p.checkpoint(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = fs::remove_file(&path);
        bytes
    }

    /// Load `bytes`; if it loads, print every node.  Either outcome is fine;
    /// a panic is not.
    fn load_and_touch(bytes: &[u8]) -> bool {
        std::panic::catch_unwind(|| {
            if let Ok(Some(q)) = load_bytes(bytes) {
                for i in 0..q.len() {
                    let _ = q.display(ExprId(i as u32)).to_string();
                }
            }
        })
        .is_ok()
    }

    /// Every truncation and every single-bit flip of a valid file loads as
    /// `Ok` or `Err` — never a panic.  (The Python suite re-runs this in a
    /// subprocess, where an allocation abort would also be caught.)
    #[test]
    fn truncated_and_bit_flipped_files_never_panic() {
        let good = every_tag_file();
        assert!(load_bytes(&good).unwrap().is_some());
        let mut bad = Vec::new();
        for n in 0..good.len() {
            if !load_and_touch(&good[..n]) {
                bad.push(format!("truncated to {n}"));
            }
        }
        for i in 0..good.len() {
            for bit in 0..8 {
                let mut b = good.clone();
                b[i] ^= 1 << bit;
                if !load_and_touch(&b) {
                    bad.push(format!("byte {i} bit {bit}"));
                }
            }
        }
        assert!(bad.is_empty(), "panicked on: {bad:?}");
    }
}
