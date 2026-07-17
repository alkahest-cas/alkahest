//! Bounded Merkle / content-addressed expression pool prototype (RFC 0001).
//!
//! Design note: `docs/rfcs/0001-incremental-merkle-pool.md`.
//!
//! This module is **experimental**: it does not replace [`crate::ExprPool`], does
//! not integrate with Salsa or egglog, and is covered by unit tests only.
//!
//! # Design
//!
//! Each node is keyed by a stable content digest of its tag, payload, and
//! child hashes (Merkle-style). Lookup is by digest. The digest algorithm is
//! a versioned FNV-1a 128-bit fold into 16 bytes — enough for tests and local
//! spikes, not a cryptographic CID.

use crate::kernel::domain::Domain;
use std::collections::HashMap;
use std::fmt;

/// Content address of a Merkle expression node (16-byte digest).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprHash([u8; 16]);

impl ExprHash {
    /// Raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl fmt::Debug for ExprHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExprHash(")?;
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for ExprHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Structural payload stored under an [`ExprHash`].
///
/// Deliberately a small subset of [`crate::kernel::ExprData`] so the prototype
/// stays bounded. Children are content hashes, not arena ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleNode {
    Symbol {
        name: String,
        domain: Domain,
    },
    /// Small integers only (prototype); arbitrary-precision can map later.
    Integer(i64),
    Add(Vec<ExprHash>),
    Mul(Vec<ExprHash>),
    Pow {
        base: ExprHash,
        exp: ExprHash,
    },
    Func {
        name: String,
        args: Vec<ExprHash>,
    },
}

/// Content-addressed expression store.
///
/// Interning is append-only: identical structure always yields the same hash
/// and a single stored node.
#[derive(Debug, Default)]
pub struct MerklePool {
    nodes: HashMap<ExprHash, MerkleNode>,
}

impl MerklePool {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Number of distinct interned nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up a node by content hash.
    pub fn lookup(&self, hash: &ExprHash) -> Option<&MerkleNode> {
        self.nodes.get(hash)
    }

    /// Whether `hash` is present.
    pub fn contains(&self, hash: &ExprHash) -> bool {
        self.nodes.contains_key(hash)
    }

    /// Intern `node`, returning its content hash (shared if already present).
    pub fn intern(&mut self, node: MerkleNode) -> ExprHash {
        let hash = hash_node(&node);
        self.nodes.entry(hash).or_insert(node);
        hash
    }

    // -- Convenience constructors (mirror a tiny slice of ExprPool) -----------

    pub fn symbol(&mut self, name: impl Into<String>, domain: Domain) -> ExprHash {
        self.intern(MerkleNode::Symbol {
            name: name.into(),
            domain,
        })
    }

    pub fn integer(&mut self, n: i64) -> ExprHash {
        self.intern(MerkleNode::Integer(n))
    }

    pub fn add(&mut self, args: Vec<ExprHash>) -> ExprHash {
        self.intern(MerkleNode::Add(args))
    }

    pub fn mul(&mut self, args: Vec<ExprHash>) -> ExprHash {
        self.intern(MerkleNode::Mul(args))
    }

    pub fn pow(&mut self, base: ExprHash, exp: ExprHash) -> ExprHash {
        self.intern(MerkleNode::Pow { base, exp })
    }

    pub fn func(&mut self, name: impl Into<String>, args: Vec<ExprHash>) -> ExprHash {
        self.intern(MerkleNode::Func {
            name: name.into(),
            args,
        })
    }
}

// ---------------------------------------------------------------------------
// Stable content digest (FNV-1a 128-bit, version-tagged)
// ---------------------------------------------------------------------------

const DIGEST_VERSION: u8 = 1;

const FNV_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV_PRIME: u128 = 0x0000000001000000000000000000013B;

fn fnv1a_128(data: &[u8]) -> u128 {
    let mut hash = FNV_OFFSET;
    for &b in data {
        hash ^= u128::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_node(node: &MerkleNode) -> ExprHash {
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    buf.push(DIGEST_VERSION);
    match node {
        MerkleNode::Symbol { name, domain } => {
            buf.push(0);
            buf.push(domain_tag(*domain));
            write_str(&mut buf, name);
        }
        MerkleNode::Integer(n) => {
            buf.push(1);
            buf.extend_from_slice(&n.to_le_bytes());
        }
        MerkleNode::Add(args) => {
            buf.push(2);
            write_hashes(&mut buf, args);
        }
        MerkleNode::Mul(args) => {
            buf.push(3);
            write_hashes(&mut buf, args);
        }
        MerkleNode::Pow { base, exp } => {
            buf.push(4);
            buf.extend_from_slice(base.as_bytes());
            buf.extend_from_slice(exp.as_bytes());
        }
        MerkleNode::Func { name, args } => {
            buf.push(5);
            write_str(&mut buf, name);
            write_hashes(&mut buf, args);
        }
    }
    let h = fnv1a_128(&buf);
    ExprHash(h.to_le_bytes())
}

fn domain_tag(d: Domain) -> u8 {
    match d {
        Domain::Real => 0,
        Domain::Complex => 1,
        Domain::Integer => 2,
        Domain::Positive => 3,
        Domain::NonNegative => 4,
        Domain::NonZero => 5,
    }
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_hashes(buf: &mut Vec<u8>, hashes: &[ExprHash]) {
    buf.extend_from_slice(&(hashes.len() as u32).to_le_bytes());
    for h in hashes {
        buf.extend_from_slice(h.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_intern_and_lookup() {
        let mut pool = MerklePool::new();
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(pool.len(), 1);
        assert!(pool.contains(&x));
        match pool.lookup(&x) {
            Some(MerkleNode::Symbol { name, domain }) => {
                assert_eq!(name, "x");
                assert_eq!(*domain, Domain::Real);
            }
            other => panic!("expected Symbol, got {other:?}"),
        }
        // Re-intern is a hit.
        let x2 = pool.symbol("x", Domain::Real);
        assert_eq!(x, x2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn domain_distinguishes_symbols() {
        let mut pool = MerklePool::new();
        let a = pool.symbol("x", Domain::Real);
        let b = pool.symbol("x", Domain::Complex);
        assert_ne!(a, b);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn dag_sharing_via_child_hashes() {
        let mut pool = MerklePool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2);
        let x2 = pool.pow(x, two);
        // (x^2) + (x^2) shares the pow node by hash.
        let sum = pool.add(vec![x2, x2]);
        assert_eq!(pool.len(), 4); // x, 2, pow, add
        match pool.lookup(&sum) {
            Some(MerkleNode::Add(args)) => {
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], x2);
                assert_eq!(args[1], x2);
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn identical_dags_same_root_hash_across_pools() {
        fn build() -> (MerklePool, ExprHash) {
            let mut p = MerklePool::new();
            let x = p.symbol("x", Domain::Real);
            let y = p.symbol("y", Domain::Real);
            let s = p.func("sin", vec![x]);
            let root = p.add(vec![s, y]);
            (p, root)
        }
        let (p1, h1) = build();
        let (p2, h2) = build();
        assert_eq!(h1, h2);
        assert_eq!(p1.lookup(&h1), p2.lookup(&h2));
    }

    #[test]
    fn structural_change_changes_hash() {
        let mut pool = MerklePool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let one = pool.integer(1);
        let a = pool.add(vec![x, one]);
        let b = pool.add(vec![y, one]);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_pow_func_roundtrip() {
        let mut pool = MerklePool::new();
        let x = pool.symbol("x", Domain::Real);
        let three = pool.integer(3);
        let prod = pool.mul(vec![x, three]);
        let f = pool.func("log", vec![prod]);
        assert!(matches!(
            pool.lookup(&f),
            Some(MerkleNode::Func { name, args }) if name == "log" && args == &[prod]
        ));
    }

    #[test]
    fn missing_hash_lookup_is_none() {
        let pool = MerklePool::new();
        let ghost = ExprHash([0u8; 16]);
        assert!(pool.lookup(&ghost).is_none());
        assert!(!pool.contains(&ghost));
    }
}
