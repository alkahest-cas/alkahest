//! CUDA-accelerated Gröbner basis via Macaulay-matrix row reduction (V1-7).
//!
//! # Algorithm
//!
//! Each iteration of the Buchberger loop produces a batch of S-polynomials.
//! Instead of reducing them one-by-one with polynomial division, we build a
//! **Macaulay matrix** — rows are the S-polynomials plus all basis multiples
//! needed to reduce them (symbolic preprocessing) — then row-reduce that
//! matrix over GF(p) for several 31-bit primes and lift the result back to ℚ
//! via CRT and rational reconstruction.
//!
//! The GPU accelerates the inner row-elimination loop: for each pivot, one
//! CUDA thread per row applies the elimination in parallel. The outer pivot
//! loop (O(n) sequential steps) runs on the CPU.
//!
//! # Feature gate
//!
//! Enabled by `--features groebner-cuda` (implies `groebner` + `cuda`).
//! The `compute_groebner_basis_gpu` entry point is always available; it
//! falls back to pure-Rust row reduction when no CUDA device is found.

use crate::poly::groebner::ideal::GbPoly;
use crate::poly::groebner::monomial_order::MonomialOrder;
use crate::poly::groebner::reduce::s_polynomial;
use rug::{Integer, Rational};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// PTX kernel — one thread per row, eliminates pivot column from that row.
// All arithmetic is mod p where p < 2^31, so factor * val < 2^62 ⊂ u64.
// ---------------------------------------------------------------------------

static ELIMINATE_ROW_PTX: &str = r#"
.version 7.5
.target sm_86
.address_size 64

// eliminate_row_kernel(mat, pivot_row, pivot_col, inv_pivot, prime, n_cols, n_rows)
// Thread gid handles row gid (skips pivot row).
// factor = mat[gid*n_cols+pivot_col] * inv_pivot % prime
// row[c] = (row[c] + prime - factor*piv[c]%prime) % prime  for each c
.visible .entry eliminate_row_kernel(
    .param .u64 p_mat,
    .param .u32 p_pivot_row,
    .param .u32 p_pivot_col,
    .param .u64 p_inv_pivot,
    .param .u64 p_prime,
    .param .u32 p_n_cols,
    .param .u32 p_n_rows
){
    .reg .pred   %pr;
    .reg .u32    %gid, %col, %pivot_row, %pivot_col, %n_cols, %n_rows;
    .reg .u64    %mat, %inv_pivot, %prime, %ncols64;
    .reg .u64    %row_base, %piv_base, %off64;
    .reg .u64    %row_addr, %piv_addr;
    .reg .u64    %factor, %rv, %pv, %prod, %result;

    ld.param.u64  %mat,       [p_mat];
    ld.param.u32  %pivot_row, [p_pivot_row];
    ld.param.u32  %pivot_col, [p_pivot_col];
    ld.param.u64  %inv_pivot, [p_inv_pivot];
    ld.param.u64  %prime,     [p_prime];
    ld.param.u32  %n_cols,    [p_n_cols];
    ld.param.u32  %n_rows,    [p_n_rows];

    .reg .u32 %tid_x, %ctaid_x, %ntid_x;
    mov.u32   %tid_x,   %tid.x;
    mov.u32   %ctaid_x, %ctaid.x;
    mov.u32   %ntid_x,  %ntid.x;
    mad.lo.u32 %gid, %ctaid_x, %ntid_x, %tid_x;

    setp.ge.u32  %pr, %gid, %n_rows;     @%pr bra END;
    setp.eq.u32  %pr, %gid, %pivot_row;  @%pr bra END;

    cvt.u64.u32  %row_base, %gid;
    cvt.u64.u32  %ncols64,  %n_cols;
    mul.lo.u64   %row_base, %row_base, %ncols64;
    shl.b64      %row_base, %row_base, 3;
    add.u64      %row_base, %mat, %row_base;

    cvt.u64.u32  %piv_base, %pivot_row;
    mul.lo.u64   %piv_base, %piv_base, %ncols64;
    shl.b64      %piv_base, %piv_base, 3;
    add.u64      %piv_base, %mat, %piv_base;

    cvt.u64.u32  %off64, %pivot_col;
    shl.b64      %off64, %off64, 3;
    add.u64      %row_addr, %row_base, %off64;
    ld.global.u64 %rv, [%row_addr];
    setp.eq.u64  %pr, %rv, 0;  @%pr bra END;

    mul.lo.u64   %factor, %rv, %inv_pivot;
    rem.u64      %factor, %factor, %prime;

    mov.u32 %col, 0;
LOOP:
    setp.ge.u32  %pr, %col, %n_cols;  @%pr bra END;
    cvt.u64.u32  %off64, %col;
    shl.b64      %off64, %off64, 3;
    add.u64      %row_addr, %row_base, %off64;
    add.u64      %piv_addr, %piv_base, %off64;
    ld.global.u64 %rv, [%row_addr];
    ld.global.u64 %pv, [%piv_addr];
    mul.lo.u64   %prod, %factor, %pv;
    rem.u64      %prod, %prod, %prime;
    add.u64      %result, %rv, %prime;
    sub.u64      %result, %result, %prod;
    rem.u64      %result, %result, %prime;
    st.global.u64 [%row_addr], %result;
    add.u32 %col, %col, 1;
    bra LOOP;
END:
    ret;
}
"#;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from GPU-accelerated Gröbner basis computation.
#[derive(Debug, Clone)]
pub enum GpuGroebnerError {
    /// CUDA device initialisation or kernel launch failed.
    CudaError(String),
    /// CRT/rational-reconstruction step failed (likely too few primes).
    CrtFailed(String),
}

impl std::fmt::Display for GpuGroebnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuGroebnerError::CudaError(s) => write!(f, "CUDA error in Gröbner backend: {s}"),
            GpuGroebnerError::CrtFailed(s) => write!(f, "CRT reconstruction failed: {s}"),
        }
    }
}

impl std::error::Error for GpuGroebnerError {}

impl crate::errors::AlkahestError for GpuGroebnerError {
    fn code(&self) -> &'static str {
        match self {
            GpuGroebnerError::CudaError(_) => "E-SOLVE-010",
            GpuGroebnerError::CrtFailed(_) => "E-SOLVE-011",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            GpuGroebnerError::CudaError(_) => {
                Some("check GPU availability; pass device_id=None to fall back to CPU")
            }
            GpuGroebnerError::CrtFailed(_) => {
                Some("CRT reconstruction failed; try adding more equations or use CPU path")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mod-p arithmetic helpers
// ---------------------------------------------------------------------------

/// Modular exponentiation: base^exp mod m.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Modular inverse: a^{-1} mod p (p must be prime, a != 0 mod p).
fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}

/// Convert a `rug::Rational` coefficient to its representative in `[0, p)`.
fn rational_mod_p(r: &Rational, p: u64) -> u64 {
    let p_int = Integer::from(p);
    // Numerator mod p (non-negative)
    let numer = r.numer();
    let n_abs = Integer::from(numer.abs_ref());
    let n_mod = u64::try_from(n_abs % &p_int).unwrap_or(0);
    let n_mod = if *numer < 0 && n_mod != 0 {
        p - n_mod
    } else {
        n_mod
    };
    // Denominator mod p
    let denom = r.denom();
    let d_mod = u64::try_from(Integer::from(denom % &p_int)).unwrap_or(0);
    if d_mod == 0 {
        return 0; // p divides denom — this prime is "unlucky", caller should skip
    }
    ((n_mod as u128 * mod_inv(d_mod, p) as u128) % p as u128) as u64
}

/// Rational reconstruction: given `a` in `[0, M)`, find `p/q` with
/// `p * q^{-1} ≡ a (mod M)`, `|p| ≤ sqrt(M/2)`, `0 < q ≤ sqrt(M/2)`.
/// Returns `None` if no such fraction exists within the bound.
fn rational_reconstruction(a: &Integer, modulus: &Integer) -> Option<Rational> {
    // Half-GCD / extended Euclidean (Wang 1981)
    let bound = {
        let mut half = modulus.clone();
        half >>= 1u32; // half = modulus / 2
        let mut b = half.sqrt();
        b += 1u32;
        b
    };

    let (mut u0, mut u1) = (modulus.clone(), Integer::ZERO.clone());
    let (mut v0, mut v1) = (a.clone(), Integer::from(1));

    while Integer::from(v0.abs_ref()) > bound {
        let q = Integer::from(&u0 / &v0);
        let new_v0 = Integer::from(&u0 - &q * &v0);
        let new_v1 = Integer::from(&u1 - &q * &v1);
        (u0, u1) = (v0, v1);
        (v0, v1) = (new_v0, new_v1);
    }

    if v1 == 0 || Integer::from(v1.abs_ref()) > bound {
        return None;
    }

    let (p, q) = if v1 < 0 {
        (Integer::from(-&v0), Integer::from(-&v1))
    } else {
        (v0, v1)
    };

    Some(Rational::from((p, q)))
}

// ---------------------------------------------------------------------------
// Dense Macaulay matrix over GF(p)
// ---------------------------------------------------------------------------

/// A dense matrix over GF(p) together with the monomial ordering for columns.
pub struct MacaulayMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// monomials[col] = exponent vector for that column, sorted descending by `order`.
    pub monomials: Vec<Vec<u32>>,
    /// Row-major data: `data[row * n_cols + col]` is the coefficient mod p.
    pub data: Vec<u64>,
    pub n_vars: usize,
    pub order: MonomialOrder,
    pub prime: u64,
}

impl MacaulayMatrix {
    /// Build from a slice of `GbPoly`, reducing coefficients mod `p`.
    /// Returns `None` if `p` divides any denominator (unlucky prime).
    pub fn build(polys: &[GbPoly], order: MonomialOrder, p: u64) -> Option<Self> {
        let n_vars = polys
            .iter()
            .find(|g| !g.is_zero())
            .map(|g| g.n_vars)
            .unwrap_or(0);

        // Collect all monomials
        let mut mon_set: BTreeSet<Vec<u32>> = BTreeSet::new();
        for poly in polys {
            for e in poly.terms.keys() {
                mon_set.insert(e.clone());
            }
        }

        // Sort descending by the given order
        let mut monomials: Vec<Vec<u32>> = mon_set.into_iter().collect();
        monomials.sort_by(|a, b| order.cmp(b, a));

        let col_map: HashMap<Vec<u32>, usize> = monomials
            .iter()
            .enumerate()
            .map(|(i, e)| (e.clone(), i))
            .collect();

        let n_rows = polys.len();
        let n_cols = monomials.len();
        let mut data = vec![0u64; n_rows * n_cols];

        for (row, poly) in polys.iter().enumerate() {
            for (exp, coeff) in &poly.terms {
                let v = rational_mod_p(coeff, p);
                if v == 0 && coeff.denom() != &Integer::from(1u32) {
                    // p divides denominator — unlucky prime
                    return None;
                }
                let col = col_map[exp];
                data[row * n_cols + col] = v;
            }
        }

        Some(MacaulayMatrix {
            n_rows,
            n_cols,
            monomials,
            data,
            n_vars,
            order,
            prime: p,
        })
    }

    /// CPU in-place Gaussian elimination over GF(p) (reduced row echelon form).
    pub fn reduce_cpu(&mut self) {
        let p = self.prime;
        let mut pivot_row = 0usize;

        for col in 0..self.n_cols {
            // Find first nonzero entry in this column at or below pivot_row
            let found = (pivot_row..self.n_rows).find(|&r| self.data[r * self.n_cols + col] != 0);
            let pr = match found {
                Some(r) => r,
                None => continue,
            };

            // Swap pr to pivot_row
            if pr != pivot_row {
                for c in 0..self.n_cols {
                    self.data
                        .swap(pivot_row * self.n_cols + c, pr * self.n_cols + c);
                }
            }

            // Normalise pivot row
            let piv_val = self.data[pivot_row * self.n_cols + col];
            let inv_piv = mod_inv(piv_val, p);
            for c in 0..self.n_cols {
                let v = self.data[pivot_row * self.n_cols + c];
                self.data[pivot_row * self.n_cols + c] =
                    ((v as u128 * inv_piv as u128) % p as u128) as u64;
            }

            // Eliminate column from all other rows
            for r in 0..self.n_rows {
                if r == pivot_row {
                    continue;
                }
                let factor = self.data[r * self.n_cols + col];
                if factor == 0 {
                    continue;
                }
                for c in 0..self.n_cols {
                    let pv = self.data[pivot_row * self.n_cols + c];
                    let rv = self.data[r * self.n_cols + c];
                    let prod = ((factor as u128 * pv as u128) % p as u128) as u64;
                    self.data[r * self.n_cols + c] = (rv + p - prod) % p;
                }
            }

            pivot_row += 1;
        }
    }

    /// GPU in-place reduced row echelon form via PTX kernel.
    /// Falls back to `reduce_cpu` if the CUDA device cannot be opened.
    pub fn reduce_gpu(&mut self, device_id: usize) -> Result<(), GpuGroebnerError> {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::Ptx;

        let ctx = CudaContext::new(device_id)
            .map_err(|e| GpuGroebnerError::CudaError(format!("context {device_id}: {e:?}")))?;

        let module = ctx
            .load_module(Ptx::from_src(ELIMINATE_ROW_PTX))
            .map_err(|e| GpuGroebnerError::CudaError(format!("module load: {e:?}")))?;

        let kernel = module
            .load_function("eliminate_row_kernel")
            .map_err(|e| GpuGroebnerError::CudaError(format!("load_function: {e:?}")))?;

        let stream = ctx.default_stream();

        let mut mat_dev = stream
            .clone_htod(&self.data)
            .map_err(|e| GpuGroebnerError::CudaError(format!("H2D: {e:?}")))?;

        let p = self.prime;
        let n_cols = self.n_cols as u32;
        let n_rows = self.n_rows as u32;
        let mut pivot_row_idx = 0usize;

        for col in 0..self.n_cols {
            // Pivot finding and row swap on CPU (needs a small D2H read)
            let host_col: Vec<u64> = {
                // Copy just the current state of this column
                let host = stream
                    .clone_dtoh(&mat_dev)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("D2H col: {e:?}")))?;
                (0..self.n_rows)
                    .map(|r| host[r * self.n_cols + col])
                    .collect()
            };

            let found = (pivot_row_idx..self.n_rows).find(|&r| host_col[r] != 0);
            let pr = match found {
                Some(r) => r,
                None => continue,
            };

            // Swap rows pr ↔ pivot_row_idx (CPU side, then re-upload)
            if pr != pivot_row_idx {
                let mut host = stream
                    .clone_dtoh(&mat_dev)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("D2H swap: {e:?}")))?;
                for c in 0..self.n_cols {
                    host.swap(pivot_row_idx * self.n_cols + c, pr * self.n_cols + c);
                }
                mat_dev = stream
                    .clone_htod(&host)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("H2D swap: {e:?}")))?;
            }

            // Normalise pivot row (CPU)
            let piv_val = {
                let h = stream
                    .clone_dtoh(&mat_dev)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("D2H piv: {e:?}")))?;
                h[pivot_row_idx * self.n_cols + col]
            };
            let inv_piv = mod_inv(piv_val, p);
            {
                let mut h = stream
                    .clone_dtoh(&mat_dev)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("D2H norm: {e:?}")))?;
                let base = pivot_row_idx * self.n_cols;
                for c in 0..self.n_cols {
                    h[base + c] = ((h[base + c] as u128 * inv_piv as u128) % p as u128) as u64;
                }
                mat_dev = stream
                    .clone_htod(&h)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("H2D norm: {e:?}")))?;
            }

            // GPU: eliminate column from all other rows in parallel
            let block: u32 = 256;
            let grid = ((self.n_rows as u32) + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid.max(1), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let pivot_row_u32 = pivot_row_idx as u32;
            let pivot_col_u32 = col as u32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut mat_dev)
                    .arg(&pivot_row_u32)
                    .arg(&pivot_col_u32)
                    .arg(&inv_piv)
                    .arg(&p)
                    .arg(&n_cols)
                    .arg(&n_rows)
                    .launch(cfg)
                    .map_err(|e| GpuGroebnerError::CudaError(format!("launch: {e:?}")))?;
            }

            pivot_row_idx += 1;
        }

        // Copy reduced matrix back
        stream
            .synchronize()
            .map_err(|e| GpuGroebnerError::CudaError(format!("sync: {e:?}")))?;
        self.data = stream
            .clone_dtoh(&mat_dev)
            .map_err(|e| GpuGroebnerError::CudaError(format!("D2H final: {e:?}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CRT + rational reconstruction
// ---------------------------------------------------------------------------

/// Accumulates per-prime reduced-matrix images and lifts them to GbPoly via CRT.
pub struct CrtLifter {
    n_rows: usize,
    n_cols: usize,
    monomials: Vec<Vec<u32>>,
    n_vars: usize,
    /// Running CRT state per cell: (current integer value, current modulus).
    vals: Vec<(Integer, Integer)>,
}

impl CrtLifter {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        monomials: Vec<Vec<u32>>,
        n_vars: usize,
        _order: MonomialOrder,
    ) -> Self {
        let vals = (0..n_rows * n_cols)
            .map(|_| (Integer::ZERO.clone(), Integer::from(1)))
            .collect();
        CrtLifter {
            n_rows,
            n_cols,
            monomials,
            n_vars,
            vals,
        }
    }

    /// Merge one prime image (reduced matrix mod `p`) into the running CRT state.
    pub fn add_image(&mut self, matrix: &MacaulayMatrix) {
        let p = matrix.prime;
        let p_int = Integer::from(p);
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                let r = matrix.data[row * self.n_cols + col];
                let (cur_val, cur_mod) = &mut self.vals[row * self.n_cols + col];
                // CRT step: find x ≡ cur_val (mod cur_mod) and x ≡ r (mod p)
                // x = cur_val + cur_mod * ((r - cur_val) * inv(cur_mod, p) mod p)
                let diff = {
                    let rv = Integer::from(r);
                    let cv_abs = Integer::from(cur_val.abs_ref());
                    let cv_mod = cv_abs % &p_int;
                    // Adjust sign
                    let cv_mod = if *cur_val < 0 && cv_mod != 0 {
                        Integer::from(&p_int - cv_mod)
                    } else {
                        cv_mod
                    };
                    let mut d = Integer::from(&rv - cv_mod);
                    if d < 0 {
                        d += &p_int;
                    }
                    d
                };
                let inv_m = {
                    let m_abs = Integer::from(cur_mod.abs_ref());
                    let m_mod = m_abs % &p_int;
                    let m_mod_u64 = u64::try_from(&m_mod).unwrap_or(1);
                    mod_inv(m_mod_u64, p)
                };
                let step = Integer::from(&diff * inv_m) % &p_int;
                let add = Integer::from(&*cur_mod * step);
                *cur_val += &add;
                *cur_mod *= &p_int;
                // Reduce to centered representation
                let cur_mod_abs = Integer::from(cur_mod.abs_ref());
                let half = cur_mod_abs / 2u32;
                let cur_val_abs = Integer::from(cur_val.abs_ref());
                if cur_val_abs > half {
                    if *cur_val > 0 {
                        *cur_val -= &*cur_mod;
                    } else {
                        *cur_val += &*cur_mod;
                    }
                }
            }
        }
    }

    /// Attempt rational reconstruction of every cell. Returns `None` if any
    /// cell fails (caller should add more primes and retry).
    pub fn reconstruct(&self) -> Option<Vec<GbPoly>> {
        let modulus = if self.vals.is_empty() {
            return Some(vec![]);
        } else {
            self.vals[0].1.clone()
        };

        let mut polys = Vec::with_capacity(self.n_rows);
        for row in 0..self.n_rows {
            let mut poly = GbPoly::zero(self.n_vars);
            for col in 0..self.n_cols {
                let (val, _) = &self.vals[row * self.n_cols + col];
                if *val == 0 {
                    continue;
                }
                let r = rational_reconstruction(val, &modulus)?;
                poly.terms.insert(self.monomials[col].clone(), r);
            }
            polys.push(poly);
        }
        Some(polys)
    }
}

// ---------------------------------------------------------------------------
// Symbolic preprocessing
// ---------------------------------------------------------------------------

/// F4 symbolic preprocessing: build the set of basis multiples (the "upper"
/// part of the Macaulay matrix) needed to fully reduce `targets` (the "lower"
/// part).
///
/// Returns `(upper, basis_lms)` where:
/// - `upper` is the list of shifted basis elements that serve as reductors,
/// - `basis_lms` is the set of their leading monomials (used after RREF to
///    identify which rows are "new" vs. already-basis-covered).
///
/// Termination: each iteration adds one monomial to `covered` (LMs of upper
/// rows). All new terms introduced have total degree ≤ the monomial being
/// covered (shift × basis, lower-degree terms). Since no new monomial can
/// exceed the max total degree of the original targets, the set is finite.
fn symbolic_preprocess_upper(
    targets: &[GbPoly],
    basis: &[GbPoly],
    order: MonomialOrder,
) -> (Vec<GbPoly>, HashSet<Vec<u32>>) {
    // LMs of the upper rows we've already added (the "done" set).
    let mut covered: HashSet<Vec<u32>> = HashSet::new();

    // Monomials appearing in any row (targets + upper) that might need reduction.
    let mut todo: HashSet<Vec<u32>> = targets
        .iter()
        .flat_map(|p| p.terms.keys().cloned())
        .collect();

    let mut upper: Vec<GbPoly> = Vec::new();

    loop {
        // Find a monomial that is NOT covered by any upper row's LM
        // AND is divisible by some basis element's LM.
        let m = todo
            .iter()
            .filter(|m| !covered.contains(*m))
            .find(|m| {
                basis.iter().any(|g| {
                    g.leading_exp(order).map_or(false, |lm| {
                        lm.len() == m.len() && lm.iter().zip(m.iter()).all(|(a, b)| a <= b)
                    })
                })
            })
            .cloned();

        let m = match m {
            Some(m) => m,
            None => break,
        };

        // Pick the first basis element whose LM divides m.
        let g = basis
            .iter()
            .find(|g| {
                g.leading_exp(order).map_or(false, |lm| {
                    lm.len() == m.len() && lm.iter().zip(m.iter()).all(|(a, b)| a <= b)
                })
            })
            .unwrap();

        let lm = g.leading_exp(order).unwrap();
        let shift: Vec<u32> = m.iter().zip(lm.iter()).map(|(a, b)| a - b).collect();
        let shifted = g.mul_monomial(&shift, &Rational::from(1));

        covered.insert(m); // m = LM of this upper row
        for e in shifted.terms.keys() {
            todo.insert(e.clone());
        }
        upper.push(shifted);
    }

    (upper, covered)
}

// ---------------------------------------------------------------------------
// Fixed 31-bit primes for CRT
// ---------------------------------------------------------------------------

const PRIMES: &[u64] = &[
    2_147_483_647, // 2^31 - 1 (Mersenne prime)
    2_147_483_629,
    2_147_483_587,
    2_147_483_579,
    2_147_483_563,
    2_147_483_549,
    2_147_483_543,
    2_147_483_477,
    2_147_483_423,
    2_147_483_399,
    2_147_483_353,
    2_147_483_323,
    2_147_483_269,
    2_147_483_249,
    2_147_483_237,
];

// ---------------------------------------------------------------------------
// Batch reduction
// ---------------------------------------------------------------------------

/// Reduce `targets` (a batch of S-polynomials) by `basis` using GPU-accelerated
/// Macaulay-matrix row reduction with CRT rational reconstruction.
///
/// Returns the non-zero reduced forms (remainders), equivalent to calling
/// `reduce(sp, basis, order)` for each sp individually.
///
/// `device_id` controls which CUDA device to use. Pass `None` to force CPU.
pub fn reduce_batch(
    targets: &[GbPoly],
    basis: &[GbPoly],
    order: MonomialOrder,
    device_id: Option<usize>,
) -> Result<Vec<GbPoly>, GpuGroebnerError> {
    if targets.is_empty() {
        return Ok(vec![]);
    }

    // Build upper (basis multiples) + lower (targets) parts of the Macaulay matrix.
    // `basis_lms` = leading monomials of upper rows; after RREF, rows with LM
    // in this set are "basis" rows and should be discarded. Rows with LM NOT
    // in this set are the reduced remainders of the targets.
    let (upper, basis_lms) = symbolic_preprocess_upper(targets, basis, order);

    // Full row list: upper rows first, then target rows.
    let _n_upper = upper.len();
    let mut all_rows: Vec<GbPoly> = upper;
    all_rows.extend_from_slice(targets);
    let n_rows = all_rows.len();

    if n_rows == 0 || all_rows[0].n_vars == 0 {
        return Ok(targets.iter().filter(|p| !p.is_zero()).cloned().collect());
    }

    let n_vars = all_rows[0].n_vars;

    // Build the shared column layout once.
    let mut mon_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    for poly in &all_rows {
        for e in poly.terms.keys() {
            mon_set.insert(e.clone());
        }
    }
    let mut monomials: Vec<Vec<u32>> = mon_set.into_iter().collect();
    monomials.sort_by(|a, b| order.cmp(b, a));

    let n_cols = monomials.len();
    let mut lifter = CrtLifter::new(n_rows, n_cols, monomials.clone(), n_vars, order);
    let mut prev: Option<Vec<GbPoly>> = None;
    let mut prime_count = 0usize;

    for &p in PRIMES {
        let mat = match MacaulayMatrix::build(&all_rows, order, p) {
            Some(m) => m,
            None => continue, // unlucky prime
        };
        let mut mat = mat;

        if let Some(dev) = device_id {
            match mat.reduce_gpu(dev) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("alkahest: GPU row reduction failed ({e}), using CPU fallback");
                    mat.reduce_cpu();
                }
            }
        } else {
            mat.reduce_cpu();
        }

        lifter.add_image(&mat);
        prime_count += 1;

        if prime_count >= 3 {
            if let Some(reconstructed) = lifter.reconstruct() {
                // After RREF, extract rows whose LM is NOT a basis LM.
                // These are the reduced S-poly remainders.
                let new_elements: Vec<GbPoly> = reconstructed
                    .into_iter()
                    .filter(|p| {
                        if p.is_zero() {
                            return false;
                        }
                        let lm = p.leading_exp(order);
                        lm.map_or(false, |l| !basis_lms.contains(&l))
                    })
                    .map(|p| p.make_monic(order))
                    .collect();

                let stable = match &prev {
                    None => false,
                    Some(prev_elems) => {
                        new_elements.len() == prev_elems.len()
                            && new_elements
                                .iter()
                                .zip(prev_elems.iter())
                                .all(|(a, b)| a.terms == b.terms)
                    }
                };
                if stable || prime_count >= PRIMES.len() {
                    let mut result = new_elements;
                    result.dedup_by(|a, b| a.terms == b.terms);
                    return Ok(result);
                }
                prev = Some(new_elements);
            }
        }
    }

    Err(GpuGroebnerError::CrtFailed(format!(
        "could not reconstruct after {} primes",
        prime_count
    )))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute a Gröbner basis using GPU-accelerated Macaulay-matrix row reduction.
///
/// The Buchberger loop structure mirrors `compute_buchberger_basis` in `buchberger.rs`;
/// the S-polynomial reduction step is replaced by batched GPU row reduction
/// with CRT rational reconstruction.
///
/// `device_id` is the CUDA device ordinal (0-indexed). Pass `None` to run
/// entirely on CPU (useful for testing correctness without a GPU).
pub fn compute_groebner_basis_gpu(
    generators: Vec<GbPoly>,
    order: MonomialOrder,
    device_id: Option<usize>,
) -> Result<Vec<GbPoly>, GpuGroebnerError> {
    let mut basis: Vec<GbPoly> = generators
        .into_iter()
        .filter(|g| !g.is_zero())
        .map(|g| g.make_monic(order))
        .collect();

    if basis.is_empty() {
        return Ok(basis);
    }

    let mut pairs: Vec<(usize, usize)> = vec![];
    for i in 0..basis.len() {
        for j in (i + 1)..basis.len() {
            if !product_criterion(&basis[i], &basis[j], order) {
                pairs.push((i, j));
            }
        }
    }

    while !pairs.is_empty() {
        let s_polys: Vec<GbPoly> = pairs
            .iter()
            .map(|&(i, j)| s_polynomial(&basis[i], &basis[j], order))
            .filter(|sp| !sp.is_zero())
            .collect();

        if !s_polys.is_empty() {
            let reduced = reduce_batch(&s_polys, &basis, order, device_id)?;
            let new_start = basis.len();
            for r in reduced {
                if !r.is_zero() {
                    basis.push(r.make_monic(order));
                }
            }

            pairs.clear();
            for i in 0..basis.len() {
                for j in (i + 1)..basis.len() {
                    if !product_criterion(&basis[i], &basis[j], order) {
                        if i >= new_start || j >= new_start {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        } else {
            pairs.clear();
        }
    }

    Ok(interreduce_gpu(basis, order, device_id)?)
}

fn product_criterion(f: &GbPoly, g: &GbPoly, order: MonomialOrder) -> bool {
    let lf = match f.leading_exp(order) {
        Some(e) => e,
        None => return true,
    };
    let lg = match g.leading_exp(order) {
        Some(e) => e,
        None => return true,
    };
    lf.iter().zip(lg.iter()).all(|(&a, &b)| a == 0 || b == 0)
}

fn interreduce_gpu(
    mut basis: Vec<GbPoly>,
    order: MonomialOrder,
    device_id: Option<usize>,
) -> Result<Vec<GbPoly>, GpuGroebnerError> {
    let mut i = 0;
    while i < basis.len() {
        let others: Vec<GbPoly> = basis
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, g)| g.clone())
            .collect();
        let reduced = reduce_batch(&[basis[i].clone()], &others, order, device_id)?;
        let r = reduced
            .into_iter()
            .next()
            .unwrap_or_else(|| GbPoly::zero(basis[i].n_vars));
        if r.is_zero() {
            basis.remove(i);
        } else {
            basis[i] = r.make_monic(order);
            i += 1;
        }
    }
    Ok(basis)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::groebner::reduce::reduce as cpu_reduce;

    fn rat(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    fn poly1(terms: &[(&[u32], i64)]) -> GbPoly {
        let n_vars = terms.first().map(|(e, _)| e.len()).unwrap_or(1);
        GbPoly {
            terms: terms
                .iter()
                .map(|(e, c)| (e.to_vec(), rat(*c, 1)))
                .collect(),
            n_vars,
        }
    }

    #[test]
    fn mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 7), 1);
    }

    #[test]
    fn mod_inv_basic() {
        let p = 7u64;
        for a in 1..p {
            let inv = mod_inv(a, p);
            assert_eq!((a * inv) % p, 1, "inv({a}, {p}) = {inv}");
        }
    }

    #[test]
    fn rational_mod_p_basic() {
        let r = Rational::from((3, 2));
        let p = 7u64;
        // 3/2 mod 7 = 3 * inv(2,7) = 3 * 4 = 12 mod 7 = 5
        assert_eq!(rational_mod_p(&r, p), 5);
    }

    #[test]
    fn rational_reconstruction_basic() {
        // 5/1: we should get 5
        let m = Integer::from(1_000_000_007u64);
        let a = Integer::from(5u32);
        let r = rational_reconstruction(&a, &m).unwrap();
        assert_eq!(r, Rational::from(5));
    }

    #[test]
    fn macaulay_matrix_build_and_reduce_cpu() {
        // Two polynomials: x^2 - 1 and x - 1 in 1 variable
        let f = poly1(&[(&[2], 1), (&[0], -1)]); // x^2 - 1
        let g = poly1(&[(&[1], 1), (&[0], -1)]); // x - 1
        let p = 2_147_483_647u64;
        let mut mat = MacaulayMatrix::build(&[f, g], MonomialOrder::Lex, p).unwrap();
        assert_eq!(mat.n_rows, 2);
        assert_eq!(mat.n_cols, 3); // {x^2, x, 1}
        mat.reduce_cpu();
        // After reduction the two rows should be in RREF: {x^2: row 0, x: row 1}
        // Check that both rows are non-zero
        let r0_nz = (0..mat.n_cols).any(|c| mat.data[c] != 0);
        let r1_nz = (0..mat.n_cols).any(|c| mat.data[mat.n_cols + c] != 0);
        assert!(r0_nz && r1_nz);
    }

    #[test]
    fn crt_lifter_roundtrip() {
        // Single polynomial 3x + 2 in 1 variable, verify CRT round-trips
        let poly = poly1(&[(&[1], 3), (&[0], 2)]);
        let order = MonomialOrder::Lex;

        let mut mon_set: BTreeSet<Vec<u32>> = BTreeSet::new();
        for e in poly.terms.keys() {
            mon_set.insert(e.clone());
        }
        let mut monomials: Vec<Vec<u32>> = mon_set.into_iter().collect();
        monomials.sort_by(|a, b| order.cmp(b, a));

        let n_cols = monomials.len();
        let mut lifter = CrtLifter::new(1, n_cols, monomials, 1, order);

        for &p in &PRIMES[..5] {
            let mat = MacaulayMatrix::build(&[poly.clone()], order, p).unwrap();
            lifter.add_image(&mat);
        }

        let result = lifter.reconstruct().expect("reconstruction failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].terms.get(&vec![1u32]), Some(&Rational::from(3)));
        assert_eq!(result[0].terms.get(&vec![0u32]), Some(&Rational::from(2)));
    }

    #[test]
    fn groebner_gpu_linear_system_cpu_path() {
        // (x + y - 1, x - y) → basis contains x - 1/2 and y - 1/2
        let f = poly1(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let g = poly1(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let basis =
            compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
                .expect("gpu groebner failed");
        assert!(!basis.is_empty());
        // Verify: original generators reduce to zero mod the computed basis
        let rf = cpu_reduce(&f, &basis, MonomialOrder::Lex);
        let rg = cpu_reduce(&g, &basis, MonomialOrder::Lex);
        assert!(rf.is_zero(), "f not in ideal: {rf:?}");
        assert!(rg.is_zero(), "g not in ideal: {rg:?}");
    }

    #[test]
    fn groebner_gpu_x_squared_minus_1_cpu_path() {
        // (x^2 - 1, x - 1) → {x - 1}
        let f = poly1(&[(&[2], 1), (&[0], -1)]);
        let g = poly1(&[(&[1], 1), (&[0], -1)]);
        let basis =
            compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
                .expect("gpu groebner failed");
        assert_eq!(basis.len(), 1);
        let rf = cpu_reduce(&f, &basis, MonomialOrder::Lex);
        let rg = cpu_reduce(&g, &basis, MonomialOrder::Lex);
        assert!(rf.is_zero());
        assert!(rg.is_zero());
    }

    #[test]
    fn groebner_gpu_circle_line_cpu_path() {
        // x^2 + y^2 - 1 = 0, y - x = 0 → solutions (±√2/2, ±√2/2)
        let f = poly1(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]);
        let g = poly1(&[(&[0, 1], 1), (&[1, 0], -1)]);
        let basis =
            compute_groebner_basis_gpu(vec![f.clone(), g.clone()], MonomialOrder::Lex, None)
                .expect("gpu groebner failed");
        assert!(!basis.is_empty());
        let rf = cpu_reduce(&f, &basis, MonomialOrder::Lex);
        let rg = cpu_reduce(&g, &basis, MonomialOrder::Lex);
        assert!(rf.is_zero(), "circle not in ideal");
        assert!(rg.is_zero(), "line not in ideal");
    }

    #[test]
    fn groebner_gpu_matches_cpu_f4() {
        use crate::poly::groebner::buchberger::compute_buchberger_basis;
        // (x + y - 1, x - y) — compare GPU (CPU-fallback) vs pure-Rust Buchberger
        let f = poly1(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]);
        let g = poly1(&[(&[1, 0], 1), (&[0, 1], -1)]);
        let order = MonomialOrder::Lex;

        let basis_gpu =
            compute_groebner_basis_gpu(vec![f.clone(), g.clone()], order, None).unwrap();
        let basis_cpu = compute_buchberger_basis(vec![f.clone(), g.clone()], order);

        // Both bases are Gröbner — each element of one should reduce to 0 mod the other
        for p in &basis_cpu {
            let r = cpu_reduce(p, &basis_gpu, order);
            assert!(r.is_zero(), "CPU basis element not in GPU basis: {p:?}");
        }
        for p in &basis_gpu {
            let r = cpu_reduce(p, &basis_cpu, order);
            assert!(r.is_zero(), "GPU basis element not in CPU basis: {p:?}");
        }
    }
}
