/// Dependency-free SVG polyline renderer for symbolic expressions.
///
/// Evaluates `expr` in `var` over `[lo, hi]` using the tree-walking interpreter
/// (no JIT required) and emits a standalone `<svg>` element containing a
/// `<polyline>`. NaN / infinite samples are silently skipped so that
/// asymptotes don't corrupt the path.
use crate::jit::eval_interp;
use crate::kernel::{ExprId, ExprPool};
use std::collections::HashMap;

/// Default number of sample points along the x axis.
const DEFAULT_N: usize = 300;

/// Render `expr(var)` over `[lo, hi]` as a standalone SVG string.
///
/// * `width` / `height` — output dimensions in pixels.
/// * `n_pts`   — number of evenly-spaced sample points (default: `300`).
/// * `padding` — pixel margin inside the SVG viewport (default: `10`).
///
/// Returns a UTF-8 SVG string suitable for embedding in HTML, saving as
/// a `.svg` file, or encoding as `data:image/svg+xml;base64,...`.
pub fn render_svg(
    pool: &ExprPool,
    expr: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
    width: u32,
    height: u32,
) -> String {
    render_svg_opts(pool, expr, var, lo, hi, width, height, DEFAULT_N, 10)
}

/// Like [`render_svg`] but exposes sampling density and padding.
#[allow(clippy::too_many_arguments)]
pub fn render_svg_opts(
    pool: &ExprPool,
    expr: ExprId,
    var: ExprId,
    lo: f64,
    hi: f64,
    width: u32,
    height: u32,
    n_pts: usize,
    padding: u32,
) -> String {
    let n = n_pts.max(2);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    let mut env = HashMap::with_capacity(1);
    for i in 0..n {
        let t = lo + (hi - lo) * (i as f64) / ((n - 1) as f64);
        env.insert(var, t);
        if let Some(y) = eval_interp(expr, &env, pool) {
            if y.is_finite() {
                xs.push(t);
                ys.push(y);
            }
        }
    }

    if xs.is_empty() {
        return empty_svg(width, height);
    }

    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = if (y_max - y_min).abs() < 1e-14 {
        1.0
    } else {
        y_max - y_min
    };

    let pad = padding as f64;
    let inner_w = (width as f64) - 2.0 * pad;
    let inner_h = (height as f64) - 2.0 * pad;
    let x_range = hi - lo;

    let points: String = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let px = pad + (x - lo) / x_range * inner_w;
            // SVG y-axis is inverted: y_max maps to top (pad), y_min to bottom.
            let py = pad + (y_max - y) / y_range * inner_h;
            format!("{:.2},{:.2}", px, py)
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#ffffff"/>
  <polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
</svg>"##,
        w = width,
        h = height,
        pts = points,
    )
}

fn empty_svg(width: u32, height: u32) -> String {
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#ffffff"/>
  <text x="{cx}" y="{cy}" font-size="12" fill="#888888" text-anchor="middle">no finite values</text>
</svg>"##,
        w = width,
        h = height,
        cx = width / 2,
        cy = height / 2,
    )
}
