"""Draw the alkahest logo: Legendre polynomials P2..P5 on [-1, 1], evaluated by alkahest.

Writes website/src/logo.svg. Run with an environment that has alkahest installed:
    python website/scripts/make_logo.py
"""

from pathlib import Path

import alkahest as ak

COLORS = ["#c41e3a", "#2563eb", "#0f8a6c", "#d97706"]  # P2, P3, P4, P5
SIZE, PAD, SAMPLES = 64, 5, 97
X0 = 0.9  # trim the ends, where every P_n runs off to ±1 together


def samples(n: int) -> list[tuple[float, float]]:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.compile_expr(ak.legendre_p(n, x), [x])
    ts = [-X0 + 2 * X0 * i / (SAMPLES - 1) for i in range(SAMPLES)]
    return [(t, f([t])) for t in ts]


def path_for(pts: list[tuple[float, float]], lo: float, hi: float) -> str:
    span = SIZE - 2 * PAD
    xy = [(PAD + (t + X0) / (2 * X0) * span, PAD + (hi - y) / (hi - lo) * span) for t, y in pts]
    return "M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in xy)


def main() -> None:
    curves = {n: samples(n) for n in range(2, 6)}
    ys = [y for pts in curves.values() for _, y in pts]
    lo, hi = min(ys), max(ys)
    paths = "\n".join(
        f'  <path d="{path_for(curves[n], lo, hi)}" stroke="{c}"/>'
        for n, c in zip(range(2, 6), COLORS)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" fill="none" '
        'stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">\n'
        "  <title>alkahest</title>\n"
        f"{paths}\n</svg>\n"
    )
    out = Path(__file__).resolve().parents[1] / "src" / "logo.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
