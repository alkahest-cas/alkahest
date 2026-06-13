"""Demo: acausal RC-circuit modeling + Laplace transform (experimental).

Part 1 — Acausal (Modelica-like) modeling
------------------------------------------
Builds a simple RC circuit (ideal voltage source, resistor, capacitor) by
creating components, registering them on an :class:`alkahest.AcausalSystem`,
wiring up the ports with :meth:`AcausalSystem.connect`, and flattening the
network into a :class:`alkahest.DAE`.

Part 2 — Laplace transform (experimental)
------------------------------------------
Shows that ``laplace_transform`` / ``inverse_laplace_transform`` are
discoverable from ``alkahest.experimental`` and compute textbook results:

    L{1}(s)        = 1/s
    L{exp(-a t)}(s) = 1/(s + a)

Run with::

    PYTHONPATH=python python examples/acausal_and_laplace.py
"""

from __future__ import annotations

import alkahest as ak
from alkahest.experimental import inverse_laplace_transform, laplace_transform


def part1_acausal_rc_circuit() -> None:
    print("=" * 70)
    print("Part 1: Acausal RC circuit")
    print("=" * 70)

    pool = ak.ExprPool()
    t = pool.symbol("t")

    # Component constructors return {"name", "n_equations", "n_ports",
    # "component"}; "component" is an alkahest.Component.
    src = ak.voltage_source("V1", pool.symbol("Vs"))
    res = ak.resistor("R1", pool.symbol("R"))
    cap = ak.capacitor("C1", pool.symbol("C"))

    src_comp = src["component"]
    res_comp = res["component"]
    cap_comp = cap["component"]

    print(f"voltage_source -> {src_comp!r}")
    print(f"resistor       -> {res_comp!r}")
    print(f"capacitor      -> {cap_comp!r}")

    sys = ak.AcausalSystem(pool)
    sys.add_component(src_comp)
    sys.add_component(res_comp)
    sys.add_component(cap_comp)

    # Wire: Vs.p -> R1.p, R1.n -> C1.p, C1.n -> Vs.n (ground loop)
    sys.connect(src_comp.port("V1.p"), res_comp.port("R1.p"))
    sys.connect(res_comp.port("R1.n"), cap_comp.port("C1.p"))
    sys.connect(cap_comp.port("C1.n"), src_comp.port("V1.n"))

    dae = sys.flatten(t)

    print(f"\nFlattened DAE: {dae.n_equations()} equations, "
          f"{dae.n_variables()} variables")
    print(dae)


def part2_laplace_transform() -> None:
    print()
    print("=" * 70)
    print("Part 2: Laplace transform (alkahest.experimental)")
    print("=" * 70)

    pool = ak.ExprPool()
    t = pool.symbol("t")
    s = pool.symbol("s")
    a = pool.symbol("a")

    one = pool.integer(1)
    L_one = laplace_transform(one, t, s)
    print(f"L{{1}}(s)          = {L_one}")  # expect 1/s

    exp_neg_at = ak.exp(pool.mul([pool.integer(-1), a, t]))
    L_exp = laplace_transform(exp_neg_at, t, s)
    print(f"L{{exp(-a t)}}(s)  = {L_exp}")  # expect 1/(s + a)

    # Round trip: inverse Laplace of 1/s should give back 1 (Heaviside(t)).
    inv = inverse_laplace_transform(L_one, s, t)
    print(f"L^-1{{1/s}}(t)     = {inv}")


if __name__ == "__main__":
    part1_acausal_rc_circuit()
    part2_laplace_transform()
