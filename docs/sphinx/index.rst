Alkahest — Python API reference
================================

Installation
------------

**Python** — install from PyPI. Supported versions: **3.10 through 3.14**:

::

   python -m pip install -U pip
   pip install alkahest

That wheel includes the Gröbner solver (``groebner``), egglog (``egraph``), the Cranelift JIT
and the ``parallel`` multi-core paths, so ``alkahest.solve`` and related APIs work out of the
box. It omits the LLVM JIT and ``cuda``. There is **no** pip extra
that swaps in a different native binary.

NumPy is not installed with ``alkahest``. The experimental ``alkahest.fex`` module
imports it at import time, so install it first (``pip install numpy``) or
``import alkahest.fex`` fails with ``ModuleNotFoundError``.

For **Linux x86_64** opt-in wheels with local versions ``+jit`` or ``+full``, use assets attached to
`GitHub Releases <https://github.com/alkahest-cas/alkahest/releases>`_. Other platforms: build from
source with ``maturin``.

**Rust** — add ``alkahest-cas`` to your ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   alkahest-cas = "3"
   # alkahest-cas = { version = "3", features = ["groebner", "parallel", "egraph"] }

Requires ``libflint-dev`` / ``libgmp-dev`` / ``libmpfr-dev`` at build time (``apt-get`` or
``brew install flint``). The opt-in ``system-gmp`` feature (GMP ≥ 6.3 and MPFR ≥ 4.2) links rug
against the same GMP as FLINT instead of a bundled copy. See `docs.rs/alkahest-cas <https://docs.rs/alkahest-cas>`_ for the
full Rust API reference.

For optional Cargo features (``jit``, ``parallel``, ``cuda``, …) and full developer setup, see the
`Getting started <../getting-started.html>`_ chapter of the user guide. The ``groebner`` and
``egraph`` features are defaults and do not need to be specified explicitly.

.. toctree::
   :maxdepth: 2
   :caption: Core

   api/core
   api/simplify
   api/diff
   api/poly
   api/numerics
   api/transform
   api/pari
   api/units
   api/tensor
   api/controls
   api/orthopoly
   api/noncommutative
   api/specfun

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   api/matrix
   api/ode
   api/solve
   api/linprog
   api/logic
   api/codegen
   api/workload
   api/autoresearch
   api/errors

For the conceptual guide (kernel design, rule engine, e-graph, derivation logs)
see the `mdBook user guide <../>`_.
