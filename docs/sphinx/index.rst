Alkahest — Python API reference
================================

Installation
------------

**Python** — install from PyPI. Supported versions: **3.9 through 3.13**:

::

   python -m pip install -U pip
   pip install alkahest

That wheel omits LLVM JIT and the optional Rust features ``groebner``, ``egraph``, and ``parallel``
(the interpreter still handles numeric APIs). There is **no** pip extra that swaps in a different
native binary.

For **Linux x86_64** opt-in wheels with local versions ``+jit`` or ``+full``, use assets attached to
`GitHub Releases <https://github.com/alkahest-cas/alkahest/releases>`_. Other platforms: build from
source with ``maturin``.

**Rust** — add ``alkahest-cas`` to your ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   alkahest-cas = "2"
   # alkahest-cas = { version = "2", features = ["groebner", "parallel", "egraph"] }

Requires ``libflint-dev`` / ``libgmp-dev`` / ``libmpfr-dev`` at build time (``apt-get`` or
``brew install flint``). See `docs.rs/alkahest-cas <https://docs.rs/alkahest-cas>`_ for the
full Rust API reference.

For optional Cargo features (``jit``, ``groebner``, ``cuda``, …) and full developer setup, see the
`Getting started <../getting-started.html>`_ chapter of the user guide.

.. toctree::
   :maxdepth: 2
   :caption: Core

   api/core
   api/simplify
   api/diff
   api/poly
   api/numerics
   api/transform

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   api/matrix
   api/ode
   api/solve
   api/codegen
   api/errors

For the conceptual guide (kernel design, rule engine, e-graph, derivation logs)
see the `mdBook user guide <../>`_.
