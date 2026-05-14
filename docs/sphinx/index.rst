Alkahest — Python API reference
================================

Installation
------------

Install the release wheel from PyPI (Python ≥ 3.9)::

   pip install alkahest

For optional LLVM JIT wheels, extra Cargo features, and building from source with `maturin`, see the
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
