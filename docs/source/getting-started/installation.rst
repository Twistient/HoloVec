Installation
============

HoloVec can be installed via pip or from source.

Quick Install
-------------

.. code-block:: bash

    pip install holovec

This installs HoloVec with the NumPy backend (CPU-only).

Optional Dependencies
---------------------

GPU Support (PyTorch)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install holovec[torch]

Or install PyTorch separately:

.. code-block:: bash

    pip install torch
    pip install holovec

JIT/TPU Support (JAX)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install holovec[jax]

Or install JAX separately:

.. code-block:: bash

    pip install jax jaxlib
    pip install holovec

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install holovec[dev]

This includes testing, linting, and documentation tools.

Install from Source
-------------------

.. code-block:: bash

    git clone https://github.com/twistient/holovec.git
    cd holovec
    pip install -e .

Verify Installation
-------------------

.. code-block:: python

    import holovec
    print(holovec.__version__)

    from holovec import VSA
    model = VSA.create('MAP', dim=1000, seed=42)
    print(f"Model created: {model.model_name}")

Requirements
------------

* Python >= 3.9
* NumPy >= 1.20
* Optional: PyTorch >= 2.0, JAX >= 0.4

Next Steps
----------

Continue to :doc:`quickstart` for a 5-minute introduction.
