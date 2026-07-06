"""Per-framework model inspectors.

Each inspector reads a model built in its target framework and emits a
:class:`~modelvision.core.ir.ModelGraph`. All framework imports are
lazy — importing this package (or any submodule) does not pull in
torch/tensorflow/jax.
"""
