"""
Optional PyMFEM-backed solver modules for the MFEM physics wrappers.

`advection`, `wave_equation`, `klein_gordon`, `telegraphers`, and `nonlinear_elasticity` contain
parallel MFEM operators, initial-condition coefficients, and `Simulate` functions. These modules
require the optional MFEM/mpi4py dependencies and are imported by the matching physics wrappers.
"""

__all__ = [    "advection",
               "wave_equation",
               "klein_gordon",
               "telegraphers",
               "nonlinear_elasticity"];
