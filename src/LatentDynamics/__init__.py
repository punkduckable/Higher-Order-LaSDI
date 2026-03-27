"""
Latent dynamics models for LaSDI.

This file exists to make `LatentDynamics/` a proper Python package and to ensure
imports like `from LatentDynamics import LatentDynamics` resolve to the *class*
defined in `LatentDynamics.py`, not the submodule.
"""

from .LatentDynamics import LatentDynamics

__all__ = ["LatentDynamics"]

