"""
Physics models for LaSDI.

This file exists to make `Physics/` a proper Python package and to ensure
imports like `from Physics import Physics` resolve to the *class* defined in
`Physics.py`, not the submodule.
"""

from .Physics import Physics

__all__ = ["Physics"]

