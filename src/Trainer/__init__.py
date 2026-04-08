"""
Trainer package.

This repository historically used a single-module Trainer implementation (e.g. `src/Trainer.py`)
and many modules import the base class via:

    from Trainer import Trainer

After refactoring the trainer code into a directory (`src/Trainer/`), `Trainer` becomes a package
in the context where `src/` is on `sys.path` (for example when running `src/Workflow.py`).

This `__init__.py` preserves the import contract by re-exporting the base `Trainer` class at the
package level.
"""

from .Trainer import Trainer;

__all__ = ["Trainer"];

