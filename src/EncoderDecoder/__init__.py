"""
Encoder/decoder models for LaSDI.

This file exists to make `EncoderDecoder/` a proper Python package and to ensure
imports like `from EncoderDecoder import EncoderDecoder` resolve to the *class*
defined in `EncoderDecoder.py`, not the submodule.
"""

from .EncoderDecoder import EncoderDecoder

__all__ = ["EncoderDecoder"]

