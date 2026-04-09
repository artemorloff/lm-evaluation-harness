"""Shims for transformers API differences across versions.
Transformers v5 removed ``AutoModelForVision2Seq``; VLMs are loaded via
``AutoModelForImageTextToText`` instead (same underlying model mappings).
"""
try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # transformers >= 5
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
__all__ = ["AutoModelForVision2Seq"]