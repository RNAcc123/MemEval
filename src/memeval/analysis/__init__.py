"""Structured analysis utilities."""

from .metrics import compare_records, record_key
from .matching import compare_files

__all__ = ["compare_files", "compare_records", "record_key"]
