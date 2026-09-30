"""Shared docstring processing engine."""

from docwrap._core._discover import ParsedDocstring
from docwrap._core._parse import Region
from docwrap._core._process import (
    analyze_source,
    check,
    wrap_file_report,
    wrap_parsed_docstring,
    wrap_source,
    wrap_source_report,
    wrap_source_report_eligible,
)
from docwrap._core._wrap import simulate_wrap_delta

__all__ = [
    'ParsedDocstring',
    'Region',
    'analyze_source',
    'check',
    'simulate_wrap_delta',
    'wrap_file_report',
    'wrap_parsed_docstring',
    'wrap_source',
    'wrap_source_report',
    'wrap_source_report_eligible',
]
