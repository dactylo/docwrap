"""Shared docstring processing engine."""

from docwrap._core._discover import ParsedDocstring as ParsedDocstring
from docwrap._core._parse import Region as Region
from docwrap._core._process import (
    analyze_source as analyze_source,
)
from docwrap._core._process import (
    check as check,
)
from docwrap._core._process import (
    wrap_file_report as wrap_file_report,
)
from docwrap._core._process import (
    wrap_parsed_docstring as wrap_parsed_docstring,
)
from docwrap._core._process import (
    wrap_source as wrap_source,
)
from docwrap._core._process import (
    wrap_source_report as wrap_source_report,
)
from docwrap._core._process import (
    wrap_source_report_eligible as wrap_source_report_eligible,
)
from docwrap._core._wrap import simulate_wrap_delta as simulate_wrap_delta
