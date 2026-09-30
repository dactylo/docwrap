"""Display formatting -- diff, JSON, summary tables, stats rendering, print orchestration."""

from docwrap._display._diff import iter_diff
from docwrap._display._json import format_json, json_error_record
from docwrap._display._print import print_preview, print_summary
from docwrap._display._stats import print_stats
from docwrap._display._summary import format_summary

__all__ = [
    'format_json',
    'format_summary',
    'iter_diff',
    'json_error_record',
    'print_preview',
    'print_stats',
    'print_summary',
]
