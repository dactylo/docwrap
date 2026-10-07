"""JSON output formatting and error records."""

import json
from typing import TYPE_CHECKING, TypedDict

from docwrap._types import JsonStatus, SkipCode, StyleName

if TYPE_CHECKING:
    from docwrap._types import WrapReport


class _JsonChange(TypedDict):
    """Describe one changed docstring in JSON output."""

    qualname: str
    start_line: int
    end_line: int
    before: str | None
    after: str | None
    summary_skipped: bool
    summary_width: int | None


class _JsonSkip(TypedDict):
    """Describe one skipped docstring in JSON output."""

    line: int
    qualname: str
    code: SkipCode
    message: str


class _JsonCounts(TypedDict):
    """JSON record for per-file docstring counts."""

    wrapped: int
    total: int
    skipped_summaries: int


class JsonFileRecord(TypedDict):
    """JSON version 1 record for one file.

    Attributes:
        path: Display path of the processed file.
        status: Processing outcome.
        warning: Failure diagnostic, or `None`.
        style: Resolved docstring style, or `None`.
        line_length: Effective line length, or `None` after failure.
        docstrings: Per-file docstring counts.
        skips: Safety records for skipped docstrings.
        changes: Records for changed docstrings.
    """

    path: str
    status: JsonStatus
    warning: str | None
    style: StyleName | None
    line_length: int | None
    docstrings: _JsonCounts
    skips: list[_JsonSkip]
    changes: list[_JsonChange]


def format_json(records: list[JsonFileRecord]) -> str:
    """Format JSON records with version envelope.

    Args:
        records: Per-file JSON records.

    Returns:
        JSON document with version, file records, and total skip count.
    """
    total_skips = sum(len(record.get('skips', [])) for record in records)
    return json.dumps({'version': 1, 'files': records, 'total_skips': total_skips}, indent=2)


def json_error_record(path: str, status: JsonStatus, warning: str) -> JsonFileRecord:
    """Build a JSON record for a file that could not be processed.

    Args:
        path: Display path for the file.
        status: Error status to record.
        warning: Diagnostic text for the failure.

    Returns:
        A JSON file record with empty result collections.
    """
    return {
        'path': path,
        'status': status,
        'warning': warning,
        'style': None,
        'line_length': None,
        'docstrings': {'wrapped': 0, 'total': 0, 'skipped_summaries': 0},
        'skips': [],
        'changes': [],
    }


def json_success_record(
    path: str, report: 'WrapReport', line_length: int, *, changed: bool
) -> JsonFileRecord:
    """Build a JSON version 1 record for a successfully processed file.

    Args:
        path: Display path for the file.
        report: Wrapping results and per-docstring records.
        line_length: Effective wrapping width.
        changed: Whether the result differs from the original source.

    Returns:
        A file record preserving the JSON version 1 field order.
    """
    changes: list[_JsonChange] = [
        {
            'qualname': dc.qualname,
            'start_line': dc.start_line,
            'end_line': dc.end_line,
            'before': dc.before,
            'after': dc.after,
            'summary_skipped': dc.summary_skipped,
            'summary_width': dc.summary_width,
        }
        for dc in report.docstrings
        if dc.changed
    ]
    skips: list[_JsonSkip] = [
        {
            'line': skip.line,
            'qualname': skip.qualname,
            'code': skip.code,
            'message': skip.message,
        }
        for skip in report.skips
    ]

    return {
        'path': path,
        'status': 'changed' if changed else 'unchanged',
        'warning': None,
        'style': report.style,
        'line_length': line_length,
        'docstrings': {
            'wrapped': report.wrapped,
            'total': report.total,
            'skipped_summaries': report.skipped_summaries,
        },
        'skips': skips,
        'changes': changes,
    }
