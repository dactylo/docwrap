"""Columnar formatting for the default CLI summary display."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from docwrap._color import BOLD, colorize, path_text, severity_color, visible_controls

if TYPE_CHECKING:
    from docwrap._types import FileResult


_HEADERS = ('Name', 'Skip', 'Total', 'Wrap', 'Wrap%')


@dataclass(frozen=True, slots=True)
class _SummaryAggregates:
    """TOTAL row inputs for format_summary."""

    n_changed: int
    agg_skip: int
    agg_wrapped: int
    agg_total: int
    show_total: bool


class _SummaryRow(NamedTuple):
    """Display path and counts for one included file."""

    path: str
    skipped: int
    wrapped: int
    total: int
    changed: bool


@dataclass(frozen=True, slots=True)
class _SummaryWidths:
    """Column widths for format_summary."""

    name_w: int
    skip_w: int
    tot_w: int
    wrapped_w: int
    pct_w: int


def _collect_summary_rows(results: list['FileResult'], *, verbose: bool) -> list[_SummaryRow]:
    """Select display rows from file results."""
    rows: list[_SummaryRow] = []
    for r in results:
        total = r.total
        if total is None:
            continue
        changed = r.wrapped > 0
        has_skips = r.skipped_summaries > 0
        if not changed and not (verbose and (has_skips or total > 0)):
            continue
        rows.append(
            _SummaryRow(
                path=visible_controls(path_text(str(r.path))),
                skipped=r.skipped_summaries,
                wrapped=r.wrapped,
                total=total,
                changed=changed,
            )
        )
    return rows


def _colorize_pct_row(
    row: str,
    pct: str,
    severity: int,
    *,
    name: str | None = None,
) -> str:
    """Apply severity color to Wrap% and optionally the filename in a row."""
    if not pct:
        return row
    sev = severity_color(severity)
    if sev is None:
        return row
    if name:
        row = colorize(name, sev) + row[len(name) :]
    return row[: -len(pct)] + colorize(pct, sev)


def _compute_summary_aggregates(
    rows: list[_SummaryRow],
) -> _SummaryAggregates:
    """Compute TOTAL values from all displayed rows."""
    n_changed = sum(row.changed for row in rows)
    agg_skip = sum(row.skipped for row in rows)
    agg_wrapped = sum(row.wrapped for row in rows)
    agg_total = sum(row.total for row in rows)
    return _SummaryAggregates(
        n_changed=n_changed,
        agg_skip=agg_skip,
        agg_wrapped=agg_wrapped,
        agg_total=agg_total,
        show_total=n_changed >= 2,  # noqa: PLR2004
    )


def _format_summary_row(  # noqa: PLR0913
    name: str,
    skip_s: str,
    tot_s: str,
    wrapped_s: str,
    pct_s: str,
    *,
    widths: _SummaryWidths,
) -> str:
    """Format one summary row to align by widths."""
    return (
        f'{name:<{widths.name_w}}  '
        f'{skip_s:>{widths.skip_w}}  '
        f'{tot_s:>{widths.tot_w}}  '
        f'{wrapped_s:>{widths.wrapped_w}}  '
        f'{pct_s:>{widths.pct_w}}'
    )


def _pct_str(wrapped: int, total: int) -> str:
    """Format wrap percentage, or empty string if total is zero."""
    if total == 0:
        return ''
    if wrapped == 0:
        return '0%'
    percentage = wrapped / total * 100
    if percentage < 1:
        return '<1%'
    return f'{round(percentage)}%'


def _compute_summary_widths(
    rows: list[_SummaryRow],
    aggs: _SummaryAggregates,
) -> _SummaryWidths:
    """Size columns to fit displayed rows and the optional TOTAL row."""
    name_w = max(len(_HEADERS[0]), *(len(row.path) for row in rows))
    if aggs.show_total:
        name_w = max(name_w, len('TOTAL'))
    skip_w = max(len(_HEADERS[1]), *(len(str(row.skipped)) for row in rows))
    tot_w = max(len(_HEADERS[2]), *(len(str(row.total)) for row in rows))
    wrapped_w = max(len(_HEADERS[3]), *(len(str(row.wrapped)) for row in rows))
    pct_w = max(
        len(_HEADERS[4]),
        *(len(_pct_str(row.wrapped, row.total)) for row in rows),
    )
    if aggs.show_total:
        skip_w = max(skip_w, len(str(aggs.agg_skip)))
        wrapped_w = max(wrapped_w, len(str(aggs.agg_wrapped)))
        tot_w = max(tot_w, len(str(aggs.agg_total)))
        pct_w = max(pct_w, len(_pct_str(aggs.agg_wrapped, aggs.agg_total)))
    return _SummaryWidths(
        name_w=name_w,
        skip_w=skip_w,
        tot_w=tot_w,
        wrapped_w=wrapped_w,
        pct_w=pct_w,
    )


def _format_summary_lines(
    rows: list[_SummaryRow],
    aggs: _SummaryAggregates,
    widths: _SummaryWidths,
    *,
    color: bool,
) -> str:
    """Render the header, file rows, and optional TOTAL row."""
    header_line = _format_summary_row(*_HEADERS, widths=widths)
    if color:
        header_line = colorize(header_line, BOLD)
    lines = [header_line]
    for item in rows:
        pct = _pct_str(item.wrapped, item.total)
        row = _format_summary_row(
            item.path,
            str(item.skipped),
            str(item.total),
            str(item.wrapped),
            pct,
            widths=widths,
        )
        if color:
            row = _colorize_pct_row(row, pct, 1 if item.wrapped > 0 else 0, name=item.path)
        lines.append(row)
    if aggs.show_total:
        agg_pct = _pct_str(aggs.agg_wrapped, aggs.agg_total)
        row = _format_summary_row(
            'TOTAL',
            str(aggs.agg_skip),
            str(aggs.agg_total),
            str(aggs.agg_wrapped),
            agg_pct,
            widths=widths,
        )
        if color:
            row = _colorize_pct_row(row, agg_pct, 1 if aggs.agg_wrapped > 0 else 0)
            row = colorize(row, BOLD)
        lines.append(row)
    return '\n'.join(lines)


def format_summary(
    results: list['FileResult'], *, verbose: bool = False, color: bool = False
) -> str:
    """Format columnar summary table.

    Args:
        results: File results to summarize.
        verbose: Whether to include unchanged files with docstrings or skipped
            summaries.
        color: Whether to apply ANSI color.

    Returns:
        The rendered summary table, or an empty string when there are no rows.
    """
    rows = _collect_summary_rows(results, verbose=verbose)
    if not rows:
        return ''
    aggregates = _compute_summary_aggregates(rows)
    widths = _compute_summary_widths(rows, aggregates)
    return _format_summary_lines(rows, aggregates, widths, color=color)
