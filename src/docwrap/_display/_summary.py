"""Columnar formatting for the default CLI summary display."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from docwrap._color import BOLD, colorize, severity_color, visible_controls

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


@dataclass(frozen=True, slots=True)
class _SummaryWidths:
    """Column widths for format_summary."""

    name_w: int
    skip_w: int
    tot_w: int
    wrapped_w: int
    pct_w: int


def _collect_summary_rows(
    results: list['FileResult'], *, verbose: bool
) -> list[tuple[str, int, int, int, bool]]:
    """Pass 1: collect (path, skip, wrapped, total, changed) tuples from results."""
    rows: list[tuple[str, int, int, int, bool]] = []
    for r in results:
        total = r.total
        if total is None:
            continue
        changed = r.wrapped > 0
        has_skips = r.skipped_summaries > 0
        if not changed and not (verbose and (has_skips or total > 0)):
            continue
        rows.append(
            (
                visible_controls(str(r.path)),
                r.skipped_summaries,
                r.wrapped,
                total,
                changed,
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
    rows: list[tuple[str, int, int, int, bool]],
) -> _SummaryAggregates:
    """Pass 2: compute TOTAL row values from all displayed rows."""
    n_changed = sum(1 for _, _, _, _, ch in rows if ch)
    agg_skip = sum(s for _, s, _, _, _ in rows)
    agg_wrapped = sum(w for _, _, w, _, _ in rows)
    agg_total = sum(t for _, _, _, t, _ in rows)
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
    rows: list[tuple[str, int, int, int, bool]],
    aggs: _SummaryAggregates,
) -> _SummaryWidths:
    """Pass 3: compute column widths for the summary table."""
    name_w = max(len(_HEADERS[0]), *(len(p) for p, _, _, _, _ in rows))
    if aggs.show_total:
        name_w = max(name_w, len('TOTAL'))
    skip_w = max(len(_HEADERS[1]), *(len(str(s)) for _, s, _, _, _ in rows))
    tot_w = max(len(_HEADERS[2]), *(len(str(t)) for _, _, _, t, _ in rows))
    wrapped_w = max(len(_HEADERS[3]), *(len(str(w)) for _, _, w, _, _ in rows))
    pct_w = max(
        len(_HEADERS[4]),
        *(len(_pct_str(w, t)) for _, _, w, t, _ in rows),
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
    rows: list[tuple[str, int, int, int, bool]],
    aggs: _SummaryAggregates,
    widths: _SummaryWidths,
    *,
    color: bool,
) -> str:
    """Pass 4: format header + body rows + optional TOTAL row."""
    header_line = _format_summary_row(*_HEADERS, widths=widths)
    if color:
        header_line = colorize(header_line, BOLD)
    lines = [header_line]
    for path_str, skip, wrapped, total, _ in rows:
        pct = _pct_str(wrapped, total)
        row = _format_summary_row(
            path_str, str(skip), str(total), str(wrapped), pct, widths=widths
        )
        if color:
            row = _colorize_pct_row(row, pct, 1 if wrapped > 0 else 0, name=path_str)
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
