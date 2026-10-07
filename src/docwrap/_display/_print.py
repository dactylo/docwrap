"""Print orchestrator -- mode-specific output functions."""

from typing import TYPE_CHECKING

from docwrap._color import BOLD, GREEN, RED, colorize, path_text, visible_controls
from docwrap._display._summary import format_summary
from docwrap._types import DEFAULT_LINE_LENGTH

_PREVIEW_RULER_CAP = 200

if TYPE_CHECKING:
    from docwrap._types import FileResult


def _format_preview_label(label: str, line_length: int | None) -> str:
    """Add a bounded ruler and its truthful configured-length marker."""
    if line_length is None or line_length <= len(label):
        return label
    ruler_width = min(line_length, _PREVIEW_RULER_CAP)
    padding = ' ' * (ruler_width - len(label))
    if line_length <= _PREVIEW_RULER_CAP:
        return f'{label}{padding}|{line_length}'
    return f'{label}{padding}...|{line_length}'


def _format_preview(results: list['FileResult'], *, color: bool = False) -> str:
    """Format preview report: per-docstring before/after snippets."""
    blocks: list[str] = []
    for r in results:
        for dc in r.docstrings:
            show_skip_note = r.show_skips and dc.summary_skipped
            if not dc.changed and not show_skip_note:
                continue

            header = visible_controls(f'{path_text(r.path)}:{dc.start_line}:{dc.qualname}')
            parts: list[str] = [colorize(header, BOLD) if color else header]

            # Skip diagnostic
            if show_skip_note:
                target = r.line_length or DEFAULT_LINE_LENGTH
                width_info = f' ({dc.summary_width} > {target})' if dc.summary_width else ''
                parts.append(f'  skip: summary exceeds line length{width_info}')

            # Before/after (only for changed docstrings)
            if dc.changed:
                before_label = _format_preview_label('  before:', r.line_length)
                after_label = _format_preview_label('  after:', r.line_length)
                before = visible_controls(dc.before or '', preserve_lf=True)
                after = visible_controls(dc.after or '', preserve_lf=True)
                before_section = f'{before_label}\n{before}'
                after_section = f'{after_label}\n{after}'
                if color:
                    before_section = colorize(before_section, RED)
                    after_section = colorize(after_section, GREEN)
                parts.append(before_section)
                parts.append(after_section)

            blocks.append('\n'.join(parts))
    return '\n\n'.join(blocks)


def print_preview(
    *,
    results: list['FileResult'],
    color: bool,
) -> None:
    """Format and print preview output.

    Args:
        results: File results to preview.
        color: Whether to apply ANSI color codes.
    """
    output = _format_preview(results, color=color)
    if output:
        print(output)


def print_summary(
    *,
    results: list['FileResult'],
    verbose: bool,
    color: bool,
) -> None:
    """Format and print summary table output.

    Args:
        results: File results to summarize.
        verbose: Whether to include unchanged files with relevant docstrings.
        color: Whether to apply ANSI color codes.
    """
    output = format_summary(results, verbose=verbose, color=color)
    if output:
        print(output)
