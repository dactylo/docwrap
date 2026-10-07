"""Unified diff formatting with optional git-style colorization."""

import difflib
from collections.abc import Iterator

from docwrap._color import BOLD, GREEN, RED, colorize, path_text, visible_controls


def _colorize_diff_line(line: str) -> str:
    """Apply git-style color to a unified diff line."""
    suffix = '\n' if line.endswith('\n') else ''
    content = line.rstrip('\n')
    if content.startswith(('---', '+++')):
        return colorize(content, BOLD) + suffix
    if content.startswith('@@'):
        return colorize(content, BOLD) + suffix
    if content.startswith('-'):
        return colorize(content, RED) + suffix
    if content.startswith('+'):
        return colorize(content, GREEN) + suffix
    return line


def _split_physical_lines(text: str) -> list[str]:
    """Split on LF only, keeping terminators, so hunks count physical lines."""
    lines = text.split('\n')
    result = [f'{line}\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def iter_diff(
    source: str,
    result: str,
    path: str,
    *,
    color: bool,
    terminal: bool = False,
) -> Iterator[str]:
    """Generate unified diff output with optional colorization.

    Args:
        source: Original text.
        result: Modified text.
        path: File path for diff header.
        color: Whether to apply ANSI color codes.
        terminal: Whether to use human-facing path separators in the header.

    Yields:
        Visible-control-safe unified diff lines, optionally colorized.
    """
    source_lines = _split_physical_lines(source)
    result_lines = _split_physical_lines(result)
    visible_path = visible_controls(path_text(path) if terminal else str(path))
    diff = difflib.unified_diff(
        source_lines,
        result_lines,
        fromfile=visible_path,
        tofile=visible_path,
    )
    for line in diff:
        visible_line = visible_controls(line, preserve_lf=True)
        yield _colorize_diff_line(visible_line) if color else visible_line
