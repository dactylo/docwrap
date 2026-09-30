"""ANSI color support for CLI output."""

import os
from typing import TextIO

from docwrap._types import ColorMode

BOLD = '\033[1m'
GREEN = '\033[32m'
RED = '\033[31m'
RESET = '\033[0m'
YELLOW = '\033[33m'
_BLUE = '\033[34m'
_CYAN = '\033[36m'
_MAGENTA = '\033[35m'

HISTOGRAM_SPECTRUM = [_MAGENTA, _BLUE, _CYAN, GREEN, YELLOW, RED]


def colorize(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text.

    Args:
        text: The text to colorize.
        *codes: ANSI escape code strings to prepend.

    Returns:
        Text wrapped in ANSI codes with a trailing reset, or the original text
        if no codes are given.
    """
    if not codes:
        return text
    return ''.join(codes) + text + RESET


def bold_if(text: str, color: bool) -> str:
    """Apply BOLD if color is enabled, otherwise return text unchanged.

    Args:
        text: Text to format.
        color: Whether to apply the `BOLD` ANSI code.

    Returns:
        The bold-formatted text when color is enabled, otherwise the original
        text.
    """
    return colorize(text, BOLD) if color else text


def percentile_color(value: int, limit: int | None) -> str | None:
    """Return color for a value relative to a configured limit.

    Args:
        value: The value to check (percentile, char count, or target).
        limit: The configured line length, or None if not set.

    Returns:
        RED if value exceeds limit, YELLOW if equal, else None.
    """
    if limit is None:
        return None
    if value > limit:
        return RED
    if value == limit:
        return YELLOW
    return None


def severity_color(percentage: int) -> str | None:
    """Return color for default mode severity percentage.

    Args:
        percentage: Wrap percentage (0-100).

    Returns:
        YELLOW if > 0%, else None.
    """
    return YELLOW if percentage > 0 else None


def should_color(
    stream: TextIO,
    *,
    cli_flag: ColorMode | None = None,
    config_value: ColorMode | None = None,
) -> bool:
    """Determine whether to colorize output for the given stream.

    CLI `always` or `never` takes precedence. Otherwise, the presence of
    `NO_COLOR` disables color, followed by a configured `always` or `never`.
    Remaining cases use the stream's terminal status.

    Args:
        stream: Output stream, checked with `isatty` in automatic mode.
        cli_flag: CLI `--color` value, or `None` if unspecified. `auto` allows
            environment and configuration settings to decide.
        config_value: `[tool.docwrap].color` value, or `None`.

    Returns:
        `True` if output should be colored.
    """
    if cli_flag == 'always':
        return True
    if cli_flag == 'never':
        return False
    if os.environ.get('NO_COLOR') is not None:
        return False
    if config_value == 'always':
        return True
    if config_value == 'never':
        return False
    return hasattr(stream, 'isatty') and stream.isatty()


def visible_controls(text: str, *, preserve_lf: bool = False) -> str:
    r"""Render terminal control characters as visible lowercase escapes.

    Args:
        text: Text to make safe for human-facing terminal output.
        preserve_lf: Whether to retain LF characters as structural newlines.

    Returns:
        Text with C0, DEL, and C1 controls rendered as `\xNN`, except for LF
        when `preserve_lf` is true.
    """
    visible: list[str] = []
    for character in text:
        if character == '\n' and preserve_lf:
            visible.append(character)
        elif character < ' ' or '\x7f' <= character <= '\x9f':
            code = ord(character)
            visible.append(f'\\x{code:02x}')
        else:
            visible.append(character)
    return ''.join(visible)
