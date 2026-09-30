"""Terminal formatting for docstring safety skips."""

from typing import TYPE_CHECKING

from docwrap._color import visible_controls

if TYPE_CHECKING:
    from docwrap._types import DocstringSkip


def format_safety_skip(path: str, skip: 'DocstringSkip') -> str:
    """Format a safety skip with visible control characters for terminal output."""
    return (
        f'{visible_controls(path)}:{skip.line}: '
        f'{visible_controls(skip.qualname)} skipped '
        f'({visible_controls(skip.code)}): {visible_controls(skip.message)}'
    )
