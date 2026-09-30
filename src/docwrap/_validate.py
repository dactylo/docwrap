"""Shared construction-invariant validation for the public configuration."""

from typing import TYPE_CHECKING

from docwrap._exceptions import InvalidLineLengthError, UnknownStyleError
from docwrap._styles import get_styles
from docwrap._types import FILL_STYLES, REFLOW_MODES, SUMMARY_STARTS

if TYPE_CHECKING:
    from docwrap._types import DocwrapConfig


def validate_config(config: 'DocwrapConfig') -> None:
    """Validate every construction invariant of a `DocwrapConfig`.

    Called by `DocwrapConfig.__post_init__` at construction time and by the
    public operations as boundary defense for objects supplied by untyped
    callers or subclasses. Operations must not validate a different subset.

    Args:
        config: Configuration object to validate.

    Raises:
        InvalidLineLengthError: If `line_length` is not an int of at least 1,
            or is a bool.
        UnknownStyleError: If `style` or `convert` names an unknown style.
        TypeError: If `strip_types` or `wrap_summary` is not a bool.
        ValueError: If `smart_slack` is not a non-negative int, or
            `fill_style`, `reflow`, or `summary_start` is not a member of its
            published vocabulary.
    """
    ll = config.line_length
    if isinstance(ll, bool) or not isinstance(ll, int) or ll < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidLineLengthError(ll)
    if (
        isinstance(config.smart_slack, bool)
        or not isinstance(config.smart_slack, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or config.smart_slack < 0
    ):
        msg = f'smart_slack must be a non-negative integer, got {config.smart_slack!r}'
        raise ValueError(msg)
    if not isinstance(config.strip_types, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = f'strip_types must be a bool, got {config.strip_types!r}'
        raise TypeError(msg)
    if not isinstance(config.wrap_summary, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = f'wrap_summary must be a bool, got {config.wrap_summary!r}'
        raise TypeError(msg)
    if config.style is not None and config.style not in get_styles():
        raise UnknownStyleError(config.style)
    if config.convert is not None and config.convert not in get_styles():
        raise UnknownStyleError(config.convert)
    if config.fill_style not in FILL_STYLES:
        valid = '/'.join(sorted(FILL_STYLES))
        msg = f'fill_style must be one of {valid}, got {config.fill_style!r}'
        raise ValueError(msg)
    if config.reflow not in REFLOW_MODES:
        msg = f'reflow must be one of {"/".join(sorted(REFLOW_MODES))}, got {config.reflow!r}'
        raise ValueError(msg)
    if config.summary_start not in SUMMARY_STARTS:
        valid = '/'.join(SUMMARY_STARTS)
        msg = f'summary_start must be one of {valid}, got {config.summary_start!r}'
        raise ValueError(msg)
