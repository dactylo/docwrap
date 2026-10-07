"""Public facade for docwrap's supported library API.

Example:
    >>> import docwrap
    >>> docwrap.wrap_source('def f(): pass')
    'def f(): pass'
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docwrap._library import (
        DEFAULT_LINE_LENGTH,
        DetailLevel,
        DocstringChange,
        DocstringSkip,
        DocwrapConfig,
        DocwrapConfigWarning,
        DocwrapSkipWarning,
        FillStyle,
        InvalidLineLengthError,
        ReflowMode,
        SkipCode,
        StyleName,
        SummaryStart,
        UnknownStyleError,
        WrapCheckResult,
        WrapReport,
        WrapStatus,
        check,
        clear_config_cache,
        get_styles,
        resolve_config,
        wrap_file_report,
        wrap_source,
        wrap_source_report,
    )

__all__ = [
    'DEFAULT_LINE_LENGTH',
    'DetailLevel',
    'DocstringChange',
    'DocstringSkip',
    'DocwrapConfig',
    'DocwrapConfigWarning',
    'DocwrapSkipWarning',
    'FillStyle',
    'InvalidLineLengthError',
    'ReflowMode',
    'SkipCode',
    'StyleName',
    'SummaryStart',
    'UnknownStyleError',
    'WrapCheckResult',
    'WrapReport',
    'WrapStatus',
    'check',
    'clear_config_cache',
    'get_styles',
    'resolve_config',
    'wrap_file_report',
    'wrap_source',
    'wrap_source_report',
]
_EXPORTS = frozenset(__all__)


def __dir__() -> list[str]:
    """List package attributes, including exports that are not loaded yet."""
    return sorted({*globals(), *__all__})


def __getattr__(name: str) -> object:
    """Load the library implementation the first time an exported name is used.

    Args:
        name: Attribute requested from the package.

    Returns:
        The exported object, cached on the package for later lookups.

    Raises:
        AttributeError: If `name` is not an exported name.
    """
    if name not in _EXPORTS:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)
    value = getattr(importlib.import_module(f'{__name__}._library'), name)
    globals()[name] = value
    return value
