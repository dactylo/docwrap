"""Library API implementation behind the package facade."""

from docwrap._config import clear_config_cache, resolve_config
from docwrap._core import check, wrap_file_report, wrap_source, wrap_source_report
from docwrap._exceptions import (
    DocwrapConfigWarning,
    DocwrapSkipWarning,
    InvalidLineLengthError,
    UnknownStyleError,
)
from docwrap._styles import get_styles
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    DetailLevel,
    DocstringChange,
    DocstringSkip,
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    SkipCode,
    StyleName,
    SummaryStart,
    WrapCheckResult,
    WrapReport,
    WrapStatus,
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
