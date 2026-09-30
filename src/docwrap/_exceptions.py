"""Custom exceptions for the docwrap public API."""


class DocwrapConfigWarning(UserWarning):
    """Category for configuration-resolution diagnostics.

    Emitted by `resolve_config()` and the internal resolvers when a discovered
    configuration file cannot take effect: unreadable or malformed files,
    non-table sections, unknown keys, and invalid values for known keys.
    """


class DocwrapSkipWarning(UserWarning):
    """Warning emitted by `wrap_source` when a docstring is skipped for safety."""


class InvalidLineLengthError(ValueError):
    """Raised when `line_length` is invalid.

    Attributes:
        line_length: Invalid line length supplied by the caller.
    """

    def __init__(self, line_length: object) -> None:
        """Store the invalid value and build a descriptive message.

        Args:
            line_length: Invalid line length supplied by the caller.
        """
        if isinstance(line_length, bool) or not isinstance(line_length, int):
            msg = f'line_length must be an int, got {type(line_length).__name__} ({line_length!r})'
        else:
            msg = f'line_length must be >= 1, got {line_length!r}'
        super().__init__(msg)
        self.line_length: object = line_length


class UnknownStyleError(ValueError):
    """Raised when a style name is not recognized.

    Attributes:
        style: Unrecognized style name supplied by the caller.
    """

    def __init__(self, style: object) -> None:
        """Store the unrecognized style name and build a descriptive message.

        Args:
            style: Unrecognized style name supplied by the caller.
        """
        super().__init__(f'unknown style: {style!r}')
        self.style: object = style
