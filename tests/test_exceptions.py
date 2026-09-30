import pytest

from docwrap._exceptions import (
    DocwrapConfigWarning,
    DocwrapSkipWarning,
    InvalidLineLengthError,
    UnknownStyleError,
)


class TestInvalidLineLengthError:
    def test_boolean_is_rejected_as_wrong_type(self) -> None:
        err = InvalidLineLengthError(line_length=True)
        assert err.line_length is True
        assert str(err) == 'line_length must be an int, got bool (True)'

    def test_out_of_range_integer_message(self) -> None:
        err = InvalidLineLengthError(-1)
        assert err.line_length == -1
        assert str(err) == 'line_length must be >= 1, got -1'

    @pytest.mark.parametrize('value', ['80', None, 80.5, [80]])
    def test_wrong_type_retains_payload_and_reports_type(self, value: object) -> None:
        err = InvalidLineLengthError(value)
        assert err.line_length is value
        assert str(err) == f'line_length must be an int, got {type(value).__name__} ({value!r})'


class TestUnknownStyleError:
    def test_message(self) -> None:
        err = UnknownStyleError('bogus')
        assert 'bogus' in str(err)

    def test_non_string_payload_is_retained(self) -> None:
        value = ['bogus']
        err = UnknownStyleError(value)
        assert err.style is value
        assert str(err) == "unknown style: ['bogus']"


def test_docwrap_config_warning_is_user_warning() -> None:
    assert issubclass(DocwrapConfigWarning, UserWarning)


def test_docwrap_skip_warning_is_user_warning() -> None:
    assert issubclass(DocwrapSkipWarning, UserWarning)
