import io
import sys
from pathlib import PurePath, PurePosixPath, PureWindowsPath

import pytest

from docwrap import _color
from docwrap._color import (
    _BLUE,
    _CYAN,
    _MAGENTA,
    BOLD,
    GREEN,
    HISTOGRAM_SPECTRUM,
    RED,
    RESET,
    YELLOW,
    colorize,
    percentile_color,
    severity_color,
    should_color,
    visible_controls,
)


class TestColorize:
    def test_empty_string(self) -> None:
        result = colorize('', GREEN)
        assert result == f'{GREEN}{RESET}'

    def test_multiple_codes(self) -> None:
        result = colorize('hello', BOLD, GREEN)
        assert result == f'{BOLD}{GREEN}hello{RESET}'

    def test_no_codes_returns_text(self) -> None:
        assert colorize('hello') == 'hello'

    def test_single_code(self) -> None:
        result = colorize('hello', RED)
        assert result == f'{RED}hello{RESET}'


class TestConstants:
    def test_histogram_spectrum_contents(self) -> None:
        assert set(HISTOGRAM_SPECTRUM) == {_MAGENTA, _BLUE, _CYAN, GREEN, YELLOW, RED}

    def test_histogram_spectrum_length(self) -> None:
        assert len(HISTOGRAM_SPECTRUM) == 6


class TestPercentileColor:
    def test_returns_none_when_below_limit(self) -> None:
        assert percentile_color(50, 79) is None

    def test_returns_none_when_limit_is_none(self) -> None:
        assert percentile_color(90, None) is None

    def test_returns_red_when_above_limit(self) -> None:
        assert percentile_color(80, 79) == RED

    def test_returns_yellow_when_at_limit(self) -> None:
        assert percentile_color(79, 79) == YELLOW


class TestSeverityColor:
    def test_returns_none_when_zero(self) -> None:
        assert severity_color(0) is None

    def test_returns_yellow_when_1(self) -> None:
        assert severity_color(1) == YELLOW


class TestShouldColor:
    def test_always_overrides_no_color_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('NO_COLOR', '1')
        assert should_color(io.StringIO(), cli_flag='always') is True

    def test_always_returns_true(self) -> None:
        assert should_color(io.StringIO(), cli_flag='always') is True

    def test_auto_non_tty_returns_false(self) -> None:
        assert should_color(io.StringIO()) is False

    def test_auto_tty_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('NO_COLOR', raising=False)
        tty = io.StringIO()
        tty.isatty = lambda: True
        assert should_color(tty) is True

    def test_auto_with_no_color_env_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('NO_COLOR', '1')
        assert should_color(sys.stdout) is False

    def test_config_always(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('NO_COLOR', raising=False)
        assert should_color(io.StringIO(), config_value='always') is True

    def test_config_never(self) -> None:
        tty = io.StringIO()
        tty.isatty = lambda: True
        assert should_color(tty, config_value='never') is False

    def test_never_returns_false(self) -> None:
        assert should_color(sys.stdout, cli_flag='never') is False

    def test_no_color_env_overrides_config_always(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('NO_COLOR', '')
        assert should_color(io.StringIO(), config_value='always') is False


class TestVisibleControls:
    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('\x1b]0;title\x07', '\\x1b]0;title\\x07'),
            ('\r', '\\x0d'),
            ('\t', '\\x09'),
            ('\n', '\\x0a'),
            ('\x7f', '\\x7f'),
            ('\x9b', '\\x9b'),
            ('printable ASCII ~', 'printable ASCII ~'),
            ('printable Unicode: café ☃', 'printable Unicode: café ☃'),
        ],
    )
    def test_control_characters_are_rendered_literally(self, text: str, expected: str) -> None:
        assert visible_controls(text) == expected

    @pytest.mark.parametrize(
        ('preserve_lf', 'expected'),
        [(False, 'first\\x0asecond\\x0d'), (True, 'first\nsecond\\x0d')],
    )
    def test_control_lf_is_preserved_only_when_requested(
        self, preserve_lf: bool, expected: str
    ) -> None:
        assert visible_controls('first\nsecond\r', preserve_lf=preserve_lf) == expected


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        (PureWindowsPath('src/package/file.py'), 'src/package/file.py'),
        (PureWindowsPath('C:/my project/file.py'), 'C:/my project/file.py'),
        (PureWindowsPath('C:relative/file.py'), 'C:relative/file.py'),
        (PureWindowsPath('//server/share/file.py'), '//server/share/file.py'),
        (PureWindowsPath('//?/C:/repo/file.py'), str(PureWindowsPath('//?/C:/repo/file.py'))),
        (PureWindowsPath('//./pipe/tool'), str(PureWindowsPath('//./pipe/tool'))),
        (PurePosixPath(r'src/literal\name.py'), r'src/literal\name.py'),
        (PurePosixPath('/repo/my project/file.py'), '/repo/my project/file.py'),
    ],
)
def test_display_paths_preserve_meaning(path: PurePath, expected: str) -> None:
    assert _color.path_text(path) == expected


def test_native_windows_text_keeps_relative_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_color, 'PurePath', PureWindowsPath)
    assert _color.path_text(r'.\src\file.py') == './src/file.py'
    assert _color.path_text(r'\??\C:\repo\file.py') == r'\??\C:\repo\file.py'
