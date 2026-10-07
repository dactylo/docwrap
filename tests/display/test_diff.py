from pathlib import PureWindowsPath

import pytest

from docwrap import _color
from docwrap._color import BOLD, GREEN, RED, RESET
from docwrap._display._diff import iter_diff


class TestIterDiff:
    def test_diff_output_with_color(self) -> None:
        lines = list(iter_diff('old\n', 'new\n', 'test.py', color=True))
        colored = ''.join(lines)
        # ANSI escape codes should be present
        assert '\033[' in colored
        # Content should still be there
        assert 'old' in colored
        assert 'new' in colored

    def test_diff_output_without_color(self) -> None:
        lines = list(iter_diff('old\n', 'new\n', 'test.py', color=False))
        assert any('---' in line for line in lines)
        assert any('-old' in line for line in lines)
        assert any('+new' in line for line in lines)

    def test_no_diff_when_identical(self) -> None:
        lines = list(iter_diff('hello\n', 'hello\n', 'test.py', color=False))
        assert lines == []


class TestPhysicalLineCoordinates:
    def test_colorized_output_still_emits(self) -> None:
        source = 'a\u2028b\nchanged\n'
        result = 'a\u2028b\nCHANGED\n'
        diff = ''.join(iter_diff(source, result, 'mod.py', color=True))
        assert 'CHANGED' in diff

    def test_no_final_newline_preserved(self) -> None:
        source = 'line one\nline two'
        result = 'line one\nline TWO'
        diff = ''.join(iter_diff(source, result, 'mod.py', color=False))
        assert '-line two' in diff
        assert '+line TWO' in diff

    def test_u2028_does_not_shift_hunk_header(self) -> None:
        source = '# comment with\u2028embedded separator\ndef f():\n    """Old docstring."""\n'
        result = source.replace('Old', 'New')
        diff = ''.join(iter_diff(source, result, 'mod.py', color=False))
        assert '@@ -1,3 +1,3 @@' in diff


class TestVisibleControls:
    def test_control_characters_are_escaped_after_raw_diff_generation(self) -> None:
        source = 'first\x1b]0;title\x07\nold\tvalue\x9b\n'
        result = source.replace('old', 'new')

        output = ''.join(iter_diff(source, result, 'mod\n\r.py', color=False))

        assert '@@ -1,2 +1,2 @@' in output
        assert 'mod\\x0a\\x0d.py' in output
        assert ' first\\x1b]0;title\\x07\n' in output
        assert '-old\\x09value\\x9b\n' in output
        assert '+new\\x09value\\x9b\n' in output
        assert all(
            character == '\n' or (ord(character) >= 32 and not 127 <= ord(character) <= 159)
            for character in output
        )

    def test_control_characters_are_escaped_before_diff_color(self) -> None:
        output = ''.join(iter_diff('old\x1b\n', 'new\x9b\n', 'test.py', color=True))

        assert f'{BOLD}--- test.py{RESET}\n' in output
        assert f'{RED}-old\\x1b{RESET}\n' in output
        assert f'{GREEN}+new\\x9b{RESET}\n' in output


@pytest.mark.parametrize('terminal', [False, True])
def test_windows_diff_path_preserves_redirected_output(
    terminal: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_color, 'PurePath', PureWindowsPath)
    output = ''.join(
        iter_diff('old\n', 'new\n', r'C:\repo\file.py', color=False, terminal=terminal)
    )
    expected = 'C:/repo/file.py' if terminal else r'C:\repo\file.py'
    assert output.startswith(f'--- {expected}\n')
