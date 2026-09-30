import contextlib
import errno
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import assert_type
from unittest.mock import Mock, patch

import pytest

import docwrap._cli as cli_mod
import docwrap._display as display_mod
import docwrap._downstream as downstream_mod
import docwrap._stats as stats_mod
from docwrap._cli import (
    _EXIT_CHANGED,
    _EXIT_OK,
    _argv_requests_help,
    _build_parser,
    _compute_file_plan,
    _compute_incremental_context,
    _config_field_source,
    _discover_incremental_repositories,
    _FileModeOptions,
    _FilePlan,
    _format_default,
    _IncrementalContext,
    _IncrementalDiagnostic,
    _IncrementalOutcome,
    _process_file,
    _process_file_json,
    _resolve_help_config,
    _wrap_file_once,
    _WrappedFile,
    main,
)
from docwrap._color import BOLD, GREEN, RED, YELLOW
from docwrap._config import ResolvedConfig
from docwrap._core._incremental import DiffHunk
from docwrap._display._json import JsonFileRecord
from docwrap._files import ReadResult, read_file_result
from docwrap._git import RepoDiscoveryResult, RepoDiscoveryStatus
from docwrap._types import (
    DocstringSkip,
    DocwrapConfig,
    IncrementalLevel,
    LineLengthSource,
    Newline,
    WrapOverrides,
    WrapReport,
)
from tests.helpers import (
    decode_text,
    git_commit_all,
    invoke_cli,
    process_transcript,
    run_cli,
    run_closing_after_first_line,
    run_setup,
    run_with_closed_output,
)

BACKSLASH = chr(92)
_BIG_WRAPPABLE = ''.join(
    f'def f{index}():\n    """Summary.\n\n    {" ".join(["wrapping"] * 40)}.\n    """\n\n\n'
    for index in range(3000)
)
_BODY_WRAPPABLE_SOURCE = (
    'def foo():\n'
    '    """Summary.\n'
    '\n'
    '    This body prose is deliberately long enough to exceed the configured default '
    'line length and require wrapping by docwrap.\n'
    '    """\n'
)
_CLOSED_STDOUT_STATUS = {
    'large-before-write': _EXIT_CHANGED,
    'small-before-write': _EXIT_CHANGED,
    'after-first-line': _EXIT_CHANGED,
    'help-before-write': _EXIT_OK,
}
_CONTROL_TEXT = 'esc\x1bcr\rtab\tdel\x7fc1\x85'
_GREEDY_WRAPPED_SOURCE = (
    'def f():\n'
    '    """Summary.\n'
    '\n'
    '    Handles rename/copy advancement (3 null-separated parts) vs normal (2\n'
    '    parts). Yields (header, dest_path) pairs.\n'
    '    """\n'
)
_SKIP_ONLY_SOURCE = 'def skipme():\n    """Line one ' + BACKSLASH + '\n    continued."""\n'
_MIXED_SKIP_SOURCE = (
    _SKIP_ONLY_SOURCE + '\n\n' + 'def wrapme(value):\n'
    '    """Return the value.\n\n'
    '    Args:\n'
    '        value: This is a very long argument description that should '
    'definitely be wrapped.\n'
    '    """\n'
)
_STATS_BODY_HEADINGS = (
    'Source code',
    'Docstrings (wrappable)',
    'Docstrings (not wrappable)',
    'Distribution',
    'Longest wrappable lines',
    'Wrapping impact',
    'Recommended targets',
)
_STATS_SOURCE = (
    'def alpha(first):\n'
    '    """Alpha summary with enough words to provide a useful measured line.\n'
    '\n'
    '    Alpha prose with enough words to provide a useful measured line.\n'
    '\n'
    '    Args:\n'
    '        first: Alpha argument text with enough words to provide a useful measured line.\n'
    '    """\n'
    '\n'
    '\n'
    'def beta(second):\n'
    '    """Beta summary with enough words to provide another useful measured line.\n'
    '\n'
    '    Beta prose with enough words to provide another useful measured line.\n'
    '\n'
    '    Args:\n'
    '        second: Beta argument text with enough words to provide another measured line.\n'
    '    """\n'
)
_VISIBLE_CONTROL_TEXT = r'esc\x1bcr\x0dtab\x09del\x7fc1\x85'


class TestArgvRequestsHelp:
    def test_dash_h(self) -> None:
        assert _argv_requests_help(['-h']) is True

    def test_double_dash_help(self) -> None:
        assert _argv_requests_help(['--help']) is True

    def test_empty(self) -> None:
        assert _argv_requests_help([]) is False

    def test_help_after_double_dash(self) -> None:
        assert _argv_requests_help(['--', '-h']) is False

    def test_help_before_double_dash(self) -> None:
        assert _argv_requests_help(['-h', '--', 'src/']) is True

    def test_help_with_other_args(self) -> None:
        assert _argv_requests_help(['--diff', '--help', 'src/']) is True

    def test_no_help(self) -> None:
        assert _argv_requests_help(['--diff', 'src/']) is False


class TestCaseAliasReporting:
    _WRAPPABLE = (
        'def f():\n'
        '    """Summary.\n'
        '\n'
        '    This body paragraph is deliberately far too long for the configured'
        ' line length and will wrap.\n'
        '    """\n'
    )

    def _alias_pair(self, tmp_path: Path) -> tuple[Path, Path]:
        target = tmp_path / 'CaseFile.py'
        target.write_text(self._WRAPPABLE)
        alias = tmp_path / 'casefile.py'
        if not alias.exists():
            pytest.skip('requires a case-insensitive filesystem')
        return target, alias

    def test_dry_run_counts_one_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target, alias = self._alias_pair(tmp_path)
        assert main(['-s', str(target), str(alias)]) == 0
        out = capsys.readouterr().out
        assert 'across 1 file' in out

    def test_write_invokes_writer_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target, alias = self._alias_pair(tmp_path)
        calls: list[Path] = []
        real_write = cli_mod.write_preserving_newline

        def counting_write(
            path: Path,
            source: str,
            encoding: str,
            *,
            raw: bytes | None = None,
            fallback_newline: Newline = '\n',
        ) -> None:
            calls.append(path)
            real_write(
                path,
                source,
                encoding,
                raw=raw,
                fallback_newline=fallback_newline,
            )

        monkeypatch.setattr(cli_mod, 'write_preserving_newline', counting_write)
        assert main(['-w', str(target), str(alias)]) == 0
        assert len(calls) == 1


class TestClosedStdout:
    @pytest.mark.parametrize('case', list(_CLOSED_STDOUT_STATUS))
    def test_closure_is_quiet(self, case: str, tmp_path: Path) -> None:
        result = _closed_stdout_result(case, tmp_path)
        assert result.stderr == b'', process_transcript(result)

    @pytest.mark.parametrize('case', list(_CLOSED_STDOUT_STATUS))
    def test_closure_keeps_the_established_status(self, case: str, tmp_path: Path) -> None:
        result = _closed_stdout_result(case, tmp_path)
        assert result.returncode == _CLOSED_STDOUT_STATUS[case], process_transcript(result)


class TestClosedStdoutInProcess:
    def test_closure_stops_before_processing_the_next_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _ClosedStream(io.StringIO):
            def write(self, _text: str) -> int:
                raise BrokenPipeError

        first = tmp_path / 'first.py'
        second = tmp_path / 'second.py'
        first.write_text(_BODY_WRAPPABLE_SOURCE, encoding='utf-8')
        second.write_text('def invalid syntax', encoding='utf-8')
        closed = _ClosedStream()
        monkeypatch.setattr(sys, 'stdout', closed)

        try:
            with patch.object(
                cli_mod, '_wrap_file_once', wraps=cli_mod._wrap_file_once
            ) as wrap_file:
                assert main(['-d', str(first), str(second)]) == _EXIT_CHANGED
            assert [call.args[0] for call in wrap_file.call_args_list] == [first]
            assert capsys.readouterr().err == ''
        finally:
            if sys.stdout is not closed:
                sys.stdout.close()

    def test_einval_from_main_work_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_cli(_argv: Sequence[str] | None) -> int:
            raise OSError(errno.EINVAL, 'main work')

        monkeypatch.setattr(cli_mod, '_run_cli', fail_cli)
        monkeypatch.setattr(downstream_mod, '_WINDOWS', True)

        with pytest.raises(OSError, match='main work') as error:
            main([])
        assert error.value.errno == errno.EINVAL

    def test_main_returns_the_status_after_closure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _ClosedStream(io.StringIO):
            def write(self, _text: str) -> int:
                raise BrokenPipeError

        def run_cli(_argv: Sequence[str] | None) -> int:
            print('discarded')
            return _EXIT_CHANGED

        monkeypatch.setattr(cli_mod, '_run_cli', run_cli)
        closed = _ClosedStream()
        monkeypatch.setattr(sys, 'stdout', closed)

        try:
            assert main([]) == _EXIT_CHANGED
            assert sys.stdout is not closed
            assert sys.stdout.write('discarded') == len('discarded')
            sys.stdout.flush()
        finally:
            if sys.stdout is not closed:
                sys.stdout.close()

    def test_windows_einval_during_output_keeps_the_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _WindowsClosedStream(io.StringIO):
            def write(self, _text: str) -> int:
                raise OSError(errno.EINVAL, 'Invalid argument')

        def run_cli(_argv: Sequence[str] | None) -> int:
            print('discarded')
            return _EXIT_CHANGED

        monkeypatch.setattr(downstream_mod, '_WINDOWS', True)
        monkeypatch.setattr(cli_mod, '_run_cli', run_cli)
        closed = _WindowsClosedStream()
        monkeypatch.setattr(sys, 'stdout', closed)

        try:
            assert main([]) == _EXIT_CHANGED
            assert sys.stdout is not closed
        finally:
            if sys.stdout is not closed:
                sys.stdout.close()


class TestColorOutput:
    def test_color_always_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--color=always'])
        assert exit_code == 0

    def test_color_always_overrides_no_color_env(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('NO_COLOR', '1')
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--color=always'])
        captured = capsys.readouterr()
        assert '\033[' in captured.out

    def test_color_auto_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--color=auto'])
        assert exit_code == 0

    def test_color_never_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--color=never'])
        assert exit_code == 0

    def test_diff_color_added_lines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--diff', '--color=always'])
        captured = capsys.readouterr()
        green_lines = [line for line in captured.out.splitlines() if GREEN in line]
        assert green_lines
        for line in green_lines:
            assert '+' in line

    def test_diff_color_file_headers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--diff', '--color=always'])
        captured = capsys.readouterr()
        file_headers = [
            line for line in captured.out.splitlines() if '---' in line or '+++' in line
        ]
        assert file_headers
        for line in file_headers:
            assert BOLD in line  # bold

    def test_diff_color_hunk_headers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--diff', '--color=always'])
        captured = capsys.readouterr()
        hunk_headers = [line for line in captured.out.splitlines() if '@@' in line]
        assert hunk_headers
        for line in hunk_headers:
            assert BOLD in line  # bold

    def test_diff_color_removed_lines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--diff', '--color=always'])
        captured = capsys.readouterr()
        red_lines = [line for line in captured.out.splitlines() if RED in line]
        assert red_lines
        for line in red_lines:
            assert '-' in line

    def test_diff_no_color(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--diff', '--color=never'])
        captured = capsys.readouterr()
        assert captured.out
        assert '\033[' not in captured.out

    def test_invalid_color_rejected(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--color=rainbow'])

    def test_no_color_env_disables_color(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('NO_COLOR', '1')
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f)])
        captured = capsys.readouterr()
        assert captured.out
        assert '\033[' not in captured.out

    def test_preview_color_after_green(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview', '--color=always'])
        captured = capsys.readouterr()
        assert GREEN in captured.out  # green for after

    def test_preview_color_before_red(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview', '--color=always'])
        captured = capsys.readouterr()
        assert RED in captured.out  # red for before

    def test_preview_color_header(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview', '--color=always'])
        captured = capsys.readouterr()
        assert BOLD in captured.out  # bold

    def test_preview_no_color(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview', '--color=never'])
        captured = capsys.readouterr()
        assert captured.out
        assert '\033[' not in captured.out

    def test_preview_ruler_colored_with_section(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview', '--color=always', '-l', '79'])
        captured = capsys.readouterr()
        # The |79 marker is inside the colorized section, so it should be
        # surrounded by the section's ANSI codes
        assert '|79' in captured.out

    def test_summary_bold_filename_for_changed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--color=always'])
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # Header (line 0) is bold
        assert BOLD in lines[0]
        # Data row (line 1) has severity color: any nonzero -> YELLOW
        data_row = lines[1]
        assert YELLOW in data_row
        assert RED not in data_row

    def test_summary_color_never(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--color=never'])
        captured = capsys.readouterr()
        assert captured.out
        assert '\033[' not in captured.out

    def test_summary_color_severity_yellow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(
            _BODY_WRAPPABLE_SOURCE + '\ndef b():\n    """Short."""\n'
            '\ndef c():\n    """Also short."""\n'
        )
        main([str(f), '--color=always'])
        captured = capsys.readouterr()
        data_row = captured.out.strip().splitlines()[1]
        # 1/3 = 33% < 50% -> YELLOW
        assert YELLOW in data_row
        assert RED not in data_row

    def test_summary_color_severity_yellow_at_100(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--color=always'])
        captured = capsys.readouterr()
        data_row = captured.out.strip().splitlines()[1]
        assert YELLOW in data_row
        assert RED not in data_row

    def test_summary_total_row_bold_with_severity_pct(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        a.write_text(_BODY_WRAPPABLE_SOURCE)
        b = tmp_path / 'b.py'
        b.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(a), str(b), '--color=always'])
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        total_row = lines[-1]
        # TOTAL row is bold
        assert BOLD in total_row
        # 100% -> YELLOW on percentage
        assert YELLOW in total_row
        assert RED not in total_row

    def test_summary_unchanged_not_bold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        main([str(f), '-v', '--color=always'])
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        data_row = lines[1]
        # Data row for unchanged file: no severity color, not bold
        assert BOLD not in data_row
        assert RED not in data_row
        assert YELLOW not in data_row

    def test_summary_zero_percent_no_severity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        main([str(f), '-v', '--color=always'])
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        data_row = lines[1]
        # No RED or YELLOW on 0% row
        assert RED not in data_row
        assert YELLOW not in data_row


class TestConfigDiagnosticsSurviveWarningFilters:
    def test_cli_warns_despite_pythonwarnings_ignore(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "bogus"\n')
        f = tmp_path / 'mod.py'
        f.write_text('def foo():\n    """Short."""\n')
        env = os.environ | {'PYTHONPATH': 'src', 'PYTHONWARNINGS': 'ignore'}
        result = run_cli([str(f)], env=env)
        assert 'warning: invalid [tool.docwrap].fill-style' in result.stderr


class TestConvertFlags:
    def test_convert_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--convert', 'numpy', 'src/'])
        assert args.convert == 'numpy'

    def test_no_strip_types_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['src/'])
        assert args.strip_types is False

    def test_strip_types_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--strip-types', 'src/'])
        assert args.strip_types is True


class TestDefaultOutput:
    def test_bare_write_preserves_long_summary_and_wraps_body(self, tmp_path: Path) -> None:
        long_summary = 'Summary ' + 'stays ' * 15 + 'unchanged.'
        long_body = 'Body prose ' + 'wrap ' * 20 + 'finishes here.'
        source = f'def foo():\n    """{long_summary}\n\n    {long_body}\n    """\n'
        path = tmp_path / 'test.py'
        path.write_text(source)

        assert main(['--write', str(path)]) == 0

        result = path.read_text()
        assert not (tmp_path / 'pyproject.toml').exists()
        assert f'    """{long_summary}\n' in result
        assert long_body not in result
        assert '\n    wrap wrap ' in result

    def test_summary_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f)])
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # Header row with new column order
        header_fields = lines[0].split()
        assert header_fields == ['Name', 'Skip', 'Total', 'Wrap', 'Wrap%']
        # Data row contains file path and stats
        assert str(f) in lines[1]
        assert '100%' in lines[1]
        # No TOTAL row for single file
        assert len(lines) == 2

    def test_summary_no_changes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == ''

    def test_summary_per_file_breakdown(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        a.write_text(_BODY_WRAPPABLE_SOURCE)
        b = tmp_path / 'b.py'
        b.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(a), str(b)])
        assert exit_code == 1
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # Header + 2 data rows + TOTAL
        assert len(lines) == 4
        assert 'TOTAL' in lines[3]

    def test_summary_with_changes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def unchanged():\n    """Short."""\n\n' + _BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f)])
        assert exit_code == 1
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # Header + 1 data row (no TOTAL for single file)
        assert len(lines) == 2
        # Data row: 1 wrapped, 2 total, 50%
        data_row = lines[1]
        assert '1' in data_row
        assert '2' in data_row
        assert '50%' in data_row


class TestDefaultPath:
    def test_defaults_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        monkeypatch.chdir(tmp_path)
        exit_code = main([])
        assert exit_code == 0

    def test_summary_without_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        monkeypatch.chdir(tmp_path)
        exit_code = main(['-v'])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert 'Name' in lines[0]
        assert 'test.py' in lines[1]
        assert '0' in lines[1]  # 0 wrapped


class TestDefaultSummaryVisibility:
    def test_default_hides_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        a.write_text(_BODY_WRAPPABLE_SOURCE)
        b = tmp_path / 'b.py'
        b.write_text('def bar():\n    """Short."""\n')
        exit_code = main([str(a), str(b)])
        assert exit_code == 1
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # a.py shown (has changes), b.py hidden (no changes)
        assert any(str(a) in line for line in lines)
        assert not any(str(b) in line for line in lines)

    def test_default_shows_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f)])
        assert exit_code == 1
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert 'Name' in lines[0]
        assert str(f) in lines[1]

    def test_write_shows_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-w'])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert 'Name' in lines[0]
        assert str(f) in lines[1]


class TestDiffMode:
    def test_diff_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '--diff'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert '---' in captured.out
        assert '+++' in captured.out


class TestDirectorySkipRelativePaths:
    def test_ancestor_named_build_not_skipped(self, tmp_path: Path) -> None:
        # "build" is an ancestor of the search root, not a child
        build_dir = tmp_path / 'build'
        project_dir = build_dir / 'project'
        project_dir.mkdir(parents=True)
        f = project_dir / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        # Pass "project" as root -- "build" is an ancestor, should not skip
        exit_code = main([str(project_dir)])
        assert exit_code == 1  # file should be found and have changes


class TestErrorHandling:
    def test_bad_coding_cookie_exit_2(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_bytes(b'# -*- coding: nonexistent -*-\ndef foo(): pass\n')
        exit_code = main([str(f)])
        assert exit_code == 2

    def test_binary_file_exit_2(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_bytes(b'\x80\x81\x82\x83')
        exit_code = main([str(f)])
        assert exit_code == 2

    def test_directory_recursion(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(tmp_path)])
        assert exit_code == 0

    @pytest.mark.parametrize(
        'dirname',
        [
            '.git',
            '.hg',
            '.mypy_cache',
            '.pytest_cache',
            '.ruff_cache',
            '.svn',
            '.tox',
            '.venv',
            '__pycache__',
            'build',
            'dist',
            'venv',
        ],
    )
    def test_directory_skip_list(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        dirname: str,
    ) -> None:
        # Files in skipped directories should not be processed
        skip_dir = tmp_path / dirname
        skip_dir.mkdir()
        f = skip_dir / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        included = tmp_path / 'included.py'
        included.write_text(_BODY_WRAPPABLE_SOURCE)
        # Only the parent dir is passed; dirname should be skipped
        exit_code = main([str(tmp_path)])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert str(included) in captured.out
        assert str(f) not in captured.out

    def test_exit_2_takes_precedence_over_1(self, tmp_path: Path) -> None:
        # One file with syntax error, one with pending changes
        bad = tmp_path / 'bad.py'
        bad.write_text('def foo(\n')
        good = tmp_path / 'good.py'
        good.write_text(_BODY_WRAPPABLE_SOURCE)
        assert main([str(good)]) == 1
        exit_code = main([str(bad), str(good)])
        # Exit code 2 (error) takes precedence over 1 (pending changes)
        assert exit_code == 2

    def test_syntax_error_exit_2(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo(\n')
        exit_code = main([str(f)])
        assert exit_code == 2


class TestExitCodes:
    def test_changes_pending_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert str(f) in captured.out

    def test_no_changes_exit_0(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f)])
        assert exit_code == 0


class TestFileFiltering:
    def test_mixed_py_and_non_py_files(self, tmp_path: Path) -> None:
        readme = tmp_path / 'README.md'
        readme.write_text('This is **not** valid Python.\n')
        py_file = tmp_path / 'test.py'
        py_file.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(readme), str(py_file)])
        # README.md skipped, test.py unchanged -> exit 0
        assert exit_code == 0

    def test_non_py_file_silently_skipped(self, tmp_path: Path) -> None:
        readme = tmp_path / 'README.md'
        readme.write_text('This is **not** valid Python.\n')
        exit_code = main([str(readme)])
        # Should skip silently (exit 0), not error (exit 2)
        assert exit_code == 0

    def test_nonexistent_non_py_path_errors(self, tmp_path: Path) -> None:
        exit_code = main([str(tmp_path / 'nonexistent')])
        assert exit_code == 2

    def test_nonexistent_py_path_errors(self, tmp_path: Path) -> None:
        exit_code = main([str(tmp_path / 'nonexistent.py')])
        assert exit_code == 2


class TestFilePlan:
    @staticmethod
    def _context(
        path: Path,
        *,
        level: IncrementalLevel,
        hunks: list[DiffHunk],
    ) -> _IncrementalContext:
        resolved = path.resolve()
        return _IncrementalContext(
            level=level,
            all_hunks={resolved: hunks} if hunks else {},
            untracked=set(),
            repo_roots={resolved: str(path.parent)},
            diff_failed=set(),
        )

    @pytest.mark.parametrize(
        'case',
        [
            ('read', 'error', 'read failed', False),
            ('decode', 'error', 'cannot decode', False),
            ('syntax', 'invalid', 'invalid syntax', True),
            ('token', 'invalid', 'token failed', True),
            ('invalid-source', 'invalid', 'analysis failed', True),
        ],
    )
    def test_failure_state_matrix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[str, str, str, bool],
    ) -> None:
        failure, expected_kind, expected_diagnostic, has_source = case
        path = tmp_path / 'a.py'
        source = 'def broken(:\n' if failure == 'syntax' else 'x = 1\n'
        path.write_text(source)
        reads = 0

        def tracked_read(_path: Path) -> bytes:
            nonlocal reads
            reads += 1
            if failure == 'read':
                raise OSError('read failed')
            if failure == 'decode':
                raise UnicodeDecodeError('utf-8', b'bad', 0, 1, 'cannot decode')
            return source.encode()

        monkeypatch.setattr(Path, 'read_bytes', tracked_read)
        if failure == 'token':
            monkeypatch.setattr(
                cli_mod,
                'get_eligible_ranges',
                Mock(side_effect=tokenize.TokenError('token failed', (1, 0))),
            )
        elif failure == 'invalid-source':
            monkeypatch.setattr(
                cli_mod,
                'get_eligible_ranges',
                Mock(side_effect=ValueError('analysis failed')),
            )

        plan = _compute_file_plan(
            path,
            self._context(path, level='definition', hunks=[(1, 1, [])]),
        )

        assert plan.state == 'failed'
        assert plan.eligible_ranges is None
        assert plan.failure_kind == expected_kind
        assert plan.diagnostic is not None
        assert expected_diagnostic in plan.diagnostic
        assert plan.prepared == (
            ReadResult(source=source, encoding='utf-8', raw=source.encode())
            if has_source
            else None
        )
        assert reads == 1

    def test_file_level_invalid_source_fails_at_planning_boundary(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / 'a.py'
        raw = b'def broken(:\r\n'
        path.write_bytes(raw)

        plan = _compute_file_plan(
            path,
            self._context(path, level='file', hunks=[(1, 1, [])]),
        )

        assert plan.state == 'failed'
        assert plan.eligible_ranges is None
        assert plan.prepared == ReadResult(
            source='def broken(:\n',
            encoding='utf-8',
            raw=raw,
        )
        assert plan.failure_kind == 'invalid'
        assert plan.diagnostic is not None
        assert 'invalid syntax' in plan.diagnostic

    @pytest.mark.parametrize(
        'case',
        [
            ('full', 'file', [(1, 1, list[bytes]())], 'x = 1\n', None, 1),
            ('ranges', 'definition', [(2, 2, list[bytes]())], 'def f():\n    pass\n', {(1, 2)}, 1),
            ('no-hunks', 'definition', list[DiffHunk](), 'x = 1\n', None, 0),
            (
                'no-eligible',
                'docstring',
                [(1, 1, list[bytes]())],
                'x = 1\n',
                set[tuple[int, int]](),
                1,
            ),
        ],
    )
    def test_ready_and_no_work_state_matrix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[
            str,
            IncrementalLevel,
            list[DiffHunk],
            str,
            set[tuple[int, int]] | None,
            int,
        ],
    ) -> None:
        state, level, hunks, source, expected_ranges, expected_reads = case
        path = tmp_path / 'a.py'
        path.write_text(source)
        reads = 0

        def tracked_read(_path: Path) -> bytes:
            nonlocal reads
            reads += 1
            return source.encode()

        monkeypatch.setattr(Path, 'read_bytes', tracked_read)

        plan = _compute_file_plan(path, self._context(path, level=level, hunks=hunks))

        assert plan.state == state
        assert plan.eligible_ranges == expected_ranges
        assert plan.failure_kind is None
        assert plan.diagnostic is None
        assert plan.prepared == (
            ReadResult(source=source, encoding='utf-8', raw=source.encode())
            if expected_reads
            else None
        )
        assert reads == expected_reads


class TestFillStyleFlag:
    def test_fill_style_changes_wrapping_behavior(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        py_file = tmp_path / 'x.py'
        py_file.write_text(_GREEDY_WRAPPED_SOURCE)
        balanced_exit = main(
            [
                str(py_file),
                '--diff',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'balanced',
            ]
        )
        balanced_output = capsys.readouterr().out
        greedy_exit = main(
            [
                str(py_file),
                '--diff',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'greedy',
            ]
        )
        greedy_output = capsys.readouterr().out
        assert balanced_exit == 1
        assert greedy_exit == 0
        assert balanced_output != greedy_output

    def test_fill_style_greedy_accepted(self) -> None:
        args = _build_parser().parse_args(['--fill-style', 'greedy', '.'])
        assert args.fill_style == 'greedy'

    def test_fill_style_invalid_rejected(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            _build_parser().parse_args(['--fill-style', 'ragged', '.'])

    def test_fill_style_smart_accepted(self) -> None:
        args = _build_parser().parse_args(['--fill-style', 'smart', '.'])
        assert args.fill_style == 'smart'

    def test_negative_smart_slack_rejected(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--smart-slack', '-1', '.'])

    def test_smart_slack_accepted(self) -> None:
        args = _build_parser().parse_args(['--smart-slack', '6', '.'])
        assert args.smart_slack == 6

    def test_stats_respects_fill_style(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        py_file = tmp_path / 'x.py'
        py_file.write_text(_GREEDY_WRAPPED_SOURCE)
        main(
            [
                str(py_file),
                '--stats',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'balanced',
            ]
        )
        balanced_output = capsys.readouterr().out
        main(
            [
                str(py_file),
                '--stats',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'greedy',
            ]
        )
        greedy_output = capsys.readouterr().out
        assert balanced_output != greedy_output

    def test_stats_respects_smart_slack(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        py_file = tmp_path / 'x.py'
        py_file.write_text(_GREEDY_WRAPPED_SOURCE)
        main(
            [
                str(py_file),
                '--stats',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'smart',
                '--smart-slack',
                '4',
            ]
        )
        repaired_output = capsys.readouterr().out
        main(
            [
                str(py_file),
                '--stats',
                '--length',
                '75',
                '--reflow',
                'full',
                '--fill-style',
                'smart',
                '--smart-slack',
                '0',
            ]
        )
        greedy_output = capsys.readouterr().out
        assert repaired_output != greedy_output


class TestHelpDefaultHelpers:
    def test_config_field_source_default_fill_style(self) -> None:
        cfg = DocwrapConfig()
        assert _config_field_source(cfg, 'fill_style') == 'default'

    def test_config_field_source_default_reflow(self) -> None:
        cfg = DocwrapConfig()
        assert _config_field_source(cfg, 'reflow') == 'default'

    def test_config_field_source_default_smart_slack(self) -> None:
        cfg = DocwrapConfig()
        assert _config_field_source(cfg, 'smart_slack') == 'default'

    def test_config_field_source_default_summary_start(self) -> None:
        cfg = DocwrapConfig()
        assert _config_field_source(cfg, 'summary_start') == 'default'

    def test_config_field_source_overridden_fill_style(self) -> None:
        cfg = DocwrapConfig(fill_style='greedy')
        assert _config_field_source(cfg, 'fill_style') == '[tool.docwrap]'

    def test_config_field_source_overridden_reflow(self) -> None:
        cfg = DocwrapConfig(reflow='full')
        assert _config_field_source(cfg, 'reflow') == '[tool.docwrap]'

    def test_config_field_source_overridden_smart_slack(self) -> None:
        cfg = DocwrapConfig(smart_slack=10)
        assert _config_field_source(cfg, 'smart_slack') == '[tool.docwrap]'

    def test_config_field_source_overridden_summary_start(self) -> None:
        cfg = DocwrapConfig(summary_start='second-line')
        assert _config_field_source(cfg, 'summary_start') == '[tool.docwrap]'

    def test_format_default_builtin(self) -> None:
        assert _format_default('none', 'default') == '(default: none)'

    def test_format_default_with_source(self) -> None:
        assert _format_default(88, '[tool.docwrap]') == '(default: 88, from [tool.docwrap])'


class TestHelpDefaults:
    def test_builtin_defaults(self) -> None:
        help_text = _build_parser().format_help()
        assert _help_contains(help_text, 'Target line length (default: 79).')
        assert _help_contains(help_text, 'Color output (default: auto).')
        assert _help_contains(
            help_text, 'Summary position relative to opening quotes (default: preserve).'
        )
        assert _help_contains(
            help_text, 'Reflow mode: none, cautious, or full (default: cautious).'
        )
        assert _help_contains(
            help_text, 'Fill style: balanced, greedy, or smart (default: smart).'
        )
        assert _help_contains(help_text, 'Smart-fill break-repair budget (default: 8).')

    def test_builtin_wrap_summary_help(self) -> None:
        help_text = _build_parser().format_help()
        assert _help_contains(
            help_text,
            'Wrap summary lines that exceed the target length (default: False).',
        )

    def test_config_file_defaults_in_help(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nreflow = "full"\nline-length = 100\n'
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit, match='0'):
            main(['--help'])
        out = capsys.readouterr().out
        normalized = ' '.join(out.split())
        assert '(default: 100, from [tool.docwrap])' in normalized
        assert '(default: full, from [tool.docwrap])' in normalized

    def test_config_sourced_defaults(self) -> None:
        cfg = ResolvedConfig(
            config=DocwrapConfig(
                fill_style='greedy',
                line_length=88,
                reflow='full',
                smart_slack=10,
                summary_start='second-line',
            ),
            color='never',
            line_length_source=LineLengthSource.DOCWRAP,
        )
        help_text = _build_parser(cfg).format_help()
        assert _help_contains(help_text, 'Target line length (default: 88, from [tool.docwrap]).')
        assert _help_contains(help_text, 'Color output (default: never, from [tool.docwrap]).')
        assert _help_contains(
            help_text,
            'Summary position relative to opening quotes'
            ' (default: second-line, from [tool.docwrap]).',
        )
        assert _help_contains(
            help_text,
            'Reflow mode: none, cautious, or full (default: full, from [tool.docwrap]).',
        )
        assert _help_contains(
            help_text,
            'Fill style: balanced, greedy, or smart (default: greedy, from [tool.docwrap]).',
        )
        assert _help_contains(
            help_text,
            'Smart-fill break-repair budget (default: 10, from [tool.docwrap]).',
        )

    def test_configured_wrap_summary_help(self) -> None:
        cfg = ResolvedConfig(config=DocwrapConfig(wrap_summary=True))
        help_text = _build_parser(cfg).format_help()
        assert _help_contains(
            help_text,
            'Wrap summary lines that exceed the target length'
            ' (default: True, from [tool.docwrap]).',
        )

    def test_docwrap_source_line_length(self) -> None:
        cfg = ResolvedConfig(
            config=DocwrapConfig(line_length=99),
            line_length_source=LineLengthSource.DOCWRAP,
        )
        help_text = _build_parser(cfg).format_help()
        assert _help_contains(help_text, 'Target line length (default: 99, from [tool.docwrap]).')

    def test_hook_options_explain_event_and_failure_policy(self) -> None:
        help_text = cli_mod._build_hook_parser().format_help()
        for description in (
            'Host event name; required only for hosts without an event in the payload.',
            'Enable incremental wrapping at the configured level (default: definition).',
            'Disable incremental wrapping, even when configured.',
            'Allow accepted commands on hook failures; unsupported commits still deny.',
        ):
            assert _help_contains(help_text, description)

    def test_incremental_help(self) -> None:
        help_text = _build_parser().format_help()
        assert _help_contains(
            help_text,
            'Enable incremental mode with the configured level (default when unset: definition).',
        )

    def test_provider_help(self) -> None:
        assert _help_contains(_build_parser().format_help(), 'Agent hook entry point')
        assert _help_contains(
            cli_mod._build_hook_parser().format_help(),
            'Agent hook entry point; reads the host payload as JSON from stdin.',
        )

    def test_stats_options_describe_order_and_limit(self) -> None:
        help_text = _build_parser().format_help()
        assert '--stats-sections SECTION[,SECTION...]' in help_text
        assert '--stats-longest-per-category N' in help_text
        assert _help_contains(
            help_text,
            'Stats body sections, shown once in the listed order.',
        )


class TestHelpFlagOrder:
    def test_help_flags_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit, match='0'):
            main(['--help'])
        out = capsys.readouterr().out
        for flag in ('-p,', '-s,', '-d,', '-j,', '-w,', '-l,', '-v,'):
            assert flag in out
        assert '--show-skips' in out
        assert '--hide-skips' in out

    def test_top_level_help_section_membership(self) -> None:
        help_text = _build_parser().format_help()
        assert _help_section_options(help_text, 'options') == [
            ('-h', '--help'),
            ('--version',),
            ('-p', '--preview'),
            ('-s', '--stats'),
            ('-d', '--diff'),
            ('-j', '--json'),
            ('-w', '--write'),
            ('-l', '--length'),
            ('-i', '--incremental'),
            ('-v', '--verbose'),
            ('--color',),
            ('--no-incremental',),
            ('--show-skips',),
            ('--hide-skips',),
        ]
        assert _help_section_options(help_text, 'style options') == [
            ('--style',),
            ('--convert',),
            ('--strip-types',),
        ]
        assert _help_section_options(help_text, 'wrapping policy options') == [
            ('--incremental-level',),
            ('--summary-start',),
            ('--wrap-summary', '--no-wrap-summary'),
            ('--fill-style',),
            ('--smart-slack',),
            ('--reflow',),
        ]
        assert _help_section_options(help_text, 'statistics options') == [
            ('--stats-sections',),
            ('--stats-longest-per-category',),
        ]

    def test_top_level_help_sections(self) -> None:
        help_text = _build_parser().format_help()
        headings = [
            line.removesuffix(':')
            for line in help_text.splitlines()
            if line and not line.startswith(' ') and line.endswith(':')
        ]
        assert headings == [
            'positional arguments',
            'options',
            'wrapping policy options',
            'style options',
            'statistics options',
            'subcommands',
        ]

    def test_top_level_usage_follows_section_priority(self) -> None:
        usage = _build_parser().format_help().partition('\n\n')[0]
        assert re.findall(r'(?<![\w-])--?[a-z][a-z-]*', usage) == [
            '-h',
            '--version',
            '-p',
            '-s',
            '-d',
            '-j',
            '-w',
            '-l',
            '-i',
            '-v',
            '--color',
            '--no-incremental',
            '--show-skips',
            '--hide-skips',
            '--incremental-level',
            '--summary-start',
            '--wrap-summary',
            '--no-wrap-summary',
            '--fill-style',
            '--smart-slack',
            '--reflow',
            '--style',
            '--convert',
            '--strip-types',
            '--stats-sections',
            '--stats-longest-per-category',
        ]


@pytest.mark.usefixtures('registered_hosts')
class TestHookStdin:
    def test_hook_avoids_cli_only_imports(self, distribution_root: Path) -> None:
        src_dir = str(Path(__file__).resolve().parent.parent / 'src')
        script = (
            'import sys\n'
            'from docwrap._cli import main\n'
            "status = main(['hook', '--agent', 'claude'])\n"
            'print(status)\n'
            "print('docwrap._display' in sys.modules)\n"
            "print('docwrap._stats' in sys.modules)\n"
        )
        result = run_setup(
            [sys.executable, '-c', script],
            stdin=b'{}',
            check=False,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join([src_dir, str(distribution_root)])},
        )
        assert result.returncode == 0, process_transcript(result)
        assert result.stderr == b''
        assert decode_text(result.stdout).splitlines() == ['0', 'False', 'False']

    def test_stdin_ingestion(self, tmp_path: Path, distribution_root: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        before = f.read_text()
        event = json.dumps(
            {
                'hook_event_name': 'PostToolUse',
                'tool_name': 'Edit',
                'tool_input': {'file_path': str(f)},
            }
        )
        src_dir = str(Path(__file__).resolve().parent.parent / 'src')
        env = {
            **os.environ,
            'PYTHONPATH': os.pathsep.join([src_dir, str(distribution_root)]),
        }
        result = run_setup(
            [
                sys.executable,
                '-c',
                (
                    'from docwrap._cli import main; import sys; '
                    'sys.exit(main(["hook", "--agent", "claude"]))'
                ),
            ],
            stdin=event.encode('utf-8'),
            check=False,
            env=env,
        )
        assert result.returncode == 0, process_transcript(result)
        # File should be modified by the hook
        assert f.read_text() != before


class TestHookSubcommand:
    @pytest.fixture(autouse=True)
    def _enable_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def hook_enabled(_cwd: str | None = None) -> bool:
            return True

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', hook_enabled)

    @pytest.mark.parametrize('flag', ['-a', '--agent'])
    def test_accepts_agent_selection(self, flag: str) -> None:
        assert cli_mod._build_hook_parser().parse_args([flag, 'codex']).agent == 'codex'

    @pytest.mark.parametrize('flag', ['-e', '--event'])
    def test_accepts_event_selection(self, flag: str) -> None:
        args = cli_mod._build_hook_parser().parse_args(['-a', 'agy', flag, 'PreToolUse'])
        assert args.event == 'PreToolUse'

    def test_main_hook_allow_on_failure_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def tracking_run_hook(
            _stdin_data: str,
            *,
            host: str,
            event: str | None = None,
            allow_on_failure: bool = False,
            incremental: bool | None = None,  # noqa: ARG001
        ) -> int:
            captured['host'] = host
            captured['event'] = event
            captured['allow_on_failure'] = allow_on_failure
            return 0

        monkeypatch.setattr('docwrap._hook.run_hook', tracking_run_hook)
        monkeypatch.setattr('sys.stdin', io.StringIO('{}'))
        main(['hook', '--agent', 'claude', '--allow-on-failure'])
        assert captured['allow_on_failure'] is True

    @pytest.mark.usefixtures('registered_hosts')
    def test_main_hook_dispatches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        before = f.read_text()
        event = json.dumps(
            {
                'hook_event_name': 'PostToolUse',
                'tool_name': 'Edit',
                'tool_input': {'file_path': str(f)},
            }
        )
        monkeypatch.setattr('sys.stdin', io.StringIO(event))
        result = main(['hook', '--agent', 'claude'])
        assert result == 0
        assert f.read_text() != before

    def test_main_hook_incremental_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def tracking_run_hook(
            _stdin_data: str,
            *,
            host: str,
            event: str | None = None,
            allow_on_failure: bool = False,  # noqa: ARG001
            incremental: bool | None = None,
        ) -> int:
            captured['host'] = host
            captured['event'] = event
            captured['incremental'] = incremental
            return 0

        monkeypatch.setattr('docwrap._hook.run_hook', tracking_run_hook)
        monkeypatch.setattr('sys.stdin', io.StringIO('{}'))
        main(['hook', '--agent', 'claude', '-i'])
        assert captured['incremental'] is True
        assert captured['host'] == 'claude'
        assert captured['event'] is None

    def test_main_hook_mutual_exclusion(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(['hook', '--agent', 'claude', '-i', '--no-incremental'])
        assert exc_info.value.code == 2

    def test_main_hook_no_incremental_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def tracking_run_hook(
            _stdin_data: str,
            *,
            host: str,
            event: str | None = None,
            allow_on_failure: bool = False,  # noqa: ARG001
            incremental: bool | None = None,
        ) -> int:
            captured['host'] = host
            captured['event'] = event
            captured['incremental'] = incremental
            return 0

        monkeypatch.setattr('docwrap._hook.run_hook', tracking_run_hook)
        monkeypatch.setattr('sys.stdin', io.StringIO('{}'))
        main(['hook', '--agent', 'claude', '--no-incremental'])
        assert captured['incremental'] is False
        assert captured['host'] == 'claude'
        assert captured['event'] is None

    def test_main_hook_tty_prints_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tty_stdin = io.StringIO()
        tty_stdin.isatty = lambda: True
        monkeypatch.setattr('sys.stdin', tty_stdin)
        result = main(['hook', '--agent', 'claude'])
        assert result == 0
        captured = capsys.readouterr()
        assert 'reads the host hook payload as JSON from stdin' in captured.out

    def test_module_entrypoint_requires_explicit_agent(self) -> None:
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        result = run_cli(['hook'], stdin='{}', env=env, expect=2)
        assert '--agent' in result.stderr

    @pytest.mark.parametrize(
        'args',
        [['--host', 'codex'], ['--agent', 'codex', '--host', 'codex']],
    )
    def test_rejects_host_option(self, args: list[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_mod._build_hook_parser().parse_args(args)
        assert exc_info.value.code == 2


class TestIncrementalAbsolutePath:
    def test_absolute_path_not_misclassified_as_untracked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'sub' / 'foo.py'
        f.parent.mkdir()
        # A docstring that needs wrapping -- if the file is misclassified as
        # untracked, incremental mode treats it as new and wraps it (exit 1).
        # A correctly classified tracked-unchanged file is skipped (exit 0).
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
            ],
            cwd=str(tmp_path),
        )
        assert main([str(f)]) == 1
        capsys.readouterr()

        # Tracked, unchanged -- incremental should skip it entirely (exit 0).
        # Must also NOT emit "not in a git repository" warning -- that would
        # mean the repo was not found (git commands likely ran from wrong cwd).
        exit_code = main(['-i', str(f)])
        captured = capsys.readouterr()
        assert 'not in a git repository' not in captured.err, (
            'git repo not detected (git commands likely ran from wrong cwd)'
        )
        assert exit_code == 0, (
            'tracked unchanged file was processed (likely misclassified as untracked '
            'due to absolute vs relative path mismatch)'
        )


class TestIncrementalCommand:
    @pytest.mark.parametrize('json_mode', [False, True])
    def test_implicit_diff_failure_diagnostic_is_protocol_aware(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        json_mode: bool,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "definition"\n')
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')

        def discover(_path: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', _no_diff)

        flags = ['--json'] if json_mode else []
        exit_code = main([*flags, str(f)])
        captured = capsys.readouterr()
        assert exit_code == 0
        if json_mode:
            assert json.loads(captured.out)['files'][0]['status'] == 'unchanged'
            assert captured.err == ''
        else:
            assert captured.err == (
                f'warning: git diff failed for {tmp_path}, falling back to full mode\n'
            )

    def test_incremental_skips_non_git(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'foo.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        assert main([str(f)]) == 1
        capsys.readouterr()
        exit_code = main(['-i', str(f)])
        assert exit_code == 0  # Skipped (per spec: skip + warning, not error)
        captured = capsys.readouterr()
        assert 'not in a git repository' in captured.err

    @pytest.mark.parametrize('json_mode', [False, True])
    def test_non_repository_diagnostic_is_protocol_aware(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        json_mode: bool,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "definition"\n')
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        monkeypatch.setattr(cli_mod, 'discover_repo_root', _not_worktree)

        flags = ['--json'] if json_mode else []
        exit_code = main([*flags, str(f)])
        captured = capsys.readouterr()
        assert exit_code == 0
        if json_mode:
            assert json.loads(captured.out) == {
                'version': 1,
                'files': [],
                'total_skips': 0,
            }
            assert captured.err == ''
        else:
            assert captured.err == (
                'warning: not in a git repository, skipping incremental files\n'
            )


class TestIncrementalConfigPrecedence:
    def test_cli_flag_overrides_config(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "definition"\n')
        f = tmp_path / 'foo.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)

        # --no-incremental should override config
        exit_code = main(['--write', '--no-incremental', str(f)])
        assert exit_code == 0
        assert f.read_text() != _BODY_WRAPPABLE_SOURCE

    @pytest.mark.parametrize(
        'flag',
        [
            ['--no-incremental'],
            ['--incremental-level', 'file'],
        ],
    )
    def test_explicit_level_options_bypass_incremental_config_lookup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        flag: list[str],
    ) -> None:
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')

        def unexpected_lookup(_path: Path) -> IncrementalLevel | bool | None:
            raise AssertionError('incremental config lookup must be bypassed')

        monkeypatch.setattr(cli_mod, 'find_incremental_config', unexpected_lookup)
        monkeypatch.setattr(cli_mod, 'discover_repo_root', _not_worktree)
        assert main([*flag, str(f)]) == 0

    @pytest.mark.parametrize(
        ('configured', 'expected'),
        [('docstring', 'docstring'), (None, 'definition')],
    )
    def test_incremental_flag_uses_configured_level_or_definition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        configured: IncrementalLevel | None,
        expected: IncrementalLevel,
    ) -> None:
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        captured: dict[str, object] = {}

        def configured_lookup(_path: Path) -> IncrementalLevel | bool | None:
            return configured

        monkeypatch.setattr(cli_mod, 'find_incremental_config', configured_lookup)
        monkeypatch.setattr(cli_mod, 'discover_repo_root', _not_worktree)

        def tracking_compute(
            files: list[Path],
            incr_level: IncrementalLevel | None,
            incr_explicit: bool,
        ) -> _IncrementalOutcome:
            captured['level'] = incr_level
            captured['explicit'] = incr_explicit
            return _compute_incremental_context(files, incr_level, incr_explicit)

        monkeypatch.setattr(cli_mod, '_compute_incremental_context', tracking_compute)
        assert main(['-i', str(f)]) == 0
        assert captured == {'level': expected, 'explicit': True}


class TestIncrementalConfiguration:
    def test_configured_git_failure_falls_back_for_affected_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', _no_diff)

        outcome = _compute_incremental_context(
            [path],
            'definition',
            incr_explicit=False,
        )

        assert outcome.context is not None
        assert path.resolve() in outcome.context.diff_failed
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.fatal is False
        assert _compute_file_plan(path, outcome.context).state == 'full'

    def test_configured_tracked_failure_falls_back_for_affected_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def empty_diff(_root: str) -> str:
            return ''

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        def tracked_failure(_root: str, _paths: Collection[str]) -> None:
            return None

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', empty_diff)
        monkeypatch.setattr(cli_mod, 'get_tracked_files', tracked_failure)

        outcome = _compute_incremental_context([path], 'definition', incr_explicit=False)

        assert outcome.context is not None
        assert path.resolve() in outcome.context.diff_failed
        assert _compute_file_plan(path, outcome.context).state == 'full'
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.message == (
            f'git tracked-file discovery failed for {tmp_path}, falling back to full mode'
        )

    def test_explicit_git_failure_remains_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', _no_diff)

        outcome = _compute_incremental_context(
            [path],
            'definition',
            incr_explicit=True,
        )

        assert outcome.context is None
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.fatal is True

    def test_explicit_malformed_diff_path_remains_fatal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        def malformed_diff(_root: str) -> str:
            return (
                'diff --git ignored ignored\n'
                '--- "a/bad\\q.py"\n'
                '+++ "b/bad\\q.py"\n'
                '@@ -1 +1 @@\n'
                '+x\n'
            )

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', malformed_diff)

        outcome = _compute_incremental_context([path], 'definition', incr_explicit=True)

        assert outcome.context is None
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.message == 'git state collection failed for incremental mode'
        assert outcome.diagnostic.fatal is True

    def test_explicit_tracked_failure_remains_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def empty_diff(_root: str) -> str:
            return ''

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        def tracked_failure(_root: str, _paths: Collection[str]) -> None:
            return None

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', empty_diff)
        monkeypatch.setattr(cli_mod, 'get_tracked_files', tracked_failure)

        outcome = _compute_incremental_context([path], 'definition', incr_explicit=True)

        assert outcome.context is None
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.message == 'git state collection failed for incremental mode'
        assert outcome.diagnostic.fatal is True

    def test_implicit_malformed_diff_path_falls_back_for_affected_repository(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('x = 1\n')

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        def malformed_diff(_root: str) -> str:
            return (
                'diff --git ignored ignored\n'
                '--- "a/bad\\q.py"\n'
                '+++ "b/bad\\q.py"\n'
                '@@ -1 +1 @@\n'
                '+x\n'
            )

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', malformed_diff)

        outcome = _compute_incremental_context([path], 'definition', incr_explicit=False)

        assert outcome.context is not None
        assert path.resolve() in outcome.context.diff_failed
        assert _compute_file_plan(path, outcome.context).state == 'full'
        assert outcome.diagnostic is not None
        assert outcome.diagnostic.message == (
            f'git diff failed for {tmp_path}, falling back to full mode'
        )


class TestIncrementalErrors:
    @pytest.mark.parametrize(
        'scenario',
        [
            (False, 'read', 'read failed'),
            (True, 'read', 'read failed'),
            (False, 'decode', 'cannot decode'),
            (True, 'decode', 'cannot decode'),
            (False, 'syntax', 'invalid syntax'),
            (True, 'syntax', 'invalid syntax'),
            (False, 'token', 'token failed'),
            (True, 'token', 'token failed'),
        ],
        ids=[
            'explicit-read',
            'configured-read',
            'explicit-decode',
            'configured-decode',
            'explicit-syntax',
            'configured-syntax',
            'explicit-token',
            'configured-token',
        ],
    )
    def test_file_failure_is_reported_once(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        scenario: tuple[bool, str, str],
    ) -> None:
        configured, case, expected_fragment = scenario
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case=case,
            configured=configured,
        )
        flags = [] if configured else ['--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        assert exit_code == 2
        assert captured.out == ''
        assert captured.err.startswith(f'error: {path}: ')
        assert expected_fragment in captured.err
        assert reads == {'count': 1}

    @pytest.mark.parametrize('configured', [False, True], ids=['explicit', 'configured'])
    def test_file_level_invalid_source_stops_before_wrapping(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        configured: bool,
    ) -> None:
        path, downstream = _prepare_invalid_file_level_case(
            tmp_path,
            monkeypatch,
            configured=configured,
        )
        flags = [] if configured else ['--incremental-level', 'file']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        assert exit_code == 2
        assert captured.out == ''
        assert captured.err.startswith(f'error: {path}: ')
        assert 'invalid syntax' in captured.err
        downstream.assert_not_called()

    @pytest.mark.parametrize(
        'scenario',
        [
            (False, 'no-hunks', 0),
            (True, 'no-hunks', 0),
            (False, 'no-eligible', 1),
            (True, 'no-eligible', 1),
        ],
        ids=[
            'explicit-no-hunks',
            'configured-no-hunks',
            'explicit-no-eligible',
            'configured-no-eligible',
        ],
    )
    def test_no_work_is_silent_success(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        scenario: tuple[bool, str, int],
    ) -> None:
        configured, case, expected_reads = scenario
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case=case,
            configured=configured,
        )
        flags = [] if configured else ['--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ''
        assert captured.err == ''
        assert reads == {'count': expected_reads}

    def test_ranged_write_reads_once_and_preserves_crlf(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        path = tmp_path / 'a.py'
        path.write_bytes(b'def f():\r\n    """Short."""\r\n')
        git_commit_all(tmp_path, 'init')
        changed = _BODY_WRAPPABLE_SOURCE.replace('\n', '\r\n').encode()
        path.write_bytes(changed)

        original_read_file = cli_mod.read_file
        original_read_bytes = Path.read_bytes
        original_write_bytes = Path.write_bytes
        reads = {'bytes': 0, 'source': 0}
        writes = {'count': 0}
        resolved = path.resolve()

        def tracked_read_file(candidate: Path) -> tuple[str, str]:
            if candidate.resolve() == resolved:
                reads['source'] += 1
            return original_read_file(candidate)

        def tracked_read_bytes(candidate: Path) -> bytes:
            if candidate.resolve() == resolved:
                reads['bytes'] += 1
            return original_read_bytes(candidate)

        def tracked_write_bytes(candidate: Path, data: bytes) -> int:
            if candidate.resolve() == resolved:
                writes['count'] += 1
            return original_write_bytes(candidate, data)

        monkeypatch.setattr(cli_mod, 'read_file', tracked_read_file)
        monkeypatch.setattr(Path, 'read_bytes', tracked_read_bytes)
        monkeypatch.setattr(Path, 'write_bytes', tracked_write_bytes)
        args = ['--write', '--incremental-level', 'docstring', str(path)]

        assert main(args) == 0
        capsys.readouterr()
        first = original_read_bytes(path)
        assert sum(reads.values()) == 1
        assert writes == {'count': 1}
        assert first != changed
        assert b'\r\n' in first
        assert b'\n' not in first.replace(b'\r\n', b'')

        reads = {'bytes': 0, 'source': 0}
        writes = {'count': 0}
        assert main(args) == 0
        capsys.readouterr()
        assert sum(reads.values()) == 1
        assert writes == {'count': 0}
        assert original_read_bytes(path) == first

    @pytest.mark.parametrize('configured', [False, True], ids=['explicit', 'configured'])
    def test_ready_ranges_reuse_single_read(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        configured: bool,
    ) -> None:
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case='ranges',
            configured=configured,
        )
        flags = [] if configured else ['--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ''
        assert captured.err == ''
        assert reads == {'count': 1}


class TestIncrementalFlags:
    def test_incremental_flag_accepted(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['-i', '.'])
        assert args.incremental is True

    def test_incremental_level_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--incremental-level', 'file', '.'])
        assert args.incremental_level == 'file'

    def test_level_and_no_incremental_conflict(self) -> None:
        with pytest.raises(SystemExit):
            main(['--incremental-level', 'file', '--no-incremental', '.'])

    def test_level_implies_incremental(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--incremental-level', 'docstring', '.'])
        assert args.incremental_level == 'docstring'

    def test_no_incremental_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--no-incremental', '.'])
        assert args.no_incremental is True


class TestIncrementalPathScoping:
    def test_directory_discovery_excludes_ignored_files(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        (tmp_path / '.gitignore').write_text('ignored.py\n')
        tracked = tmp_path / 'tracked.py'
        tracked.write_text(_BODY_WRAPPABLE_SOURCE)
        git_commit_all(tmp_path, 'baseline')
        ignored = tmp_path / 'ignored.py'
        ignored.write_text(_BODY_WRAPPABLE_SOURCE)
        assert main(['--write', '--incremental', str(tmp_path)]) == 0
        assert ignored.read_text() == _BODY_WRAPPABLE_SOURCE
        assert tracked.read_text() == _BODY_WRAPPABLE_SOURCE

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='POSIX bytes-path behavior is not available on Windows',
    )
    def test_explicit_file_ignores_unrelated_non_utf8_modified_path(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        target = tmp_path / 'requested.py'
        target.write_text('def requested():\n    """Short."""\n')
        unrelated_bytes = b'unrelated-\xff.py'
        unrelated_path = os.fsencode(tmp_path / os.fsdecode(unrelated_bytes))
        try:
            descriptor = os.open(
                unrelated_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
        except OSError as exc:
            if exc.errno == errno.EILSEQ:
                pytest.skip('filesystem cannot create non-UTF-8 filenames')
            raise
        try:
            os.write(descriptor, b'x = 1\n')
        finally:
            os.close(descriptor)
        git_commit_all(tmp_path, 'init')
        original = _BODY_WRAPPABLE_SOURCE.replace('def foo', 'def requested')
        target.write_text(original)
        descriptor = os.open(unrelated_path, os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(descriptor, b'x = 2\n')
        finally:
            os.close(descriptor)

        exit_code = main(['--write', '--incremental', str(target)])

        assert exit_code == 0
        assert target.read_text() != original

    @pytest.mark.parametrize('ignored', [False, True])
    def test_explicit_untracked_file_is_eligible(self, tmp_path: Path, ignored: bool) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        (tmp_path / '.gitignore').write_text('new.py\n' if ignored else '')
        git_commit_all(tmp_path, 'baseline')
        target = tmp_path / 'new.py'
        target.write_text(_BODY_WRAPPABLE_SOURCE)
        assert main(['--write', '--incremental', str(target)]) == 0
        assert target.read_text() != _BODY_WRAPPABLE_SOURCE

    @pytest.mark.parametrize('level', ['definition', 'file'])
    def test_rename_preserves_unchanged_definition(
        self, tmp_path: Path, level: IncrementalLevel
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        old = tmp_path / 'old.py'
        first = _BODY_WRAPPABLE_SOURCE + '    return 1\n'
        untouched = first.replace('def foo', 'def untouched')
        old.write_text(first + '\n\n' + untouched)
        unrelated = tmp_path / 'unselected.py'
        unrelated.write_text(first)
        git_commit_all(tmp_path, 'baseline')
        target = tmp_path / 'new.py'
        old.rename(target)
        changed = first.replace('return 1', 'return 2')
        target.write_text(changed + '\n\n' + untouched)
        run_setup(['git', 'add', '-A'], cwd=tmp_path)
        unrelated.write_text(changed)
        rename = run_setup(
            ['git', 'diff', '--cached', '--name-status', '-M'],
            cwd=tmp_path,
        )
        status, source_name, destination_name = decode_text(rename.stdout).strip().split('\t')
        assert status.startswith('R')
        assert int(status[1:]) >= 90
        assert (source_name, destination_name) == ('old.py', 'new.py')

        outcome = _compute_incremental_context([target], level, incr_explicit=True)
        assert outcome.context is not None
        plan = _compute_file_plan(target, outcome.context)
        if level == 'definition':
            assert plan.eligible_ranges == {(1, len(changed.splitlines()))}
        else:
            assert plan.state == 'full'
        assert main(['--write', '--incremental-level', level, str(target)]) == 0
        result = target.read_text()
        assert result.split('def untouched')[0] != changed + '\n\n'
        assert (result.endswith(untouched)) is (level == 'definition')
        assert unrelated.read_text() == changed

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='Windows cannot represent a backslash as a filename character',
    )
    def test_tracked_quoted_backslash_filename_wraps_changed_definition(
        self,
        tmp_path: Path,
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        target = tmp_path / 'slash\\name.py'
        target.write_text('def requested():\n    """Short."""\n')
        git_commit_all(tmp_path, 'init')
        original = _BODY_WRAPPABLE_SOURCE.replace('def foo', 'def requested')
        target.write_text(original)

        exit_code = main(['--write', '--incremental', str(target)])

        assert exit_code == 0
        assert target.read_text() != original


class TestIncrementalRepositoryDiscovery:
    @pytest.mark.parametrize('marker', ['directory', 'file'])
    def test_nested_and_unrelated_worktrees_keep_distinct_roots(
        self,
        git_repo: Path,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        marker: str,
    ) -> None:
        nested = git_repo / 'nested'
        unrelated = tmp_path_factory.mktemp('unrelated')
        if marker == 'file':
            run_setup(
                ['git', 'worktree', 'add', '--detach', str(nested), 'HEAD'],
                cwd=git_repo,
            )
        else:
            nested.mkdir()
            run_setup(['git', 'init'], cwd=nested)
        run_setup(['git', 'init'], cwd=unrelated)
        paths = [git_repo / 'top.py', nested / 'a.py', nested / 'sub' / 'b.py', unrelated / 'c.py']
        (nested / 'sub').mkdir()
        discover = Mock(wraps=cli_mod.discover_repo_root)
        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)

        roots, groups, failed = _discover_incremental_repositories(paths)

        assert {
            path: Path(root) if root is not None else None for path, root in roots.items()
        } == dict(zip(paths, [git_repo, nested, nested, unrelated], strict=True))
        assert list(map(Path, groups)) == [git_repo, nested, unrelated]
        assert not failed
        assert discover.call_count == 3

    def test_incremental_repo_discovery_error_keeps_configured_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(
            cli_mod,
            'get_repo_diff',
            Mock(side_effect=AssertionError('Git diff called after discovery error')),
        )
        monkeypatch.setattr(
            cli_mod,
            'get_tracked_files',
            Mock(side_effect=AssertionError('tracked lookup called after discovery error')),
        )
        outcome = _compute_incremental_context([path], 'definition', incr_explicit=False)

        assert calls == [str(tmp_path.resolve())]
        assert outcome.context is not None
        assert outcome.context.repo_roots == {path.resolve(): None}
        assert outcome.context.diff_failed == {path.resolve()}
        assert _compute_file_plan(path, outcome.context).state == 'full'
        assert outcome.diagnostic == _IncrementalDiagnostic(
            'git repository discovery failed, falling back to full mode'
        )

    def test_incremental_repo_discovery_error_keeps_fatal_decision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(
            cli_mod,
            'get_repo_diff',
            Mock(side_effect=AssertionError('Git diff called after discovery error')),
        )
        monkeypatch.setattr(
            cli_mod,
            'get_tracked_files',
            Mock(side_effect=AssertionError('tracked lookup called after discovery error')),
        )
        outcome = _compute_incremental_context([path], 'definition', incr_explicit=True)

        assert calls == [str(tmp_path.resolve())]
        assert outcome.context is None
        assert outcome.diagnostic == _IncrementalDiagnostic(
            'git state collection failed for incremental mode', fatal=True
        )

    def test_incremental_repo_discovery_not_worktree_keeps_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'a.py'
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        outcome = _compute_incremental_context([path], 'definition', incr_explicit=True)

        assert calls == [str(tmp_path.resolve())]
        assert outcome.context is not None
        assert outcome.context.repo_roots == {path.resolve(): None}
        assert outcome.diagnostic == _IncrementalDiagnostic(
            'not in a git repository, skipping incremental files'
        )

    def test_incremental_repo_discovery_once_for_shared_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = tmp_path / 'a.py'
        second = tmp_path / 'b.py'
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        repo_roots, by_root, discovery_failed = _discover_incremental_repositories([first, second])

        assert calls == [str(tmp_path.resolve())]
        assert repo_roots == {
            first.resolve(): str(tmp_path),
            second.resolve(): str(tmp_path),
        }
        assert by_root == {str(tmp_path): [first.resolve(), second.resolve()]}
        assert discovery_failed == set()

    def test_incremental_repo_discovery_reuses_root_across_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = tmp_path / 'first' / 'a.py'
        second = tmp_path / 'second' / 'b.py'
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        repo_roots, by_root, discovery_failed = _discover_incremental_repositories([first, second])

        assert calls == [str(first.parent.resolve())]
        assert repo_roots == {
            first.resolve(): str(tmp_path),
            second.resolve(): str(tmp_path),
        }
        assert by_root == {str(tmp_path): [first.resolve(), second.resolve()]}
        assert discovery_failed == set()

    @pytest.mark.parametrize(
        'status', [RepoDiscoveryStatus.ERROR, RepoDiscoveryStatus.NOT_WORKTREE]
    )
    def test_failed_or_absent_discovery_does_not_hide_descendant_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: RepoDiscoveryStatus
    ) -> None:
        nested = tmp_path / 'nested'
        paths = [tmp_path / 'a.py', nested / 'b.py']
        discover = Mock(
            side_effect=[
                RepoDiscoveryResult(status),
                RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(nested)),
            ]
        )
        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)

        roots, groups, failed = _discover_incremental_repositories(paths)

        assert roots == {paths[0]: None, paths[1]: str(nested)}
        assert groups == {str(nested): [paths[1]]}
        assert failed == ({paths[0]} if status is RepoDiscoveryStatus.ERROR else set())
        assert discover.call_count == 2

    def test_unchanged_files_across_many_directories_discover_once(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths: list[Path] = []
        for index in range(80):
            path = git_repo / f'dir-{index}' / 'source.py'
            path.parent.mkdir()
            path.write_text('pass\n')
            paths.append(path)
        git_commit_all(git_repo, 'unchanged files')
        discover = Mock(wraps=cli_mod.discover_repo_root)
        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)

        outcome = _compute_incremental_context(paths, 'definition', incr_explicit=True)

        assert outcome.context is not None
        assert outcome.diagnostic is None
        assert all(_compute_file_plan(path, outcome.context).state == 'no-hunks' for path in paths)
        assert discover.call_count == 1


class TestInvalidConfigEntrypoints:
    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        assert main([str(empty)]) == 0

    @pytest.mark.parametrize('mode', [[], ['--preview'], ['--diff'], ['--write']])
    def test_invalid_config_style_warns_and_processes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        mode: list[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstyle = "bogus"\n')
        f = tmp_path / 'a.py'
        f.write_text('x = 1\n')
        assert main([*mode, str(f)]) == 0
        captured = capsys.readouterr()
        assert captured.err == "warning: invalid [tool.docwrap].style value 'bogus', using None\n"
        assert 'Traceback' not in captured.err


class TestJsonIncrementalErrors:
    @pytest.mark.parametrize(
        'scenario',
        [
            (False, 'read', 'error', 'read failed'),
            (True, 'read', 'error', 'read failed'),
            (False, 'decode', 'error', 'cannot decode'),
            (True, 'decode', 'error', 'cannot decode'),
            (False, 'syntax', 'invalid', 'invalid syntax'),
            (True, 'syntax', 'invalid', 'invalid syntax'),
            (False, 'token', 'invalid', 'token failed'),
            (True, 'token', 'invalid', 'token failed'),
        ],
        ids=[
            'explicit-read',
            'configured-read',
            'explicit-decode',
            'configured-decode',
            'explicit-syntax',
            'configured-syntax',
            'explicit-token',
            'configured-token',
        ],
    )
    def test_file_failure_is_structured_once(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        scenario: tuple[bool, str, str, str],
    ) -> None:
        configured, case, expected_status, expected_fragment = scenario
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case=case,
            configured=configured,
        )
        flags = ['--json'] if configured else ['--json', '--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        record = json.loads(captured.out)['files'][0]
        assert exit_code == 2
        assert record['path'] == str(path)
        assert record['status'] == expected_status
        assert expected_fragment in record['warning']
        assert captured.err == ''
        assert reads == {'count': 1}

    @pytest.mark.parametrize('configured', [False, True], ids=['explicit', 'configured'])
    def test_file_level_invalid_source_stops_before_wrapping(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        configured: bool,
    ) -> None:
        path, downstream = _prepare_invalid_file_level_case(
            tmp_path,
            monkeypatch,
            configured=configured,
        )
        flags = ['--json'] if configured else ['--json', '--incremental-level', 'file']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        record = json.loads(captured.out)['files'][0]
        assert exit_code == 2
        assert record['path'] == str(path)
        assert record['status'] == 'invalid'
        assert 'invalid syntax' in record['warning']
        assert captured.err == ''
        downstream.assert_not_called()

    @pytest.mark.parametrize(
        'scenario',
        [
            (False, 'no-hunks', 0),
            (True, 'no-hunks', 0),
            (False, 'no-eligible', 1),
            (True, 'no-eligible', 1),
        ],
        ids=[
            'explicit-no-hunks',
            'configured-no-hunks',
            'explicit-no-eligible',
            'configured-no-eligible',
        ],
    )
    def test_no_work_is_empty_success(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        scenario: tuple[bool, str, int],
    ) -> None:
        configured, case, expected_reads = scenario
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case=case,
            configured=configured,
        )
        flags = ['--json'] if configured else ['--json', '--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert json.loads(captured.out) == {
            'version': 1,
            'files': [],
            'total_skips': 0,
        }
        assert captured.err == ''
        assert reads == {'count': expected_reads}

    @pytest.mark.parametrize('configured', [False, True], ids=['explicit', 'configured'])
    def test_ready_ranges_reuse_single_read(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        configured: bool,
    ) -> None:
        path, reads = _prepare_incremental_file_case(
            tmp_path,
            monkeypatch,
            case='ranges',
            configured=configured,
        )
        flags = ['--json'] if configured else ['--json', '--incremental-level', 'docstring']

        exit_code = main([*flags, str(path)])

        captured = capsys.readouterr()
        record = json.loads(captured.out)['files'][0]
        assert exit_code == 0
        assert record['status'] == 'unchanged'
        assert captured.err == ''
        assert reads == {'count': 1}


class TestJsonMode:
    def test_change_record_always_has_skip_fields(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text(
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', str(p)])
        change = json.loads(buf.getvalue())['files'][0]['changes'][0]
        assert 'summary_skipped' in change
        assert 'summary_width' in change

    def test_changed_record_always_has_before_after(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text(
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', str(p)])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['status'] == 'changed'
        change = rec['changes'][0]
        assert change['before'] is not None
        assert change['after'] is not None
        assert 'qualname' in change
        assert 'start_line' in change
        assert 'end_line' in change

    def test_empty_directory_returns_empty_envelope(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        assert main(['--json', str(empty)]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {
            'version': 1,
            'files': [],
            'total_skips': 0,
        }
        assert captured.err == ''

    def test_explicit_incremental_diff_failure_is_structured(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / 'a.py'
        f.write_text('x = 1\n')

        def discover(_path: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', _no_diff)

        assert main(['--json', '-i', str(f)]) == 2
        captured = capsys.readouterr()
        record = json.loads(captured.out)['files'][0]
        assert record['path'] == str(f)
        assert record['status'] == 'error'
        assert record['warning'] == 'git state collection failed for incremental mode'
        assert captured.err == ''

    def test_invalid_file_record(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('def f(:\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', str(p)])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['status'] == 'invalid'
        assert rec['warning'] is not None

    def test_invalid_style_warns_after_incremental_setup(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nincremental = "definition"\nstyle = "bogus"\n'
        )
        f = tmp_path / 'a.py'
        f.write_text('x = 1\n')
        setup_calls: list[str] = []

        def discover(_path: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)

        def track_diff(_path: str) -> str:
            setup_calls.append('diff')
            return ''

        monkeypatch.setattr(cli_mod, 'get_repo_diff', track_diff)
        monkeypatch.setattr(cli_mod, 'get_tracked_files', _no_tracked_files)

        assert main(['--json', str(f)]) == 0
        captured = capsys.readouterr()
        assert setup_calls == ['diff']
        record = json.loads(captured.out)['files'][0]
        assert record['status'] == 'unchanged'
        assert record['warning'] is None
        assert captured.err == (
            "warning: invalid [tool.docwrap].style value 'bogus', using None\n"
        )

    def test_invalid_style_warns_and_emits_normal_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstyle = "bogus"\n')
        f = tmp_path / 'a.py'
        f.write_text('x = 1\n')
        assert main(['--json', str(f)]) == 0
        captured = capsys.readouterr()
        records = json.loads(captured.out)['files']
        assert len(records) == 1
        assert records[0]['status'] == 'unchanged'
        assert records[0]['warning'] is None
        assert captured.err == (
            "warning: invalid [tool.docwrap].style value 'bogus', using None\n"
        )

    def test_json_exit_0_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        assert main(['-j', str(p)]) == 0

    def test_json_exit_1_changed(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text(
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        assert main(['-j', str(p)]) == 1

    def test_json_exit_2_invalid(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('def f(:\n')
        assert main(['-j', str(p)]) == 2

    def test_json_exit_2_precedence(self, tmp_path: Path) -> None:
        good = tmp_path / 'good.py'
        good.write_text(
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        bad = tmp_path / 'bad.py'
        bad.write_text('def f(:\n')
        assert main(['-j', str(good), str(bad)]) == 2

    def test_json_exit_2_unreadable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')

        def unreadable(_path: Path) -> ReadResult:
            raise PermissionError('permission denied')

        monkeypatch.setattr(cli_mod, 'read_file_result', unreadable)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = main(['-j', str(p)])

        assert exit_code == 2
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['status'] == 'error'
        assert rec['warning'] == 'permission denied'
        assert rec['docstrings'] == {'wrapped': 0, 'total': 0, 'skipped_summaries': 0}
        assert rec['line_length'] is None
        assert rec['style'] is None

    def test_json_mutually_exclusive_with_diff(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--json', '--diff', '.'])

    def test_json_no_stderr(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / 'a.py'
        p.write_text('def f(:\n')
        main(['-j', str(p)])
        assert capsys.readouterr().err == ''

    def test_json_short_flag(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        result = main(['-j', str(p)])
        assert result == 0

    def test_no_ansi_in_json(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', '--color', 'always', str(p)])
        output = buf.getvalue()
        assert '\033[' not in output

    def test_path_relative_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        monkeypatch.chdir(tmp_path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', 'a.py'])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['path'] == 'a.py'

    def test_skipped_summaries_in_json(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text(
            'def f():\n'
            '    """This summary is longer than forty characters by quite a bit.\n\n'
            '    Args:\n'
            '        x: A value.\n'
            '    """\n'
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', '-l', '40', '--no-wrap-summary', str(p)])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['docstrings']['skipped_summaries'] >= 1

    def test_style_with_cli_override(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', '--style', 'numpy', str(p)])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['style'] == 'numpy'

    def test_unchanged_record(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('def f():\n    """Short."""\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['-j', str(p)])
        rec = json.loads(buf.getvalue())['files'][0]
        assert rec['status'] == 'unchanged'
        assert rec['warning'] is None
        assert rec['changes'] == []
        assert rec['docstrings']['total'] == 1
        assert rec['docstrings']['wrapped'] == 0
        assert isinstance(rec['line_length'], int)

    def test_warning_level_invalid_config_warns_with_normal_json_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\ncolor = "rainbow"\nline-length = 0\n'
        )
        f = tmp_path / 'a.py'
        f.write_text('x = 1\n')
        assert main(['--json', str(f)]) == 0
        captured = capsys.readouterr()
        record = json.loads(captured.out)['files'][0]
        assert record['status'] == 'unchanged'
        assert record['line_length'] == 79
        assert captured.err == (
            'warning: invalid [tool.docwrap].line-length value 0, using 79\n'
            "warning: invalid [tool.docwrap].color value 'rainbow', using 'auto'\n"
        )


class TestJsonSafetySkip:
    def test_json_output_reports_duplicate_field(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / 'mod.py'
        target.write_text(
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    :returns: First thing.\n'
            '    :returns: Second thing.\n'
            '    """\n'
        )
        assert main(['-j', '--convert', 'google', str(target)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert 'duplicate-field' in json.dumps(payload)


class TestJsonSkips:
    def test_json_includes_skips(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'a.py'
        f.write_text(_SKIP_ONLY_SOURCE)
        code = main(['-j', str(f)])
        data = json.loads(capsys.readouterr().out)
        assert code == 0
        assert data['total_skips'] == 1
        record = data['files'][0]
        assert record['status'] == 'unchanged'
        assert record['skips'] == [
            {
                'line': 2,
                'qualname': 'skipme',
                'code': 'line-continuation',
                'message': 'backslash-newline line continuation at index 9',
            }
        ]


class TestLineLengthFlag:
    def test_l_for_length(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-l', '10'])
        assert exit_code == 1

    @pytest.mark.parametrize('value', [0, -1])
    def test_non_positive_rejected(self, value: int, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--length', str(value)])
        assert '--length must be a positive integer' in capsys.readouterr().err

    def test_override(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '--length', '10'])
        assert exit_code == 1


class TestModeExclusivity:
    @pytest.mark.parametrize(
        'flags',
        [
            ['-p', '-d'],
            ['-p', '-w'],
            ['-d', '-w'],
            ['-s', '-p'],
            ['-s', '-d'],
            ['-s', '-w'],
            ['-j', '-p'],
            ['-j', '-d'],
            ['-j', '-w'],
            ['-j', '-s'],
        ],
    )
    def test_mutually_exclusive_modes(self, flags: list[str]) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['.', *flags])


class TestModuleInvocation:
    @pytest.mark.parametrize('module', ['docwrap', 'docwrap._cli'])
    def test_help_via_module(self, module: str) -> None:
        result = run_setup(
            [sys.executable, '-m', module, '--help'],
            env={**os.environ, 'PYTHONPATH': 'src'},
        )
        assert 'usage: docwrap' in decode_text(result.stdout)


class TestMultiRepositoryIncremental:
    def test_explicit_diff_failure_is_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_doc = '"""Summary.\n\n    ' + 'word ' * 40 + '\n    """'
        repo = _make_repo(tmp_path / 'repo')
        target = repo / 'mod.py'
        target.write_text('def f():\n    """Doc."""\n', encoding='utf-8')
        git_commit_all(repo, 'initial')
        original = f'def f():\n    {long_doc}\n'
        target.write_text(original, encoding='utf-8')

        def failed_diff(_root: str) -> str | None:
            return None

        monkeypatch.setattr('docwrap._cli.get_repo_diff', failed_diff)
        main(['--write', '--incremental', str(target)])

        assert target.read_text(encoding='utf-8') == original

    def test_file_outside_any_repo_mixed_with_repo_file(self, tmp_path: Path) -> None:
        long_doc = '"""Summary.\n\n    ' + 'word ' * 40 + '\n    """'
        repo = _make_repo(tmp_path / 'repo')
        in_repo = repo / 'mod.py'
        in_repo.write_text('def f():\n    """Doc."""\n', encoding='utf-8')
        git_commit_all(repo, 'initial')
        in_repo.write_text(f'def f():\n    {long_doc}\n', encoding='utf-8')

        loose = tmp_path / 'loose'
        loose.mkdir()
        outside = loose / 'mod.py'
        outside.write_text(f'def g():\n    {long_doc}\n', encoding='utf-8')

        main(['--write', '--incremental', str(outside), str(in_repo)])

        assert in_repo.read_text(encoding='utf-8') != f'def f():\n    {long_doc}\n'

    def test_implicit_diff_failure_wraps_that_repo_in_full_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_doc = '"""Summary.\n\n    ' + 'word ' * 40 + '\n    """'
        repo = _make_repo(tmp_path / 'repo')
        target = repo / 'mod.py'
        target.write_text('def f():\n    """Doc."""\n', encoding='utf-8')
        git_commit_all(repo, 'initial')
        target.write_text(f'def f():\n    {long_doc}\n', encoding='utf-8')
        (repo / 'pyproject.toml').write_text(
            '[tool.docwrap]\nincremental = "definition"\n', encoding='utf-8'
        )

        def failed_diff(_root: str) -> str | None:
            return None

        monkeypatch.setattr('docwrap._cli.get_repo_diff', failed_diff)
        main(['--write', str(target)])

        assert target.read_text(encoding='utf-8') != f'def f():\n    {long_doc}\n'

    def test_one_repo_fails_while_the_other_stays_incremental(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_doc = '"""Summary.\n\n    ' + 'word ' * 40 + '\n    """'
        repos: list[tuple[Path, Path]] = []
        for name in ('a', 'b'):
            repo = _make_repo(tmp_path / name)
            target = repo / 'mod.py'
            target.write_text('def f():\n    """Doc."""\n', encoding='utf-8')
            git_commit_all(repo, 'initial')
            target.write_text(f'def f():\n    {long_doc}\n', encoding='utf-8')
            (repo / 'pyproject.toml').write_text(
                '[tool.docwrap]\nincremental = "definition"\n', encoding='utf-8'
            )
            repos.append((repo, target))

        failing_root = str(repos[0][0].resolve())

        def selective_diff(root: str) -> str | None:
            return None if str(Path(root).resolve()) == failing_root else ''

        monkeypatch.setattr('docwrap._cli.get_repo_diff', selective_diff)
        main(['--write', str(repos[0][1]), str(repos[1][1])])

        assert repos[0][1].read_text(encoding='utf-8') != f'def f():\n    {long_doc}\n'
        assert repos[1][1].read_text(encoding='utf-8') == f'def f():\n    {long_doc}\n'

    def test_same_relative_path_in_two_repos(self, tmp_path: Path) -> None:
        repo_a = _make_repo(tmp_path / 'a')
        repo_b = _make_repo(tmp_path / 'b')
        file_a = repo_a / 'src' / 'a.py'
        file_b = repo_b / 'src' / 'a.py'
        changed_a = '"""Alpha changed summary.\n\n    ' + 'alpha ' * 30 + '\n    """'
        sentinel_a = '"""Alpha sentinel.\n\n    ' + 'sentinel-a ' * 20 + '\n    """'
        changed_b = '"""Bravo changed summary.\n\n    ' + 'bravo ' * 30 + '\n    """'
        sentinel_b = '"""Bravo sentinel.\n\n    ' + 'sentinel-b ' * 20 + '\n    """'

        initial_a = (
            f'def changed_alpha():\n    """Doc."""\n\n\ndef sentinel_alpha():\n    {sentinel_a}\n'
        )
        initial_b = (
            f'def sentinel_bravo():\n    {sentinel_b}\n\n\ndef changed_bravo():\n    """Doc."""\n'
        )
        for file, source in ((file_a, initial_a), (file_b, initial_b)):
            file.parent.mkdir()
            file.write_text(source, encoding='utf-8')
        git_commit_all(repo_a, 'initial')
        git_commit_all(repo_b, 'initial')

        file_a.write_text(
            initial_a.replace('    """Doc."""\n', f'    {changed_a}\n'),
            encoding='utf-8',
        )
        file_b.write_text(
            initial_b.replace('    """Doc."""\n', f'    {changed_b}\n'),
            encoding='utf-8',
        )

        main(['--write', '--incremental', str(file_a), str(file_b)])

        result_a = file_a.read_text(encoding='utf-8')
        result_b = file_b.read_text(encoding='utf-8')
        assert f'    {changed_a}\n' not in result_a
        assert f'    {sentinel_a}\n' in result_a
        assert f'    {changed_b}\n' not in result_b
        assert f'    {sentinel_b}\n' in result_b


class TestNewlinePreservation:
    def test_coding_cookie_encoding_roundtrip(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        content = (
            '# -*- coding: utf-8 -*-\n'
            'def foo():\n'
            '    """Summary.\n\n'
            '    This body prose includes Renée at a café and is deliberately long '
            'enough to wrap at forty columns.\n'
            '    """\n'
        )
        original = content.encode('utf-8')
        f.write_bytes(original)
        assert main([str(f), '--write', '--length', '40']) == 0
        raw = f.read_bytes()

        assert raw != original
        result = raw.decode('utf-8')
        assert 'Ren\u00e9e' in result
        assert 'caf\u00e9' in result

    def test_crlf_preserved_on_write(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        original = _BODY_WRAPPABLE_SOURCE.replace('\n', '\r\n').encode()
        f.write_bytes(original)
        main([str(f), '--write'])
        raw = f.read_bytes()
        # Verify write actually changed the file (guards against no-op)
        assert raw != original
        # Newline style must be preserved (requirement 7.2)
        assert b'\r\n' in raw
        # After stripping \r\n pairs, no bare \n should remain
        assert raw.replace(b'\r\n', b'').count(b'\n') == 0

    def test_lf_preserved_on_write(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        original = _BODY_WRAPPABLE_SOURCE.encode()
        f.write_bytes(original)
        main([str(f), '--write'])
        raw = f.read_bytes()
        # Verify write actually changed the file (guards against no-op)
        assert raw != original
        assert b'\r\n' not in raw


class TestPathDeduplication:
    def test_overlapping_dir_and_file_no_duplicate_processing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        # Pass both the directory and the file -- without dedup, file appears twice
        exit_code = main([str(tmp_path), str(f), '--diff'])
        assert exit_code == 1
        captured = capsys.readouterr()
        # The diff header for this file should appear exactly once
        diff_headers = [line for line in captured.out.splitlines() if line.startswith('---')]
        assert len(diff_headers) == 1


class TestPerFileConfig:
    def test_different_line_lengths_per_directory(self, tmp_path: Path) -> None:
        narrow_width = 40
        wide_width = 200

        # Tree A: line-length 40 (will trigger wrapping)
        dir_a = tmp_path / 'a'
        dir_a.mkdir()
        (dir_a / 'pyproject.toml').write_text(f'[tool.docwrap]\nline-length = {narrow_width}\n')
        fa = dir_a / 'test.py'
        fa.write_text(_BODY_WRAPPABLE_SOURCE)

        # Tree B: line-length 200 (won't trigger wrapping)
        dir_b = tmp_path / 'b'
        dir_b.mkdir()
        (dir_b / 'pyproject.toml').write_text(f'[tool.docwrap]\nline-length = {wide_width}\n')
        fb = dir_b / 'test.py'
        fb.write_text(_BODY_WRAPPABLE_SOURCE)

        # Process both files; only fa should change
        exit_code = main(['--write', str(fa), str(fb)])
        wrapped_a = fa.read_text()
        unchanged_b = fb.read_text()
        assert exit_code == 0
        assert wrapped_a != _BODY_WRAPPABLE_SOURCE
        assert max(map(len, wrapped_a.splitlines())) <= narrow_width
        assert unchanged_b == _BODY_WRAPPABLE_SOURCE
        assert max(map(len, unchanged_b.splitlines())) < wide_width


class TestPreviewMode:
    def test_p_for_preview(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-p'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert 'before:' in captured.out

    def test_preview_preserves_original_indent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        # Nested docstring at 8-space indent
        f.write_text(
            'class Foo:\n'
            '    def bar(self):\n'
            '        """Summary.\n'
            '\n'
            '        This body prose is deliberately long enough to exceed the configured '
            'default line length and require wrapping by docwrap.\n'
            '        """\n'
        )
        main([str(f), '--preview'])
        captured = capsys.readouterr()
        # Before/after should show 8-space indent, not normalized 4-space
        assert '        """' in captured.out

    def test_preview_ruler_marker_at_target_column(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '-p', '-l', '79'])
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        before_lines = [x for x in lines if x.lstrip().startswith('before:')]
        after_lines = [x for x in lines if x.lstrip().startswith('after:')]
        assert len(before_lines) == 1
        assert len(after_lines) == 1
        # The | marks the boundary after column 79 (0-indexed position 79)
        assert before_lines[0][79] == '|'
        assert '|79' in before_lines[0]
        assert after_lines[0][79] == '|'
        assert '|79' in after_lines[0]

    def test_preview_ruler_uses_config_line_length(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '-p'])
        captured = capsys.readouterr()
        # Default config resolves to 79, so ruler should show |79
        assert '|79' in captured.out

    def test_preview_shows_before_after(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '--preview'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert 'foo' in captured.out
        assert 'before:' in captured.out
        assert 'after:' in captured.out


class TestPreviewSkipNotes:
    def test_hide_skips_from_config_omits_skip_only_preview_diagnostic(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nwrap-summary = false\nshow-skips = false\n'
        )
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--preview'])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ''

    def test_hide_skips_keeps_diff_for_changed_docstring(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long_summary = 'A ' + 'very ' * 15 + 'long summary.'
        long_desc = 'Description ' + 'word ' * 20 + 'end.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long_summary}\n\n    {long_desc}\n    """\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary', '--hide-skips'])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'skip:' not in captured.out
        assert 'before:' in captured.out
        assert 'after:' in captured.out

    def test_hide_skips_omits_skip_only_preview_diagnostic(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary', '--hide-skips'])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ''

    def test_no_skip_no_note(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--preview'])
        captured = capsys.readouterr()
        assert 'skip:' not in captured.out

    def test_show_skips_overrides_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nwrap-summary = false\nshow-skips = false\n'
        )
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--preview', '--show-skips'])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert 'skip: summary exceeds line length' in captured.out
        assert 'not wrapped' not in captured.out
        assert 'foo' in captured.out

    def test_show_skips_shows_skip_only_diagnostic(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary', '--show-skips'])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert 'skip: summary exceeds line length' in captured.out
        assert 'not wrapped' not in captured.out
        assert 'foo' in captured.out

    def test_show_skips_with_changes_shows_skip_and_diff(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long_summary = 'A ' + 'very ' * 15 + 'long summary.'
        long_desc = 'Description ' + 'word ' * 20 + 'end.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long_summary}\n\n    {long_desc}\n    """\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary', '--show-skips'])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'skip: summary exceeds line length' in captured.out
        assert 'not wrapped' not in captured.out
        assert 'before:' in captured.out
        assert 'after:' in captured.out

    def test_skip_only_hides_diagnostic_by_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary'])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ''

    def test_skip_with_changes_hides_note_by_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long_summary = 'A ' + 'very ' * 15 + 'long summary.'
        long_desc = 'Description ' + 'word ' * 20 + 'end.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long_summary}\n\n    {long_desc}\n    """\n')
        exit_code = main([str(f), '--preview', '--no-wrap-summary'])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert 'skip:' not in captured.out
        assert 'before:' in captured.out
        assert 'after:' in captured.out


class TestProcessFile:
    def test_caught_error_controls_are_visible(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = Path(f'path-{_CONTROL_TEXT}.py')

        def raise_os_error(*_args: object, **_kwargs: object) -> None:
            raise OSError(f'failure-{_CONTROL_TEXT}')

        monkeypatch.setattr(cli_mod, '_wrap_file_once', raise_os_error)

        result = _process_file(
            path,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='summary',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 2
        assert result.total is None
        assert result.docstrings == ()
        assert capsys.readouterr().err == (
            f'error: path-{_VISIBLE_CONTROL_TEXT}.py: failure-{_VISIBLE_CONTROL_TEXT}\n'
        )

    def test_changed_diff_retains_counts_without_docstrings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / 'changed.py'
        path.write_text(_BODY_WRAPPABLE_SOURCE)

        result = _process_file(
            path,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='diff',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 1
        assert result.total == 1
        assert result.wrapped == 1
        assert result.skipped_summaries == 0
        assert result.docstrings == ()
        assert '---' in capsys.readouterr().out

    def test_changed_preview_retains_docstring_changes(self, tmp_path: Path) -> None:
        path = tmp_path / 'changed.py'
        path.write_text(_BODY_WRAPPABLE_SOURCE)

        result = _process_file(
            path,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='preview',
                detail='diff',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 1
        assert result.total == 1
        assert result.wrapped == 1
        assert len(result.docstrings) == 1
        assert result.docstrings[0].before is not None
        assert result.docstrings[0].after is not None

    def test_changed_summary_retains_counts_without_docstrings(self, tmp_path: Path) -> None:
        path = tmp_path / 'changed.py'
        path.write_text(_BODY_WRAPPABLE_SOURCE)

        result = _process_file(
            path,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='summary',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 1
        assert result.total == 1
        assert result.wrapped == 1
        assert result.skipped_summaries == 0
        assert result.docstrings == ()

    def test_file_result_records_effective_line_length(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        result = _process_file(
            f,
            WrapOverrides(line_length=72),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='summary',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )
        assert result.line_length == 72

    def test_skip_controls_are_visible(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = Path(f'path-{_CONTROL_TEXT}.py')
        report = WrapReport(
            source='',
            total=1,
            wrapped=0,
            skips=(
                DocstringSkip(
                    path=str(path),
                    line=1,
                    qualname=f'name-{_CONTROL_TEXT}',
                    code='line-continuation',
                    message=f'message-{_CONTROL_TEXT}',
                ),
            ),
        )
        wrapped = _WrappedFile(
            source='',
            encoding='utf-8',
            config=DocwrapConfig(),
            report=report,
            raw=b'',
            show_skips=False,
        )

        def return_wrapped(*_args: object, **_kwargs: object) -> _WrappedFile:
            return wrapped

        monkeypatch.setattr(cli_mod, '_wrap_file_once', return_wrapped)

        result = _process_file(
            path,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='summary',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 0
        assert capsys.readouterr().err == (
            f'path-{_VISIBLE_CONTROL_TEXT}.py:1: name-{_VISIBLE_CONTROL_TEXT} '
            f'skipped (line-continuation): message-{_VISIBLE_CONTROL_TEXT}\n'
        )

    def test_successful_write_retains_counts_without_docstrings(self, tmp_path: Path) -> None:
        path = tmp_path / 'changed.py'
        source = _BODY_WRAPPABLE_SOURCE
        path.write_text(source)

        result = _process_file(
            path,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='write',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 0
        assert result.total == 1
        assert result.wrapped == 1
        assert result.skipped_summaries == 0
        assert result.docstrings == ()
        assert path.read_text() != source

    def test_unchanged_result_retains_counts_without_docstrings(self, tmp_path: Path) -> None:
        path = tmp_path / 'unchanged.py'
        path.write_text('def foo():\n    """Short."""\n')

        result = _process_file(
            path,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='summary',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 0
        assert result.total == 1
        assert result.wrapped == 0
        assert result.skipped_summaries == 0
        assert result.docstrings == ()

    def test_write_error_retains_post_wrap_counts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / 'changed.py'
        path.write_text(_BODY_WRAPPABLE_SOURCE)

        def raise_write_error(*_args: object, **_kwargs: object) -> None:
            raise OSError('write failed')

        monkeypatch.setattr(cli_mod, 'write_preserving_newline', raise_write_error)

        result = _process_file(
            path,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
            _FileModeOptions(
                mode='write',
                detail='summary',
                verbose=False,
                color=False,
            ),
        )

        assert result.status == 2
        assert result.total == 1
        assert result.wrapped == 1
        assert result.skipped_summaries == 0
        assert result.docstrings == ()
        assert capsys.readouterr().err.endswith('changed.py: write failed\n')


class TestProcessFileJson:
    def test_control_text_round_trips_raw_in_version_1_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = Path(f'path-{_CONTROL_TEXT}.py')

        def raise_os_error(*_args: object, **_kwargs: object) -> None:
            raise OSError(f'failure-{_CONTROL_TEXT}')

        monkeypatch.setattr(cli_mod, '_wrap_file_once', raise_os_error)

        record = _process_file_json(
            path,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )
        decoded = json.loads(display_mod.format_json([record]))

        assert decoded['version'] == 1
        assert decoded['files'][0]['path'] == str(path)
        assert decoded['files'][0]['warning'] == f'failure-{_CONTROL_TEXT}'

    def test_process_file_json_changed(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text(
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        record = assert_type(
            _process_file_json(
                p,
                WrapOverrides(line_length=79),
                _FilePlan(state='full', eligible_ranges=None),
                None,
            ),
            JsonFileRecord,
        )
        assert record['status'] == 'changed'
        assert len(record['changes']) > 0
        change = record['changes'][0]
        assert 'qualname' in change
        assert 'before' in change
        assert 'after' in change
        assert 'summary_skipped' in change
        assert 'summary_width' in change

    def test_process_file_json_error_permission(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')

        def unreadable(_path: Path) -> ReadResult:
            raise PermissionError('permission denied')

        monkeypatch.setattr(cli_mod, 'read_file_result', unreadable)
        record = _process_file_json(
            p,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )

        assert record['status'] == 'error'
        assert record['warning'] == 'permission denied'
        assert record['docstrings'] == {'wrapped': 0, 'total': 0, 'skipped_summaries': 0}
        assert record['line_length'] is None
        assert record['style'] is None

    def test_process_file_json_invalid_syntax(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('def f(:\n')
        record = _process_file_json(
            p,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )
        assert record['status'] == 'invalid'
        assert record['warning'] is not None
        assert record['docstrings'] == {'wrapped': 0, 'total': 0, 'skipped_summaries': 0}
        assert record['line_length'] is None
        assert record['style'] is None

    def test_process_file_json_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / 'a.py'
        p.write_text('x = 1\n')
        record = _process_file_json(
            p,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )
        assert record['status'] == 'unchanged'
        assert record['warning'] is None
        assert record['changes'] == []
        assert record['docstrings']['total'] == 0
        assert record['docstrings']['wrapped'] == 0
        assert isinstance(record['line_length'], int)


class TestReadyWriteReads:
    @pytest.mark.parametrize(
        'case',
        [
            pytest.param(
                (
                    'ordinary',
                    b'\xef\xbb\xbf' + _BODY_WRAPPABLE_SOURCE.replace('\n', '\r\n').encode(),
                    b'\xef\xbb\xbf',
                    b'\r\n',
                ),
                id='ordinary-bom-crlf',
            ),
            pytest.param(
                (
                    'untracked',
                    (
                        b'# -*- coding: latin-1 -*-\n\ndef f():\n'
                        b'    """Caf\xe9 summary.\n\n'
                        b'    This body prose is deliberately long enough to exceed the '
                        b'configured default line length and require wrapping by docwrap.\n'
                        b'    """\n'
                    ),
                    b'# -*- coding: latin-1 -*-\n',
                    b'\n',
                ),
                id='untracked-cookie-lf',
            ),
            pytest.param(
                (
                    'configured-fallback',
                    _BODY_WRAPPABLE_SOURCE.replace('def foo', 'def f')
                    .replace('\n', '\r\n')
                    .encode(),
                    b'def f():\r\n',
                    b'\r\n',
                ),
                id='configured-fallback-utf8-crlf',
            ),
        ],
    )
    def test_full_write_reuses_first_raw_bytes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[str, bytes, bytes, bytes],
    ) -> None:
        route, changed, prefix, newline = case
        path = tmp_path / 'a.py'
        flags = _configure_full_write_route(tmp_path, monkeypatch, route)
        path.write_bytes(changed)

        original_read_file = cli_mod.read_file
        original_read_bytes = Path.read_bytes
        original_writer = cli_mod.write_preserving_newline
        resolved = path.resolve()
        reads = {'bytes': 0, 'source': 0}
        writer_raw: list[bytes | None] = []

        def tracked_read_file(candidate: Path) -> tuple[str, str]:
            if candidate.resolve() == resolved:
                reads['source'] += 1
            return original_read_file(candidate)

        def tracked_read_bytes(candidate: Path) -> bytes:
            if candidate.resolve() == resolved:
                reads['bytes'] += 1
            return original_read_bytes(candidate)

        def tracked_writer(
            candidate: Path,
            source: str,
            encoding: str,
            *,
            raw: bytes | None = None,
        ) -> None:
            if candidate.resolve() == resolved:
                writer_raw.append(raw)
            original_writer(candidate, source, encoding, raw=raw)

        monkeypatch.setattr(cli_mod, 'read_file', tracked_read_file)
        monkeypatch.setattr(Path, 'read_bytes', tracked_read_bytes)
        monkeypatch.setattr(cli_mod, 'write_preserving_newline', tracked_writer)
        args = [*flags, '--write', str(path)]

        assert main(args) == 0
        capsys.readouterr()
        first = original_read_bytes(path)
        assert reads == {'bytes': 1, 'source': 0}
        assert writer_raw == [changed]
        assert first != changed
        assert first.startswith(prefix)
        _assert_newline_form(first, newline)

        reads = {'bytes': 0, 'source': 0}
        writer_raw.clear()
        assert main(args) == 0
        capsys.readouterr()
        assert reads == {'bytes': 1, 'source': 0}
        assert writer_raw == []
        assert original_read_bytes(path) == first

    def test_full_write_uses_shared_reader(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        path.write_bytes(_BODY_WRAPPABLE_SOURCE.replace('\n', '\r\n').encode())
        calls: list[Path] = []

        def tracked_reader(candidate: Path) -> ReadResult:
            calls.append(candidate)
            return read_file_result(candidate)

        monkeypatch.setattr(cli_mod, 'read_file_result', tracked_reader)

        assert main(['--write', str(path)]) == 0
        capsys.readouterr()
        assert calls == [path]


class TestReflowFlag:
    def test_reflow_flag_invalid(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            _build_parser().parse_args(['--reflow', 'aggressive'])

    def test_reflow_none_preserves_short_lines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        py_file = tmp_path / 'x.py'
        py_file.write_text(
            'def f():\n    """Summary.\n\n    First sentence.\n    Second sentence.\n    """\n'
        )
        assert main(['--reflow', 'none', '--diff', str(py_file)]) == 0
        assert capsys.readouterr().out == ''
        assert main(['--reflow', 'full', '--diff', str(py_file)]) == 1
        assert '+    First sentence. Second sentence.' in capsys.readouterr().out

    def test_stats_respects_reflow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        py_file = tmp_path / 'x.py'
        py_file.write_text(
            'def f():\n    """Summary.\n\n    First sentence.\n    Second sentence.\n    """\n'
        )
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        for reflow, changed in (('none', '0'), ('full', '1')):
            assert main(['--reflow', reflow, '--stats', str(py_file)]) == 0
            output = capsys.readouterr().out
            impact = output[output.index('Wrapping impact') :]
            row = re.search(r'^\s+79\s+\d+\s+\d+\s+[+-]?\d+\s+(\d+)\s', impact, re.MULTILINE)
            assert row is not None
            assert row[1] == changed


class TestReportErrors:
    def test_partial_report_on_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / 'bad.py'
        bad.write_text('def foo(\n')  # syntax error
        good = tmp_path / 'good.py'
        good.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(bad), str(good)])
        assert exit_code == 2  # error takes precedence
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert 'Name' in lines[0]
        assert str(good) in lines[1]
        assert '1' in lines[1]  # 1 wrapped
        assert 'error' in captured.err  # error message for bad file


class TestResolveHelpConfig:
    def test_invalid_style_keeps_other_resolved_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 88\nstyle = "bogus"\n'
        )
        monkeypatch.chdir(tmp_path)
        cfg = _resolve_help_config()
        assert cfg.config.line_length == 88
        assert cfg.config.style is None

    def test_reads_pyproject_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 88\n')
        monkeypatch.chdir(tmp_path)
        resolved = _resolve_help_config()
        assert resolved.config.line_length == 88
        assert resolved.line_length_source is LineLengthSource.DOCWRAP

    def test_returns_defaults_when_no_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        monkeypatch.chdir(empty)
        resolved = _resolve_help_config()
        assert resolved.config.line_length == 79
        assert resolved.line_length_source is LineLengthSource.DEFAULT


class TestSafetySkipStderr:
    def test_check_mode_skip_only_exits_zero_with_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'a.py'
        f.write_text(_SKIP_ONLY_SOURCE)
        code = main([str(f)])
        captured = capsys.readouterr()
        assert code == 0
        assert f'{f}:2: skipme skipped (line-continuation):' in captured.err

    def test_hide_skips_does_not_hide_safety_warnings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'a.py'
        f.write_text(_MIXED_SKIP_SOURCE)
        code = main(['-p', '--hide-skips', str(f)])
        captured = capsys.readouterr()
        assert code == 1
        assert 'skipped (line-continuation)' in captured.err

    @pytest.mark.parametrize(
        ('source', 'skip_code'),
        [
            (_SKIP_ONLY_SOURCE, 'line-continuation'),
            ('def skipme():\n    """First.""" """Second."""\n', 'implicit-concatenation'),
        ],
    )
    def test_stats_mode_reports_safety_skips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, skip_code: str
    ) -> None:
        f = tmp_path / 'a.py'
        f.write_text(source)
        code = main(['-s', str(f)])
        captured = capsys.readouterr()
        assert code == 0
        assert f'skipped ({skip_code})' in captured.err
        assert 'Skipped 1 docstring for safety.' in captured.out

    def test_write_mode_wraps_rest_and_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'a.py'
        f.write_text(_MIXED_SKIP_SOURCE)
        code = main(['-w', str(f)])
        captured = capsys.readouterr()
        assert code == 0
        assert f'{f}:2: skipme skipped (line-continuation):' in captured.err
        text = f.read_text()
        assert 'Line one ' + BACKSLASH + '\n' in text  # skipped docstring untouched
        assert 'definitely\n' in text  # other docstring wrapped


class TestShortOptions:
    def test_d_for_diff(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-d'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert '---' in captured.out

    def test_w_for_write(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        before = f.read_text()
        exit_code = main([str(f), '-w'])
        assert exit_code == 0
        assert f.read_text() != before


class TestShowSkipsFlag:
    def test_hide_skips_accepted(self) -> None:
        args = _build_parser().parse_args(['--hide-skips', '.'])
        assert args.show_skips is False

    def test_show_skips_accepted(self) -> None:
        args = _build_parser().parse_args(['--show-skips', '.'])
        assert args.show_skips is True


class TestSkippedSummariesOutput:
    def test_skip_only_file_hidden_in_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        main([str(f), '--no-wrap-summary'])
        captured = capsys.readouterr()
        assert 'Summaries skipped' not in captured.out

    def test_skip_only_file_shown_in_verbose(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        main([str(f), '--no-wrap-summary', '-v'])
        captured = capsys.readouterr()
        assert 'Skip' in captured.out
        assert str(f) in captured.out


class TestStatsEffectiveConfig:
    def test_stats_convert_tie_impact_matches_reality(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / 'mod.py'
        target.write_text(
            'def f():\n'
            '    """Summary line.\n'
            '\n'
            '    :param alpha: Short description.\n'
            '\n'
            '    Returns:\n'
            '        Something useful.\n'
            '    """\n'
        )
        monkeypatch.chdir(tmp_path)
        assert main(['--stats', '--convert', 'google', str(target)]) == 0
        out = capsys.readouterr().out
        impact_section = out[out.index('Wrapping impact') :]
        row_re = re.compile(r'^\s+\d+\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+(\d+)\s', re.MULTILINE)
        rows = row_re.findall(impact_section)
        assert rows
        assert all(int(docs) == 0 for _, _, _, docs in rows)

    def test_stats_impact_reflects_pyproject_summary_start(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "second-line"\n')
        target = tmp_path / 'mod.py'
        target.write_text(
            'def f():\n    """Summary line that is fine.\n\n    More prose here.\n    """\n'
        )
        monkeypatch.chdir(tmp_path)
        assert main(['--stats', str(target)]) == 0
        out = capsys.readouterr().out
        impact_section = out[out.index('Wrapping impact') :]
        row_re = re.compile(r'^\s+\d+\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+(\d+)\s', re.MULTILINE)
        rows = row_re.findall(impact_section)
        assert rows
        assert any(int(docs) > 0 for _, _, _, docs in rows)


class TestStatsMode:
    def test_collect_stats_without_impact_returns_empty_mapping(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)
        impact_spy = Mock(side_effect=AssertionError('impact should not be computed'))
        monkeypatch.setattr(stats_mod, 'compute_impact', impact_spy)

        data = cli_mod._collect_stats(
            [f],
            WrapOverrides(wrap_summary=True),
            needs_impact=False,
        )

        assert data.impact == {}
        assert impact_spy.call_count == 0

    @pytest.mark.parametrize(
        ('flag', 'expected_wrappability'),
        [
            (None, [False, True]),
            ('--wrap-summary', [True, True]),
            ('--no-wrap-summary', [False, False]),
        ],
    )
    def test_each_file_effective_config_controls_stats(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        flag: str | None,
        expected_wrappability: list[bool],
    ) -> None:
        first_dir = tmp_path / 'first'
        first_dir.mkdir()
        (first_dir / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'line-length = 91\n'
            'convention = "google"\n'
            'convert = "numpy"\n'
            'fill-style = "greedy"\n'
            'reflow = "none"\n'
            'smart-slack = 0\n'
            'strip-types = true\n'
            'summary-start = "second-line"\n'
            'wrap-summary = false\n'
        )
        first = first_dir / 'first.py'
        first.write_text('def first():\n    """First summary."""\n')

        second_dir = tmp_path / 'second'
        second_dir.mkdir()
        (second_dir / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'line-length = 97\n'
            'convention = "numpy"\n'
            'convert = "google"\n'
            'fill-style = "balanced"\n'
            'reflow = "full"\n'
            'smart-slack = 3\n'
            'summary-start = "first-line"\n'
            'wrap-summary = true\n'
        )
        second = second_dir / 'second.py'
        second.write_text('def second():\n    """Second summary."""\n')

        impact_spy = Mock(return_value=[])
        print_spy = Mock(wraps=display_mod.print_stats)
        monkeypatch.setattr(stats_mod, 'compute_impact', impact_spy)
        monkeypatch.setattr(display_mod, 'print_stats', print_spy)

        argv = ['--stats', '--stats-sections=docstrings,impact']
        if flag is not None:
            argv.append(flag)
        argv.extend((str(first), str(second)))
        assert main(argv) == 0

        expected_configs = [
            DocwrapConfig(
                convert='numpy',
                fill_style='greedy',
                line_length=91,
                reflow='none',
                smart_slack=0,
                strip_types=True,
                style='google',
                summary_start='second-line',
                wrap_summary=expected_wrappability[0],
            ),
            DocwrapConfig(
                convert='google',
                fill_style='balanced',
                line_length=97,
                reflow='full',
                smart_slack=3,
                style='numpy',
                summary_start='first-line',
                wrap_summary=expected_wrappability[1],
            ),
        ]
        assert [call.args[1] for call in impact_spy.call_args_list] == expected_configs
        data = print_spy.call_args.args[0]
        summaries = [m for m in data.measurements if m.category == 'summary']
        assert [m.wrappable for m in summaries] == expected_wrappability

    def test_first_file_controls_layout_but_analysis_stays_per_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first_dir = tmp_path / 'first'
        first_dir.mkdir()
        (first_dir / 'pyproject.toml').write_text(
            '[tool.docwrap]\nreflow = "none"\n[tool.docwrap.stats]\nsections = ["impact"]\n'
        )
        first = first_dir / 'first.py'
        first.write_text(_STATS_SOURCE)

        second_dir = tmp_path / 'second'
        second_dir.mkdir()
        (second_dir / 'pyproject.toml').write_text(
            '[tool.docwrap]\nreflow = "full"\n[tool.docwrap.stats]\nsections = ["source"]\n'
        )
        second = second_dir / 'second.py'
        second.write_text(_STATS_SOURCE)

        impact_spy = Mock(wraps=stats_mod.compute_impact)
        monkeypatch.setattr(stats_mod, 'compute_impact', impact_spy)

        assert main(['--stats', str(first), str(second)]) == 0
        output = capsys.readouterr().out

        assert [call.args[1].reflow for call in impact_spy.call_args_list] == [
            'none',
            'full',
        ]
        assert 'Wrapping impact' in output
        assert all(
            heading not in output
            for heading in _STATS_BODY_HEADINGS
            if heading != 'Wrapping impact'
        )

    @pytest.mark.parametrize('invalid_first', [True, False])
    def test_invalid_project_config_warns_and_processes_all_files(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        invalid_first: bool,
    ) -> None:
        good_dir = tmp_path / 'good'
        good_dir.mkdir()
        (good_dir / 'pyproject.toml').write_text('[tool.docwrap]\n')
        good = good_dir / 'good.py'
        good.write_text('def good():\n    """Good."""\n')

        bad_dir = tmp_path / 'bad'
        bad_dir.mkdir()
        (bad_dir / 'pyproject.toml').write_text('[tool.docwrap]\nstyle = "bogus"\n')
        bad = bad_dir / 'bad.py'
        bad.write_text('def bad():\n    """Bad."""\n')
        files = [bad, good] if invalid_first else [good, bad]

        assert main(['--stats', *(str(path) for path in files)]) == 0
        captured = capsys.readouterr()
        assert 'Scanned: 2 docstrings across 2 files' in captured.out
        assert captured.err == (
            "warning: invalid [tool.docwrap].style value 'bogus', using None\n"
        )
        assert 'Traceback' not in captured.err

    def test_invalid_project_stats_config_warns_but_exits_zero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap.stats]\nsections = ["unknown"]\n')
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert main(['--stats', str(f)]) == 0
        captured = capsys.readouterr()

        assert '[tool.docwrap.stats].sections' in captured.err

    @pytest.mark.parametrize('value', ['0', '-1', 'not-an-integer'])
    def test_invalid_stats_longest_cli_exits_two(self, value: str) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--stats', f'--stats-longest-per-category={value}'])

    @pytest.mark.parametrize(
        'value',
        ['unknown', 'source,source', 'source,,impact'],
    )
    def test_invalid_stats_sections_cli_exits_two(self, value: str) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['--stats', f'--stats-sections={value}'])

    def test_recommendations_keep_configured_nonstandard_length(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert (
            main(
                [
                    '--stats',
                    '--stats-sections=recommendations',
                    '--length=75',
                    str(f),
                ]
            )
            == 0
        )
        output = capsys.readouterr().out

        assert 'Recommended targets' in output
        assert 'Configured: 75' in output

    def test_s_for_stats(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '-s'])
        assert exit_code == 0

    @pytest.mark.parametrize(
        'incremental_flags',
        [pytest.param([], id='configured'), pytest.param(['--incremental'], id='explicit')],
    )
    def test_stats_bypasses_incremental_setup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        incremental_flags: list[str],
    ) -> None:
        path = tmp_path / 'a.py'
        path.write_text('def f():\n    """Short."""\n')
        lookup = Mock(side_effect=AssertionError('incremental config lookup must not run'))
        compute = Mock(side_effect=AssertionError('incremental state collection must not run'))
        monkeypatch.setattr(cli_mod, 'find_incremental_config', lookup)
        monkeypatch.setattr(cli_mod, '_compute_incremental_context', compute)

        assert main(['--stats', *incremental_flags, str(path)]) == 0
        lookup.assert_not_called()
        compute.assert_not_called()

    def test_stats_config_length_override(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        main([str(f), '--stats', '--length', '80'])
        out = capsys.readouterr().out
        assert 'Configured length: 80 (from --length)' in out

    def test_stats_exits_0_for_valid_files(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '--stats'])
        assert exit_code == 0

    def test_stats_header_has_docstring_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = 'def foo():\n    """Doc1."""\n\ndef bar():\n    """Doc2."""\n'
        f = tmp_path / 'test.py'
        f.write_text(source)
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'Scanned: 2 docstrings across 1 file' in out

    def test_stats_histograms_separated_by_blank_lines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = 'def foo(x):\n    """Do something.\n\n    Args:\n        x: The value.\n    """\n'
        f = tmp_path / 'test.py'
        f.write_text(source)
        main([str(f), '--stats'])
        captured = capsys.readouterr()
        assert 'Distribution' in captured.out
        # Histogram blocks should be separated by blank lines
        assert '\n\n' in captured.out

    def test_stats_impact_columns_aligned(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long description.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """Summary.\n\n    {long}\n    """\n')
        main([str(f), '--stats'])
        captured = capsys.readouterr()
        # Extract only impact table data rows (after the Impact header)
        lines = captured.out.split('\n')
        impact_start = next(i for i, ln in enumerate(lines) if 'Wrapping impact' in ln)
        impact_lines = [
            line
            for line in lines[impact_start:]
            if line.strip() and line.strip()[0].isdigit() and '%' in line
        ]
        assert len(impact_lines) >= 2
        # All data rows should have the same length (padded alignment)
        widths = [len(line) for line in impact_lines]
        assert len(set(widths)) == 1, f'Row widths vary: {widths}'

    def test_stats_impact_table_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'Wrapping impact' in out
        assert '+Lines' in out
        assert 'Net' in out
        assert 'Docs' in out

    def test_stats_impact_target_colored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        main([str(f), '--stats', '--length', '72', '--color', 'always'])
        out = capsys.readouterr().out
        assert RED in out or YELLOW in out

    def test_stats_long_flag(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--stats'])
        assert exit_code == 0

    def test_stats_longest_cli_overrides_project(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'wrap-summary = true\n'
            '[tool.docwrap.stats]\n'
            'sections = ["longest"]\n'
            'longest-per-category = 4\n'
        )
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert (
            main(
                [
                    '--stats',
                    '--stats-longest-per-category=1',
                    str(f),
                ]
            )
            == 0
        )
        output = capsys.readouterr().out
        location_counts = _stats_location_counts(output)

        assert len(location_counts) >= 2
        assert all(count == 1 for count in location_counts)

    def test_stats_metadata_at_top(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Summary.\n\n    Description here.\n    """\n')
        main([str(f), '--stats'])
        output = capsys.readouterr().out
        lines = output.strip().splitlines()
        assert 'Scanned:' in lines[0]
        assert 'Configured length:' in lines[1]

    def test_stats_mutually_exclusive_with_preview(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['.', '-s', '-p'])

    def test_stats_mutually_exclusive_with_write(self) -> None:
        with pytest.raises(SystemExit, match='2'):
            main(['.', '-s', '-w'])

    def test_stats_no_statistics_title(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        main([str(f), '--stats'])
        output = capsys.readouterr().out
        first_line = output.strip().split('\n')[0]
        assert not first_line.startswith('Statistics')

    @pytest.mark.parametrize(
        'option',
        ['--stats-sections=source', '--stats-longest-per-category=1'],
    )
    @pytest.mark.parametrize('mode', [(), ('--preview',)])
    def test_stats_only_options_require_stats(
        self,
        option: str,
        mode: tuple[str, ...],
    ) -> None:
        with pytest.raises(SystemExit, match='2'):
            main([*mode, option])

    def test_stats_outlier_shortened_labels(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = 'def foo(x):\n    """Do something.\n\n    Args:\n        x: The value.\n    """\n'
        f = tmp_path / 'test.py'
        f.write_text(source)
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'Longest wrappable lines' in out
        assert 'Longest wrappable lines (docstring):' not in out
        assert 'Longest prose lines:' not in out

    def test_stats_project_longest_applies_without_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'wrap-summary = true\n'
            '[tool.docwrap.stats]\n'
            'sections = ["longest"]\n'
            'longest-per-category = 1\n'
        )
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert main(['--stats', str(f)]) == 0
        output = capsys.readouterr().out
        location_counts = _stats_location_counts(output)

        assert len(location_counts) >= 2
        assert all(count <= 1 for count in location_counts)

    def test_stats_project_sections_apply_without_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap.stats]\nsections = ["recommendations", "source"]\n'
        )
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert main(['--stats', str(f)]) == 0
        output = capsys.readouterr().out

        assert output.index('Recommended targets') < output.index('Source code')
        assert all(
            heading not in output
            for heading in _STATS_BODY_HEADINGS
            if heading not in {'Recommended targets', 'Source code'}
        )

    def test_stats_section_labels(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Summary.\n\n    Description here.\n    """\n')
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'Source code' in out
        assert 'Docstrings (wrappable)' in out
        assert 'Wrapping impact' in out
        assert 'Distribution' in out
        assert 'Longest wrappable lines' in out
        assert 'Configured length:' in out

    def test_stats_sections_cli_preserves_order_and_replaces_config(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap.stats]\nsections = ["impact", "docstrings"]\n'
        )
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert (
            main(
                [
                    '--stats',
                    '--stats-sections=recommendations,source',
                    str(f),
                ]
            )
            == 0
        )
        output = capsys.readouterr().out

        assert output.index('Recommended targets') < output.index('Source code')
        assert all(
            heading not in output
            for heading in _STATS_BODY_HEADINGS
            if heading not in {'Recommended targets', 'Source code'}
        )

    def test_stats_sections_cli_strips_surrounding_whitespace(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert (
            main(
                [
                    '--stats',
                    '--stats-sections',
                    'source, recommendations',
                    str(f),
                ]
            )
            == 0
        )
        output = capsys.readouterr().out

        assert output.index('Source code') < output.index('Recommended targets')
        assert all(
            heading not in output
            for heading in _STATS_BODY_HEADINGS
            if heading not in {'Recommended targets', 'Source code'}
        )

    @pytest.mark.parametrize(
        ('sections', 'expected_per_file'),
        [
            (('source',), 0),
            (('impact',), 1),
            (('recommendations',), 1),
            (('recommendations', 'source'), 1),
        ],
    )
    def test_stats_sections_control_impact_demand(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sections: tuple[str, ...],
        expected_per_file: int,
    ) -> None:
        files = [tmp_path / 'first.py', tmp_path / 'second.py']
        for f in files:
            f.write_text(_STATS_SOURCE)
        impact_spy = Mock(return_value=[])
        monkeypatch.setattr(stats_mod, 'compute_impact', impact_spy)

        assert (
            main(
                [
                    '--stats',
                    f'--stats-sections={",".join(sections)}',
                    *(str(f) for f in files),
                ]
            )
            == 0
        )

        assert impact_spy.call_count == expected_per_file * len(files)

    def test_stats_sections_empty_cli_value(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_STATS_SOURCE)

        assert main(['--stats', '--stats-sections=', str(f)]) == 0
        output = capsys.readouterr().out

        assert 'Scanned:' in output
        assert 'Configured length:' in output
        assert all(heading not in output for heading in _STATS_BODY_HEADINGS)

    def test_stats_shows_histogram(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        long = 'A ' + 'very ' * 15 + 'long description.'
        f = tmp_path / 'test.py'
        f.write_text(
            'def foo(short, long):\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        short: Brief.\n'
            f'        long: {long}\n'
            '    """\n'
        )
        main([str(f), '--stats'])
        captured = capsys.readouterr()
        assert '1-40' in captured.out  # histogram range labels
        assert '#' in captured.out  # histogram bars

    def test_stats_shows_recommendations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(
            'def foo():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: This argument description is deliberately long enough to exceed'
            ' the seventy-nine character line length limit.\n'
            '    """\n'
        )
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'Recommended targets' in out
        assert 'Low churn' in out

    def test_stats_skipped_footer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / 'bad.py'
        bad.write_bytes(b'# coding: nosuchcodec\n')
        good = tmp_path / 'good.py'
        good.write_text('def foo():\n    """Short."""\n')
        main([str(good), str(bad), '--stats'])
        out = capsys.readouterr().out
        assert 'Skipped 1 file (unreadable).' in out

    def test_stats_summary_shows_percentiles(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        main([str(f), '--stats', '--no-wrap-summary'])
        out = capsys.readouterr().out
        assert 'Docstrings (not wrappable)' in out
        assert 'Median' in out
        assert 'P75' in out
        assert 'P90' in out
        assert 'P95' in out
        assert 'P99' in out

    def test_stats_zero_docstrings_early_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('x = 1\n')
        main([str(f), '--stats'])
        out = capsys.readouterr().out
        assert 'No docstrings found.' in out
        assert 'Scanned:' in out
        assert '0 docstrings' in out
        assert 'Source code' not in out


class TestStyleOption:
    def test_convention_from_config(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nconvention = "numpy"\n')
        f = tmp_path / 'test.py'
        long_summary = 'Summary.'
        long_desc = 'A very long description ' * 4 + 'that needs wrapping.'
        original = (
            f'def foo(x):\n    """{long_summary}\n\n    Parameters\n    ----------\n'
            f'    x : int\n        {long_desc}\n    """\n'
        )
        f.write_text(original)
        main([str(f), '--write'])
        result = f.read_text()
        assert result != original  # wrapping happened
        # Re-run: idempotent under configured convention (exit 0)
        exit_code = main([str(f)])
        assert exit_code == 0

    def test_invalid_style_exits_2(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        with pytest.raises(SystemExit) as exc_info:
            main([str(f), '--style', 'bogus'])
        assert exc_info.value.code == 2

    def test_style_flag_passes_convention(self, tmp_path: Path) -> None:
        long_summary = 'Summary.'
        long_desc = 'A very long description ' * 4 + 'that needs wrapping.'
        f = tmp_path / 'test.py'
        # Put colon-free section prose at indent 0 so Google can promote it
        # safely while NumPy retains it in the top-level description.
        original = f'def foo(x):\n    """{long_summary}\n\n    Args:\n    {long_desc}\n    """\n'
        f.write_text(original)
        main([str(f), '--style', 'google', '--write'])
        result_google = f.read_text()
        assert result_google != original  # wrapping happened
        # Re-run: idempotent under forced style
        exit_code = main([str(f), '--style', 'google'])
        assert exit_code == 0
        # Cross-style comparison: same original, different forced style
        f.write_text(original)
        main([str(f), '--style', 'numpy', '--write'])
        result_numpy = f.read_text()
        # Different style = different output, proving convention was forwarded
        assert result_google != result_numpy

    def test_style_flag_sphinx(self, tmp_path: Path) -> None:
        long_summary = 'Summary.'
        long_desc = 'A very long description ' * 4 + 'that needs wrapping.'
        f = tmp_path / 'test.py'
        original = f'def foo(x):\n    """{long_summary}\n\n    :param x: {long_desc}\n    """\n'
        f.write_text(original)
        main([str(f), '--style', 'sphinx', '--write'])
        result = f.read_text()
        assert result != original  # wrapping happened
        # Re-run: idempotent under forced style
        exit_code = main([str(f), '--style', 'sphinx'])
        assert exit_code == 0


class TestSummaryMode:
    def test_file_without_docstrings_is_omitted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        a.write_text(_BODY_WRAPPABLE_SOURCE)
        b = tmp_path / 'b.py'
        b.write_text('x = 1\n')  # no docstrings -> 0 total
        main([str(a), str(b), '-v'])
        captured = capsys.readouterr()
        # b.py has total=0 and no skips, so it should not appear
        assert str(a) in captured.out
        assert str(b) not in captured.out


class TestSummaryStartFlag:
    @pytest.mark.parametrize(
        ('choice', 'configured', 'opening', 'expected_opening'),
        [
            ('first-line', 'second-line', '"""\n    ', '"""'),
            ('second-line', 'first-line', '"""', '"""\n    '),
            ('preserve', 'second-line', '"""', '"""'),
        ],
    )
    def test_summary_start_overrides_project_config(
        self,
        tmp_path: Path,
        choice: str,
        configured: str,
        opening: str,
        expected_opening: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            f'[tool.docwrap]\nsummary-start = "{configured}"\n'
        )
        f = tmp_path / 'test.py'
        body = 'Summary.\n\n    A separate body paragraph stays present.\n    """\n'
        f.write_text(f'def foo():\n    {opening}{body}')

        assert main([str(f), '--summary-start', choice, '--write']) == 0
        assert f.read_text() == f'def foo():\n    {expected_opening}{body}'


class TestVerboseIgnoredFiles:
    def test_verbose_ignored_with_diff(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-d', '-v'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert '---' in captured.out  # diff still works

    def test_verbose_ignored_with_preview(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        exit_code = main([str(f), '-p', '-v'])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert 'before:' in captured.out  # preview still works


class TestVerboseSummary:
    def test_v_for_verbose(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '-v'])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # With -v, header + data row shown even for unchanged file
        assert 'Name' in lines[0]
        assert str(f) in lines[1]
        assert '0' in lines[1]  # 0 wrapped

    def test_verbose_includes_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        a.write_text(_BODY_WRAPPABLE_SOURCE)
        b = tmp_path / 'b.py'
        b.write_text('def bar():\n    """Short."""\n')
        exit_code = main([str(a), str(b), '-v'])
        assert exit_code == 1
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        # Both files shown with -v
        assert any(str(a) in line for line in lines)
        assert any(str(b) in line for line in lines)


class TestVersion:
    def test_help_goes_to_stdout(self) -> None:
        status, stdout, stderr = invoke_cli(['--help'])
        assert (status, stderr) == (0, '')
        assert stdout.startswith('usage: docwrap')
        assert '--version' in stdout

    def test_missing_metadata_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, 'version', missing)

        status, stdout, stderr = invoke_cli(['--version'])
        assert (status, stdout) == (2, '')
        assert 'installed distribution metadata for docwrap is unavailable' in stderr

    def test_normal_runs_do_not_read_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def forbidden(name: str) -> str:
            raise AssertionError(f'metadata read for {name}')

        monkeypatch.setattr(importlib.metadata, 'version', forbidden)
        (tmp_path / 'mod.py').write_text('def a():\n    pass\n', encoding='utf-8')

        assert invoke_cli([str(tmp_path / 'mod.py')])[0] == _EXIT_OK

    def test_version_ignores_other_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def version(_name: str) -> str:
            return '9.8.7'

        monkeypatch.setattr(importlib.metadata, 'version', version)

        assert invoke_cli(['--version', '-d', 'missing.py']) == (0, 'docwrap 9.8.7\n', '')

    def test_version_prints_the_installed_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def version(name: str) -> str:
            assert name == 'docwrap'
            return '9.8.7'

        monkeypatch.setattr(importlib.metadata, 'version', version)

        assert invoke_cli(['--version']) == (0, 'docwrap 9.8.7\n', '')


class TestWrapFileOnce:
    def test_accepts_prepared_source_without_reading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = tmp_path / 'test.py'
        source = 'def foo():\n    """Short."""\n'
        p.write_text(source)
        prepared = ReadResult(source=source, encoding='utf-8', raw=source.encode())
        read = Mock(side_effect=AssertionError('prepared source must not be read again'))
        monkeypatch.setattr(cli_mod, 'read_file_result', read)

        result = _wrap_file_once(
            p,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None, prepared=prepared),
            None,
        )

        assert result.source == source
        assert result.encoding == 'utf-8'
        read.assert_not_called()

    def test_returns_wrapped_file(self, tmp_path: Path) -> None:
        p = tmp_path / 'test.py'
        p.write_text(
            'def foo():\n'
            '    """Hello world this is a very long line that should be wrapped'
            ' to the target."""\n'
        )
        result = _wrap_file_once(
            p,
            WrapOverrides(line_length=40),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )
        assert result.report is not None
        assert result.config.line_length == 40

    def test_unprepared_plan_reads_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        p = tmp_path / 'test.py'
        source = 'def foo():\n    """Short."""\n'
        p.write_text(source)
        read = Mock(wraps=read_file_result)
        monkeypatch.setattr(cli_mod, 'read_file_result', read)

        result = _wrap_file_once(
            p,
            WrapOverrides(),
            _FilePlan(state='full', eligible_ranges=None),
            None,
        )

        assert result.source == source
        read.assert_called_once_with(p)


class TestWrapSummaryFlag:
    def test_no_wrap_summary_flag_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--no-wrap-summary'])
        assert exit_code == 0

    def test_no_wrap_summary_overrides_config(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = true\n')
        f = tmp_path / 'test.py'
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--no-wrap-summary'])
        assert exit_code == 0  # No changes (summary not wrapped)

    def test_wrap_summary_flag_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        exit_code = main([str(f), '--wrap-summary'])
        assert exit_code == 0

    def test_wrap_summary_overrides_config(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = false\n')
        f = tmp_path / 'test.py'
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f.write_text(f'def foo():\n    """{long}"""\n')
        exit_code = main([str(f), '--wrap-summary'])
        assert exit_code == 1  # Changes would be made (summary wrapped)


class TestWriteMode:
    def test_write_modifies_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text(_BODY_WRAPPABLE_SOURCE)
        before = f.read_text()
        exit_code = main([str(f), '--write'])
        assert exit_code == 0
        result = f.read_text()
        assert result != before


def _assert_newline_form(raw: bytes, newline: bytes) -> None:
    if newline == b'\r\n':
        assert b'\r\n' in raw
        assert b'\n' not in raw.replace(b'\r\n', b'')
    else:
        assert b'\n' in raw
        assert b'\r' not in raw


def _closed_stdout_result(case: str, tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    if case == 'help-before-write':
        return run_with_closed_output(['--help'])
    target = tmp_path / 'target.py'
    source = _BODY_WRAPPABLE_SOURCE if case == 'small-before-write' else _BIG_WRAPPABLE
    target.write_bytes(source.encode('utf-8'))
    if case == 'after-first-line':
        return run_closing_after_first_line(['-d', str(target)])
    return run_with_closed_output(['-d', str(target)])


def _help_contains(help_text: str, substring: str) -> bool:
    return ' '.join(substring.split()) in ' '.join(help_text.split())


def _help_section_options(help_text: str, heading: str) -> list[tuple[str, ...]]:
    lines = help_text.splitlines()
    start = lines.index(f'{heading}:') + 1
    end = next(
        (
            index
            for index, line in enumerate(lines[start:], start)
            if line and not line.startswith(' ') and line.endswith(':')
        ),
        len(lines),
    )
    return [
        tuple(re.findall(r'(?<![\w-])--?[a-z][a-z-]*', line))
        for line in lines[start:end]
        if line.startswith('  -')
    ]


def _incremental_case_source(case: str) -> str:
    if case == 'syntax':
        return 'def broken(:\n'
    if case == 'ranges':
        return '"""Short."""\n'
    return 'x = 1\n'


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    run_setup(['git', 'init'], cwd=str(root))
    run_setup(
        [
            'git',
            '-c',
            'user.name=Test',
            '-c',
            'user.email=test@test.com',
            'commit',
            '-m',
            'init',
            '--allow-empty',
        ],
        cwd=str(root),
    )
    return root


def _no_diff(_path: str) -> None:
    return


def _configure_full_write_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> list[str]:
    if route == 'untracked':
        run_setup(['git', 'init'], cwd=tmp_path)
        (tmp_path / 'baseline.py').write_text('# baseline\n')
        git_commit_all(tmp_path, 'init')
        return ['--incremental-level', 'docstring']
    if route == 'configured-fallback':
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "docstring"\n')

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

        monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(cli_mod, 'get_repo_diff', _no_diff)
    return []


def _no_tracked_files(_path: str, _paths: Collection[str]) -> set[str]:
    return set()


def _not_worktree(_path: str) -> RepoDiscoveryResult:
    return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)


def _prepare_incremental_file_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case: str,
    configured: bool,
) -> tuple[Path, dict[str, int]]:
    path = tmp_path / 'a.py'
    source = _incremental_case_source(case)
    path.write_text(source)
    if configured:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "docstring"\n')

    def discover(_cwd: str) -> RepoDiscoveryResult:
        return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

    monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
    diff = (
        ''
        if case == 'no-hunks'
        else 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n'
    )

    def batch_diff(_root: str) -> str:
        return diff

    def tracked(_root: str, paths: Collection[str]) -> set[str]:
        return set(paths)

    monkeypatch.setattr(cli_mod, 'get_repo_diff', batch_diff)
    monkeypatch.setattr(cli_mod, 'get_tracked_files', tracked)
    reads = {'count': 0}

    def tracked_read(_path: Path) -> bytes:
        reads['count'] += 1
        if case == 'read':
            raise OSError('read failed')
        if case == 'decode':
            raise UnicodeDecodeError('utf-8', b'bad', 0, 1, 'cannot decode')
        return source.encode()

    monkeypatch.setattr(Path, 'read_bytes', tracked_read)
    if case == 'token':
        monkeypatch.setattr(
            cli_mod,
            'get_eligible_ranges',
            Mock(side_effect=tokenize.TokenError('token failed', (1, 0))),
        )
    return path, reads


def _prepare_invalid_file_level_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    configured: bool,
) -> tuple[Path, Mock]:
    path = tmp_path / 'a.py'
    path.write_text('def broken(:\n')
    if configured:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "file"\n')

    def discover(_cwd: str) -> RepoDiscoveryResult:
        return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(tmp_path))

    def batch_diff(_root: str) -> str:
        return 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n'

    def tracked(_root: str, paths: Collection[str]) -> set[str]:
        return set(paths)

    monkeypatch.setattr(cli_mod, 'discover_repo_root', discover)
    monkeypatch.setattr(cli_mod, 'get_repo_diff', batch_diff)
    monkeypatch.setattr(cli_mod, 'get_tracked_files', tracked)
    downstream = Mock(side_effect=AssertionError('downstream parser was called'))
    monkeypatch.setattr(cli_mod, 'wrap_source_report_eligible', downstream)
    return path, downstream


def _stats_location_counts(output: str) -> list[int]:
    counts: list[int] = []
    in_category = False
    for line in output.splitlines():
        if line.startswith('    ') and not line.startswith('      ') and line.endswith(':'):
            counts.append(0)
            in_category = True
        elif in_category and line.startswith('      '):
            counts[-1] += 1
        elif line and not line.startswith(' '):
            in_category = False
    return counts
