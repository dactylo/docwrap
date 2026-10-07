from pathlib import Path, PureWindowsPath
from unittest.mock import Mock

import pytest

from docwrap import _color
from docwrap._color import YELLOW
from docwrap._display import _summary as summary_module
from docwrap._display._summary import format_summary
from docwrap._types import FileResult

_PERCENTAGE_CASES = [
    (0, 0, '', 0),
    (0, 100, '0%', 0),
    (1, 101, '<1%', 1),
    (1, 1000, '<1%', 1),
    (1, 100, '1%', 1),
    (50, 100, '50%', 1),
]


class TestFormatSummary:
    def test_control_characters_are_escaped_before_widths_and_color(self) -> None:
        results = [
            _result('tab\t.py', 1, 1),
            _result('longer.py', 1, 1),
        ]

        plain_lines = format_summary(results, color=False).splitlines()
        colored = format_summary(results, color=True)

        assert plain_lines[1].startswith('tab\\x09.py  ')
        assert plain_lines[2].startswith('longer.py   ')
        numeric_columns = '     0      1     1   100%'
        assert plain_lines[1].index(numeric_columns) == plain_lines[2].index(numeric_columns)
        assert f'{YELLOW}tab\\x09.py' in colored
        assert '\t' not in colored

    def test_skip_column_in_header(self) -> None:
        results = [_result('x.py', 1, 2, skipped_summaries=1)]
        output = format_summary(results, verbose=False, color=False)
        header = output.splitlines()[0]
        assert 'Skip' in header

    def test_skip_only_file_hidden_in_default_mode(self) -> None:
        results = [_result('x.py', 0, 3, skipped_summaries=2)]
        output = format_summary(results, verbose=False, color=False)
        assert output == ''

    def test_skip_only_file_shown_in_verbose_mode(self) -> None:
        results = [_result('x.py', 0, 3, skipped_summaries=2)]
        output = format_summary(results, verbose=True, color=False)
        assert 'x.py' in output
        assert '2' in output  # skip count

    def test_total_row_sums_all_displayed_rows(self) -> None:
        results = [
            _result('a.py', 2, 5, skipped_summaries=1),
            _result('b.py', 1, 4, skipped_summaries=3),
            _result('c.py', 0, 3, skipped_summaries=2),
        ]
        output = format_summary(results, verbose=True, color=False)
        lines = output.splitlines()
        total_line = next(line for line in lines if 'TOTAL' in line)
        fields = total_line.split()
        assert fields[0] == 'TOTAL'
        assert fields[1] == '6'  # skip: 1+3+2
        assert fields[2] == '12'  # total: 5+4+3
        assert fields[3] == '3'  # wrap: 2+1+0
        assert fields[4] == '25%'  # 3/12


class TestPercentageColor:
    @pytest.mark.parametrize(
        ('wrapped', 'total', 'display', 'severity'),
        _PERCENTAGE_CASES,
    )
    def test_ordinary_and_total_rows_use_underlying_count(
        self,
        wrapped: int,
        total: int,
        display: str,
        severity: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        colorize_spy = Mock(wraps=summary_module._colorize_pct_row)
        monkeypatch.setattr(summary_module, '_colorize_pct_row', colorize_spy)
        results = [_result('a.py', wrapped, total)]
        if wrapped > 0:
            results.append(_result('b.py', wrapped, total))

        lines = format_summary(results, verbose=True, color=True).splitlines()
        ordinary = lines[1]
        observed = [
            (spy_call.args[1], spy_call.args[2], spy_call.kwargs.get('name'))
            for spy_call in colorize_spy.call_args_list
        ]
        expected: list[tuple[str, int, str | None]] = [(display, severity, 'a.py')]
        if wrapped > 0:
            expected.extend(
                [
                    (display, 1, 'b.py'),
                    (display, 1, None),
                ]
            )

        assert observed == expected
        assert (YELLOW in ordinary) is bool(severity)
        if display:
            assert display in ordinary
        if wrapped > 0:
            total_row = lines[-1]
            assert 'TOTAL' in total_row
            assert YELLOW in total_row
            assert display in total_row
        else:
            assert all('TOTAL' not in line for line in lines)


class TestPercentageFormatting:
    @pytest.mark.parametrize(
        ('wrapped', 'total', 'display', '_severity'),
        _PERCENTAGE_CASES,
    )
    def test_ordinary_and_total_rows_preserve_small_positive_values(
        self,
        wrapped: int,
        total: int,
        display: str,
        _severity: int,
    ) -> None:
        results = [_result('a.py', wrapped, total)]
        if wrapped > 0:
            results.append(_result('b.py', wrapped, total))

        lines = format_summary(results, verbose=True, color=False).splitlines()
        ordinary_fields = lines[1].split()
        if display:
            assert ordinary_fields[-1] == display
        else:
            assert ordinary_fields == ['a.py', '1', '0', '0']
        if wrapped > 0:
            total_fields = lines[-1].split()
            assert total_fields[0] == 'TOTAL'
            assert total_fields[-1] == display


class TestSummaryRounding:
    def test_percentage_uses_round_not_truncate(self) -> None:
        # 2 wrapped / 3 total = 66.66...% -> round() = 67, int() = 66 (different!)
        output = format_summary([_result('test.py', 2, 3)], color=False)
        # 2/3 = 66.67% -> round() = 67%
        assert '67%' in output
        # Not 66% (which int() truncation would produce)
        assert '66%' not in output


class TestSummaryTotal:
    def test_single_file_no_total_row(self) -> None:
        output = format_summary([_result('test.py', 1, 1)], color=False)
        assert 'test.py' in output
        assert 'TOTAL' not in output

    def test_total_aggregates_all_displayed_rows(self) -> None:
        results = [
            _result('a.py', 1, 2),
            _result('b.py', 1, 1),
            _result('c.py', 0, 1, skipped_summaries=0),
        ]
        output = format_summary(results, verbose=True, color=False)
        lines = output.strip().splitlines()
        total_line = lines[-1]
        assert 'TOTAL' in total_line
        # a.py: 1 wrapped / 2 total; b.py: 1 wrapped / 1 total
        # c.py: 0 wrapped / 1 total (unchanged, but displayed in -v)
        # TOTAL: 2 wrapped, 4 total
        assert '50%' in total_line  # round(2/4*100) = 50

    def test_two_changed_files_total_row(self) -> None:
        results = [
            _result('a.py', 1, 1),
            _result('b.py', 1, 1),
        ]
        output = format_summary(results, color=False)
        lines = output.strip().splitlines()
        assert 'TOTAL' in lines[-1]
        # TOTAL aggregates: 2 wrapped, 2 total, 100%
        assert '2' in lines[-1]
        assert '100%' in lines[-1]

    def test_verbose_with_one_changed_no_total(self) -> None:
        results = [
            _result('a.py', 1, 1),
            _result('b.py', 0, 1, skipped_summaries=0),
        ]
        output = format_summary(results, verbose=True, color=False)
        assert 'a.py' in output
        assert 'TOTAL' not in output


def _result(
    name: str,
    wrapped: int,
    total: int,
    *,
    skipped_summaries: int | None = None,
) -> FileResult:
    if skipped_summaries is None:
        skipped_summaries = 1 if wrapped == 0 else 0
    return FileResult(
        status=0,
        path=Path(name),
        line_length=79,
        total=total,
        wrapped=wrapped,
        skipped_summaries=skipped_summaries,
    )


def test_windows_summary_normalizes_only_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_color, 'PurePath', PureWindowsPath)
    result = _result(r'C:\repo\file.py', 1, 1)
    original = result.path
    output = format_summary([result], color=False)
    assert 'C:/repo/file.py' in output
    assert result.path == original
