from pathlib import Path

import pytest

from docwrap._color import BOLD, GREEN, RED, RESET
from docwrap._display._print import _format_preview, print_preview, print_summary
from docwrap._types import DocstringChange, FileResult


class TestPrintPreview:
    def test_control_characters_are_escaped_before_preview_color(self) -> None:
        result = _preview_result(
            path='dir\x1b/file\t.py',
            qualname='Thing\rmethod\x9b',
            before='old\tline\nnext\x7f',
            after='new\x00line\né',
        )

        output = _format_preview([result], color=True)

        path = str(Path('dir\x1b/file\t.py')).replace('\x1b', r'\x1b').replace('\t', r'\x09')
        assert f'{BOLD}{path}:3:Thing\\x0dmethod\\x9b{RESET}' in output
        assert f'{RED}  before:\nold\\x09line\nnext\\x7f{RESET}' in output
        assert f'{GREEN}  after:\nnew\\x00line\né{RESET}' in output

    def test_ordinary_preview_ruler_is_unchanged(self) -> None:
        result = _preview_result(line_length=12)

        output = _format_preview([result], color=False)

        assert output == ('test.py:3:func\n  before:   |12\nold\n  after:    |12\nnew')

    def test_preview_ruler_caps_padding_and_retains_configured_length(self) -> None:
        result = _preview_result(line_length=1_000_000)

        output = _format_preview([result], color=False)
        before_label, after_label = (
            line for line in output.splitlines() if line.startswith(('  before:', '  after:'))
        )

        assert before_label.index('...') == 200
        assert after_label.index('...') == 200
        assert before_label.endswith('...|1000000')
        assert after_label.endswith('...|1000000')
        assert len(output) < 1_000

    def test_prints_nothing_for_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_preview(results=[], color=False)
        assert capsys.readouterr().out == ''


class TestPrintSummary:
    def test_no_skipped_footer(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            FileResult(
                status=0,
                path=Path('x.py'),
                line_length=79,
                total=1,
                skipped_summaries=3,
            )
        ]
        print_summary(results=results, verbose=True, color=False)
        out = capsys.readouterr().out
        assert 'Summaries skipped' not in out

    def test_prints_nothing_for_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_summary(results=[], verbose=False, color=False)
        assert capsys.readouterr().out == ''


def _preview_result(
    *,
    path: str = 'test.py',
    qualname: str = 'func',
    before: str = 'old',
    after: str = 'new',
    line_length: int | None = None,
) -> FileResult:
    change = DocstringChange(
        qualname=qualname,
        start_line=3,
        end_line=3,
        changed=True,
        before=before,
        after=after,
    )
    return FileResult(
        status=1,
        path=Path(path),
        line_length=line_length,
        total=1,
        wrapped=1,
        docstrings=(change,),
    )
