import ast

import pytest

from docwrap._core._normalize import (
    normalize_content as _raw_normalize,
)
from docwrap._core._normalize import (
    parse_literal,
    reassemble,
    split_source_lines,
)
from docwrap._core._process import wrap_source, wrap_source_report
from docwrap._types import DocwrapConfig

BACKSLASH = chr(92)


class TestBackslashWidthAccounting:
    def test_backslash_heavy_idempotent(self) -> None:
        pattern = r'\d+ ' * 12 + 'text'
        escaped = pattern.replace('\\', '\\\\')
        source = f'def f():\n    """{escaped}"""\n'
        first = wrap_source(source, config=DocwrapConfig(line_length=50))
        second = wrap_source(first, config=DocwrapConfig(line_length=50))
        assert first == second

    def test_backslash_heavy_stays_within_limit(self) -> None:
        pattern = r'\d+ ' * 12 + 'text'
        escaped = pattern.replace('\\', '\\\\')
        source = f'def f():\n    """{escaped}"""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=50, wrap_summary=True))
        for line in result.splitlines():
            assert len(line) <= 50, f'Line exceeds 50: {line!r} ({len(line)} chars)'

    def test_raw_prefix_no_width_reduction(self) -> None:
        pattern = r'\d+ ' * 8 + 'text'
        source = f'def f():\n    r"""{pattern}"""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=60))
        # Raw prefix preserved, no doubling
        assert 'r"""' in result or "r'''" in result


class TestEscapedDelimiterCollision:
    def test_escaped_delimiter_keeps_quote_style(self) -> None:
        content = 'shows ' + BACKSLASH + '"' + '""' + ' spelled inline'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        assert result.startswith('"""')

    def test_escaped_double_run_with_real_single_run_switches_to_double(self) -> None:
        content = 'shows ' + BACKSLASH + '"' + '""' + ' and ' + "'''" + ' text'
        result = reassemble(content, "'''", '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        assert result.startswith('"""')

    def test_no_safe_quote_returns_none(self) -> None:
        content = 'both ' + '"""' + ' and ' + "'''" + ' present'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is None

    def test_unescaped_delimiter_switches_quotes(self) -> None:
        content = 'holds ' + '"""' + ' inside'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        assert result.startswith("'''")


class TestIndentation:
    def test_content_lines_indented(self) -> None:
        content = 'Summary.\n\nDescription.'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        lines = result.split('\n')
        # Lines between first and last should be indented by base_indent
        for line in lines[1:-1]:
            if line.strip():
                assert line.startswith('    ')

    def test_exact_source_margin_indents_multiline_literal(self) -> None:
        content = 'Summary.\nContinuation.'
        result = reassemble(
            content,
            '"""',
            '',
            base_indent=8,
            config=DocwrapConfig(),
            source_margin='\t ',
        )

        assert result == '"""Summary.\n\t Continuation.\n\t """'


class TestInlineDocstringWidth:
    def test_oneliner_fit_uses_col_offset(self) -> None:
        # 'def foo(): """Short."""' is 22 chars. With line_length=20 and
        # col_offset=11, the oneliner (11 + 12 = 23) should NOT fit.
        # But with base_indent=0 only, it would (0 + 12 = 12 <= 20).
        result = reassemble(
            'Short.', '"""', '', base_indent=0, config=DocwrapConfig(line_length=20), col_offset=11
        )
        assert result is not None
        # Should be multiline since 11 + len('"""Short."""') = 23 > 20
        assert '\n' in result


class TestMultiline:
    def test_backslash_not_doubled_for_non_raw_string(self) -> None:
        content = 'Regex: ' + BACKSLASH + BACKSLASH + 'd+'
        result = reassemble(
            content, '"""', '', base_indent=4, config=DocwrapConfig(line_length=99)
        )
        assert result is not None
        assert BACKSLASH + BACKSLASH + 'd+' in result
        assert BACKSLASH * 4 not in result

    def test_backslash_not_doubled_for_raw_string(self) -> None:
        content = 'Regex: \\d+'
        result = reassemble(
            content, '"""', 'r', base_indent=4, config=DocwrapConfig(line_length=99)
        )
        assert result is not None
        assert '\\\\' not in result  # raw strings keep single backslash

    def test_closing_quotes_aligned_to_base_indent(self) -> None:
        content = 'Summary.\n\nDescription.'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        lines = result.split('\n')
        # Closing quotes aligned to base_indent per requirements 15.2
        assert lines[-1] == '    """'

    def test_content_ending_with_quote_uses_multiline(self) -> None:
        result = reassemble(
            'Returns "value"', '"""', '', base_indent=4, config=DocwrapConfig(line_length=99)
        )
        assert result is not None
        # Must be multiline -- one-liner would produce '"""Returns "value""""'
        assert '\n' in result
        assert result.endswith('"""')

    def test_drops_r_prefix_trailing_backslash(self) -> None:
        # The backslash touches the closing delimiter in a one-line literal.
        result = reassemble(
            'Ends with \\', '"""', 'r', base_indent=4, config=DocwrapConfig(line_length=99)
        )
        assert result is not None
        assert not result.startswith('r')
        assert '\\\\' in result  # backslash escaped
        assert '\n' not in result
        assert ast.literal_eval(result) == 'Ends with \\'

    def test_multiline_format(self) -> None:
        content = 'Summary.\n\nDescription.'
        result = reassemble(content, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        lines = result.split('\n')
        assert lines[0] == '"""Summary.'
        assert lines[-1] == '    """'

    def test_one_line_raw_fallback_rechecks_quote_collision(self) -> None:
        content = 'Shows ' + BACKSLASH + '""" then ends ' + BACKSLASH

        result = reassemble(
            content, '"""', 'r', base_indent=4, config=DocwrapConfig(line_length=99)
        )

        assert result is not None
        assert '\n' not in result
        assert result.startswith("'''")
        assert ast.literal_eval(result) == content

    def test_preserves_prefix(self) -> None:
        result = reassemble('Short.', '"""', 'r', base_indent=4, config=DocwrapConfig())
        assert result == 'r"""Short."""'

    def test_preserves_r_prefix_with_safe_backslash(self) -> None:
        # Backslash in middle of content is valid in raw string
        result = reassemble(
            'Has a \\n newline.', '"""', 'r', base_indent=4, config=DocwrapConfig()
        )
        assert result is not None
        assert result.startswith('r')

    def test_preserves_raw_prefix_when_newline_separates_trailing_backslash(self) -> None:
        result = reassemble(
            'Summary.\n\nRoot is C:\\', '"""', 'r', base_indent=4, config=DocwrapConfig()
        )

        assert result == 'r"""Summary.\n\n    Root is C:\\\n    """'

    def test_preserves_single_quotes(self) -> None:
        result = reassemble('Short.', "'''", '', base_indent=4, config=DocwrapConfig())
        assert result == "'''Short.'''"

    def test_quote_collision_both_styles_returns_none(self) -> None:
        # Content contains both triple-quote styles -- unwrappable
        content = 'Has ' + '"""' + ' and ' + "'''" + ' inside.'
        result = reassemble(
            content, '"""', '', base_indent=4, config=DocwrapConfig(line_length=99)
        )
        assert result is None


class TestNormalizeContentOffset:
    def test_all_whitespace_strips_all_elements(self) -> None:
        _text, offset = _raw_normalize('   \n   \n', source_margin='')
        assert offset == 3  # splits to ['   ', '   ', ''], all blank

    def test_empty_string_strips_empty_element(self) -> None:
        _text, offset = _raw_normalize('', source_margin='')
        assert offset == 1  # '' splits to [''], one blank stripped

    def test_no_leading_blanks_returns_zero(self) -> None:
        _text, offset = _raw_normalize('Hello.', source_margin='')
        assert offset == 0

    def test_one_leading_blank_returns_one(self) -> None:
        _text, offset = _raw_normalize('\nHello.', source_margin='')
        assert offset == 1

    def test_two_leading_blanks_returns_two(self) -> None:
        _text, offset = _raw_normalize('\n\nHello.', source_margin='')
        assert offset == 2

    def test_whitespace_only_leading_blank_stripped(self) -> None:
        _text, offset = _raw_normalize('  \nHello.', source_margin='')
        assert offset == 1


class TestOneLiner:
    def test_leading_whitespace_takes_multiline_layout(self) -> None:
        result = reassemble('  Compact summary.', '"""', '', 4, DocwrapConfig())
        assert result is not None
        assert '\n' in result
        assert '  Compact summary.' in result

    def test_short_single_line(self) -> None:
        result = reassemble('Short.', '"""', '', base_indent=4, config=DocwrapConfig())
        assert result == '"""Short."""'

    def test_too_long_becomes_multiline(self) -> None:
        text = 'A ' + 'very ' * 20 + 'long summary.'
        result = reassemble(text, '"""', '', base_indent=4, config=DocwrapConfig())
        assert result is not None
        assert result.startswith('"""A ')
        assert result.endswith('"""')
        assert '\n' in result


class TestOriginalPositionDetection:
    def test_d212_docstring_preserved(self) -> None:
        long_desc = 'Description ' * 10 + 'end.'
        source = f'def foo():\n    """Short summary.\n\n    {long_desc}\n    """\n'
        config = DocwrapConfig(summary_start='preserve')
        report = wrap_source_report(source, config=config)
        result_lines = report.source.split('\n')
        # Summary on same line as opening quotes
        assert result_lines[1].strip().startswith('"""Short')

    def test_d213_docstring_preserved(self) -> None:
        source = (
            'def foo():\n'
            '    """\n'
            '    Short summary.\n'
            '\n'
            '    ' + 'Description ' * 10 + 'end.\n'
            '    """\n'
        )
        config = DocwrapConfig(summary_start='preserve')
        report = wrap_source_report(source, config=config)
        result_lines = report.source.split('\n')
        # Opening quotes should be alone on their line
        assert result_lines[1].strip() == '"""'
        # Summary on next line
        assert 'Short summary.' in result_lines[2]

    def test_d213_summary_width_no_quote_overhead(self) -> None:
        # In D212 mode, first-line width = 79 - 4 (indent) - 3 (""") = 72.
        # In D213 mode, summary-line width = 79 - 4 (indent) = 75.
        # A 73-char summary with a break point would be wrapped in D212
        # but should fit in D213.
        summary = 'A' * 30 + ' ' + 'B' * 42  # 73 chars
        source = f'def foo():\n    """\n    {summary}\n\n    Description.\n    """\n'
        config = DocwrapConfig(summary_start='preserve', wrap_summary=True)
        report = wrap_source_report(source, config=config)
        # Summary should NOT be wrapped because in D213 mode it fits
        assert summary in report.source


class TestParseLiteral:
    def test_content_is_raw_body_for_escape(self) -> None:
        raw = '"""Line with ' + BACKSLASH + 'n escape."""'
        info = parse_literal(raw)
        assert info is not None
        assert info.content == 'Line with ' + BACKSLASH + 'n escape.'

    def test_double_quotes(self) -> None:
        info = parse_literal('"""Hello."""')
        assert info is not None
        assert info.prefix == ''
        assert info.quote_char == '"""'
        assert info.content == 'Hello.'

    def test_implicit_concatenation_skipped(self) -> None:
        assert parse_literal('"""a""" """b"""') is None

    def test_r_prefix(self) -> None:
        info = parse_literal('r"""Hello."""')
        assert info is not None
        assert info.prefix == 'r'

    def test_single_quote_string_skipped(self) -> None:
        assert parse_literal('"Hello."') is None

    def test_single_quotes(self) -> None:
        info = parse_literal("'''Hello.'''")
        assert info is not None
        assert info.quote_char == "'''"

    def test_single_single_quote_skipped(self) -> None:
        assert parse_literal("'Hello.'") is None

    def test_u_prefix(self) -> None:
        info = parse_literal('u"""Hello."""')
        assert info is not None
        assert info.prefix == 'u'


class TestSummaryStartD213:
    def test_first_line_layout(self) -> None:
        config = DocwrapConfig(summary_start='first-line')
        result = reassemble(
            'Summary.\n\nDescription.',
            '"""',
            '',
            base_indent=4,
            config=config,
        )
        assert result is not None
        lines = result.split('\n')
        assert lines[0] == '"""Summary.'

    def test_oneliner_unaffected_by_summary_start(self) -> None:
        config = DocwrapConfig(summary_start='second-line')
        result = reassemble('Short.', '"""', '', base_indent=0, config=config)
        assert result == '"""Short."""'

    def test_preserve_defaults_to_first_line(self) -> None:
        config = DocwrapConfig(summary_start='preserve')
        result = reassemble(
            'Summary.\n\nDescription.',
            '"""',
            '',
            base_indent=4,
            config=config,
        )
        assert result is not None
        lines = result.split('\n')
        assert lines[0] == '"""Summary.'

    def test_preserve_keeps_original_position(self) -> None:
        config = DocwrapConfig(summary_start='preserve')
        # original_position='second-line' -> D213 layout
        result = reassemble(
            'Summary.\n\nDescription.',
            '"""',
            '',
            base_indent=4,
            config=config,
            original_position='second-line',
        )
        assert result is not None
        lines = result.split('\n')
        assert lines[0] == '"""'
        assert lines[1].strip() == 'Summary.'

    def test_second_line_layout(self) -> None:
        config = DocwrapConfig(summary_start='second-line')
        result = reassemble(
            'Summary.\n\nDescription.',
            '"""',
            '',
            base_indent=4,
            config=config,
        )
        assert result is not None
        lines = result.split('\n')
        assert lines[0] == '"""'
        assert lines[1].strip() == 'Summary.'


def _content(raw: str, *, source_margin: str = '') -> str:
    text, _offset = _raw_normalize(raw, source_margin=source_margin)
    return text


class TestNormalizeContent:
    def test_blank_lines_ignored_for_dedent(self) -> None:
        content = '    Hello.\n\n    World.'
        assert _content(content, source_margin='    ') == 'Hello.\n\nWorld.'

    def test_crlf_converted(self) -> None:
        assert _content('Hello.\r\nWorld.') == 'Hello.\nWorld.'

    def test_d213_relative_indentation_preserved(self) -> None:
        assert (
            _raw_normalize(
                '\n        code_line()\n    prose\n',
                source_margin='    ',
            )[0]
            == '    code_line()\nprose'
        )

    def test_dedent(self) -> None:
        content = '    Hello.\n    World.'
        assert _content(content, source_margin='    ') == 'Hello.\nWorld.'

    def test_empty_string(self) -> None:
        assert _content('') == ''

    def test_lone_cr_converted(self) -> None:
        assert _content('Hello.\rWorld.') == 'Hello.\nWorld.'

    def test_mixed_indent_preserves_whitespace_beyond_margin(self) -> None:
        content = '    Hello.\n        World.'
        assert _content(content, source_margin='    ') == 'Hello.\n    World.'

    def test_opening_line_leading_whitespace_is_idempotent(self) -> None:
        source = '''\
def f():
    """    indented first line
words at zero indent that are here
    """
'''
        expected = (
            'def f():\n'
            '    """indented first line\n'
            '    words at zero indent that are here\n'
            '    """\n'
        )

        first = wrap_source(source, config=DocwrapConfig())
        second = wrap_source(first, config=DocwrapConfig())

        assert first == expected
        assert second == expected

    @pytest.mark.parametrize(
        ('content', 'source_margin', 'expected'),
        [
            ('- item\n      continuation\n    ', '    ', '- item\n  continuation'),
            ('\titem\n\t\tcontinuation\n\t', '\t', 'item\n\tcontinuation'),
            ('\n    item\n\tcontinuation\n', '    ', '    item\n\tcontinuation'),
            ('Summary.\n        body\n    ', '    ', 'Summary.\n    body'),
        ],
    )
    def test_removes_only_exact_source_margin(
        self,
        content: str,
        source_margin: str,
        expected: str,
    ) -> None:
        normalized, _ = _raw_normalize(content, source_margin=source_margin)
        assert normalized == expected

    def test_single_line(self) -> None:
        assert _content('Hello.') == 'Hello.'

    def test_strip_leading_blank_lines(self) -> None:
        assert _content('\n\nHello.') == 'Hello.'

    def test_strip_trailing_blank_lines(self) -> None:
        assert _content('Hello.\n\n') == 'Hello.'

    def test_summary_first_line_body_indented(self) -> None:
        content = 'Summary.\n    Body text.'
        assert _content(content, source_margin='    ') == 'Summary.\nBody text.'

    def test_whitespace_only(self) -> None:
        assert _content('   \n   \n') == ''


class TestSplitSourceLines:
    @pytest.mark.parametrize(
        ('source', 'expected'),
        [
            ('', []),
            ('a\r\nb\nc\rZ', ['a\r\n', 'b\n', 'c\r', 'Z']),
            ('a\r\nb\nc\r', ['a\r\n', 'b\n', 'c\r']),
            ('a\u0085b\u2028c', ['a\u0085b\u2028c']),
        ],
    )
    def test_retains_only_python_line_terminators(self, source: str, expected: list[str]) -> None:
        lines = split_source_lines(source)
        assert lines == expected
        assert ''.join(lines) == source
