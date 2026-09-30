import ast
import textwrap
from collections.abc import Iterable

import pytest

import docwrap._core._wrap as wrap_module
from docwrap._core._discover import find_docstrings, parse_docstrings
from docwrap._core._parse import Region, parse_regions
from docwrap._core._process import wrap_source, wrap_source_report
from docwrap._core._wrap import (
    _effective_column,
    _fill,
    compute_quote_overhead,
    detect_summary_skip,
    wrap_regions,
)
from docwrap._styles import get_style, get_styles
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    SummaryStart,
)
from tests.helpers import assert_region_projection

BACKSLASH = chr(92)
SUMMARY_CLASSIFIER_CASES = (
    (('lead', ':param x: detail'), (1,)),
    (('lead', ':return: detail'), (1,)),
    (('lead', 'Returns:'), (1,)),
    (('lead', 'Returns', '-------', 'detail'), (1,)),
    (('lead', '- item'), (1,)),
    (('lead', '1. item'), (1,)),
    (('lead', '```python', 'value = 1', '```'), (1, 2, 3)),
    (('lead', '>>> value'), (1,)),
    (('lead::', '', '    value = 1'), (1, 2)),
    (('lead', '    indented continuation'), (1,)),
)
SUMMARY_STRUCTURAL_CASES = (
    pytest.param(
        'w' * 66 + ' :param x: detail',
        ':param',
        id='sphinx-field',
    ),
    pytest.param(
        'w' * 66 + ' :return: detail',
        ':return:',
        id='sphinx-field-role-alias',
    ),
    pytest.param(
        'w' * 68 + ' Returns: ' + 'x' * 72,
        'Returns:',
        id='google-returns',
    ),
    pytest.param(
        'w' * 68 + ' Returns ' + '-' * 72 + ' detail',
        'Returns',
        id='numpy-returns',
    ),
)


class TestBodyMarkerBoundaries:
    @pytest.mark.parametrize('reflow', ['cautious', 'full'])
    def test_bare_marker_reflow_is_idempotent(self, reflow: ReflowMode) -> None:
        source = textwrap.dedent('''\
            def f():
                """Summary.

                -
                alpha
                  gamma delta
                """
        ''')
        config = DocwrapConfig(reflow=reflow)

        first = wrap_source(source, config=config)

        assert wrap_source(first, config=config) == first

    @pytest.mark.parametrize('field', [':param x:', ':return:', ':raises ValueError:'])
    @pytest.mark.parametrize('fill_style', ['greedy', 'balanced', 'smart'])
    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_forced_inline_field_preserves_body_regions(
        self, field: str, fill_style: FillStyle, reflow: ReflowMode
    ) -> None:
        source = (
            'def f():\n    """Summary.\n\n'
            f'    alpha beta gamma delta eta {field} value with further explanatory words.\n'
            '    """\n'
        )
        config = DocwrapConfig(line_length=30, fill_style=fill_style, reflow=reflow)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.wrapped == 1
        assert first.skips == second.skips == ()
        assert first.source == second.source
        assert_region_projection(source, config)

    @pytest.mark.parametrize('field', [':param x:', ':return:', ':raises ValueError:'])
    @pytest.mark.parametrize('fill_style', ['greedy', 'balanced', 'smart'])
    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_initial_authored_field_remains_an_entry(
        self, field: str, fill_style: FillStyle, reflow: ReflowMode
    ) -> None:
        source = (
            'def f():\n    """Summary.\n\n'
            f'    {field} A detailed description with further explanatory words.\n'
            '    """\n'
        )
        config = DocwrapConfig(line_length=40, fill_style=fill_style, reflow=reflow)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.wrapped == 1
        assert first.skips == second.skips == ()
        assert f'    {field} ' in first.source
        assert first.source == second.source
        assert_region_projection(source, config)

    @pytest.mark.parametrize('fill_style', ['greedy', 'balanced', 'smart'])
    def test_inline_field_binding_preserves_gap(self, fill_style: FillStyle) -> None:
        assert _fill('alpha beta  :param x: value', 12, fill_style=fill_style) == (
            'alpha\nbeta  :param\nx: value'
        )

    @pytest.mark.parametrize('fill_style', ['greedy', 'balanced', 'smart'])
    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_inline_sphinx_field_preserves_body_regions(
        self, fill_style: FillStyle, reflow: ReflowMode
    ) -> None:
        source = (
            'def f():\n    """Summary.\n\n'
            '    This function formats the supplied field list and preserves the '
            'names and types used by the calling application when :param x: '
            'description with some detailed words and further explanatory '
            'material after the marker.\n    """\n'
        )
        config = DocwrapConfig(line_length=72, fill_style=fill_style, reflow=reflow)
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert first.wrapped == 1
        assert first.skips == second.skips == ()
        assert first.source == second.source
        assert_region_projection(source, config)

    @pytest.mark.parametrize(
        ('word', 'expected'),
        [
            (':class:', 'alpha beta\n:class: value'),
            (':parameterized', 'alpha beta\n:parameterized\nvalue'),
        ],
    )
    @pytest.mark.parametrize('fill_style', ['greedy', 'balanced', 'smart'])
    def test_ordinary_colon_word_can_start_a_line(
        self, word: str, expected: str, fill_style: FillStyle
    ) -> None:
        result = _fill(f'alpha beta {word} value', 13, fill_style=fill_style)

        assert result == expected


class TestD213InlineWidth:
    def test_inline_d213_uses_base_indent_not_col_offset(self) -> None:
        # Inline D213: def foo(): """\n    Summary\n    """\n
        # col_offset=11 (where """ starts), base_indent=0 (def's indent).
        # D213 summary sits at base_indent, so width = 79 - 0 = 79.
        # Bug: code uses col_offset -> width = 79 - 11 = 68, wrongly wrapping.
        summary = 'A' * 30 + ' ' + 'B' * 42  # 73 chars, fits in 79 but not 68
        source = f'def foo(): """\n    {summary}\n\n    Description.\n    """\n'
        config = DocwrapConfig(summary_start='preserve', wrap_summary=True)
        report = wrap_source_report(source, config=config)
        # Summary should NOT be wrapped -- 73 chars fits in 79-char width
        assert summary in report.source


class TestEscapeSkipWidth:
    def test_source_escape_spelling_counts_toward_skip_width(self) -> None:
        escape_part = (BACKSLASH + 'x41 ') * 5
        filler = 'x' * (71 - len(escape_part))
        summary = escape_part + filler
        assert len(summary) == 71
        # Indent + opening quotes + source body + closing quotes = 81.
        assert 4 + 3 + len(summary) + 3 == 81
        source = f'def foo():\n    """{summary}"""\n'
        config = DocwrapConfig(wrap_summary=False)
        report = wrap_source_report(source, config=config)

        assert report.source == source
        assert report.skipped_summaries == 1


class TestEscapeWidthExactness:
    def test_escape_token_moves_whole_at_boundary(self) -> None:
        body = (
            'Words fill this summary until the escape `' + BACKSLASH + 'n` lands right '
            'at the wrap boundary of the line.'
        )
        source = 'def f():\n    """' + body + '"""\n'
        cfg = DocwrapConfig(line_length=50, wrap_summary=True)
        result = wrap_source(source, config=cfg)
        for line in result.splitlines():
            assert len(line) <= 50, f'line exceeds 50: {line!r}'
            if BACKSLASH in line:
                assert BACKSLASH + 'n' in line  # token never split
        assert wrap_source(result, config=cfg) == result

    def test_named_escape_never_splits(self) -> None:
        spelling = BACKSLASH + 'N{LATIN CAPITAL LETTER A}'
        body = 'Some leading filler words push ' + spelling + ' against the limit here.'
        source = 'def f():\n    """' + body + '"""\n'
        cfg = DocwrapConfig(line_length=45, wrap_summary=True)
        result = wrap_source(source, config=cfg)
        assert spelling in result
        assert 'N{LATIN' not in result.replace(spelling, '')  # appears only intact
        for line in result.splitlines():
            assert len(line) <= 45, f'line exceeds 45: {line!r}'
        assert wrap_source(result, config=cfg) == result


class TestFillStyle:
    def test_balanced_budget_falls_back_to_greedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '(2 parts). Yields (header, dest_path) pairs.'
        )
        monkeypatch.setattr(wrap_module, '_BALANCED_CANDIDATE_BUDGET', 1, raising=False)
        expected = _fill(text, 75, initial='    ', fill_style='greedy')

        result = _fill(text, 75, initial='    ', fill_style='balanced')

        assert result == expected
        assert _fill(result, 75, initial='    ', fill_style='balanced') == result

    def test_balanced_uses_minimum_line_count(self) -> None:
        assert _fill('aaaa bbbb cccc dddd', 14, fill_style='balanced') == ('aaaa bbbb cccc\ndddd')

    def test_balanced_uses_weighted_raggedness_between_two_line_plans(self) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '(2 parts). Yields (header, dest_path) pairs.'
        )
        assert _fill(text, 75, initial='    ', fill_style='balanced') == (
            '    Handles rename/copy advancement (3 null-separated parts) vs normal\n'
            '    (2 parts). Yields (header, dest_path) pairs.'
        )

    def test_empty_input_returns_empty_string(self) -> None:
        assert _fill('', 20, initial='    ', fill_style='balanced') == ''
        assert _fill('   ', 20, initial='    ', fill_style='balanced') == ''
        assert _fill('', 20, initial='    ', fill_style='smart') == ''
        assert _fill('   ', 20, initial='    ', fill_style='smart') == ''

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_expands_tabs_without_splitting_the_line(self, fill_style: FillStyle) -> None:
        assert _fill('a\tb', 40, fill_style=fill_style) == 'a       b'

    @pytest.mark.parametrize(
        ('fill_style', 'smart_slack'),
        [('greedy', 8), ('smart', 0)],
    )
    def test_greedy_compatible_modes_preserve_leading_boundary_whitespace(
        self,
        fill_style: FillStyle,
        smart_slack: int,
    ) -> None:
        assert (
            _fill(
                '  alpha beta  ',
                8,
                fill_style=fill_style,
                smart_slack=smart_slack,
            )
            == '  alpha\nbeta'
        )

    def test_greedy_emits_expected_physical_lines(self) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '(2 parts). Yields (header, dest_path) pairs.'
        )
        assert _fill(text, 75, initial='    ', fill_style='greedy') == (
            '    Handles rename/copy advancement (3 null-separated parts) vs normal (2\n'
            '    parts). Yields (header, dest_path) pairs.'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_internal_marker_shaped_word_cannot_start_a_line(
        self,
        fill_style: FillStyle,
    ) -> None:
        assert _fill('alpha beta 1. gamma delta', 12, fill_style=fill_style) == (
            'alpha\nbeta 1.\ngamma delta'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_long_single_word_is_not_split(self, fill_style: FillStyle) -> None:
        assert _fill('supercalifragilistic', 8, initial='  ', fill_style=fill_style) == (
            '  supercalifragilistic'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_keeps_unbreakable_long_word(self, fill_style: FillStyle) -> None:
        assert (
            _fill(
                'supercalifragilistic  tail',
                8,
                initial='  ',
                fill_style=fill_style,
            )
            == '  supercalifragilistic\n  tail'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_replans_with_different_indents(self, fill_style: FillStyle) -> None:
        assert (
            _fill(
                'aa  bbb cccc',
                11,
                initial='>>>> ',
                subsequent='>',
                fill_style=fill_style,
            )
            == '>>>> aa bbb\n>cccc'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_selected_boundaries_stabilize_in_one_call(
        self,
        fill_style: FillStyle,
    ) -> None:
        assert (
            _fill(
                'aaaa  bbbb   cccc    dddd',
                8,
                fill_style=fill_style,
            )
            == 'aaaa\nbbbb\ncccc\ndddd'
        )

    @pytest.mark.parametrize('gap', ['  ', '   '])
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_selected_boundary_becomes_newline(
        self,
        fill_style: FillStyle,
        gap: str,
    ) -> None:
        assert _fill(f'aaaa{gap}bbbb', 8, fill_style=fill_style) == 'aaaa\nbbbb'

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_selected_gap_reunites_as_one_space(self, fill_style: FillStyle) -> None:
        assert _fill('aa  bbb c', 6, fill_style=fill_style) == 'aa bbb\nc'

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_multispace_unselected_gap_remains_exact(self, fill_style: FillStyle) -> None:
        assert _fill('aa  bb ccccc', 7, fill_style=fill_style) == 'aa  bb\nccccc'

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_non_structural_word_retains_ordinary_fill(
        self,
        fill_style: FillStyle,
    ) -> None:
        assert _fill('alpha beta gamma ^^^^', 12, fill_style=fill_style) == (
            'alpha beta\ngamma ^^^^'
        )

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_preserves_interior_double_spaces(self, fill_style: FillStyle) -> None:
        assert _fill('hello  world  foo', 40, fill_style=fill_style) == 'hello  world  foo'

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_respects_different_indent_widths(self, fill_style: FillStyle) -> None:
        assert (
            _fill(
                'alpha beta gamma delta',
                12,
                initial='>>>> ',
                subsequent='>',
                fill_style=fill_style,
            )
            == '>>>> alpha\n>beta gamma\n>delta'
        )

    def test_single_line_fit_fast_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        words = [f'w{i}' for i in range(2000)]
        text = ' '.join(words)
        width = len(text) + 20

        def fail_planner(*_args: object, **_kwargs: object) -> None:
            pytest.fail('single-line input reached the fill planner')

        monkeypatch.setattr(wrap_module, '_stabilize_fill_plan', fail_planner)

        result = _fill(text, width, initial='    ', fill_style='balanced')

        assert result == '    ' + text

    def test_smart_default_repairs_medium_parenthesized_fragment(self) -> None:
        text = 'aaaa bbbb (e.g. parts) tail'
        assert _fill(text, 18, fill_style='smart') == 'aaaa bbbb\n(e.g. parts) tail'

    def test_smart_does_not_back_off_to_empty_line(self) -> None:
        text = '(2 parts). remaining words'
        assert _fill(text, 3, fill_style='smart') == _fill(text, 3, fill_style='greedy')

    def test_smart_does_not_move_ordinary_word(self) -> None:
        text = 'Resolved config, or plain defaults if the config file has an unrecognized style.'
        assert _fill(text, 79, initial='        ', fill_style='smart') == (
            '        Resolved config, or plain defaults if the config file has an\n'
            '        unrecognized style.'
        )

    def test_smart_keeps_bracketed_phrase_together(self) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '[2 parts]. Yields (header, dest_path) pairs.'
        )
        result = _fill(text, 75, initial='    ', fill_style='smart')
        lines = result.splitlines()
        assert lines[0].endswith('normal')
        assert lines[1].lstrip().startswith('[2 parts].')

    def test_smart_keeps_parenthesized_phrase_together(self) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '(2 parts). Yields (header, dest_path) pairs.'
        )
        result = _fill(text, 75, initial='    ', fill_style='smart')
        lines = result.splitlines()
        assert lines[0].endswith('normal')
        assert lines[1].lstrip().startswith('(2 parts).')

    def test_smart_narrow_width_backoff_may_increase_lines(self) -> None:
        text = 'xxxx (2 parts) yy'
        greedy = _fill(text, 10, fill_style='greedy')
        smart = _fill(text, 10, fill_style='smart')
        # Smart may produce more lines than greedy; this is expected
        # for narrow widths -- local repair, not global optimization.
        assert smart.count('\n') >= greedy.count('\n')

    def test_smart_nested_opener_does_not_false_positive_backoff(self) -> None:
        text = 'aaaa bbbb (x(y z) tail'
        result = _fill(text, 16, fill_style='smart', smart_slack=5)
        greedy = _fill(text, 16, fill_style='greedy')
        assert result == greedy

    def test_smart_rejects_backoff_over_slack_budget(self) -> None:
        text = 'aaaa bbbb (12345678 parts) x'
        assert _fill(text, 19, fill_style='smart') == 'aaaa bbbb (12345678\nparts) x'

    def test_smart_rejects_backoff_that_exposes_previous_line_repair(self) -> None:
        text = 'aaaa (x (zz qq) y)'
        assert _fill(text, 12, fill_style='smart') == _fill(text, 12, fill_style='greedy')

    def test_smart_rejects_backoff_that_pushes_content_later(self) -> None:
        text = 'aaaa bbbb (x y) c d e f g'
        assert _fill(text, 12, fill_style='smart') == _fill(text, 12, fill_style='greedy')

    def test_smart_rejects_backoff_that_would_skip_next_repair(self) -> None:
        text = 'aaaa bbbb (x y) cc (zz qq)'
        assert _fill(text, 12, fill_style='smart') == 'aaaa bbbb (x\ny) cc\n(zz qq)'

    def test_smart_rejects_backoff_when_delimiter_group_would_still_wrap(self) -> None:
        text = 'aaaa bbbb (x m n yyyyyyyyyy)'
        assert _fill(text, 12, fill_style='smart') == _fill(text, 12, fill_style='greedy')

    def test_smart_slack_propagates_through_wrap_source(self) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    Handles rename/copy advancement (3 null-separated parts) vs normal (2 parts).'
            ' Yields (header, dest_path) pairs.\n'
            '    """\n'
        )
        slack_0 = wrap_source(source, config=DocwrapConfig(smart_slack=0))
        slack_4 = wrap_source(source, config=DocwrapConfig(smart_slack=4))
        assert slack_0 != slack_4

    def test_smart_slack_zero_matches_greedy(self) -> None:
        text = (
            'Handles rename/copy advancement (3 null-separated parts) vs normal '
            '(2 parts). Yields (header, dest_path) pairs.'
        )
        assert _fill(text, 75, initial='    ', fill_style='smart', smart_slack=0) == _fill(
            text, 75, initial='    ', fill_style='greedy'
        )

    def test_smart_unrelated_closer_does_not_trigger_backoff(self) -> None:
        text = 'aaaa bbbb cccc (x ddd (done) eeee'
        result = _fill(text, 20, fill_style='smart')
        greedy = _fill(text, 20, fill_style='greedy')
        # Greedy puts '(x' at line end.  Subsequent words include
        # '(done)' whose ')' belongs to a different group.
        # Smart should NOT back off -- '(x' has no matching closer.
        assert result == greedy

    @pytest.mark.parametrize('adornment', ['=========', '---------', '~~~~~~~~~'])
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_structural_word_cannot_start_a_line(
        self,
        adornment: str,
        fill_style: FillStyle,
    ) -> None:
        text = f'alpha beta gamma {adornment} delta'

        result = _fill(text, 16, fill_style=fill_style)

        assert all(not line.startswith(adornment) for line in result.splitlines())
        assert result.split() == text.split()

    @pytest.mark.parametrize('adornment', ['=========', '---------', '~~~~~~~~~'])
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_structural_word_final_token_cannot_start_a_line(
        self,
        adornment: str,
        fill_style: FillStyle,
    ) -> None:
        text = f'alpha beta gamma {adornment}'

        result = _fill(text, 16, fill_style=fill_style)

        assert all(not line.startswith(adornment) for line in result.splitlines())
        assert result.split() == text.split()

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_terminal_marker_shaped_word_cannot_start_a_line(
        self,
        fill_style: FillStyle,
    ) -> None:
        assert _fill('alpha beta 1.', 10, fill_style=fill_style) == 'alpha\nbeta 1.'

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_terminal_marker_wrap_is_idempotent(self, fill_style: FillStyle) -> None:
        source = 'def f():\n    """- [value] 1. y\n    """\n'
        config = DocwrapConfig(
            fill_style=fill_style,
            line_length=10,
            reflow='none',
            wrap_summary=True,
        )

        first = wrap_source(source, config=config)

        assert wrap_source(first, config=config) == first

    def test_text_fitting_one_line_is_not_wrapped(self) -> None:
        assert _fill('short line', 40, initial='  ', fill_style='balanced') == '  short line'
        assert _fill('short line', 40, initial='  ', fill_style='smart') == '  short line'

    def test_zero_or_negative_capacity_places_one_word_per_line(self) -> None:
        assert (
            _fill(
                'one two three',
                2,
                initial='>>>>',
                subsequent='>>>',
                fill_style='balanced',
            )
            == '>>>>one\n>>>two\n>>>three'
        )


class TestGoogleEntryWrap:
    def test_description_matching_param_name(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: x\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        assert '        x: x\n' in result

    def test_entry_idempotent(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = f'def foo(x):\n    """Summary.\n\n    Args:\n        x: {long_desc}\n    """\n'
        result1 = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        result2 = wrap_source(
            result1, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        assert result1 == result2

    def test_entry_no_description(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x:\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        assert '        x:\n' in result

    def test_entry_with_type_wraps(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = (
            f'def foo(x):\n    """Summary.\n\n    Args:\n        x (int): {long_desc}\n    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        lines = result.splitlines()
        assert all(len(line) <= DEFAULT_LINE_LENGTH for line in lines)
        # First line should retain the entry marker
        entry_line = next(ln for ln in lines if 'x (int):' in ln)
        assert entry_line.startswith('        ')

    def test_long_entry_description_wraps(self) -> None:
        long_desc = 'A very long description ' * 4 + 'that must wrap.'
        source = f'def foo(x):\n    """Summary.\n\n    Args:\n        x: {long_desc}\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        lines = result.splitlines()
        over = [ln for ln in lines if len(ln) > DEFAULT_LINE_LENGTH]
        assert all(len(line) <= DEFAULT_LINE_LENGTH for line in lines), (
            f'Lines exceed {DEFAULT_LINE_LENGTH}: {over}'
        )
        # Continuation lines should be at indent 12 (base 4 + entry cont 8)
        cont_lines = [ln for ln in lines if ln.startswith('            ') and 'x:' not in ln]
        assert len(cont_lines) > 0

    def test_multiple_entries_wrap_independently(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = (
            f'def foo(x, y):\n    """Summary.\n\n    Args:\n'
            f'        x: {long_desc}\n'
            f'        y: {long_desc}\n    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        lines = result.splitlines()
        assert all(len(line) <= DEFAULT_LINE_LENGTH for line in lines)
        # Both entry markers should be present
        assert sum(1 for ln in lines if 'x:' in ln and ln.strip().startswith('x:')) == 1
        assert sum(1 for ln in lines if 'y:' in ln and ln.strip().startswith('y:')) == 1

    def test_short_entry_untouched(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: Short desc.\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        assert result == source


class TestInjectiveProseRuns:
    def test_colon_free_google_prose_keeps_safe_promotion(self) -> None:
        regions = [
            Region(kind='section_header', text='Notes:'),
            Region(
                kind='prose',
                text='ordinary prose',
                indent=2,
                original_lines=('  ordinary prose',),
            ),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == ['Notes:', '    ordinary prose']

    def test_contextual_google_args_rejects_entry_reclassification(self) -> None:
        regions = [
            Region(kind='section_header', text='Args:', role='param'),
            Region(
                kind='prose',
                text='callback: A callable.',
                indent=2,
                original_lines=('  callback: A callable.',),
                role='param',
            ),
            Region(
                kind='prose',
                text='continuation text.',
                indent=12,
                original_lines=('            continuation text.',),
                role='param',
            ),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == [
            'Args:',
            '  callback: A callable.',
            '            continuation text.',
        ]

    def test_distinct_indents_receive_one_shared_delta(self) -> None:
        regions = [
            Region(kind='prose', text='first paragraph', indent=1),
            Region(kind='prose', text='second paragraph', indent=3),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == [
            '    first paragraph',
            '      second paragraph',
        ]

    @pytest.mark.parametrize(
        'breaker',
        [
            pytest.param(Region(kind='blank', text=''), id='blank'),
            pytest.param(Region(kind='entry', text='    x: value'), id='entry'),
            pytest.param(Region(kind='list_item', text='- item'), id='list-item'),
            pytest.param(Region(kind='code_block', text='    code()'), id='code-block'),
            pytest.param(Region(kind='section_header', text='Notes:'), id='header'),
            pytest.param(Region(kind='title', text='Title\n-----'), id='title'),
            pytest.param(Region(kind='summary', text='Summary.'), id='summary'),
            pytest.param(Region(kind='description', text='Description.'), id='description'),
            pytest.param(Region(kind='prose', text='----------'), id='structural-underline'),
        ],
    )
    def test_every_structural_boundary_breaks_a_run(self, breaker: Region) -> None:
        regions = [
            Region(kind='prose', text='before', indent=0),
            breaker,
            Region(kind='prose', text='after', indent=2),
        ]
        assert wrap_module._prose_runs(regions) == [(0, 1), (2, 3)]

    def test_following_google_entry_rejects_indent_collision(self) -> None:
        style = get_style('google')
        regions = [
            Region(kind='prose', text='prose', indent=0),
            Region(
                kind='entry',
                text='    x: value',
                original_lines=('    x: value',),
            ),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=style,
        )
        assert wrapped.splitlines() == ['prose', '    x: value']

    def test_google_bracketed_return_rejects_conservative_entry_shape(self) -> None:
        regions = [
            Region(kind='section_header', text='Returns:', role='returns'),
            Region(
                kind='prose',
                text='dict[str, int]: description',
                indent=2,
                original_lines=('  dict[str, int]: description',),
                role='returns',
            ),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == [
            'Returns:',
            '  dict[str, int]: description',
        ]

    def test_numpy_zero_floor_is_a_noop(self) -> None:
        wrapped = wrap_regions(
            [Region(kind='prose', text='prose', indent=2)],
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('numpy'),
        )
        assert wrapped == '  prose'

    def test_preceding_entry_rejects_only_a_threshold_crossing(self) -> None:
        sphinx_wrapped = wrap_regions(
            [
                Region(
                    kind='entry',
                    text=':param x: value',
                    original_lines=(':param x: value',),
                ),
                Region(kind='prose', text='prose', indent=0),
            ],
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('sphinx'),
        )
        google_wrapped = wrap_regions(
            [
                Region(
                    kind='entry',
                    text='    x: value',
                    original_lines=('    x: value',),
                ),
                Region(kind='prose', text='prose', indent=0),
            ],
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert sphinx_wrapped.splitlines()[-1] == 'prose'
        assert google_wrapped.splitlines()[-1] == '    prose'

    def test_preceding_list_rejects_only_a_threshold_crossing(self) -> None:
        crossing = wrap_regions(
            [
                Region(kind='list_item', text='- item'),
                Region(kind='prose', text='first', indent=0),
                Region(kind='prose', text='second', indent=2),
            ],
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        safe = wrap_regions(
            [
                Region(kind='list_item', text='100. item'),
                Region(kind='prose', text='first', indent=0),
                Region(kind='prose', text='second', indent=2),
            ],
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert crossing.splitlines() == ['- item', 'first', '  second']
        assert safe.splitlines() == ['100. item', '    first', '      second']

    def test_scanner_guard_keeps_authored_code_protected(self) -> None:
        regions = [
            Region(kind='prose', text='Example::', indent=0, source_start=0),
            Region(kind='blank', text='', source_start=1),
            Region(
                kind='code_block',
                text='    code()',
                original_lines=('    code()',),
                source_scanner_protected=True,
                source_start=2,
            ),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == ['Example::', '', '    code()']

    def test_scanner_guard_rejects_new_literal_block_membership(self) -> None:
        regions = [
            Region(kind='description', text='Example::', source_start=0),
            Region(kind='blank', text='', source_start=1),
            Region(kind='prose', text='ordinary prose', indent=0, source_start=2),
        ]
        wrapped = wrap_regions(
            regions,
            DocwrapConfig(line_length=99),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped.splitlines() == ['Example::', '', 'ordinary prose']

    def test_verbatim_regions_bypass_target_dispatch(self) -> None:
        region = Region(
            kind='prose',
            text='reconstructed text',
            indent=2,
            original_lines=('  authored line', '      authored continuation'),
            preserve_verbatim=True,
        )
        wrapped = wrap_regions(
            [region],
            DocwrapConfig(line_length=20, reflow='full'),
            base_indent=0,
            style=get_style('google'),
        )
        assert wrapped == '  authored line\n      authored continuation'


class TestListContinuationBug:
    def test_continuation_indent_preserved(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Two-pass approach:
                1. Neutralize ``"`` and ``'`` inside heredoc bodies so ``DQUOTE_RE``
                   cannot terminate early at an inner quote.
                2. Blank double-quoted and single-quoted string contents.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='full')
        report = wrap_source_report(source, config=config)
        body_lines = [
            ln
            for ln in report.source.splitlines()
            if ln.strip() and not ln.strip().startswith(('"""', "'''", 'def '))
        ]
        for line in body_lines:
            stripped = line.lstrip()
            # Skip indent prefix, then check for stray internal
            # whitespace (4+ consecutive spaces).  For list items,
            # the marker prefix (e.g., "1. ") naturally has 1-3
            # spaces; stray whitespace appears AFTER the marker.
            if stripped.startswith(('1.', '2.')):
                # Check the text after "N. " for stray internal spaces
                after_marker = stripped.split('. ', 1)[-1]
                assert '    ' not in after_marker, f'Stray whitespace after marker in: {line!r}'
            else:
                assert '    ' not in stripped, f'Stray whitespace in: {line!r}'

    def test_list_reflow_full_joins_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                - First part of
                  the item.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='full')
        report = wrap_source_report(source, config=config)
        assert '- First part of the item.' in report.source

    def test_list_reflow_none_preserves_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                - First part of
                  the item.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='none')
        report = wrap_source_report(source, config=config)
        assert '- First part of\n' in report.source
        assert '  the item.' in report.source

    def test_nested_list_marker_terminates_parent(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                1. Parent item
                   - nested bullet
                   - another nested
                2. Second parent.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='full')
        report = wrap_source_report(source, config=config)
        # Nested bullets should NOT be joined into parent.
        # Check exact line structure: each nested bullet on its own line.
        lines = report.source.splitlines()
        nested_lines = [ln.strip() for ln in lines if '- nested' in ln or '- another' in ln]
        assert nested_lines == ['- nested bullet', '- another nested']

    def test_numbered_marker_preserves_stable_prose_boundary(self) -> None:
        source = '''\
def f():
    """Summary.

    Args:
        x (int): parameter description here.
    100. numbered item
    trailing prose line
    """
'''
        first = wrap_source(source)
        second = wrap_source(first)
        assert '    100. numbered item\n        trailing prose line' in first
        assert second == first

    def test_short_marker_preserves_prose_boundary(self) -> None:
        source = '''\
def f():
    """Summary.

    Args:
        x (int): parameter description here.
    - bullet item
    trailing prose line
    """
'''
        first = wrap_source(source)
        second = wrap_source(first)
        assert '    - bullet item\n    trailing prose line' in first
        assert second == first


class TestNumpyCompoundTypeWrap:
    def test_compound_type_desc_wraps(self) -> None:
        long_desc = 'A very long description ' * 4 + 'that must wrap.'
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    Returns\n'
            '    -------\n'
            f'    int or None\n'
            f'        {long_desc}\n'
            '    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        lines = result.splitlines()
        over = [ln for ln in lines if len(ln) > DEFAULT_LINE_LENGTH]
        assert all(len(ln) <= DEFAULT_LINE_LENGTH for ln in lines), (
            f'Lines exceed {DEFAULT_LINE_LENGTH}: {over}'
        )
        # Type line preserved verbatim
        assert '    int or None' in result

    def test_compound_type_idempotent(self) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    Returns\n'
            '    -------\n'
            '    int or None\n'
            '        The result value with a somewhat longish description.\n'
            '    """\n'
        )
        first = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        second = wrap_source(
            first, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        assert first == second


class TestNumpyEntryWrap:
    def test_entry_idempotent(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = (
            f'def foo(x):\n    """Summary.\n\n    Parameters\n    ----------\n'
            f'    x : int\n        {long_desc}\n    """\n'
        )
        result1 = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        result2 = wrap_source(
            result1, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        assert result1 == result2

    def test_entry_name_only_no_desc(self) -> None:
        source = (
            'def foo(x):\n    """Summary.\n\n    Parameters\n    ----------\n'
            '    x : int\n    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        assert result == source

    def test_long_entry_description_wraps(self) -> None:
        long_desc = 'A very long description ' * 4 + 'that must wrap.'
        source = (
            f'def foo(x):\n    """Summary.\n\n    Parameters\n    ----------\n'
            f'    x : int\n        {long_desc}\n    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        lines = result.splitlines()
        over = [ln for ln in lines if len(ln) > DEFAULT_LINE_LENGTH]
        assert all(len(line) <= DEFAULT_LINE_LENGTH for line in lines), (
            f'Lines exceed {DEFAULT_LINE_LENGTH}: {over}'
        )
        # name : type line should still be present
        assert any('x : int' in ln for ln in lines)

    def test_short_entry_untouched(self) -> None:
        source = (
            'def foo(x):\n    """Summary.\n\n    Parameters\n    ----------\n'
            '    x : int\n        Short desc.\n    """\n'
        )
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='numpy')
        )
        assert result == source


class TestOpeningTitleWrapping:
    TITLE = Region(
        kind='title',
        text='Heading\n-------',
        source_line_count=2,
        original_lines=('Heading', '-------'),
    )

    def test_title_region_never_counts_as_skipped_summary(self) -> None:
        assert detect_summary_skip(
            [self.TITLE],
            DocwrapConfig(line_length=5, wrap_summary=False),
            base_indent=0,
            col_offset=0,
            quote_overhead=3,
        ) == (False, None)

    @pytest.mark.parametrize(
        'config',
        [
            DocwrapConfig(line_length=5, wrap_summary=True),
            DocwrapConfig(line_length=120, wrap_summary=False),
            DocwrapConfig(fill_style='balanced'),
            DocwrapConfig(fill_style='greedy'),
            DocwrapConfig(fill_style='smart'),
            DocwrapConfig(reflow='none'),
            DocwrapConfig(reflow='cautious'),
            DocwrapConfig(reflow='full'),
            DocwrapConfig(smart_slack=0),
            DocwrapConfig(smart_slack=20),
        ],
    )
    def test_title_region_round_trips_verbatim(
        self,
        config: DocwrapConfig,
    ) -> None:
        assert (
            wrap_regions(
                [self.TITLE],
                config,
                base_indent=0,
            )
            == 'Heading\n-------'
        )


class TestParagraphReflowProvenance:
    @pytest.mark.parametrize('reflow', ['none', 'cautious'])
    @pytest.mark.parametrize(
        ('authored_line', 'expected_line'),
        [
            pytest.param('  promoted prose.', '    promoted prose.', id='promoted'),
            pytest.param('    four-space prose.', '    four-space prose.', id='four-space'),
            pytest.param('      six-space prose.', '      six-space prose.', id='six-space'),
            pytest.param('    ----------', '    ----------', id='structural-underline'),
        ],
    )
    def test_applies_chosen_indent_once(
        self,
        reflow: ReflowMode,
        authored_line: str,
        expected_line: str,
    ) -> None:
        style = get_style('google')
        regions = parse_regions(
            f'Summary.\n\nNotes:\n{authored_line}',
            style=style,
            code_block_lines=set(),
        )
        prose = regions[-1]
        assert prose.kind == 'prose'
        assert prose.original_lines == (authored_line,)

        result = wrap_regions(
            regions,
            DocwrapConfig(reflow=reflow),
            base_indent=0,
            style=style,
        )
        assert result.splitlines()[-1] == expected_line


class TestReflowCautious:
    def test_bullet_not_stripped_as_markup(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Some text
                * not a continuation.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    * not a continuation.' in report.source

    @pytest.mark.parametrize('count', [100, 200])
    def test_cautious_join_scanning_is_linear(
        self, count: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = (
            'def f():\n    """Summary.\n\n'
            + '    this description continues with ordinary words through each physical line\n'
            * count
            + '    """\n'
        )
        original = wrap_module._is_semantically_joinable
        scanned_characters = 0
        calls = 0

        def track_scanning(line_a: str, line_b: str) -> bool:
            nonlocal scanned_characters, calls
            scanned_characters += len(line_a) + len(line_b)
            calls += 1
            return original(line_a, line_b)

        monkeypatch.setattr(wrap_module, '_is_semantically_joinable', track_scanning)
        report = wrap_source_report(source)

        assert report.source == source
        assert report.skips == ()
        assert calls > 0
        assert scanned_characters <= 4 * len(source)

    def test_conjunction_before_enumeration_stays_split(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Pick one of these, or
                (a) the first option
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    Pick one of these, or\n    (a) the first option\n' in report.source

    def test_fills_joinable_run_across_original_boundary(self) -> None:
        # Module-level docstring: col_offset=0, width=79
        # Each line fits individually (66, 61) but combined (128) exceeds 79.
        # Old algorithm: keeps separate. New algorithm: fills as one run.
        source = (
            '"""Summary.\n'
            '\n'
            'The configuration is resolved from pyproject.toml with fallback to\n'
            'sensible defaults that work for most projects out of the box.\n'
            '"""\n'
        )
        config = DocwrapConfig(line_length=79, reflow='cautious')
        report = wrap_source_report(source, config=config)
        # Words redistribute: "to sensible" now on line 1 instead of "to\nsensible"
        assert 'fallback to sensible' in report.source
        # No line exceeds target
        for line in report.source.splitlines():
            assert len(line) <= 79 or not line.strip()

    @pytest.mark.parametrize(
        ('body', 'expected_body'),
        [
            pytest.param(
                '    First fragment\n    ``\n    continues here.\n',
                '    First fragment `` continues here.\n',
                id='markup-only-chunk',
            ),
            pytest.param(
                '    Use e.g.\n    this option.\n',
                '    Use e.g.\n    this option.\n',
                id='abbreviation-final-period-is-terminal',
            ),
            pytest.param(
                '    See ``done.``\n    for more details.\n',
                '    See ``done.``\n    for more details.\n',
                id='terminal-punctuation-before-markup',
            ),
            pytest.param(
                '    Set the option\n    --flag\n    for this command.\n',
                '    Set the option --flag for this command.\n',
                id='long-option',
            ),
            pytest.param(
                '    Use \\t\n    for the separator.\n',
                '    Use \\t for the separator.\n',
                id='opaque-escape',
            ),
            pytest.param(
                '    Heading\n    ----------\n    more details.\n',
                '    Heading\n    ----------\n    more details.\n',
                id='underline',
            ),
        ],
    )
    def test_join_boundary_characterization(self, body: str, expected_body: str) -> None:
        prefix = 'def f():\n    """Summary.\n\n'
        suffix = '    """\n'
        source = prefix + body + suffix
        expected = prefix + expected_body + suffix
        config = DocwrapConfig(line_length=99, reflow='cautious')

        report = wrap_source_report(source, config=config)

        assert report.source == expected
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == expected

    def test_joins_across_continuation_token(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Combines the first value, and
                also the second value.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'value, and also' in report.source

    def test_joins_across_or_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Returns a list of items, or
                None to indicate failure.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'items, or None' in report.source

    def test_joins_inline_code_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Unsafe operators include
                ``|&`` for piping.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'include ``|&``' in report.source

    def test_joins_long_cli_option_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                The CLI explicitly set it to False via
                --no-incremental. This should stay readable.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=70, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'False via --no-incremental.' in report.source

    def test_joins_mid_sentence_fragment(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                This function returns the
                computed value.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'returns the computed value' in report.source

    def test_joins_parenthetical_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                The stripped strings
                (empty "" and '') are not valid.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'strings (empty' in report.source

    def test_joins_quoted_digit_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Zero counts display as
                '0%' in the output.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert "display as '0%'" in report.source

    def test_joins_uppercase_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First part without punctuation
                New sentence here.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'without punctuation New sentence' in report.source

    def test_joins_uppercase_mid_sentence_true(self) -> None:
        source = (
            '"""Summary.\n'
            '\n'
            'Returns/Yields entries whose type marker is the only structural content\n'
            'become prose regions (or are dropped) when strip_types is\n'
            'True.  Sphinx type/rtype/vartype entries are dropped entirely.\n'
            '"""\n'
        )
        config = DocwrapConfig(line_length=79, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'strip_types is True.' in report.source

    def test_no_orphan_from_original_short_line(self) -> None:
        # Module-level docstring: col_offset=0, width=79
        source = (
            '"""Summary.\n'
            '\n'
            'short\n'
            "already overwritten to 'direct' by the bug. Live resolution is the best available\n"
            'approximation.\n'
            '"""\n'
        )
        config = DocwrapConfig(line_length=79, reflow='cautious')
        report = wrap_source_report(source, config=config)
        lines = report.source.splitlines()
        body_lines = [ln.strip() for ln in lines if ln.strip() and ln.strip() != '"""']
        # "short" must have joined with the head of the split next line
        assert 'short' not in body_lines, f"Orphan 'short' found: {lines}"

    def test_no_orphan_from_split_tail(self) -> None:
        # Module-level docstring: col_offset=0, width=79
        source = (
            '"""Summary.\n'
            '\n'
            'This is an intentional lossy reclassification -- the true original provenance was\n'
            "already overwritten to 'direct' by the bug. Live resolution is the best available\n"
            'approximation.\n'
            '"""\n'
        )
        config = DocwrapConfig(line_length=79, reflow='cautious')
        report = wrap_source_report(source, config=config)
        lines = report.source.splitlines()
        body_lines = [ln.strip() for ln in lines if ln.strip() and ln.strip() != '"""']
        assert 'was' not in body_lines, f"Orphan 'was' found: {lines}"

    def test_orphan_fix_is_idempotent(self) -> None:
        # Module-level docstring: col_offset=0, width=79
        source = (
            '"""Summary.\n'
            '\n'
            'This is an intentional lossy reclassification -- the true original provenance was\n'
            "already overwritten to 'direct' by the bug. Live resolution is the best available\n"
            'approximation.\n'
            '"""\n'
        )
        config = DocwrapConfig(line_length=79, reflow='cautious')
        pass1 = wrap_source_report(source, config=config).source
        pass2 = wrap_source_report(pass1, config=config).source
        assert pass1 == pass2

    def test_preserves_colon_ending(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                The following items:
                are listed here.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    The following items:\n' in report.source

    def test_preserves_letter_enumeration_markers(self) -> None:
        for marker in ('a)', 'a.', '1)'):
            source = textwrap.dedent(f"""\
                def f():
                    \"\"\"Summary.

                    Choices without terminal punctuation
                    {marker} first choice
                    \"\"\"
            """)
            config = DocwrapConfig(line_length=99, reflow='cautious')
            report = wrap_source_report(source, config=config)
            assert f'    {marker} first choice' in report.source

    def test_preserves_paren_enumeration_markers(self) -> None:
        for marker in ('(a)', '(1)'):
            source = textwrap.dedent(f"""\
                def f():
                    \"\"\"Summary.

                    Choices without terminal punctuation
                    {marker} first choice
                    \"\"\"
            """)
            config = DocwrapConfig(line_length=99, reflow='cautious')
            report = wrap_source_report(source, config=config)
            assert f'    {marker} first choice' in report.source

    def test_preserves_semicolon_ending(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First clause here;
                second clause here.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    First clause here;\n' in report.source

    def test_preserves_sentence_boundary(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First sentence ends here.
                Second sentence starts here.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    First sentence ends here.\n' in report.source
        assert '    Second sentence starts here.' in report.source

    def test_preserves_slash_ending(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                /c -> C:/
                Already-Windows paths pass through unchanged.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert '    /c -> C:/\n' in report.source

    def test_strips_leading_markup_for_lowercase_check(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                This returns the
                ``computed`` value.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        assert 'the ``computed`` value' in report.source

    def test_strips_trailing_markup_for_punctuation_check(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                See ``done.``
                for more details.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='cautious')
        report = wrap_source_report(source, config=config)
        # The backtick-wrapped period should be detected as terminal
        assert '``done.``\n' in report.source


class TestReflowNone:
    def test_description_preserves_short_lines(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First sentence on its own line.
                Second sentence on its own line.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=99, reflow='none')
        report = wrap_source_report(source, config=config)
        assert 'First sentence on its own line.\n' in report.source
        assert '    Second sentence on its own line.' in report.source

    def test_description_splits_overlong_line(self) -> None:
        long_word = 'word ' * 30  # ~150 chars
        source = textwrap.dedent(f"""\
            def f():
                \"\"\"Summary.

                Short line.
                {long_word.strip()}
                Another short line.
                \"\"\"
        """)
        config = DocwrapConfig(line_length=79, reflow='none')
        report = wrap_source_report(source, config=config)
        assert '    Short line.' in report.source
        assert '    Another short line.' in report.source
        for line in report.source.splitlines():
            assert len(line) <= 79 or not line.strip()

    def test_full_mode_joins_lines(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First sentence.
                Second sentence.
                \"\"\"
        """)
        config_full = DocwrapConfig(line_length=99, reflow='full')
        config_none = DocwrapConfig(line_length=99, reflow='none')
        report_full = wrap_source_report(source, config=config_full)
        report_none = wrap_source_report(source, config=config_none)
        # full joins them; none preserves them
        assert 'First sentence. Second sentence.' in report_full.source
        assert 'First sentence.\n' in report_none.source


class TestSphinxEntryWrap:
    def test_field_idempotent(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = f'def foo(x):\n    """Summary.\n\n    :param x: {long_desc}\n    """\n'
        result1 = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        result2 = wrap_source(
            result1, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        assert result1 == result2

    def test_field_no_description(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    :param x:\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        assert '    :param x:\n' in result

    def test_hanging_indent_from_marker(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = f'def foo(x):\n    """Summary.\n\n    :param x: {long_desc}\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        lines = result.splitlines()
        # Find continuation lines (not the :param line, but indented lines after it)
        param_idx = next(i for i, ln in enumerate(lines) if ':param x:' in ln)
        cont_lines: list[str] = []
        for i in range(param_idx + 1, len(lines)):
            line = str(lines[i])
            if line.strip() == '"""' or not line.strip():
                break
            cont_lines.append(line)
        assert len(cont_lines) > 0
        # All continuation lines should have consistent indent
        cont_indents = {len(ln) - len(ln.lstrip()) for ln in cont_lines}
        assert len(cont_indents) == 1

    def test_long_field_description_wraps(self) -> None:
        long_desc = 'A very long description ' * 4 + 'end.'
        source = f'def foo(x):\n    """Summary.\n\n    :param x: {long_desc}\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        lines = result.splitlines()
        over = [ln for ln in lines if len(ln) > DEFAULT_LINE_LENGTH]
        assert all(len(line) <= DEFAULT_LINE_LENGTH for line in lines), (
            f'Lines exceed {DEFAULT_LINE_LENGTH}: {over}'
        )
        # First line should still have the field marker
        assert any(':param x:' in ln for ln in lines)

    def test_short_field_untouched(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    :param x: Short.\n    """\n'
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='sphinx')
        )
        assert result == source


class TestSummaryMarkerBoundaries:
    @staticmethod
    def _source(marker: str, position: str, adjacent_text: str = 'Adjacent text.') -> str:
        if position == 'beginning':
            summary_lines = [marker, adjacent_text]
        else:
            summary_lines = ['Opening text.', marker, adjacent_text]
        first, *remaining = summary_lines
        body = ''.join(f'    {line}\n' for line in remaining)
        return f'def f():\n    """{first}\n{body}    """\n'

    @pytest.mark.parametrize('marker', ['+', '-', '*', '1.'])
    @pytest.mark.parametrize('position', ['beginning', 'middle'])
    @pytest.mark.parametrize('wrap_summary', [False, True])
    def test_bare_marker_remains_a_summary_line(
        self,
        marker: str,
        position: str,
        wrap_summary: bool,
    ) -> None:
        adjacent_text = 'Adjacent text.'
        config = DocwrapConfig(wrap_summary=wrap_summary)
        first = wrap_source(
            self._source(marker, position, adjacent_text),
            config=config,
        )
        reparsed = parse_docstrings(first, find_docstrings(first), config)[0]
        summary = reparsed.regions[0]
        assert summary.source_start is not None
        summary_lines = reparsed.content.splitlines()[
            summary.source_start : summary.source_start + summary.source_line_count
        ]
        assert summary.kind == 'summary'
        assert marker in summary_lines
        assert adjacent_text in summary_lines
        assert f'{marker} {adjacent_text}' not in summary_lines

    def test_consecutive_and_trailing_markers_preserve_fragment_boundaries(self) -> None:
        summary = Region(
            kind='summary',
            text='Alpha beta + - 123456 123456 *',
            original_lines=('Alpha', 'beta', '+', '-', '123456 123456', '*'),
        )
        assert wrap_regions(
            [summary],
            DocwrapConfig(line_length=18, wrap_summary=True),
            base_indent=4,
            quote_overhead=3,
        ).splitlines() == ['Alpha beta', '+', '-', '123456 123456', '*']
        assert detect_summary_skip(
            [summary],
            DocwrapConfig(line_length=16, wrap_summary=False),
            base_indent=4,
            col_offset=4,
            quote_overhead=3,
        ) == (True, 17)

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_marker_first_gives_later_fragment_full_width(
        self,
        fill_style: FillStyle,
    ) -> None:
        adjacent_text = '123456 123456'
        config = DocwrapConfig(
            fill_style=fill_style,
            line_length=18,
            wrap_summary=True,
        )
        first = wrap_source(
            self._source('+', 'beginning', adjacent_text),
            config=config,
        )
        reparsed = parse_docstrings(first, find_docstrings(first), config)[0]
        summary = reparsed.regions[0]
        assert summary.source_start is not None
        summary_lines = reparsed.content.splitlines()[
            summary.source_start : summary.source_start + summary.source_line_count
        ]
        assert summary_lines == ['+', adjacent_text]

    def test_no_wrap_summary_uses_original_lines(self) -> None:
        summary = Region(
            kind='summary',
            text='First line. Second line.',
            original_lines=('First line.', 'Second line.'),
        )
        assert wrap_regions(
            [summary],
            DocwrapConfig(line_length=99, wrap_summary=False),
            base_indent=0,
        ).splitlines() == ['First line.', 'Second line.']

    def test_no_wrap_summary_without_provenance_uses_logical_text(self) -> None:
        summary = Region(kind='summary', text='First line. Second line.')
        assert (
            wrap_regions(
                [summary],
                DocwrapConfig(line_length=99, wrap_summary=False),
                base_indent=0,
            )
            == 'First line. Second line.'
        )

    def test_ordinary_multiline_summary_still_refills_when_enabled(self) -> None:
        summary = Region(
            kind='summary',
            text='First line. Second line.',
            original_lines=('First line.', 'Second line.'),
        )
        assert wrap_regions(
            [summary],
            DocwrapConfig(line_length=18, wrap_summary=True),
            base_indent=0,
            quote_overhead=3,
        ).splitlines() == ['First line.', 'Second line.']

    def test_overlong_marker_summary_only_returns_unchanged(self) -> None:
        source = self._source('+', 'beginning', '1234567890')
        report = wrap_source_report(
            source,
            config=DocwrapConfig(line_length=13, wrap_summary=False),
            detail='list',
        )
        assert report.source == source
        assert report.skipped_summaries == 1
        assert report.docstrings[0].summary_width == 14

    def test_single_physical_line_alone_counts_closing_quotes(self) -> None:
        summary = Region(kind='summary', text='abcde')
        config = DocwrapConfig(line_length=14, wrap_summary=False)
        assert detect_summary_skip(
            [summary],
            config,
            base_indent=4,
            col_offset=4,
            quote_overhead=3,
        ) == (True, 15)
        assert detect_summary_skip(
            [summary, Region(kind='description', text='Body.')],
            config,
            base_indent=4,
            col_offset=4,
            quote_overhead=3,
        ) == (False, None)

    @pytest.mark.parametrize('with_body', [False, True])
    def test_skip_measurement_uses_marker_physical_lines(self, with_body: bool) -> None:
        regions = [
            Region(
                kind='summary',
                text='+ 1234567890',
                original_lines=('+', '1234567890'),
            )
        ]
        if with_body:
            regions.append(Region(kind='description', text='Body.'))

        assert detect_summary_skip(
            regions,
            DocwrapConfig(line_length=14, wrap_summary=False),
            base_indent=4,
            col_offset=4,
            quote_overhead=3,
        ) == (False, None)
        assert detect_summary_skip(
            regions,
            DocwrapConfig(line_length=13, wrap_summary=False),
            base_indent=4,
            col_offset=4,
            quote_overhead=3,
        ) == (True, 14)

    def test_skip_measurement_uses_ordinary_multiline_logical_text(self) -> None:
        summary = Region(
            kind='summary',
            text='12345 67890',
            original_lines=('12345', '67890'),
        )
        regions = [summary, Region(kind='description', text='Body.')]

        assert detect_summary_skip(
            regions,
            DocwrapConfig(line_length=10, wrap_summary=False),
            base_indent=0,
            col_offset=0,
            quote_overhead=0,
        ) == (True, 11)
        assert detect_summary_skip(
            regions,
            DocwrapConfig(line_length=11, wrap_summary=False),
            base_indent=0,
            col_offset=0,
            quote_overhead=0,
        ) == (False, None)


class TestSummaryStructuralGuard:
    @pytest.mark.parametrize(
        ('summary', 'structural_token'),
        SUMMARY_STRUCTURAL_CASES,
    )
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_summary_parser_boundary_never_starts_a_continuation(
        self,
        summary: str,
        structural_token: str,
        fill_style: FillStyle,
    ) -> None:
        source = f'def f():\n    """{summary}"""\n'
        config = DocwrapConfig(fill_style=fill_style, wrap_summary=True)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        function = ast.parse(first).body[0]
        assert isinstance(function, ast.FunctionDef)
        value = ast.get_docstring(function, clean=True)
        assert value is not None
        lines = value.splitlines()
        styles = tuple(get_style(name) for name in sorted(get_styles()))
        assert structural_token in value.split()
        assert wrap_module._summary_break_line_indexes(lines, styles) == ()
        assert _docstring_words(first) == _docstring_words(source)
        assert second == first

    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_summary_structural_binding_preserves_preceding_two_space_gap(
        self,
        fill_style: FillStyle,
    ) -> None:
        lead = 'w' * 66
        summary = lead + '  :param x: detail'
        source = f'def f():\n    """{summary}"""\n'
        config = DocwrapConfig(fill_style=fill_style, wrap_summary=True)

        first = wrap_source(source, config=config)

        assert f'{lead}  :param' in first
        assert _docstring_words(first) == _docstring_words(source)
        assert wrap_source(first, config=config) == first

    @pytest.mark.parametrize(
        ('lines', 'expected'),
        SUMMARY_CLASSIFIER_CASES,
    )
    def test_summary_structural_classifier_returns_unsafe_physical_lines(
        self,
        lines: tuple[str, ...],
        expected: tuple[int, ...],
    ) -> None:
        styles = tuple(get_style(name) for name in sorted(get_styles()))
        assert wrap_module._summary_break_line_indexes(list(lines), styles) == expected


class TestUnderlineGuardDoesNotOverreach:
    def test_docstring_without_underline_is_unchanged(self) -> None:
        source = (
            'def f(x):\n    """Summary.\n\n    Args:\n        x: A short parameter.\n    """\n'
        )
        report = wrap_source_report(source, config=DocwrapConfig())
        assert report.source == source
        assert report.skips == ()

    @pytest.mark.parametrize(
        ('underline', 'line_length'),
        [
            ('---', 88),
            ('-' * 120, 40),
        ],
    )
    def test_literal_underline_in_prose_is_preserved(
        self, underline: str, line_length: int
    ) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    Before the break.\n'
            f'    {underline}\n'
            '    After the break.\n'
            '    """\n'
        )
        config = DocwrapConfig(line_length=line_length)
        first = wrap_source_report(source, config=config)
        assert f'{underline} After' not in first.source
        assert f'    {underline}\n' in first.source
        assert first.skips == ()
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source


class TestUnderlineJoinGuard:
    def _wrap_lines(self, *lines: str) -> str:
        region = Region(
            kind='description',
            text=' '.join(lines),
            original_lines=lines,
            source_line_count=len(lines),
        )
        return wrap_regions([region], DocwrapConfig(), base_indent=0)

    def test_ordinary_prose_still_joins(self) -> None:
        assert self._wrap_lines('alpha', 'beta') == 'alpha beta'

    def test_underline_not_joined_as_second_line(self) -> None:
        assert self._wrap_lines('Parameters', '----------') == 'Parameters\n----------'

    def test_underline_not_joined_with_following_word(self) -> None:
        assert self._wrap_lines('----------', 'x') == '----------\nx'


class TestUnderlinePreservationMatrix:
    FOR_CONVERSION = (
        'def f(x):\n'
        '    """Summary.\n'
        '\n'
        '    :type orphan: int\n'
        '\n'
        '    :param x: Kept parameter.\n'
        '    """\n'
    )
    HAND_WRITTEN = (
        'def f(x):\n'
        '    """Summary.\n'
        '\n'
        '    Parameters\n'
        '    ----------\n'
        '    x\n'
        '        Kept parameter.\n'
        '    :type orphan: int\n'
        '    """\n'
    )

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_conversion_produced_output_preserved(
        self, reflow: ReflowMode, fill_style: FillStyle
    ) -> None:
        config = DocwrapConfig(convert='numpy', reflow=reflow, fill_style=fill_style)
        first = wrap_source_report(self.FOR_CONVERSION, config=config)
        assert '---------- x' not in first.source
        assert '    ----------\n' in first.source
        assert first.skips == ()
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    @pytest.mark.parametrize('fill_style', ['balanced', 'greedy', 'smart'])
    def test_hand_written_mixed_style_preserved(
        self, reflow: ReflowMode, fill_style: FillStyle
    ) -> None:
        config = DocwrapConfig(reflow=reflow, fill_style=fill_style)
        first = wrap_source_report(self.HAND_WRITTEN, config=config)
        assert '---------- x' not in first.source
        assert '    ----------\n' in first.source
        assert first.skips == ()
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source


class TestWrapBlankLines:
    def test_blank_lines_preserved(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='description', text='Description.'),
        ]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=4)
        assert '\n\n' in result


class TestWrapDeeplyNested:
    def test_deeply_nested_does_not_raise(self) -> None:
        # base_indent > line_length: width would go negative without floor
        regions = [Region(kind='summary', text='Short.')]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=80)
        assert 'Short.' in result


class TestWrapDescription:
    def test_description_wrapped(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='description', text='A ' + 'very ' * 20 + 'long description.'),
        ]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=4)
        assert 'Summary.' in result
        lines = result.split('\n')
        for line in lines:
            assert len('    ' + line) <= DEFAULT_LINE_LENGTH or ' ' not in line.strip()


class TestWrapListItems:
    def test_list_item_with_hanging_indent(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='list_item', text='- A ' + 'very ' * 20 + 'long list item.'),
        ]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=4)
        lines = result.split('\n')
        # First line starts with "- ", continuation lines indented past bullet
        item_lines = [s for s in lines if 'very' in s or s.strip().startswith('-')]
        assert item_lines[0].strip().startswith('- ')


class TestWrapPreservesCodeBlocks:
    def test_code_block_verbatim(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='code_block', text='    long_code_line = "' + 'x' * 100 + '"'),
        ]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=4)
        assert 'x' * 100 in result


class TestWrapSummary:
    def test_long_summary_wrapped(self) -> None:
        text = 'A ' + 'very ' * 20 + 'long summary.'
        regions = [Region(kind='summary', text=text)]
        result = wrap_regions(
            regions, DocwrapConfig(wrap_summary=True), base_indent=4, quote_overhead=3
        )
        lines = result.split('\n')
        # First line must fit with indent + quotes: 4 + 3 + len(line) <= DEFAULT_LINE_LENGTH
        assert len('    """' + lines[0]) <= DEFAULT_LINE_LENGTH or ' ' not in lines[0].strip()
        # Subsequent lines fit with indent only
        for line in lines[1:]:
            assert len('    ' + line) <= DEFAULT_LINE_LENGTH or ' ' not in line.strip()

    def test_raw_prefix_quote_overhead(self) -> None:
        text = 'A ' + 'very ' * 20 + 'long summary.'
        regions = [Region(kind='summary', text=text)]
        result = wrap_regions(
            regions, DocwrapConfig(wrap_summary=True), base_indent=4, quote_overhead=4
        )
        lines = result.split('\n')
        # First line: 4 (indent) + 4 (r""") + len(line) <= DEFAULT_LINE_LENGTH
        assert len('    r"""' + lines[0]) <= DEFAULT_LINE_LENGTH or ' ' not in lines[0].strip()

    def test_short_summary_unchanged(self) -> None:
        regions = [Region(kind='summary', text='Short.')]
        result = wrap_regions(regions, DocwrapConfig(), base_indent=4)
        assert result == 'Short.'


class TestWrapSummaryGate:
    def test_wrap_summary_false_passes_through_already_joined_summary(self) -> None:
        regions = [Region(kind='summary', text='A short summary.')]
        config = DocwrapConfig(wrap_summary=False)
        result = wrap_regions(regions, config, base_indent=4)
        assert result == 'A short summary.'

    def test_wrap_summary_false_preserves_long_summary(self) -> None:
        long_text = 'A ' + 'very ' * 20 + 'long summary.'
        regions = [Region(kind='summary', text=long_text)]
        config = DocwrapConfig(wrap_summary=False)
        result = wrap_regions(regions, config, base_indent=4)
        # Summary should be one line, not wrapped
        assert result == long_text

    def test_wrap_summary_true_wraps_long_summary(self) -> None:
        long_text = 'A ' + 'very ' * 20 + 'long summary.'
        regions = [Region(kind='summary', text=long_text)]
        config = DocwrapConfig(wrap_summary=True)
        result = wrap_regions(regions, config, base_indent=4)
        # Summary should be wrapped across multiple lines
        assert '\n' in result


def _call_cautious_reflow_selector(
    original: tuple[str, ...],
    emissions: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    return wrap_module._select_cautious_reflow_state(original, emissions)


def _docstring_words(source: str) -> list[str]:
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    value = ast.get_docstring(function, clean=False)
    assert value is not None
    return value.split()


def _summary_boundary_source(
    summary: str,
    *,
    original_position: SummaryStart,
    placement: str,
    prefix: str,
) -> tuple[str, int, int]:
    opening = prefix + '"""'
    if original_position == 'first-line':
        literal_lines = [opening + summary, '', 'Body.', '"""']
    else:
        literal_lines = [opening, summary, '', 'Body.', '"""']

    if placement == 'module':
        return '\n'.join(literal_lines) + '\n', 0, 0
    if placement == 'nested-definition':
        literal = '\n'.join('        ' + line for line in literal_lines)
        return 'def outer():\n    def inner():\n' + literal + '\n', 8, 8
    literal_lines[0] = 'def f(): ' + literal_lines[0]
    return '\n'.join(literal_lines) + '\n', 0, 9


def _summary_boundary_cases() -> list[
    tuple[str, SummaryStart, tuple[int, int], tuple[bool, int | None]]
]:
    cases: list[tuple[str, SummaryStart, tuple[int, int], tuple[bool, int | None]]] = []
    for summary_start in ('first-line', 'second-line', 'preserve'):
        for original_position in ('first-line', 'second-line'):
            for prefix in ('', 'r'):
                for placement in ('module', 'nested-definition', 'inline-suite'):
                    _, base_indent, col_offset = _summary_boundary_source(
                        '',
                        original_position=original_position,
                        placement=placement,
                        prefix=prefix,
                    )
                    effective_position = (
                        original_position if summary_start == 'preserve' else summary_start
                    )
                    quote_overhead = 0 if effective_position == 'second-line' else len(prefix) + 3
                    effective_column = base_indent if quote_overhead == 0 else col_offset
                    capacity = 79 - effective_column - quote_overhead
                    for boundary_delta in (-1, 0, 1):
                        source, _, _ = _summary_boundary_source(
                            's' * (capacity + boundary_delta),
                            original_position=original_position,
                            placement=placement,
                            prefix=prefix,
                        )
                        expected_skipped = boundary_delta == 1
                        cases.append(
                            (
                                source,
                                summary_start,
                                (base_indent, col_offset),
                                (expected_skipped, 80 if expected_skipped else None),
                            )
                        )
    return cases


class TestSummarySkipDetection:
    def test_long_summary_skip_detected(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        config = DocwrapConfig(wrap_summary=False)
        report = wrap_source_report(source, config=config, detail='list')
        assert report.skipped_summaries == 1
        assert report.docstrings[0].summary_skipped is True
        assert report.docstrings[0].summary_width is not None

    def test_multiline_with_skip_and_other_changes(self) -> None:
        long_summary = 'A ' + 'very ' * 15 + 'long summary.'
        long_desc = 'Description ' + 'word ' * 20 + 'end.'
        source = f'def foo():\n    """{long_summary}\n\n    {long_desc}\n    """\n'
        config = DocwrapConfig(wrap_summary=False)
        report = wrap_source_report(source, config=config, detail='list')
        assert report.skipped_summaries == 1
        # Description should still be wrapped, so the docstring is modified
        assert report.wrapped == 1
        assert report.docstrings[0].changed is True
        assert report.docstrings[0].summary_skipped is True

    def test_oneliner_skip_returns_unchanged(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        config = DocwrapConfig(wrap_summary=False)
        report = wrap_source_report(source, config=config)
        # Source unchanged -- one-liner can't be wrapped without D200 violation
        assert report.source == source

    def test_second_line_inline_summary_uses_base_indent(self) -> None:
        source = 'def f(): """' + 's' * 71 + '\n\n' + 'body ' * 20 + '\n"""\n'
        config = DocwrapConfig(
            line_length=79,
            summary_start='second-line',
            wrap_summary=False,
        )
        report = wrap_source_report(source, config=config, detail='list')

        assert report.docstrings[0].changed is True
        assert report.docstrings[0].summary_skipped is False
        assert report.docstrings[0].summary_width is None
        assert report.skipped_summaries == 0

    def test_short_summary_no_skip(self) -> None:
        source = 'def foo():\n    """Short summary."""\n'
        config = DocwrapConfig(wrap_summary=False)
        report = wrap_source_report(source, config=config, detail='list')
        assert report.skipped_summaries == 0

    @pytest.mark.parametrize(
        (
            'source',
            'summary_start',
            'expected_geometry',
            'expected_report',
        ),
        _summary_boundary_cases(),
    )
    def test_summary_skip_boundary_uses_rendering_geometry(
        self,
        source: str,
        summary_start: SummaryStart,
        expected_geometry: tuple[int, int],
        expected_report: tuple[bool, int | None],
    ) -> None:
        expected_base_indent, expected_col_offset = expected_geometry
        expected_skipped, expected_width = expected_report
        config = DocwrapConfig(
            line_length=79,
            summary_start=summary_start,
            wrap_summary=False,
        )
        parsed = parse_docstrings(source, find_docstrings(source), config)

        assert len(parsed) == 1
        assert parsed[0].base_indent == expected_base_indent
        assert parsed[0].col_offset == expected_col_offset

        report = wrap_source_report(source, config=config, detail='list')

        assert report.docstrings[0].summary_skipped is expected_skipped
        assert report.docstrings[0].summary_width == expected_width
        assert report.skipped_summaries == int(expected_skipped)

    def test_wrap_summary_true_no_skip(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        config = DocwrapConfig(wrap_summary=True)
        report = wrap_source_report(source, config=config, detail='list')
        assert report.skipped_summaries == 0
        assert report.wrapped == 1


def test_adornment_fixed_point() -> None:
    source = 'def f():\n    """Summary.\n\n    alpha beta gamma =========\n    """\n'
    config = DocwrapConfig(line_length=16, fill_style='balanced', reflow='full')

    first = wrap_source(source, config=config)
    second = wrap_source(first, config=config)

    ast.parse(first)
    assert first == second
    assert all(line.strip() != '=========' for line in first.splitlines())
    assert _docstring_words(first) == _docstring_words(source)


def test_cautious_balanced_stabilization() -> None:
    source = textwrap.dedent('''\
        def f():
            """Summary.

            alpha beta zeta and
            finish right. abcdefghijklmnopqrst
            """
    ''')
    cfg = DocwrapConfig(line_length=20, reflow='cautious', fill_style='balanced')

    first = wrap_source(source, config=cfg)
    second = wrap_source(first, config=cfg)

    assert first != source
    assert _docstring_words(first) == _docstring_words(source)
    assert first == second


def test_cautious_reflow_cap_returns_original() -> None:
    original = ('original',)
    emissions = ((f'emission-{index}',) for index in range(10))

    assert _call_cautious_reflow_selector(original, emissions) == original


def test_cautious_reflow_cycle_selects_canonical_state() -> None:
    original = ('original',)
    emissions = iter([('b',), ('a',), ('b',)])

    assert _call_cautious_reflow_selector(original, emissions) == ('a',)


def test_cautious_reflow_fixed_point_returns_stable_state() -> None:
    original = ('original',)
    emissions = iter([('stable',), ('stable',)])

    assert _call_cautious_reflow_selector(original, emissions) == ('stable',)


def test_compute_quote_overhead_first_line() -> None:
    source = 'def foo():\n    """Hello."""\n'
    cfg = DocwrapConfig(line_length=79, summary_start='first-line')
    docs = find_docstrings(source)
    parsed = parse_docstrings(source, docs, cfg)[0]
    overhead = compute_quote_overhead(parsed, cfg)
    assert overhead == len(parsed.lit.prefix) + 3  # prefix + """


def test_compute_quote_overhead_second_line() -> None:
    source = 'def foo():\n    """Hello."""\n'
    cfg = DocwrapConfig(line_length=79, summary_start='second-line')
    docs = find_docstrings(source)
    parsed = parse_docstrings(source, docs, cfg)[0]
    overhead = compute_quote_overhead(parsed, cfg)
    assert overhead == 0


@pytest.mark.parametrize(
    ('base_indent', 'col_offset', 'quote_overhead', 'expected'),
    [
        (4, 20, 0, 4),
        (4, 20, 3, 20),
        (4, None, 3, 4),
    ],
)
def test_effective_column_rule(
    base_indent: int,
    col_offset: int | None,
    quote_overhead: int,
    expected: int,
) -> None:
    assert _effective_column(base_indent, col_offset, quote_overhead) == expected


def test_marker_shaped_description_continuation_is_idempotent() -> None:
    source = (
        "def generated_261(value: str = 'x') -> str:\n"
        '    """Generated summary 261 with preserve **bold** markup and '
        '(parenthesized) phrases.\n'
        '\n'
        '    1. use `value` with --line-length and grouped words; keep "quoted" '
        'digits 123 and option=value stable; case 261.\n'
        '    use `value` with --line-length and grouped words; keep "quoted" '
        'digits 123 and option=value stable; case 261.\n'
        '    trailing prose use `value` with --line-length and grouped words; '
        'keep "quoted" digits 123 and option=value stable; case 261.\n'
        '    """\n'
        '    return value\n'
    )
    config = DocwrapConfig(line_length=60, fill_style='balanced', reflow='full')
    first = wrap_source(source, config=config)
    second = wrap_source(first, config=config)

    assert _docstring_words(first) == _docstring_words(source)
    assert second == first


def test_marker_shaped_emitted_continuation_is_idempotent() -> None:
    source = (
        "def generated_78(value: str = 'x') -> str:\n"
        '    """Generated summary 78 with preserve **bold** markup and '
        '(parenthesized) phrases.\n'
        '\n'
        '    Args:\n'
        '        value (str): preserve **bold** markup and (parenthesized) '
        'phrases; carry a deliberately long description across original '
        'boundaries; case 78.\n'
        '    - preserve **bold** markup and (parenthesized) phrases; carry a '
        'deliberately long description across original boundaries; case 78.\n'
        '        preserve **bold** markup and (parenthesized) phrases; carry a '
        'deliberately long description across original boundaries; case 78.\n'
        '    """\n'
        '    return value\n'
    )
    config = DocwrapConfig(line_length=72, fill_style='balanced', reflow='full')
    first = wrap_source(source, config=config)
    second = wrap_source(first, config=config)

    assert _docstring_words(first) == _docstring_words(source)
    assert second == first


def test_nested_numbered_list_remains_separate_and_idempotent() -> None:
    source = '''\
def f():
    """Summary.

    - outer item
      1. nested numbered item
    """
'''
    config = DocwrapConfig(line_length=72, fill_style='balanced', reflow='full')
    first = wrap_source(source, config=config)
    second = wrap_source(first, config=config)

    assert _docstring_words(first) == _docstring_words(source)
    assert '\n      1. nested numbered item' in first
    assert second == first
