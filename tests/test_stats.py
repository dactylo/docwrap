import pathlib
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

import docwrap._stats as stats_mod
from docwrap import wrap_source_report
from docwrap._core import analyze_source
from docwrap._core._discover import ParsedDocstring, find_docstrings, parse_docstrings
from docwrap._core._parse import Region
from docwrap._core._process import wrap_parsed_docstring
from docwrap._stats import (
    STANDARD_LENGTHS,
    ImpactRow,
    LineMeasurement,
    ZoneBoundaries,
    _categorize_region,
    _is_width_invariant,
    _wrappable_categories,
    _zone_index,
    allocate_percentages,
    build_histogram,
    compute_impact,
    compute_percentiles,
    find_outliers,
    measure_docstring,
    measure_source_lines,
    source_boundaries,
)
from docwrap._types import DocwrapConfig, RegionKind, StyleName
from tests.helpers import make_measurement

if TYPE_CHECKING:
    from collections.abc import Callable


class TestBucketIsOverflow:
    def test_normal_buckets_not_overflow(self) -> None:
        ms = [make_measurement(30)]
        h = build_histogram(ms)
        non_overflow = [b for b in h.buckets if not b.is_overflow]
        assert all(b.upper > 0 for b in non_overflow)

    def test_overflow_bucket_has_flag(self) -> None:
        ms = [make_measurement(110)]
        h = build_histogram(ms)
        overflow = [b for b in h.buckets if b.is_overflow]
        assert len(overflow) == 1
        assert overflow[0].count == 1

    def test_overflow_upper_is_zero(self) -> None:
        ms = [make_measurement(110)]
        h = build_histogram(ms)
        overflow = next(b for b in h.buckets if b.is_overflow)
        assert overflow.upper == 0


class TestCategorizeRegion:
    def test_blank_is_none(self) -> None:
        assert _categorize_region('blank') is None

    def test_code_block_is_code_block(self) -> None:
        assert _categorize_region('code_block') == 'code_block'

    def test_description_is_prose(self) -> None:
        assert _categorize_region('description') == 'prose'

    def test_entry_is_args(self) -> None:
        assert _categorize_region('entry') == 'args'

    def test_list_item_is_prose(self) -> None:
        assert _categorize_region('list_item') == 'prose'

    def test_prose_is_prose(self) -> None:
        assert _categorize_region('prose') == 'prose'

    def test_section_header_is_none(self) -> None:
        assert _categorize_region('section_header') is None

    def test_summary_is_summary(self) -> None:
        assert _categorize_region('summary') == 'summary'

    def test_title_is_excluded(self) -> None:
        assert _categorize_region('title') is None


class TestComputeImpact:
    def test_candidate_configs_change_only_line_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = (
            'def f(value):\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        value (int): A description with enough words to wrap at narrower widths.\n'
            '    """\n'
        )
        config = DocwrapConfig(
            convert='numpy',
            fill_style='balanced',
            line_length=91,
            reflow='full',
            smart_slack=3,
            strip_types=True,
            style='google',
            summary_start='second-line',
            wrap_summary=True,
        )
        parsed = analyze_source(source, config).parsed
        simulation_spy = Mock(wraps=stats_mod.simulate_wrap_delta)
        wrapping_spy = Mock(wraps=stats_mod.wrap_parsed_docstring)
        monkeypatch.setattr(stats_mod, 'simulate_wrap_delta', simulation_spy)
        monkeypatch.setattr(stats_mod, 'wrap_parsed_docstring', wrapping_spy)

        compute_impact(parsed, config, extra_lengths=(91,))

        candidate_configs = [call.args[1] for call in wrapping_spy.call_args_list]
        assert {cfg.line_length for cfg in candidate_configs} == {
            *STANDARD_LENGTHS,
            91,
        }
        assert all(
            cfg == replace(config, line_length=cfg.line_length) for cfg in candidate_configs
        )
        assert simulation_spy.call_count == 0

        invariant_config = DocwrapConfig(
            fill_style='balanced',
            line_length=91,
            reflow='full',
            smart_slack=3,
            style='google',
        )
        invariant_parsed = analyze_source(
            'def invariant():\n    """Summary."""\n',
            invariant_config,
        ).parsed
        simulation_spy.reset_mock()
        compute_impact(invariant_parsed, invariant_config)

        assert simulation_spy.call_count == 1
        assert simulation_spy.call_args.args[1] == replace(invariant_config, line_length=72)

    def test_fill_style_is_propagated_to_reconstructed_configs(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    Handles rename/copy advancement (3 null-separated parts) vs normal (2\n'
            '    parts). Yields (header, dest_path) pairs.\n'
            '    """\n'
        )
        parsed = _parse_for_impact(source)
        balanced = compute_impact(
            parsed,
            DocwrapConfig(fill_style='balanced', reflow='full'),
            extra_lengths=(75,),
        )
        greedy = compute_impact(
            parsed,
            DocwrapConfig(fill_style='greedy', reflow='full'),
            extra_lengths=(75,),
        )
        by_length_balanced = {row.line_length: row for row in balanced}
        by_length_greedy = {row.line_length: row for row in greedy}
        assert by_length_balanced[75].docstrings_changed == 1
        assert by_length_greedy[75].docstrings_changed == 0

    def test_line_delta_positive_at_tight_target(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long description.'
        source = f'def foo():\n    """Summary.\n\n    {long}\n    """\n'
        parsed = _parse_for_impact(source)
        rows = compute_impact(parsed)
        by_length = {r.line_length: r for r in rows}
        assert by_length[72].lines_added > 0  # splitting adds lines
        assert by_length[72].lines_removed == 0

    def test_no_docstrings(self) -> None:
        parsed = _parse_for_impact('x = 1\n')
        rows = compute_impact(parsed)
        assert all(r.docstrings_changed == 0 for r in rows)

    def test_no_docstrings_line_delta_zero(self) -> None:
        parsed = _parse_for_impact('x = 1\n')
        rows = compute_impact(parsed)
        assert all(r.lines_added == 0 and r.lines_removed == 0 for r in rows)

    def test_returns_rows_for_standard_lengths(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        parsed = _parse_for_impact(source)
        rows = compute_impact(parsed)
        lengths = [r.line_length for r in rows]
        assert 72 in lengths
        assert 79 in lengths
        assert 88 in lengths
        assert 99 in lengths

    def test_shorter_length_changes_more(self) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    Body prose with enough words to wrap at seventy-two while staying within '
            'ninety-nine.\n'
            '    """\n'
        )
        config = DocwrapConfig()
        parsed = _parse_for_impact(source)
        rows = compute_impact(parsed, config)
        by_length = {r.line_length: r for r in rows}
        assert by_length[72].docstrings_changed == 1
        assert by_length[99].docstrings_changed == 0
        assert (
            wrap_source_report(
                source,
                config=replace(config, line_length=72),
            ).source
            != source
        )
        assert (
            wrap_source_report(
                source,
                config=replace(config, line_length=99),
            ).source
            == source
        )

    def test_smart_slack_is_propagated_to_reconstructed_configs(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    Handles rename/copy advancement (3 null-separated parts) vs normal (2\n'
            '    parts). Yields (header, dest_path) pairs.\n'
            '    """\n'
        )
        parsed = _parse_for_impact(source)
        repaired = compute_impact(
            parsed,
            DocwrapConfig(fill_style='smart', smart_slack=4, reflow='full'),
            extra_lengths=(75,),
        )
        greedy_equivalent = compute_impact(
            parsed,
            DocwrapConfig(fill_style='smart', smart_slack=0, reflow='full'),
            extra_lengths=(75,),
        )
        by_length_repaired = {row.line_length: row for row in repaired}
        by_length_greedy = {row.line_length: row for row in greedy_equivalent}
        assert by_length_repaired[75].docstrings_changed == 1
        assert by_length_greedy[75].docstrings_changed == 0

    def test_wrap_summary_accepts_keyword(self) -> None:
        source = 'def foo():\n    """Short summary."""\n'
        parsed = _parse_for_impact(source)
        rows = compute_impact(parsed, DocwrapConfig(wrap_summary=True))
        assert len(rows) > 0

    def test_wrap_summary_default_does_not_wrap_summary(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        parsed = _parse_for_impact(source)
        rows_default = compute_impact(parsed)
        rows_no_wrap = compute_impact(parsed, DocwrapConfig(wrap_summary=False))
        rows_wrap = compute_impact(parsed, DocwrapConfig(wrap_summary=True))
        by_len_default = {r.line_length: r for r in rows_default}
        by_len_no_wrap = {r.line_length: r for r in rows_no_wrap}
        by_len_wrap = {r.line_length: r for r in rows_wrap}
        assert by_len_default[72].lines_added == by_len_no_wrap[72].lines_added
        assert by_len_wrap[72].lines_added >= by_len_no_wrap[72].lines_added


class TestComputeImpactEquivalence:
    def _assert_equivalence(
        self,
        source: str,
        *,
        style: StyleName | None = None,
        wrap_summary: bool = False,
    ) -> None:
        parsed = _parse_for_impact(source, style=style)
        if not parsed:
            return
        reference = _reference_per_docstring(parsed, style=style, wrap_summary=wrap_summary)
        rows = compute_impact(parsed, DocwrapConfig(style=style, wrap_summary=wrap_summary))
        for li, _length in enumerate(STANDARD_LENGTHS):
            ref_changed = sum(1 for di in range(len(parsed)) if reference[(li, di)][0])
            ref_delta = sum(reference[(li, di)][1] for di in range(len(parsed)))
            row = rows[li]
            assert row.docstrings_changed == ref_changed
            assert row.lines_added - row.lines_removed == ref_delta

    def test_backslash_in_content(self) -> None:
        source = 'def foo():\n    """Summary with \\\\n newline."""\n'
        self._assert_equivalence(source)

    def test_d213_second_line_summary(self) -> None:
        source = (
            'def foo():\n    """\n    Summary on second line.\n\n    Description text.\n    """\n'
        )
        self._assert_equivalence(source)

    def test_docwrap_own_source(self) -> None:
        stats_path = pathlib.Path('src/docwrap/_stats.py')
        source = stats_path.read_text()
        self._assert_equivalence(source)

    def test_google_style(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Do something.\n'
            '\n'
            '    Args:\n'
            '        x: A very long argument description that should wrap.\n'
            '    """\n'
        )
        self._assert_equivalence(source, style='google')

    def test_mixed_invariant_and_dependent(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long description.'
        source = (
            f'def foo():\n    """Short."""\n\ndef bar():\n    """Summary.\n\n    {long}\n    """\n'
        )
        self._assert_equivalence(source)

    def test_quote_collision(self) -> None:
        source = 'def foo():\n    """Has both \' and \\" quotes."""\n'
        self._assert_equivalence(source)

    def test_raw_docstring(self) -> None:
        source = 'def foo():\n    r"""Raw summary with \\n."""\n'
        self._assert_equivalence(source)

    def test_width_dependent_prose(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long description.'
        source = f'def foo():\n    """Summary.\n\n    {long}\n    """\n'
        self._assert_equivalence(source)

    def test_width_invariant_code_block(self) -> None:
        source = 'def foo():\n    """Summary.\n\n    Example::\n\n        code = here\n    """\n'
        self._assert_equivalence(source)

    def test_width_invariant_summary_only(self) -> None:
        source = 'def foo():\n    """Short summary."""\n'
        self._assert_equivalence(source)

    def test_width_invariant_summary_with_verbatim_body(self) -> None:
        source = 'def foo():\n    """' + 's' * 71 + '\n\n        value = 1\n    """\n'
        self._assert_equivalence(source)

    def test_wrap_summary_false(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        self._assert_equivalence(source, wrap_summary=False)

    def test_wrap_summary_true(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        self._assert_equivalence(source, wrap_summary=True)


class TestFindOutliers:
    def test_default_returns_3(self) -> None:
        lines = [make_measurement(50 + i) for i in range(10)]
        result = find_outliers(lines)
        assert len(result) == 3

    def test_empty(self) -> None:
        result = find_outliers([], n=5)
        assert result == []

    def test_fewer_than_n(self) -> None:
        lines = [make_measurement(50)]
        result = find_outliers(lines, n=5)
        assert len(result) == 1

    def test_returns_top_n(self) -> None:
        lines = [make_measurement(50 + i) for i in range(10)]
        result = find_outliers(lines, n=3)
        assert len(result) == 3
        assert [measurement.length for measurement in result] == [59, 58, 57]


class TestImpactEffectiveConfig:
    _SOURCE = 'def f():\n    """Summary line that is fine.\n\n    More prose here.\n    """\n'

    _TIE_SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    :param alpha: Short description.\n'
        '\n'
        '    Returns:\n'
        '        Something useful.\n'
        '    """\n'
    )

    def _assert_rows_match_actual(
        self, source: str, rows: list[ImpactRow], config: DocwrapConfig
    ) -> None:
        for row in rows:
            cfg = replace(config, line_length=row.line_length)
            actual = wrap_source_report(source, config=cfg).source != source
            assert (row.docstrings_changed > 0) == actual

    def test_combined_transforms_match_actual(self) -> None:
        source = 'def f(a):\n    """Summary.\n\n    Args:\n        a (int): Value.\n    """\n'
        cfg = DocwrapConfig(summary_start='second-line', convert='numpy')
        parsed = analyze_source(source, cfg).parsed
        rows = compute_impact(parsed, cfg)
        assert all(row.docstrings_changed >= 1 for row in rows)
        self._assert_rows_match_actual(source, rows, cfg)

    def test_convert_changes_impact(self) -> None:
        source = 'def f(a):\n    """Summary.\n\n    Args:\n        a: Value.\n    """\n'
        cfg = DocwrapConfig(convert='numpy')
        parsed = analyze_source(source, cfg).parsed
        rows = compute_impact(parsed, cfg)
        assert all(row.docstrings_changed >= 1 for row in rows)
        self._assert_rows_match_actual(source, rows, cfg)

    def test_convert_tie_matches_actual_outcome(self) -> None:
        cfg = DocwrapConfig(convert='google')
        parsed = analyze_source(self._TIE_SOURCE, cfg).parsed
        rows = compute_impact(parsed, cfg)
        assert all(row.docstrings_changed == 0 for row in rows)
        self._assert_rows_match_actual(self._TIE_SOURCE, rows, cfg)

    def test_default_impact_unchanged(self) -> None:
        parsed = analyze_source(self._SOURCE, DocwrapConfig()).parsed
        rows = compute_impact(parsed)
        assert all(row.docstrings_changed == 0 for row in rows)

    def test_strip_types_changes_impact(self) -> None:
        source = 'def f(a):\n    """Summary.\n\n    Args:\n        a (int): Value.\n    """\n'
        cfg = DocwrapConfig(strip_types=True)
        parsed = analyze_source(source, cfg).parsed
        rows = compute_impact(parsed, cfg)
        assert all(row.docstrings_changed >= 1 for row in rows)
        self._assert_rows_match_actual(source, rows, cfg)

    def test_summary_start_changes_impact(self) -> None:
        cfg = DocwrapConfig(summary_start='second-line')
        parsed = analyze_source(self._SOURCE, cfg).parsed
        rows = compute_impact(parsed, cfg)
        assert all(row.docstrings_changed >= 1 for row in rows)
        self._assert_rows_match_actual(self._SOURCE, rows, cfg)


class TestIsWidthInvariant:
    def _region(self, kind: RegionKind) -> Region:
        return Region(kind=kind, text='x')

    def test_blank_is_invariant(self) -> None:
        assert _is_width_invariant((self._region('blank'),), wrap_summary=False)

    def test_code_block_is_invariant(self) -> None:
        assert _is_width_invariant((self._region('code_block'),), wrap_summary=False)

    def test_description_is_dependent(self) -> None:
        assert not _is_width_invariant((self._region('description'),), wrap_summary=False)

    def test_empty_regions_is_invariant(self) -> None:
        assert _is_width_invariant((), wrap_summary=False)

    def test_entry_is_dependent(self) -> None:
        assert not _is_width_invariant((self._region('entry'),), wrap_summary=False)

    def test_list_item_is_dependent(self) -> None:
        assert not _is_width_invariant((self._region('list_item'),), wrap_summary=False)

    def test_mixed_invariant_regions(self) -> None:
        regions = (self._region('blank'), self._region('summary'), self._region('code_block'))
        assert _is_width_invariant(regions, wrap_summary=False)

    def test_mixed_with_one_dependent(self) -> None:
        regions = (self._region('summary'), self._region('description'))
        assert not _is_width_invariant(regions, wrap_summary=False)

    def test_prose_is_dependent(self) -> None:
        assert not _is_width_invariant((self._region('prose'),), wrap_summary=False)

    def test_section_header_is_invariant(self) -> None:
        assert _is_width_invariant((self._region('section_header'),), wrap_summary=False)

    def test_summary_dependent_when_wrap_summary_true(self) -> None:
        assert not _is_width_invariant((self._region('summary'),), wrap_summary=True)

    def test_summary_invariant_when_wrap_summary_false(self) -> None:
        assert _is_width_invariant((self._region('summary'),), wrap_summary=False)

    def test_title_is_invariant(self) -> None:
        assert _is_width_invariant(
            (self._region('title'),),
            wrap_summary=False,
        )
        assert _is_width_invariant(
            (self._region('title'),),
            wrap_summary=True,
        )

    def test_unknown_kind_is_dependent(self) -> None:
        unknown = cast('RegionKind', 'future_kind')
        assert not _is_width_invariant((self._region(unknown),), wrap_summary=False)


class TestLargestRemainder:
    def test_all_zero_counts(self) -> None:
        result = allocate_percentages([0, 0, 0, 0, 0, 0, 0, 0])
        assert result == [0, 0, 0, 0, 0, 0, 0, 0]

    def test_empty_input(self) -> None:
        result = allocate_percentages([])
        assert result == []

    def test_even_distribution_sums_to_100(self) -> None:
        result = allocate_percentages([25, 25, 25, 25])
        assert result == [25, 25, 25, 25]
        nums = [r for r in result if r is not None]
        assert sum(nums) == 100

    def test_less_than_1_percent_excluded_from_sum(self) -> None:
        # 200 lines: 199 in bucket A, 1 in bucket B (0.5%)
        result = allocate_percentages([199, 1, 0, 0, 0, 0, 0, 0])
        assert result[1] is None
        # The non-<1% values should still sum to 100%
        nums = [r for r in result if r is not None and r != 0]
        assert sum(nums) == 100

    def test_remainder_distributed_to_largest_fractional(self) -> None:
        # 3 items across 3 buckets: each is 33.33%. The first rounds to 34%.
        result = allocate_percentages([1, 1, 1])
        nums = [r for r in result if r is not None]
        assert sum(nums) == 100
        assert sorted(nums, reverse=True) == [34, 33, 33]

    def test_single_bucket_gets_100(self) -> None:
        result = allocate_percentages([42])
        assert result == [100]

    def test_tie_breaker_favors_lower_index(self) -> None:
        result = allocate_percentages([1, 1, 1, 1, 1, 1, 1])
        nums = [r for r in result if r is not None]
        assert sum(nums) == 100
        # First two should be 15%, rest should be 14%
        assert nums[0] == 15
        assert nums[1] == 15
        assert all(n == 14 for n in nums[2:])

    def test_zero_count_shows_0_percent(self) -> None:
        result = allocate_percentages([10, 0, 0, 0, 0, 0, 0, 0])
        assert result[0] == 100
        assert all(r == 0 for r in result[1:])


class TestLineMeasurement:
    def test_wrappability_is_required(self) -> None:
        constructor = cast('Callable[..., LineMeasurement]', LineMeasurement)
        with pytest.raises(TypeError):
            constructor(
                length=40,
                category='prose',
                file_path='f.py',
                line_number=1,
                text='prose',
            )


class TestMeasureDocstring:
    def test_code_blocks_categorized(self) -> None:
        source = 'def foo():\n    """Example.\n\n    Example::\n\n        code here\n    """\n'
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        code_ms = [m for m in measurements if m.category == 'code_block']
        assert any('code here' in m.text for m in code_ms)

    def test_d212_line_numbers_exact(self) -> None:
        source = (
            'def foo():\n'  # line 1
            '    """Summary.\n'  # line 2 - summary here
            '\n'  # line 3
            '    Description line one.\n'  # line 4
            '    Description line two.\n'  # line 5
            '    """\n'  # line 6
        )
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        summary_ms = [m for m in measurements if m.category == 'summary']
        assert len(summary_ms) == 1
        assert summary_ms[0].line_number == 2
        prose_ms = [m for m in measurements if m.category == 'prose']
        assert len(prose_ms) == 2
        assert prose_ms[0].line_number == 4
        assert prose_ms[1].line_number == 5

    def test_d213_line_numbers_offset(self) -> None:
        source = (
            'def foo():\n'  # line 1
            '    """\n'  # line 2 (info.lineno)
            '    Summary.\n'  # line 3 - should be reported here
            '\n'  # line 4
            '    Details.\n'  # line 5
            '    """\n'  # line 6
        )
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        # Summary is on source line 3, not line 2
        summary_ms = [m for m in measurements if m.category == 'summary']
        assert len(summary_ms) == 1
        assert summary_ms[0].line_number == 3
        # Details is on source line 5
        prose_ms = [m for m in measurements if m.category == 'prose']
        assert len(prose_ms) == 1
        assert prose_ms[0].line_number == 5

    def test_multiline_prose_measures_each_line(self) -> None:
        source = (
            'def foo():\n'
            '    """First line of summary.\n'
            '\n'
            '    This is a longer description that\n'
            '    spans multiple lines in the source.\n'
            '    """\n'
        )
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        # Should have 3 prose measurements (summary + 2 description lines)
        assert len(measurements) == 3

    def test_no_docstrings_returns_empty(self) -> None:
        source = 'x = 1\n'
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        assert measurements == []

    def test_simple_prose(self) -> None:
        source = 'def foo():\n    """Short summary."""\n'
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        assert len(measurements) >= 1
        assert measurements[0].category == 'summary'
        assert measurements[0].length > 0

    def test_with_args(self) -> None:
        source = 'def foo(x):\n    """Do something.\n\n    Args:\n        x: The value.\n    """\n'
        parsed = _parse_source_full(source)
        measurements = measure_docstring(parsed, file_path='test.py')
        categories = {m.category for m in measurements}
        assert 'summary' in categories
        assert 'args' in categories


class TestMeasureSourceLines:
    def test_all_source_category(self) -> None:
        source = 'x = 1\ny = 2\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        assert all(m.category == 'source' for m in measurements)

    def test_empty_source(self) -> None:
        measurements = measure_source_lines('', [], file_path='test.py')
        assert measurements == []

    def test_excludes_blank_lines(self) -> None:
        source = 'x = 1\n\ny = 2\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        assert len(measurements) == 2

    def test_excludes_comments(self) -> None:
        source = 'x = 1\n# comment\ny = 2\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        assert len(measurements) == 2

    def test_excludes_docstring_lines(self) -> None:
        source = (
            'def foo():\n    """This is a docstring.\n\n    More detail.\n    """\n    return 1\n'
        )
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        texts = [m.text for m in measurements]
        assert not any('docstring' in t for t in texts)
        assert not any('More detail' in t for t in texts)
        assert any('return 1' in t for t in texts)

    def test_inline_comment_included(self) -> None:
        source = 'x = 1  # inline comment\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        assert len(measurements) == 1
        assert measurements[0].length == len('x = 1  # inline comment')

    def test_length_includes_indentation(self) -> None:
        source = 'def foo():\n    return 1\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        ret = next(m for m in measurements if 'return' in m.text)
        assert ret.length == len('    return 1')

    def test_line_numbers_correct(self) -> None:
        source = 'x = 1\n# comment\ny = 2\n'
        measurements = measure_source_lines(source, _docstring_ranges(source), file_path='test.py')
        assert measurements[0].line_number == 1
        assert measurements[1].line_number == 3

    def test_no_docstring_ranges_measures_all_code(self) -> None:
        source = 'def foo(:\n'
        measurements = measure_source_lines(source, [], file_path='test.py')
        assert len(measurements) == 1
        assert measurements[0].text == 'def foo(:'


class TestWrappableCategories:
    def test_default_excludes_summary(self) -> None:
        result = _wrappable_categories(wrap_summary=False)
        assert result == ('prose', 'args')

    def test_wrap_summary_includes_summary(self) -> None:
        result = _wrappable_categories(wrap_summary=True)
        assert result == ('prose', 'args', 'summary')


class TestZoneIndex:
    def test_above_max(self) -> None:
        # c > max -> zone 5
        assert _zone_index(100, ZoneBoundaries(32, 55, 78, 85, 99)) == 5

    def test_at_median(self) -> None:
        # c <= median -> zone 0
        assert _zone_index(32, ZoneBoundaries(32, 55, 78, 85, 99)) == 0

    def test_below_median(self) -> None:
        assert _zone_index(10, ZoneBoundaries(32, 55, 78, 85, 99)) == 0

    def test_collapsed_boundaries(self) -> None:
        # p90 == p95 == max: zones 3 and 4 have zero width
        assert _zone_index(78, ZoneBoundaries(32, 55, 78, 78, 78)) == 2  # at p90
        assert _zone_index(79, ZoneBoundaries(32, 55, 78, 78, 78)) == 5  # above max

    def test_median_to_p75(self) -> None:
        # median < c <= p75 -> zone 1
        assert _zone_index(33, ZoneBoundaries(32, 55, 78, 85, 99)) == 1
        assert _zone_index(55, ZoneBoundaries(32, 55, 78, 85, 99)) == 1

    def test_p75_to_p90(self) -> None:
        # p75 < c <= p90 -> zone 2
        assert _zone_index(56, ZoneBoundaries(32, 55, 78, 85, 99)) == 2
        assert _zone_index(78, ZoneBoundaries(32, 55, 78, 85, 99)) == 2

    def test_p90_to_p95(self) -> None:
        # p90 < c <= p95 -> zone 3
        assert _zone_index(79, ZoneBoundaries(32, 55, 78, 85, 99)) == 3
        assert _zone_index(85, ZoneBoundaries(32, 55, 78, 85, 99)) == 3

    def test_p95_to_max(self) -> None:
        # p95 < c <= max -> zone 4
        assert _zone_index(86, ZoneBoundaries(32, 55, 78, 85, 99)) == 4
        assert _zone_index(99, ZoneBoundaries(32, 55, 78, 85, 99)) == 4


def _docstring_ranges(source: str) -> list[tuple[int, int]]:
    try:
        docstrings = find_docstrings(source)
    except SyntaxError:
        return []
    return [(info.lineno, info.end_lineno) for info in docstrings]


def _parse_source_full(
    source: str,
    *,
    style: StyleName | None = None,
) -> list[ParsedDocstring]:
    try:
        docstrings = find_docstrings(source)
    except SyntaxError:
        return []
    cfg = DocwrapConfig(style=style)
    return parse_docstrings(source, docstrings, cfg, skips=[])


class TestBuildHistogram:
    @staticmethod
    def _make(length: int) -> LineMeasurement:
        return make_measurement(length, text='x')

    def test_empty(self) -> None:
        h = build_histogram([])
        assert h.total == 0
        assert all(b.count == 0 for b in h.buckets)

    def test_over_100(self) -> None:
        h = build_histogram([self._make(142)])
        assert h.total == 1
        over = next(b for b in h.buckets if b.upper == 0)
        assert over.count == 1

    def test_single_line(self) -> None:
        h = build_histogram([self._make(55)])
        assert h.total == 1
        bucket_60 = next(b for b in h.buckets if b.upper == 60)
        assert bucket_60.count == 1

    def test_zone_counts_above_max(self) -> None:
        # Line at 95, boundaries [32, 55, 78, 85, 90] -> zone 5 (above max=90)
        lines = [self._make(95)]
        h = build_histogram(lines, boundaries=ZoneBoundaries(32, 55, 78, 85, 90))
        bucket_100 = next(b for b in h.buckets if b.upper == 100)
        assert bucket_100.zone_counts == (0, 0, 0, 0, 0, 1)

    def test_zone_counts_correct_zones(self) -> None:
        # All lines at length 30, boundaries [50, 60, 70, 80, 90]
        # 30 <= 50 -> zone 0
        lines = [self._make(30)] * 5
        h = build_histogram(lines, boundaries=ZoneBoundaries(50, 60, 70, 80, 90))
        bucket_40 = next(b for b in h.buckets if b.upper == 40)
        assert bucket_40.zone_counts == (5, 0, 0, 0, 0, 0)

    def test_zone_counts_none_without_boundaries(self) -> None:
        h = build_histogram([self._make(55)])
        assert all(b.zone_counts is None for b in h.buckets)

    def test_zone_counts_present_with_boundaries(self) -> None:
        lines = [self._make(n) for n in [30, 55, 65, 75, 85, 95]]
        h = build_histogram(lines, boundaries=ZoneBoundaries(50, 60, 70, 80, 90))
        for b in h.buckets:
            if b.count > 0:
                assert b.zone_counts is not None
                assert sum(b.zone_counts) == b.count
            else:
                assert b.zone_counts is not None  # still a tuple, but all zeros
                assert sum(b.zone_counts) == 0

    def test_zone_counts_tuple_length(self) -> None:
        lines = [self._make(55)]
        h = build_histogram(lines, boundaries=ZoneBoundaries(32, 55, 78, 85, 99))
        for b in h.buckets:
            assert b.zone_counts is not None
            assert len(b.zone_counts) == 6


class TestCodeBlockMeasurements:
    @pytest.mark.parametrize(
        ('source', 'expected_region_kinds', 'expected_code_lines'),
        [
            (
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    ```python\n'
                    '        code()\n'
                    '        \n'
                    '        next_code()\n'
                    '    ```\n'
                    '    """\n'
                ),
                (
                    'summary',
                    'blank',
                    'code_block',
                    'code_block',
                    'code_block',
                    'code_block',
                    'code_block',
                ),
                ('    ```python', '        code()', '        next_code()', '    ```'),
            ),
            (
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    >>> first()\n'
                    '        \n'
                    '    >>> second()\n'
                    '    """\n'
                ),
                ('summary', 'blank', 'code_block', 'blank', 'code_block'),
                ('    >>> first()', '    >>> second()'),
            ),
            (
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    Example::\n\n'
                    '        first()\n'
                    '        \n'
                    '        second()\n'
                    '    """\n'
                ),
                (
                    'summary',
                    'blank',
                    'description',
                    'blank',
                    'code_block',
                    'code_block',
                    'code_block',
                ),
                ('        first()', '        second()'),
            ),
        ],
        ids=['fenced', 'doctest', 'rest-literal'],
    )
    def test_protected_blank_is_not_measured(
        self,
        source: str,
        expected_region_kinds: tuple[RegionKind, ...],
        expected_code_lines: tuple[str, ...],
    ) -> None:
        parsed = _parse_source_full(source)
        region_kinds = tuple(region.kind for region in parsed[0].regions)

        measurements = measure_docstring(parsed, file_path='test.py')
        code_measurements = [item for item in measurements if item.category == 'code_block']

        assert region_kinds == expected_region_kinds
        assert tuple(item.text for item in code_measurements) == expected_code_lines
        assert tuple(item.length for item in code_measurements) == tuple(
            len(line) for line in expected_code_lines
        )


class TestComputePercentiles:
    @staticmethod
    def _make(length: int) -> LineMeasurement:
        return make_measurement(length, text='x')

    def test_empty_returns_none(self) -> None:
        assert compute_percentiles([]) is None

    def test_even_count_median_picks_lower(self) -> None:
        lines = [self._make(v) for v in [10, 20, 30, 40]]
        p = compute_percentiles(lines)
        assert p is not None
        # nearest-rank: index = max(0, (50*4+99)//100 - 1) = 1 -> value 20
        assert p.levels[50] == 20  # lower-middle, not 25 (interpolated)

    def test_multiple_elements(self) -> None:
        lines = [self._make(i) for i in range(1, 21)]
        p = compute_percentiles(lines)
        assert p is not None
        assert p.max == 20
        # nearest-rank: index = max(0, (50*20+99)//100 - 1) = 9 -> value 10
        assert p.levels[50] == 10
        # nearest-rank: index = max(0, (75*20+99)//100 - 1) = 14 -> value 15
        assert p.levels[75] == 15
        # index = max(0, (90*20+99)//100 - 1) = 17 -> value 18
        assert p.levels[90] == 18
        # index = max(0, (95*20+99)//100 - 1) = 18 -> value 19
        assert p.levels[95] == 19

    def test_p75_computed(self) -> None:
        lines = [self._make(i * 10) for i in range(1, 11)]  # 10, 20, ..., 100
        p = compute_percentiles(lines)
        assert p is not None
        # nearest-rank: index = max(0, (75*10+99)//100 - 1) = 7 -> value 80
        assert p.levels[75] == 80

    def test_p99_computed(self) -> None:
        ms = [self._make(i) for i in range(1, 101)]
        p = compute_percentiles(ms)
        assert p is not None
        assert p.levels[99] == 99
        assert p.max == 100

    def test_single_element(self) -> None:
        p = compute_percentiles([self._make(50)])
        assert p is not None
        assert p.levels[50] == 50
        assert p.levels[75] == 50
        assert p.levels[90] == 50
        assert p.levels[95] == 50
        assert p.max == 50


class TestOpeningTitleMeasurements:
    @staticmethod
    def _measure(source: str) -> list[LineMeasurement]:
        docstrings = find_docstrings(source)
        parsed = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        return measure_docstring(parsed)

    @pytest.mark.parametrize('fence', ['```', '~~~'])
    def test_ambiguous_complete_fence_is_excluded_as_title(
        self,
        fence: str,
    ) -> None:
        source = f'def f():\n    """{fence}\n    foo\n    {fence}\n    """\n'
        assert self._measure(source) == []

    def test_title_is_excluded_and_body_is_prose(self) -> None:
        source = 'def f():\n    """Heading\n    -------\n    body\n    """\n'
        measurements = self._measure(source)
        assert [(item.category, item.text.strip()) for item in measurements] == [('prose', 'body')]


class TestPhysicalSourceLengths:
    def test_d213_summary_uses_second_physical_line(self) -> None:
        source = 'def f():\n    r"""\n    Summary second.\n    """\n'
        expected = '    Summary second.'

        measurements = measure_docstring(_parse_source_full(source), file_path='test.py')

        assert [(item.text, item.length, item.line_number) for item in measurements] == [
            (expected, len(expected), 3)
        ]

    def test_generated_region_has_no_authored_measurement(self) -> None:
        parsed = _parse_source_full('def f():\n    """Summary."""\n')[0]
        generated = Region(
            kind='prose',
            text='Generated conversion text.',
            source_line_count=1,
            original_lines=('Generated conversion text.',),
        )
        converted = replace(parsed, regions=(generated,))

        assert measure_docstring([converted], file_path='test.py') == []

    def test_multiline_regions_use_indentation_and_closing_source(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '        indented code\n'
            '    Final details."""; marker = 1\n'
        )
        expected = [
            ('summary', '    """Summary.'),
            ('code_block', '        indented code'),
            ('prose', '    Final details."""; marker = 1'),
        ]

        measurements = measure_docstring(_parse_source_full(source), file_path='test.py')

        assert [(item.category, item.text) for item in measurements] == expected
        assert [item.length for item in measurements] == [len(text) for _, text in expected]

    @pytest.mark.parametrize(
        ('source', 'category', 'expected_text'),
        [
            (
                'def f():\n    r"""Raw summary."""; marker = 1\n',
                'summary',
                '    r"""Raw summary."""; marker = 1',
            ),
            (
                'def f():\n    """One-line summary."""\n',
                'summary',
                '    """One-line summary."""',
            ),
            (
                'def f():\n    """Use \\x41 spelling."""\n',
                'summary',
                '    """Use \\x41 spelling."""',
            ),
        ],
        ids=['d212-prefix-trailing', 'one-line-delimiters', 'escape-spelling'],
    )
    def test_single_physical_line_is_measured_exactly(
        self,
        source: str,
        category: str,
        expected_text: str,
    ) -> None:
        measurements = measure_docstring(_parse_source_full(source), file_path='test.py')

        assert [(item.category, item.text, item.length) for item in measurements] == [
            (category, expected_text, len(expected_text))
        ]

    def test_statement_f_string_is_not_discovered(self) -> None:
        source = 'def f():\n    f"""Not a docstring {1}."""\n'

        assert _parse_source_full(source) == []
        assert measure_docstring(_parse_source_full(source), file_path='test.py') == []


class TestSourceBoundaries:
    @staticmethod
    def _make(length: int) -> LineMeasurement:
        return make_measurement(length, category='source')

    def test_empty_returns_none(self) -> None:
        pctls, bounds = source_boundaries([])
        assert pctls is None
        assert bounds is None

    def test_none_returns_none(self) -> None:
        pctls, bounds = source_boundaries(None)
        assert pctls is None
        assert bounds is None

    def test_returns_percentiles_and_boundaries(self) -> None:
        ms = [self._make(i) for i in range(1, 101)]
        pctls, bounds = source_boundaries(ms)
        assert pctls is not None
        assert bounds is not None
        assert len(bounds) == 5
        assert bounds == ZoneBoundaries(*(pctls.levels[p] for p in (50, 75, 90, 95, 99)))


def _parse_for_impact(
    source: str,
    style: StyleName | None = None,
) -> list[ParsedDocstring]:
    docstrings = find_docstrings(source)
    cfg = DocwrapConfig(style=style)
    return parse_docstrings(source, docstrings, cfg, skips=[])


def _reference_per_docstring(
    parsed: list[ParsedDocstring],
    *,
    style: StyleName | None = None,
    wrap_summary: bool = False,
) -> dict[tuple[int, int], tuple[bool, int]]:
    detail: dict[tuple[int, int], tuple[bool, int]] = {}
    for li, length in enumerate(STANDARD_LENGTHS):
        cfg = DocwrapConfig(line_length=length, style=style, wrap_summary=wrap_summary)
        for di, pd in enumerate(parsed):
            result = wrap_parsed_docstring(pd, cfg, style_override=None)
            changed = result.new_literal is not None
            ld = 0
            if result.new_literal is not None:
                ld = result.new_literal.count('\n') - pd.lit.raw_literal.count('\n')
            detail[(li, di)] = (changed, ld)
    return detail
