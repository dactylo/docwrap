import io
import re
from contextlib import redirect_stdout
from dataclasses import replace

import pytest

from docwrap._color import BOLD, GREEN, HISTOGRAM_SPECTRUM, RED, RESET, YELLOW
from docwrap._display import print_stats
from docwrap._display._stats import (
    _build_histogram_rows,
    _compute_bucket_labels,
    _format_histogram,
    _ImpactTableOptions,
    _print_configured_length,
    _print_distributions,
    _print_docstring_tables,
    _print_impact_table,
    _print_outliers,
    _print_recommendations,
    _print_skipped_footer,
    _print_source_table,
    _render_bar,
    _StatsRenderCtx,
    _trim_histogram_rows,
    _zone_annotation,
)
from docwrap._stats import (
    PERCENTILE_LEVELS,
    ImpactRow,
    LineMeasurement,
    Percentiles,
    StatsData,
    ZoneBoundaries,
    build_histogram,
    source_boundaries,
)
from docwrap._types import LineLengthSource, MeasurementCategory, StatsSection
from tests.helpers import make_measurement


class TestColor:
    def test_configured_length_no_yellow_without_color(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_configured_length(79, LineLengthSource.DEFAULT, color=False)
        output = buf.getvalue()
        assert YELLOW not in output
        assert '79' in output

    def test_configured_length_yellow_when_color(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_configured_length(79, LineLengthSource.DEFAULT, color=True)
        output = buf.getvalue()
        assert YELLOW in output
        assert '79' in output

    def test_impact_negative_delta_no_yellow(self) -> None:
        agg: dict[int, ImpactRow] = {
            80: ImpactRow(
                line_length=80,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=2,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True))
        output = buf.getvalue()
        assert YELLOW not in output

    def test_impact_no_ansi_when_no_color(self) -> None:
        agg: dict[int, ImpactRow] = {
            72: ImpactRow(
                line_length=72,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=False))
        output = buf.getvalue()
        assert RED not in output
        assert BOLD not in output

    def test_impact_positive_net_no_color(self) -> None:
        agg: dict[int, ImpactRow] = {
            80: ImpactRow(
                line_length=80,
                docstrings_changed=0,
                files_changed=0,
                lines_added=5,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True))
        output = buf.getvalue()
        assert '+5' in output
        assert YELLOW not in output

    def test_impact_target_no_color_when_under_limit(self) -> None:
        agg: dict[int, ImpactRow] = {
            72: ImpactRow(
                line_length=72,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True, line_length=79))
        output = buf.getvalue()
        assert YELLOW not in output
        assert RED not in output

    def test_impact_target_red_when_over_limit(self) -> None:
        agg: dict[int, ImpactRow] = {
            80: ImpactRow(
                line_length=80,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=2,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True, line_length=79))
        output = buf.getvalue()
        assert RED in output

    def test_impact_target_yellow_when_at_limit(self) -> None:
        agg: dict[int, ImpactRow] = {
            79: ImpactRow(
                line_length=79,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True, line_length=79))
        output = buf.getvalue()
        assert YELLOW in output

    def test_impact_zero_delta_no_yellow(self) -> None:
        agg: dict[int, ImpactRow] = {
            80: ImpactRow(
                line_length=80,
                docstrings_changed=0,
                files_changed=0,
                lines_added=0,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True))
        output = buf.getvalue()
        assert YELLOW not in output

    def test_impact_zero_net_no_yellow(self) -> None:
        agg: dict[int, ImpactRow] = {
            80: ImpactRow(
                line_length=80,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=0,
            )
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, None, _impact_options(color=True))
        output = buf.getvalue()
        assert YELLOW not in output

    def test_recommendations_green_when_color(self) -> None:
        impact: dict[int, ImpactRow] = {
            79: ImpactRow(
                line_length=79,
                docstrings_changed=1,
                files_changed=1,
                lines_added=2,
                lines_removed=0,
            ),
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_recommendations(
                impact,
                None,
                line_length_source=LineLengthSource.DEFAULT,
                configured_length=79,
                color=True,
            )
        output = buf.getvalue()
        assert GREEN in output
        assert '79' in output

    def test_recommendations_no_green_without_color(self) -> None:
        impact: dict[int, ImpactRow] = {
            79: ImpactRow(
                line_length=79,
                docstrings_changed=1,
                files_changed=1,
                lines_added=2,
                lines_removed=0,
            ),
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_recommendations(
                impact,
                None,
                line_length_source=LineLengthSource.DEFAULT,
                configured_length=79,
                color=False,
            )
        output = buf.getvalue()
        assert GREEN not in output
        assert '79' in output

    def test_summary_table_bold_headers_when_color(self) -> None:
        ms = [_make(40), _make(50, category='args')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=True,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert BOLD in output
        assert RESET in output

    def test_summary_table_no_ansi_when_no_color(self) -> None:
        ms = [_make(40), _make(50, category='args')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert BOLD not in output
        assert RESET not in output


class TestComputeBucketLabels:
    def test_coincident_boundaries(self) -> None:
        boundaries = ZoneBoundaries(50, 80, 80, 80, 99)
        ms = [_make(78)]
        h = build_histogram(ms, boundaries=boundaries)
        buckets = tuple(
            row.bucket for row in _trim_histogram_rows(_build_histogram_rows(h.buckets))
        )
        pctls = Percentiles(
            levels={50: 50, 75: 80, 90: 80, 95: 80, 99: 99},
            max=100,
        )
        _labels, boundary_sets = _compute_bucket_labels(buckets, pctls)
        b71 = next(i for i, b in enumerate(buckets) if b.upper == 80)
        assert boundary_sets[b71] == frozenset({1, 2, 3})

    def test_no_boundaries_without_percentiles(self) -> None:
        ms = [_make(50)]
        h = build_histogram(ms)
        buckets = tuple(
            row.bucket for row in _trim_histogram_rows(_build_histogram_rows(h.buckets))
        )
        _labels, boundary_sets = _compute_bucket_labels(buckets, None)
        assert all(len(bs) == 0 for bs in boundary_sets)

    def test_overflow_bucket_no_boundaries(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(30), _make(110)]
        h = build_histogram(ms, boundaries=boundaries)
        buckets = tuple(
            row.bucket for row in _trim_histogram_rows(_build_histogram_rows(h.buckets))
        )
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=110,
        )
        _labels, boundary_sets = _compute_bucket_labels(buckets, pctls)
        overflow_idx = next(i for i, b in enumerate(buckets) if b.is_overflow)
        assert len(boundary_sets[overflow_idx]) == 0

    def test_returns_boundary_sets(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(i) for i in range(71, 81)]
        h = build_histogram(ms, boundaries=boundaries)
        buckets = tuple(
            row.bucket for row in _trim_histogram_rows(_build_histogram_rows(h.buckets))
        )
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=100,
        )
        _labels, boundary_sets = _compute_bucket_labels(buckets, pctls)
        b71 = next(i for i, b in enumerate(buckets) if b.upper == 80)
        assert 3 in boundary_sets[b71]


class TestDistributions:
    def test_header_no_colon(self) -> None:
        ms = [_make(50)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'Distribution' in output
        assert 'Distribution:' not in output

    def test_label_has_no_line_count(self) -> None:
        ms = [_make(50)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'prose:' in output
        assert 'lines)' not in output

    def test_only_wrappable_categories_shown(self) -> None:
        ms = [
            _make(50),
            _make(60, category='args'),
            _make(70, category='code_block'),
            _make(40, category='summary'),
        ]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'prose' in output
        assert 'args' in output
        assert 'code_block' not in output
        assert 'TOTAL' in output

    def test_subtitle_plain_without_source(self) -> None:
        ms = [_make(50)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'prose:' in output
        assert '(source:' not in output

    def test_subtitle_with_source_percentiles(self) -> None:
        ms = [_make(50)]
        source_ms = [_make(n, category='source') for n in range(30, 100)]
        sp, sb = source_boundaries(source_ms)
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=source_ms,
            color=False,
            line_length=None,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert '(source:' in output
        assert 'median=' in output
        assert 'p75=' in output

    def test_summary_excluded_by_default(self) -> None:
        ms = [_make(50), _make(40, category='summary')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'summary' not in output

    def test_total_label_used(self) -> None:
        ms = [_make(50), _make(60, category='args')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'TOTAL' in output
        assert 'all wrappable' not in output

    def test_wrap_summary_adds_summary_histogram(self) -> None:
        ms = [_make(50), _make(40, category='summary', wrappable=True)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_distributions(ctx)
        output = buf.getvalue()
        assert 'summary' in output


class TestHistogramLabels:
    def test_collapsed_boundaries_combined_label(self) -> None:
        boundaries = ZoneBoundaries(50, 80, 80, 80, 99)
        ms = [_make(78)]
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 50, 75: 80, 90: 80, 95: 80, 99: 99},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        bucket_line = next(ln for ln in lines if '71-80' in ln)
        assert 'p75, p90, p95' in bucket_line

    def test_label_for_p99_inside_bucket(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(i) for i in range(91, 101)]  # 10 lines in 91-100
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        bucket_line = next(ln for ln in lines if '91-100' in ln)
        assert 'p99' in bucket_line

    def test_label_when_boundary_at_bucket_upper(self) -> None:
        boundaries = ZoneBoundaries(50, 75, 90, 95, 99)
        ms = [_make(45), _make(48)]
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 50, 75: 75, 90: 90, 95: 95, 99: 99},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        bucket_line = next(ln for ln in lines if '41-50' in ln)
        assert 'median' in bucket_line

    def test_label_when_boundary_inside_bucket(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(i) for i in range(71, 81)]  # 10 lines in 71-80
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        bucket_line = next(ln for ln in lines if '71-80' in ln)
        assert 'p95' in bucket_line

    def test_leading_boundary_relocated_to_previous_row(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(85)] * 5 + [_make(98)] * 3
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        row_81 = next(ln for ln in lines if '81-90' in ln)
        row_91 = next(ln for ln in lines if '91-100' in ln)
        assert '|' in row_81, 'Expected trailing | on 81-90 row'
        assert 'p99' in row_81, 'Expected p99 label on 81-90 row'
        assert 'p99' not in row_91, 'p99 label should not be on 91-100 row'
        stripped_91 = re.sub(r'\033\[[0-9;]*m', '', row_91)
        bar_91 = stripped_91.split()[1] if len(stripped_91.split()) > 1 else ''
        assert not bar_91.startswith('|'), 'No leading | on 91-100 row'

    def test_multiple_labels_relocated_to_same_row(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 92, 95)
        ms = [_make(85)] * 5 + [_make(98)] * 3
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 92, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        row_81 = next(ln for ln in lines if '81-90' in ln)
        assert 'p95' in row_81, 'Expected relocated p95 label on 81-90'
        assert 'p99' in row_81, 'Expected relocated p99 label on 81-90'
        p95_pos = row_81.index('p95')
        p99_pos = row_81.index('p99')
        assert p95_pos < p99_pos, 'p95 should appear before p99'

    def test_no_label_on_empty_bucket(self) -> None:
        boundaries = ZoneBoundaries(50, 75, 90, 95, 99)
        ms = [_make(30)]
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 50, 75: 75, 90: 90, 95: 95, 99: 99},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        data_rows = output.split('\n')[2:]
        for row in data_rows:
            assert 'p75' not in row
            assert 'p90' not in row

    def test_no_label_on_overflow_bucket(self) -> None:
        boundaries = ZoneBoundaries(30, 50, 70, 80, 90)
        ms = [_make(25), _make(110)]
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 30, 75: 50, 90: 70, 95: 80, 99: 90},
            max=110,
        )
        output = _format_histogram(
            h,
            label='test',
            color=False,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        overflow_line = next(ln for ln in lines if ln.lstrip().startswith('>'))
        assert 'p99' not in overflow_line

    def test_no_label_without_source_percentiles(self) -> None:
        ms = [_make(45)]
        h = build_histogram(ms)
        output = _format_histogram(h, label='test', color=False)
        assert 'median' not in output
        assert 'p75' not in output

    def test_no_pipe_without_source_percentiles(self) -> None:
        ms = [_make(i) for i in range(71, 80)] * 3
        h = build_histogram(ms, boundaries=ZoneBoundaries(37, 52, 68, 79, 95))
        output = _format_histogram(h, label='test', color=True)
        lines = output.split('\n')
        bucket_line = next(ln for ln in lines if '71-80' in ln)
        assert '|' not in bucket_line

    def test_overlapping_percentiles_natural_separator(self) -> None:
        boundaries = ZoneBoundaries(50, 50, 55, 70, 90)
        ms = [_make(52)] * 5 + [_make(58)] * 5
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 50, 75: 50, 90: 55, 95: 70, 99: 90},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        row_51 = next(ln for ln in lines if '51-60' in ln)
        assert '|' in row_51, 'Expected natural | for two-sided boundary'
        pipe_count = row_51.count('|')
        assert pipe_count == 1, f'Expected 1 natural |, got {pipe_count}'

    def test_scan_fail_drops_pipe_keeps_label(self) -> None:
        boundaries = ZoneBoundaries(5, 10, 15, 20, 95)
        ms = [_make(98)] * 5
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 5, 75: 10, 90: 15, 95: 20, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        row_91 = next(ln for ln in lines if '91-100' in ln)
        assert 'p99' in row_91
        stripped = row_91.lstrip()
        bar_start = stripped.split()[1] if len(stripped.split()) > 1 else ''
        assert not bar_start.startswith('|')

    def test_spectrum_matches_percentile_levels(self) -> None:
        assert len(HISTOGRAM_SPECTRUM) == len(PERCENTILE_LEVELS) + 1

    def test_trailing_boundary_on_current_row(self) -> None:
        boundaries = ZoneBoundaries(37, 52, 68, 79, 95)
        ms = [_make(i) for i in range(71, 80)] * 3
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 37, 75: 52, 90: 68, 95: 79, 99: 95},
            max=100,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )
        lines = output.split('\n')
        row_71 = next(ln for ln in lines if '71-80' in ln)
        assert '|' in row_71, 'Expected trailing | on 71-80 row'
        assert 'p95' in row_71, 'Expected p95 label on 71-80 row'

    def test_trailing_pipe_at_end(self) -> None:
        ms = [_make(i) for i in range(71, 80)] * 3  # 27 lines, zone 3 only
        h = build_histogram(ms, boundaries=ZoneBoundaries(37, 52, 68, 79, 95))
        bucket = next(b for b in h.buckets if b.upper == 80)

        bar_without = _render_bar(bucket, 27, color=True)
        bar_with = _render_bar(bucket, 27, color=True, trailing_pipe=True)
        vis_without = bar_without.count('#') + bar_without.count('|')
        vis_with = bar_with.count('#') + bar_with.count('|')
        assert vis_without == vis_with, f'{vis_without} != {vis_with}'
        assert bar_with.endswith('|')
        assert bar_with.count('#') == bar_without.count('#') - 1

    def test_trailing_pipe_false_no_pipe(self) -> None:
        ms = [_make(75)] * 10
        h = build_histogram(ms, boundaries=ZoneBoundaries(37, 52, 68, 79, 95))
        bucket = next(b for b in h.buckets if b.upper == 80)

        bar = _render_bar(bucket, 10, color=True)
        assert not bar.endswith('|')


class TestImpactTableMarkers:
    def _run_impact(
        self,
        source_pctls: Percentiles | None,
        targets: tuple[int, ...] = (72, 79, 80, 88, 99, 100),
    ) -> str:
        agg: dict[int, ImpactRow] = {
            t: ImpactRow(
                line_length=t,
                docstrings_changed=1,
                files_changed=1,
                lines_added=0,
                lines_removed=0,
            )
            for t in targets
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_impact_table(agg, source_pctls, _impact_options(color=False))
        return buf.getvalue()

    def test_always_separate_marker_row(self) -> None:
        pctls = Percentiles(
            levels={50: 50, 75: 65, 90: 80, 95: 88, 99: 96},
            max=100,
        )
        output = self._run_impact(pctls)
        lines = output.split('\n')
        data_80 = [ln for ln in lines if ln.strip().startswith('80') and '- - -' not in ln]
        marker_80 = [ln for ln in lines if '80' in ln and '- - -' in ln]
        assert len(data_80) >= 1
        assert len(marker_80) >= 1  # separate marker row
        assert '(source' not in data_80[0]

    def test_collapse_adjacent_different_values(self) -> None:
        pctls = Percentiles(
            levels={50: 50, 75: 65, 90: 73, 95: 77, 99: 95},
            max=100,
        )
        output = self._run_impact(pctls)
        assert 'p90=73' in output
        assert 'p95=77' in output

    def test_collapse_adjacent_same_value(self) -> None:
        pctls = Percentiles(
            levels={50: 50, 75: 65, 90: 85, 95: 85, 99: 95},
            max=100,
        )
        output = self._run_impact(pctls)
        assert '(source, p90, p95)' in output

    def test_header_text(self) -> None:
        output = self._run_impact(None)
        assert 'Wrapping impact' in output

    def test_marker_label_outside_table(self) -> None:
        pctls = Percentiles(
            levels={50: 50, 75: 65, 90: 78, 95: 85, 99: 95},
            max=99,
        )
        output = self._run_impact(pctls)
        lines = output.strip().split('\n')
        # Find the column header line to measure table width
        hdr_line = next(ln for ln in lines if 'Target' in ln and 'Files%' in ln)
        table_width = len(hdr_line)
        # Marker dashes should extend to (at least) table width
        marker_line = next(ln for ln in lines if '(source, p90)' in ln)
        annotation_start = marker_line.index('(source')
        assert annotation_start >= table_width, (
            f'Annotation starts at {annotation_start} but table width is {table_width}'
        )

    def test_marker_range_filter(self) -> None:
        pctls = Percentiles(
            levels={50: 30, 75: 45, 90: 55, 95: 60, 99: 130},
            max=150,
        )
        output = self._run_impact(pctls)
        assert '(source' not in output

    def test_markers_shown_between_rows(self) -> None:
        pctls = Percentiles(levels={50: 50, 75: 65, 90: 78, 95: 85, 99: 95}, max=99)
        output = self._run_impact(pctls)
        lines = output.strip().split('\n')
        marker_lines = [line for line in lines if '(source, p90)' in line]
        assert len(marker_lines) == 1
        assert '- - -' in marker_lines[0]

    def test_no_source_measurements_no_markers(self) -> None:
        output = self._run_impact(None)
        assert '(source' not in output

    def test_p99_replaces_max_in_markers(self) -> None:
        pctls = Percentiles(
            levels={50: 50, 75: 65, 90: 85, 95: 92, 99: 98},
            max=105,
        )
        output = self._run_impact(pctls)
        assert 'p99' in output
        assert '(source, max' not in output

    def test_spanning_headers_dynamic_dashes(self) -> None:
        output = self._run_impact(None)
        # Dynamic format must be present (dashes expand to fill span width)
        lines = output.strip().split('\n')
        spanning_line = next(line for line in lines if 'Line change' in line and 'Scope' in line)
        assert '- Line change -' in spanning_line
        assert '- Scope -' in spanning_line
        # Spanning headers use dynamic dashes, not the old fixed-width labels
        # Verify the dashes extend beyond the minimal '- Label -' format
        lines_part = spanning_line.split('Scope')[0]
        assert lines_part.count('-') > 2


class TestOutliers:
    def test_control_characters_are_escaped_before_outlier_color(self) -> None:
        ms = [_make(90, file_path='bad\x1b\t\x9b.py', line_number=7)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=True,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()

        with redirect_stdout(buf):
            _print_outliers(ctx)

        assert f'{BOLD}bad\\x1b\\x09\\x9b.py:7{RESET}' in buf.getvalue()

    def test_header_no_colon_no_docstring(self) -> None:
        ms = [_make(90)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert 'Longest wrappable lines' in output
        assert 'Longest wrappable lines (docstring):' not in output

    def test_only_wrappable_categories_shown(self) -> None:
        ms = [_make(90), _make(95, category='code_block'), _make(85, category='summary')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert 'prose' in output
        assert 'code_block' not in output
        assert 'summary' not in output

    def test_outlier_chars_no_color_when_under_limit(self) -> None:
        ms = [_make(78)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=True,
            line_length=79,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert RED not in output
        assert YELLOW not in output

    def test_outlier_chars_red_when_over_limit(self) -> None:
        ms = [_make(90)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=True,
            line_length=79,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert RED in output

    def test_outlier_chars_yellow_when_at_limit(self) -> None:
        ms = [_make(79)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=True,
            line_length=79,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert YELLOW in output

    def test_outlier_no_annotation_below_p75(self) -> None:
        ms = [_make(40)]
        src = [_make(i, category='source') for i in range(30, 100)]
        output = _run_stats(ms, src)
        assert '>p75' not in output
        assert '>median' not in output

    def test_outlier_no_color_when_color_false(self) -> None:
        src = [_make(i, category='source') for i in range(30, 100)]
        ms = [_make(120)]
        sp, sb = source_boundaries(src)
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=src,
            color=False,
            line_length=79,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert YELLOW not in output
        assert RED not in output
        assert BOLD not in output

    def test_outlier_zone_annotation(self) -> None:
        ms = [
            _make(100),  # above p99
            _make(90),
            _make(50),
        ]
        src = [_make(i, category='source') for i in range(30, 100)]
        output = _run_stats(ms, src)
        assert '>p99' in output or '>p95' in output  # longest line gets annotation

    def test_outlier_zone_p90_no_color(self) -> None:
        src = [_make(i, category='source') for i in range(30, 100)]
        # Outlier at 94 chars (above p90=92, below p95=96 -> zone 3)
        ms = [_make(94)]
        sp, sb = source_boundaries(src)
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=src,
            color=True,
            line_length=None,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        # Should have BOLD (for loc) but not YELLOW or RED for annotation
        assert YELLOW not in output
        assert RED not in output

    def test_outlier_zone_p95_no_color(self) -> None:
        src = [_make(i, category='source') for i in range(30, 100)]
        ms = [_make(97)]
        sp, sb = source_boundaries(src)
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=src,
            color=True,
            line_length=None,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert '>p95' in output
        assert YELLOW not in output

    def test_outlier_zone_p99_no_color(self) -> None:
        src = [_make(i, category='source') for i in range(30, 100)]
        ms = [_make(120)]
        sp, sb = source_boundaries(src)
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=src,
            color=True,
            line_length=None,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert '>p99' in output
        assert YELLOW not in output
        assert RED not in output

    def test_sections_separated(self) -> None:
        ms = [_make(90, category='prose'), _make(95, category='args')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()

        with redirect_stdout(buf):
            _print_outliers(ctx)

        output = buf.getvalue()
        prose_heading = '    prose:'
        args_heading = '    args:'
        assert prose_heading in output
        assert args_heading in output
        longest_lines = output[output.index('Longest wrappable lines') :]
        between = longest_lines[
            longest_lines.index(prose_heading) : longest_lines.index(args_heading)
        ]
        assert '\n\n' in between

    def test_wrap_summary_includes_summary(self) -> None:
        ms = [_make(90), _make(85, category='summary', wrappable=True)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=None,
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_outliers(ctx)
        output = buf.getvalue()
        assert 'summary' in output


class TestPrintSkippedFooter:
    def test_skipped_footer_pluralizes(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_skipped_footer(2)
        assert 'Skipped 2 files' in buf.getvalue()

    def test_skipped_footer_singular(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_skipped_footer(1)
        assert 'Skipped 1 file' in buf.getvalue()


class TestSafetySkipFooter:
    def test_footer_printed_when_safety_skips(self, capsys: pytest.CaptureFixture[str]) -> None:
        data = StatsData(
            docstring_count=0,
            docstring_file_count=0,
            impact={},
            measurements=[],
            skipped_count=0,
            source_measurements=[],
            safety_skip_count=2,
        )
        print_stats(
            data,
            line_length=79,
            color=False,
            line_length_source=LineLengthSource.DEFAULT,
        )
        assert 'Skipped 2 docstrings for safety.' in capsys.readouterr().out

    def test_no_footer_without_safety_skips(self, capsys: pytest.CaptureFixture[str]) -> None:
        data = StatsData(
            docstring_count=0,
            docstring_file_count=0,
            impact={},
            measurements=[],
            skipped_count=0,
            source_measurements=[],
        )
        print_stats(
            data,
            line_length=79,
            color=False,
            line_length_source=LineLengthSource.DEFAULT,
        )
        assert 'for safety' not in capsys.readouterr().out


class TestSummaryTables:
    def test_column_headers_present(self) -> None:
        ms = [_make(40)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Category' in output
        assert 'Lines' in output
        assert 'Median' in output
        assert 'P75' in output
        assert 'P90' in output
        assert 'P95' in output
        assert 'P99' in output
        assert 'Max' in output

    def test_not_wrappable_before_wrappable(self) -> None:
        ms = [_make(40), _make(60, category='summary')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        nw_pos = output.index('not wrappable')
        w_pos = output.index('Docstrings (wrappable)')
        assert nw_pos < w_pos

    def test_not_wrappable_omitted_when_empty(self) -> None:
        ms = [_make(40)]  # no summary or code_block
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'not wrappable' not in output.lower()

    def test_not_wrappable_shown_with_code_block(self) -> None:
        ms = [_make(40), _make(60, category='code_block')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Docstrings (not wrappable)' in output

    def test_not_wrappable_shown_with_summary_when_wrap_summary_false(self) -> None:
        ms = [_make(40), _make(60, category='summary')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Docstrings (not wrappable)' in output
        assert 'summary' in output

    def test_red_on_values_exceeding_line_length(self) -> None:
        ms = [_make(90)]  # p90/p95/max all 90, exceeds default 79
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=True,
            line_length=79,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert RED in output

    def test_reference_header(self) -> None:
        source_ms = [_make(50, category='source'), _make(60, category='source')]
        doc_ms = [_make(40)]
        sp, sb = source_boundaries(source_ms)
        ctx = _StatsRenderCtx(
            measurements=doc_ms,
            source_measurements=source_ms,
            color=False,
            line_length=None,
            source_percentiles=sp,
            source_zone_boundaries=sb,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_source_table(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Source code' in output
        assert 'Source code reference:' not in output

    def test_source_table_omitted_when_empty(self) -> None:
        ms = [_make(40)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_source_table(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Source code' not in output

    def test_spanning_header_present(self) -> None:
        ms = [_make(40)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        # Dynamic format must be present
        assert 'Number of chars' in output
        lines = output.strip().split('\n')
        spanning = next(line for line in lines if 'Number of chars' in line)
        assert '- Number of chars -' in spanning

    def test_total_row_shown(self) -> None:
        ms = [_make(40), _make(50, category='args')]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'TOTAL' in output
        assert 'all' not in output.split('TOTAL')[0].split('\n')[-1]  # no 'all' row

    def test_total_row_shown_with_single_wrappable_category(self) -> None:
        ms = [_make(40), _make(50)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'TOTAL' in output

    def test_wrap_summary_moves_summary_to_wrappable(self) -> None:
        ms = [_make(40, category='summary', wrappable=True), _make(50)]
        ctx = _StatsRenderCtx(
            measurements=ms,
            source_measurements=[],
            color=False,
            line_length=None,
            source_percentiles=None,
            source_zone_boundaries=None,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_docstring_tables(ctx, leading_blank=False)
        output = buf.getvalue()
        assert 'Docstrings (wrappable)' in output
        lines = output.split('\n')
        wrappable_idx = next(
            i
            for i, line in enumerate(lines)
            if 'wrappable' in line.lower() and 'not' not in line.lower()
        )
        not_wrappable_indices = [
            i for i, line in enumerate(lines) if 'not wrappable' in line.lower()
        ]
        end_idx = not_wrappable_indices[0] if not_wrappable_indices else len(lines)
        wrappable_section = '\n'.join(lines[wrappable_idx:end_idx])
        assert 'summary' in wrappable_section


class TestTrimHistogramRows:
    def test_all_empty_returns_empty(self) -> None:
        h = build_histogram([])
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        assert len(trimmed) == 0

    def test_interior_zeros_preserved(self) -> None:
        ms = [_make(30), _make(65)]
        h = build_histogram(ms)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        # 1-40, 41-50 (empty), 51-60 (empty), 61-70, >70
        assert len(trimmed) == 5
        assert trimmed[1].bucket.count == 0  # 41-50 interior zero
        assert trimmed[2].bucket.count == 0  # 51-60 interior zero

    def test_leading_empty_trimmed(self) -> None:
        ms = [_make(65)]
        h = build_histogram(ms)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        assert trimmed[0].label == '61-70'

    def test_mixed_normal_and_overflow(self) -> None:
        ms = [_make(30), _make(110)]
        h = build_histogram(ms)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        # Should include: 1-40 (with data), ..., >100 (with data)
        assert trimmed[0].bucket.count == 1  # 1-40 bucket preserved
        assert trimmed[-1].bucket.is_overflow
        assert trimmed[-1].bucket.count == 1  # overflow bucket preserved

    def test_overflow_only(self) -> None:
        ms = [_make(110)]
        h = build_histogram(ms)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        assert len(trimmed) == 1
        assert trimmed[0].bucket.is_overflow
        assert trimmed[0].bucket.count == 1

    def test_overflow_row_zone_counts_preserved(self) -> None:
        boundaries = ZoneBoundaries(30, 50, 70, 80, 90)
        ms = [_make(35)]
        h = build_histogram(ms, boundaries=boundaries)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        overflow = trimmed[-1].bucket
        assert overflow.is_overflow
        assert overflow.zone_counts is not None
        assert len(overflow.zone_counts) == 6
        assert all(c == 0 for c in overflow.zone_counts)

    def test_trailing_empty_trimmed(self) -> None:
        ms = [_make(30)]
        h = build_histogram(ms)
        trimmed = _trim_histogram_rows(_build_histogram_rows(h.buckets))
        assert len(trimmed) == 2  # 1-40 + >40
        assert trimmed[0].bucket.count == 1
        assert trimmed[1].label == '>40'
        assert trimmed[1].bucket.is_overflow
        assert trimmed[1].bucket.count == 0


class TestZoneAnnotation:
    def test_zone_0_no_annotation(self) -> None:
        assert _zone_annotation(30, ZoneBoundaries(50, 75, 90, 95, 99)) is None

    def test_zone_1_no_annotation(self) -> None:
        assert _zone_annotation(60, ZoneBoundaries(50, 75, 90, 95, 99)) is None

    def test_zone_2_above_p75(self) -> None:
        assert _zone_annotation(80, ZoneBoundaries(50, 75, 90, 95, 99)) == '>p75'

    def test_zone_3_above_p90(self) -> None:
        assert _zone_annotation(92, ZoneBoundaries(50, 75, 90, 95, 99)) == '>p90'

    def test_zone_4_above_p95(self) -> None:
        assert _zone_annotation(97, ZoneBoundaries(50, 75, 90, 95, 99)) == '>p95'

    def test_zone_5_above_p99(self) -> None:
        assert _zone_annotation(100, ZoneBoundaries(50, 75, 90, 95, 99)) == '>p99'


def _make(
    length: int,
    *,
    category: MeasurementCategory = 'prose',
    file_path: str = 'f.py',
    line_number: int = 1,
    wrappable: bool | None = None,
) -> LineMeasurement:
    measurement = make_measurement(
        length,
        category=category,
        file_path=file_path,
        line_number=line_number,
    )
    if wrappable is None:
        wrappable = category in ('prose', 'args')
    return replace(measurement, wrappable=wrappable)


class TestFormatHistogram:
    @staticmethod
    def _make(length: int, category: MeasurementCategory = 'prose') -> LineMeasurement:
        return _make(length, category=category)

    def test_ascii_only(self) -> None:
        lines = [self._make(n) for n in range(30, 120, 5)]
        h = build_histogram(lines)
        output = _format_histogram(h, label='test')
        assert all(ord(c) < 128 for c in output)

    def test_bars_use_hashes(self) -> None:
        lines = [self._make(55)] * 10
        h = build_histogram(lines)
        output = _format_histogram(h, label='test')
        assert '#' in output

    def test_color_false_no_ansi(self) -> None:
        lines = [self._make(n) for n in [30, 55, 75]]
        h = build_histogram(lines, boundaries=ZoneBoundaries(50, 60, 70, 80, 90))
        output = _format_histogram(h, label='test', color=False)
        assert '\033[' not in output

    def test_color_true_with_zones_has_ansi(self) -> None:
        lines = [self._make(n) for n in [30, 55, 75]]
        h = build_histogram(lines, boundaries=ZoneBoundaries(50, 60, 70, 80, 90))
        output = _format_histogram(h, label='test', color=True)
        assert '\033[' in output

    def test_color_true_without_zones_bars_monochrome(self) -> None:
        lines = [self._make(55)]
        h = build_histogram(lines)
        output = _format_histogram(h, label='test', color=True)
        data_lines = [ln for ln in output.split('\n') if '#' in ln]
        assert data_lines
        for ln in data_lines:
            assert '\033[' not in ln

    def test_colored_column_alignment(self) -> None:
        boundaries = ZoneBoundaries(40, 55, 70, 85, 95)
        ms = [self._make(i) for i in range(20, 100)]
        h = build_histogram(ms, boundaries=boundaries)
        pctls = Percentiles(
            levels={50: 40, 75: 55, 90: 70, 95: 85, 99: 95},
            max=99,
        )
        output = _format_histogram(
            h,
            label='test',
            color=True,
            source_percentiles=pctls,
        )

        # Strip ANSI escape codes
        ansi_re = re.compile(r'\x1b\[[0-9;]*m')
        lines = output.split('\n')

        # Find data rows (skip subtitle and header)
        data_rows = [ln for ln in lines if '#' in ln or (ln.strip() and ln.strip()[0].isdigit())]
        assert len(data_rows) >= 2, 'Need at least 2 data rows for alignment check'

        # Strip ANSI and check core column positions are consistent
        stripped = [ansi_re.sub('', row) for row in data_rows]

        # All rows should have the same length up to the Lines% column
        core_widths: list[int] = []
        for row in stripped:
            pct_pos = row.rfind('%')
            if pct_pos >= 0:
                core_widths.append(pct_pos + 1)

        assert len(set(core_widths)) == 1, f'Column misalignment: core widths = {core_widths}'

    def test_columns_aligned(self) -> None:
        lines = [self._make(55), self._make(105)]
        h = build_histogram(lines)
        output = _format_histogram(h, label='test')
        rows = output.strip().split('\n')[2:]  # skip label line and header line
        widths = [len(row) for row in rows]
        assert len(set(widths)) == 1, f'Row widths vary: {widths}'

    def test_no_separator_single_zone(self) -> None:
        lines = [self._make(30)] * 5
        h = build_histogram(lines, boundaries=ZoneBoundaries(50, 60, 70, 80, 90))
        output = _format_histogram(h, label='test', color=True)
        bucket_line = next(ln for ln in output.split('\n') if '1-40' in ln)
        assert '|' not in bucket_line

    def test_output_has_all_buckets(self) -> None:
        lines = [self._make(55), self._make(75), self._make(110)]
        h = build_histogram(lines)
        output = _format_histogram(h, label='test')
        assert '51-60' in output
        assert '>100' in output

    def test_output_has_column_headers(self) -> None:
        h = build_histogram([self._make(55)])
        output = _format_histogram(h, label='test')
        assert 'Chars' in output
        assert 'Lines' in output
        assert 'Lines%' in output

    def test_output_has_label(self) -> None:
        h = build_histogram([self._make(55)])
        output = _format_histogram(h, label='prose')
        assert 'prose:' in output

    def test_separators_between_zones(self) -> None:
        lines = [self._make(41), self._make(48)]
        h = build_histogram(lines, boundaries=ZoneBoundaries(40, 45, 50, 55, 60))
        output = _format_histogram(h, label='test', color=True)
        bucket_line = next(ln for ln in output.split('\n') if '41-50' in ln)
        assert '|' in bucket_line

    def test_subtitle_none_without_source_percentiles(self) -> None:
        lines = [self._make(55)]
        h = build_histogram(lines)
        output = _format_histogram(h, label='prose', source_percentiles=None)
        assert 'prose:' in output
        assert '(source:' not in output

    def test_subtitle_with_source_percentiles(self) -> None:
        lines = [self._make(55)]
        h = build_histogram(lines)
        pctls = Percentiles(levels={50: 32, 75: 55, 90: 78, 95: 85, 99: 95}, max=99)
        output = _format_histogram(h, label='prose', source_percentiles=pctls)
        assert 'prose (source: median=32, p75=55, p90=78, p95=85, p99=95):' in output


class TestHistogramRowLabels:
    def test_labels_are_range_format(self) -> None:
        rows = _build_histogram_rows(build_histogram([_make(55)]).buckets)
        assert [row.label for row in rows] == [
            '1-40',
            '41-50',
            '51-60',
            '61-70',
            '71-80',
            '81-90',
            '91-100',
            '>100',
        ]

    @pytest.mark.parametrize(
        ('lengths', 'expected'),
        [
            (
                [30, 55, 65, 75, 85, 95, 110],
                ['15%', '0%', '15%', '14%', '14%', '14%', '14%', '14%'],
            ),
            ([30] * 199 + [55], ['100%', '0%', '<1%', '0%', '0%', '0%', '0%', '0%']),
            ([], ['0%'] * 8),
        ],
    )
    def test_percentage_strings(self, lengths: list[int], expected: list[str]) -> None:
        rows = _build_histogram_rows(build_histogram([_make(n) for n in lengths]).buckets)
        assert [row.percentage for row in rows] == expected


class TestPrintStatsSections:
    _BODY_HEADINGS = (
        'Source code',
        'Docstrings (wrappable)',
        'Distribution',
        'Longest wrappable lines',
        'Wrapping impact',
        'Recommended targets',
    )

    @staticmethod
    def _data(
        *,
        measurements: list[LineMeasurement] | None = None,
        source_measurements: list[LineMeasurement] | None = None,
        docstring_count: int = 2,
        skipped_count: int = 0,
        safety_skip_count: int = 0,
    ) -> StatsData:
        impact = {
            length: ImpactRow(
                line_length=length,
                docstrings_changed=1,
                files_changed=1,
                lines_added=2,
                lines_removed=1,
            )
            for length in (79, 80)
        }
        return StatsData(
            docstring_count=docstring_count,
            docstring_file_count=0 if docstring_count == 0 else 1,
            impact=impact,
            measurements=(
                measurements
                if measurements is not None
                else [_make(70), _make(80, category='args')]
            ),
            skipped_count=skipped_count,
            source_measurements=(
                source_measurements
                if source_measurements is not None
                else [_make(60, category='source')]
            ),
            safety_skip_count=safety_skip_count,
        )

    @staticmethod
    def _render(
        data: StatsData,
        *,
        sections: tuple[StatsSection, ...],
        longest_per_category: int = 3,
    ) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_stats(
                data,
                line_length=79,
                color=False,
                line_length_source=LineLengthSource.DEFAULT,
                sections=sections,
                longest_per_category=longest_per_category,
            )
        return buf.getvalue()

    def test_configured_order_is_render_order(self) -> None:
        output = self._render(self._data(), sections=('recommendations', 'source', 'impact'))

        assert output.index('Recommended targets') < output.index('Source code')
        assert output.index('Source code') < output.index('Wrapping impact')

    def test_default_arguments_keep_all_sections_in_current_order(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_stats(
                self._data(),
                line_length=79,
                color=False,
                line_length_source=LineLengthSource.DEFAULT,
            )
        output = buf.getvalue()

        for index in range(len(self._BODY_HEADINGS) - 1):
            assert output.index(self._BODY_HEADINGS[index]) < output.index(
                self._BODY_HEADINGS[index + 1]
            )

    def test_empty_sections_keeps_preamble_and_footers(self) -> None:
        output = self._render(self._data(skipped_count=2, safety_skip_count=1), sections=())

        assert 'Scanned: 2 docstrings across 1 file' in output
        assert 'Configured length: 79 (from default)' in output
        assert 'Skipped 2 files (unreadable).' in output
        assert 'Skipped 1 docstring for safety.' in output
        assert all(heading not in output for heading in self._BODY_HEADINGS)

    def test_longest_per_category_limits_each_category(self) -> None:
        measurements = [
            *[
                _make(
                    70 + index,
                    file_path=f'prose_{index}.py',
                    line_number=index,
                )
                for index in range(1, 5)
            ],
            *[
                _make(
                    80 + index,
                    category='args',
                    file_path=f'args_{index}.py',
                    line_number=index,
                )
                for index in range(1, 5)
            ],
        ]
        output = self._render(
            self._data(measurements=measurements),
            sections=('longest',),
            longest_per_category=2,
        )

        assert output.count('prose_') == 2
        assert output.count('args_') == 2

    @pytest.mark.parametrize(
        ('sections', 'present', 'absent'),
        [
            (('source',), 'Source code', 'Docstrings (wrappable)'),
            (('docstrings',), 'Docstrings (wrappable)', 'Source code'),
        ],
    )
    def test_source_and_docstrings_are_independent(
        self,
        sections: tuple[StatsSection, ...],
        present: str,
        absent: str,
    ) -> None:
        output = self._render(self._data(), sections=sections)

        assert present in output
        assert absent not in output

    def test_subset_omits_unselected_sections(self) -> None:
        output = self._render(self._data(), sections=('source', 'longest'))

        assert 'Source code' in output
        assert 'Longest wrappable lines' in output
        assert 'Docstrings (wrappable)' not in output
        assert 'Distribution' not in output
        assert 'Wrapping impact' not in output
        assert 'Recommended targets' not in output

    def test_zero_docstrings_keeps_existing_diagnostic(self) -> None:
        output = self._render(
            self._data(
                measurements=[],
                source_measurements=[],
                docstring_count=0,
                skipped_count=2,
                safety_skip_count=1,
            ),
            sections=('recommendations', 'source'),
        )

        assert output.index('No docstrings found.') < output.index('Skipped 2 files (unreadable).')
        assert output.index('Skipped 2 files (unreadable).') < output.index(
            'Skipped 1 docstring for safety.'
        )
        assert all(heading not in output for heading in self._BODY_HEADINGS)

    def test_zero_wrappable_renders_applicable_profiles_in_order(self) -> None:
        output = self._render(
            self._data(measurements=[_make(60, category='summary')]),
            sections=('docstrings', 'source', 'distribution'),
        )

        assert output.index('Docstrings (not wrappable)') < output.index('Source code')
        assert 'Distribution' not in output
        assert output.index('Source code') < output.index('No wrappable lines found.')


class TestWrapSummaryConsistency:
    def test_categories_consistent_across_sections(self) -> None:
        ms = [
            _make(50),
            _make(50, category='args'),
            _make(50, category='summary', wrappable=True),
        ]
        output = _run_stats(ms, [])
        assert output.count('summary') >= 3  # distribution, outliers, summary table

    @pytest.mark.parametrize(('wrappable', 'message_present'), [(False, True), (True, False)])
    def test_no_wrappable_decision_uses_stored_value(
        self,
        wrappable: bool,
        message_present: bool,
    ) -> None:
        data = StatsData(
            docstring_count=1,
            docstring_file_count=1,
            impact={},
            measurements=[LineMeasurement(50, 'summary', 'f.py', 1, 'summary', wrappable)],
            skipped_count=0,
            source_measurements=[],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_stats(
                data,
                line_length=79,
                color=False,
                line_length_source=LineLengthSource.DEFAULT,
                sections=('docstrings',),
            )
        assert ('No wrappable lines found.' in buf.getvalue()) is message_present

    def test_stored_wrappability_controls_every_rendering_classification(self) -> None:
        ms = [
            LineMeasurement(50, 'summary', 'not.py', 1, 'not wrappable', wrappable=False),
            LineMeasurement(60, 'summary', 'yes.py', 2, 'wrappable', wrappable=True),
            LineMeasurement(70, 'prose', 'prose.py', 3, 'prose', wrappable=True),
        ]
        output = _run_stats(ms, [])

        not_wrappable, wrappable = output.split('Docstrings (wrappable)', maxsplit=1)
        assert 'summary' in not_wrappable
        assert 'summary' in wrappable
        assert 'yes.py:2' in output
        assert 'not.py:1' not in output


def _impact_options(*, color: bool, line_length: int | None = None) -> _ImpactTableOptions:
    return _ImpactTableOptions(
        docstring_count=10,
        docstring_file_count=5,
        color=color,
        line_length=line_length,
    )


def _run_stats(
    measurements: list[LineMeasurement],
    source_measurements: list[LineMeasurement],
    *,
    color: bool = False,
    line_length: int | None = None,
) -> str:
    sp, sb = source_boundaries(source_measurements)
    ctx = _StatsRenderCtx(
        measurements=measurements,
        source_measurements=source_measurements,
        color=color,
        line_length=line_length,
        source_percentiles=sp,
        source_zone_boundaries=sb,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        printed = _print_source_table(ctx, leading_blank=False)
        _print_docstring_tables(ctx, leading_blank=printed)
        _print_distributions(ctx)
        _print_outliers(ctx)
    return buf.getvalue()
