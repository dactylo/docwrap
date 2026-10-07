"""Statistics rendering: histograms, impact tables, outliers, percentile tables."""

from collections.abc import Collection, Sequence
from dataclasses import astuple, dataclass
from typing import assert_never

from docwrap._color import (
    BOLD,
    GREEN,
    HISTOGRAM_SPECTRUM,
    RESET,
    YELLOW,
    bold_if,
    colorize,
    path_text,
    percentile_color,
    visible_controls,
)
from docwrap._display._text import pluralize
from docwrap._stats import (
    BUCKET_BOUNDS,
    PERCENTILE_LEVELS,
    STANDARD_LENGTHS,
    WRAPPABLE_CATEGORIES,
    Bucket,
    Histogram,
    ImpactRow,
    LineMeasurement,
    Percentiles,
    StatsData,
    ZoneBoundaries,
    allocate_percentages,
    build_histogram,
    compute_percentiles,
    find_outliers,
    source_boundaries,
)
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    DEFAULT_STATS_LONGEST_PER_CATEGORY,
    DEFAULT_STATS_SECTIONS,
    LineLengthSource,
    StatsSection,
)

_MarkerPayload = tuple[int, list[str]]
_MAX_BAR_WIDTH = 30
_StatsRow = tuple[str, *tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _HistogramRow:
    """A numeric histogram bucket with its terminal labels."""

    bucket: Bucket
    label: str
    percentage: str


@dataclass(frozen=True, slots=True)
class _ImpactDataRow:
    """Display-ready impact table row."""

    length: int
    lines_added: int
    lines_removed: int
    net: str
    docstrings_changed: int
    docstrings_pct: str
    files_changed: int
    files_pct: str


@dataclass(frozen=True, slots=True)
class _ImpactLayout:
    """Column widths and total table width for impact rows."""

    widths: tuple[int, ...]
    total_width: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _ImpactTableOptions:
    """Options and denominators for impact table rendering."""

    docstring_count: int = 0
    docstring_file_count: int = 0
    color: bool = False
    line_length: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _StatsRenderCtx:
    """Rendering context shared across stats printing functions."""

    measurements: list[LineMeasurement]
    source_measurements: list[LineMeasurement] | None
    color: bool
    line_length: int | None
    source_percentiles: Percentiles | None
    source_zone_boundaries: ZoneBoundaries | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _StatsSectionOptions:
    """Options shared by ordered stats section dispatch."""

    configured_length: int
    line_length_source: LineLengthSource
    longest_per_category: int
    wrappable: bool


def _build_group_rows(
    by_cat: dict[str, list[LineMeasurement]],
    categories: tuple[str, ...],
) -> tuple[list[_StatsRow], list[LineMeasurement]]:
    """Build table rows for a group of measurement categories."""
    rows: list[_StatsRow] = []
    all_ms: list[LineMeasurement] = []
    for cat in categories:
        cat_ms = by_cat.get(cat, [])
        if not cat_ms:
            continue
        p = compute_percentiles(cat_ms)
        if p is None:
            continue
        label = 'code blocks' if cat == 'code_block' else cat
        rows.append((label, len(cat_ms), *(p.levels[lv] for lv in PERCENTILE_LEVELS), p.max))
        all_ms.extend(cat_ms)
    return rows, all_ms


def _build_histogram_rows(buckets: tuple[Bucket, ...]) -> tuple[_HistogramRow, ...]:
    """Attach range labels and allocated percentages before trimming buckets."""
    percentages = allocate_percentages([bucket.count for bucket in buckets])
    rows: list[_HistogramRow] = []
    previous = 0
    for bucket, percentage in zip(buckets, percentages, strict=True):
        label = f'>{previous}' if bucket.is_overflow else f'{previous + 1}-{bucket.upper}'
        rows.append(
            _HistogramRow(bucket, label, '<1%' if percentage is None else f'{percentage}%')
        )
        previous = bucket.upper
    return tuple(rows)


def _distribute_bar_widths(
    zones: list[tuple[int, int]],
    hash_available: int,
) -> list[int]:
    """Distribute hash-character widths across zones."""
    n_zones = len(zones)
    widths = [1] * n_zones
    remainder = hash_available - n_zones
    if remainder <= 0:
        return widths

    total_lines = sum(c for _, c in zones)
    if total_lines == 0:
        return widths

    exact = [c / total_lines * remainder for _, c in zones]
    floors = [int(e) for e in exact]
    leftover = remainder - sum(floors)
    fractionals = [(exact[j] - floors[j], j) for j in range(n_zones)]
    fractionals.sort(key=lambda x: (-x[0], x[1]))
    for k in range(leftover):
        floors[fractionals[k][1]] += 1
    for j in range(n_zones):
        widths[j] += floors[j]
    return widths


def _dynamic_spanning(label: str, span_width: int) -> str:
    """Build a spanning header with dynamic dashes."""
    min_width = len(label) + 4  # "- " + label + " -"
    if span_width < min_width:
        return label.center(span_width)
    remaining = span_width - len(label) - 2  # subtract flanking spaces
    left = remaining // 2
    right = remaining - left
    return '-' * left + ' ' + label + ' ' + '-' * right


def _build_impact_spanning(widths: Sequence[int]) -> str:
    """Build the spanning header line for the impact table."""
    # "Line change" spans +Lines/-Lines/Net (cols 1-3)
    lines_span_start = 4 + widths[0] + 2
    lines_span_end = lines_span_start + widths[1] + 2 + widths[2] + 2 + widths[3]
    lines_span_width = lines_span_end - lines_span_start
    lines_text = _dynamic_spanning('Line change', lines_span_width)

    # "Scope" spans Docs/Docs%/Files/Files% (cols 4-7)
    counts_span_start = lines_span_end + 2
    counts_span_end = counts_span_start + widths[4] + 2 + widths[5] + 2 + widths[6] + 2 + widths[7]
    counts_span_width = counts_span_end - counts_span_start
    counts_text = _dynamic_spanning('Scope', counts_span_width)

    return ' ' * lines_span_start + lines_text + '  ' + counts_text


def _find_pipe_target(
    buckets: tuple[Bucket, ...],
    start: int,
    boundary_idx: int,
) -> int | None:
    """Scan backward for the last bucket with data from zones <= boundary_idx."""
    for si in range(start, -1, -1):
        sb = buckets[si]
        if sb.zone_counts is not None and any(
            sb.zone_counts[z] > 0 for z in range(boundary_idx + 1)
        ):
            return si
    return None


def _fmt_pct(numerator: int, denominator: int) -> str:
    """Format a percentage with int() truncation and <1% rule."""
    if denominator == 0 or numerator == 0:
        return '0%'
    exact = numerator / denominator * 100
    if exact < 1.0:
        return '<1%'
    return f'{int(exact)}%'


def _build_impact_data_rows(
    targets: list[int],
    agg_impact: dict[int, ImpactRow],
    docstring_count: int,
    docstring_file_count: int,
) -> list[_ImpactDataRow]:
    """Build data rows for the impact table."""
    data_rows: list[_ImpactDataRow] = []
    for length in targets:
        row = agg_impact[length]
        net = row.lines_added - row.lines_removed
        net_str = f'+{net}' if net > 0 else str(net)
        docs_pct = _fmt_pct(row.docstrings_changed, docstring_count)
        files_pct = _fmt_pct(row.files_changed, docstring_file_count)
        data_rows.append(
            _ImpactDataRow(
                length=length,
                lines_added=row.lines_added,
                lines_removed=row.lines_removed,
                net=net_str,
                docstrings_changed=row.docstrings_changed,
                docstrings_pct=docs_pct,
                files_changed=row.files_changed,
                files_pct=files_pct,
            )
        )
    return data_rows


def _format_impact_data_row(
    row: _ImpactDataRow,
    widths: Sequence[int],
    *,
    color: bool = False,
    line_length: int | None = None,
) -> str:
    """Format a single impact table data row."""
    target_str = f'{row.length:>{widths[0]}}'
    if color:
        tc = percentile_color(row.length, line_length)
        if tc is not None:
            target_str = colorize(target_str, tc)

    return (
        f'    {target_str}'
        f'  {row.lines_added:>{widths[1]}}'
        f'  {row.lines_removed:>{widths[2]}}'
        f'  {row.net:>{widths[3]}}'
        f'  {row.docstrings_changed:>{widths[4]}}'
        f'  {row.docstrings_pct:>{widths[5]}}'
        f'  {row.files_changed:>{widths[6]}}'
        f'  {row.files_pct:>{widths[7]}}'
    )


def _format_impact_marker(
    val: int,
    labels: list[str],
    widths: Sequence[int],
    total_width: int,
) -> str:
    """Format a source percentile marker line for the impact table."""
    label_str = ', '.join(labels)
    annotation = f'(source, {label_str})'
    val_str = f'    {val:>{widths[0]}}'
    dash_start = len(val_str) + 2
    avail = total_width - dash_start
    if avail > 0:
        dash_content = '- ' * ((avail + 1) // 2)
        dash_content = dash_content[:avail].rstrip()
    else:
        dash_content = '- - -'
    return f'{val_str}  {dash_content}  {annotation}'


def _emit_impact_rows(
    data_rows: list[_ImpactDataRow],
    marker_map: dict[int, list[str]],
    layout: _ImpactLayout,
    options: _ImpactTableOptions,
) -> None:
    """Emit impact data rows with interleaved source percentile markers."""
    # Build interleaved event list: (value, is_marker, payload)
    events: list[tuple[int, bool, _ImpactDataRow | _MarkerPayload]] = [
        (row.length, False, row) for row in data_rows
    ]
    events.extend((val, True, (val, labels)) for val, labels in marker_map.items())
    events.sort(key=lambda e: (e[0], e[1]))

    # Emit with collapse logic
    i = 0
    while i < len(events):
        _val, _is_marker, payload = events[i]
        if isinstance(payload, _ImpactDataRow):
            print(
                _format_impact_data_row(
                    payload,
                    layout.widths,
                    color=options.color,
                    line_length=options.line_length,
                )
            )
            i += 1
            continue

        # Collect consecutive markers
        group: list[_MarkerPayload] = [payload]
        j = i + 1
        while j < len(events) and events[j][1]:
            marker = events[j][2]
            if isinstance(marker, tuple):
                group.append(marker)
                j += 1
            else:
                break

        # Collapse group
        first_val = group[0][0]
        all_same = all(g[0] == first_val for g in group)
        if all_same:
            all_labels = [lbl for g in group for lbl in g[1]]
            marker_line = _format_impact_marker(
                first_val, all_labels, layout.widths, layout.total_width
            )
        else:
            labeled = [f'{lbl}={g_val}' for g_val, g_labels in group for lbl in g_labels]
            marker_line = _format_impact_marker(
                group[0][0], labeled, layout.widths, layout.total_width
            )

        print(bold_if(marker_line, options.color))
        i = j


def _format_outlier_chars(length: int, color: bool, line_length: int | None) -> str:
    """Render an outlier's character count with optional percentile color."""
    chars_str = str(length)
    if color:
        pc = percentile_color(length, line_length)
        if pc is not None:
            chars_str = colorize(chars_str, pc)
    return chars_str


def _level_display_name(level: int) -> str:
    """Name the median specially and prefix other percentile levels with `p`."""
    return 'median' if level == 50 else f'p{level}'  # noqa: PLR2004


def _compute_bucket_labels(
    buckets: tuple[Bucket, ...],
    source_percentiles: Percentiles | None,
) -> tuple[list[list[str]], list[frozenset[int]]]:
    """Compute right-side labels and active boundary indices for histogram rows."""
    n = len(buckets)
    if source_percentiles is None:
        return [[] for _ in range(n)], [frozenset()] * n

    bvalues = [source_percentiles.levels[p] for p in PERCENTILE_LEVELS]

    # Compute correct initial prev_upper for first trimmed bucket
    prev_upper = 0
    if buckets and not buckets[0].is_overflow:
        b0_upper = buckets[0].upper
        if b0_upper in BUCKET_BOUNDS:
            idx = BUCKET_BOUNDS.index(b0_upper)
            if idx > 0:
                prev_upper = BUCKET_BOUNDS[idx - 1]

    bucket_labels: list[list[str]] = []
    bucket_boundaries: list[frozenset[int]] = []
    for bucket in buckets:
        if bucket.is_overflow or bucket.count == 0:
            bucket_labels.append([])
            bucket_boundaries.append(frozenset())
            if not bucket.is_overflow:
                prev_upper = bucket.upper
            continue
        names: list[str] = []
        active: set[int] = set()
        for j, p in enumerate(PERCENTILE_LEVELS):
            if prev_upper < bvalues[j] <= bucket.upper:
                names.append(_level_display_name(p))
                active.add(j)
        bucket_labels.append(names)
        bucket_boundaries.append(frozenset(active))
        prev_upper = bucket.upper

    return bucket_labels, bucket_boundaries


def _format_stats_table(
    header: str,
    rows: list[_StatsRow],
    *,
    color: bool = False,
    bold_row: str | None = None,
    line_length: int | None = None,
) -> str:
    """Format a stats summary table with aligned columns and spanning header."""
    if not rows:
        return ''

    pct_headers = tuple(_level_display_name(p).capitalize() for p in PERCENTILE_LEVELS)
    col_headers = ('Category', 'Lines', *pct_headers, 'Max')
    cat_width = max(len(col_headers[0]), *(len(r[0]) for r in rows))
    num_widths = [
        max(len(col_headers[i + 1]), *(len(str(r[i + 1])) for r in rows))
        for i in range(len(col_headers) - 1)
    ]

    # Column header line
    hdr_line = f'    {col_headers[0]:<{cat_width}}'
    for i, w in enumerate(num_widths):
        hdr_line += f'  {col_headers[i + 1]:>{w}}'

    # Spanning header: "--- Number of chars ---" centered over Median through Max
    # Calculate the span width: from start of Median column to end of Max column
    lines_col_end = 4 + cat_width + 2 + num_widths[0]  # after Lines column
    span_start = lines_col_end + 2  # start of Median column (after 2-space gap)
    span_end = len(hdr_line)  # end of Max column
    span_width = span_end - span_start
    spanning = ' ' * span_start + _dynamic_spanning('Number of chars', span_width)

    lines: list[str] = []
    lines.append(bold_if(header, color))
    lines.append(bold_if(spanning, color))
    lines.append(bold_if(hdr_line, color))

    for cat, *nums in rows:
        is_bold = bool(color and bold_row and cat == bold_row)
        row_str = f'    {cat:<{cat_width}}'
        for i, n in enumerate(nums):
            val_str = f'{n:>{num_widths[i]}}'
            pct_color = percentile_color(n, line_length) if i >= 1 else None
            if color and pct_color is not None:
                if is_bold:
                    val_str = colorize(val_str, BOLD, pct_color)
                else:
                    val_str = colorize(val_str, pct_color)
            row_str += f'  {val_str}'
        if is_bold:
            row_str = colorize(row_str, BOLD)
        lines.append(row_str)

    return '\n'.join(lines)


def _group_by_category(
    measurements: list[LineMeasurement],
    categories: Collection[str] | None = None,
) -> dict[str, list[LineMeasurement]]:
    """Group measurements by category, optionally filtering to a subset."""
    by_cat: dict[str, list[LineMeasurement]] = {}
    for m in measurements:
        if categories is None or m.category in categories:
            by_cat.setdefault(m.category, []).append(m)
    return by_cat


def _place_boundaries(
    buckets: tuple[Bucket, ...],
    raw_labels: list[list[str]],
    bucket_boundaries: list[frozenset[int]],
    *,
    color: bool,
) -> tuple[list[list[str]], list[bool]]:
    """Place boundary pipes and relocate labels for one-sided boundaries."""
    final_labels: list[list[str]] = [list(names) for names in raw_labels]
    trailing_pipes: list[bool] = [False] * len(buckets)

    if not color:
        return final_labels, trailing_pipes

    for bi, bucket in enumerate(buckets):
        if not bucket_boundaries[bi] or bucket.zone_counts is None:
            continue
        zc = bucket.zone_counts
        for b in sorted(bucket_boundaries[bi]):
            has_before = any(zc[z] > 0 for z in range(b + 1))
            has_after = any(zc[z] > 0 for z in range(b + 1, len(zc)))
            if has_before and has_after:
                continue  # natural | from zone coloring

            label_name = _level_display_name(PERCENTILE_LEVELS[b])
            target = _find_pipe_target(buckets, bi, b)

            if target is not None:
                trailing_pipes[target] = True
                if target != bi:
                    if label_name in final_labels[bi]:
                        final_labels[bi].remove(label_name)
                    if label_name not in final_labels[target]:
                        final_labels[target].append(label_name)

    return final_labels, trailing_pipes


def _print_configured_length(
    line_length: int | None,
    line_length_source: LineLengthSource | None,
    *,
    color: bool,
) -> None:
    """Print the configured line length header."""
    val = line_length if line_length is not None else DEFAULT_LINE_LENGTH
    src = line_length_source if line_length_source is not None else 'default'
    val_str = colorize(str(val), YELLOW) if color else str(val)
    print(f'Configured length: {val_str} (from {src})')


def _print_docstring_tables(ctx: _StatsRenderCtx, *, leading_blank: bool) -> bool:
    """Print the applicable docstring profiles."""
    wrappable_ms = [m for m in ctx.measurements if m.wrappable]
    not_wrappable_ms = [m for m in ctx.measurements if not m.wrappable]
    tables: list[str] = []

    nw_rows, _ = _build_group_rows(
        _group_by_category(not_wrappable_ms),
        ('summary', 'code_block'),
    )
    if nw_rows:
        tables.append(
            _format_stats_table(
                'Docstrings (not wrappable)',
                nw_rows,
                color=ctx.color,
                line_length=ctx.line_length,
            )
        )

    wrappable_rows, all_wrappable = _build_group_rows(
        _group_by_category(wrappable_ms),
        WRAPPABLE_CATEGORIES,
    )
    if wrappable_rows:
        percentiles = compute_percentiles(all_wrappable)
        if percentiles is not None:
            wrappable_rows.append(
                (
                    'TOTAL',
                    len(all_wrappable),
                    *(percentiles.levels[level] for level in PERCENTILE_LEVELS),
                    percentiles.max,
                )
            )
        tables.append(
            _format_stats_table(
                'Docstrings (wrappable)',
                wrappable_rows,
                color=ctx.color,
                bold_row='TOTAL',
                line_length=ctx.line_length,
            )
        )

    if not tables:
        return False

    separator = '\n' if leading_blank else ''
    print(separator + '\n\n'.join(tables))
    return True


def _print_impact_table(
    agg_impact: dict[int, ImpactRow],
    source_percentiles: Percentiles | None = None,
    options: _ImpactTableOptions | None = None,
    *,
    leading_blank: bool = True,
) -> bool:
    """Print the wrapping-impact section when applicable."""
    if not agg_impact:
        return False

    render_options = options if options is not None else _ImpactTableOptions()
    targets = sorted(agg_impact)

    # Build marker map from source percentiles
    marker_map: dict[int, list[str]] = {}
    if source_percentiles is not None:
        lo = STANDARD_LENGTHS[0] - 10
        hi = STANDARD_LENGTHS[-1] + 20
        for p in PERCENTILE_LEVELS:
            if p < 90:  # noqa: PLR2004
                continue
            val = source_percentiles.levels[p]
            if lo <= val <= hi:
                label = _level_display_name(p)
                marker_map.setdefault(val, []).append(label)

    separator = '\n' if leading_blank else ''
    print(f'{separator}{bold_if("Wrapping impact", render_options.color)}')

    # Compute data rows and column widths
    data_rows = _build_impact_data_rows(
        targets,
        agg_impact,
        render_options.docstring_count,
        render_options.docstring_file_count,
    )
    col_names = ('Target', '+Lines', '-Lines', 'Net', 'Docs', 'Docs%', 'Files', 'Files%')
    widths = [len(name) for name in col_names]
    for row in data_rows:
        for ci, val in enumerate(astuple(row)):
            widths[ci] = max(widths[ci], len(str(val)))

    # Build and print column headers with spanning
    col_hdr = f'    {col_names[0]:>{widths[0]}}'
    for i in range(1, 8):
        col_hdr += f'  {col_names[i]:>{widths[i]}}'
    spanning = _build_impact_spanning(widths)

    print(bold_if(spanning, render_options.color))
    print(bold_if(col_hdr, render_options.color))

    # Emit rows with interleaved markers
    _emit_impact_rows(
        data_rows,
        marker_map,
        _ImpactLayout(tuple(widths), len(col_hdr)),
        render_options,
    )
    return True


def _print_safety_skip_footer(count: int) -> None:
    """Report docstrings left unchanged for safety."""
    label = pluralize('docstring', count)
    print(f'\n  Skipped {count} {label} for safety.')


def _print_skipped_footer(count: int) -> None:
    """Report unreadable files omitted from statistics."""
    fs = pluralize('file', count)
    print(f'\n  Skipped {count} {fs} (unreadable).')


def _print_source_table(ctx: _StatsRenderCtx, *, leading_blank: bool) -> bool:
    """Print the source-code profile when applicable."""
    if not ctx.source_measurements:
        return False

    rows, _ = _build_group_rows({'source': ctx.source_measurements}, ('source',))
    if not rows:
        return False

    separator = '\n' if leading_blank else ''
    print(
        separator
        + _format_stats_table('Source code', rows, color=ctx.color, line_length=ctx.line_length)
    )
    return True


def _trim_histogram_rows(rows: tuple[_HistogramRow, ...]) -> tuple[_HistogramRow, ...]:
    """Trim empty display rows, retaining interior zeros and overflow."""
    non_empty = [i for i, row in enumerate(rows) if row.bucket.count > 0]
    if not non_empty:
        return ()
    first, last = non_empty[0], non_empty[-1]
    if rows[last].bucket.is_overflow:
        return rows[first : last + 1]

    last_upper = rows[last].bucket.upper
    zones = rows[-1].bucket.zone_counts
    overflow = Bucket(
        upper=0,
        count=0,
        zone_counts=tuple(0 for _ in zones) if zones is not None else None,
        is_overflow=True,
    )
    return (*rows[first : last + 1], _HistogramRow(overflow, f'>{last_upper}', '0%'))


def _zone_annotation(char_count: int, boundaries: ZoneBoundaries) -> str | None:
    """Return an outlier annotation above p75, or None in the first two zones."""
    for level, bound in reversed(tuple(zip(PERCENTILE_LEVELS[1:], boundaries[1:], strict=True))):
        if char_count > bound:
            return f'>{_level_display_name(level)}'
    return None


def _format_outlier_annotation(
    length: int,
    boundaries: ZoneBoundaries | None,
) -> str:
    """Format a zone annotation suffix for an outlier line."""
    if boundaries is None:
        return ''
    zone_label = _zone_annotation(length, boundaries)
    if zone_label is None:
        return ''
    return f', {zone_label}'


def _print_outliers(
    ctx: _StatsRenderCtx,
    *,
    limit: int = DEFAULT_STATS_LONGEST_PER_CATEGORY,
    leading_blank: bool = True,
) -> bool:
    """Print the longest-lines section when applicable."""
    boundaries = ctx.source_zone_boundaries

    any_printed = False
    for cat in WRAPPABLE_CATEGORIES:
        cat_ms = [m for m in ctx.measurements if m.wrappable and m.category == cat]
        outliers = find_outliers(cat_ms, n=limit)
        if not outliers:
            continue
        if not any_printed:
            header = 'Longest wrappable lines'
            separator = '\n' if leading_blank else ''
            print(f'{separator}{bold_if(header, ctx.color)}')
            any_printed = True
        print(f'\n    {cat}:')
        for m in outliers:
            loc = visible_controls(f'{path_text(m.file_path)}:{m.line_number}')
            if ctx.color:
                loc = colorize(loc, BOLD)
            annotation = _format_outlier_annotation(m.length, boundaries)
            chars_str = _format_outlier_chars(m.length, ctx.color, ctx.line_length)
            print(f'      {loc} ({chars_str} chars{annotation})')
    return any_printed


def _print_recommendations(  # noqa: PLR0913
    impact: dict[int, ImpactRow],
    source_percentiles: Percentiles | None,
    *,
    line_length_source: LineLengthSource,
    configured_length: int,
    color: bool,
    leading_blank: bool = True,
) -> bool:
    """Print the recommendations section when applicable."""
    from docwrap._display._recommend import compute_recommendations  # noqa: PLC0415

    if not impact:
        return False

    source_p95 = source_percentiles.levels[95] if source_percentiles is not None else None

    recs = compute_recommendations(
        impact=impact,
        source_p95=source_p95,
        line_length_source=line_length_source,
        configured_length=configured_length,
    )
    if not recs:
        return False

    separator = '\n' if leading_blank else ''
    print(f'{separator}{bold_if("Recommended targets", color)}\n')

    for rec in recs:
        label_str = bold_if(f'{rec.label}:', color)
        target_str = colorize(str(rec.target), GREEN) if color else str(rec.target)
        print(f'  {label_str} {target_str}')
        print(f'    {rec.rationale}')
    return True


def _render_bar(
    bucket: Bucket,
    max_count: int,
    color: bool,
    trailing_pipe: bool = False,
) -> str:
    """Render a single histogram bar, optionally with zone colors."""
    if bucket.count == 0 or max_count == 0:
        return ''

    if not color or bucket.zone_counts is None:
        bar_len = max(1, round(bucket.count / max_count * _MAX_BAR_WIDTH))
        if trailing_pipe:
            return '#' * max(1, bar_len - 1) + '|'
        return '#' * bar_len

    zone_counts = bucket.zone_counts
    non_empty = [(i, c) for i, c in enumerate(zone_counts) if c > 0]
    if not non_empty:
        return ''

    n_zones = len(non_empty)
    n_seps = n_zones - 1
    total_width = max(1, round(bucket.count / max_count * _MAX_BAR_WIDTH))

    pipe_cost = 1 if trailing_pipe else 0
    hash_available = total_width - n_seps - pipe_cost
    if hash_available < n_zones:
        total_width = n_zones + n_seps + pipe_cost
        hash_available = n_zones

    zone_widths = _distribute_bar_widths(non_empty, hash_available)

    parts: list[str] = []
    for j, (zone_idx, _count) in enumerate(non_empty):
        if j > 0:
            parts.append('|')
        color_code = HISTOGRAM_SPECTRUM[zone_idx]
        parts.append(f'{color_code}{"#" * zone_widths[j]}{RESET}')

    if trailing_pipe:
        parts.append('|')

    return ''.join(parts)


def _format_histogram(
    histogram: Histogram,
    *,
    label: str,
    color: bool = False,
    source_percentiles: Percentiles | None = None,
) -> str:
    """Format a histogram as an ASCII bar chart."""
    if source_percentiles is not None:
        sp = source_percentiles
        parts = ', '.join(f'{_level_display_name(p)}={sp.levels[p]}' for p in PERCENTILE_LEVELS)
        subtitle = f'{label} (source: {parts})'
    else:
        subtitle = label

    rows = _trim_histogram_rows(_build_histogram_rows(histogram.buckets))
    if not rows:
        return ''

    buckets = tuple(row.bucket for row in rows)
    max_count = max(b.count for b in buckets)

    label_width = max(len(row.label) for row in rows)
    count_width = max(len(str(b.count)) for b in buckets)
    pct_width = max(len(row.percentage) for row in rows)

    count_width = max(count_width, len('Lines'))
    pct_width = max(pct_width, len('Lines%'))

    raw_labels, bucket_boundaries = _compute_bucket_labels(buckets, source_percentiles)
    final_labels, trailing_pipes = _place_boundaries(
        buckets,
        raw_labels,
        bucket_boundaries,
        color=color,
    )

    bar_col_width = _MAX_BAR_WIDTH

    hdr = (
        f'    {"Chars":>{label_width}}  '
        f'{"":>{bar_col_width}}  '
        f'{"Lines":>{count_width}}  '
        f'{"Lines%":>{pct_width}}'
    )

    lines: list[str] = [f'  {subtitle}:', bold_if(hdr, color)]

    for bi, row in enumerate(rows):
        bucket = row.bucket
        bar = _render_bar(bucket, max_count, color, trailing_pipe=trailing_pipes[bi])
        visible_len = bar.count('#') + bar.count('|')

        pad = ' ' * max(0, bar_col_width - visible_len)
        line = (
            f'    {row.label:>{label_width}}  {bar}{pad}'
            f'  {bucket.count:>{count_width}}  {row.percentage:>{pct_width}}'
        )
        label_str = ', '.join(final_labels[bi])
        if label_str:
            line += f'  {label_str}'
        lines.append(line)

    return '\n'.join(lines)


def _print_distributions(
    ctx: _StatsRenderCtx,
    *,
    leading_blank: bool = True,
) -> bool:
    """Print the distribution section when applicable."""
    source_pctls = ctx.source_percentiles
    boundaries = ctx.source_zone_boundaries

    by_cat = _group_by_category(
        [m for m in ctx.measurements if m.wrappable],
        WRAPPABLE_CATEGORIES,
    )

    all_wrappable = [m for cat_ms in by_cat.values() for m in cat_ms]
    if not all_wrappable:
        return False

    separator = '\n' if leading_blank else ''
    print(f'{separator}{bold_if("Distribution", ctx.color)}\n')

    for cat in WRAPPABLE_CATEGORIES:
        cat_ms = by_cat.get(cat, [])
        if not cat_ms:
            continue
        output = _format_histogram(
            build_histogram(cat_ms, boundaries=boundaries),
            label=cat,
            color=ctx.color,
            source_percentiles=source_pctls,
        )
        for line in output.split('\n'):
            print(f'  {line}')
        print()

    output = _format_histogram(
        build_histogram(all_wrappable, boundaries=boundaries),
        label='TOTAL',
        color=ctx.color,
        source_percentiles=source_pctls,
    )
    for line in output.split('\n'):
        print(f'  {line}')
    return True


def _print_stats_section(
    section: StatsSection,
    data: StatsData,
    ctx: _StatsRenderCtx,
    options: _StatsSectionOptions,
    *,
    leading_blank: bool,
) -> bool:
    """Print one selected stats body section."""
    if not options.wrappable and section not in ('source', 'docstrings'):
        return False

    match section:
        case 'source':
            rendered = _print_source_table(ctx, leading_blank=leading_blank)
        case 'docstrings':
            rendered = _print_docstring_tables(ctx, leading_blank=leading_blank)
        case 'distribution':
            rendered = _print_distributions(ctx, leading_blank=leading_blank)
        case 'longest':
            rendered = _print_outliers(
                ctx,
                limit=options.longest_per_category,
                leading_blank=leading_blank,
            )
        case 'impact':
            rendered = _print_impact_table(
                data.impact,
                ctx.source_percentiles,
                _ImpactTableOptions(
                    docstring_count=data.docstring_count,
                    docstring_file_count=data.docstring_file_count,
                    color=ctx.color,
                    line_length=ctx.line_length,
                ),
                leading_blank=leading_blank,
            )
        case 'recommendations':
            rendered = _print_recommendations(
                data.impact,
                ctx.source_percentiles,
                line_length_source=options.line_length_source,
                configured_length=options.configured_length,
                color=ctx.color,
                leading_blank=leading_blank,
            )
        case _:
            assert_never(section)
    return rendered


def print_stats(  # noqa: PLR0913
    data: StatsData,
    *,
    line_length: int,
    color: bool,
    line_length_source: LineLengthSource,
    sections: tuple[StatsSection, ...] = DEFAULT_STATS_SECTIONS,
    longest_per_category: int = DEFAULT_STATS_LONGEST_PER_CATEGORY,
) -> None:
    """Print stats output.

    Args:
        data: Collected stats data (includes impact).
        line_length: Resolved line length.
        color: Whether to apply ANSI color.
        line_length_source: Where line_length came from.
        sections: Selected body sections in display order.
        longest_per_category: Maximum longest-line rows per wrappable category.
    """
    ds = data.docstring_count
    ds_files = data.docstring_file_count
    ds_label = pluralize('docstring', ds)
    fs_label = pluralize('file', ds_files)

    # Header (always shown, at top)
    print(f'Scanned: {ds} {ds_label} across {ds_files} {fs_label}')
    _print_configured_length(line_length, line_length_source, color=color)
    print()

    # Zero docstrings early exit
    if ds == 0:
        print('No docstrings found.')
        if data.skipped_count > 0:
            _print_skipped_footer(data.skipped_count)
        if data.safety_skip_count > 0:
            _print_safety_skip_footer(data.safety_skip_count)
        return

    src_pctls, src_bounds = source_boundaries(data.source_measurements)

    ctx = _StatsRenderCtx(
        measurements=data.measurements,
        source_measurements=data.source_measurements,
        color=color,
        line_length=line_length,
        source_percentiles=src_pctls,
        source_zone_boundaries=src_bounds,
    )

    has_wrappable = any(m.wrappable for m in data.measurements)
    section_options = _StatsSectionOptions(
        configured_length=line_length,
        line_length_source=line_length_source,
        longest_per_category=longest_per_category,
        wrappable=has_wrappable,
    )

    printed_body = False
    for section in sections:
        rendered = _print_stats_section(
            section,
            data,
            ctx,
            section_options,
            leading_blank=printed_body,
        )
        printed_body = rendered or printed_body

    if not has_wrappable:
        separator = '\n' if printed_body else ''
        print(f'{separator}  No wrappable lines found.')

    if data.skipped_count > 0:
        _print_skipped_footer(data.skipped_count)
    if data.safety_skip_count > 0:
        _print_safety_skip_footer(data.safety_skip_count)
