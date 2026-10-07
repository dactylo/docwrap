"""Line length statistics for docstring analysis."""

import heapq
from dataclasses import dataclass, replace
from typing import NamedTuple

from docwrap._core import (
    ParsedDocstring,
    Region,
    simulate_wrap_delta,
    wrap_parsed_docstring,
)
from docwrap._core._wrap import VERBATIM_REGION_KINDS
from docwrap._types import (
    DocwrapConfig,
    MeasurementCategory,
    RegionKind,
)

BUCKET_BOUNDS = (40, 50, 60, 70, 80, 90, 100)
PERCENTILE_LEVELS = (50, 75, 90, 95, 99)
STANDARD_LENGTHS = (72, 79, 80, 88, 99, 100)
WRAPPABLE_CATEGORIES = ('prose', 'args', 'summary')
_ARGS_KINDS = frozenset({'entry'})
_CODE_BLOCK_KINDS = frozenset({'code_block'})
_DEFAULT_CFG = DocwrapConfig()
_PROSE_KINDS = frozenset({'description', 'list_item', 'prose'})
_SUMMARY_KINDS = frozenset({'summary'})
_WIDTH_INVARIANT_KINDS = VERBATIM_REGION_KINDS | frozenset({'blank'})
_WIDTH_INVARIANT_KINDS_NO_WRAP = _WIDTH_INVARIANT_KINDS | frozenset({'summary'})


@dataclass(frozen=True, slots=True)
class Bucket:
    """A single histogram bucket.

    Attributes:
        upper: Upper bound of the bucket, or 0 for the overflow bucket.
        count: Number of lines falling into this bucket.
        zone_counts: One count per boundary zone plus overflow, or None if no
            zone boundaries were provided.
        is_overflow: Whether this is the overflow (>N) bucket.
    """

    upper: int  # 0 for overflow buckets
    count: int
    zone_counts: tuple[int, ...] | None
    is_overflow: bool


@dataclass(frozen=True, slots=True)
class Histogram:
    """Line length histogram.

    Attributes:
        buckets: Tuple of buckets from lowest to highest bound, plus overflow.
        total: Total number of lines measured.
    """

    buckets: tuple[Bucket, ...]
    total: int


@dataclass(frozen=True, slots=True)
class ImpactRow:
    """Impact of wrapping at a specific line length.

    Attributes:
        line_length: The line length threshold.
        docstrings_changed: Number of docstrings that would change.
        files_changed: Always 0 or 1 for single-source; aggregated by CLI.
        lines_added: Lines added by wrapping (non-negative).
        lines_removed: Lines removed by reflowing (non-negative).
    """

    line_length: int
    docstrings_changed: int
    files_changed: int
    lines_added: int
    lines_removed: int


@dataclass(frozen=True, slots=True)
class LineMeasurement:
    """A single measured line from a docstring.

    Attributes:
        length: Absolute line length including indentation.
        category: The measurement category classification for this line.
        file_path: Path to the source file.
        line_number: 1-based line number in the source file.
        text: Measured text content of the line.
        wrappable: Whether the line is wrappable under its file's effective
            configuration.
    """

    length: int
    category: MeasurementCategory
    file_path: str
    line_number: int
    text: str
    wrappable: bool


@dataclass(frozen=True, slots=True)
class Percentiles:
    """Percentile statistics computed from line measurements.

    Attributes:
        levels: Mapping from percentile level to value. Keys match
            PERCENTILE_LEVELS (e.g. {50: 37, 75: 53, 90: 68, 95: 80, 99: 95}).
        max: Absolute maximum value (not a percentile).
    """

    levels: dict[int, int]
    max: int


@dataclass(frozen=True, slots=True)
class StatsData:
    """Aggregated statistics from CLI file scanning.

    Attributes:
        docstring_count: Total number of docstrings found.
        docstring_file_count: Number of files containing docstrings.
        impact: Mapping of line length to ImpactRow.
        measurements: All docstring line measurements.
        skipped_count: Number of unreadable files skipped.
        source_measurements: All source code line measurements.
        safety_skip_count: Number of docstrings skipped for safety.
        unparseable_count: Number of files omitted because they could not be
            parsed.
    """

    docstring_count: int
    docstring_file_count: int
    impact: dict[int, ImpactRow]
    measurements: list[LineMeasurement]
    skipped_count: int
    source_measurements: list[LineMeasurement]
    safety_skip_count: int = 0
    unparseable_count: int = 0


class ZoneBoundaries(NamedTuple):
    """Percentile boundaries for line-length zone classification.

    Attributes:
        median: Median line length.
        p75: 75th-percentile line length.
        p90: 90th-percentile line length.
        p95: 95th-percentile line length.
        p99: 99th-percentile line length.
    """

    median: int
    p75: int
    p90: int
    p95: int
    p99: int


def _bucket_bound(length: int) -> int:
    """Return the bucket bound for a line length."""
    for bound in BUCKET_BOUNDS:
        if length <= bound:
            return bound
    return 0


def _categorize_region(kind: RegionKind) -> MeasurementCategory | None:
    """Map a region kind to a stats category."""
    if kind in _SUMMARY_KINDS:
        return 'summary'
    if kind in _PROSE_KINDS:
        return 'prose'
    if kind in _ARGS_KINDS:
        return 'args'
    if kind in _CODE_BLOCK_KINDS:
        return 'code_block'
    return None


def _is_width_invariant(regions: tuple[Region, ...], *, wrap_summary: bool) -> bool:
    """Check if regions produce identical wrap_regions output at any width."""
    invariant = _WIDTH_INVARIANT_KINDS if wrap_summary else _WIDTH_INVARIANT_KINDS_NO_WRAP
    return all(r.kind in invariant for r in regions)


def _wrappable_categories(*, wrap_summary: bool) -> tuple[str, ...]:
    """Return ordered wrappable categories for the selected summary policy."""
    if wrap_summary:
        return WRAPPABLE_CATEGORIES
    return WRAPPABLE_CATEGORIES[:-1]


def _measure_from_source_lines(
    parsed: ParsedDocstring,
    file_path: str,
    wrap_summary: bool,
) -> list[LineMeasurement]:
    """Measure exact authored source lines using regions for categorization."""
    measurements: list[LineMeasurement] = []
    for region in parsed.regions:
        cat = _categorize_region(region.kind)
        if cat is None or region.source_start is None or region.original_lines is None:
            continue
        for offset, original_line in enumerate(region.original_lines):
            if not original_line.strip():
                continue
            source_index = parsed.content_start_offset + region.source_start + offset
            if source_index < len(parsed.source_lines):
                source_line = parsed.source_lines[source_index]
                measurements.append(
                    LineMeasurement(
                        length=len(source_line),
                        category=cat,
                        file_path=file_path,
                        line_number=parsed.info.lineno + source_index,
                        text=source_line,
                        wrappable=cat in _wrappable_categories(wrap_summary=wrap_summary),
                    )
                )
    return measurements


def _zone_index(char_count: int, boundaries: ZoneBoundaries) -> int:
    """Return the boundary zone index for a character count."""
    for i, bound in enumerate(boundaries):
        if char_count <= bound:
            return i
    return len(boundaries)


def allocate_percentages(counts: list[int]) -> list[int | None]:
    """Allocate integer percentages with stable largest-remainder rounding.

    Returns:
        One allocation per count: zero for empty buckets, None for nonempty
        buckets below one percent, and integers for participating buckets.
        Participating buckets are rescaled to sum to 100, with ties favoring
        the lower input index.
    """
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)

    result: list[int | None] = [None] * len(counts)
    exact = [c / total * 100 for c in counts]

    # Identify <1% and 0% entries
    participating: list[int] = []  # indices that participate in rounding
    for i, (count, pct) in enumerate(zip(counts, exact, strict=True)):
        if count == 0:
            result[i] = 0
        elif pct < 1.0:
            result[i] = None
        else:
            participating.append(i)

    if not participating:
        return result

    # Recompute percentages over participating buckets only
    participating_total = sum(counts[i] for i in participating)
    participating_exact = [counts[i] / participating_total * 100 for i in participating]
    floors = [int(p) for p in participating_exact]
    remainder = 100 - sum(floors)

    # Distribute remainder to largest fractional parts (tie-break: lower index)
    fractionals = [(participating_exact[j] - floors[j], j) for j in range(len(participating))]
    fractionals.sort(key=lambda x: (-x[0], x[1]))
    for k in range(remainder):
        floors[fractionals[k][1]] += 1

    for j, idx in enumerate(participating):
        result[idx] = floors[j]

    return result


def build_histogram(
    measurements: list[LineMeasurement],
    boundaries: ZoneBoundaries | None = None,
) -> Histogram:
    """Build a histogram from line measurements.

    Lines are placed into decade buckets (1-40, 41-50, ..., 91-100) with an
    overflow bucket for lines exceeding 100 characters.

    Args:
        measurements: List of line measurements to histogram.
        boundaries: Optional zone boundaries (median, p75, p90, p95, p99) for
            zone classification. When provided, each bucket tracks per-zone
            line counts.

    Returns:
        Histogram with numeric bounds, counts, and optional zone counts.
    """
    counts: dict[int, int] = dict.fromkeys(BUCKET_BOUNDS, 0)
    counts[0] = 0  # overflow bucket

    # Track each boundary zone plus overflow for every bucket.
    zone_tracking: dict[int, list[int]] | None = None
    if boundaries is not None:
        zone_tracking = {b: [0] * (len(boundaries) + 1) for b in BUCKET_BOUNDS}
        zone_tracking[0] = [0] * (len(boundaries) + 1)

    for m in measurements:
        bound = _bucket_bound(m.length)
        counts[bound] += 1
        if zone_tracking is not None and boundaries is not None:
            zone_tracking[bound][_zone_index(m.length, boundaries)] += 1

    total = len(measurements)
    buckets: list[Bucket] = []
    all_bounds = [*BUCKET_BOUNDS, 0]
    for bound in all_bounds:
        zc: tuple[int, ...] | None = None
        if zone_tracking is not None:
            zc = tuple(zone_tracking[bound])
        buckets.append(
            Bucket(
                upper=bound,
                count=counts[bound],
                zone_counts=zc,
                is_overflow=bound == 0,
            )
        )

    return Histogram(buckets=tuple(buckets), total=total)


def compute_impact(
    parsed: list[ParsedDocstring],
    config: DocwrapConfig = _DEFAULT_CFG,
    *,
    extra_lengths: tuple[int, ...] = (),
) -> list[ImpactRow]:
    """Compute wrapping impact at standard line lengths for pre-parsed docstrings.

    Pre-classifies width-invariant docstrings (no fill-dependent regions) and
    caches their wrap_regions output, reusing it across all standard lengths to
    avoid redundant wrapping calls.

    Args:
        parsed: Pre-parsed docstrings from parse_docstrings.
        config: Complete effective configuration for wrapping.
        extra_lengths: Additional line lengths to include beyond the standard
            set.

    Returns:
        List of ImpactRow, one per line length (sorted ascending).
    """
    transforms_active = (
        config.summary_start != _DEFAULT_CFG.summary_start
        or config.strip_types != _DEFAULT_CFG.strip_types
        or config.convert is not None
    )

    # Pre-classify: cache wrapped output for width-invariant docstrings.
    pre_wrapped_cache: dict[int, str] = {}
    for i, pd in enumerate(parsed):
        if not transforms_active and _is_width_invariant(
            pd.regions,
            wrap_summary=config.wrap_summary,
        ):
            # Use any width (72) since output is width-independent.
            cfg = replace(config, line_length=72)
            pre_wrapped_cache[i] = simulate_wrap_delta(pd, cfg)

    all_lengths = sorted(set(STANDARD_LENGTHS) | set(extra_lengths))
    cfgs = {length: replace(config, line_length=length) for length in all_lengths}
    rows: list[ImpactRow] = []
    for length in all_lengths:
        cfg = cfgs[length]
        docstrings_changed = 0
        lines_added = 0
        lines_removed = 0
        for i, pd in enumerate(parsed):
            cached = pre_wrapped_cache.get(i)
            result = wrap_parsed_docstring(pd, cfg, style_override=None, pre_wrapped=cached)
            if result.new_literal is not None:
                docstrings_changed += 1
                delta = result.new_literal.count('\n') - pd.lit.raw_literal.count('\n')
                if delta > 0:
                    lines_added += delta
                elif delta < 0:
                    lines_removed += abs(delta)
        rows.append(
            ImpactRow(
                line_length=length,
                docstrings_changed=docstrings_changed,
                files_changed=1 if docstrings_changed > 0 else 0,
                lines_added=lines_added,
                lines_removed=lines_removed,
            )
        )
    return rows


def compute_percentiles(measurements: list[LineMeasurement]) -> Percentiles | None:
    """Compute percentile statistics from line measurements.

    Uses nearest-rank method with integer-only math to avoid float imprecision.

    Args:
        measurements: List of line measurements.

    Returns:
        Percentiles dataclass, or None if empty.
    """
    if not measurements:
        return None
    lengths = sorted(m.length for m in measurements)
    n = len(lengths)

    def _rank(p: int) -> int:
        return max(0, (p * n + 99) // 100 - 1)

    return Percentiles(
        levels={p: lengths[_rank(p)] for p in PERCENTILE_LEVELS},
        max=lengths[-1],
    )


def find_outliers(measurements: list[LineMeasurement], *, n: int = 3) -> list[LineMeasurement]:
    """Return the N longest lines, sorted descending by length.

    Args:
        measurements: List of line measurements to search.
        n: Maximum number of outliers to return.

    Returns:
        Up to `n` measurements ordered from longest to shortest.
    """
    return heapq.nlargest(n, measurements, key=lambda m: m.length)


def measure_docstring(
    parsed: list[ParsedDocstring],
    *,
    file_path: str = '',
    wrap_summary: bool = False,
) -> list[LineMeasurement]:
    """Measure line lengths for pre-parsed docstrings.

    Args:
        parsed: Pre-parsed docstrings from parse_docstrings.
        file_path: Path to the source file (for reporting).
        wrap_summary: Whether summary lines are wrappable.

    Returns:
        List of LineMeasurement for each measurable line.
    """
    all_measurements: list[LineMeasurement] = []

    for pd in parsed:
        measurements = _measure_from_source_lines(
            pd,
            file_path,
            wrap_summary,
        )
        all_measurements.extend(measurements)

    return all_measurements


def measure_source_lines(
    source: str,
    docstring_ranges: list[tuple[int, int]],
    *,
    file_path: str = '',
) -> list[LineMeasurement]:
    """Measure line lengths for non-docstring source code.

    Excludes blank lines, comment-only lines, and lines inside docstrings.

    Args:
        source: Python source code string.
        docstring_ranges: List of (lineno, end_lineno) pairs identifying
            docstring line ranges to exclude.
        file_path: Path to the source file (for reporting).

    Returns:
        List of LineMeasurement for each measurable source line.
    """
    if not source.strip():
        return []

    docstring_lines: set[int] = set()
    for start, end in docstring_ranges:
        docstring_lines.update(range(start, end + 1))

    measurements: list[LineMeasurement] = []
    for i, line in enumerate(source.split('\n'), start=1):
        if i in docstring_lines:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        measurements.append(
            LineMeasurement(
                length=len(line),
                category='source',
                file_path=file_path,
                line_number=i,
                text=stripped,
                wrappable=False,
            )
        )

    return measurements


def source_boundaries(
    measurements: list[LineMeasurement] | None,
) -> tuple[Percentiles | None, ZoneBoundaries | None]:
    """Compute source percentiles and zone boundaries.

    Args:
        measurements: Optional source line measurements.

    Returns:
        Tuple of (source_percentiles, boundaries).
    """
    if not measurements:
        return None, None
    pctls = compute_percentiles(measurements)
    if pctls is None:
        return None, None
    return pctls, ZoneBoundaries(*(pctls.levels[p] for p in PERCENTILE_LEVELS))
