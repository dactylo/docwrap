"""Line-length recommendation engine for stats output."""

from dataclasses import dataclass

from docwrap._display._text import pluralize
from docwrap._stats import STANDARD_LENGTHS, ImpactRow
from docwrap._types import LineLengthSource


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A line-length recommendation prepared for display.

    Attributes:
        label: Display label, combining labels when targets coincide.
        target: Recommended integer line width in characters.
        rationale: Human-readable reason for selecting this target.
    """

    label: str
    target: int
    rationale: str


def _format_rationale(
    prefix: str,
    row: ImpactRow,
) -> str:
    """Describe a target's projected docstring and line changes."""
    lines_added = row.lines_added
    lines_removed = row.lines_removed
    docs = row.docstrings_changed
    files = row.files_changed
    if docs == 0 and lines_added == 0 and lines_removed == 0:
        return f'{prefix} No changes needed.'
    if lines_added == 0 and docs > 0:
        ds = pluralize('docstring', docs)
        fs = pluralize('file', files)
        return f'{prefix} Reformats {docs} {ds} across {files} {fs}.'
    ds = pluralize('docstring', docs)
    fs = pluralize('file', files)
    if lines_removed > 0:
        return (
            f'{prefix} Adds {lines_added}, removes {lines_removed} lines'
            f' in {docs} {ds} across {files} {fs}.'
        )
    return f'{prefix} Adds {lines_added} lines in {docs} {ds} across {files} {fs}.'


def _best_fit(
    impact: dict[int, ImpactRow],
    source_p95: int,
) -> tuple[int, str] | None:
    """Find the standard length closest to source p95."""
    candidates = [length for length in STANDARD_LENGTHS if length in impact]
    if not candidates:
        return None

    best = min(candidates, key=lambda ln: (abs(ln - source_p95), ln))

    row = impact[best]
    rationale = _format_rationale('Closest target to p95.', row)
    return best, rationale


def _configured(
    impact: dict[int, ImpactRow],
    configured_length: int,
) -> tuple[int, str] | None:
    """Build a recommendation for the user's configured length."""
    if configured_length not in impact:
        return None
    row = impact[configured_length]
    rationale = _format_rationale('Your configured length.', row)
    return configured_length, rationale


def _low_churn(
    impact: dict[int, ImpactRow],
) -> tuple[int, str]:
    """Find the standard length with minimum total line churn."""
    candidates = [length for length in STANDARD_LENGTHS if length in impact]
    best = min(
        candidates,
        key=lambda ln: (
            impact[ln].lines_added + impact[ln].lines_removed,
            impact[ln].files_changed,
            impact[ln].docstrings_changed,
            ln,
        ),
    )
    row = impact[best]
    rationale = _format_rationale('', row)
    return best, rationale.lstrip()


def compute_recommendations(
    *,
    impact: dict[int, ImpactRow],
    source_p95: int | None,
    line_length_source: LineLengthSource,
    configured_length: int,
) -> list[Recommendation]:
    """Compute recommended line-length targets.

    Args:
        impact: Mapping of line length to `ImpactRow`, containing at least one
            target from `STANDARD_LENGTHS`.
        source_p95: Source code 95th percentile, or `None` if unavailable.
        line_length_source: Where the configured length came from.
        configured_length: The user's configured line length.

    Returns:
        Available recommendations in Best fit, Low churn, Configured order.
        Equal target lengths are combined, joining their labels and retaining
        the first recommendation's rationale.
    """
    raw: list[tuple[str, int, str]] = []

    # Best fit
    if source_p95 is not None:
        result = _best_fit(impact, source_p95)
        if result is not None:
            raw.append(('Best fit', result[0], result[1]))

    # Low churn
    lc_target, lc_rationale = _low_churn(impact)
    raw.append(('Low churn', lc_target, lc_rationale))

    # Configured
    if line_length_source in (LineLengthSource.CLI, LineLengthSource.DOCWRAP):
        result = _configured(impact, configured_length)
        if result is not None:
            raw.append(('Configured', result[0], result[1]))

    # Dedup by target, combine labels, highest-precedence rationale wins
    seen: dict[int, int] = {}
    deduped: list[Recommendation] = []
    for label, target, rationale in raw:
        if target in seen:
            idx = seen[target]
            old = deduped[idx]
            deduped[idx] = Recommendation(
                label=f'{old.label}, {label}',
                target=target,
                rationale=old.rationale,
            )
        else:
            seen[target] = len(deduped)
            deduped.append(Recommendation(label=label, target=target, rationale=rationale))

    return deduped
