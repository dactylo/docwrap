"""Per-docstring style auto-detection."""

from collections.abc import Collection

from docwrap._styles import get_style
from docwrap._styles._sphinx import FIELD_MARKER_RE
from docwrap._types import StyleName

_GOOGLE = get_style('google')
_NUMPY = get_style('numpy')


def _count_style_markers(
    lines: list[str],
    code_block_lines: set[int] | None,
    eligible_line_indices: Collection[int],
) -> dict[StyleName, int]:
    """Count style markers on eligible non-code lines."""
    counts: dict[StyleName, int] = {'google': 0, 'numpy': 0, 'sphinx': 0}

    for i, line in enumerate(lines):
        if i not in eligible_line_indices:
            continue
        if code_block_lines and i in code_block_lines:
            continue
        stripped = line.strip()

        # Sphinx: field list markers
        if FIELD_MARKER_RE.match(line):
            counts['sphinx'] += 1
            continue

        # NumPy: header followed by dashed underline (indent 0 only)
        # Pass raw line (not stripped) so is_section_header's indent-0 guard works.
        if stripped and not line[0].isspace():
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if _NUMPY.is_section_header(line, next_line) is not None:
                counts['numpy'] += 1
                continue

        # Google: header name followed by colon at indent 0
        if _GOOGLE.is_section_header(line) is not None:
            counts['google'] += 1

    return counts


def _resolve_style_counts(
    counts: dict[StyleName, int], *, preferred: StyleName | None = None
) -> StyleName | None:
    """Return the unique winner or the preferred style when it shares the lead."""
    highest = max(counts.values())
    if highest == 0:
        return None

    leaders: list[StyleName] = [style for style, count in counts.items() if count == highest]
    if len(leaders) == 1:
        return leaders[0]
    return preferred if preferred in leaders else None


def detect_style(
    content: str,
    code_block_lines: set[int] | None = None,
    *,
    primary_start: int = 0,
    fallback_line_indices: Collection[int] = (),
    preferred: StyleName | None = None,
) -> StyleName | None:
    """Detect the docstring style from primary evidence, then fallback.

    Args:
        content: Normalized docstring content.
        code_block_lines: Line indices to skip (code blocks).
        primary_start: First absolute line index eligible for primary evidence.
        fallback_line_indices: Absolute line indices eligible for fallback
            evidence when primary evidence has no markers.
        preferred: Style to select when it shares the highest marker count.

    Returns:
        The detected style name, or `None` if neither pass finds markers or the
        applicable tie-breaking rules leave no selected style.
    """
    lines = content.split('\n')
    primary_counts = _count_style_markers(
        lines,
        code_block_lines,
        range(primary_start, len(lines)),
    )
    if sum(primary_counts.values()) > 0:
        return _resolve_style_counts(primary_counts, preferred=preferred)

    fallback_counts = _count_style_markers(
        lines,
        code_block_lines,
        fallback_line_indices,
    )
    return _resolve_style_counts(fallback_counts, preferred=preferred)
