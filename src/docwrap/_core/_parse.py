"""Parse normalized docstring content into typed regions."""

import re
import string
from dataclasses import dataclass, replace
from typing import Literal

from docwrap._styles import EntryMatch, Style
from docwrap._types import CanonicalRole, RegionKind, SectionKind, SphinxFieldRole

LIST_RE = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')
_MIN_UNDERLINE_LENGTH = 3
_TITLE_ADORNMENT_CHARS = frozenset(string.punctuation)
_TITLE_LINE_COUNT_OVERLINED = 3
_TITLE_LINE_COUNT_UNDERLINED = 2
_UNDERLINE_CHARS = frozenset('-=~')


@dataclass(frozen=True, slots=True)
class Region:
    """A parsed region within a docstring.

    Args:
        kind: Semantic region classification.
        text: Normalized region text.
        entry_match: Parsed structured-entry fields, when applicable.
        indent: Number of leading whitespace characters, including tabs.
        role: Canonical or Sphinx field role, when applicable.
        section_name: Authored section header name, when applicable.
        source_line_count: Number of physical source lines represented.
        original_lines: Original physical lines, when retained.
        source_start: Zero-based start line in normalized content, when known.
        preserve_verbatim: Whether conversion must retain the authored text.
        source_scanner_protected: Whether source scanning protected the region.
    """

    kind: RegionKind
    text: str
    entry_match: EntryMatch | None = None
    indent: int = 0
    role: CanonicalRole | SphinxFieldRole | Literal[''] = ''
    section_name: str | None = None
    source_line_count: int = 1
    original_lines: tuple[str, ...] | None = None
    source_start: int | None = None
    preserve_verbatim: bool = False
    source_scanner_protected: bool = False


def _is_bare_list_marker(line: str) -> bool:
    """Return whether a line is a supported list marker without content."""
    return LIST_RE.fullmatch(f'{line} ') is not None


def _is_section_header(style: Style, line: str, lines: list[str], i: int) -> bool:
    """Check if a line is a section header using the style protocol."""
    next_line = lines[i + 1] if i + 1 < len(lines) else None
    return style.is_section_header(line, next_line) is not None


def _is_title_adornment(line: str) -> bool:
    """Return whether a line is a flush-left repeated punctuation adornment."""
    return (
        bool(line)
        and line == line.lstrip()
        and line[0] in _TITLE_ADORNMENT_CHARS
        and line == line[0] * len(line)
    )


def _parse_entry(
    lines: list[str],
    start: int,
    style: Style,
    code_block_lines: set[int],
    role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
) -> tuple[Region, int]:
    """Parse a section entry line with its continuation lines."""
    entry_lines = [lines[start]]
    entry_match = style.match_entry(lines[start], role=role)
    entry_prefix = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    cont_indent = style.entry_continuation_indent
    i = start + 1
    while i < len(lines):
        nline = lines[i]
        if not nline.strip() or i in code_block_lines:
            break
        line_indent = len(nline) - len(nline.lstrip())
        if style.uses_field_list_entries and style.match_entry(nline, role=role) is not None:
            line_prefix = nline[:line_indent]
            if line_prefix != entry_prefix and line_prefix.startswith(entry_prefix):
                entry_lines.append(nline)
                i += 1
                continue
            break
        if line_indent < cont_indent:
            break
        # List items at deeper indent are not continuations (req 12.4)
        if LIST_RE.match(nline):
            break
        # New entries in section-based styles remain separate entries.
        if not style.uses_field_list_entries and style.match_entry(nline, role=role) is not None:
            break
        entry_lines.append(nline)
        i += 1
    # Join full description from first-line match + continuation lines
    if entry_match is not None and len(entry_lines) > 1:
        full_desc_parts: list[str] = []
        if entry_match.description:
            full_desc_parts.append(entry_match.description)
        full_desc_parts.extend(cl.strip() for cl in entry_lines[1:])
        entry_match = replace(entry_match, description=' '.join(full_desc_parts))
    return Region(
        kind='entry',
        text='\n'.join(entry_lines),
        source_line_count=len(entry_lines),
        entry_match=entry_match,
        original_lines=tuple(entry_lines),
        source_start=start,
    ), i


def _consume_entry(  # noqa: PLR0913
    lines: list[str],
    i: int,
    style: Style,
    code_block_lines: set[int],
    cur_section: str | None,
    *,
    section_kind: SectionKind | None,
    regions: list[Region],
) -> int | None:
    """Try to consume a section or Sphinx entry at `lines[i]`."""
    line = lines[i]
    role: CanonicalRole | SphinxFieldRole | Literal[''] = (
        style.section_to_canonical.get(cur_section, '') if cur_section is not None else ''
    )
    em: EntryMatch | None = None
    if section_kind == 'entry' or style.uses_field_list_entries:
        em = style.match_entry(line, role=role)
    if em is None and not (
        style.supports_compound_type_entries and section_kind == 'entry' and line == line.lstrip()
    ):
        return None
    region, next_i = _parse_entry(
        lines,
        i,
        style,
        code_block_lines,
        role=role,
    )
    if region.entry_match is not None and region.entry_match.role:
        # Sphinx: role already set by match_entry
        region = replace(region, role=region.entry_match.role)
    elif role:
        # Google/NumPy: derive from section
        region = replace(region, role=role)
    regions.append(region)
    return next_i


def _parse_list_item(
    lines: list[str], start: int, code_block_lines: set[int]
) -> tuple[Region, int]:
    """Parse a list item starting at the given index."""
    m = LIST_RE.match(lines[start])
    if m is None:  # pragma: no cover - caller guarantees match
        return Region(
            kind='list_item',
            text=lines[start],
            original_lines=(lines[start],),
            source_start=start,
        ), start + 1
    list_lines = [lines[start]]
    cont_indent = len(m.group(0))
    parent_indent = len(lines[start]) - len(lines[start].lstrip())
    i = start + 1
    while i < len(lines) and lines[i].strip():
        if i in code_block_lines:
            break
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        # Sibling marker at parent indent level
        if LIST_RE.match(lines[i]) and line_indent <= parent_indent:
            break
        # Nested marker at continuation indent
        if LIST_RE.match(lines[i]) and line_indent >= cont_indent:
            break
        if line_indent >= cont_indent:
            list_lines.append(lines[i])
            i += 1
        else:
            break
    return Region(
        kind='list_item',
        text='\n'.join(list_lines),
        source_line_count=len(list_lines),
        original_lines=tuple(list_lines),
        source_start=start,
    ), i


def _peek_section_body_lines(
    lines: list[str],
    start: int,
    code_block_lines: set[int],
    *,
    max_count: int = 2,
) -> tuple[str, ...]:
    """Return up to max_count non-blank, non-code-block body lines starting at index `start`."""
    out: list[str] = []
    i = start
    while i < len(lines) and len(out) < max_count:
        if i not in code_block_lines and lines[i].strip():
            out.append(lines[i])
        i += 1
    return tuple(out)


def _consume_section_header(
    lines: list[str],
    i: int,
    style: Style,
    code_block_lines: set[int],
    regions: list[Region],
) -> tuple[int, str, SectionKind | None] | None:
    """Try to consume a section header at `lines[i]`."""
    next_line = lines[i + 1] if i + 1 < len(lines) else None
    section_name = style.is_section_header(lines[i], next_line)
    if section_name is None:
        return None
    regions.append(
        Region(
            kind='section_header',
            text=lines[i],
            section_name=section_name,
            original_lines=(lines[i],),
            source_start=i,
        )
    )
    next_i = i + 1
    for _ in range(1, style.section_header_line_count):
        if next_i >= len(lines):
            break
        regions.append(
            Region(
                kind='section_header',
                text=lines[next_i],
                section_name=section_name,
                original_lines=(lines[next_i],),
                source_start=next_i,
            )
        )
        next_i += 1
    # Classify section for entry vs prose dispatch
    peek = _peek_section_body_lines(lines, next_i, code_block_lines, max_count=4)
    section_kind = style.classify_section(section_name, peek)
    return next_i, section_name, section_kind


def _reset_section_after_dedent(
    lines: list[str],
    i: int,
    style: Style | None,
    cur_section: str | None,
    section_kind: SectionKind | None,
) -> tuple[str | None, SectionKind | None]:
    """Clear section state when a body line dedents after a blank line."""
    if (
        cur_section is None
        or style is None
        or not style.supports_section_headers
        or i == 0
        or lines[i - 1].strip()
    ):
        return cur_section, section_kind
    line_indent = len(lines[i]) - len(lines[i].lstrip())
    if line_indent < style.section_body_indent:
        return None, None
    return cur_section, section_kind


def is_structural_underline(line: str) -> bool:
    """Report whether a line is a section underline rather than prose.

    Args:
        line: A single physical line, with or without indentation.

    Returns:
        True when the stripped line is at least three repeated `-`, `=`, or `~`
        characters.
    """
    s = line.strip()
    return len(s) >= _MIN_UNDERLINE_LENGTH and len(set(s)) == 1 and s[0] in _UNDERLINE_CHARS


def _parse_paragraph(
    lines: list[str],
    start: int,
    style: Style | None,
    code_block_lines: set[int],
    *,
    in_section: bool,
) -> tuple[Region, int]:
    """Parse a prose/description paragraph starting at the given index."""
    line = lines[start]
    para_indent = len(line) - len(line.lstrip())
    para_lines = [line]
    if _is_bare_list_marker(line) or is_structural_underline(line):
        kind = 'prose' if in_section else 'description'
        region = Region(
            kind=kind,
            text=line.strip(),
            indent=para_indent,
            source_line_count=1,
            original_lines=(line,),
            source_start=start,
        )
        return region, start + 1
    i = start + 1
    while i < len(lines):
        nline = lines[i]
        if not nline.strip() or i in code_block_lines:
            break
        if style is not None and _is_section_header(style, nline, lines, i):
            break
        if _is_bare_list_marker(nline):
            break
        if LIST_RE.match(nline):
            break
        if (
            style is not None
            and style.uses_field_list_entries
            and style.match_entry(nline) is not None
        ):
            break
        if is_structural_underline(nline):
            break
        if (len(nline) - len(nline.lstrip())) != para_indent:
            break
        para_lines.append(nline)
        i += 1
    kind = 'prose' if in_section else 'description'
    region = Region(
        kind=kind,
        text=' '.join(s.strip() for s in para_lines),
        indent=para_indent,
        source_line_count=len(para_lines),
        original_lines=tuple(para_lines),
        source_start=start,
    )
    return region, i


def _parse_body(
    lines: list[str],
    start: int,
    style: Style | None,
    code_block_lines: set[int],
    regions: list[Region],
) -> None:
    """Parse the body of a docstring (everything after summary + blank)."""
    i = start
    cur_section: str | None = None
    section_kind: SectionKind | None = None
    while i < len(lines):
        line = lines[i]

        if i in code_block_lines:
            regions.append(
                Region(
                    kind='code_block',
                    text=line,
                    original_lines=(line,),
                    source_scanner_protected=True,
                    source_start=i,
                )
            )
            i += 1
            continue

        if not line.strip():
            regions.append(
                Region(
                    kind='blank',
                    text='',
                    original_lines=(line,),
                    source_start=i,
                )
            )
            i += 1
            continue

        cur_section, section_kind = _reset_section_after_dedent(
            lines, i, style, cur_section, section_kind
        )

        # Section header detection (delegated to style)
        if style is not None:
            header = _consume_section_header(lines, i, style, code_block_lines, regions)
            if header is not None:
                i, cur_section, section_kind = header
                continue

        if LIST_RE.match(line):
            region, i = _parse_list_item(lines, i, code_block_lines)
            regions.append(region)
            continue

        # Section entry or Sphinx field-list entry.
        if style is not None:
            next_i = _consume_entry(
                lines,
                i,
                style,
                code_block_lines,
                cur_section,
                section_kind=section_kind,
                regions=regions,
            )
            if next_i is not None:
                i = next_i
                continue

        # Body parsing, not the code-block scanner, owns indented description
        # blocks.
        line_indent = len(line) - len(line.lstrip())
        if line_indent > 0 and cur_section is None:
            regions.append(
                Region(
                    kind='code_block',
                    text=line,
                    original_lines=(line,),
                    source_start=i,
                )
            )
            i += 1
            continue

        # Regular prose/description
        region, i = _parse_paragraph(
            lines, i, style, code_block_lines, in_section=cur_section is not None
        )
        regions.append(region)


def is_summary_break(
    line: str,
    i: int,
    lines: list[str],
    style: Style | None,
    code_block_lines: set[int],
) -> bool:
    """Return whether a physical line should end the summary.

    Args:
        line: Current physical line to classify.
        i: Index of the current line in `lines`.
        lines: Complete physical lines for the candidate content.
        style: Active style grammar, or `None` for generic syntax.
        code_block_lines: Physical line indexes owned by code blocks.

    Returns:
        `True` when the line starts parser-visible non-summary structure.
    """
    if not line.strip():
        return True
    if i in code_block_lines:
        return True
    if line != line.lstrip():
        return len(lines) != 1 or line.startswith(('    ', '\t'))
    if LIST_RE.match(line):
        return True
    if style is not None and _is_section_header(style, line, lines, i):
        return True
    return (
        style is not None and style.uses_field_list_entries and style.match_entry(line) is not None
    )


def match_opening_title(lines: list[str], style: Style | None) -> int | None:
    """Return the physical-line count of a complete opening title.

    Args:
        lines: Normalized docstring content lines.
        style: Resolved active style, or None for syntax-only matching.

    Returns:
        Two for an underline-only title, three for a matching overlined title,
        or None when the opening lines are not a title.
    """
    if (
        len(lines) >= _TITLE_LINE_COUNT_OVERLINED
        and _is_title_adornment(lines[0])
        and lines[2] == lines[0]
        and lines[1].strip()
        and len(lines[0]) >= len(lines[1].strip())
    ):
        return _TITLE_LINE_COUNT_OVERLINED

    if len(lines) < _TITLE_LINE_COUNT_UNDERLINED:
        return None
    title = lines[0]
    underline = lines[1]
    if (
        not title.strip()
        or title != title.lstrip()
        or not _is_title_adornment(underline)
        or len(underline) < len(title.strip())
    ):
        return None
    if style is not None and _is_section_header(style, title, lines, 0):
        return None
    return _TITLE_LINE_COUNT_UNDERLINED


def parse_regions(
    content: str,
    style: Style | None,
    code_block_lines: set[int],
) -> list[Region]:
    """Parse normalized content into typed regions.

    Args:
        content: Normalized docstring content.
        style: A Style instance or None for style-agnostic parsing.
        code_block_lines: Line indices to treat as code blocks.

    Returns:
        List of Region objects.
    """
    if not content.strip():
        return []

    lines = content.split('\n')
    regions: list[Region] = []
    title_line_count = match_opening_title(lines, style)
    if title_line_count is not None:
        title_lines = lines[:title_line_count]
        regions.append(
            Region(
                kind='title',
                text='\n'.join(title_lines),
                source_line_count=title_line_count,
                original_lines=tuple(title_lines),
                source_start=0,
            )
        )
        _parse_body(lines, title_line_count, style, code_block_lines, regions)
        return regions

    i = 0
    summary_lines: list[str] = []

    # Parse summary (first paragraph -- lines at indent 0 only)
    while i < len(lines):
        if is_summary_break(lines[i], i, lines, style, code_block_lines):
            break
        summary_lines.append(lines[i])
        i += 1

    if summary_lines:
        regions.append(
            Region(
                kind='summary',
                text=' '.join(s.strip() for s in summary_lines),
                source_line_count=len(summary_lines),
                original_lines=tuple(summary_lines),
                source_start=0,
            )
        )

    if i >= len(lines):
        return regions

    # D205 enforcement: exactly one blank line between summary and rest.
    blank_start = i
    blank_count = 0
    while i < len(lines) and not lines[i].strip():
        blank_count += 1
        i += 1

    if summary_lines and i < len(lines):
        regions.append(
            Region(
                kind='blank',
                text='',
                source_line_count=blank_count,
                original_lines=tuple(lines[blank_start:i]) if blank_count else None,
                source_start=blank_start if blank_count else None,
            )
        )

    _parse_body(lines, i, style, code_block_lines, regions)
    return regions
