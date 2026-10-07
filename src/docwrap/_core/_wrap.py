"""Wrap parsed docstring regions into source text."""

import re
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from docwrap._core._codeblock import find_code_block_lines
from docwrap._core._normalize import TRIPLE_QUOTE_LEN, resolve_summary_position
from docwrap._core._parse import (
    LIST_RE,
    Region,
    is_structural_underline,
    is_summary_break,
    match_opening_title,
)
from docwrap._styles import EntryMatch, Style, get_style, get_styles
from docwrap._styles._sphinx import FIELD_MARKER_RE
from docwrap._types import (
    CanonicalRole,
    DocstringChange,
    DocstringSkip,
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    SphinxFieldRole,
    StyleName,
)

if TYPE_CHECKING:
    from docwrap._core._discover import ParsedDocstring

VERBATIM_REGION_KINDS = frozenset({'code_block', 'section_header', 'title'})
_ASCII_WHITESPACE = ' \t'
_BALANCED_CANDIDATE_BUDGET = 100_000
_BODY_SKIP_KINDS = frozenset({'summary', 'blank'})
_CAUTIOUS_REFLOW_MAX_ITERATIONS = 10
_CLI_LONG_OPTION_RE = re.compile(r'^--[A-Za-z0-9][A-Za-z0-9_-]*(?:[=.,)]|\s|$)')
_FINAL_LINE_SLACK_WEIGHT = 1
_INLINE_TOKEN_RE = re.compile(r'^(?:`{1,2}(?!`)|:[A-Za-z][\w-]*:(?:`|[,.)]))')
_LEADING_MARKUP_RE = re.compile(r'^[`\[\{<"\'*_]+')
_LINE_MARKER_RE = re.compile(r'^(?:[-*+]\s+|\d+[.)]\s+|\((?:[A-Za-z]|\d+)\)\s+|[A-Za-z][.)]\s+)')
_NONFINAL_LINE_SLACK_WEIGHT = 4
_SPHINX_FALLBACK_HANGING = 4
_TERMINAL_RE = re.compile(r'[.!?;:/]$')
_TEXTWRAP_WHITESPACE_TRANS = str.maketrans(dict.fromkeys('\t\n\x0b\x0c\r', ' '))
_TRAILING_MARKUP_RE = re.compile(r'[\)\]\}\>"`\'*_\s]+$')

type _PlanValidator = Callable[[list[str], _FillPlan], tuple[int, ...]]


@dataclass(slots=True)
class WrapState:
    """Mutable state accumulated while wrapping docstrings in one source.

    Attributes:
        changes: Per-docstring change records accumulated during wrapping.
        detected_styles: Style names detected in processed docstrings.
        converted_count: Number of docstrings converted between styles.
        wrapped: Number of docstrings whose source changed.
        skipped_summaries: Number of overlong summaries left unwrapped.
        skips: Safety records for docstrings left unchanged.
    """

    changes: list[DocstringChange]
    detected_styles: set[StyleName]
    converted_count: int = 0
    wrapped: int = 0
    skipped_summaries: int = 0
    skips: list[DocstringSkip] = field(default_factory=list[DocstringSkip])


@dataclass(frozen=True, slots=True)
class _Chunk:
    """One logical line with its raw form and split parameters."""

    raw: str
    body: str
    initial: str
    subsequent: str


@dataclass(frozen=True, slots=True)
class _FillOptions:
    """Configuration shared by internal fill and reflow helpers."""

    fill_style: FillStyle
    smart_slack: int


@dataclass(frozen=True, slots=True)
class _FillPlan:
    """Exclusive word indexes ending each planned output line."""

    ends: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _FillWords:
    """Normalized words, gaps, and marker-binding state for one fill."""

    gaps: tuple[str, ...]
    leading: str
    marker_bound: bool
    normalized: str
    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RenderOptions:
    """Scalar layout inputs for provenance-aware region rendering."""

    base_indent: int
    style: Style | None = None
    quote_overhead: int = 3
    col_offset: int | None = None


@dataclass(frozen=True, slots=True)
class _RenderedLine:
    """One emitted physical line and its planned region identity."""

    text: str
    region_index: int


@dataclass(frozen=True, slots=True)
class _SummaryOptions:
    """Scalar layout inputs for marker-aware summary rendering."""

    fill: _FillOptions
    quote_overhead: int
    width: int


@dataclass(frozen=True, slots=True)
class _WrapCtx:
    """Shared per-call wrapping context passed to region dispatch handlers."""

    config: DocwrapConfig
    style: Style | None
    width: int
    quote_overhead: int
    section_body_indent: int


def _candidate_crosses_preceding_boundary(
    regions: list[Region],
    start: int,
    candidate_delta: int,
    style: Style | None,
) -> bool:
    """Check list and entry continuation thresholds before a prose run."""
    if start == 0:
        return False
    previous = regions[start - 1]
    threshold: int | None = None
    if previous.kind == 'list_item':
        previous_line = (
            previous.original_lines[0]
            if previous.original_lines
            else previous.text.partition('\n')[0]
        )
        match = LIST_RE.match(previous_line)
        if match is not None:
            threshold = len(match.group(0))
    elif previous.kind == 'entry' and style is not None:
        threshold = style.entry_continuation_indent
    if threshold is None:
        return False
    authored_indent = regions[start].indent
    return authored_indent < threshold <= authored_indent + candidate_delta


def _containing_role(
    regions: list[Region],
    index: int,
    style: Style,
) -> CanonicalRole | SphinxFieldRole | Literal['']:
    """Return a region's role or its nearest active section role."""
    if regions[index].role:
        return regions[index].role
    for previous in reversed(regions[:index]):
        if previous.kind == 'section_header' and previous.section_name is not None:
            return style.section_to_canonical.get(previous.section_name, '')
    return ''


def _delimiter_pair(word: str) -> tuple[str, str] | None:
    """Return the delimiter pair for a smart-fill opener word."""
    if word.startswith('('):
        return '(', ')'
    if word.startswith('['):
        return '[', ']'
    return None


def _effective_column(
    base_indent: int,
    col_offset: int | None,
    quote_overhead: int,
) -> int:
    """Return the physical column used for docstring content."""
    if quote_overhead == 0:
        return base_indent
    if col_offset is not None:
        return col_offset
    return base_indent


def _entry_chunks(
    original_lines: tuple[str, ...],
    style: Style,
    role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
) -> list[_Chunk] | None:
    """Decompose entry original_lines into Chunks."""
    cont = ' ' * style.entry_continuation_indent
    first = original_lines[0]
    first_em = style.match_entry(first, role=role)
    if first_em is None:
        return None
    chunks: list[_Chunk] = []
    if first_em.description:
        if first_em.description_start is not None:
            desc_start = first_em.description_start
            marker = first[:desc_start]
            chunks.append(
                _Chunk(
                    raw=first,
                    body=first[desc_start:].strip(_ASCII_WHITESPACE),
                    initial=marker,
                    subsequent=cont,
                )
            )
        else:
            indent = first[: len(first) - len(first.lstrip())]
            chunks.append(
                _Chunk(
                    raw=first,
                    body=first.strip(_ASCII_WHITESPACE),
                    initial=indent,
                    subsequent=cont,
                )
            )
    else:
        return None
    chunks.extend(
        _Chunk(raw=line, body=line.strip(_ASCII_WHITESPACE), initial=cont, subsequent=cont)
        for line in original_lines[1:]
    )
    return chunks


def _fill_words_text(words: tuple[str, ...], gaps: tuple[str, ...]) -> str:
    """Reconstruct exact text from alternating words and gaps."""
    if not words:
        return ''
    parts = [words[0]]
    for index, word in enumerate(words[1:]):
        parts.extend((gaps[index], word))
    return ''.join(parts)


def _bind_plan_boundaries(
    fill_words: _FillWords,
    boundaries: tuple[int, ...],
) -> _FillWords:
    """Bind requested plan boundaries into their preceding words."""
    gaps = list(fill_words.gaps)
    words = list(fill_words.words)
    for end in sorted(boundaries, reverse=True):
        words[end - 1] += gaps[end - 1] + words[end]
        del words[end]
        del gaps[end - 1]
    bound_gaps = tuple(gaps)
    bound_words = tuple(words)
    return replace(
        fill_words,
        gaps=bound_gaps,
        normalized=_fill_words_text(bound_words, bound_gaps),
        words=bound_words,
    )


def _greedy_end(
    words: tuple[str, ...],
    gaps: tuple[str, ...],
    start: int,
    capacity: int,
) -> int:
    """Return the word index after greedily packing one line."""
    end = start + 1
    content_length = len(words[start])
    while end < len(words):
        next_length = content_length + len(gaps[end - 1]) + len(words[end])
        if next_length > capacity:
            break
        content_length = next_length
        end += 1
    return end


def _greedy_plan(
    fill_words: _FillWords,
    width: int,
    *,
    initial: str,
    subsequent: str,
) -> _FillPlan:
    """Plan greedy line endings without breaking marker-bound words."""
    ends: list[int] = []
    start = 0
    while start < len(fill_words.words):
        indent = initial if not ends else subsequent
        end = _greedy_end(
            fill_words.words,
            fill_words.gaps,
            start,
            width - len(indent),
        )
        ends.append(end)
        start = end
    return _FillPlan(ends=tuple(ends))


def _balanced_plan(
    fill_words: _FillWords,
    width: int,
    *,
    initial: str,
    subsequent: str,
) -> _FillPlan:
    """Plan line endings by line count, then weighted squared slack."""
    word_count = len(fill_words.words)
    candidate_count = 0
    scores: list[tuple[int, int] | None] = [None] * (word_count + 1)
    next_break: list[int] = [word_count] * word_count
    scores[word_count] = (0, 0)

    for start in range(word_count - 1, -1, -1):
        indent = initial if start == 0 else subsequent
        capacity = width - len(indent)
        content_length = 0

        for end in range(start, word_count):
            candidate_count += 1
            if candidate_count > _BALANCED_CANDIDATE_BUDGET:
                return _greedy_plan(
                    fill_words,
                    width,
                    initial=initial,
                    subsequent=subsequent,
                )
            if end == start:
                content_length = len(fill_words.words[end])
            else:
                content_length += len(fill_words.gaps[end - 1]) + len(fill_words.words[end])

            if end > start and content_length > capacity:
                break

            remainder = scores[end + 1]
            if remainder is None:
                continue

            slack = max(capacity - content_length, 0)
            weight = (
                _FINAL_LINE_SLACK_WEIGHT if end + 1 == word_count else _NONFINAL_LINE_SLACK_WEIGHT
            )
            candidate = (
                remainder[0] + 1,
                remainder[1] + weight * slack**2,
            )

            current = scores[start]
            if current is None or candidate < current:
                scores[start] = candidate
                next_break[start] = end + 1

    ends: list[int] = []
    start = 0
    while start < word_count:
        end = next_break[start]
        ends.append(end)
        start = end

    return _FillPlan(ends=tuple(ends))


def _has_body_regions(regions: list[Region]) -> bool:
    """Check whether regions contain content beyond summary and blanks."""
    return any(r.kind not in _BODY_SKIP_KINDS and r.text.strip() for r in regions)


def _is_bare_summary_marker(line: str) -> bool:
    """Return whether a line is an exact flush-left bare list marker."""
    return line == line.strip() and LIST_RE.fullmatch(f'{line} ') is not None


def _is_conservative_entry_candidate(line: str, ctx: _WrapCtx) -> bool:
    """Ask the style whether a line at entry indent could become an entry."""
    if ctx.style is None:
        return False
    indent = len(line) - len(line.lstrip())
    if indent != ctx.section_body_indent:
        return False
    return ctx.style.is_conservative_entry_candidate(line[indent:])


def _is_semantically_joinable(line_a: str, line_b: str) -> bool:
    """Check if two consecutive lines may be joined under cautious rules."""
    a = line_a.rstrip()
    b_raw = line_b.lstrip()
    if not a or not b_raw:
        return False

    if is_structural_underline(a) or is_structural_underline(b_raw):
        return False

    if _LINE_MARKER_RE.match(b_raw):
        return False

    a_stripped = _TRAILING_MARKUP_RE.sub('', a)
    if _TERMINAL_RE.search(a_stripped):
        return False

    b_stripped = _LEADING_MARKUP_RE.sub('', b_raw)
    return bool(
        (b_stripped and b_stripped[0].isalpha())
        or _INLINE_TOKEN_RE.match(b_raw)
        or _CLI_LONG_OPTION_RE.match(b_raw)
        or b_raw.startswith('(')
        or (b_raw[0] in {'"', "'"} and b_stripped and b_stripped[0].isdigit())
    )


def _list_chunks(original_lines: tuple[str, ...]) -> tuple[list[_Chunk], str, str]:
    """Decompose list item original_lines into Chunks."""
    m = LIST_RE.match(original_lines[0])
    if m is None:
        return (
            [
                _Chunk(
                    raw=ln,
                    body=ln.strip(_ASCII_WHITESPACE),
                    initial='',
                    subsequent='',
                )
                for ln in original_lines
            ],
            '',
            '',
        )
    prefix = m.group(0)
    hanging = ' ' * len(prefix)
    chunks = [
        _Chunk(
            raw=original_lines[0],
            body=original_lines[0][len(prefix) :].strip(_ASCII_WHITESPACE),
            initial=prefix,
            subsequent=hanging,
        )
    ]
    chunks.extend(
        _Chunk(
            raw=line,
            body=line.strip(_ASCII_WHITESPACE),
            initial=hanging,
            subsequent=hanging,
        )
        for line in original_lines[1:]
    )
    return chunks, prefix, hanging


def _matching_closer_end(
    words: tuple[str, ...],
    start: int,
    opening: str,
    closing: str,
) -> int | None:
    """Return the word index after the matching closer for `words[start]`."""
    if closing in words[start]:
        return None

    depth = words[start].count(opening)
    for index in range(start + 1, len(words)):
        depth += words[index].count(opening) - words[index].count(closing)
        if depth <= 0:
            return index + 1
    return None


def _merge_structural_words(
    words: tuple[str, ...],
    gaps: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind parser-structural words to their preceding words."""
    merged_words: list[str] = []
    merged_gaps: list[str] = []
    for index, word in enumerate(words):
        bind_list_marker = LIST_RE.fullmatch(f'{word} ') is not None
        bind_underline = is_structural_underline(word)
        bind_field = FIELD_MARKER_RE.match(word) is not None
        if index > 0 and (bind_list_marker or bind_underline or bind_field):
            merged_words[-1] += gaps[index - 1] + word
            continue
        if merged_words:
            merged_gaps.append(gaps[index - 1])
        merged_words.append(word)
    return tuple(merged_words), tuple(merged_gaps)


def _normalize_plan_boundary_gaps(
    fill_words: _FillWords,
    plan: _FillPlan,
) -> _FillWords | None:
    """Canonicalize only wide gaps selected as plan boundaries."""
    gaps = list(fill_words.gaps)
    changed = False
    for end in plan.ends[:-1]:
        if len(gaps[end - 1]) > 1:
            gaps[end - 1] = ' '
            changed = True
    if not changed:
        return None
    normalized_gaps = tuple(gaps)
    return replace(
        fill_words,
        gaps=normalized_gaps,
        normalized=_fill_words_text(fill_words.words, normalized_gaps),
    )


def _para_chunks(
    original_lines: tuple[str, ...],
    authored_indent: int,
    emitted_indent: str,
    *,
    hanging: str | None = None,
) -> list[_Chunk]:
    """Build chunks for a uniform-indent paragraph."""
    if not original_lines:
        return []
    hang = emitted_indent if hanging is None else hanging
    chunks = [
        _Chunk(
            raw=emitted_indent + original_lines[0][authored_indent:],
            body=original_lines[0][authored_indent:],
            initial=emitted_indent,
            subsequent=hang,
        )
    ]
    chunks.extend(
        _Chunk(
            raw=emitted_indent + line[authored_indent:],
            body=line[authored_indent:],
            initial=hang,
            subsequent=hang,
        )
        for line in original_lines[1:]
    )
    return chunks


def _prose_runs(regions: list[Region]) -> list[tuple[int, int]]:
    """Return half-open spans for maximal wrappable prose runs."""
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(regions):
        region = regions[index]
        if (
            region.kind != 'prose'
            or region.preserve_verbatim
            or is_structural_underline(region.text)
        ):
            index += 1
            continue
        end = index + 1
        while end < len(regions):
            next_region = regions[end]
            if (
                next_region.kind != 'prose'
                or next_region.preserve_verbatim
                or is_structural_underline(next_region.text)
            ):
                break
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _render_fill_plan(
    fill_words: _FillWords,
    plan: _FillPlan,
    *,
    initial: str,
    subsequent: str,
) -> list[str]:
    """Render a fill plan while preserving every selected same-line gap."""
    lines: list[str] = []
    start = 0
    for end in plan.ends:
        indent = initial if not lines else subsequent
        parts = [fill_words.words[start]]
        for index in range(start + 1, end):
            parts.extend((fill_words.gaps[index - 1], fill_words.words[index]))
        lines.append(indent + ''.join(parts))
        start = end
    return lines


def _select_cautious_reflow_state(
    original: tuple[str, ...],
    emissions: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    """Select a deterministic bounded cautious-reflow state."""
    states = [original]
    seen = {original: 0}
    for iteration, emitted in enumerate(emissions, start=1):
        cycle_start = seen.get(emitted)
        if cycle_start is not None:
            return min(states[cycle_start:])
        states.append(emitted)
        seen[emitted] = len(states) - 1
        if iteration == _CAUTIOUS_REFLOW_MAX_ITERATIONS:
            return original
    return original


def _smart_backoff_cost(
    words: tuple[str, ...],
    gaps: tuple[str, ...],
    candidate_index: int,
) -> int:
    """Return columns removed from the greedy line by moving `candidate_index`."""
    return len(gaps[candidate_index - 1]) + len(words[candidate_index])


def _repairable_line_end(
    words: tuple[str, ...],
    gaps: tuple[str, ...],
    line_start: int,
    line_end: int,
    smart_slack: int,
) -> bool:
    """Return True if a line ends with a separately repairable opener."""
    if line_end >= len(words) or line_end <= line_start + 1:
        return False

    candidate_index = line_end - 1
    pair = _delimiter_pair(words[candidate_index])
    if pair is None:
        return False

    opening, closing = pair
    if _matching_closer_end(words, candidate_index, opening, closing) is None:
        return False
    return _smart_backoff_cost(words, gaps, candidate_index) <= smart_slack


def _smart_backoff_candidate_end(
    words: tuple[str, ...],
    gaps: tuple[str, ...],
    line_span: tuple[int, int],
    next_capacity: int,
    smart_slack: int,
) -> int | None:
    """Return candidate line-B end if smart backoff should be accepted."""
    line_start, greedy_end = line_span
    candidate_index = greedy_end - 1
    accepted_end: int | None = None
    if greedy_end < len(words) and candidate_index > line_start:
        pair = _delimiter_pair(words[candidate_index])
        if pair is not None:
            opening, closing = pair
            matching_closer_end = _matching_closer_end(words, candidate_index, opening, closing)
            if (
                matching_closer_end is not None
                and _smart_backoff_cost(words, gaps, candidate_index) <= smart_slack
            ):
                candidate_next_end = _greedy_end(words, gaps, candidate_index, next_capacity)
                greedy_next_end = _greedy_end(words, gaps, greedy_end, next_capacity)
                if (
                    candidate_next_end >= matching_closer_end
                    and candidate_next_end >= greedy_next_end
                    and not _repairable_line_end(
                        words,
                        gaps,
                        line_start,
                        candidate_index,
                        smart_slack,
                    )
                    and not _repairable_line_end(
                        words,
                        gaps,
                        candidate_index,
                        candidate_next_end,
                        smart_slack,
                    )
                ):
                    accepted_end = candidate_next_end
    return accepted_end


def _smart_plan(
    fill_words: _FillWords,
    width: int,
    *,
    initial: str,
    options: _FillOptions,
    subsequent: str,
) -> _FillPlan:
    """Plan greedily while repairing inexpensive delimiter-fragment breaks."""
    if options.smart_slack == 0:
        return _greedy_plan(
            fill_words,
            width,
            initial=initial,
            subsequent=subsequent,
        )

    word_count = len(fill_words.words)
    ends: list[int] = []
    start = 0

    while start < word_count:
        indent = initial if not ends else subsequent
        capacity = width - len(indent)
        end = _greedy_end(fill_words.words, fill_words.gaps, start, capacity)

        candidate_next_end = _smart_backoff_candidate_end(
            fill_words.words,
            fill_words.gaps,
            (start, end),
            width - len(subsequent),
            options.smart_slack,
        )
        if candidate_next_end is not None:
            ends.extend((end - 1, candidate_next_end))
            start = candidate_next_end
            continue

        ends.append(end)
        start = end

    return _FillPlan(ends=tuple(ends))


def _plan_fill(
    fill_words: _FillWords,
    width: int,
    *,
    initial: str,
    options: _FillOptions,
    subsequent: str,
) -> _FillPlan:
    """Dispatch fill planning to the configured line-breaking strategy."""
    if options.fill_style == 'greedy':
        return _greedy_plan(
            fill_words,
            width,
            initial=initial,
            subsequent=subsequent,
        )
    if options.fill_style == 'balanced':
        return _balanced_plan(
            fill_words,
            width,
            initial=initial,
            subsequent=subsequent,
        )
    return _smart_plan(
        fill_words,
        width,
        initial=initial,
        options=options,
        subsequent=subsequent,
    )


def _split_fill_words(
    text: str,
) -> _FillWords:
    """Normalize fill text and bind list/field markers and structural underlines."""
    expanded = text.expandtabs().translate(_TEXTWRAP_WHITESPACE_TRANS)
    leading = expanded[: len(expanded) - len(expanded.lstrip(' '))]
    normalized = expanded.strip(' ')
    if not normalized:
        return _FillWords(gaps=(), leading='', marker_bound=False, normalized='', words=())
    parts = re.split(r'( +)', normalized)
    words = tuple(parts[::2])
    gaps = tuple(parts[1::2])
    merged_words, merged_gaps = _merge_structural_words(words, gaps)
    return _FillWords(
        gaps=merged_gaps,
        leading=leading,
        marker_bound=len(merged_words) != len(words),
        normalized=_fill_words_text(merged_words, merged_gaps),
        words=merged_words,
    )


def _stabilize_fill_plan(  # noqa: PLR0913
    fill_words: _FillWords,
    width: int,
    *,
    initial: str,
    options: _FillOptions,
    subsequent: str,
    validator: _PlanValidator | None = None,
) -> tuple[_FillWords, _FillPlan, list[str]]:
    """Plan and render until boundary repairs reach a fixed point."""
    while True:
        plan = _plan_fill(
            fill_words,
            width,
            initial=initial,
            options=options,
            subsequent=subsequent,
        )
        lines = _render_fill_plan(
            fill_words,
            plan,
            initial=initial,
            subsequent=subsequent,
        )
        if validator is not None:
            boundaries = validator(lines, plan)
            if boundaries:
                fill_words = _bind_plan_boundaries(fill_words, boundaries)
                continue
        normalized_fill_words = _normalize_plan_boundary_gaps(fill_words, plan)
        if normalized_fill_words is not None:
            fill_words = normalized_fill_words
            continue
        return fill_words, plan, lines


def _fill(  # noqa: PLR0913
    text: str,
    width: int,
    *,
    initial: str = '',
    subsequent: str | None = None,
    options: _FillOptions,
    validator: _PlanValidator | None = None,
) -> str:
    """Fill text using the configured line-breaking strategy."""
    subsequent_indent = initial if subsequent is None else subsequent
    fill_words = _split_fill_words(text)
    if not fill_words.normalized:
        return ''
    preserve_leading = options.fill_style == 'greedy' or (
        options.fill_style == 'smart' and options.smart_slack == 0
    )
    first_capacity = width - len(initial)
    if (
        preserve_leading
        and fill_words.leading
        and not fill_words.marker_bound
        and len(fill_words.leading) + len(fill_words.words[0]) <= first_capacity
    ):
        words = (fill_words.leading + fill_words.words[0], *fill_words.words[1:])
        fill_words = replace(
            fill_words,
            normalized=_fill_words_text(words, fill_words.gaps),
            words=words,
        )
    if len(fill_words.normalized) <= width - len(initial):
        return initial + fill_words.normalized

    _, _, lines = _stabilize_fill_plan(
        fill_words,
        width,
        initial=initial,
        options=options,
        subsequent=subsequent_indent,
        validator=validator,
    )
    return '\n'.join(lines)


def _cautious_reflow(  # noqa: C901
    chunks: list[_Chunk],
    width: int,
    options: _FillOptions,
) -> list[str]:
    """Reflow semantically joinable runs until cautious output is stable."""
    if not chunks:
        return []

    template = chunks[0]

    def _joined_body(run: list[_Chunk]) -> str:
        parts: list[str] = []
        for chunk in run:
            body = chunk.body.strip(_ASCII_WHITESPACE)
            if body:
                parts.append(body)
        return ' '.join(parts)

    def _emit_run(run: list[_Chunk]) -> list[str]:
        first = run[0]

        if len(run) == 1 and len(first.raw) <= width:
            return [first.raw]

        body = _joined_body(run)
        if not body:
            return [first.initial.rstrip()]

        filled = _fill(
            body,
            width,
            initial=first.initial,
            subsequent=first.subsequent,
            options=options,
        )
        return filled.split('\n')

    def _emit_once(current: list[_Chunk]) -> list[str]:
        result: list[str] = []
        run: list[_Chunk] = [current[0]]

        for chunk in current[1:]:
            if _is_semantically_joinable(run[-1].body, chunk.body):
                run.append(chunk)
                continue

            result.extend(_emit_run(run))
            run = [chunk]

        result.extend(_emit_run(run))
        return result

    def _reparse_emitted(lines: list[str]) -> list[_Chunk]:
        reparsed: list[_Chunk] = []
        for index, line in enumerate(lines):
            initial = template.initial if index == 0 else template.subsequent
            body = (
                ''
                if line == initial.rstrip()
                else line.removeprefix(initial).strip(_ASCII_WHITESPACE)
            )
            reparsed.append(
                _Chunk(
                    raw=line,
                    body=body,
                    initial=initial,
                    subsequent=template.subsequent,
                )
            )
        return reparsed

    def _emissions() -> Iterable[tuple[str, ...]]:
        current = chunks
        while True:
            emitted = _emit_once(current)
            yield tuple(emitted)
            current = _reparse_emitted(emitted)

    original = tuple(chunk.raw for chunk in chunks)
    selected = _select_cautious_reflow_state(original, _emissions())
    return list(selected)


def _emit_chunks(chunks: list[_Chunk], width: int, options: _FillOptions) -> list[str]:
    """Emit chunks: raw if it fits, split via _fill if it overflows."""
    result: list[str] = []
    for chunk in chunks:
        if len(chunk.raw) <= width:
            result.append(chunk.raw)
        else:
            filled = _fill(
                chunk.body,
                width,
                initial=chunk.initial,
                subsequent=chunk.subsequent,
                options=options,
            )
            result.extend(filled.split('\n'))
    return result


def _emit_reflowed_chunks(
    chunks: list[_Chunk],
    width: int,
    options: _FillOptions,
    reflow: ReflowMode,
) -> list[str] | None:
    """Emit chunks using the appropriate reflow strategy."""
    if reflow == 'none':
        return _emit_chunks(chunks, width, options)
    if reflow == 'cautious':
        return _cautious_reflow(chunks, width, options)
    return None


def _summary_break_line_indexes(
    lines: list[str],
    styles: Collection[Style],
) -> tuple[int, ...]:
    """Return candidate continuation lines that would end a summary."""
    code_block_lines = find_code_block_lines(lines)
    unsafe_indexes: list[int] = []
    for index, line in enumerate(lines[1:], start=1):
        unsafe = is_summary_break(line, index, lines, None, code_block_lines)
        for style in styles:
            if is_summary_break(line, index, lines, style, code_block_lines):
                unsafe = True
        if unsafe:
            unsafe_indexes.append(index)
    return tuple(unsafe_indexes)


def _protect_authored_summary_lines(lines: list[str], style: Style) -> list[str]:
    """Bind target-structural continuation tokens to their predecessors."""
    protected = lines.copy()
    while unsafe_indexes := _summary_break_line_indexes(protected, (style,)):
        line_index = unsafe_indexes[0]
        continuation = protected[line_index]
        token_match = re.search(r'\S+', continuation)
        if token_match is None:
            break
        token = token_match.group()
        protected[line_index - 1] = f'{protected[line_index - 1].rstrip()} {token}'
        remainder = continuation[token_match.end() :].lstrip()
        if remainder:
            protected[line_index] = remainder
        else:
            del protected[line_index]
    return protected


def _summary_fragments(region: Region) -> Iterator[tuple[str, bool]]:
    """Yield joined prose or intact marker lines with their bare-marker flag."""
    original_lines = region.original_lines
    if original_lines is None or not any(_is_bare_summary_marker(line) for line in original_lines):
        yield region.text, False
        return

    fragment: list[str] = []
    for line in original_lines:
        if not _is_bare_summary_marker(line):
            fragment.append(line)
            continue
        if fragment:
            yield ' '.join(item.strip(_ASCII_WHITESPACE) for item in fragment), False
            fragment = []
        yield line, True
    if fragment:
        yield ' '.join(item.strip(_ASCII_WHITESPACE) for item in fragment), False


def _summary_measurement_lines(region: Region) -> list[str]:
    """Return the established logical lines used for skip reporting."""
    return [text for text, _is_marker in _summary_fragments(region)]


def _wrap_entry_continuation(  # noqa: PLR0913
    first_line: str,
    cont_lines: list[str],
    width: int,
    style: Style,
    *,
    options: _FillOptions,
    reflow: ReflowMode = 'full',
    original_lines: tuple[str, ...] | None = None,
) -> list[str]:
    """Wrap continuation lines under a header line with hanging indent."""
    cont_indent = ' ' * style.entry_continuation_indent
    if not cont_lines:
        return [first_line]

    if reflow != 'full' and original_lines is not None:
        chunks: list[_Chunk] = [
            _Chunk(
                raw=line,
                body=line.strip(_ASCII_WHITESPACE),
                initial=cont_indent,
                subsequent=cont_indent,
            )
            for line in original_lines[1:]
        ]
        result = _emit_reflowed_chunks(chunks, width, options, reflow)
        if result is not None:
            return [first_line, *result]

    body = ' '.join(line.strip(_ASCII_WHITESPACE) for line in cont_lines)
    wrapped = _fill(
        body,
        width,
        initial=cont_indent,
        subsequent=cont_indent,
        options=options,
    )
    return [first_line, *wrapped.split('\n')]


def _wrap_entry(  # noqa: PLR0913
    text: str,
    width: int,
    style: Style,
    entry_match: EntryMatch | None = None,
    *,
    options: _FillOptions,
    role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
    reflow: ReflowMode = 'full',
    original_lines: tuple[str, ...] | None = None,
) -> list[str]:
    """Wrap an entry region with style-appropriate indentation."""
    lines = text.split('\n')
    first_line = lines[0]

    em = entry_match if entry_match is not None else style.match_entry(first_line, role=role)

    if em is not None and original_lines is not None and reflow != 'full':
        chunks = _entry_chunks(original_lines, style, role)
        if chunks is not None:
            result = _emit_reflowed_chunks(chunks, width, options, reflow)
            if result is not None:
                return result

    # The stored match contains the joined multi-line description, so physical
    # first-line offsets require a fresh match.
    first_em = style.match_entry(first_line, role=role) if entry_match is not None else em

    if first_em is not None and first_em.description and first_em.description_start is not None:
        desc_start = first_em.description_start
        marker = first_line[:desc_start]
        hanging = ' ' * style.entry_continuation_indent

        if original_lines is not None:
            parts = [original_lines[0][desc_start:].strip(_ASCII_WHITESPACE)]
            parts.extend(ln.strip(_ASCII_WHITESPACE) for ln in original_lines[1:])
            desc_text = ' '.join(p for p in parts if p)
        else:
            all_desc = [
                first_em.description,
                *(cl.strip(_ASCII_WHITESPACE) for cl in lines[1:]),
            ]
            desc_text = ' '.join(all_desc)

        if not desc_text.strip():
            return [first_line.rstrip()]

        wrapped = _fill(
            desc_text,
            width,
            initial=marker,
            subsequent=hanging,
            options=options,
        )
        return wrapped.split('\n')

    return _wrap_entry_continuation(
        first_line,
        lines[1:],
        width,
        style,
        options=options,
        reflow=reflow,
        original_lines=original_lines,
    )


def _wrap_entry_region(region: Region, ctx: _WrapCtx) -> list[str]:
    """Wrap a section entry, falling back to verbatim when no style is active."""
    if ctx.style is None:
        return [region.text]
    return _wrap_entry(
        region.text,
        ctx.width,
        ctx.style,
        entry_match=region.entry_match,
        options=_FillOptions(ctx.config.fill_style, ctx.config.smart_slack),
        role=region.role,
        reflow=ctx.config.reflow,
        original_lines=region.original_lines,
    )


def _wrap_list_item(
    text: str,
    width: int,
    *,
    options: _FillOptions,
    reflow: ReflowMode = 'full',
    original_lines: tuple[str, ...] | None = None,
) -> list[str]:
    """Wrap a list item with hanging indent."""
    m = LIST_RE.match(text)
    if m is None:
        return [text]
    prefix = m.group(0)
    hanging = ' ' * len(prefix)
    if original_lines is not None and reflow != 'full':
        chunks, _, _ = _list_chunks(original_lines)
        result = _emit_reflowed_chunks(chunks, width, options, reflow)
        if result is not None:
            return result

    # Full mode: join original_lines (stripped) and reflow
    if original_lines is not None:
        parts = [original_lines[0][len(prefix) :].strip(_ASCII_WHITESPACE)]
        parts.extend(ln.strip(_ASCII_WHITESPACE) for ln in original_lines[1:])
        rest = ' '.join(p for p in parts if p)
    else:
        rest = text[len(prefix) :].replace('\n', ' ').strip(_ASCII_WHITESPACE)
    if not rest:
        # Empty list item (e.g., bare "- "): preserve bullet without wrapping.
        # _fill('') drops initial indentation, so guard explicitly.
        return [prefix.rstrip()]
    wrapped = _fill(
        rest,
        width,
        initial=prefix,
        subsequent=hanging,
        options=options,
    )
    return wrapped.split('\n')


def _wrap_list_item_region(region: Region, ctx: _WrapCtx) -> list[str]:
    """Wrap a list item region using the list-item helper."""
    return _wrap_list_item(
        region.text,
        ctx.width,
        options=_FillOptions(ctx.config.fill_style, ctx.config.smart_slack),
        reflow=ctx.config.reflow,
        original_lines=region.original_lines,
    )


def _wrap_paragraph(region: Region, ctx: _WrapCtx, indent: str) -> list[str]:
    """Wrap a paragraph region with the given indent prefix."""
    hanging = indent
    options = _FillOptions(ctx.config.fill_style, ctx.config.smart_slack)
    if ctx.style is None and FIELD_MARKER_RE.match(region.text):
        hanging = indent + ' ' * _SPHINX_FALLBACK_HANGING
    if region.original_lines is not None and ctx.config.reflow != 'full':
        result = _emit_reflowed_chunks(
            _para_chunks(region.original_lines, region.indent, indent, hanging=hanging),
            ctx.width,
            options,
            ctx.config.reflow,
        )
        if result is not None:
            return result
    return _fill(
        region.text,
        ctx.width,
        initial=indent,
        subsequent=hanging,
        options=options,
    ).split('\n')


def _wrap_description(region: Region, ctx: _WrapCtx) -> list[str]:
    """Wrap a top-level description paragraph."""
    return _wrap_paragraph(region, ctx, indent='')


def _wrap_prose(
    region: Region,
    ctx: _WrapCtx,
    indent_delta: int,
) -> list[str]:
    """Wrap an in-section prose paragraph with the appropriate indent."""
    prose_indent = ' ' * (region.indent + indent_delta)
    return _wrap_paragraph(region, ctx, indent=prose_indent)


def _wrap_summary_fragment(
    text: str,
    options: _SummaryOptions,
) -> list[str]:
    """Wrap one marker-free summary fragment."""
    # Summary first line shares space with opening quotes.  Use
    # initial_indent to account for quote overhead on line 1; subsequent
    # lines get the full width.  Strip fake indent afterward.
    fake_indent = ' ' * options.quote_overhead
    styles = tuple(get_style(name) for name in sorted(get_styles()))

    def validate_summary_plan(lines: list[str], plan: _FillPlan) -> tuple[int, ...]:
        candidate_lines = lines.copy()
        candidate_lines[0] = candidate_lines[0][len(fake_indent) :]
        return tuple(
            plan.ends[line_index - 1]
            for line_index in _summary_break_line_indexes(candidate_lines, styles)
        )

    wrapped = _fill(
        text,
        options.width,
        initial=fake_indent,
        subsequent='',
        options=options.fill,
        validator=validate_summary_plan,
    )
    lines = wrapped.split('\n')
    lines[0] = lines[0][options.quote_overhead :]
    return lines


def _render_wrapped_summary_lines(
    region: Region,
    options: _SummaryOptions,
) -> list[str]:
    """Render marker-aware physical lines with the summary fill engine."""
    physical_lines: list[str] = []
    for text, is_marker in _summary_fragments(region):
        if is_marker:
            physical_lines.append(text)
        else:
            physical_lines.extend(
                _wrap_summary_fragment(
                    text,
                    replace(
                        options,
                        quote_overhead=options.quote_overhead if not physical_lines else 0,
                    ),
                )
            )
    return physical_lines


def _wrap_summary_region(region: Region, ctx: _WrapCtx) -> list[str]:
    """Render a summary from physical provenance or the fill engine."""
    if not ctx.config.wrap_summary:
        if region.original_lines is None:
            return [region.text]
        authored_lines = list(region.original_lines)
        if ctx.config.convert is not None and ctx.style is not None:
            return _protect_authored_summary_lines(authored_lines, ctx.style)
        return authored_lines
    return _render_wrapped_summary_lines(
        region,
        _SummaryOptions(
            fill=_FillOptions(ctx.config.fill_style, ctx.config.smart_slack),
            quote_overhead=ctx.quote_overhead,
            width=ctx.width,
        ),
    )


def _wrap_region(region: Region, ctx: _WrapCtx, indent_delta: int) -> list[str]:
    """Dispatch one region while honoring verbatim preservation."""
    if region.preserve_verbatim:
        if region.original_lines is not None:
            wrapped = list(region.original_lines)
        else:
            wrapped = [region.text]
    else:
        match region.kind:
            case 'blank':
                wrapped = ['']
            case kind if kind in VERBATIM_REGION_KINDS:
                wrapped = [region.text]
            case 'description':
                wrapped = _wrap_description(region, ctx)
            case 'entry':
                wrapped = _wrap_entry_region(region, ctx)
            case 'list_item':
                wrapped = _wrap_list_item_region(region, ctx)
            case 'prose':
                wrapped = _wrap_prose(region, ctx, indent_delta)
            case 'summary':
                wrapped = _wrap_summary_region(region, ctx)
            case _:
                raise ValueError(f'Unknown region kind: {region.kind}')
    return wrapped


def _candidate_run_gains_entry_shape(
    regions: list[Region],
    start: int,
    end: int,
    candidate_delta: int,
    ctx: _WrapCtx,
) -> bool:
    """Check whether candidate rendering turns prose into entry syntax."""
    if ctx.style is None:
        return False
    for index in range(start, end):
        region = regions[index]
        role = _containing_role(regions, index, ctx.style)
        zero_lines = [
            physical
            for line in _wrap_region(region, ctx, indent_delta=0)
            for physical in line.split('\n')
        ]
        if any(ctx.style.match_entry(line, role=role) is not None for line in zero_lines):
            continue
        candidate_lines = [
            physical
            for line in _wrap_region(region, ctx, indent_delta=candidate_delta)
            for physical in line.split('\n')
        ]
        if any(
            ctx.style.match_entry(line, role=role) is not None
            or _is_conservative_entry_candidate(line, ctx)
            for line in candidate_lines
        ):
            return True
    return False


def _following_entry_collision(
    regions: list[Region],
    end: int,
    candidate_delta: int,
    ctx: _WrapCtx,
) -> bool:
    """Check whether a shifted run would collide with a following entry."""
    if (
        ctx.style is None
        or ctx.style.uses_field_list_entries
        or end >= len(regions)
        or regions[end].kind != 'entry'
    ):
        return False
    entry_lines = _wrap_region(regions[end], ctx, indent_delta=0)
    if not entry_lines:
        return False
    entry_indent = len(entry_lines[0]) - len(entry_lines[0].lstrip())
    candidate_indent = regions[end - 1].indent + candidate_delta
    return candidate_indent == entry_indent


def _render_regions(
    regions: list[Region],
    ctx: _WrapCtx,
    deltas: list[int],
) -> list[_RenderedLine]:
    """Render every region with physical-line provenance."""
    rendered: list[_RenderedLine] = []
    for index, region in enumerate(regions):
        for line in _wrap_region(region, ctx, deltas[index]):
            rendered.extend(
                _RenderedLine(text=physical_line, region_index=index)
                for physical_line in line.split('\n')
            )
    return rendered


def _scanner_projection_is_stable(
    regions: list[Region],
    ctx: _WrapCtx,
    deltas: list[int],
) -> bool:
    """Check emitted scanner protection against every originating region."""
    rendered = _render_regions(regions, ctx, deltas)
    lines = [line.text for line in rendered]
    title_line_count = match_opening_title(lines, ctx.style)
    protected = find_code_block_lines(lines, start=title_line_count or 0)
    return all(
        (line_index in protected) is regions[line.region_index].source_scanner_protected
        for line_index, line in enumerate(rendered)
    )


def _choose_prose_deltas(regions: list[Region], ctx: _WrapCtx) -> list[int]:
    """Choose one full or zero indentation delta for each maximal prose run."""
    chosen = [0] * len(regions)
    for start, end in _prose_runs(regions):
        minimum_indent = min(region.indent for region in regions[start:end])
        candidate_delta = max(0, ctx.section_body_indent - minimum_indent)
        if candidate_delta == 0:
            continue
        if (
            _candidate_crosses_preceding_boundary(
                regions,
                start,
                candidate_delta,
                ctx.style,
            )
            or _candidate_run_gains_entry_shape(
                regions,
                start,
                end,
                candidate_delta,
                ctx,
            )
            or _following_entry_collision(regions, end, candidate_delta, ctx)
        ):
            continue
        candidate = chosen.copy()
        candidate[start:end] = [candidate_delta] * (end - start)
        if _scanner_projection_is_stable(regions, ctx, candidate):
            chosen = candidate
    return chosen


def _render_with_deltas(
    regions: list[Region],
    config: DocwrapConfig,
    options: _RenderOptions,
) -> list[_RenderedLine]:
    """Render regions injectively with physical-line provenance."""
    effective_col = _effective_column(
        options.base_indent,
        options.col_offset,
        options.quote_overhead,
    )
    ctx = _WrapCtx(
        config=config,
        style=options.style,
        width=max(config.line_length - effective_col, 1),
        quote_overhead=options.quote_overhead,
        section_body_indent=options.style.section_body_indent if options.style is not None else 4,
    )
    return _render_regions(regions, ctx, _choose_prose_deltas(regions, ctx))


def aggregate_style(
    config_style: StyleName | None,
    detected_styles: set[StyleName],
) -> StyleName | None:
    """Aggregate per-docstring style detections into a file-level style.

    Args:
        config_style: Explicit style override from config/CLI, or None.
        detected_styles: Set of style names detected across docstrings.

    Returns:
        The explicit style, the sole detected style, or `None` when styles are
        mixed or absent.
    """
    if config_style is not None:
        return config_style
    if len(detected_styles) == 1:
        return next(iter(detected_styles))
    return None


def compute_quote_overhead(parsed: 'ParsedDocstring', config: DocwrapConfig) -> int:
    """Width consumed by leading quote chars on the first content line.

    Args:
        parsed: Parsed docstring with prefix metadata.
        config: docwrap configuration carrying summary_start.

    Returns:
        0 when the summary starts on its own (second) line; otherwise the
        prefix length plus triple-quote width.
    """
    effective_position = resolve_summary_position(config.summary_start, parsed.original_position)
    if effective_position == 'second-line':
        return 0
    return len(parsed.lit.prefix) + TRIPLE_QUOTE_LEN


def detect_summary_skip(
    regions: list[Region],
    config: DocwrapConfig,
    *,
    base_indent: int,
    col_offset: int | None,
    quote_overhead: int,
) -> tuple[bool, int | None]:
    """Detect an overlong summary preserved because summary wrapping is disabled.

    Args:
        regions: Parsed docstring regions.
        config: Wrapping configuration.
        base_indent: Base indentation of the docstring's source line.
        col_offset: Column offset of the docstring literal, when known.
        quote_overhead: Length of the literal prefix and opening quotes.

    Returns:
        `(True, width)` when a preserved summary exceeds the target, otherwise
        `(False, None)`. Widths include the effective source column, with
        opening literal overhead on the first summary line. The summary-only
        one-line check also counts closing quotes when it triggers the skip.
    """
    if config.wrap_summary:
        return False, None
    effective_col = _effective_column(base_indent, col_offset, quote_overhead)
    for region in regions:
        if region.kind == 'summary':
            summary_lines = _summary_measurement_lines(region)
            summary_line_width = max(
                effective_col + len(line) + (quote_overhead if line_index == 0 else 0)
                for line_index, line in enumerate(summary_lines)
            )
            if summary_line_width > config.line_length:
                return True, summary_line_width
            # For summary-only docstrings, reassemble uses a one-liner which
            # also includes closing quotes.  Check that width too.
            if not _has_body_regions(regions) and len(summary_lines) == 1:
                oneliner_width = summary_line_width + TRIPLE_QUOTE_LEN
                if oneliner_width > config.line_length:
                    return True, oneliner_width
            break
    return False, None


def is_oneliner_skip(
    summary_skipped: bool,
    new_literal: str | None,
    regions: list[Region],
) -> bool:
    """Check if a summary-only docstring should be left unchanged.

    When wrap_summary=False and the summary exceeds the target, a summary-only
    docstring can't be meaningfully changed without creating a D200 violation
    (multi-line docstring with summary on first line when entire content fits
    on one logical line).

    Args:
        summary_skipped: Whether wrapping skipped an overlong summary.
        new_literal: Rewritten docstring literal, or `None` when unchanged.
        regions: Parsed regions from the docstring.

    Returns:
        Whether the rewritten summary-only docstring should be discarded.
    """
    if not summary_skipped or new_literal is None:
        return False
    return not _has_body_regions(regions) and '\n' in new_literal


def wrap_regions(  # noqa: PLR0913
    regions: list[Region],
    config: DocwrapConfig,
    base_indent: int,
    *,
    style: Style | None = None,
    quote_overhead: int = 3,
    col_offset: int | None = None,
) -> str:
    """Wrap parsed regions into final docstring content.

    Args:
        regions: List of parsed regions.
        config: Wrapping configuration.
        base_indent: Base indentation (character count, from leading whitespace
            on the docstring's source line).
        style: Style instance for style-specific indentation rules.
        quote_overhead: Length of prefix + opening quotes (e.g., 3 for
            triple-quote, 4 for r-prefixed triple-quote) to subtract from the
            summary's first-line width.
        col_offset: Column where the docstring starts. Used for width
            calculations. Defaults to base_indent when not provided.

    Returns:
        Wrapped docstring content (without quotes or base indentation).
    """
    return '\n'.join(
        line.text
        for line in _render_with_deltas(
            regions,
            config,
            _RenderOptions(
                base_indent=base_indent,
                style=style,
                quote_overhead=quote_overhead,
                col_offset=col_offset,
            ),
        )
    )


def simulate_wrap_delta(parsed: 'ParsedDocstring', config: DocwrapConfig) -> str:
    """Wrap a parsed docstring's regions and return the joined text.

    Stats-facing helper: lets callers pre-compute width-invariant wrapping
    output without reaching into private symbols or reassembling source.

    Args:
        parsed: Parsed docstring with regions, base_indent, style, and
            col_offset.
        config: Wrapping configuration.

    Returns:
        Wrapped docstring content as a string suitable for passing back as
        wrap_parsed_docstring's pre_wrapped argument.
    """
    return wrap_regions(
        list(parsed.regions),
        config,
        parsed.base_indent,
        style=parsed.style,
        quote_overhead=compute_quote_overhead(parsed, config),
        col_offset=parsed.col_offset,
    )
