"""Parse, normalize, and reassemble docstring literals."""

import io
import re
from dataclasses import dataclass

from docwrap._types import DocwrapConfig, SummaryStart

PREFIX_RE = re.compile(r'^([brBRuU]*)("""|\'\'\'|"|\')')
TRIPLE_QUOTE_LEN = 3


@dataclass(frozen=True, slots=True)
class LiteralInfo:
    """Parsed docstring literal components.

    Attributes:
        prefix: String prefix (`r`, `u`, etc.).
        quote_char: Quote delimiter characters (triple).
        content: Raw source body between the delimiters, LF-normalized.
        raw_literal: Exact source text of the literal.
    """

    prefix: str
    quote_char: str
    content: str
    raw_literal: str


def _ends_with_unescaped_quote(content: str, quote: str) -> bool:
    """Check whether `content` ends with an unescaped quote character."""
    if not content.endswith(quote):
        return False
    before = content[: len(content) - 1]
    backslashes = len(before) - len(before.rstrip('\\'))
    return backslashes % 2 == 0


def _has_unescaped_run(content: str, quote_char: str) -> bool:
    """Check whether an unescaped delimiter run occurs in `content`."""
    start = 0
    while True:
        idx = content.find(quote_char, start)
        if idx == -1:
            return False
        backslashes = 0
        j = idx - 1
        while j >= 0 and content[j] == '\\':
            backslashes += 1
            j -= 1
        if backslashes % 2 == 0:
            return True
        start = idx + 1


def _shared_source_prefix(lines: list[str], source_margin: str) -> str:
    """Return the exact source-margin prefix shared by every nonblank line."""
    prefix = source_margin
    for line in lines:
        if not line.strip():
            continue
        while prefix and not line.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def byte_col_to_char_col(line: str, byte_col: int) -> int:
    """Convert AST byte-based col_offset to character-based col_offset.

    Args:
        line: Source line containing the offset.
        byte_col: Zero-based UTF-8 byte offset.

    Returns:
        The corresponding zero-based character offset.
    """
    if byte_col == 0 or line[:byte_col].isascii():
        return byte_col
    return len(line.encode('utf-8')[:byte_col].decode('utf-8'))


def is_own_line(col_offset: int, source_margin: str) -> bool:
    """Return whether the literal starts immediately after its source margin."""
    return col_offset == len(source_margin)


def normalize_content(
    content: str,
    *,
    source_margin: str,
    source_margin_candidate: str | None = None,
) -> tuple[str, int]:
    """Normalize docstring content within its exact source margin.

    Args:
        content: Raw or escape-shielded docstring source content.
        source_margin: Exact leading whitespace of the literal's source line.
        source_margin_candidate: Alternate generated margin for a non-own-line
            literal. Used only when every selected nonblank line shares it and
            does not share the exact source margin.

    Returns:
        Tuple of (normalized content, number of leading blank lines stripped).
    """
    text = content.replace('\r\n', '\n').replace('\r', '\n')

    lines = text.split('\n')
    if lines:
        lines[0] = lines[0].lstrip()

    # Strip leading/trailing blank lines
    leading_stripped = 0
    while lines and not lines[0].strip():
        lines.pop(0)
        leading_stripped += 1
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return '', leading_stripped

    selected_start = 0 if lines[0] != lines[0].lstrip() else 1
    selected = lines[selected_start:]
    prefix = _shared_source_prefix(selected, source_margin)
    if source_margin_candidate is not None and prefix != source_margin:
        candidate_prefix = _shared_source_prefix(selected, source_margin_candidate)
        if candidate_prefix == source_margin_candidate:
            prefix = candidate_prefix
    if prefix:
        lines[selected_start:] = [
            line.removeprefix(prefix) if line.startswith(prefix) else line for line in selected
        ]

    normalized_lines = ['' if not line.strip() else line for line in lines]
    return '\n'.join(normalized_lines), leading_stripped


def parse_literal(raw_literal: str) -> LiteralInfo | None:
    """Parse a docstring literal into prefix, quote style, and content.

    Args:
        raw_literal: The exact source text of the literal.

    Returns:
        LiteralInfo with the raw LF-normalized body as content, or None if not
        a single triple-quoted string.
    """
    m = PREFIX_RE.match(raw_literal)
    if m is None:
        return None

    prefix = m.group(1)
    quote_char = m.group(2)

    # Only process triple-quoted docstrings
    if len(quote_char) != TRIPLE_QUOTE_LEN:
        return None

    # Reject implicit string concatenation: walk the inner content checking
    # for unescaped triple-quote sequences
    inner = raw_literal[len(prefix) + len(quote_char) : -len(quote_char)]
    j = 0
    while j < len(inner):
        if inner[j] == '\\':
            j += 2
            continue
        if inner[j : j + TRIPLE_QUOTE_LEN] == quote_char:
            return None
        j += 1

    body = inner.replace('\r\n', '\n').replace('\r', '\n')

    return LiteralInfo(
        prefix=prefix,
        quote_char=quote_char,
        content=body,
        raw_literal=raw_literal,
    )


def resolve_summary_position(configured: SummaryStart, original: SummaryStart) -> SummaryStart:
    """Resolve 'preserve' to the original summary position."""
    return original if configured == 'preserve' else configured


def reassemble(  # noqa: PLR0913
    content: str,
    quote_char: str,
    prefix: str,
    base_indent: int,
    config: DocwrapConfig,
    *,
    col_offset: int | None = None,
    original_position: SummaryStart = 'first-line',
    source_margin: str | None = None,
) -> str | None:
    """Build final docstring literal from wrapped content.

    Args:
        content: Wrapped docstring content with restored source-level escape
            spellings (no base indentation).
        quote_char: Original quote characters (triple).
        prefix: Original prefix (r, u, etc.).
        base_indent: Indentation level.
        config: Wrapping configuration.
        col_offset: Column where the docstring starts (for one-liner fit
            check). Defaults to base_indent when not provided.
        original_position: Detected summary position from the original
            docstring. Used when config.summary_start is 'preserve'.
        source_margin: Exact leading whitespace before an own-line literal.
            Defaults to spaces derived from base_indent when not provided.

    Returns:
        Complete docstring literal, or None if the content cannot be safely
        wrapped (e.g., both triple-quote styles appear in the content).
    """
    indent_str = source_margin if source_margin is not None else ' ' * base_indent

    # Handle quote collisions: only unescaped delimiter runs collide
    if _has_unescaped_run(content, quote_char):
        alt = "'''" if quote_char == '"""' else '"""'
        if _has_unescaped_run(content, alt):
            return None
        quote_char = alt

    # Try one-liner.  Skip if content ends with the quote character --
    # e.g., content 'Returns "value"' with quote_char '"""' would produce
    # '"""Returns "value""""' which is a SyntaxError.
    oneliner_prefix = prefix
    oneliner_content = content
    if 'r' in prefix.lower() and (len(content) - len(content.rstrip('\\'))) % 2:
        # Only a one-line raw literal puts the trailing slash against its delimiter.
        oneliner_prefix = ''
        oneliner_content = content.replace('\\', '\\\\')
    oneliner_quote = quote_char
    if _has_unescaped_run(oneliner_content, oneliner_quote):
        alt = "'''" if oneliner_quote == '"""' else '"""'
        oneliner_quote = None if _has_unescaped_run(oneliner_content, alt) else alt
    if (
        oneliner_quote is not None
        and '\n' not in content
        and content == content.lstrip()
        and not _ends_with_unescaped_quote(oneliner_content, oneliner_quote[0])
    ):
        oneliner = f'{oneliner_prefix}{oneliner_quote}{oneliner_content}{oneliner_quote}'
        effective_col = col_offset if col_offset is not None else base_indent
        if effective_col + len(oneliner) <= config.line_length:
            return oneliner

    # Multiline layout
    lines = content.split('\n')

    # Resolve effective summary position
    position = resolve_summary_position(config.summary_start, original_position)

    if position == 'second-line':
        # D213: opening quotes alone, summary on next line
        result_lines = [f'{prefix}{quote_char}']
        # Guard against trailing whitespace on empty content
        result_lines.append(f'{indent_str}{lines[0]}' if lines[0] else '')
    else:
        # D212 (default): summary on opening quotes line
        result_lines = [f'{prefix}{quote_char}{lines[0]}']

    for line in lines[1:]:
        if line.strip():
            result_lines.append(f'{indent_str}{line}')
        else:
            result_lines.append('')
    result_lines.append(f'{indent_str}{quote_char}')
    return '\n'.join(result_lines)


def split_source_lines(source: str) -> list[str]:
    """Split Python physical lines while retaining their terminators.

    Returns:
        Lines ending in LF, CRLF, or CR, plus any unterminated final line.
        Empty source produces an empty list. Joining the result without a
        separator reproduces source exactly.
    """
    with io.StringIO(source, newline='') as stream:
        return stream.readlines()


def detect_original_position(
    raw_literal: str,
    lit: LiteralInfo,
) -> SummaryStart:
    """Detect whether the original docstring uses D212 or D213 format.

    Args:
        raw_literal: Original source text of the string literal.
        lit: Parsed literal (prefix, quote_char, content).

    Returns:
        'second-line' if the summary starts on the line after the opening
        quotes (D213), 'first-line' otherwise (D212).
    """
    if not lit.content:
        return 'first-line'
    first_line = split_source_lines(raw_literal)[0]
    after_quotes = first_line[len(lit.prefix) + TRIPLE_QUOTE_LEN :]
    if not after_quotes.strip():
        return 'second-line'
    return 'first-line'
