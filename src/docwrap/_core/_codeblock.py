"""Detect code blocks and structured text that must remain verbatim."""

import re

_ALIGNED_ROW_RE = re.compile(r'(?P<indent> *)`[^`\s]+` {2,}(?P<description>\S.*)')
_TABLE_BORDER_RE = re.compile(r'(?:={2,}(?: +={2,})+|-{2,}(?: +-{2,})+)')


def _find_tilde_closer(lines: list[str], start: int, opener_len: int) -> int | None:
    """Find the first tilde-only closer at least as long as the opener."""
    for i in range(start + 1, len(lines)):
        closer = lines[i].strip()
        if len(closer) >= opener_len and closer == '~' * len(closer):
            return i
    return None


def _is_table_border(stripped: str) -> bool:
    """Check whether a stripped line is a reST simple-table border."""
    return _TABLE_BORDER_RE.fullmatch(stripped) is not None


def _scan_aligned_rows(lines: list[str], start: int, code_block_lines: set[int]) -> int:
    """Protect consecutive code-labelled rows sharing a description column."""
    first = _ALIGNED_ROW_RE.fullmatch(lines[start])
    if first is None:
        return start + 1
    column = first.start('description')
    indent = first.group('indent')
    rows = 1
    i = start + 1
    while i < len(lines):
        line = lines[i]
        row = _ALIGNED_ROW_RE.fullmatch(line)
        if (
            row is not None
            and row.group('indent') == indent
            and row.start('description') == column
        ):
            rows += 1
        elif not line.strip() or len(line) - len(line.lstrip(' ')) != column:
            break
        i += 1
    if rows == 1:
        return start + 1
    code_block_lines.update(range(start, i))
    return i


def _scan_doctest(lines: list[str], start: int, code_block_lines: set[int]) -> int:
    """Scan a doctest block starting at a >>> prompt."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    i = start

    # Consume >>> and ... prompt lines at the same indent level
    while i < len(lines):
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        s = lines[i].lstrip()
        if line_indent == indent and s.startswith(('>>>', '...')):
            code_block_lines.add(i)
            i += 1
        else:
            break

    # Consume expected output lines (until blank line, next >>>, or dedent)
    while i < len(lines):
        if not lines[i].strip():
            break
        if lines[i].strip().startswith('>>>'):
            break
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent < indent:
            break
        code_block_lines.add(i)
        i += 1

    return i


def _scan_fenced(lines: list[str], start: int, code_block_lines: set[int]) -> int:
    """Scan a backtick- or tilde-fenced code block."""
    stripped = lines[start].strip()
    fence_char = stripped[0]
    opener_len = len(stripped) - len(stripped.lstrip(fence_char))

    # Reject backtick inline code spans (``` x = 1 ```)
    if fence_char == '`' and '`' in stripped[opener_len:]:
        return start + 1

    code_block_lines.add(start)
    i = start + 1
    while i < len(lines):
        code_block_lines.add(i)
        closer = lines[i].strip()
        if closer and set(closer) == {fence_char} and len(closer) >= opener_len:
            break
        i += 1
    return i + 1


def _scan_literal(lines: list[str], start: int, code_block_lines: set[int]) -> int:
    """Scan a reST literal block (introduced by ::)."""
    introducer_indent = len(lines[start]) - len(lines[start].lstrip())

    # Literal block requires a blank line after ::
    if start + 1 >= len(lines) or lines[start + 1].strip():
        return start + 1

    j = start + 2
    while j < len(lines):
        if not lines[j].strip():
            # Peek ahead: only include blank lines followed by more indented code
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines):
                peek_indent = len(lines[k]) - len(lines[k].lstrip())
                if peek_indent > introducer_indent:
                    for b in range(j, k):
                        code_block_lines.add(b)
                    j = k
                    continue
            break
        line_indent = len(lines[j]) - len(lines[j].lstrip())
        if line_indent > introducer_indent:
            code_block_lines.add(j)
            j += 1
        else:
            break
    return j


def _scan_simple_table(lines: list[str], start: int, code_block_lines: set[int]) -> int:
    """Scan a reST simple table opened by a border line."""
    closer: int | None = None
    opener_char = lines[start].strip()[0]
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if (
            _is_table_border(stripped)
            and stripped[0] == opener_char
            and (j + 1 >= len(lines) or not lines[j + 1].strip())
        ):
            closer = j
            if opener_char == '=':
                break
    if closer is not None:
        code_block_lines.update(range(start, closer + 1))
        return closer + 1
    return start + 1


def find_code_block_lines(lines: list[str], *, start: int = 0) -> set[int]:
    """Find line indices that are inside scanned code blocks.

    Detects fenced blocks (``` or ~~~), doctest blocks (>>>), and literal
    blocks (::), bordered simple tables, and aligned code-labelled rows.
    Body parsing separately classifies indented description blocks.

    Args:
        lines: Docstring content lines.
        start: Absolute line index where scanning begins.

    Returns:
        Set of 0-based line indices inside code blocks.
    """
    code_block_lines: set[int] = set()
    i = start
    while i < len(lines):
        stripped = lines[i].strip()

        # Tilde-fenced block. A tilde-only run directly under nonblank text
        # is a heading underline, not a fence opener.
        if stripped.startswith('~~~'):
            if set(stripped) == {'~'} and i > 0 and lines[i - 1].strip():
                i += 1
                continue
            opener_len = len(stripped) - len(stripped.lstrip('~'))
            closer = _find_tilde_closer(lines, i, opener_len)
            if closer is not None:
                code_block_lines.update(range(i, closer + 1))
                i = closer + 1
                continue
            i += 1
            continue

        # Backtick-fenced block
        if stripped.startswith('```'):
            i = _scan_fenced(lines, i, code_block_lines)
            continue

        # Doctest block
        if stripped.startswith('>>>'):
            i = _scan_doctest(lines, i, code_block_lines)
            continue

        # Literal block
        if stripped.endswith('::'):
            i = _scan_literal(lines, i, code_block_lines)
            continue

        # reST simple table (multi-run = or - border after a blank line)
        if _is_table_border(stripped) and (i == 0 or not lines[i - 1].strip()):
            i = _scan_simple_table(lines, i, code_block_lines)
            continue

        # Require a block boundary so inline-code prose is not an opener.
        if i == start or not lines[i - 1].strip():
            i = _scan_aligned_rows(lines, i, code_block_lines)
            continue

        i += 1

    return code_block_lines
