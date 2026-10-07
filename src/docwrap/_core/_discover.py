"""Discover and parse docstrings from Python syntax trees."""

import ast
import warnings
from dataclasses import dataclass

from docwrap._core._codeblock import find_code_block_lines
from docwrap._core._escape import EscapeScanError, shield_escapes
from docwrap._core._normalize import (
    PREFIX_RE,
    TRIPLE_QUOTE_LEN,
    LiteralInfo,
    byte_col_to_char_col,
    detect_original_position,
    is_own_line,
    normalize_content,
    parse_literal,
    split_source_lines,
)
from docwrap._core._parse import (
    Region,
    match_opening_title,
    parse_regions,
)
from docwrap._detect import detect_style
from docwrap._styles import Style, get_style, resolve_style_override
from docwrap._types import DocstringSkip, DocwrapConfig, StyleName, SummaryStart

_MODULE_QUALNAME = '<module>'
_UNDERLINE_TITLE_LINE_COUNT = 2


@dataclass(frozen=True, slots=True)
class DocstringInfo:
    """Information about a discovered docstring.

    Column offsets use AST expression coordinates, which can differ from the
    underlying literal coordinates for parenthesized docstrings. Literal
    slicing uses the coordinates on `info.node.value`.

    Attributes:
        node: AST expression containing the docstring literal.
        lineno: First source line of the literal.
        end_lineno: Final source line of the literal.
        col_offset: Byte offset of the expression on its first source line.
        end_col_offset: Byte offset immediately after the expression on its
            final source line.
        def_end_lineno: Final source line of the owning definition, or zero for
            a module docstring.
        def_lineno: First source line of the owning definition, or zero for a
            module docstring.
        qualname: Qualified name of the docstring owner.
    """

    node: ast.Expr
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    def_end_lineno: int = 0
    def_lineno: int = 0
    qualname: str = ''


@dataclass(frozen=True, slots=True)
class ParsedDocstring:
    """Pre-parsed docstring data ready for wrapping.

    Attributes:
        info: Discovery metadata from AST traversal.
        lit: Parsed literal components (prefix, quote char, content).
        regions: Typed regions parsed from the normalized content.
        style: Resolved style instance, or None if undetected.
        detected_name: Resolved style name, or None if undetected.
        base_indent: Indentation level of the line containing the docstring.
        col_offset: Character-based column offset of the string literal.
        content: Normalized docstring content.
        code_block_lines: Line indices treated as code blocks.
        original_position: Whether summary is on first or second line.
        content_start_offset: Number of leading blank lines stripped during
            normalization. Added to info.lineno to get the source line where
            normalized content begins.
        source_lines: Physical source lines spanned by the literal, without
            CR/LF terminators.
        source_margin: Exact leading whitespace before an own-line literal.
        escape_map: Ordered (identity, spelling) pairs shielded in content.
    """

    info: DocstringInfo
    lit: LiteralInfo
    regions: tuple[Region, ...]
    style: Style | None
    detected_name: StyleName | None
    base_indent: int
    col_offset: int
    content: str
    code_block_lines: frozenset[int]
    original_position: SummaryStart
    content_start_offset: int
    source_lines: tuple[str, ...]
    source_margin: str
    escape_map: tuple[tuple[str, str], ...] = ()


def _detect_style_after_title(
    content: str,
    code_block_lines: set[int],
    provisional_title_line_count: int | None,
    *,
    preferred: StyleName | None = None,
) -> StyleName | None:
    """Detect style using post-title evidence and eligible title fallback."""
    primary_start = provisional_title_line_count or 0
    fallback_line_indices = (
        range(_UNDERLINE_TITLE_LINE_COUNT)
        if provisional_title_line_count == _UNDERLINE_TITLE_LINE_COUNT
        else ()
    )
    return detect_style(
        content,
        code_block_lines,
        primary_start=primary_start,
        fallback_line_indices=fallback_line_indices,
        preferred=preferred,
    )


def _literal_source_context(
    source_lines: list[str],
    raw_lineno: int,
    raw_end_lineno: int,
) -> tuple[str, int, str, tuple[str, ...]]:
    """Return the literal source line, indent, margin, and physical lines."""
    doc_line = source_lines[raw_lineno - 1].rstrip('\r\n')
    base_indent = len(doc_line) - len(doc_line.lstrip())
    return (
        doc_line,
        base_indent,
        doc_line[:base_indent],
        tuple(line.rstrip('\r\n') for line in source_lines[raw_lineno - 1 : raw_end_lineno]),
    )


def _record_unmappable_literal(
    info: DocstringInfo,
    is_triple: bool,
    skips: list[DocstringSkip],
) -> None:
    """Record an unmappable triple-quoted literal."""
    if not is_triple:
        return
    skips.append(
        DocstringSkip(
            path='',
            line=info.lineno,
            qualname=info.qualname,
            code='implicit-concatenation',
            message='implicitly concatenated docstring literal',
        )
    )


def _shield_literal_content(
    lit: LiteralInfo,
    info: DocstringInfo,
    skips: list[DocstringSkip],
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """Shield a non-raw literal body, or return None when scanning fails."""
    work_body = lit.content
    escape_map: tuple[tuple[str, str], ...] = ()
    if 'r' in lit.prefix.lower() or '\\' not in work_body:
        return work_body, escape_map
    try:
        shielded = shield_escapes(work_body)
    except EscapeScanError as error:
        skips.append(
            DocstringSkip(
                path='',
                line=info.lineno,
                qualname=info.qualname,
                code=error.code,
                message=str(error),
            )
        )
        return None
    return shielded.text, shielded.escape_map


def _target_style_has_body_evidence(
    content: str,
    code_block_lines: set[int],
    source_style: Style,
    target: StyleName,
) -> bool:
    """Return whether target markers remain after the source-style summary."""
    source_regions = parse_regions(
        content,
        style=source_style,
        code_block_lines=code_block_lines,
    )
    summary = next((region for region in source_regions if region.kind == 'summary'), None)
    if summary is None or summary.source_start is None:
        return True
    return (
        detect_style(
            content,
            code_block_lines,
            primary_start=summary.source_start + summary.source_line_count,
            preferred=target,
        )
        == target
    )


def _resolve_parsing_style(
    content: str,
    code_block_lines: set[int],
    provisional_title_line_count: int | None,
    config: DocwrapConfig,
    style_override: Style | None,
) -> tuple[Style | None, StyleName | None]:
    """Resolve the parser style and reported detection for one docstring."""
    if style_override is not None and config.convert is None:
        return style_override, config.style

    detected_name = _detect_style_after_title(
        content,
        code_block_lines,
        provisional_title_line_count,
        preferred=config.convert,
    )
    if (
        style_override is not None
        and detected_name == config.convert
        and config.convert is not None
        and not _target_style_has_body_evidence(
            content,
            code_block_lines,
            style_override,
            config.convert,
        )
    ):
        return style_override, config.style
    if style_override is not None and detected_name != config.convert:
        return style_override, config.style
    return (get_style(detected_name) if detected_name else None), detected_name


def definition_key(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int]:
    """Return the decorator-inclusive definition extent, or (0, 0) for a module."""
    if isinstance(node, ast.Module):
        return (0, 0)
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    return (start, node.end_lineno or node.lineno)


def docstring_expression(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    """Return the leading string expression of a docstring-owning scope, if any."""
    if (
        node.body
        and isinstance(first := node.body[0], ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def parse_docstrings(
    source: str,
    docstrings: list[DocstringInfo],
    config: DocwrapConfig,
    source_lines: list[str] | None = None,
    *,
    skips: list[DocstringSkip],
) -> list[ParsedDocstring]:
    """Parse discovered docstrings into structured representations.

    Extracts the parse phase from docstring processing: literal parsing,
    content normalization, style detection, region parsing, and position
    detection. Does not perform wrapping or reassembly.

    Args:
        source: Python source code string.
        docstrings: Docstring info list from `find_docstrings`.
        config: Wrapping configuration.
        source_lines: Physical lines retaining their LF, CRLF, or CR
            terminators, as returned by `split_source_lines`. If `None`, splits
            source internally. Pass this to avoid redundant splitting when the
            caller already has the lines.
        skips: Accumulator for `DocstringSkip` records with an empty path for
            escape-safety skips and unmappable implicitly concatenated
            triple-quoted literals.

    Returns:
        List of `ParsedDocstring` for each processable triple-quoted docstring.
        Single-quoted and unmappable docstrings are skipped.
    """
    if source_lines is None:
        source_lines = split_source_lines(source)

    style_override = resolve_style_override(config.style)

    results: list[ParsedDocstring] = []

    for info in docstrings:
        value_node = info.node.value
        if not isinstance(value_node, ast.Constant):
            continue

        raw_lineno = value_node.lineno
        raw_end_lineno = value_node.end_lineno or raw_lineno
        raw_col = value_node.col_offset
        raw_end_col = value_node.end_col_offset or 0
        doc_line, base_indent, source_margin, literal_source_lines = _literal_source_context(
            source_lines,
            raw_lineno,
            raw_end_lineno,
        )
        # Use the literal's character column to distinguish own-line source
        # margins from inline numeric-space continuation geometry.
        col_offset = byte_col_to_char_col(doc_line, value_node.col_offset)
        if raw_lineno == raw_end_lineno:
            raw_literal = (
                source_lines[raw_lineno - 1].encode('utf-8')[raw_col:raw_end_col].decode('utf-8')
            )
        else:
            first_part = source_lines[raw_lineno - 1].encode('utf-8')[raw_col:].decode('utf-8')
            last_part = (
                source_lines[raw_end_lineno - 1].encode('utf-8')[:raw_end_col].decode('utf-8')
            )
            middle = source_lines[raw_lineno : raw_end_lineno - 1]
            raw_literal = ''.join([first_part, *middle, last_part])

        # Distinguish single-quoted literals from unmappable triples.
        m = PREFIX_RE.match(raw_literal)
        is_triple = m is not None and len(m.group(2)) == TRIPLE_QUOTE_LEN

        if not isinstance(value_node.value, str):
            continue
        lit = parse_literal(raw_literal)
        if lit is None:
            # Record unmappable triples; skip single-quoted strings silently.
            _record_unmappable_literal(info, is_triple, skips)
            continue

        # Shield escapes so the engine wraps them as unbreakable words
        shielded_content = _shield_literal_content(lit, info, skips)
        if shielded_content is None:
            continue
        work_body, escape_map = shielded_content

        # Normalize, resolve title precedence, detect code blocks and style
        content, content_start_offset = normalize_content(
            work_body,
            source_margin=source_margin,
            source_margin_candidate=(
                None if is_own_line(col_offset, source_margin) else ' ' * base_indent
            ),
        )
        content_lines = content.split('\n')
        provisional_title_line_count = match_opening_title(
            content_lines,
            style=None,
        )
        code_block_lines = find_code_block_lines(
            content_lines,
            start=provisional_title_line_count or 0,
        )

        style, detected_name = _resolve_parsing_style(
            content,
            code_block_lines,
            provisional_title_line_count,
            config,
            style_override,
        )

        if (
            provisional_title_line_count is not None
            and match_opening_title(content_lines, style=style) is None
        ):
            code_block_lines = find_code_block_lines(content_lines, start=0)

        regions = parse_regions(content, style=style, code_block_lines=code_block_lines)

        original_position = detect_original_position(raw_literal, lit)

        results.append(
            ParsedDocstring(
                info=info,
                lit=lit,
                regions=tuple(regions),
                style=style,
                detected_name=detected_name,
                base_indent=base_indent,
                col_offset=col_offset,
                content=content,
                code_block_lines=frozenset(code_block_lines),
                original_position=original_position,
                content_start_offset=content_start_offset,
                source_lines=literal_source_lines,
                source_margin=source_margin,
                escape_map=escape_map,
            )
        )

    return results


def parse_source(source: str | bytes) -> ast.Module:
    """Parse source without emitting compiler warnings about user code.

    Args:
        source: Python source text or bytes with their original encoding
            cookie.

    Returns:
        Parsed module tree.

    Raises:
        SyntaxError: If the source is invalid independently of warning filters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return ast.parse(source)


def find_docstrings(source: str) -> list[DocstringInfo]:
    """Find all docstrings in Python source via AST.

    Args:
        source: Python source code string.

    Returns:
        List of DocstringInfo sorted by (lineno, col_offset).

    Raises:
        SyntaxError: If the source cannot be parsed.
    """
    tree = parse_source(source)
    stack: list[tuple[ast.AST, tuple[str, ...]]] = [(tree, ())]
    results: list[DocstringInfo] = []

    while stack:
        node, parts = stack.pop()

        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if isinstance(node, ast.Module):
                next_parts = parts
                qualname = _MODULE_QUALNAME
            else:
                next_parts = (*parts, node.name)
                qualname = '.'.join(next_parts)

            if (first := docstring_expression(node)) is not None:
                d_lineno, d_end_lineno = definition_key(node)
                results.append(
                    DocstringInfo(
                        node=first,
                        lineno=first.value.lineno,
                        end_lineno=first.value.end_lineno or first.value.lineno,
                        col_offset=first.col_offset,
                        end_col_offset=first.end_col_offset or 0,
                        def_end_lineno=d_end_lineno,
                        def_lineno=d_lineno,
                        qualname=qualname,
                    )
                )

        else:
            next_parts = parts

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                stack.append((child, next_parts))
            elif isinstance(child, ast.ExceptHandler | ast.match_case):
                stack.extend((statement, next_parts) for statement in child.body)

    results.sort(key=lambda d: (d.lineno, d.col_offset))
    return results
