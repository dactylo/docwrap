"""Orchestrate docstring parsing, wrapping, and source reassembly."""

import ast
import os
import tokenize
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from docwrap._config import resolve_config
from docwrap._convert import (
    DuplicateFieldError,
    UnrepresentableReturnError,
    convert_regions,
    preserve_multi_name_attribute_spans,
)
from docwrap._core._discover import (
    DocstringInfo,
    ParsedDocstring,
    find_docstrings,
    parse_docstrings,
)
from docwrap._core._escape import (
    EscapeRestoreError,
    EscapeScanError,
    restore_escapes,
    scan_escape_spellings,
)
from docwrap._core._normalize import (
    byte_col_to_char_col,
    is_own_line,
    reassemble,
    split_source_lines,
)
from docwrap._core._wrap import (
    WrapState,
    aggregate_style,
    compute_quote_overhead,
    detect_summary_skip,
    is_oneliner_skip,
    wrap_regions,
)
from docwrap._detect import detect_style
from docwrap._exceptions import DocwrapSkipWarning
from docwrap._files import read_file_result, write_preserving_newline
from docwrap._styles import Style, get_style, resolve_style_override
from docwrap._types import (
    DetailLevel,
    DocstringChange,
    DocstringSkip,
    DocwrapConfig,
    EligibleTarget,
    SkipCode,
    WrapCheckResult,
    WrapReport,
)
from docwrap._validate import validate_config

if TYPE_CHECKING:
    from docwrap._core._parse import Region

_VALID_DETAILS = frozenset({'summary', 'list', 'diff'})


@dataclass(frozen=True, slots=True)
class DocstringWrapResult:
    """Result of wrapping one parsed docstring.

    Args:
        new_literal: Rewritten literal, or None if unchanged.
        summary_skipped: Whether the summary was preserved over wrap target.
        summary_width: Width of the skipped summary, or None.
        converted: Whether the docstring was converted between styles.
        skip: Safety-skip record when validation reverted this docstring.
    """

    new_literal: str | None
    summary_skipped: bool
    summary_width: int | None
    converted: bool
    skip: DocstringSkip | None = None

    @property
    def changed(self) -> bool:
        """Whether wrapping changed the literal.

        Returns:
            Whether a rewritten literal is present.
        """
        return self.new_literal is not None


@dataclass(frozen=True, slots=True)
class SourceAnalysis:
    """Result of source-level docstring discovery and parsing.

    Args:
        docstrings: Discovered docstring metadata.
        parsed: Parsed docstrings with regions and content.
        parse_error: Whether a syntax error prevented parsing.
        skips: Safety-skip records emitted during parsing.
    """

    docstrings: list[DocstringInfo]
    parsed: list[ParsedDocstring]
    parse_error: bool
    skips: tuple[DocstringSkip, ...] = ()


def _build_change_record(
    *,
    info: DocstringInfo,
    parsed: ParsedDocstring,
    result: DocstringWrapResult,
    detail: DetailLevel,
) -> DocstringChange:
    """Build a per-docstring change record for the WrapReport."""
    before: str | None = None
    after: str | None = None
    if detail == 'diff' and result.new_literal is not None:
        indent = (
            parsed.source_margin
            if is_own_line(parsed.col_offset, parsed.source_margin)
            else ' ' * parsed.col_offset
        )
        before = indent + parsed.lit.raw_literal
        after = indent + result.new_literal
    return DocstringChange(
        qualname=info.qualname,
        start_line=info.lineno,
        end_line=info.end_lineno,
        changed=result.changed,
        before=before,
        after=after,
        converted=result.converted,
        summary_skipped=result.summary_skipped,
        summary_width=result.summary_width,
    )


def _convert_parsed_regions(
    parsed: ParsedDocstring,
    config: DocwrapConfig,
    wrap_style: Style | None,
) -> 'tuple[list[Region], Style | None, bool, list[str]]':
    """Apply requested conversion or type stripping to parsed regions."""
    regions: list[Region] = [
        replace(region, preserve_verbatim=True)
        if config.convert is not None
        and parsed.detected_name == config.convert
        and region.kind in ('description', 'prose')
        and (marker_style := detect_style(region.text)) is not None
        and marker_style != config.convert
        else region
        for region in parsed.regions
    ]
    deleted_text: list[str] = []
    if parsed.style is None or parsed.detected_name is None:
        return regions, wrap_style, False, deleted_text
    if config.convert is not None and parsed.detected_name == config.convert:
        regions = preserve_multi_name_attribute_spans(
            regions,
            parsed.style,
            get_style(config.convert),
        )
    if config.convert is not None and parsed.detected_name != config.convert:
        target_style = get_style(config.convert)
        regions = convert_regions(
            regions,
            parsed.style,
            target_style,
            strip_types=config.strip_types,
            deleted_text=deleted_text,
        )
        return regions, target_style, True, deleted_text
    if config.strip_types:
        regions = convert_regions(
            regions,
            parsed.style,
            parsed.style,
            strip_types=True,
            deleted_text=deleted_text,
        )
    return regions, wrap_style, False, deleted_text


def _filter_eligible(
    docstrings: list[DocstringInfo],
    eligible_ranges: set[tuple[int, int]] | None,
    eligible_target: EligibleTarget,
) -> list[DocstringInfo]:
    """Filter docstrings by eligible (start, end) line ranges."""
    if eligible_ranges is None:
        return docstrings
    if eligible_target == 'definition':
        return [d for d in docstrings if (d.def_lineno, d.def_end_lineno) in eligible_ranges]
    return [d for d in docstrings if (d.lineno, d.end_lineno) in eligible_ranges]


def _has_protected_region(regions: 'list[Region]') -> bool:
    """Report whether any region preserves semantic leading whitespace."""
    return any(
        r.preserve_verbatim or r.source_scanner_protected or r.kind == 'code_block'
        for r in regions
    )


def _parses_cleanly(source: str) -> bool:
    """Check whether source parses, suppressing escape warnings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


def _skip_record(parsed: ParsedDocstring, code: SkipCode, message: str) -> DocstringSkip:
    """Build a path-less DocstringSkip for one parsed docstring."""
    return DocstringSkip(
        path='',
        line=parsed.info.lineno,
        qualname=parsed.info.qualname,
        code=code,
        message=message,
    )


def _splice_literal(
    source_lines: list[str],
    info: DocstringInfo,
    new_literal: str,
) -> None:
    """Splice a new literal into source_lines using byte-offset-to-char conversion."""
    value_node = info.node.value
    v_lineno = value_node.lineno
    v_end_lineno = value_node.end_lineno or v_lineno
    v_col_offset = value_node.col_offset
    v_end_col_offset = value_node.end_col_offset or 0

    first_line_idx = v_lineno - 1
    last_line_idx = v_end_lineno - 1

    first_line = source_lines[first_line_idx]
    char_start = byte_col_to_char_col(first_line, v_col_offset)

    last_line = source_lines[last_line_idx]
    char_end = byte_col_to_char_col(last_line, v_end_col_offset)

    before = first_line[:char_start]
    after = last_line[char_end:]

    spliced = before + new_literal + after
    source_lines[first_line_idx : last_line_idx + 1] = split_source_lines(spliced)


def _validate_pre_wrapped_options(config: DocwrapConfig, pre_wrapped: str | None) -> None:
    """Validate options that cannot be combined with pre-wrapped content."""
    if pre_wrapped is None:
        return
    if config.convert is not None:
        msg = 'pre_wrapped cannot be used with convert'
        raise ValueError(msg)
    if config.strip_types:
        msg = 'pre_wrapped cannot be used with strip_types'
        raise ValueError(msg)


def _validate_report_args(config: DocwrapConfig, detail: DetailLevel) -> None:
    """Revalidate config invariants and the detail level at the operation boundary."""
    validate_config(config)
    if detail not in _VALID_DETAILS:
        msg = f'detail must be one of {sorted(_VALID_DETAILS)}, got {detail!r}'
        raise ValueError(msg)


def _validate_rewrite(
    parsed: ParsedDocstring, new_literal: str, *, allowed_spellings: 'Counter[str]'
) -> DocstringSkip | None:
    """Validate escape preservation and syntax for a rewritten literal."""
    if 'r' not in parsed.lit.prefix.lower():
        try:
            spellings = scan_escape_spellings(new_literal)
        except EscapeScanError:
            return _skip_record(
                parsed, 'validation-failed', 'escape scan failed on rewritten literal'
            )
        expected = Counter(spelling for _, spelling in parsed.escape_map)
        actual = Counter(spellings)
        extra = actual - expected
        lost = expected - actual
        if extra or lost - allowed_spellings:
            return _skip_record(
                parsed, 'validation-failed', 'escape spellings changed during rewrite'
            )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            ast.parse(new_literal, mode='eval')
    except (SyntaxError, ValueError):
        return _skip_record(parsed, 'validation-failed', 'rewritten literal does not compile')
    return None


def analyze_source(source: str, config: DocwrapConfig) -> SourceAnalysis:
    """Discover and parse all docstrings in source.

    Combines find_docstrings and parse_docstrings into a single call.
    Does NOT filter by eligible ranges (use wrap_source_report_eligible for
    that).

    Args:
        source: Python source code string.
        config: Wrapping configuration (for style resolution).

    Returns:
        SourceAnalysis with discovered docstrings, parsed results, and parse
        error flag.
    """
    try:
        docstrings = find_docstrings(source)
    except SyntaxError:
        return SourceAnalysis(docstrings=[], parsed=[], parse_error=True)
    parse_skips: list[DocstringSkip] = []
    parsed = parse_docstrings(source, docstrings, config, skips=parse_skips)
    return SourceAnalysis(
        docstrings=docstrings,
        parsed=parsed,
        parse_error=False,
        skips=tuple(parse_skips),
    )


def wrap_parsed_docstring(
    parsed: ParsedDocstring,
    config: DocwrapConfig,
    style_override: Style | None,
    pre_wrapped: str | None = None,
) -> DocstringWrapResult:
    """Wrap a pre-parsed docstring without re-parsing source.

    Takes the output of parse_docstrings and applies the wrapping phase:
    conversion, region wrapping, summary skip detection, reassembly, one-liner
    skip, and change detection.

    Args:
        parsed: Pre-parsed docstring from parse_docstrings.
        config: Wrapping configuration.
        style_override: Explicit style instance, or None for auto-detected.
        pre_wrapped: Pre-computed wrapped content to use instead of calling
            wrap_regions. Cannot be combined with convert or strip_types.

    Returns:
        DocstringWrapResult with the rewritten literal (or None if unchanged),
        summary skip status and width, and conversion flag.

    Raises:
        ValueError: If `pre_wrapped` is combined with conversion or type
            stripping.
    """
    # Preserve the target parser's geometry after conversion.
    if config.convert is not None and parsed.detected_name == config.convert:
        wrap_style = parsed.style
    else:
        wrap_style = style_override if style_override is not None else parsed.style

    # Guard: pre_wrapped bypasses conversion, so combining is an error.
    _validate_pre_wrapped_options(config, pre_wrapped)

    # Convert regions when --convert or --strip-types is active.
    # Work on a local copy so parsed.regions stays untouched.
    try:
        regions, wrap_style, converted, deleted_text = _convert_parsed_regions(
            parsed, config, wrap_style
        )
    except (DuplicateFieldError, UnrepresentableReturnError) as error:
        skip_code = (
            'duplicate-field'
            if isinstance(error, DuplicateFieldError)
            else 'unrepresentable-return'
        )
        return DocstringWrapResult(
            new_literal=None,
            summary_skipped=False,
            summary_width=None,
            converted=False,
            skip=_skip_record(parsed, skip_code, str(error)),
        )

    deleted = ''.join(deleted_text)
    allowed_identities = frozenset(
        identity for identity, _ in parsed.escape_map if identity in deleted
    )

    quote_overhead = compute_quote_overhead(parsed, config)

    if pre_wrapped is not None:
        wrapped = pre_wrapped
    else:
        wrapped = wrap_regions(
            regions,
            config,
            parsed.base_indent,
            style=wrap_style,
            quote_overhead=quote_overhead,
            col_offset=parsed.col_offset,
        )

    summary_skipped, summary_width = detect_summary_skip(
        regions,
        config,
        base_indent=parsed.base_indent,
        col_offset=parsed.col_offset,
        quote_overhead=quote_overhead,
    )

    if parsed.escape_map:
        try:
            wrapped = restore_escapes(wrapped, parsed.escape_map, allowed_loss=allowed_identities)
        except EscapeRestoreError as error:
            return DocstringWrapResult(
                new_literal=None,
                summary_skipped=summary_skipped,
                summary_width=summary_width,
                converted=False,
                skip=_skip_record(parsed, 'validation-failed', str(error)),
            )

    if '\n' not in wrapped and not _has_protected_region(regions):
        wrapped = wrapped.lstrip()

    source_margin = (
        parsed.source_margin if is_own_line(parsed.col_offset, parsed.source_margin) else None
    )
    new_literal = reassemble(
        wrapped,
        parsed.lit.quote_char,
        parsed.lit.prefix,
        parsed.base_indent,
        config,
        col_offset=parsed.col_offset,
        original_position=parsed.original_position,
        source_margin=source_margin,
    )

    # A None reassembly means no safe quote style exists: a structured skip,
    # never a silent no-op.
    if new_literal is None:
        return DocstringWrapResult(
            new_literal=None,
            summary_skipped=summary_skipped,
            summary_width=summary_width,
            converted=False,
            skip=_skip_record(
                parsed, 'quote-collision', 'no safe quote style for rewritten content'
            ),
        )
    # Suppress change for summary-only docstrings to avoid D200 violation,
    # or when reassemble found no meaningful change.
    if new_literal == parsed.lit.raw_literal.replace('\r\n', '\n').replace(
        '\r', '\n'
    ) or is_oneliner_skip(summary_skipped, new_literal, regions):
        return DocstringWrapResult(
            new_literal=None,
            summary_skipped=summary_skipped,
            summary_width=summary_width,
            converted=False,
        )
    validation_skip = _validate_rewrite(
        parsed,
        new_literal,
        allowed_spellings=Counter(
            spelling for identity, spelling in parsed.escape_map if identity in allowed_identities
        ),
    )
    if validation_skip is not None:
        return DocstringWrapResult(
            new_literal=None,
            summary_skipped=summary_skipped,
            summary_width=summary_width,
            converted=False,
            skip=validation_skip,
        )
    return DocstringWrapResult(
        new_literal=new_literal,
        summary_skipped=summary_skipped,
        summary_width=summary_width,
        converted=converted,
    )


def _apply_wraps(  # noqa: PLR0913
    docstrings: list[DocstringInfo],
    parsed_map: dict[int, ParsedDocstring],
    source_lines: list[str],
    config: DocwrapConfig,
    style_override: Style | None,
    *,
    detail: DetailLevel,
) -> WrapState:
    """Process docstrings bottom-up, wrapping each and splicing the source."""
    state = WrapState(changes=[], detected_styles=set())
    for info in reversed(docstrings):
        parsed = parsed_map.get(id(info))
        if parsed is None:
            # Unparseable docstring (single-quoted, non-string, unmappable)
            if detail != 'summary':
                state.changes.append(
                    DocstringChange(
                        qualname=info.qualname,
                        start_line=info.lineno,
                        end_line=info.end_lineno,
                        changed=False,
                    )
                )
            continue

        result = wrap_parsed_docstring(parsed, config, style_override)
        if result.skip is not None:
            state.skips.append(result.skip)
        if parsed.detected_name is not None:
            state.detected_styles.add(parsed.detected_name)

        if (new_literal := result.new_literal) is not None:
            state.wrapped += 1
            _splice_literal(source_lines, info, new_literal)
        if result.converted:
            state.converted_count += 1
        if result.summary_skipped:
            state.skipped_summaries += 1

        if detail != 'summary':
            state.changes.append(
                _build_change_record(
                    info=info,
                    parsed=parsed,
                    result=result,
                    detail=detail,
                )
            )

    # Changes were collected in reverse order; restore original order
    state.changes.reverse()
    return state


def wrap_source_report_eligible(
    source: str,
    *,
    config: DocwrapConfig,
    detail: DetailLevel = 'summary',
    eligible_ranges: set[tuple[int, int]] | None,
    eligible_target: EligibleTarget,
) -> WrapReport:
    """Wrap docstrings with hook/CLI eligibility filtering.

    Internal core API for the CLI and hook incremental paths. Public callers
    use `wrap_source_report`, which wraps every docstring.

    Args:
        source: Python source code string.
        config: Wrapping configuration.
        detail: `summary` omits per-docstring records; `list` includes records
            without literal text; `diff` also includes before/after text for
            changed docstrings.
        eligible_ranges: Set of inclusive `(start, end)` line tuples
            identifying eligible definitions or docstrings. `None` selects all
            docstrings;
            an empty set selects none.
        eligible_target: Whether `eligible_ranges` refers to `definition` or
            `docstring` line ranges.

    Returns:
        `WrapReport` with transformed source and change metadata. A Python
        parse failure returns unchanged source with `parse_error=True`.
        Safety skips leave affected content unchanged and populate `skips` at
        every detail level.

    Raises:
        InvalidLineLengthError: If `line_length` is invalid.
        TypeError: If a boolean-typed configuration field is not a bool.
        UnknownStyleError: If `style` or `convert` is not recognized.
        ValueError: If `detail` or a configuration field value is invalid.
    """
    _validate_report_args(config, detail)

    # Parse -- on failure, return error report
    try:
        docstrings = find_docstrings(source)
    except SyntaxError:
        return WrapReport(source=source, total=0, wrapped=0, parse_error=True, style=config.style)

    docstrings = _filter_eligible(docstrings, eligible_ranges, eligible_target)
    if not docstrings:
        return WrapReport(source=source, total=0, wrapped=0, style=config.style)

    style_override = resolve_style_override(config.style)

    # Parse phase: extract all docstring data in one pass
    source_lines = split_source_lines(source)
    parse_skips: list[DocstringSkip] = []
    parsed_list = parse_docstrings(
        source, docstrings, config, source_lines=source_lines, skips=parse_skips
    )
    parsed_map = {id(p.info): p for p in parsed_list}
    state = _apply_wraps(
        docstrings,
        parsed_map,
        source_lines,
        config,
        style_override,
        detail=detail,
    )

    # Aggregate file-level style
    file_style = aggregate_style(config.style, state.detected_styles)

    skips = sorted([*parse_skips, *state.skips], key=lambda s: s.line)
    new_source = ''.join(source_lines)
    if state.wrapped and not _parses_cleanly(new_source):
        skips.append(
            DocstringSkip(
                path='',
                line=0,
                qualname='',
                code='file-validation-failed',
                message='rewritten file does not parse; all changes reverted',
            )
        )
        return WrapReport(
            source=source,
            total=len(docstrings),
            wrapped=0,
            convert_target=config.convert,
            style=file_style,
            skips=tuple(skips),
        )

    return WrapReport(
        source=new_source,
        total=len(docstrings),
        wrapped=state.wrapped,
        convert_target=config.convert,
        converted=state.converted_count,
        docstrings=tuple(state.changes),
        skipped_summaries=state.skipped_summaries,
        style=file_style,
        skips=tuple(skips),
    )


def wrap_source_report(
    source: str,
    *,
    config: DocwrapConfig | None = None,
    detail: DetailLevel = 'summary',
) -> WrapReport:
    """Wrap docstrings in Python source, returning transformed source and metadata.

    Args:
        source: Python source code string.
        config: Wrapping configuration, or `None` to use `DocwrapConfig()`.
        detail: `summary` omits per-docstring records; `list` includes records
            without literal text; `diff` also includes before/after text for
            changed docstrings.

    Returns:
        `WrapReport` with transformed source and change metadata. A Python
        parse failure returns unchanged source with `parse_error=True`.
        Safety skips leave affected content unchanged and populate `skips` at
        every detail level.

    Raises:
        InvalidLineLengthError: If `line_length` is invalid.
        TypeError: If a boolean-typed configuration field is not a bool.
        UnknownStyleError: If `style` or `convert` is not recognized.
        ValueError: If `detail` or a configuration field value is invalid.
    """
    effective = config if config is not None else DocwrapConfig()
    return wrap_source_report_eligible(
        source,
        config=effective,
        detail=detail,
        eligible_ranges=None,
        eligible_target='definition',
    )


def check(
    source: str,
    *,
    config: DocwrapConfig | None = None,
) -> WrapCheckResult:
    """Return whether wrap_source() would change source, without modifying it.

    Args:
        source: Python source code to analyze.
        config: Resolved DocwrapConfig; defaults to DocwrapConfig() when None.

    Returns:
        WrapCheckResult with status 'invalid', 'changed', or 'unchanged'.

    Raises:
        InvalidLineLengthError: If line_length is invalid.
        TypeError: If a boolean-typed configuration field is not a bool.
        UnknownStyleError: If style is not recognized.
        ValueError: If a configuration field value is invalid.
    """
    base = config if config is not None else DocwrapConfig()
    try:
        report = wrap_source_report(source, config=base)
    except (SyntaxError, tokenize.TokenError):
        return WrapCheckResult(status='invalid')
    if report.parse_error:
        return WrapCheckResult(status='invalid')
    if report.source != source:
        return WrapCheckResult(status='changed', skips=report.skips)
    return WrapCheckResult(status='unchanged', skips=report.skips)


def wrap_file_report(
    path: str | os.PathLike[str],
    *,
    config: DocwrapConfig | None = None,
    detail: DetailLevel = 'summary',
    write: bool = False,
) -> WrapReport:
    """Wrap docstrings in a Python file.

    Reads the file respecting its encoding cookie and normalizes source
    newlines to LF. Optional writes use the original encoding and selected
    newline style: CRLF when the input contains CRLF, otherwise LF.

    Args:
        path: Path to a Python source file.
        config: Wrapping configuration, or `None` to resolve from
            `pyproject.toml`.
        detail: `summary` omits per-docstring records; `list` includes records
            without literal text; `diff` also includes before/after text for
            changed docstrings.
        write: If `True`, write changed source back when parsing succeeded.

    Returns:
        `WrapReport` with LF-normalized source and change metadata. A Python
        parse failure returns the unmodified normalized source with
        `parse_error=True` and does not write the file. Safety skips populate
        `skips` at every detail level, with their path set to the supplied
        file.

    Raises:
        InvalidLineLengthError: If `line_length` is invalid.
        OSError: If the file cannot be read or written.
        SyntaxError: If the Python source has a malformed or unknown encoding
            cookie.
        TypeError: If a boolean-typed configuration field is not a bool.
        UnicodeDecodeError: If source bytes are invalid for the detected
            encoding.
        UnknownStyleError: If `style` or `convert` is not recognized.
        ValueError: If `detail` or a configuration field value is invalid.
    """
    file_path = Path(path)
    effective = config if config is not None else resolve_config(file_path)

    read_result = read_file_result(file_path)
    source = read_result.source

    report = wrap_source_report(
        source,
        config=effective,
        detail=detail,
    )

    if report.skips:
        report = replace(
            report,
            skips=tuple(replace(s, path=str(file_path)) for s in report.skips),
        )

    if write and report.source != source and not report.parse_error:
        write_preserving_newline(
            file_path,
            report.source,
            read_result.encoding,
            raw=read_result.raw,
        )

    return report


def wrap_source(
    source: str,
    *,
    config: DocwrapConfig | None = None,
) -> str:
    """Wrap docstrings in Python source.

    Args:
        source: Python source code string.
        config: Wrapping configuration (None = use defaults).

    Returns:
        Source with docstrings wrapped. Returns source unchanged on parse
        errors. Emits `DocwrapSkipWarning` for each safety-skipped docstring;
        use `wrap_source_report` for structured skip records.

    Raises:
        InvalidLineLengthError: If line_length is invalid.
        TypeError: If a boolean-typed configuration field is not a bool.
        UnknownStyleError: If style is not recognized.
        ValueError: If a configuration field value is invalid.
    """
    effective = config if config is not None else DocwrapConfig()
    report = wrap_source_report(
        source,
        config=effective,
    )
    for skip in report.skips:
        warnings.warn(
            f'line {skip.line}: {skip.qualname} skipped ({skip.code}): {skip.message}',
            DocwrapSkipWarning,
            stacklevel=2,
        )
    return report.source
