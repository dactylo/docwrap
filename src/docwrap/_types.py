"""Public data types for the docwrap library API."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast, get_args

CanonicalRole = Literal[
    'param',
    'keyword_param',
    'other_param',
    'returns',
    'yields',
    'raises',
    'warns',
    'attributes',
    'methods',
    'see_also',
    'notes',
    'references',
    'examples',
    'todo',
    'warnings',
]
ColorMode = Literal['auto', 'always', 'never']
DetailLevel = Literal['summary', 'list', 'diff']
EligibleTarget = Literal['definition', 'docstring']
FillStyle = Literal['balanced', 'greedy', 'smart']
IncrementalLevel = Literal['file', 'definition', 'docstring']
JsonStatus = Literal['changed', 'unchanged', 'error', 'invalid']
MeasurementCategory = Literal['summary', 'prose', 'args', 'code_block', 'source']
Newline = Literal['\n', '\r\n']
OutputMode = Literal['preview', 'diff', 'stats', 'write', 'json', 'summary']
RegionKind = Literal[
    'summary',
    'blank',
    'description',
    'section_header',
    'entry',
    'prose',
    'code_block',
    'list_item',
    'title',
]
SphinxFieldRole = Literal[
    'param',
    'keyword',
    'type',
    'vartype',
    'rtype',
    'raises',
    'returns',
    'yields',
    'warns',
    'ivar',
    'cvar',
    'var',
]
StatsSection = Literal[
    'source',
    'docstrings',
    'distribution',
    'longest',
    'impact',
    'recommendations',
]
StyleName = Literal['google', 'numpy', 'sphinx']
ReflowMode = Literal['none', 'cautious', 'full']
SectionKind = Literal['entry', 'prose']
SkipCode = Literal[
    'duplicate-field',
    'file-validation-failed',
    'implicit-concatenation',
    'line-continuation',
    'quote-collision',
    'sentinel-unavailable',
    'unrepresentable-return',
    'validation-failed',
]
SummaryStart = Literal['preserve', 'first-line', 'second-line']
WrapStatus = Literal['changed', 'unchanged', 'invalid']

COLOR_MODES = cast('tuple[ColorMode, ...]', get_args(ColorMode))
DEFAULT_LINE_LENGTH = 79
DEFAULT_STATS_LONGEST_PER_CATEGORY = 3
DEFAULT_STATS_SECTIONS = cast('tuple[StatsSection, ...]', get_args(StatsSection))
FILL_STYLES = cast('tuple[FillStyle, ...]', get_args(FillStyle))
INCREMENTAL_LEVELS = cast('tuple[IncrementalLevel, ...]', get_args(IncrementalLevel))
REFLOW_MODES = cast('tuple[ReflowMode, ...]', get_args(ReflowMode))
SUMMARY_STARTS = cast('tuple[SummaryStart, ...]', get_args(SummaryStart))


class BaselineStatus(enum.Enum):
    """Disposition of a Git-index baseline read.

    Attributes:
        CONTENT: The baseline was read and decoded.
        DECODE_FAILED: The baseline bytes could not be decoded as Python.
        INFRA_ERROR: Git or filesystem infrastructure failed.
        MISSING: The path has no baseline entry.
    """

    CONTENT = 'content'
    DECODE_FAILED = 'decode_failed'
    INFRA_ERROR = 'infra_error'
    MISSING = 'missing'


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineResult:
    """Per-file result of a git-index baseline read.

    Args:
        status: Disposition of the read.
        content: Decoded source text (only when status is CONTENT).
        encoding: Detected encoding (defaults to utf-8).
        newline: Original blob newline style (defaults to LF).
    """

    status: BaselineStatus
    content: str | None = None
    encoding: str = 'utf-8'
    newline: Newline = '\n'


@dataclass(frozen=True, slots=True, kw_only=True)
class DocstringChange:
    """Information about a single docstring's wrapping result.

    Args:
        qualname: Qualified name of the docstring owner (e.g. "Cls.method").
        start_line: Starting line number of the docstring.
        end_line: Ending line number of the docstring.
        changed: Whether the docstring was modified.
        before: Original docstring text for a changed record at diff detail.
        after: Wrapped docstring text for a changed record at diff detail.
        converted: Whether this docstring was converted to a different style.
        summary_skipped: Whether summary wrapping was skipped for this
            docstring.
        summary_width: Width of the overlong summary when `summary_skipped` is
            true.

    Raises:
        ValueError: If `before` or `after` is supplied when `changed` is false,
            or `summary_width` is supplied when `summary_skipped` is false.
    """

    qualname: str
    start_line: int
    end_line: int
    changed: bool
    before: str | None = None
    after: str | None = None
    converted: bool = False
    summary_skipped: bool = False
    summary_width: int | None = None

    def __post_init__(self) -> None:
        """Validate the change record's field invariants.

        Raises:
            ValueError: If `before` or `after` is supplied when `changed` is
                false, or `summary_width` is supplied when `summary_skipped` is
                false.
        """
        if (self.before is not None or self.after is not None) and not self.changed:
            msg = 'before/after require changed=True'
            raise ValueError(msg)
        if self.summary_width is not None and not self.summary_skipped:
            msg = 'summary_width requires summary_skipped=True'
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocstringSkip:
    """Safety/losslessness record for a docstring left unchanged.

    Args:
        path: File path when known, or an empty string for in-memory sources.
        line: 1-based source line of the docstring literal, or 0 for file-level
            records.
        qualname: Qualified name of the docstring owner, or an empty string for
            file-level records.
        code: Machine-readable skip reason. This is the stable contract;
            branch on `SkipCode` values.
        message: Human-readable explanation. Wording may improve between
            releases; do not compare it programmatically.
    """

    path: str
    line: int
    qualname: str
    code: SkipCode
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DocwrapConfig:
    """Public configuration type for docwrap.

    Args:
        convert: Target style for conversion, or None to skip conversion.
        fill_style: Filling algorithm used to select line breaks.
        line_length: Maximum line length for wrapping.
        reflow: Reflow mode controlling line joining behavior.
        smart_slack: Maximum space smart fill may give up to repair a
            recognized awkward break.
        strip_types: Whether to strip inline type annotations.
        style: Docstring style name, or None for auto-detection.
        summary_start: Summary position relative to opening quotes.
        wrap_summary: Whether to wrap summary lines that exceed the target.

    Raises:
        InvalidLineLengthError: If `line_length` is not an integer of at least
            one, or is a boolean.
        UnknownStyleError: If `style` or `convert` names an unknown style.
        TypeError: If `strip_types` or `wrap_summary` is not a boolean.
        ValueError: If `smart_slack` is not a non-negative integer, or
            `fill_style`, `reflow`, or `summary_start` is not a supported
            value.
    """

    convert: StyleName | None = None
    fill_style: FillStyle = 'smart'
    line_length: int = DEFAULT_LINE_LENGTH
    reflow: ReflowMode = 'cautious'
    smart_slack: int = 8
    strip_types: bool = False
    style: StyleName | None = None
    summary_start: SummaryStart = 'preserve'
    wrap_summary: bool = False

    def __post_init__(self) -> None:
        """Validate every configuration construction invariant.

        Raises:
            InvalidLineLengthError: If `line_length` is not an int of at least
                1, or is a bool.
            UnknownStyleError: If `style` or `convert` names an unknown style.
            TypeError: If `strip_types` or `wrap_summary` is not a bool.
            ValueError: If `smart_slack` is not a non-negative int or is a
                bool, or `fill_style`, `reflow`, or `summary_start` is outside
                its published vocabulary.
        """
        from docwrap._validate import validate_config  # noqa: PLC0415

        validate_config(self)


class LineLengthSource(enum.StrEnum):
    """Provenance label for the resolved line length.

    Attributes:
        CLI: The `--length` command-line option supplied the value.
        DEFAULT: The built-in default supplied the value.
        DOCWRAP: The `[tool.docwrap]` configuration supplied the value.
    """

    CLI = '--length'
    DEFAULT = 'default'
    DOCWRAP = '[tool.docwrap]'


@dataclass(frozen=True, slots=True)
class PreCommitAdd:
    """Parsed Git add command that precedes a commit.

    Args:
        tokens: Parsed arguments for the Git add command.
        exec_cwd: Directory in which the command segment executes.
        git_cwd: Directory Git uses after applying any `-C` option.
    """

    tokens: tuple[str, ...]
    exec_cwd: str
    git_cwd: str


@dataclass(frozen=True, slots=True)
class WorktreeWrapContext:
    """Bundle of state passed through pre-tool-use hook handlers.

    Internal to the hook layer; not part of the public library API (omitted
    from __all__).

    Args:
        repo_root: Absolute path to the Git worktree root.
        incr_level: Resolved incremental level, or `None` when disabled.
        allow_on_failure: Whether a wrapping failure permits the operation.
    """

    repo_root: Path
    incr_level: IncrementalLevel | None
    allow_on_failure: bool


class WorktreeWrite(NamedTuple):
    """Prepared working-tree file write.

    Args:
        path: File path to update.
        source: Transformed source text.
        encoding: Encoding to use for the write.
        authorized_raw: Exact working-tree bytes authorized for replacement.
    """

    path: Path
    source: str
    encoding: str
    authorized_raw: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapCheckResult:
    """Result of a non-mutating wrap check.

    Args:
        status: 'changed', 'unchanged', or 'invalid'.
        skips: Safety-skip records observed during the check.
    """

    status: WrapStatus
    skips: tuple[DocstringSkip, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapOverrides:
    """CLI flag overrides applied on top of a resolved DocwrapConfig.

    Internal to the CLI layer; not part of the public library API (omitted from
    __all__).

    Args:
        line_length: Maximum line length override.
        style: Docstring style override.
        convert: Target conversion style override.
        fill_style: Filling algorithm override.
        smart_slack: Smart-fill repair budget override.
        show_skips: Whether preview output shows skipped-summary diagnostics.
        strip_types: Whether conversion strips inline type annotations.
        summary_start: Summary position override.
        wrap_summary: Whether to wrap overlong summaries.
        reflow: Reflow mode override.
    """

    line_length: int | None = None
    style: StyleName | None = None
    convert: StyleName | None = None
    fill_style: FillStyle | None = None
    smart_slack: int | None = None
    show_skips: bool | None = None
    strip_types: bool | None = None
    summary_start: SummaryStart | None = None
    wrap_summary: bool | None = None
    reflow: ReflowMode | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WrapReport:
    """Result of wrapping, containing transformed source and metadata.

    Args:
        source: The transformed source code.
        total: Total number of docstrings found.
        wrapped: Number of docstrings that were wrapped.
        convert_target: Target style used for conversion, or None.
        converted: Number of docstrings converted to a different style.
        docstrings: Per-docstring change details.
        parse_error: Whether a parse error occurred.
        skipped_summaries: Number of docstrings whose summary wrapping was
            skipped.
        style: Resolved docstring style for the file, or None if undetermined.
        skips: Safety-skip records for docstrings left unchanged.
    """

    source: str
    total: int
    wrapped: int
    convert_target: StyleName | None = None
    converted: int = 0
    docstrings: tuple[DocstringChange, ...] = ()
    parse_error: bool = False
    skipped_summaries: int = 0
    style: StyleName | None = None
    skips: tuple[DocstringSkip, ...] = ()

    @property
    def percent_wrapped(self) -> float:
        """Calculate the percentage of docstrings that were wrapped.

        Returns:
            Percentage of docstrings that were wrapped, or `0.0` when `total`
            is zero.
        """
        return (self.wrapped / self.total * 100) if self.total else 0.0


@dataclass(slots=True)
class FileResult:
    """Per-file processing result used by CLI output formatting.

    Args:
        status: Exit code for this file (0 unchanged, 1 changed, 2 error).
        path: File path.
        line_length: Effective line length used for wrapping.
        total: Total number of docstrings found, or None when no report was
            produced.
        wrapped: Number of docstrings that were wrapped.
        skipped_summaries: Number of docstrings whose summary wrapping was
            skipped.
        docstrings: Per-docstring change details retained for preview output.
        show_skips: Whether preview output should show skipped-summary
            diagnostics.
    """

    status: int
    path: Path
    line_length: int | None = None
    total: int | None = None
    wrapped: int = 0
    skipped_summaries: int = 0
    docstrings: tuple[DocstringChange, ...] = ()
    show_skips: bool = False
