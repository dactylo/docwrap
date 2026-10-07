"""Non-hook CLI commands for docwrap."""

import argparse
import logging
import os
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override

from docwrap._arguments import incremental_override
from docwrap._color import path_text, should_color, visible_controls
from docwrap._config import (
    ResolvedConfig,
    effective_config,
    find_color_config,
    find_config,
    find_incremental_config,
    find_stats_config,
    render_config_warnings,
    resolve_config,
)
from docwrap._core import analyze_source, wrap_source_report_eligible
from docwrap._core._incremental import (
    DiffHunk,
    derive_eligible_target,
    get_eligible_ranges,
    parse_diff_hunks,
    resolve_incremental_level,
)
from docwrap._downstream import stdout_closed
from docwrap._exceptions import UnknownStyleError
from docwrap._files import (
    ReadResult,
    collect_files,
    read_file,
    read_file_result,
    write_preserving_newline,
)
from docwrap._git import (
    RepoDiscoveryStatus,
    can_reuse_repo_root,
    discover_repo_root,
    get_repo_diff,
    get_tracked_files,
    is_head_unborn,
)
from docwrap._main import main as entry_main
from docwrap._styles import get_styles
from docwrap._types import (
    COLOR_MODES,
    DEFAULT_STATS_SECTIONS,
    FILL_STYLES,
    INCREMENTAL_LEVELS,
    REFLOW_MODES,
    SUMMARY_STARTS,
    DetailLevel,
    DocwrapConfig,
    FileResult,
    IncrementalLevel,
    LineLengthSource,
    OutputMode,
    StatsSection,
    WrapOverrides,
    WrapReport,
)

if TYPE_CHECKING:
    from docwrap._display._json import JsonFileRecord
    from docwrap._stats import ImpactRow, LineMeasurement, StatsData

_COMMAND_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ('commands', (('hook', 'run as a coding-agent hook (see docwrap hook --help)'),)),
)

_DEFAULT_CONFIG = DocwrapConfig()
_DEFAULT_RESOLVED = ResolvedConfig(config=DocwrapConfig())
_EXIT_CHANGED = 1
_EXIT_ERROR = 2
_EXIT_OK = 0

type _FileFailureKind = Literal['error', 'invalid']
type _FilePlanState = Literal['failed', 'full', 'no-eligible', 'no-hunks', 'ranges']


@dataclass(frozen=True, slots=True)
class _FileModeOptions:
    """Resolved output mode options for file processing."""

    mode: OutputMode
    detail: DetailLevel
    verbose: bool
    color: bool


@dataclass(frozen=True, slots=True)
class _FilePlan:
    """Pre-computed file plan from incremental analysis."""

    state: _FilePlanState
    eligible_ranges: set[tuple[int, int]] | None
    prepared: ReadResult | None = None
    failure_kind: _FileFailureKind | None = None
    diagnostic: str | None = None


@dataclass
class _IncrementalContext:
    """Computed incremental context for CLI file processing."""

    level: IncrementalLevel | None
    all_hunks: dict[Path, list[DiffHunk]]
    untracked: set[Path]
    repo_roots: dict[Path, str | None]
    diff_failed: set[Path]


@dataclass(frozen=True, slots=True)
class _IncrementalDiagnostic:
    """Diagnostic returned for caller-specific rendering."""

    message: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class _IncrementalOutcome:
    """Incremental context plus an optional caller-owned diagnostic."""

    context: _IncrementalContext | None
    diagnostic: _IncrementalDiagnostic | None = None


@dataclass
class _IncrementalRepositoryState:
    """Mutable incremental state collected across repositories."""

    all_hunks: dict[Path, list[DiffHunk]]
    diff_failed: set[Path]
    path_to_rel: dict[Path, str]
    untracked: set[Path]
    warnings: list[str]


class _RootParser(argparse.ArgumentParser):
    """Render the option groups, then the commands that have their own parsers."""

    @override
    def format_help(self, formatter: argparse.HelpFormatter | None = None) -> str:
        if formatter is None:
            formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)  # noqa: SLF001
            formatter.end_section()
        for title, commands in _COMMAND_GROUPS:
            formatter.start_section(title)
            formatter.add_arguments(
                [argparse.Action([], name, help=text) for name, text in commands]
            )
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()


@dataclass(frozen=True, slots=True)
class _WrappedFile:
    """Result of reading, resolving config, and wrapping a single file."""

    source: str
    encoding: str
    config: DocwrapConfig
    report: WrapReport
    raw: bytes
    show_skips: bool


def _aggregate_impact(target: dict[int, 'ImpactRow'], rows: list['ImpactRow']) -> None:
    """Merge per-file impact rows into aggregate impact dict."""
    for row in rows:
        prev = target.get(row.line_length)
        if prev is None:
            target[row.line_length] = row
            continue
        target[row.line_length] = replace(
            prev,
            docstrings_changed=prev.docstrings_changed + row.docstrings_changed,
            files_changed=prev.files_changed + row.files_changed,
            lines_added=prev.lines_added + row.lines_added,
            lines_removed=prev.lines_removed + row.lines_removed,
        )


def _argv_requests_help(args_list: list[str]) -> bool:
    """Check whether the argument list requests help output."""
    for arg in args_list:
        if arg == '--':
            return False
        if arg in {'-h', '--help'}:
            return True
    return False


def _classify_file_failure(error: Exception) -> _FileFailureKind:
    """Classify a file failure for planning and JSON output."""
    return 'error' if isinstance(error, (OSError, UnicodeDecodeError)) else 'invalid'


def _classify_missing_incremental_paths(
    state: _IncrementalRepositoryState,
    root: str,
    missing: list[Path],
    incr_explicit: bool,
) -> bool:
    """Classify paths missing from diff hunks using Git tracked membership."""
    if not missing:
        return True

    tracked_set = get_tracked_files(root, [state.path_to_rel[path] for path in missing])
    if tracked_set is None:
        if incr_explicit:
            return False
        state.diff_failed.update(missing)
        state.warnings.append(
            f'git tracked-file discovery failed for {root}, falling back to full mode'
        )
        return True

    for resolved in missing:
        if state.path_to_rel[resolved] not in tracked_set:
            state.untracked.add(resolved)
    return True


def _collect_incremental_repository_state(
    by_root: dict[str, list[Path]],
    incr_explicit: bool,
) -> _IncrementalRepositoryState | None:
    """Collect diff and untracked state once per repository."""
    state = _IncrementalRepositoryState(
        all_hunks={},
        diff_failed=set(),
        path_to_rel={},
        untracked=set(),
        warnings=[],
    )

    for root, root_files in by_root.items():
        relative_paths = tuple(
            os.path.relpath(str(resolved), root).replace(os.sep, '/') for resolved in root_files
        )
        state.path_to_rel.update(zip(root_files, relative_paths, strict=True))
        diff_output = get_repo_diff(root)
        if diff_output is None and is_head_unborn(root):
            # Without a commit, every file is new.
            state.untracked.update(root_files)
            continue
        root_hunks = None if diff_output is None else parse_diff_hunks(diff_output)
        if root_hunks is None:
            if incr_explicit:
                return None
            state.diff_failed.update(root_files)
            state.warnings.append(f'git diff failed for {root}, falling back to full mode')
            continue

        for resolved, rel in zip(root_files, relative_paths, strict=True):
            if rel in root_hunks:
                state.all_hunks[resolved] = root_hunks[rel]

        missing = [resolved for resolved in root_files if resolved not in state.all_hunks]
        if not _classify_missing_incremental_paths(state, root, missing, incr_explicit):
            return None

    return state


def _collect_stats(
    files: list[Path],
    overrides: WrapOverrides,
    *,
    extra_lengths: tuple[int, ...] = (),
    needs_impact: bool = True,
) -> 'StatsData':
    """Collect stats in one pass, optionally computing wrapping impact."""
    from docwrap._display._skips import format_safety_skip  # noqa: PLC0415
    from docwrap._stats import (  # noqa: PLC0415
        StatsData,
        compute_impact,
        measure_docstring,
        measure_source_lines,
    )

    all_measurements: list[LineMeasurement] = []
    all_source_measurements: list[LineMeasurement] = []
    agg_impact: dict[int, ImpactRow] = {}
    total_docstrings = 0
    docstring_files = 0
    skipped = 0
    safety_skips = 0
    unparseable = 0

    for path in files:
        try:
            source, _ = read_file(path)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            print(
                f'error: {visible_controls(path_text(path))}: {visible_controls(str(error))}',
                file=sys.stderr,
            )
            skipped += 1
            continue

        with render_config_warnings():
            cfg = resolve_config(path)
        effective_cfg = effective_config(cfg, overrides)

        analysis = analyze_source(source, effective_cfg)
        if analysis.parse_error:
            print(f'error: {visible_controls(path_text(path))}: failed to parse', file=sys.stderr)
            unparseable += 1
            continue

        for skip in analysis.skips:
            print(
                format_safety_skip(str(path), skip),
                file=sys.stderr,
            )
        safety_skips += len(analysis.skips)

        docstrings = analysis.docstrings
        parsed = analysis.parsed

        n_docstrings = len(docstrings)
        total_docstrings += n_docstrings
        if n_docstrings > 0:
            docstring_files += 1

        measurements = measure_docstring(
            parsed,
            file_path=str(path),
            wrap_summary=effective_cfg.wrap_summary,
        )
        all_measurements.extend(measurements)

        docstring_ranges = [(info.lineno, info.end_lineno) for info in docstrings]
        source_ms = measure_source_lines(source, docstring_ranges, file_path=str(path))
        all_source_measurements.extend(source_ms)

        if needs_impact:
            rows = compute_impact(
                parsed,
                effective_cfg,
                extra_lengths=extra_lengths,
            )
            _aggregate_impact(agg_impact, rows)

    return StatsData(
        docstring_count=total_docstrings,
        docstring_file_count=docstring_files,
        impact=agg_impact,
        measurements=all_measurements,
        skipped_count=skipped,
        source_measurements=all_source_measurements,
        safety_skip_count=safety_skips,
        unparseable_count=unparseable,
    )


def _compute_file_plan(
    path: Path,
    incr_ctx: _IncrementalContext,
) -> _FilePlan:
    """Compute a discriminated file plan for incremental mode."""
    resolved = path.resolve()
    if (
        incr_ctx.level is None
        or resolved in incr_ctx.diff_failed
        or resolved in incr_ctx.untracked
    ):
        plan = _FilePlan(state='full', eligible_ranges=None)
    else:
        hunks = incr_ctx.all_hunks.get(resolved, [])
        if incr_ctx.repo_roots.get(resolved) is None or not hunks:
            plan = _FilePlan(state='no-hunks', eligible_ranges=None)
        else:
            try:
                read_result = read_file_result(path)
            except (
                OSError,
                UnicodeDecodeError,
                SyntaxError,
                tokenize.TokenError,
                ValueError,
            ) as error:
                plan = _FilePlan(
                    state='failed',
                    eligible_ranges=None,
                    failure_kind=_classify_file_failure(error),
                    diagnostic=str(error),
                )
            else:
                prepared = read_result
                try:
                    ranges = get_eligible_ranges(
                        prepared.source,
                        hunks,
                        level=incr_ctx.level,
                        encoding=prepared.encoding,
                    )
                except (SyntaxError, tokenize.TokenError, ValueError) as error:
                    plan = _FilePlan(
                        state='failed',
                        eligible_ranges=None,
                        prepared=prepared,
                        failure_kind=_classify_file_failure(error),
                        diagnostic=str(error),
                    )
                else:
                    if ranges is None:
                        state: _FilePlanState = 'full'
                    elif ranges:
                        state = 'ranges'
                    else:
                        state = 'no-eligible'
                    plan = _FilePlan(
                        state=state,
                        eligible_ranges=ranges,
                        prepared=prepared,
                    )
    return plan


def _config_field_source(config: DocwrapConfig, field_name: str) -> str:
    """Return the source label for a config field."""
    if getattr(config, field_name) == getattr(_DEFAULT_CONFIG, field_name):
        return 'default'
    return '[tool.docwrap]'


def _discover_incremental_repositories(
    files: list[Path],
) -> tuple[dict[Path, str | None], dict[str, list[Path]], set[Path]]:
    """Map files to repository roots and group files by root."""
    discovery_failed: set[Path] = set()
    repo_roots: dict[Path, str | None] = {}
    root_cache: dict[Path, tuple[str | None, RepoDiscoveryStatus]] = {}
    discovered_roots: dict[str, None] = {}
    for path in files:
        resolved = path.resolve()
        parent = resolved.parent
        if parent not in root_cache:
            reusable = next(
                (root for root in discovered_roots if can_reuse_repo_root(parent, root)),
                None,
            )
            if reusable is not None:
                root_cache[parent] = reusable, RepoDiscoveryStatus.WORKTREE
            else:
                discovery = discover_repo_root(str(parent))
                root_cache[parent] = discovery.root, discovery.status
                if discovery.root is not None:
                    discovered_roots[discovery.root] = None
        root, status = root_cache[parent]
        repo_roots[resolved] = root
        if status is RepoDiscoveryStatus.ERROR:
            discovery_failed.add(resolved)

    by_root: dict[str, list[Path]] = {}
    for resolved, root in repo_roots.items():
        if root is not None:
            by_root.setdefault(root, []).append(resolved)
    return repo_roots, by_root, discovery_failed


def _compute_incremental_context(
    files: list[Path],
    incr_level: IncrementalLevel | None,
    incr_explicit: bool,
) -> _IncrementalOutcome:
    """Compute diff hunks and untracked file set for incremental mode."""
    if incr_level is None or not files:
        return _IncrementalOutcome(
            context=_IncrementalContext(
                level=incr_level,
                all_hunks={},
                untracked=set(),
                repo_roots={},
                diff_failed=set(),
            )
        )

    repo_roots, by_root, discovery_failed = _discover_incremental_repositories(files)
    if discovery_failed and incr_explicit:
        return _IncrementalOutcome(
            context=None,
            diagnostic=_IncrementalDiagnostic(
                'git state collection failed for incremental mode', fatal=True
            ),
        )

    state = _collect_incremental_repository_state(by_root, incr_explicit)
    if state is None:
        return _IncrementalOutcome(
            context=None,
            diagnostic=_IncrementalDiagnostic(
                'git state collection failed for incremental mode', fatal=True
            ),
        )

    if discovery_failed:
        state.diff_failed.update(discovery_failed)
        state.warnings.append('git repository discovery failed, falling back to full mode')
    if any(
        root is None and resolved not in discovery_failed for resolved, root in repo_roots.items()
    ):
        state.warnings.append('not in a git repository, skipping incremental files')
    diagnostic = (
        _IncrementalDiagnostic('; '.join(sorted(state.warnings))) if state.warnings else None
    )

    return _IncrementalOutcome(
        context=_IncrementalContext(
            level=incr_level,
            all_hunks=state.all_hunks,
            untracked=state.untracked,
            repo_roots=repo_roots,
            diff_failed=state.diff_failed,
        ),
        diagnostic=diagnostic,
    )


def _format_default(value: object, source: str) -> str:
    """Format a default-value string for help text."""
    if source == 'default':
        return f'(default: {value})'
    return f'(default: {value}, from {source})'


def _on_off(value: bool) -> str:
    """Return the help-text spelling of a boolean default."""
    return 'on' if value else 'off'


def _parse_stats_longest(value: str) -> int:
    """Parse a positive stats longest-line limit."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _parse_stats_sections(value: str) -> tuple[StatsSection, ...]:
    """Parse an ordered comma-separated stats section list."""
    if value == '':
        return ()
    names = [part.strip() for part in value.split(',')]
    if any(not name for name in names):
        raise argparse.ArgumentTypeError('stats sections cannot contain an empty item')
    unknown = [name for name in names if name not in DEFAULT_STATS_SECTIONS]
    if unknown:
        raise argparse.ArgumentTypeError(f'unknown stats section: {unknown[0]}')
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError('stats sections cannot contain duplicates')
    return cast('tuple[StatsSection, ...]', tuple(names))


def _build_parser(resolved: ResolvedConfig | None = None) -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    resolved_cfg = _DEFAULT_RESOLVED if resolved is None else resolved
    help_config = resolved_cfg.config

    color_default = _format_default(
        resolved_cfg.color,
        'default' if resolved_cfg.color == _DEFAULT_RESOLVED.color else '[tool.docwrap]',
    )
    length_default = _format_default(
        help_config.line_length,
        resolved_cfg.line_length_source,
    )
    fill_style_default = _format_default(
        help_config.fill_style,
        _config_field_source(help_config, 'fill_style'),
    )
    smart_slack_default = _format_default(
        help_config.smart_slack,
        _config_field_source(help_config, 'smart_slack'),
    )
    reflow_default = _format_default(
        help_config.reflow,
        _config_field_source(help_config, 'reflow'),
    )
    summary_start_default = _format_default(
        help_config.summary_start,
        _config_field_source(help_config, 'summary_start'),
    )
    wrap_summary_default = _format_default(
        _on_off(help_config.wrap_summary),
        _config_field_source(help_config, 'wrap_summary'),
    )
    show_skips_default = _format_default(
        _on_off(resolved_cfg.show_skips),
        'default' if resolved_cfg.show_skips == _DEFAULT_RESOLVED.show_skips else '[tool.docwrap]',
    )
    incremental_level = (
        f' (level: {resolved_cfg.incremental}, from [tool.docwrap])'
        if isinstance(resolved_cfg.incremental, str)
        else ''
    )

    parser = _RootParser(
        prog='docwrap',
        usage='%(prog)s [options] [PATH ...]',
        description='Wrap Python docstrings to a configured line length.',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        default=['.'],
        metavar='PATH',
        help='files or directories to check (default: current directory)',
    )
    parser.add_argument('--version', action='store_true', help='show the version and exit')
    output = parser.add_argument_group('output (choose one)')
    modes = output.add_mutually_exclusive_group()
    modes.add_argument(
        '-p', '--preview', action='store_true', help='show each docstring that would change'
    )
    modes.add_argument('-d', '--diff', action='store_true', help='show a unified diff')
    modes.add_argument('-w', '--write', action='store_true', help='rewrite files in place')
    modes.add_argument('-j', '--json', action='store_true', help='print results as JSON')
    modes.add_argument(
        '-s',
        '--stats',
        action='store_true',
        help='report docstring line lengths and help choose a line length',
    )
    wrapping = parser.add_argument_group('wrapping')
    wrapping.add_argument(
        '-l',
        '--length',
        type=int,
        dest='line_length',
        metavar='N',
        help=f'maximum line length {length_default}',
    )
    wrapping.add_argument(
        '--reflow',
        choices=REFLOW_MODES,
        default=None,
        help=f'how freely to rejoin lines before wrapping {reflow_default}',
    )
    wrapping.add_argument(
        '--fill-style',
        choices=FILL_STYLES,
        default=None,
        help=f'how to choose line breaks {fill_style_default}',
    )
    wrapping.add_argument(
        '--smart-slack',
        type=int,
        default=None,
        metavar='N',
        help=(
            f'columns smart fill may give up to avoid a stranded fragment {smart_slack_default}'
        ),
    )
    wrapping.add_argument(
        '--summary-start',
        choices=SUMMARY_STARTS,
        default=None,
        help=f'where a multi-line summary starts {summary_start_default}',
    )
    wrapping.add_argument(
        '--wrap-summary',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f'wrap summaries longer than the line length {wrap_summary_default}',
    )
    wrapping.add_argument(
        '--style',
        choices=sorted(get_styles()),
        help="use this docstring style instead of detecting each docstring's style",
    )
    conversion = parser.add_argument_group('conversion')
    conversion.add_argument(
        '--convert',
        choices=sorted(get_styles()),
        help='rewrite docstrings in another style',
    )
    conversion.add_argument(
        '--strip-types',
        action='store_true',
        default=False,
        help='remove inline types from structured entries, such as param (str):',
    )
    selection = parser.add_argument_group('selection')
    selection.add_argument(
        '-i',
        '--incremental',
        action='store_true',
        default=False,
        help=f'wrap only docstrings affected by changes since the last commit{incremental_level}',
    )
    selection.add_argument(
        '--incremental-level',
        choices=INCREMENTAL_LEVELS,
        default=None,
        help='how much around each change to wrap; implies --incremental',
    )
    selection.add_argument(
        '--no-incremental',
        action='store_true',
        default=False,
        help='wrap every docstring, even when configuration enables incremental mode',
    )
    display = parser.add_argument_group('display')
    display.add_argument('-v', '--verbose', action='store_true', help='also list unchanged files')
    display.add_argument(
        '--color',
        choices=COLOR_MODES,
        default='auto',
        help=f'when to use color {color_default}',
    )
    skips = display.add_mutually_exclusive_group()
    skips.add_argument(
        '--show-skips',
        action='store_true',
        default=None,
        dest='show_skips',
        help=(
            'in preview, list summaries left unwrapped because they are too long '
            f'{show_skips_default}'
        ),
    )
    skips.add_argument(
        '--hide-skips',
        action='store_false',
        dest='show_skips',
        help='in preview, do not list summaries left unwrapped because they are too long',
    )
    statistics = parser.add_argument_group('statistics')
    statistics.add_argument(
        '--stats-sections',
        type=_parse_stats_sections,
        default=None,
        metavar='SECTION[,SECTION...]',
        help='report sections to show, in order',
    )
    statistics.add_argument(
        '--stats-longest-per-category',
        type=_parse_stats_longest,
        default=None,
        metavar='N',
        help='longest lines to list per category (default: 3)',
    )
    return parser


def _print_version() -> int:
    """Print the installed distribution version without other work."""
    import importlib.metadata  # noqa: PLC0415

    try:
        version = importlib.metadata.version('docwrap')
    except importlib.metadata.PackageNotFoundError:
        print('error: installed distribution metadata for docwrap is unavailable', file=sys.stderr)
        return _EXIT_ERROR
    print(f'docwrap {version}')
    return _EXIT_OK


def _resolve_help_config() -> ResolvedConfig:
    """Resolve composed config from cwd for help-text defaults."""
    with render_config_warnings():
        return find_config(Path())


def _resolve_incremental(
    args: argparse.Namespace,
    file_config_incremental: IncrementalLevel | Literal[False] | None,
) -> tuple[IncrementalLevel | None, bool]:
    """Resolve effective incremental level and whether it was explicit."""
    level = resolve_incremental_level(
        override=incremental_override(args),
        explicit_level=args.incremental_level,
        config_value=file_config_incremental,
    )
    is_explicit = bool(
        args.no_incremental or args.incremental_level is not None or args.incremental
    )
    return level, is_explicit


def _resolve_mode(args: argparse.Namespace) -> tuple[OutputMode, DetailLevel]:
    """Determine output mode and detail level from parsed args."""
    if args.preview:
        return 'preview', 'diff'
    if args.diff:
        return 'diff', 'summary'
    if args.stats:
        return 'stats', 'summary'
    if args.write:
        return 'write', 'summary'
    if args.json:
        return 'json', 'diff'
    return 'summary', 'summary'


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate parsed arguments that argparse cannot constrain directly."""
    if args.line_length is not None and args.line_length < 1:
        parser.error('--length must be a positive integer')

    if not args.stats and (
        args.stats_sections is not None or args.stats_longest_per_category is not None
    ):
        parser.error('--stats-sections and --stats-longest-per-category require --stats')

    if args.smart_slack is not None and args.smart_slack < 0:
        parser.error('--smart-slack must be a non-negative integer')

    incr_count = sum([args.incremental, args.incremental_level is not None, args.no_incremental])
    if incr_count > 1:
        parser.error(
            '-i/--incremental, --incremental-level, and --no-incremental are mutually exclusive'
        )


def _wrap_file_once(
    path: Path,
    overrides: WrapOverrides,
    file_plan: _FilePlan,
    incr_level: IncrementalLevel | None,
    detail: DetailLevel = 'summary',
) -> _WrappedFile:
    """Read, resolve config, and wrap a single file."""
    prepared = file_plan.prepared
    if prepared is None:
        read_result = read_file_result(path)
        raw = read_result.raw
        source = read_result.source
        encoding = read_result.encoding
    else:
        raw = prepared.raw
        source = prepared.source
        encoding = prepared.encoding
    with render_config_warnings():
        resolved = find_config(path)
    eff_cfg = effective_config(resolved.config, overrides)
    show_skips = overrides.show_skips if overrides.show_skips is not None else resolved.show_skips
    eligible_target = derive_eligible_target(incr_level)
    report = wrap_source_report_eligible(
        source,
        config=eff_cfg,
        detail=detail,
        eligible_ranges=file_plan.eligible_ranges,
        eligible_target=eligible_target,
    )
    return _WrappedFile(
        source=source,
        encoding=encoding,
        config=eff_cfg,
        report=report,
        raw=raw,
        show_skips=show_skips,
    )


def _process_file(
    path: Path,
    overrides: WrapOverrides,
    file_plan: _FilePlan,
    incr_level: IncrementalLevel | None,
    options: _FileModeOptions,
) -> FileResult:
    """Wrap one selected file and render its requested output."""
    try:
        wf = _wrap_file_once(
            path,
            overrides,
            file_plan,
            incr_level,
            detail=options.detail,
        )
    except UnknownStyleError as error:
        print(f'error: {visible_controls(str(error))}', file=sys.stderr)
        return FileResult(status=_EXIT_ERROR, path=path)
    except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as e:
        print(
            f'error: {visible_controls(path_text(path))}: {visible_controls(str(e))}',
            file=sys.stderr,
        )
        return FileResult(status=_EXIT_ERROR, path=path)

    report = wf.report
    if report.parse_error:
        print(f'error: {visible_controls(path_text(path))}: failed to parse', file=sys.stderr)
        return FileResult(status=_EXIT_ERROR, path=path)

    from docwrap._display._skips import format_safety_skip  # noqa: PLC0415

    for skip in report.skips:
        print(
            format_safety_skip(str(path), skip),
            file=sys.stderr,
        )

    effective_length = wf.config.line_length
    show_skips = wf.show_skips
    result = report.source
    docstrings = report.docstrings if options.mode == 'preview' else ()

    def file_result(status: int) -> FileResult:
        return FileResult(
            status=status,
            path=path,
            line_length=effective_length,
            total=report.total,
            wrapped=report.wrapped,
            skipped_summaries=report.skipped_summaries,
            docstrings=docstrings,
            show_skips=show_skips,
        )

    if result == wf.source:
        return file_result(_EXIT_OK)

    if options.mode == 'write':
        try:
            write_preserving_newline(path, result, wf.encoding, raw=wf.raw)
        except OSError as e:
            print(
                f'error: {visible_controls(path_text(path))}: {visible_controls(str(e))}',
                file=sys.stderr,
            )
            return file_result(_EXIT_ERROR)

    if options.mode == 'diff':
        from docwrap._display import iter_diff  # noqa: PLC0415

        sys.stdout.writelines(
            iter_diff(
                wf.source, result, str(path), color=options.color, terminal=sys.stdout.isatty()
            )
        )

    status = _EXIT_OK if options.mode == 'write' else _EXIT_CHANGED
    return file_result(status)


def _process_file_json(
    path: Path,
    overrides: WrapOverrides,
    file_plan: _FilePlan,
    incr_level: IncrementalLevel | None,
) -> 'JsonFileRecord':
    """Process a single file for JSON output."""
    from docwrap._display._json import (  # noqa: PLC0415
        json_error_record,
        json_success_record,
    )

    str_path = str(path)

    try:
        wf = _wrap_file_once(
            path,
            overrides,
            file_plan,
            incr_level,
            detail='diff',
        )
    except (OSError, UnicodeDecodeError, SyntaxError, tokenize.TokenError, ValueError) as error:
        return json_error_record(str_path, _classify_file_failure(error), str(error))

    report = wf.report
    if report.parse_error:
        return json_error_record(str_path, 'invalid', f'{path}: failed to parse')

    return json_success_record(
        str_path, report, wf.config.line_length, changed=report.source != wf.source
    )


def _run_file_mode(
    overrides: WrapOverrides,
    files: list[Path],
    incr_ctx: _IncrementalContext,
    opts: _FileModeOptions,
) -> int:
    """Handle non-JSON, non-stats output modes (preview, diff, write, summary)."""
    from docwrap._display import print_preview, print_summary  # noqa: PLC0415

    results: list[FileResult] = []
    for path in files:
        if stdout_closed():
            break
        file_plan = _compute_file_plan(path, incr_ctx)
        if file_plan.state in ('no-hunks', 'no-eligible'):
            continue
        if file_plan.state == 'failed':
            assert file_plan.diagnostic is not None  # noqa: S101 -- failed-plan invariant
            print(
                f'error: {visible_controls(path_text(path))}: '
                f'{visible_controls(file_plan.diagnostic)}',
                file=sys.stderr,
            )
            results.append(FileResult(status=_EXIT_ERROR, path=path))
            continue
        result = _process_file(
            path,
            overrides,
            file_plan,
            incr_ctx.level,
            opts,
        )
        results.append(result)

    if opts.mode != 'diff':
        if opts.mode in ('summary', 'write'):
            print_summary(results=results, verbose=opts.verbose, color=opts.color)
        elif opts.mode == 'preview':
            print_preview(results=results, color=opts.color)

    has_error = any(r.status == _EXIT_ERROR for r in results)
    has_changes = any(r.status == _EXIT_CHANGED for r in results)

    if has_error:
        return _EXIT_ERROR
    if has_changes:
        return _EXIT_CHANGED
    return _EXIT_OK


def _run_json_mode(
    overrides: WrapOverrides,
    files: list[Path],
    incr_ctx: _IncrementalContext,
) -> int:
    """Collect per-file records and emit the JSON result envelope."""
    from docwrap._display import format_json, json_error_record  # noqa: PLC0415

    logging.getLogger('docwrap').setLevel(logging.CRITICAL)
    records: list[JsonFileRecord] = []
    for path in files:
        file_plan = _compute_file_plan(path, incr_ctx)
        if file_plan.state in ('no-hunks', 'no-eligible'):
            continue
        if file_plan.state == 'failed':
            assert file_plan.failure_kind is not None  # noqa: S101 -- failed-plan invariant
            assert file_plan.diagnostic is not None  # noqa: S101 -- failed-plan invariant
            records.append(
                json_error_record(
                    str(path),
                    file_plan.failure_kind,
                    file_plan.diagnostic,
                )
            )
            continue
        records.append(
            _process_file_json(
                path,
                overrides,
                file_plan,
                incr_ctx.level,
            )
        )
    print(format_json(records))
    has_error = any(r['status'] in ('error', 'invalid') for r in records)
    has_changes = any(r['status'] == 'changed' for r in records)
    if has_error:
        return _EXIT_ERROR
    if has_changes:
        return _EXIT_CHANGED
    return _EXIT_OK


def _wrap_overrides(args: argparse.Namespace) -> WrapOverrides:
    """Translate CLI wrapping options into configuration overrides."""
    return WrapOverrides(
        line_length=args.line_length,
        style=args.style,
        convert=args.convert,
        fill_style=args.fill_style,
        smart_slack=args.smart_slack,
        show_skips=args.show_skips,
        strip_types=args.strip_types,
        summary_start=args.summary_start,
        wrap_summary=args.wrap_summary,
        reflow=args.reflow,
    )


def _print_stats(
    *,
    files: list[Path],
    args: argparse.Namespace,
) -> bool:
    """Collect and print stats output, reporting whether any file failed."""
    if not files:
        return False

    with render_config_warnings():
        stats_cfg = find_stats_config(files[0])
    sections = args.stats_sections if args.stats_sections is not None else stats_cfg.sections
    longest_per_category = (
        args.stats_longest_per_category
        if args.stats_longest_per_category is not None
        else stats_cfg.longest_per_category
    )
    needs_impact = 'impact' in sections or 'recommendations' in sections

    from docwrap._display import print_stats  # noqa: PLC0415

    with render_config_warnings():
        resolved = find_config(files[0])
    first_cfg = resolved.config
    overrides = _wrap_overrides(args)
    first_effective_cfg = effective_config(first_cfg, overrides)

    if args.line_length is not None:
        effective_line_length: int = args.line_length
        line_length_source: LineLengthSource = LineLengthSource.CLI
    else:
        effective_line_length = first_effective_cfg.line_length
        line_length_source = resolved.line_length_source

    extra_lengths: tuple[int, ...] = ()
    if needs_impact and line_length_source in (
        LineLengthSource.CLI,
        LineLengthSource.DOCWRAP,
    ):
        extra_lengths = (effective_line_length,)

    data = _collect_stats(
        files,
        overrides,
        extra_lengths=extra_lengths,
        needs_impact=needs_impact,
    )
    with render_config_warnings():
        color_config = find_color_config()
    color = should_color(sys.stdout, cli_flag=args.color, config_value=color_config)

    print_stats(
        data,
        line_length=effective_line_length,
        color=color,
        line_length_source=line_length_source,
        sections=sections,
        longest_per_category=longest_per_category,
    )
    return data.skipped_count > 0 or data.unparseable_count > 0


def _run_stats_mode(args: argparse.Namespace, files: list[Path]) -> int:
    """Measure selected files and render the requested statistics."""
    try:
        failed = _print_stats(files=files, args=args)
    except UnknownStyleError as error:
        print(f'error: {visible_controls(str(error))}', file=sys.stderr)
        return _EXIT_ERROR
    return _EXIT_ERROR if failed else _EXIT_OK


def run_cli(args: list[str]) -> int:
    """Process CLI arguments and run the requested command."""
    help_cfg = _resolve_help_config() if _argv_requests_help(args) else None
    parser = _build_parser(help_cfg)
    options = parser.parse_args(args)
    if options.version:
        return _print_version()
    _validate_args(parser, options)

    mode, detail = _resolve_mode(options)
    files = collect_files(options.paths)
    if mode == 'stats':
        return _run_stats_mode(options, files)

    overrides = _wrap_overrides(options)

    if options.no_incremental or options.incremental_level is not None:
        file_config_incr = None
    else:
        with render_config_warnings():
            file_config_incr = find_incremental_config(files[0]) if files else None
    incr_level, incr_explicit = _resolve_incremental(options, file_config_incr)
    incr_outcome = _compute_incremental_context(files, incr_level, incr_explicit)
    diagnostic = incr_outcome.diagnostic

    if mode == 'json':
        if diagnostic is not None and diagnostic.fatal:
            from docwrap._display import format_json, json_error_record  # noqa: PLC0415

            print(
                format_json(
                    [
                        json_error_record(
                            str(files[0]),
                            'error',
                            diagnostic.message,
                        )
                    ]
                )
            )
            status = _EXIT_ERROR
        else:
            assert incr_outcome.context is not None  # noqa: S101 -- nonfatal outcome contract
            status = _run_json_mode(overrides, files, incr_outcome.context)
        return status

    if diagnostic is not None:
        prefix = 'error' if diagnostic.fatal else 'warning'
        print(f'{prefix}: {visible_controls(diagnostic.message)}', file=sys.stderr)
        if diagnostic.fatal:
            return _EXIT_ERROR

    assert incr_outcome.context is not None  # noqa: S101 -- nonfatal outcome contract
    incr_ctx = incr_outcome.context

    with render_config_warnings():
        color_config = find_color_config()
    color = should_color(sys.stdout, cli_flag=options.color, config_value=color_config)

    opts = _FileModeOptions(
        mode=mode,
        detail=detail,
        verbose=options.verbose,
        color=color,
    )
    return _run_file_mode(overrides, files, incr_ctx, opts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface through the shared entry point.

    Args:
        argv: Arguments to parse, or `None` to use `sys.argv`.

    Returns:
        The process exit status.
    """
    return entry_main(argv)


if __name__ == '__main__':
    sys.exit(main())
