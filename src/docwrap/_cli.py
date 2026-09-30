"""CLI entry point for docwrap."""

import argparse
import logging
import os
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from docwrap._color import should_color, visible_controls
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
from docwrap._downstream import closable_stdout, stdout_closed
from docwrap._exceptions import UnknownStyleError
from docwrap._files import (
    ReadResult,
    collect_files,
    ensure_utf8_stdio,
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
)
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


@dataclass(frozen=True, slots=True)
class _WrappedFile:
    """Result of reading, resolving config, and wrapping a single file."""

    source: str
    encoding: str
    config: DocwrapConfig
    report: WrapReport
    raw: bytes | None
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


def _build_hook_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the hook subcommand."""
    parser = argparse.ArgumentParser(
        prog='docwrap hook',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Agent hook entry point; reads the host payload as JSON from stdin.',
    )
    parser.add_argument(
        '-a',
        '--agent',
        required=True,
        metavar='NAME',
        help='Installed coding-agent adapter name.',
    )
    parser.add_argument(
        '-e',
        '--event',
        metavar='NAME',
        help='Host event name; required only for hosts without an event in the payload.',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-i',
        '--incremental',
        action='store_true',
        help='Enable incremental wrapping at the configured level (default: definition).',
    )
    group.add_argument(
        '--no-incremental',
        action='store_true',
        help='Disable incremental wrapping, even when configured.',
    )
    parser.add_argument(
        '--allow-on-failure',
        action='store_true',
        help='Allow accepted commands on hook failures; unsupported commits still deny.',
    )
    return parser


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

    for path in files:
        try:
            source, _ = read_file(path)
        except (OSError, UnicodeDecodeError, SyntaxError):
            skipped += 1
            continue

        with render_config_warnings():
            cfg = resolve_config(path)
        effective_cfg = effective_config(cfg, overrides)

        analysis = analyze_source(source, effective_cfg)
        if analysis.parse_error:
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
            except (OSError, UnicodeDecodeError) as error:
                plan = _FilePlan(
                    state='failed',
                    eligible_ranges=None,
                    failure_kind='error',
                    diagnostic=str(error),
                )
            except (SyntaxError, tokenize.TokenError, ValueError) as error:
                plan = _FilePlan(
                    state='failed',
                    eligible_ranges=None,
                    failure_kind='invalid',
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
                        failure_kind='invalid',
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


def _incremental_override(args: argparse.Namespace) -> bool | None:
    """Map CLI incremental switches to a tri-state override."""
    if args.no_incremental:
        return False
    if args.incremental:
        return True
    return None


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
        help_config.wrap_summary,
        _config_field_source(help_config, 'wrap_summary'),
    )

    parser = argparse.ArgumentParser(
        prog='docwrap',
        description='Wrap Python docstrings to a configured line length.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('subcommands:\n  hook    Agent hook entry point (see docwrap hook --help)'),
    )
    wrapping_policy_group = parser.add_argument_group('wrapping policy options')
    style_group = parser.add_argument_group('style options')
    statistics_group = parser.add_argument_group('statistics options')
    parser.add_argument('paths', nargs='*', default=['.'], help='Python files or directories.')
    parser.add_argument('--version', action='store_true', help='Show the installed version.')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-p', '--preview', action='store_true', help='Show per-docstring before/after.'
    )
    mode_group.add_argument(
        '-s',
        '--stats',
        action='store_true',
        help='Show line length statistics.',
    )
    mode_group.add_argument('-d', '--diff', action='store_true', help='Show unified diff.')
    mode_group.add_argument('-j', '--json', action='store_true', help='Output to JSON.')
    mode_group.add_argument('-w', '--write', action='store_true', help='Modify files in place.')
    parser.add_argument(
        '-l',
        '--length',
        type=int,
        dest='line_length',
        help=f'Target line length {length_default}.',
    )
    parser.add_argument(
        '-i',
        '--incremental',
        action='store_true',
        default=False,
        help=(
            'Enable incremental mode with the configured level (default when unset: definition).'
        ),
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Include unchanged files.')
    parser.add_argument(
        '--color',
        choices=COLOR_MODES,
        default='auto',
        help=f'Color output {color_default}.',
    )
    parser.add_argument(
        '--no-incremental',
        action='store_true',
        default=False,
        help='Explicitly disable incremental mode.',
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        '--show-skips',
        action='store_true',
        default=None,
        dest='show_skips',
        help='Show skipped-summary diagnostics in preview output.',
    )
    skip_group.add_argument(
        '--hide-skips',
        action='store_false',
        dest='show_skips',
        help='Hide skipped-summary diagnostics in preview output.',
    )
    wrapping_policy_group.add_argument(
        '--incremental-level',
        choices=INCREMENTAL_LEVELS,
        default=None,
        help='Override incremental level (enables incremental).',
    )
    wrapping_policy_group.add_argument(
        '--summary-start',
        choices=SUMMARY_STARTS,
        default=None,
        help=f'Summary position relative to opening quotes {summary_start_default}.',
    )
    wrapping_policy_group.add_argument(
        '--wrap-summary',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f'Wrap summary lines that exceed the target length {wrap_summary_default}.',
    )
    wrapping_policy_group.add_argument(
        '--fill-style',
        choices=FILL_STYLES,
        default=None,
        help=f'Fill style: balanced, greedy, or smart {fill_style_default}.',
    )
    wrapping_policy_group.add_argument(
        '--smart-slack',
        type=int,
        default=None,
        help=f'Smart-fill break-repair budget {smart_slack_default}.',
    )
    wrapping_policy_group.add_argument(
        '--reflow',
        choices=REFLOW_MODES,
        default=None,
        help=f'Reflow mode: none, cautious, or full {reflow_default}.',
    )
    style_group.add_argument(
        '--style',
        choices=sorted(get_styles()),
        help='Override docstring style.',
    )
    style_group.add_argument(
        '--convert',
        choices=sorted(get_styles()),
        help='Convert docstrings to the specified style.',
    )
    style_group.add_argument(
        '--strip-types',
        action='store_true',
        default=False,
        help='Strip inline type annotations from docstrings.',
    )
    statistics_group.add_argument(
        '--stats-sections',
        type=_parse_stats_sections,
        default=None,
        metavar='SECTION[,SECTION...]',
        help='Stats body sections, shown once in the listed order.',
    )
    statistics_group.add_argument(
        '--stats-longest-per-category',
        type=_parse_stats_longest,
        default=None,
        metavar='N',
        help='Longest lines shown per wrappable category in stats mode (default: 3).',
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
    file_config_incremental: IncrementalLevel | bool | None,
) -> tuple[IncrementalLevel | None, bool]:
    """Resolve effective incremental level and whether it was explicit."""
    level = resolve_incremental_level(
        override=_incremental_override(args),
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


def _run_hook(argv: list[str]) -> int:
    """Handle the hook subcommand."""
    hook_parser = _build_hook_parser()
    args = hook_parser.parse_args(argv)

    if sys.stdin.isatty():
        print('docwrap hook: reads the host hook payload as JSON from stdin')
        print('See docs/hook.md for configuration.')
        return 0

    try:
        stdin_data = sys.stdin.read()
    except (ValueError, OSError):
        return 0

    from docwrap._hook import run_hook  # noqa: PLC0415

    return run_hook(
        stdin_data,
        host=args.agent,
        event=args.event,
        allow_on_failure=args.allow_on_failure,
        incremental=_incremental_override(args),
    )


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
    """Process a single file."""
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
            f'error: {visible_controls(str(path))}: {visible_controls(str(e))}',
            file=sys.stderr,
        )
        return FileResult(status=_EXIT_ERROR, path=path)

    report = wf.report
    if report.parse_error:
        print(f'error: {visible_controls(str(path))}: failed to parse', file=sys.stderr)
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

    if result == wf.source:
        return FileResult(
            status=_EXIT_OK,
            path=path,
            line_length=effective_length,
            total=report.total,
            wrapped=report.wrapped,
            skipped_summaries=report.skipped_summaries,
            docstrings=docstrings,
            show_skips=show_skips,
        )

    if options.mode == 'write':
        try:
            write_preserving_newline(path, result, wf.encoding, raw=wf.raw)
        except OSError as e:
            print(
                f'error: {visible_controls(str(path))}: {visible_controls(str(e))}',
                file=sys.stderr,
            )
            return FileResult(
                status=_EXIT_ERROR,
                path=path,
                line_length=effective_length,
                total=report.total,
                wrapped=report.wrapped,
                skipped_summaries=report.skipped_summaries,
                docstrings=docstrings,
                show_skips=show_skips,
            )

    if options.mode == 'diff':
        from docwrap._display import iter_diff  # noqa: PLC0415

        sys.stdout.writelines(iter_diff(wf.source, result, str(path), color=options.color))

    status = _EXIT_OK if options.mode == 'write' else _EXIT_CHANGED
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
    except (OSError, UnicodeDecodeError) as e:
        return json_error_record(str_path, 'error', str(e))
    except (SyntaxError, tokenize.TokenError, ValueError) as e:
        return json_error_record(str_path, 'invalid', str(e))

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
                f'error: {visible_controls(str(path))}: {visible_controls(file_plan.diagnostic)}',
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
    """Handle JSON output mode."""
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
) -> None:
    """Collect and print stats output."""
    if not files:
        return

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


def _run_stats_mode(args: argparse.Namespace, files: list[Path]) -> int:
    """Handle stats output mode."""
    try:
        _print_stats(files=files, args=args)
    except UnknownStyleError as error:
        print(f'error: {visible_controls(str(error))}', file=sys.stderr)
        return _EXIT_ERROR
    return _EXIT_OK


def _run_cli(argv: Sequence[str] | None) -> int:
    """Process CLI arguments and run the requested command."""
    ensure_utf8_stdio()
    args_list = sys.argv[1:] if argv is None else list(argv)
    if args_list and args_list[0] == 'hook':
        return _run_hook(args_list[1:])
    help_cfg = _resolve_help_config() if _argv_requests_help(args_list) else None
    parser = _build_parser(help_cfg)
    args = parser.parse_args(args_list)
    if args.version:
        return _print_version()
    _validate_args(parser, args)

    mode, detail = _resolve_mode(args)
    files = collect_files(args.paths)
    if mode == 'stats':
        return _run_stats_mode(args, files)

    overrides = _wrap_overrides(args)

    if args.no_incremental or args.incremental_level is not None:
        file_config_incr = None
    else:
        with render_config_warnings():
            file_config_incr = find_incremental_config(files[0]) if files else None
    incr_level, incr_explicit = _resolve_incremental(args, file_config_incr)
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
    color = should_color(sys.stdout, cli_flag=args.color, config_value=color_config)

    opts = _FileModeOptions(
        mode=mode,
        detail=detail,
        verbose=args.verbose,
        color=color,
    )
    return _run_file_mode(overrides, files, incr_ctx, opts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI command and emit its output and diagnostics.

    Configure standard streams for UTF-8 where supported before processing
    arguments. If stdout closes, discard remaining output, examine no further
    files, and return the status of work completed so far.

    Args:
        argv: Command-line arguments, or `None` to use `sys.argv[1:]`.

    Returns:
        For file and statistics commands, 0 on success, including successful
        writes; 1 when changes remain pending; or 2 on processing errors.
        Hook commands use host-specific exit statuses, or 2 when no functioning
        adapter is available.

    Raises:
        SystemExit: When argument parsing displays help or rejects arguments.
    """
    with closable_stdout():
        return _run_cli(argv)


if __name__ == '__main__':
    sys.exit(main())
