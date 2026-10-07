"""Engine-backed hook work: commit wrapping and PostToolUse editing."""

import contextlib
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from docwrap._color import path_text, visible_controls
from docwrap._config import find_config, render_config_warnings, resolve_config
from docwrap._core import wrap_source_report_eligible
from docwrap._core._incremental import (
    DiffHunk,
    derive_eligible_target,
    get_eligible_ranges,
    parse_diff_hunks,
    resolve_incremental_level,
)
from docwrap._files import RestoreError, read_file_result, write_preserving_newline
from docwrap._git import (
    IndexEntry,
    RepoDiscoveryStatus,
    StagedFileInfo,
    apply_index_entries,
    can_reuse_repo_root,
    discover_repo_root,
    get_modified_tracked_py_files,
    get_repo_diff,
    get_staged_py_info,
    get_tracked_files,
    is_hook_enabled,
    prepare_index_entry,
    read_staged_blobs,
    resolve_pathspec_py_files,
    run_pre_commit_adds,
)
from docwrap._hook_front import Outcome, allow, fail
from docwrap._types import (
    BaselineStatus,
    DocstringSkip,
    WorktreeWrapContext,
    WorktreeWrite,
)
from docwrap._vendor.hookshell._grammar import CommitPlan, PreCommitAdd
from docwrap._vendor.hookshell._shell import normalize_msys_path

if TYPE_CHECKING:
    from docwrap._types import BaselineResult, DocwrapConfig, IncrementalLevel
    from docwrap._vendor.hookshell._grammar import CommitMode


@dataclass(frozen=True, slots=True)
class _PathspecResult:
    """Distinct success and failure paths for pathspec processing."""

    failure: Outcome | None = None
    wrapped_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PostToolFile:
    """Prepared PostToolUse file state."""

    path: Path
    raw: bytes
    source: str
    encoding: str
    config: 'DocwrapConfig'
    incr_level: 'IncrementalLevel | None'
    repo_root: str


@dataclass(frozen=True, slots=True)
class _PreToolContext:
    """Resolved state after classification and config resolution."""

    cwd: str
    repo_root: str
    incr_level: 'IncrementalLevel | None'
    allow_on_failure: bool
    mode: 'CommitMode'
    pre_commit_adds: tuple[PreCommitAdd, ...]


@dataclass(frozen=True, slots=True)
class _ProcessedStagedBlob:
    """Prepared changes for one staged blob."""

    index_entry: IndexEntry
    worktree_write: WorktreeWrite | None = None


@dataclass(frozen=True, slots=True)
class _WrapResult:
    """Result of wrapping a single source."""

    source: str
    changed: bool
    failed: bool = False
    skips: tuple[DocstringSkip, ...] = ()


@dataclass(frozen=True, slots=True)
class _WrapWorktreeResult:
    """Result of wrapping working-tree files."""

    failure: Outcome | None = None


def _eligible_from_hunks(
    source: str,
    hunks: list[DiffHunk] | None,
    incr_level: 'IncrementalLevel | None',
    encoding: str,
) -> set[tuple[int, int]] | None:
    """Convert preselected diff hunks to eligible source ranges."""
    if incr_level is None or hunks is None:
        return None
    try:
        return get_eligible_ranges(source, hunks, level=incr_level, encoding=encoding)
    except SyntaxError:
        # Invalid Python falls back to full-file processing, which reports
        # the parse failure as a per-file skip instead of denying the hook.
        return None


def _hook_config(cfg: 'DocwrapConfig') -> 'DocwrapConfig':
    """Suppress convert/strip_types for hook wrapping."""
    return replace(cfg, convert=None, strip_types=False)


def _load_staged_info(
    ctx: _PreToolContext,
) -> list[StagedFileInfo] | Outcome:
    """Load staged Python file info, returning Outcome on failure."""
    info_list = get_staged_py_info(ctx.cwd)
    if info_list is None:
        return fail('docwrap: failed to list staged files', allow_on_failure=ctx.allow_on_failure)
    return info_list


def _prepare_post_tool_file(
    path: Path,
    incremental_override: bool | None,
    *,
    repo_root: str,
) -> _PostToolFile:
    """Read and resolve per-file PostToolUse state once."""
    read_result = read_file_result(path)
    with render_config_warnings():
        resolved = find_config(path)
    return _PostToolFile(
        path=path,
        raw=read_result.raw,
        source=read_result.source,
        encoding=read_result.encoding,
        config=_hook_config(resolved.config),
        incr_level=resolve_incremental_level(
            override=incremental_override,
            config_value=resolved.incremental,
        ),
        repo_root=repo_root,
    )


def _print_skips(path: str, skips: Sequence[DocstringSkip]) -> None:
    """Print informational safety-skip lines to stderr."""
    if not skips:
        return
    from docwrap._display._skips import format_safety_skip  # noqa: PLC0415

    for skip in skips:
        print(
            f'docwrap: {format_safety_skip(path, skip)}',
            file=sys.stderr,
        )


def _receipt(count: int) -> str:
    """Return the allow reason naming how many staged files were rewritten."""
    noun = 'file' if count == 1 else 'files'
    return f'docwrap: wrapped {count} staged Python {noun} for this commit'


def _apply_commit_results(
    repo_root: str,
    entries: list[IndexEntry],
    wt_writes: list[WorktreeWrite],
    *,
    allow_on_failure: bool,
) -> Outcome:
    """Apply wrapped index results and safe working-tree synchronization."""
    if entries and not apply_index_entries(repo_root, entries):
        return fail('docwrap: failed to update git index', allow_on_failure=allow_on_failure)

    for wt_write in wt_writes:
        try:
            if not wt_write.path.is_symlink():
                raw_bytes = wt_write.path.read_bytes()
                if raw_bytes != wt_write.authorized_raw:
                    continue
                write_preserving_newline(
                    wt_write.path, wt_write.source, wt_write.encoding, raw=raw_bytes
                )
        except OSError as error:
            print(
                f'WARNING: docwrap: failed to synchronize '
                f'{visible_controls(path_text(wt_write.path))}: {visible_controls(str(error))}',
                file=sys.stderr,
            )

    return Outcome('allow', _receipt(len(entries))) if entries else allow()


def _report_post_tool_error(path: Path, error: Exception) -> None:
    """Report expected file failures tersely and unexpected failures with their traceback."""
    if isinstance(error, (OSError, UnicodeDecodeError, SyntaxError)):
        detail = f': {visible_controls(str(error))}' if isinstance(error, RestoreError) else ''
        print(
            f'docwrap: PostToolUse error processing {visible_controls(path_text(path))}{detail}',
            file=sys.stderr,
        )
    else:
        traceback.print_exc()


def _resolve_discovery_error_allow_on_failure(cwd: str, allow_on_failure: bool) -> bool:
    """Resolve failure policy without swallowing malformed configuration."""
    try:
        config_cwd = Path(cwd).resolve()
    except (OSError, ValueError):
        return allow_on_failure

    try:
        with render_config_warnings():
            resolved = find_config(config_cwd)
    except OSError:
        return allow_on_failure
    return allow_on_failure or resolved.allow_on_failure


def _resolve_edited_paths(paths: Sequence[str], cwd: str) -> list[str]:
    """Resolve reported paths to absolute Python file paths, preserving order."""
    base = Path(normalize_msys_path(cwd))
    resolved: list[str] = []
    for path in paths:
        normalized = normalize_msys_path(path)
        if not normalized.endswith('.py'):
            continue
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = base / candidate
        text = str(candidate)
        if text not in resolved:
            resolved.append(text)
    return resolved


def _resolve_has_pathspec(ctx: _PreToolContext) -> bool:
    """Determine whether the commit has effective pathspecs."""
    if ctx.mode.pathspecs:
        return True
    if ctx.mode.pathspec_from_file is None or ctx.mode.pathspec_from_file == '-':
        return False
    try:
        pf_path = Path(normalize_msys_path(ctx.mode.pathspec_from_file))
        if not pf_path.is_absolute():
            pf_path = Path(ctx.cwd) / pf_path
        return pf_path.stat().st_size > 0
    except OSError:
        return True  # Let resolve_pathspec_py_files handle the error


def _resolve_pre_tool_context(
    plan: 'CommitPlan',
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> _PreToolContext | Outcome | None:
    """Resolve repository state for an accepted commit plan."""
    cwd = plan.cwd
    discovery = discover_repo_root(cwd)
    if discovery.status is RepoDiscoveryStatus.NOT_WORKTREE:
        return None
    if discovery.status is RepoDiscoveryStatus.ERROR:
        merged_allow = _resolve_discovery_error_allow_on_failure(cwd, allow_on_failure)
        return fail(
            'docwrap: failed to discover git work tree',
            allow_on_failure=merged_allow,
        )
    repo_root = discovery.root
    assert repo_root is not None  # noqa: S101 -- narrowed by discovery invariant
    with render_config_warnings():
        resolved = find_config(Path(repo_root))
    merged_allow = allow_on_failure or resolved.allow_on_failure
    incr_level = resolve_incremental_level(
        override=incremental,
        config_value=resolved.incremental,
    )
    return _PreToolContext(
        cwd=cwd,
        repo_root=repo_root,
        incr_level=incr_level,
        allow_on_failure=merged_allow,
        mode=plan.mode,
        pre_commit_adds=plan.pre_commit_adds,
    )


def _to_wt_context(context: _PreToolContext) -> WorktreeWrapContext:
    """Build a worktree-wrapping context from pre-tool state."""
    return WorktreeWrapContext(
        repo_root=Path(context.repo_root),
        incr_level=context.incr_level,
        allow_on_failure=context.allow_on_failure,
    )


def _wrap_with_config(
    source: str,
    config: 'DocwrapConfig',
    eligible_ranges: set[tuple[int, int]] | None,
    incr_level: 'IncrementalLevel | None',
) -> _WrapResult:
    """Wrap source with config, suppressing convert/strip_types."""
    cfg = _hook_config(config)
    eligible_target = derive_eligible_target(incr_level)
    try:
        report = wrap_source_report_eligible(
            source,
            config=cfg,
            eligible_ranges=eligible_ranges,
            eligible_target=eligible_target,
        )
    except (SyntaxError, ValueError):
        return _WrapResult(source=source, changed=False, failed=True)
    return _WrapResult(
        source=report.source,
        changed=report.source != source,
        skips=report.skips,
    )


def _process_post_tool_group(files: Sequence[_PostToolFile]) -> None:
    """Share repository state while processing prepared PostToolUse files."""
    incremental_files = [file for file in files if file.incr_level is not None]
    incremental_groups: dict[str, list[_PostToolFile]] = {}
    for file in incremental_files:
        incremental_groups.setdefault(file.repo_root, []).append(file)
    relative_paths = {
        file.path: file.path.resolve().relative_to(file.repo_root).as_posix()
        for file in incremental_files
    }
    diff_outputs = {root: get_repo_diff(root) for root in incremental_groups}
    hunks_by_root = {
        root: parse_diff_hunks(output) if output is not None else None
        for root, output in diff_outputs.items()
    }
    missing_by_root = {
        root: [
            relative_paths[file.path]
            for file in incremental_groups[root]
            if relative_paths[file.path] not in hunks
        ]
        for root, hunks in hunks_by_root.items()
        if hunks is not None
    }
    tracked_by_root: dict[str, set[str] | None] = {
        root: get_tracked_files(root, missing) if missing else set()
        for root, missing in missing_by_root.items()
    }

    for file in files:
        try:
            relative_path = relative_paths.get(file.path)
            hunks_by_path = hunks_by_root.get(file.repo_root)
            tracked = tracked_by_root.get(file.repo_root)
            diff_hunks = None
            if relative_path is not None and hunks_by_path is not None:
                diff_hunks = hunks_by_path.get(relative_path)
                if diff_hunks is None and tracked is not None and relative_path in tracked:
                    diff_hunks = []
            eligible_ranges = _eligible_from_hunks(
                file.source,
                diff_hunks,
                file.incr_level,
                file.encoding,
            )

            result = _wrap_with_config(
                file.source,
                file.config,
                eligible_ranges,
                file.incr_level,
            )
            _print_skips(str(file.path), result.skips)
            if result.failed or not result.changed:
                continue
            write_preserving_newline(
                file.path,
                result.source,
                file.encoding,
                raw=file.raw,
            )
        except Exception as error:  # noqa: BLE001
            _report_post_tool_error(file.path, error)


def _process_staged_blob(
    ctx: _PreToolContext,
    info: StagedFileInfo,
    baseline: 'BaselineResult',
    diff_hunks: list[DiffHunk] | None,
) -> _ProcessedStagedBlob | Outcome | None:
    """Wrap one staged blob and prepare its index/worktree updates."""
    source = baseline.content
    assert source is not None  # noqa: S101 -- narrowed after content status check
    encoding = baseline.encoding
    newline = baseline.newline
    eligible_ranges = _eligible_from_hunks(source, diff_hunks, ctx.incr_level, encoding)
    with render_config_warnings():
        file_cfg = resolve_config(Path(ctx.repo_root) / info.path)
    wr = _wrap_with_config(source, file_cfg, eligible_ranges, ctx.incr_level)
    _print_skips(info.path, wr.skips)
    if wr.failed or not wr.changed:
        return None

    entry = prepare_index_entry(
        ctx.cwd, info.path, wr.source, info.mode, encoding=encoding, newline=newline
    )
    if entry is None:
        return fail(
            f'docwrap: failed to prepare index entry for {info.path}',
            allow_on_failure=ctx.allow_on_failure,
        )

    wt_write: WorktreeWrite | None = None
    wt_path = Path(ctx.repo_root) / info.path
    wt_raw: bytes | None = None
    try:
        wt_raw = wt_path.read_bytes()
        wt_content = wt_raw.decode(encoding)
        wt_matches_staged = wt_content.replace('\r\n', '\n').replace('\r', '\n') == source
    except (OSError, UnicodeDecodeError):
        wt_matches_staged = False

    if wt_matches_staged and wt_raw is not None:
        wt_write = WorktreeWrite(
            path=wt_path,
            source=wr.source,
            encoding=encoding,
            authorized_raw=wt_raw,
        )

    return _ProcessedStagedBlob(index_entry=entry, worktree_write=wt_write)


def _wrap_staged_blobs(
    ctx: _PreToolContext,
    info_list: list[StagedFileInfo],
) -> tuple[list[IndexEntry], list[WorktreeWrite]] | Outcome:
    """Wrap staged blobs and prepare index entries."""
    contents = read_staged_blobs(ctx.cwd, info_list)
    if contents is None:
        return fail(
            'docwrap: failed to read staged contents', allow_on_failure=ctx.allow_on_failure
        )

    diff_output = (
        get_repo_diff(ctx.repo_root, cached=True)
        if ctx.incr_level is not None and info_list
        else None
    )
    # Git and parser failures both give every staged file full-wrap eligibility.
    hunks_by_path = parse_diff_hunks(diff_output) if diff_output is not None else None

    entries: list[IndexEntry] = []
    wt_writes: list[WorktreeWrite] = []

    for info in info_list:
        baseline = contents.get(info.path)
        if baseline is None:
            return fail(
                f'docwrap: failed to read staged content for {info.path}',
                allow_on_failure=ctx.allow_on_failure,
            )
        if baseline.status is BaselineStatus.MISSING:
            continue
        if baseline.status is BaselineStatus.DECODE_FAILED:
            continue
        if baseline.status is BaselineStatus.INFRA_ERROR:
            return fail(
                f'docwrap: failed to read staged content for {info.path}',
                allow_on_failure=ctx.allow_on_failure,
            )
        result = _process_staged_blob(
            ctx,
            info,
            baseline,
            None if hunks_by_path is None else hunks_by_path.get(info.path, []),
        )
        if isinstance(result, Outcome):
            return result
        if result is None:
            continue
        entries.append(result.index_entry)
        if result.worktree_write is not None:
            wt_writes.append(result.worktree_write)

    return entries, wt_writes


def _wrap_worktree_files(
    ctx: WorktreeWrapContext,
    paths: Sequence[str],
) -> _WrapWorktreeResult:
    """Wrap working-tree files with rollback on fail-closed write errors."""
    repo_root = str(ctx.repo_root)
    incr_level = ctx.incr_level
    allow_on_failure = ctx.allow_on_failure
    diff_output = get_repo_diff(repo_root) if incr_level is not None and paths else None
    # Git and parser failures both give every worktree file full-wrap eligibility.
    hunks_by_path = parse_diff_hunks(diff_output) if diff_output is not None else None
    # Pass 1: read and wrap
    pending: list[tuple[Path, bytes, str, str]] = []

    for rel_path in paths:
        wt_path = ctx.repo_root / rel_path
        if not wt_path.is_file() or wt_path.is_symlink():
            continue
        try:
            read_result = read_file_result(wt_path)
            original_bytes = read_result.raw
            source = read_result.source
            encoding = read_result.encoding
        except (OSError, UnicodeDecodeError, SyntaxError):
            decision = fail(
                f'docwrap: cannot read {rel_path}',
                allow_on_failure=allow_on_failure,
            )
            if not allow_on_failure:
                return _WrapWorktreeResult(failure=decision)
            continue

        with render_config_warnings():
            file_cfg = resolve_config(wt_path)

        diff_hunks = None if hunks_by_path is None else hunks_by_path.get(rel_path, [])
        eligible_ranges = _eligible_from_hunks(source, diff_hunks, incr_level, encoding)

        wr = _wrap_with_config(source, file_cfg, eligible_ranges, incr_level)
        _print_skips(rel_path, wr.skips)
        if wr.failed or not wr.changed:
            continue

        pending.append((wt_path, original_bytes, wr.source, encoding))

    # Pass 2: write with rollback
    written: list[tuple[Path, bytes]] = []
    for wt_path, original_bytes, new_source, encoding in pending:
        try:
            write_preserving_newline(wt_path, new_source, encoding, raw=original_bytes)
            written.append((wt_path, original_bytes))
        except OSError as error:
            detail = f': {visible_controls(str(error))}' if isinstance(error, RestoreError) else ''
            if not allow_on_failure:
                for written_path, orig_bytes in written:
                    with contextlib.suppress(OSError):
                        written_path.write_bytes(orig_bytes)
                # Rollback is non-recoverable -- force deny regardless of
                # caller's allow_on_failure.
                return _WrapWorktreeResult(
                    failure=fail(
                        f'docwrap: failed to write {path_text(wt_path)}{detail}',
                        allow_on_failure=False,
                    )
                )
            print(
                'WARNING: docwrap: failed to write '
                f'{visible_controls(path_text(wt_path))}{detail}',
                file=sys.stderr,
            )

    return _WrapWorktreeResult()


def _handle_auto_stage_commit(ctx: _PreToolContext) -> Outcome:
    """Wrap modified tracked files for an auto-stage commit."""
    wt_paths = get_modified_tracked_py_files(ctx.cwd)
    if wt_paths is None:
        return fail(
            'docwrap: failed to discover modified files',
            allow_on_failure=ctx.allow_on_failure,
        )
    if not wt_paths:
        return allow()
    wt_result = _wrap_worktree_files(_to_wt_context(ctx), wt_paths)
    if wt_result.failure is not None:
        return wt_result.failure
    return allow()


def _resolve_and_wrap_pathspec(
    ctx: WorktreeWrapContext,
    cwd: str,
    mode: 'CommitMode',
) -> _PathspecResult:
    """Resolve modified pathspec files and wrap them on the working tree."""
    wt_paths = resolve_pathspec_py_files(
        cwd,
        list(mode.pathspecs),
        repo_root=str(ctx.repo_root),
        pathspec_from_file=mode.pathspec_from_file,
        pathspec_file_nul=mode.pathspec_file_nul,
    )
    if wt_paths is None:
        return _PathspecResult(
            failure=fail(
                'docwrap: pathspec resolution failed',
                allow_on_failure=ctx.allow_on_failure,
            )
        )
    if not wt_paths:
        return _PathspecResult()
    modified = get_modified_tracked_py_files(cwd)
    if modified is None:
        return _PathspecResult(
            failure=fail(
                'docwrap: modified-file discovery failed',
                allow_on_failure=ctx.allow_on_failure,
            )
        )
    modified_set = frozenset(modified)
    filtered = [p for p in wt_paths if p in modified_set]
    if not filtered:
        return _PathspecResult()
    wt_result = _wrap_worktree_files(ctx, filtered)
    if wt_result.failure is not None:
        return _PathspecResult(failure=wt_result.failure)
    return _PathspecResult(wrapped_paths=tuple(filtered))


def _handle_pathspec_only_commit(ctx: _PreToolContext) -> Outcome:
    """Wrap working-tree files for a pathspec-only commit."""
    result = _resolve_and_wrap_pathspec(_to_wt_context(ctx), ctx.cwd, ctx.mode)
    if result.failure is not None:
        return result.failure
    return allow()


def _wrap_include_pathspecs(
    ctx: _PreToolContext,
    has_pathspec: bool,
    info_list: list[StagedFileInfo],
) -> list[StagedFileInfo] | Outcome:
    """Wrap include pathspecs and filter staged-blob handling."""
    if not (ctx.mode.include and has_pathspec):
        return info_list

    incl_result = _resolve_and_wrap_pathspec(_to_wt_context(ctx), ctx.cwd, ctx.mode)
    if incl_result.failure is not None:
        return incl_result.failure

    if incl_result.wrapped_paths:
        wt_set = frozenset(incl_result.wrapped_paths)
        info_list = [info for info in info_list if info.path not in wt_set]
    return info_list


def _handle_normal_commit(
    ctx: _PreToolContext,
    *,
    has_pathspec: bool,
) -> Outcome:
    """Handle a normal commit (including --include mode)."""
    loaded = _load_staged_info(ctx)
    if isinstance(loaded, Outcome):
        return loaded

    include_result = _wrap_include_pathspecs(ctx, has_pathspec, loaded)
    if isinstance(include_result, Outcome):
        return include_result
    info_list = include_result

    if info_list:
        blob_result = _wrap_staged_blobs(ctx, info_list)
        if isinstance(blob_result, Outcome):
            return blob_result
        entries, wt_writes = blob_result
    else:
        entries, wt_writes = [], []

    return _apply_commit_results(
        ctx.repo_root,
        entries,
        wt_writes,
        allow_on_failure=ctx.allow_on_failure,
    )


def _dispatch_pre_tool_commit(ctx: _PreToolContext, has_pathspec: bool) -> Outcome:
    """Dispatch a resolved commit to its mode-specific handler."""
    if ctx.mode.auto_stage:
        return _handle_auto_stage_commit(ctx)
    if has_pathspec and not ctx.mode.include:
        return _handle_pathspec_only_commit(ctx)
    return _handle_normal_commit(ctx, has_pathspec=has_pathspec)


def _format_commit(
    plan: 'CommitPlan',
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> Outcome:
    """Replay the add, then format the content an accepted commit will consume."""
    ctx: _PreToolContext | None = None
    try:
        resolved = _resolve_pre_tool_context(
            plan,
            allow_on_failure=allow_on_failure,
            incremental=incremental,
        )
        if isinstance(resolved, Outcome):
            return resolved
        ctx = resolved
        if ctx is None:
            return allow()
        if ctx.pre_commit_adds and (detail := run_pre_commit_adds(ctx.pre_commit_adds)):
            return fail(
                f'docwrap: failed to replay git add: {detail}',
                allow_on_failure=ctx.allow_on_failure,
            )
        return _dispatch_pre_tool_commit(ctx, _resolve_has_pathspec(ctx))
    except Exception:  # noqa: BLE001
        effective_allow = ctx.allow_on_failure if ctx is not None else allow_on_failure
        return fail(
            'docwrap: unexpected error during wrapping',
            allow_on_failure=effective_allow,
        )


def run_commit(
    plan: CommitPlan,
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> Outcome:
    """Process a supported commit when the hook is enabled for its repository.

    Args:
        plan: Commit plan accepted by classification.
        allow_on_failure: Whether infrastructure failures warn instead of
            denying.
        incremental: Override incremental mode, or defer to configuration when
            None.

    Returns:
        Allow when the hook is disabled for the plan's directory; otherwise the
        outcome of processing the commit.
    """
    if not is_hook_enabled(plan.cwd):
        return allow()
    return _format_commit(plan, allow_on_failure=allow_on_failure, incremental=incremental)


def run_post_tool(
    edited_paths: Sequence[str],
    cwd: str,
    *,
    incremental: bool | None = None,
) -> None:
    """Wrap the Python files a post-tool payload reports as written.

    Args:
        edited_paths: Paths reported by the host as edited files.
        cwd: Directory used to resolve relative edited paths.
        incremental: Override incremental mode, or defer to configuration when
            None.
    """
    paths = list(
        filter(
            lambda path: not path.is_symlink() and path.is_file(),
            map(Path, _resolve_edited_paths(edited_paths, cwd)),
        )
    )

    groups: dict[str, list[Path]] = {}
    discovered_roots: dict[str, None] = {}
    roots_by_path: dict[Path, str] = {}
    for path in paths:
        resolved_path = path.resolve()
        repo_root = next(
            (root for root in discovered_roots if can_reuse_repo_root(resolved_path.parent, root)),
            None,
        )
        if repo_root is None:
            discovery = discover_repo_root(str(path.parent))
            if discovery.root is not None:
                discovered_roots[discovery.root] = None
            repo_root = discovery.root or str(resolved_path.parent)
        groups.setdefault(repo_root, []).append(path)
        roots_by_path[path] = repo_root

    enabled = {root: is_hook_enabled(root) for root in groups}
    prepared: list[_PostToolFile] = []
    for path in paths:
        repo_root = roots_by_path[path]
        if not enabled[repo_root]:
            continue
        try:
            prepared.append(
                _prepare_post_tool_file(
                    path,
                    incremental,
                    repo_root=repo_root,
                )
            )
        except Exception as error:  # noqa: BLE001
            _report_post_tool_error(path, error)
    _process_post_tool_group(prepared)
