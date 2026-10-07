"""Provide Git subprocess and index operations for docwrap hooks."""

import os
import subprocess
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from docwrap._files import decode_python_source, detect_newline, encode_with_newline
from docwrap._git_utils import (
    decode_git_quoted_path,
    git_executable,
    git_failure_detail,
    run_git_command_bytes,
)
from docwrap._types import BaselineResult, BaselineStatus, Newline
from docwrap._vendor.hookshell._shell import normalize_msys_path

if TYPE_CHECKING:
    from docwrap._vendor.hookshell._grammar import PreCommitAdd

_CAT_FILE_HEADER_FIELD_COUNT = 3
_GIT_RAW_DIFF_FIELD_COUNT = 5
_REGULAR_FILE_MODES = frozenset(('100644', '100755'))


class IndexEntry(NamedTuple):
    """Prepared Git index update.

    Attributes:
        mode: Git file mode for the index entry.
        blob_hash: Object ID of the prepared blob.
        path: Repository-relative path to update.
    """

    mode: str
    blob_hash: str
    path: str


class RepoDiscoveryStatus(Enum):
    """Outcome of checked Git worktree discovery.

    Attributes:
        WORKTREE: A worktree root was discovered.
        NOT_WORKTREE: The path is not inside a Git worktree.
        ERROR: Discovery failed because of an infrastructure error.
    """

    WORKTREE = 'worktree'
    NOT_WORKTREE = 'not_worktree'
    ERROR = 'error'


@dataclass(frozen=True, slots=True)
class RepoDiscoveryResult:
    """Checked Git discovery outcome with a usable root only for a worktree.

    A nonempty root is present if and only if `status` is `WORKTREE`.

    Attributes:
        status: Whether discovery found a worktree, found no worktree, or
            failed.
        root: Usable worktree root for `WORKTREE`; otherwise `None` or empty.
    """

    status: RepoDiscoveryStatus
    root: str | None = None

    def __post_init__(self) -> None:
        """Validate that a nonempty root accompanies exactly the worktree status.

        Raises:
            ValueError: If `WORKTREE` has no nonempty root, or another status
                has a nonempty root.
        """
        has_root = bool(self.root)
        if (self.status is RepoDiscoveryStatus.WORKTREE) is not has_root:
            raise ValueError('invalid repository discovery result')


class StagedFileInfo(NamedTuple):
    """Information about a staged file from `git diff --cached --raw`.

    Attributes:
        path: Repository-relative staged path.
        mode: New Git file mode.
        blob_hash: New staged object ID.
        old_hash: Previous object ID.
        status: Git raw-diff status code.
    """

    path: str
    mode: str
    blob_hash: str
    old_hash: str
    status: str


def _decode_git_pathspec_record(record: bytes) -> bytes | None:
    """Decode one LF-delimited Git pathspec record."""
    return decode_git_quoted_path(record) if record.startswith(b'"') else record


def _decode_git_text(data: bytes) -> str:
    """Decode non-path Git protocol text without losing bytes."""
    return data.decode('utf-8', errors='surrogateescape')


def _git_marker_present(cwd: str) -> bool | None:
    """Return whether cwd or a parent has `.git`, or `None` on error."""
    try:
        path = Path(cwd).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not path.is_dir():
        return None

    for directory in (path, *path.parents):
        try:
            (directory / '.git').lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return None
        return True
    return False


def _iter_raw_diff_records(output: bytes) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Yield raw diff headers with their path fields."""
    parts = output.split(b'\0')
    i = 0
    while i + 1 < len(parts):
        header = _decode_git_text(parts[i])
        fields = header.lstrip(':').split()
        if len(fields) < _GIT_RAW_DIFF_FIELD_COUNT:
            i += 2
            continue

        status = fields[4]
        if status[0] in ('R', 'C'):
            if i + 2 >= len(parts):
                break
            yield header, (os.fsdecode(parts[i + 1]), os.fsdecode(parts[i + 2]))
            i += 3
        else:
            yield header, (os.fsdecode(parts[i + 1]),)
            i += 2


def _mark_remaining_infra(
    results: dict[str, BaselineResult],
    info_list: list[StagedFileInfo],
    start_idx: int,
) -> None:
    """Mark all entries from start_idx onward as infra errors."""
    for remaining in info_list[start_idx:]:
        results[remaining.path] = BaselineResult(status=BaselineStatus.INFRA_ERROR)


def _parse_cat_file_output(
    data: bytes,
    info_list: list[StagedFileInfo],
) -> dict[str, BaselineResult]:
    """Parse raw git cat-file --batch output into BaselineResult entries."""
    results: dict[str, BaselineResult] = {}
    pos = 0
    for idx, info in enumerate(info_list):
        nl_idx = data.find(b'\n', pos)
        if nl_idx == -1:
            _mark_remaining_infra(results, info_list, idx)
            return results

        header_bytes = data[pos:nl_idx]
        requested_object = info.blob_hash.encode()

        if header_bytes == requested_object + b' missing':
            results[info.path] = BaselineResult(status=BaselineStatus.MISSING)
            pos = nl_idx + 1
            continue

        parts = header_bytes.rsplit(b' ', 2)
        if (
            len(parts) < _CAT_FILE_HEADER_FIELD_COUNT
            or parts[0] != requested_object
            or parts[1] != b'blob'
        ):
            _mark_remaining_infra(results, info_list, idx)
            return results

        try:
            size = int(parts[2])
        except ValueError:
            _mark_remaining_infra(results, info_list, idx)
            return results
        if size < 0:
            _mark_remaining_infra(results, info_list, idx)
            return results

        content_start = nl_idx + 1
        content_end = content_start + size
        if content_end >= len(data) or data[content_end : content_end + 1] != b'\n':
            _mark_remaining_infra(results, info_list, idx)
            return results
        raw = data[content_start:content_end]
        pos = content_end + 1

        try:
            decoded = decode_python_source(raw)
            results[info.path] = BaselineResult(
                status=BaselineStatus.CONTENT,
                content=decoded.source,
                encoding=decoded.encoding,
                newline=detect_newline(raw),
            )
        except (UnicodeDecodeError, SyntaxError):
            results[info.path] = BaselineResult(status=BaselineStatus.DECODE_FAILED)

    return results


def _parse_null_separated_py_paths(
    result: subprocess.CompletedProcess[bytes] | None,
) -> list[str] | None:
    """Extract sorted .py paths from null-separated git output."""
    if result is None or result.returncode != 0:
        return None
    return sorted(
        os.fsdecode(name) for name in result.stdout.split(b'\0') if name.endswith(b'.py')
    )


def _read_git_pathspec_file(
    cwd: str, pathspec_from_file: str, *, pathspec_file_nul: bool
) -> list[str] | None:
    """Read literal or Git-quoted pathspec-file records."""
    try:
        path = Path(normalize_msys_path(pathspec_from_file))
        if not path.is_absolute():
            path = Path(cwd) / path
        raw = path.read_bytes()
    except OSError:
        return None

    delimiter = b'\0' if pathspec_file_nul else b'\n'
    records: list[bytes] = []
    for part in raw.split(delimiter):
        record = part if pathspec_file_nul else part.removesuffix(b'\r')
        if not record:
            continue
        if not pathspec_file_nul:
            decoded = _decode_git_pathspec_record(record)
            if decoded is None:
                return None
            record = decoded
        records.append(record)
    return [os.fsdecode(record) for record in records]


def _collect_pathspecs(
    cwd: str,
    pathspecs: list[str],
    *,
    pathspec_from_file: str | None = None,
    pathspec_file_nul: bool = False,
) -> list[str] | None:
    """Combine inline and file-provided pathspecs."""
    if pathspec_from_file == '-':
        return []

    all_pathspecs = list(pathspecs)
    if pathspec_from_file is not None:
        file_pathspecs = _read_git_pathspec_file(
            cwd,
            pathspec_from_file,
            pathspec_file_nul=pathspec_file_nul,
        )
        if file_pathspecs is None:
            return None
        all_pathspecs.extend(file_pathspecs)
    return all_pathspecs


def _resolve_git_pathspecs(
    cwd: str,
    pathspecs: list[str] | None,
    repo_root: str,
) -> list[str] | None:
    """Resolve prepared pathspecs to tracked Python files through Git."""
    if pathspecs is None:
        return None
    if not pathspecs:
        return []

    result = run_git_command_bytes(
        [
            'git',
            'ls-files',
            '-z',
            '--full-name',
            '--',
            *(os.fsencode(pathspec) for pathspec in pathspecs),
        ],
        cwd,
    )
    if result is None or result.returncode != 0:
        return None

    py_paths: list[str] = []
    for name_bytes in result.stdout.split(b'\0'):
        if not name_bytes or not name_bytes.endswith(b'.py'):
            continue
        name = os.fsdecode(name_bytes)
        full_path = Path(repo_root) / name
        if full_path.exists() and not full_path.is_symlink():
            py_paths.append(name)

    return sorted(py_paths)


def _run_git_command(
    args: list[str],
    cwd: str | None = None,
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run a text-mode git command with standard capture settings."""
    try:
        executable = git_executable()
        if executable is None:
            return None
        return subprocess.run(  # noqa: S603
            [executable, *args[1:]],
            input=stdin,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=cwd,
            check=False,
        )
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _get_empty_tree_hash(cwd: str) -> str:
    """Return the hash of the empty tree for this repository."""
    result = _run_git_command(['git', 'mktree'], cwd, stdin='')
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return '4b825dc642cb6eb9a060e54bf8d69288fbee4904'


def _git_config_bool(key: str, cwd: str | None = None) -> bool | None:
    """Read a git config boolean, returning None if not set."""
    result = _run_git_command(['git', 'config', '--type=bool', '--get', key], cwd)
    if result is not None and result.returncode == 0:
        return result.stdout.strip() == 'true'
    return None


def apply_index_entries(cwd: str, entries: list[IndexEntry]) -> bool:
    """Atomically update multiple index entries.

    Args:
        cwd: Working directory for the Git command.
        entries: Prepared index entries to update.

    Returns:
        `True` when every entry is applied, otherwise `False`.
    """
    if not entries:
        return True
    stdin_data = b''.join(
        e.mode.encode('ascii')
        + b' '
        + e.blob_hash.encode('ascii')
        + b'\t'
        + os.fsencode(e.path)
        + b'\0'
        for e in entries
    )
    result = run_git_command_bytes(
        ['git', 'update-index', '-z', '--index-info'],
        cwd,
        stdin=stdin_data,
    )
    return result is not None and result.returncode == 0


def can_reuse_repo_root(directory: Path, root: str) -> bool:
    """Check whether a directory remains inside an already discovered worktree.

    Args:
        directory: Resolved directory containing an input file.
        root: Successfully discovered worktree root; never a fallback
            directory.

    Returns:
        True when the directory is below the root without an intervening `.git`
        file or directory marking a nested worktree.
    """
    root_path = Path(root)
    if not directory.is_relative_to(root_path):
        return False
    while directory != root_path:
        if (directory / '.git').exists():
            return False
        directory = directory.parent
    return True


def discover_repo_root(cwd: str) -> RepoDiscoveryResult:
    """Discover a worktree root without conflating absence with failure.

    Args:
        cwd: Directory from which Git should discover a worktree.

    Returns:
        Explicit worktree, non-worktree, or infrastructure-error result.
    """
    result = run_git_command_bytes(['git', 'rev-parse', '--show-toplevel'], cwd)

    if result is None:
        return RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)
    if result.returncode == 0:
        root = os.fsdecode(result.stdout.removesuffix(b'\n'))
        if root:
            return RepoDiscoveryResult(
                RepoDiscoveryStatus.WORKTREE,
                normalize_msys_path(root),
            )
        return RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    bare_result = _run_git_command(
        ['git', 'rev-parse', '--is-bare-repository'],
        cwd,
    )
    if (
        bare_result is not None
        and bare_result.returncode == 0
        and bare_result.stdout.strip() == 'true'
    ):
        return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    marker_present = None if bare_result is None else _git_marker_present(cwd)
    if marker_present is False:
        return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)
    return RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)


def get_modified_tracked_py_files(cwd: str) -> list[str] | None:
    """Return sorted list of modified tracked .py files (staged + unstaged).

    For initial commits (no HEAD), diffs against the empty tree.

    Args:
        cwd: Working directory for Git commands.

    Returns:
        Sorted modified Python paths, or `None` on Git failure.
    """
    head_check = _run_git_command(['git', 'rev-parse', '--verify', 'HEAD'], cwd)
    diff_target = (
        'HEAD'
        if head_check is not None and head_check.returncode == 0
        else _get_empty_tree_hash(cwd)
    )

    result = run_git_command_bytes(
        [
            'git',
            'diff',
            '-z',
            '--no-relative',
            '--no-renames',
            '--diff-filter=d',
            '--name-only',
            diff_target,
        ],
        cwd,
    )
    return _parse_null_separated_py_paths(result)


def get_repo_diff(
    repo_root: str,
    *,
    cached: bool = False,
) -> str | None:
    """Diff the repository against `HEAD`, preserving rename pairing.

    Args:
        repo_root: Repository root directory.
        cached: If True, diff the index against `HEAD`.

    Returns:
        Raw diff output string, or None on git failure.
    """
    cmd: list[str | bytes] = [
        'git',
        '-c',
        'core.quotePath=false',
        'diff',
        '--no-color',
        '--no-ext-diff',
        '--no-textconv',
        '--src-prefix=a/',
        '--dst-prefix=b/',
        '-U0',
        '--inter-hunk-context=0',
        '-M',
    ]
    if cached:
        cmd.append('--cached')
    cmd.extend(['HEAD', '--'])
    result = run_git_command_bytes(cmd, repo_root)
    if result is None or result.returncode != 0:
        return None
    return _decode_git_text(result.stdout)


def get_staged_py_info(cwd: str) -> list[StagedFileInfo] | None:
    """Get staged .py file info (path, mode, blob hash) in a single git call.

    Args:
        cwd: Working directory for the Git command.

    Returns:
        Sorted staged Python file metadata, or `None` on Git failure.
    """
    result = run_git_command_bytes(
        [
            'git',
            'diff',
            '--cached',
            '--raw',
            '-z',
            '--diff-filter=d',
            '-M',
            '--no-relative',
            '--no-abbrev',
        ],
        cwd,
    )
    if result is None or result.returncode != 0:
        return None

    output = result.stdout
    if not output:
        return []

    entries: list[StagedFileInfo] = []
    for header, paths in _iter_raw_diff_records(output):
        path = paths[-1]
        fields = header.lstrip(':').split()
        new_mode = fields[1]
        old_hash = fields[2]
        new_hash = fields[3]
        status = fields[4]

        if path.endswith('.py') and new_mode in _REGULAR_FILE_MODES:
            entries.append(StagedFileInfo(path, new_mode, new_hash, old_hash, status))

    return sorted(entries, key=lambda e: e.path)


def get_tracked_files(repo_root: str, paths: Collection[str]) -> set[str] | None:
    """Select requested tracked paths from one bounded repository query.

    Args:
        repo_root: Repository root directory.
        paths: Repository-relative paths whose tracked state is requested.

    Returns:
        Requested tracked paths, or `None` on Git failure.
    """
    if not paths:
        return set()
    result = run_git_command_bytes(
        [
            'git',
            'ls-files',
            '-z',
            '--',
        ],
        repo_root,
    )
    if result is None or result.returncode != 0:
        return None
    return set(paths).intersection(
        os.fsdecode(name) for name in result.stdout.split(b'\0') if name
    )


def is_head_unborn(cwd: str) -> bool:
    """Report whether `HEAD` names a branch that has no commits yet.

    Args:
        cwd: Working directory for Git commands.

    Returns:
        `True` only when Git confirms an unborn branch; `False` for any other
        state, including Git failure.
    """
    branch = _run_git_command(['git', 'symbolic-ref', '--quiet', 'HEAD'], cwd)
    if branch is None or branch.returncode != 0:
        return False
    head = _run_git_command(['git', 'rev-parse', '--verify', '--quiet', 'HEAD'], cwd)
    return head is not None and head.returncode == 1


def is_hook_enabled(cwd: str | None = None) -> bool:
    """Check if the docwrap hook is enabled via git config.

    The hook defaults to enabled. Set `docwrap.hook=false` to disable it for a
    repository or user.

    Args:
        cwd: Repository directory for local configuration, or `None` for the
            current Git context.

    Returns:
        The configured hook state, defaulting to `True` when unset.
    """
    enabled = _git_config_bool('docwrap.hook', cwd=cwd)
    return True if enabled is None else enabled


def prepare_index_entry(  # noqa: PLR0913
    cwd: str,
    path: str,
    content: str,
    mode: str,
    *,
    encoding: str = 'utf-8',
    newline: Newline = '\n',
) -> IndexEntry | None:
    """Create a blob for content without touching the index.

    Args:
        cwd: Working directory for the Git command.
        path: Repository-relative path for the index entry.
        content: Text to store in the blob.
        mode: Git file mode for the index entry.
        encoding: Encoding for the blob content.
        newline: Line ending to apply to the blob content.

    Returns:
        The prepared index entry, or `None` on Git failure.
    """
    hash_result = run_git_command_bytes(
        ['git', 'hash-object', '-w', '--stdin'],
        cwd,
        stdin=encode_with_newline(content, encoding, newline),
    )
    if hash_result is None or hash_result.returncode != 0:
        return None
    return IndexEntry(mode=mode, blob_hash=hash_result.stdout.decode('ascii').strip(), path=path)


def read_staged_blobs(
    cwd: str,
    info_list: list[StagedFileInfo],
) -> dict[str, BaselineResult] | None:
    r"""Read multiple staged blob contents in a single git cat-file call.

    Args:
        cwd: Working directory for the Git command.
        info_list: Staged file metadata identifying blobs to read.

    Returns:
        A mapping from path to baseline result, or `None` on Git failure.
    """
    if not info_list:
        return {}

    input_data = ('\n'.join(info.blob_hash for info in info_list) + '\n').encode()
    proc = run_git_command_bytes(
        ['git', 'cat-file', '--batch'],
        cwd,
        stdin=input_data,
    )
    if proc is None or proc.returncode != 0:
        return None

    return _parse_cat_file_output(proc.stdout, info_list)


def resolve_pathspec_py_files(
    cwd: str,
    pathspecs: list[str],
    *,
    repo_root: str,
    pathspec_from_file: str | None = None,
    pathspec_file_nul: bool = False,
) -> list[str] | None:
    """Resolve pathspecs to tracked .py files that exist on disk.

    Args:
        cwd: Working directory for Git and pathspec-file resolution.
        pathspecs: Command-line pathspecs to resolve.
        repo_root: Repository root used to verify resolved paths.
        pathspec_from_file: Optional file containing additional pathspecs.
        pathspec_file_nul: Whether the pathspec file uses NUL separators.

    Returns:
        Sorted tracked Python paths, or `None` on failure.
    """
    collected = _collect_pathspecs(
        cwd,
        pathspecs,
        pathspec_from_file=pathspec_from_file,
        pathspec_file_nul=pathspec_file_nul,
    )
    return _resolve_git_pathspecs(cwd, collected, repo_root)


def run_pre_commit_adds(adds: 'tuple[PreCommitAdd, ...]') -> str | None:
    """Replay preceding git adds and report the first failure.

    Args:
        adds: Parsed git add commands preceding the commit.

    Returns:
        `None` when every add succeeds; otherwise the first Git failure detail.
    """
    for add in adds:
        result = run_git_command_bytes(list(add.tokens), add.exec_cwd, catch_value_error=False)
        if result is None or result.returncode != 0:
            return git_failure_detail(result)
    return None
