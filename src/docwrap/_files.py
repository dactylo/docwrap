"""Shared file I/O utilities for docwrap."""

import io
import os
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

from docwrap._color import visible_controls
from docwrap._git_utils import git_failure_detail, run_git_command_bytes
from docwrap._types import Newline

_SKIP_DIRS = frozenset(
    {
        '.git',
        '.hg',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        '.svn',
        '.tox',
        '.venv',
        '__pycache__',
        'build',
        'dist',
        'venv',
    }
)


@dataclass(frozen=True, slots=True)
class ReadResult:
    """Decoded Python source and the exact byte snapshot it came from.

    Attributes:
        raw: Original source bytes, retaining encoding and newline spellings.
        source: Source decoded with `encoding` and universal newlines, so CRLF
            and CR line endings become LF.
        encoding: Python source encoding detected from the byte snapshot.
    """

    raw: bytes
    source: str
    encoding: str


class RestoreError(OSError):
    """A failed write whose original bytes could not be restored.

    The original write error remains available as `__cause__`, and `errno`
    retains its error number. The message includes both the write and restore
    failures so callers that display the exception can describe the damage.
    """


def _filter_git_ignored(directory: Path, files: list[Path]) -> tuple[list[Path], str | None]:
    """Remove Git-ignored paths and return a failure detail when filtering fails."""
    if not files:
        return files, None
    relative_files = [(path, os.fsencode(os.path.relpath(path, directory))) for path in files]
    result = run_git_command_bytes(
        ['git', 'check-ignore', '-z', '--stdin'],
        directory,
        stdin=b'\0'.join(raw_path for _, raw_path in relative_files) + b'\0',
        catch_value_error=False,
    )
    if result is None or result.returncode not in (0, 1):
        return files, git_failure_detail(result)
    ignored = frozenset(result.stdout.split(b'\0'))
    return [path for path, raw_path in relative_files if raw_path not in ignored], None


def _included_nested_worktrees(
    directory: Path,
    nested_worktrees: list[Path],
    failures: dict[Path, str],
) -> set[Path]:
    """Find nested worktrees retained by their parent's Git exclusions."""
    included = {directory}
    for root in nested_worktrees:
        parent = next(
            (
                candidate
                for candidate in reversed(nested_worktrees)
                if candidate != root and root.is_relative_to(candidate)
            ),
            directory,
        )
        if parent in included:
            retained, detail = _filter_git_ignored(parent, [root])
            if detail is not None:
                failures.setdefault(parent, detail)
            if retained:
                included.add(root)
    return included


def _inside_worktree(directory: Path) -> bool:
    """Detect a Git marker in the directory or one of its parents."""
    resolved = directory.resolve()
    return any(
        (root / '.git').is_dir() or (root / '.git').is_file()
        for root in (resolved, *resolved.parents)
    )


def _collect_directory(directory: Path) -> list[Path]:
    """Collect Python files from one directory operand."""
    children: list[Path] = []
    nested_worktrees: list[Path] = []
    for root, dirnames, filenames in os.walk(
        directory,
        topdown=True,
        followlinks=False,
    ):
        current = Path(root)
        if current != directory and ('.git' in dirnames or '.git' in filenames):
            nested_worktrees.append(current)
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        children.extend(
            child
            for name in sorted(filenames)
            if name.endswith('.py') and not (child := current / name).is_symlink()
        )

    by_worktree: dict[Path, list[Path]] = {directory: []}
    failures: dict[Path, str] = {}
    included_worktrees = _included_nested_worktrees(directory, nested_worktrees, failures)
    for path in sorted(children):
        owner = next(
            (root for root in reversed(nested_worktrees) if path.is_relative_to(root)),
            directory,
        )
        if owner in included_worktrees:
            by_worktree.setdefault(owner, []).append(path)
    retained_files: list[Path] = []
    for root, worktree_files in by_worktree.items():
        retained, detail = _filter_git_ignored(root, worktree_files)
        if detail is not None:
            failures.setdefault(root, detail)
        retained_files.extend(retained)
    for root, detail in failures.items():
        if root != directory or _inside_worktree(directory):
            warning = (
                f'warning: cannot apply Git exclusions in {root}: {detail}; '
                'keeping every discovered Python file'
            )
            print(visible_controls(warning), file=sys.stderr)
    return sorted(retained_files)


def collect_files(paths: list[str]) -> list[Path]:
    """Expand explicit paths into retained Python file paths.

    Directory operands filter discovered files through standard Git exclusions
    when available and retain filesystem results on Git failure. Existing
    non-Python file paths are ignored, while missing explicit paths are
    retained.

    Args:
        paths: Explicit file and directory paths to expand.

    Returns:
        Retained Python and missing explicit paths.
    """
    result: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            result.extend(_collect_directory(p))
        else:
            result.extend((p,) if p.suffix == '.py' or not p.exists() else ())
    seen_ids: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    deduped: list[Path] = []
    for p in result:
        try:
            st = p.stat()
        except OSError:
            pass
        else:
            if st.st_ino != 0:
                key = (st.st_dev, st.st_ino)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                deduped.append(p)
                continue
        resolved = p.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            deduped.append(p)
    return deduped


def decode_python_source(raw: bytes) -> ReadResult:
    """Decode raw Python source with PEP 263 and universal newlines.

    Args:
        raw: Exact Python source bytes.

    Returns:
        Decoded source, detected encoding, and the original bytes.

    Raises:
        SyntaxError: If the encoding cookie is malformed or unknown.
        UnicodeDecodeError: If the bytes are invalid for the detected encoding.
    """
    encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
    with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding) as stream:
        source = stream.read()
    return ReadResult(raw=raw, source=source, encoding=encoding)


def detect_newline(raw: bytes) -> Newline:
    """Select CRLF when raw bytes contain any CRLF, otherwise select LF.

    Args:
        raw: Source bytes to inspect.

    Returns:
        `CRLF` when any CRLF sequence is present, otherwise `LF`.
    """
    if b'\r\n' in raw:
        return '\r\n'
    return '\n'


def encode_with_newline(source: str, encoding: str, newline: Newline) -> bytes:
    r"""Encode source after normalizing CRLF and applying the requested newline.

    CRLF sequences are normalized to LF before the requested newline is
    applied. Lone `\r` characters are unchanged.

    Args:
        source: The text to encode.
        encoding: Target encoding, such as `utf-8`.
        newline: Requested `\n` or `\r\n` line ending.

    Returns:
        Encoded source bytes.
    """
    output = source.replace('\r\n', '\n')
    if newline == '\r\n':
        output = output.replace('\n', '\r\n')
    return output.encode(encoding)


def read_file_result(path: Path) -> ReadResult:
    """Read one Python source snapshot with encoding metadata.

    Args:
        path: Python source path.

    Returns:
        Decoded source, detected encoding, and exact original bytes.

    Raises:
        OSError: If the path cannot be read.
        SyntaxError: If the encoding cookie is malformed or unknown.
        UnicodeDecodeError: If the bytes are invalid for the detected encoding.
    """
    return decode_python_source(path.read_bytes())


def read_file(path: Path) -> tuple[str, str]:
    """Read a Python file respecting its encoding cookie.

    Args:
        path: Python source path.

    Returns:
        Tuple of decoded source text and detected encoding.

    Raises:
        OSError: If the path cannot be read.
        SyntaxError: If the encoding cookie is malformed or unknown.
        UnicodeDecodeError: If the bytes are invalid for the detected encoding.
    """
    result = read_file_result(path)
    return result.source, result.encoding


def write_preserving_newline(
    path: Path,
    source: str,
    encoding: str,
    *,
    raw: bytes,
) -> None:
    """Write source using the input's newline style or LF for empty input.

    Nonempty input selects CRLF when any CRLF exists, otherwise LF.

    Args:
        path: Target file path.
        source: Transformed source text to write.
        encoding: File encoding (from tokenize detection).
        raw: Pre-read raw bytes used to detect the newline style.

    Raises:
        OSError: If the new bytes cannot be written; the original bytes are
            restored first when possible.
        RestoreError: If the original bytes also cannot be restored.
    """
    newline = detect_newline(raw)
    try:
        path.write_bytes(encode_with_newline(source, encoding, newline))
    except OSError as write_error:
        try:
            path.write_bytes(raw)
        except OSError as restore_error:
            message = f'{write_error}; restoring the original bytes also failed: {restore_error}'
            raise RestoreError(write_error.errno, message) from write_error
        raise
