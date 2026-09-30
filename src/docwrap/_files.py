"""Shared file I/O utilities for docwrap."""

import io
import os
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

from docwrap._git_utils import git_executable
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


def _filter_git_ignored(directory: Path, files: list[Path]) -> list[Path]:
    """Remove Git-ignored paths while retaining the input on Git failure."""
    if not files:
        return files
    relative_files = [(path, os.fsencode(os.path.relpath(path, directory))) for path in files]
    try:
        executable = git_executable()
        if executable is None:
            return files
        result = subprocess.run(
            [  # noqa: S607
                'git',
                'check-ignore',
                '-z',
                '--stdin',
            ],
            executable=executable,
            capture_output=True,
            cwd=directory,
            input=b'\0'.join(raw_path for _, raw_path in relative_files) + b'\0',
            check=False,
        )
    except OSError:
        return files
    if result.returncode not in (0, 1):
        return files
    ignored = frozenset(result.stdout.split(b'\0'))
    return [path for path, raw_path in relative_files if raw_path not in ignored]


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
    included_worktrees = {directory}
    for root in nested_worktrees:
        parent = next(
            (
                candidate
                for candidate in reversed(nested_worktrees)
                if candidate != root and root.is_relative_to(candidate)
            ),
            directory,
        )
        if parent in included_worktrees and _filter_git_ignored(parent, [root]):
            included_worktrees.add(root)
    for path in sorted(children):
        owner = next(
            (root for root in reversed(nested_worktrees) if path.is_relative_to(root)),
            directory,
        )
        if owner in included_worktrees:
            by_worktree.setdefault(owner, []).append(path)
    return sorted(
        path
        for root, worktree_files in by_worktree.items()
        for path in _filter_git_ignored(root, worktree_files)
    )


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


def ensure_utf8_stdio() -> None:
    """Reconfigure supported standard streams to UTF-8 on every platform.

    Each standard stream exposing `reconfigure()` retains its existing error
    handler.
    """
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            encoding='utf-8',
            errors=sys.stdin.errors,
        )
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            encoding='utf-8',
            errors=sys.stdout.errors,
        )
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            encoding='utf-8',
            errors=sys.stderr.errors,
        )


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
    raw: bytes | None = None,
    fallback_newline: Newline = '\n',
) -> None:
    """Write source using the input's selected newline or empty-file fallback.

    Nonempty input selects CRLF when any CRLF exists, otherwise LF.

    Args:
        path: Target file path.
        source: Transformed source text to write.
        encoding: File encoding (from tokenize detection).
        raw: Pre-read raw bytes. If None, reads path to detect newline.
        fallback_newline: Newline to use when raw is empty.

    Raises:
        OSError: If the existing path cannot be read or the new bytes cannot be
            written.
    """
    raw_bytes = path.read_bytes() if raw is None else raw
    newline = detect_newline(raw_bytes) if raw_bytes else fallback_newline
    path.write_bytes(encode_with_newline(source, encoding, newline))
