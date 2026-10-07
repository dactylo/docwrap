"""Normalize cross-platform paths and resolve repository routes."""

import os
import re
import stat
import sys
from pathlib import Path

_MSYS_MOUNT_RE = re.compile(r'^/([a-zA-Z])(/.*)?$')


def is_msys_mount_path(path_str: str) -> bool:
    """Return whether the entire path uses the single-letter MSYS mount syntax."""
    return _MSYS_MOUNT_RE.fullmatch(path_str) is not None


def normalize_msys_path(path_str: str) -> str:
    """Convert MSYS2/Git Bash mount paths to Windows format.

    /c/Users/... -> C:/Users/...
    /d/ -> D:/
    /c -> C:/
    Already-Windows paths and non-matching paths pass through unchanged.

    Args:
        path_str: Path string that may be in POSIX mount format.

    Returns:
        The normalized Windows path on Windows, otherwise the original path.
    """
    if sys.platform != 'win32':
        return path_str
    m = _MSYS_MOUNT_RE.match(path_str)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2) or '/'
        return f'{drive}:{rest}'
    return path_str


def resolve_route(route: str, base: str | None, *, logical: bool) -> str | None:
    """Resolve and validate one literal repository route.

    Args:
        route: Literal directory route, including an MSYS mount path.
        base: Known directory a relative route is resolved against, or `None`
            when the invocation directory is unknown; an absolute route never
            reads it.
        logical: Whether to normalize path components without resolving
            symlinks.

    Returns:
        The resolved directory path, or `None` if it is not an existing
        directory.

    Raises:
        ValueError: If the route is relative and `base` is `None`; the grammar
            rejects that combination before calling.
    """
    normalized = normalize_msys_path(route)
    path = Path(normalized)
    if not path.is_absolute():
        if base is None:
            raise ValueError('a relative route requires a known base directory')
        path = Path(base) / path
    path = Path(os.path.normpath(path.absolute())) if logical else path.resolve()
    try:
        mode = path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not stat.S_ISDIR(mode):
        return None
    return str(path)
