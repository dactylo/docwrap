"""Provide Git executable resolution and quoted-path decoding."""

import os
import shutil
import subprocess
from pathlib import Path

_GIT_C_STYLE_ESCAPES = {
    ord('"'): ord('"'),
    ord('\\'): ord('\\'),
    ord('a'): 7,
    ord('b'): 8,
    ord('f'): 12,
    ord('n'): 10,
    ord('r'): 13,
    ord('t'): 9,
    ord('v'): 11,
}
_GIT_OCTAL_ESCAPE_DIGITS = 3


def decode_git_quoted_path(raw: bytes) -> bytes | None:
    """Decode a complete Git C-quoted byte token, or return `None` if malformed.

    Args:
        raw: Token including its opening and closing double quotes.
    """
    if not raw.startswith(b'"'):
        return None

    decoded = bytearray()
    index = 1
    while index < len(raw):
        byte = raw[index]
        if byte == ord('"'):
            return bytes(decoded) if index == len(raw) - 1 else None
        if byte != ord('\\'):
            decoded.append(byte)
            index += 1
            continue

        index += 1
        if index == len(raw):
            return None
        escaped = raw[index]
        replacement = _GIT_C_STYLE_ESCAPES.get(escaped)
        if replacement is not None:
            decoded.append(replacement)
            index += 1
            continue

        digits = raw[index : index + _GIT_OCTAL_ESCAPE_DIGITS]
        if (
            len(digits) != _GIT_OCTAL_ESCAPE_DIGITS
            or digits[0] not in b'0123'
            or any(digit not in b'01234567' for digit in digits[1:])
        ):
            return None
        decoded.append(int(digits, 8))
        index += _GIT_OCTAL_ESCAPE_DIGITS
    return None


def git_executable() -> str | None:
    """Find Git in absolute PATH directories without implicit cwd lookup.

    Absolute PATH entries are trusted in their configured order, including
    wrappers. Empty and relative entries are ignored. Return `None` when no
    executable qualifies; retain the platform's PATHEXT handling.
    """
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        if not Path(directory).is_absolute():
            continue
        executable = shutil.which(str(Path(directory) / 'git'))
        if executable is not None:
            return executable
    return None


def git_failure_detail(result: subprocess.CompletedProcess[bytes] | None) -> str:
    """Describe why a Git command could not provide a usable result.

    Args:
        result: Failed Git command, or `None` when Git could not be run.

    Returns:
        The first nonblank stderr line, or a stable fallback when stderr is
        empty or Git could not be run.
    """
    if result is None:
        return 'git could not be run'
    for line in result.stderr.decode(errors='replace').splitlines():
        if detail := line.strip():
            return detail
    return f'git exited with status {result.returncode}'


def run_git_command_bytes(
    args: list[str | bytes],
    cwd: str | Path,
    *,
    stdin: bytes | None = None,
    catch_value_error: bool = True,
) -> subprocess.CompletedProcess[bytes] | None:
    """Run a binary-mode Git command with standard capture settings.

    Args:
        args: Git command and arguments.
        cwd: Directory in which to run the command.
        stdin: Optional bytes to send to Git.
        catch_value_error: Whether invalid subprocess arguments return `None`.

    Returns:
        Completed command, or `None` when Git is unavailable or launching
        fails.
    """
    try:
        executable = git_executable()
        if executable is None:
            return None
        return subprocess.run(  # noqa: S603
            [executable, *args[1:]],
            input=stdin,
            capture_output=True,
            cwd=cwd,
            check=False,
        )
    except OSError:
        return None
    except ValueError:
        if catch_value_error:
            return None
        raise
