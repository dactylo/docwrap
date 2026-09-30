"""Provide Git executable resolution and quoted-path decoding."""

import os
import shutil
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
