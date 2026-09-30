"""Shield and restore escapes in non-raw docstring bodies."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from docwrap._types import SkipCode

_HEX_DIGITS = frozenset('0123456789ABCDEFabcdef')
_HEX_ESCAPE_DIGIT_COUNTS = {'U': 8, 'u': 4, 'x': 2}
_MAX_OCTAL_DIGITS = 3
_OCTAL_DIGITS = frozenset('01234567')
_SENTINEL_MAX = 0xF8FF
_SENTINEL_MIN = 0xE000
_SIMPLE_ESCAPES = frozenset('abfnrtv' + chr(34) + chr(39) + chr(92))


class EscapeRestoreError(ValueError):
    """Raised when transformed text cannot be mapped back to its escape spellings."""


class EscapeScanError(ValueError):
    """Raised when a docstring body cannot be shielded for safe wrapping.

    Attributes:
        code: Skip code identifying the reason.
    """

    def __init__(self, code: SkipCode, message: str) -> None:
        """Store the skip code and build the message.

        Args:
            code: Skip code identifying the reason.
            message: Human-readable explanation.
        """
        super().__init__(message)
        self.code: SkipCode = code


@dataclass(frozen=True, slots=True)
class ShieldedBody:
    """A docstring body with escapes replaced by unique identity placeholders.

    Restoration associates each identity with its original spelling even if
    wrapping reorders the runs. Each identity must remain one contiguous run of
    the original spelling's length, unless its loss is explicitly allowed.

    Attributes:
        text: Shielded body with each escape replaced by an equal-length run of
            a unique character absent from the original body.
        escape_map: Ordered `(identity_char, spelling)` pairs in source order;
            empty when the body contains no escapes.
    """

    text: str
    escape_map: tuple[tuple[str, str], ...]


def _hex_escape_end(text: str, digits_start: int, count: int) -> int:
    """Return the end index of a hex escape, or the two-char fallback end."""
    end = digits_start + count
    if end <= len(text) and all(ch in _HEX_DIGITS for ch in text[digits_start:end]):
        return end
    return digits_start


def _escape_end(text: str, start: int) -> int:
    """Return the end index of the escape token starting at `start`."""
    nxt = start + 1
    if nxt >= len(text):
        return nxt
    ch = text[nxt]
    if ch == '\n':
        msg = f'backslash-newline line continuation at index {start}'
        raise EscapeScanError('line-continuation', msg)
    if ch in _SIMPLE_ESCAPES:
        return nxt + 1
    if ch in _OCTAL_DIGITS:
        end = nxt + 1
        while end < len(text) and end - nxt < _MAX_OCTAL_DIGITS and text[end] in _OCTAL_DIGITS:
            end += 1
        return end
    if ch in _HEX_ESCAPE_DIGIT_COUNTS:
        return _hex_escape_end(text, nxt + 1, _HEX_ESCAPE_DIGIT_COUNTS[ch])
    if ch == 'N' and text[nxt + 1 : nxt + 2] == '{':
        brace = text.find('}', nxt + 2)
        if brace != -1:
            return brace + 1
    return nxt + 1


def _iter_escape_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) index pairs for each escape token in `text`."""
    i = 0
    while i < len(text):
        if text[i] == '\\':
            end = _escape_end(text, i)
            yield i, end
            i = end
        else:
            i += 1


def restore_escapes(
    text: str,
    escape_map: Sequence[tuple[str, str]],
    *,
    allowed_loss: frozenset[str] = frozenset(),
) -> str:
    """Replace identity runs in `text` with their original spellings.

    Args:
        text: Transformed text containing identity placeholder runs.
        escape_map: Ordered (identity_char, spelling) pairs from shielding.
        allowed_loss: Identity characters whose text an explicitly requested
            transformation deleted; only these may be absent.

    Returns:
        Text with every identity run replaced by its own spelling.

    Raises:
        EscapeRestoreError: If an identity appears more than once, a run length
            does not match its spelling, or an identity outside `allowed_loss`
            is missing.
    """
    spelling_by_identity = dict(escape_map)
    seen: set[str] = set()
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch not in spelling_by_identity:
            out.append(ch)
            i += 1
            continue
        if ch in seen:
            msg = 'identity appears more than once after transformation'
            raise EscapeRestoreError(msg)
        run_end = i
        while run_end < len(text) and text[run_end] == ch:
            run_end += 1
        spelling = spelling_by_identity[ch]
        if run_end - i != len(spelling):
            msg = 'identity run length does not match its spelling'
            raise EscapeRestoreError(msg)
        seen.add(ch)
        out.append(spelling)
        i = run_end
    missing = {identity for identity, _ in escape_map} - seen - allowed_loss
    if missing:
        msg = 'identity missing after transformation'
        raise EscapeRestoreError(msg)
    return ''.join(out)


def scan_escape_spellings(text: str) -> tuple[str, ...]:
    """Return the ordered escape spellings found in `text`.

    Args:
        text: Text to scan (a literal body or a complete literal).

    Returns:
        Tuple of escape spellings in source order.

    Raises:
        EscapeScanError: If the text contains a backslash-newline line
            continuation.
    """
    return tuple(text[start:end] for start, end in _iter_escape_spans(text))


def shield_escapes(body: str) -> ShieldedBody:
    """Replace each escape in `body` with a unique equal-length identity run.

    Args:
        body: Raw literal body (LF-normalized, quotes stripped).

    Returns:
        `ShieldedBody` with placeholder text and ordered identity pairs
        (empty map when the body contains no escapes).

    Raises:
        EscapeScanError: If the body contains a line continuation, or if the
            private-use identity range is exhausted.
    """
    spans = list(_iter_escape_spans(body))
    if not spans:
        return ShieldedBody(text=body, escape_map=())
    used = {ord(ch) for ch in body if _SENTINEL_MIN <= ord(ch) <= _SENTINEL_MAX}
    candidates = (cp for cp in range(_SENTINEL_MIN, _SENTINEL_MAX + 1) if cp not in used)
    parts: list[str] = []
    pairs: list[tuple[str, str]] = []
    cursor = 0
    for start, end in spans:
        code_point = next(candidates, None)
        if code_point is None:
            msg = 'no identity character available for escape shielding'
            raise EscapeScanError('sentinel-unavailable', msg)
        identity = chr(code_point)
        parts.append(body[cursor:start])
        parts.append(identity * (end - start))
        pairs.append((identity, body[start:end]))
        cursor = end
    parts.append(body[cursor:])
    return ShieldedBody(text=''.join(parts), escape_map=tuple(pairs))
