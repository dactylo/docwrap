"""Parse POSIX shell command text into a structural tree without executing it."""

# pyright: reportPrivateUsage=false

import re
from dataclasses import dataclass
from string import ascii_letters, digits

_MAX_SUBSTITUTION_DEPTH = 8
NAME_CHARACTERS = frozenset(ascii_letters + digits + '_')
NAME_START = frozenset(ascii_letters + '_')
_OPERATORS = (
    '&>>',
    '<<-',
    '<<<',
    '&&',
    '&>',
    '||',
    ';;',
    '<<',
    '<&',
    '>>',
    '>&',
    '|&',
    ';',
    '|',
    '&',
    '<',
    '>',
    '(',
    ')',
)
_RECOVERY_ANSI_C = 'unsupported ANSI-C quoting'
_RECOVERY_BACKTICK = 'unterminated backtick'
_RECOVERY_EXPANSION = 'unterminated expansion'
_RECOVERY_HEREDOC = 'unterminated heredoc'
_RECOVERY_NESTING = 'substitution nesting limit'
_RECOVERY_QUOTES = 'unbalanced quotes'
_RECOVERY_SUBSTITUTION = 'unterminated substitution'
_SPECIAL_PARAMETERS = frozenset('@*#?-$!0123456789')
_TERMINATORS = frozenset({'&&', '||', ';', ';;', '|', '|&', '&', '\n'})
_WORD_BREAKS = frozenset(' \t\n')

type WordPart = LiteralPart | SubstitutionPart | ExpansionPart | GlobPart
type _Element = Word | Operator | Heredoc | _Comment


@dataclass(frozen=True, slots=True)
class LiteralPart:
    """Literal text of a word and whether quoting or an escape produced it."""

    text: str
    quoted: bool


@dataclass(frozen=True, slots=True)
class SubstitutionPart:
    """Command substitution with its parsed body and surrounding quote context.

    Attributes:
        command: Parsed nested command. An empty tree does not establish an
            empty substitution when `truncated` is true.
        raw: Source spelling of the substitution.
        in_double_quotes: Whether the substitution occurs inside double quotes.
        truncated: Whether the parser omitted the nested tree because its depth
            limit was reached. Consumers must treat its contents as unknown.
    """

    command: 'Command'
    raw: str
    in_double_quotes: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ExpansionPart:
    """Parameter, tilde, or brace expansion with any substitutions nested inside it."""

    raw: str
    children: tuple[SubstitutionPart, ...] = ()


@dataclass(frozen=True, slots=True)
class GlobPart:
    """Unquoted glob character."""

    raw: str


@dataclass(frozen=True, slots=True)
class Word:
    """Shell word made of parts, keeping its raw text."""

    raw: str
    parts: tuple[WordPart, ...]

    @property
    def literal_text(self) -> str | None:
        """Return the argv text of a word made only of literal parts, else `None`."""
        texts: list[str] = []
        for part in self.parts:
            if not isinstance(part, LiteralPart):
                return None
            texts.append(part.text)
        return ''.join(texts)


@dataclass(frozen=True, slots=True)
class Operator:
    """Operator element that does not terminate its segment."""

    text: str


@dataclass(frozen=True, slots=True)
class Heredoc:
    """Heredoc body and expansions, with recovery scan ownership.

    Attributes:
        body_inlined: Whether detector recovery parsed the body as following
            segments. Its token text and expansions may be shared with an
            earlier recovered heredoc, in which case these fields are empty.
    """

    delimiter: str
    quoted: bool
    body: str
    expansions: tuple[SubstitutionPart, ...] = ()
    body_inlined: bool = False


class _CommandState:
    """Mutable state of one command level while its segments are parsed."""

    __slots__ = ('_elements', '_finished', '_nested', '_paren_depth', '_pending', '_terminators')

    def __init__(self, *, nested: bool) -> None:
        """Initialize the segments and pending heredocs for one command level."""
        self._nested = nested
        self._elements: list[list[_Element]] = []
        self._terminators: list[str | None] = []
        self._pending: list[_PendingHeredoc] = []
        self._paren_depth = 0
        self._finished = False


@dataclass(frozen=True, slots=True)
class _Comment:
    """Comment element from `#` to the end of its line."""

    text: str


@dataclass(frozen=True, slots=True)
class Segment:
    """Elements of one simple command and the operator that terminated it."""

    elements: tuple[_Element, ...]
    terminator: str | None


@dataclass(frozen=True, slots=True)
class Command:
    """Parsed command tree with the recovery reasons a total parse recorded."""

    segments: tuple[Segment, ...]
    recovery: tuple[str, ...] = ()


@dataclass(slots=True)
class _PendingHeredoc:
    """Heredoc awaiting its body at the next line feed."""

    _elements: list[_Element]
    _index: int
    _delimiter: str
    _quoted: bool
    _strip_tabs: bool


class _WordState:
    """Mutable state while one word's parts are collected."""

    __slots__ = ('_parts', '_run', '_start')

    def __init__(self, start: int) -> None:
        """Begin collecting one word at its raw-text offset."""
        self._start = start
        self._parts: list[WordPart] = []
        self._run: list[str] = []


class _Parser:
    """Recursive-descent parser over one command text."""

    def __init__(self, text: str, depth: int = 0, *, recover_heredocs: bool = False) -> None:
        """Bind the input and inherited substitution depth."""
        self._text = text
        self._pos = 0
        self._depth = depth
        self._recovery: list[str] = []
        self._recover_heredocs = recover_heredocs
        self._heredoc_ends: dict[tuple[bool, str], int] | None = None
        self._recovered_body = False
        self._recovered_expansions = False
        self._recovered_tabs = False

    def _at_end(self) -> bool:
        """Return whether the cursor has consumed the input."""
        return self._pos >= len(self._text)

    def _flush(self, state: _WordState, *, quoted: bool) -> None:
        """Append a pending literal run with its quoting provenance."""
        if state._run:
            state._parts.append(LiteralPart(''.join(state._run), quoted))
            state._run.clear()

    def _has_heredoc_end(self, pending: _PendingHeredoc) -> bool:
        """Check remaining delimiter lines using one index for this script."""
        if self._heredoc_ends is None:
            self._heredoc_ends = {}
            offset = 0
            for line in self._text.split('\n'):
                if offset == len(self._text):
                    break
                self._heredoc_ends[False, line] = offset
                self._heredoc_ends[True, line.lstrip('\t')] = offset
                offset += len(line) + 1
        return self._heredoc_ends.get((pending._strip_tabs, pending._delimiter), -1) >= self._pos

    def _match_operator(self) -> str | None:
        """Find the longest operator at the current cursor."""
        for operator in _OPERATORS:
            if self._text.startswith(operator, self._pos):
                return operator
        return None

    def _nested_substitution(
        self,
        inner: str,
        raw: str,
        *,
        in_double_quotes: bool = False,
    ) -> SubstitutionPart:
        """Parse detached substitution text at the next depth or record truncation."""
        if self._depth >= _MAX_SUBSTITUTION_DEPTH:
            self._recovery.append(_RECOVERY_NESTING)
            return SubstitutionPart(Command(()), raw, in_double_quotes, truncated=True)
        parser = _Parser(inner, self._depth + 1, recover_heredocs=self._recover_heredocs)
        command = parser._parse()
        self._recovery.extend(parser._recovery)
        return SubstitutionPart(command, raw, in_double_quotes)

    def _parse_ansi_c_quoted(self) -> LiteralPart | ExpansionPart:
        """Recover ANSI-C quote boundaries, retaining only unescaped literal contents."""
        start = self._pos
        self._pos += 2
        content_start = self._pos
        escaped = False
        self._recovery.append(_RECOVERY_ANSI_C)
        while not self._at_end():
            char = self._text[self._pos]
            if char == "'":
                break
            if char == '\\':
                escaped = True
                self._pos = min(self._pos + 2, len(self._text))
            else:
                self._pos += 1
        content = self._text[content_start : self._pos]
        if self._at_end():
            self._recovery.append(_RECOVERY_QUOTES)
        else:
            self._pos += 1
        if escaped:
            return ExpansionPart(self._text[start : self._pos])
        return LiteralPart(content, quoted=True)

    def _parse_backtick(self, *, in_double_quotes: bool) -> SubstitutionPart:
        """Parse a backtick region with its inherited quote context."""
        start = self._pos
        close = _scan_backtick(self._text, start + 1)
        if close < 0:
            self._recovery.append(_RECOVERY_BACKTICK)
            close = len(self._text)
            self._pos = close
        else:
            self._pos = close + 1
        # Remove one backtick escape layer before parsing the nested shell text.
        escapes = r'\\([$`\\"])|\\\n' if in_double_quotes else r'\\([$`\\])|\\\n'
        inner = re.sub(escapes, r'\1', self._text[start + 1 : close])
        return self._nested_substitution(
            inner,
            self._text[start : self._pos],
            in_double_quotes=in_double_quotes,
        )

    def _parse_brace_expansion(self, *, in_double_quotes: bool) -> ExpansionPart:
        """Retain parameter-expansion text and every nested substitution."""
        start = self._pos
        self._pos += 2
        children: list[SubstitutionPart] = []
        depth = 1
        while not self._at_end():
            char = self._text[self._pos]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    self._pos += 1
                    return ExpansionPart(self._text[start : self._pos], tuple(children))
            elif char == '`' or self._text.startswith('$(', self._pos):
                part = self._parse_dynamic(in_double_quotes=in_double_quotes)
                if isinstance(part, SubstitutionPart):
                    children.append(part)
                continue
            self._pos += 1
        self._recovery.append(_RECOVERY_EXPANSION)
        return ExpansionPart(self._text[start : self._pos], tuple(children))

    def _parse_double_quoted(self, state: _WordState) -> None:
        """Read double-quoted parts with shell escape and substitution rules."""
        self._flush(state, quoted=False)
        self._pos += 1
        run: list[str] = []
        emitted = False
        while True:
            if self._at_end():
                self._recovery.append(_RECOVERY_QUOTES)
                break
            char = self._text[self._pos]
            if char == '"':
                self._pos += 1
                break
            if char == '\\':
                if self._text.startswith('\\\n', self._pos):
                    self._pos += 2
                    continue
                following = self._text[self._pos + 1 : self._pos + 2]
                if following in ('$', '`', '"', '\\'):
                    run.append(following)
                    self._pos += 2
                    continue
                run.append('\\')
                self._pos += 1
                continue
            if char in '$`':
                if run:
                    state._parts.append(LiteralPart(''.join(run), quoted=True))
                    run.clear()
                state._parts.append(self._parse_dynamic(in_double_quotes=True))
                emitted = True
                continue
            run.append(char)
            self._pos += 1
        if run or not emitted:
            state._parts.append(LiteralPart(''.join(run), quoted=True))

    def _parse_dynamic(self, *, in_double_quotes: bool) -> WordPart:
        """Read the expansion or substitution beginning at the cursor."""
        if self._text[self._pos] == '`':
            return self._parse_backtick(in_double_quotes=in_double_quotes)
        following = self._text[self._pos + 1 : self._pos + 2]
        if following == '(':
            return self._parse_substitution(in_double_quotes=in_double_quotes)
        if following == '{':
            return self._parse_brace_expansion(in_double_quotes=in_double_quotes)
        if following in _SPECIAL_PARAMETERS:
            self._pos += 2
            return ExpansionPart(f'${following}')
        if following in NAME_START:
            end = self._pos + 1
            while end < len(self._text) and self._text[end] in NAME_CHARACTERS:
                end += 1
            raw = self._text[self._pos : end]
            self._pos = end
            return ExpansionPart(raw)
        self._pos += 1
        return LiteralPart('$', quoted=False)

    def _parse_escape(self, state: _WordState) -> bool:
        """Consume an escape and return whether it quoted a character."""
        if self._text.startswith('\\\n', self._pos):
            self._pos += 2
            return False
        self._flush(state, quoted=False)
        if self._pos + 1 < len(self._text):
            state._parts.append(LiteralPart(self._text[self._pos + 1], quoted=True))
            self._pos += 2
            return True
        state._parts.append(LiteralPart('\\', quoted=False))
        self._pos += 1
        return False

    def _parse_single_quoted(self, state: _WordState) -> None:
        """Read literal single-quoted text, recovering an absent closing quote."""
        self._flush(state, quoted=False)
        self._pos += 1
        end = self._text.find("'", self._pos)
        if end < 0:
            self._recovery.append(_RECOVERY_QUOTES)
            text = self._text[self._pos :]
            self._pos = len(self._text)
        else:
            text = self._text[self._pos : end]
            self._pos = end + 1
        state._parts.append(LiteralPart(text, quoted=True))

    def _parse_substitution(self, *, in_double_quotes: bool) -> SubstitutionPart:
        """Parse a nested command until its structural closing parenthesis."""
        start = self._pos
        self._pos += 2
        if self._depth >= _MAX_SUBSTITUTION_DEPTH:
            self._pos = _scan_balanced(self._text, start + 1, '(', ')')
            self._recovery.append(_RECOVERY_NESTING)
            raw = self._text[start : self._pos]
            return SubstitutionPart(Command(()), raw, in_double_quotes, truncated=True)
        recovery_start = len(self._recovery)
        self._depth += 1
        segments = self._parse_segments(nested=True)
        self._depth -= 1
        child_recovery = tuple(self._recovery[recovery_start:])
        if self._at_end():
            self._recovery.append(_RECOVERY_SUBSTITUTION)
        else:
            self._pos += 1
        raw = self._text[start : self._pos]
        return SubstitutionPart(
            Command(tuple(segments), child_recovery),
            raw,
            in_double_quotes,
        )

    def _parse_word(self) -> tuple[Word, bool]:
        """Collect word parts without discarding quote and expansion boundaries."""
        state = _WordState(self._pos)
        quoted = False
        while not self._at_end():
            char = self._text[self._pos]
            if char in _WORD_BREAKS or self._match_operator() is not None:
                break
            if char == "'":
                quoted = True
                self._parse_single_quoted(state)
            elif char == '"':
                quoted = True
                self._parse_double_quoted(state)
            elif char == '\\':
                quoted = self._parse_escape(state) or quoted
            elif self._text.startswith("$'", self._pos):
                quoted = True
                self._flush(state, quoted=False)
                state._parts.append(self._parse_ansi_c_quoted())
            elif char in '$`':
                self._flush(state, quoted=False)
                state._parts.append(self._parse_dynamic(in_double_quotes=False))
            elif char in '*?[':
                self._flush(state, quoted=False)
                state._parts.append(GlobPart(char))
                self._pos += 1
            elif char in '{}' or (char == '~' and self._pos == state._start):
                self._flush(state, quoted=False)
                state._parts.append(ExpansionPart(char))
                self._pos += 1
            else:
                state._run.append(char)
                self._pos += 1
        self._flush(state, quoted=False)
        return Word(self._text[state._start : self._pos], tuple(state._parts)), quoted

    def _read_heredoc_body(self, pending: _PendingHeredoc) -> str:
        """Consume one heredoc body through its matching delimiter line."""
        lines: list[str] = []
        while not self._at_end():
            end = self._text.find('\n', self._pos)
            line = self._text[self._pos :] if end < 0 else self._text[self._pos : end]
            self._pos = len(self._text) if end < 0 else end + 1
            candidate = line.lstrip('\t') if pending._strip_tabs else line
            if candidate == pending._delimiter:
                return ''.join(f'{text}\n' for text in lines)
            lines.append(candidate)
        self._recovery.append(_RECOVERY_HEREDOC)
        return ''.join(f'{text}\n' for text in lines)

    def _read_to_line_end(self) -> str:
        """Consume comment text while leaving its line feed for segment parsing."""
        end = self._text.find('\n', self._pos)
        end = len(self._text) if end < 0 else end
        text = self._text[self._pos : end]
        self._pos = end
        return text

    def _scan_heredoc_expansions(self, body: str) -> tuple[SubstitutionPart, ...]:
        """Scan body expansions with structural parsing at each substitution opener."""
        scanner = _Parser(body, self._depth, recover_heredocs=self._recover_heredocs)
        found: list[SubstitutionPart] = []
        while not scanner._at_end():
            char = scanner._text[scanner._pos]
            if char == '\\':
                following = scanner._text[scanner._pos + 1 : scanner._pos + 2]
                scanner._pos += 2 if following in ('$', '`', '\\', '\n') else 1
            elif scanner._text.startswith('$(', scanner._pos) or char == '`':
                part = scanner._parse_dynamic(in_double_quotes=False)
                if isinstance(part, SubstitutionPart):
                    found.append(part)
            else:
                scanner._pos += 1
        self._recovery.extend(scanner._recovery)
        return tuple(found)

    def _recover_heredoc(self, pending: _PendingHeredoc) -> Heredoc:
        """Leave an unmatched body's script in place and share its tail scans."""
        self._recovery.append(_RECOVERY_HEREDOC)
        if pending._strip_tabs and not self._recovered_tabs:
            # Preserve <<- semantics even within continued words and substitutions.
            tail = '\n'.join(line.lstrip('\t') for line in self._text[self._pos :].split('\n'))
            self._text = self._text[: self._pos] + tail
            self._heredoc_ends = None
            self._recovered_expansions = False
            self._recovered_tabs = True
        body = ''
        expansions: tuple[SubstitutionPart, ...] = ()
        if not self._recovered_body:
            body = self._text[self._pos :]
            self._recovered_body = True
        if not pending._quoted and not self._recovered_expansions:
            expansions = self._scan_heredoc_expansions(self._text[self._pos :])
            self._recovered_expansions = True
        return Heredoc(pending._delimiter, pending._quoted, body, expansions, body_inlined=True)

    def _drain_heredocs(self, state: _CommandState) -> None:
        """Attach pending bodies and their expansion trees to their segments."""
        for pending in state._pending:
            if self._recover_heredocs and not self._has_heredoc_end(pending):
                pending._elements[pending._index] = self._recover_heredoc(pending)
                continue
            body = self._read_heredoc_body(pending)
            expansions = () if pending._quoted else self._scan_heredoc_expansions(body)
            pending._elements[pending._index] = Heredoc(
                pending._delimiter,
                pending._quoted,
                body,
                expansions,
            )
        state._pending.clear()

    def _parse_segments(self, *, nested: bool) -> list[Segment]:
        """Collect segments until this command level ends."""
        state = _CommandState(nested=nested)
        while not state._finished:
            self._parse_segment(state)
        if state._pending:
            self._drain_heredocs(state)
        return [
            Segment(tuple(elements), terminator)
            for elements, terminator in zip(
                state._elements,
                state._terminators,
                strict=True,
            )
            if elements or terminator not in (None, '\n')
        ]

    def _parse(self) -> Command:
        """Parse the complete text as a top-level command."""
        segments = self._parse_segments(nested=False)
        return Command(tuple(segments), tuple(self._recovery))

    def _skip_blanks(self) -> None:
        """Skip unquoted spaces, tabs, and continued line feeds."""
        while not self._at_end():
            if self._text[self._pos] in ' \t':
                self._pos += 1
            elif self._text.startswith('\\\n', self._pos):
                self._pos += 2
            else:
                return

    def _parse_operator(
        self,
        operator: str,
        elements: list[_Element],
        state: _CommandState,
    ) -> None:
        """Record an operator and collect any heredoc delimiter it introduces."""
        if operator == '(':
            state._paren_depth += 1
        elif operator == ')':
            state._paren_depth -= 1
        elements.append(Operator(operator))
        if operator in ('<<', '<<-'):
            self._skip_blanks()
            word, quoted = self._parse_word()
            delimiter = _quote_removed_text(word)
            state._pending.append(
                _PendingHeredoc(
                    elements,
                    len(elements),
                    delimiter,
                    quoted,
                    _strip_tabs=operator == '<<-',
                )
            )
            elements.append(Heredoc(delimiter, quoted, ''))

    def _parse_segment(self, state: _CommandState) -> None:
        """Read one segment while preserving words, operators, and heredocs."""
        elements: list[_Element] = []
        terminator: str | None = None
        while True:
            self._skip_blanks()
            if self._at_end():
                state._finished = True
                break
            char = self._text[self._pos]
            if char == '\n':
                self._pos += 1
                self._drain_heredocs(state)
                terminator = '\n'
                break
            if char == '#':
                elements.append(_Comment(self._read_to_line_end()))
                continue
            operator = self._match_operator()
            if operator is None:
                word, _ = self._parse_word()
                elements.append(word)
                continue
            if operator == ')' and state._nested and state._paren_depth == 0:
                state._finished = True
                break
            self._pos += len(operator)
            if operator in _TERMINATORS:
                terminator = operator
                break
            self._parse_operator(operator, elements, state)
        state._elements.append(elements)
        state._terminators.append(terminator)


def _quote_removed_text(word: Word) -> str:
    """Return a word's text after quote removal without performing expansions."""
    return ''.join(part.text if isinstance(part, LiteralPart) else part.raw for part in word.parts)


def _scan_backtick(text: str, start: int) -> int:
    """Return the index of the next unescaped backtick at or after `start`, or -1."""
    index = start
    while index < len(text):
        if text[index] == '\\':
            index += 2
            continue
        if text[index] == '`':
            return index
        index += 1
    return -1


def _scan_balanced(text: str, start: int, opener: str, closer: str) -> int:
    """Return the index after the closer matching `opener` at `start`, ignoring quotes."""
    depth = 0
    index = start
    while index < len(text):
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
            continue
        if text.startswith(closer, index):
            depth -= 1
            index += len(closer)
            if depth == 0:
                return index
            continue
        index += 1
    return len(text)


def parse_command(text: str, *, recover_heredocs: bool = False) -> Command:
    """Parse shell command text into a total command tree.

    Args:
        text: Raw shell command text as the host delivered it.
        recover_heredocs: For conservative detector script scans only, parse
            unmatched heredoc bodies as following segments and share their
            token and expansion tails instead of reparsing nested suffixes.

    Returns:
        The command tree. A parse that needed recovery lists its reasons in
        `Command.recovery`; the tree is never absent.
    """
    return _Parser(text, recover_heredocs=recover_heredocs)._parse()
