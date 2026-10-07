"""Detect commit invocations in commands the grammar rejected."""

from collections.abc import Iterator

from ._shellparse import (
    Command,
    ExpansionPart,
    Heredoc,
    Operator,
    Segment,
    SubstitutionPart,
    Word,
    parse_command,
)

_EVALUATOR = 'eval'
_GIT_WORDS = frozenset({'git', 'git.exe'})
_INTERPRETER_PREFIX = 'python'
_INTERPRETERS = frozenset({'node', 'perl', 'ruby'})
_SCRIPT_OPTIONS = frozenset({'-c', '-e'})
_SHELLS = frozenset({'bash', 'dash', 'ksh', 'sh', 'zsh'})
_TOKEN_BREAKS = str.maketrans(dict.fromkeys('\'"[](){},;=', ' '))


def _basename(text: str) -> str:
    """Return the lower-cased final path component of a literal word."""
    return text.replace('\\', '/').rsplit('/', 1)[-1].lower()


def _is_data_sink(segment: Segment) -> bool:
    """Identify a plain cat or tee segment that writes its input to a file."""
    elements = segment.elements
    if (
        not elements
        or not isinstance(elements[0], Word)
        or elements[0].literal_text not in {'cat', 'tee'}
        or segment.terminator in {'|', '|&'}
    ):
        return False

    writes_file = False
    for index, element in enumerate(elements):
        previous = elements[index - 1] if index else None
        if isinstance(element, Operator):
            if element.text not in {'<<', '<<-', '>', '>>'}:
                return False
            if element.text in {'>', '>>'}:
                if index + 1 >= len(elements) or not isinstance(elements[index + 1], Word):
                    return False
                writes_file = True
        elif (
            elements[0].literal_text == 'tee'
            and index > 0
            and isinstance(element, Word)
            and not (isinstance(previous, Operator) and previous.text in {'>', '>>'})
            and element.literal_text is not None
            and not element.literal_text.startswith('-')
        ):
            writes_file = True
    return writes_file


def _is_interpreter(text: str) -> bool:
    """Return whether a literal word names a listed script interpreter."""
    name = _basename(text).removesuffix('.exe')
    return name in _INTERPRETERS or name.startswith(_INTERPRETER_PREFIX)


def _is_script_option(text: str) -> bool:
    """Return whether a word is a short-option cluster containing `c`."""
    return text.startswith('-') and not text.startswith('--') and 'c' in text[1:]


def _evaluator_at(texts: list[str | None], start: int) -> Iterator[str]:
    """Yield scripts selected by the evaluator rule at one literal word."""
    first = texts[start]
    if first is None:
        return
    name = _basename(first)
    if name == _EVALUATOR:
        yield ' '.join(text for text in texts[start + 1 :] if text is not None)
        return
    if name.removesuffix('.exe') not in _SHELLS:
        return
    for index, text in enumerate(texts[start + 1 :], start + 1):
        if text is not None and _is_script_option(text):
            yield from (later for later in texts[index + 1 :] if later is not None)
            return


def _word_substitutions(word: Word) -> Iterator[SubstitutionPart]:
    """Yield every substitution nested anywhere inside a word."""
    for part in word.parts:
        if isinstance(part, SubstitutionPart):
            yield part
        elif isinstance(part, ExpansionPart):
            yield from part.children


def is_git_executable(text: str) -> bool:
    r"""Return whether a literal word names Git by its executable basename.

    Args:
        text: Literal text of one shell word, possibly path-qualified with
            POSIX or Windows separators.

    Returns:
        `True` for `git` or `git.exe` in any letter case after the last path
        separator, so `/usr/bin/git`, `./git`, and `C:\Git\cmd\GIT.EXE` all
        match. A word carrying an expansion has no literal text and is never
        passed here.
    """
    return _basename(text) in _GIT_WORDS


def _tokens_commit(text: str) -> bool:
    """Return whether quote-stripped tokens hold a git word followed by commit."""
    tokens = text.translate(_TOKEN_BREAKS).split()
    return any(
        is_git_executable(token) and 'commit' in tokens[index + 1 :]
        for index, token in enumerate(tokens)
    )


def _interpreter_commits(texts: list[str | None]) -> bool:
    """Scan the literal script after a listed interpreter's -c or -e option."""
    for index, text in enumerate(texts):
        if text is None or not _is_interpreter(text):
            continue
        for option_index in range(index + 1, len(texts)):
            if texts[option_index] in _SCRIPT_OPTIONS:
                script = texts[option_index + 1] if option_index + 1 < len(texts) else None
                return script is not None and _tokens_commit(script)
    return False


def _segment_commits(
    segment: Segment, pending: list[Segment | SubstitutionPart | str], *, sink: bool
) -> bool:
    """Apply direct detector rules and queue nested work for one segment."""
    words = [element for element in segment.elements if isinstance(element, Word)]
    texts = [word.literal_text for word in words]
    for index, text in enumerate(texts):
        if (text is None or is_git_executable(text)) and 'commit' in texts[index + 1 :]:
            return True
    nested: list[SubstitutionPart | str] = []
    for element in segment.elements:
        if isinstance(element, Word):
            nested.extend(_word_substitutions(element))
        elif isinstance(element, Heredoc) and not (element.quoted and sink):
            if _tokens_commit(element.body):
                return True
            nested.extend(element.expansions)
            if not element.body_inlined:
                nested.append(element.body)
    if _interpreter_commits(texts):
        return True
    for start in range(len(texts)):
        nested.extend(_evaluator_at(texts, start))
    pending.extend(reversed(nested))
    return False


def detect_commit(command: Command) -> bool:
    """Return whether a command tree matches conservative commit-detection rules.

    A quoted heredoc written by plain `cat` or `tee` is exempt only in a
    one-segment command.

    Args:
        command: Parsed command tree, clean or recovered.

    Returns:
        `True` when a segment, nested substitution, heredoc body, or evaluator
        script matches a possible commit. Unknown executable words, truncated
        substitutions, and commit-looking script text can also match; a
        positive result does not establish that Git will execute.
    """
    sink = (
        command.segments[0]
        if len(command.segments) == 1 and _is_data_sink(command.segments[0])
        else None
    )
    pending: list[Segment | SubstitutionPart | str] = list(reversed(command.segments))
    # Any positive scan ends the invocation, so visited scripts need no cached result.
    scripts: set[str] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if item not in scripts:
                scripts.add(item)
                parsed = parse_command(item, recover_heredocs=True)
                pending.extend(reversed(parsed.segments))
        elif isinstance(item, SubstitutionPart):
            if item.truncated:
                return True
            pending.extend(reversed(item.command.segments))
        # Only the original segment can receive the quoted-heredoc exemption.
        elif _segment_commits(item, pending, sink=item is sink):
            return True
    return False
