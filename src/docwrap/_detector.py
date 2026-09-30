"""Detect commit invocations in commands the grammar rejected."""

from pathlib import PurePosixPath

from docwrap._shellparse import (
    Command,
    ExpansionPart,
    Heredoc,
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


def _detect_commit(command: Command, scripts: dict[str, bool]) -> bool:
    """Scan a tree with the script results shared by this detector invocation."""
    return any(_segment_commits(segment, scripts) for segment in command.segments)


def _evaluator_commits(texts: list[str | None], scripts: dict[str, bool]) -> bool:
    """Scan listed evaluator words after prefixes without accepting those prefixes."""
    return any(_evaluator_at(texts, start, scripts) for start in range(len(texts)))


def _is_interpreter(text: str) -> bool:
    """Return whether a literal word names a listed script interpreter."""
    name = _basename(text)
    return name in _INTERPRETERS or name.startswith(_INTERPRETER_PREFIX)


def _is_script_option(text: str) -> bool:
    """Return whether a word is a short-option cluster containing `c`."""
    return text.startswith('-') and not text.startswith('--') and 'c' in text[1:]


def _evaluator_at(texts: list[str | None], start: int, scripts: dict[str, bool]) -> bool:
    """Apply the evaluator rule beginning at one literal word."""
    first = texts[start]
    if first is None:
        return False
    name = PurePosixPath(first).name
    if name == _EVALUATOR:
        joined = ' '.join(text for text in texts[start + 1 :] if text is not None)
        return _script_commits(joined, scripts)
    if name not in _SHELLS:
        return False
    for index, text in enumerate(texts[start + 1 :], start + 1):
        if text is not None and _is_script_option(text):
            return any(
                _script_commits(later, scripts)
                for later in texts[index + 1 :]
                if later is not None
            )
    return False


def _script_commits(text: str, scripts: dict[str, bool]) -> bool:
    """Parse and scan each distinct script at most once per detector invocation."""
    if text not in scripts:
        scripts[text] = _detect_commit(parse_command(text), scripts)
    return scripts[text]


def _substitution_commits(part: SubstitutionPart, scripts: dict[str, bool]) -> bool:
    """Classify a substitution, treating a truncated one as a commit."""
    return part.truncated or _detect_commit(part.command, scripts)


def _word_commits(word: Word, scripts: dict[str, bool]) -> bool:
    """Classify every substitution nested anywhere inside a word."""
    for part in word.parts:
        if isinstance(part, SubstitutionPart) and _substitution_commits(part, scripts):
            return True
        if isinstance(part, ExpansionPart) and any(
            _substitution_commits(child, scripts) for child in part.children
        ):
            return True
    return False


def detect_commit(command: Command) -> bool:
    """Return whether a command tree matches conservative commit-detection rules.

    Args:
        command: Parsed command tree, clean or recovered.

    Returns:
        `True` when a segment, nested substitution, heredoc body, or evaluator
        script matches a possible commit. Unknown executable words, truncated
        substitutions, and commit-looking script text can also match; a
        positive result does not establish that Git will execute.
    """
    return _detect_commit(command, {})


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


def _heredoc_commits(heredoc: Heredoc, scripts: dict[str, bool]) -> bool:
    """Apply the expansion scan, the script scan, and the token scan to one heredoc body."""
    if any(_substitution_commits(part, scripts) for part in heredoc.expansions):
        return True
    return _script_commits(heredoc.body, scripts) or _tokens_commit(heredoc.body)


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


def _segment_commits(segment: Segment, scripts: dict[str, bool]) -> bool:
    """Apply every detector rule to one segment."""
    words = [element for element in segment.elements if isinstance(element, Word)]
    texts = [word.literal_text for word in words]
    for index, text in enumerate(texts):
        if (text is None or is_git_executable(text)) and 'commit' in texts[index + 1 :]:
            return True
    for element in segment.elements:
        if isinstance(element, Word) and _word_commits(element, scripts):
            return True
        if isinstance(element, Heredoc) and _heredoc_commits(element, scripts):
            return True
    return bool(texts) and (_evaluator_commits(texts, scripts) or _interpreter_commits(texts))
