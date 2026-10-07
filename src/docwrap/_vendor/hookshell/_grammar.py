"""Accept the closed commit command grammar over a parsed shell tree."""

# pyright: reportPrivateUsage=false

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from ._detector import is_git_executable
from ._shell import is_msys_mount_path, normalize_msys_path, resolve_route
from ._shellparse import (
    NAME_CHARACTERS,
    NAME_START,
    Command,
    ExpansionPart,
    GlobPart,
    Heredoc,
    LiteralPart,
    Operator,
    Segment,
    SubstitutionPart,
    Word,
    WordPart,
)

ADD_SELECTORS = frozenset({'-A', '-u', '--all', '--update'})
_AMEND_DATE_COMMAND = ('git', 'log', '-1', '--format=%aI')
_ASSIGNMENT_NAME = 'GIT_COMMITTER_DATE'
_CD_SEGMENT_SIZE = 2
_COMMIT_BOOL_LONG = frozenset(
    {
        '--ahead-behind',
        '--allow-empty',
        '--allow-empty-message',
        '--amend',
        '--branch',
        '--edit',
        '--gpg-sign',
        '--no-ahead-behind',
        '--no-allow-empty',
        '--no-allow-empty-message',
        '--no-branch',
        '--no-edit',
        '--no-gpg-sign',
        '--no-post-rewrite',
        '--no-quiet',
        '--no-signoff',
        '--no-status',
        '--no-untracked-files',
        '--no-verbose',
        '--no-verify',
        '--quiet',
        '--reset-author',
        '--signoff',
        '--status',
        '--untracked-files',
        '--verbose',
        '--verify',
    }
)
_COMMIT_BOOL_LONG_JOINED = frozenset({'--gpg-sign', '--untracked-files'})
_COMMIT_BOOL_SHORT = frozenset('enqsv')
_COMMIT_HELP = frozenset({'-h', '--help'})
_COMMIT_INTERACTIVE_LONG = {
    '--interactive': True,
    '--no-interactive': False,
    '--no-patch': False,
    '--patch': True,
}
_COMMIT_OPTIONAL_VALUE_SHORT = frozenset('Su')
_COMMIT_SELECTION_LONG = {
    '--all': ('_auto_stage', True),
    '--include': ('_include', True),
    '--no-all': ('_auto_stage', False),
    '--no-include': ('_include', False),
    '--no-only': ('_only', False),
    '--only': ('_only', True),
}
_COMMIT_SELECTION_SHORT = {
    'a': ('_auto_stage', True),
    'i': ('_include', True),
    'o': ('_only', True),
}
_COMMIT_STATUS_LONG = frozenset({'--dry-run', '--long', '--null', '--porcelain', '--short'})
_COMMIT_VALUE_LONG = frozenset(
    {
        '--author',
        '--cleanup',
        '--date',
        '--file',
        '--fixup',
        '--inter-hunk-context',
        '--message',
        '--pathspec-from-file',
        '--reedit-message',
        '--reuse-message',
        '--squash',
        '--template',
        '--trailer',
        '--unified',
    }
)
_COMMIT_VALUE_SHORT = frozenset('CFcmt')
_ELEMENT_RELATIVE_ROUTE = 'relative route without a known working directory'
_ELEMENT_UNKNOWN_CWD = 'unknown working directory'
_FRAGMENT_LIMIT = 40
_GIT_ROUTE_INDEX = 2
_GIT_SUBCOMMANDS = frozenset({'add', 'commit', 'diff', 'log', 'status'})
_HEREDOC_ELEMENT_COUNT = 3
_HINT_ADD = (
    'supported add: git add [--] PATHSPEC... or git add (-A|--all|-u|--update) [--] [PATHSPEC...]'
)
_HINT_COMMIT = 'supported commit options are listed in docs/hook.md'
_HINT_EXECUTABLE = 'use the bare executable name git; PATH resolves the installed git'
_HINT_PRECHECK = (
    'run the check or observation in a separate tool call, then run the add/commit command '
    'only after success; for a check after staging, run add plus check without a real commit, '
    'then submit the commit separately'
)
_HINT_READ_ONLY = 'read-only segments accept only the documented git status, diff, and log options'
_HINT_ROUTE = (
    'supported form: git -C /absolute/repo commit ... or cd /absolute/repo && git commit ...'
)
_HINT_SEQUENCE = 'supported form: [cd DIR &&] [git add ...;] git commit ...'
_LOG_COUNT_RE = re.compile(r'-[0-9]+')
_READ_ONLY_OPTIONS = {
    'diff': frozenset(
        {'--cached', '--check', '--name-only', '--name-status', '--staged', '--stat'}
    ),
    'log': frozenset({'--oneline', '--stat'}),
    'status': frozenset(
        {'--branch', '--porcelain', '--porcelain=v1', '--short', '--untracked-files=all'}
    ),
}
_SEQUENCE_TERMINATORS = frozenset({'&&', ';', '\n'})


@dataclass(frozen=True, slots=True)
class PreCommitAdd:
    """Parsed Git add command that precedes a commit.

    Attributes:
        tokens: Parsed arguments for the Git add command.
        exec_cwd: Directory in which the command segment executes.
        git_cwd: Git working directory after `-C` or `cd` routing.
    """

    tokens: tuple[str, ...]
    exec_cwd: str
    git_cwd: str


class CommitMode(NamedTuple):
    """Selection state of an accepted `git commit` command.

    Attributes:
        auto_stage: Whether the commit stages tracked changes automatically.
        include: Whether named paths are staged in addition to the index.
        only: Whether only named paths are committed.
        pathspecs: Inline pathspec arguments.
        pathspec_from_file: Path to a file containing pathspecs, or `None`.
        pathspec_file_nul: Whether pathspec-file records are NUL-delimited.
    """

    auto_stage: bool
    include: bool
    only: bool
    pathspecs: tuple[str, ...]
    pathspec_from_file: str | None
    pathspec_file_nul: bool


@dataclass(frozen=True, slots=True)
class CommitPlan:
    """Accepted command that will create a commit, resolved to its execution directory.

    Attributes:
        cwd: Established directory in which Git runs the commit. It is never a
            substituted session or process directory.
        mode: Selection state of the commit.
        pre_commit_adds: The supported `git add` replayed before inspection.
    """

    cwd: str
    mode: CommitMode
    pre_commit_adds: tuple[PreCommitAdd, ...]


@dataclass(frozen=True, slots=True)
class NoCommit:
    """Accepted command proven not to create a commit; it carries no directory."""


@dataclass(frozen=True, slots=True)
class Rejection:
    """First unsupported element of a command and the nearest accepted form.

    Attributes:
        element: Short phrase naming the unsupported element.
        hint: Nearest accepted form.
    """

    element: str
    hint: str


class _CommitState:
    """Mutable option state while a commit segment is parsed."""

    __slots__ = (
        '_auto_stage',
        '_heredoc_messages',
        '_include',
        '_interactive',
        '_interactive_option',
        '_only',
        '_options_ended',
        '_pathspec_file_nul',
        '_pathspec_from_file',
        '_pathspecs',
        '_status_only',
    )

    def __init__(self) -> None:
        """Initialize selection and metadata state before reading commit options."""
        self._auto_stage = False
        self._heredoc_messages = 0
        self._include = False
        self._interactive = False
        self._interactive_option = ''
        self._only = False
        self._options_ended = False
        self._pathspec_file_nul = False
        self._pathspec_from_file: str | None = None
        self._pathspecs: list[str] = []
        self._status_only = False


class _GitSegment(NamedTuple):
    """One accepted git segment with its route, argv, and parsed commit state."""

    subcommand: str
    route: str | None
    tokens: tuple[str, ...]
    text: str
    commit: _CommitState | None = None


def _amend_value_accepted(rest: str, remaining: Sequence[WordPart]) -> bool:
    """Return whether an assignment value is literal or the exact log substitution."""
    if all(isinstance(part, LiteralPart) for part in remaining):
        return True
    if rest or len(remaining) != 1 or not isinstance(remaining[0], SubstitutionPart):
        return False
    segments = remaining[0].command.segments
    if len(segments) != 1 or segments[0].terminator is not None:
        return False
    elements = segments[0].elements
    texts = tuple(element.literal_text for element in elements if isinstance(element, Word))
    return len(texts) == len(elements) and texts == _AMEND_DATE_COMMAND


def _assignment_name(word: Word) -> tuple[str, str] | None:
    """Return an unquoted assignment's name and literal remainder, or `None`."""
    first = word.parts[0] if word.parts else None
    if not isinstance(first, LiteralPart) or first.quoted:
        return None
    name, separator, rest = first.text.partition('=')
    valid = (
        bool(separator)
        and bool(name)
        and name[0] in NAME_START
        and all(character in NAME_CHARACTERS for character in name)
    )
    return (name, rest) if valid else None


def _check_shape(cd_route: str | None, parsed: Sequence[_GitSegment]) -> Rejection | None:
    """Check segment order, add count, and routing uniformity."""
    commit = parsed[-1]
    if commit.subcommand != 'commit':
        return Rejection(f"final segment '{commit.text}'", _HINT_SEQUENCE)
    seen_add = False
    for git in parsed[:-1]:
        if git.subcommand == 'commit' or (git.subcommand == 'add' and seen_add):
            return Rejection(f"segment '{git.text}'", _HINT_SEQUENCE)
        seen_add = seen_add or git.subcommand == 'add'
    if len({git.route for git in parsed}) > 1:
        return Rejection('-C route differs between segments', _HINT_SEQUENCE)
    if cd_route is not None and parsed[0].route is not None:
        return Rejection('cd combined with -C', _HINT_SEQUENCE)
    return None


def _commit_finish(state: _CommitState) -> _CommitState | Rejection:
    """Check the final selection state of a commit segment."""
    has_inline = bool(state._pathspecs)
    has_file = state._pathspec_from_file is not None
    has_pathspec = has_inline or has_file
    defects = (
        (state._interactive, f'option {state._interactive_option}'),
        (
            state._pathspec_file_nul and not has_file,
            '--pathspec-file-nul without --pathspec-from-file',
        ),
        (state._auto_stage and has_pathspec, 'pathspec with --all'),
        (has_inline and has_file, '--pathspec-from-file with pathspecs'),
        (state._include and state._only, '--include with --only'),
        (state._include and not has_pathspec, '--include without pathspec'),
        (state._only and not has_pathspec, '--only without pathspec'),
    )
    for defect, element in defects:
        if defect:
            return Rejection(element, _HINT_COMMIT)
    return state


def _commit_flag(name: str, state: _CommitState) -> bool:
    """Apply one valueless long option and return whether it is in the inventory."""
    selection = _COMMIT_SELECTION_LONG.get(name)
    interactive = _COMMIT_INTERACTIVE_LONG.get(name)
    if selection is not None:
        setattr(state, selection[0], selection[1])
    elif interactive is not None:
        state._interactive = interactive
        state._interactive_option = name
    elif name in _COMMIT_STATUS_LONG:
        state._status_only = True
    elif name == '--pathspec-file-nul':
        state._pathspec_file_nul = True
    else:
        return name in _COMMIT_BOOL_LONG
    return True


def _commit_short_flag(character: str, state: _CommitState) -> bool:
    """Apply one valueless short option and return whether it is in the inventory."""
    selection = _COMMIT_SELECTION_SHORT.get(character)
    if selection is not None:
        setattr(state, selection[0], selection[1])
    elif character == 'z':
        state._status_only = True
    elif character == 'p':
        state._interactive = True
        state._interactive_option = '-p'
    else:
        return character in _COMMIT_BOOL_SHORT
    return True


def _fragment(text: str) -> str:
    """Return at most the first 40 characters of a reason fragment."""
    return text[:_FRAGMENT_LIMIT].rstrip()


def _amend_prefix(word: Word) -> bool | Rejection:
    """Classify a commit segment's first word as a supported assignment or not one."""
    assignment = _assignment_name(word)
    if assignment is None:
        return False
    name, rest = assignment
    if name == _ASSIGNMENT_NAME and _amend_value_accepted(rest, word.parts[1:]):
        return True
    return Rejection(f"assignment '{_fragment(word.raw)}'", _HINT_SEQUENCE)


def _carriage_return(words: Sequence[Word]) -> Rejection | None:
    """Reject carriage returns in literal words without inspecting opaque bodies."""
    for word in words:
        if '\r' in (word.literal_text or ''):
            return Rejection(f"carriage return in '{_fragment(word.raw)}'", _HINT_SEQUENCE)
    return None


def _git_route(words: Sequence[Word]) -> tuple[str | None, int] | None:
    """Return the `-C` route and the subcommand index, or `None` for a malformed `-C`."""
    if len(words) > 1 and words[1].literal_text == '-C':
        route = words[_GIT_ROUTE_INDEX].literal_text if len(words) > _GIT_ROUTE_INDEX else None
        if not route:
            return None
        return route, 3
    return None, 1


def _is_absolute_route(route: str) -> bool:
    """Return whether a literal route needs no base directory."""
    return Path(normalize_msys_path(route)).is_absolute()


def _parse_add(texts: Sequence[str]) -> Rejection | None:
    """Check a `git add` segment against the closed add forms."""
    if not texts:
        return Rejection('git add without pathspec', _HINT_ADD)
    selector = texts[0] in ADD_SELECTORS
    index = 1 if selector else 0
    options_ended = index < len(texts) and texts[index] == '--'
    if options_ended:
        index += 1
    pathspecs = texts[index:]
    if not selector and not pathspecs:
        return Rejection('git add without pathspec', _HINT_ADD)
    if not options_ended:
        for pathspec in pathspecs:
            if pathspec.startswith('-'):
                return Rejection(f"option '{_fragment(pathspec)}' on git add", _HINT_ADD)
    return None


def _parse_read_only(subcommand: str, texts: Sequence[str]) -> Rejection | None:
    """Check a status, diff, or log segment against its option allowlist."""
    allowed = _READ_ONLY_OPTIONS[subcommand]
    index = 0
    while index < len(texts):
        text = texts[index]
        if text in allowed:
            index += 1
            continue
        if (
            subcommand == 'status'
            and len(text) > 1
            and text.startswith('-')
            and not text.startswith('--')
            and all(character in 'sb' for character in text[1:])
        ):
            index += 1
            continue
        if subcommand == 'log':
            if text.startswith('--format=') or _LOG_COUNT_RE.fullmatch(text):
                index += 1
                continue
            if text == '-n' and index + 1 < len(texts) and texts[index + 1].isdigit():
                index += 2
                continue
        kind = 'option' if text.startswith('-') and text != '--' else 'argument'
        return Rejection(f"{kind} '{_fragment(text)}' on git {subcommand}", _HINT_READ_ONLY)
    return None


def _reject_word(word: Word, hint: str) -> Rejection:
    """Name the first non-literal part of a word."""
    for part in word.parts:
        if isinstance(part, SubstitutionPart):
            return Rejection(f"substitution in '{_fragment(word.raw)}'", hint)
        if isinstance(part, ExpansionPart):
            return Rejection(f"expansion '{_fragment(part.raw)}'", hint)
        if isinstance(part, GlobPart):
            return Rejection(f"glob '{part.raw}'", hint)
    return Rejection(f"word '{_fragment(word.raw)}'", hint)


def _heredoc_message(word: Word) -> Rejection | None:
    """Accept the double-quoted heredoc message idiom, else name the defect."""
    if len(word.parts) != 1 or not isinstance(word.parts[0], SubstitutionPart):
        return _reject_word(word, _HINT_COMMIT)
    part = word.parts[0]
    segments = part.command.segments
    if len(segments) != 1 or segments[0].terminator not in (None, '\n'):
        return _reject_word(word, _HINT_COMMIT)
    elements = segments[0].elements
    shape = (
        len(elements) == _HEREDOC_ELEMENT_COUNT
        and isinstance(elements[0], Word)
        and elements[0].literal_text == 'cat'
        and isinstance(elements[1], Operator)
        and elements[1].text == '<<'
        and isinstance(elements[2], Heredoc)
        and elements[2].quoted
    )
    if not shape:
        return _reject_word(word, _HINT_COMMIT)
    if not part.in_double_quotes:
        return Rejection('unquoted heredoc message', _HINT_COMMIT)
    return None


def _commit_message_word(value: Word, state: _CommitState) -> Rejection | None:
    """Accept one heredoc message word, at most once per commit segment."""
    rejection = _heredoc_message(value)
    if rejection is not None:
        return rejection
    state._heredoc_messages += 1
    if state._heredoc_messages > 1:
        return Rejection('second heredoc message', _HINT_COMMIT)
    return None


def _commit_value(
    name: str,
    value: Word | str,
    state: _CommitState,
    next_index: int,
) -> int | Rejection:
    """Record one option value and return the index after it."""
    if isinstance(value, Word) and value.literal_text is None:
        if name not in ('-m', '--message'):
            return _reject_word(value, _HINT_COMMIT)
        rejection = _commit_message_word(value, state)
        return next_index if rejection is None else rejection
    literal = value if isinstance(value, str) else (value.literal_text or '')
    if name == '--fixup' and literal.startswith('reword:'):
        return Rejection('option --fixup=reword:', _HINT_COMMIT)
    if name == '--pathspec-from-file':
        if literal in ('', '-'):
            return Rejection('option --pathspec-from-file=-', _HINT_COMMIT)
        state._pathspec_from_file = literal
    return next_index


def _commit_long_option(
    text: str,
    words: Sequence[Word],
    index: int,
    state: _CommitState,
) -> int | Rejection:
    """Apply one long commit option and return the index after it."""
    name, separator, joined = text.partition('=')
    if separator and name in _COMMIT_VALUE_LONG:
        return _commit_value(name, joined, state, index + 1)
    if separator:
        joined_rejection = Rejection(f'option {_fragment(text)}', _HINT_COMMIT)
        return index + 1 if name in _COMMIT_BOOL_LONG_JOINED else joined_rejection
    if _commit_flag(name, state):
        return index + 1
    if name not in _COMMIT_VALUE_LONG:
        return Rejection(f'option {_fragment(text)}', _HINT_COMMIT)
    if index + 1 >= len(words):
        return Rejection(f'missing value for {name}', _HINT_COMMIT)
    return _commit_value(name, words[index + 1], state, index + 2)


def _commit_short_cluster(
    text: str,
    words: Sequence[Word],
    index: int,
    state: _CommitState,
) -> int | Rejection:
    """Apply one short option cluster and return the index after it."""
    position = 1
    while position < len(text):
        character = text[position]
        if _commit_short_flag(character, state):
            position += 1
            continue
        if character in _COMMIT_OPTIONAL_VALUE_SHORT:
            return index + 1
        if character not in _COMMIT_VALUE_SHORT:
            return Rejection(f'option -{character}', _HINT_COMMIT)
        remainder = text[position + 1 :]
        if remainder:
            return _commit_value(f'-{character}', remainder, state, index + 1)
        if index + 1 >= len(words):
            return Rejection(f'missing value for -{character}', _HINT_COMMIT)
        return _commit_value(f'-{character}', words[index + 1], state, index + 2)
    return index + 1


def _literal_texts(words: Sequence[Word], hint: str) -> tuple[str, ...] | Rejection:
    """Return every word's argv text, rejecting the first non-literal word."""
    texts: list[str] = []
    for word in words:
        text = word.literal_text
        if text is None:
            return _reject_word(word, hint)
        texts.append(text)
    return tuple(texts)


def _parse_commit(words: Sequence[Word]) -> _CommitState | Rejection:
    """Parse the words after `git commit` against the option inventory."""
    state = _CommitState()
    index = 0
    while index < len(words):
        word = words[index]
        text = word.literal_text
        if text is None:
            return _reject_word(word, _HINT_COMMIT)
        if state._options_ended or not text.startswith('-') or text == '-':
            state._pathspecs.append(text)
            index += 1
            continue
        if text == '--':
            state._options_ended = True
            index += 1
            continue
        if text in _COMMIT_HELP:
            return Rejection(f'option {text}', _HINT_COMMIT)
        if text.startswith('--'):
            consumed = _commit_long_option(text, words, index, state)
        else:
            consumed = _commit_short_cluster(text, words, index, state)
        if isinstance(consumed, Rejection):
            return consumed
        index = consumed
    return _commit_finish(state)


def _parse_git_arguments(
    git: _GitSegment,
    words: Sequence[Word],
    index: int,
) -> _GitSegment | Rejection:
    """Parse the arguments after a git subcommand into the segment record."""
    if git.subcommand == 'commit':
        parsed = _parse_commit(words[index:])
        if isinstance(parsed, Rejection):
            return parsed
        return _GitSegment(git.subcommand, git.route, (), git.text, parsed)
    texts = _literal_texts(words, _HINT_ADD if git.subcommand == 'add' else _HINT_READ_ONLY)
    if isinstance(texts, Rejection):
        return texts
    rejection = (
        _parse_add(texts[index:])
        if git.subcommand == 'add'
        else _parse_read_only(git.subcommand, texts[index:])
    )
    if rejection is not None:
        return rejection
    return _GitSegment(git.subcommand, git.route, texts, git.text)


def _resolve_plan(
    cd_route: str | None,
    parsed: Sequence[_GitSegment],
    command_cwd: str | None,
) -> CommitPlan | NoCommit | Rejection:
    """Resolve the route and build the plan for a shape-checked sequence."""
    state = parsed[-1].commit
    if state is None:
        raise ValueError('a validated commit sequence must contain commit state')
    if state._status_only:
        return NoCommit()
    route = cd_route if cd_route is not None else parsed[0].route
    if route is None:
        if command_cwd is None:
            return Rejection(_ELEMENT_UNKNOWN_CWD, _HINT_ROUTE)
        cwd = command_cwd
    else:
        if command_cwd is None and not _is_absolute_route(route):
            return Rejection(_ELEMENT_RELATIVE_ROUTE, _HINT_ROUTE)
        try:
            resolved = resolve_route(route, command_cwd, logical=cd_route is not None)
        except (OSError, ValueError):
            return Rejection('unresolvable directory route', _HINT_ROUTE)
        if resolved is None:
            return NoCommit()
        cwd = resolved
    mode = CommitMode(
        auto_stage=state._auto_stage,
        include=state._include,
        only=state._only,
        pathspecs=tuple(state._pathspecs),
        pathspec_from_file=state._pathspec_from_file,
        pathspec_file_nul=state._pathspec_file_nul,
    )
    exec_cwd = command_cwd if cd_route is None and command_cwd is not None else cwd
    pre_commit_adds = tuple(
        PreCommitAdd(tokens=git.tokens, exec_cwd=exec_cwd, git_cwd=cwd)
        for git in parsed[:-1]
        if git.subcommand == 'add'
    )
    return CommitPlan(cwd, mode, pre_commit_adds)


def _segment_words(segment: Segment) -> tuple[Word, ...] | Rejection:
    """Return a segment's words, rejecting other elements and separators."""
    if segment.terminator is not None and segment.terminator not in _SEQUENCE_TERMINATORS:
        return Rejection(f"separator '{segment.terminator}'", _HINT_SEQUENCE)
    words: list[Word] = []
    for element in segment.elements:
        if isinstance(element, Word):
            words.append(element)
        elif isinstance(element, Operator):
            return Rejection(f"operator '{element.text}'", _HINT_SEQUENCE)
        elif isinstance(element, Heredoc):
            return Rejection('heredoc', _HINT_SEQUENCE)
        else:
            return Rejection('comment', _HINT_SEQUENCE)
    if not words:
        return Rejection('empty segment', _HINT_SEQUENCE)
    return _carriage_return(words) or tuple(words)


def _word_text(word: Word) -> str:
    """Return a word's argv text, or its raw text when it is not literal."""
    text = word.literal_text
    return word.raw if text is None else text


def _has_msys_path(words: Sequence[Word]) -> bool:
    """Return whether a Windows command carries an MSYS mount path."""
    return sys.platform == 'win32' and any(is_msys_mount_path(_word_text(word)) for word in words)


def _segment_text(words: Sequence[Word]) -> str:
    """Join the words of a segment for a reason fragment."""
    return _fragment(' '.join(_word_text(word) for word in words))


def _parse_cd(words: Sequence[Word], terminator: str | None) -> str | Rejection:
    """Return the literal route of a routing `cd` segment."""
    texts = _literal_texts(words, _HINT_SEQUENCE)
    if isinstance(texts, Rejection):
        return texts
    if len(texts) != _CD_SEGMENT_SIZE or not texts[1]:
        return Rejection(f"segment '{_segment_text(words)}'", _HINT_SEQUENCE)
    route = texts[1]
    if not (
        route.startswith(('/', './', '../')) or Path(normalize_msys_path(route)).is_absolute()
    ):
        return Rejection(f"cd target '{_fragment(route)}'", _HINT_SEQUENCE)
    if terminator == '\n':
        return Rejection('line feed after cd', _HINT_SEQUENCE)
    if terminator != '&&':
        return Rejection(f"separator '{terminator or 'end'}' after cd", _HINT_SEQUENCE)
    return route


def _parse_git_segment(segment: Segment, *, final: bool) -> _GitSegment | Rejection:
    """Parse one git segment, allowing the amend prefix only on the final segment."""
    words = _segment_words(segment)
    if isinstance(words, Rejection):
        return words
    text = _segment_text(words)
    assignment_word = words[0]
    prefix = _amend_prefix(words[0]) if final else False
    if isinstance(prefix, Rejection):
        return prefix
    if prefix:
        words = words[1:]
    segment_rejection = Rejection(f"segment '{text}'", _HINT_SEQUENCE)
    is_git = bool(words) and words[0].literal_text == 'git'
    first_text = words[0].literal_text if words else None
    if first_text is not None and first_text != 'git' and is_git_executable(first_text):
        segment_rejection = Rejection(f"executable '{_fragment(first_text)}'", _HINT_EXECUTABLE)
        is_git = False
    if is_git and _has_msys_path(words):
        segment_rejection = Rejection(
            f"MSYS-style word in Windows git segment '{text}'",
            'Windows git segments reject /<letter> and /<letter>/... words; '
            'use native Windows spelling such as C:/... for paths',
        )
        is_git = False
    routing = _git_route(words) if is_git else None
    if routing is None:
        return segment_rejection
    route, index = routing
    subcommand = words[index].literal_text if index < len(words) else None
    if subcommand is None or subcommand not in _GIT_SUBCOMMANDS:
        return segment_rejection
    if prefix and subcommand != 'commit':
        return Rejection(f"assignment '{_fragment(assignment_word.raw)}'", _HINT_SEQUENCE)
    return _parse_git_arguments(_GitSegment(subcommand, route, (), text), words, index + 1)


def _sequence(segments: Sequence[Segment]) -> tuple[str | None, list[_GitSegment]] | Rejection:
    """Parse the routing `cd` and every git segment of a command."""
    remaining = list(segments)
    cd_route: str | None = None
    first = _segment_words(remaining[0])
    if not isinstance(first, Rejection) and first[0].literal_text == 'cd':
        route = _parse_cd(first, remaining[0].terminator)
        if isinstance(route, Rejection):
            return route
        cd_route = route
        remaining.pop(0)
        if not remaining:
            return Rejection('cd without git commit', _HINT_SEQUENCE)
    parsed: list[_GitSegment] = []
    for position, segment in enumerate(remaining):
        git = _parse_git_segment(segment, final=position == len(remaining) - 1)
        if isinstance(git, Rejection):
            return git
        parsed.append(git)
    return cd_route, parsed


def classify_grammar(
    command: Command, command_cwd: str | None
) -> CommitPlan | NoCommit | Rejection:
    """Accept a command in the closed commit grammar or name its first defect.

    Args:
        command: Parsed command tree.
        command_cwd: Absolute directory the command executes in, or `None` when
            the host cannot establish it. A committing command with an unknown
            directory is accepted only through an absolute route.

    Returns:
        The resolved commit plan, `NoCommit` for a status-only commit or a
        route whose target does not exist, or the rejection naming the first
        unsupported element.
    """
    if command.recovery:
        return Rejection(command.recovery[0], _HINT_SEQUENCE)
    if not command.segments:
        return Rejection('empty command', _HINT_SEQUENCE)
    sequence = _sequence(command.segments)
    if isinstance(sequence, Rejection):
        return sequence
    cd_route, parsed = sequence
    rejection = _check_shape(cd_route, parsed)
    if rejection is not None:
        return rejection
    plan = _resolve_plan(cd_route, parsed, command_cwd)
    if isinstance(plan, CommitPlan):
        for git in parsed[:-1]:
            if git.subcommand in _READ_ONLY_OPTIONS:
                return Rejection(
                    f'read-only git {git.subcommand} before git commit', _HINT_PRECHECK
                )
    return plan
