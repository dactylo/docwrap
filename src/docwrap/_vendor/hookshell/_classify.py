"""Classify a pre-tool shell command into one of three outcomes."""

from dataclasses import dataclass
from typing import Literal

from ._detector import detect_commit
from ._grammar import CommitPlan, NoCommit, Rejection, classify_grammar
from ._shellparse import parse_command

_C0_LIMIT = 0x20
_C1_END = 0x9F
_DEL_CODEPOINT = 0x7F

type _Outcome = Literal['unrelated', 'supported', 'unsupported']


@dataclass(frozen=True, slots=True)
class Classification:
    """Outcome of classifying one command.

    Attributes:
        outcome: `unrelated`, `supported`, or `unsupported`.
        plan: Resolved commit plan, or `NoCommit`, for a supported command.
        reason: Deny reason for an unsupported command.
    """

    outcome: _Outcome
    plan: CommitPlan | NoCommit | None = None
    reason: str | None = None


def _escape_controls(text: str) -> str:
    """Render C0, DEL, and C1 controls as lowercase hexadecimal escapes."""
    return ''.join(
        f'\\x{codepoint:02x}'
        if codepoint < _C0_LIMIT or _DEL_CODEPOINT <= codepoint <= _C1_END
        else character
        for character in text
        for codepoint in (ord(character),)
    )


def _deny_reason(rejection: Rejection, reason_prefix: str) -> str:
    """Render a sanitized deny reason for a rejected command."""
    return _escape_controls(f'{reason_prefix}: {rejection.element}; {rejection.hint}')


def classify_command(
    text: str,
    command_cwd: str | None,
    *,
    reason_prefix: str,
) -> Classification:
    """Classify a shell command as unrelated, supported, or unsupported.

    Args:
        text: Shell command text from the hook payload.
        command_cwd: Absolute directory the command executes in, or `None` when
            unknown.
        reason_prefix: Caller-specific prefix for unsupported-command reasons.

    Returns:
        The classification. The grammar decides first; a rejected command is
        unsupported when the detector finds a commit and unrelated otherwise.
        A committing command with an unknown directory and no absolute route is
        unsupported.
    """
    command = parse_command(text)
    result = classify_grammar(command, command_cwd)
    if isinstance(result, Rejection):
        if detect_commit(command):
            return Classification('unsupported', reason=_deny_reason(result, reason_prefix))
        return Classification('unrelated')
    return Classification('supported', plan=result)
