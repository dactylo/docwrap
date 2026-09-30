"""Classify a pre-tool shell command into one of three outcomes."""

from dataclasses import dataclass
from typing import Literal

from docwrap._color import visible_controls
from docwrap._detector import detect_commit
from docwrap._grammar import CommitPlan, NoCommit, Rejection, classify_grammar
from docwrap._shellparse import parse_command

type _OUTCOME = Literal['unrelated', 'supported', 'unsupported']

_REASON_PREFIX = 'docwrap: unsupported commit command'


@dataclass(frozen=True, slots=True)
class Classification:
    """Outcome of classifying one command.

    Args:
        outcome: `unrelated`, `supported`, or `unsupported`.
        plan: Resolved commit plan, or `NoCommit`, for a supported command.
        reason: Deny reason for an unsupported command.
    """

    outcome: _OUTCOME
    plan: CommitPlan | NoCommit | None = None
    reason: str | None = None


def _deny_reason(rejection: Rejection) -> str:
    """Render a sanitized deny reason for a rejected command."""
    return visible_controls(f'{_REASON_PREFIX}: {rejection.element}; {rejection.hint}')


def classify_command(text: str, command_cwd: str | None) -> Classification:
    """Classify a shell command as unrelated, supported, or unsupported.

    Args:
        text: Shell command text from the hook payload.
        command_cwd: Absolute directory the command executes in, or `None` when
            unknown.

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
            return Classification('unsupported', reason=_deny_reason(result))
        return Classification('unrelated')
    return Classification('supported', plan=result)
