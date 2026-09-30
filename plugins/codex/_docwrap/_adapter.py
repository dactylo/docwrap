"""Translate Codex hook events for docwrap."""

import json
from collections.abc import Mapping

from docwrap.hooks import (
    API_VERSION,
    HookResult,
    HostInput,
    HostResponse,
    notes_text,
    object_field,
    payload_cwd,
)

_APPLY_PATCH_FILE_PREFIXES = (
    '*** Add File: ',
    '*** Delete File: ',
    '*** Move to: ',
    '*** Update File: ',
)
_EDIT_TOOL = 'apply_patch'


def _allow_response(result: HookResult) -> HostResponse:
    """Render Codex allow feedback."""
    stdout = f'{json.dumps({"systemMessage": result.reason})}\n' if result.reason else ''
    return HostResponse(stdout, notes_text(result), 0)


def _apply_patch_paths(command: str) -> tuple[str, ...]:
    """Extract touched file paths from an `apply_patch` command."""
    paths: list[str] = []
    for line in command.splitlines():
        for prefix in _APPLY_PATCH_FILE_PREFIXES:
            if line.startswith(prefix):
                path = line.removeprefix(prefix).strip()
                if path and path not in paths:
                    paths.append(path)
                break
    return tuple(paths)


def _permission_deny(reason: str, stderr_prefix: str = '') -> HostResponse:
    """Render Codex's `hookSpecificOutput` deny shape."""
    payload = {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        },
    }
    return HostResponse(f'{json.dumps(payload)}\n', f'{stderr_prefix}{reason}\n', 0)


class CodexAdapter:
    """Translate Codex `PreToolUse` and `PostToolUse` payloads."""

    name = 'codex'
    api_version = API_VERSION
    requires_event = False

    def parse(self, payload: Mapping[str, object], event: str | None) -> HostInput | None:
        """Read `hook_event_name`, `tool_name`, `tool_input`, and `cwd`.

        Codex defines top-level `cwd` as the session directory and exposes no
        execution directory for a shell call, so pre-tool `command_cwd` is
        always `None`. Undocumented directory-like fields are ignored.

        Args:
            payload: Decoded Codex hook payload.
            event: Always `None`; Codex payloads carry their event.

        Returns:
            Neutral input for a `PreToolUse` or `PostToolUse` payload, else
            `None`. Post-tool paths come from the `apply_patch` command text.
        """
        del event
        hook_event = payload.get('hook_event_name')
        tool_name = payload.get('tool_name')
        tool_input = object_field(payload, 'tool_input')
        cwd = payload_cwd(payload)
        if hook_event == 'PreToolUse':
            command = tool_input.get('command') if tool_name == 'Bash' else None
            return HostInput('pre_tool', command if isinstance(command, str) else None, cwd, None)
        if hook_event == 'PostToolUse':
            command = tool_input.get('command') if tool_name == _EDIT_TOOL else None
            paths = _apply_patch_paths(command) if isinstance(command, str) else ()
            return HostInput('post_tool', None, cwd, None, paths)
        return None

    def render(self, result: HookResult) -> HostResponse:
        """Render allow feedback or a denial.

        Args:
            result: Decision, reason, and notes from docwrap.

        Returns:
            Empty stdout for a silent allow, or one system message for allow
            feedback; notes on stderr and exit 0. For deny, the deny JSON on
            stdout, the reason on stderr, and exit 0.
        """
        notes = notes_text(result)
        if result.decision == 'deny':
            return _permission_deny(result.reason or 'docwrap denied the operation', notes)
        return _allow_response(result)

    def render_failure(self, reason: str) -> HostResponse:
        """Render the deny JSON that blocks the tool call.

        Args:
            reason: Reason for the failure.

        Returns:
            The deny JSON on stdout, the reason on stderr, and exit 0.
        """
        return _permission_deny(reason)
