"""Translate Claude Code hook events for docwrap."""

import json
from collections.abc import Mapping

from docwrap.hooks import (
    API_VERSION,
    HookResult,
    HostInput,
    HostResponse,
    absolute_directory,
    notes_text,
    object_field,
    payload_cwd,
)

_EDIT_TOOLS = frozenset({'Edit', 'Write'})


def _allow_response(result: HookResult) -> HostResponse:
    """Render Claude Code allow feedback."""
    stdout = f'{json.dumps({"systemMessage": result.reason})}\n' if result.reason else ''
    return HostResponse(stdout, notes_text(result), 0)


def _permission_deny(reason: str, stderr_prefix: str = '') -> HostResponse:
    """Render Claude Code's `hookSpecificOutput` deny shape."""
    payload = {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny',
            'permissionDecisionReason': reason,
        },
    }
    return HostResponse(f'{json.dumps(payload)}\n', f'{stderr_prefix}{reason}\n', 0)


class ClaudeAdapter:
    """Translate Claude Code `PreToolUse` and `PostToolUse` payloads."""

    name = 'claude'
    api_version = API_VERSION
    requires_event = False

    def parse(self, payload: Mapping[str, object], event: str | None) -> HostInput | None:
        """Read `hook_event_name`, `tool_name`, `tool_input`, and `cwd`.

        For Bash pre-tool calls, `command_cwd` is the top-level `cwd` when it
        is an absolute path; Claude Code documents it as the current directory
        of the tool call. Otherwise it is `None`.

        Args:
            payload: Decoded Claude Code hook payload.
            event: Always `None`; Claude Code payloads carry their event.

        Returns:
            Neutral input for a `PreToolUse` or `PostToolUse` payload, else
            `None`.
        """
        del event
        hook_event = payload.get('hook_event_name')
        tool_name = payload.get('tool_name')
        tool_input = object_field(payload, 'tool_input')
        cwd = payload_cwd(payload)
        if hook_event == 'PreToolUse':
            command = tool_input.get('command') if tool_name == 'Bash' else None
            command_cwd = absolute_directory(payload, 'cwd') if tool_name == 'Bash' else None
            return HostInput(
                'pre_tool',
                command if isinstance(command, str) else None,
                cwd,
                command_cwd,
            )
        if hook_event == 'PostToolUse':
            editing = isinstance(tool_name, str) and tool_name in _EDIT_TOOLS
            path = tool_input.get('file_path') if editing else None
            paths = (path,) if isinstance(path, str) and path else ()
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
