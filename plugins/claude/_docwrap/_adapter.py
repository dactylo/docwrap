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
    """Translate Claude Code `PreToolUse` and `PostToolUse` payloads.

    Call `parse()` before `render()` on the same instance. Parsing each payload
    replaces the event context used to choose the response format.
    """

    name = 'claude'
    api_version = API_VERSION
    requires_event = False
    _post_tool = False

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
        self._post_tool = False
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
            self._post_tool = True
            editing = isinstance(tool_name, str) and tool_name in _EDIT_TOOLS
            path = tool_input.get('file_path') if editing else None
            paths = (path,) if isinstance(path, str) and path else ()
            return HostInput('post_tool', None, cwd, None, paths)
        return None

    def render(self, result: HookResult) -> HostResponse:
        """Render feedback using the event recorded by `parse()`.

        Args:
            result: Decision, reason, and notes from docwrap.

        Returns:
            For pre-tool input, empty stdout for a silent allow, a system
            message for allow feedback, or deny JSON with the denial reason on
            stderr. For post-tool input, distinct nonempty diagnostic lines as
            additional context, or empty stdout when there are none.
            Notes remain on stderr and the exit status is 0.
        """
        if self._post_tool:
            return _post_tool_response(result)
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


def _post_tool_response(result: HookResult) -> HostResponse:
    """Render distinct diagnostic lines as post-tool context."""
    lines = '\n'.join((*result.notes, result.reason or '')).split('\n')
    context = '\n'.join(dict.fromkeys(line for line in lines if line))
    payload = {
        'hookSpecificOutput': {
            'hookEventName': 'PostToolUse',
            'additionalContext': context,
        },
    }
    stdout = f'{json.dumps(payload)}\n' if context else ''
    return HostResponse(stdout, notes_text(result), 0)
