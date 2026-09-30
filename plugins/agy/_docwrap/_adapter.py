"""Translate Antigravity hook events for docwrap."""

import json
from collections.abc import Mapping
from typing import cast

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

_COMMAND_TOOL = 'run_command'
_EDIT_TOOLS = frozenset({'replace_file_content', 'write_to_file'})
_EVENTS = {'PreToolUse': 'pre_tool', 'PostToolUse': 'post_tool'}


def _allow_response(result: HookResult) -> HostResponse:
    """Render Antigravity allow feedback."""
    stdout = f'{json.dumps({"systemMessage": result.reason})}\n' if result.reason else ''
    return HostResponse(stdout, notes_text(result), 0)


def _base_cwd(payload: Mapping[str, object], args: Mapping[str, object]) -> str:
    """Resolve the target directory from `Cwd`, `workspacePaths`, `cwd`, or the process."""
    cwd = args.get('Cwd')
    if isinstance(cwd, str) and cwd:
        return cwd
    workspaces = payload.get('workspacePaths')
    if isinstance(workspaces, list):
        first = cast('list[object]', workspaces)[:1]
        if first and isinstance(first[0], str) and first[0]:
            return first[0]
    return payload_cwd(payload)


def _deny(reason: str, stderr_prefix: str = '') -> HostResponse:
    """Render Antigravity's top-level deny object."""
    stdout = json.dumps({'decision': 'deny', 'reason': reason})
    return HostResponse(f'{stdout}\n', f'{stderr_prefix}{reason}\n', 0)


class AntigravityAdapter:
    """Translate Antigravity `toolCall` payloads, which carry no event name."""

    name = 'agy'
    api_version = API_VERSION
    requires_event = True

    def parse(self, payload: Mapping[str, object], event: str | None) -> HostInput | None:
        """Read `toolCall.name` and `toolCall.args` under the given event.

        For `run_command`, pre-tool `command_cwd` is `toolCall.args.Cwd` when
        it is an absolute path; `workspacePaths` and the process directory are
        never invocation evidence. Other tools report `None`.

        Args:
            payload: Decoded Antigravity hook payload.
            event: `PreToolUse` or `PostToolUse` from `--event`.

        Returns:
            Neutral input; a tool other than `run_command` under `PreToolUse`
            or other than the file writers under `PostToolUse` yields input
            with nothing to act on.

        Raises:
            ValueError: If `event` is not an Antigravity hook event.
        """
        neutral_event = _EVENTS.get(event or '')
        if neutral_event is None:
            raise ValueError(
                f'unknown Antigravity event {event!r}; expected PreToolUse or PostToolUse'
            )
        tool_call = object_field(payload, 'toolCall')
        args = object_field(tool_call, 'args')
        cwd = _base_cwd(payload, args)
        name = tool_call.get('name')
        if neutral_event == 'pre_tool':
            command = args.get('CommandLine') if name == _COMMAND_TOOL else None
            command_cwd = absolute_directory(args, 'Cwd') if name == _COMMAND_TOOL else None
            return HostInput(
                'pre_tool',
                command if isinstance(command, str) else None,
                cwd,
                command_cwd,
            )
        editing = isinstance(name, str) and name in _EDIT_TOOLS
        target = args.get('TargetFile') if editing else None
        paths = (target,) if isinstance(target, str) and target else ()
        return HostInput('post_tool', None, cwd, None, paths)

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
            return _deny(result.reason or 'docwrap denied the operation', notes)
        return _allow_response(result)

    def render_failure(self, reason: str) -> HostResponse:
        """Render the top-level deny object.

        Args:
            reason: Reason for the failure.

        Returns:
            The deny object on stdout, the reason on stderr, and exit 0.
        """
        return _deny(reason)
