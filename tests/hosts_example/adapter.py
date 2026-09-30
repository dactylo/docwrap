import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal, Protocol, cast


class _Result(Protocol):
    @property
    def decision(self) -> Literal['allow', 'deny']: ...

    @property
    def reason(self) -> str | None: ...

    @property
    def notes(self) -> Sequence[str]: ...


@dataclass(frozen=True)
class ExampleInput:
    event: Literal['pre_tool', 'post_tool']
    command: str | None
    cwd: str
    command_cwd: str | None
    edited_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExampleResponse:
    stdout: str
    stderr: str
    exit_code: int


class ExampleAdapter:
    name = 'example'
    api_version = 1
    requires_event = False

    def parse(self, payload: Mapping[str, object], event: str | None) -> ExampleInput | None:
        del event
        tool_input = payload.get('tool_input')
        fields = cast('dict[str, object]', tool_input) if isinstance(tool_input, dict) else {}
        cwd = _string(payload, 'cwd') or '.'
        command_cwd = cwd if PurePath(cwd).is_absolute() else None
        if payload.get('hook_event_name') == 'PreToolUse':
            command = _string(fields, 'command') if payload.get('tool_name') == 'Bash' else None
            return ExampleInput('pre_tool', command, cwd, command_cwd)
        if payload.get('hook_event_name') == 'PostToolUse':
            path = _string(fields, 'file_path')
            return ExampleInput('post_tool', None, cwd, None, (path,) if path else ())
        return None

    def render(self, result: _Result) -> ExampleResponse:
        notes = ''.join(f'{note}\n' for note in result.notes)
        if result.decision == 'deny':
            return _deny(result.reason or 'denied', notes)
        return ExampleResponse('', notes, 0)

    def render_failure(self, reason: str) -> ExampleResponse:
        return _deny(reason)


def _deny(reason: str, notes: str = '') -> ExampleResponse:
    payload = {'example': {'decision': 'deny', 'reason': reason}}
    return ExampleResponse(f'{json.dumps(payload)}\n', f'{notes}{reason}\n', 0)


class StaleAdapter(ExampleAdapter):
    name = 'stale'
    api_version = 0


class RequiresEventAdapter(ExampleAdapter):
    name = 'requires-event'
    requires_event = True


class BrokenParseAdapter(ExampleAdapter):
    name = 'broken-parse'

    def parse(self, payload: Mapping[str, object], event: str | None) -> ExampleInput | None:
        del payload, event
        raise RuntimeError('parse exploded')


class BrokenRenderAdapter(ExampleAdapter):
    name = 'broken-render'

    def render(self, result: _Result) -> ExampleResponse:
        del result
        raise RuntimeError('render exploded')


class BrokenBothAdapter(BrokenRenderAdapter):
    name = 'broken-both'

    def render_failure(self, reason: str) -> ExampleResponse:
        del reason
        raise RuntimeError('render_failure exploded')


class EmptyCwdAdapter(ExampleAdapter):
    name = 'empty-cwd'

    def parse(self, payload: Mapping[str, object], event: str | None) -> ExampleInput | None:
        parsed = super().parse(payload, event)
        if parsed is None:
            return None
        return ExampleInput(parsed.event, parsed.command, parsed.cwd, '', parsed.edited_paths)


class RelativeCwdAdapter(ExampleAdapter):
    name = 'relative-cwd'

    def parse(self, payload: Mapping[str, object], event: str | None) -> ExampleInput | None:
        parsed = super().parse(payload, event)
        if parsed is None:
            return None
        return ExampleInput(parsed.event, parsed.command, parsed.cwd, 'rel', parsed.edited_paths)


class TypedCwdAdapter(ExampleAdapter):
    name = 'typed-cwd'

    def parse(self, payload: Mapping[str, object], event: str | None) -> ExampleInput | None:
        parsed = super().parse(payload, event)
        if parsed is None:
            return None
        bad = cast('str', 7)
        return ExampleInput(parsed.event, parsed.command, parsed.cwd, bad, parsed.edited_paths)


def _string(fields: Mapping[str, object], key: str) -> str | None:
    value = fields.get(key)
    return value if isinstance(value, str) else None
