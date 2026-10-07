"""Light hook parsing, classification, rendering, and engine dispatch."""

import argparse
import contextlib
import io
import json
import sys
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override

from docwrap._arguments import incremental_override
from docwrap._color import visible_controls
from docwrap._hosts import AdapterLoadError, load_adapter
from docwrap._vendor.hookshell._classify import classify_command
from docwrap._vendor.hookshell._grammar import CommitPlan
from docwrap._vendor.hookshell._shell import normalize_msys_path
from docwrap.hooks import HostInput

if TYPE_CHECKING:
    from docwrap._vendor.hookshell._classify import Classification
    from docwrap.hooks import HookAdapter, HookEvent, HookInput

_LAST_RESORT_EXIT = 2
_REASON_PREFIX = 'docwrap: unsupported commit command'


class _ClassificationError(Exception):
    """Signal a failure before command eligibility has been established."""


class _LastResortError(Exception):
    """Signal that no functioning host adapter exists for this invocation."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """Decision of the hook core before a host adapter renders it.

    Attributes:
        decision: Whether the host should allow or deny the tool call.
        reason: Optional explanation rendered by the host adapter.
        notes: Diagnostic lines captured while processing the invocation.
    """

    decision: Literal['allow', 'deny']
    reason: str | None = None
    notes: tuple[str, ...] = ()


def _check_command_cwd(host: str, value: object) -> None:
    """Reject an adapter directory that is neither `None` nor an absolute path string."""
    if value is None:
        return
    if isinstance(value, str) and value and Path(normalize_msys_path(value)).is_absolute():
        return
    raise _LastResortError(
        f"host '{host}' supplied an invalid command_cwd {value!r};"
        ' expected None or an absolute path'
    )


def _classify_for_hook(command: str, command_cwd: str | None) -> 'Classification':
    """Translate classifier exceptions into a failure that always blocks."""
    try:
        normalized = None if command_cwd is None else normalize_msys_path(command_cwd)
        return classify_command(command, normalized, reason_prefix=_REASON_PREFIX)
    except Exception as error:
        raise _ClassificationError('docwrap: unexpected error during classification') from error


def _last_resort(diagnostic: str) -> int:
    """Write one diagnostic line and exit 2 when no functioning adapter exists."""
    print(f'docwrap: {visible_controls(diagnostic)}', file=sys.stderr)
    return _LAST_RESORT_EXIT


def _post_tool_paths(cwd: object, paths: Iterable[object]) -> tuple[str, tuple[str, ...]]:
    """Validate the PostToolUse path fields consumed by the hook core."""
    if not isinstance(cwd, str):
        raise TypeError('post-tool cwd must be a string')
    edited_paths: list[str] = []
    for path in paths:
        if not isinstance(path, str):
            raise TypeError('post-tool edited paths must be strings')
        edited_paths.append(path)
    return cwd, tuple(edited_paths)


def _response_fields(stdout: object, stderr: object, exit_code: object) -> tuple[str, str, int]:
    """Validate every response field before either output stream is written."""
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise TypeError('adapter response streams must be strings')
    if not isinstance(exit_code, int):
        raise TypeError('adapter response exit_code must be an integer')
    return stdout, stderr, exit_code


def _render_response(adapter: 'HookAdapter', host: str, result: Outcome) -> tuple[str, str, int]:
    """Translate adapter render exceptions into the typed recovery boundary."""
    try:
        response = adapter.render(result)
        return _response_fields(response.stdout, response.stderr, response.exit_code)
    except Exception as error:
        raise _LastResortError(f"host '{host}' failed to render the response") from error


def _render(adapter: 'HookAdapter', host: str, result: Outcome) -> tuple[str, str, int]:
    """Render a result, falling back to the adapter's failure response."""
    try:
        return _render_response(adapter, host, result)
    except _LastResortError:
        try:
            response = adapter.render_failure(
                f"docwrap: host '{visible_controls(host)}' failed to render the response"
            )
            return _response_fields(response.stdout, response.stderr, response.exit_code)
        except Exception as error:
            raise _LastResortError(
                f"host '{host}' failed to render the failure response"
            ) from error


def _select_adapter(host: str, event: str | None) -> 'HookAdapter':
    """Load the selected adapter and enforce its `--event` requirement."""
    try:
        adapter = load_adapter(host)
        requires_event = adapter.requires_event
    except AdapterLoadError as error:
        raise _LastResortError(str(error)) from error
    except Exception as error:
        raise _LastResortError(f"host '{host}' failed to load: {error}") from error
    if requires_event and event is None:
        raise _LastResortError(f"host '{host}' requires --event")
    if not requires_event and event is not None:
        raise _LastResortError(f"host '{host}' does not accept --event")
    return adapter


def _validated_adapter_fields(event: object, command: object) -> tuple['HookEvent', str | None]:
    """Check adapter values at the runtime boundary."""
    if event not in ('pre_tool', 'post_tool'):
        raise TypeError(f'invalid hook event {event!r}')
    if command is not None and not isinstance(command, str):
        raise TypeError(f'invalid hook command {command!r}')
    return event, command


def _parse_input(
    adapter: 'HookAdapter',
    host: str,
    stdin_data: str,
    event: str | None,
) -> 'HookInput | None':
    """Decode the payload and translate it through the adapter."""
    try:
        payload = json.loads(stdin_data)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        hook_input = adapter.parse(cast('dict[str, object]', payload), event)
        if hook_input is None:
            return None
        command_cwd = hook_input.command_cwd
        _check_command_cwd(host, command_cwd)
        hook_event, command = _validated_adapter_fields(hook_input.event, hook_input.command)
        if hook_event == 'post_tool':
            cwd, edited_paths = _post_tool_paths(hook_input.cwd, hook_input.edited_paths)
            return HostInput(hook_event, None, cwd, command_cwd, edited_paths)
        return HostInput(hook_event, command, '', command_cwd)
    except _LastResortError:
        raise
    except Exception as error:
        raise _LastResortError(f"host '{host}' failed to parse the payload: {error}") from error


def allow() -> Outcome:
    """Return the silent allow outcome."""
    return Outcome('allow')


def run_hook(
    stdin_data: str,
    *,
    host: str,
    event: str | None = None,
    allow_on_failure: bool = False,
    incremental: bool | None = None,
) -> int:
    """Run one hook invocation through the selected host adapter.

    Args:
        stdin_data: Raw JSON text the host wrote to standard input.
        host: Adapter name given to `--agent`.
        event: `--event` value for a host whose payload carries no event.
        allow_on_failure: If True, warn instead of denying on infrastructure
            failures after a command is accepted.
        incremental: Override incremental mode. True enables, False disables,
            and None defers to configuration, then to full-file processing when
            unset.

    Returns:
        The exit status the adapter rendered. Invalid JSON and payloads the
        adapter reports as inapplicable exit 0 silently. When no functioning
        adapter exists, one diagnostic line goes to standard error and the
        status is 2.
    """
    try:
        adapter = _select_adapter(host, event)
        hook_input = _parse_input(adapter, host, stdin_data, event)
        if hook_input is None:
            return 0
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            outcome = _run_event(
                hook_input,
                allow_on_failure=allow_on_failure,
                incremental=incremental,
            )
        notes = tuple(captured.getvalue().splitlines())
        stdout, stderr, exit_code = _render(
            adapter, host, Outcome(outcome.decision, outcome.reason, notes)
        )
    except _LastResortError as error:
        return _last_resort(str(error))
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return exit_code


def _run_post_tool(
    edited_paths: Sequence[str],
    cwd: str,
    *,
    incremental: bool | None = None,
) -> Outcome:
    """Load the engine and process edited files after a tool call."""
    try:
        from docwrap._hook import run_post_tool  # noqa: PLC0415
    except ImportError:
        traceback.print_exc()
        print('docwrap: unexpected error during PostToolUse', file=sys.stderr)
        return allow()

    run_post_tool(edited_paths, cwd, incremental=incremental)
    return allow()


def _build_hook_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the hook subcommand."""
    parser = _HookParser(
        prog='docwrap hook',
        usage='%(prog)s --agent NAME [options]',
        description=(
            "Run docwrap as a coding-agent hook, reading the agent's hook payload as JSON "
            'from stdin.'
        ),
    )
    parser.add_argument('-a', '--agent', required=True, metavar='NAME', help='coding agent')
    parser.add_argument(
        '-e',
        '--event',
        metavar='NAME',
        help='hook event, for agents whose payload does not name it (agy)',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-i',
        '--incremental',
        action='store_true',
        help='wrap only docstrings affected by changes since the last commit',
    )
    group.add_argument(
        '--no-incremental',
        action='store_true',
        help='wrap every docstring, even when configuration enables incremental mode',
    )
    parser.add_argument(
        '--allow-on-failure',
        action='store_true',
        help='allow the command when docwrap itself fails; unsupported commits are still denied',
    )
    return parser


def fail(reason: str, *, allow_on_failure: bool) -> Outcome:
    """Handle an infrastructure failure as a deny or a warned allow.

    Args:
        reason: Diagnostic text, escaped for terminal-safe rendering.
        allow_on_failure: Whether to warn and allow instead of denying.

    Returns:
        The decision the selected host adapter will render.
    """
    if allow_on_failure:
        print(f'WARNING: {visible_controls(reason)}', file=sys.stderr)
        return allow()
    return Outcome('deny', visible_controls(reason))


def _run_commit(
    plan: CommitPlan,
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> Outcome:
    """Load the engine and process a supported commit."""
    try:
        from docwrap._hook import run_commit  # noqa: PLC0415
    except ImportError:
        traceback.print_exc()
        return fail('docwrap: unexpected error during wrapping', allow_on_failure=allow_on_failure)

    return run_commit(plan, allow_on_failure=allow_on_failure, incremental=incremental)


def _run_pre_tool_use(
    command: str,
    command_cwd: str | None,
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> Outcome:
    """Classify a pre-tool command before repository access and format a supported commit."""
    try:
        classification = _classify_for_hook(command, command_cwd)
    except _ClassificationError as error:
        return Outcome('deny', str(error))
    if classification.outcome == 'unsupported':
        return Outcome('deny', classification.reason)
    plan = classification.plan
    if not isinstance(plan, CommitPlan):
        return allow()
    return _run_commit(plan, allow_on_failure=allow_on_failure, incremental=incremental)


def _run_event(
    hook_input: 'HookInput',
    *,
    allow_on_failure: bool,
    incremental: bool | None,
) -> Outcome:
    """Dispatch neutral input to the pre-tool or post-tool core."""
    if hook_input.event == 'post_tool':
        return _run_post_tool(hook_input.edited_paths, hook_input.cwd, incremental=incremental)
    if hook_input.command is None:
        return allow()
    return _run_pre_tool_use(
        hook_input.command,
        hook_input.command_cwd,
        allow_on_failure=allow_on_failure,
        incremental=incremental,
    )


class _HookParser(argparse.ArgumentParser):
    """List the registered coding agents when hook help is shown."""

    @override
    def format_help(self, formatter: argparse.HelpFormatter | None = None) -> str:
        from docwrap._hosts import registered_host_names  # noqa: PLC0415

        agents = ', '.join(registered_host_names()) or 'none'
        for action in self._actions:
            if action.dest == 'agent':
                action.help = f'coding agent: {agents}'
        if sys.version_info >= (3, 15):
            return super().format_help(formatter=formatter)
        return super().format_help()


def run_hook_command(argv: list[str]) -> int:
    """Parse hook options and dispatch the selected host adapter.

    Args:
        argv: Hook arguments after the subcommand name.

    Returns:
        The selected adapter's exit status, or zero when input is unavailable.
    """
    hook_parser = _build_hook_parser()
    args = hook_parser.parse_args(argv)

    if sys.stdin.isatty():
        print('docwrap hook: reads the host hook payload as JSON from stdin')
        print('See docs/hook.md for configuration.')
        return 0

    try:
        stdin_data = sys.stdin.read()
    except (ValueError, OSError):
        return 0

    return run_hook(
        stdin_data,
        host=args.agent,
        event=args.event,
        allow_on_failure=args.allow_on_failure,
        incremental=incremental_override(args),
    )
