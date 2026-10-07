"""Public host adapter contract for `docwrap hook`.

A host adapter translates one agent host's hook protocol to docwrap's neutral
hook input and translates docwrap's neutral result back to the host's response.
Adapters are discovered through the `docwrap.hook_adapters` entry-point group;
each entry point names a class whose instances satisfy `HookAdapter`. The
`HookInput`, `HookResult`, `HookResponse`, and `HookAdapter` types are
structural protocols. `HostInput` and `HostResponse` are optional concrete
records for adapters that do not need their own record types.

Pre-tool input carries two directories. `cwd` is event context consumed only by
post-tool handling. `command_cwd` is the directory the command executes in, or
`None` when the host contract does not establish it; the core then denies a
committing command unless it names an absolute route.

docwrap selects the adapter named by `docwrap hook --agent NAME`, refuses an
adapter whose `api_version` differs from `API_VERSION`, and never infers the
host from the payload. Classification, Git operations, and the allow or deny
decision never depend on the adapter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

API_VERSION = 1

type HookDecisionName = Literal['allow', 'deny']
type HookEvent = Literal['pre_tool', 'post_tool']


@dataclass(frozen=True, slots=True)
class HostInput:
    """Immutable neutral input produced by a host adapter.

    Attributes:
        event: Whether the payload describes `pre_tool` or `post_tool`.
        command: Shell command for a pre-tool event, or `None`.
        cwd: Event directory used to resolve relative post-tool edited paths.
            It does not establish the pre-tool command's execution directory.
        command_cwd: Absolute directory in which a pre-tool command executes,
            or `None` when the host contract does not establish it. Do not
            substitute a process, session, or repository directory. Post-tool
            input uses `None`.
        edited_paths: Paths reported as written by a post-tool event. Relative
            paths are resolved against `cwd`.
    """

    event: HookEvent
    command: str | None
    cwd: str
    command_cwd: str | None
    edited_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HostResponse:
    """Immutable output and exit status produced by a host adapter.

    Attributes:
        stdout: Exact text to write to standard output, including any required
            trailing newline.
        stderr: Exact text to write to standard error, including any required
            trailing newline.
        exit_code: Process exit status returned by the hook command.
    """

    stdout: str
    stderr: str
    exit_code: int


class HookInput(Protocol):
    """Neutral hook input an adapter produces from a host payload."""

    @property
    def event(self) -> HookEvent:
        """Return whether the payload precedes or follows a tool call."""
        ...

    @property
    def command(self) -> str | None:
        """Return the shell command of a pre-tool payload, or `None`."""
        ...

    @property
    def cwd(self) -> str:
        """Return the event directory the host reports; consumed only after a tool call."""
        ...

    @property
    def edited_paths(self) -> Sequence[str]:
        """Return the paths a post-tool payload reports as written."""
        ...

    @property
    def command_cwd(self) -> str | None:
        """Return the absolute directory a pre-tool command executes in, or `None`.

        `None` means the host contract does not establish that directory. It
        never means the process, session, or repository directory. Post-tool
        input always reports `None`.
        """
        ...


class HookResult(Protocol):
    """Neutral result docwrap passes to an adapter for rendering."""

    @property
    def decision(self) -> HookDecisionName:
        """Return whether the host should allow or deny the tool call."""
        ...

    @property
    def reason(self) -> str | None:
        """Return a deny explanation or allow receipt; `None` means silent allow."""
        ...

    @property
    def notes(self) -> Sequence[str]:
        """Return informational lines, such as safety skips, for the host's diagnostics."""
        ...


class HookResponse(Protocol):
    """Host response an adapter renders from a neutral result."""

    @property
    def stdout(self) -> str:
        """Return the exact text to write to standard output."""
        ...

    @property
    def stderr(self) -> str:
        """Return the exact text to write to standard error."""
        ...

    @property
    def exit_code(self) -> int:
        """Return the process exit status."""
        ...


class HookAdapter(Protocol):
    """Translator between one host's hook protocol and docwrap's neutral types."""

    @property
    def name(self) -> str:
        """Return the host name selected by `--agent`."""
        ...

    @property
    def api_version(self) -> int:
        """Return the `API_VERSION` the adapter implements."""
        ...

    def parse(self, payload: Mapping[str, object], event: str | None) -> HookInput | None:
        """Translate a host payload to neutral input.

        Args:
            payload: Decoded JSON object the host wrote to standard input.
            event: The `--event` value when the adapter requires one, else
                `None`.

        Returns:
            Neutral input, or `None` when the payload describes nothing docwrap
            acts on, which docwrap treats as a silent allow.

        Raises:
            ValueError: If the payload or event is not one the host produces.
        """
        ...

    def render(self, result: HookResult) -> HookResponse:
        """Translate a neutral result to the host's response.

        Args:
            result: Decision, reason, and notes from docwrap.

        Returns:
            The host response carrying the decision.
        """
        ...

    def render_failure(self, reason: str) -> HookResponse:
        """Render the host's fail-closed response for a reason.

        Args:
            reason: Reason for the failure.

        Returns:
            The host response that blocks the tool call where the host's
            protocol allows blocking.
        """
        ...

    @property
    def requires_event(self) -> bool:
        """Return whether `--event` is required because the payload carries no event."""
        ...


def absolute_directory(fields: Mapping[str, object], key: str) -> str | None:
    """Return the absolute directory string at `key`, or `None`.

    Args:
        fields: Payload object that may carry the host's invocation directory.
        key: Field name the host contract defines as that directory.

    Returns:
        The field value when it is a non-empty absolute path string. Any other
        value, including a relative path, is not invocation evidence and yields
        `None`.
    """
    value = fields.get(key)
    if isinstance(value, str) and value and Path(value).is_absolute():
        return value
    return None


def notes_text(result: HookResult) -> str:
    """Render a result's notes as stderr lines."""
    return ''.join(f'{note}\n' for note in result.notes)


def object_field(payload: Mapping[str, object], key: str) -> dict[str, object]:
    """Return a nested object field, or an empty object when absent or not an object."""
    value = payload.get(key)
    if isinstance(value, dict):
        return cast('dict[str, object]', value)
    return {}


def payload_cwd(payload: Mapping[str, object], key: str = 'cwd') -> str:
    """Return the event directory for post-tool handling, falling back to the process directory.

    Never use this value as a pre-tool `command_cwd`; use `absolute_directory`
    instead.
    """
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    return str(Path.cwd())
