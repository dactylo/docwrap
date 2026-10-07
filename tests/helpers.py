import contextlib
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol
from unittest import mock

import pytest

import docwrap._core._process as process_module
import docwrap._core._wrap as wrap_module
from _build import _plugin_entry_points
from docwrap._cli import main
from docwrap._config import clear_config_cache
from docwrap._core._codeblock import find_code_block_lines
from docwrap._core._discover import ParsedDocstring, find_docstrings, parse_docstrings
from docwrap._core._parse import Region, is_structural_underline, match_opening_title
from docwrap._stats import LineMeasurement
from docwrap._styles import EntryMatch, Style, resolve_style_override
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    CanonicalRole,
    DocwrapConfig,
    FillStyle,
    MeasurementCategory,
    ReflowMode,
    SphinxFieldRole,
)
from tests.hosts_example.adapter import (
    BrokenBothAdapter,
    BrokenParseAdapter,
    BrokenRenderAdapter,
    EmptyCwdAdapter,
    ExampleAdapter,
    RelativeCwdAdapter,
    RequiresEventAdapter,
    StaleAdapter,
    TypedCwdAdapter,
)

ROOT = Path(__file__).resolve().parents[1]
BUILTIN_ENTRY_POINTS = _plugin_entry_points(ROOT)['docwrap.hook_adapters']
# Hang guard, not a performance assertion. Allow for scheduling on shared CI hosts.
CHILD_TIMEOUT_SECONDS = 300
_COLLECT_AFTER_KILL_SECONDS = 10
_CONTROL_CHARACTERS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')
PROCESS_GROUP_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
_TRANSCRIPT_FIELD_CHARACTERS = 2048
_TRANSCRIPT_HEAD_BYTES = 2048
_TRANSCRIPT_TAIL_BYTES = 6144
_ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT / 'src')])}
CASES = [
    pytest.param(
        textwrap.dedent("""\
            def foo(x, y):
                \"\"\"A very long summary that must be wrapped because it exceeds the line limit.

                Args:
                    x: The x value with a long description.
                    y: The y value.

                Returns:
                    The computed result.
                \"\"\"
                return x + y
        """),
        DEFAULT_LINE_LENGTH,
        id='google-basic',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo(x):
                \"\"\"Summary.

                Parameters
                ----------
                x : int
                    The x value with a very long description that needs wrapping across lines.

                Returns
                -------
                int
                    The result.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='numpy-basic',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo(x):
                \"\"\"Summary.

                :param x: The x value with a very long description that needs wrapping across.
                :returns: The result.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='sphinx-basic',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"Just a summary with no sections at all.\"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='summary-only',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"Summary.

                Description with code:

                    ./tool --flag
                    ./tool --other

                And more text.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='description-with-code-block',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"Summary.

                Example:

                    >>> print("Hello")
                    ... # continuation
                    Hello

                And some closing text.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='doctest-block',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"Summary.

                Returns
                -------
                int or None
                    The result value with a very long description that wraps.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='numpy-compound-type',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"Regex patterns \\\\d+ \\\\d+ \\\\d+ \\\\d+ \\\\d+ \\\\d+ \\\\d+ end.\"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='backslash-heavy',
    ),
    pytest.param(
        textwrap.dedent("""\
            def foo():
                \"\"\"- first item
                - second item
                - third item with a longer description
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='list-only-docstring',
    ),
    pytest.param(
        textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First sentence on its own line.
                Second sentence on its own line.
                Third sentence that is a bit longer.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='sembr-description',
    ),
    pytest.param(
        textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                1. First item with a continuation
                   line that wraps.
                2. Second item.
                \"\"\"
                pass
        """),
        DEFAULT_LINE_LENGTH,
        id='numbered-list-continuation',
    ),
]
EXAMPLE_ENTRY_POINTS = {
    **{
        adapter.name: f'{adapter.__module__}:{adapter.__name__}'
        for adapter in (
            ExampleAdapter,
            EmptyCwdAdapter,
            RelativeCwdAdapter,
            TypedCwdAdapter,
            StaleAdapter,
            RequiresEventAdapter,
            BrokenParseAdapter,
            BrokenRenderAdapter,
            BrokenBothAdapter,
        )
    },
    'missing': 'tests.hosts_example.missing:Adapter',
}
FILL_STYLES: list[FillStyle] = ['balanced', 'greedy', 'smart']
REFLOW_MODES: list[ReflowMode] = ['none', 'cautious', 'full']
SMART_SLACK_VALUES = [0, 4, 8]


class _Outcome(Protocol):
    @property
    def args(self) -> object: ...

    @property
    def returncode(self) -> int | None: ...

    @property
    def stdout(self) -> str | bytes | None: ...

    @property
    def stderr(self) -> str | bytes | None: ...


@dataclass(frozen=True, slots=True)
class BlankBoundaryInvariant:
    after: str
    before: str


@dataclass(frozen=True, slots=True)
class LineIndentInvariant:
    fragment: str
    indent: int
    exact: bool = False


@dataclass(frozen=True, slots=True)
class ReviewRemediationCase:
    blank_boundaries: tuple[BlankBoundaryInvariant, ...]
    line_invariants: tuple[LineIndentInvariant, ...]
    name: str
    source: str


class UnvalidatedConfig(DocwrapConfig):
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _ProjectedLine:
    region_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _ProjectionPlan:
    lines: tuple[_ProjectedLine, ...]
    regions: tuple[Region, ...]
    style: Style | None


@dataclass(frozen=True, slots=True)
class _ProjectionRegion:
    canonical_role: str
    canonical_section: str | None
    end: int
    entry_name: str  # noqa: V107
    entry_type: str  # noqa: V107
    kind: str
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class _ProjectionToken:
    end: int
    kind: str
    start: int


def _canonical_role(
    region: Region,
    style: Style | None,
    entry_match: EntryMatch | None,
    section_role: CanonicalRole | SphinxFieldRole | Literal[''],
) -> str:
    role = region.role or (entry_match.role if entry_match is not None else '')
    if not role and region.kind == 'entry':
        role = section_role
    if style is None:
        return role
    return style.canonicalize_role(role)


def _collapse_whitespace(text: str) -> str:
    return ' '.join(text.split())


def _describe_status(returncode: int | None) -> str:
    if returncode is None:
        return 'not collected'
    if returncode < 0:
        try:
            return f'{returncode} ({signal.Signals(-returncode).name})'
        except ValueError:
            return str(returncode)
    return str(returncode)


def _entry_match_for_projection(region: Region, style: Style | None) -> EntryMatch | None:
    if region.entry_match is not None or style is None or region.kind != 'entry':
        return region.entry_match
    lines = region.text.split('\n')
    match = style.match_entry(lines[0])
    if match is None:
        return None
    descriptions = [match.description] if match.description else []
    descriptions.extend(line.strip() for line in lines[1:] if line.strip())
    return replace(match, description=' '.join(descriptions))


def _escape_transcript_line(line: str) -> str:
    line = _CONTROL_CHARACTERS.sub(lambda match: f'\\x{ord(match.group()):02x}', line)
    line = line.replace('##[', '#\\#[')
    body = line.lstrip()
    if body.startswith('::'):
        line = line[: len(line) - len(body)] + '\\' + body
    return line


def _exit_status(error: SystemExit) -> int:
    code = error.code
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def _opaque_spans(regions: tuple[Region, ...]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(regions):
        if not regions[index].preserve_verbatim:
            index += 1
            continue
        end = index + 1
        while end < len(regions) and regions[end].preserve_verbatim:
            end += 1
        spans.append((index, end))
        index = end
    return tuple(spans)


def _projection_regions(
    regions: tuple[Region, ...],
    style: Style | None,
    spans: dict[int, tuple[int, int]],
) -> dict[int, _ProjectionRegion]:
    result: dict[int, _ProjectionRegion] = {}
    section_role: CanonicalRole | SphinxFieldRole | Literal[''] = ''
    for index, region in enumerate(regions):
        if region.kind in {'summary', 'title', 'description'}:
            section_role = ''
        if region.kind == 'section_header' and region.section_name is not None:
            if style is None:
                section_role = ''
            else:
                section_role = style.section_to_canonical.get(
                    region.section_name,
                    '',
                )

        entry_match = _entry_match_for_projection(region, style)
        canonical_section = (
            (region.section_name if style is None else section_role)
            if region.kind == 'section_header'
            else None
        )
        if entry_match is not None:
            text = _collapse_whitespace(entry_match.description)
        elif region.kind in {'description', 'entry', 'list_item', 'prose', 'summary'}:
            text = (
                region.text
                if is_structural_underline(region.text)
                else _collapse_whitespace(region.text)
            )
        else:
            text = region.text
        start, end = spans[index]
        result[index] = _ProjectionRegion(
            canonical_role=_canonical_role(region, style, entry_match, section_role),
            canonical_section=canonical_section,
            end=end,
            entry_name=entry_match.name if entry_match is not None else '',
            entry_type=entry_match.type_ann if entry_match is not None else '',
            kind=region.kind,
            start=start,
            text=text,
        )
    return result


def _rendered_spans(
    lines: tuple[_ProjectedLine, ...],
    region_count: int,
) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    for line_index, line in enumerate(lines):
        assert 0 <= line.region_index < region_count
        existing = spans.get(line.region_index)
        if existing is None:
            spans[line.region_index] = (line_index, line_index + 1)
            continue
        start, end = existing
        assert end == line_index, f'region {line.region_index} emitted non-contiguously'
        spans[line.region_index] = (start, line_index + 1)
    assert set(spans) == set(range(region_count)), 'every planned region must emit physical lines'
    return spans


def _target_config(config: DocwrapConfig, style: Style | None) -> DocwrapConfig:
    return replace(
        config,
        convert=None,
        strip_types=False,
        style=style.name if style is not None else None,
    )


def _terminate_tree[T: (str, bytes)](process: subprocess.Popen[T]) -> None:
    if sys.platform == 'win32':
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                capture_output=True,
                timeout=_COLLECT_AFTER_KILL_SECONDS,
                check=False,
            )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        process.kill()


def _transcript_field(label: str, value: object) -> str:
    field = repr(value)
    if len(field) > _TRANSCRIPT_FIELD_CHARACTERS:
        omitted = len(field) - _TRANSCRIPT_FIELD_CHARACTERS
        field = f'{field[:_TRANSCRIPT_FIELD_CHARACTERS]} [... {omitted} characters omitted ...]'
    return f'{label}: {_escape_transcript_line(field)}'


def _transcript_stream(label: str, data: str | bytes | None) -> list[str]:
    if data is None:
        return [f'{label}: <not collected>']
    raw = data if isinstance(data, bytes) else data.encode('utf-8', 'backslashreplace')
    if not raw:
        return [f'{label}: <empty>']
    if len(raw) > _TRANSCRIPT_HEAD_BYTES + _TRANSCRIPT_TAIL_BYTES:
        omitted = len(raw) - _TRANSCRIPT_HEAD_BYTES - _TRANSCRIPT_TAIL_BYTES
        field = (
            raw[:_TRANSCRIPT_HEAD_BYTES].decode('utf-8', 'backslashreplace')
            + f'\n[... {omitted} bytes omitted ...]\n'
            + raw[-_TRANSCRIPT_TAIL_BYTES:].decode('utf-8', 'backslashreplace')
        )
    else:
        field = raw.decode('utf-8', 'backslashreplace')
    body = field.replace('\r\n', '\n').removesuffix('\n')
    return [f'{label}:', *(f'  | {_escape_transcript_line(line)}' for line in body.split('\n'))]


def _verify_opaque_spans(
    plan: _ProjectionPlan,
    scanner_lines: set[int],
    spans: dict[int, tuple[int, int]],
    opaque_spans: tuple[tuple[int, int], ...],
) -> None:
    for start_region, end_region in opaque_spans:
        expected_lines: list[str] = []
        first_line = spans[start_region][0]
        last_line = spans[end_region - 1][1]
        for region_index in range(start_region, end_region):
            region = plan.regions[region_index]
            assert region.original_lines is not None
            expected_lines.extend(region.original_lines)
            emitted_start, emitted_end = spans[region_index]
            emitted = tuple(line.text for line in plan.lines[emitted_start:emitted_end])
            assert emitted == region.original_lines
            expected_protected = region.source_scanner_protected
            for line_index in range(emitted_start, emitted_end):
                assert (line_index in scanner_lines) is expected_protected
        assert tuple(line.text for line in plan.lines[first_line:last_line]) == tuple(
            expected_lines
        )


def _verify_projection(
    plan: _ProjectionPlan,
    reparsed: ParsedDocstring,
) -> None:
    assert reparsed.content == '\n'.join(line.text for line in plan.lines)
    spans = _rendered_spans(plan.lines, len(plan.regions))
    opaque_spans = _opaque_spans(plan.regions)
    opaque_by_region = {
        region_index: (start, end)
        for start, end in opaque_spans
        for region_index in range(start, end)
    }

    content_lines = [line.text for line in plan.lines]
    title_line_count = match_opening_title(content_lines, plan.style)
    scanner_lines = find_code_block_lines(
        content_lines,
        start=title_line_count or 0,
    )
    assert scanner_lines == set(reparsed.code_block_lines)
    for line_index, rendered_line in enumerate(plan.lines):
        region = plan.regions[rendered_line.region_index]
        assert (line_index in scanner_lines) is region.source_scanner_protected
    _verify_opaque_spans(plan, scanner_lines, spans, opaque_spans)

    expected_sequence: list[_ProjectionToken] = []
    index = 0
    while index < len(plan.regions):
        opaque = opaque_by_region.get(index)
        if opaque is not None:
            expected_sequence.append(
                _ProjectionToken(end=opaque[1], kind='opaque', start=opaque[0])
            )
            index = opaque[1]
            continue
        expected_sequence.append(_ProjectionToken(end=index + 1, kind='region', start=index))
        index += 1

    actual_sequence: list[_ProjectionToken] = []
    actual_regions: dict[int, Region] = {}
    actual_spans: dict[int, tuple[int, int]] = {}
    seen_opaque: set[tuple[int, int]] = set()
    for actual_region in reparsed.regions:
        assert actual_region.source_start is not None
        start = actual_region.source_start
        end = start + actual_region.source_line_count
        identities = {line.region_index for line in plan.lines[start:end]}
        assert identities, 'reparsed regions must cover emitted physical lines'
        opaque_matches = {opaque_by_region[i] for i in identities if i in opaque_by_region}
        if opaque_matches:
            assert len(opaque_matches) == 1
            opaque = next(iter(opaque_matches))
            assert identities <= set(range(opaque[0], opaque[1]))
            if opaque not in seen_opaque:
                actual_sequence.append(
                    _ProjectionToken(end=opaque[1], kind='opaque', start=opaque[0])
                )
                seen_opaque.add(opaque)
            continue

        assert len(identities) == 1, (
            f'reparsed region crossed planned identities: {sorted(identities)}'
        )
        region_index = next(iter(identities))
        actual_sequence.append(
            _ProjectionToken(end=region_index + 1, kind='region', start=region_index)
        )
        assert region_index not in actual_regions, (
            f'planned region {region_index} split across reparsed regions'
        )
        actual_regions[region_index] = actual_region
        actual_spans[region_index] = (start, end)

    assert actual_sequence == expected_sequence
    planned_regions = _projection_regions(plan.regions, plan.style, spans)
    projected_actual = _projection_regions(
        tuple(actual_regions[index] for index in sorted(actual_regions)),
        plan.style,
        {
            new_index: actual_spans[region_index]
            for new_index, region_index in enumerate(sorted(actual_regions))
        },
    )
    for actual_index, region_index in enumerate(sorted(actual_regions)):
        assert projected_actual[actual_index] == planned_regions[region_index]


def assert_region_projection(
    source: str, config: DocwrapConfig, *, emitted_source: str | None = None
) -> None:
    analysis = process_module.analyze_source(source, config)
    assert not analysis.parse_error
    assert analysis.skips == ()
    assert len(analysis.parsed) == len(analysis.docstrings)

    render = wrap_module._render_with_deltas
    render_options = wrap_module._RenderOptions
    convert = process_module._convert_parsed_regions

    plans: list[_ProjectionPlan] = []
    for parsed in analysis.parsed:
        if config.convert is not None and parsed.detected_name == config.convert:
            wrap_style = parsed.style
        else:
            wrap_style = resolve_style_override(config.style) or parsed.style
        regions, target_style, _, _ = convert(parsed, config, wrap_style)
        rendered = render(
            regions,
            config,
            render_options(
                base_indent=parsed.base_indent,
                style=target_style,
                quote_overhead=wrap_module.compute_quote_overhead(parsed, config),
                col_offset=parsed.col_offset,
            ),
        )
        projected_lines = tuple(
            _ProjectedLine(region_index=line.region_index, text=line.text) for line in rendered
        )
        plans.append(
            _ProjectionPlan(
                lines=projected_lines,
                regions=tuple(regions),
                style=target_style,
            )
        )

    if emitted_source is None:
        emitted_source = process_module.wrap_source_report(source, config=config).source
    emitted_docstrings = find_docstrings(emitted_source)
    assert len(emitted_docstrings) == len(plans)

    for info, plan in zip(emitted_docstrings, plans, strict=True):
        reparsed = parse_docstrings(
            emitted_source, [info], _target_config(config, plan.style), skips=[]
        )
        assert len(reparsed) == 1
        _verify_projection(plan, reparsed[0])


def decode_text(data: bytes) -> str:
    return data.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


def hook_event(
    event_name: str,
    tool_name: str,
    tool_input: dict[str, object],
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'hook_event_name': event_name,
        'tool_name': tool_name,
        'tool_input': tool_input,
    }
    payload.update(extra)
    return payload


def process_transcript(
    result: _Outcome,
    *,
    cwd: Path | str | None = None,
    note: str | None = None,
) -> str:
    lines = [_transcript_field('command', result.args)]
    if cwd is not None:
        lines.append(_transcript_field('cwd', str(cwd)))
    if note is not None:
        lines.append(f'note: {_escape_transcript_line(note)}')
    lines.append(f'status: {_describe_status(result.returncode)}')
    lines.extend(_transcript_stream('stdout', result.stdout))
    lines.extend(_transcript_stream('stderr', result.stderr))
    return '\n'.join(lines)


def _kill_and_describe[T: (str, bytes)](
    process: subprocess.Popen[T],
    *,
    cwd: Path | str | None,
    reason: str,
) -> str:
    _terminate_tree(process)
    try:
        stdout, stderr = process.communicate(timeout=_COLLECT_AFTER_KILL_SECONDS)
    except subprocess.TimeoutExpired:
        stdout = stderr = None
        reason = f'{reason}; a descendant outside the process group still holds the output pipes'
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_COLLECT_AFTER_KILL_SECONDS)
    killed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    return process_transcript(killed, cwd=cwd, note=reason)


def finish_process[T: (str, bytes)](
    process: subprocess.Popen[T],
    *,
    stdin: T | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[T]:
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            _kill_and_describe(process, cwd=cwd, reason=f'timed out after {timeout:g} s')
        ) from None
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def invoke_cli(
    argv: Sequence[str],
    *,
    stdin: str = '',
    cwd: Path | str | None = None,
) -> tuple[int, str, str]:
    stdout_buffer = io.BytesIO()
    stderr_buffer = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_buffer, encoding='utf-8', newline='')
    stderr = io.TextIOWrapper(
        stderr_buffer, encoding='utf-8', errors='backslashreplace', newline=''
    )
    stdin_stream = io.TextIOWrapper(io.BytesIO(stdin.encode('utf-8')), encoding='utf-8')
    clear_config_cache()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {'COLUMNS': '80'}))
        stack.enter_context(contextlib.redirect_stdout(stdout))
        stack.enter_context(contextlib.redirect_stderr(stderr))
        stack.enter_context(mock.patch.object(sys, 'stdin', stdin_stream))
        if cwd is not None:
            stack.enter_context(contextlib.chdir(cwd))
        try:
            status = main(argv)
        except SystemExit as error:
            status = _exit_status(error)
        stdout.flush()
        stderr.flush()
    return (
        status,
        stdout_buffer.getvalue().decode('utf-8'),
        stderr_buffer.getvalue().decode('utf-8'),
    )


def json_stdout(result: subprocess.CompletedProcess[str]) -> object:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(f'stdout is not JSON\n{process_transcript(result)}') from None


def register_distribution(root: Path, name: str, entry_points: Mapping[str, str]) -> None:
    dist = root / f'{name.replace("-", "_")}-0.0.0.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.0\n', encoding='utf-8'
    )
    lines = [
        '[docwrap.hook_adapters]',
        *(f'{host} = {target}' for host, target in entry_points.items()),
    ]
    (dist / 'entry_points.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_cli(  # noqa: PLR0913
    argv: Sequence[str],
    *,
    expect: int | None = None,
    stdin: str = '',
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [sys.executable, '-m', 'docwrap', *argv],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        env=dict(_ENV if env is None else env),
        cwd=cwd,
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    result = finish_process(process, stdin=stdin, timeout=timeout, cwd=cwd)
    if expect is not None and result.returncode != expect:
        raise AssertionError(f'expected status {expect}\n{process_transcript(result, cwd=cwd)}')
    return result


def run_closing_after_first_line(
    argv: Sequence[str],
    *,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        [sys.executable, '-m', 'docwrap', *argv],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_ENV,
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    stdout = process.stdout
    assert stdout is not None
    descriptor = stdout.fileno()

    def read_first_line() -> None:
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk or b'\n' in chunk:
                return

    reader = threading.Thread(target=read_first_line, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        raise AssertionError(
            _kill_and_describe(process, cwd=None, reason=f'no stdout line within {timeout:g} s')
        )
    stdout.close()
    process.stdout = None  # communicate() must not read the closed pipe on Windows
    return finish_process(process, timeout=timeout)


def run_setup(  # noqa: PLR0913
    command: Sequence[str | bytes],
    *,
    check: bool = True,
    cwd: Path | str | bytes | None = None,
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=cwd,
        start_new_session=True,
        creationflags=PROCESS_GROUP_FLAGS,
    )
    display_cwd = os.fsdecode(cwd) if isinstance(cwd, bytes) else cwd
    result = finish_process(process, stdin=stdin, timeout=timeout, cwd=display_cwd)
    if check and result.returncode != 0:
        raise AssertionError(
            f'setup command failed\n{process_transcript(result, cwd=display_cwd)}'
        )
    return result


def run_with_closed_output(
    argv: Sequence[str],
    *,
    closed: Literal['stdout', 'stderr'] = 'stdout',
    stdin: bytes | None = None,
    timeout: float = CHILD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # every write to the closed stream happens after its reader closed
    try:
        process = subprocess.Popen(
            [sys.executable, '-m', 'docwrap', *argv],
            stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
            stdout=write_fd if closed == 'stdout' else subprocess.PIPE,
            stderr=write_fd if closed == 'stderr' else subprocess.PIPE,
            env=_ENV,
            start_new_session=True,
            creationflags=PROCESS_GROUP_FLAGS,
        )
    finally:
        os.close(write_fd)
    return finish_process(process, stdin=stdin, timeout=timeout)


def git_commit_all(repo: Path, message: str = 'commit') -> None:
    run_setup(
        ['git', 'add', '.'],
        cwd=str(repo),
    )
    run_setup(
        ['git', '-c', 'user.name=Test', '-c', 'user.email=test@test.com', 'commit', '-m', message],
        cwd=str(repo),
    )


def make_measurement(
    length: int,
    *,
    category: MeasurementCategory = 'prose',
    file_path: str = 'f.py',
    line_number: int = 1,
    text: str | None = None,
) -> LineMeasurement:
    return LineMeasurement(
        length=length,
        category=category,
        file_path=file_path,
        line_number=line_number,
        text=text if text is not None else 'x' * length,
        wrappable=category in ('prose', 'args'),
    )


def shell_path(path: Path) -> str:
    return shlex.quote(path.as_posix())


REVIEW_REMEDIATION_CASES: tuple[ReviewRemediationCase, ...] = (
    ReviewRemediationCase(
        name='difflib.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Returns HTML file of side by side comparison with change highlights
                    fromlines -- list of "from" lines
                        which shows full differences).
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='fromlines -- list of "from" lines', indent=0),
            LineIndentInvariant(fragment='which shows full differences).', indent=4),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='email/policy.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"+
                    Header folding is controlled by the refold_source policy setting.  A
                    value is considered to be a 'source value' if and only if it does not

                    Source values are split into lines using splitlines.  If the value is
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='+', indent=0, exact=True),
            LineIndentInvariant(
                fragment='Header folding is controlled by the refold_source policy setting.',
                indent=0,
            ),
            LineIndentInvariant(fragment='Source values are split into lines', indent=0),
        ),
        blank_boundaries=(
            BlankBoundaryInvariant(
                before='does not',
                after='Source values are split into lines',
            ),
        ),
    ),
    ReviewRemediationCase(
        name='encodings/cp720.py',
        source=textwrap.dedent("""\
            \"\"\"Python Character Mapping Codec cp720 generated on Windows:
            Vista 6.0.6002 SP2 Multiprocessor Free with the command:
              python Tools/unicode/genwincodec.py 720
            \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='python Tools/unicode/genwincodec.py 720', indent=2),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='filecmp.py',
        source=textwrap.dedent("""\
            class Sample:
                \"\"\"A class that manages the comparison of 2 directories.
                 left_list, right_list: The files in dir1 and dir2,
                    filtered by hide and ignore.
                 \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='left_list, right_list:', indent=1),
            LineIndentInvariant(fragment='filtered by hide and ignore.', indent=4),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='ftplib.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Retrieve data in binary mode.  A new port is created for you.

                    Args:
                      callback: A single parameter callable to be called on each
                                block of data read.
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='callback: A single parameter', indent=2),
            LineIndentInvariant(fragment='block of data read.', indent=12),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='getpass.py',
        source=textwrap.dedent("""\
            class Sample:
                \"\"\"Prompt for a password, with echo turned off.

                Args:
                  stream: A writable file object to display the prompt.  Defaults to
                          the tty.  If no tty is available defaults to sys.stderr.
                \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='stream: A writable file object', indent=2),
            LineIndentInvariant(fragment='the tty.  If no tty is available', indent=10),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='http/cookies.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Load cookies from a string (presumably HTTP_COOKIE) or
                    from a dictionary.  Loading cookies from a dictionary 'd'
                        map(Cookie.__setitem__, d.keys(), d.values())
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(
                fragment='map(Cookie.__setitem__, d.keys(), d.values())',
                indent=4,
            ),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='imaplib.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Setup a stream connection.
                    This connection will be used by the routines:
                        read, readline, send, shutdown.
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='read, readline, send, shutdown.', indent=4),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='pty.py',
        source=textwrap.dedent("""\
            class Sample:
                \"\"\"Parent copy loop.
                Copies
                        pty master -> standard output   (master_read)
                        standard input -> pty master    (stdin_read)\"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(
                fragment='pty master -> standard output   (master_read)',
                indent=8,
            ),
            LineIndentInvariant(
                fragment='standard input -> pty master    (stdin_read)',
                indent=8,
            ),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='random.py',
        source=textwrap.dedent("""\
            class Sample:
                \"\"\"Alternate random number generator using sources provided
                by the operating system (such as /dev/urandom on Unix or
                 Not available on all systems (see os.urandom() for details).
                \"\"\"
        """),
        line_invariants=(LineIndentInvariant(fragment='Not available on all systems', indent=1),),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='smtplib.py',
        source=textwrap.dedent("""\
            class Sample:
                \"\"\"This class manages a connection to an SMTP or ESMTP server.
                SMTP Objects:
                    SMTP objects have the following attributes:
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='SMTP Objects:', indent=0),
            LineIndentInvariant(fragment='SMTP objects have the following attributes:', indent=4),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='turtle.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Adds a turtle shape to TurtleScreen's shapelist.
                    (2) name is an arbitrary string and shape is a tuple
                        of pairs of coordinates. Installs the corresponding
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='(2) name is an arbitrary string', indent=0),
            LineIndentInvariant(fragment='of pairs of coordinates.', indent=4),
        ),
        blank_boundaries=(),
    ),
    ReviewRemediationCase(
        name='unittest/case.py',
        source=textwrap.dedent("""\
            class Sample:
                def method(self):
                    \"\"\"Asserts that two iterables have the same elements, the same number of
                    times, without regard to order.
                        self.assertEqual(Counter(list(first)),
                    \"\"\"
        """),
        line_invariants=(
            LineIndentInvariant(fragment='self.assertEqual(Counter(list(first)),', indent=4),
        ),
        blank_boundaries=(),
    ),
)
