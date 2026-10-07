# pyright: reportPrivateUsage=false

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from docwrap._hook_front import _build_hook_parser, run_hook
from docwrap._hosts import load_adapter
from docwrap.hooks import (
    API_VERSION,
    HookAdapter,
    HookResponse,
    HookResult,
    HostInput,
    HostResponse,
)
from plugins.claude._docwrap._adapter import ClaudeAdapter
from tests.helpers import decode_text, git_commit_all, run_setup

ROOT = Path(__file__).resolve().parents[3]
DENY_JSON = {
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': 'docwrap: reason',
    }
}


@dataclass(frozen=True)
class _Result:
    decision: Literal['allow', 'deny']
    reason: str | None = None
    notes: tuple[str, ...] = ()


def _decision(stdout: str) -> str | None:
    if not stdout:
        return None
    data = json.loads(stdout)
    if 'systemMessage' in data:
        return None
    if 'hookSpecificOutput' in data:
        return str(data['hookSpecificOutput']['permissionDecision'])
    return str(data['example']['decision'])


def _payload(event: str, tool: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {'hook_event_name': event, 'tool_name': tool, 'tool_input': tool_input, 'cwd': '/r'}


def _result(decision: str, reason: str | None = None, notes: tuple[str, ...] = ()) -> HookResult:
    return _Result('deny' if decision == 'deny' else 'allow', reason, notes)


class TestClaudeAdapter:
    def test_contract(self) -> None:
        adapter: HookAdapter = ClaudeAdapter()
        assert (adapter.name, adapter.api_version, adapter.requires_event) == (
            'claude',
            API_VERSION,
            False,
        )

    @pytest.mark.usefixtures('example_hosts')
    @pytest.mark.parametrize('host', ['claude', 'example'])
    @pytest.mark.parametrize(
        ('command', 'outcome'),
        [
            pytest.param('ls', 'unrelated', id='unrelated'),
            pytest.param('git commit --future-option -m x', 'denied', id='denied'),
            pytest.param('git commit -m x', 'wrapped', id='supported'),
        ],
    )
    def test_hook_decisions_preserve_adapter_independence(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        host: str,
        command: str,
        outcome: Literal['unrelated', 'denied', 'wrapped'],
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=tmp_path)
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 40\nfill-style = "greedy"\n', encoding='utf-8'
        )
        target = tmp_path / 'sample.py'
        target.write_text('pass\n', encoding='utf-8')
        git_commit_all(tmp_path)
        source = (
            'def f():\n    """Summary.\n\n'
            '    one two three four five six seven eight nine ten eleven twelve\n'
            '    """\n'
        )
        target.write_text(source, encoding='utf-8')
        run_setup(['git', 'add', 'sample.py'], cwd=tmp_path)
        payload = json.dumps(
            _payload('PreToolUse', 'Bash', {'command': command}) | {'cwd': str(tmp_path)}
        )

        status = run_hook(payload, host=host)

        captured = capsys.readouterr()
        expected_decision = 'deny' if outcome == 'denied' else None
        assert (status, _decision(captured.out)) == (0, expected_decision)
        if expected_decision == 'deny':
            assert 'unsupported commit command' in captured.err
        else:
            assert captured.err == ''
        expected = (
            'def f():\n    """Summary.\n\n'
            '    one two three four five six seven\n'
            '    eight nine ten eleven twelve\n'
            '    """\n'
            if outcome == 'wrapped'
            else source
        )
        staged = run_setup(['git', 'show', ':sample.py'], cwd=tmp_path)
        assert decode_text(staged.stdout) == expected
        assert target.read_text(encoding='utf-8') == expected

    def test_parse_missing_cwd_uses_process_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        payload: dict[str, object] = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {},
        }
        parsed = ClaudeAdapter().parse(payload, None)
        assert parsed is not None
        assert parsed.cwd == str(tmp_path)

    def test_parse_other_event_is_inapplicable(self) -> None:
        assert ClaudeAdapter().parse({'hook_event_name': 'Notification'}, None) is None

    @pytest.mark.parametrize('tool', ['Edit', 'Write'])
    def test_parse_post_tool_edit(self, tool: str) -> None:
        payload = _payload('PostToolUse', tool, {'file_path': 'a.py'})
        assert ClaudeAdapter().parse(payload, None) == HostInput(
            'post_tool', None, '/r', None, ('a.py',)
        )

    def test_parse_pre_tool_bash(self, tmp_path: Path) -> None:
        payload = _payload('PreToolUse', 'Bash', {'command': 'git commit -m x'})
        payload['cwd'] = str(tmp_path)
        assert ClaudeAdapter().parse(payload, None) == HostInput(
            'pre_tool', 'git commit -m x', str(tmp_path), str(tmp_path)
        )

    @pytest.mark.parametrize('cwd', ['rel', '', 123, None])
    def test_parse_pre_tool_non_absolute_cwd_is_unknown(
        self, cwd: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        payload = _payload('PreToolUse', 'Bash', {'command': 'ls'})
        payload['cwd'] = cwd
        parsed = ClaudeAdapter().parse(payload, None)
        assert parsed is not None
        assert parsed.command_cwd is None
        assert parsed.cwd == (cwd if isinstance(cwd, str) and cwd else str(tmp_path))

    def test_parse_pre_tool_other_tool_has_no_command(self) -> None:
        payload = _payload('PreToolUse', 'Read', {'file_path': 'a.py'})
        assert ClaudeAdapter().parse(payload, None) == HostInput('pre_tool', None, '/r', None)

    def test_render_allow_feedback_is_one_system_message(self) -> None:
        response = ClaudeAdapter().render(_result('allow', 'first\nsecond', ('diagnostic',)))
        assert json.loads(response.stdout) == {'systemMessage': 'first\nsecond'}
        assert response.stderr == 'diagnostic\n'
        assert response.exit_code == 0

    def test_render_allow_is_silent_with_notes_on_stderr(self) -> None:
        response: HookResponse = ClaudeAdapter().render(
            _result('allow', notes=('docwrap: a.py:1: f skipped',))
        )
        assert response == HostResponse('', 'docwrap: a.py:1: f skipped\n', 0)

    def test_render_deny_is_json_and_stderr_reason(self) -> None:
        response = ClaudeAdapter().render(_result('deny', 'docwrap: reason', ('note',)))
        assert json.loads(response.stdout) == DENY_JSON
        assert response.stderr == 'note\ndocwrap: reason\n'
        assert response.exit_code == 0

    def test_render_failure(self) -> None:
        response = ClaudeAdapter().render_failure('docwrap: reason')
        assert json.loads(response.stdout) == DENY_JSON
        assert response.stderr == 'docwrap: reason\n'
        assert response.exit_code == 0

    @pytest.mark.parametrize(
        ('reason', 'notes', 'context'),
        [
            (None, ('Cannot preserve layout.',), 'Cannot preserve layout.'),
            ('Cannot wrap a.py: invalid syntax.', (), 'Cannot wrap a.py: invalid syntax.'),
            ('first\nsecond', ('first', 'diagnostic'), 'first\ndiagnostic\nsecond'),
        ],
    )
    def test_render_post_tool_context(
        self, reason: str | None, notes: tuple[str, ...], context: str
    ) -> None:
        adapter = ClaudeAdapter()
        assert adapter.parse(_payload('PostToolUse', 'Edit', {'file_path': 'a.py'}), None)
        response = adapter.render(_result('allow', reason, notes))
        assert json.loads(response.stdout) == {
            'hookSpecificOutput': {
                'hookEventName': 'PostToolUse',
                'additionalContext': context,
            }
        }
        assert response.stderr == ''.join(f'{note}\n' for note in notes)
        assert response.exit_code == 0

    @pytest.mark.parametrize('event', ['PreToolUse', 'Notification', None])
    def test_render_resets_post_tool_context(self, event: str | None) -> None:
        adapter = ClaudeAdapter()
        adapter.parse(_payload('PostToolUse', 'Edit', {'file_path': 'a.py'}), None)
        assert adapter.render(_result('allow')) == HostResponse('', '', 0)
        adapter.parse({'hook_event_name': event}, None)
        response = adapter.render(_result('deny', 'docwrap: reason'))
        assert json.loads(response.stdout) == DENY_JSON
        receipt = adapter.render(_result('allow', 'staged rewrite receipt'))
        assert json.loads(receipt.stdout) == {'systemMessage': 'staged rewrite receipt'}


@pytest.mark.usefixtures('registered_hosts')
def test_native_manifest_command_loads_this_adapter() -> None:
    argv = shlex.split('docwrap hook --agent claude')
    args = _build_hook_parser().parse_args(argv[2:])
    adapter = load_adapter(args.agent)
    assert adapter.name == 'claude'
    assert args.event is None
    assert not adapter.requires_event


def test_native_manifest_is_unchanged() -> None:
    manifest = ROOT / 'plugins' / 'claude' / 'hooks' / 'hooks.json'
    assert json.loads(manifest.read_text(encoding='utf-8')) == {
        'hooks': {
            'PreToolUse': [
                {
                    'matcher': '^Bash$',
                    'hooks': [{'type': 'command', 'command': 'docwrap hook --agent claude'}],
                }
            ]
        }
    }


def test_native_plugin_identity_and_marketplace_source() -> None:
    plugin = json.loads(
        (ROOT / 'plugins' / 'claude' / '.claude-plugin' / 'plugin.json').read_text(
            encoding='utf-8'
        )
    )
    marketplace = json.loads(
        (ROOT / '.claude-plugin' / 'marketplace.json').read_text(encoding='utf-8')
    )
    assert plugin['name'] == 'docwrap'
    assert marketplace['plugins'][0]['source'] == './plugins/claude'
    assert not (ROOT / 'plugins' / 'claude' / 'hooks.json').exists()
    assert not (ROOT / 'plugins' / 'claude' / '.codex-plugin').exists()
