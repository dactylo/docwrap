# pyright: reportPrivateUsage=false

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from docwrap._cli import _build_hook_parser
from docwrap._hook import run_hook
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

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / 'tests' / 'data' / 'commit_classification.json'
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

    def test_decisions_match_the_third_party_example_over_the_corpus(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        example_hosts: Path,
    ) -> None:
        del example_hosts
        (tmp_path / 'repo').mkdir()
        (tmp_path / ';').mkdir()

        def enabled(_cwd: str | None = None) -> bool:
            return True

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', enabled)
        entries = json.loads(CORPUS.read_text(encoding='utf-8'))['entries']
        for entry in entries:
            payload = json.dumps(
                _payload('PreToolUse', 'Bash', {'command': entry['command']})
                | {'cwd': str(tmp_path)}
            )
            decisions = [
                (run_hook(payload, host=host), _decision(capsys.readouterr().out))
                for host in ('claude', 'example')
            ]
            assert decisions[0] == decisions[1], entry['id']
            # This native payload establishes cwd, unlike the unknown-cwd corpus variants.
            if (
                entry['outcome'] != 'supported'
                and entry.get('command_cwd', str(tmp_path)) is not None
            ):
                expected = 'deny' if entry['outcome'] == 'unsupported' else None
                assert decisions[0] == (0, expected), entry['id']

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
