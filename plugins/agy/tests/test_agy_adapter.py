# pyright: reportPrivateUsage=false

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from docwrap._hook_front import _build_hook_parser
from docwrap._hosts import load_adapter
from docwrap.hooks import API_VERSION, HookAdapter, HookResult, HostInput, HostResponse
from plugins.agy._docwrap._adapter import AntigravityAdapter

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _Result:
    decision: Literal['allow', 'deny']
    reason: str | None = None
    notes: tuple[str, ...] = ()


def _payload(tool: str, args: dict[str, object]) -> dict[str, object]:
    return {'toolCall': {'name': tool, 'args': args}}


def _result(decision: str, reason: str | None = None, notes: tuple[str, ...] = ()) -> HookResult:
    return _Result('deny' if decision == 'deny' else 'allow', reason, notes)


class TestAntigravityAdapter:
    def test_contract(self) -> None:
        adapter: HookAdapter = AntigravityAdapter()
        assert (adapter.name, adapter.api_version, adapter.requires_event) == (
            'agy',
            API_VERSION,
            True,
        )

    @pytest.mark.parametrize('tool', ['write_to_file', 'replace_file_content'])
    def test_parse_post_tool_file_writers(self, tool: str) -> None:
        payload = _payload(tool, {'TargetFile': '/t/a.py', 'Cwd': '/t'})
        assert AntigravityAdapter().parse(payload, 'PostToolUse') == HostInput(
            'post_tool', None, '/t', None, ('/t/a.py',)
        )

    def test_parse_pre_tool_other_tool_has_no_command(self) -> None:
        payload = _payload('write_to_file', {'TargetFile': 'a.py', 'Cwd': '/t'})
        assert AntigravityAdapter().parse(payload, 'PreToolUse') == HostInput(
            'pre_tool', None, '/t', None
        )

    def test_parse_rejects_unknown_event(self) -> None:
        with pytest.raises(ValueError, match='unknown Antigravity event'):
            AntigravityAdapter().parse({}, 'Stop')

    @pytest.mark.parametrize('cwd', ['rel', '', 7])
    def test_parse_run_command_non_absolute_cwd_is_unknown(self, cwd: object) -> None:
        payload = _payload('run_command', {'CommandLine': 'git commit -m x', 'Cwd': cwd})
        payload['workspacePaths'] = ['/ws']
        parsed = AntigravityAdapter().parse(payload, 'PreToolUse')
        assert parsed is not None
        assert parsed.command_cwd is None

    def test_parse_run_command_prefers_cwd_over_workspace(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / 'cwd')
        payload = _payload('run_command', {'CommandLine': 'git commit -m x', 'Cwd': cwd})
        payload['workspacePaths'] = [str(tmp_path / 'workspace')]
        assert AntigravityAdapter().parse(payload, 'PreToolUse') == HostInput(
            'pre_tool', 'git commit -m x', cwd, cwd
        )

    def test_parse_run_command_without_cwd_is_unknown(self) -> None:
        payload = _payload('run_command', {'CommandLine': 'git commit -m x'})
        payload['workspacePaths'] = ['/ws']
        assert AntigravityAdapter().parse(payload, 'PreToolUse') == HostInput(
            'pre_tool', 'git commit -m x', '/ws', None
        )

    def test_render_allow_and_failure(self) -> None:
        assert AntigravityAdapter().render(_result('allow')) == HostResponse('', '', 0)
        failure = AntigravityAdapter().render_failure('docwrap: reason')
        assert json.loads(failure.stdout) == {'decision': 'deny', 'reason': 'docwrap: reason'}
        assert (failure.stderr, failure.exit_code) == ('docwrap: reason\n', 0)

    def test_render_allow_feedback_is_one_system_message(self) -> None:
        response = AntigravityAdapter().render(_result('allow', 'first\nsecond', ('diagnostic',)))
        assert json.loads(response.stdout) == {'systemMessage': 'first\nsecond'}
        assert (response.stderr, response.exit_code) == ('diagnostic\n', 0)

    def test_render_deny_is_top_level_json(self) -> None:
        response = AntigravityAdapter().render(_result('deny', 'docwrap: reason', ('note',)))
        assert json.loads(response.stdout) == {'decision': 'deny', 'reason': 'docwrap: reason'}
        assert (response.stderr, response.exit_code) == ('note\ndocwrap: reason\n', 0)


@pytest.mark.usefixtures('registered_hosts')
def test_native_manifest_command_loads_this_adapter() -> None:
    argv = shlex.split('docwrap hook --agent agy --event PreToolUse')
    args = _build_hook_parser().parse_args(argv[2:])
    adapter = load_adapter(args.agent)
    assert adapter.name == 'agy'
    assert args.event == 'PreToolUse'
    assert adapter.requires_event


def test_native_manifest_is_unchanged() -> None:
    assert json.loads((ROOT / 'plugins' / 'agy' / 'hooks.json').read_text(encoding='utf-8')) == {
        'docwrap': {
            'PreToolUse': [
                {
                    'matcher': 'run_command',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': 'docwrap hook --agent agy --event PreToolUse',
                        }
                    ],
                }
            ]
        }
    }


def test_native_plugin_identity_and_layout() -> None:
    plugin = json.loads((ROOT / 'plugins' / 'agy' / 'plugin.json').read_text(encoding='utf-8'))
    assert set(plugin) == {'$schema', 'name', 'description'}
    assert plugin['$schema'] == 'https://antigravity.google/schemas/v1/plugin.json'
    assert plugin['name'] == 'docwrap-agy'
    assert isinstance(plugin['description'], str)
    assert plugin['description'].strip()
    assert not (ROOT / 'plugins' / 'agy' / 'hooks').exists()
