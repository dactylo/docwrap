# pyright: reportPrivateUsage=false

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from docwrap._cli import _build_hook_parser
from docwrap._hosts import load_adapter
from docwrap.hooks import API_VERSION, HookAdapter, HookResult, HostInput, HostResponse
from plugins.codex._docwrap._adapter import CodexAdapter

ROOT = Path(__file__).resolve().parents[3]
DENY_JSON = {
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': 'docwrap: reason',
    }
}


class TestCodexAdapter:
    def test_contract(self) -> None:
        adapter: HookAdapter = CodexAdapter()
        assert (adapter.name, adapter.api_version, adapter.requires_event) == (
            'codex',
            API_VERSION,
            False,
        )

    def test_parse_post_tool_apply_patch_paths(self) -> None:
        command = (
            '*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** Add File: b/c.py\n'
            '*** Move to: d.py\n*** Delete File: e.txt\n*** Update File: a.py\n*** End Patch\n'
        )
        payload = _payload('PostToolUse', 'apply_patch', {'command': command})
        assert CodexAdapter().parse(payload, None) == HostInput(
            'post_tool', None, '/r', None, ('a.py', 'b/c.py', 'd.py', 'e.txt')
        )

    def test_parse_pre_tool_bash_has_no_invocation_directory(self) -> None:
        payload = _payload('PreToolUse', 'Bash', {'command': 'ls'})
        assert CodexAdapter().parse(payload, None) == HostInput('pre_tool', 'ls', '/r', None)

    def test_parse_pre_tool_ignores_directory_lookalikes(self) -> None:
        payload = _payload(
            'PreToolUse', 'Bash', {'command': 'git commit -m x', 'workdir': '/b', 'cwd': '/b'}
        )
        payload['workdir'] = '/b'
        assert CodexAdapter().parse(payload, None) == HostInput(
            'pre_tool', 'git commit -m x', '/r', None
        )

    def test_render_allow_and_failure(self) -> None:
        assert CodexAdapter().render(_result('allow')) == HostResponse('', '', 0)
        failure = CodexAdapter().render_failure('docwrap: reason')
        assert json.loads(failure.stdout) == DENY_JSON
        assert (failure.stderr, failure.exit_code) == ('docwrap: reason\n', 0)

    def test_render_allow_feedback_is_one_system_message(self) -> None:
        response = CodexAdapter().render(_result('allow', 'first\nsecond', ('diagnostic',)))
        assert json.loads(response.stdout) == {'systemMessage': 'first\nsecond'}
        assert (response.stderr, response.exit_code) == ('diagnostic\n', 0)

    def test_render_deny_matches_claude_shape(self) -> None:
        response = CodexAdapter().render(_result('deny', 'docwrap: reason'))
        assert json.loads(response.stdout) == DENY_JSON
        assert (response.stderr, response.exit_code) == ('docwrap: reason\n', 0)


@dataclass(frozen=True)
class _Result:
    decision: Literal['allow', 'deny']
    reason: str | None = None
    notes: tuple[str, ...] = ()


def _payload(event: str, tool: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {'hook_event_name': event, 'tool_name': tool, 'tool_input': tool_input, 'cwd': '/r'}


def _result(decision: str, reason: str | None = None, notes: tuple[str, ...] = ()) -> HookResult:
    return _Result('deny' if decision == 'deny' else 'allow', reason, notes)


@pytest.mark.usefixtures('registered_hosts')
def test_native_manifest_command_loads_this_adapter() -> None:
    argv = shlex.split('docwrap hook --agent codex')
    args = _build_hook_parser().parse_args(argv[2:])
    adapter = load_adapter(args.agent)
    assert adapter.name == 'codex'
    assert args.event is None
    assert not adapter.requires_event


def test_native_manifest_is_unchanged() -> None:
    manifest = ROOT / 'plugins' / 'codex' / 'hooks' / 'hooks.json'
    assert json.loads(manifest.read_text(encoding='utf-8')) == {
        'hooks': {
            'PreToolUse': [
                {
                    'matcher': '^Bash$',
                    'hooks': [{'type': 'command', 'command': 'docwrap hook --agent codex'}],
                }
            ]
        }
    }


def test_native_plugin_identity_and_marketplace_source() -> None:
    plugin = json.loads(
        (ROOT / 'plugins' / 'codex' / '.codex-plugin' / 'plugin.json').read_text(encoding='utf-8')
    )
    marketplace = json.loads(
        (ROOT / '.agents' / 'plugins' / 'marketplace.json').read_text(encoding='utf-8')
    )
    assert plugin['name'] == 'docwrap'
    assert 'hooks' not in plugin
    assert marketplace['plugins'][0]['source'] == {'source': 'local', 'path': './plugins/codex'}
    assert not (ROOT / 'plugins' / 'codex' / 'hooks.json').exists()
    assert not (ROOT / 'plugins' / 'codex' / '.claude-plugin').exists()
