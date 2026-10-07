import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from docwrap._hosts import AdapterLoadError, load_adapter, registered_host_names
from docwrap.hooks import HookAdapter, HookInput, HookResponse, HookResult
from tests.helpers import BUILTIN_ENTRY_POINTS, EXAMPLE_ENTRY_POINTS, register_distribution
from tests.hosts_example.adapter import (
    ExampleAdapter,
    ExampleInput,
    ExampleResponse,
)

PAYLOAD = json.dumps(
    {
        'hook_event_name': 'PreToolUse',
        'tool_name': 'Bash',
        'tool_input': {'command': 'ls'},
        'cwd': '.',
    }
)


@pytest.mark.usefixtures('registered_hosts')
class TestDiscovery:
    def test_bundled_adapters_are_registered_through_plugin_manifests(self) -> None:
        assert registered_host_names() == sorted(BUILTIN_ENTRY_POINTS)
        for name in BUILTIN_ENTRY_POINTS:
            assert load_adapter(name).name == name

    def test_duplicate_name_is_refused(self, example_hosts: Path) -> None:
        register_distribution(
            example_hosts,
            'docwrap-host-duplicate',
            {'example': EXAMPLE_ENTRY_POINTS['example']},
        )
        with pytest.raises(AdapterLoadError, match="host 'example' is registered 2 times"):
            load_adapter('example')

    def test_third_party_adapter_is_discovered(self, example_hosts: Path) -> None:
        del example_hosts
        adapter = load_adapter('example')
        assert adapter.name == 'example'
        assert 'example' in registered_host_names()

    def test_unimportable_adapter_is_refused(self, example_hosts: Path) -> None:
        del example_hosts
        with pytest.raises(AdapterLoadError, match="host 'missing' failed to load"):
            load_adapter('missing')

    def test_unknown_host_names_the_registered_hosts(self) -> None:
        hosts = ', '.join(sorted(BUILTIN_ENTRY_POINTS))
        with pytest.raises(
            AdapterLoadError, match=rf"unknown host 'nope'; registered hosts: {hosts}"
        ):
            load_adapter('nope')

    def test_version_mismatch_is_refused(self, example_hosts: Path) -> None:
        del example_hosts
        with pytest.raises(
            AdapterLoadError,
            match='hook adapter API',
        ):
            load_adapter('stale')


class TestThirdPartyContract:
    def test_example_classes_satisfy_the_public_protocols(self) -> None:
        adapter: HookAdapter = ExampleAdapter()
        parsed: HookInput = ExampleInput('pre_tool', 'ls', '.', None)
        response: HookResponse = ExampleResponse('', '', 0)
        payload = json.loads(PAYLOAD)
        assert adapter.parse(payload, None) == parsed
        assert parsed.edited_paths == ()
        assert adapter.render(_result('allow')) == response


@dataclass(frozen=True)
class _Result:
    decision: Literal['allow', 'deny']
    reason: str | None = None
    notes: tuple[str, ...] = ()


def _result(decision: str, reason: str | None = None) -> HookResult:
    return _Result('deny' if decision == 'deny' else 'allow', reason)
