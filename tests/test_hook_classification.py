import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from docwrap import _hook as hook_mod
from docwrap._hook import run_hook


def _payload(command: str, cwd: Path) -> str:
    return json.dumps(
        {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': command},
            'cwd': str(cwd),
        }
    )


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    [
        'git init && git commit -m x',
        'git commit --future-option -m x',
        'git diff --check && git add sample.py && git commit -m x',
        'git add sample.py && git status --short && git commit -m x',
        'git log --oneline && git commit -m x',
    ],
)
@pytest.mark.parametrize('allow_on_failure', [False, True])
def test_unsupported_commit_denies_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    allow_on_failure: bool,
) -> None:
    def repository_work_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail('unsupported classification reached repository work')

    disabled = Mock(return_value=False)
    monkeypatch.setattr(hook_mod, 'is_hook_enabled', disabled)
    for name in ('discover_repo_root', 'run_pre_commit_adds', '_format_commit'):
        monkeypatch.setattr(hook_mod, name, repository_work_forbidden)
    result = run_hook(
        _payload(command, tmp_path), host='claude', allow_on_failure=allow_on_failure
    )
    captured = capsys.readouterr()
    assert result == 0
    disabled.assert_not_called()
    assert captured.out, 'unsupported commit must produce a native denial'
    assert json.loads(captured.out)['hookSpecificOutput']['permissionDecision'] == 'deny'
    if '--check' in command or 'status' in command or 'log' in command:
        assert 'separate tool call' in captured.out


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    [
        'git diff --check',
        'git add sample.py && git diff --cached --check',
        'git diff --check && git add sample.py && git commit --dry-run',
        'cd ./missing && git log --oneline && git add sample.py && git commit -m x',
    ],
)
def test_noncommitting_prechecks_do_no_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    def repository_work_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail('noncommitting precheck reached repository work')

    for name in ('is_hook_enabled', 'discover_repo_root', 'run_pre_commit_adds'):
        monkeypatch.setattr(hook_mod, name, repository_work_forbidden)
    assert run_hook(_payload(command, tmp_path), host='claude') == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ('', '')


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('allow_on_failure', [False, True])
def test_classifier_exception_always_denies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow_on_failure: bool,
) -> None:
    def classification_failed(_command: str, _cwd: str) -> None:
        raise RuntimeError('classifier exploded')

    def repository_work_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail('failed classification reached repository work')

    monkeypatch.setattr(hook_mod, 'classify_command', classification_failed)
    for name in ('is_hook_enabled', 'discover_repo_root', 'run_pre_commit_adds'):
        monkeypatch.setattr(hook_mod, name, repository_work_forbidden)
    result = run_hook(
        _payload('git commit -m x', tmp_path),
        host='claude',
        allow_on_failure=allow_on_failure,
    )
    captured = capsys.readouterr()
    assert result == 0
    output = json.loads(captured.out)['hookSpecificOutput']
    assert output['permissionDecision'] == 'deny'
    assert output['permissionDecisionReason'] == 'docwrap: unexpected error during classification'
    assert captured.err == 'docwrap: unexpected error during classification\n'
