import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest

from docwrap import wrap_source
from docwrap._config import resolve_config
from tests.helpers import (
    BUILTIN_ENTRY_POINTS,
    decode_text,
    json_stdout,
    process_transcript,
    register_distribution,
    run_cli,
    run_setup,
    shell_path,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
AMEND_COMMAND = (
    'GIT_COMMITTER_DATE="$(git log -1 --format=%aI)" git commit --amend'
    " -m 'docs: amended subject' -m 'Amended body.'"
)
CORPUS = Path(__file__).parent / 'data' / 'commit_classification.json'
DIRECT_COMMAND = "git commit -m 'probe: direct commit'"
HEREDOC_COMMAND = (
    "git add sample.py && git commit -m \"$(cat <<'EOF'\n"
    'feat: sample\n'
    '\n'
    'Body mentioning git commit forms.\n'
    'EOF\n'
    ')"'
)
INCIDENT_FILES = (
    'src/sequential_hooks/_adapters/_common.py',
    'src/sequential_hooks/_containment/__init__.py',
    'src/sequential_hooks/_containment/_posix.py',
    'src/sequential_hooks/_containment/_uncontained.py',
    'src/sequential_hooks/_containment/_windows.py',
    'src/sequential_hooks/_containment/_windows_api.py',
    'src/sequential_hooks/_doctor/__init__.py',
    'src/sequential_hooks/_doctor/_types.py',
    'src/sequential_hooks/_runner.py',
)
INCIDENT_COMMAND = (
    "git commit -m 'docs: correct reviewed docstring contracts' -m 'Apply the 15 accepted"
    ' docstring proposals for failure conditions,\n'
    'completion sentinels, process lifecycle, version fallback, and timeout\n'
    'semantics. Executable code is unchanged.\n'
    '\n'
    'Validation: exact proposal matching, non-docstring AST comparison, and\n'
    "make check passed. Native Windows tests remain platform-skipped.'"
)
INCIDENT_PRECHECK = 'git add ' + ' '.join(INCIDENT_FILES) + ' && git diff --cached --check'
LONG_DOCSTRING = (
    'def {name}(value):\n'
    '    """Return the value.\n'
    '\n'
    '    Args:\n'
    '        value: This is a very long argument description that should definitely be'
    ' wrapped by docwrap before the staged file is committed ({name}).\n'
    '    """\n'
    '    return value\n'
)
SELECTION_MODES = {
    'auto-stage-short': 'auto',
    'only-short': 'only',
    'include-long': 'include',
    'pathspec-file-separate': 'only',
    'pathspec-file-nul-separate': 'only',
    'include-pathspec-file': 'include',
    'auto-stage-reset': 'normal',
}


class Harness:
    def __init__(self, repo: Path, distributions: Path) -> None:
        self.repo = repo
        self.env = dict(os.environ)
        self.env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(SRC), str(distributions)])
        self.env['GIT_AUTHOR_NAME'] = 'Test'
        self.env['GIT_AUTHOR_EMAIL'] = 'test@test.com'
        self.env['GIT_COMMITTER_NAME'] = 'Test'
        self.env['GIT_COMMITTER_EMAIL'] = 'test@test.com'

    def git(self, *args: str) -> str:
        return decode_text(run_setup(['git', *args], cwd=self.repo, env=self.env).stdout)

    def hook(
        self, command: str, *extra: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        payload = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': command},
            'cwd': str(self.repo),
        }
        merged = dict(self.env)
        merged.update(env or {})
        return run_cli(
            ['hook', '--agent', 'claude', *extra],
            stdin=json.dumps(payload),
            cwd=self.repo,
            env=merged,
        )

    def git_bytes(self, *args: str) -> bytes:
        return run_setup(['git', *args], cwd=self.repo, env=self.env).stdout

    def assert_fixed_point(self, path: str, ref: str = 'HEAD') -> str:
        blob = self.git('show', f'{ref}:{path}')
        config = resolve_config(self.repo / path)
        assert wrap_source(blob, config=config) == blob, path
        return blob

    def run_agent_command(self, command: str) -> None:
        run_setup(['sh', '-c', command], cwd=self.repo, env=self.env)

    def stage(self, path: str, source: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.encode('utf-8'))
        self.git('add', path)


class TestDenials:
    def test_infrastructure_failure_denies_under_fail_closed(self, harness: Harness) -> None:
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        result = harness.hook(DIRECT_COMMAND, env={'GIT_OBJECT_DIRECTORY': '/nonexistent'})
        assert result.returncode == 0, process_transcript(result)
        decision = cast('dict[str, dict[str, str]]', json_stdout(result))['hookSpecificOutput']
        assert decision['permissionDecision'] == 'deny'
        assert decision['permissionDecisionReason'].startswith('docwrap: failed to')

    def test_infrastructure_failure_warns_under_allow_on_failure(self, harness: Harness) -> None:
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        result = harness.hook(
            DIRECT_COMMAND, '--allow-on-failure', env={'GIT_OBJECT_DIRECTORY': '/nonexistent'}
        )
        assert (result.returncode, result.stdout) == (0, ''), process_transcript(result)
        assert result.stderr.startswith('WARNING: docwrap: failed to')

    @pytest.mark.parametrize('case', ['short', 'other-error', 'cached', 'cached-add', 'unborn'])
    def test_precheck_denial_preserves_repository(self, harness: Harness, case: str) -> None:
        short = 'def example():\n    """Return a value."""\n    return 1\n'
        harness.stage('sample.py', short)
        harness.stage('other.txt', 'clean\n')
        harness.git('commit', '-qm', 'precheck baseline')
        if case == 'unborn':
            harness.git('checkout', '--orphan', 'unborn')
            harness.git('rm', '-rf', '.')
        dirty = short.replace('return 1\n', 'return 2  \n')
        check_args = ['diff', '--check']
        command = 'git diff --check && git add sample.py && git commit -m update'
        if case.startswith('cached'):
            dirty = LONG_DOCSTRING.replace('({name}).\n', '({name}).  \n').format(name='sample')
            harness.stage('sample.py', dirty)
            check_args = ['diff', '--cached', '--check']
            command = 'git diff --cached --check && git commit -m update'
            if case == 'cached-add':
                command = 'git add sample.py && ' + command
        else:
            (harness.repo / 'sample.py').write_text(dirty, encoding='utf-8')
        if case == 'other-error':
            (harness.repo / 'other.txt').write_text('bad  \n', encoding='utf-8')
        if case == 'unborn':
            check_args = ['log', '--oneline']
            command = 'git log --oneline && git add sample.py && git commit -m update'

        def pending_check() -> subprocess.CompletedProcess[bytes]:
            return run_setup(
                ['git', *check_args],
                cwd=harness.repo,
                check=False,
                env=harness.env,
            )

        before_check = pending_check()
        assert before_check.returncode == (128 if case == 'unborn' else 2), process_transcript(
            before_check
        )
        head = None if case == 'unborn' else harness.git('rev-parse', 'HEAD')
        head_bytes = (harness.repo / '.git' / 'HEAD').read_bytes()
        staged = harness.git_bytes('ls-files', '--stage')
        index = (harness.repo / '.git' / 'index').read_bytes()
        working = {path: path.read_bytes() for path in harness.repo.iterdir() if path.is_file()}
        result = harness.hook(command)
        # Check bytes before any further Git command can refresh the index.
        assert (harness.repo / '.git' / 'index').read_bytes() == index
        assert {
            path: path.read_bytes() for path in harness.repo.iterdir() if path.is_file()
        } == working
        assert (harness.repo / '.git' / 'HEAD').read_bytes() == head_bytes
        assert harness.git_bytes('ls-files', '--stage') == staged
        if head is not None:
            assert harness.git('rev-parse', 'HEAD') == head
        else:
            assert not (harness.repo / '.git' / 'refs' / 'heads' / 'unborn').exists()
        after_check = pending_check()
        assert (after_check.returncode, after_check.stdout, after_check.stderr) == (
            before_check.returncode,
            before_check.stdout,
            before_check.stderr,
        ), process_transcript(after_check)
        assert result.returncode == 0, process_transcript(result)
        decision = cast('dict[str, dict[str, str]]', json_stdout(result))['hookSpecificOutput']
        assert decision['permissionDecision'] == 'deny'
        assert 'separate tool call' in decision['permissionDecisionReason']

    def test_unsupported_commit_denies_and_changes_nothing(self, harness: Harness) -> None:
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        head = harness.git('rev-parse', 'HEAD')
        index = harness.git('diff', '--cached')
        result = harness.hook("git commit -m 'x' && git push")
        assert result.returncode == 0, process_transcript(result)
        decision = cast('dict[str, dict[str, str]]', json_stdout(result))['hookSpecificOutput']
        assert decision['permissionDecision'] == 'deny'
        assert 'unsupported commit command' in decision['permissionDecisionReason']
        assert result.stderr.startswith('docwrap: unsupported commit command')
        assert harness.git('rev-parse', 'HEAD') == head
        assert harness.git('diff', '--cached') == index


class TestIncremental:
    def _enable_incremental(self, harness: Harness) -> None:
        (harness.repo / 'pyproject.toml').write_text(
            '[tool.docwrap]\nincremental = true\n', encoding='utf-8'
        )
        harness.git('add', 'pyproject.toml')
        harness.git('commit', '-qm', 'config')

    def test_deletion_only_hunk_leaves_the_blob_byte_identical(self, harness: Harness) -> None:
        self._enable_incremental(harness)
        original = LONG_DOCSTRING.format(name='sample') + '\n\nTRAILING = 1\nEXTRA = 2\n'
        harness.stage('sample.py', original)
        harness.git('commit', '-qm', 'baseline', '--no-verify')
        edited = LONG_DOCSTRING.format(name='sample') + '\n\nTRAILING = 1\n'
        harness.stage('sample.py', edited)
        _assert_allow(harness.hook(DIRECT_COMMAND), None)
        assert harness.git_bytes('show', ':sample.py') == edited.encode('utf-8')
        harness.run_agent_command(DIRECT_COMMAND)
        assert harness.git_bytes('show', 'HEAD:sample.py') == edited.encode('utf-8')

    def test_touched_docstring_is_wrapped(self, harness: Harness) -> None:
        self._enable_incremental(harness)
        harness.stage('sample.py', 'x = 1\n')
        harness.git('commit', '-qm', 'baseline')
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        _assert_allow(harness.hook(DIRECT_COMMAND), 1)
        harness.run_agent_command(DIRECT_COMMAND)
        harness.assert_fixed_point('sample.py')


class TestMandatedForms:
    def test_absolute_route_touches_only_the_selected_repository(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        other_root = tmp_path / 'other'
        other_root.mkdir()
        other = Harness(other_root, tmp_path / 'hosts')
        other.git('init', '-q')
        other.git('config', 'docwrap.hook', 'true')
        other.stage('base.py', 'x = 1\n')
        other.git('commit', '-qm', 'init')
        target = LONG_DOCSTRING.format(name='target')
        other.stage('target.py', target)
        sample = LONG_DOCSTRING.format(name='sample')
        harness.stage('sample.py', sample)
        command = f"git -C {shell_path(other_root)} commit -m 'probe: cross repository'"
        _assert_allow(harness.hook(command), 1)
        assert harness.git('show', ':sample.py') == sample
        assert other.git('show', ':target.py') != target
        harness.run_agent_command(command)
        assert other.assert_fixed_point('target.py') != target
        assert harness.git('rev-list', '--count', 'HEAD') == '1\n'

    def test_direct_commit(self, harness: Harness) -> None:
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        _assert_allow(harness.hook(DIRECT_COMMAND), 1)
        harness.run_agent_command(DIRECT_COMMAND)
        blob = harness.assert_fixed_point('sample.py')
        assert blob != LONG_DOCSTRING.format(name='sample')

    def test_heredoc_commit_with_add(self, harness: Harness) -> None:
        (harness.repo / 'sample.py').write_text(
            LONG_DOCSTRING.format(name='sample'), encoding='utf-8'
        )
        _assert_allow(harness.hook(HEREDOC_COMMAND), 1)
        harness.run_agent_command(HEREDOC_COMMAND)
        harness.assert_fixed_point('sample.py')
        assert harness.git('log', '-1', '--format=%B').startswith(
            'feat: sample\n\nBody mentioning'
        )

    def test_incident_command(self, harness: Harness) -> None:
        for index, path in enumerate(INCIDENT_FILES):
            (harness.repo / path).parent.mkdir(parents=True, exist_ok=True)
            (harness.repo / path).write_bytes(LONG_DOCSTRING.format(name=f'f{index}').encode())
        _assert_allow(harness.hook(INCIDENT_PRECHECK), None)
        harness.run_agent_command(INCIDENT_PRECHECK)
        _assert_allow(harness.hook(INCIDENT_COMMAND), len(INCIDENT_FILES))
        harness.run_agent_command(INCIDENT_COMMAND)
        for path in INCIDENT_FILES:
            harness.assert_fixed_point(path)
        assert (
            harness.git('log', '-1', '--format=%s')
            == 'docs: correct reviewed docstring contracts\n'
        )

    def test_mandated_amend(self, harness: Harness) -> None:
        harness.stage('first.py', 'y = 2\n')
        harness.git('commit', '-qm', 'first')
        author_date = harness.git('log', '-1', '--format=%aI')
        harness.stage('sample.py', LONG_DOCSTRING.format(name='sample'))
        _assert_allow(harness.hook(AMEND_COMMAND), 1)
        harness.run_agent_command(AMEND_COMMAND)
        harness.assert_fixed_point('sample.py')
        assert harness.git('log', '-1', '--format=%cI') == author_date
        assert harness.git('log', '-1', '--format=%s') == 'docs: amended subject\n'

    def test_partial_staging_keeps_the_unstaged_edit_out_of_the_commit(
        self, harness: Harness
    ) -> None:
        source = LONG_DOCSTRING.format(name='sample')
        harness.stage('sample.py', source)
        (harness.repo / 'sample.py').write_text(source + '\n# unstaged note\n', encoding='utf-8')
        _assert_allow(harness.hook(DIRECT_COMMAND), 1)
        assert harness.git('show', ':sample.py') != source
        harness.run_agent_command(DIRECT_COMMAND)
        committed = harness.assert_fixed_point('sample.py')
        assert committed != source
        assert '# unstaged note' not in committed
        assert '# unstaged note' in (harness.repo / 'sample.py').read_text(encoding='utf-8')

    def test_separate_check_then_add_commit_after_whitespace_correction(
        self, harness: Harness
    ) -> None:
        dirty = 'def example():\n    """Return a value."""\n    return 2  \n'
        harness.stage('sample.py', dirty.replace('return 2  ', 'return 1'))
        harness.git('commit', '-qm', 'baseline')
        (harness.repo / 'sample.py').write_bytes(dirty.encode('utf-8'))
        _assert_allow(harness.hook('git diff --check'), None)
        with pytest.raises(AssertionError, match='setup command failed'):
            harness.run_agent_command('git diff --check')
        corrected = dirty.replace('return 2  \n', 'return 2\n')
        (harness.repo / 'sample.py').write_bytes(corrected.encode('utf-8'))
        _assert_allow(harness.hook('git diff --check'), None)
        harness.run_agent_command('git diff --check')
        add_commit = 'git add sample.py && git commit -m update'
        _assert_allow(harness.hook(add_commit), None)
        harness.run_agent_command(add_commit)
        assert harness.git('show', 'HEAD:sample.py') == corrected


def _assert_allow(result: subprocess.CompletedProcess[str], wrapped_files: int | None) -> None:
    assert (result.returncode, result.stderr) == (0, ''), process_transcript(result)
    if wrapped_files is None:
        assert result.stdout == ''
        return
    noun = 'file' if wrapped_files == 1 else 'files'
    assert json_stdout(result) == {
        'systemMessage': f'docwrap: wrapped {wrapped_files} staged Python {noun} for this commit'
    }


def _selection_cases() -> list[tuple[str, str, str]]:
    entries = json.loads(CORPUS.read_text(encoding='utf-8'))['entries']
    by_id = {entry['id']: entry for entry in entries}
    return [
        (entry_id, by_id[entry_id]['command'], mode) for entry_id, mode in SELECTION_MODES.items()
    ]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    repo = tmp_path / 'repo'
    repo.mkdir()
    hosts = tmp_path / 'hosts'
    hosts.mkdir()
    register_distribution(hosts, 'docwrap', BUILTIN_ENTRY_POINTS)
    harness = Harness(repo, hosts)
    harness.git('init', '-q')
    harness.git('config', 'docwrap.hook', 'true')
    (repo / 'base.py').write_text('x = 1\n', encoding='utf-8')
    harness.git('add', 'base.py')
    harness.git('commit', '-qm', 'init')
    return harness


class TestSelectionForms:
    @pytest.mark.parametrize(('entry_id', 'command', 'mode'), _selection_cases())
    def test_selected_blobs_are_wrapped(
        self, harness: Harness, entry_id: str, command: str, mode: str
    ) -> None:
        baseline = 'x = 1\n'
        for path in ('a.py', 'b.py'):
            harness.stage(path, baseline)
        harness.git('commit', '-qm', 'selection baseline')
        selected = LONG_DOCSTRING.format(name='selected')
        other = LONG_DOCSTRING.format(name='other')
        (harness.repo / 'a.py').write_text(selected, encoding='utf-8')
        if mode == 'normal':
            harness.git('add', 'a.py')
        harness.stage('b.py', other)
        (harness.repo / 'paths.txt').write_bytes(b'a.py\n')
        (harness.repo / 'paths.nul').write_bytes(b'a.py\0')
        index_before = harness.git('write-tree')
        _assert_allow(harness.hook(command), {'normal': 2, 'include': 1}.get(mode))
        if mode in ('only', 'auto'):
            assert harness.git('write-tree') == index_before, entry_id
        if mode in ('include', 'normal'):
            assert harness.assert_fixed_point('b.py', '') != other
        if mode == 'normal':
            assert harness.assert_fixed_point('a.py', '') != selected
        else:
            working = (harness.repo / 'a.py').read_text(encoding='utf-8')
            assert working != selected
            assert wrap_source(working, config=resolve_config(harness.repo / 'a.py')) == working
        harness.run_agent_command(command)
        assert harness.assert_fixed_point('a.py') != selected
        if mode == 'only':
            assert harness.git('show', 'HEAD:b.py') == baseline
            assert harness.git('show', ':b.py') == other
        else:
            assert harness.assert_fixed_point('b.py') != other
