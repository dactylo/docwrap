import errno
import os
import subprocess
from pathlib import Path

import pytest

import docwrap._git as git_mod
from docwrap._core._incremental import get_eligible_ranges, parse_diff_hunks
from docwrap._files import ReadResult, decode_python_source
from docwrap._git import (
    IndexEntry,
    RepoDiscoveryResult,
    RepoDiscoveryStatus,
    StagedFileInfo,
    _decode_git_pathspec_record,
    _get_empty_tree_hash,
    _git_config_bool,
    _iter_raw_diff_records,
    _parse_cat_file_output,
    _parse_null_separated_py_paths,
    _read_git_pathspec_file,
    _run_git_command,
    _run_git_command_bytes,
    apply_index_entries,
    discover_repo_root,
    get_modified_tracked_py_files,
    get_repo_diff,
    get_repo_root,
    get_staged_py_info,
    is_hook_enabled,
    prepare_index_entry,
    read_staged_blobs,
    resolve_pathspec_py_files,
    run_pre_commit_adds,
)
from docwrap._grammar import CommitPlan, classify_grammar
from docwrap._shellparse import parse_command
from docwrap._types import BaselineStatus, PreCommitAdd
from tests.helpers import git_commit_all, run_setup


class TestApplyIndexEntries:
    def test_empty_is_noop(self, tmp_path: Path) -> None:
        assert apply_index_entries(str(tmp_path), []) is True

    def test_encodes_path_as_nul_delimited_filesystem_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path_bytes = _native_path_bytes(b'non-utf8-\xff.py', b'caf\xc3\xa9.py')
        path = os.fsdecode(path_bytes)
        captured: dict[str, object] = {}

        def capture_run(
            args: list[str | bytes], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            captured['args'] = args
            captured['input'] = kwargs.get('input')
            return subprocess.CompletedProcess(args, 0, b'', b'')

        monkeypatch.setattr(subprocess, 'run', capture_run)

        assert apply_index_entries(str(tmp_path), [IndexEntry('100644', 'a' * 40, path)])
        assert captured['input'] == b'100644 ' + b'a' * 40 + b'\t' + path_bytes + b'\0'

    def test_updates_index(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        git_commit_all(tmp_path, 'init')
        entry = prepare_index_entry(str(tmp_path), 'foo.py', 'x = 1\n', '100644')
        assert entry is not None
        assert apply_index_entries(str(tmp_path), [entry]) is True


class TestBatchProtocolParsing:
    @staticmethod
    def _info(path: str) -> StagedFileInfo:
        return StagedFileInfo(path, '100644', 'a' * 40, '0' * 40, 'M')

    @staticmethod
    def _record(
        content: bytes = b'x = 1\n',
        object_identity: bytes = b'a' * 40,
    ) -> bytes:
        header = object_identity + b' blob ' + str(len(content)).encode('ascii') + b'\n'
        return header + content + b'\n'

    @pytest.mark.parametrize(
        ('payload', 'expected_content', 'expected_encoding', 'expected_newline'),
        [
            pytest.param(
                b'\xef\xbb\xbfx = 1\r\n',
                'x = 1\n',
                'utf-8-sig',
                '\r\n',
                id='bom-crlf',
            ),
            pytest.param(
                b'# coding: latin-1\nvalue = "\xe9"\n',
                '# coding: latin-1\nvalue = "\xe9"\n',
                'iso-8859-1',
                '\n',
                id='latin1-lf',
            ),
            pytest.param(
                b'x = 1\ry = 2\r',
                'x = 1\ny = 2\n',
                'utf-8',
                '\n',
                id='lone-cr',
            ),
        ],
    )
    def test_content_payload_preserves_decode_and_newline_contract(
        self,
        payload: bytes,
        expected_content: str,
        expected_encoding: str,
        expected_newline: str,
    ) -> None:
        result = _parse_cat_file_output(self._record(payload), [self._info('file.py')])['file.py']

        assert result.status is BaselineStatus.CONTENT
        assert result.content == expected_content
        assert result.encoding == expected_encoding
        assert result.newline == expected_newline

    def test_content_payload_uses_shared_decoder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = b'x = 1\r\n'
        calls: list[bytes] = []

        def tracked_decode(raw: bytes) -> ReadResult:
            calls.append(raw)
            return decode_python_source(raw)

        monkeypatch.setattr(git_mod, 'decode_python_source', tracked_decode)

        result = _parse_cat_file_output(self._record(payload), [self._info('file.py')])

        assert calls == [payload]
        assert result['file.py'].content == 'x = 1\n'

    @pytest.mark.parametrize(
        ('record', 'expected_status'),
        [
            (b'a' * 40 + b' missing\n', BaselineStatus.MISSING),
            (_record(), BaselineStatus.CONTENT),
            (_record(b'\xff'), BaselineStatus.DECODE_FAILED),
        ],
        ids=['missing', 'content', 'decode-failed'],
    )
    def test_exact_object_identity_retains_valid_status(
        self,
        record: bytes,
        expected_status: BaselineStatus,
    ) -> None:
        results = _parse_cat_file_output(record, [self._info('file.py')])

        assert results['file.py'].status is expected_status

    @pytest.mark.parametrize(
        'failure',
        [
            b'a' * 40 + b' blob 6\nx =',
            b'a' * 40 + b' blob invalid\n',
            b'a' * 40 + b' blob -1\n',
        ],
        ids=['truncated-body', 'non-integer-length', 'negative-length'],
    )
    def test_invalid_content_record_marks_affected_and_unresolved_as_infrastructure_errors(
        self,
        failure: bytes,
    ) -> None:
        info = [self._info(f'{name}.py') for name in ('first', 'middle', 'last')]
        data = self._record() + failure

        results = _parse_cat_file_output(data, info)

        assert results['first.py'].status is BaselineStatus.CONTENT
        assert results['middle.py'].status is BaselineStatus.INFRA_ERROR
        assert results['last.py'].status is BaselineStatus.INFRA_ERROR

    @pytest.mark.parametrize('failure_index', [0, 1, 2], ids=['first', 'middle', 'last'])
    def test_malformed_header_marks_affected_and_unresolved_as_infrastructure_errors(
        self,
        failure_index: int,
    ) -> None:
        info = [self._info(f'{name}.py') for name in ('first', 'middle', 'last')]
        data = self._record() * failure_index + b'malformed header\n'

        results = _parse_cat_file_output(data, info)

        for resolved in info[:failure_index]:
            assert results[resolved.path].status is BaselineStatus.CONTENT
        for unresolved in info[failure_index:]:
            assert results[unresolved.path].status is BaselineStatus.INFRA_ERROR

    def test_missing_object_header_is_a_skip_and_parsing_continues(self) -> None:
        info = [self._info(f'{name}.py') for name in ('first', 'middle', 'last')]
        missing = b'a' * 40 + b' missing\n'
        data = self._record() + missing + self._record()

        results = _parse_cat_file_output(data, info)

        assert results['first.py'].status is BaselineStatus.CONTENT
        assert results['middle.py'].status is BaselineStatus.MISSING
        assert results['last.py'].status is BaselineStatus.CONTENT

    @pytest.mark.parametrize('failure_index', [0, 1, 2], ids=['first', 'middle', 'last'])
    @pytest.mark.parametrize(
        ('record_kind', 'object_identity'),
        [
            ('missing', b'garbage'),
            ('missing', b'b' * 40),
            ('content', b'garbage'),
            ('content', b'b' * 40),
        ],
        ids=['garbage-missing', 'different-missing', 'garbage-content', 'different-content'],
    )
    def test_unexpected_object_identity_marks_affected_and_unresolved_as_infrastructure_errors(
        self,
        failure_index: int,
        record_kind: str,
        object_identity: bytes,
    ) -> None:
        info = [self._info(f'{name}.py') for name in ('first', 'middle', 'last')]
        if record_kind == 'missing':
            failure = object_identity + b' missing\n'
        else:
            failure = self._record(object_identity=object_identity)
        data = self._record() * failure_index + failure

        results = _parse_cat_file_output(data, info)

        for resolved in info[:failure_index]:
            assert results[resolved.path].status is BaselineStatus.CONTENT
        for unresolved in info[failure_index:]:
            assert results[unresolved.path].status is BaselineStatus.INFRA_ERROR

    def test_unknown_cookie_remains_decode_failure(self) -> None:
        result = _parse_cat_file_output(
            self._record(b'# coding: not-a-codec\nx = 1\n'),
            [self._info('file.py')],
        )

        assert result['file.py'].status is BaselineStatus.DECODE_FAILED


class TestDiscoverRepoRoot:
    def test_bare_repository_is_not_worktree(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '--bare'], cwd=tmp_path)

        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    def test_embedded_nul_working_directory_is_error(self) -> None:
        result = discover_repo_root('invalid\0cwd')

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_file_working_directory_is_error(self, tmp_path: Path) -> None:
        target = tmp_path / 'file'
        target.write_text('not a directory')

        result = discover_repo_root(str(target))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_git_launch_failure_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def fail_git(*_args: object, **_kwargs: object) -> None:
            return

        monkeypatch.setattr('docwrap._git._run_git_command', fail_git)

        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_git_output_decode_failure_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def raise_decode_error(*_args: object, **_kwargs: object) -> None:
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

        monkeypatch.setattr(subprocess, 'run', raise_decode_error)

        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    @pytest.mark.parametrize(
        ('status', 'root'),
        [
            (RepoDiscoveryStatus.WORKTREE, None),
            (RepoDiscoveryStatus.WORKTREE, ''),
            (RepoDiscoveryStatus.NOT_WORKTREE, '/repo'),
            (RepoDiscoveryStatus.ERROR, '/repo'),
        ],
    )
    def test_invalid_result_state_is_rejected(
        self, status: RepoDiscoveryStatus, root: str | None
    ) -> None:
        with pytest.raises(ValueError, match='repository discovery'):
            RepoDiscoveryResult(status=status, root=root)

    def test_missing_directory_is_error(self, tmp_path: Path) -> None:
        result = discover_repo_root(str(tmp_path / 'missing'))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_nested_bare_repository_is_not_worktree(self, tmp_path: Path) -> None:
        outer = tmp_path / 'outer'
        outer.mkdir()
        run_setup(['git', 'init'], cwd=outer)
        bare = outer / 'nested.git'
        run_setup(
            ['git', 'init', '--bare', str(bare)],
            cwd=outer,
        )

        result = discover_repo_root(str(bare))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    def test_non_repository_is_not_worktree(self, tmp_path: Path) -> None:
        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    def test_repo_marker_inspection_failure_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_lstat = Path.lstat

        def fail_marker(path: Path) -> os.stat_result:
            if path.name == '.git':
                raise PermissionError
            return original_lstat(path)

        monkeypatch.setattr(Path, 'lstat', fail_marker)

        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_repo_marker_with_git_failure_is_error(self, tmp_path: Path) -> None:
        (tmp_path / '.git').mkdir()

        result = discover_repo_root(str(tmp_path))

        assert result == RepoDiscoveryResult(RepoDiscoveryStatus.ERROR)

    def test_subdirectory_returns_worktree_root(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        subdir = tmp_path / 'nested'
        subdir.mkdir()

        result = discover_repo_root(str(subdir))

        assert result.status is RepoDiscoveryStatus.WORKTREE
        assert result.root is not None
        assert Path(result.root).resolve() == tmp_path.resolve()

    def test_text_git_decode_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def raise_decode_error(*_args: object, **_kwargs: object) -> None:
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

        monkeypatch.setattr(subprocess, 'run', raise_decode_error)

        assert _run_git_command(['git', 'status']) is None

    @pytest.mark.parametrize('repo_name', ['repo', 'repo '], ids=['ordinary', 'trailing-space'])
    def test_toplevel_preserves_exact_root_path(self, tmp_path: Path, repo_name: str) -> None:
        repo = tmp_path / repo_name
        repo.mkdir()
        run_setup(['git', 'init'], cwd=repo)

        result = discover_repo_root(str(repo))

        assert result.status is RepoDiscoveryStatus.WORKTREE
        assert result.root is not None
        assert Path(result.root).resolve() == repo.resolve()
        assert Path(result.root).name == repo.resolve().name

    def test_worktree_returns_normalized_root(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)

        result = discover_repo_root(str(tmp_path))

        assert result.status is RepoDiscoveryStatus.WORKTREE
        assert result.root is not None
        assert Path(result.root).resolve() == tmp_path.resolve()


class TestGetEmptyTreeHash:
    def test_returns_hash(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        result = _get_empty_tree_hash(str(tmp_path))
        assert len(result) >= 40
        assert result != '0' * len(result)


class TestGetModifiedTrackedPyFiles:
    def test_modified_file(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        git_commit_all(tmp_path, 'init')
        f.write_text('x = 1\n')
        result = get_modified_tracked_py_files(str(tmp_path))
        assert result is not None
        assert 'foo.py' in result


class TestGetRepoDiff:
    def test_batch_diff_command_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def fail_run(
            args: list[str | bytes], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(args, 1, b'', b'failure')

        monkeypatch.setattr(subprocess, 'run', fail_run)

        assert get_repo_diff('/repo', cached=True) is None

    def test_diff_includes_all_modified_paths(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        requested = tmp_path / 'requested.py'
        unrelated = tmp_path / 'unrelated.py'
        requested.write_text('x = 1\n')
        unrelated.write_text('y = 1\n')
        git_commit_all(tmp_path, 'init')
        requested.write_text('x = 2\n')
        unrelated.write_text('y = 2\n')

        result = get_repo_diff(str(tmp_path))

        assert result is not None
        assert 'requested.py' in result
        assert 'unrelated.py' in result

    @pytest.mark.parametrize(
        ('cached', 'snapshot_args'),
        [(False, []), (True, ['--cached'])],
        ids=['worktree', 'cached'],
    )
    def test_repo_diff_builds_bounded_snapshot_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        cached: bool,
        snapshot_args: list[str],
    ) -> None:
        captured_args: list[str | bytes] = []

        def capture_run(
            args: list[str | bytes], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            captured_args.extend(args)
            return subprocess.CompletedProcess(args, 0, b'', b'')

        monkeypatch.setattr(subprocess, 'run', capture_run)

        result = get_repo_diff('/repo', cached=cached)

        assert result == ''
        assert isinstance(captured_args[0], str)
        assert Path(captured_args[0]).is_absolute()
        assert captured_args[1:] == [
            '-c',
            'core.quotePath=false',
            'diff',
            '--no-color',
            '--src-prefix=a/',
            '--dst-prefix=b/',
            '-U0',
            '--inter-hunk-context=0',
            '-M',
            *snapshot_args,
            'HEAD',
            '--',
        ]

    def test_initial_commit_returns_none(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        result = get_repo_diff(str(tmp_path))
        assert result is None

    def test_no_changes_returns_empty(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        git_commit_all(tmp_path, 'init')
        result = get_repo_diff(str(tmp_path))
        assert result is not None
        assert result == ''

    def test_non_repo_returns_none(self, tmp_path: Path) -> None:
        result = get_repo_diff(str(tmp_path))
        assert result is None

    @pytest.mark.parametrize('cached', [False, True])
    def test_diff_uses_requested_snapshot(self, tmp_path: Path, cached: bool) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        path = tmp_path / 'sample.py'
        path.write_bytes(b'value = 1\n')
        git_commit_all(tmp_path, 'init')
        path.write_bytes(b'value = 2\n')
        run_setup(['git', 'add', '.'], cwd=tmp_path)
        path.write_bytes(b'value = 3\n')

        result = get_repo_diff(str(tmp_path), cached=cached)

        assert result is not None
        assert '-value = 1\n' in result
        assert f'+value = {2 if cached else 3}\n' in result

    def test_returns_diff_output(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        git_commit_all(tmp_path, 'init')
        f.write_text('x = 2\n')
        result = get_repo_diff(str(tmp_path))
        assert result is not None
        assert 'foo.py' in result

    def test_surrogateescapes_protocol_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[str | bytes] = []

        def capture_run(
            args: list[str | bytes], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            captured_args.extend(args)
            return subprocess.CompletedProcess(args, 0, b'protocol-\xfe', b'')

        monkeypatch.setattr(subprocess, 'run', capture_run)

        result = get_repo_diff('/repo')

        assert captured_args[-1] == '--'
        assert result == 'protocol-\udcfe'


class TestGetStagedPyInfo:
    def test_new_file(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
                '--allow-empty',
            ],
            cwd=str(tmp_path),
        )
        f = tmp_path / 'new.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        result = get_staged_py_info(str(tmp_path))
        assert result is not None
        info = [i for i in result if i.path == 'new.py']
        assert len(info) == 1
        assert info[0].status == 'A'
        assert info[0].old_hash
        assert set(info[0].old_hash) == {'0'}

    def test_no_staged(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        result = get_staged_py_info(str(tmp_path))
        assert result is not None
        assert result == []

    def test_staged_file_info(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        git_commit_all(tmp_path, 'init')
        f.write_text('x = 1\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        result = get_staged_py_info(str(tmp_path))
        assert result is not None
        assert len(result) == 1
        assert result[0].path == 'foo.py'
        assert result[0].mode in ('100644', '100755')
        assert result[0].status == 'M'


class TestGetTrackedFiles:
    def test_get_tracked_files_builds_bounded_query_and_selects_literal_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path_bytes = _native_path_bytes(b'non-utf8-\xff.py', b'caf\xc3\xa9.py')
        captured_args: list[str | bytes] = []

        def capture_run(
            args: list[str | bytes], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            captured_args.extend(args)
            return subprocess.CompletedProcess(
                args,
                0,
                b'tracked.py\0*.py\0unselected.py\0' + path_bytes + b'\0',
                b'',
            )

        monkeypatch.setattr(subprocess, 'run', capture_run)

        result = git_mod.get_tracked_files(
            '/repo',
            ['tracked.py', 'untracked.py', 'ignored.py', '*.py', os.fsdecode(path_bytes)],
        )

        assert result == {'tracked.py', '*.py', os.fsdecode(path_bytes)}
        assert isinstance(captured_args[0], str)
        assert Path(captured_args[0]).is_absolute()
        assert captured_args[1:] == [
            'ls-files',
            '-z',
            '--',
        ]

    def test_get_tracked_files_command_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def fail_run(
            args: list[str | bytes], **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(args, 1, b'', b'failure')

        monkeypatch.setattr(subprocess, 'run', fail_run)

        assert git_mod.get_tracked_files('/repo', ['tracked.py']) is None

    def test_get_tracked_files_returns_only_requested_tracked_paths(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        tracked = tmp_path / 'tracked.py'
        literal = tmp_path / '[literal].py'
        tracked.write_text('tracked = True\n')
        literal.write_text('literal = True\n')
        (tmp_path / '.gitignore').write_text('ignored.py\n')
        git_commit_all(tmp_path, 'tracked files')
        (tmp_path / 'untracked.py').write_text('untracked = True\n')
        (tmp_path / 'ignored.py').write_text('ignored = True\n')

        result = git_mod.get_tracked_files(
            str(tmp_path),
            ['tracked.py', 'untracked.py', 'ignored.py', '[literal].py'],
        )

        assert result == {'tracked.py', '[literal].py'}


class TestGitConfigBool:
    def test_false(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def mock_run(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='false\n')

        monkeypatch.setattr(subprocess, 'run', mock_run)
        assert _git_config_bool('test.key') is False

    def test_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def mock_run(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=1, stdout='')

        monkeypatch.setattr(subprocess, 'run', mock_run)
        assert _git_config_bool('test.key') is None

    def test_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def mock_run(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise OSError('no git')

        monkeypatch.setattr(subprocess, 'run', mock_run)
        assert _git_config_bool('test.key') is None

    def test_reads_key_from_explicit_cwd(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        run_setup(
            ['git', 'config', 'test.key', 'true'],
            cwd=tmp_path,
        )
        assert _git_config_bool('test.key', cwd=str(tmp_path)) is True

    def test_true(self, monkeypatch: pytest.MonkeyPatch) -> None:

        def mock_run(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout='true\n')

        monkeypatch.setattr(subprocess, 'run', mock_run)
        assert _git_config_bool('test.key') is True

    def test_uses_type_bool_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[list[str]] = []

        def mock_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
            captured.append(list(cmd))
            return subprocess.CompletedProcess(args=[], returncode=1, stdout='')

        monkeypatch.setattr(subprocess, 'run', mock_run)
        _git_config_bool('test.key')
        assert any('--type=bool' in cmd for cmd in captured)


class TestGitDiffPresentation:
    @pytest.mark.parametrize('cached', [False, True], ids=['worktree', 'cached'])
    @pytest.mark.parametrize(
        ('config_key', 'config_value'),
        [('diff.noprefix', 'true'), ('color.ui', 'always'), ('diff.interHunkContext', '1')],
        ids=['no-prefix', 'color', 'inter-hunk-context'],
    )
    def test_presentation_config_does_not_change_parsed_hunks_or_eligibility(
        self,
        tmp_path: Path,
        *,
        cached: bool,
        config_key: str,
        config_value: str,
    ) -> None:
        source = 'def f():\n    """Summary."""\n    before = 1\n    if before:\n        return 1\n'
        modified = source.replace('= 1', '= 2').replace('return 1', 'return 2')
        run_setup(['git', 'init'], cwd=tmp_path)
        path = tmp_path / 'sample.py'
        path.write_bytes(source.encode('utf-8'))
        git_commit_all(tmp_path, 'init')
        run_setup(
            ['git', 'config', '--local', config_key, config_value],
            cwd=tmp_path,
        )
        path.write_bytes(modified.encode('utf-8'))
        if cached:
            run_setup(
                ['git', 'add', 'sample.py'],
                cwd=tmp_path,
            )

        batch_diff = get_repo_diff(str(tmp_path), cached=cached)

        expected = [(3, 3, [b'    before = 1']), (5, 5, [b'        return 1'])]
        assert batch_diff is not None
        hunks = parse_diff_hunks(batch_diff)
        assert hunks is not None
        assert hunks == {'sample.py': expected}
        assert get_eligible_ranges(modified, hunks['sample.py'], level='definition') == {(1, 5)}


class TestGetRepoRoot:
    def test_fallback_on_non_repo(self, tmp_path: Path) -> None:
        result = get_repo_root(str(tmp_path))
        assert result == str(tmp_path)

    @pytest.mark.parametrize('repo_name', ['repo', 'repo '], ids=['ordinary', 'trailing-space'])
    def test_preserves_exact_toplevel_path(self, tmp_path: Path, repo_name: str) -> None:
        repo = tmp_path / repo_name
        repo.mkdir()
        run_setup(['git', 'init'], cwd=repo)

        result = get_repo_root(str(repo))

        assert Path(result).resolve() == repo.resolve()
        assert Path(result).name == repo.resolve().name

    def test_returns_toplevel(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        result = get_repo_root(str(tmp_path))
        assert Path(result).resolve() == tmp_path.resolve()


class TestGitExecutableBoundary:
    @pytest.mark.parametrize('boundary', ['text', 'bytes', 'add'])
    def test_uses_absolute_git_and_preserves_arguments(
        self, boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[tuple[list[str | bytes], object]] = []

        def capture_run(
            args: list[str | bytes], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            captured.append((args, kwargs.get('cwd')))
            return subprocess.CompletedProcess(args, 0, b'', b'')

        monkeypatch.setattr(subprocess, 'run', capture_run)
        cwd = str(tmp_path / 'child directory')
        tail = ['-C', 'nested repo', 'add', '--', 'space name.py']
        if boundary == 'text':
            assert _run_git_command(['git', *tail], cwd) is not None
        elif boundary == 'bytes':
            assert _run_git_command_bytes(['git', *tail, b'raw-\xff.py'], cwd) is not None
        else:
            add = PreCommitAdd(tokens=('git', *tail), exec_cwd=cwd, git_cwd=str(tmp_path))
            assert run_pre_commit_adds((add,))
        assert len(captured) == 1
        argv, child_cwd = captured[0]
        assert isinstance(argv[0], str)
        assert Path(argv[0]).is_absolute()
        assert argv[1:] == ([*tail, b'raw-\xff.py'] if boundary == 'bytes' else tail)
        assert child_cwd == cwd

    @pytest.mark.parametrize('boundary', ['text', 'bytes', 'add'])
    def test_missing_git_retains_failure_result(
        self, boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PATH', str(tmp_path))

        def unexpected_run(*_args: object, **_kwargs: object) -> None:
            pytest.fail('No executable should be launched when Git cannot be resolved')

        monkeypatch.setattr(subprocess, 'run', unexpected_run)
        cwd = str(tmp_path)
        if boundary == 'text':
            assert _run_git_command(['git', '--version'], cwd) is None
        elif boundary == 'bytes':
            assert _run_git_command_bytes(['git', '--version'], cwd) is None
        else:
            add = PreCommitAdd(tokens=('git', 'add', '.'), exec_cwd=cwd, git_cwd=cwd)
            assert run_pre_commit_adds((add,)) is False


class TestGitPathParsing:
    def test_modified_path_fixture_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path_bytes = _native_path_bytes(b'modified-\xff.py', b'modified-caf\xc3\xa9.py')

        def text_run(
            _args: list[str], _cwd: str | None = None, **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, 'HEAD\n', '')

        def bytes_run(
            _args: list[str | bytes], _cwd: str, **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess([], 0, path_bytes + b'\0', b'')

        monkeypatch.setattr('docwrap._git._run_git_command', text_run)
        monkeypatch.setattr('docwrap._git._run_git_command_bytes', bytes_run)

        result = get_modified_tracked_py_files('/repo')

        assert result is not None
        assert os.fsencode(result[0]) == path_bytes
        assert '\ufffd' not in result[0]

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='POSIX bytes-path behavior is not available on Windows',
    )
    def test_non_utf8_paths_round_trip_across_git_discovery(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        root = os.fsencode(tmp_path)
        tracked_bytes = b'tracked-\xff.py'
        staged_bytes = b'staged-\xfe.py'
        untracked_bytes = b'untracked-\xfd.py'

        _write_bytes_file(tmp_path, tracked_bytes, b'x = 1\n')
        git_commit_all(tmp_path, 'init')
        _write_bytes_file(tmp_path, tracked_bytes, b'x = 2\n')
        _write_bytes_file(tmp_path, staged_bytes, b'y = 1\n')
        run_setup(
            [b'git', b'add', b'--', staged_bytes],
            cwd=root,
        )
        _write_bytes_file(tmp_path, untracked_bytes, b'z = 1\n')

        tracked = os.fsdecode(tracked_bytes)
        staged = os.fsdecode(staged_bytes)
        untracked = os.fsdecode(untracked_bytes)
        resolved = resolve_pathspec_py_files(str(tmp_path), [tracked], repo_root=str(tmp_path))
        modified = get_modified_tracked_py_files(str(tmp_path))
        staged_info = get_staged_py_info(str(tmp_path))
        tracked_paths = git_mod.get_tracked_files(str(tmp_path), [tracked, staged, untracked])

        assert resolved is not None
        assert os.fsencode(resolved[0]) == tracked_bytes
        assert modified is not None
        assert tracked in modified
        assert staged_info is not None
        assert any(info.path == staged for info in staged_info)
        assert tracked_paths == {tracked, staged}
        for actual, expected in (
            (tracked, tracked_bytes),
            (staged, staged_bytes),
            (untracked, untracked_bytes),
        ):
            assert '\ufffd' not in actual
            assert os.fsencode(actual) == expected

    def test_staged_path_fixture_round_trips(self) -> None:
        path_bytes = _native_path_bytes(b'staged-\xfe.py', b'staged-caf\xc3\xa9.py')
        result = subprocess.CompletedProcess([], 0, path_bytes + b'\0', b'')

        paths = _parse_null_separated_py_paths(result)

        assert paths is not None
        assert os.fsencode(paths[0]) == path_bytes
        assert '\ufffd' not in paths[0]

    def test_tracked_path_fixture_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path_bytes = _native_path_bytes(b'tracked-\xfd.py', b'tracked-caf\xc3\xa9.py')

        def bytes_run(
            _args: list[str | bytes], _cwd: str, **_kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess([], 0, path_bytes + b'\0', b'')

        def exists(_path: Path) -> bool:
            return True

        def is_symlink(_path: Path) -> bool:
            return False

        monkeypatch.setattr('docwrap._git._run_git_command_bytes', bytes_run)
        monkeypatch.setattr(Path, 'exists', exists)
        monkeypatch.setattr(Path, 'is_symlink', is_symlink)

        paths = resolve_pathspec_py_files('/repo', ['*.py'], repo_root='/repo')

        assert paths is not None
        assert os.fsencode(paths[0]) == path_bytes
        assert '\ufffd' not in paths[0]


class TestIsHookEnabled:
    def test_docwrap_hook_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def disabled(key: str) -> bool | None:
            return False if key == 'docwrap.hook' else None

        monkeypatch.setattr(
            'docwrap._git._git_config_bool',
            disabled,
        )
        assert is_hook_enabled() is False

    def test_docwrap_hook_false_uses_explicit_cwd(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        run_setup(
            ['git', 'config', 'docwrap.hook', 'false'],
            cwd=tmp_path,
        )
        assert is_hook_enabled(cwd=str(tmp_path)) is False

    def test_docwrap_hook_true_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def enabled(key: str) -> bool | None:
            return True if key == 'docwrap.hook' else None

        monkeypatch.setattr(
            'docwrap._git._git_config_bool',
            enabled,
        )
        assert is_hook_enabled() is True

    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unset(_key: str) -> bool | None:
            return None

        monkeypatch.setattr(
            'docwrap._git._git_config_bool',
            unset,
        )
        assert is_hook_enabled() is True


class TestPrepareIndexEntry:
    def test_creates_blob(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        result = prepare_index_entry(str(tmp_path), 'foo.py', 'x = 1\n', '100644')
        assert result is not None
        mode, blob_hash, path = result
        assert mode == '100644'
        assert len(blob_hash) >= 40
        assert path == 'foo.py'

    def test_crlf_round_trip(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        entry = prepare_index_entry(
            str(tmp_path), 'foo.py', 'x = 1\ny = 2\n', '100644', newline='\r\n'
        )
        assert entry is not None
        result = run_setup(
            ['git', 'cat-file', '-p', entry.blob_hash],
            cwd=str(tmp_path),
        )
        raw = result.stdout
        assert b'\r\n' in raw
        assert raw.replace(b'\r\n', b'') == raw.replace(b'\r\n', b'').replace(b'\n', b'')


class TestRawDiffParsing:
    def test_metadata_uses_surrogateescape_separately_from_paths(self) -> None:
        path_bytes = _native_path_bytes(b'module-\xfe.py', b'module-caf\xc3\xa9.py')
        header_bytes = b':100644 100644 ' + b'a' * 40 + b' ' + b'b' * 40 + b' M\xff'

        records = list(_iter_raw_diff_records(header_bytes + b'\0' + path_bytes + b'\0'))

        assert len(records) == 1
        header, paths = records[0]
        assert header.endswith('M\udcff')
        assert os.fsencode(paths[0]) == path_bytes
        assert '\ufffd' not in header

    def test_rename_paths_are_decoded_individually(self) -> None:
        source = _native_path_bytes(b'old-\xff.py', b'old-caf\xc3\xa9.py')
        target = _native_path_bytes(b'new-\xfe.py', b'new-caf\xc3\xa9.py')
        header = b':100644 100644 ' + b'a' * 40 + b' ' + b'b' * 40 + b' R100'

        records = list(_iter_raw_diff_records(header + b'\0' + source + b'\0' + target + b'\0'))

        assert len(records) == 1
        _, paths = records[0]
        assert tuple(os.fsencode(path) for path in paths) == (source, target)
        assert all('\ufffd' not in path for path in paths)


class TestReadStagedBlobs:
    def test_empty_list(self, tmp_path: Path) -> None:
        blobs = read_staged_blobs(str(tmp_path), [])
        assert blobs == {}

    def test_read_blob(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        git_commit_all(tmp_path, 'init')
        f.write_text('x = 2\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        info = get_staged_py_info(str(tmp_path))
        assert info is not None
        blobs = read_staged_blobs(str(tmp_path), info)
        assert blobs is not None
        baseline = blobs['foo.py']
        assert baseline.status is BaselineStatus.CONTENT
        assert baseline.content == 'x = 2\n'
        assert baseline.encoding == 'utf-8'

    def test_read_blob_crlf_metadata(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
                '--allow-empty',
            ],
            cwd=str(tmp_path),
        )
        crlf_content = b'x = 1\r\nprint("hello")\r\n'
        result = run_setup(
            ['git', 'hash-object', '-w', '--stdin'],
            stdin=crlf_content,
            cwd=str(tmp_path),
        )
        blob_hash = result.stdout.decode('ascii').strip()
        run_setup(
            ['git', 'update-index', '--add', '--cacheinfo', f'100644,{blob_hash},foo.py'],
            cwd=str(tmp_path),
        )
        info = [StagedFileInfo('foo.py', '100644', blob_hash, '0' * 40, 'A')]
        blobs = read_staged_blobs(str(tmp_path), info)
        assert blobs is not None
        baseline = blobs['foo.py']
        assert baseline.status is BaselineStatus.CONTENT
        assert baseline.newline == '\r\n'
        assert '\r' not in (baseline.content or '')


class TestResolvePathspecPyFiles:
    @pytest.mark.parametrize(
        ('record', 'expected'),
        [
            (b'"tab\\tname.py"', b'tab\tname.py'),
            (b'plain-\xff.py', b'plain-\xff.py'),
            (b'"quote\\"slash\\\\.py"', b'quote"slash\\.py'),
            (b'"caf\\303\\251.py"', b'caf\xc3\xa9.py'),
        ],
    )
    def test_decodes_c_style_quoted_record(self, record: bytes, expected: bytes) -> None:
        assert _decode_git_pathspec_record(record) == expected

    def test_filter_non_py(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        (tmp_path / 'readme.md').write_text('# hi\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        result = resolve_pathspec_py_files(str(tmp_path), ['readme.md'], repo_root=str(tmp_path))
        assert result is not None
        assert result == []

    @pytest.mark.parametrize(
        'record',
        [
            b'"bad\\q.py"',
            b'"bad\\12"',
            b'"bad.py',
            b'"bad.py"x',
        ],
    )
    def test_invalid_c_style_quoted_record_fails(self, record: bytes) -> None:
        assert _decode_git_pathspec_record(record) is None

    def test_nul_pathspec_file_keeps_double_quotes_literal(self, tmp_path: Path) -> None:
        (tmp_path / 'paths.txt').write_bytes(b'"quoted".py\0')

        assert _read_git_pathspec_file(str(tmp_path), 'paths.txt', pathspec_file_nul=True) == [
            '"quoted".py'
        ]

    def test_nul_pathspec_file_keeps_metacharacters_literal(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        filename = '[quoted].py'
        (tmp_path / filename).write_text('value = 1\n')
        run_setup(['git', 'add', '--', filename], cwd=tmp_path)
        (tmp_path / 'paths.txt').write_bytes(os.fsencode(filename) + b'\0')

        assert resolve_pathspec_py_files(
            str(tmp_path),
            [],
            repo_root=str(tmp_path),
            pathspec_from_file='paths.txt',
            pathspec_file_nul=True,
        ) == [filename]

    def test_resolve_file(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        result = resolve_pathspec_py_files(str(tmp_path), ['foo.py'], repo_root=str(tmp_path))
        assert result is not None
        assert 'foo.py' in result

    @pytest.mark.parametrize('line_ending', [b'\n', b'\r\n'], ids=['lf', 'crlf'])
    def test_resolve_quoted_space_from_newline_pathspec_file(
        self, tmp_path: Path, line_ending: bytes
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        (tmp_path / 'foo bar.py').write_text('value = 1\n')
        (tmp_path / 'paths.txt').write_bytes(b'"foo bar.py"' + line_ending)
        run_setup(
            ['git', 'add', '--pathspec-from-file=paths.txt'],
            cwd=tmp_path,
        )

        assert resolve_pathspec_py_files(
            str(tmp_path),
            [],
            repo_root=str(tmp_path),
            pathspec_from_file='paths.txt',
        ) == ['foo bar.py']

    def test_stdin_returns_empty(self, tmp_path: Path) -> None:
        result = resolve_pathspec_py_files(
            str(tmp_path), [], repo_root=str(tmp_path), pathspec_from_file='-'
        )
        assert result == []


class TestRunPreCommitAdds:
    def test_adds_files_to_index(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
                '--allow-empty',
            ],
            cwd=str(tmp_path),
        )
        f = tmp_path / 'foo.py'
        f.write_text('x = 1\n')
        add = PreCommitAdd(
            tokens=('git', 'add', 'foo.py'),
            exec_cwd=str(tmp_path),
            git_cwd=str(tmp_path),
        )
        result = run_pre_commit_adds((add,))
        staged = get_staged_py_info(str(tmp_path))
        assert result is True
        assert staged is not None
        assert any(info.path == 'foo.py' for info in staged)

    def test_empty_list_is_noop(self) -> None:
        assert run_pre_commit_adds(()) is True

    def test_failed_add_does_not_raise(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        add = PreCommitAdd(
            tokens=('git', 'add', 'nonexistent.py'),
            exec_cwd=str(tmp_path),
            git_cwd=str(tmp_path),
        )
        assert run_pre_commit_adds((add,)) is False

    def test_failure_preserves_prior_successful_add(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
                '--allow-empty',
            ],
            cwd=tmp_path,
        )
        (tmp_path / 'first.py').write_text('first = 1\n')
        adds = (
            PreCommitAdd(
                tokens=('git', 'add', 'first.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
            PreCommitAdd(
                tokens=('git', 'add', 'missing.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
        )

        assert run_pre_commit_adds(adds) is False
        staged = get_staged_py_info(str(tmp_path))
        assert staged is not None
        assert any(info.path == 'first.py' for info in staged)

    def test_failure_stops_before_next_add(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[list[str]] = []

        def fail_first(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[list[str]]:
            calls.append(args)
            return subprocess.CompletedProcess(args, 1)

        monkeypatch.setattr(subprocess, 'run', fail_first)
        adds = (
            PreCommitAdd(
                tokens=('git', 'add', 'first.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
            PreCommitAdd(
                tokens=('git', 'add', 'second.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
        )

        assert run_pre_commit_adds(adds) is False
        assert len(calls) == 1
        assert Path(calls[0][0]).is_absolute()
        assert calls[0][1:] == ['add', 'first.py']

    def test_multiple_add_lists(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'init',
                '--allow-empty',
            ],
            cwd=str(tmp_path),
        )
        (tmp_path / 'a.py').write_text('a = 1\n')
        (tmp_path / 'b.py').write_text('b = 2\n')
        adds = (
            PreCommitAdd(
                tokens=('git', 'add', 'a.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
            PreCommitAdd(
                tokens=('git', 'add', 'b.py'),
                exec_cwd=str(tmp_path),
                git_cwd=str(tmp_path),
            ),
        )
        result = run_pre_commit_adds(adds)
        staged = get_staged_py_info(str(tmp_path))
        assert result is True
        assert staged is not None
        assert any(info.path == 'a.py' for info in staged)
        assert any(info.path == 'b.py' for info in staged)

    def test_os_error_returns_false(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

        def raise_os_error(*_args: object, **_kwargs: object) -> None:
            raise OSError('injected failure')

        monkeypatch.setattr(subprocess, 'run', raise_os_error)
        add = PreCommitAdd(
            tokens=('git', 'add', 'foo.py'),
            exec_cwd=str(tmp_path),
            git_cwd=str(tmp_path),
        )

        assert run_pre_commit_adds((add,)) is False

    def test_replays_parser_tokens_in_execution_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        target = tmp_path / 'file with spaces.py'
        target.write_text('value = 1\n')
        plan = classify_grammar(
            parse_command('git add -- "file with spaces.py" && git commit -m "test"'),
            str(tmp_path),
        )
        assert isinstance(plan, CommitPlan)
        add = plan.pre_commit_adds[0]
        calls: list[list[str]] = []
        original_run = subprocess.run

        def record_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            assert kwargs == {
                'capture_output': True,
                'check': False,
                'cwd': add.exec_cwd,
            }
            calls.append(args)
            return original_run(
                args,
                cwd=add.exec_cwd,
                capture_output=True,
                check=False,
            )

        with monkeypatch.context() as patch:
            patch.setattr(subprocess, 'run', record_run)
            result = run_pre_commit_adds(plan.pre_commit_adds)

        assert result is True
        assert len(calls) == 1
        assert Path(calls[0][0]).is_absolute()
        assert calls[0][1:] == list(add.tokens[1:])
        staged = get_staged_py_info(str(tmp_path))
        assert staged is not None
        assert [info.path for info in staged] == ['file with spaces.py']


def _native_path_bytes(posix: bytes, windows: bytes) -> bytes:
    return posix if os.name != 'nt' else windows


def _write_bytes_file(root: Path, name: bytes, content: bytes) -> None:
    try:
        descriptor = os.open(
            os.fsencode(root / os.fsdecode(name)),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
    except OSError as exc:
        if exc.errno == errno.EILSEQ:
            pytest.skip('filesystem cannot create non-UTF-8 filenames')
        raise
    try:
        os.write(descriptor, content)
    finally:
        os.close(descriptor)
