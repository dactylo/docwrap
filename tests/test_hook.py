import errno
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

import docwrap._hook as hook_mod
import docwrap._hook_front as front_mod
from docwrap._core._incremental import DiffHunk
from docwrap._exceptions import UnknownStyleError
from docwrap._files import ReadResult, read_file_result, write_preserving_newline
from docwrap._git import IndexEntry, RepoDiscoveryResult, RepoDiscoveryStatus, StagedFileInfo
from docwrap._hook import (
    _PreToolContext,
    _print_skips,
    _resolve_edited_paths,
    _wrap_staged_blobs,
    _wrap_worktree_files,
    _WrapResult,
)
from docwrap._hook_front import _REASON_PREFIX, Outcome, fail, run_hook
from docwrap._hosts import load_adapter
from docwrap._types import (
    BaselineResult,
    BaselineStatus,
    DocstringSkip,
    WorktreeWrapContext,
    WorktreeWrite,
)
from docwrap._vendor.hookshell._grammar import (
    _ELEMENT_UNKNOWN_CWD,
    _HINT_COMMIT,
    _HINT_ROUTE,
    CommitMode,
    PreCommitAdd,
)
from docwrap.hooks import HookAdapter, HookResult, HostInput, HostResponse
from tests.helpers import (
    BUILTIN_ENTRY_POINTS,
    decode_text,
    git_commit_all,
    hook_event,
    run_setup,
    shell_path,
)
from tests.hosts_example.adapter import ExampleAdapter

BACKSLASH = chr(92)
_CONTROL_TEXT = 'esc\x1bcr\rtab\tdel\x7fc1\x85'
_ESCAPE_MIXED_SOURCE = (
    'def skipme():\n'
    '    """Line one ' + BACKSLASH + '\n'
    '    continued."""\n'
    '\n'
    '\n'
    'def wrapme(value):\n'
    '    """Return the value.\n\n'
    '    Args:\n'
    '        value: This is a very long argument description that should '
    'definitely be wrapped.\n'
    '    """\n'
    '    return value\n'
)
_LAST_RESORT_PAYLOAD = json.dumps(
    {
        'hook_event_name': 'PreToolUse',
        'tool_name': 'Bash',
        'tool_input': {'command': 'ls'},
        'cwd': '.',
    }
)
_POST_TOOL_WRAPPED_SOURCE = (
    'def f():\n'
    '    """A cafe summary.\n'
    '\n'
    '    This body prose is deliberately long enough to exceed the configured\n'
    '    default line length and require wrapping by docwrap.\n'
    '    """\n'
    '    pass\n'
)
_RECEIPT_ONE = {'systemMessage': 'docwrap: wrapped 1 staged Python file for this commit'}
_UNKNOWN_CWD_REASON = f'{_REASON_PREFIX}: {_ELEMENT_UNKNOWN_CWD}; {_HINT_ROUTE}'
_VISIBLE_CONTROL_TEXT = r'esc\x1bcr\x0dtab\x09del\x7fc1\x85'


class TestEngineImportFailure:
    def test_commit_denies_when_engine_import_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, 'docwrap._hook', None)
        outcome = front_mod._run_pre_tool_use(
            'git commit -m message', str(tmp_path), allow_on_failure=False, incremental=None
        )
        assert (outcome.decision, outcome.reason) == (
            'deny',
            'docwrap: unexpected error during wrapping',
        )

    def test_commit_engine_call_import_error_is_not_handled_as_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def raise_import_error(*_args: object, **_kwargs: object) -> None:
            raise ImportError('engine call failed')

        monkeypatch.setattr(hook_mod, 'run_commit', raise_import_error)
        with pytest.raises(ImportError, match='engine call failed'):
            front_mod._run_pre_tool_use(
                'git commit -m message', str(tmp_path), allow_on_failure=False, incremental=None
            )

    def test_commit_warns_and_allows_with_allow_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setitem(sys.modules, 'docwrap._hook', None)
        outcome = front_mod._run_pre_tool_use(
            'git commit -m message', str(tmp_path), allow_on_failure=True, incremental=None
        )
        assert outcome.decision == 'allow'
        assert 'WARNING: docwrap: unexpected error during wrapping' in capsys.readouterr().err

    def test_post_tool_allows_with_diagnostic_when_engine_import_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setitem(sys.modules, 'docwrap._hook', None)
        outcome = front_mod._run_post_tool((), str(tmp_path), incremental=None)
        assert outcome.decision == 'allow'
        assert 'docwrap: unexpected error during PostToolUse' in capsys.readouterr().err

    def test_post_tool_engine_call_import_error_is_not_handled_as_import_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def raise_import_error(*_args: object, **_kwargs: object) -> None:
            raise ImportError('engine call failed')

        monkeypatch.setattr(hook_mod, 'run_post_tool', raise_import_error)
        with pytest.raises(ImportError, match='engine call failed'):
            front_mod._run_post_tool((), str(tmp_path), incremental=None)


class TestPreToolUseDispatchers:
    def test_staged_diff_batches_cached_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        info_list = [
            StagedFileInfo('a.py', '100644', 'a' * 40, '0' * 40, 'M'),
            StagedFileInfo('b.py', '100644', 'b' * 40, '0' * 40, 'M'),
        ]
        contents = {
            'a.py': BaselineResult(
                status=BaselineStatus.CONTENT,
                content='def a():\n    """Short."""\n    pass\n',
            ),
            'b.py': BaselineResult(
                status=BaselineStatus.CONTENT,
                content='def b():\n    """Short."""\n    pass\n',
            ),
        }
        batch_calls: list[tuple[str, bool]] = []
        parse_calls: list[str] = []
        eligible_by_file: dict[str, set[tuple[int, int]] | None] = {}

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> str:
            batch_calls.append((repo_root, cached))
            return 'cached snapshot'

        def parse_diff(diff_output: str) -> dict[str, list[DiffHunk]]:
            parse_calls.append(diff_output)
            return {'a.py': [(1, 1, [])]}

        def read_staged(
            _cwd: str,
            _info_list: list[StagedFileInfo],
        ) -> dict[str, BaselineResult]:
            return contents

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            name = 'a.py' if 'def a' in source else 'b.py'
            eligible_by_file[name] = eligible_ranges
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, 'read_staged_blobs', read_staged)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        mode = CommitMode(
            auto_stage=False,
            include=False,
            only=False,
            pathspecs=(),
            pathspec_from_file=None,
            pathspec_file_nul=False,
        )
        ctx = _PreToolContext(
            cwd=str(tmp_path),
            repo_root=str(tmp_path),
            incr_level='definition',
            allow_on_failure=False,
            mode=mode,
            pre_commit_adds=(),
        )

        assert _wrap_staged_blobs(ctx, info_list) == ([], [])
        assert batch_calls == [(str(tmp_path), True)]
        assert parse_calls == ['cached snapshot']
        assert eligible_by_file == {'a.py': {(1, 3)}, 'b.py': set()}

    def test_staged_diff_failure_falls_back_to_full_wrap(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        info_list = [
            StagedFileInfo('a.py', '100644', 'a' * 40, '0' * 40, 'M'),
            StagedFileInfo('b.py', '100644', 'b' * 40, '0' * 40, 'M'),
        ]
        contents = {
            'a.py': BaselineResult(status=BaselineStatus.CONTENT, content='x = 1\n'),
            'b.py': BaselineResult(status=BaselineStatus.CONTENT, content='y = 1\n'),
        }
        batch_calls: list[tuple[str, bool]] = []
        eligible_ranges_seen: list[set[tuple[int, int]] | None] = []

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> None:
            batch_calls.append((repo_root, cached))

        def fail_parse(_diff: str) -> dict[str, list[DiffHunk]]:
            pytest.fail('failed diff must not be parsed')

        def read_staged(
            _cwd: str,
            _info_list: list[StagedFileInfo],
        ) -> dict[str, BaselineResult]:
            return contents

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            eligible_ranges_seen.append(eligible_ranges)
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', fail_parse)
        monkeypatch.setattr(hook_mod, 'read_staged_blobs', read_staged)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        mode = CommitMode(
            auto_stage=False,
            include=False,
            only=False,
            pathspecs=(),
            pathspec_from_file=None,
            pathspec_file_nul=False,
        )
        ctx = _PreToolContext(
            cwd=str(tmp_path),
            repo_root=str(tmp_path),
            incr_level='definition',
            allow_on_failure=False,
            mode=mode,
            pre_commit_adds=(),
        )

        assert _wrap_staged_blobs(ctx, info_list) == ([], [])
        assert batch_calls == [(str(tmp_path), True)]
        assert eligible_ranges_seen == [None, None]


class TestProcessFileReadEfficiency:
    def test_process_file_uses_shared_reader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        path.write_text(_post_tool_long_source())
        calls: list[Path] = []

        def tracked_reader(candidate: Path) -> ReadResult:
            calls.append(candidate)
            return read_file_result(candidate)

        monkeypatch.setattr(hook_mod, 'read_file_result', tracked_reader)

        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        assert calls == [path]


class TestProtocolHelpers:
    def test_fail_controls_are_visible_in_warning_and_deny(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reason = f'git failure {_CONTROL_TEXT}'

        assert fail(reason, allow_on_failure=True) == Outcome('allow')
        captured = capsys.readouterr()
        assert captured.err == f'WARNING: git failure {_VISIBLE_CONTROL_TEXT}\n'
        assert captured.out == ''

        assert fail(reason, allow_on_failure=False) == Outcome(
            'deny', f'git failure {_VISIBLE_CONTROL_TEXT}'
        )
        assert capsys.readouterr() == ('', '')

    def test_fail_deny(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert fail('git error', allow_on_failure=False) == Outcome('deny', 'git error')
        assert capsys.readouterr() == ('', '')

    def test_fail_warns_and_allows(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = fail('git error', allow_on_failure=True)
        assert result == Outcome('allow')
        captured = capsys.readouterr()
        assert 'WARNING' in captured.err
        assert 'git error' in captured.err
        assert captured.out == ''  # No JSON (implicit allow)

    def test_skip_controls_are_visible(self, capsys: pytest.CaptureFixture[str]) -> None:
        skip = DocstringSkip(
            path='',
            line=3,
            qualname=f'name-{_CONTROL_TEXT}',
            code='line-continuation',
            message=f'message-{_CONTROL_TEXT}',
        )

        _print_skips(f'path-{_CONTROL_TEXT}.py', (skip,))

        assert capsys.readouterr().err == (
            f'docwrap: path-{_VISIBLE_CONTROL_TEXT}.py:3: name-{_VISIBLE_CONTROL_TEXT} '
            f'skipped (line-continuation): message-{_VISIBLE_CONTROL_TEXT}\n'
        )


class TestWrapWorktreeFiles:
    @pytest.mark.parametrize(
        'filename',
        ['slash\\name.py', 'space name.py'] if os.name != 'nt' else ['space name.py'],
    )
    def test_batched_worktree_wraps_quoted_filename(self, tmp_path: Path, filename: str) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / filename
        path.write_text(_post_tool_long_source(text='coffee'))
        git_commit_all(tmp_path, 'add quoted path')
        path.write_text(_post_tool_long_source())

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            [filename],
        )

        assert result.failure is None
        assert path.read_text() == _POST_TOOL_WRAPPED_SOURCE

    def test_changed_file_uses_shared_reader(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        original = _post_tool_long_source().replace('\n', '\r\n').encode()
        path.write_bytes(original)
        real_read_bytes = Path.read_bytes
        real_reader = read_file_result
        real_writer = hook_mod.write_preserving_newline
        reader_calls: list[Path] = []
        writer_raw: list[bytes | None] = []
        physical_reads = 0

        def tracked_reader(candidate: Path) -> ReadResult:
            reader_calls.append(candidate)
            return real_reader(candidate)

        def tracked_read_bytes(candidate: Path) -> bytes:
            nonlocal physical_reads
            if candidate == path:
                physical_reads += 1
            return real_read_bytes(candidate)

        def tracked_writer(
            candidate: Path,
            source: str,
            encoding: str,
            *,
            raw: bytes,
        ) -> None:
            writer_raw.append(raw)
            real_writer(candidate, source, encoding, raw=raw)

        monkeypatch.setattr(hook_mod, 'read_file_result', tracked_reader)
        monkeypatch.setattr(Path, 'read_bytes', tracked_read_bytes)
        monkeypatch.setattr(hook_mod, 'write_preserving_newline', tracked_writer)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=False,
            ),
            ['source.py'],
        )

        assert result.failure is None
        assert reader_calls == [path]
        assert physical_reads == 1
        assert writer_raw == [original]

    def test_empty_paths_succeeds(self, tmp_path: Path) -> None:
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            [],
        )
        assert result.failure is None

    def test_malformed_diff_path_falls_back_to_full_wrap(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        path.write_text(_post_tool_long_source())
        eligible_ranges_seen: list[set[tuple[int, int]] | None] = []
        malformed_diff = (
            'diff --git ignored ignored\n--- "a/bad\\q.py"\n+++ "b/bad\\q.py"\n@@ -1 +1 @@\n+x\n'
        )

        def batch_diff(
            _repo_root: str,
            *,
            cached: bool = False,  # noqa: ARG001
        ) -> str:
            return malformed_diff

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            eligible_ranges_seen.append(eligible_ranges)
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['source.py'],
        )

        assert result.failure is None
        assert eligible_ranges_seen == [None]

    def test_no_changes_succeeds(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('def foo():\n    """Short."""\n    pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        f.write_text('def foo():\n    """Short."""\n    x = 1\n')

        before = f.read_text()
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['foo.py'],
        )
        assert result.failure is None
        assert f.read_text() == before

    def test_skips_symlinks(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        real_file = tmp_path / 'real.py'
        _write_long_arg_docstring(real_file)
        link = tmp_path / 'link.py'
        link.symlink_to(real_file)
        regular_file = tmp_path / 'regular.py'
        regular_file.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )

        _write_long_arg_docstring(regular_file)
        regular_source = regular_file.read_text()
        regular_result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['regular.py'],
        )
        assert regular_result.failure is None
        assert regular_file.read_text() != regular_source

        original_content = real_file.read_bytes()
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=False,
            ),
            ['link.py'],
        )
        assert result.failure is None
        assert real_file.read_bytes() == original_content
        assert link.is_symlink()

    def test_worktree_diff_batches_snapshot_and_distributes_hunks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / 'a.py').write_text('def a():\n    """Short."""\n    pass\n')
        (tmp_path / 'b.py').write_text('def b():\n    """Short."""\n    pass\n')
        batch_calls: list[tuple[str, bool]] = []
        parse_calls: list[str] = []
        eligible_by_file: dict[str, set[tuple[int, int]] | None] = {}

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> str:
            batch_calls.append((repo_root, cached))
            return 'worktree snapshot'

        def parse_diff(diff_output: str) -> dict[str, list[DiffHunk]]:
            parse_calls.append(diff_output)
            return {'b.py': [(1, 1, [])]}

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            name = 'a.py' if 'def a' in source else 'b.py'
            eligible_by_file[name] = eligible_ranges
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff, raising=False)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff, raising=False)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['a.py', 'b.py'],
        )

        assert result.failure is None
        assert batch_calls == [(str(tmp_path), False)]
        assert parse_calls == ['worktree snapshot']
        assert eligible_by_file == {'a.py': set(), 'b.py': {(1, 3)}}

    def test_worktree_diff_failure_falls_back_to_full_wrap(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / 'a.py').write_text('def a():\n    """Short."""\n    pass\n')
        (tmp_path / 'b.py').write_text('def b():\n    """Short."""\n    pass\n')
        batch_calls: list[tuple[str, bool]] = []
        eligible_ranges_seen: list[set[tuple[int, int]] | None] = []

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> None:
            batch_calls.append((repo_root, cached))

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            eligible_ranges_seen.append(eligible_ranges)
            return _WrapResult(source=source, changed=False)

        def fail_parse(_diff: str) -> dict[str, list[DiffHunk]]:
            pytest.fail('failed diff must not be parsed')

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff, raising=False)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', fail_parse, raising=False)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['a.py', 'b.py'],
        )

        assert result.failure is None
        assert batch_calls == [(str(tmp_path), False)]
        assert eligible_ranges_seen == [None, None]

    def test_wraps_modified_file(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        _write_long_arg_docstring(f)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level='definition',
                allow_on_failure=False,
            ),
            ['foo.py'],
        )
        assert result.failure is None
        content = f.read_text()
        lines = content.splitlines()
        assert all(len(line) <= 79 for line in lines)


class TestWrapWorktreeFilesReadError:
    @pytest.mark.parametrize('allow_on_failure', [False, True])
    def test_bad_encoding_cookie_preserves_failure_policy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        *,
        allow_on_failure: bool,
    ) -> None:
        bad = tmp_path / 'bad.py'
        bad.write_bytes(b'# coding: unknown-encoding\npass\n')
        good = tmp_path / 'good.py'
        _write_long_arg_docstring(good)
        original = good.read_bytes()

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=allow_on_failure,
            ),
            ['bad.py', 'good.py'],
        )

        assert bad.read_bytes() == b'# coding: unknown-encoding\npass\n'
        if allow_on_failure:
            assert result.failure is None
            assert good.read_bytes() != original
            assert 'WARNING: docwrap: cannot read bad.py' in capsys.readouterr().err
        else:
            assert result.failure is not None
            assert result.failure.decision == 'deny'
            assert good.read_bytes() == original

    def test_read_error_allow_warns_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')

        original_read_bytes = Path.read_bytes

        def failing_read_bytes(self_path: Path) -> bytes:
            if self_path.name == 'foo.py':
                raise OSError('simulated read failure')
            return original_read_bytes(self_path)

        monkeypatch.setattr(Path, 'read_bytes', failing_read_bytes)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=True,
            ),
            ['foo.py'],
        )
        assert result.failure is None
        captured = capsys.readouterr()
        assert 'WARNING' in captured.err
        assert 'foo.py' in captured.err

    def test_read_error_deny_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / 'foo.py'
        _write_long_arg_docstring(f)

        original_read_bytes = Path.read_bytes

        def failing_read_bytes(self_path: Path) -> bytes:
            if self_path.name == 'foo.py':
                raise OSError('simulated read failure')
            return original_read_bytes(self_path)

        monkeypatch.setattr(Path, 'read_bytes', failing_read_bytes)

        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=False,
            ),
            ['foo.py'],
        )
        assert result.failure is not None
        assert result.failure.decision == 'deny'


class TestWrapWorktreeRollback:
    def test_allow_on_failure_skips_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        long_docstring = _post_tool_long_source()
        a.write_text(long_docstring)
        b.write_text(long_docstring)

        original_b = b.read_bytes()

        def fail_second_write(path: Path, source: str, encoding: str, *, raw: bytes) -> None:
            if path == b:
                raise OSError('simulated write failure')
            write_preserving_newline(path, source, encoding, raw=raw)

        monkeypatch.setattr(hook_mod, 'write_preserving_newline', fail_second_write)
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=True,
            ),
            ['a.py', 'b.py'],
        )
        assert result.failure is None
        assert a.read_text() == _POST_TOOL_WRAPPED_SOURCE
        assert b.read_bytes() == original_b
        assert capsys.readouterr().err == f'WARNING: docwrap: failed to write {b.as_posix()}\n'

    def test_failed_restore_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        original = _post_tool_long_source().encode()
        a.write_bytes(original)
        b.write_bytes(original)
        original_write_bytes = Path.write_bytes
        calls = 0

        def failed_writes(path: Path, data: bytes) -> int:
            nonlocal calls
            if path == b:
                calls += 1
                if calls == 1:
                    original_write_bytes(path, data[: len(data) // 2])
                    raise OSError(errno.ENOSPC, 'No space left on device')
                raise OSError('restore failed')
            return original_write_bytes(path, data)

        monkeypatch.setattr(Path, 'write_bytes', failed_writes)
        result = _wrap_worktree_files(
            WorktreeWrapContext(repo_root=tmp_path, incr_level=None, allow_on_failure=True),
            ['a.py', 'b.py'],
        )

        assert result.failure is None
        assert calls == 2
        assert 'restoring the original bytes also failed' in capsys.readouterr().err

    @pytest.mark.parametrize('allow_on_failure', [False, True])
    def test_failing_file_is_restored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        allow_on_failure: bool,
    ) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        original = _post_tool_long_source().encode()
        a.write_bytes(original)
        b.write_bytes(original)
        original_write_bytes = Path.write_bytes
        calls = 0

        def partial_write(path: Path, data: bytes) -> int:
            nonlocal calls
            if path == b:
                calls += 1
                if calls == 1:
                    original_write_bytes(path, data[: len(data) // 2])
                    raise OSError(errno.ENOSPC, 'No space left on device')
            return original_write_bytes(path, data)

        monkeypatch.setattr(Path, 'write_bytes', partial_write)
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path, incr_level=None, allow_on_failure=allow_on_failure
            ),
            ['a.py', 'b.py'],
        )

        assert calls == 2
        assert b.read_bytes() == original
        if allow_on_failure:
            assert result.failure is None
            assert a.read_text() == _POST_TOOL_WRAPPED_SOURCE
            assert capsys.readouterr().err == f'WARNING: docwrap: failed to write {b.as_posix()}\n'
        else:
            assert result.failure is not None
            assert result.failure.decision == 'deny'
            assert a.read_bytes() == original

    def test_rollback_on_write_failure(self, tmp_path: Path) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        long_docstring = _post_tool_long_source()
        a.write_text(long_docstring)
        b.write_text(long_docstring)
        original_a_bytes = a.read_bytes()

        # Make b.py read-only so the write fails
        b.chmod(0o444)
        try:
            result = _wrap_worktree_files(
                WorktreeWrapContext(
                    repo_root=tmp_path,
                    incr_level=None,
                    allow_on_failure=False,
                ),
                ['a.py', 'b.py'],
            )
            assert result.failure is not None
            assert result.failure.decision == 'deny'
            # a.py should be rolled back to original bytes
            assert a.read_bytes() == original_a_bytes
        finally:
            b.chmod(0o644)

    def test_rollback_suppresses_os_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        long_docstring = _post_tool_long_source()
        a.write_text(long_docstring)
        b.write_text(long_docstring)

        # First write_bytes (a.py) succeeds, second (b.py) fails,
        # rollback write for a.py also fails (suppressed by contextlib.suppress).
        call_count = 0
        original_write_bytes = Path.write_bytes

        def failing_write_bytes(self_path: Path, data: bytes) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_write_bytes(self_path, data)
            raise OSError('simulated write failure')

        monkeypatch.setattr(Path, 'write_bytes', failing_write_bytes)

        # Should return a failure result (failure set) without raising
        result = _wrap_worktree_files(
            WorktreeWrapContext(
                repo_root=tmp_path,
                incr_level=None,
                allow_on_failure=False,
            ),
            ['a.py', 'b.py'],
        )
        assert result.failure is not None
        assert result.failure.decision == 'deny'


class _AdapterFields:
    def __init__(self, fields: dict[str, object]) -> None:
        self._fields = fields

    def __getattr__(self, name: str) -> object:
        if name not in self._fields:
            raise AttributeError(name)
        value = self._fields[name]
        if isinstance(value, BaseException):
            raise value
        return value


class _AdapterProbe:
    requires_event = False

    def __init__(self, parsed: object, response: object) -> None:
        self._parsed = parsed
        self._response = response

    def parse(self, payload: Mapping[str, object], event: str | None) -> object:
        del payload, event
        if isinstance(self._parsed, BaseException):
            raise self._parsed
        return self._parsed

    def render(self, result: HookResult) -> object:
        del result
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    def render_failure(self, reason: str) -> object:
        del reason
        return self._response


def _antigravity_payload(tool_name: str, args: dict[str, object], workspace: Path) -> str:
    return json.dumps(
        {
            'toolCall': {'name': tool_name, 'args': args},
            'workspacePaths': [str(workspace)],
        }
    )


def _antigravity_pretooluse_payload(cmd: str, cwd: Path) -> str:
    return _antigravity_payload('run_command', {'CommandLine': cmd, 'Cwd': str(cwd)}, cwd)


def _assert_uniform_newlines(raw: bytes, *, crlf: bool) -> None:
    if crlf:
        assert b'\r\n' in raw
        assert b'\n' not in raw.replace(b'\r\n', b'')
        assert b'\r' not in raw.replace(b'\r\n', b'')
    else:
        assert b'\n' in raw
        assert b'\r' not in raw


def _fail_on_repository_work(monkeypatch: pytest.MonkeyPatch) -> None:

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail('repository-dependent operation was called')

    for name in (
        'apply_index_entries',
        'discover_repo_root',
        'find_config',
        'get_staged_py_info',
        'is_hook_enabled',
        'read_staged_blobs',
        'run_pre_commit_adds',
    ):
        monkeypatch.setattr(hook_mod, name, fail_if_called)


def _git_file_list_failure(_cwd: str) -> None:
    return


def _index_listing(repo: Path) -> bytes:
    return run_setup(['git', 'ls-files', '-s'], cwd=str(repo)).stdout


def _init_hook_repo(repo: Path, *, hook_enabled: bool) -> None:
    repo.mkdir(exist_ok=True)
    run_setup(['git', 'init'], cwd=str(repo))
    run_setup(
        ['git', 'config', 'docwrap.hook', str(hook_enabled).lower()],
        cwd=str(repo),
    )
    (repo / 'baseline.py').write_text('# baseline\n')
    git_commit_all(repo, 'initial')


def _install_adapter_probe(monkeypatch: pytest.MonkeyPatch, adapter: object) -> None:
    def load_probe(_host: str) -> HookAdapter:
        return cast('HookAdapter', adapter)

    monkeypatch.setattr(front_mod, 'load_adapter', load_probe)


def _isolate_global_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    global_config: Path,
) -> None:
    home = tmp_path / 'home'
    xdg = tmp_path / 'xdg'
    home.mkdir(exist_ok=True)
    xdg.mkdir(exist_ok=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(xdg))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(global_config))
    monkeypatch.delenv('GIT_CONFIG_SYSTEM', raising=False)
    monkeypatch.delenv('GIT_CONFIG_COUNT', raising=False)


def _run_agy(
    stdin_data: str,
    *,
    event: str = 'PreToolUse',
    allow_on_failure: bool = False,
    incremental: bool | None = None,
) -> int:
    return run_hook(
        stdin_data,
        host='agy',
        event=event,
        allow_on_failure=allow_on_failure,
        incremental=incremental,
    )


@pytest.mark.usefixtures('registered_hosts')
class TestAntigravityProtocol:
    def test_antigravity_commit_uses_target_cwd_not_process_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        process_repo = tmp_path / 'process'
        target_repo = tmp_path / 'target'
        process_repo.mkdir()
        target_repo.mkdir()
        run_setup(['git', 'init'], cwd=str(process_repo))
        run_setup(['git', 'config', 'docwrap.hook', 'false'], cwd=str(process_repo))
        run_setup(['git', 'init'], cwd=str(target_repo))
        run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=str(target_repo))
        (target_repo / 'baseline.py').write_text('# baseline\n')
        git_commit_all(target_repo, 'initial')
        f = target_repo / 'foo.py'
        f.write_text(
            'def foo(value):\n'
            '    """Return the value.\n\n'
            '    Args:\n'
            '        value: This is a very long argument description that should '
            'definitely be wrapped.\n'
            '    """\n'
            '    return value\n'
        )
        run_setup(['git', 'add', '.'], cwd=str(target_repo))
        monkeypatch.chdir(process_repo)

        result = _run_agy(_antigravity_pretooluse_payload('git commit -m "test"', target_repo))

        assert result == 0
        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        staged = run_setup(
            ['git', 'show', ':foo.py'],
            cwd=str(target_repo),
        ).stdout
        assert 'definitely\n' in decode_text(staged)

    def test_antigravity_commit_wraps_staged_file_with_a_receipt(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=str(tmp_path))
        (tmp_path / 'baseline.py').write_text('# baseline\n')
        git_commit_all(tmp_path, 'initial')
        f = tmp_path / 'foo.py'
        f.write_text(
            'def foo(value):\n'
            '    """Return the value.\n\n'
            '    Args:\n'
            '        value: This is a very long argument description that should '
            'definitely be wrapped.\n'
            '    """\n'
            '    return value\n'
        )
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))

        result = _run_agy(_antigravity_pretooluse_payload('git commit -m "test"', tmp_path))

        assert result == 0
        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        staged = run_setup(
            ['git', 'show', ':foo.py'],
            cwd=str(tmp_path),
        ).stdout
        assert 'definitely\n' in decode_text(staged)

    def test_antigravity_infrastructure_denial_uses_top_level_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=str(tmp_path))
        monkeypatch.setattr('docwrap._hook.get_staged_py_info', _git_file_list_failure)

        result = _run_agy(_antigravity_pretooluse_payload('git commit -m "test"', tmp_path))

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output == {'decision': 'deny', 'reason': 'docwrap: failed to list staged files'}

    def test_antigravity_non_commit_run_command_is_silent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        result = _run_agy(_antigravity_pretooluse_payload('pytest -q', tmp_path))

        assert result == 0
        assert capsys.readouterr().out == ''

    def test_antigravity_null_tool_call_is_noop(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _run_agy(json.dumps({'toolCall': None}))
        assert result == 0
        assert capsys.readouterr().out == ''

    def test_antigravity_post_tool_use_ignores_non_python_and_unsupported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        text_file = tmp_path / 'notes.txt'
        text_file.write_text('unchanged')

        non_python = _run_agy(
            _antigravity_payload('write_to_file', {'TargetFile': str(text_file)}, tmp_path),
            event='PostToolUse',
        )
        unsupported = _run_agy(
            _antigravity_payload('read_file', {'TargetFile': str(text_file)}, tmp_path),
            event='PostToolUse',
        )

        assert non_python == 0
        assert unsupported == 0
        assert capsys.readouterr().out == ''
        assert text_file.read_text() == 'unchanged'

    def test_antigravity_replace_file_content_resolves_relative_target(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.chdir('/')
        f = tmp_path / 'foo.py'
        original = _post_tool_long_source()
        f.write_text(original)

        result = _run_agy(
            _antigravity_payload('replace_file_content', {'TargetFile': 'foo.py'}, tmp_path),
            event='PostToolUse',
        )

        assert result == 0
        assert capsys.readouterr().out == ''
        assert f.read_text() != original

    def test_antigravity_workspace_without_command_cwd_is_denied(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=str(tmp_path))
        (tmp_path / 'baseline.py').write_text('# baseline\n')
        git_commit_all(tmp_path, 'initial')
        f = tmp_path / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        payload = _antigravity_payload(
            'run_command', {'CommandLine': 'git commit -m "test"'}, tmp_path
        )

        before = (_index_listing(tmp_path), f.read_bytes())
        result = _run_agy(payload)

        assert result == 0
        assert _assert_denied(capsys, 'antigravity') == _UNKNOWN_CWD_REASON
        assert (_index_listing(tmp_path), f.read_bytes()) == before

    def test_antigravity_write_to_file_wraps_python_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        f = tmp_path / 'foo.py'
        original = _post_tool_long_source()
        f.write_text(original)

        result = _run_agy(
            _antigravity_payload('write_to_file', {'TargetFile': str(f)}, tmp_path),
            event='PostToolUse',
        )

        assert result == 0
        assert capsys.readouterr().out == ''
        assert f.read_text() != original


class TestLastResort:
    def test_adapter_parse_failure(
        self, capsys: pytest.CaptureFixture[str], example_hosts: Path
    ) -> None:
        del example_hosts
        assert run_hook(_LAST_RESORT_PAYLOAD, host='broken-parse') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == (
            "docwrap: host 'broken-parse' failed to parse the payload: parse exploded\n"
        )

    @pytest.mark.parametrize('exception_type', [KeyboardInterrupt, SystemExit])
    @pytest.mark.parametrize('boundary', ['metadata', 'parse', 'input', 'render', 'response'])
    def test_adapter_process_control_exceptions_propagate(
        self,
        boundary: str,
        exception_type: type[BaseException],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        failure = exception_type('stop')
        parsed: object = HostInput('pre_tool', None, '.', None)
        response: object = HostResponse('', '', 0)
        if boundary == 'parse':
            parsed = failure
        elif boundary == 'input':
            parsed = _AdapterFields({'command_cwd': failure})
        elif boundary == 'render':
            response = failure
        elif boundary == 'response':
            response = _AdapterFields({'stdout': 'partial', 'stderr': failure, 'exit_code': 0})
        adapter = _AdapterProbe(parsed, response)
        _install_adapter_probe(
            monkeypatch,
            _AdapterFields({'requires_event': failure}) if boundary == 'metadata' else adapter,
        )

        with pytest.raises(exception_type, match='stop'):
            run_hook('{}', host='probe')
        assert capsys.readouterr() == ('', '')

    def test_both_render_paths_failing_is_the_last_resort(
        self, capsys: pytest.CaptureFixture[str], example_hosts: Path
    ) -> None:
        del example_hosts
        assert run_hook(_LAST_RESORT_PAYLOAD, host='broken-both') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert (
            captured.err == "docwrap: host 'broken-both' failed to render the failure response\n"
        )

    @pytest.mark.parametrize('exception_type', [KeyboardInterrupt, SystemExit])
    def test_failure_renderer_process_control_exceptions_propagate(
        self,
        exception_type: type[BaseException],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class InterruptedFailureAdapter(_AdapterProbe):
            def render_failure(self, reason: str) -> object:
                del reason
                raise exception_type('stop')

        adapter = InterruptedFailureAdapter(
            HostInput('pre_tool', None, '.', None), RuntimeError('render failed')
        )
        _install_adapter_probe(monkeypatch, adapter)
        with pytest.raises(exception_type, match='stop'):
            run_hook('{}', host='probe')
        assert capsys.readouterr() == ('', '')

    def test_invalid_json_and_inapplicable_payloads_are_silent(
        self, capsys: pytest.CaptureFixture[str], example_hosts: Path
    ) -> None:
        del example_hosts
        assert run_hook('not json', host='example') == 0
        assert run_hook('[1]', host='example') == 0
        assert run_hook(json.dumps({'hook_event_name': 'Notification'}), host='example') == 0
        assert capsys.readouterr() == ('', '')

    @pytest.mark.parametrize('raises', [False, True])
    @pytest.mark.parametrize(
        'field', ['requires_event', 'command_cwd', 'event', 'command', 'cwd', 'edited_paths']
    )
    def test_malformed_adapter_fields_fail_closed(
        self,
        field: str,
        raises: bool,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fields: dict[str, object] = {
            'event': 'post_tool' if field in {'cwd', 'edited_paths'} else 'pre_tool',
            'command': 'ls',
            'command_cwd': None,
            'cwd': '.',
            'edited_paths': (),
        }
        if field != 'requires_event':
            fields.pop(field)
            if raises:
                fields[field] = RuntimeError('first\nsecond')
        adapter = _AdapterProbe(_AdapterFields(fields), HostResponse('', '', 0))
        if field == 'requires_event':
            metadata: dict[str, object] = {'parse': adapter.parse}
            if raises:
                metadata[field] = RuntimeError('first\nsecond')
            _install_adapter_probe(monkeypatch, _AdapterFields(metadata))
        else:
            _install_adapter_probe(monkeypatch, adapter)

        assert run_hook('{}', host='probe\nname') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert len(captured.err.splitlines()) == 1
        assert captured.err.startswith("docwrap: host 'probe\\x0aname'")
        assert 'Traceback' not in captured.err

    @pytest.mark.parametrize(
        ('field', 'value'),
        [('event', 'PostToolUse'), ('event', 42), ('command', 42)],
    )
    def test_invalid_adapter_values_fail_closed(
        self,
        field: str,
        value: object,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fields: dict[str, object] = {
            'event': 'pre_tool',
            'command': 'ls',
            'command_cwd': None,
            'cwd': '.',
            'edited_paths': (),
        }
        fields[field] = value
        _install_adapter_probe(
            monkeypatch, _AdapterProbe(_AdapterFields(fields), HostResponse('', '', 0))
        )

        assert run_hook('{}', host='probe') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err.startswith("docwrap: host 'probe' failed to parse the payload:")

    @pytest.mark.parametrize(
        'case',
        [
            ('cwd', 42, 'post-tool cwd must be a string'),
            ('edited_paths', (42,), 'post-tool edited paths must be strings'),
        ],
    )
    def test_malformed_post_tool_paths_fail_before_repository_work(
        self,
        case: tuple[str, object, str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        field, value, detail = case
        source = tmp_path / 'mod.py'
        source.write_bytes(b'def zulu():\n    pass\n')
        fields: dict[str, object] = {
            'event': 'post_tool',
            'command': None,
            'command_cwd': None,
            'cwd': str(tmp_path),
            'edited_paths': ('mod.py',),
        }
        fields[field] = value
        adapter = _AdapterProbe(_AdapterFields(fields), HostResponse('partial', '', 0))
        _install_adapter_probe(monkeypatch, adapter)

        def fail_if_called(*_args: object, **_kwargs: object) -> None:
            pytest.fail('repository or source work was called')

        for name in ('discover_repo_root', 'read_file_result', 'write_preserving_newline'):
            monkeypatch.setattr(hook_mod, name, fail_if_called)

        assert run_hook('{}', host='probe\nname') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert len(captured.err.splitlines()) == 1
        assert captured.err.startswith("docwrap: host 'probe\\x0aname'")
        assert detail in captured.err
        assert 'Traceback' not in captured.err
        assert source.read_bytes() == b'def zulu():\n    pass\n'

    @pytest.mark.parametrize('field', ['stdout', 'stderr', 'exit_code'])
    @pytest.mark.parametrize('failure', ['missing', 'raising', 'wrong_type'])
    def test_malformed_response_emits_no_partial_output(
        self,
        field: str,
        failure: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fields: dict[str, object] = {
            'stdout': '{"decision":"allow"}\n',
            'stderr': 'note\n',
            'exit_code': 0,
        }
        fields.pop(field)
        if failure == 'raising':
            fields[field] = RuntimeError('first\nsecond')
        elif failure == 'wrong_type':
            fields[field] = 'zero' if field == 'exit_code' else 7
        adapter = _AdapterProbe(HostInput('pre_tool', None, '.', None), _AdapterFields(fields))
        _install_adapter_probe(monkeypatch, adapter)

        assert run_hook('{}', host='probe\nname') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert len(captured.err.splitlines()) == 1
        assert captured.err.startswith("docwrap: host 'probe\\x0aname'")
        assert 'Traceback' not in captured.err

    def test_malformed_response_recovers_with_valid_denial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class RecoveringAdapter(_AdapterProbe):
            def render_failure(self, reason: str) -> HostResponse:
                return HostResponse('denied', reason, 2)

        adapter = RecoveringAdapter(
            HostInput('pre_tool', None, '.', None),
            _AdapterFields({'stdout': 'partial allow', 'stderr': ''}),
        )
        _install_adapter_probe(monkeypatch, adapter)
        assert run_hook('{}', host='probe') == 2
        assert capsys.readouterr() == (
            'denied',
            "docwrap: host 'probe' failed to render the response",
        )

    def test_missing_and_unexpected_event(
        self, capsys: pytest.CaptureFixture[str], example_hosts: Path
    ) -> None:
        del example_hosts
        assert run_hook(_LAST_RESORT_PAYLOAD, host='requires-event') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == "docwrap: host 'requires-event' requires --event\n"
        assert run_hook(_LAST_RESORT_PAYLOAD, host='example', event='PreToolUse') == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == "docwrap: host 'example' does not accept --event\n"

    def test_render_failure_falls_back_to_the_failure_response(
        self, capsys: pytest.CaptureFixture[str], example_hosts: Path
    ) -> None:
        del example_hosts
        assert run_hook(_LAST_RESORT_PAYLOAD, host='broken-render') == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {
            'example': {
                'decision': 'deny',
                'reason': "docwrap: host 'broken-render' failed to render the response",
            }
        }
        assert captured.err == "docwrap: host 'broken-render' failed to render the response\n"

    @pytest.mark.parametrize('failure', ['missing', 'raising'])
    def test_unusable_renderers_fail_closed(
        self,
        failure: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        adapter = _AdapterProbe(HostInput('pre_tool', None, '.', None), HostResponse('', '', 0))
        fields: dict[str, object] = {'requires_event': False, 'parse': adapter.parse}
        if failure == 'raising':
            fields.update(
                render=RuntimeError('render failed'), render_failure=RuntimeError('failure failed')
            )
        _install_adapter_probe(monkeypatch, _AdapterFields(fields))
        assert run_hook('{}', host='probe') == 2
        assert capsys.readouterr() == (
            '',
            "docwrap: host 'probe' failed to render the failure response\n",
        )

    @pytest.mark.usefixtures('example_hosts')
    def test_raising_adapter_version_is_contained(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fail_version(_self: object) -> int:
            raise RuntimeError('first\nsecond')

        monkeypatch.setattr(ExampleAdapter, 'api_version', property(fail_version))
        assert run_hook(_LAST_RESORT_PAYLOAD, host='example') == 2
        assert capsys.readouterr() == (
            '',
            "docwrap: host 'example' failed to load: first\\x0asecond\n",
        )

    @pytest.mark.usefixtures('example_hosts')
    @pytest.mark.parametrize('host', ['empty-cwd', 'relative-cwd', 'typed-cwd'])
    def test_invalid_command_cwd_is_a_last_resort_failure(
        self,
        host: str,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError('repository code must not run')

        monkeypatch.setattr('docwrap._hook_front._run_event', explode)
        assert run_hook(_LAST_RESORT_PAYLOAD, host=host) == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'invalid command_cwd' in captured.err

    @pytest.mark.usefixtures('registered_hosts')
    def test_unknown_host(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run_hook(_LAST_RESORT_PAYLOAD, host='nope') == 2
        captured = capsys.readouterr()
        hosts = ', '.join(sorted(BUILTIN_ENTRY_POINTS))
        assert captured.out == ''
        assert captured.err == f"docwrap: unknown host 'nope'; registered hosts: {hosts}\n"


@pytest.mark.usefixtures('registered_hosts')
class TestPathspecFailOpenOnModifiedNone:
    def test_pathspec_commit_modified_none_allows_with_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(
            'docwrap._hook.get_modified_tracked_py_files',
            _git_file_list_failure,
        )
        original = f.read_bytes()

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        _run_claude(json.dumps(event), allow_on_failure=True)

        assert f.read_bytes() == original
        captured = capsys.readouterr()
        assert 'WARNING' in captured.err

    def test_pathspec_commit_modified_none_denies(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(
            'docwrap._hook.get_modified_tracked_py_files',
            _git_file_list_failure,
        )
        original = f.read_bytes()

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        _run_claude(json.dumps(event))
        output = capsys.readouterr().out

        assert f.read_bytes() == original
        assert output.strip()
        data = json.loads(output)
        assert data['hookSpecificOutput']['permissionDecision'] == 'deny'


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolUseIncremental:
    def test_default_wraps_whole_selected_file(self, tmp_path: Path) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / 'foo.py'
        path.write_text(_hook_selection_source(1))
        git_commit_all(tmp_path, 'add selection fixture')
        path.write_text(_hook_selection_source(2))
        event = hook_event(
            'PostToolUse',
            'Edit',
            {'file_path': str(path)},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert path.read_text() == _hook_selection_source(2, wrapped=True)

    @pytest.mark.parametrize('ignored', [False, True])
    def test_explicit_untracked_post_edit_is_eligible(self, tmp_path: Path, ignored: bool) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        (tmp_path / '.gitignore').write_text('new.py\n' if ignored else '')
        path = tmp_path / 'new.py'
        original = _post_tool_long_source()
        path.write_text(original)
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(path)}, cwd=str(tmp_path))
        assert _run_claude(json.dumps(event), incremental=True) == 0
        assert path.read_text() != original

    @pytest.mark.parametrize('relative', ['module.py', 'pkg/module.py'])
    @pytest.mark.parametrize('changed', [False, True])
    def test_native_path_preserves_untouched_bytes(
        self, tmp_path: Path, relative: str, changed: bool
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        first = _post_tool_long_source() + '    return 1\n'
        untouched = first.replace('def f(', 'def untouched(')
        original = first + '\n\n' + untouched
        path.write_bytes(original.encode())
        git_commit_all(tmp_path, 'baseline')
        source = original.replace('return 1', 'return 2', 1) if changed else original
        path.write_bytes(source.encode())
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(path)}, cwd=str(tmp_path))
        assert _run_claude(json.dumps(event), incremental=True) == 0
        result = path.read_bytes()
        assert result.endswith(untouched.encode())
        assert (result != source.encode()) is changed

    def test_post_tool_wraps_quoted_filename(self, tmp_path: Path) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / ('tab\tname.py' if os.name != 'nt' else 'space name.py')
        path.write_text(_post_tool_long_source(text='coffee'))
        git_commit_all(tmp_path, 'add quoted path')
        path.write_text(_post_tool_long_source())
        event = hook_event(
            'PostToolUse',
            'Edit',
            {'file_path': str(path)},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert path.read_text() == _POST_TOOL_WRAPPED_SOURCE

    @pytest.mark.parametrize('route', ['post-edit', 'staged', 'worktree'])
    def test_rename_preserves_unchanged_docstring(self, tmp_path: Path, route: str) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        old = tmp_path / 'old.py'
        first = _post_tool_long_source() + '    return 1\n'
        untouched = first.replace('def f(', 'def untouched(')
        old.write_text(first + '\n\n' + untouched)
        unrelated = tmp_path / 'unselected.py'
        unrelated.write_text(first)
        git_commit_all(tmp_path, 'baseline')
        target = tmp_path / 'new.py'
        old.rename(target)
        changed = first.replace('return 1', 'return 2')
        target.write_text(changed + '\n\n' + untouched)
        run_setup(['git', 'add', '-A'], cwd=tmp_path)
        unrelated.write_text(changed)
        rename = run_setup(
            ['git', 'diff', '--cached', '--name-status', '-M'],
            cwd=tmp_path,
        )
        status, source_name, destination_name = decode_text(rename.stdout).strip().split('\t')
        assert status.startswith('R')
        assert int(status[1:]) >= 90
        assert (source_name, destination_name) == ('old.py', 'new.py')
        if route == 'post-edit':
            event = hook_event(
                'PostToolUse', 'Edit', {'file_path': str(target)}, cwd=str(tmp_path)
            )
        else:
            command = 'git commit -m test' if route == 'staged' else 'git commit -m test new.py'
            event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))
        assert _run_claude(json.dumps(event), incremental=True) == 0
        result = _staged_text(tmp_path, 'new.py') if route == 'staged' else target.read_text()
        assert result.split('def untouched')[0] != changed + '\n\n'
        assert result.endswith(untouched)
        assert unrelated.read_text() == changed


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolUseSkipReporting:
    def test_mixed_skip_and_change_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        f = tmp_path / 'foo.py'
        f.write_text(_ESCAPE_MIXED_SOURCE)
        payload = hook_event('PostToolUse', 'Write', {'file_path': str(f)})
        assert _run_claude(json.dumps(payload)) == 0
        text = f.read_text()
        assert 'Line one ' + BACKSLASH + '\n' in text  # skipped docstring untouched
        assert 'definitely\n' in text  # other docstring wrapped
        captured = capsys.readouterr()
        assert f'docwrap: {f.as_posix()}:2: skipme skipped (line-continuation):' in captured.err
        assert json.loads(captured.out) == {
            'hookSpecificOutput': {
                'hookEventName': 'PostToolUse',
                'additionalContext': captured.err.rstrip('\n'),
            }
        }


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolUseUntrackedFile:
    def test_untracked_file_gets_full_wrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        f = tmp_path / 'new_file.py'
        f.write_text(
            'def foo(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: This is a very long argument description that should'
            ' definitely be wrapped by docwrap.\n'
            '    """\n'
            '    pass\n'
        )
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.chdir(tmp_path)

        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Write',
            'tool_input': {'file_path': str(f)},
        }
        _run_claude(json.dumps(event))

        content = f.read_text()
        # The long arg line should have been wrapped (broken across lines)
        lines = content.splitlines()
        arg_lines = [line for line in lines if 'definitely' in line]
        assert arg_lines, 'Expected wrapped output to contain "definitely"'
        assert all(len(line) <= 79 for line in arg_lines)


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUse:
    def test_claude_permission_mode_commit_wraps_with_a_receipt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
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
        f.write_text(
            'def foo(value):\n'
            '    """Return the value.\n'
            '\n'
            '    Args:\n'
            '        value: This is a very long argument description that should'
            ' definitely be wrapped by docwrap before the staged file is committed.\n'
            '    """\n'
            '    return value\n'
        )
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = {
            'session_id': 'session-test',
            'transcript_path': str(tmp_path / 'transcript.jsonl'),
            'cwd': str(tmp_path),
            'permission_mode': 'default',
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_use_id': 'toolu_test',
            'tool_input': {'command': 'git commit -m "test"'},
        }

        result = _run_claude(json.dumps(event))
        assert result == 0
        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        assert 'definitely\n' in _staged_text(tmp_path, 'foo.py')

    def test_codex_commit_receipt_counts_rewritten_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
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
        long_docstring = (
            'def {name}(value):\n'
            '    """Return the value.\n'
            '\n'
            '    Args:\n'
            '        value: This is a very long argument description that should'
            ' definitely be wrapped by docwrap before the staged file is committed.\n'
            '    """\n'
            '    return value\n'
        )
        (tmp_path / 'one.py').write_text(long_docstring.format(name='one'), encoding='utf-8')
        (tmp_path / 'two.py').write_text(long_docstring.format(name='two'), encoding='utf-8')
        (tmp_path / 'clean.py').write_text('X = 1\n', encoding='utf-8')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        event = {
            'hook_event_name': 'PreToolUse',
            'turn_id': 'turn-test',
            'tool_name': 'Bash',
            'tool_use_id': 'tool-test',
            'tool_input': {'command': f'git -C {shell_path(tmp_path)} commit -m "test"'},
            'cwd': str(tmp_path),
        }
        assert _run_codex(json.dumps(event)) == 0
        assert json.loads(capsys.readouterr().out) == {
            'systemMessage': 'docwrap: wrapped 2 staged Python files for this commit'
        }

    def test_codex_commit_with_nothing_to_wrap_stays_silent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
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
        (tmp_path / 'clean.py').write_text('X = 1\n', encoding='utf-8')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        event = {
            'hook_event_name': 'PreToolUse',
            'turn_id': 'turn-test',
            'tool_name': 'Bash',
            'tool_use_id': 'tool-test',
            'tool_input': {'command': f'git -C {shell_path(tmp_path)} commit -m "test"'},
            'cwd': str(tmp_path),
        }
        assert _run_codex(json.dumps(event)) == 0
        assert capsys.readouterr().out == ''

    def test_commit_preserves_crlf_staged_blob(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        crlf_source = _post_tool_long_source().replace('\n', '\r\n').encode()
        result = run_setup(
            ['git', 'hash-object', '-w', '--stdin'],
            stdin=crlf_source,
            cwd=str(tmp_path),
        )
        blob_hash = result.stdout.decode('ascii').strip()
        run_setup(
            ['git', 'update-index', '--add', '--cacheinfo', f'100644,{blob_hash},foo.py'],
            cwd=str(tmp_path),
        )
        (tmp_path / 'foo.py').write_bytes(crlf_source)

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit -m "test"'},
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        assert result == 0

        after = run_setup(
            ['git', 'show', ':foo.py'],
            cwd=str(tmp_path),
        ).stdout
        assert after != crlf_source, 'wrapping should have changed the staged content'
        assert b'\r\n' in after, 'CRLF should be preserved in the staged blob'
        stripped = after.replace(b'\r\n', b'')
        assert b'\n' not in stripped, 'no bare LF should remain (all newlines must be CRLF)'

    def test_default_wraps_whole_staged_file_and_preserves_partial_staging(
        self,
        tmp_path: Path,
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / 'foo.py'
        path.write_text(_hook_selection_source(1))
        git_commit_all(tmp_path, 'add selection fixture')
        staged_source = _hook_selection_source(2)
        path.write_text(staged_source)
        run_setup(['git', 'add', 'foo.py'], cwd=tmp_path)
        worktree_source = '# unrelated unstaged change\n' + staged_source
        path.write_text(worktree_source)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert _staged_text(tmp_path, 'foo.py') == _hook_selection_source(2, wrapped=True)
        assert path.read_text() == worktree_source

    def test_non_bash_ignored(self) -> None:
        event = hook_event('PreToolUse', 'Edit', {'command': 'git commit -m "test"'})
        result = _run_claude(json.dumps(event))
        assert result == 0

    def test_non_commit_ignored(self) -> None:
        event = hook_event('PreToolUse', 'Bash', {'command': 'git push'}, cwd='/tmp')  # noqa: S108
        result = _run_claude(json.dumps(event))
        assert result == 0


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseEmptyPathspecFile:
    def test_empty_pathspec_from_file_processes_staged(
        self,
        git_repo_with_file: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tmp_path, f = git_repo_with_file
        # Stage a long docstring
        f.write_text(
            'def foo(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: This is a very long argument description that should'
            ' definitely be wrapped by docwrap.\n'
            '    """\n'
            '    pass\n'
        )
        run_setup(['git', 'add', 'foo.py'], cwd=str(tmp_path))

        # Create empty pathspec file
        pathspec_file = tmp_path / 'pathspecs.txt'
        pathspec_file.write_text('')
        pathspec_file_word = shell_path(pathspec_file)

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {
                'command': f'git commit --pathspec-from-file={pathspec_file_word} -m "test"',
            },
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        assert result == 0
        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        # Verify the staged file was actually wrapped (normal commit path,
        # not empty pathspec-only path that skips processing).
        content = f.read_text()
        lines = [line for line in content.splitlines() if 'definitely' in line]
        assert lines, 'staged file should have been processed'
        assert all(len(line) <= 79 for line in lines)


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseIncludeExcludePathspec:
    def test_include_mode_excludes_pathspec_from_staged(
        self,
        git_repo_with_file: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tmp_path, f = git_repo_with_file
        # Stage foo.py with a long docstring
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(tmp_path))

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        original_read = hook_mod.read_staged_blobs
        captured_paths: list[list[str]] = []

        def tracking_read(
            cwd: str,
            info_list: list[StagedFileInfo],
        ) -> dict[str, BaselineResult] | None:
            captured_paths.append([i.path for i in info_list])
            return original_read(cwd, info_list)

        monkeypatch.setattr('docwrap._hook.read_staged_blobs', tracking_read)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit --include foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        _run_claude(json.dumps(event))

        # foo.py should NOT appear in the paths passed to read_staged_blobs
        for paths in captured_paths:
            assert 'foo.py' not in paths


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseInvalidPythonIncremental:
    _VALID = 'def f():\n    """Ok docstring."""\n    return 1\n'
    _INVALID = 'def f(:\n    """Broken on purpose."""\n'
    _WRAPPABLE = (
        'def g():\n'
        '    """Summary.\n'
        '\n'
        '    This body paragraph is deliberately far too long for the configured'
        ' line length and will wrap.\n'
        '    """\n'
    )

    def _repo_with_staged_invalid(self, repo: Path) -> None:
        run_setup(['git', 'init'], cwd=str(repo))
        (repo / 'mod.py').write_bytes(self._VALID.encode('utf-8'))
        git_commit_all(repo, 'add mod')
        (repo / 'mod.py').write_bytes(self._INVALID.encode('utf-8'))
        run_setup(['git', 'add', 'mod.py'], cwd=str(repo))

    def test_initial_commit_invalid_python_allows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        run_setup(['git', 'init'], cwd=str(tmp_path))
        (tmp_path / 'bad.py').write_text(self._INVALID)
        run_setup(['git', 'add', 'bad.py'], cwd=str(tmp_path))
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload), incremental=True) == 0
        _assert_success(capsys)
        assert _staged_text(tmp_path, 'bad.py') == self._INVALID

    def test_mixed_commit_wraps_valid_and_skips_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        self._repo_with_staged_invalid(tmp_path)
        (tmp_path / 'good.py').write_bytes(self._WRAPPABLE.encode('utf-8'))
        run_setup(['git', 'add', 'good.py'], cwd=str(tmp_path))
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload), incremental=True) == 0
        _assert_success(capsys, 1)
        staged = run_setup(
            ['git', 'show', ':good.py'],
            cwd=str(tmp_path),
        ).stdout.decode('utf-8')
        assert staged != self._WRAPPABLE
        staged_invalid = run_setup(
            ['git', 'show', ':mod.py'],
            cwd=str(tmp_path),
        ).stdout.decode('utf-8')
        assert staged_invalid == self._INVALID

    def test_posttooluse_working_tree_invalid_python_skips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        run_setup(['git', 'init'], cwd=str(tmp_path))
        target = tmp_path / 'mod.py'
        target.write_text(self._VALID)
        git_commit_all(tmp_path, 'add mod')
        target.write_text(self._INVALID)
        payload = hook_event('PostToolUse', 'Edit', {'file_path': str(target)}, cwd=str(tmp_path))
        assert _run_claude(json.dumps(payload), incremental=True) == 0
        assert target.read_text() == self._INVALID
        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'PostToolUse error processing' not in captured.err

    def test_staged_invalid_python_allows_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        self._repo_with_staged_invalid(tmp_path)
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload), incremental=True) == 0

        _assert_success(capsys)
        assert _staged_text(tmp_path, 'mod.py') == self._INVALID


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseModificationFiltering:
    def test_unmodified_pathspec_file_not_processed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        _write_long_arg_docstring(f)
        git_commit_all(tmp_path, 'init')
        # foo.py is tracked but NOT modified since last commit

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        original = f.read_bytes()

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        _run_claude(json.dumps(event))

        assert f.read_bytes() == original


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseNonUtf8Pyproject:
    def test_commit_allowed_with_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        run_setup(['git', 'init'], cwd=str(tmp_path))
        (tmp_path / 'baseline.py').write_text('# baseline\n')
        git_commit_all(tmp_path, 'initial')
        (tmp_path / 'pyproject.toml').write_bytes(
            b'[tool.docwrap]\n# caf\xe9 comment\nline-length = 100\n'
        )
        (tmp_path / 'mod.py').write_text('def f():\n    """Fine."""\n')
        run_setup(
            ['git', 'add', 'pyproject.toml', 'mod.py'],
            cwd=str(tmp_path),
        )
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload)) == 0


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUsePathspecCommit:
    def test_include_keeps_staged_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        staged_file = tmp_path / 'staged.py'
        staged_file.write_text('pass\n')
        included_file = tmp_path / 'included.py'
        included_file.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        # Stage a long docstring in staged.py
        staged_file.write_text(
            'def bar(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: This is a very long argument description that should'
            ' definitely be wrapped by docwrap.\n'
            '    """\n'
            '    pass\n'
        )
        run_setup(['git', 'add', 'staged.py'], cwd=str(tmp_path))
        # Modify included.py but DON'T stage it
        included_file.write_text(
            'def baz(y):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        y: This is another very long argument description that should'
            ' also definitely be wrapped.\n'
            '    """\n'
            '    pass\n'
        )
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git commit --include included.py -m "test"'},
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))
        assert result == 0
        # Both files should have been wrapped
        staged_content = staged_file.read_text()
        staged_lines = [line for line in staged_content.splitlines() if 'definitely' in line]
        assert staged_lines, 'staged.py should still contain "definitely"'
        assert all(len(line) <= 79 for line in staged_lines)

        included_content = included_file.read_text()
        included_lines = [line for line in included_content.splitlines() if 'definitely' in line]
        assert included_lines, 'included.py should have been wrapped'
        assert all(len(line) <= 79 for line in included_lines)


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUsePathspecFailClosed:
    # NOTE: The staged filter code path is dead post-restructure (Task 6).
    # Only the two dispatcher fail-closed sites need D9 coverage.

    def test_include_dispatcher_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        f.write_text('x = 1\n')
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr('docwrap._hook.resolve_pathspec_py_files', _fail_pathspecs)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit --include foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        output = capsys.readouterr()
        assert result == 2 or 'deny' in output.out

    def test_pathspec_only_dispatcher_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        f.write_text('x = 1\n')
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr('docwrap._hook.resolve_pathspec_py_files', _fail_pathspecs)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit foo.py -m "test"'},
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        output = capsys.readouterr()
        assert result == 2 or 'deny' in output.out


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUsePathspecResolution:
    def test_directory_pathspec_resolves_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        sub = tmp_path / 'sub'
        sub.mkdir()
        f = sub / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        f.write_text(
            'def foo(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: This is a very long argument description that should'
            ' definitely be wrapped by docwrap.\n'
            '    """\n'
            '    pass\n'
        )

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit sub/ -m "test"'},
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        assert result == 0
        content = f.read_text()
        lines = [line for line in content.splitlines() if 'definitely' in line]
        assert lines, 'directory pathspec should resolve to .py files inside it'
        assert all(len(line) <= 79 for line in lines)

    def test_pathspec_from_file_only_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )
        f.write_text(
            'def foo(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: This is a very long argument description that should'
            ' definitely be wrapped by docwrap.\n'
            '    """\n'
            '    pass\n'
        )

        pathspec_file = tmp_path / 'pathspecs.txt'
        pathspec_file.write_text('foo.py\n')
        pathspec_file_word = shell_path(pathspec_file)

        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {
                'command': f'git commit --pathspec-from-file={pathspec_file_word} -m "test"',
            },
            'cwd': str(tmp_path),
        }
        result = _run_claude(json.dumps(event))
        assert result == 0
        content = f.read_text()
        lines = [line for line in content.splitlines() if 'definitely' in line]
        assert lines, 'pathspec-from-file should resolve and wrap the file'
        assert all(len(line) <= 79 for line in lines)


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseStagedSkipReporting:
    def _init_repo(self, repo: Path) -> None:
        run_setup(['git', 'init'], cwd=str(repo))
        run_setup(
            ['git', 'config', 'core.autocrlf', 'false'],
            cwd=str(repo),
        )
        (repo / 'baseline.py').write_text('# baseline\n')
        git_commit_all(repo, 'initial')

    def _staged_bytes(self, repo: Path, name: str) -> bytes:
        return run_setup(
            ['git', 'show', f':{name}'],
            cwd=str(repo),
        ).stdout

    def test_crlf_staged_blob_preserves_escape_and_newlines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        self._init_repo(tmp_path)
        f = tmp_path / 'foo.py'
        f.write_bytes(_ESCAPE_MIXED_SOURCE.replace('\n', '\r\n').encode('utf-8'))
        run_setup(['git', 'add', 'foo.py'], cwd=str(tmp_path))
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload)) == 0
        staged = self._staged_bytes(tmp_path, 'foo.py')
        assert b'\r\n' in staged
        text = staged.decode('utf-8').replace('\r\n', '\n')
        assert 'Line one ' + BACKSLASH + '\n' in text
        assert 'definitely\n' in text
        assert 'skipped (line-continuation)' in capsys.readouterr().err

    def test_staged_blob_skips_wraps_and_preserves_partial_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        self._init_repo(tmp_path)
        f = tmp_path / 'foo.py'
        f.write_bytes(_ESCAPE_MIXED_SOURCE.encode('utf-8'))
        run_setup(['git', 'add', 'foo.py'], cwd=str(tmp_path))
        # Partial staging with a LINE OFFSET: the worktree gains two unstaged
        # leading comment lines, so the skipped docstring sits at line 2 in the
        # staged blob but line 4 in the worktree. The diagnostic must cite the
        # staged-blob line.
        f.write_bytes(('# lead one\n# lead two\n' + _ESCAPE_MIXED_SOURCE).encode('utf-8'))
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload)) == 0
        text = self._staged_bytes(tmp_path, 'foo.py').decode('utf-8')
        assert 'Line one ' + BACKSLASH + '\n' in text  # skipped docstring untouched
        assert 'definitely\n' in text  # wrapped in the staged blob
        assert '# lead one' not in text  # staged blob does not gain worktree edits
        assert f.read_text().startswith('# lead one\n')  # worktree edit preserved
        err = capsys.readouterr().err
        assert 'docwrap: foo.py:2: skipme skipped (line-continuation):' in err
        assert 'foo.py:4:' not in err  # never the worktree line number


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseTargetHookConfig:
    def test_cd_target_enabled_process_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process_repo = tmp_path / 'process'
        target_repo = tmp_path / 'target'
        _init_hook_repo(process_repo, hook_enabled=False)
        _init_hook_repo(target_repo, hook_enabled=True)
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(target_repo))
        monkeypatch.chdir(process_repo)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': f'cd {shell_path(target_repo)} && git commit -m "test"'},
            cwd=str(process_repo),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')

    def test_dash_c_target_disabled_process_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process_repo = tmp_path / 'process'
        target_repo = tmp_path / 'target'
        _init_hook_repo(process_repo, hook_enabled=True)
        _init_hook_repo(target_repo, hook_enabled=False)
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(target_repo))
        before = _staged_text(target_repo, 'foo.py')
        monkeypatch.chdir(process_repo)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': f'git -C {shell_path(target_repo)} commit -m "test"'},
            cwd=str(process_repo),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert _staged_text(target_repo, 'foo.py') == before

    def test_dash_c_target_enabled_process_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        process_repo = tmp_path / 'process'
        target_repo = tmp_path / 'target'
        _init_hook_repo(process_repo, hook_enabled=False)
        _init_hook_repo(target_repo, hook_enabled=True)
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(target_repo))
        monkeypatch.chdir(process_repo)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': f'git -C {shell_path(target_repo)} commit -m "test"'},
            cwd=str(process_repo),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUseWorktreeSkipReporting:
    def test_commit_a_worktree_path_reports_skips(
        self,
        git_repo_with_file: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tmp_path, f = git_repo_with_file
        f.write_text(_ESCAPE_MIXED_SOURCE)
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        payload = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -am "msg"'}, cwd=str(tmp_path)
        )
        assert _run_claude(json.dumps(payload)) == 0
        text = f.read_text()
        assert 'Line one ' + BACKSLASH + '\n' in text  # skipped docstring untouched
        assert 'definitely\n' in text  # other docstring wrapped on the worktree
        err = capsys.readouterr().err
        assert 'skipped (line-continuation)' in err


def _assert_success(capsys: pytest.CaptureFixture[str], wrapped_files: int | None = None) -> None:
    stdout = capsys.readouterr().out
    if wrapped_files is None:
        assert stdout == ''
        return
    noun = 'file' if wrapped_files == 1 else 'files'
    assert json.loads(stdout) == {
        'systemMessage': f'docwrap: wrapped {wrapped_files} staged Python {noun} for this commit'
    }


def _run_claude(
    stdin_data: str,
    *,
    allow_on_failure: bool = False,
    incremental: bool | None = None,
) -> int:
    return run_hook(
        stdin_data,
        host='claude',
        allow_on_failure=allow_on_failure,
        incremental=incremental,
    )


def _run_codex(
    stdin_data: str,
    *,
    allow_on_failure: bool = False,
    incremental: bool | None = None,
) -> int:
    return run_hook(
        stdin_data,
        host='codex',
        allow_on_failure=allow_on_failure,
        incremental=incremental,
    )


def _run_provider(provider: str, stdin_data: str, *, event: str = 'PreToolUse') -> int:
    if provider == 'antigravity':
        return _run_agy(stdin_data, event=event)
    if provider == 'codex':
        return _run_codex(stdin_data)
    return _run_claude(stdin_data)


def _assert_denied(
    capsys: pytest.CaptureFixture[str],
    provider: str = 'claude',
    prefix: str = _REASON_PREFIX,
) -> str:
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    if provider == 'antigravity':
        assert data['decision'] == 'deny'
        reason = str(data['reason'])
    else:
        hook = data['hookSpecificOutput']
        assert hook['permissionDecision'] == 'deny'
        reason = str(hook['permissionDecisionReason'])
    assert reason.startswith(prefix)
    assert captured.err == f'{reason}\n'
    return reason


@pytest.mark.usefixtures('registered_hosts')
class TestAntigravityPreToolUseTargetHookConfig:
    @pytest.mark.parametrize(
        'command_template',
        [
            'git -C {target} commit -m "test"',
            'cd {target} && git commit -m "test"',
        ],
        ids=['dash-c', 'cd'],
    )
    @pytest.mark.parametrize(
        ('workspace_enabled', 'target_enabled', 'expect_change'),
        [(True, False, False), (False, True, True)],
    )
    def test_redirected_commit_uses_target_repository(
        self,
        tmp_path: Path,
        command_template: str,
        workspace_enabled: bool,
        target_enabled: bool,
        expect_change: bool,
    ) -> None:
        workspace = tmp_path / 'workspace'
        target = tmp_path / 'target'
        _init_hook_repo(workspace, hook_enabled=workspace_enabled)
        _init_hook_repo(target, hook_enabled=target_enabled)
        path = target / 'foo.py'
        _write_long_arg_docstring(path)
        run_setup(['git', 'add', 'foo.py'], cwd=target)
        before = _staged_text(target, 'foo.py')

        payload = _antigravity_pretooluse_payload(
            command_template.format(target=shell_path(target)),
            workspace,
        )
        assert _run_agy(payload) == 0
        after = _staged_text(target, 'foo.py')
        if expect_change:
            assert 'definitely\n' in after
        else:
            assert after == before


@pytest.mark.usefixtures('registered_hosts')
class TestNonUtf8GitPaths:
    @pytest.mark.skipif(
        os.name == 'nt',
        reason='POSIX bytes-path behavior is not available on Windows',
    )
    def test_staged_blob_wraps_with_non_utf8_filename(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        (tmp_path / 'baseline.py').write_text('# baseline\n')
        git_commit_all(tmp_path, 'init')
        name_bytes = b'staged-\xff.py'
        source = (
            b'def example(value):\n'
            b'    """Return the value.\n\n'
            b'    Args:\n'
            b'        value: This is a very long argument description that should '
            b'definitely be wrapped before commit.\n'
            b'    """\n'
            b'    return value\n'
        )
        path_bytes = os.fsencode(tmp_path / os.fsdecode(name_bytes))
        try:
            descriptor = os.open(
                path_bytes,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
        except OSError as exc:
            if exc.errno == errno.EILSEQ:
                pytest.skip('filesystem cannot create non-UTF-8 filenames')
            raise
        try:
            os.write(descriptor, source)
        finally:
            os.close(descriptor)
        run_setup(
            [b'git', b'add', b'--', name_bytes],
            cwd=os.fsencode(tmp_path),
        )
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        staged = run_setup(
            [b'git', b'show', b':' + name_bytes],
            cwd=os.fsencode(tmp_path),
        ).stdout
        assert staged != source
        assert b'definitely\n' in staged


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolBatch:
    @pytest.mark.parametrize('failed_call', ['diff', 'tracking'])
    def test_post_tool_batch_failure_falls_back_to_full_wrap(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failed_call: str,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        first = repo / 'first.py'
        second = repo / 'second.py'
        first.write_text('def first():\n    """Short."""\n    pass\n')
        second.write_text('def second():\n    """Short."""\n    pass\n')
        eligible_ranges_seen: list[set[tuple[int, int]] | None] = []
        tracked_calls: list[tuple[str, tuple[str, ...]]] = []

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(repo))

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {}

        def batch_diff(_repo_root: str) -> str | None:
            return None if failed_call == 'diff' else 'snapshot'

        def tracked_files(repo_root: str, paths: list[str]) -> set[str] | None:
            tracked_calls.append((repo_root, tuple(paths)))
            return None

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            eligible_ranges_seen.append(eligible_ranges)
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'get_tracked_files', tracked_files, raising=False)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {first}\n'
            f'*** Update File: {second}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert eligible_ranges_seen == [None, None]
        expected_tracked_calls = (
            [] if failed_call == 'diff' else [(str(repo), ('first.py', 'second.py'))]
        )
        assert tracked_calls == expected_tracked_calls

    def test_post_tool_batch_interleaved_repositories_keep_input_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first_repo = tmp_path / 'first-repo'
        second_repo = tmp_path / 'second-repo'
        first_repo.mkdir()
        second_repo.mkdir()
        first = first_repo / 'one.py'
        second = second_repo / 'two.py'
        third = first_repo / 'three.py'
        first.write_text('def one():\n    """Short."""\n    pass\n')
        second.write_text('def two():\n    """Short."""\n    pass\n')
        third.write_text('def three():\n    """Short."""\n    pass\n')
        discovery_calls: list[str] = []
        enablement_calls: list[str] = []
        diff_calls: list[str] = []
        tracked_calls: list[tuple[str, tuple[str, ...]]] = []
        processed: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            discovery_calls.append(cwd)
            root = first_repo if Path(cwd).is_relative_to(first_repo) else second_repo
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(root))

        def hook_enabled(cwd: str | None = None) -> bool:
            assert cwd is not None
            enablement_calls.append(cwd)
            return True

        def batch_diff(repo_root: str) -> str:
            diff_calls.append(repo_root)
            return 'snapshot'

        def tracked_files(repo_root: str, paths: list[str]) -> set[str]:
            tracked_calls.append((repo_root, tuple(paths)))
            return set(paths)

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {}

        def wrap_source(
            source: str,
            _config: object,
            _eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            processed.append(source.partition('(')[0].removeprefix('def '))
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'get_tracked_files', tracked_files)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {first}\n'
            f'*** Update File: {second}\n'
            f'*** Update File: {third}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert discovery_calls == [str(first_repo), str(second_repo)]
        assert enablement_calls == [str(first_repo), str(second_repo)]
        assert diff_calls == [str(first_repo), str(second_repo)]
        assert tracked_calls == [
            (str(first_repo), ('one.py', 'three.py')),
            (str(second_repo), ('two.py',)),
        ]
        assert processed == ['one', 'two', 'three']

    def test_post_tool_batch_nested_worktree_uses_own_hook_setting(
        self,
        tmp_path: Path,
    ) -> None:
        outer = tmp_path / 'outer'
        nested = outer / 'nested'
        _init_hook_repo(outer, hook_enabled=True)
        _init_hook_repo(nested, hook_enabled=False)
        outer_path = outer / 'outer.py'
        nested_path = nested / 'nested.py'
        _write_long_arg_docstring(outer_path)
        _write_long_arg_docstring(nested_path)
        outer_before = outer_path.read_text()
        nested_before = nested_path.read_text()
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {outer_path}\n'
            f'*** Update File: {nested_path}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event)) == 0
        assert outer_path.read_text() != outer_before
        assert nested_path.read_text() == nested_before

    def test_post_tool_batch_normalizes_dot_dot_path_for_git(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        (tmp_path / 'sub').mkdir()
        path = tmp_path / 'source.py'
        path.write_text('def source():\n    """Short."""\n    pass\n')
        git_commit_all(tmp_path, 'add source')
        path.write_text('def source():\n    """Changed."""\n    pass\n')
        diff_calls: list[str] = []

        def batch_diff(repo_root: str) -> str:
            diff_calls.append(repo_root)
            return 'snapshot'

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {'source.py': [(1, 1, [])]}

        def wrap_source(
            source: str,
            _config: object,
            _eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = '*** Begin Patch\n*** Update File: sub/../source.py\n*** End Patch\n'
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert len(diff_calls) == 1
        assert Path(diff_calls[0]) == tmp_path

    def test_post_tool_batch_one_repository_distributes_hunks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        first = repo / 'first.py'
        second = repo / 'second.py'
        first.write_text('def first():\n    """Short."""\n    pass\n')
        second.write_text('def second():\n    """Short."""\n    pass\n')
        discovery_calls: list[str] = []
        enablement_calls: list[str] = []
        diff_calls: list[tuple[str, bool]] = []
        tracked_calls: list[tuple[str, tuple[str, ...]]] = []
        config_calls: list[Path] = []
        eligible_by_file: dict[str, set[tuple[int, int]] | None] = {}
        real_find_config = hook_mod.find_config

        def discover(cwd: str) -> RepoDiscoveryResult:
            discovery_calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(repo))

        def hook_enabled(cwd: str | None = None) -> bool:
            assert cwd is not None
            enablement_calls.append(cwd)
            return True

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> str:
            diff_calls.append((repo_root, cached))
            return 'snapshot'

        def tracked_files(repo_root: str, paths: list[str]) -> set[str]:
            tracked_calls.append((repo_root, tuple(paths)))
            return {'second.py'}

        def find_config(path: Path) -> object:
            config_calls.append(path)
            return real_find_config(path)

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {'first.py': [(1, 1, [])]}

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            name = 'first.py' if 'def first' in source else 'second.py'
            eligible_by_file[name] = eligible_ranges
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'get_tracked_files', tracked_files, raising=False)
        monkeypatch.setattr(hook_mod, 'find_config', find_config)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {first}\n'
            f'*** Update File: {second}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert discovery_calls == [str(repo)]
        assert enablement_calls == [str(repo)]
        assert diff_calls == [(str(repo), False)]
        assert tracked_calls == [(str(repo), ('second.py',))]
        assert config_calls == [first, second]
        assert eligible_by_file == {'first.py': {(1, 3)}, 'second.py': set()}

    def test_post_tool_batch_singleton_reuses_discovered_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        path = repo / 'source.py'
        path.write_text('def source():\n    """Short."""\n    pass\n')
        discovery_calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            discovery_calls.append(cwd)
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(repo))

        def batch_diff(_repo_root: str) -> None:
            return

        def wrap_source(
            source: str,
            _config: object,
            _eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = f'*** Begin Patch\n*** Update File: {path}\n*** End Patch\n'
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event)) == 0
        assert discovery_calls == [str(repo)]

    def test_post_tool_batch_two_repositories_retain_path_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first_repo = tmp_path / 'first-repo'
        second_repo = tmp_path / 'second-repo'
        first_repo.mkdir()
        second_repo.mkdir()
        first = first_repo / 'first.py'
        second = second_repo / 'second.py'
        first.write_text('def first():\n    """Short."""\n    pass\n')
        second.write_text('def second():\n    """Short."""\n    pass\n')
        discovery_calls: list[str] = []
        enablement_calls: list[str] = []
        diff_calls: list[tuple[str, bool]] = []
        tracked_calls: list[tuple[str, tuple[str, ...]]] = []
        processed: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            discovery_calls.append(cwd)
            root = first_repo if Path(cwd).is_relative_to(first_repo) else second_repo
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(root))

        def hook_enabled(cwd: str | None = None) -> bool:
            assert cwd is not None
            enablement_calls.append(cwd)
            return True

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {}

        def batch_diff(
            repo_root: str,
            *,
            cached: bool = False,
        ) -> str:
            diff_calls.append((repo_root, cached))
            return 'snapshot'

        def tracked_files(repo_root: str, paths: list[str]) -> set[str]:
            tracked_calls.append((repo_root, tuple(paths)))
            return set(paths)

        def wrap_source(
            source: str,
            _config: object,
            _eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            processed.append('first.py' if 'def first' in source else 'second.py')
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'get_tracked_files', tracked_files, raising=False)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {second}\n'
            f'*** Update File: {first}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert discovery_calls == [str(second_repo), str(first_repo)]
        assert enablement_calls == [str(second_repo), str(first_repo)]
        assert diff_calls == [
            (str(second_repo), False),
            (str(first_repo), False),
        ]
        assert tracked_calls == [
            (str(second_repo), ('second.py',)),
            (str(first_repo), ('first.py',)),
        ]
        assert processed == ['second.py', 'first.py']

    def test_post_tool_batch_untracked_file_gets_full_eligibility(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        tracked = repo / 'tracked.py'
        untracked = repo / 'untracked.py'
        tracked.write_text('def tracked():\n    """Short."""\n    pass\n')
        untracked.write_text('def untracked():\n    """Short."""\n    pass\n')
        tracked_calls: list[tuple[str, tuple[str, ...]]] = []
        eligible_by_file: dict[str, set[tuple[int, int]] | None] = {}

        def discover(_cwd: str) -> RepoDiscoveryResult:
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(repo))

        def batch_diff(_repo_root: str) -> str:
            return 'snapshot'

        def parse_diff(_output: str) -> dict[str, list[DiffHunk]]:
            return {}

        def tracked_files(repo_root: str, paths: list[str]) -> set[str]:
            tracked_calls.append((repo_root, tuple(paths)))
            return {'tracked.py'}

        def wrap_source(
            source: str,
            _config: object,
            eligible_ranges: set[tuple[int, int]] | None,
            _incr_level: object,
        ) -> _WrapResult:
            name = 'tracked.py' if 'def tracked' in source else 'untracked.py'
            eligible_by_file[name] = eligible_ranges
            return _WrapResult(source=source, changed=False)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        monkeypatch.setattr(hook_mod, 'is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(hook_mod, 'get_repo_diff', batch_diff)
        monkeypatch.setattr(hook_mod, 'get_tracked_files', tracked_files, raising=False)
        monkeypatch.setattr(hook_mod, 'parse_diff_hunks', parse_diff)
        monkeypatch.setattr(hook_mod, '_wrap_with_config', wrap_source)
        patch = (
            '*** Begin Patch\n'
            f'*** Update File: {tracked}\n'
            f'*** Update File: {untracked}\n'
            '*** End Patch\n'
        )
        event = hook_event('PostToolUse', 'apply_patch', {'command': patch}, cwd=str(tmp_path))

        assert _run_codex(json.dumps(event), incremental=True) == 0
        assert tracked_calls == [(str(repo), ('tracked.py', 'untracked.py'))]
        assert eligible_by_file == {'tracked.py': set(), 'untracked.py': None}


class TestPostToolUseEncoding:
    @pytest.mark.parametrize(
        'case',
        [
            ('utf-8-sig', '', 'cafe', '\n', b'\xef\xbb\xbf', b'cafe'),
            ('iso-8859-1', '# coding: latin-1\n', 'caf\xe9', '\r\n', b'', b'caf\xe9'),
        ],
        ids=['utf-8-bom', 'latin-1-cookie'],
    )
    def test_changed_file_preserves_encoding_and_is_noop_on_pass_two(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        case: tuple[str, str, str, str, bytes, bytes],
    ) -> None:
        encoding, header, text, newline, expected_prefix, expected_bytes = case
        source = _post_tool_long_source(header=header, text=text).replace('\n', newline)
        raw = source.encode(encoding)
        path = tmp_path / 'encoded.py'
        path.write_bytes(raw)

        counts, read_bytes = _track_post_tool_io(path, monkeypatch)
        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        first = read_bytes(path)
        assert counts == {'reads': 1, 'writes': 1}
        assert first != raw
        assert first.startswith(expected_prefix)
        assert expected_bytes in first
        _assert_uniform_newlines(first, crlf=newline == '\r\n')

        counts.update(reads=0, writes=0)
        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        assert counts == {'reads': 1, 'writes': 0}
        assert read_bytes(path) == first


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolUseExceptionHandling:
    @pytest.fixture(autouse=True)
    def _enable_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

    @pytest.mark.parametrize('marker', ['directory', 'file'])
    def test_nested_worktree_marker_keeps_repository_grouping(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch, marker: str
    ) -> None:
        nested = git_repo / 'nested'
        if marker == 'file':
            run_setup(
                ['git', 'worktree', 'add', '--detach', str(nested), 'HEAD'],
                cwd=git_repo,
            )
        else:
            nested.mkdir()
            run_setup(['git', 'init'], cwd=nested)
        (nested / 'sub').mkdir()
        paths = [git_repo / 'top.py', nested / 'a.py', nested / 'sub' / 'b.py']
        for path in paths:
            path.write_text(_post_tool_long_source())
        discovered: list[str] = []
        discover = hook_mod.discover_repo_root

        def track_discovery(cwd: str) -> RepoDiscoveryResult:
            discovered.append(cwd)
            return discover(cwd)

        monkeypatch.setattr(hook_mod, 'discover_repo_root', track_discovery)
        hook_mod.run_post_tool(list(map(str, paths)), str(git_repo), incremental=False)

        assert discovered == [str(git_repo), str(nested)]
        assert all(path.read_text() != _post_tool_long_source() for path in paths)

    def test_oserror_prints_stderr_returns_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('pass\n')

        def raise_oserror(
            _path: Path,
            _incremental_override: bool | None,
            *,
            repo_root: str | None = None,
        ) -> None:
            assert repo_root is not None
            raise OSError('disk failure')

        monkeypatch.setattr('docwrap._hook._prepare_post_tool_file', raise_oserror)
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(f)})
        result = _run_claude(json.dumps(event))
        assert result == 0
        assert 'PostToolUse error processing' in capsys.readouterr().err

    def test_post_tool_failed_restore_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / 'test.py'
        path.write_text(_post_tool_long_source())
        original_write_bytes = Path.write_bytes
        calls = 0

        def failed_writes(candidate: Path, data: bytes) -> int:
            nonlocal calls
            if candidate == path:
                calls += 1
                if calls == 1:
                    original_write_bytes(candidate, data[: len(data) // 2])
                    raise OSError(errno.ENOSPC, 'No space left on device')
                raise OSError('restore failed')
            return original_write_bytes(candidate, data)

        monkeypatch.setattr(Path, 'write_bytes', failed_writes)
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(path)})
        assert _run_claude(json.dumps(event)) == 0

        captured = capsys.readouterr()
        assert calls == 2
        assert 'restoring the original bytes also failed' in captured.err
        context = json.loads(captured.out)['hookSpecificOutput']['additionalContext']
        assert 'restoring the original bytes also failed' in context

    @pytest.mark.parametrize('file_count', [1, 2])
    def test_preparation_failure_has_common_diagnostic_and_preserves_later_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        file_count: int,
    ) -> None:
        paths = [tmp_path / f'file-{index}.py' for index in range(file_count)]
        source = _post_tool_long_source()
        for path in paths:
            path.write_text(source)
        prepare = hook_mod._prepare_post_tool_file

        def prepare_or_fail(
            path: Path, incremental_override: bool | None, *, repo_root: str
        ) -> hook_mod._PostToolFile:
            if path == paths[0]:
                raise OSError('injected preparation error')
            return prepare(path, incremental_override, repo_root=repo_root)

        monkeypatch.setattr(hook_mod, '_prepare_post_tool_file', prepare_or_fail)
        hook_mod.run_post_tool(list(map(str, paths)), str(tmp_path), incremental=False)

        assert (
            capsys.readouterr().err
            == f'docwrap: PostToolUse error processing {paths[0].as_posix()}\n'
        )
        assert paths[0].read_text() == source
        for path in paths[1:]:
            assert path.read_text() != source

    @pytest.mark.parametrize(
        'status', [RepoDiscoveryStatus.ERROR, RepoDiscoveryStatus.NOT_WORKTREE]
    )
    def test_failed_or_absent_discovery_does_not_reuse_fallback_for_descendants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: RepoDiscoveryStatus
    ) -> None:
        nested = tmp_path / 'nested'
        nested.mkdir()
        paths = [tmp_path / 'a.py', nested / 'b.py']
        for path in paths:
            path.write_text(_post_tool_long_source())
        calls: list[str] = []

        def discover(cwd: str) -> RepoDiscoveryResult:
            calls.append(cwd)
            if cwd == str(tmp_path):
                return RepoDiscoveryResult(status)
            return RepoDiscoveryResult(RepoDiscoveryStatus.WORKTREE, str(nested))

        monkeypatch.setattr(hook_mod, 'discover_repo_root', discover)
        hook_mod.run_post_tool(list(map(str, paths)), str(tmp_path), incremental=False)

        assert calls == [str(tmp_path), str(nested)]
        assert all(path.read_text() != _post_tool_long_source() for path in paths)

    @pytest.mark.parametrize('tool_name', [None, [], {}, 1, True], ids=repr)
    @pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
    def test_non_string_tool_name_is_an_unsupported_event(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        provider: str,
        tool_name: object,
    ) -> None:
        path = tmp_path / 'test.py'
        original = _post_tool_long_source().encode()
        path.write_bytes(original)

        if provider == 'codex':
            direct_payload: dict[str, object] = {
                'hook_event_name': 'PostToolUse',
                'tool_name': tool_name,
                'tool_input': {'command': f'*** Update File: {path}'},
                'cwd': str(tmp_path),
            }
            raw_payload = direct_payload
        elif provider == 'antigravity':
            direct_payload = {
                'hook_event_name': 'PostToolUse',
                'tool_name': tool_name,
                'tool_input': {'file_path': str(path)},
                'cwd': str(tmp_path),
                'model': 'antigravity',
            }
            raw_payload = {
                'toolCall': {'name': tool_name, 'args': {'TargetFile': str(path)}},
                'workspacePaths': [str(tmp_path)],
            }
        else:
            direct_payload = {
                'hook_event_name': 'PostToolUse',
                'tool_name': tool_name,
                'tool_input': {'file_path': str(path)},
                'cwd': str(tmp_path),
            }
            raw_payload = direct_payload

        assert _run_provider(provider, json.dumps(raw_payload), event='PostToolUse') == 0
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == ''
        assert path.read_bytes() == original

    @pytest.mark.parametrize(
        ('tool_name', 'tool_input'),
        [
            ('Edit', {'file_path': 'test.py'}),
            ('Write', {'file_path': 'test.py'}),
            ('apply_patch', {'command': '*** Update File: test.py'}),
        ],
    )
    def test_supported_string_tool_names_keep_routing(
        self,
        tmp_path: Path,
        tool_name: str,
        tool_input: dict[str, object],
    ) -> None:
        payload: dict[str, object] = {
            'hook_event_name': 'PostToolUse',
            'tool_name': tool_name,
            'tool_input': tool_input,
            'cwd': str(tmp_path),
        }
        adapter = load_adapter('codex' if tool_name == 'apply_patch' else 'claude')
        parsed = adapter.parse(payload, None)

        assert parsed is not None
        assert _resolve_edited_paths(parsed.edited_paths, str(tmp_path)) == [
            str(tmp_path / 'test.py')
        ]

    def test_syntax_error_prints_stderr_returns_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_bytes(b'# -*- coding: bogus -*-\npass\n')
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(f)})
        result = _run_claude(json.dumps(event))
        assert result == 0
        assert 'PostToolUse error processing' in capsys.readouterr().err

    def test_unexpected_exception_prints_traceback_returns_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        f.write_text('pass\n')

        def raise_unexpected(
            _path: Path,
            _incremental_override: bool | None,
            *,
            repo_root: str | None = None,
        ) -> None:
            assert repo_root is not None
            raise RuntimeError('unexpected bug')

        monkeypatch.setattr('docwrap._hook._prepare_post_tool_file', raise_unexpected)
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(f)})
        result = _run_claude(json.dumps(event))
        assert result == 0
        err = capsys.readouterr().err
        assert 'RuntimeError' in err
        assert 'unexpected bug' in err


@pytest.mark.usefixtures('registered_hosts')
class TestPostToolUseIncrementalOverride:
    @pytest.fixture(autouse=True)
    def _enable_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

    def test_incremental_false_forces_full_wrap(self, tmp_path: Path) -> None:
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
        f.write_text('pass\n')
        git_commit_all(tmp_path, 'add foo')
        _write_long_arg_docstring(f)

        event = hook_event('PostToolUse', 'Edit', {'file_path': str(f)})
        _run_claude(json.dumps(event), incremental=False)

        lines = f.read_text().splitlines()
        assert all(len(line) <= 79 for line in lines)

    def test_incremental_true_preserves_unchanged_definition(self, tmp_path: Path) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        path = tmp_path / 'foo.py'
        path.write_text(_hook_selection_source(1))
        git_commit_all(tmp_path, 'add selection fixture')
        changed = _hook_selection_source(2)
        path.write_text(changed)
        event = hook_event(
            'PostToolUse',
            'Edit',
            {'file_path': str(path)},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event), incremental=True) == 0
        assert path.read_text() == changed


class TestPostToolUseNewlines:
    @pytest.mark.parametrize(
        ('newline', 'crlf'),
        [
            ('\n', False),
            ('\r\n', True),
            ('\r', False),
            ('mixed', True),
        ],
        ids=['lf', 'crlf', 'lone-cr', 'mixed'],
    )
    def test_changed_file_writes_once_then_is_noop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        newline: str,
        crlf: bool,
    ) -> None:
        source = _post_tool_long_source()
        if newline == 'mixed':
            raw = source.replace('\n', '\r\n', 1).encode()
        else:
            raw = source.replace('\n', newline).encode()
        path = tmp_path / 'changed.py'
        path.write_bytes(raw)
        counts, read_bytes = _track_post_tool_io(path, monkeypatch)

        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        first = read_bytes(path)
        assert counts == {'reads': 1, 'writes': 1}
        assert first != raw
        _assert_uniform_newlines(first, crlf=crlf)

        counts.update(reads=0, writes=0)
        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        assert counts == {'reads': 1, 'writes': 0}
        assert read_bytes(path) == first

    @pytest.mark.parametrize(
        'raw',
        [
            _POST_TOOL_WRAPPED_SOURCE.encode(),
            _POST_TOOL_WRAPPED_SOURCE.replace('\n', '\r\n').encode(),
            _POST_TOOL_WRAPPED_SOURCE.replace('\n', '\r').encode(),
            _POST_TOOL_WRAPPED_SOURCE.replace('\n', '\r\n', 1).encode(),
        ],
        ids=['lf', 'crlf', 'lone-cr', 'mixed'],
    )
    def test_unchanged_file_reads_once_without_writing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raw: bytes,
    ) -> None:
        path = tmp_path / 'unchanged.py'
        path.write_bytes(raw)
        counts, read_bytes = _track_post_tool_io(path, monkeypatch)

        hook_mod.run_post_tool([str(path)], str(path.parent), incremental=False)

        assert counts == {'reads': 1, 'writes': 0}
        assert read_bytes(path) == raw


class TestPreToolUseErrorHandling:
    @pytest.mark.usefixtures('registered_hosts')
    def test_allow_on_failure(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(
            'docwrap._hook.get_staged_py_info',
            _git_file_list_failure,
        )
        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit -m "test"'},
            'cwd': str(git_repo),
        }
        _run_claude(json.dumps(event), allow_on_failure=True)
        captured = capsys.readouterr()
        assert 'WARNING' in captured.err

    @pytest.mark.usefixtures('registered_hosts')
    def test_catchall_honors_config_allow_on_failure(
        self,
        git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (git_repo / 'pyproject.toml').write_text('[tool.docwrap]\nallow-on-failure = true\n')
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        def _boom(*_args: object, **_kwargs: object) -> int:
            raise RuntimeError('injected downstream failure')

        monkeypatch.setattr('docwrap._hook._handle_normal_commit', _boom)

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit -m "test"'},
            'cwd': str(git_repo),
        }
        result = _run_claude(json.dumps(event))  # no allow_on_failure on CLI
        captured = capsys.readouterr()

        assert result == 0, f'expected passthrough (0), got {result}'
        assert 'WARNING' in captured.err

    @pytest.mark.usefixtures('registered_hosts')
    def test_git_failure_denies(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)
        monkeypatch.setattr(
            'docwrap._hook.get_staged_py_info',
            _git_file_list_failure,
        )

        event = {
            'hook_event_name': 'PreToolUse',
            'tool_name': 'Bash',
            'tool_input': {'command': 'git commit -m "test"'},
            'cwd': str(git_repo),
        }
        assert _run_claude(json.dumps(event)) == 0
        assert json.loads(capsys.readouterr().out) == {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': 'docwrap: failed to list staged files',
            }
        }

    @pytest.mark.parametrize('allow_on_failure', [False, True], ids=['deny', 'allow'])
    @pytest.mark.parametrize(
        'status',
        [
            BaselineStatus.CONTENT,
            BaselineStatus.MISSING,
            BaselineStatus.DECODE_FAILED,
            BaselineStatus.INFRA_ERROR,
            None,
        ],
        ids=['content', 'missing', 'decode-failed', 'infra-error', 'absent'],
    )
    def test_staged_blob_status_uses_explicit_failure_policy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        status: BaselineStatus | None,
        allow_on_failure: bool,
    ) -> None:
        info = StagedFileInfo('target.py', '100644', 'a' * 40, '0' * 40, 'M')
        contents = (
            {}
            if status is None
            else {
                info.path: BaselineResult(
                    status=status,
                    content='x = 1\n' if status is BaselineStatus.CONTENT else None,
                )
            }
        )
        processed: list[str] = []

        def track_process(*_args: object, **_kwargs: object) -> None:
            processed.append(info.path)

        def staged_contents(
            _cwd: str,
            _info_list: list[StagedFileInfo],
        ) -> dict[str, BaselineResult]:
            return contents

        monkeypatch.setattr('docwrap._hook.read_staged_blobs', staged_contents)
        monkeypatch.setattr('docwrap._hook._process_staged_blob', track_process)
        mode = CommitMode(
            auto_stage=False,
            include=False,
            only=False,
            pathspecs=(),
            pathspec_from_file=None,
            pathspec_file_nul=False,
        )
        ctx = _PreToolContext(
            cwd=str(tmp_path),
            repo_root=str(tmp_path),
            incr_level=None,
            allow_on_failure=allow_on_failure,
            mode=mode,
            pre_commit_adds=(),
        )

        result = _wrap_staged_blobs(ctx, [info])
        captured = capsys.readouterr()

        if status is BaselineStatus.CONTENT:
            assert result == ([], [])
            assert processed == ['target.py']
            assert captured.out == ''
            assert captured.err == ''
        elif status in {BaselineStatus.MISSING, BaselineStatus.DECODE_FAILED}:
            assert result == ([], [])
            assert processed == []
            assert captured.out == ''
            assert captured.err == ''
        elif allow_on_failure:
            assert result == Outcome('allow')
            assert processed == []
            assert captured.out == ''
            assert 'WARNING' in captured.err
            assert 'target.py' in captured.err
        else:
            assert isinstance(result, Outcome)
            assert result.decision == 'deny'
            assert result.reason is not None
            assert 'target.py' in result.reason
            assert processed == []
            assert captured.out == ''
            assert captured.err == ''


@pytest.mark.usefixtures('registered_hosts')
class TestPreToolUsePreCommitAdds:
    @pytest.mark.parametrize(
        'command',
        [
            'GIT_INDEX_FILE=/tmp/alternate-index git add target.py && git commit',
            'env GIT_INDEX_FILE=/tmp/alternate-index git add target.py && git commit',
            'GIT_INDEX_FILE=/tmp/alternate-index git commit',
            'env GIT_INDEX_FILE=/tmp/alternate-index git commit',
        ],
        ids=['add-direct', 'add-env', 'commit-direct', 'commit-env'],
    )
    def test_alternate_index_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        def fail_if_called(*_args: object, **_kwargs: object) -> None:
            pytest.fail('repository-dependent operation was called')

        for name in (
            'discover_repo_root',
            'run_pre_commit_adds',
        ):
            monkeypatch.setattr(hook_mod, name, fail_if_called)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': command},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        _assert_denied(capsys)

    def test_attr_source_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _fail_on_repository_work(monkeypatch)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': ('git --attr-source=HEAD add \':(attr:sync)\' && git commit -m "test"')},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        _assert_denied(capsys)

    def test_cd_target_commit_wraps_staged_blob(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_repo = tmp_path / 'target'
        target_repo.mkdir()
        run_setup(['git', 'init'], cwd=str(target_repo))
        (target_repo / 'baseline.py').write_text('# baseline\n')
        git_commit_all(target_repo, 'initial')
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(target_repo))
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': f'cd {shell_path(target_repo)} && git commit -m "test"'},
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')

    @pytest.mark.parametrize(
        ('newline', 'crlf'),
        [(b'\n', False), (b'\r\n', True), (b'\r', False)],
        ids=['lf', 'crlf', 'lone-cr'],
    )
    def test_compound_add_sync_uses_supported_newline_policy(
        self, tmp_path: Path, newline: bytes, crlf: bool
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        run_setup(
            ['git', 'config', 'core.autocrlf', 'false'],
            cwd=tmp_path,
        )
        target = tmp_path / 'target.py'
        source = (
            b'def foo(value):\n'
            b'    """Return the value.\n'
            b'\n'
            b'    Args:\n'
            b'        value: This is a very long argument description that should'
            b' definitely be wrapped by docwrap before the staged file is committed.\n'
            b'    """\n'
            b'    return value\n'
        ).replace(b'\n', newline)
        target.write_bytes(source)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add target.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        staged = run_setup(
            ['git', 'show', ':target.py'],
            cwd=tmp_path,
        ).stdout
        worktree = target.read_bytes()
        expected_newline = b'\r\n' if crlf else b'\n'
        assert b'definitely' + expected_newline in staged
        assert b'definitely' + expected_newline in worktree
        _assert_uniform_newlines(staged, crlf=crlf)
        _assert_uniform_newlines(worktree, crlf=crlf)

    def test_cross_repository_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        add_repo = tmp_path / 'add-repo'
        commit_repo = tmp_path / 'commit-repo'
        add_repo.mkdir()
        commit_repo.mkdir()
        _fail_on_repository_work(monkeypatch)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {
                'command': (
                    f'git -C {shell_path(add_repo)} add foo.py && '
                    f'git -C {shell_path(commit_repo)} commit -m "test"'
                )
            },
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        _assert_denied(capsys)

    def test_dash_c_add_and_dash_c_commit_wraps_and_syncs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_repo = tmp_path / 'target'
        target_repo.mkdir()
        run_setup(['git', 'init'], cwd=str(target_repo))
        (target_repo / 'baseline.py').write_text('# baseline\n')
        git_commit_all(target_repo, 'initial')
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {
                'command': (
                    f'git -C {shell_path(target_repo)} add foo.py && '
                    f'git -C {shell_path(target_repo)} commit -m "test"'
                )
            },
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')
        assert 'definitely\n' in f.read_text()

    def test_directory_add_preserves_unrelated_partial_staging(self, tmp_path: Path) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        src = tmp_path / 'src'
        src.mkdir()
        target = src / 'target.py'
        target.write_text('target = 1\n')
        other = tmp_path / 'other.py'
        other.write_text('other = 1\n')
        git_commit_all(tmp_path, 'tracked files')
        _write_long_arg_docstring(target)
        _write_long_arg_docstring(other)
        run_setup(['git', 'add', 'other.py'], cwd=tmp_path)
        other.write_text('# unstaged prefix\n' + other.read_text())
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add src && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert 'definitely\n' in _staged_text(tmp_path, 'other.py')
        assert other.read_text().startswith('# unstaged prefix\n')
        assert 'definitely\n' in _staged_text(tmp_path, 'src/target.py')
        assert 'definitely\n' in target.read_text()

    def test_dry_run_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _fail_on_repository_work(monkeypatch)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add --dry-run src && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        _assert_denied(capsys)

    def test_env_prefixed_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _fail_on_repository_work(monkeypatch)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'FOO=1 git add foo.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        _assert_denied(capsys)

    def test_filter_divergence_leaves_worktree_untouched(self, tmp_path: Path) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        filtered = tmp_path / 'filtered.py'
        (tmp_path / '.gitattributes').write_text('filtered.py ident\n')
        filtered.write_text(
            '# $Id$\n\ndef filtered(value):\n    """Return the value."""\n    return value\n'
        )
        git_commit_all(tmp_path, 'track filtered file')
        filtered.unlink()
        run_setup(
            ['git', 'checkout', '--', 'filtered.py'],
            cwd=tmp_path,
        )
        expanded_source = filtered.read_text()
        assert expanded_source.startswith('# $Id: ')
        long_docstring = (
            '    """Return the value.\n'
            '\n'
            '    Args:\n'
            '        value: This is a very long argument description that should'
            ' definitely be wrapped by docwrap before the staged file is committed.\n'
            '    """\n'
        )
        clean_source = '# $Id$\n\ndef filtered(value):\n' + long_docstring + '    return value\n'
        filtered.write_text(
            expanded_source.replace(
                '    """Return the value."""\n',
                long_docstring,
            )
        )
        worktree_before = filtered.read_bytes()
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add filtered.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert 'definitely\n' in _staged_text(tmp_path, 'filtered.py')
        assert filtered.read_bytes() == worktree_before

        run_setup(['git', 'add', 'filtered.py'], cwd=tmp_path)
        assert _staged_text(tmp_path, 'filtered.py') == clean_source

    def test_interactive_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_if_called(*_args: object, **_kwargs: object) -> None:
            pytest.fail('repository-dependent operation was called')

        for name in (
            'discover_repo_root',
            'run_pre_commit_adds',
        ):
            monkeypatch.setattr(hook_mod, name, fail_if_called)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add --interactive target.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        _assert_denied(capsys)

    def test_literal_pathspec_add_preserves_unrelated_partial_staging(
        self, tmp_path: Path
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        literal_path = tmp_path / '[literal].py'
        literal_path.write_text('literal = 1\n')
        other = tmp_path / 'other.py'
        other.write_text('other = 1\n')
        git_commit_all(tmp_path, 'tracked files')
        _write_long_arg_docstring(literal_path)
        _write_long_arg_docstring(other)
        run_setup(['git', 'add', 'other.py'], cwd=tmp_path)
        other.write_text('# unstaged prefix\n' + other.read_text())
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add -- \':(literal)[literal].py\' && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        assert 'definitely\n' in _staged_text(tmp_path, '[literal].py')
        assert 'definitely\n' in literal_path.read_text()
        assert 'definitely\n' in _staged_text(tmp_path, 'other.py')
        assert other.read_text().startswith('# unstaged prefix\n')

    def test_mixed_dash_c_add_is_denied_before_repository_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / 'sub').mkdir()
        _fail_on_repository_work(monkeypatch)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git -C sub add foo.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        _assert_denied(capsys)

    def test_replay_failure_reports_the_git_cause(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)
        (tmp_path / 'a.py').write_text('value = 1\n')
        (tmp_path / '.git' / 'index.lock').touch()
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add a.py && git commit -m x'},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0
        output = json.loads(capsys.readouterr().out)
        reason = output['hookSpecificOutput']['permissionDecisionReason']
        assert reason.startswith('docwrap: failed to replay git add: ')
        assert 'index.lock' in reason

    @pytest.mark.parametrize('allow_on_failure', [False, True], ids=['deny', 'allow'])
    def test_replay_failure_stops_before_hook_mutation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        allow_on_failure: bool,
    ) -> None:
        _init_hook_repo(tmp_path, hook_enabled=True)

        def fail_if_called(*_args: object, **_kwargs: object) -> None:
            pytest.fail('downstream hook work was called')

        def fail_replay(_adds: tuple[PreCommitAdd, ...]) -> str:
            return 'simulated failure'

        monkeypatch.setattr(hook_mod, 'run_pre_commit_adds', fail_replay)
        for name in (
            'get_staged_py_info',
            'get_modified_tracked_py_files',
            'resolve_pathspec_py_files',
            'read_staged_blobs',
            'apply_index_entries',
        ):
            monkeypatch.setattr(hook_mod, name, fail_if_called)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add target.py && git commit -m "test"'},
            cwd=str(tmp_path),
        )

        result = _run_claude(json.dumps(event), allow_on_failure=allow_on_failure)
        captured = capsys.readouterr()

        if allow_on_failure:
            assert result == 0
            assert captured.out == ''
            assert captured.err == (
                'WARNING: docwrap: failed to replay git add: simulated failure\n'
            )
        else:
            assert result == 0
            output = json.loads(captured.out)
            assert output['hookSpecificOutput']['permissionDecisionReason'] == (
                'docwrap: failed to replay git add: simulated failure'
            )
            assert captured.err == 'docwrap: failed to replay git add: simulated failure\n'

    def test_unsafe_cd_passes_through_without_touching_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_repo = tmp_path / 'target'
        target_repo.mkdir()
        run_setup(['git', 'init'], cwd=str(target_repo))
        (target_repo / 'baseline.py').write_text('# baseline\n')
        git_commit_all(target_repo, 'initial')
        f = target_repo / 'foo.py'
        _write_long_arg_docstring(f)
        run_setup(['git', 'add', 'foo.py'], cwd=str(target_repo))
        before = _staged_text(target_repo, 'foo.py')
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': f'cd {shell_path(target_repo)} ; git commit -m "test"'},
            cwd=str(tmp_path),
        )
        result = _run_claude(json.dumps(event))

        assert result == 0
        assert _staged_text(target_repo, 'foo.py') == before


@pytest.mark.usefixtures('registered_hosts')
class TestProcessFilePostToolUse:
    @pytest.fixture(autouse=True)
    def _enable_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

    def test_normal_file_wrapped(self, tmp_path: Path) -> None:
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
        f.write_text('pass\n')
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
        run_setup(
            [
                'git',
                '-c',
                'user.name=Test',
                '-c',
                'user.email=test@test.com',
                'commit',
                '-m',
                'add foo',
            ],
            cwd=str(tmp_path),
        )
        # Now modify with a long docstring
        _write_long_arg_docstring(f)

        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Edit',
            'tool_input': {'file_path': str(f)},
        }
        _run_claude(json.dumps(event))

        lines = f.read_text().splitlines()
        assert all(len(line) <= 79 for line in lines)

    def test_symlink_skipped_in_post_tool_use(self, tmp_path: Path) -> None:
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
        real_file = tmp_path / 'real.py'
        _write_long_arg_docstring(real_file)
        link = tmp_path / 'link.py'
        link.symlink_to(real_file)
        regular_file = tmp_path / 'regular.py'
        _write_long_arg_docstring(regular_file)

        regular_source = regular_file.read_text()
        regular_event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Edit',
            'tool_input': {'file_path': str(regular_file)},
        }
        _run_claude(json.dumps(regular_event))
        assert regular_file.read_text() != regular_source

        original_content = real_file.read_text()
        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Edit',
            'tool_input': {'file_path': str(link)},
        }
        _run_claude(json.dumps(event))

        assert real_file.read_text() == original_content

    def test_untracked_file_falls_back_to_full_wrap(self, tmp_path: Path) -> None:
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
        f = tmp_path / 'new_file.py'
        _write_long_arg_docstring(f)

        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Edit',
            'tool_input': {'file_path': str(f)},
        }
        _run_claude(json.dumps(event))

        lines = f.read_text().splitlines()
        assert all(len(line) <= 79 for line in lines)


class TestProtocol:
    # Pin is_hook_enabled to True so tests are not affected by the
    # developer's local git config (docwrap.hook).
    @pytest.fixture(autouse=True)
    def _enable_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('docwrap._hook.is_hook_enabled', _hook_enabled)

    @pytest.mark.usefixtures('registered_hosts')
    def test_always_exits_0(self) -> None:
        result = _run_claude('invalid json')
        assert result == 0

    @pytest.mark.usefixtures('registered_hosts')
    def test_codex_apply_patch_processes_python_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        f = tmp_path / 'test.py'
        original = _post_tool_long_source()
        f.write_text(original)
        event = {
            'hook_event_name': 'PostToolUse',
            'tool_name': 'apply_patch',
            'tool_input': {
                'command': (
                    '*** Begin Patch\n'
                    '*** Update File: test.py\n'
                    '@@\n'
                    '-def foo():\n'
                    '+def foo():\n'
                    '*** End Patch\n'
                )
            },
            'cwd': str(tmp_path),
            'model': 'gpt-5.5',
            'turn_id': 'turn_test',
        }
        result = _run_codex(json.dumps(event))
        assert result == 0
        assert capsys.readouterr().out == ''
        assert f.read_text() != original

    @pytest.mark.usefixtures('registered_hosts')
    def test_edit_event_processes_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        original = _post_tool_long_source()
        f.write_text(original)
        event = hook_event('PostToolUse', 'Edit', {'file_path': str(f)})
        result = _run_claude(json.dumps(event))
        assert result == 0
        # File should be modified
        assert f.read_text() != original

    def test_hook_ignores_convert_config(self, tmp_path: Path) -> None:
        # Create pyproject.toml with convert = "numpy"
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nconvert = "numpy"\nstrip-types = true\n'
        )
        # Create a file with Google docstring
        py_file = tmp_path / 'example.py'
        py_file.write_text(
            'def f(x):\n    """Summary.\n\n    Args:\n        x (int): The value.\n    """\n'
        )

        hook_mod.run_post_tool([str(py_file)], str(py_file.parent))

        # The file should NOT be converted to NumPy
        result = py_file.read_text()
        assert 'Args:' in result  # Still Google style
        assert 'Parameters' not in result  # Not converted to NumPy
        assert '(int)' in result  # Types not stripped

    @pytest.mark.usefixtures('registered_hosts')
    def test_non_edit_write_ignored(self) -> None:
        event = hook_event('PostToolUse', 'Read', {'file_path': '/tmp/test.py'})  # noqa: S108
        result = _run_claude(json.dumps(event))
        assert result == 0

    @pytest.mark.usefixtures('registered_hosts')
    def test_non_python_ignored(self) -> None:
        event = hook_event('PostToolUse', 'Edit', {'file_path': '/tmp/test.txt'})  # noqa: S108
        result = _run_claude(json.dumps(event))
        assert result == 0

    @pytest.mark.usefixtures('registered_hosts')
    def test_post_tool_use_preserves_long_summary_and_wraps_body(self, tmp_path: Path) -> None:
        long_summary = 'Summary ' + 'stays ' * 15 + 'unchanged.'
        long_body = 'Body prose ' + 'wrap ' * 20 + 'finishes here.'
        source = f'def foo():\n    """{long_summary}\n\n    {long_body}\n    """\n'
        path = tmp_path / 'test.py'
        path.write_text(source)
        event = hook_event(
            'PostToolUse',
            'Write',
            {'file_path': str(path)},
            cwd=str(tmp_path),
        )

        assert _run_claude(json.dumps(event)) == 0

        result = path.read_text()
        assert not (tmp_path / 'pyproject.toml').exists()
        assert f'    """{long_summary}\n' in result
        assert long_body not in result
        assert '\n    wrap wrap ' in result

    @pytest.mark.usefixtures('registered_hosts')
    def test_write_event_processes_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        original = _post_tool_long_source()
        f.write_text(original)
        event = hook_event('PostToolUse', 'Write', {'file_path': str(f)})
        result = _run_claude(json.dumps(event))
        assert result == 0
        assert f.read_text() != original


def _fail_pathspecs(
    _cwd: str,
    _pathspecs: list[str],
    *,
    repo_root: str,
    pathspec_from_file: str | None = None,
    pathspec_file_nul: bool = False,
) -> None:
    del repo_root, pathspec_from_file, pathspec_file_nul


def _hook_enabled(_cwd: str | None = None) -> bool:
    return True


def _post_tool_long_source(*, header: str = '', text: str = 'cafe') -> str:
    return (
        header
        + 'def f():\n'
        + f'    """A {text} summary.\n'
        + '\n'
        + '    This body prose is deliberately long enough to exceed the configured '
        + 'default line length and require wrapping by docwrap.\n'
        + '    """\n'
        + '    pass\n'
    )


def _hook_selection_source(value: int, *, wrapped: bool = False) -> str:
    first_definition = _POST_TOOL_WRAPPED_SOURCE if wrapped else _post_tool_long_source()
    return first_definition + f'\n\ndef changed():\n    return {value}\n'


def _staged_text(repo: Path, path: str) -> str:
    return decode_text(
        run_setup(
            ['git', 'show', f':{path}'],
            cwd=str(repo),
        ).stdout
    )


def _track_post_tool_io(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, int], Callable[[Path], bytes]]:
    original_read_bytes = Path.read_bytes
    counts = {'reads': 0, 'writes': 0}
    resolved = path.resolve()

    def tracked_read_bytes(candidate: Path) -> bytes:
        if candidate.resolve() == resolved:
            counts['reads'] += 1
        return original_read_bytes(candidate)

    def tracked_write(
        candidate: Path,
        source: str,
        encoding: str,
        *,
        raw: bytes,
    ) -> None:
        if candidate.resolve() == resolved:
            counts['writes'] += 1
        write_preserving_newline(candidate, source, encoding, raw=raw)

    monkeypatch.setattr(Path, 'read_bytes', tracked_read_bytes)
    monkeypatch.setattr('docwrap._hook.write_preserving_newline', tracked_write)
    return counts, original_read_bytes


def _write_long_arg_docstring(path: Path) -> None:
    path.write_text(
        'def foo(value):\n'
        '    """Return the value.\n'
        '\n'
        '    Args:\n'
        '        value: This is a very long argument description that should'
        ' definitely be wrapped by docwrap before the staged file is committed.\n'
        '    """\n'
        '    return value\n'
    )


def _stage_long_docstring(repo: Path) -> None:
    _init_hook_repo(repo, hook_enabled=True)
    _write_long_arg_docstring(repo / 'foo.py')
    run_setup(['git', 'add', 'foo.py'], cwd=str(repo))


def _write_permission_candidate(path: Path, *, needs_wrap: bool) -> None:
    if needs_wrap:
        _write_long_arg_docstring(path)
    else:
        path.write_text('def changed():\n    """Short."""\n    pass\n')


@pytest.mark.usefixtures('registered_hosts')
def test_antigravity_non_worktree_commit_is_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _antigravity_pretooluse_payload(
        'git commit -m "test"',
        tmp_path,
    )

    assert _run_agy(payload) == 0
    assert capsys.readouterr().out == ''


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    ('workspace_enabled', 'target_enabled', 'expect_change'),
    [(False, True, True), (True, False, False)],
)
def test_antigravity_post_tool_use_uses_target_repository(
    tmp_path: Path,
    workspace_enabled: bool,
    target_enabled: bool,
    expect_change: bool,
) -> None:
    workspace = tmp_path / 'workspace'
    target = tmp_path / 'target'
    _init_hook_repo(workspace, hook_enabled=workspace_enabled)
    _init_hook_repo(target, hook_enabled=target_enabled)
    path = target / 'target.py'
    _write_long_arg_docstring(path)
    before = path.read_text()

    payload = _antigravity_payload(
        'write_to_file',
        {'TargetFile': str(path)},
        workspace,
    )
    assert _run_agy(payload, event='PostToolUse') == 0
    assert (path.read_text() != before) is expect_change


@pytest.mark.usefixtures('registered_hosts')
def test_bare_commit_preserves_concurrent_worktree_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_hook_repo(tmp_path, hook_enabled=True)
    path = tmp_path / 'example.py'
    _write_long_arg_docstring(path)
    run_setup(['git', 'add', 'example.py'], cwd=tmp_path)
    real_apply_index_entries = hook_mod.apply_index_entries
    marker = '# concurrent unstaged marker\n'

    def apply_index_entries_then_edit(repo_root: str, entries: list[IndexEntry]) -> bool:
        applied = real_apply_index_entries(repo_root, entries)
        if applied:
            path.write_text(marker + path.read_text())
        return applied

    monkeypatch.setattr(hook_mod, 'apply_index_entries', apply_index_entries_then_edit)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    assert path.read_text().startswith(marker)
    assert 'definitely\n' in _staged_text(tmp_path, 'example.py')


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('needs_wrap', [False, True])
def test_bare_commit_success_is_silent(
    git_repo_with_file: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    needs_wrap: bool,
) -> None:
    repo, path = git_repo_with_file
    run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=repo)
    _write_permission_candidate(path, needs_wrap=needs_wrap)
    run_setup(['git', 'add', 'foo.py'], cwd=repo)
    event = hook_event('PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(repo))
    assert _run_claude(json.dumps(event)) == 0
    _assert_success(capsys, 1 if needs_wrap else None)
    staged = _staged_text(repo, 'foo.py')
    assert ('definitely\n' in staged) is needs_wrap


@pytest.mark.usefixtures('registered_hosts')
def test_bootstrap_chain_is_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail('repository-dependent operation was called')

    monkeypatch.setattr('docwrap._hook.is_hook_enabled', fail_if_called)
    monkeypatch.setattr('docwrap._hook.discover_repo_root', fail_if_called)
    monkeypatch.setattr('docwrap._hook.run_pre_commit_adds', fail_if_called)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git init child && git -C child add . && git -C child commit -m "initial"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    _assert_denied(capsys)


@pytest.mark.usefixtures('registered_hosts')
def test_branch_commit_reaches_repository_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def record_discovery(cwd: str) -> RepoDiscoveryResult:
        seen.append(cwd)
        return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    monkeypatch.setattr(hook_mod, 'is_hook_enabled', _hook_enabled)
    monkeypatch.setattr(hook_mod, 'discover_repo_root', record_discovery)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit --branch -m "test"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    assert seen == [str(tmp_path)]


@pytest.mark.usefixtures('registered_hosts')
def test_codex_post_tool_use_gates_each_repository(
    tmp_path: Path,
) -> None:
    enabled_repo = tmp_path / 'enabled'
    disabled_repo = tmp_path / 'disabled'
    _init_hook_repo(enabled_repo, hook_enabled=True)
    _init_hook_repo(disabled_repo, hook_enabled=False)
    enabled_path = enabled_repo / 'enabled.py'
    disabled_path = disabled_repo / 'disabled.py'
    _write_long_arg_docstring(enabled_path)
    _write_long_arg_docstring(disabled_path)
    enabled_before = enabled_path.read_text()
    disabled_before = disabled_path.read_text()
    patch = (
        '*** Begin Patch\n'
        f'*** Update File: {enabled_path}\n'
        f'*** Update File: {disabled_path}\n'
        '*** End Patch\n'
    )
    event = hook_event(
        'PostToolUse',
        'apply_patch',
        {'command': patch},
        cwd=str(tmp_path),
    )

    assert _run_codex(json.dumps(event)) == 0
    assert enabled_path.read_text() != enabled_before
    assert disabled_path.read_text() == disabled_before


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('needs_wrap', [False, True])
def test_compound_add_success_is_silent(
    git_repo_with_file: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    needs_wrap: bool,
) -> None:
    repo, path = git_repo_with_file
    run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=repo)
    _write_permission_candidate(path, needs_wrap=needs_wrap)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git add foo.py && git commit -m "test"'},
        cwd=str(repo),
    )
    assert _run_claude(json.dumps(event)) == 0
    _assert_success(capsys, 1 if needs_wrap else None)
    staged = _staged_text(repo, 'foo.py')
    assert ('definitely\n' in staged) is needs_wrap


@pytest.mark.usefixtures('registered_hosts')
def test_existing_nested_repo_standalone_commit_keeps_processing(
    tmp_path: Path,
) -> None:
    _init_hook_repo(tmp_path, hook_enabled=True)
    child = tmp_path / 'child'
    _init_hook_repo(child, hook_enabled=True)
    path = child / 'example.py'
    _write_long_arg_docstring(path)
    run_setup(
        ['git', 'add', 'example.py'],
        cwd=child,
    )

    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git -C child commit -m "next"'},
        cwd=str(tmp_path),
    )
    assert _run_claude(json.dumps(event)) == 0
    staged = _staged_text(child, 'example.py')
    assert 'definitely\n' in staged


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
def test_false_predecessor_is_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    _fail_on_repository_work(monkeypatch)
    command = 'false && git add -A && git commit -m "x"'
    if provider == 'antigravity':
        payload = _antigravity_pretooluse_payload(command, tmp_path)
    else:
        event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))
        if provider == 'codex':
            event.update(
                {
                    'model': 'codex-test',
                    'turn_id': 'turn-test',
                    'tool_use_id': 'tool-test',
                }
            )
        payload = json.dumps(event)

    assert _run_provider(provider, payload) == 0
    _assert_denied(capsys, provider)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    [
        'git --version init && git commit -m "next"',
        'git --definitely-not-real init && git commit -m "x"',
        'git --definitely-not-real commit -m "x"',
    ],
    ids=['terminating', 'unknown-predecessor', 'unknown-selected-command'],
)
@pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
def test_git_global_option_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    _fail_on_repository_work(monkeypatch)

    if provider == 'antigravity':
        payload = _antigravity_pretooluse_payload(command, tmp_path)
    else:
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': command},
            cwd=str(tmp_path),
        )
        if provider == 'codex':
            event.update(
                {
                    'model': 'codex-test',
                    'turn_id': 'turn-test',
                    'tool_use_id': 'tool-test',
                }
            )
        payload = json.dumps(event)

    assert _run_provider(provider, payload) == 0
    _assert_denied(capsys, provider)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('include_wraps', [False, True])
@pytest.mark.parametrize('staged_wraps', [False, True])
def test_include_success_is_silent(
    git_repo_with_file: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    include_wraps: bool,
    staged_wraps: bool,
) -> None:
    repo, included = git_repo_with_file
    staged = repo / 'staged.py'
    staged.write_text('pass\n')
    git_commit_all(repo, 'add staged file')
    run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=repo)
    _write_permission_candidate(included, needs_wrap=include_wraps)
    _write_permission_candidate(staged, needs_wrap=staged_wraps)
    run_setup(['git', 'add', 'staged.py'], cwd=repo)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit --include foo.py -m "test"'},
        cwd=str(repo),
    )
    assert _run_claude(json.dumps(event)) == 0
    _assert_success(capsys, 1 if staged_wraps else None)
    assert ('definitely\n' in included.read_text()) is include_wraps
    assert ('definitely\n' in _staged_text(repo, 'staged.py')) is staged_wraps


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    [
        'git commit -a target.py -m test',
        'git commit --pathspec-from-file=/dev/null target.py -m test',
        'git commit -i -o target.py -m test',
        'git commit -i -m test',
    ],
    ids=[
        'auto-stage-with-pathspec',
        'inline-and-file-pathspec',
        'include-and-only',
        'include-without-pathspec',
    ],
)
@pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
def test_invalid_commit_mode_is_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    provider: str,
) -> None:
    _fail_on_repository_work(monkeypatch)

    if provider == 'antigravity':
        payload = _antigravity_pretooluse_payload(command, tmp_path)
    else:
        event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))
        if provider == 'codex':
            event.update(
                {
                    'model': 'codex-test',
                    'turn_id': 'turn-test',
                    'tool_use_id': 'tool-test',
                }
            )
        payload = json.dumps(event)

    assert _run_provider(provider, payload) == 0
    _assert_denied(capsys, provider)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    [
        'git add *.py && git commit -m "test"',
        'git add "$FILES" && git commit -m "test"',
        'cat <<\'EOF\'\ngit add secret.py\nEOF\ngit commit -m "test"',
        'export GIT_INDEX_FILE=/tmp/alternate-index; git commit -m "test"',
    ],
    ids=['glob-add', 'variable-add', 'heredoc-add', 'export-alternate-index'],
)
def test_literal_boundary_is_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    _fail_on_repository_work(monkeypatch)
    event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))

    assert _run_claude(json.dumps(event)) == 0
    _assert_denied(capsys)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
def test_multiline_continuation_is_supported_and_touches_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: str,
) -> None:
    _init_hook_repo(tmp_path, hook_enabled=True)
    path = tmp_path / 'baseline.py'
    path.write_bytes(b'# staged provider bytes\n')
    run_setup(['git', 'add', 'baseline.py'], cwd=tmp_path)
    path.write_bytes(b'# working-tree provider bytes\n')
    index_before = run_setup(
        ['git', 'diff', '--cached', '--binary'],
        cwd=tmp_path,
    ).stdout
    file_before = path.read_bytes()
    command = 'git commit -m "test" ' + BACKSLASH + '\n    -- narrow.py'
    if provider == 'codex':
        command = command.replace('git commit', f'git -C {shell_path(tmp_path)} commit', 1)
    if provider == 'antigravity':
        payload = _antigravity_pretooluse_payload(command, tmp_path)
    else:
        event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))
        if provider == 'codex':
            event.update(
                {
                    'model': 'codex-test',
                    'turn_id': 'turn-test',
                    'tool_use_id': 'tool-test',
                }
            )
        payload = json.dumps(event)

    assert _run_provider(provider, payload) == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
    index_after = run_setup(
        ['git', 'diff', '--cached', '--binary'],
        cwd=tmp_path,
    ).stdout
    assert index_after == index_before
    assert path.read_bytes() == file_before


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('provider', ['claude', 'codex', 'antigravity'])
def test_nested_bootstrap_is_denied_and_does_not_touch_outer_repo(
    tmp_path: Path,
    provider: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_hook_repo(tmp_path, hook_enabled=True)
    child = tmp_path / 'child'
    child.mkdir()
    path = child / 'example.py'
    _write_long_arg_docstring(path)
    before = path.read_text()
    command = 'git init child && git -C child add example.py && git -C child commit -m "initial"'

    if provider == 'antigravity':
        payload = _antigravity_pretooluse_payload(command, tmp_path)
    else:
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': command},
            cwd=str(tmp_path),
        )
        if provider == 'codex':
            event.update(
                {
                    'model': 'codex-test',
                    'turn_id': 'turn-test',
                    'tool_use_id': 'tool-test',
                }
            )
        payload = json.dumps(event)

    assert _run_provider(provider, payload) == 0
    _assert_denied(capsys, provider)
    staged = run_setup(
        ['git', 'diff', '--cached', '--name-only'],
        cwd=tmp_path,
    )
    assert decode_text(staged.stdout) == ''
    assert path.read_text() == before
    assert not (child / '.git').exists()


@pytest.mark.usefixtures('registered_hosts')
def test_no_staged_python_files_is_silent(
    git_repo_with_file: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = git_repo_with_file
    run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=repo)
    event = hook_event('PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(repo))
    assert _run_claude(json.dumps(event)) == 0
    _assert_success(capsys)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('lookup_fails', [False, True])
def test_non_repo_target_defaults_enabled_without_global_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lookup_fails: bool,
) -> None:
    workspace = tmp_path / 'workspace'
    _init_hook_repo(workspace, hook_enabled=False)
    global_config = tmp_path / 'global.gitconfig'
    if lookup_fails:
        global_config.mkdir()
    else:
        global_config.write_text('')
    _isolate_global_git(monkeypatch, tmp_path, global_config)
    path = tmp_path / 'outside' / 'target.py'
    path.parent.mkdir()
    _write_long_arg_docstring(path)
    before = path.read_text()
    event = hook_event('PostToolUse', 'Edit', {'file_path': str(path)}, cwd=str(workspace))

    assert _run_claude(json.dumps(event)) == 0
    assert path.read_text() != before


@pytest.mark.usefixtures('registered_hosts')
def test_non_repo_target_honors_global_disable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config = tmp_path / 'global.gitconfig'
    global_config.write_text('[docwrap]\n\thook = false\n')
    _isolate_global_git(monkeypatch, tmp_path, global_config)
    path = tmp_path / 'outside' / 'target.py'
    path.parent.mkdir()
    _write_long_arg_docstring(path)
    before = path.read_text()
    event = hook_event('PostToolUse', 'Edit', {'file_path': str(path)}, cwd=str(tmp_path))

    assert _run_claude(json.dumps(event)) == 0
    assert path.read_text() == before


@pytest.mark.usefixtures('registered_hosts')
def test_non_worktree_commit_is_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = hook_event(
        'PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(tmp_path)
    )

    assert _run_claude(json.dumps(event)) == 0
    assert capsys.readouterr().out == ''


@pytest.mark.usefixtures('registered_hosts')
def test_pre_tool_context_uses_hook_config_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_unknown_style(_path: Path) -> None:
        raise UnknownStyleError('invalid')

    _init_hook_repo(tmp_path, hook_enabled=True)
    monkeypatch.setattr('docwrap._hook.find_config', raise_unknown_style)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['hookSpecificOutput']['permissionDecisionReason'] == (
        'docwrap: unexpected error during wrapping'
    )


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_decode_failure_denies_exact_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    def raise_decode_error(*_args: object, **_kwargs: object) -> None:
        raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')

    monkeypatch.setattr(subprocess, 'run', raise_decode_error)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['hookSpecificOutput']['permissionDecisionReason'] == (
        'docwrap: failed to discover git work tree'
    )


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_error_denies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / '.git').mkdir()
    event = hook_event(
        'PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(tmp_path)
    )

    assert _run_claude(json.dumps(event)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert output['hookSpecificOutput']['permissionDecisionReason'] == (
        'docwrap: failed to discover git work tree'
    )


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_error_honors_config_allow_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / '.git').mkdir()
    (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nallow-on-failure = true\n')
    event = hook_event(
        'PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(tmp_path)
    )

    assert _run_claude(json.dumps(event)) == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'WARNING: docwrap: failed to discover git work tree' in captured.err


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_error_preserves_malformed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_unknown_style(_path: Path) -> None:
        raise UnknownStyleError('invalid')

    (tmp_path / '.git').mkdir()
    monkeypatch.setattr('docwrap._hook.find_config', raise_unknown_style)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['hookSpecificOutput']['permissionDecisionReason'] == (
        'docwrap: unexpected error during wrapping'
    )


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_invalid_cwd_denies_exact_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path / 'missing-directory'),
    )

    assert _run_claude(json.dumps(event)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['hookSpecificOutput']['permissionDecisionReason'] == (
        'docwrap: failed to discover git work tree'
    )


@pytest.mark.usefixtures('registered_hosts')
def test_repository_discovery_invalid_cwd_honors_explicit_allow_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git commit -m "test"'},
        cwd=str(tmp_path / 'missing-directory'),
    )

    assert _run_claude(json.dumps(event), allow_on_failure=True) == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'WARNING: docwrap: failed to discover git work tree\n'


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command_template',
    ['git -C {target} commit -m "test"', 'cd {target} && git commit -m "test"'],
)
def test_repository_discovery_uses_resolved_invocation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_template: str,
) -> None:
    target = tmp_path / 'target'
    target.mkdir()
    seen: list[str] = []

    def record_target(cwd: str) -> RepoDiscoveryResult:
        seen.append(cwd)
        return RepoDiscoveryResult(RepoDiscoveryStatus.NOT_WORKTREE)

    monkeypatch.setattr('docwrap._hook.discover_repo_root', record_target)
    command = command_template.format(target=shell_path(target))
    event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(tmp_path))

    assert _run_claude(json.dumps(event)) == 0
    assert seen == [str(target)]


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize('allow_on_failure', [False, True], ids=['deny', 'allow'])
def test_route_permission_error_always_denies_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow_on_failure: bool,
) -> None:
    target = tmp_path / 'blocked'
    real_stat = Path.stat

    def raise_for_target(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path == target:
            raise PermissionError('blocked')
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', raise_for_target)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': f'git -C {shell_path(target)} commit -m "x"'},
        cwd=str(tmp_path),
    )

    result = _run_claude(json.dumps(event), allow_on_failure=allow_on_failure)
    captured = capsys.readouterr()

    assert result == 0
    output = json.loads(captured.out)
    assert output['hookSpecificOutput']['permissionDecision'] == 'deny'
    reason = f'{_REASON_PREFIX}: unresolvable directory route; {_HINT_ROUTE}'
    assert output['hookSpecificOutput']['permissionDecisionReason'] == reason
    assert captured.err == f'{reason}\n'


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'commit_args',
    ['--dry-run', '--short', '--porcelain', '--long', '--null', '-z'],
    ids=['dry-run', 'short', 'porcelain', 'long', 'null', 'null-short'],
)
def test_status_only_commit_is_silent_without_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    commit_args: str,
) -> None:
    _fail_on_repository_work(monkeypatch)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': f'git commit {commit_args}'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    ('commit_args', 'element'),
    [
        ('--dry-run --no-dry-run', 'option --no-dry-run'),
        ('--future-option', 'option --future-option'),
        ('-m', 'missing value for -m'),
    ],
    ids=['dry-run-negated', 'unknown-commit-option', 'missing-message'],
)
def test_unknown_commit_option_or_malformed_commit_is_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    commit_args: str,
    element: str,
) -> None:
    _fail_on_repository_work(monkeypatch)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': f'git commit {commit_args}'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    reason = _assert_denied(capsys)
    assert reason == f'{_REASON_PREFIX}: {element}; {_HINT_COMMIT}'


@pytest.mark.usefixtures('registered_hosts')
def test_unsupported_predecessors_are_denied_before_repository_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail('repository-dependent operation was called')

    monkeypatch.setattr('docwrap._hook.is_hook_enabled', fail_if_called)
    monkeypatch.setattr('docwrap._hook.discover_repo_root', fail_if_called)
    event = hook_event(
        'PreToolUse',
        'Bash',
        {'command': 'git --future status ; git init child ; git commit -m "x"'},
        cwd=str(tmp_path),
    )

    assert _run_claude(json.dumps(event)) == 0
    _assert_denied(capsys)


@pytest.mark.usefixtures('registered_hosts')
@pytest.mark.parametrize(
    'command',
    ['git commit -am "test"', 'git commit foo.py -m "test"'],
)
@pytest.mark.parametrize('needs_wrap', [False, True])
def test_worktree_commit_success_is_silent(
    git_repo_with_file: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    command: str,
    needs_wrap: bool,
) -> None:
    repo, path = git_repo_with_file
    run_setup(['git', 'config', 'docwrap.hook', 'true'], cwd=repo)
    _write_permission_candidate(path, needs_wrap=needs_wrap)
    event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(repo))
    assert _run_claude(json.dumps(event)) == 0
    _assert_success(capsys)
    assert ('definitely\n' in path.read_text()) is needs_wrap


class TestUnknownInvocationDirectory:
    @pytest.mark.usefixtures('registered_hosts')
    @pytest.mark.parametrize('form', ['dash_c', 'cd'])
    def test_absolute_route_formats_only_the_target_regardless_of_process_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        form: str,
    ) -> None:
        session_repo = tmp_path / 'a'
        target_repo = tmp_path / 'b'
        elsewhere = tmp_path / 'c'
        elsewhere.mkdir()
        _stage_long_docstring(session_repo)
        _stage_long_docstring(target_repo)
        session_before = _index_listing(session_repo)
        command = (
            f'git -C {shell_path(target_repo)} commit -m "test"'
            if form == 'dash_c'
            else f'cd {shell_path(target_repo)} && git commit -m "test"'
        )
        monkeypatch.chdir(elsewhere)
        event = hook_event('PreToolUse', 'Bash', {'command': command}, cwd=str(session_repo))

        assert _run_codex(json.dumps(event)) == 0

        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')
        assert _index_listing(session_repo) == session_before

    @pytest.mark.usefixtures('registered_hosts')
    def test_absolute_route_replays_an_add_that_stages_new_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_repo = tmp_path / 'a'
        target_repo = tmp_path / 'b'
        _stage_long_docstring(session_repo)
        _init_hook_repo(target_repo, hook_enabled=True)
        _write_long_arg_docstring(target_repo / 'foo.py')
        staged_names = run_setup(
            ['git', 'diff', '--cached', '--name-only'],
            cwd=str(target_repo),
        ).stdout
        assert decode_text(staged_names) == ''
        session_before = _index_listing(session_repo)
        monkeypatch.chdir(session_repo)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {
                'command': (
                    f'git -C {shell_path(target_repo)} add foo.py && '
                    f'git -C {shell_path(target_repo)} commit -m "test"'
                )
            },
            cwd=str(session_repo),
        )

        assert _run_codex(json.dumps(event)) == 0

        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        assert 'definitely\n' in _staged_text(target_repo, 'foo.py')
        assert _index_listing(session_repo) == session_before

    @pytest.mark.usefixtures('registered_hosts')
    def test_claude_trusts_its_absolute_cwd_over_the_process_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tracked_repo = tmp_path / 'b'
        other_repo = tmp_path / 'a'
        _stage_long_docstring(tracked_repo)
        _stage_long_docstring(other_repo)
        other_before = _index_listing(other_repo)
        monkeypatch.chdir(other_repo)
        event = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(tracked_repo)
        )

        assert _run_claude(json.dumps(event)) == 0

        assert json.loads(capsys.readouterr().out) == _RECEIPT_ONE
        assert 'definitely\n' in _staged_text(tracked_repo, 'foo.py')
        assert _index_listing(other_repo) == other_before

    @pytest.mark.usefixtures('registered_hosts')
    def test_codex_bare_commit_is_denied_and_touches_neither_repository(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_repo = tmp_path / 'a'
        target_repo = tmp_path / 'b'
        _stage_long_docstring(session_repo)
        _stage_long_docstring(target_repo)
        before = (_index_listing(session_repo), _index_listing(target_repo))
        before_files = (
            (session_repo / 'foo.py').read_bytes(),
            (target_repo / 'foo.py').read_bytes(),
        )
        monkeypatch.chdir(target_repo)
        event = hook_event(
            'PreToolUse', 'Bash', {'command': 'git commit -m "test"'}, cwd=str(session_repo)
        )

        assert _run_codex(json.dumps(event)) == 0

        captured = capsys.readouterr()
        assert json.loads(captured.out) == {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': _UNKNOWN_CWD_REASON,
            }
        }
        assert (_index_listing(session_repo), _index_listing(target_repo)) == before
        assert (
            (session_repo / 'foo.py').read_bytes(),
            (target_repo / 'foo.py').read_bytes(),
        ) == before_files

    @pytest.mark.usefixtures('registered_hosts')
    def test_denial_precedes_hook_disable_and_allow_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_repo = tmp_path / 'a'
        target_repo = tmp_path / 'b'
        _stage_long_docstring(session_repo)
        _stage_long_docstring(target_repo)
        for repo in (session_repo, target_repo):
            run_setup(['git', 'config', 'docwrap.hook', 'false'], cwd=str(repo))
        before = (_index_listing(session_repo), _index_listing(target_repo))
        monkeypatch.chdir(target_repo)
        event = hook_event(
            'PreToolUse',
            'Bash',
            {'command': 'git add foo.py && git commit -m "test"'},
            cwd=str(session_repo),
        )

        assert _run_codex(json.dumps(event), allow_on_failure=True) == 0

        decision = json.loads(capsys.readouterr().out)['hookSpecificOutput']['permissionDecision']
        assert decision == 'deny'
        assert (_index_listing(session_repo), _index_listing(target_repo)) == before

    @pytest.mark.usefixtures('registered_hosts')
    @pytest.mark.parametrize(
        'command',
        [
            'git -C {repo} commit -a -m "test"',
            'git -C {repo} commit -m "test" -- foo.py',
            'git -C {repo} commit -i foo.py -m "test"',
        ],
    )
    def test_routed_modes_commit_wrapped_blobs_with_unknown_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        command: str,
    ) -> None:
        session_repo = tmp_path / 'a'
        target_repo = tmp_path / 'b'
        _stage_long_docstring(session_repo)
        _init_hook_repo(target_repo, hook_enabled=True)
        (target_repo / 'foo.py').write_text(
            'def foo(value):\n    """Return the value."""\n    return value\n'
        )
        git_commit_all(target_repo, 'seed')
        run_setup(
            ['git', 'config', '--local', 'user.useConfigOnly', 'true'],
            cwd=target_repo,
        )
        for key, value in [('user.name', 'Test'), ('user.email', 'test@test.com')]:
            run_setup(
                ['git', 'config', '--local', key, value],
                cwd=target_repo,
            )
        _write_long_arg_docstring(target_repo / 'foo.py')
        rendered = command.format(repo=shell_path(target_repo))
        session_before = _index_listing(session_repo)
        monkeypatch.chdir(session_repo)
        event = hook_event('PreToolUse', 'Bash', {'command': rendered}, cwd=str(session_repo))

        assert _run_codex(json.dumps(event)) == 0
        assert capsys.readouterr().out == ''
        run_setup(shlex.split(rendered), cwd=str(session_repo))

        committed = run_setup(
            ['git', 'show', 'HEAD:foo.py'],
            cwd=str(target_repo),
        ).stdout
        assert 'definitely\n' in decode_text(committed)
        assert _index_listing(session_repo) == session_before

    def test_unknown_directory_denial_performs_no_repository_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError('repository access is forbidden before a target exists')

        for name in (
            'discover_repo_root',
            'is_hook_enabled',
            'run_pre_commit_adds',
            'find_config',
        ):
            monkeypatch.setattr(hook_mod, name, explode)

        outcome = front_mod._run_pre_tool_use(
            'git commit -m x', None, allow_on_failure=True, incremental=None
        )

        assert outcome == Outcome('deny', _UNKNOWN_CWD_REASON)


def test_worktree_sync_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'staged.py'
    raw = b'original\n'
    path.write_bytes(raw)

    def apply_entries(_repo_root: str, _entries: list[IndexEntry]) -> bool:
        return True

    monkeypatch.setattr(hook_mod, 'apply_index_entries', apply_entries)

    def fail_sync(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(hook_mod, 'write_preserving_newline', fail_sync)
    result = hook_mod._apply_commit_results(
        str(tmp_path),
        [IndexEntry('100644', 'hash', 'staged.py')],
        [WorktreeWrite(path, 'replacement\n', 'utf-8', raw)],
        allow_on_failure=False,
    )

    assert result == Outcome('allow', 'docwrap: wrapped 1 staged Python file for this commit')
    assert (
        capsys.readouterr().err
        == f'WARNING: docwrap: failed to synchronize {path.as_posix()}: disk full\n'
    )
