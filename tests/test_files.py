import errno
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import docwrap._files as files_mod
from docwrap._files import (
    _SKIP_DIRS,
    ReadResult,
    collect_files,
    decode_python_source,
    detect_newline,
    encode_with_newline,
    read_file,
    read_file_result,
    write_preserving_newline,
)
from tests.helpers import run_setup


class TestCollectFiles:
    def test_collects_py_files(self, tmp_path: Path) -> None:
        (tmp_path / 'a.py').write_text('pass\n')
        (tmp_path / 'b.py').write_text('pass\n')
        (tmp_path / 'c.txt').write_text('not python\n')
        files = collect_files([str(tmp_path)])
        names = [f.name for f in files]
        assert 'a.py' in names
        assert 'b.py' in names
        assert 'c.txt' not in names

    def test_deduplicates(self, tmp_path: Path) -> None:
        f = tmp_path / 'dup.py'
        f.write_text('pass\n')
        files = collect_files([str(f), str(f)])
        assert len(files) == 1

    def test_directory_excludes_git_ignored_nested_worktree(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        nested = tmp_path / 'nested'
        nested.mkdir()
        run_setup(['git', 'init', '-q'], cwd=nested)
        inside = nested / 'inside.py'
        inside.write_text('x = 1\n', encoding='utf-8')
        (tmp_path / '.gitignore').write_text('nested/\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == []

    def test_directory_excludes_git_ignored_python(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        kept = tmp_path / 'kept.py'
        kept.write_text('x = 1\n', encoding='utf-8')
        ignored = tmp_path / 'ignored.py'
        ignored.write_text('x = 2\n', encoding='utf-8')
        (tmp_path / '.gitignore').write_text('ignored.py\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == [kept]

    def test_directory_keeps_tracked_ignored_python(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        source = tmp_path / 'tracked.py'
        source.write_text('x = 1\n', encoding='utf-8')
        run_setup(['git', 'add', 'tracked.py'], cwd=tmp_path)
        (tmp_path / '.gitignore').write_text('tracked.py\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == [source]

    def test_directory_keeps_untracked_nonignored_python(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        source = tmp_path / 'untracked.py'
        source.write_text('x = 1\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == [source]

    def test_directory_recurses_into_nested_git_worktree(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        top = tmp_path / 'top.py'
        top.write_text('x = 1\n', encoding='utf-8')
        nested = tmp_path / 'nested'
        nested.mkdir()
        run_setup(['git', 'init', '-q'], cwd=nested)
        inside = nested / 'inside.py'
        inside.write_text('x = 2\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == [inside, top]

    def test_explicit_file_path(self, tmp_path: Path) -> None:
        f = tmp_path / 'single.py'
        f.write_text('pass\n')
        files = collect_files([str(f)])
        assert len(files) == 1
        assert files[0].name == 'single.py'

    def test_explicit_git_ignored_python_is_included(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        ignored = tmp_path / 'ignored.py'
        ignored.write_text('x = 1\n', encoding='utf-8')
        (tmp_path / '.gitignore').write_text('ignored.py\n', encoding='utf-8')

        assert collect_files([str(ignored)]) == [ignored]

    @pytest.mark.parametrize('failure', ['nonzero', 'oserror'])
    def test_git_failure_falls_back_to_filesystem(
        self,
        failure: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / 'source.py'
        source.write_text('x = 1\n', encoding='utf-8')

        def fail_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            if failure == 'oserror':
                raise OSError
            return subprocess.CompletedProcess([], 128, b'source.py\0', b'failure')

        monkeypatch.setattr(subprocess, 'run', fail_git)

        assert collect_files([str(tmp_path)]) == [source]
        assert capsys.readouterr().err == ''

    @pytest.mark.parametrize(
        ('git_stderr', 'detail'),
        [
            (
                b"fatal: detected dubious ownership in repository at '/x'\nhint\n",
                "fatal: detected dubious ownership in repository at '/x'",
            ),
            (b'', 'git exited with status 128'),
        ],
    )
    def test_git_failure_inside_worktree_warns(
        self,
        git_stderr: bytes,
        detail: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / '.git').mkdir()
        source = tmp_path / 'source.py'
        source.write_text('x = 1\n', encoding='utf-8')

        def fail_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess([], 128, b'', git_stderr)

        monkeypatch.setattr(subprocess, 'run', fail_git)

        assert collect_files([str(tmp_path)]) == [source]
        assert capsys.readouterr().err == (
            f'warning: cannot apply Git exclusions in {tmp_path}: {detail}; '
            'keeping every discovered Python file\n'
        )

    def test_git_filter_uses_absolute_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / 'space name.py'
        source.write_text('pass\n')

        def capture_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            assert Path(args[0]).is_absolute()
            assert args[1:] == ['check-ignore', '-z', '--stdin']
            assert kwargs['cwd'] == tmp_path
            assert kwargs['input'] == b'space name.py\0'
            return subprocess.CompletedProcess(args, 0, b'space name.py\0', b'')

        monkeypatch.setattr(subprocess, 'run', capture_run)
        assert collect_files([str(tmp_path)]) == []

    def test_git_value_error_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'source.py').write_text('x = 1\n')

        def invalid_run(*_args: object, **_kwargs: object) -> None:
            raise ValueError('invalid subprocess argument')

        monkeypatch.setattr(subprocess, 'run', invalid_run)

        with pytest.raises(ValueError, match='invalid subprocess argument'):
            collect_files([str(tmp_path)])

    def test_missing_git_falls_back_to_filesystem(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / 'source.py'
        source.write_text('pass\n')
        monkeypatch.setenv('PATH', str(tmp_path))

        def unexpected_run(*_args: object, **_kwargs: object) -> None:
            pytest.fail('No executable should be launched when Git cannot be resolved')

        monkeypatch.setattr(subprocess, 'run', unexpected_run)
        assert collect_files([str(tmp_path)]) == [source]
        assert capsys.readouterr().err == ''

    def test_missing_git_inside_worktree_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / '.git').mkdir()
        source = tmp_path / 'source.py'
        source.write_text('pass\n')
        monkeypatch.setenv('PATH', str(tmp_path))

        assert collect_files([str(tmp_path)]) == [source]
        assert capsys.readouterr().err == (
            f'warning: cannot apply Git exclusions in {tmp_path}: git could not be run; '
            'keeping every discovered Python file\n'
        )

    def test_nested_git_failures_warn_once_per_root_in_first_seen_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / '.git').mkdir()
        outer_file = tmp_path / 'outer.py'
        outer_file.write_text('x = 1\n', encoding='utf-8')
        nested_roots = [tmp_path / 'nested_a', tmp_path / 'nested_b']
        nested_files: list[Path] = []
        for root in nested_roots:
            root.mkdir()
            (root / '.git').mkdir()
            source = root / 'inside.py'
            source.write_text('x = 2\n', encoding='utf-8')
            nested_files.append(source)

        calls: list[Path] = []

        def fail_git(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            root = kwargs['cwd']
            assert isinstance(root, Path)
            calls.append(root)
            detail = f'{root.name} failure {calls.count(root)}'.encode()
            return subprocess.CompletedProcess(args, 128, b'', detail)

        monkeypatch.setattr(subprocess, 'run', fail_git)

        assert collect_files([str(tmp_path)]) == sorted([outer_file, *nested_files])
        assert calls == [tmp_path, tmp_path, tmp_path, *nested_roots]
        assert capsys.readouterr().err.splitlines() == [
            f'warning: cannot apply Git exclusions in {root}: {root.name} failure 1; '
            'keeping every discovered Python file'
            for root in [tmp_path, *nested_roots]
        ]

    def test_nested_git_worktree_uses_own_tracked_state(self, tmp_path: Path) -> None:
        run_setup(['git', 'init', '-q'], cwd=tmp_path)
        nested = tmp_path / 'nested'
        nested.mkdir()
        run_setup(['git', 'init', '-q'], cwd=nested)
        tracked = nested / 'tracked.py'
        tracked.write_text('x = 1\n', encoding='utf-8')
        run_setup(['git', 'add', 'tracked.py'], cwd=nested)
        ignored = nested / 'ignored.py'
        ignored.write_text('x = 2\n', encoding='utf-8')
        (nested / '.gitignore').write_text('*.py\n', encoding='utf-8')

        assert collect_files([str(tmp_path)]) == [tracked]

    def test_prunes_skipped_directories_before_descent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        def controlled_walk(
            root: Path,
            *,
            topdown: bool,
            followlinks: bool,
        ) -> Iterator[tuple[str, list[str], list[str]]]:
            assert root == tmp_path
            assert topdown is True
            assert followlinks is False
            dirnames = ['.git', 'retained']
            yield str(tmp_path), dirnames, []
            assert dirnames == ['retained']
            yield str(tmp_path / 'retained'), [], ['kept.py']

        monkeypatch.setattr(files_mod.os, 'walk', controlled_walk)

        assert collect_files([str(tmp_path)]) == [tmp_path / 'retained' / 'kept.py']

    def test_skips_pycache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / '__pycache__'
        cache_dir.mkdir()
        (cache_dir / 'cached.py').write_text('pass\n')
        (tmp_path / 'real.py').write_text('pass\n')
        files = collect_files([str(tmp_path)])
        names = [f.name for f in files]
        assert 'real.py' in names
        assert 'cached.py' not in names

    def test_skips_symlinked_py_files_during_directory_recursion(self, tmp_path: Path) -> None:
        target_dir = tmp_path / 'external'
        target_dir.mkdir()
        target = target_dir / 'external.py'
        target.write_text('def ext():\n    """External."""\n')

        project = tmp_path / 'project'
        project.mkdir()
        real = project / 'real.py'
        real.write_text('def foo():\n    """Short."""\n')
        link = project / 'linked.py'
        link.symlink_to(target)

        files = collect_files([str(project)])
        names = [f.name for f in files]
        assert 'real.py' in names
        assert 'linked.py' not in names


class TestDecodePythonSource:
    @pytest.mark.parametrize(
        ('raw', 'expected_source', 'expected_encoding'),
        [
            pytest.param(
                b'x = 1\r\ny = 2\r\n',
                'x = 1\ny = 2\n',
                'utf-8',
                id='crlf',
            ),
            pytest.param(
                b'x = 1\ry = 2\r',
                'x = 1\ny = 2\n',
                'utf-8',
                id='lone-cr',
            ),
            pytest.param(
                b'\xef\xbb\xbfx = 1\n',
                'x = 1\n',
                'utf-8-sig',
                id='utf8-bom',
            ),
            pytest.param(
                b'# coding: latin-1\r\nname = "caf\xe9"\r\n',
                '# coding: latin-1\nname = "caf\xe9"\n',
                'iso-8859-1',
                id='latin1-cookie',
            ),
        ],
    )
    def test_decodes_with_universal_newlines(
        self,
        raw: bytes,
        expected_source: str,
        expected_encoding: str,
    ) -> None:
        result = decode_python_source(raw)

        assert result == ReadResult(
            raw=raw,
            source=expected_source,
            encoding=expected_encoding,
        )

    def test_invalid_cookie_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            decode_python_source(b'# coding: not-a-codec\nx = 1\n')

    def test_invalid_declared_bytes_raise_unicode_error(self) -> None:
        with pytest.raises(UnicodeDecodeError):
            decode_python_source(b'# coding: ascii\nvalue = "\xff"\n')


class TestDetectNewline:
    def test_crlf(self) -> None:
        assert detect_newline(b'line1\r\nline2\r\n') == '\r\n'

    def test_empty(self) -> None:
        assert detect_newline(b'') == '\n'

    def test_lf(self) -> None:
        assert detect_newline(b'line1\nline2\n') == '\n'

    def test_mixed_returns_crlf(self) -> None:
        assert detect_newline(b'line1\r\nline2\n') == '\r\n'


class TestEncodeWithNewline:
    def test_crlf_conversion(self) -> None:
        result = encode_with_newline('line1\nline2\n', 'utf-8', '\r\n')
        assert result == b'line1\r\nline2\r\n'

    def test_encoding_respected(self) -> None:
        result = encode_with_newline('cafe\n', 'ascii', '\n')
        assert result == b'cafe\n'

    def test_lf_passthrough(self) -> None:
        result = encode_with_newline('line1\nline2\n', 'utf-8', '\n')
        assert result == b'line1\nline2\n'

    def test_mixed_input_normalized(self) -> None:
        result = encode_with_newline('line1\r\nline2\n', 'utf-8', '\r\n')
        assert result == b'line1\r\nline2\r\n'
        assert b'\r\r\n' not in result


class TestFileIdentityDedup:
    def test_case_alias_dedups_to_first(self, tmp_path: Path) -> None:
        target = tmp_path / 'CaseFile.py'
        target.write_text('x = 1\n')
        alias = tmp_path / 'casefile.py'
        if not alias.exists():
            pytest.skip('requires a case-insensitive filesystem')
        result = collect_files([str(target), str(alias)])
        assert result == [target]

    def test_distinct_files_kept_in_order(self, tmp_path: Path) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        a.write_text('x = 1\n')
        b.write_text('x = 2\n')
        assert collect_files([str(b), str(a)]) == [b, a]

    def test_hardlink_alias_dedups(self, tmp_path: Path) -> None:
        a = tmp_path / 'a.py'
        a.write_text('x = 1\n')
        b = tmp_path / 'b.py'
        os.link(a, b)
        assert collect_files([str(a), str(b)]) == [a]

    def test_missing_paths_dedup_lexically(self, tmp_path: Path) -> None:
        missing = tmp_path / 'missing.py'
        other = tmp_path / 'other.py'
        result = collect_files([str(missing), str(missing), str(other)])
        assert result == [missing, other]

    def test_recursive_discovery_with_explicit_alias(self, tmp_path: Path) -> None:
        target = tmp_path / 'CaseFile.py'
        target.write_text('x = 1\n')
        alias = tmp_path / 'casefile.py'
        if not alias.exists():
            pytest.skip('requires a case-insensitive filesystem')
        result = collect_files([str(tmp_path), str(alias)])
        assert [p.name for p in result] == ['CaseFile.py']

    def test_symlink_alias_dedups(self, tmp_path: Path) -> None:
        a = tmp_path / 'a.py'
        a.write_text('x = 1\n')
        link = tmp_path / 'link.py'
        link.symlink_to(a)
        assert collect_files([str(a), str(link)]) == [a]

    def test_zero_inode_keeps_distinct_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        a.write_text('x = 1\n')
        b.write_text('x = 2\n')
        real_stat = Path.stat

        def zero_inode_stat(
            path: Path,
            *,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            stat = real_stat(path, follow_symlinks=follow_symlinks)
            values = list(stat)
            values[1] = 0
            return os.stat_result(values)

        monkeypatch.setattr(Path, 'stat', zero_inode_stat)

        assert collect_files([str(a), str(b)]) == [a, b]


class TestReadFile:
    def test_plain_utf8_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'plain.py'
        f.write_text('y = 2\n', encoding='utf-8')
        source, encoding = read_file(f)
        assert source == 'y = 2\n'
        assert encoding == 'utf-8'

    def test_returns_source_and_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / 'hello.py'
        f.write_text('# -*- coding: utf-8 -*-\nx = 1\n', encoding='utf-8')
        source, encoding = read_file(f)
        assert 'x = 1' in source
        assert encoding == 'utf-8'


class TestReadFileResult:
    def test_read_file_delegates_to_snapshot_reader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = Path('source.py')
        result = ReadResult(
            raw=b'# coding: latin-1\nvalue = "\xe9"\n',
            source='# coding: latin-1\nvalue = "\xe9"\n',
            encoding='iso-8859-1',
        )
        calls: list[Path] = []

        def tracked_reader(candidate: Path) -> ReadResult:
            calls.append(candidate)
            return result

        monkeypatch.setattr(files_mod, 'read_file_result', tracked_reader)

        assert read_file(path) == (result.source, result.encoding)
        assert calls == [path]

    def test_reads_once_and_retains_raw_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        raw = b'x = 1\r\n'
        path.write_bytes(raw)
        original_read_bytes = Path.read_bytes
        reads = 0

        def tracked_read_bytes(candidate: Path) -> bytes:
            nonlocal reads
            if candidate == path:
                reads += 1
            return original_read_bytes(candidate)

        monkeypatch.setattr(Path, 'read_bytes', tracked_read_bytes)

        result = read_file_result(path)

        assert reads == 1
        assert result.raw == raw
        assert result.source == 'x = 1\n'
        assert result.encoding == 'utf-8'


class TestSkipDirs:
    def test_contains_expected_entries(self) -> None:
        critical = {'.git', '__pycache__', '.venv', 'build', 'dist'}
        assert critical <= _SKIP_DIRS


class TestWritePreservingNewline:
    def test_empty_original_bytes_use_lf(self, tmp_path: Path) -> None:
        path = tmp_path / 'empty.py'
        path.write_bytes(b'')

        write_preserving_newline(path, 'new\nline\n', 'utf-8', raw=b'')

        assert path.read_bytes() == b'new\nline\n'

    def test_failed_restore_reports_both_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'partial.py'
        raw = b'original\n'
        path.write_bytes(raw)
        original_write_bytes = Path.write_bytes
        write_error = OSError(errno.ENOSPC, 'No space left on device')
        calls = 0

        def failed_writes(candidate: Path, data: bytes) -> int:
            nonlocal calls
            if candidate == path:
                calls += 1
                if calls == 1:
                    original_write_bytes(candidate, data[: len(data) // 2])
                    raise write_error
                raise OSError('restore failed')
            return original_write_bytes(candidate, data)

        monkeypatch.setattr(Path, 'write_bytes', failed_writes)

        with pytest.raises(files_mod.RestoreError) as exc_info:
            write_preserving_newline(path, 'replacement\n', 'utf-8', raw=raw)

        assert exc_info.value.errno == errno.ENOSPC
        assert exc_info.value.__cause__ is write_error
        assert 'No space left on device' in str(exc_info.value)
        assert 'restore failed' in str(exc_info.value)
        assert calls == 2

    def test_failed_write_restores_original_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / 'partial.py'
        raw = b'original\n'
        path.write_bytes(raw)
        original_write_bytes = Path.write_bytes
        calls = 0

        def partial_write(candidate: Path, data: bytes) -> int:
            nonlocal calls
            if candidate == path:
                calls += 1
                if calls == 1:
                    original_write_bytes(candidate, data[: len(data) // 2])
                    raise OSError(errno.ENOSPC, 'No space left on device')
            return original_write_bytes(candidate, data)

        monkeypatch.setattr(Path, 'write_bytes', partial_write)

        with pytest.raises(OSError, match='No space left on device') as exc_info:
            write_preserving_newline(path, 'replacement\n', 'utf-8', raw=raw)

        assert exc_info.value.errno == errno.ENOSPC
        assert calls == 2
        assert path.read_bytes() == raw

    def test_original_bytes_select_any_crlf_policy(self, tmp_path: Path) -> None:
        path = tmp_path / 'mixed.py'
        path.write_bytes(b'old\r\nline\n')

        write_preserving_newline(path, 'new\nline\n', 'utf-8', raw=b'old\r\nline\n')

        assert path.read_bytes() == b'new\r\nline\r\n'

    def test_original_lone_cr_selects_lf_policy(self, tmp_path: Path) -> None:
        path = tmp_path / 'lone-cr.py'
        path.write_bytes(b'old\rline\r')

        write_preserving_newline(path, 'new\nline\n', 'utf-8', raw=b'old\rline\r')

        assert path.read_bytes() == b'new\nline\n'
