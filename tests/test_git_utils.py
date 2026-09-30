import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import docwrap._git_utils as git_utils
from docwrap._git_utils import decode_git_quoted_path, git_executable
from tests.helpers import CHILD_TIMEOUT_SECONDS


class TestDecodeGitQuotedPath:
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            (b'""', b''),
            (b'"raw-\xff.py"', b'raw-\xff.py'),
            (b'"\\000\\377"', b'\0\xff'),
            (b'"\\a\\b\\f\\n\\r\\t\\v\\\\\\""', b'\a\b\f\n\r\t\v\\"'),
            (b'plain.py', None),
            (b'"\\400"', None),
            (b'"\\378"', None),
            (b'"\\12"', None),
            (b'"\\q"', None),
            (b'"trailing\\', None),
            (b'"missing', None),
            (b'"closed"suffix', None),
        ],
    )
    def test_decodes_complete_byte_tokens(self, raw: bytes, expected: bytes | None) -> None:
        assert decode_git_quoted_path(raw) == expected


class TestGitExecutable:
    def test_ignores_cwd_and_relative_entries_and_preserves_path_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = tmp_path / 'repository'
        repository.mkdir()
        (repository / 'git.exe').write_bytes(b'decoy; never executed')
        missing = tmp_path / 'missing'
        wrapper_dir = tmp_path / 'trusted wrapper directory'
        other_dir = tmp_path / 'other trusted directory'
        monkeypatch.chdir(repository)
        monkeypatch.setenv(
            'PATH',
            os.pathsep.join(
                ['', '.', 'relative/bin', str(missing), str(wrapper_dir), str(other_dir)]
            ),
        )
        calls: list[str] = []

        def find_candidate(candidate: str) -> str | None:
            calls.append(candidate)
            assert Path(candidate).is_absolute()
            if Path(candidate).parent == missing:
                return None
            # which owns executable extension handling; retain its returned wrapper spelling.
            return candidate + '.CMD'

        monkeypatch.setattr(git_utils.shutil, 'which', find_candidate)
        assert git_executable() == str(wrapper_dir / 'git') + '.CMD'
        assert calls == [str(missing / 'git'), str(wrapper_dir / 'git')]
        monkeypatch.setenv('PATH', str(other_dir))
        assert git_executable() == str(other_dir / 'git') + '.CMD'

    @pytest.mark.parametrize('path', ['', '.', 'relative/bin'])
    def test_no_absolute_entries_returns_none(
        self, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PATH', path)
        assert git_executable() is None

    def test_no_git_in_absolute_entry_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PATH', str(tmp_path))
        assert git_executable() is None

    @pytest.mark.skipif(sys.platform == 'win32', reason='POSIX executable wrapper permissions')
    def test_preserves_absolute_path_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper = tmp_path / 'wrapper directory' / 'git'
        wrapper.parent.mkdir()
        wrapper.write_text('#!/bin/sh\nexit 0\n')
        wrapper.chmod(0o755)
        monkeypatch.setenv('PATH', str(wrapper.parent))
        assert git_executable() == str(wrapper)

    def test_unset_path_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PATH', raising=False)
        assert git_executable() is None

    @pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows executable search')
    @pytest.mark.parametrize('explicit_executable', [False, True], ids=['argv', 'executable'])
    def test_windows_ignores_cwd_decoy_when_launching_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, explicit_executable: bool
    ) -> None:
        real_git = git_executable()
        assert real_git is not None, 'Git must be installed in an absolute PATH directory'
        # Only an OS executable is copied as the decoy, never a repository-supplied binary.
        os_executable = Path(os.environ['SYSTEMROOT']) / 'System32' / 'where.exe'
        shutil.copyfile(os_executable, tmp_path / 'git.exe')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('PATH', str(Path(real_git).parent))
        executable = git_executable()
        assert executable is not None
        assert executable == real_git
        result = subprocess.run(
            ['git' if explicit_executable else executable, '--version'],
            executable=executable if explicit_executable else None,
            timeout=CHILD_TIMEOUT_SECONDS,
            capture_output=True,
            check=True,
        )
        assert result.stdout.startswith(b'git version ')

    @pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows PATHEXT resolution')
    def test_windows_preserves_pathext_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for suffix in ('.CMD', '.EXE'):
            (tmp_path / f'git{suffix}').write_bytes(b'not executed')
        monkeypatch.setenv('PATH', str(tmp_path))
        monkeypatch.setenv('PATHEXT', '.CMD;.EXE')
        assert git_executable() == str(tmp_path / 'git.CMD')
