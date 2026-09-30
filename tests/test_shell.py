import sys
from pathlib import Path

import pytest

from docwrap._shell import normalize_msys_path, resolve_route


class TestNormalizeMsysPath:
    def test_drive_lowercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/c/Users/user') == 'C:/Users/user'

    def test_drive_root_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/c') == 'C:/'

    def test_drive_uppercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/D/repo') == 'D:/repo'

    def test_drive_with_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/c/') == 'C:/'

    def test_noop_on_unix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'darwin')
        assert normalize_msys_path('/c/Users/user') == '/c/Users/user'

    def test_passthrough_relative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('src/foo.py') == 'src/foo.py'

    def test_passthrough_unix_absolute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/home/user') == '/home/user'

    def test_passthrough_windows_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('C:/Users/user') == 'C:/Users/user'

    def test_single_letter_treated_as_drive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'win32')
        assert normalize_msys_path('/h/user') == 'H:/user'


class TestResolveRoute:
    def test_absolute_route(self, tmp_path: Path) -> None:
        assert resolve_route(str(tmp_path), '/elsewhere', logical=False) == str(tmp_path.resolve())

    def test_file_route_is_none(self, tmp_path: Path) -> None:
        (tmp_path / 'file').write_text('x\n', encoding='utf-8')
        assert resolve_route('./file', str(tmp_path), logical=True) is None

    def test_logical_resolution_keeps_symlinked_parents(self, tmp_path: Path) -> None:
        target = tmp_path / 'target'
        target.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(target, target_is_directory=True)
        assert resolve_route('./link', str(tmp_path), logical=True) == str(link)
        assert resolve_route('./link', str(tmp_path), logical=False) == str(target.resolve())

    def test_missing_route_is_none(self, tmp_path: Path) -> None:
        assert resolve_route('./missing', str(tmp_path), logical=True) is None

    def test_relative_route_resolves_against_the_payload_directory(self, tmp_path: Path) -> None:
        (tmp_path / 'repo').mkdir()
        assert resolve_route('./repo', str(tmp_path), logical=True) == str(tmp_path / 'repo')


def test_absolute_route_resolves_without_a_base(tmp_path: Path) -> None:
    assert resolve_route(str(tmp_path), None, logical=True) == str(tmp_path)


def test_relative_route_without_a_base_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match='relative route requires a known base'):
        resolve_route('./repo', None, logical=False)
