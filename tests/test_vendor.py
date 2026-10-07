import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / 'src' / 'docwrap' / '_vendor' / 'hookshell'
_MANIFEST = _VENDOR / 'manifest.json'
_VENDORED_SUFFIXES = frozenset({'.json', '.py'})


def _vendored_paths() -> set[str]:
    return {
        path.relative_to(_ROOT).as_posix()
        for root in (_VENDOR, _ROOT / 'tests' / 'hookshell')
        for path in root.rglob('*')
        if path.is_file()
        and path.suffix in _VENDORED_SUFFIXES
        and '__pycache__' not in path.parts
        and path != _MANIFEST
    }


def test_vendored_hookshell_matches_its_manifest() -> None:
    files: dict[str, str] = json.loads(_MANIFEST.read_text(encoding='utf-8'))['files']
    assert set(files) == _vendored_paths()
    for path, digest in files.items():
        assert hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() == digest, path
