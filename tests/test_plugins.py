import configparser
import json
import os
import shutil
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import cast

import pytest

from _build import PluginMetadataHook
from tests.helpers import BUILTIN_ENTRY_POINTS, process_transcript, run_cli, run_setup

_ROOT = Path(__file__).resolve().parents[1]
_MARKETPLACES = (
    _ROOT / '.agents' / 'plugins' / 'marketplace.json',
    _ROOT / '.claude-plugin' / 'marketplace.json',
)
_PLUGINS = _ROOT / 'plugins'


def _build(project: Path, target: str) -> Path:
    run_setup(
        [sys.executable, '-m', 'hatchling', 'build', '-t', target],
        cwd=project,
    )
    suffix = '*.whl' if target == 'wheel' else '*.tar.gz'
    return next((project / 'dist').glob(suffix))


def _copy_project(destination: Path) -> None:
    destination.mkdir()
    for name in ('pyproject.toml', 'README.md', 'LICENSE', '_build.py'):
        source = _ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for name in ('src', 'plugins'):
        source = _ROOT / name
        if source.is_dir():
            shutil.copytree(
                source,
                destination / name,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
            )


def _declared_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for manifest in sorted(_PLUGINS.rglob('plugin.json')):
        data = cast('dict[str, object]', json.loads(manifest.read_text(encoding='utf-8')))
        if 'version' in data:
            versions[manifest.relative_to(_ROOT).as_posix()] = cast('str', data['version'])
    for marketplace in _MARKETPLACES:
        data = cast('dict[str, object]', json.loads(marketplace.read_text(encoding='utf-8')))
        entries = cast('list[dict[str, object]]', data['plugins'])
        for index, entry in enumerate(entries):
            if 'version' in entry:
                key = f'{marketplace.relative_to(_ROOT).as_posix()}:plugins[{index}]'
                versions[key] = cast('str', entry['version'])
    return versions


def _entry_points(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith('/entry_points.txt'))
        config = configparser.ConfigParser()
        config.read_string(archive.read(metadata).decode())
    return dict(config['docwrap.hook_adapters'])


def test_declared_versions_match_package_version() -> None:
    with (_ROOT / 'pyproject.toml').open('rb') as stream:
        package_version = cast('str', tomllib.load(stream)['project']['version'])
    versions = _declared_versions()
    assert versions
    assert versions == dict.fromkeys(versions, package_version)


def test_duplicate_plugin_registration_fails_the_metadata_build(tmp_path: Path) -> None:
    for directory in ('first', 'second'):
        plugin = tmp_path / 'plugins' / directory
        plugin.mkdir(parents=True)
        (plugin / 'adapter.toml').write_text(
            '[entry-points."docwrap.hook_adapters"]\nexample = "example:Adapter"\n',
            encoding='utf-8',
        )
    hook = PluginMetadataHook(str(tmp_path), {})
    with pytest.raises(ValueError, match=r'Duplicate entry point docwrap\.hook_adapters:example'):
        hook.update({})


def test_installed_wheel_reports_its_version(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    _copy_project(project)
    expected = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8'))['project'][
        'version'
    ]
    wheel = _build(project, 'wheel')
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            archive.extract(name, installed)

    result = run_cli(
        ['--version'],
        expect=0,
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(installed)},
    )
    assert result.stdout == f'docwrap {expected}\n', process_transcript(result)


def test_plugin_directory_addition_survives_wheel_and_sdist(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    _copy_project(project)
    plugin = project / 'plugins' / 'example'
    package = plugin / '_docwrap'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / '_adapter.py').write_text('ADAPTER = "example"\n', encoding='utf-8')
    (plugin / 'adapter.toml').write_text(
        '[entry-points."docwrap.hook_adapters"]\n'
        'example = "plugins.example._docwrap._adapter:ADAPTER"\n',
        encoding='utf-8',
    )
    native_manifest = plugin / '.claude-plugin' / 'plugin.json'
    native_manifest.parent.mkdir()
    native_manifest.write_text('{}', encoding='utf-8')
    tests = plugin / 'tests'
    tests.mkdir()
    (tests / 'test_adapter.py').write_text('def test_example():\n    pass\n', encoding='utf-8')
    wheel = _build(project, 'wheel')
    expected = {**BUILTIN_ENTRY_POINTS, 'example': 'plugins.example._docwrap._adapter:ADAPTER'}
    assert _entry_points(wheel) == expected
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'plugins/example/_docwrap/_adapter.py' in names
        assert all(
            target.partition(':')[0].replace('.', '/') + '.py' in names
            for target in expected.values()
        )
        assert 'plugins/example/tests/test_adapter.py' not in names
        assert 'plugins/example/adapter.toml' not in names
        assert 'plugins/example/.claude-plugin/plugin.json' not in names
        installed = tmp_path / 'installed'
        for name in names:
            archive.extract(name, installed)
    run_setup(
        [
            sys.executable,
            '-c',
            (
                'from importlib.metadata import distribution; '
                'entries = distribution("docwrap").entry_points; '
                'loaded = {ep.name: ep.load() for ep in entries '
                'if ep.group == "docwrap.hook_adapters"}; '
                'assert loaded["example"] == "example"; '
                'assert set(loaded) == set(' + repr(sorted(expected)) + ')'
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(installed)},
    )
    sdist = _build(project, 'sdist')
    source = tmp_path / 'source'
    with tarfile.open(sdist) as archive:
        archive.extractall(source, filter='data')
    unpacked = next(source.iterdir())
    assert (unpacked / 'plugins/example/adapter.toml').is_file()
    assert (unpacked / 'plugins/example/.claude-plugin/plugin.json').is_file()
    rebuilt = _build(unpacked, 'wheel')
    assert _entry_points(rebuilt) == _entry_points(wheel)
