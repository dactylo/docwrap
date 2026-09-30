"""Provide repository-wide pytest fixtures for core and bundled plugin tests."""

import os
import sys
from pathlib import Path

import pytest

from tests.helpers import (
    BUILTIN_ENTRY_POINTS,
    EXAMPLE_ENTRY_POINTS,
    register_distribution,
)


@pytest.fixture(autouse=True)
def _isolate_git_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep inherited Git configuration out of test repositories."""
    for name in tuple(os.environ):
        if name == 'GIT_CONFIG' or name.startswith('GIT_CONFIG_'):
            monkeypatch.delenv(name)
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')


@pytest.fixture
def distribution_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepend an isolated distribution metadata directory to `sys.path`."""
    root = tmp_path / 'distributions'
    root.mkdir()
    monkeypatch.setattr(sys, 'path', [str(root), *sys.path])
    return root


@pytest.fixture
def registered_hosts(distribution_root: Path) -> Path:
    """Register every bundled host through synthesized distribution metadata."""
    register_distribution(distribution_root, 'docwrap', BUILTIN_ENTRY_POINTS)
    return distribution_root


@pytest.fixture
def example_hosts(registered_hosts: Path) -> Path:
    """Register the generic test adapters beside the bundled adapters."""
    register_distribution(registered_hosts, 'docwrap-host-example', EXAMPLE_ENTRY_POINTS)
    return registered_hosts
