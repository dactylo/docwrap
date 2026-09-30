from pathlib import Path

import pytest

from tests.helpers import git_commit_all, run_setup


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
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
    return tmp_path


@pytest.fixture
def git_repo_with_file(git_repo: Path) -> tuple[Path, Path]:
    path = git_repo / 'foo.py'
    path.write_text('pass\n')
    git_commit_all(git_repo, 'add foo')
    return git_repo, path
