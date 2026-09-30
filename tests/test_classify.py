import json
from pathlib import Path

import pytest

from docwrap._classify import _REASON_PREFIX, _deny_reason, classify_command
from docwrap._detector import detect_commit
from docwrap._grammar import CommitPlan, NoCommit, Rejection, classify_grammar
from docwrap._shellparse import parse_command

CORPUS = Path(__file__).parent / 'data' / 'commit_classification.json'


class TestReason:
    def test_fragments_are_bounded_and_sanitized(self, corpus_root: Path) -> None:
        long_segment = 'make ' + 'x' * 80 + '\x1b[31m'
        classification = classify_command(f'{long_segment} && git commit -m x', str(corpus_root))
        assert classification.reason is not None
        element = classification.reason.removeprefix(f'{_REASON_PREFIX}: ').split(';')[0]
        assert element == f"segment '{long_segment[:40]}'"
        assert '\x1b' not in classification.reason
        assert _deny_reason(Rejection('x\x07', 'hint')) == f'{_REASON_PREFIX}: x\\x07; hint'

    def test_precheck_reason_names_the_command_and_separate_call_remedy(
        self, corpus_root: Path
    ) -> None:
        classification = classify_command(
            'git diff --check && git add a.py && git commit -m x', str(corpus_root)
        )
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f'{_REASON_PREFIX}: read-only git diff before git commit; '
            'run the check or observation in a separate tool call, then run the add/commit '
            'command only after success; for a check after staging, run add plus check '
            'without a real commit, then submit the commit separately'
        )

    def test_reason_names_the_element_and_the_accepted_form(self, corpus_root: Path) -> None:
        classification = classify_command('git commit -m x && git push', str(corpus_root))
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f"{_REASON_PREFIX}: segment 'git push'; supported form: "
            '[cd DIR &&] [git add ...;] git commit ...'
        )

    def test_unknown_directory_reason_is_one_line_with_a_placeholder_route(self) -> None:
        classification = classify_command('git commit -m x', None)
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f'{_REASON_PREFIX}: unknown working directory; supported form: '
            'git -C /absolute/repo commit ... or cd /absolute/repo && git commit ...'
        )

    def test_unrelated_and_supported_carry_no_reason(self, corpus_root: Path) -> None:
        assert classify_command('ls', str(corpus_root)).reason is None
        assert classify_command('git commit -m x', str(corpus_root)).reason is None


def _entries() -> list[dict[str, object]]:
    data = json.loads(CORPUS.read_text(encoding='utf-8'))
    return data['entries']


def _ids() -> list[str]:
    return [str(entry['id']) for entry in _entries()]


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    (tmp_path / 'repo').mkdir()
    (tmp_path / ';').mkdir()
    return tmp_path


class TestCorpus:
    @pytest.mark.parametrize('entry', _entries(), ids=_ids())
    def test_expected_outcome_and_detection(
        self, entry: dict[str, object], corpus_root: Path
    ) -> None:
        command = str(entry['command'])
        cwd = _cwd(entry, corpus_root)
        classification = classify_command(command, cwd)
        assert classification.outcome == entry['outcome']
        assert detect_commit(parse_command(command)) is entry['detected']
        if classification.outcome == 'unsupported':
            assert classification.plan is None
        if classification.plan is not None:
            assert isinstance(classification.plan, CommitPlan) is entry.get('commits', True)
        if 'element' in entry:
            result = classify_grammar(parse_command(command), cwd)
            assert isinstance(result, Rejection)
            assert result.element == entry['element']

    def test_ids_are_unique(self) -> None:
        ids = _ids()
        assert len(ids) == len(set(ids))

    def test_supported_inventory_covers_selection_modes(self, corpus_root: Path) -> None:
        covered: set[str] = set()
        for entry in _entries():
            if entry['outcome'] != 'supported':
                continue
            plan = classify_grammar(parse_command(str(entry['command'])), _cwd(entry, corpus_root))
            assert not isinstance(plan, Rejection)
            if isinstance(plan, NoCommit):
                continue
            for mode in (
                'auto_stage',
                'include',
                'only',
                'pathspec_from_file',
                'pathspec_file_nul',
            ):
                if getattr(plan.mode, mode):
                    covered.add(mode)
        assert covered == {
            'auto_stage',
            'include',
            'only',
            'pathspec_from_file',
            'pathspec_file_nul',
        }


def _cwd(entry: dict[str, object], corpus_root: Path) -> str | None:
    if 'command_cwd' in entry and entry['command_cwd'] is None:
        return None
    return str(corpus_root)
