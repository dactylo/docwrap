import json
import shlex
from pathlib import Path

import pytest

from docwrap._vendor.hookshell._classify import (
    _deny_reason,
    classify_command,
)
from docwrap._vendor.hookshell._detector import (
    detect_commit,
)
from docwrap._vendor.hookshell._grammar import (
    CommitPlan,
    NoCommit,
    Rejection,
    classify_grammar,
)
from docwrap._vendor.hookshell._shellparse import (
    parse_command,
)

CORPUS = Path(__file__).parent / 'data' / 'commit_classification.json'
PREFIX = 'tool: unsupported commit command'


class TestReason:
    def test_all_control_ranges_are_escaped_in_the_reason(self) -> None:
        rejection = Rejection('a\x00\x1f\x7f\x80\x9fb', 'hint\x9f')
        assert _deny_reason(rejection, PREFIX) == (
            f'{PREFIX}: a\\x00\\x1f\\x7f\\x80\\x9fb; hint\\x9f'
        )

    def test_fragments_are_bounded_and_sanitized(self, corpus_root: Path) -> None:
        long_segment = 'make ' + 'x' * 80 + '\x1b[31m'
        classification = classify_command(
            f'{long_segment} && git commit -m x', str(corpus_root), reason_prefix=PREFIX
        )
        assert classification.reason is not None
        element = classification.reason.removeprefix(f'{PREFIX}: ').split(';')[0]
        assert element == f"segment '{long_segment[:40]}'"
        assert '\x1b' not in classification.reason
        assert _deny_reason(Rejection('x\x07', 'hint'), PREFIX) == f'{PREFIX}: x\\x07; hint'

    def test_precheck_reason_names_the_command_and_separate_call_remedy(
        self, corpus_root: Path
    ) -> None:
        classification = classify_command(
            'git diff --check && git add a.py && git commit -m x',
            str(corpus_root),
            reason_prefix=PREFIX,
        )
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f'{PREFIX}: read-only git diff before git commit; '
            'run the check or observation in a separate tool call, then run the add/commit '
            'command only after success; for a check after staging, run add plus check '
            'without a real commit, then submit the commit separately'
        )

    def test_reason_names_the_element_and_the_accepted_form(self, corpus_root: Path) -> None:
        classification = classify_command(
            'git commit -m x && git push', str(corpus_root), reason_prefix=PREFIX
        )
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f"{PREFIX}: segment 'git push'; supported form: "
            '[cd DIR &&] [git add ...;] git commit ...'
        )

    def test_reason_uses_the_caller_prefix(self, corpus_root: Path) -> None:
        classification = classify_command(
            'git commit -m x && git push',
            str(corpus_root),
            reason_prefix='acme: unsupported commit command',
        )
        assert classification.reason is not None
        assert classification.reason.startswith(
            "acme: unsupported commit command: segment 'git push'; "
        )

    def test_source_heredoc_with_many_shift_operators_is_unrelated(self) -> None:
        source = 'std::cout << "hello";\n' * 400
        result = classify_command(
            f"cat > t.cpp <<'EOF'\n{source}EOF\n", None, reason_prefix=PREFIX
        )
        assert result.outcome == 'unrelated'
        assert result.reason is None

    def test_unknown_directory_reason_is_one_line_with_a_placeholder_route(self) -> None:
        classification = classify_command('git commit -m x', None, reason_prefix=PREFIX)
        assert classification.outcome == 'unsupported'
        assert classification.reason == (
            f'{PREFIX}: unknown working directory; supported form: '
            'git -C /absolute/repo commit ... or cd /absolute/repo && git commit ...'
        )

    def test_unrelated_and_supported_carry_no_reason(self, corpus_root: Path) -> None:
        assert classify_command('ls', str(corpus_root), reason_prefix=PREFIX).reason is None
        assert (
            classify_command('git commit -m x', str(corpus_root), reason_prefix=PREFIX).reason
            is None
        )


class TestTotality:
    @pytest.mark.parametrize(
        'template', ['git -C {route} commit -m x', 'cd {route} && git commit -m x']
    )
    @pytest.mark.parametrize('component', ['\0', 'a' * 256], ids=['invalid', 'overlong'])
    def test_unusable_absolute_route_never_produces_a_commit_plan(
        self, template: str, component: str, tmp_path: Path
    ) -> None:
        route = shlex.quote(tmp_path.as_posix() + '/' + component)
        result = classify_command(template.format(route=route), None, reason_prefix=PREFIX)
        # Windows can report an overlong route as missing; that cannot create a commit.
        if isinstance(result.plan, NoCommit):
            assert result.outcome == 'supported'
            assert result.reason is None
            return
        assert result.outcome == 'unsupported'
        assert result.plan is None
        assert result.reason is not None
        assert 'git -C /absolute/repo' in result.reason
        assert 'cd /absolute/repo' in result.reason

    @pytest.mark.parametrize(
        'template', ['git -C {route} commit -m x', 'cd {route} && git commit -m x']
    )
    def test_route_lookup_error_is_unsupported(
        self, template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_stat(_path: Path) -> None:
            raise PermissionError('route lookup refused')

        route = shlex.quote(tmp_path.as_posix())
        monkeypatch.setattr(Path, 'stat', fail_stat)
        result = classify_command(template.format(route=route), None, reason_prefix=PREFIX)
        assert result.outcome == 'unsupported'
        assert result.plan is None
        assert result.reason is not None
        assert 'unresolvable directory route' in result.reason

    @pytest.mark.parametrize(
        ('suffix', 'outcome'),
        [('echo x', 'unrelated'), ('true; git commit -m x', 'unsupported')],
    )
    def test_deep_eval_chain_returns_a_classification(self, suffix: str, outcome: str) -> None:
        result = classify_command('eval ' * 180 + suffix, None, reason_prefix=PREFIX)
        assert result.outcome == outcome
        assert result.plan is None


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
        classification = classify_command(command, cwd, reason_prefix=PREFIX)
        assert classification.outcome == entry['outcome']
        assert detect_commit(parse_command(command)) is entry['detected']
        if classification.outcome == 'unsupported':
            assert classification.plan is None
        if isinstance(classification.plan, CommitPlan):
            assert entry['detected'] is True
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
