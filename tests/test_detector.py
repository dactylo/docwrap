import pytest

from docwrap._detector import _tokens_commit, detect_commit, is_git_executable
from docwrap._shellparse import Command, parse_command


class TestHeredocRules:
    def test_expansion_scan_ignores_quotes_in_an_unquoted_body(self) -> None:
        assert _detected("cat <<EOF\n'$(git commit -m x)'\nEOF")

    def test_heredoc_without_commit(self) -> None:
        assert not _detected("cat <<'EOF'\nplain text about commits\nEOF")

    def test_prose_about_commits_in_a_heredoc_body(self) -> None:
        assert not _detected("cat <<'EOF'\nplain text about commits and git history\nEOF")

    def test_script_scan_applies_to_every_delimiter(self) -> None:
        assert _detected("sh <<'EOF'\ngit commit -m x\nEOF")
        assert _detected("cat <<'EOF'\n$(git commit -m x)\nEOF")

    def test_subprocess_commit_in_a_heredoc_body(self) -> None:
        body = (
            "python - <<'PY'\n"
            'import subprocess\n'
            "subprocess.run(['git','commit','-m','x'],check=True)\n"
            'PY'
        )
        assert _detected(body)

    def test_unquoted_body_substitution_preserves_quoted_parentheses(self) -> None:
        assert _detected("cat <<EOF\n'$(printf ')'; git commit -m x)'\nEOF")

    def test_unterminated_heredoc_is_scanned(self) -> None:
        assert _detected('sh <<EOF\ngit commit -m x')


class TestWordRule:
    @pytest.mark.parametrize(
        'text',
        [
            'git commit -m x',
            'git --no-pager commit -m x',
            'git.exe commit -m x',
            'git.EXE commit -m x',
            'Git.Exe commit -m x',
            "'git' 'commit'",
            'g""it com""mit',
            "git -C ';' commit -m x",
            "git add a.py ';' git commit -m x",
            'env FOO=1 git commit -m x',
        ],
    )
    def test_git_then_commit_in_one_segment(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            'git status',
            'git commitx',
            'commit git',
            'echo "git commit"',
            "grep 'git commit' docs/hook.md",
            'ls # git commit',
            'git $ACTION',
        ],
    )
    def test_no_commit_word_pair(self, text: str) -> None:
        assert not _detected(text)


class TestExecutableRule:
    @pytest.mark.parametrize(
        'text',
        [
            '/usr/bin/git commit -m x',
            '/opt/toolchain/usr/bin/git -C /abs/repo commit -m x',
            './git commit -m x',
            '/usr/bin/git.exe commit -m x',
            '"C:/Program Files/Git/cmd/git.exe" commit -m x',
            "'C:\\Program Files\\Git\\cmd\\git.exe' commit -m x",
            "'.\\git.exe' commit -m x",
            "'\\\\server\\share\\GIT.EXE' commit -m x",
            'gittoken git-exec -- /usr/bin/git -C /abs/repo commit -m x',
            'env -u X /Users/someone/.local/bin/git -C /abs/repo commit -m x',
        ],
    )
    def test_path_qualified_git_is_a_commit(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            '/usr/bin/gitk commit',
            '/usr/bin/git/ commit',
            'gits commit',
            "grep 'git commit' README.md",
            'echo "/usr/bin/git commit"',
        ],
    )
    def test_other_basenames_are_not_git(self, text: str) -> None:
        assert not _detected(text)

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('git', True),
            ('GIT', True),
            ('git.exe', True),
            ('Git.Exe', True),
            ('/usr/bin/git', True),
            ('C:\\Git\\cmd\\git.exe', True),
            ('./git', True),
            ('git commit', False),
            ('gitk', False),
            ('git/', False),
            ('', False),
        ],
    )
    def test_is_git_executable(self, text: str, *, expected: bool) -> None:
        assert is_git_executable(text) is expected


class TestExpansionRule:
    @pytest.mark.parametrize(
        'text',
        [
            '$GIT commit -m x',
            '"$(command -v git)" -C /abs/repo commit -m x',
            '${GIT:-git} commit',
            'env FOO=1 $GIT commit -m x',
        ],
    )
    def test_expansion_before_commit_is_a_commit(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            'git $ACTION',
            'echo $MESSAGE',
            'commit $X',
            'grep commit $FILE',
        ],
    )
    def test_expansion_without_a_later_commit_word(self, text: str) -> None:
        assert not _detected(text)


class TestScriptDataRule:
    @pytest.mark.parametrize(
        'text',
        [
            "python -c \"import subprocess; subprocess.run(['git', 'commit', '-m', 'x'])\"",
            'python3 -c \'import subprocess; subprocess.run(["git", "commit"])\'',
            'python3.13 -c \'import subprocess; subprocess.check_call(("git", "commit"))\'',
            '/usr/bin/python -c \'import os; os.system("git commit -m x")\'',
            'python -X utf8 -c \'import subprocess; subprocess.run(["git", "commit"])\'',
            'perl -e \'system("git", "commit", "-m", "x")\'',
            'ruby -e \'system("git", "commit", "-m", "x")\'',
            'node -e \'require("child_process").execFileSync("git", ["commit", "-m", "x"])\'',
            'python -c "print(\'git commit\')"',
            "python -c 'git commit -m x'",
        ],
    )
    def test_inline_scripts_that_commit(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            'python -c "print(\'hello\')"',
            'python script.py',
            'python -c \'import subprocess; subprocess.run(["git", "status"])\'',
            'python -c "$SCRIPT"',
            'python -m pytest -k commit',
            'perl -e \'print "commit"\'',
            'node -e \'console.log("gitcommit")\'',
        ],
    )
    def test_inline_scripts_that_do_not_commit(self, text: str) -> None:
        assert not _detected(text)

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ("subprocess.run(['git','commit'])", True),
            ('system("git", "commit")', True),
            ('execFileSync("git", ["commit"])', True),
            ("os.system('git commit -m x')", True),
            ('git=commit', True),
            ('plain text about commits', False),
            ("subprocess.run(['git', 'status'])", False),
            ('commit git', False),
            ('', False),
        ],
    )
    def test_tokens_commit(self, text: str, *, expected: bool) -> None:
        assert _tokens_commit(text) is expected


class TestSubstitutionRule:
    @pytest.mark.parametrize(
        'text',
        [
            'echo "$(git commit -m x)"',
            'echo $(git commit -m x)',
            'echo `git commit -m x`',
            'echo "${VALUE:-$(git commit -m x)}"',
            'echo "$(echo $(git commit -m x))"',
            'echo "$(printf "%s" "$(printf ")")"; sh -c \'git commit -m x\')"',
        ],
    )
    def test_nested_commits(self, text: str) -> None:
        assert _detected(text)

    def test_single_quoted_text_is_not_a_substitution(self) -> None:
        assert not _detected("echo '$(git commit -m x)'")

    def test_truncated_substitution_is_a_commit(self) -> None:
        assert _detected('echo ' + '$(echo ' * 9 + 'deep' + ')' * 9)


class TestEvaluatorRule:
    @pytest.mark.parametrize('suffix', ['', '; git commit --interactive'])
    def test_eval_chain_reparses_only_distinct_suffixes(
        self, suffix: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        command = parse_command('eval ' * 24 + 'true' + suffix)
        reparsed: list[str] = []

        def counted_parse(text: str) -> Command:
            assert text not in reparsed, 'repeated evaluator script parse'
            reparsed.append(text)
            return parse_command(text)

        monkeypatch.setattr('docwrap._detector.parse_command', counted_parse)
        for _ in range(2):
            reparsed.clear()
            assert detect_commit(command) is bool(suffix)
            assert len(reparsed) == 24

    @pytest.mark.parametrize(
        'text',
        [
            "sh -c 'git commit -m x'",
            "A=x bash -c 'git commit -m x'",
            "command bash -c 'git commit -m x'",
            "if true; then bash -c 'git commit -m x'; fi",
            "sh -e -c 'git commit -m x'",
            "zsh -lc 'git commit -m x'",
            'bash -c "git commit -m x"',
            "bash -c -- 'git commit -m x'",
            "bash -c -e 'git commit -m x'",
            "bash -c -o posix 'git commit -m x'",
            "bash -c +e 'git commit -m x'",
            "dash -c 'git commit -m x'",
            "ksh -c 'git commit -m x'",
            "/bin/sh -c 'git commit -m x'",
            "eval 'git' 'commit -m x'",
            "eval 'true' ';' 'git commit -m x'",
            'eval "git commit -m x"',
        ],
    )
    def test_scripts_are_classified(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            "sh 'git commit -m x'",
            "sh --c 'git commit -m x'",
            "fish -c 'git commit -m x'",
            'eval "$CMD"',
            "sh -c 'echo commit'",
        ],
    )
    def test_other_forms_are_not_scripts(self, text: str) -> None:
        assert not _detected(text)


class TestRecoveryRule:
    @pytest.mark.parametrize(
        'text',
        [
            'g""it com""mit -m "x\ry"',
            "git -C ';' commit -m 'x\ry'",
            "git commit -m 'unterminated",
            'git commit -m "$(echo x"',
        ],
    )
    def test_recovered_trees_are_classified(self, text: str) -> None:
        assert _detected(text)


def _detected(text: str) -> bool:
    return detect_commit(parse_command(text))
