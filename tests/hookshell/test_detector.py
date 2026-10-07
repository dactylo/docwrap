import pytest

from docwrap._vendor.hookshell import (
    _detector,
)
from docwrap._vendor.hookshell._detector import (
    _tokens_commit,
    detect_commit,
    is_git_executable,
)
from docwrap._vendor.hookshell._shellparse import (
    Command,
    parse_command,
)


class TestDataSinkRule:
    @pytest.mark.parametrize(
        'text',
        [
            "cat > notes.md <<'EOF'\ngit log then commit\nEOF\n",
            "cat >> notes.md <<'EOF'\ngit commit -m x\nEOF\n",
            "cat <<'EOF' > notes.md\ngit commit -m x\nEOF\n",
            "cat <<-'EOF' > notes.md\n\tgit commit -m x\n\tEOF\n",
            "tee notes.md <<'EOF'\ngit commit -m x\nEOF\n",
            "tee -a notes.md <<'EOF'\ngit commit -m x\nEOF\n",
        ],
    )
    def test_quoted_heredoc_written_to_a_file_is_data(self, text: str) -> None:
        assert not _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            "cat <<'EOF'\ngit commit -m x\nEOF\n",
            "cat > a.md <<'EOF' && echo done\ngit commit -m x\nEOF\n",
            "cat > s.sh <<'EOF' && sh s.sh\ngit commit -m x\nEOF\n",
            "cat > s.sh <<'EOF'\ngit commit -m x\nEOF\nsh s.sh\n",
            'eval "$(tee s.sh <<\'EOF\'\ngit commit -m x\nEOF\n)"\n',
            "cat > f <<'EOF' | sh\ngit commit -m x\nEOF\n",
            "tee f <<'EOF' | sh\ngit commit -m x\nEOF\n",
            "tee f <<'EOF' |& sh\ngit commit -m x\nEOF\n",
            "cat <<'EOF' > >(sh)\ngit commit -m x\nEOF\n",
            "tee >(sh) <<'EOF'\ngit commit -m x\nEOF\n",
            "cat <<'EOF' >&2\ngit commit -m x\nEOF\n",
            "tee <<'EOF'\ngit commit -m x\nEOF\n",
            "tee -a <<'EOF'\ngit commit -m x\nEOF\n",
            "sudo tee f <<'EOF'\ngit commit -m x\nEOF\n",
            "FOO=1 cat > f <<'EOF'\ngit commit -m x\nEOF\n",
            "/bin/cat > f <<'EOF'\ngit commit -m x\nEOF\n",
            "cat > f < in <<'EOF'\ngit commit -m x\nEOF\n",
            'cat > f <<EOF\n$(git commit -m x)\nEOF\n',
        ],
    )
    def test_consumed_or_unquoted_heredocs_are_scanned(self, text: str) -> None:
        assert _detected(text)

    @pytest.mark.parametrize(
        'text',
        [
            'cat > "$(git commit -m x)" <<\'EOF\'\nbody\nEOF\n',
            "cat > f <<'EOF' && git commit -m x\nbody\nEOF\n",
        ],
    )
    def test_sink_words_and_later_segments_are_still_scanned(self, text: str) -> None:
        assert _detected(text)


class TestHeredocRules:
    def test_evaluator_recovered_heredoc_retains_token_scan(self) -> None:
        assert _detected("eval \"cat <<MISSING\nsubprocess.run(['git','commit'])\n\"")

    def test_expansion_scan_ignores_quotes_in_an_unquoted_body(self) -> None:
        assert _detected("cat <<EOF\n'$(git commit -m x)'\nEOF")

    def test_heredoc_without_commit(self) -> None:
        assert not _detected("cat <<'EOF'\nplain text about commits\nEOF")

    def test_long_unmatched_heredoc_chain_is_detected(self) -> None:
        command = parse_command(
            "python - <<'PY'\n" + 'value = 1 << 2\n' * 2000 + "$GIT com''mit\nPY\n"
        )
        assert detect_commit(command)

    def test_nested_heredoc_scans_quoted_fragments(self) -> None:
        assert _detected('cat <<\'OUTER\'\nsh <<\'INNER\'\ng""it com""mit -m x\nINNER\nOUTER\n')

    @pytest.mark.parametrize(
        'body',
        [
            "cat <<MISSING\n$GIT com''mit\n",
            "cat <<'MISSING'\n$GIT com''mit\n",
            "cat <<-MISSING\n\t$GIT com''mit\n",
            'cat <<-MISSING\n$GIT com\\\n\tmit\n',
            "cat <<FIRST\ncat <<-SECOND\n'$( $GIT com\\\n\tmit )'\n",
            "cat <<MISSING; true\n$GIT com''mit\n",
            "cat <<MISSING <<OTHER\n$GIT com''mit\n",
            'cat <<MISSING\n\'$( $GIT com""mit )\'\n',
            "cat <<'FIRST'\ncat <<SECOND\n'$( $GIT com\"\"mit )'\n",
            'cat <<MISSING\npython -c \'os.system("git commit")\'\n',
            'cat <<MISSING\nsh -c \'$GIT com""mit\'\n',
            "cat <<MISSING\nsubprocess.run(['git','commit'])\n",
            "cat <<'INNER'\n$GIT com''mit\nINNER\n",
        ],
    )
    def test_nested_script_detection_survives_heredoc_recovery(self, body: str) -> None:
        assert _detected(f"cat <<'OUTER'\n{body}OUTER\n")

    def test_prose_about_commits_in_a_heredoc_body(self) -> None:
        assert not _detected("cat <<'EOF'\nplain text about commits and git history\nEOF")

    def test_script_scan_applies_to_every_delimiter(self) -> None:
        assert _detected("sh <<'EOF'\ngit commit -m x\nEOF")
        assert _detected("cat <<'EOF'\n$(git commit -m x)\nEOF")

    @pytest.mark.parametrize(
        ('prefix', 'delimiter'),
        [("python - <<'PY'\n", 'PY'), ("cat <<'CPP'\n", 'CPP')],
    )
    @pytest.mark.parametrize('suffix', ['', "$GIT com''mit\n"])
    @pytest.mark.parametrize('operand', ['"hello"', '""'])
    def test_shift_operators_do_not_reparse_successive_body_suffixes(
        self,
        prefix: str,
        delimiter: str,
        suffix: str,
        operand: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        body = f'std::cout << {operand};\n' * 400 + suffix
        command = parse_command(f'{prefix}{body}{delimiter}\n')
        parsed_characters = 0

        def counted_parse(text: str, **kwargs: bool) -> Command:
            nonlocal parsed_characters
            parsed_characters += len(text)
            assert parsed_characters <= 2 * len(body), 'reparsed overlapping heredoc suffixes'
            return parse_command(text, **kwargs)

        monkeypatch.setattr(_detector, 'parse_command', counted_parse)
        assert detect_commit(command) is bool(suffix)

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
    @pytest.mark.parametrize('body', [r'g\\it commit -m x', r'git com\\mit -m x'])
    @pytest.mark.parametrize('quoted', [False, True])
    def test_backtick_escape_layer_cannot_hide_commit(self, body: str, *, quoted: bool) -> None:
        substitution = '`' + body + '`'
        text = 'echo ' + ('"' + substitution + '"' if quoted else substitution)
        assert _detected(text)

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

        def counted_parse(text: str, **kwargs: bool) -> Command:
            assert text not in reparsed, 'repeated evaluator script parse'
            reparsed.append(text)
            return parse_command(text, **kwargs)

        monkeypatch.setattr(_detector, 'parse_command', counted_parse)
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
            "eval '#' eval 'git commit -m x'",
            'eval \'# ignored\' sh -c \'g""it com""mit -m x\'',
            'eval \'eval\' \'"g\\"\\"it com\\"\\"mit -m x"\'',
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


class TestEvaluatorSpellings:
    @pytest.mark.parametrize(
        'name', ['bash', 'dash', 'ksh', 'sh', 'zsh', 'node', 'perl', 'ruby', 'python3.14']
    )
    @pytest.mark.parametrize(
        'spelling', ['{name}', '{name}.exe', '/usr/bin/{name}', 'C:\\tools\\{name}.EXE']
    )
    def test_listed_evaluators_match_path_case_and_exe_variants(
        self, name: str, spelling: str
    ) -> None:
        executable = spelling.format(name=name.upper())
        option = '-e' if name in {'node', 'perl', 'ruby'} else '-c'
        assert _detected(f"'{executable}' {option} 'git commit -m x'")

    @pytest.mark.parametrize('name', ['bash.exe', 'perl.exe', 'C:\\tools\\SH.EXE'])
    def test_listed_evaluator_without_commit_is_not_detected(self, name: str) -> None:
        assert not _detected(f"'{name}' -c 'git status'")

    @pytest.mark.parametrize('name', ['bash.exe.bak', 'mybash.exe', 'perl.exe.bak', 'myperl.exe'])
    def test_unlisted_executables_are_not_evaluators(self, name: str) -> None:
        assert not _detected(f"'{name}' -c 'git commit'")


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
