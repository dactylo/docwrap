from pathlib import Path

import pytest

from docwrap._core._incremental import DiffHunk, get_eligible_ranges, parse_diff_hunks
from tests.helpers import decode_text, run_setup


class TestControlFlowNestedDefinition:
    def test_class_inside_try_is_eligible(self) -> None:
        source = (
            'try:\n'
            '    class Foo:\n'
            '        """Foo doc."""\n'
            '        x = 1\n'
            'except Exception:\n'
            '    pass\n'
        )
        changed: list[DiffHunk] = [(4, 4, [])]
        ranges = get_eligible_ranges(source, changed, level='definition')
        assert ranges is not None
        assert (2, 4) in ranges

    def test_function_inside_if_is_eligible(self) -> None:
        source = (
            'def outer():\n'
            '    """Outer doc."""\n'
            '    if True:\n'
            '        def inner():\n'
            '            """Inner doc."""\n'
            '            x = 1\n'
            '    pass\n'
        )
        # Change inner's body (line 6)
        changed: list[DiffHunk] = [(6, 6, [])]
        ranges = get_eligible_ranges(source, changed, level='definition')
        assert ranges is not None
        # inner's range (4, 6) should be in eligible, not just outer's
        assert (4, 6) in ranges


class TestDiffHeaderPathWithSpace:
    def test_path_containing_b_slash(self) -> None:
        diff = (
            'diff --git a/src b/file.py b/src b/file.py\n'
            'index abc..def 100644\n'
            '--- a/src b/file.py\n'
            '+++ b/src b/file.py\n'
            '@@ -1,1 +1,2 @@\n'
            '+new line\n'
        )
        result = parse_diff_hunks(diff)
        assert result is not None
        assert 'src b/file.py' in result

    def test_unquoted_patch_header_tab_delimiter_is_not_part_of_filename(self) -> None:
        diff = (
            'diff --git a/space name.py b/space name.py\n'
            'index abc..def 100644\n'
            '--- a/space name.py\t\n'
            '+++ b/space name.py\t\n'
            '@@ -2 +2 @@\n'
            '-    """A coffee summary.\r\n'
            '+    """A cafe summary.\r\n'
        )

        assert parse_diff_hunks(diff) == {
            'space name.py': [(2, 2, [b'    """A coffee summary.\r'])]
        }


class TestDiffHeaderQuoting:
    @pytest.mark.parametrize(
        ('header_path', 'expected'),
        [
            ('"a/tab\\tname.py"', 'tab\tname.py'),
            ('"a/slash\\\\name.py"', 'slash\\name.py'),
            ('"a/quote\\"name.py"', 'quote"name.py'),
            ('"a/caf\\303\\251.py"', 'café.py'),
            ('"a/café.py"', 'café.py'),
            ('"a/raw\\377.py"', 'raw\udcff.py'),
        ],
    )
    def test_c_quoted_plus_path_is_decoded(
        self,
        header_path: str,
        expected: str,
    ) -> None:
        new_path = header_path.replace('a/', 'b/', 1)
        diff = f'diff --git ignored ignored\n--- {header_path}\n+++ {new_path}\n@@ -1 +1 @@\n+x\n'

        assert parse_diff_hunks(diff) == {expected: [(1, 1, [])]}

    @pytest.mark.parametrize(
        'header_path',
        [
            '"a/bad\\q.py"',
            '"a/bad\\12.py"',
            '"a/bad.py',
            '"a/bad.py"x',
        ],
        ids=['unknown-escape', 'short-octal', 'missing-quote', 'trailing-text'],
    )
    def test_malformed_diff_path_returns_none(self, header_path: str) -> None:
        new_path = header_path.replace('a/', 'b/', 1)
        diff = f'diff --git ignored ignored\n--- {header_path}\n+++ {new_path}\n@@ -1 +1 @@\n+x\n'

        assert parse_diff_hunks(diff) is None

    def test_quoted_dev_null_is_not_accepted(self) -> None:
        diff = 'diff --git ignored ignored\n--- "/dev/null"\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n'

        assert parse_diff_hunks(diff) == {}


class TestEligibleRanges:
    def test_definition_level_class_attribute(self) -> None:
        source = (
            'class Foo:\n'  # 1
            '    """Class doc."""\n'  # 2
            '    x = 1\n'  # 3
            '    def bar(self):\n'  # 4
            '        pass\n'  # 5
        )
        changed: list[DiffHunk] = [(3, 3, [])]  # class attribute
        result = get_eligible_ranges(source, changed, level='definition')
        assert result is not None
        assert (1, 5) in result  # Foo (full definition extent)

    def test_definition_level_method_body(self) -> None:
        source = (
            'class Foo:\n'  # 1
            '    """Class doc."""\n'  # 2
            '    def bar(self):\n'  # 3
            '        """Bar."""\n'  # 4
            '        x = 1\n'  # 5
        )
        changed: list[DiffHunk] = [(5, 5, [])]  # change in method body
        result = get_eligible_ranges(source, changed, level='definition')
        assert result == {(3, 5)}

    def test_definition_level_module_code(self) -> None:
        source = (
            '"""Module doc."""\n'  # 1
            'x = 1\n'  # 2
            'def foo():\n'  # 3
            '    pass\n'  # 4
        )
        changed: list[DiffHunk] = [(2, 2, [])]
        result = get_eligible_ranges(source, changed, level='definition')
        assert result is not None
        assert (0, 0) in result  # module (key is always (0, 0))

    def test_docstring_level(self) -> None:
        source = (
            'def foo():\n'  # 1
            '    """Old doc."""\n'  # 2
            '    pass\n'  # 3
        )
        changed: list[DiffHunk] = [(2, 2, [])]  # docstring itself changed
        result = get_eligible_ranges(source, changed, level='docstring')
        assert result is not None
        assert (2, 2) in result

    def test_docstring_level_body_change_not_eligible(self) -> None:
        source = (
            'def foo():\n'  # 1
            '    """Doc."""\n'  # 2
            '    x = 1\n'  # 3
        )
        changed: list[DiffHunk] = [(3, 3, [])]  # body change, not docstring
        result = get_eligible_ranges(source, changed, level='docstring')
        assert result is not None
        assert len(result) == 0

    def test_file_level_any_change(self) -> None:
        source = 'def foo():\n    """Doc."""\n    pass\n'
        changed: list[DiffHunk] = [(2, 2, [])]
        result = get_eligible_ranges(source, changed, level='file')
        assert result is None  # None means "all eligible"

    def test_file_level_no_changes(self) -> None:
        source = 'def foo():\n    """Doc."""\n    pass\n'
        result = get_eligible_ranges(source, [], level='file')
        # With no changes, even file level should return empty
        assert result is not None
        assert len(result) == 0

    def test_no_changes(self) -> None:
        source = 'def foo():\n    """Doc."""\n    pass\n'
        result = get_eligible_ranges(source, [], level='definition')
        assert result is not None
        assert len(result) == 0


class TestIncrementalDiffWorkflow:
    @pytest.mark.parametrize(
        ('removed', 'gap', 'expected'),
        [
            ('old detail', 3, {(2, 4)}),
            ('    obsolete = 1', 4, set[tuple[int, int]]()),
        ],
        ids=['inside-docstring', 'after-docstring'],
    )
    def test_docstring_deletion_selects_only_edited_literal(
        self, removed: str, gap: int, expected: set[tuple[int, int]]
    ) -> None:
        source = 'def target():\n    """Summary.\n    Detail.\n    """\n    return 1\n'
        diff = (
            'diff --git a/sample.py b/sample.py\n'
            '--- a/sample.py\n'
            '+++ b/sample.py\n'
            f'@@ -{gap + 1} +{gap},0 @@\n'
            f'-{removed}\n'
        )
        hunks = parse_diff_hunks(diff)
        assert hunks is not None

        assert get_eligible_ranges(source, hunks['sample.py'], level='docstring') == expected

    def test_git_diff_definition_level_ranges(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=str(tmp_path))
        f = tmp_path / 'foo.py'
        long_ds = 'Long docstring that exceeds the default line length limit'
        original = (
            'def foo():\n'
            f'    """{long_ds}."""\n'
            '    pass\n'
            '\n'
            'def bar():\n'
            f'    """{long_ds}."""\n'
            '    pass\n'
        )
        f.write_text(original)
        run_setup(['git', 'add', '.'], cwd=str(tmp_path))
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
            ],
            cwd=str(tmp_path),
        )

        # Modify foo's body only
        modified = original.replace('    pass\n', '    x = 1\n', 1)
        f.write_text(modified)

        # Get diff and compute eligible ranges
        result = run_setup(
            ['git', 'diff', '-U0', '-M', 'HEAD', '--', str(f)],
            cwd=str(tmp_path),
        )

        hunks = parse_diff_hunks(decode_text(result.stdout))
        assert hunks is not None
        file_hunks = hunks.get('foo.py', [])
        ranges = get_eligible_ranges(modified, file_hunks, level='definition')

        assert ranges == {(1, 3)}

    def test_leading_deletion_makes_module_eligible(self, tmp_path: Path) -> None:
        run_setup(['git', 'init'], cwd=tmp_path)
        path = tmp_path / 'foo.py'
        original = 'obsolete = True\n"""Module docstring."""\n'
        path.write_text(original)
        run_setup(['git', 'add', '.'], cwd=tmp_path)
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
            ],
            cwd=tmp_path,
        )

        modified = '"""Module docstring."""\n'
        path.write_text(modified)
        diff = decode_text(
            run_setup(
                ['git', 'diff', '-U0', 'HEAD', '--', path.name],
                cwd=tmp_path,
            ).stdout
        )

        hunks = parse_diff_hunks(diff)
        assert hunks is not None
        ranges = get_eligible_ranges(
            modified,
            hunks.get(path.name, []),
            level='definition',
        )

        assert ranges == {(0, 0)}

    def test_parent_deletion_after_child_selects_only_parent(self) -> None:
        source = (
            'def parent():\n'
            '    """Parent docstring."""\n'
            '    def child():\n'
            '        """Child docstring."""\n'
            '        return 1\n'
            '    return child()\n'
        )
        diff = (
            'diff --git a/sample.py b/sample.py\n'
            '--- a/sample.py\n'
            '+++ b/sample.py\n'
            '@@ -6 +5,0 @@\n'
            '-    obsolete = 1\n'
        )
        hunks = parse_diff_hunks(diff)
        assert hunks is not None

        assert get_eligible_ranges(source, hunks['sample.py'], level='definition') == {(1, 6)}

    def test_removed_decorator_preserves_source_encoding(self) -> None:
        source = '# coding: latin-1\n\ndef target():\n    """Café docstring."""\n    return 1\n'
        patch = (
            b'diff --git a/sample.py b/sample.py\n'
            b'--- a/sample.py\n'
            b'+++ b/sample.py\n'
            b'@@ -3 +2,0 @@\n'
            b'-@decorate("caf\xe9")\n'
        )
        hunks = parse_diff_hunks(patch.decode('utf-8', errors='surrogateescape'))
        assert hunks is not None

        assert get_eligible_ranges(
            source, hunks['sample.py'], level='definition', encoding='latin-1'
        ) == {(3, 5)}

    def test_removed_decorator_selects_its_function(self) -> None:
        source = (
            'def first():\n'
            '    pass\n'
            '\n'
            '\n'
            'def target():\n'
            '    """Target docstring."""\n'
            '    return 1\n'
        )
        diff = (
            'diff --git a/sample.py b/sample.py\n'
            '--- a/sample.py\n'
            '+++ b/sample.py\n'
            '@@ -5 +4,0 @@\n'
            '-@decorator\n'
        )
        hunks = parse_diff_hunks(diff)
        assert hunks is not None

        assert get_eligible_ranges(source, hunks['sample.py'], level='definition') == {(5, 7)}


class TestInvalidSource:
    def test_definition_level_propagates_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            get_eligible_ranges('def broken(:\n', [(1, 1, [])], level='definition')

    def test_docstring_level_propagates_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            get_eligible_ranges('def broken(:\n', [(1, 1, [])], level='docstring')

    def test_file_level_propagates_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            get_eligible_ranges('def broken(:\n', [(1, 1, [])], level='file')


class TestParseDiffHunks:
    def test_empty_diff(self) -> None:
        result = parse_diff_hunks('')
        assert result == {}

    def test_hunk_body_paths_do_not_replace_current_file(self) -> None:
        diff = (
            'diff --git a/real.py b/real.py\n'
            '--- a/real.py\n'
            '+++ b/real.py\n'
            '@@ -1 +1,2 @@\n'
            '--- a/decoy.py\n'
            '+++ b/decoy.py\n'
            '@@ -10 +11 @@\n'
            '+changed = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'real.py': [(1, 2, [b'-- a/decoy.py']), (11, 11, [])]}

    def test_hunk_body_paths_do_not_replace_multiple_files(self) -> None:
        diff = (
            'diff --git a/real.py b/real.py\n'
            '--- a/real.py\n'
            '+++ b/real.py\n'
            '@@ -1 +1,2 @@\n'
            '--- a/decoy.py\n'
            '+++ b/decoy.py\n'
            '@@ -10 +11 @@\n'
            '+changed = True\n'
            'diff --git a/second.py b/second.py\n'
            '--- a/second.py\n'
            '+++ b/second.py\n'
            '@@ -3,0 +4 @@\n'
            '+other = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {
            'real.py': [(1, 2, [b'-- a/decoy.py']), (11, 11, [])],
            'second.py': [(4, 4, [])],
        }

    def test_incomplete_record_does_not_create_or_borrow_file(self) -> None:
        diff = (
            'diff --git a/real.py b/real.py\n'
            '--- a/real.py\n'
            '+++ b/real.py\n'
            '@@ -1 +1 @@\n'
            '+changed = True\n'
            'diff --git a/missing.py b/fabricated.py\n'
            '+++ b/fabricated.py\n'
            '@@ -4,0 +5 @@\n'
            '+other = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'real.py': [(1, 1, [])]}

    def test_incomplete_record_hunk_blocks_deceptive_header_pair(self) -> None:
        diff = (
            'diff --git a/real.py b/real.py\n'
            '--- a/real.py\n'
            '+++ b/real.py\n'
            '@@ -1 +1 @@\n'
            '+changed = True\n'
            'diff --git a/missing.py b/missing.py\n'
            '@@ -4,0 +5,2 @@\n'
            '--- a/decoy.py\n'
            '+++ b/decoy.py\n'
            '@@ -10 +11 @@\n'
            '+other = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'real.py': [(1, 1, [])]}

    def test_leading_deletion_retains_empty_interval(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -1 +0,0 @@\n'
            '-obsolete = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(1, 0, [b'obsolete = True'])]}

    def test_modification(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -5,2 +5,3 @@ class Foo:\n'
            '-    x = 1\n'
            '-    y = 2\n'
            '+    x = 10\n'
            '+    y = 20\n'
            '+    z = 30\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(5, 7, [b'    x = 1', b'    y = 2'])]}

    def test_multi_line_addition(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -4,0 +5,3 @@ def foo():\n'
            '+    x = 1\n'
            '+    y = 2\n'
            '+    z = 3\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(5, 7, [])]}

    def test_multiple_files(self) -> None:
        diff = (
            'diff --git a/a.py b/a.py\n'
            'index abc..def 100644\n'
            '--- a/a.py\n'
            '+++ b/a.py\n'
            '@@ -1,0 +2 @@\n'
            '+    x = 1\n'
            'diff --git a/b.py b/b.py\n'
            'index abc..def 100644\n'
            '--- a/b.py\n'
            '+++ b/b.py\n'
            '@@ -3,0 +4 @@\n'
            '+    y = 2\n'
        )
        result = parse_diff_hunks(diff)
        assert result is not None
        assert 'a.py' in result
        assert 'b.py' in result

    def test_multiple_hunks_same_file(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -2,0 +3 @@\n'
            '+    pass\n'
            '@@ -10,0 +12 @@\n'
            '+    return x\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(3, 3, []), (12, 12, [])]}

    def test_new_file(self) -> None:
        diff = (
            'diff --git a/new.py b/new.py\n'
            'new file mode 100644\n'
            'index 0000000..abc 100644\n'
            '--- /dev/null\n'
            '+++ b/new.py\n'
            '@@ -0,0 +1,3 @@\n'
            '+def foo():\n'
            '+    """Doc."""\n'
            '+    pass\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'new.py': [(1, 3, [])]}

    def test_path_with_spaces(self) -> None:
        diff = (
            'diff --git a/path with spaces.py b/path with spaces.py\n'
            'index abc..def 100644\n'
            '--- a/path with spaces.py\n'
            '+++ b/path with spaces.py\n'
            '@@ -4,0 +5 @@\n'
            '+changed = True\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'path with spaces.py': [(5, 5, [])]}

    def test_pure_deletion(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -5,2 +4,0 @@ def foo():\n'
            '-    x = 1\n'
            '-    y = 2\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(5, 4, [b'    x = 1', b'    y = 2'])]}

    def test_rename_with_changes(self) -> None:
        diff = (
            'diff --git a/old.py b/new.py\n'
            'similarity index 90%\n'
            'rename from old.py\n'
            'rename to new.py\n'
            'index abc..def 100644\n'
            '--- a/old.py\n'
            '+++ b/new.py\n'
            '@@ -5,0 +5,2 @@\n'
            '+    x = 1\n'
            '+    y = 2\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'new.py': [(5, 6, [])]}

    def test_single_addition(self) -> None:
        diff = (
            'diff --git a/foo.py b/foo.py\n'
            'index abc..def 100644\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -4,0 +5 @@ def foo():\n'
            '+    x = 1\n'
        )
        result = parse_diff_hunks(diff)
        assert result == {'foo.py': [(5, 5, [])]}
