from typing import ClassVar, cast

import pytest

from docwrap._core._codeblock import (
    find_code_block_lines,
)


class TestAlignedRows:
    @pytest.mark.parametrize('indent', ['', '    '])
    def test_rows_and_description_continuations(self, indent: str) -> None:
        lines = [
            '',
            indent + '`scalar`   a number',
            indent + '`mapping`  a rollup',
            indent + '           with nested values',
            indent + '`error`    a failure',
            'Following prose.',
        ]
        assert find_code_block_lines(lines) == {1, 2, 3, 4}

    def test_continuation_before_second_row_and_blank_boundary(self) -> None:
        lines = [
            '`scalar`   a number',
            '           continued',
            '`mapping`  a rollup',
            '',
            'Ordinary prose.',
        ]
        assert find_code_block_lines(lines) == {0, 1, 2}

    @pytest.mark.parametrize(
        'lines',
        [
            ['`scalar`  a number'],
            ['`scalar`  a number', '`mapping`  a rollup'],
            ['`scalar` a number', '`otherx` a rollup'],
            ['ordinary  prose continues', 'anotherx  sentence follows'],
            ['Prose starts with', '`scalar`   a number', '`mapping`  a rollup'],
        ],
    )
    def test_ambiguous_prose_is_not_protected(self, lines: list[str]) -> None:
        assert find_code_block_lines(lines) == set()


class TestDoctestBlocks:
    def test_doctest_with_continuation_and_output(self) -> None:
        lines = ['>>> for i in range(3):', '...     print(i)', '0', '1', '2', '', 'More.']
        # Commands + expected output until blank line
        assert find_code_block_lines(lines) == {0, 1, 2, 3, 4}

    def test_doctest_with_output(self) -> None:
        # Blank line separates output from subsequent prose (standard doctest format)
        lines = ['Text.', '>>> x = 1', '>>> x + 1', '2', '', 'More.']
        # Expected output '2' must also be protected; 'More.' is prose
        assert find_code_block_lines(lines) == {1, 2, 3}

    def test_indent_0_doctest_output_ends_at_blank(self) -> None:
        lines = ['>>> 1 + 1', '2', 'Not output but no blank separator.']
        # All three lines are protected (line 2 has no blank before it)
        assert find_code_block_lines(lines) == {0, 1, 2}

    def test_indented_doctest_with_output(self) -> None:
        lines = ['Text.', '    >>> x = 1', '    >>> x + 1', '    2', 'More.']
        assert find_code_block_lines(lines) == {1, 2, 3}


class TestFencedBlocks:
    def test_backtick_fence(self) -> None:
        lines = ['Text.', '```', 'code()', '```', 'More text.']
        assert find_code_block_lines(lines) == {1, 2, 3}

    def test_fence_with_language(self) -> None:
        lines = ['Text.', '```python', 'code()', '```', 'More.']
        assert find_code_block_lines(lines) == {1, 2, 3}

    def test_indented_fence(self) -> None:
        lines = ['Text.', '    ```', '    code()', '    ```', 'More.']
        assert find_code_block_lines(lines) == {1, 2, 3}

    def test_longer_closing_fence(self) -> None:
        lines = ['Text.', '```python', 'code()', '`````', 'More.']
        assert find_code_block_lines(lines) == {1, 2, 3}

    def test_short_closer_does_not_close_long_opener(self) -> None:
        # 5-char opener requires at least 5-char closer
        lines = ['`````python', 'code()', '```', 'still code', '`````', 'More.']
        assert find_code_block_lines(lines) == {0, 1, 2, 3, 4}

    def test_tilde_fence(self) -> None:
        lines = ['Text.', '', '~~~', 'code()', '~~~', 'More text.']
        assert find_code_block_lines(lines) == {2, 3, 4}

    def test_tilde_longer_closer(self) -> None:
        lines = ['Text.', '', '~~~', 'code()', '~~~~~', 'More.']
        assert find_code_block_lines(lines) == {2, 3, 4}

    @pytest.mark.parametrize(
        ('lines', 'protected'),
        cast(
            'list[tuple[list[str], set[int]]]',
            [
                (['Text.', '', '~~~', 'code()', '~~~', 'After.'], {2, 3, 4}),
                (['~~~python', 'code()', '~~~~', 'After.'], {0, 1, 2}),
                (['Heading', '~~~', 'After.'], set()),
                (['', '~~~python', 'After.'], set()),
                (['Heading', '~~~~', 'After.'], set()),
            ],
        ),
    )
    def test_tilde_requires_matching_closer(
        self,
        lines: list[str],
        protected: set[int],
    ) -> None:
        assert find_code_block_lines(lines) == protected

    def test_unmatched_backtick_fence_protects_through_eof(self) -> None:
        lines = ['Text.', '```', 'code()', 'trailing code']
        assert find_code_block_lines(lines) == {1, 2, 3}


class TestLiteralBlocks:
    def test_double_colon_block(self) -> None:
        lines = ['Example::', '', '    code here', '    more code', '', 'Text.']
        assert find_code_block_lines(lines) == {2, 3}

    def test_double_colon_no_blank_line(self) -> None:
        lines = ['Example::', '    code here', 'Text.']
        assert find_code_block_lines(lines) == set()

    def test_double_colon_with_internal_blank(self) -> None:
        lines = ['Example::', '', '    code', '', '    more code', '', 'Text.']
        # Lines 2, 3, 4 are the code block (including internal blank at 3)
        # Line 5 is a trailing blank before dedent -- not part of the block
        assert find_code_block_lines(lines) == {2, 3, 4}


class TestNoCodeBlocks:
    def test_empty(self) -> None:
        assert find_code_block_lines([]) == set()

    def test_plain_text(self) -> None:
        lines = ['Just some text.', 'More text.']
        assert find_code_block_lines(lines) == set()


class TestSimpleTableProtection:
    _TABLE: ClassVar[list[str]] = [
        'Summary line.',
        '',
        '==========  ================================================',
        'Option      Meaning',
        '==========  ================================================',
        "'r'         open for reading with a very long descriptive cell",
        "'w'         open for writing, truncating the file first",
        '==========  ================================================',
        '',
        'Tail prose.',
    ]

    def test_blank_after_hyphen_separator_keeps_following_rows(self) -> None:
        lines = [
            '=========  ==========================',
            'Character  Meaning',
            "'a'        first row",
            '---------  --------------------------',
            '',
            "'r'        open for reading with a very long descriptive cell",
            '=========  ==========================',
            '',
        ]
        assert set(range(7)) <= find_code_block_lines(lines)

    def test_blank_row_group_included(self) -> None:
        lines = [
            '====  ====',
            'a     b',
            '',
            'c     d',
            '====  ====',
            '',
        ]
        assert set(range(5)) <= find_code_block_lines(lines)

    def test_border_directly_under_text_not_opening(self) -> None:
        lines = ['Heading text', '====  ====', 'prose continues here']
        assert find_code_block_lines(lines) == set()

    def test_hyphen_only_table_protected(self) -> None:
        lines = [
            '---------  --------------------------',
            "'r'        open for reading with a very long descriptive cell",
            '---------  --------------------------',
        ]
        assert find_code_block_lines(lines) == {0, 1, 2}

    def test_hyphen_separator_shape_protected(self) -> None:
        lines = [
            '=========  ==========================',
            'Character  Meaning',
            '---------  --------------------------',
            "'r'        open for reading (default)",
            '=========  ==========================',
        ]
        assert set(range(5)) <= find_code_block_lines(lines)

    def test_table_span_protected(self) -> None:
        protected = find_code_block_lines(self._TABLE)
        assert set(range(2, 8)) <= protected
        assert 0 not in protected
        assert 9 not in protected

    def test_two_tables_do_not_protect_intervening_prose(self) -> None:
        table = ['=====  =====', 'one    two', '=====  =====']
        lines = [*table, '', 'Normal prose between tables.', '', *table]
        assert find_code_block_lines(lines) == {0, 1, 2, 6, 7, 8}

    def test_unclosed_border_unprotected(self) -> None:
        lines = ['====  ====', 'a     b', 'c     d']
        assert find_code_block_lines(lines) == set()

    def test_unclosed_mixed_border_unprotected(self) -> None:
        lines = ['====  ====', 'a     b', '----  ----', '', 'Tail prose.']
        assert find_code_block_lines(lines) == set()


class TestStartingBoundary:
    def test_reused_fence_character_restarts_pairing(self) -> None:
        lines = [
            'Heading',
            '~~~~~~~',
            'body',
            '',
            '~~~~~~~',
            'code',
            '~~~~~~~',
            'after',
        ]
        assert find_code_block_lines(lines, start=2) == {4, 5, 6}

    def test_start_ignores_title_like_tilde_prefix(self) -> None:
        lines = ['Heading', '~~~~~~~', 'body']
        assert find_code_block_lines(lines, start=2) == set()

    def test_start_preserves_absolute_indices(self) -> None:
        lines = ['Heading', '-------', '>>> 1 + 1', '2', '', 'body']
        assert find_code_block_lines(lines, start=2) == {2, 3}

    def test_start_scans_real_fence_after_title(self) -> None:
        lines = ['Heading', '-------', '', '```', 'code()', '```', 'body']
        assert find_code_block_lines(lines, start=2) == {3, 4, 5}


class TestTildeUnderlineContext:
    def test_adjacent_heading_underlines_are_not_a_fence(self) -> None:
        lines = [
            'More',
            '~~~~',
            '',
            'An overlong paragraph that should stay unprotected prose.',
            '',
            'Details',
            '~~~~~~~',
        ]
        assert find_code_block_lines(lines) == set()

    def test_tilde_fence_after_blank_still_protects(self) -> None:
        lines = ['Intro.', '', '~~~~', 'code here', '~~~~', '']
        assert set(range(2, 5)) <= find_code_block_lines(lines)

    def test_tilde_fence_with_info_string_after_text_is_a_fence(self) -> None:
        lines = ['Intro.', '~~~python', 'code here', '~~~']
        assert set(range(1, 4)) <= find_code_block_lines(lines)

    def test_unmatched_tilde_after_blank_stays_unprotected(self) -> None:
        lines = ['Intro.', '', '~~~~', 'text with no closer']
        assert find_code_block_lines(lines) == set()
