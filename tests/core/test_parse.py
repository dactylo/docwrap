import string
import textwrap

import pytest

from docwrap._core._codeblock import find_code_block_lines
from docwrap._core._discover import find_docstrings, parse_docstrings
from docwrap._core._normalize import normalize_content
from docwrap._core._parse import Region, is_structural_underline, parse_regions
from docwrap._core._process import wrap_source, wrap_source_report
from docwrap._detect import detect_style
from docwrap._styles import get_style
from docwrap._styles._google import GoogleStyle
from docwrap._styles._numpy import NumpyStyle
from docwrap._styles._sphinx import (
    FIELD_ROLES,
    _SphinxStyle,
)
from docwrap._types import CanonicalRole, DocwrapConfig, RegionKind, StyleName

GOOGLE_FALLBACK_ANNOTATION_CASES = (
    ('    dict[str, int]: Mapping result.', 'dict[str, int]', 'Mapping result.'),
    ('    list[str]: Items.', 'list[str]', 'Items.'),
    ('    Optional[int]: Optional result.', 'Optional[int]', 'Optional result.'),
    ('    int | None: Optional result.', 'int | None', 'Optional result.'),
    ('    Callable[[int], str]: Callback.', 'Callable[[int], str]', 'Callback.'),
    ('    Literal[1]: Literal result.', 'Literal[1]', 'Literal result.'),
    ('    Literal[1 | 2]: Literal union result.', 'Literal[1 | 2]', 'Literal union result.'),
    ('    "pkg.Type:Alias": Forward reference.', '"pkg.Type:Alias"', 'Forward reference.'),
    ('    ...: Ellipsis result.', '...', 'Ellipsis result.'),
)
GOOGLE_NON_ANNOTATION_LINES = (
    '    value or fallback: explanation',
    '    factory(): explanation',
    '    left + right: explanation',
    '    left < right: explanation',
    '    lambda x: x',
    '    value if ready else fallback: explanation',
    '    [x for x in values]: explanation',
    '    {"key": value}: explanation',
)
GOOGLE_RETURN_SECTIONS: tuple[tuple[str, CanonicalRole], ...] = (
    ('Returns', 'returns'),
    ('Yields', 'yields'),
)
SPHINX_EXCEPTION_ROLES = frozenset({'except', 'raise', 'raises'})
SPHINX_NAMED_ROLES = frozenset({'cvar', 'ivar', 'keyword', 'param', 'type', 'var', 'vartype'})


class TestCodeBlockPreservation:
    def test_four_space_indented_one_line_is_code_block(self) -> None:
        content = '    result = compute_value()'

        regions = parse_regions(content, style=None, code_block_lines=set())

        assert [(region.kind, region.text) for region in regions] == [('code_block', content)]

    def test_indented_block_in_description(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Usage:
                ./tool --flag""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        code_regions = [r for r in regions if r.kind == 'code_block']
        assert len(code_regions) >= 1

    @pytest.mark.parametrize('indent', [1, 2, 3])
    def test_shallow_indented_one_line_artifact_is_summary(self, indent: int) -> None:
        content = ' ' * indent + 'Compact summary.'

        regions = parse_regions(content, style=None, code_block_lines=set())

        assert [(region.kind, region.text) for region in regions] == [
            ('summary', 'Compact summary.')
        ]


class TestGoogleEntrySections:
    def setup_method(self) -> None:
        self.style = GoogleStyle()

    def test_args_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                name (str): The name.
                count: The count.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 2

    def test_entry_with_continuation(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                name (str): A very long description that
                    continues on the next line.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 1
        assert 'continues' in entries[0].text

    @pytest.mark.parametrize(('section', 'role'), GOOGLE_RETURN_SECTIONS)
    @pytest.mark.parametrize(
        ('line', 'annotation', 'description'),
        GOOGLE_FALLBACK_ANNOTATION_CASES,
    )
    def test_fallback_return_annotation(
        self,
        section: str,
        role: CanonicalRole,
        line: str,
        annotation: str,
        description: str,
    ) -> None:
        regions = parse_regions(
            f'Summary.\n\n{section}:\n{line}',
            style=self.style,
            code_block_lines=set(),
        )
        entries = [region for region in regions if region.kind == 'entry']
        assert len(entries) == 1
        entry = entries[0]
        assert entry.role == role
        assert entry.original_lines == (line,)
        assert entry.entry_match is not None
        assert entry.entry_match.name == ''
        assert entry.entry_match.type_ann == annotation
        assert entry.entry_match.description == description

    @pytest.mark.parametrize(('section', 'role'), GOOGLE_RETURN_SECTIONS)
    @pytest.mark.parametrize('line', GOOGLE_NON_ANNOTATION_LINES)
    def test_non_annotation_expression_remains_prose(
        self,
        section: str,
        role: CanonicalRole,
        line: str,
    ) -> None:
        regions = parse_regions(
            f'Summary.\n\n{section}:\n{line}',
            style=self.style,
            code_block_lines=set(),
        )
        assert not [region for region in regions if region.kind == 'entry']
        prose = [region for region in regions if region.kind == 'prose']
        assert len(prose) == 1
        assert prose[0].role == ''
        assert prose[0].original_lines == (line,)
        assert self.style.match_entry(line, role=role) is None

    @pytest.mark.parametrize(('section', 'role'), GOOGLE_RETURN_SECTIONS)
    def test_none_uses_normal_name_as_type_representation(
        self,
        section: str,
        role: CanonicalRole,
    ) -> None:
        line = '    None: No result.'
        regions = parse_regions(
            f'Summary.\n\n{section}:\n{line}',
            style=self.style,
            code_block_lines=set(),
        )
        entries = [region for region in regions if region.kind == 'entry']
        assert len(entries) == 1
        entry = entries[0]
        assert entry.role == role
        assert entry.entry_match is not None
        assert entry.entry_match.name == 'None'
        assert entry.entry_match.type_ann == ''
        assert entry.entry_match.description == 'No result.'

    def test_raises_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Raises:
                ValueError: If bad.
                TypeError: If wrong type.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 2


class TestGoogleMultipleSections:
    def setup_method(self) -> None:
        self.style = GoogleStyle()

    def test_args_and_returns(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                x: The x.

            Returns:
                The result.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        headers = [r for r in regions if r.kind == 'section_header']
        assert len(headers) == 2


class TestGoogleProseSections:
    def setup_method(self) -> None:
        self.style = GoogleStyle()

    def test_note_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Note:
                This is important.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        prose = [r for r in regions if r.kind == 'prose']
        assert len(prose) == 1

    def test_returns_prose(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Returns:
                The result value.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        prose = [r for r in regions if r.kind == 'prose']
        assert len(prose) == 1


class TestIsStructuralUnderline:
    @pytest.mark.parametrize(
        'line',
        [
            '-' * 120,
            '==========',
            '~~~~~~~~~~',
            '---',
            '    ----------',
            '----------   ',
        ],
    )
    def test_recognizes_adornment_runs(self, line: str) -> None:
        assert is_structural_underline(line) is True

    @pytest.mark.parametrize(
        'line',
        [
            '',
            '   ',
            '--',
            '-',
            '-=-=-=',
            '--- text',
            'text ---',
            '^^^^^^',
            '******',
            '######',
            '++++++',
            'Parameters',
            'x',
        ],
    )
    def test_rejects_non_underlines(self, line: str) -> None:
        assert is_structural_underline(line) is False


class TestListAsFirstContent:
    def test_list_only_docstring_preserves_structure(self) -> None:
        content = '- first\n- second'
        regions = parse_regions(content, style=None, code_block_lines=set())
        # List items should NOT be flattened into summary
        kinds = [r.kind for r in regions]
        assert 'list_item' in kinds
        # If a summary exists, it should not contain both items
        summaries = [r for r in regions if r.kind == 'summary']
        if summaries:
            assert '- second' not in summaries[0].text

    def test_numbered_list_only(self) -> None:
        content = '1. first\n2. second'
        regions = parse_regions(content, style=None, code_block_lines=set())
        kinds = [r.kind for r in regions]
        assert 'list_item' in kinds

    def test_summary_blank_then_list(self) -> None:
        content = 'Summary.\n\n- first\n- second'
        regions = parse_regions(content, style=None, code_block_lines=set())
        assert regions[0].kind == 'summary'
        assert regions[0].text == 'Summary.'
        list_regions = [r for r in regions if r.kind == 'list_item']
        assert len(list_regions) == 2

    def test_summary_then_list(self) -> None:
        content = 'Summary line.\n- first item\n- second item'
        regions = parse_regions(content, style=None, code_block_lines=set())
        assert regions[0].kind == 'summary'
        assert regions[0].text == 'Summary line.'
        list_regions = [r for r in regions if r.kind == 'list_item']
        assert len(list_regions) == 2


class TestListDetection:
    def test_bullet_list(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            - First item.
            - Second item.""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        list_regions = [r for r in regions if r.kind == 'list_item']
        assert len(list_regions) == 2

    def test_numbered_list(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            1. First item.
            2. Second item.""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        list_regions = [r for r in regions if r.kind == 'list_item']
        assert len(list_regions) == 2


class TestNoSummary:
    def test_code_block_without_summary(self) -> None:
        content = '```\ncode()\n```'
        regions = parse_regions(content, style=None, code_block_lines={0, 1, 2})
        assert regions[0].kind == 'code_block'

    def test_doctest_without_summary(self) -> None:
        content = '>>> 1 + 1\n2'
        regions = parse_regions(content, style=None, code_block_lines={0, 1})
        assert regions[0].kind == 'code_block'

    def test_section_header_without_summary(self) -> None:
        style = GoogleStyle()
        content = 'Args:\n    x: val.'
        regions = parse_regions(content, style=style, code_block_lines=set())
        assert regions[0].kind != 'blank'


class TestNumpyCompoundTypeEntries:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_compound_type_with_multiple_entries(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Raises
            ------
            ValueError
                If bad input.
            TypeError or KeyError
                If wrong type.
        """)
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entry_regions = [r for r in regions if r.kind == 'entry']
        assert len(entry_regions) == 2

    def test_returns_bare_type_parsed_as_entry(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Returns
            -------
            bool
                True if valid.
        """)
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entry_regions = [r for r in regions if r.kind == 'entry']
        assert len(entry_regions) == 1
        assert 'bool' in entry_regions[0].text

    def test_returns_compound_type_parsed_as_entry(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Returns
            -------
            int or None
                The result value.
        """)
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entry_regions = [r for r in regions if r.kind == 'entry']
        assert len(entry_regions) == 1
        assert 'int or None' in entry_regions[0].text


class TestNumpyEntrySections:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_dashed_underline_consumed(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            x : int
                The x.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        headers = [r for r in regions if r.kind == 'section_header']
        # Both "Parameters" and "----------" should be section_header regions
        assert len(headers) == 2

    def test_entry_with_multiline_desc(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            name : str
                A very long description that wraps
                across multiple lines.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 1
        assert 'wraps' in entries[0].text

    def test_parameters_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            name : str
                The name.
            count : int
                The count.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 2


class TestOpeningTitle:
    def test_adornment_shorter_than_title_does_not_match(self) -> None:
        regions = parse_regions(
            'Title\n----',
            style=None,
            code_block_lines=set(),
        )
        assert all(region.kind != 'title' for region in regions)

    def test_alphanumeric_rule_does_not_match(self) -> None:
        regions = parse_regions(
            'Title\nAAAAA',
            style=None,
            code_block_lines=set(),
        )
        assert all(region.kind != 'title' for region in regions)

    def test_conflicting_style_accepts_title(self) -> None:
        regions = parse_regions(
            'Parameters\n----------',
            style=get_style('sphinx'),
            code_block_lines=set(),
        )
        assert regions[0].kind == 'title'

    @pytest.mark.parametrize('adornment', string.punctuation)
    def test_every_ascii_adornment_matches_at_parser_boundary(self, adornment: str) -> None:
        regions = parse_regions(
            f'A\n{adornment}',
            style=None,
            code_block_lines=set(),
        )
        assert regions[0].kind == 'title'

    def test_google_header_wins_under_google_style(self) -> None:
        regions = parse_regions(
            'Args:\n-----',
            style=get_style('google'),
            code_block_lines=set(),
        )
        assert regions[0].kind == 'section_header'
        assert all(region.kind != 'title' for region in regions)

    def test_indented_underline_form_does_not_match(self) -> None:
        regions = parse_regions(
            '  Title\n-------',
            style=None,
            code_block_lines=set(),
        )
        assert all(region.kind != 'title' for region in regions)

    def test_mismatched_overline_and_underline_do_not_match(self) -> None:
        regions = parse_regions(
            '=======\nTitle\n-------',
            style=None,
            code_block_lines=set(),
        )
        assert all(region.kind != 'title' for region in regions)

    def test_mixed_adornment_does_not_match(self) -> None:
        regions = parse_regions(
            'Title\n-=-=-',
            style=None,
            code_block_lines=set(),
        )
        assert all(region.kind != 'title' for region in regions)

    def test_numpy_header_wins_under_numpy_style(self) -> None:
        regions = parse_regions(
            'Parameters\n----------',
            style=get_style('numpy'),
            code_block_lines=set(),
        )
        assert regions[0].kind == 'section_header'
        assert all(region.kind != 'title' for region in regions)

    def test_one_character_title_and_adornment_match(self) -> None:
        regions = parse_regions(
            'A\n=',
            style=None,
            code_block_lines=set(),
        )
        assert regions[0].kind == 'title'

    def test_ordinary_summary_is_unchanged(self) -> None:
        regions = parse_regions(
            'A summary\ncontinued here.\n\nBody.',
            style=None,
            code_block_lines=set(),
        )
        assert [(region.kind, region.text) for region in regions] == [
            ('summary', 'A summary continued here.'),
            ('blank', ''),
            ('description', 'Body.'),
        ]

    def test_overline_and_underline_match(self) -> None:
        regions = parse_regions(
            '=======\n  Title\n=======',
            style=None,
            code_block_lines=set(),
        )
        assert regions[0].kind == 'title'
        assert regions[0].source_line_count == 3

    def test_overlined_title_wins_under_every_style(self) -> None:
        for style_name in ('google', 'numpy', 'sphinx'):
            regions = parse_regions(
                '-----\nArgs:\n-----',
                style=get_style(style_name),
                code_block_lines=set(),
            )
            assert regions[0].kind == 'title'

    def test_parser_emits_exact_title_then_body(self) -> None:
        regions = parse_regions(
            'Heading\n-------\nbody',
            style=None,
            code_block_lines=set(),
        )
        assert [
            (
                region.kind,
                region.text,
                region.original_lines,
                region.source_line_count,
            )
            for region in regions
        ] == [
            (
                'title',
                'Heading\n-------',
                ('Heading', '-------'),
                2,
            ),
            ('description', 'body', ('body',), 1),
        ]

    @pytest.mark.parametrize(
        ('separator', 'expected_kinds'),
        [
            ('', ['title', 'description']),
            ('\n', ['title', 'blank', 'description']),
            ('\n\n', ['title', 'blank', 'blank', 'description']),
        ],
    )
    def test_parser_preserves_authored_blank_count(
        self,
        separator: str,
        expected_kinds: list[RegionKind],
    ) -> None:
        regions = parse_regions(
            f'Heading\n-------\n{separator}body',
            style=None,
            code_block_lines=set(),
        )
        assert [region.kind for region in regions] == expected_kinds

    def test_parser_preserves_inset_overlined_title(self) -> None:
        regions = parse_regions(
            '=======\n  Title\n=======',
            style=None,
            code_block_lines=set(),
        )
        assert regions == [
            Region(
                kind='title',
                text='=======\n  Title\n=======',
                source_line_count=3,
                original_lines=('=======', '  Title', '======='),
                source_start=0,
            )
        ]


class TestOriginalLines:
    def test_all_authored_regions_preserve_original_lines(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Args:
                    x: Value.
                \"\"\"
        """)
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        assert [(region.kind, region.original_lines) for region in results[0].regions] == [
            ('summary', ('Summary.',)),
            ('blank', ('',)),
            ('section_header', ('Args:',)),
            ('entry', ('    x: Value.',)),
        ]

    def test_entry_preserves_original_lines(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x: First line of description
                        continuation of description.
                \"\"\"
        """)
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        entry_regions = [r for r in results[0].regions if r.kind == 'entry']
        assert len(entry_regions) == 1
        assert entry_regions[0].original_lines == (
            '    x: First line of description',
            '        continuation of description.',
        )

    def test_list_item_preserves_original_lines(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                - First line of item
                  continuation of item.
                \"\"\"
        """)
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        list_regions = [r for r in results[0].regions if r.kind == 'list_item']
        assert len(list_regions) == 1
        assert list_regions[0].original_lines == (
            '- First line of item',
            '  continuation of item.',
        )

    def test_paragraph_preserves_original_lines(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                First line of description.
                Second line of description.
                \"\"\"
        """)
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        desc_regions = [r for r in results[0].regions if r.kind == 'description']
        assert len(desc_regions) == 1
        original = desc_regions[0].original_lines
        assert original is not None
        assert original == (
            'First line of description.',
            'Second line of description.',
        )
        assert desc_regions[0].source_line_count == len(original)


class TestParagraphMarkerBoundary:
    @pytest.mark.parametrize('marker', ['+', '-', '*', '1.'])
    def test_bare_marker_owns_one_region(self, marker: str) -> None:
        regions = parse_regions(
            f'Summary.\n\nalpha\n{marker}\nbeta',
            style=None,
            code_block_lines=set(),
        )

        assert [
            (
                region.kind,
                region.text,
                region.original_lines,
                region.source_line_count,
            )
            for region in regions[2:]
        ] == [
            ('description', 'alpha', ('alpha',), 1),
            ('description', marker, (marker,), 1),
            ('description', 'beta', ('beta',), 1),
        ]


class TestParagraphUnderlineBoundary:
    def _regions(self, content: str) -> list[Region]:
        norm, _ = normalize_content(content, source_margin='')
        lines = norm.split('\n')
        return parse_regions(norm, style=None, code_block_lines=find_code_block_lines(lines))

    def test_no_underline_leaves_paragraph_joined(self) -> None:
        regions = self._regions('Summary.\n\nalpha\nbeta\ngamma')
        assert 'alpha beta gamma' in [r.text for r in regions]

    def test_paragraph_before_underline_is_preserved(self) -> None:
        regions = self._regions('Summary.\n\nParameters\n----------\nx')
        assert 'Parameters' in [r.text for r in regions]

    def test_underline_alone_in_its_region(self) -> None:
        regions = self._regions('Summary.\n\nParameters\n----------\nx')
        underlines = [r for r in regions if r.text == '----------']
        assert len(underlines) == 1
        assert underlines[0].original_lines == ('----------',)

    def test_underline_not_joined_into_paragraph(self) -> None:
        regions = self._regions('Summary.\n\nParameters\n----------\nx')
        texts = [r.text for r in regions if r.kind in ('description', 'prose')]
        assert 'Parameters ---------- x' not in texts
        assert 'Parameters' in texts
        assert '----------' in texts

    def test_unmatched_tilde_underline_does_not_protect_later_prose(self) -> None:
        regions = self._regions('Summary.\n\nHeading\n~~~\nAfter.')
        assert [(region.kind, region.text) for region in regions[2:]] == [
            ('description', 'Heading'),
            ('description', '~~~'),
            ('description', 'After.'),
        ]
        assert is_structural_underline(regions[3].text) is True


class TestRegionEnrichment:
    def test_entry_has_entry_match(self) -> None:
        style = get_style('google')
        regions = parse_regions(
            'Summary.\n\nArgs:\n    x (int): The value.\n        More detail.',
            style=style,
            code_block_lines=set(),
        )
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 1
        assert entries[0].entry_match is not None
        assert entries[0].entry_match.name == 'x'
        assert entries[0].entry_match.type_ann == 'int'
        assert 'More detail' in entries[0].entry_match.description

    def test_entry_has_role_from_section(self) -> None:
        style = get_style('google')
        regions = parse_regions(
            'Summary.\n\nArgs:\n    x: val', style=style, code_block_lines=set()
        )
        entries = [r for r in regions if r.kind == 'entry']
        assert entries[0].role == 'param'

    def test_every_authored_region_retains_its_exact_source_span(self) -> None:
        content = (
            'Summary.\n'
            '\n'
            'Parameters\n'
            '----------\n'
            'x : int\n'
            '    First detail.\n'
            '    More detail.\n'
            '\n'
            'Notes\n'
            '-----\n'
            '    Prose detail.\n'
            '    - list detail\n'
            '    ======\n'
            '    code()'
        )
        content_lines = content.split('\n')
        code_line = len(content_lines) - 1
        regions = parse_regions(
            content,
            style=get_style('numpy'),
            code_block_lines={code_line},
        )

        assert {
            'blank',
            'code_block',
            'entry',
            'list_item',
            'prose',
            'section_header',
            'summary',
        } <= {region.kind for region in regions}
        assert any(is_structural_underline(region.text) for region in regions)
        for region in regions:
            assert region.source_start is not None
            assert region.source_line_count == len(region.original_lines or ())
            assert region.original_lines == tuple(
                content_lines[region.source_start : region.source_start + region.source_line_count]
            )
            assert region.preserve_verbatim is False

    def test_numpy_compound_type_entry_match_is_none(self) -> None:
        style = get_style('numpy')
        regions = parse_regions(
            'Summary.\n\nReturns\n-------\nint or None\n    The result.',
            style=style,
            code_block_lines=set(),
        )
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 1
        assert entries[0].entry_match is None

    def test_section_header_has_section_name(self) -> None:
        style = get_style('google')
        regions = parse_regions(
            'Summary.\n\nArgs:\n    x: val', style=style, code_block_lines=set()
        )
        headers = [r for r in regions if r.kind == 'section_header']
        assert len(headers) == 1
        assert headers[0].section_name == 'Args'

    def test_sphinx_entry_has_role_from_match(self) -> None:
        style = get_style('sphinx')
        regions = parse_regions('Summary.\n\n:param x: val', style=style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert entries[0].role == 'param'


class TestScannerProvenance:
    def test_generated_region_uses_unprotected_default(self) -> None:
        region = Region(kind='code_block', text='generated()')
        assert region.source_start is None
        assert region.source_scanner_protected is False

    def test_scanner_and_parser_owned_code_blocks_are_distinct(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Usage:
                parser_owned()

            ```
            scanner_owned()
            ```""")
        lines = content.splitlines()
        code_block_lines = find_code_block_lines(lines)
        regions = parse_regions(
            content,
            style=None,
            code_block_lines=code_block_lines,
        )

        parser_owned = next(region for region in regions if region.text == '    parser_owned()')
        scanner_owned = next(region for region in regions if region.text == 'scanner_owned()')
        assert parser_owned.kind == 'code_block'
        assert scanner_owned.kind == 'code_block'
        assert parser_owned.source_scanner_protected is False
        assert scanner_owned.source_scanner_protected is True

        for region in regions:
            assert region.source_start is not None
            assert region.source_scanner_protected is (region.source_start in code_block_lines)


class TestSectionExit:
    def test_google_indent0_after_blank_exits_section(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: A value.\n'
            '\n'
            '    Some note at base indent.\n'
            '    """\n'
        )
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        kinds = [r.kind for r in results[0].regions]
        # The note should be 'description', not 'prose'
        assert 'description' in kinds
        assert kinds[-1] == 'description'

    def test_google_indent4_after_blank_stays_in_section(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: A value.\n'
            '\n'
            '        Note about x continued.\n'
            '    """\n'
        )
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        # After normalization, "Note about x continued." is at indent 4 (relative
        # to body). That is >= section_body_indent (4), so it stays as prose.
        prose_regions = [r for r in results[0].regions if r.kind == 'prose']
        assert len(prose_regions) >= 1

    def test_numpy_indent0_entries_not_broken(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Summary.\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    x\n'
            '        A value.\n'
            '\n'
            '    y\n'
            '        Another value.\n'
            '    """\n'
        )
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig())
        entry_regions = [r for r in results[0].regions if r.kind == 'entry']
        # Both x and y should be entries (not descriptions)
        assert len(entry_regions) == 2


class TestSectionExitWrapOutput:
    def test_indent0_after_args_stays_at_indent0(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: A value.\n'
            '\n'
            '    /c/Users/... -> C:/Users/...\n'
            '    /d/ -> D:/\n'
            '    """\n'
        )
        result = wrap_source(source)
        # These lines must NOT be re-indented into the section
        assert '    /c/Users/... -> C:/Users/...\n' in result
        assert '    /d/ -> D:/\n' in result
        # They must NOT appear at entry indent (8 spaces)
        assert '        /c/Users' not in result

    def test_indent0_returns_note_after_args_stays_at_indent0(self) -> None:
        source = (
            'def foo(x):\n'
            '    """Do something.\n'
            '\n'
            '    Args:\n'
            '        x: A value.\n'
            '\n'
            '    Returns True if successful.\n'
            '    """\n'
        )
        result = wrap_source(source)
        assert '    Returns True if successful.\n' in result
        assert '        Returns True' not in result


class TestSphinxFieldLists:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    @pytest.mark.parametrize('raw_role', sorted(FIELD_ROLES))
    def test_deeper_field_roles_remain_continuations(self, raw_role: str) -> None:
        field_line = _sphinx_field_line(raw_role)
        content = f'Summary.\n\n:param value: Lead.\n {field_line}'
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [region for region in regions if region.kind == 'entry']

        assert len(entries) == 1
        entry = entries[0]
        assert entry.role == 'param'
        assert entry.entry_match is not None
        assert entry.entry_match.description == f'Lead. {field_line}'
        assert entry.original_lines == (':param value: Lead.', f' {field_line}')
        assert entry.source_start == 2
        assert entry.source_line_count == 2

    def test_field_list_with_description_before(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Some description text.

            :param x: The x.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        desc = [r for r in regions if r.kind == 'description']
        entries = [r for r in regions if r.kind == 'entry']
        assert len(desc) == 1
        assert len(entries) == 1

    def test_field_with_continuation(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            :param name: A very long description that
                continues on the next line.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 1
        assert 'continues' in entries[0].text

    def test_indented_fields_not_merged(self) -> None:
        content = 'Summary.\n\n    :param name: The name.\n    :param count: The count.'
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 2

    def test_mixed_fields(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            :param x: The x.
            :type x: int
            :returns: The result.
            :raises ValueError: If bad.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 4

    @pytest.mark.parametrize(
        ('active_prefix', 'candidate_prefix'),
        [
            ('\t', '    '),
            ('    ', '    '),
            ('    ', ''),
        ],
    )
    def test_non_deeper_field_prefix_starts_sibling(
        self,
        active_prefix: str,
        candidate_prefix: str,
    ) -> None:
        first_line = f'{active_prefix}:param value: Lead.'
        second_line = f'{candidate_prefix}:rtype: int'
        content = f'Summary.\n\n{first_line}\n{second_line}'
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [region for region in regions if region.kind == 'entry']

        assert [entry.role for entry in entries] == ['param', 'rtype']
        assert [entry.original_lines for entry in entries] == [(first_line,), (second_line,)]
        assert [entry.source_start for entry in entries] == [2, 3]
        assert [entry.source_line_count for entry in entries] == [1, 1]

    def test_param_fields(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            :param name: The name.
            :param count: The count.""")
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [r for r in regions if r.kind == 'entry']
        assert len(entries) == 2

    def test_tab_prefix_extended_by_space_remains_continuation(self) -> None:
        content = 'Summary.\n\n\t:param value: Lead.\n\t :rtype:.'
        regions = parse_regions(content, style=self.style, code_block_lines=set())
        entries = [region for region in regions if region.kind == 'entry']

        assert len(entries) == 1
        assert entries[0].role == 'param'
        assert entries[0].original_lines == (
            '\t:param value: Lead.',
            '\t :rtype:.',
        )
        assert entries[0].source_start == 2
        assert entries[0].source_line_count == 2


class TestSummaryAndDescription:
    def test_d205_inserts_blank_before_section(self) -> None:
        style = GoogleStyle()
        content = 'Summary.\nArgs:\n    x: val.'
        regions = parse_regions(content, style=style, code_block_lines=set())
        kinds = [r.kind for r in regions]
        # D205: blank must be inserted between summary and Args section
        assert kinds[0] == 'summary'
        assert kinds[1] == 'blank'

    def test_d205_normalizes_multiple_blanks(self) -> None:
        content = textwrap.dedent("""\
            Summary.


            Description.""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        kinds = [r.kind for r in regions]
        assert kinds == ['summary', 'blank', 'description']

    def test_multi_paragraph_description(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            First paragraph.

            Second paragraph.""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        kinds = [r.kind for r in regions]
        assert kinds == ['summary', 'blank', 'description', 'blank', 'description']

    def test_with_blank_separator(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Description paragraph.""")
        regions = parse_regions(content, style=None, code_block_lines=set())
        kinds = [r.kind for r in regions]
        assert kinds == ['summary', 'blank', 'description']
        assert regions[2].text == 'Description paragraph.'


class TestSummaryNotFalsePositive:
    def test_numpy_bare_word_summary(self) -> None:
        style = NumpyStyle()
        content = 'Summary.\n\nParameters\n----------\nx : int\n    A param.'
        regions = parse_regions(content, style=style, code_block_lines=set())
        assert regions[0].kind == 'summary'
        assert regions[0].text == 'Summary.'


class TestSummaryOnly:
    def test_multi_line_summary(self) -> None:
        content = 'A long summary that was\nwrapped across two lines.'
        regions = parse_regions(content, style=None, code_block_lines=set())
        assert len(regions) == 1
        assert regions[0].kind == 'summary'
        assert regions[0].text == 'A long summary that was wrapped across two lines.'

    def test_single_line(self) -> None:
        regions = parse_regions('Short summary.', style=None, code_block_lines=set())
        assert len(regions) == 1
        assert regions[0].kind == 'summary'
        assert regions[0].text == 'Short summary.'


class TestUnderlineBoundaryRiskValidation:
    MIXED = (
        'def f(x):\n'
        '    """Summary.\n'
        '\n'
        '    Parameters\n'
        '    ----------\n'
        '    x\n'
        '        Kept parameter.\n'
        '    :type orphan: int\n'
        '    """\n'
    )

    def test_body_is_split_into_exact_five_region_shape(self) -> None:
        content = 'Summary.\n\nParameters\n----------\nx'
        norm, _ = normalize_content(content, source_margin='')
        lines = norm.split('\n')
        regions = parse_regions(norm, style=None, code_block_lines=find_code_block_lines(lines))
        assert len(regions) == 5
        assert [r.text for r in regions[2:]] == ['Parameters', '----------', 'x']

    def test_conversion_provenance_reports_no_spurious_skip(self) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Before.\n'
            '    ----------\n'
            '    After.\n'
            '\n'
            '    :param x: Kept.\n'
            '    :returns: Done.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert report.converted == 1
        assert '    ----------\n' in report.source
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source

    def test_escape_spellings_preserved_across_boundary(self) -> None:
        content = (
            'Summary.\n\nParameters\n----------\nUses `' + chr(92) + 'n` here.\n:type orphan: int'
        )
        assert detect_style(content) is None
        norm, _ = normalize_content(content, source_margin='')
        lines = norm.split('\n')
        regions = parse_regions(norm, style=None, code_block_lines=find_code_block_lines(lines))
        assert [(region.kind, region.text, region.original_lines) for region in regions[2:]] == [
            ('description', 'Parameters', ('Parameters',)),
            ('description', '----------', ('----------',)),
            (
                'description',
                'Uses `' + chr(92) + 'n` here. :type orphan: int',
                ('Uses `' + chr(92) + 'n` here.', ':type orphan: int'),
            ),
        ]
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    Uses `' + chr(92) + 'n` here.\n'
            '    :type orphan: int\n'
            '    """\n'
        )
        config = DocwrapConfig()
        report = wrap_source_report(source, config=config)
        assert chr(92) + 'n' in report.source
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source

    def test_no_content_lost_across_the_boundary(self) -> None:
        report = wrap_source_report(self.MIXED, config=DocwrapConfig())
        for token in ('Parameters', '----------', 'Kept parameter.', ':type orphan: int'):
            assert token in report.source

    def test_no_region_text_spans_an_underline(self) -> None:
        content = 'Summary.\n\nParameters\n----------\nx'
        norm, _ = normalize_content(content, source_margin='')
        lines = norm.split('\n')
        regions = parse_regions(norm, style=None, code_block_lines=find_code_block_lines(lines))
        for region in regions:
            if any(is_structural_underline(part) for part in region.text.split()):
                assert is_structural_underline(region.text)


def _sphinx_field_line(raw_role: str) -> str:
    if raw_role in SPHINX_NAMED_ROLES:
        return f':{raw_role} value: payload'
    if raw_role in SPHINX_EXCEPTION_ROLES:
        return f':{raw_role} ValueError: payload'
    return f':{raw_role}: payload'


@pytest.mark.parametrize(
    ('style_name', 'content', 'expected_lines'),
    [
        (
            'google',
            'Summary.\n\nArgs:\n    x: First.\n    y: Second.',
            (('    x: First.',), ('    y: Second.',)),
        ),
        (
            'numpy',
            'Summary.\n\nParameters\n----------\nx : int\n    First.\ny : str\n    Second.',
            (
                ('x : int', '    First.'),
                ('y : str', '    Second.'),
            ),
        ),
    ],
)
def test_non_field_entry_boundaries_unchanged(
    style_name: StyleName,
    content: str,
    expected_lines: tuple[tuple[str, ...], ...],
) -> None:
    regions = parse_regions(
        content,
        style=get_style(style_name),
        code_block_lines=set(),
    )
    entries = [region for region in regions if region.kind == 'entry']

    assert tuple(entry.original_lines for entry in entries) == expected_lines
