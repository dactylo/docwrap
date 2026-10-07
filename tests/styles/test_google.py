import logging

import pytest

from docwrap._styles import EntryMatch, get_style
from docwrap._styles._google import _compile_entry_re, _GoogleStyle
from docwrap._types import CanonicalRole

FALLBACK_ANNOTATION_CASES = (
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
NON_ANNOTATION_LINES = (
    '    value or fallback: explanation',
    '    factory(): explanation',
    '    left + right: explanation',
    '    left < right: explanation',
    '    lambda x: x',
    '    value if ready else fallback: explanation',
    '    [x for x in values]: explanation',
    '    {"key": value}: explanation',
)
RETURN_HEADERS = ('Returns', 'Yields')
RETURN_ROLES: tuple[CanonicalRole, ...] = ('returns', 'yields')


class TestCanonicalSections:
    def test_google_args(self) -> None:
        assert get_style('google').section_to_canonical['Args'] == 'param'

    def test_google_arguments(self) -> None:
        assert get_style('google').section_to_canonical['Arguments'] == 'param'

    def test_raises_aliases(self) -> None:
        assert get_style('google').section_to_canonical['Raises'] == 'raises'
        assert get_style('google').section_to_canonical['Raise'] == 'raises'
        assert get_style('google').section_to_canonical['Except'] == 'raises'
        assert get_style('google').section_to_canonical['Exceptions'] == 'raises'

    def test_unknown_returns_none(self) -> None:
        assert get_style('google').section_to_canonical.get('UnknownHeader') is None


class TestClassification:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    def test_args_is_entry(self) -> None:
        assert self.style.classify_section('Args') == 'entry'

    def test_note_is_prose(self) -> None:
        assert self.style.classify_section('Note') == 'prose'

    @pytest.mark.parametrize('header', RETURN_HEADERS)
    @pytest.mark.parametrize(('line', '_annotation', '_description'), FALLBACK_ANNOTATION_CASES)
    def test_return_annotation_is_entry(
        self,
        header: str,
        line: str,
        _annotation: str,
        _description: str,
    ) -> None:
        assert self.style.classify_section(header, (line,)) == 'entry'

    @pytest.mark.parametrize('header', RETURN_HEADERS)
    @pytest.mark.parametrize('line', NON_ANNOTATION_LINES)
    def test_return_expression_is_prose(self, header: str, line: str) -> None:
        assert self.style.classify_section(header, (line,)) == 'prose'

    @pytest.mark.parametrize('header', RETURN_HEADERS)
    def test_return_none_uses_normal_entry_grammar(self, header: str) -> None:
        assert self.style.classify_section(header, ('    None: No result.',)) == 'entry'

    def test_returns_entry_when_body_matches(self) -> None:
        assert self.style.classify_section('Returns', ('    bool: True if ok.',)) == 'entry'

    def test_returns_prose_by_default(self) -> None:
        assert self.style.classify_section('Returns') == 'prose'

    def test_wide_return_union_is_prose(self) -> None:
        line = '    ' + ' | '.join(['int'] * 1000) + ': Result.'

        assert self.style.classify_section('Returns', (line,)) == 'prose'
        assert self.style.match_entry(line, role='returns') is None

    def test_yields_entry_when_body_matches(self) -> None:
        assert self.style.classify_section('Yields', ('    str: Each line.',)) == 'entry'


class TestEntryMatching:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    def test_double_star_kwargs(self) -> None:
        match = self.style.match_entry('    **kwargs: Keyword args.')
        assert match is not None
        assert match.name == '**kwargs'

    def test_entry_no_description(self) -> None:
        match = self.style.match_entry('    name:')
        assert match is not None
        assert match.description == ''

    def test_entry_pattern_uses_requested_body_indent(self) -> None:
        pattern = _compile_entry_re(2)

        assert pattern.match('  name: Description.') is not None
        assert pattern.match('    name: Description.') is None

    def test_exception_name(self) -> None:
        match = self.style.match_entry('    ValueError: If bad.')
        assert match is not None
        assert match.name == 'ValueError'

    @pytest.mark.parametrize('role', RETURN_ROLES)
    @pytest.mark.parametrize(('line', 'annotation', 'description'), FALLBACK_ANNOTATION_CASES)
    def test_fallback_return_annotation(
        self,
        role: CanonicalRole,
        line: str,
        annotation: str,
        description: str,
    ) -> None:
        match = self.style.match_entry(line, role=role)
        assert match is not None
        assert match.name == ''
        assert match.type_ann == annotation
        assert match.description == description

    @pytest.mark.parametrize(('line', '_annotation', '_description'), FALLBACK_ANNOTATION_CASES)
    def test_fallback_return_annotation_is_not_arg(
        self,
        line: str,
        _annotation: str,
        _description: str,
    ) -> None:
        assert self.style.match_entry(line, role='param') is None

    def test_match_entry_role_empty(self) -> None:
        m = self.style.match_entry('    name (str): desc')
        assert m is not None
        assert m.role == ''

    def test_name_only(self) -> None:
        match = self.style.match_entry('    name: The description.')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == ''
        assert match.description == 'The description.'

    def test_name_with_type(self) -> None:
        match = self.style.match_entry('    name (str): The description.')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == 'str'
        assert match.description == 'The description.'

    @pytest.mark.parametrize('role', RETURN_ROLES)
    @pytest.mark.parametrize('line', NON_ANNOTATION_LINES)
    def test_non_annotation_expression(
        self,
        role: CanonicalRole,
        line: str,
    ) -> None:
        assert self.style.match_entry(line, role=role) is None

    def test_none_remains_valid_arg(self) -> None:
        match = self.style.match_entry('    None: No result.', role='param')
        assert match is not None
        assert match.name == 'None'
        assert match.type_ann == ''

    @pytest.mark.parametrize('role', RETURN_ROLES)
    def test_none_uses_normal_name_as_type_representation(self, role: CanonicalRole) -> None:
        match = self.style.match_entry('    None: No result.', role=role)
        assert match is not None
        assert match.name == 'None'
        assert match.type_ann == ''
        assert match.description == 'No result.'

    def test_not_an_entry(self) -> None:
        assert self.style.match_entry('    Just a line.') is None

    def test_star_args(self) -> None:
        match = self.style.match_entry('    *args: Positional args.')
        assert match is not None
        assert match.name == '*args'


class TestFormatEntry:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    def test_param_empty_description(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='')
        result = self.style.format_entry(em, 'param')
        assert result == ['    x (int):']

    def test_param_strip_types(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param', strip_types=True) == ['    x: The value.']

    def test_param_with_type(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param') == ['    x (int): The value.']

    def test_param_without_type(self) -> None:
        em = EntryMatch(name='x', type_ann='', description='The value.')
        assert self.style.format_entry(em, 'param') == ['    x: The value.']

    def test_raises(self) -> None:
        em = EntryMatch(name='ValueError', type_ann='', description='If empty.')
        assert self.style.format_entry(em, 'raises') == ['    ValueError: If empty.']

    def test_returns_entry(self) -> None:
        em = EntryMatch(name='int', type_ann='', description='The count.')
        assert self.style.format_entry(em, 'returns') == ['    int: The count.']

    def test_returns_named_with_type(self) -> None:
        em = EntryMatch(name='out', type_ann='ndarray', description='Output array.')
        assert self.style.format_entry(em, 'returns') == ['    out (ndarray): Output array.']

    def test_returns_no_type_no_name(self) -> None:
        em = EntryMatch(name='', type_ann='', description='The count.')
        assert self.style.format_entry(em, 'returns') == ['    The count.']

    def test_returns_normalized_type_in_type_ann(self) -> None:
        em = EntryMatch(name='', type_ann='int', description='The count.')
        assert self.style.format_entry(em, 'returns') == ['    int: The count.']

    def test_returns_strip_types_no_stray_colon(self) -> None:
        em = EntryMatch(name='int', type_ann='', description='The count.')
        result = self.style.format_entry(em, 'returns', strip_types=True)
        assert result == ['    The count.']


class TestMultiNameSplit:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    def test_multi_name_param_expands_to_reparseable_entries(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        match = EntryMatch(
            name='x1, x2',
            type_ann='array_like',
            description='The values.',
            role='param',
        )

        with caplog.at_level(logging.WARNING, logger='docwrap._styles._google'):
            lines = self.style.format_entry(match, 'param')

        assert lines == [
            '    x1 (array_like): The values.',
            '    x2 (array_like): The values.',
        ]
        reparsed = [self.style.match_entry(line, role='param') for line in lines]
        assert all(entry is not None for entry in reparsed)
        assert [
            (entry.name, entry.type_ann, entry.description)
            for entry in reparsed
            if entry is not None
        ] == [
            ('x1', 'array_like', 'The values.'),
            ('x2', 'array_like', 'The values.'),
        ]
        assert len(caplog.records) == 1
        assert 'shared description was duplicated and may need review' in caplog.text

    @pytest.mark.parametrize('role', ['attributes', 'returns', 'yields', 'raises', 'warns'])
    def test_other_roles_do_not_expand_comma_names(
        self,
        role: CanonicalRole,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        match = EntryMatch(
            name='x1, x2',
            type_ann='array_like',
            description='The values.',
            role=role,
        )

        with caplog.at_level(logging.WARNING, logger='docwrap._styles._google'):
            lines = self.style.format_entry(match, role)

        assert lines == ['    x1, x2 (array_like): The values.']
        assert caplog.records == []

    def test_single_name_param_is_not_expanded(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        match = EntryMatch(
            name='x',
            type_ann='array_like',
            description='The value.',
            role='param',
        )

        with caplog.at_level(logging.WARNING, logger='docwrap._styles._google'):
            lines = self.style.format_entry(match, 'param')

        assert lines == ['    x (array_like): The value.']
        assert caplog.records == []


class TestSectionHeaders:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    def test_all_known_headers_recognized(self) -> None:
        for header in self.style.known_headers:
            assert self.style.is_section_header(f'{header}:') == header

    def test_args_header(self) -> None:
        assert self.style.is_section_header('Args:') == 'Args'

    def test_format_examples(self) -> None:
        assert self.style.format_section_header('examples') == ['Examples:']

    def test_format_param(self) -> None:
        assert self.style.format_section_header('param') == ['Args:']

    def test_format_raises(self) -> None:
        assert self.style.format_section_header('raises') == ['Raises:']

    def test_indented_not_header(self) -> None:
        assert self.style.is_section_header('    Args:') is None

    def test_not_a_header(self) -> None:
        assert self.style.is_section_header('Not a header:') is None

    def test_returns_header(self) -> None:
        assert self.style.is_section_header('Returns:') == 'Returns'

    def test_unknown_key_returns_empty(self) -> None:
        assert self.style.format_section_header('unknown_key') == []


class TestTargetRoleSupport:
    def setup_method(self) -> None:
        self.style = _GoogleStyle()

    @pytest.mark.parametrize(
        'role',
        [
            'param',
            'keyword_param',
            'returns',
            'yields',
            'raises',
            'warns',
            'attributes',
            'methods',
            'see_also',
            'notes',
            'references',
            'examples',
            'todo',
            'warnings',
        ],
    )
    def test_supported_role(self, role: CanonicalRole) -> None:
        assert self.style.supports_canonical_role(role)

    def test_unsupported_role(self) -> None:
        assert not self.style.supports_canonical_role('other_param')
