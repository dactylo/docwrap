import pytest

from docwrap._styles import EntryMatch, get_style
from docwrap._styles._numpy import NumpyStyle
from docwrap._types import CanonicalRole


class TestCanonicalSections:
    def test_numpy_params(self) -> None:
        assert get_style('numpy').section_to_canonical['Params'] == 'param'


class TestClassification:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_notes_is_prose(self) -> None:
        assert self.style.classify_section('Notes') == 'prose'

    def test_parameters_is_entry(self) -> None:
        assert self.style.classify_section('Parameters') == 'entry'

    def test_returns_entry_compound_type(self) -> None:
        assert (
            self.style.classify_section('Returns', ('int or None', '    The result.')) == 'entry'
        )

    def test_returns_entry_when_body_matches(self) -> None:
        assert self.style.classify_section('Returns', ('bool',)) == 'entry'

    def test_returns_prose_by_default(self) -> None:
        assert self.style.classify_section('Returns') == 'prose'

    def test_returns_prose_when_compound_not_indented(self) -> None:
        assert self.style.classify_section('Returns', ('int or None', 'More text.')) == 'prose'

    def test_yields_entry_compound_type(self) -> None:
        assert self.style.classify_section('Yields', ('list of str', '    Each item.')) == 'entry'


class TestEntryMatching:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_comma_separated_names(self) -> None:
        match = self.style.match_entry('x1, x2 : array_like')
        assert match is not None
        assert match.name == 'x1, x2'
        assert match.type_ann == 'array_like'

    def test_complex_type(self) -> None:
        match = self.style.match_entry('data : dict[str, list[int]]')
        assert match is not None
        assert match.type_ann == 'dict[str, list[int]]'

    def test_exception_type(self) -> None:
        match = self.style.match_entry('ValueError')
        assert match is not None
        assert match.name == 'ValueError'

    def test_indented_is_not_entry(self) -> None:
        assert self.style.match_entry('    Indented description.') is None

    def test_match_entry_role_empty(self) -> None:
        m = self.style.match_entry('name : str')
        assert m is not None
        assert m.role == ''

    def test_match_entry_role_keyword(self) -> None:
        match = self.style.match_entry('value : int', role='returns')
        assert match is not None
        assert match.name == 'value'
        assert match.type_ann == 'int'

    def test_name_and_type(self) -> None:
        match = self.style.match_entry('name : str')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == 'str'
        assert match.description == ''

    def test_name_only(self) -> None:
        match = self.style.match_entry('name')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == ''


class TestFormatEntry:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_param_empty_description(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='')
        assert self.style.format_entry(em, 'param') == ['x : int']

    def test_param_strip_types(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param', strip_types=True) == ['x\n    The value.']

    def test_param_with_type(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param') == ['x : int\n    The value.']

    def test_param_without_type(self) -> None:
        em = EntryMatch(name='x', type_ann='', description='The value.')
        assert self.style.format_entry(em, 'param') == ['x\n    The value.']

    def test_raises(self) -> None:
        em = EntryMatch(name='ValueError', type_ann='', description='If empty.')
        assert self.style.format_entry(em, 'raises') == ['ValueError\n    If empty.']

    def test_returns_entry(self) -> None:
        em = EntryMatch(name='', type_ann='int', description='The count.')
        assert self.style.format_entry(em, 'returns') == ['int\n    The count.']

    def test_returns_named_with_type(self) -> None:
        em = EntryMatch(name='out', type_ann='ndarray', description='Output array.')
        assert self.style.format_entry(em, 'returns') == ['out : ndarray\n    Output array.']

    def test_returns_no_type_no_name(self) -> None:
        em = EntryMatch(name='', type_ann='', description='The count.')
        result = self.style.format_entry(em, 'returns')
        assert result == ['    The count.']
        assert not result[0].startswith('\n')  # no blank first line


class TestSectionHeaders:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    def test_all_known_headers_recognized(self) -> None:
        for header in self.style.known_headers:
            dashes = '-' * len(header)
            assert self.style.is_section_header(header, dashes) == header

    def test_format_dashes_match_length(self) -> None:
        result = self.style.format_section_header('see_also')
        assert result == ['See Also', '--------']

    def test_format_param(self) -> None:
        result = self.style.format_section_header('param')
        assert result == ['Parameters', '----------']

    def test_format_raises(self) -> None:
        result = self.style.format_section_header('raises')
        assert result == ['Raises', '------']

    def test_indented_header_not_matched(self) -> None:
        assert self.style.is_section_header('    Parameters', '    ----------') is None

    def test_indented_underline_not_matched(self) -> None:
        assert self.style.is_section_header('Parameters', '    ----------') is None

    def test_no_dashes_no_match(self) -> None:
        assert self.style.is_section_header('Parameters') is None

    def test_parameters_with_dashes(self) -> None:
        assert self.style.is_section_header('Parameters', '----------') == 'Parameters'

    def test_returns_with_dashes(self) -> None:
        assert self.style.is_section_header('Returns', '-------') == 'Returns'

    def test_short_dashes_no_match(self) -> None:
        assert self.style.is_section_header('Parameters', '--') is None

    def test_unknown_header(self) -> None:
        assert self.style.is_section_header('Unknown', '-------') is None


class TestTargetRoleSupport:
    def setup_method(self) -> None:
        self.style = NumpyStyle()

    @pytest.mark.parametrize(
        'role',
        [
            'param',
            'other_param',
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
        assert not self.style.supports_canonical_role('keyword_param')
