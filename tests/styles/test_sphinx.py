import logging

import pytest

from docwrap._styles import EntryMatch, get_style
from docwrap._styles._sphinx import _SphinxStyle
from docwrap._types import CanonicalRole


class TestClassification:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    def test_always_entry(self) -> None:
        assert self.style.classify_section('') == 'entry'


class TestFieldMatching:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    def test_except_alias_normalized(self) -> None:
        m = self.style.match_entry(':except ValueError: bad')
        assert m is not None
        assert m.role == 'raises'

    def test_ivar(self) -> None:
        match = self.style.match_entry(':ivar x: The x value.')
        assert match is not None
        assert match.name == 'x'
        assert match.type_ann == ''
        assert match.description == 'The x value.'

    def test_ivar_with_inline_type(self) -> None:
        match = self.style.match_entry(':ivar dict[str, int] registry: Values.')
        assert match is not None
        assert match.role == 'ivar'
        assert match.name == 'registry'
        assert match.type_ann == 'dict[str, int]'
        assert match.description == 'Values.'

    def test_keyword_role(self) -> None:
        m = self.style.match_entry(':keyword debug: Enable debug.')
        assert m is not None
        assert m.role == 'keyword'
        assert m.name == 'debug'

    def test_keyword_with_type(self) -> None:
        m = self.style.match_entry(':keyword bool debug: Enable debug.')
        assert m is not None
        assert m.role == 'keyword'
        assert m.name == 'debug'
        assert m.type_ann == 'bool'

    @pytest.mark.parametrize(
        ('line', 'expected_role', 'expected_name'),
        [
            pytest.param(
                ':raises Custom Error: Bad.',
                'raises',
                'Custom Error',
                id='raises',
            ),
            pytest.param(
                ':warns Runtime Warning: Careful.',
                'warns',
                'Runtime Warning',
                id='warns',
            ),
            pytest.param(
                ':returns result value: Produced.',
                'returns',
                'result value',
                id='returns',
            ),
            pytest.param(
                ':yields next value: Produced.',
                'yields',
                'next value',
                id='yields',
            ),
            pytest.param(
                ':cvar class value: Stored.',
                'cvar',
                'class value',
                id='cvar',
            ),
            pytest.param(
                ':var instance value: Stored.',
                'var',
                'instance value',
                id='var',
            ),
        ],
    )
    def test_non_ivar_roles_keep_entire_middle_as_name(
        self,
        line: str,
        expected_role: str,
        expected_name: str,
    ) -> None:
        match = self.style.match_entry(line)
        assert match is not None
        assert match.role == expected_role
        assert match.name == expected_name
        assert match.type_ann == ''

    def test_not_a_field(self) -> None:
        assert self.style.match_entry('Just a line.') is None

    def test_param(self) -> None:
        match = self.style.match_entry(':param name: The description.')
        assert match is not None
        assert match.name == 'name'
        assert match.description == 'The description.'

    def test_param_role(self) -> None:
        m = self.style.match_entry(':param name: desc')
        assert m is not None
        assert m.role == 'param'

    def test_param_with_complex_type(self) -> None:
        match = self.style.match_entry(':param dict[str, int] name: The description.')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == 'dict[str, int]'

    def test_param_with_type(self) -> None:
        match = self.style.match_entry(':param str name: The description.')
        assert match is not None
        assert match.name == 'name'
        assert match.type_ann == 'str'

    def test_raise_alias_normalized(self) -> None:
        m = self.style.match_entry(':raise ValueError: bad')
        assert m is not None
        assert m.role == 'raises'

    def test_raises(self) -> None:
        match = self.style.match_entry(':raises ValueError: If bad.')
        assert match is not None
        assert match.name == 'ValueError'
        assert match.description == 'If bad.'

    def test_raises_role(self) -> None:
        m = self.style.match_entry(':raises ValueError: bad')
        assert m is not None
        assert m.role == 'raises'

    def test_return_alias_normalized(self) -> None:
        m = self.style.match_entry(':return: value')
        assert m is not None
        assert m.role == 'returns'

    def test_returns(self) -> None:
        match = self.style.match_entry(':returns: The return value.')
        assert match is not None
        assert match.name == ''
        assert match.description == 'The return value.'

    def test_rtype(self) -> None:
        match = self.style.match_entry(':rtype: bool')
        assert match is not None

    def test_rtype_role(self) -> None:
        m = self.style.match_entry(':rtype: int')
        assert m is not None
        assert m.role == 'rtype'

    def test_type_field(self) -> None:
        match = self.style.match_entry(':type name: str')
        assert match is not None
        assert match.name == 'name'
        assert match.description == 'str'

    def test_type_role(self) -> None:
        m = self.style.match_entry(':type name: str')
        assert m is not None
        assert m.role == 'type'

    def test_vartype(self) -> None:
        match = self.style.match_entry(':vartype attr: float')
        assert match is not None
        assert match.role == 'vartype'
        assert match.name == 'attr'
        assert match.type_ann == ''
        assert match.description == 'float'

    def test_warn_alias_normalized(self) -> None:
        m = self.style.match_entry(':warn UserWarning: Something.')
        assert m is not None
        assert m.role == 'warns'

    def test_warns_role(self) -> None:
        m = self.style.match_entry(':warns UserWarning: Something.')
        assert m is not None
        assert m.role == 'warns'
        assert m.name == 'UserWarning'

    def test_yield_alias_normalized(self) -> None:
        m = self.style.match_entry(':yield: Each line.')
        assert m is not None
        assert m.role == 'yields'

    def test_yields_role(self) -> None:
        m = self.style.match_entry(':yields: Each line.')
        assert m is not None
        assert m.role == 'yields'
        assert m.description == 'Each line.'


class TestFormatEntry:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    def test_ivar(self) -> None:
        em = EntryMatch(name='count', type_ann='int', description='Item count.')
        assert self.style.format_entry(em, 'ivar') == [
            ':ivar count: Item count.',
            ':vartype count: int',
        ]

    def test_ivar_strip_types(self) -> None:
        em = EntryMatch(name='count', type_ann='int', description='Item count.')
        assert self.style.format_entry(em, 'ivar', strip_types=True) == [
            ':ivar count: Item count.'
        ]

    def test_ivar_without_type(self) -> None:
        em = EntryMatch(name='count', type_ann='', description='Item count.')
        assert self.style.format_entry(em, 'ivar') == [':ivar count: Item count.']

    def test_param_empty_desc_no_trailing_space(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='')
        result = self.style.format_entry(em, 'param')
        assert result == [':param int x:']

    def test_param_strip_types(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param', strip_types=True) == [':param x: The value.']

    def test_param_with_type(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert self.style.format_entry(em, 'param') == [':param int x: The value.']

    def test_param_without_type(self) -> None:
        em = EntryMatch(name='x', type_ann='', description='The value.')
        assert self.style.format_entry(em, 'param') == [':param x: The value.']

    def test_raises(self) -> None:
        em = EntryMatch(name='ValueError', type_ann='', description='If empty.')
        assert self.style.format_entry(em, 'raises') == [':raises ValueError: If empty.']

    def test_returns_empty_desc_no_trailing_space(self) -> None:
        em = EntryMatch(name='', type_ann='int', description='')
        result = self.style.format_entry(em, 'returns')
        for line in result:
            assert line == line.rstrip(), f'trailing space in: {line!r}'

    def test_returns_strip_types(self) -> None:
        em = EntryMatch(name='', type_ann='int', description='The count.')
        result = self.style.format_entry(em, 'returns', strip_types=True)
        assert result == [':returns: The count.']

    def test_returns_with_type(self) -> None:
        em = EntryMatch(name='', type_ann='int', description='The count.')
        result = self.style.format_entry(em, 'returns')
        assert result == [':returns: The count.', ':rtype: int']

    def test_returns_without_type(self) -> None:
        em = EntryMatch(name='', type_ann='', description='The count.')
        assert self.style.format_entry(em, 'returns') == [':returns: The count.']


class TestMultiNameSplit:
    def test_multi_name_param_splits_to_multiple_fields(self) -> None:
        style_s = get_style('sphinx')
        em = EntryMatch(name='x1, x2', type_ann='int', description='The values.')
        result = style_s.format_entry(em, 'param')
        assert len(result) == 2
        assert ':param int x1: The values.' in result[0]
        assert ':param int x2: The values.' in result[1]

    def test_multi_name_split_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        style_s = get_style('sphinx')
        em = EntryMatch(name='a, b', type_ann='int', description='Values.')
        with caplog.at_level(logging.WARNING, logger='docwrap._styles._sphinx'):
            style_s.format_entry(em, 'param')
        assert 'Splitting' in caplog.text

    def test_single_name_is_not_split(self) -> None:
        style_s = get_style('sphinx')
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        result = style_s.format_entry(em, 'param')
        assert len(result) == 1


class TestSectionHeaders:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    def test_format_returns_empty(self) -> None:
        assert self.style.format_section_header('param') == []

    def test_known_headers_empty(self) -> None:
        assert len(self.style.known_headers) == 0

    def test_no_headers(self) -> None:
        assert self.style.is_section_header('Args:') is None
        assert self.style.is_section_header('Parameters') is None


class TestTargetRoleSupport:
    def setup_method(self) -> None:
        self.style = _SphinxStyle()

    @pytest.mark.parametrize(
        'role',
        ['param', 'keyword_param', 'returns', 'raises', 'warns', 'attributes'],
    )
    def test_supported_role(self, role: CanonicalRole) -> None:
        assert self.style.supports_canonical_role(role)

    @pytest.mark.parametrize(
        'role',
        [
            'other_param',
            'yields',
            'methods',
            'see_also',
            'notes',
            'references',
            'examples',
            'todo',
            'warnings',
        ],
    )
    def test_unsupported_role(self, role: CanonicalRole) -> None:
        assert not self.style.supports_canonical_role(role)
