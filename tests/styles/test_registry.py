import importlib
import sys
from pathlib import Path

import pytest

from docwrap._styles import EntryMatch, get_style, get_styles
from docwrap._types import StyleName
from tests.helpers import run_setup


class TestEntryMatch:
    def test_entry_match_fields(self) -> None:
        match = EntryMatch(name='x', type_ann='int', description='The value.')
        assert match.name == 'x'
        assert match.type_ann == 'int'
        assert match.description == 'The value.'

    def test_entry_match_no_type(self) -> None:
        match = EntryMatch(name='x', type_ann='', description='The value.')
        assert match.type_ann == ''


class TestEntryMatchRole:
    def test_default_role_is_empty(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.')
        assert em.role == ''

    def test_role_field_preserved(self) -> None:
        em = EntryMatch(name='x', type_ann='int', description='The value.', role='param')
        assert em.role == 'param'


class TestGetStyles:
    def test_contains_known_styles(self) -> None:
        result = get_styles()
        assert result == frozenset({'google', 'numpy', 'sphinx'})

    def test_returned_key_set_is_immutable(self) -> None:
        result = get_styles()
        assert isinstance(result, frozenset)
        assert get_styles() == frozenset({'google', 'numpy', 'sphinx'})


class TestStyleRegistry:
    @pytest.mark.parametrize(
        'command',
        [
            (
                'import docwrap; assert docwrap.get_styles() '
                "== frozenset({'google', 'numpy', 'sphinx'})"
            ),
            (
                'from docwrap._styles import get_style, get_styles; '
                "assert get_styles() == frozenset({'google', 'numpy', 'sphinx'}); "
                "assert [get_style(name).name for name in ('google', 'numpy', 'sphinx')] "
                "== ['google', 'numpy', 'sphinx']"
            ),
        ],
    )
    def test_cold_import_exposes_fixed_builtin_styles(self, command: str) -> None:
        run_setup(
            [sys.executable, '-c', command],
            cwd=Path(__file__).parents[2] / 'src',
        )

    def test_fixed_builtin_key_set_survives_repeated_imports(self) -> None:
        expected = frozenset({'google', 'numpy', 'sphinx'})
        assert get_styles() == expected
        for module_name in (
            'docwrap._styles._google',
            'docwrap._styles._numpy',
            'docwrap._styles._sphinx',
        ):
            importlib.import_module(module_name)
            assert get_styles() == expected

    def test_get_style_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown'):
            get_style('unknown')  # pyright: ignore[reportArgumentType]

    def test_matchers_accept_role_keyword(self) -> None:
        assert get_style('google').match_entry('    value: Description.', role='param') is not None
        assert get_style('numpy').match_entry('value : int', role='returns') is not None
        assert (
            get_style('sphinx').match_entry(':param value: Description.', role='param') is not None
        )

    @pytest.mark.parametrize('style_name', ['google', 'numpy', 'sphinx'])
    def test_registered(self, style_name: StyleName) -> None:
        style = get_style(style_name)
        assert style.name == style_name
