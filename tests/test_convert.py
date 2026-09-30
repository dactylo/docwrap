import logging
import textwrap

import pytest

from docwrap._convert import (
    DuplicateFieldError,
    UnrepresentableReturnError,
    _convert_entry,
    _group_sphinx_entries,
    _merge_sphinx_types,
    _strip_types_entry,
    convert_regions,
)
from docwrap._core._codeblock import (
    find_code_block_lines,
)
from docwrap._core._discover import find_docstrings, parse_docstrings
from docwrap._core._normalize import normalize_content
from docwrap._core._parse import Region, match_opening_title, parse_regions
from docwrap._core._process import wrap_source, wrap_source_report
from docwrap._styles import EntryMatch, get_style
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    CanonicalRole,
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    StyleName,
)
from tests.helpers import (
    FILL_STYLES,
    REFLOW_MODES,
    SMART_SLACK_VALUES,
    assert_region_projection,
)

BACKSLASH = chr(92)
ROLES: tuple[CanonicalRole, ...] = ('returns', 'yields')
SOURCE_STYLES: tuple[StyleName, ...] = ('google', 'numpy')
TARGETS: tuple[StyleName | None, ...] = (None, 'google', 'numpy', 'sphinx')


class TestConversionCycleSeparators:
    _SOURCE = (
        'def f(a, b):\n'
        '    """Summary.\n'
        '\n'
        '    Args:\n'
        '        a: First value.\n'
        '        b: Second value.\n'
        '\n'
        '    Returns:\n'
        '        The combined result.\n'
        '\n'
        '    Raises:\n'
        '        ValueError: If inputs conflict.\n'
        '    """\n'
    )

    def _cycle(self, source: str) -> str:
        to_sphinx = wrap_source(source, config=DocwrapConfig(convert='sphinx'))
        return wrap_source(to_sphinx, config=DocwrapConfig(convert='google'))

    def test_cycles_reach_a_fixed_point(self) -> None:
        first = self._cycle(self._SOURCE)
        second = self._cycle(first)
        third = self._cycle(second)
        assert second == first
        assert third == second

    def test_no_blank_run_growth(self) -> None:
        out = self._cycle(self._cycle(self._SOURCE))
        assert '\n\n\n' not in out

    def test_same_target_conversion_idempotent(self) -> None:
        once = wrap_source(self._SOURCE, config=DocwrapConfig(convert='sphinx'))
        assert wrap_source(once, config=DocwrapConfig(convert='sphinx')) == once


class TestConversionIdempotency:
    def test_explicit_google_to_sphinx_long_field_is_idempotent(self) -> None:
        source = (
            'def f(value):\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        value (dict[str, int]): This is a deliberately long typed parameter '
            'description for wrapping.\n'
            '    """\n'
        )
        config = DocwrapConfig(line_length=32, style='google', convert='sphinx')

        first_report = wrap_source_report(source, config=config)
        second_report = wrap_source_report(first_report.source, config=config)
        reparsed = parse_docstrings(
            first_report.source,
            find_docstrings(first_report.source),
            DocwrapConfig(),
        )
        field_matches = [
            region.entry_match
            for region in reparsed[0].regions
            if region.role == 'param' and region.entry_match is not None
        ]

        assert ':param dict[str, int] value:' in first_report.source
        assert '\n        is a deliberately long' in first_report.source
        assert len(field_matches) == 1
        assert ' '.join(field_matches[0].description.split()) == (
            'This is a deliberately long typed parameter description for wrapping.'
        )
        assert second_report.source == first_report.source
        assert second_report.converted == 0

    def test_google_to_numpy_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_google_to_sphinx_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_numpy_to_google_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Parameters
                ----------
                x : int
                    The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_numpy_to_sphinx_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Parameters
                ----------
                x : int
                    The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_sphinx_complex_return_wraps_on_first_conversion(self) -> None:
        description = (
            'This is a deliberately long return description that should wrap when converted '
            'to Google style.'
        )
        source = textwrap.dedent(f"""\
            def f():
                \"\"\"Summary.

                :returns: {description}
                :rtype: dict[str, int]
                \"\"\"
        """)
        config = DocwrapConfig(convert='google', line_length=60, reflow='full')

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert max(map(len, first.source.splitlines())) <= 60
        assert f'dict[str, int]: {description}' in ' '.join(first.source.split())
        assert second.source == first.source

    @pytest.mark.parametrize('fill_style', FILL_STYLES)
    @pytest.mark.parametrize('reflow', REFLOW_MODES)
    def test_sphinx_field_shaped_continuation_idempotent(
        self,
        fill_style: FillStyle,
        reflow: ReflowMode,
    ) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Args:
                    strip_types: If True, omit type annotations and suppress :rtype:.
                \"\"\"
        """)
        config = DocwrapConfig(
            convert='sphinx',
            fill_style=fill_style,
            line_length=40,
            reflow=reflow,
            strip_types=True,
            wrap_summary=True,
        )

        first = wrap_source_report(source, config=config)
        assert '\n        suppress :rtype:.\n' in first.source
        assert first.source.count(':rtype:.') == 1
        assert first.skips == ()
        assert first.parse_error is False
        compile(first.source, '<test>', 'exec')

        docstrings = find_docstrings(first.source)
        reparsed = parse_docstrings(first.source, docstrings, config)
        assert len(reparsed) == 1
        assert [region for region in reparsed[0].regions if region.role == 'rtype'] == []

        second = wrap_source_report(first.source, config=config)

        assert second.source == first.source
        assert second.skips == ()
        assert second.parse_error is False

    def test_sphinx_to_google_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                :param int x: The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_sphinx_to_numpy_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                :param int x: The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source
        assert second.converted == 0

    def test_warns_google_sphinx_google_roundtrip(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Warns:
                    UserWarning: Something bad.
                \"\"\"
        """)
        config_s = DocwrapConfig(convert='sphinx')
        config_g = DocwrapConfig(convert='google')
        sphinx = wrap_source_report(source, config=config_s)
        assert ':warns' in sphinx.source
        back = wrap_source_report(sphinx.source, config=config_g)
        assert 'Warns:' in back.source
        assert ':warns' not in back.source

    def test_yields_google_to_sphinx_stays_authored(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Yields:
                    str: Each line.
                \"\"\"
        """)
        config_s = DocwrapConfig(convert='sphinx')
        config_g = DocwrapConfig(convert='google')
        sphinx = wrap_source_report(source, config=config_s)
        assert 'Yields:' in sphinx.source
        assert 'str: Each line.' in sphinx.source
        assert ':yields:' not in sphinx.source
        back = wrap_source_report(sphinx.source, config=config_g)
        assert back.source == sphinx.source


class TestConvertEdgeCases:
    def test_examples_section_survives_conversion(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.

                Examples:
                    >>> f(1)
                    1
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        report = wrap_source_report(source, config=config)
        assert report.converted == 1
        # Parameters should be converted to NumPy
        assert 'Parameters' in report.source
        assert '----------' in report.source
        # Examples content should survive (section header converted)
        assert 'Examples' in report.source
        assert '>>> f(1)' in report.source

    def test_mixed_style_file_converts_independently(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary for f.

                Args:
                    x (int): The value.
                \"\"\"


            def g(y):
                \"\"\"Summary for g.

                Parameters
                ----------
                y : str
                    The name.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx')
        report = wrap_source_report(source, config=config)
        assert report.converted == 2
        assert ':param int x: The value.' in report.source
        assert ':param str y: The name.' in report.source

    def test_same_source_target_is_noop(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert report.converted == 0
        assert report.convert_target == 'google'
        # Wrapping still runs (source is valid)
        compile(report.source, '<test>', 'exec')

    def test_strip_types_same_style(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google', strip_types=True)
        report = wrap_source_report(source, config=config)
        # Same-style: no conversion
        assert report.converted == 0
        # Types should be stripped
        assert '(int)' not in report.source
        assert 'x:' in report.source

    def test_style_agnostic_passthrough(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Just a summary.\"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        report = wrap_source_report(source, config=config)
        assert report.converted == 0
        assert report.source == source

    def test_unchanged_preserved_return_is_not_converted(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                Returns
                -------
                int or None
                    The result.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx')

        report = wrap_source_report(source, config=config, detail='list')

        assert report.source == source
        assert report.wrapped == 0
        assert report.converted == 0
        assert report.skips == ()
        assert len(report.docstrings) == 1
        assert report.docstrings[0].changed is False
        assert report.docstrings[0].converted is False


class TestConvertPipeline:
    def test_convert_tracks_per_docstring(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x: The value.
                \"\"\"


            def g():
                \"\"\"Just a summary.\"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        report = wrap_source_report(source, config=config, detail='list')
        # f has Args -> converted, g is agnostic -> not converted
        assert report.converted == 1
        converted_ds = [d for d in report.docstrings if d.converted]
        assert len(converted_ds) == 1
        assert converted_ds[0].qualname == 'f'

    def test_google_to_numpy_full_pipeline(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.

                Returns:
                    True if ok.
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        report = wrap_source_report(source, config=config)
        assert 'Parameters' in report.source
        assert '----------' in report.source
        assert 'x : int' in report.source
        assert report.converted == 1
        assert report.convert_target == 'numpy'

    def test_google_to_sphinx_full_pipeline(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx')
        report = wrap_source_report(source, config=config)
        assert ':param int x: The value.' in report.source
        assert report.converted == 1
        assert report.convert_target == 'sphinx'

    def test_keyword_args_google_to_numpy_full(self) -> None:
        source = textwrap.dedent("""\
            def f(**kwargs):
                \"\"\"Summary.

                Keyword Args:
                    debug (bool): Enable debug.
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy')
        report = wrap_source_report(source, config=config)
        assert 'Keyword' in report.source  # header preserved
        assert 'debug' in report.source  # entry preserved

    def test_keyword_args_to_sphinx_valid_role(self) -> None:
        source = textwrap.dedent('''\
            def f(**kwargs):
                """Summary.

                Keyword Args:
                    debug (bool): Enable debug.
                """
        ''')
        config = DocwrapConfig(convert='sphinx')
        report = wrap_source_report(source, config=config)
        assert ':keyword bool debug:' in report.source
        assert ':keyword_param' not in report.source

    def test_named_numpy_yield_is_preserved_before_sphinx_preflight(self) -> None:
        source = textwrap.dedent('''\
            def f():
                """Summary.

                Raises
                ------
                ValueError
                    If no item is available.

                Yields
                ------
                result : int
                    Named item.

                Warns
                -----
                RuntimeWarning
                    If item generation is slow.
                """
        ''')
        config = DocwrapConfig(convert='sphinx', strip_types=True)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert ':raises ValueError: If no item is available.' in first.source
        assert '\n    Yields\n    ------\n    result : int\n        Named item.\n' in first.source
        assert ':warns RuntimeWarning: If item generation is slow.' in first.source
        assert first.skips == ()
        assert second.source == first.source
        assert second.skips == ()

    def test_numpy_to_google_full_pipeline(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Parameters
                ----------
                x : int
                    The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert 'Args:' in report.source
        assert 'x (int): The value.' in report.source
        assert report.converted == 1

    def test_numpy_to_google_no_duplicate_headers(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Parameters
                ----------
                x : int
                    The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert report.source.count('Args:') == 1

    def test_sphinx_reordered_prose_positioned_correctly(self) -> None:
        source = textwrap.dedent('''\
            def f(x):
                """Summary.

                :raises ValueError: Bad.

                Note about errors.

                :param x: val.
                """
        ''')
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        lines = report.source.split('\n')
        # Args should come before Raises (canonical order)
        args_idx = next(i for i, ln in enumerate(lines) if 'Args:' in ln)
        raises_idx = next(i for i, ln in enumerate(lines) if 'Raises:' in ln)
        assert args_idx < raises_idx
        # "Note about errors" should appear after Raises (it was between raises and param)
        note_idx = next(i for i, ln in enumerate(lines) if 'Note about errors' in ln)
        assert note_idx > raises_idx

    def test_sphinx_to_google_blank_between_sections(self) -> None:
        source = textwrap.dedent('''\
            def f(x):
                """Summary.

                :param x: The value.
                :returns: The result.
                """
        ''')
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        lines = report.source.split('\n')
        # Find Args: and Returns: lines
        args_idx = next(i for i, ln in enumerate(lines) if 'Args:' in ln)
        returns_idx = next(i for i, ln in enumerate(lines) if 'Returns:' in ln)
        # There should be a blank line between the last entry of Args and Returns:
        between = lines[args_idx + 2 : returns_idx]  # skip header + entry
        assert any(ln.strip() == '' for ln in between), f'No blank between sections: {between}'

    def test_sphinx_to_google_full_pipeline(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                :param int x: The value.
                :returns: The result.
                \"\"\"
        """)
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert 'Args:' in report.source
        assert 'x (int): The value.' in report.source
        assert 'Returns:' in report.source
        assert report.converted == 1

    def test_strip_types_drops_empty_returns_sphinx_target(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                :returns:
                :rtype: int
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert ':returns:' not in report.source
        assert ':rtype:' not in report.source
        assert report.source == 'def f():\n    """Summary."""\n'
        # Idempotence: a second pass is stable.
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_keeps_bare_returns_without_rtype_sphinx_target(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                :returns:
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert ':returns:' in report.source
        assert report.source == source
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_keeps_described_returns_sphinx_target(self) -> None:
        source = textwrap.dedent("""\
            def f():
                \"\"\"Summary.

                :returns: The count.
                :rtype: int
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert ':returns: The count.' in report.source
        assert ':rtype:' not in report.source
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_with_convert(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(convert='numpy', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert 'Parameters' in report.source
        # Type should be stripped
        assert 'int' not in report.source
        assert report.converted == 1

    def test_strip_types_without_convert(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)
        assert '(int)' not in report.source
        assert 'x:' in report.source

    def test_style_override_with_convert(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): The value.
                \"\"\"
        """)
        config = DocwrapConfig(style='google', convert='numpy')
        report = wrap_source_report(source, config=config)
        assert 'Parameters' in report.source
        assert report.converted == 1


class TestConvertRegions:
    def test_compound_type_passthrough_numpy_to_google(self) -> None:
        source = 'Summary.\n\nReturns\n-------\nint or None\n    The result.'
        style_n = get_style('numpy')
        style_g = get_style('google')
        regions = _regions_from(source, 'numpy')
        # Find the compound-type entry (entry_match=None)
        compound = [r for r in regions if r.kind == 'entry' and r.entry_match is None]
        assert len(compound) >= 1
        converted = convert_regions(regions, style_n, style_g)
        # Compound type should be re-indented for Google (body indent 4 vs NumPy 0)
        compound_out = [r for r in converted if r.kind == 'entry' and r.entry_match is None]
        assert len(compound_out) >= 1
        # Check that indentation was added (Google uses 4-space body indent)
        for r in compound_out:
            first_line = r.text.split('\n')[0]
            assert first_line.startswith('    ') or first_line == first_line.strip()
            assert r.role == 'returns'

    def test_converted_header_has_target_section_name(self) -> None:
        source = 'Summary.\n\nParameters\n----------\nx : int\n    val'
        style_n = get_style('numpy')
        style_g = get_style('google')
        regions = _regions_from(source, 'numpy')
        converted = convert_regions(regions, style_n, style_g, strip_types=False)
        headers = [r for r in converted if r.kind == 'section_header']
        args_headers = [r for r in headers if 'Args' in r.text]
        assert args_headers[0].section_name == 'Args'  # not 'Parameters'

    def test_entry_match_none_on_converted(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        # Source entry has entry_match set
        src_entries = [r for r in regions if r.kind == 'entry']
        assert src_entries[0].entry_match is not None
        converted = convert_regions(regions, style_g, style_n)
        entries = [r for r in converted if r.kind == 'entry']
        for e in entries:
            assert e.entry_match is None
            assert e.role == 'param'

    @pytest.mark.parametrize(
        ('source_name', 'target_name', 'expected_roles'),
        [
            pytest.param('sphinx', 'google', ('returns',), id='google-target'),
            pytest.param(
                'google',
                'sphinx',
                ('returns', 'rtype'),
                id='sphinx-target',
            ),
        ],
    )
    def test_generated_return_entries_retain_target_roles(
        self,
        source_name: StyleName,
        target_name: StyleName,
        expected_roles: tuple[str, ...],
    ) -> None:
        target_style = get_style(target_name)
        entry = Region(
            kind='entry',
            text='source entry',
            entry_match=EntryMatch(
                name='',
                type_ann='dict[str, int]',
                description='Mapping result.',
                role='returns',
            ),
            role='returns',
        )

        converted = _convert_entry(
            entry,
            get_style(source_name),
            target_style,
            strip_types=False,
            deleted_text=None,
        )

        assert converted
        assert tuple(region.role for region in converted) == expected_roles
        assert all(
            target_style.match_entry(region.text, role=region.role) is not None
            for region in converted
        )

    def test_google_raises_to_numpy(self) -> None:
        source = 'Summary.\n\nRaises:\n    ValueError: If x is bad.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n)
        headers = [r for r in converted if r.kind == 'section_header']
        assert any('Raises' in r.text for r in headers)
        entries = [r for r in converted if r.kind == 'entry']
        assert any('ValueError' in r.text for r in entries)

    def test_google_to_numpy_args(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n, strip_types=False)
        headers = [r for r in converted if r.kind == 'section_header']
        assert any('Parameters' in r.text for r in headers)
        entries = [r for r in converted if r.kind == 'entry']
        assert any('x : int' in r.text for r in entries)

    def test_google_to_sphinx(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        style_s = get_style('sphinx')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_s)
        entries = [r for r in converted if r.kind == 'entry']
        assert any(':param int x: The value.' in r.text for r in entries)
        # Sphinx has no section headers
        headers = [r for r in converted if r.kind == 'section_header']
        assert len(headers) == 0

    def test_keyword_args_preserved_when_target_lacks_mapping(self) -> None:
        source = 'Summary.\n\nKeyword Args:\n    debug (bool): Enable debug.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n, strip_types=False)
        headers = [r for r in converted if r.kind == 'section_header']
        # Header should be preserved, not dropped
        assert any('Keyword' in r.text for r in headers)

    def test_keyword_args_span_preserved_for_numpy(self) -> None:
        source = 'Summary.\n\nKeyword Args:\n    debug (bool): Enable debug.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n, strip_types=False)
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert [region.original_lines for region in authored_span] == [
            ('Keyword Args:',),
            ('    debug (bool): Enable debug.',),
        ]
        assert all(region.preserve_verbatim for region in authored_span)

    @pytest.mark.parametrize(
        ('target_name', 'expected_entry'),
        [
            pytest.param(
                'google',
                '    result (int): Named result.',
                id='google',
            ),
            pytest.param(
                'numpy',
                'result : int\n    Named result.',
                id='numpy',
            ),
        ],
    )
    def test_named_numpy_return_keeps_semantics_for_representable_target(
        self,
        target_name: StyleName,
        expected_entry: str,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                'Summary.\n\nReturns\n-------\nresult : int\n    Named result.',
                'numpy',
            ),
            get_style('numpy'),
            get_style(target_name),
        )

        return_entries = [
            region.text
            for region in converted
            if region.kind == 'entry' and region.role == 'returns'
        ]
        assert return_entries == [expected_entry]

    def test_named_numpy_return_to_sphinx_raises_before_deletion(self) -> None:
        deleted_text = ['existing deletion']

        with pytest.raises(UnrepresentableReturnError) as caught:
            convert_regions(
                _regions_from(
                    'Summary.\n\nReturns\n-------\nresult : int\n    Named result.',
                    'numpy',
                ),
                get_style('numpy'),
                get_style('sphinx'),
                deleted_text=deleted_text,
            )

        assert deleted_text == ['existing deletion']
        assert 'Returns' in str(caught.value)
        assert "'result'" in str(caught.value)
        assert 'sphinx' in str(caught.value).lower()

    def test_numpy_to_google_no_duplicate_headers(self) -> None:
        source = 'Summary.\n\nParameters\n----------\nx : int\n    The value.'
        style_n = get_style('numpy')
        style_g = get_style('google')
        regions = _regions_from(source, 'numpy')
        converted = convert_regions(regions, style_n, style_g, strip_types=False)
        headers = [r for r in converted if r.kind == 'section_header']
        # Should be exactly one Args: header, not two
        assert len(headers) == 1
        assert headers[0].text == 'Args:'

    def test_passthrough_non_entry_regions(self) -> None:
        source = 'Summary.\n\nSome description.\n\nArgs:\n    x: The value.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n)
        summaries = [r for r in converted if r.kind == 'summary']
        assert len(summaries) == 1
        assert summaries[0].text == 'Summary.'
        blanks = [r for r in converted if r.kind == 'blank']
        assert len(blanks) >= 1
        descs = [r for r in converted if r.kind == 'description']
        assert len(descs) == 1
        assert descs[0].text == 'Some description.'

    def test_returns_normalization_google_to_numpy(self) -> None:
        source = 'Summary.\n\nReturns:\n    int: The count.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        # Google parses Returns entry as name='int', type_ann=''
        returns_entries = [r for r in regions if r.kind == 'entry' and r.role == 'returns']
        assert len(returns_entries) == 1
        assert returns_entries[0].entry_match is not None
        assert returns_entries[0].entry_match.name == 'int'
        assert returns_entries[0].entry_match.type_ann == ''
        # After conversion, NumPy should show type only (no name)
        converted = convert_regions(regions, style_g, style_n)
        entries = [r for r in converted if r.kind == 'entry']
        assert len(entries) == 1
        # NumPy format_entry with name='' type_ann='int' produces
        # 'int\n    The count.'
        assert 'int' in entries[0].text

    def test_returns_normalization_google_to_sphinx(self) -> None:
        source = 'Summary.\n\nReturns:\n    int: The count.'
        style_g = get_style('google')
        style_s = get_style('sphinx')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_s)
        entries = [r for r in converted if r.kind == 'entry']
        # _SphinxStyle.format_entry returns [':returns: desc', ':rtype: type']
        assert any(':returns: The count.' in r.text for r in entries)
        assert any(':rtype: int' in r.text for r in entries)

    def test_same_style_passthrough(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_g)
        entries = [r for r in converted if r.kind == 'entry']
        assert len(entries) == 1
        # Should be in Google format
        assert 'x (int): The value.' in entries[0].text

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_authored_blank_seals_preserved_span_once(
        self,
        target_name: StyleName,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n:param x: Before.\n\n:cvar registry: Opaque.\n\n:returns: After.',
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        opaque_index = next(
            index for index, region in enumerate(converted) if ':cvar registry:' in region.text
        )
        assert converted[opaque_index - 1].kind == 'blank'
        assert converted[opaque_index - 2].kind != 'blank'
        assert converted[opaque_index + 1].kind == 'blank'
        assert converted[opaque_index + 2].kind != 'blank'

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_authored_blank_separates_opaque_vartype_from_target_header(
        self,
        role: str,
        target_name: StyleName,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n'
                f':{role} registry: Opaque.\n'
                ':param x: Between.\n'
                ':vartype registry: dict[str, int]\n'
                '\n'
                ':returns: After.',
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        vartype_index = next(
            index for index, region in enumerate(converted) if ':vartype registry:' in region.text
        )
        next_header_index = next(
            index
            for index, region in enumerate(converted[vartype_index + 1 :], vartype_index + 1)
            if region.kind == 'section_header'
        )
        assert next_header_index == vartype_index + 2
        assert converted[vartype_index + 1].kind == 'blank'
        assert converted[vartype_index + 1].original_lines == ('',)

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_consecutive_opaque_family_fields_stay_contiguous(
        self,
        target_name: StyleName,
    ) -> None:
        expected_lines = [
            ':cvar first: First.',
            ':var second: Second.',
            ':vartype first: int',
            ':vartype second: str',
        ]
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n' + '\n'.join(expected_lines),
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        field_indexes = [
            index for index, region in enumerate(converted) if region.text in expected_lines
        ]
        assert field_indexes == list(range(field_indexes[0], field_indexes[-1] + 1))
        assert [converted[index].text for index in field_indexes] == expected_lines
        assert all(converted[index].preserve_verbatim for index in field_indexes)

    def test_sphinx_param_no_type_to_google(self) -> None:
        source = 'Summary.\n\n:param x: The value.'
        style_s = get_style('sphinx')
        style_g = get_style('google')
        regions = _regions_from(source, 'sphinx')
        converted = convert_regions(regions, style_s, style_g)
        entries = [r for r in converted if r.kind == 'entry']
        assert any('x: The value.' in r.text for r in entries)
        # No parenthesized type
        assert not any('(' in r.text for r in entries)

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_preserved_field_span_stays_between_supported_neighbors(
        self,
        role: str,
        target_name: StyleName,
    ) -> None:
        body = (
            'Summary.\n\n'
            ':param int before: Converted before.\n'
            f':{role} registry: Authored  value.\n'
            '    continuation  spacing.\n'
            ':vartype registry: dict[str, int]\n'
            '\n'
            'Associated  prose after field.\n'
            '\n'
            ':returns: Converted after.\n'
            ':rtype: bool'
        )
        converted = convert_regions(
            _regions_from(body, 'sphinx'),
            get_style('sphinx'),
            get_style(target_name),
        )

        preserved_indexes = [
            index for index, region in enumerate(converted) if region.preserve_verbatim
        ]
        assert preserved_indexes
        assert preserved_indexes == list(range(preserved_indexes[0], preserved_indexes[-1] + 1))
        preserved = [converted[index] for index in preserved_indexes]
        assert [region.original_lines for region in preserved] == [
            (f':{role} registry: Authored  value.', '    continuation  spacing.'),
            (':vartype registry: dict[str, int]',),
            ('',),
            ('Associated  prose after field.',),
            ('',),
        ]
        before_index = next(
            index for index, region in enumerate(converted) if region.role == 'param'
        )
        after_index = next(
            index for index, region in enumerate(converted) if region.role == 'returns'
        )
        assert before_index < preserved_indexes[0]
        assert preserved_indexes[-1] < after_index

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_preserved_vartype_name_matching_is_global(
        self,
        role: str,
        target_name: StyleName,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n'
                f':{role} registry: Opaque.\n'
                ':param x: Between.\n'
                ':vartype registry: dict[str, int]',
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        preserved_text = {region.text for region in converted if region.preserve_verbatim}
        assert f':{role} registry: Opaque.' in preserved_text
        assert ':vartype registry: dict[str, int]' in preserved_text
        between = next(region for region in converted if 'Between.' in region.text)
        assert not between.preserve_verbatim

    def test_sphinx_to_google_entry(self) -> None:
        source = 'Summary.\n\n:param int x: The value.'
        style_s = get_style('sphinx')
        style_g = get_style('google')
        regions = _regions_from(source, 'sphinx')
        converted = convert_regions(regions, style_s, style_g)
        headers = [r for r in converted if r.kind == 'section_header']
        assert any('Args' in r.text for r in headers)
        entries = [r for r in converted if r.kind == 'entry']
        assert any('x (int): The value.' in r.text for r in entries)

    @pytest.mark.parametrize(
        ('body', 'expected_deleted'),
        [
            pytest.param(
                'Summary.\n\n:ivar attr: Value.\n:vartype attr: float',
                [':vartype attr: float'],
                id='paired',
            ),
            pytest.param(
                'Summary.\n\n:ivar float attr: Value.',
                ['float'],
                id='inline-compatibility',
            ),
        ],
    )
    def test_sphinx_typed_ivar_strip_types_preserves_attribute(
        self,
        body: str,
        expected_deleted: list[str],
    ) -> None:
        style = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(body, 'sphinx'),
            style,
            style,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [region.text for region in converted if region.kind == 'entry'] == [
            ':ivar attr: Value.'
        ]
        assert deleted_text == expected_deleted

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_sphinx_unrelated_vartype_forms_distinct_preserved_span(
        self,
        role: str,
        target_name: StyleName,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                f'Summary.\n\n:{role} registry: Opaque.\nAssociated prose.\n:vartype orphan: int',
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        opaque_index = next(
            index for index, region in enumerate(converted) if f':{role} registry:' in region.text
        )
        associated_index = next(
            index for index, region in enumerate(converted) if region.text == 'Associated prose.'
        )
        orphan_index = next(
            index for index, region in enumerate(converted) if ':vartype orphan:' in region.text
        )
        assert converted[opaque_index].preserve_verbatim
        assert converted[associated_index].preserve_verbatim
        assert converted[orphan_index].preserve_verbatim
        separator = converted[associated_index + 1 : orphan_index]
        assert [
            (region.kind, region.preserve_verbatim, region.original_lines) for region in separator
        ] == [
            ('blank', False, None),
        ]

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize(
        ('target_name', 'preceding_kind'),
        [
            pytest.param('google', 'blank', id='google'),
            pytest.param('numpy', 'entry', id='numpy'),
        ],
    )
    def test_sphinx_unsafe_prose_extends_preserved_span_backward(
        self,
        role: str,
        target_name: StyleName,
        preceding_kind: str,
    ) -> None:
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n'
                ':param x: Before.\n'
                'Authored prose for the parameter group.\n'
                f':{role} registry: Opaque.\n'
                ':returns: Done.',
                'sphinx',
            ),
            get_style('sphinx'),
            get_style(target_name),
        )

        prose_index = next(
            index
            for index, region in enumerate(converted)
            if region.text == 'Authored prose for the parameter group.'
        )
        opaque_index = next(
            index for index, region in enumerate(converted) if f':{role} registry:' in region.text
        )
        assert converted[prose_index].preserve_verbatim
        assert converted[opaque_index].preserve_verbatim
        assert converted[prose_index - 1].kind == preceding_kind
        assert opaque_index == prose_index + 1

    def test_strip_types_google_to_numpy(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        style_n = get_style('numpy')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_n, strip_types=True)
        entries = [r for r in converted if r.kind == 'entry']
        assert len(entries) == 1
        # With strip_types, no type annotation should appear
        assert ' : int' not in entries[0].text
        assert 'x' in entries[0].text

    @pytest.mark.parametrize(
        ('source_name', 'body', 'expected_entry'),
        [
            pytest.param(
                'google',
                'Summary.\n\nAttributes:\n    attr (float): Value.',
                '    attr (float): Value.',
                id='google',
            ),
            pytest.param(
                'numpy',
                'Summary.\n\nAttributes\n----------\nattr : float\n    Value.',
                'attr : float\n    Value.',
                id='numpy',
            ),
        ],
    )
    def test_typed_attribute_round_trips_through_sphinx(
        self,
        source_name: StyleName,
        body: str,
        expected_entry: str,
    ) -> None:
        source_style = get_style(source_name)
        sphinx_style = get_style('sphinx')
        sphinx_regions = convert_regions(
            _regions_from(body, source_name),
            source_style,
            sphinx_style,
        )
        assert [region.text for region in sphinx_regions if region.kind == 'entry'] == [
            ':ivar attr: Value.',
            ':vartype attr: float',
        ]

        sphinx_body = '\n'.join(region.text for region in sphinx_regions)
        round_trip = convert_regions(
            _regions_from(sphinx_body, 'sphinx'),
            sphinx_style,
            source_style,
        )
        assert [
            region.text
            for region in round_trip
            if region.kind == 'entry' and region.role == 'attributes'
        ] == [expected_entry]

    def test_yields_google_to_sphinx_stays_verbatim(self) -> None:
        source = 'Summary.\n\nYields:\n    str: Next item.'
        style_g = get_style('google')
        style_s = get_style('sphinx')
        regions = _regions_from(source, 'google')
        converted = convert_regions(regions, style_g, style_s)
        authored_span = [region for region in converted if region.preserve_verbatim]
        assert [(region.kind, region.text) for region in authored_span] == [
            ('section_header', 'Yields:'),
            ('entry', '    str: Next item.'),
        ]


class TestDuplicateTypelessYields:
    _SOURCE = (
        'def g():\n'
        '    """Summary.\n'
        '\n'
        '    :yields: First thing.\n'
        '    :yields: Second thing.\n'
        '    """\n'
    )

    def test_duplicate_returns_message_unchanged(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    :returns: First thing.\n'
            '    :returns: Second thing.\n'
            '    """\n'
        )
        report = wrap_source_report(source, config=DocwrapConfig(convert='google'))
        assert report.skips[0].message == 'target cannot represent multiple typeless returns'

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_duplicate_yields_skip_preserves_source(self, target: StyleName) -> None:
        report = wrap_source_report(self._SOURCE, config=DocwrapConfig(convert=target))
        assert [s.code for s in report.skips] == ['duplicate-field']
        assert report.skips[0].message == 'target cannot represent multiple typeless yields'
        assert report.source == self._SOURCE

    def test_single_yields_still_converts(self) -> None:
        source = 'def g():\n    """Summary.\n\n    :yields: Only thing.\n    """\n'
        out = wrap_source(source, config=DocwrapConfig(convert='google'))
        assert 'Yields:' in out


class TestGoogleMultiNameConversion:
    def test_numpy_multi_name_param_converts_to_reparseable_google_entries(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        source = (
            'def f(x1, x2, count):\n'
            '    """Summary.\n\n'
            '    Parameters\n'
            '    ----------\n'
            '    x1, x2 : array_like\n'
            '        The values.\n'
            '    count : int\n'
            '        The count.\n'
            '    """\n'
        )
        expected = (
            'def f(x1, x2, count):\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x1 (array_like): The values.\n'
            '        x2 (array_like): The values.\n'
            '        count (int): The count.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert='google')

        with caplog.at_level(logging.WARNING, logger='docwrap._styles._google'):
            first = wrap_source_report(source, config=config)
            second = wrap_source_report(first.source, config=config)

        reparsed = parse_docstrings(first.source, find_docstrings(first.source), config)
        param_matches = [
            region.entry_match
            for region in reparsed[0].regions
            if region.role == 'param' and region.entry_match is not None
        ]
        assert first.source == expected
        assert [(match.name, match.type_ann, match.description) for match in param_matches] == [
            ('x1', 'array_like', 'The values.'),
            ('x2', 'array_like', 'The values.'),
            ('count', 'int', 'The count.'),
        ]
        assert len(caplog.records) == 1
        assert 'shared description was duplicated and may need review' in caplog.text
        compile(first.source, '<test>', 'exec')
        assert first.converted == 1
        assert second.source == first.source
        assert second.converted == 0

    def test_numpy_other_param_comma_name_is_preserved_without_expansion(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        source = (
            'def f(value):\n'
            '    """Summary.\n\n'
            '    Other Parameters\n'
            '    ----------------\n'
            '    x1, x2 : array_like\n'
            '        The optional values.\n\n'
            '    Parameters\n'
            '    ----------\n'
            '    value : int\n'
            '        The value.\n'
            '    """\n'
        )
        preserved = (
            '    Other Parameters\n'
            '    ----------------\n'
            '    x1, x2 : array_like\n'
            '        The optional values.\n\n'
        )
        config = DocwrapConfig(convert='google')

        with caplog.at_level(logging.WARNING, logger='docwrap._styles._google'):
            first = wrap_source_report(source, config=config)
            second = wrap_source_report(first.source, config=config)

        assert preserved in first.source
        assert '        x1 (array_like):' not in first.source
        assert '        x2 (array_like):' not in first.source
        assert caplog.records == []
        compile(first.source, '<test>', 'exec')
        assert first.converted == 1
        assert second.source == first.source
        assert second.converted == 0


class TestGroupSphinxEntries:
    def test_canonical_order(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:raises ValueError: bad\n:returns: result\n:param x: val',
            'sphinx',
        )
        style_g = get_style('google')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_g)
        headers = [r for r in grouped if r.kind == 'section_header']
        header_texts = [r.text for r in headers]
        args_idx = header_texts.index('Args:')
        returns_idx = header_texts.index('Returns:')
        raises_idx = header_texts.index('Raises:')
        assert args_idx < returns_idx < raises_idx

    def test_convertible_out_of_order_entries_keep_group_and_relative_order(self) -> None:
        regions = _regions_from(
            'Summary.\n\n'
            ':raises ValueError: First error.\n'
            ':param z: Z value.\n'
            ':returns: Result.\n'
            ':param a: A value.\n'
            ':raises TypeError: Second error.',
            'sphinx',
        )

        grouped = _group_sphinx_entries(regions, get_style('sphinx'), get_style('google'))

        assert [region.text for region in grouped if region.kind == 'section_header'] == [
            'Args:',
            'Returns:',
            'Raises:',
        ]
        assert [
            (
                region.entry_match.role,
                region.entry_match.name,
                region.entry_match.description,
            )
            for region in grouped
            if region.kind == 'entry' and region.entry_match is not None
        ] == [
            ('param', 'z', 'Z value.'),
            ('param', 'a', 'A value.'),
            ('returns', '', 'Result.'),
            ('raises', 'ValueError', 'First error.'),
            ('raises', 'TypeError', 'Second error.'),
        ]

    def test_interleaved_params_grouped(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:param a: First.\n:returns: The result.\n:param b: Second.',
            'sphinx',
        )
        style_g = get_style('google')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_g)
        headers = [r for r in grouped if r.kind == 'section_header']
        header_texts = [r.text for r in headers]
        assert 'Args:' in header_texts
        assert 'Returns:' in header_texts

    def test_numpy_target_headers(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:param x: val\n:returns: result',
            'sphinx',
        )
        style_n = get_style('numpy')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_n)
        headers = [r for r in grouped if r.kind == 'section_header']
        header_texts = [r.text for r in headers]
        assert 'Parameters' in header_texts
        # NumPy headers include dashes line
        dashes = [r for r in grouped if r.text.startswith('---')]
        assert len(dashes) >= 1

    def test_pre_entry_regions_at_start(self) -> None:
        regions = _regions_from(
            'Summary.\n\nSome description.\n\n:param x: val',
            'sphinx',
        )
        style_g = get_style('google')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_g)
        # Summary and description should precede any section_header
        first_header_idx = next(i for i, r in enumerate(grouped) if r.kind == 'section_header')
        for r in grouped[:first_header_idx]:
            assert r.kind in ('summary', 'blank', 'description')

    def test_reachable_order_and_attachments(self) -> None:
        source = textwrap.dedent('''\
            def f(value, **kwargs):
                """Summary.

                :ivar state: Current state.
                :vartype state: str
                :warns RuntimeWarning: Slow operation.
                :raises ValueError: Invalid value.
                :yields: Next item.
                :returns: Computed result.
                :rtype: int
                :keyword bool debug: Debug flag.
                :param int value: Positional value.
                """
        ''')
        expected = textwrap.dedent('''\
            def f(value, **kwargs):
                """Summary.

                Args:
                    value (int): Positional value.

                Keyword Args:
                    debug (bool): Debug flag.

                Returns:
                    int: Computed result.

                Yields:
                    Next item.

                Raises:
                    ValueError: Invalid value.

                Warns:
                    RuntimeWarning: Slow operation.

                Attributes:
                    state (str): Current state.
                """
        ''')

        report = wrap_source_report(source, config=DocwrapConfig(convert='google'))

        assert report.source == expected

    def test_same_style_sphinx_keeps_multiple_typeless_returns(self) -> None:
        style = get_style('sphinx')
        converted = convert_regions(
            _regions_from('Summary.\n\n:returns: First.\n:returns: Second.', 'sphinx'),
            style,
            style,
        )

        assert [
            region.text
            for region in converted
            if region.kind == 'entry' and region.role == 'returns'
        ] == [':returns: First.', ':returns: Second.']

    def test_section_header_has_section_name(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:param x: val',
            'sphinx',
        )
        style_g = get_style('google')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_g)
        headers = [r for r in grouped if r.kind == 'section_header']
        assert len(headers) == 1
        assert headers[0].section_name is not None

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_single_typeless_return_remains_convertible(self, target: StyleName) -> None:
        converted = convert_regions(
            _regions_from('Summary.\n\n:returns: Result.', 'sphinx'),
            get_style('sphinx'),
            get_style(target),
        )

        assert any(region.kind == 'section_header' for region in converted)
        assert [region.text for region in converted if region.kind == 'prose'] == ['Result.']

    def test_sphinx_inter_field_blank_preserved(self) -> None:
        source = 'Summary.\n\n:param x: X value.\n\n:param y: Y value.'
        style_s = get_style('sphinx')
        style_g = get_style('google')
        regions = _regions_from(source, style_s.name)
        grouped = _group_sphinx_entries(regions, style_s, style_g)
        # Both params must survive
        entries = [r for r in grouped if r.kind == 'entry']
        assert len(entries) == 2
        # The inter-field blank must also survive (2 blanks in, 2 out)
        blanks_in = [r for r in regions if r.kind == 'blank']
        blanks_out = [r for r in grouped if r.kind == 'blank']
        assert len(blanks_out) == len(blanks_in)

    def test_sphinx_inter_field_prose_preserved(self) -> None:
        source = textwrap.dedent('''\
            def f(x, y):
                """Summary.

                :param x: X value.

                Note about params.

                :param y: Y value.
                """
        ''')
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)
        assert 'Note about params' in report.source

    def test_unmapped_roles_at_end(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:param x: val\n:type y: int',
            'sphinx',
        )
        # After _merge_sphinx_types, orphaned :type y: stays (no matching param)
        merged = _merge_sphinx_types(regions)
        style_g = get_style('google')
        grouped = _group_sphinx_entries(merged, get_style('sphinx'), style_g)
        entries = [r for r in grouped if r.kind == 'entry']
        # The param entry should come first, orphaned type at the end
        assert entries[0].entry_match is not None
        assert entries[0].entry_match.name == 'x'
        assert len(entries) == 2

    def test_within_group_order_preserved(self) -> None:
        regions = _regions_from(
            'Summary.\n\n:param z: Last.\n:param a: First.',
            'sphinx',
        )
        style_g = get_style('google')
        grouped = _group_sphinx_entries(regions, get_style('sphinx'), style_g)
        entries = [r for r in grouped if r.kind == 'entry']
        assert entries[0].entry_match is not None
        assert entries[0].entry_match.name == 'z'
        assert entries[1].entry_match is not None
        assert entries[1].entry_match.name == 'a'


class TestKeywordFieldPreservation:
    _SOURCE = (
        'def f(**kwargs):\n'
        '    """Summary.\n'
        '\n'
        '    :param base: The base value.\n'
        '    :keyword optional: An optional keyword setting.\n'
        '    """\n'
    )

    def test_google_target_still_converts_keyword(self) -> None:
        out = wrap_source(self._SOURCE, config=DocwrapConfig(convert='google'))
        assert 'Keyword Args:' in out
        assert ':keyword' not in out

    @pytest.mark.parametrize(
        'field',
        [
            pytest.param(':keyword int optional: Setting.', id='keyword-p10'),
            pytest.param(':var int value: Setting.', id='var-control'),
        ],
    )
    def test_inline_typed_opaque_field_stays_verbatim_under_strip_types(
        self,
        field: str,
    ) -> None:
        source = f'def f():\n    """Summary.\n\n    {field}\n    """\n'
        config = DocwrapConfig(convert='numpy', strip_types=True)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(f'Summary.\n\n{field}', 'sphinx'),
            get_style('sphinx'),
            get_style('numpy'),
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert first.source == source
        assert second.source == first.source
        assert [
            (region.text, region.preserve_verbatim)
            for region in converted
            if region.kind == 'entry'
        ] == [(field, True)]
        assert deleted_text == []

    def test_keyword_preserved_verbatim_in_numpy_target(self) -> None:
        out = wrap_source(self._SOURCE, config=DocwrapConfig(convert='numpy'))
        assert ':keyword optional: An optional keyword setting.' in out
        assert 'Parameters' in out
        assert '\n    optional\n' not in out

    def test_keyword_round_trip_keeps_keyword_role(self) -> None:
        to_numpy = wrap_source(self._SOURCE, config=DocwrapConfig(convert='numpy'))
        back = wrap_source(to_numpy, config=DocwrapConfig(convert='sphinx'))
        assert ':keyword optional:' in back
        assert ':param optional:' not in back

    def test_numpy_target_fixed_point(self) -> None:
        once = wrap_source(self._SOURCE, config=DocwrapConfig(convert='numpy'))
        assert wrap_source(once, config=DocwrapConfig(convert='numpy')) == once

    def test_same_name_parameter_keeps_conflict_skip(self) -> None:
        source = (
            'def f(optional, **kwargs):\n'
            '    """Summary.\n'
            '\n'
            '    :param optional: The positional value.\n'
            '    :keyword optional: An optional keyword setting.\n'
            '    :type optional: int\n'
            '    """\n'
        )
        report = wrap_source_report(source, config=DocwrapConfig(convert='numpy'))
        assert [skip.code for skip in report.skips] == ['duplicate-field']
        assert report.source == source

    def test_typed_keyword_keeps_type_field(self) -> None:
        source = (
            'def f(**kwargs):\n'
            '    """Summary.\n'
            '\n'
            '    :keyword optional: An optional keyword setting.\n'
            '    :type optional: int\n'
            '    """\n'
        )
        out = wrap_source(source, config=DocwrapConfig(convert='numpy'))
        assert ':keyword optional: An optional keyword setting.' in out
        assert ':type optional: int' in out

    def test_typed_keyword_strip_types_deletes_type_field(self) -> None:
        source = (
            'def f(**kwargs):\n'
            '    """Summary.\n'
            '\n'
            '    :keyword optional: An optional keyword setting.\n'
            '    :type optional: int\n'
            '    """\n'
        )
        out = wrap_source(
            source,
            config=DocwrapConfig(convert='numpy', strip_types=True),
        )
        assert ':keyword optional: An optional keyword setting.' in out
        assert ':type optional: int' not in out


class TestMergeSphinxTypes:
    @pytest.mark.parametrize(
        ('body', 'match'),
        [
            pytest.param(
                'Summary.\n\n:param x: desc\n:type x: int\n:type x: str',
                'duplicate :type x:',
                id='duplicate-type',
            ),
            pytest.param(
                'Summary.\n\n:returns: desc\n:rtype: int\n:rtype: str',
                'duplicate :rtype:',
                id='duplicate-rtype',
            ),
            pytest.param(
                'Summary.\n\n:ivar x: desc\n:vartype x: int\n:vartype x: str',
                'duplicate :vartype x:',
                id='duplicate-vartype',
            ),
            pytest.param(
                'Summary.\n\n:param x: First.\n:param x: Second.\n:type x: int',
                'more than one',
                id='fanout-type',
            ),
            pytest.param(
                'Summary.\n\n:returns: First.\n:returns: Second.\n:rtype: int',
                'more than one',
                id='fanout-rtype',
            ),
            pytest.param(
                'Summary.\n\n:param int x: desc\n:type x: str',
                'inline',
                id='overwrite-type',
            ),
            pytest.param(
                'Summary.\n\n:ivar int x: desc\n:vartype x: str',
                'inline',
                id='overwrite-vartype',
            ),
        ],
    )
    def test_nonconforming_merge_raises(self, body: str, match: str) -> None:
        regions = _regions_from(body, 'sphinx')
        with pytest.raises(DuplicateFieldError, match=match):
            _merge_sphinx_types(regions)

    def test_orphan_duplicate_type_fields_preserved(self) -> None:
        regions = _regions_from('Summary.\n\n:type x: int\n:type x: str', 'sphinx')
        merged = _merge_sphinx_types(regions)
        types = [r for r in merged if r.kind == 'entry' and r.role == 'type']
        assert len(types) == 2

    def test_orphaned_rtype_synthesizes_returns(self) -> None:
        regions = _regions_from('Summary.\n\n:rtype: int', 'sphinx')
        merged = _merge_sphinx_types(regions)
        returns = [r for r in merged if r.kind == 'entry' and r.role == 'returns']
        assert len(returns) == 1
        assert returns[0].entry_match is not None
        assert returns[0].entry_match.type_ann == 'int'

    def test_orphaned_type_preserved(self) -> None:
        regions = _regions_from('Summary.\n\n:type x: int', 'sphinx')
        merged = _merge_sphinx_types(regions)
        types = [r for r in merged if r.kind == 'entry' and r.role == 'type']
        assert len(types) == 1

    def test_rtype_merged_into_returns(self) -> None:
        regions = _regions_from('Summary.\n\n:returns: The count.\n:rtype: int', 'sphinx')
        merged = _merge_sphinx_types(regions)
        returns = [r for r in merged if r.kind == 'entry' and r.role == 'returns']
        assert len(returns) == 1
        assert returns[0].entry_match is not None
        assert returns[0].entry_match.type_ann == 'int'
        rtypes = [r for r in merged if r.kind == 'entry' and r.role == 'rtype']
        assert len(rtypes) == 0

    def test_type_merged_into_param(self) -> None:
        regions = _regions_from('Summary.\n\n:param x: The value.\n:type x: int', 'sphinx')
        merged = _merge_sphinx_types(regions)
        params = [r for r in merged if r.kind == 'entry' and r.role == 'param']
        assert len(params) == 1
        assert params[0].entry_match is not None
        assert params[0].entry_match.type_ann == 'int'
        # :type: region should be removed
        types = [r for r in merged if r.kind == 'entry' and r.role == 'type']
        assert len(types) == 0

    def test_vartype_merged_into_ivar(self) -> None:
        regions = _regions_from('Summary.\n\n:ivar x: The field.\n:vartype x: str', 'sphinx')
        merged = _merge_sphinx_types(regions)
        ivars = [r for r in merged if r.kind == 'entry' and r.role == 'ivar']
        assert len(ivars) == 1
        assert ivars[0].entry_match is not None
        assert ivars[0].entry_match.type_ann == 'str'
        vartypes = [r for r in merged if r.kind == 'entry' and r.role == 'vartype']
        assert len(vartypes) == 0

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    def test_vartype_not_merged_into_non_ivar(self, role: str) -> None:
        regions = _regions_from(
            f'Summary.\n\n:{role} x: The field.\n:vartype x: str',
            'sphinx',
        )
        merged = _merge_sphinx_types(regions)

        fields = [r for r in merged if r.kind == 'entry' and r.role == role]
        assert len(fields) == 1
        assert fields[0].entry_match is not None
        assert fields[0].entry_match.type_ann == ''
        vartypes = [r for r in merged if r.kind == 'entry' and r.role == 'vartype']
        assert len(vartypes) == 1
        assert vartypes[0].text == ':vartype x: str'


class TestMultiNameAttributePreservation:
    _LONG_ATTRIBUTE_SPAN = (
        '    x, y : int\n'
        '        Shared description that is deliberately long enough to wrap on a later pass when '
        'the preserved span is no longer opaque to the target parser.\n'
    )
    _LONG_SOURCE = (
        'class C:\n'
        '    """Summary.\n'
        '\n'
        '    Attributes\n'
        '    ----------\n' + _LONG_ATTRIBUTE_SPAN + '\n'
        '    Parameters\n'
        '    ----------\n'
        '    a : str\n'
        '        Neighbor parameter.\n'
        '    """\n'
    )
    _SOURCE = (
        'class C:\n'
        '    """Summary.\n'
        '\n'
        '    Attributes\n'
        '    ----------\n'
        '    x, y : int\n'
        '        Shared description.\n'
        '\n'
        '    Parameters\n'
        '    ----------\n'
        '    a, b : str\n'
        '        Parameter pair.\n'
        '    """\n'
    )

    @pytest.mark.parametrize('target_name', ['google', 'sphinx'])
    def test_complete_authored_span_preserves_deletion_provenance(
        self,
        target_name: StyleName,
    ) -> None:
        body = (
            'Summary.\n\nAttributes\n----------\nx, y : int\n    Shared authored continuation.\n'
        )
        deleted_text: list[str] = []

        converted = convert_regions(
            _regions_from(body, 'numpy'),
            get_style('numpy'),
            get_style(target_name),
            strip_types=True,
            deleted_text=deleted_text,
        )
        preserved = [region for region in converted if region.preserve_verbatim]

        assert [region.original_lines for region in preserved] == [
            ('x, y : int', '    Shared authored continuation.')
        ]
        assert deleted_text == []

    def test_google_long_attribute_span_is_fixed_and_onward_opaque(self) -> None:
        config = DocwrapConfig(convert='google', line_length=72)

        first = wrap_source(self._LONG_SOURCE, config=config)
        second = wrap_source(first, config=config)
        onward = wrap_source(first, config=DocwrapConfig(convert='sphinx', line_length=72))

        assert self._LONG_ATTRIBUTE_SPAN in first
        assert second == first
        assert self._LONG_ATTRIBUTE_SPAN in onward
        assert ':ivar' not in onward
        assert ':vartype' not in onward
        assert ':param str a: Neighbor parameter.' in onward

    def test_google_target_fixed_point(self) -> None:
        once = wrap_source(self._SOURCE, config=DocwrapConfig(convert='google'))
        assert wrap_source(once, config=DocwrapConfig(convert='google')) == once

    def test_multi_name_attributes_preserved_to_google(self) -> None:
        out = wrap_source(self._SOURCE, config=DocwrapConfig(convert='google'))
        assert 'x, y : int' in out
        assert 'x, y (int):' not in out

    def test_multi_name_attributes_preserved_to_sphinx(self) -> None:
        config = DocwrapConfig(convert='sphinx', line_length=72)

        first = wrap_source(self._LONG_SOURCE, config=config)
        second = wrap_source(first, config=config)
        onward = wrap_source(first, config=DocwrapConfig(convert='google', line_length=72))

        assert self._LONG_ATTRIBUTE_SPAN in first
        assert ':ivar' not in first
        assert ':vartype' not in first
        assert ':param str a: Neighbor parameter.' in first
        assert second == first
        assert self._LONG_ATTRIBUTE_SPAN in onward
        assert 'x, y (int):' not in onward
        assert 'a (str): Neighbor parameter.' in onward

    def test_numpy_target_keeps_multi_name_grammar(self) -> None:
        source = (
            'class C:\n'
            '    """Summary.\n'
            '\n'
            '    Attributes:\n'
            '        x (int): Single description.\n'
            '    """\n'
        )
        out = wrap_source(source, config=DocwrapConfig(convert='numpy'))
        assert 'x : int' in out

    def test_onward_conversion_keeps_preserved_attribute_span_opaque(self) -> None:
        to_google = wrap_source(self._SOURCE, config=DocwrapConfig(convert='google'))
        onward = wrap_source(to_google, config=DocwrapConfig(convert='sphinx'))
        google_span = to_google.split('\n\n    Args:', maxsplit=1)[0].split(
            '    Attributes:',
            maxsplit=1,
        )[1]
        onward_span = onward.split('\n\n    :param', maxsplit=1)[0].split(
            '    Attributes:',
            maxsplit=1,
        )[1]

        assert 'x, y : int' in onward_span
        assert onward_span == google_span
        assert ':ivar' not in onward
        assert ':param str a: Parameter pair.' in onward
        assert ':param str b: Parameter pair.' in onward

    def test_parameter_split_unchanged(self) -> None:
        out = wrap_source(self._SOURCE, config=DocwrapConfig(convert='google'))
        assert 'a (str): Parameter pair.' in out
        assert 'b (str): Parameter pair.' in out

    def test_single_name_attribute_still_converts(self) -> None:
        source = (
            'class C:\n'
            '    """Summary.\n'
            '\n'
            '    Attributes\n'
            '    ----------\n'
            '    x : int\n'
            '        Single description.\n'
            '    """\n'
        )
        out = wrap_source(source, config=DocwrapConfig(convert='google'))
        assert 'x (int): Single description.' in out


class TestStripTypesPreservesOriginalLines:
    def test_strip_types_reflow_none_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): First sentence.
                        Second sentence.
                \"\"\"
        """)
        config = DocwrapConfig(reflow='none', strip_types=True)
        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        assert first == second

    def test_strip_types_reflow_none_no_description(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int):
                \"\"\"
        """)
        config = DocwrapConfig(reflow='none', strip_types=True)
        result = wrap_source(source, config=config)
        assert '(int)' not in result
        assert 'x:' in result

    def test_strip_types_reflow_none_no_type_passthrough(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x: First sentence.
                        Second sentence.
                \"\"\"
        """)
        config = DocwrapConfig(reflow='none', strip_types=True)
        result = wrap_source(source, config=config)
        assert result == source

    def test_strip_types_reflow_none_preserves_continuation(self) -> None:
        source = textwrap.dedent("""\
            def f(x):
                \"\"\"Summary.

                Args:
                    x (int): First sentence.
                        Second sentence.
                \"\"\"
        """)
        config = DocwrapConfig(reflow='none', strip_types=True)
        report = wrap_source_report(source, config=config)
        # Type stripped but lines NOT joined
        assert 'x:' in report.source
        assert '(int)' not in report.source
        assert 'First sentence.\n' in report.source
        assert 'Second sentence.' in report.source
        # Should NOT be joined into one line
        assert 'First sentence. Second sentence.' not in report.source


class TestStripTypesStandalone:
    def test_deletion_free_empty_section_is_kept(self) -> None:
        style = get_style('google')
        converted = convert_regions(
            _regions_from('Summary.\n\nReturns:', 'google'), style, style, strip_types=True
        )

        kinds = [(region.kind, region.text) for region in converted]
        assert ('section_header', 'Returns:') in kinds

    def test_google_param_types_stripped(self) -> None:
        source = 'Summary.\n\nArgs:\n    x (int): The value.'
        style_g = get_style('google')
        converted = convert_regions(
            _regions_from(source, 'google'), style_g, style_g, strip_types=True
        )
        entries = [r for r in converted if r.kind == 'entry']
        assert any('x:' in r.text and 'int' not in r.text for r in entries)
        assert [r.role for r in entries] == ['param']

    def test_google_returns_becomes_prose(self) -> None:
        source = 'Summary.\n\nReturns:\n    int: The count.'
        style_g = get_style('google')
        converted = convert_regions(
            _regions_from(source, 'google'), style_g, style_g, strip_types=True
        )
        # The returns entry should become prose (no type marker)
        prose = [r for r in converted if r.kind == 'prose']
        assert any('The count' in r.text for r in prose)
        # No entry with 'int' should remain
        entries = [r for r in converted if r.kind == 'entry']
        assert not any('int:' in r.text for r in entries)

    def test_google_returns_no_desc_skipped(self) -> None:
        source = 'Summary.\n\nReturns:\n    int:'
        style_g = get_style('google')
        converted = convert_regions(
            _regions_from(source, 'google'), style_g, style_g, strip_types=True
        )
        entries = [r for r in converted if r.kind == 'entry']
        prose = [r for r in converted if r.kind == 'prose']
        assert not any('int' in r.text for r in entries)
        assert len(prose) == 0

    def test_google_yields_becomes_prose(self) -> None:
        source = 'Summary.\n\nYields:\n    str: Next item.'
        style_g = get_style('google')
        converted = convert_regions(
            _regions_from(source, 'google'), style_g, style_g, strip_types=True
        )
        prose = [r for r in converted if r.kind == 'prose']
        assert any('Next item' in r.text for r in prose)

    @pytest.mark.parametrize('source_style', SOURCE_STYLES)
    def test_named_parameter_does_not_trigger_return_skip(
        self,
        source_style: StyleName,
    ) -> None:
        if source_style == 'google':
            source = textwrap.dedent('''\
                def f(x):
                    """Summary.

                    Args:
                        x (bytes): Typed parameter.
                    """
            ''')
        else:
            source = textwrap.dedent('''\
                def f(x):
                    """Summary.

                    Parameters
                    ----------
                    x : bytes
                        Typed parameter.
                    """
            ''')

        first = wrap_source_report(source, config=DocwrapConfig(strip_types=True))
        second = wrap_source_report(first.source, config=DocwrapConfig(strip_types=True))

        assert first.skips == ()
        assert 'bytes' not in first.source
        assert 'Typed parameter.' in first.source
        assert second.source == first.source

    def test_named_return_type_raises_dedicated_error(self) -> None:
        style = get_style('google')

        with pytest.raises(UnrepresentableReturnError) as caught:
            convert_regions(
                _regions_from(
                    'Summary.\n\nReturns:\n    result (int): Named result.',
                    'google',
                ),
                style,
                style,
                strip_types=True,
            )

        assert 'Returns' in str(caught.value)
        assert "'result'" in str(caught.value)
        assert 'type' in str(caught.value)

    @pytest.mark.parametrize('target', TARGETS)
    @pytest.mark.parametrize('role', ROLES)
    @pytest.mark.parametrize('source_style', SOURCE_STYLES)
    def test_named_return_type_skips_whole_docstring(
        self,
        source_style: StyleName,
        role: CanonicalRole,
        target: StyleName | None,
    ) -> None:
        header = role.title()
        if source_style == 'google':
            source = textwrap.dedent(f'''\
                def f(x):
                    """Summary.

                    Args:
                        x (bytes): Typed parameter.

                    {header}:
                        result (int): Named result.
                    """
            ''')
            typed_parameter = 'x (bytes)'
        else:
            source = textwrap.dedent(f'''\
                def f(x):
                    """Summary.

                    Parameters
                    ----------
                    x : bytes
                        Typed parameter.

                    {header}
                    {'-' * len(header)}
                    result : int
                        Named result.
                    """
            ''')
            typed_parameter = 'x : bytes'
        config = DocwrapConfig(convert=target, strip_types=True)
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        if target == 'sphinx' and role == 'yields':
            assert first.source != source
            assert first.converted == 1
            assert first.skips == ()
            assert header in first.source
            assert 'result (int)' in first.source or 'result : int' in first.source
            assert typed_parameter not in first.source
            assert second.source == first.source
            assert second.converted == 0
            assert second.skips == ()
            return

        assert first.source == source
        assert first.converted == 0
        assert [skip.code for skip in first.skips] == ['unrepresentable-return']
        assert header in first.skips[0].message
        assert "'result'" in first.skips[0].message
        if target == 'sphinx' and role == 'returns':
            assert 'sphinx' in first.skips[0].message.lower()
        else:
            assert 'type' in first.skips[0].message
        assert typed_parameter in first.source
        assert second.source == source
        assert second.converted == 0
        assert second.skips == first.skips

    def test_numpy_param_types_stripped(self) -> None:
        source = 'Summary.\n\nParameters\n----------\nx : int\n    The value.'
        style_n = get_style('numpy')
        converted = convert_regions(
            _regions_from(source, 'numpy'), style_n, style_n, strip_types=True
        )
        entries = [r for r in converted if r.kind == 'entry']
        assert any('x' in r.text and 'int' not in r.text for r in entries)

    def test_numpy_returns_becomes_prose(self) -> None:
        source = 'Summary.\n\nReturns\n-------\nint\n    The count.'
        style_n = get_style('numpy')
        converted = convert_regions(
            _regions_from(source, 'numpy'), style_n, style_n, strip_types=True
        )
        prose = [r for r in converted if r.kind == 'prose']
        assert any('The count' in r.text for r in prose)

    def test_numpy_stripped_returns_indent(self) -> None:
        source = textwrap.dedent('''\
            def f():
                """Summary.

                Returns:
                    int: The count.
                """
        ''')
        config = DocwrapConfig(convert='numpy', strip_types=True)
        report = wrap_source_report(source, config=config)
        # The description should be indented at 4 spaces relative to section body
        lines = report.source.split('\n')
        desc_lines = [ln for ln in lines if 'The count' in ln]
        assert desc_lines
        for ln in desc_lines:
            stripped = ln.lstrip()
            indent = len(ln) - len(stripped)
            # Should be base_indent (8 for method body) + entry_continuation (4) = 12
            # or at minimum not at indent 0 relative to section
            assert indent >= 8, f'Indent too low ({indent}): {ln!r}'

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_orphan_rtype_conversion_leaves_no_empty_section(self, target_name: StyleName) -> None:
        style_s = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from('Summary.\n\n:rtype: Pattern' + BACKSLASH + 'x41', 'sphinx'),
            style_s,
            get_style(target_name),
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [('summary', 'Summary.')]
        assert deleted_text == [':rtype: Pattern' + BACKSLASH + 'x41']

    @pytest.mark.parametrize('role', ROLES)
    @pytest.mark.parametrize('source_style', SOURCE_STYLES)
    def test_return_description_only_does_not_trigger_skip(
        self,
        source_style: StyleName,
        role: CanonicalRole,
    ) -> None:
        header = role.title()
        if source_style == 'google':
            source = textwrap.dedent(f'''\
                def f():
                    """Summary.

                    {header}:
                        Description without an annotation.
                    """
            ''')
        else:
            source = textwrap.dedent(f'''\
                def f():
                    """Summary.

                    {header}
                    {'-' * len(header)}
                        Description without an annotation.
                    """
            ''')
        config = DocwrapConfig(strip_types=True)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.skips == ()
        assert 'Description without an annotation.' in first.source
        assert second.source == first.source

    @pytest.mark.parametrize('style_name', ['google', 'numpy'])
    @pytest.mark.parametrize('role', ['returns', 'yields'])
    def test_return_like_prose_retains_formatter_indentation(
        self,
        style_name: StyleName,
        role: CanonicalRole,
    ) -> None:
        style = get_style(style_name)
        entry = Region(
            kind='entry',
            text='source entry',
            entry_match=EntryMatch(
                name='int',
                type_ann='',
                description='First line.\n    Second line.',
                role=role,
            ),
            role=role,
        )

        converted = _strip_types_entry(
            entry,
            style,
            style,
            strip_types=True,
            deleted_text=[],
        )

        assert converted is not None
        regions, entry_deleted = converted
        assert not entry_deleted
        _assert_generated_prose(regions, ['    First line.', '    Second line.'])

    @pytest.mark.parametrize('role', ROLES)
    @pytest.mark.parametrize('source_style', SOURCE_STYLES)
    def test_return_single_slot_type_does_not_trigger_skip(
        self,
        source_style: StyleName,
        role: CanonicalRole,
    ) -> None:
        header = role.title()
        if source_style == 'google':
            source = textwrap.dedent(f'''\
                def f():
                    """Summary.

                    {header}:
                        int: Typed result.
                    """
            ''')
        else:
            source = textwrap.dedent(f'''\
                def f():
                    """Summary.

                    {header}
                    {'-' * len(header)}
                    int
                        Typed result.
                    """
            ''')
        config = DocwrapConfig(strip_types=True)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.skips == ()
        assert 'int' not in first.source
        assert 'Typed result.' in first.source
        assert second.source == first.source

    @pytest.mark.parametrize(
        ('field', 'target_name', 'expected_deleted'),
        [
            pytest.param(
                ':type x: Pattern' + BACKSLASH + 'x41',
                None,
                ':type x: Pattern' + BACKSLASH + 'x41',
                id='type-no-conversion',
            ),
            pytest.param(
                ':type x: Pattern' + BACKSLASH + 'x41',
                'google',
                ':type x: Pattern' + BACKSLASH + 'x41',
                id='type-google',
            ),
            pytest.param(
                ':type x: Pattern' + BACKSLASH + 'x41',
                'numpy',
                ':type x: Pattern' + BACKSLASH + 'x41',
                id='type-numpy',
            ),
            pytest.param(
                ':rtype: Pattern' + BACKSLASH + 'x41',
                None,
                ':rtype: Pattern' + BACKSLASH + 'x41',
                id='rtype-no-conversion',
            ),
            pytest.param(
                ':rtype: Pattern' + BACKSLASH + 'x41',
                'google',
                ':rtype: Pattern' + BACKSLASH + 'x41',
                id='rtype-google',
            ),
            pytest.param(
                ':rtype: Pattern' + BACKSLASH + 'x41',
                'numpy',
                ':rtype: Pattern' + BACKSLASH + 'x41',
                id='rtype-numpy',
            ),
            pytest.param(
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                None,
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                id='vartype-no-conversion',
            ),
            pytest.param(
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                'google',
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                id='vartype-google',
            ),
            pytest.param(
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                'numpy',
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                id='vartype-numpy',
            ),
        ],
    )
    def test_sphinx_deleted_field_keeps_one_prose_separator(
        self,
        field: str,
        target_name: StyleName | None,
        expected_deleted: str,
    ) -> None:
        style_s = get_style('sphinx')
        target_style = style_s if target_name is None else get_style(target_name)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n' + field + '\n\nAuthored trailing prose remains here.',
                'sphinx',
            ),
            style_s,
            target_style,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [
            ('summary', 'Summary.'),
            ('blank', ''),
            ('description', 'Authored trailing prose remains here.'),
        ]
        assert deleted_text == [expected_deleted]
        assert converted[-1].text == 'Authored trailing prose remains here.'
        assert sum(region.kind == 'blank' for region in converted) == 1

    def test_sphinx_inline_type_stripped(self) -> None:
        source = 'Summary.\n\n:param int x: The value.'
        style_s = get_style('sphinx')
        converted = convert_regions(
            _regions_from(source, 'sphinx'), style_s, style_s, strip_types=True
        )
        entries = [r for r in converted if r.kind == 'entry']
        assert any(':param x:' in r.text for r in entries)
        assert not any('int' in r.text for r in entries)

    @pytest.mark.parametrize(
        ('field', 'expected_deleted'),
        [
            pytest.param(
                ':type x: Pattern' + BACKSLASH + 'x41',
                ':type x: Pattern' + BACKSLASH + 'x41',
                id='type',
            ),
            pytest.param(
                ':rtype: Pattern' + BACKSLASH + 'x41',
                ':rtype: Pattern' + BACKSLASH + 'x41',
                id='rtype',
            ),
            pytest.param(
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                ':vartype x: Pattern' + BACKSLASH + 'x41',
                id='vartype',
            ),
        ],
    )
    def test_sphinx_orphan_type_field_drops_separator(
        self, field: str, expected_deleted: str
    ) -> None:
        style_s = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from('Summary.\n\n' + field, 'sphinx'),
            style_s,
            style_s,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [('summary', 'Summary.')]
        assert deleted_text == [expected_deleted]

    def test_sphinx_rtype_only_dropped(self) -> None:
        source = 'Summary.\n\n:rtype: int'
        style_s = get_style('sphinx')
        converted = convert_regions(
            _regions_from(source, 'sphinx'), style_s, style_s, strip_types=True
        )
        # After merge, orphan :rtype: becomes a returns entry with type_ann='int'
        # and no description. With strip_types, this should be dropped.
        entries = [r for r in converted if r.kind == 'entry']
        assert not any('int' in r.text for r in entries)

    def test_sphinx_rtype_region_dropped(self) -> None:
        source = 'Summary.\n\n:returns: The count.\n:rtype: int'
        style_s = get_style('sphinx')
        converted = convert_regions(
            _regions_from(source, 'sphinx'), style_s, style_s, strip_types=True
        )
        # :rtype: should be gone
        rtypes = [r for r in converted if r.role == 'rtype']
        assert len(rtypes) == 0
        # :returns: should still be present
        returns = [r for r in converted if r.kind == 'entry']
        assert any(':returns: The count.' in r.text for r in returns)

    def test_sphinx_type_regions_dropped(self) -> None:
        source = 'Summary.\n\n:param x: val\n:type x: int'
        style_s = get_style('sphinx')
        converted = convert_regions(
            _regions_from(source, 'sphinx'), style_s, style_s, strip_types=True
        )
        types = [r for r in converted if r.role == 'type']
        assert len(types) == 0

    def test_sphinx_vartype_region_dropped(self) -> None:
        source = 'Summary.\n\n:ivar x: The field.\n:vartype x: str'
        style_s = get_style('sphinx')
        converted = convert_regions(
            _regions_from(source, 'sphinx'), style_s, style_s, strip_types=True
        )
        vartypes = [r for r in converted if r.role == 'vartype']
        assert len(vartypes) == 0

    def test_strip_types_convert_sphinx_preserves_typed_yields(self) -> None:
        source = textwrap.dedent('''\
            def f():
                """Summary.

                Yields:
                    str: Each line.
                """
        ''')
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert 'Yields:' in report.source
        assert 'str: Each line.' in report.source
        assert ':yields:' not in report.source

    def test_strip_types_convert_sphinx_returns_not_corrupted(self) -> None:
        source = textwrap.dedent('''\
            def f():
                """Summary.

                Returns:
                    int: The count.
                """
        ''')
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        report = wrap_source_report(source, config=config)
        assert ':returns: The count.' in report.source
        assert ':rtype:' not in report.source  # type stripped
        assert 'The count.' in report.source

    @pytest.mark.parametrize('target_name', [None, 'google', 'numpy'])
    @pytest.mark.parametrize(
        ('kept', 'first_field', 'second_field'),
        [
            pytest.param(
                ':param x: Kept.',
                ':type x: First' + BACKSLASH + 'x41',
                ':type x: Second' + BACKSLASH + 'x42',
                id='type',
            ),
            pytest.param(
                ':returns: Kept.',
                ':rtype: First' + BACKSLASH + 'x41',
                ':rtype: Second' + BACKSLASH + 'x42',
                id='rtype',
            ),
            pytest.param(
                ':ivar x: Kept.',
                ':vartype x: First' + BACKSLASH + 'x41',
                ':vartype x: Second' + BACKSLASH + 'x42',
                id='vartype',
            ),
        ],
    )
    def test_strip_types_deletes_duplicate_fields_with_provenance(
        self,
        kept: str,
        first_field: str,
        second_field: str,
        target_name: StyleName | None,
    ) -> None:
        style_s = get_style('sphinx')
        target_style = style_s if target_name is None else get_style(target_name)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(
                'Summary.\n\n' + kept + '\n' + first_field + '\n' + second_field,
                'sphinx',
            ),
            style_s,
            target_style,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert deleted_text == [first_field, second_field]
        joined = '\n'.join(region.text for region in converted)
        assert 'First' not in joined
        assert 'Second' not in joined
        assert 'Kept.' in joined

    def test_strip_types_deletes_orphan_duplicate_fields(self) -> None:
        style_s = get_style('sphinx')
        first = ':type x: First' + BACKSLASH + 'x41'
        second = ':type x: Second' + BACKSLASH + 'x42'
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from('Summary.\n\n' + first + '\n' + second, 'sphinx'),
            style_s,
            style_s,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [('summary', 'Summary.')]
        assert deleted_text == [first, second]

    def test_strip_types_keeps_adjacent_deletion_free_same_name_header(self) -> None:
        style = get_style('google')
        body = 'Summary.\n\nReturns:\n    int:\nReturns:'
        converted = convert_regions(_regions_from(body, 'google'), style, style, strip_types=True)

        headers = [
            (region.kind, region.text) for region in converted if region.kind == 'section_header'
        ]
        assert headers == [('section_header', 'Returns:')]

    def test_strip_types_keeps_extra_authored_separator_at_deletion(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='blank', text=''),
            Region(
                kind='entry',
                text=':type x: Gone',
                entry_match=EntryMatch(name='x', type_ann='', description='Gone', role='type'),
                role='type',
            ),
            Region(kind='blank', text=''),
            Region(kind='description', text='Trailing prose.'),
        ]
        style_s = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            regions, style_s, style_s, strip_types=True, deleted_text=deleted_text
        )

        assert [(region.kind, region.text) for region in converted] == [
            ('summary', 'Summary.'),
            ('blank', ''),
            ('blank', ''),
            ('description', 'Trailing prose.'),
        ]
        assert deleted_text == [':type x: Gone']

    def test_strip_types_keeps_following_section_header(self) -> None:
        style = get_style('google')
        body = 'Summary.\n\nReturns:\n    int:\n\nRaises:\n    ValueError: When bad.'
        converted = convert_regions(_regions_from(body, 'google'), style, style, strip_types=True)

        kinds = [(region.kind, region.text) for region in converted]
        assert ('section_header', 'Returns:') not in kinds
        assert ('section_header', 'Raises:') in kinds
        assert sum(kind == 'blank' for kind, _ in kinds) == 1

    def test_strip_types_preserves_unrelated_adjacent_blanks(self) -> None:
        style_s = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            [
                Region(kind='summary', text='Summary.'),
                Region(kind='blank', text=''),
                Region(kind='blank', text=''),
                Region(kind='description', text='Authored trailing prose remains here.'),
            ],
            style_s,
            style_s,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [
            ('summary', 'Summary.'),
            ('blank', ''),
            ('blank', ''),
            ('description', 'Authored trailing prose remains here.'),
        ]
        assert deleted_text == []

    @pytest.mark.parametrize(
        ('source_name', 'target_name', 'body', 'expected_deleted'),
        [
            pytest.param(
                'google',
                None,
                'Summary.\n\nReturns:\n    int:',
                ['    int:', 'Returns:'],
                id='google-same',
            ),
            pytest.param(
                'google',
                'numpy',
                'Summary.\n\nReturns:\n    int:',
                ['    int:', 'Returns:'],
                id='google-numpy',
            ),
            pytest.param(
                'google',
                'sphinx',
                'Summary.\n\nReturns:\n    int:',
                ['    int:', 'Returns:'],
                id='google-sphinx',
            ),
            pytest.param(
                'numpy',
                None,
                'Summary.\n\nReturns\n-------\nint',
                ['int', 'Returns', '-------'],
                id='numpy-same',
            ),
            pytest.param(
                'numpy',
                'google',
                'Summary.\n\nReturns\n-------\nint',
                ['int', 'Returns', '-------'],
                id='numpy-google',
            ),
            pytest.param(
                'numpy',
                'sphinx',
                'Summary.\n\nReturns\n-------\nint',
                ['int', 'Returns', '-------'],
                id='numpy-sphinx',
            ),
        ],
    )
    def test_strip_types_removes_emptied_authored_header(
        self,
        source_name: StyleName,
        target_name: StyleName | None,
        body: str,
        expected_deleted: list[str],
    ) -> None:
        style = get_style(source_name)
        target_style = style if target_name is None else get_style(target_name)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(body, source_name),
            style,
            target_style,
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [('summary', 'Summary.')]
        assert deleted_text == expected_deleted

    def test_strip_types_removes_header_left_adjacent_to_next(self) -> None:
        regions = [
            Region(kind='summary', text='Summary.'),
            Region(kind='blank', text=''),
            Region(kind='section_header', text='Returns:', section_name='Returns'),
            Region(
                kind='entry',
                text='    int:',
                entry_match=EntryMatch(name='int', type_ann='', description='', role='returns'),
                role='returns',
            ),
            Region(kind='section_header', text='Raises:', section_name='Raises'),
            Region(
                kind='entry',
                text='    ValueError: When bad.',
                entry_match=EntryMatch(
                    name='ValueError', type_ann='', description='When bad.', role='raises'
                ),
                role='raises',
            ),
        ]
        style = get_style('google')
        converted = convert_regions(regions, style, style, strip_types=True)

        kinds = [(region.kind, region.text) for region in converted]
        assert ('section_header', 'Returns:') not in kinds
        assert ('section_header', 'Raises:') in kinds

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_strip_types_removes_matched_rtype_synthesized_header(
        self, target_name: StyleName
    ) -> None:
        style = get_style('sphinx')
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from('Summary.\n\n:returns:\n:rtype: int', 'sphinx'),
            style,
            get_style(target_name),
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [('summary', 'Summary.')]
        assert deleted_text == [':rtype: int', ':returns:']

    def test_strip_types_removes_only_first_adjacent_same_name_header(self) -> None:
        style = get_style('google')
        body = 'Summary.\n\nReturns:\n    int:\nReturns:\n    str: Kept.'
        converted = convert_regions(_regions_from(body, 'google'), style, style, strip_types=True)

        headers = [
            (region.kind, region.text) for region in converted if region.kind == 'section_header'
        ]
        assert headers == [('section_header', 'Returns:')]
        assert any(region.kind == 'prose' and region.text == 'Kept.' for region in converted)

    @pytest.mark.parametrize(
        ('source_name', 'body', 'expected_section'),
        [
            pytest.param(
                'google',
                'Summary.\n\nReturns:',
                (('section_header', 'Returns:'),),
                id='google',
            ),
            pytest.param(
                'numpy',
                'Summary.\n\nReturns\n-------',
                (('section_header', 'Returns'), ('section_header', '-------')),
                id='numpy',
            ),
        ],
    )
    def test_strip_types_target_sphinx_preserves_empty_section(
        self,
        source_name: StyleName,
        body: str,
        expected_section: tuple[tuple[str, str], ...],
    ) -> None:
        source_style = get_style(source_name)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(body, source_name),
            source_style,
            get_style('sphinx'),
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [
            ('summary', 'Summary.'),
            ('blank', ''),
            *expected_section,
        ]
        assert all(region.preserve_verbatim for region in converted[2:])
        assert deleted_text == []

    @pytest.mark.parametrize(
        ('source_name', 'body'),
        [
            pytest.param('google', 'Summary.\n\nReturns:\n    int: Kept.', id='google'),
            pytest.param('numpy', 'Summary.\n\nReturns\n-------\nint\n    Kept.', id='numpy'),
        ],
    )
    def test_strip_types_target_sphinx_surviving_section_reports_only_type(
        self, source_name: StyleName, body: str
    ) -> None:
        source_style = get_style(source_name)
        deleted_text: list[str] = []
        converted = convert_regions(
            _regions_from(body, source_name),
            source_style,
            get_style('sphinx'),
            strip_types=True,
            deleted_text=deleted_text,
        )

        assert [(region.kind, region.text) for region in converted] == [
            ('summary', 'Summary.'),
            ('blank', ''),
            ('entry', ':returns: Kept.'),
        ]
        assert deleted_text == ['int']


class TestUnderlineBoundaryConversionProvenance:
    def test_deleted_text_is_exact_with_structural_underline(self) -> None:
        content = 'Summary.\n\n:param x: Kept.\n:type x: int\n\nBefore.\n----------\nAfter.'
        norm, _ = normalize_content(content, source_margin='')
        lines = norm.split('\n')
        style = get_style('sphinx')
        title_line_count = match_opening_title(lines, style) or 0
        regions = parse_regions(
            norm,
            style=style,
            code_block_lines=find_code_block_lines(lines, start=title_line_count),
        )
        deleted_text: list[str] = []
        converted = convert_regions(
            regions,
            get_style('sphinx'),
            get_style('numpy'),
            strip_types=True,
            deleted_text=deleted_text,
        )
        assert deleted_text == [':type x: int']
        assert [
            (region.kind, region.text, region.original_lines)
            for region in converted
            if '----------' in region.text
        ] == [
            ('section_header', '----------', None),
            ('description', '----------', ('----------',)),
        ]


class TestUnmappedFieldAssociation:
    ORPHAN_FIRST = (
        'def f(x):\n'
        '    """Summary.\n'
        '\n'
        '    :type orphan: int\n'
        '\n'
        '    Authored prose after orphan type.\n'
        '\n'
        '    :param x: Kept parameter.\n'
        '    """\n'
    )
    CANONICAL_FIRST = (
        'def f(x):\n'
        '    """Summary.\n'
        '\n'
        '    :param x: Kept parameter.\n'
        '    :type orphan: int\n'
        '\n'
        '    Authored prose after orphan type.\n'
        '    """\n'
    )

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_consecutive_orphans_each_keep_their_prose(self, target: StyleName) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    :type first: int\n'
            '\n'
            '    Prose for first.\n'
            '\n'
            '    :type second: str\n'
            '\n'
            '    Prose for second.\n'
            '\n'
            '    :param x: Kept parameter.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert=target)
        report = wrap_source_report(source, config=config)
        assert 'Prose for first.' in report.source
        assert 'Prose for second.' in report.source
        assert report.source.index(':type first: int') < report.source.index('Prose for first.')
        assert report.source.index(':type second: str') < report.source.index('Prose for second.')
        assert report.source.index('Prose for first.') < report.source.index(':type second: str')
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_prose_after_orphan_is_retained_in_place_and_idempotent(
        self, target: StyleName
    ) -> None:
        config = DocwrapConfig(convert=target)
        report = wrap_source_report(self.ORPHAN_FIRST, config=config)
        assert 'Authored prose after orphan type.' in report.source
        assert report.source.index(':type orphan: int') < report.source.index('Authored prose')
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_prose_not_moved_across_orphan(self, target: StyleName) -> None:
        config = DocwrapConfig(convert=target)
        report = wrap_source_report(self.CANONICAL_FIRST, config=config)
        assert 'Authored prose after orphan type.' in report.source
        assert report.source.index(':type orphan: int') < report.source.index('Authored prose')
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    def test_trailing_orphan_keeps_its_prose(self, target: StyleName) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    :param x: Kept parameter.\n'
            '\n'
            '    :type orphan: int\n'
            '\n'
            '    Trailing prose.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert=target)
        report = wrap_source_report(source, config=config)
        assert 'Trailing prose.' in report.source
        assert report.skips == ()
        assert wrap_source_report(report.source, config=config).source == report.source


class TestUnsupportedSectionPreservation:
    def test_partially_preserved_numpy_to_sphinx_is_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def f(value):
                \"\"\"Summary.

                Parameters
                ----------
                value : list[str]
                    This is a deliberately long parameter description for wrapping.

                Returns
                -------
                dict[str, int]
                    The result mapping.
                \"\"\"
        """)
        config = DocwrapConfig(convert='sphinx', line_length=28)

        first_report = wrap_source_report(source, config=config)
        reparsed = parse_docstrings(
            first_report.source,
            find_docstrings(first_report.source),
            config,
        )
        second_report = wrap_source_report(first_report.source, config=config)

        assert ':param list[str] value:' in first_report.source
        assert '\n    Returns\n    -------\n    dict[str, int]\n' in first_report.source
        assert len(reparsed) == 1
        assert reparsed[0].detected_name == 'sphinx'
        assert second_report.source == first_report.source
        assert second_report.converted == 0
        compile(first_report.source, '<test>', 'exec')

    def test_preserved_span_stays_between_supported_neighbors(self) -> None:
        body = (
            'Summary.\n\n'
            'Args:\n'
            '    x (int): Input.\n\n'
            'Notes:\n'
            '    Keep this section.\n\n'
            'Returns:\n'
            '    int: Result.'
        )
        converted = convert_regions(
            _regions_from(body, 'google'),
            get_style('google'),
            get_style('sphinx'),
        )
        preserved_indexes = [
            index for index, region in enumerate(converted) if region.preserve_verbatim
        ]

        assert preserved_indexes == list(range(preserved_indexes[0], preserved_indexes[0] + 3))
        authored_span = [converted[index] for index in preserved_indexes]
        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == [
            ('Notes:',),
            ('    Keep this section.',),
            ('',),
        ]
        assert [region.source_start for region in authored_span] == [5, 6, 7]
        span_start = preserved_indexes[0]
        span_end = preserved_indexes[-1] + 1
        assert any(
            region.kind == 'entry' and region.text == ':param int x: Input.'
            for region in converted[:span_start]
        )
        assert any(
            region.kind == 'entry' and region.text == ':returns: Result.'
            for region in converted[span_end:]
        )

    def test_preserves_authored_trailing_blank_in_final_span(self) -> None:
        body = 'Summary.\n\nNotes:\n    Authored.\n'
        regions = _regions_from(body, 'google')
        regions.append(
            Region(
                kind='blank',
                text='',
                original_lines=('',),
                source_start=4,
            )
        )

        converted = convert_regions(
            regions,
            get_style('google'),
            get_style('sphinx'),
        )
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == [
            ('Notes:',),
            ('    Authored.',),
            ('',),
        ]
        assert [region.source_start for region in authored_span] == [2, 3, 4]
        assert converted[-1].kind == 'blank'
        assert converted[-1].preserve_verbatim

    def test_preserves_complete_numpy_header_and_protected_body(self) -> None:
        body = (
            'Summary.\n\n'
            'Notes\n'
            '-----\n\n'
            'Authored prose.\n\n'
            '```text\n'
            'fenced text\n'
            '```\n\n'
            '>>> sample()\n'
            'sample output\n\n'
            'Literal::\n\n'
            '    literal text\n\n'
            'Parameters\n'
            '----------\n'
            'x : int\n'
            '    Input.'
        )
        converted = convert_regions(
            _regions_from(body, 'numpy'),
            get_style('numpy'),
            get_style('sphinx'),
        )
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == [
            ('Notes',),
            ('-----',),
            ('',),
            ('Authored prose.',),
            ('',),
            ('```text',),
            ('fenced text',),
            ('```',),
            ('',),
            ('>>> sample()',),
            ('sample output',),
            ('',),
            ('Literal::',),
            ('',),
            ('    literal text',),
            ('',),
        ]
        assert [region.source_start for region in authored_span] == list(range(2, 18))
        assert [region.source_scanner_protected for region in authored_span] == [
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            False,
        ]
        assert any(
            region.kind == 'entry'
            and region.text == ':param int x: Input.'
            and not region.preserve_verbatim
            for region in converted
        )

    @pytest.mark.parametrize(
        ('source_style', 'body', 'expected_original_lines', 'expected_source_starts'),
        [
            pytest.param(
                'google',
                'Summary.\n\nExamples:\n    Authored example prose.',
                (('Examples:',), ('    Authored example prose.',)),
                (2, 3),
                id='google-examples',
            ),
            pytest.param(
                'google',
                'Summary.\n\nNotes:\n    Authored note prose.',
                (('Notes:',), ('    Authored note prose.',)),
                (2, 3),
                id='google-notes',
            ),
            pytest.param(
                'numpy',
                'Summary.\n\nExamples\n--------\nAuthored example prose.',
                (('Examples',), ('--------',), ('Authored example prose.',)),
                (2, 3, 4),
                id='numpy-examples',
            ),
            pytest.param(
                'numpy',
                'Summary.\n\nNotes\n-----\nAuthored note prose.',
                (('Notes',), ('-----',), ('Authored note prose.',)),
                (2, 3, 4),
                id='numpy-notes',
            ),
            pytest.param(
                'google',
                'Summary.\n\nYields:\n    str: Next item.',
                (('Yields:',), ('    str: Next item.',)),
                (2, 3),
                id='google-yields',
            ),
            pytest.param(
                'numpy',
                'Summary.\n\nYields\n------\nstr\n    Next item.',
                (('Yields',), ('------',), ('str', '    Next item.')),
                (2, 3, 4),
                id='numpy-yields',
            ),
        ],
    )
    def test_preserves_each_unsupported_prose_or_yields_region(
        self,
        source_style: StyleName,
        body: str,
        expected_original_lines: tuple[tuple[str, ...], ...],
        expected_source_starts: tuple[int, ...],
    ) -> None:
        converted = convert_regions(
            _regions_from(body, source_style),
            get_style(source_style),
            get_style('sphinx'),
        )
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == list(expected_original_lines)
        assert [region.source_start for region in authored_span] == list(expected_source_starts)

    @pytest.mark.parametrize(
        ('body', 'expected_original_lines', 'expected_source_starts'),
        [
            pytest.param(
                'Summary.\n\nReturns:',
                (('Returns:',),),
                (2,),
                id='empty-returns',
            ),
            pytest.param(
                'Summary.\n\nReturns:\n    Authored  prose remains exactly as written.',
                (('Returns:',), ('    Authored  prose remains exactly as written.',)),
                (2, 3),
                id='prose-only-returns',
            ),
            pytest.param(
                'Summary.\n\nArgs:\n    Authored  prose is not a parameter entry.',
                (('Args:',), ('    Authored  prose is not a parameter entry.',)),
                (2, 3),
                id='prose-only-args',
            ),
        ],
    )
    def test_preserves_mapped_sections_without_convertible_entries(
        self,
        body: str,
        expected_original_lines: tuple[tuple[str, ...], ...],
        expected_source_starts: tuple[int, ...],
    ) -> None:
        converted = convert_regions(
            _regions_from(body, 'google'),
            get_style('google'),
            get_style('sphinx'),
        )
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == list(expected_original_lines)
        assert [region.source_start for region in authored_span] == list(expected_source_starts)

    @pytest.mark.parametrize(
        'header',
        ['Attention', 'Caution', 'Danger', 'Hint', 'Important', 'Tip'],
    )
    def test_preserves_unmapped_google_sections_verbatim(self, header: str) -> None:
        authored_body = (
            'This  authored body remains exactly as written even though it is deliberately '
            'long enough to exceed the configured wrapping width.'
        )
        body = f'Summary.\n\n{header}:\n    {authored_body}'
        converted = convert_regions(
            _regions_from(body, 'google'),
            get_style('google'),
            get_style('sphinx'),
        )
        authored_span = [region for region in converted if region.preserve_verbatim]

        assert all(region.preserve_verbatim for region in authored_span)
        assert [region.original_lines for region in authored_span] == [
            (f'{header}:',),
            (f'    {authored_body}',),
        ]
        assert [region.source_start for region in authored_span] == [2, 3]

        source = f'def f():\n    """Summary.\n\n    {header}:\n        {authored_body}\n    """\n'
        config = DocwrapConfig(convert='sphinx', line_length=48)
        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.source == source
        assert second.source == first.source


class TestUnsupportedTargetRolePreservation:
    @pytest.mark.parametrize(
        ('target_name', 'source', 'header', 'next_header'),
        [
            pytest.param(
                'google',
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    Other Parameters\n'
                    '    ----------------\n'
                    '    debug : bool\n'
                    '        Enable debug with a continuation\n'
                    '        kept exactly.\n\n'
                    '    - Keep this list item.\n\n'
                    '    ```python\n'
                    '    print("debug")\n'
                    '    ```\n\n'
                    '    Raises\n'
                    '    ------\n'
                    '    ValueError\n'
                    '        Failed.\n'
                    '    """\n'
                ),
                '    Other Parameters\n',
                '    Raises\n',
                id='numpy-other-param-to-google',
            ),
            pytest.param(
                'sphinx',
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    Other Parameters\n'
                    '    ----------------\n'
                    '    debug : bool\n'
                    '        Enable debug with a continuation\n'
                    '        kept exactly.\n\n'
                    '    - Keep this list item.\n\n'
                    '    ```python\n'
                    '    print("debug")\n'
                    '    ```\n\n'
                    '    Raises\n'
                    '    ------\n'
                    '    ValueError\n'
                    '        Failed.\n'
                    '    """\n'
                ),
                '    Other Parameters\n',
                '    Raises\n',
                id='numpy-other-param-to-sphinx',
            ),
            pytest.param(
                'numpy',
                (
                    'def f():\n'
                    '    """Summary.\n\n'
                    '    Keyword Args:\n'
                    '        debug (bool): Enable debug with a continuation\n'
                    '            kept exactly.\n\n'
                    '        - Keep this list item.\n\n'
                    '        ```python\n'
                    '        print("debug")\n'
                    '        ```\n\n'
                    '    Raises:\n'
                    '        ValueError: Failed.\n'
                    '    """\n'
                ),
                '    Keyword Args:\n',
                '    Raises:\n',
                id='google-keyword-param-to-numpy',
            ),
        ],
    )
    def test_full_pipeline_preserves_exact_source_span(
        self,
        target_name: StyleName,
        source: str,
        header: str,
        next_header: str,
    ) -> None:
        span_start = source.index(header)
        authored_span = source[span_start : source.index(next_header, span_start)]
        config = DocwrapConfig(convert=target_name, line_length=38, strip_types=True)

        first = wrap_source_report(source, config=config)
        output_start = first.source.index(header)
        second = wrap_source_report(first.source, config=config)

        assert first.source[output_start : output_start + len(authored_span)] == authored_span
        assert first.skips == ()
        assert first.parse_error is False
        compile(first.source, '<test>', 'exec')
        assert second.source == first.source
        assert second.skips == ()
        assert second.parse_error is False

    def test_google_keyword_args_converts_to_one_sphinx_keyword_field(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    Keyword Args:\n'
            '        debug (bool): Enable debug with a continuation kept exactly.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert='sphinx')

        report = wrap_source_report(source, config=config)
        parsed = parse_docstrings(report.source, find_docstrings(report.source), config)
        keyword_fields = [
            region
            for region in parsed[0].regions
            if region.role == 'keyword' and region.entry_match is not None
        ]

        assert 'Keyword Args:' not in report.source
        assert ':keyword bool debug: Enable debug with a continuation kept exactly.' in (
            report.source
        )
        assert len(keyword_fields) == 1
        keyword_match = keyword_fields[0].entry_match
        assert keyword_match is not None
        assert keyword_match.name == 'debug'
        assert keyword_match.type_ann == 'bool'
        assert keyword_match.description == 'Enable debug with a continuation kept exactly.'
        assert not keyword_fields[0].preserve_verbatim

    @pytest.mark.parametrize(
        ('source_name', 'target_name', 'body', 'expected_original_lines'),
        [
            pytest.param(
                'numpy',
                'google',
                (
                    'Summary.\n\n'
                    'Other Parameters\n'
                    '----------------\n'
                    'debug : bool\n'
                    '    Enable debug with a continuation\n'
                    '    kept exactly.\n\n'
                    '- Keep this list item.\n\n'
                    '```python\n'
                    'print("debug")\n'
                    '```\n\n'
                    'Raises\n'
                    '------\n'
                    'ValueError\n'
                    '    Failed.'
                ),
                (
                    ('Other Parameters',),
                    ('----------------',),
                    (
                        'debug : bool',
                        '    Enable debug with a continuation',
                        '    kept exactly.',
                    ),
                    ('',),
                    ('- Keep this list item.',),
                    ('',),
                    ('```python',),
                    ('print("debug")',),
                    ('```',),
                    ('',),
                ),
                id='numpy-other-param-to-google',
            ),
            pytest.param(
                'numpy',
                'sphinx',
                (
                    'Summary.\n\n'
                    'Other Parameters\n'
                    '----------------\n'
                    'debug : bool\n'
                    '    Enable debug with a continuation\n'
                    '    kept exactly.\n\n'
                    '- Keep this list item.\n\n'
                    '```python\n'
                    'print("debug")\n'
                    '```\n\n'
                    'Raises\n'
                    '------\n'
                    'ValueError\n'
                    '    Failed.'
                ),
                (
                    ('Other Parameters',),
                    ('----------------',),
                    (
                        'debug : bool',
                        '    Enable debug with a continuation',
                        '    kept exactly.',
                    ),
                    ('',),
                    ('- Keep this list item.',),
                    ('',),
                    ('```python',),
                    ('print("debug")',),
                    ('```',),
                    ('',),
                ),
                id='numpy-other-param-to-sphinx',
            ),
            pytest.param(
                'google',
                'numpy',
                (
                    'Summary.\n\n'
                    'Keyword Args:\n'
                    '    debug (bool): Enable debug with a continuation\n'
                    '        kept exactly.\n\n'
                    '    - Keep this list item.\n\n'
                    '    ```python\n'
                    '    print("debug")\n'
                    '    ```\n\n'
                    'Raises:\n'
                    '    ValueError: Failed.'
                ),
                (
                    ('Keyword Args:',),
                    (
                        '    debug (bool): Enable debug with a continuation',
                        '        kept exactly.',
                    ),
                    ('',),
                    ('    - Keep this list item.',),
                    ('',),
                    ('    ```python',),
                    ('    print("debug")',),
                    ('    ```',),
                    ('',),
                ),
                id='google-keyword-param-to-numpy',
            ),
        ],
    )
    def test_unsupported_span_bypasses_conversion(
        self,
        source_name: StyleName,
        target_name: StyleName,
        body: str,
        expected_original_lines: tuple[tuple[str, ...], ...],
    ) -> None:
        regions = _regions_from(body, source_name)
        header_start = next(
            index for index, region in enumerate(regions) if region.kind == 'section_header'
        )
        span_end = next(
            index
            for index, region in enumerate(regions[header_start + 1 :], header_start + 1)
            if region.kind == 'section_header'
            and region.section_name != regions[header_start].section_name
        )
        source_span = regions[header_start:span_end]
        deleted_text: list[str] = []

        converted = convert_regions(
            regions,
            get_style(source_name),
            get_style(target_name),
            strip_types=True,
            deleted_text=deleted_text,
        )
        output_span = [region for region in converted if region.preserve_verbatim]

        assert [region.original_lines for region in output_span] == list(expected_original_lines)
        assert all(region.preserve_verbatim for region in output_span)
        assert [
            (
                region.kind,
                region.text,
                region.entry_match,
                region.role,
                region.section_name,
                region.original_lines,
                region.source_start,
                region.source_scanner_protected,
            )
            for region in output_span
        ] == [
            (
                region.kind,
                region.text,
                region.entry_match,
                region.role,
                region.section_name,
                region.original_lines,
                region.source_start,
                region.source_scanner_protected,
            )
            for region in source_span
        ]
        assert deleted_text == []


class TestUntypedReturnsProse:
    @pytest.mark.parametrize(
        ('source_name', 'target_name'),
        [
            pytest.param('google', 'numpy', id='google-to-numpy'),
            pytest.param('numpy', 'google', id='numpy-to-google'),
        ],
    )
    @pytest.mark.parametrize('role', ['returns', 'yields'])
    def test_converted_prose_retains_formatter_indentation(
        self,
        source_name: StyleName,
        target_name: StyleName,
        role: CanonicalRole,
    ) -> None:
        entry = Region(
            kind='entry',
            text='source entry',
            entry_match=EntryMatch(
                name='',
                type_ann='',
                description='First line.\n    Second line.',
                role=role,
            ),
            role=role,
        )

        converted = _convert_entry(
            entry,
            get_style(source_name),
            get_style(target_name),
            strip_types=False,
            deleted_text=None,
        )

        _assert_generated_prose(converted, ['    First line.', '    Second line.'])

    def test_generated_entry_retains_leading_spaces_for_rematching(self) -> None:
        target_style = get_style('google')
        entry = Region(
            kind='entry',
            text='x : int\n    The value.',
            entry_match=EntryMatch(
                name='x',
                type_ann='int',
                description='The value.',
                role='param',
            ),
            role='param',
        )

        converted = _convert_entry(
            entry,
            get_style('numpy'),
            target_style,
            strip_types=False,
            deleted_text=None,
        )

        assert len(converted) == 1
        assert converted[0].kind == 'entry'
        assert converted[0].text == '    x (int): The value.'
        assert target_style.match_entry(converted[0].text) is not None

    def test_untyped_returns_wraps_as_prose(self) -> None:
        long_desc = (
            'This is a very long description that should definitely'
            ' wrap when the line length is set to a reasonable value.'
        )
        source = textwrap.dedent(f'''\
            def f():
                """Summary.

                :returns: {long_desc}
                """
        ''')
        config = DocwrapConfig(convert='google', line_length=60)
        report = wrap_source_report(source, config=config)
        # No line should exceed 60 chars (accounting for base indent)
        for line in report.source.split('\n'):
            assert len(line) <= 60, f'Line too long ({len(line)}): {line!r}'


def _assert_generated_prose(regions: list[Region], physical_lines: list[str]) -> None:
    assert len(regions) == len(physical_lines)
    for region, physical in zip(regions, physical_lines, strict=True):
        expected_indent = len(physical) - len(physical.lstrip())
        assert region.kind == 'prose'
        assert region.indent == expected_indent
        assert region.text == physical.lstrip()
        assert region.source_start is None


def _regions_from(source: str, style_name: StyleName) -> list[Region]:
    style = get_style(style_name)
    norm, _offset = normalize_content(source, source_margin='')
    lines = norm.split('\n')
    title_line_count = match_opening_title(lines, style) or 0
    code_lines = find_code_block_lines(lines, start=title_line_count)
    return parse_regions(norm, style=style, code_block_lines=code_lines)


def test_escape_bearing_opening_title_survives_conversion() -> None:
    adornment = '=' * 30
    title = 'Escaped `' + BACKSLASH + 't` title'
    source = (
        'def f(x):\n'
        f'    """{title}\n'
        f'    {adornment}\n'
        '    Args:\n'
        '        x (int): Kept parameter.\n'
        '    """\n'
    )
    config = DocwrapConfig(convert='sphinx', line_length=35)
    first = wrap_source_report(source, config=config)
    assert f'{title}\n    {adornment}\n' in first.source
    assert ':param int x: Kept parameter.' in first.source
    assert first.converted == 1
    assert first.skips == ()
    second = wrap_source_report(first.source, config=config)
    assert second.source == first.source
    assert second.converted == 0


@pytest.mark.parametrize(
    ('title_lines', 'body', 'target', 'converted_marker'),
    [
        pytest.param(
            'Args:\n-----',
            ':param int x: Kept Sphinx.',
            'numpy',
            'Parameters',
            id='underline-google-title-sphinx-body',
        ),
        pytest.param(
            'Parameters\n----------',
            'Args:\n    x (int): Kept Google.',
            'sphinx',
            ':param int x: Kept Google.',
            id='underline-numpy-title-google-body',
        ),
        pytest.param(
            ':param title:\n-------------',
            'Parameters\n----------\nx : int\n    Kept NumPy.',
            'google',
            'Args:',
            id='underline-sphinx-title-numpy-body',
        ),
        pytest.param(
            '-----\nArgs:\n-----',
            ':param int x: Kept Sphinx.',
            'numpy',
            'Parameters',
            id='overline-google-title-sphinx-body',
        ),
        pytest.param(
            '----------\nParameters\n----------',
            'Args:\n    x (int): Kept Google.',
            'sphinx',
            ':param int x: Kept Google.',
            id='overline-numpy-title-google-body',
        ),
        pytest.param(
            '-------------\n:param title:\n-------------',
            'Parameters\n----------\nx : int\n    Kept NumPy.',
            'google',
            'Args:',
            id='overline-sphinx-title-numpy-body',
        ),
    ],
)
def test_marker_title_does_not_suppress_body_conversion(
    title_lines: str,
    body: str,
    target: StyleName,
    converted_marker: str,
) -> None:
    indented_title = title_lines.replace('\n', '\n    ')
    indented_body = body.replace('\n', '\n    ')
    source = f'def f(x):\n    """{indented_title}\n    {indented_body}\n    """\n'
    config = DocwrapConfig(convert=target, line_length=45)
    first = wrap_source_report(source, config=config)
    assert indented_title in first.source
    assert converted_marker in first.source
    assert first.converted == 1
    assert first.skips == ()
    second = wrap_source_report(first.source, config=config)
    assert second.source == first.source
    assert second.converted == 0


@pytest.mark.parametrize(
    'title_lines',
    ['Heading\n-------', '=======\n  Heading\n======='],
    ids=['underline', 'overline'],
)
@pytest.mark.parametrize(
    ('body', 'target', 'converted_marker'),
    [
        pytest.param(
            'Args:\n    x (int): Kept parameter.',
            'sphinx',
            ':param int x: Kept parameter.',
            id='google-to-sphinx',
        ),
        pytest.param(
            'Parameters\n----------\nx : int\n    Kept parameter.',
            'sphinx',
            ':param int x: Kept parameter.',
            id='numpy-to-sphinx',
        ),
        pytest.param(
            ':param int x: Kept parameter.',
            'google',
            'Args:',
            id='sphinx-to-google',
        ),
        pytest.param(
            ':param int x: Kept parameter.',
            'numpy',
            'Parameters',
            id='sphinx-to-numpy',
        ),
    ],
)
def test_opening_title_survives_conversion_idempotently(
    title_lines: str,
    body: str,
    target: StyleName,
    converted_marker: str,
) -> None:
    indented_title = title_lines.replace('\n', '\n    ')
    indented_body = body.replace('\n', '\n    ')
    source = f'def f(x):\n    """{indented_title}\n    {indented_body}\n    """\n'
    config = DocwrapConfig(convert=target, line_length=45)
    first = wrap_source_report(source, config=config)
    assert indented_title in first.source
    assert 'Kept parameter.' in first.source
    assert converted_marker in first.source
    assert first.converted == 1
    assert first.skips == ()
    second = wrap_source_report(first.source, config=config)
    assert second.source == first.source
    assert second.converted == 0


@pytest.mark.parametrize('smart_slack', SMART_SLACK_VALUES)
@pytest.mark.parametrize('fill_style', FILL_STYLES)
@pytest.mark.parametrize('reflow', REFLOW_MODES)
def test_reflow_with_convert_idempotent(
    reflow: ReflowMode,
    fill_style: FillStyle,
    smart_slack: int,
) -> None:
    source = textwrap.dedent("""\
        def f(x):
            \"\"\"Summary.

            Args:
                x: First sentence.
                    Second sentence.
            \"\"\"
            pass
    """)
    cfg = DocwrapConfig(
        line_length=DEFAULT_LINE_LENGTH,
        reflow=reflow,
        fill_style=fill_style,
        smart_slack=smart_slack,
        convert='numpy',
    )
    first = wrap_source(source, config=cfg)
    second = wrap_source(first, config=cfg)
    assert first == second
    assert_region_projection(source, cfg)


def test_regions_from_uses_title_aware_code_scan() -> None:
    regions = _regions_from(
        'Heading\n~~~~~~~\n:param x: Kept.',
        'sphinx',
    )
    assert [region.kind for region in regions] == ['title', 'entry']
    assert regions[1].text == ':param x: Kept.'
