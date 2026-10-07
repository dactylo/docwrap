import warnings
from typing import Literal

import pytest

from docwrap._core._discover import ParsedDocstring, find_docstrings, parse_docstrings
from docwrap._core._escape import _SENTINEL_MAX, _SENTINEL_MIN
from docwrap._core._normalize import split_source_lines
from docwrap._types import DocstringSkip, DocwrapConfig

BACKSLASH = chr(92)


class TestCodeBlockLines:
    def test_code_block_lines_detected(self) -> None:
        source = 'def foo():\n    """Summary.\n\n    Example::\n\n        x = 1\n    """\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert len(results[0].code_block_lines) > 0


class TestDocstringDefLineno:
    def test_class_def_lineno(self) -> None:
        source = 'class Foo:\n    """Doc."""\n    pass\n'
        results = find_docstrings(source)
        assert results[0].def_lineno == 1
        assert results[0].def_end_lineno == 3

    def test_decorated_function_def_lineno(self) -> None:
        source = '@deco\ndef foo():\n    """Doc."""\n    pass\n'
        results = find_docstrings(source)
        assert results[0].def_lineno == 1
        assert results[0].def_end_lineno == 4

    def test_function_def_lineno(self) -> None:
        source = 'def foo():\n    """Doc."""\n    pass\n'
        results = find_docstrings(source)
        assert results[0].def_lineno == 1
        assert results[0].def_end_lineno == 3

    def test_method_def_lineno(self) -> None:
        source = 'class Foo:\n    def bar(self):\n        """Bar."""\n        pass\n'
        results = find_docstrings(source)
        bar_doc = next(d for d in results if d.qualname == 'Foo.bar')
        assert bar_doc.def_lineno == 2
        assert bar_doc.def_end_lineno == 4

    def test_module_def_lineno(self) -> None:
        source = '"""Module doc."""\n'
        results = find_docstrings(source)
        module_doc = next(d for d in results if d.qualname == '<module>')
        assert module_doc.def_lineno == 0
        assert module_doc.def_end_lineno == 0


class TestDocstringQualname:
    def test_async_function(self) -> None:
        source = 'async def foo():\n    """Async doc."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'foo'

    def test_class_docstring(self) -> None:
        source = 'class Foo:\n    """Foo doc."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'Foo'

    def test_class_inside_try(self) -> None:
        source = 'try:\n    class Foo:\n        """Inside try."""\nexcept:\n    pass\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'Foo'

    def test_function_docstring(self) -> None:
        source = 'def foo():\n    """Foo doc."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'foo'

    def test_function_inside_except(self) -> None:
        source = (
            'try:\n    pass\nexcept Exception:\n    def foo():\n        """Inside except."""\n'
        )
        results = find_docstrings(source)
        assert results[0].qualname == 'foo'

    def test_function_inside_if(self) -> None:
        source = 'if True:\n    def foo():\n        """Inside if."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'foo'

    def test_function_inside_match_case(self) -> None:
        source = (
            'match 1:\n    case 1:\n        def foo():\n            """Inside match case."""\n'
        )
        results = find_docstrings(source)
        assert results[0].qualname == 'foo'

    def test_method_docstring(self) -> None:
        source = 'class Foo:\n    def bar(self):\n        """Bar doc."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'Foo.bar'

    def test_module_docstring(self) -> None:
        source = '"""Module doc."""\ndef foo(): pass\n'
        results = find_docstrings(source)
        module_docs = [d for d in results if d.qualname == '<module>']
        assert len(module_docs) == 1

    def test_nested_function(self) -> None:
        source = 'def outer():\n    def inner():\n        """Inner doc."""\n'
        results = find_docstrings(source)
        assert results[0].qualname == 'outer.inner'


class TestFindDocstrings:
    def test_class_docstring(self) -> None:
        source = 'class Foo:\n    """Foo doc."""\n'
        results = find_docstrings(source)
        assert len(results) == 1

    def test_function_docstring(self) -> None:
        source = 'def foo():\n    """Foo doc."""\n'
        results = find_docstrings(source)
        assert len(results) == 1
        assert results[0].col_offset == 4

    def test_method_docstring(self) -> None:
        source = 'class Foo:\n    def bar(self):\n        """Bar doc."""\n'
        results = find_docstrings(source)
        # class docstring is None, method has one
        assert len(results) == 1
        assert results[0].col_offset == 8

    def test_module_docstring(self) -> None:
        source = '"""Module doc."""\n'
        results = find_docstrings(source)
        assert len(results) == 1
        assert results[0].col_offset == 0

    def test_multiple_docstrings(self) -> None:
        source = '"""Module."""\ndef foo():\n    """Foo."""\ndef bar():\n    """Bar."""\n'
        results = find_docstrings(source)
        assert len(results) == 3

    def test_no_docstring(self) -> None:
        source = 'x = 1\n'
        assert find_docstrings(source) == []

    def test_non_string_first_expr(self) -> None:
        source = 'def foo():\n    x = 1\n'
        assert find_docstrings(source) == []

    def test_sorted_by_line(self) -> None:
        source = '"""Module."""\ndef foo():\n    """Foo."""\n'
        results = find_docstrings(source)
        assert results[0].lineno < results[1].lineno

    def test_sorted_by_line_and_col(self) -> None:
        source = 'def a():\n    """A."""\n\ndef b():\n    """B."""\n'
        results = find_docstrings(source)
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert (results[i].lineno, results[i].col_offset) <= (
                results[i + 1].lineno,
                results[i + 1].col_offset,
            )


class TestParseDocstrings:
    def test_fewer_results_than_input_with_mixed_quotes(self) -> None:
        source = 'def foo():\n    \'single quoted\'\n\ndef bar():\n    """triple quoted"""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert len(results) < len(docstrings)
        assert len(results) == 1

    def test_google_style_detected(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: A value.\n    """\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].detected_name == 'google'
        assert results[0].style is not None

    def test_multiple_docstrings(self) -> None:
        source = 'def foo():\n    """Foo summary."""\n\ndef bar():\n    """Bar summary."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert len(results) == 2

    def test_no_style_for_plain_docstring(self) -> None:
        source = 'def foo():\n    """Just a summary."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].detected_name is None
        assert results[0].style is None

    def test_regions_have_section_header_for_google(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: A value.\n    """\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        kinds = [r.kind for r in results[0].regions]
        assert 'section_header' in kinds

    def test_returns_parsed_for_triple_quoted(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert len(results) == 1
        assert isinstance(results[0], ParsedDocstring)

    def test_skips_single_quoted(self) -> None:
        source = "def foo():\n    'Single quoted.'\n"
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert len(results) == 0

    def test_style_override(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: A value.\n    """\n'
        docstrings = find_docstrings(source)
        config = DocwrapConfig(style='numpy')
        results = parse_docstrings(source, docstrings, config, skips=[])
        assert results[0].detected_name == 'numpy'


class TestParseSkipDiagnostics:
    def test_implicit_concatenation_recorded(self) -> None:
        source = 'def f():\n    """a""" """b"""\n'
        docstrings = find_docstrings(source)
        skips: list[DocstringSkip] = []
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=skips)
        assert results == []
        assert len(skips) == 1
        assert skips[0].code == 'implicit-concatenation'
        assert skips[0].qualname == 'f'
        assert skips[0].line == 2
        assert skips[0].path == ''

    def test_line_continuation_recorded_others_still_parse(self) -> None:
        source = (
            'def f():\n'
            '    """Line one ' + BACKSLASH + '\n'
            '    continued."""\n'
            '\n'
            '\n'
            'def g():\n'
            '    """Short."""\n'
        )
        docstrings = find_docstrings(source)
        skips: list[DocstringSkip] = []
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=skips)
        assert [r.info.qualname for r in results] == ['g']
        assert len(skips) == 1
        assert skips[0].code == 'line-continuation'
        assert skips[0].qualname == 'f'
        assert skips[0].line == 2

    def test_sentinel_exhaustion_recorded(self) -> None:
        body = (
            ''.join(chr(cp) for cp in range(_SENTINEL_MIN, _SENTINEL_MAX + 1))
            + ' '
            + BACKSLASH
            + 'n'
        )
        source = 'def f():\n    """' + body + '"""\n'
        docstrings = find_docstrings(source)
        skips: list[DocstringSkip] = []
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=skips)
        assert results == []
        assert len(skips) == 1
        assert skips[0].code == 'sentinel-unavailable'


class TestSourceWarnings:
    @pytest.mark.parametrize('action', ['always', 'error'])
    def test_syntax_warning_does_not_escape_discovery(
        self, action: Literal['always', 'error']
    ) -> None:
        source = 'def sample():\n    """A docstring."""\n    return "\\q"\n'
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter(action, SyntaxWarning)
            found = find_docstrings(source)
        assert [item.qualname for item in found] == ['sample']
        assert not emitted


class TestOpeningTitleDiscovery:
    @staticmethod
    def _parse(source: str, config: DocwrapConfig | None = None) -> ParsedDocstring:
        docstrings = find_docstrings(source)
        return parse_docstrings(source, docstrings, config or DocwrapConfig(), skips=[])[0]

    @pytest.mark.parametrize(
        ('title_lines', 'body', 'detected_name', 'body_kind'),
        [
            pytest.param(
                'Args:\n-----',
                ':param int x: Kept Sphinx.',
                'sphinx',
                'entry',
                id='underline-google-title-sphinx-body',
            ),
            pytest.param(
                'Parameters\n----------',
                'Args:\n    x (int): Kept Google.',
                'google',
                'section_header',
                id='underline-numpy-title-google-body',
            ),
            pytest.param(
                ':param title:\n-------------',
                'Parameters\n----------\nx : int\n    Kept NumPy.',
                'numpy',
                'section_header',
                id='underline-sphinx-title-numpy-body',
            ),
            pytest.param(
                '-----\nArgs:\n-----',
                ':param int x: Kept Sphinx.',
                'sphinx',
                'entry',
                id='overline-google-title-sphinx-body',
            ),
            pytest.param(
                '----------\nParameters\n----------',
                'Args:\n    x (int): Kept Google.',
                'google',
                'section_header',
                id='overline-numpy-title-google-body',
            ),
            pytest.param(
                '-------------\n:param title:\n-------------',
                'Parameters\n----------\nx : int\n    Kept NumPy.',
                'numpy',
                'section_header',
                id='overline-sphinx-title-numpy-body',
            ),
        ],
    )
    def test_body_style_wins_over_marker_title(
        self,
        title_lines: str,
        body: str,
        detected_name: str,
        body_kind: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        indented_body = body.replace('\n', '\n    ')
        source = f'def f(x):\n    """{indented_title}\n    {indented_body}\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name == detected_name
        assert parsed.regions[0].kind == 'title'
        assert any(region.kind == body_kind for region in parsed.regions[1:])
        assert parsed.code_block_lines == frozenset()

    @pytest.mark.parametrize(
        'title_lines',
        [
            'Args:\n-----',
            '-----\nArgs:\n-----',
        ],
        ids=['underline', 'overline'],
    )
    def test_body_tie_remains_unresolved(
        self,
        title_lines: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        source = (
            'def f(x, y):\n'
            f'    """{indented_title}\n'
            '    Parameters\n'
            '    ----------\n'
            '    x : int\n'
            '        Kept.\n'
            '    :param y: Also kept.\n'
            '    """\n'
        )
        parsed = self._parse(source)
        assert parsed.detected_name is None
        assert parsed.regions[0].kind == 'title'
        assert all(region.kind not in {'section_header', 'entry'} for region in parsed.regions[1:])

    @pytest.mark.parametrize('fence', ['```', '~~~'])
    def test_complete_short_fence_takes_title_precedence(
        self,
        fence: str,
    ) -> None:
        source = f'def f():\n    """{fence}\n    foo\n    {fence}\n    """\n'
        parsed = self._parse(source)
        assert [(region.kind, region.text) for region in parsed.regions] == [
            ('title', f'{fence}\nfoo\n{fence}')
        ]
        assert parsed.code_block_lines == frozenset()

    def test_conflicting_override_accepts_complete_title(self) -> None:
        source = 'def f():\n    """Parameters\n    ----------\n    body\n    """\n'
        parsed = self._parse(source, DocwrapConfig(style='sphinx'))
        assert parsed.detected_name == 'sphinx'
        assert parsed.regions[0].kind == 'title'

    @pytest.mark.parametrize(
        'config',
        [DocwrapConfig(), DocwrapConfig(style='numpy')],
        ids=['auto', 'explicit'],
    )
    def test_numpy_section_header_wins(
        self,
        config: DocwrapConfig,
    ) -> None:
        source = 'def f(x):\n    """Parameters\n    ----------\n    x\n        Kept.\n    """\n'
        parsed = self._parse(source, config)
        assert parsed.detected_name == 'numpy'
        assert parsed.regions[0].kind == 'section_header'
        assert all(region.kind != 'title' for region in parsed.regions)

    @pytest.mark.parametrize(
        'title_lines',
        [
            '-----\nArgs:\n-----',
            '----------\nParameters\n----------',
            '-------------\n:param title:\n-------------',
        ],
        ids=['google', 'numpy', 'sphinx'],
    )
    def test_overlined_marker_title_has_no_fallback_evidence(
        self,
        title_lines: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        source = f'def f():\n    """{indented_title}\n    plain body\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name is None
        assert parsed.regions[0].kind == 'title'

    def test_rejected_google_candidate_restores_full_scan(self) -> None:
        source = 'def f():\n    """Args:\n    ~~~~~\n    value\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name == 'google'
        assert all(region.kind != 'title' for region in parsed.regions)
        assert parsed.code_block_lines == frozenset()

    @pytest.mark.parametrize('adornment', ['~~~~~~~', '```````', '>>>>>>>'])
    def test_scanner_collision_adornment_is_accepted(
        self,
        adornment: str,
    ) -> None:
        source = f'def f():\n    """Heading\n    {adornment}\n    body\n    """\n'
        parsed = self._parse(source)
        assert parsed.regions[0].text == f'Heading\n{adornment}'
        assert any(
            region.kind == 'description' and region.text == 'body' for region in parsed.regions
        )
        assert parsed.code_block_lines == frozenset()

    def test_title_exempt_scan_allows_body_style_detection(self) -> None:
        source = 'def f(x):\n    """Heading\n    ~~~~~~~\n    Args:\n        x: Kept.\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name == 'google'
        assert parsed.regions[0].kind == 'title'
        assert parsed.code_block_lines == frozenset()

    def test_title_then_fence_keeps_absolute_final_set(self) -> None:
        source = (
            'def f():\n'
            '    """Heading\n'
            '    -------\n'
            '\n'
            '    ```\n'
            '    code()\n'
            '    ```\n'
            '    body\n'
            '    """\n'
        )
        parsed = self._parse(source)
        assert parsed.regions[0].kind == 'title'
        assert parsed.code_block_lines == frozenset({3, 4, 5})
        assert any(
            region.kind == 'description' and region.text == 'body' for region in parsed.regions
        )

    @pytest.mark.parametrize(
        ('title_lines', 'detected_name'),
        [
            ('Args:\n-----', 'google'),
            ('Parameters\n----------', 'numpy'),
        ],
    )
    def test_underline_candidate_is_zero_body_fallback(
        self,
        title_lines: str,
        detected_name: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        source = f'def f():\n    """{indented_title}\n    plain body\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name == detected_name
        assert all(region.kind != 'title' for region in parsed.regions)
        assert parsed.regions[0].kind == 'section_header'

    def test_unresolved_style_accepts_complete_title(self) -> None:
        source = 'def f():\n    """Heading\n    -------\n    body\n    """\n'
        parsed = self._parse(source)
        assert parsed.detected_name is None
        assert parsed.regions[0].kind == 'title'


class TestParsedDocstringFields:
    def test_base_indent(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].base_indent == 4

    def test_base_indent_module_level(self) -> None:
        source = '"""Module doc."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].base_indent == 0

    def test_col_offset(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].col_offset == 4

    def test_content_is_normalized(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].content == 'Summary line.'

    def test_non_ascii_col_offset(self) -> None:
        source = 'x = "' + chr(233) + '"\ndef foo():\n    """Summary."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        # Docstring is indented 4 spaces on its own line
        assert results[0].col_offset == 4

    def test_original_position_first_line(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].original_position == 'first-line'

    def test_original_position_second_line(self) -> None:
        source = 'def foo():\n    """\n    Summary line.\n    """\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].original_position == 'second-line'

    def test_regions_contain_summary(self) -> None:
        source = 'def foo():\n    """Summary line."""\n'
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        kinds = [r.kind for r in results[0].regions]
        assert 'summary' in kinds

    @pytest.mark.parametrize(
        ('source', 'expected'),
        [
            (
                'def f():\n    r"""Summary."""; trailing = 1\n',
                ('    r"""Summary."""; trailing = 1',),
            ),
            (
                'def f():\n    """\n    Summary.\n    """\n',
                ('    """', '    Summary.', '    """'),
            ),
        ],
    )
    def test_source_lines_preserve_exact_physical_lines(
        self,
        source: str,
        expected: tuple[str, ...],
    ) -> None:
        docstrings = find_docstrings(source)
        results = parse_docstrings(source, docstrings, DocwrapConfig(), skips=[])
        assert results[0].source_lines == expected


class TestPhysicalSourceLines:
    @pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'], ids=['lf', 'crlf', 'cr'])
    @pytest.mark.parametrize('presplit', [False, True])
    def test_literal_and_measurement_views(self, newline: str, presplit: bool) -> None:
        lines = ['    """', '    Summary.', '', '    Body.', '    """; marker = 1']
        source = '# before' + newline + 'def f():' + newline + newline.join(lines) + newline
        found = find_docstrings(source)
        assert len(found) == 1
        assert (found[0].lineno, found[0].end_lineno) == (3, 7)
        results = parse_docstrings(
            source,
            found,
            DocwrapConfig(),
            source_lines=split_source_lines(source) if presplit else None,
            skips=[],
        )
        assert len(results) == 1
        parsed = results[0]
        assert parsed.lit.raw_literal == newline.join(lines).removeprefix('    ').removesuffix(
            '; marker = 1'
        )
        assert parsed.source_lines == tuple(lines)
        assert parsed.source_margin == '    '
        assert parsed.col_offset == 4
        assert parsed.original_position == 'second-line'

    @pytest.mark.parametrize('separator', ['\u0085', '\u2028'])
    def test_parenthesized_literal_with_non_ascii_byte_columns(self, separator: str) -> None:
        literal = '"""Summary.' + separator + ' More.\r\nBody.\rEnd."""'
        prefix = 'def café(): ('
        source = '# before\r' + prefix + literal + '); marker = 1\n'
        found = find_docstrings(source)
        assert len(found) == 1
        value = found[0].node.value
        assert (value.lineno, value.end_lineno) == (2, 4)
        assert value.col_offset == len(prefix.encode('utf-8'))
        assert value.end_col_offset == len('End."""')
        parsed = parse_docstrings(source, found, DocwrapConfig(), skips=[])[0]
        assert parsed.lit.raw_literal == literal
        assert parsed.col_offset == len(prefix)
        assert parsed.source_lines == (
            prefix + '"""Summary.' + separator + ' More.',
            'Body.',
            'End."""); marker = 1',
        )
