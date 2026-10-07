import ast
import inspect
import random
import re
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

import docwrap
import docwrap._core._process as process_mod
from docwrap import DetailLevel
from docwrap._core._discover import ParsedDocstring, find_docstrings, parse_docstrings
from docwrap._core._escape import scan_escape_spellings
from docwrap._core._parse import Region
from docwrap._core._process import (
    analyze_source,
    check,
    wrap_file_report,
    wrap_parsed_docstring,
    wrap_source,
    wrap_source_report,
    wrap_source_report_eligible,
)
from docwrap._core._wrap import simulate_wrap_delta
from docwrap._exceptions import DocwrapSkipWarning, InvalidLineLengthError, UnknownStyleError
from docwrap._files import ReadResult, read_file_result
from docwrap._styles import Style, resolve_style_override
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    FILL_STYLES,
    REFLOW_MODES,
    SUMMARY_STARTS,
    DocstringSkip,
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    StyleName,
    SummaryStart,
    WrapReport,
)
from tests.helpers import (
    CASES,
    REVIEW_REMEDIATION_CASES,
    ReviewRemediationCase,
    UnvalidatedConfig,
    assert_region_projection,
)

BACKSLASH = chr(92)
ESCAPE_MATRIX = [
    BACKSLASH + 'n',
    BACKSLASH + 'r',
    BACKSLASH + 't',
    BACKSLASH + 'a',
    BACKSLASH + 'b',
    BACKSLASH + 'f',
    BACKSLASH + 'v',
    BACKSLASH + '040',
    BACKSLASH + 'x00',
    BACKSLASH + 'x41',
    BACKSLASH + 'u0041',
    BACKSLASH + 'U0001F600',
    BACKSLASH + 'N{LATIN SMALL LETTER A}',
    BACKSLASH + BACKSLASH,
    BACKSLASH + 'd',
    BACKSLASH + "'",
    BACKSLASH + '"',
]
SUMMARY_BOUNDARY_CASES = (
    pytest.param(
        'w' * 66 + ' :param x: detail',
        None,
        id='sphinx-without-initial-style',
    ),
    pytest.param(
        'w' * 68 + ' Returns: ' + 'x' * 72,
        'google',
        id='google-returns',
    ),
)
_IDEMPOTENCE_CASE_COUNT = 256
_IDEMPOTENCE_SEED = 20260708
_SLOW_IDEMPOTENCE_CASE_COUNT = 2048


class TestCheck:
    def test_changed(self) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    A very long body paragraph line that definitely exceeds the'
            ' default seventy-nine character line length limit and will be'
            ' wrapped.\n'
            '    """\n'
        )
        result = docwrap.check(source)
        assert result.status == 'changed'

    def test_config_override(self) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    A medium length body paragraph that fits in 120 chars'
            ' but not in 40.\n'
            '    """\n'
        )
        result_wide = docwrap.check(source, config=DocwrapConfig(line_length=120))
        result_narrow = docwrap.check(source, config=DocwrapConfig(line_length=40))
        assert result_wide.status == 'unchanged'
        assert result_narrow.status == 'changed'

    def test_invalid_syntax(self) -> None:
        source = 'def foo(\n'
        result = docwrap.check(source)
        assert result.status == 'invalid'

    def test_style_override(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        result = docwrap.check(source, config=DocwrapConfig(style='google'))
        assert result.status == 'unchanged'

    def test_unchanged(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        result = docwrap.check(source)
        assert result.status == 'unchanged'


class TestEligibleRangesFiltering:
    def test_eligible_definition_wraps_matching(self) -> None:
        long = 'Long ' * 10 + 'end.'
        source = (
            f'def foo():\n    """{long}"""\n    pass\ndef bar():\n    """Short."""\n    pass\n'
        )
        cfg = DocwrapConfig(line_length=40)
        result = wrap_source_report_eligible(
            source,
            config=cfg,
            eligible_ranges={(1, 3)},
            eligible_target='definition',
        )
        assert result.total == 1
        assert result.wrapped <= 1

    def test_eligible_docstring_target(self) -> None:
        long = 'Long ' * 10 + 'end.'
        source = f'def foo():\n    """{long}"""\n    pass\n'
        cfg = DocwrapConfig(line_length=40)
        result = wrap_source_report_eligible(
            source,
            config=cfg,
            eligible_ranges={(2, 2)},
            eligible_target='docstring',
        )
        assert result.total == 1

    def test_eligible_empty_wraps_none(self) -> None:
        long = 'Long ' * 10 + 'end.'
        source = f'def foo():\n    """{long}"""\n    pass\n'
        cfg = DocwrapConfig(line_length=40)
        result = wrap_source_report_eligible(
            source,
            config=cfg,
            eligible_ranges=set(),
            eligible_target='definition',
        )
        assert result.total == 0
        assert result.wrapped == 0

    def test_eligible_none_wraps_all(self) -> None:
        long = 'Long ' * 10 + 'end.'
        source = f'def foo():\n    """{long}"""\n    pass\n'
        cfg = DocwrapConfig(line_length=40)
        result = wrap_source_report(source, config=cfg)
        assert result.total == 1


class TestFileRoundTrips:
    def test_crlf_file_round_trip(self, tmp_path: Path) -> None:
        lf_source = (
            '# leading comment\n'
            'def f(x):\n'
            '    """Escape `' + BACKSLASH + 'n` plus filler words pushing well past '
            'the width limit."""\n'
            '\n'
            '\n'
            'X = 1\n'
        )
        target = tmp_path / 'crlf.py'
        target.write_bytes(lf_source.replace('\n', '\r\n').encode('utf-8'))
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        wrap_file_report(target, config=cfg, write=True)
        raw = target.read_bytes()
        assert b'\r\n' in raw
        assert b'\n' not in raw.replace(b'\r\n', b'')  # uniformly CRLF
        text = raw.decode('utf-8').replace('\r\n', '\n')
        assert text.startswith('# leading comment\n')
        assert text.endswith('X = 1\n')
        assert BACKSLASH + 'n' in text
        # Idempotent second run
        before = target.read_bytes()
        wrap_file_report(target, config=cfg, write=True)
        assert target.read_bytes() == before

    def test_lf_file_round_trip_outside_docstring_untouched(self, tmp_path: Path) -> None:
        lf_source = (
            'A = "unrelated"\n'
            'def f(x):\n'
            '    """Escape `' + BACKSLASH + 't` plus filler words pushing well past '
            'the width limit."""\n'
            'B = "also unrelated"\n'
        )
        target = tmp_path / 'lf.py'
        target.write_bytes(lf_source.encode('utf-8'))
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        wrap_file_report(target, config=cfg, write=True)
        text = target.read_bytes().decode('utf-8')
        assert text.startswith('A = "unrelated"\n')
        assert text.endswith('B = "also unrelated"\n')
        assert BACKSLASH + 't' in text


class TestInlineDocstringLineLength:
    def test_inline_docstring_respects_line_length(self) -> None:
        long = 'A ' + 'very ' * 10 + 'long summary.'
        source = f'def foo(): """{long}"""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=40, wrap_summary=True))
        for line in result.splitlines():
            assert len(line) <= 40, f'Line exceeds 40: {line!r} ({len(line)} chars)'


class TestPreWrapped:
    def test_pre_wrapped_bypasses_wrap_regions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = (
            'def foo():\n'
            '    """Summary.\n'
            '\n'
            '    This description has enough words to wrap at the configured width.\n'
            '    """\n'
        )
        docstrings = find_docstrings(source)
        cfg = DocwrapConfig(line_length=40)
        parsed_list = parse_docstrings(source, docstrings, cfg, skips=[])
        pd = parsed_list[0]
        normal_result = wrap_parsed_docstring(pd, cfg, style_override=None)
        pre_wrapped = (
            'Summary.\n\nThis description has enough words to\nwrap at the configured width.'
        )

        def fail_recomputation(*_args: object, **_kwargs: object) -> None:
            pytest.fail('pre-wrapped content was recomputed')

        monkeypatch.setattr(process_mod, 'wrap_regions', fail_recomputation)
        pre_wrapped_result = wrap_parsed_docstring(
            pd, cfg, style_override=None, pre_wrapped=pre_wrapped
        )

        assert pre_wrapped_result == normal_result

    def test_pre_wrapped_recomputes_summary_metadata_at_each_width(self) -> None:
        source = 'def foo():\n    """' + 's' * 71 + '\n\n        value = 1\n    """\n'
        parsed = _parse_one(source)
        pre_wrapped = simulate_wrap_delta(
            parsed,
            DocwrapConfig(line_length=72, wrap_summary=False),
        )

        for line_length in (72, 79, 88, 99):
            config = DocwrapConfig(line_length=line_length, wrap_summary=False)
            normal = wrap_parsed_docstring(parsed, config, style_override=None)
            cached = wrap_parsed_docstring(
                parsed,
                config,
                style_override=None,
                pre_wrapped=pre_wrapped,
            )
            assert cached == normal

    def test_pre_wrapped_with_convert_raises(self) -> None:
        source = 'def foo():\n    """Summary."""\n'
        docstrings = find_docstrings(source)
        cfg = DocwrapConfig(convert='numpy')
        parsed_list = parse_docstrings(source, docstrings, cfg, skips=[])
        pd = parsed_list[0]
        with pytest.raises(ValueError, match=r'pre_wrapped.*convert'):
            wrap_parsed_docstring(pd, cfg, style_override=None, pre_wrapped='Summary.')

    def test_pre_wrapped_with_strip_types_raises(self) -> None:
        source = 'def foo():\n    """Summary."""\n'
        docstrings = find_docstrings(source)
        cfg = DocwrapConfig(strip_types=True)
        parsed_list = parse_docstrings(source, docstrings, cfg, skips=[])
        pd = parsed_list[0]
        with pytest.raises(ValueError, match=r'pre_wrapped.*strip_types'):
            wrap_parsed_docstring(pd, cfg, style_override=None, pre_wrapped='Summary.')


class TestProcessConventionOverride:
    def test_convention_bypasses_auto_detect(self) -> None:
        source = textwrap.dedent("""\
            def foo(x):
                \"\"\"Summary.

                :param x: The x value with a long enough description that it
                    needs wrapping at the configured line length.
                \"\"\"
                pass
        """)
        # Source has Sphinx markers, but force Google convention
        result = wrap_source(
            source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        # Auto-detect would identify Sphinx; override forces Google parsing.
        # The result should still be valid Python and idempotent under Google.
        second = wrap_source(
            result, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, style='google')
        )
        assert result == second


class TestProcessMultipleDocstrings:
    def test_multiple_functions(self) -> None:
        source = textwrap.dedent("""\
            def foo():
                \"\"\"Foo doc.\"\"\"
                pass

            def bar():
                \"\"\"Bar doc.\"\"\"
                pass
        """)
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert '"""Foo doc."""' in result
        assert '"""Bar doc."""' in result


class TestProcessNonAscii:
    def test_non_ascii_before_docstring(self) -> None:
        source = chr(233) + ' = 1\ndef foo():\n    """Short."""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert chr(233) + ' = 1' in result
        assert '"""Short."""' in result

    def test_non_ascii_in_function_name(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        # One-liner: docstring is on the SAME line as 'o', so col_offset
        # (byte 12) differs from character index (11).
        source = 'def f' + chr(243) + f'o(): """{long}"""\n'
        result = wrap_source(
            source,
            config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, wrap_summary=True),
        )
        assert 'def f' + chr(243) + 'o():' in result
        assert result != source
        compile(result, '<test>', 'exec')

    def test_non_ascii_source_preserved(self) -> None:
        # UTF-8 byte offsets differ from char indices for non-ASCII
        source = 'x = "h' + chr(233) + 'llo"\ndef foo():\n    """Short."""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert result == source  # unchanged, but verifies splicing didn't corrupt

    def test_non_ascii_with_modified_docstring(self) -> None:
        # Exercises line-scoped replacement with non-ASCII before the docstring
        long = 'A ' + 'very ' * 15 + 'long summary.'
        emoji_assignment = chr(233) + 'moji = "' + chr(0x1F600) + '"'
        source = emoji_assignment + f'\ndef foo():\n    """{long}"""\n'
        result = wrap_source(
            source,
            config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, wrap_summary=True),
        )
        assert emoji_assignment in result
        assert result != source  # docstring was actually modified
        assert '"""A ' in result


class TestProcessPreservesCode:
    def test_code_unchanged(self) -> None:
        source = textwrap.dedent("""\
            x = 1
            def foo():
                \"\"\"Doc.\"\"\"
                return x + 2
            y = 3
        """)
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        # Non-docstring code must be identical
        assert 'x = 1' in result
        assert 'return x + 2' in result
        assert 'y = 3' in result


class TestProcessSimple:
    def test_no_docstrings(self) -> None:
        source = 'x = 1\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert result == source

    def test_unchanged_docstring(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        result = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert result == source

    def test_wrap_long_summary(self) -> None:
        long_text = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long_text}"""\n'
        result = wrap_source(
            source,
            config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH, wrap_summary=True),
        )
        assert result != source
        assert '"""A ' in result
        # Should be valid Python
        compile(result, '<test>', 'exec')


class TestSkipReportPlumbing:
    def _continuation_source(self) -> str:
        return 'def f():\n    """Line one ' + BACKSLASH + '\n    continued."""\n'

    def test_analyze_source_exposes_skips(self) -> None:
        analysis = analyze_source(self._continuation_source(), DocwrapConfig())
        assert len(analysis.skips) == 1
        assert analysis.skips[0].code == 'line-continuation'

    def test_check_carries_skips(self) -> None:
        result = check(self._continuation_source())
        assert result.status == 'unchanged'
        assert result.skips[0].code == 'line-continuation'

    def test_incremental_eligible_escape_docstring(self) -> None:
        source = (
            'def f(x):\n'
            '    """Escape `' + BACKSLASH + 't` plus filler words pushing well past '
            'the width limit."""\n'
            '\n'
            '\n'
            'def g(x):\n'
            '    """Another long summary line that also exceeds the configured '
            'width target."""\n'
        )
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        report = wrap_source_report_eligible(
            source, config=cfg, eligible_ranges={(1, 2)}, eligible_target='definition'
        )
        assert BACKSLASH + 't' in report.source
        assert 'Another long summary line that also exceeds' in report.source
        assert report.wrapped == 1

    def test_unrepresentable_return_is_docstring_skip(self) -> None:
        source = textwrap.dedent('''\
            def f(x):
                """Summary.

                Args:
                    x (bytes): Typed parameter.

                Returns:
                    result (int): Named result.
                """
        ''')
        config = DocwrapConfig(convert='sphinx', strip_types=True)
        analysis = analyze_source(source, config)

        result = wrap_parsed_docstring(analysis.parsed[0], config, None)

        assert result.new_literal is None
        assert result.converted is False
        assert result.skip is not None
        assert result.skip.path == ''
        assert result.skip.line == 2
        assert result.skip.qualname == 'f'
        assert result.skip.code == 'unrepresentable-return'
        assert 'Returns' in result.skip.message
        assert "'result'" in result.skip.message
        assert 'sphinx' in result.skip.message.lower()

    def test_unrepresentable_return_report_is_stable(self) -> None:
        source = textwrap.dedent('''\
            def f(x):
                """Summary.

                Parameters
                ----------
                x : bytes
                    Typed parameter.

                Returns
                -------
                result : int
                    Named result.

                Raises
                ------
                ValueError
                    If the result is unavailable.
                """
        ''')
        config = DocwrapConfig(convert='sphinx')

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.source == source
        assert first.converted == 0
        assert [skip.code for skip in first.skips] == ['unrepresentable-return']
        assert 'Returns' in first.skips[0].message
        assert "'result'" in first.skips[0].message
        assert 'sphinx' in first.skips[0].message.lower()
        assert second == first

    def test_wrap_file_report_fills_skip_path(self, tmp_path: Path) -> None:
        target = tmp_path / 'mod.py'
        target.write_text(self._continuation_source(), encoding='utf-8')
        report = wrap_file_report(target, config=DocwrapConfig())
        assert report.skips[0].path == str(target)

    def test_wrap_source_report_merges_parse_skips(self) -> None:
        report = wrap_source_report(self._continuation_source(), config=DocwrapConfig())
        assert report.skips[0].code == 'line-continuation'
        assert report.skips[0].path == ''


class TestStructuredLiteralMappingSkip:
    def test_unmappable_literal_records_without_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Implicit concatenation: ast finds docstring but get_source_segment
        # returns a literal that parse_literal cannot map (returns None).
        # The docstring should be left unmodified and recorded once.
        source = 'def foo():\n    """part1""" """part2"""\n    pass\n'
        with caplog.at_level('WARNING', logger='docwrap._core'):
            report = wrap_source_report(
                source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH)
            )
        assert report.source == source
        assert len(report.skips) == 1
        assert report.skips[0].code == 'implicit-concatenation'
        assert report.skips[0].qualname == 'foo'
        assert report.skips[0].line == 2
        assert caplog.records == []


class TestStyleTieFieldWrapping:
    _SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    :param alpha: This description is deliberately far too long for the'
        ' configured line length and must be wrapped somehow.\n'
        '\n'
        '    Returns:\n'
        '        Something useful.\n'
        '    """\n'
    )

    def test_malformed_field_like_marker_stays_flat(self) -> None:
        source = (
            'def f():\n'
            '    """Summary line.\n'
            '\n'
            '    :param beta: Short.\n'
            '\n'
            '    :notarole alpha: This description is deliberately far too long for the'
            ' configured line length and must be wrapped somehow.\n'
            '\n'
            '    Returns:\n'
            '        Something useful.\n'
            '    """\n'
        )
        out = wrap_source(source)
        lines = out.split('\n')
        marker_index = next(i for i, ln in enumerate(lines) if ':notarole alpha:' in ln)
        continuation = lines[marker_index + 1]
        assert continuation.startswith('    ')
        assert not continuation.startswith('        ')
        assert continuation.strip()

    def test_near_tie_majority_still_wins(self) -> None:
        source = (
            'def f():\n'
            '    """Summary line.\n'
            '\n'
            '    :param alpha: This description is deliberately far too long for the'
            ' configured line length and must be wrapped somehow.\n'
            '    :param beta: Short.\n'
            '\n'
            '    Returns:\n'
            '        Something useful.\n'
            '    """\n'
        )
        out = wrap_source(source)
        lines = out.split('\n')
        marker_index = next(i for i, ln in enumerate(lines) if ':param alpha:' in ln)
        assert lines[marker_index + 1].startswith('        ')

    def test_tie_field_continuation_hangs(self) -> None:
        out = wrap_source(self._SOURCE)
        lines = out.split('\n')
        marker_index = next(i for i, ln in enumerate(lines) if ':param alpha:' in ln)
        continuation = lines[marker_index + 1]
        assert continuation.startswith('        ')
        assert continuation.strip()
        assert wrap_source(out) == out

    def test_tie_plain_prose_unchanged(self) -> None:
        source = (
            'def f():\n'
            '    """Summary line.\n'
            '\n'
            '    :param alpha: Short field.\n'
            '\n'
            '    Returns:\n'
            '        Something useful.\n'
            '\n'
            '    This plain paragraph is deliberately far too long for the configured'
            ' line length and should wrap flat.\n'
            '    """\n'
        )
        out = wrap_source(source)
        lines = out.split('\n')
        wrapped_index = next(
            i for i, ln in enumerate(lines) if ln.startswith('    This plain paragraph')
        )
        assert lines[wrapped_index + 1].startswith('    ')
        assert not lines[wrapped_index + 1].startswith('        ')


class TestTildeHeadingWrapping:
    def test_paragraph_between_tilde_headings_wraps(self) -> None:
        source = (
            'def f():\n'
            '    """Summary line.\n'
            '\n'
            '    More\n'
            '    ~~~~\n'
            '\n'
            '    This paragraph is deliberately much too long for the configured line'
            ' length and should be wrapped by cautious reflow.\n'
            '\n'
            '    Details\n'
            '    ~~~~~~~\n'
            '    """\n'
        )
        out = wrap_source(source)
        assert out != source
        assert '    ~~~~\n' in out
        assert '    ~~~~~~~\n' in out
        assert wrap_source(out) == out


class TestValidationNet:
    def test_accidental_conversion_loss_fails_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: Uses `' + BACKSLASH + 'n` in its description text.\n'
            '    """\n'
        )
        original = process_mod.convert_regions

        def dropping(
            regions: list[Region],
            source_style: Style,
            target_style: Style,
            strip_types: bool = False,
            *,
            deleted_text: list[str] | None = None,
        ) -> list[Region]:
            converted = original(
                regions,
                source_style,
                target_style,
                strip_types,
                deleted_text=deleted_text,
            )
            return [r for r in converted if not any(0xE000 <= ord(ch) <= 0xF8FF for ch in r.text)]

        monkeypatch.setattr(process_mod, 'convert_regions', dropping)
        cfg = DocwrapConfig(convert='numpy', line_length=72)
        report = wrap_source_report(source, config=cfg, detail='list')
        assert report.source == source
        assert report.wrapped == 0
        assert report.converted == 0
        assert report.docstrings[0].changed is False
        assert report.docstrings[0].converted is False
        assert report.skips[0].code == 'validation-failed'
        assert report.skips[0].qualname == 'f'

    def test_accidental_loss_of_retained_methods_type_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # strip_types is active, so type deletions ARE reported elsewhere --
        # but the Methods entry is retained verbatim by the converter, so its
        # identity is never reported as deleted. An accidental drop of that
        # retained entry must therefore fail validation, not pass as an
        # allowed loss.
        source = (
            'class C:\n'
            '    """Summary.\n'
            '\n'
            '    Methods:\n'
            '        run (Pattern' + BACKSLASH + 'n): Run it.\n'
            '    """\n'
        )
        original = process_mod.convert_regions

        def dropping(
            regions: list[Region],
            source_style: Style,
            target_style: Style,
            strip_types: bool = False,
            *,
            deleted_text: list[str] | None = None,
        ) -> list[Region]:
            converted = original(
                regions,
                source_style,
                target_style,
                strip_types,
                deleted_text=deleted_text,
            )
            return [r for r in converted if not any(0xE000 <= ord(ch) <= 0xF8FF for ch in r.text)]

        monkeypatch.setattr(process_mod, 'convert_regions', dropping)
        cfg = DocwrapConfig(convert='numpy', strip_types=True, line_length=72)
        report = wrap_source_report(source, config=cfg)
        assert 'Pattern' + BACKSLASH + 'n' in report.source  # untouched
        assert report.wrapped == 0
        assert report.converted == 0
        assert report.skips[0].code == 'validation-failed'
        assert report.skips[0].qualname == 'C'

    def test_file_validation_failure_reverts_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = (
            'def f(x):\n    """Return the value with a very long summary that must wrap now."""\n'
        )

        def parse_failure(_source: str) -> bool:
            return False

        monkeypatch.setattr(process_mod, '_parses_cleanly', parse_failure)
        cfg = DocwrapConfig(line_length=50, wrap_summary=True)
        report = wrap_source_report(source, config=cfg)
        assert report.source == source
        assert report.wrapped == 0
        assert report.skips[-1].code == 'file-validation-failed'
        assert report.skips[-1].line == 0
        assert report.skips[-1].qualname == ''

    def test_no_skips_on_clean_wrap(self) -> None:
        source = (
            'def f(x):\n'
            '    """Plain filler words pushing this summary well past the width '
            'limit now."""\n'
        )
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        report = wrap_source_report(source, config=cfg)
        assert report.wrapped == 1
        assert report.skips == ()

    def test_quote_collision_recorded_as_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = (
            'def f(x):\n'
            '    """Plain filler words pushing this summary well past the width '
            'limit now."""\n'
        )

        def collide_quotes(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(process_mod, 'reassemble', collide_quotes)
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        report = wrap_source_report(source, config=cfg)
        assert report.source == source
        assert report.wrapped == 0
        assert report.skips[0].code == 'quote-collision'
        assert report.skips[0].qualname == 'f'

    def test_validation_failure_reverts_docstring_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        escape_doc = (
            'def bad():\n'
            '    """Escape `' + BACKSLASH + 't` plus filler words pushing well past '
            'the configured width limit."""\n'
        )
        plain_doc = (
            'def good():\n'
            '    """Plain filler words pushing this summary well past the width '
            'limit too."""\n'
        )
        source = escape_doc + '\n\n' + plain_doc

        def corrupting_restore(
            text: str,
            escape_map: tuple[tuple[str, str], ...],
            *,
            allowed_loss: frozenset[str] = frozenset(),
        ) -> str:
            del allowed_loss
            return text.replace(escape_map[0][0], 'Z')

        monkeypatch.setattr(process_mod, 'restore_escapes', corrupting_restore)
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        report = wrap_source_report(source, config=cfg)
        assert BACKSLASH + 't' in report.source  # bad() docstring untouched
        assert escape_doc.rstrip('\n') in report.source
        assert report.wrapped == 1  # good() still wrapped
        assert len(report.skips) == 1
        assert report.skips[0].code == 'validation-failed'
        assert report.skips[0].qualname == 'bad'
        assert report.skips[0].path == ''
        assert report.skips[0].line == 2


class TestWrapFileReport:
    def test_auto_resolves_config(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 40\nwrap-summary = true\n'
        )
        f = tmp_path / 'test.py'
        medium = 'A medium length summary for testing.'
        f.write_text(f'def foo():\n    """{medium}"""\n')
        report = wrap_file_report(f)
        assert report.wrapped == 1  # would not wrap at DEFAULT_LINE_LENGTH

    def test_detail_level(self, tmp_path: Path) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        report = wrap_file_report(f, detail='list')
        assert len(report.docstrings) == 1

    def test_invalid_declared_encoding_error_propagates(self, tmp_path: Path) -> None:
        path = tmp_path / 'invalid-bytes.py'
        path.write_bytes(b'# coding: ascii\nvalue = "\xff"\n')

        with pytest.raises(UnicodeDecodeError):
            wrap_file_report(path)

    def test_line_length_override(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        report = wrap_file_report(f, config=DocwrapConfig(line_length=10, wrap_summary=True))
        assert report.wrapped == 1  # "Short." doesn't fit in 10

    def test_missing_path_error_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            wrap_file_report(tmp_path / 'missing.py')

    def test_parse_error(self, tmp_path: Path) -> None:
        f = tmp_path / 'bad.py'
        f.write_text('def foo(\n')
        report = wrap_file_report(f)
        assert report.parse_error is True

    def test_style_override(self, tmp_path: Path) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        f.write_text(f'def foo():\n    """{long}"""\n')
        report = wrap_file_report(f, config=DocwrapConfig(style='google', wrap_summary=True))
        assert report.wrapped == 1

    def test_unchanged_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        report = wrap_file_report(f)
        assert report.total == 1
        assert report.wrapped == 0

    def test_unknown_encoding_cookie_error_propagates(self, tmp_path: Path) -> None:
        path = tmp_path / 'unknown-cookie.py'
        path.write_bytes(b'# coding: not-a-codec\nx = 1\n')

        with pytest.raises(SyntaxError):
            wrap_file_report(path)

    def test_wraps_file_returns_report(self, tmp_path: Path) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        original = f'def foo():\n    """{long}"""\n'.encode()
        f.write_bytes(original)
        report = wrap_file_report(f, config=DocwrapConfig(wrap_summary=True))
        assert isinstance(report, WrapReport)
        assert report.total == 1
        assert report.wrapped == 1
        assert report.source.encode() != original
        assert f.read_bytes() == original

    def test_write_modifies_file(self, tmp_path: Path) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        f = tmp_path / 'test.py'
        original = f'def foo():\n    """{long}"""\n'
        f.write_text(original)
        report = wrap_file_report(f, config=DocwrapConfig(wrap_summary=True), write=True)
        assert f.read_text() != original
        assert f.read_text() == report.source

    def test_write_no_change_skips_write(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('def foo():\n    """Short."""\n')
        mtime_before = f.stat().st_mtime
        wrap_file_report(f, write=True)
        assert f.stat().st_mtime == mtime_before

    def test_write_preserves_crlf(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        source = (
            'def foo():\n'
            '    """Summary.\n\n'
            '    This body prose is deliberately long enough to wrap at forty columns.\n'
            '    """\n'
        )
        original = source.replace('\n', '\r\n').encode()
        f.write_bytes(original)
        report = wrap_file_report(f, config=DocwrapConfig(line_length=40), write=True)
        raw = f.read_bytes()

        assert report.wrapped == 1
        assert raw != original
        assert b'\r\n' in raw
        assert raw.replace(b'\r\n', b'').count(b'\n') == 0

    def test_write_preserves_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        content = (
            '# -*- coding: utf-8 -*-\n'
            'def foo():\n'
            '    """Summary.\n\n'
            '    This body prose includes Renée at a café and is deliberately long '
            'enough to wrap at forty columns.\n'
            '    """\n'
        )
        original = content.encode('utf-8')
        f.write_bytes(original)
        report = wrap_file_report(f, config=DocwrapConfig(line_length=40), write=True)
        raw = f.read_bytes()

        assert report.wrapped == 1
        assert raw != original
        result = raw.decode('utf-8')
        assert 'Renée' in result
        assert 'café' in result

    def test_write_uses_original_read_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'source.py'
        original = b'\xef\xbb\xbfdef f():\r\n    """A ' + b'very ' * 20 + b'long summary."""\r\n'
        path.write_bytes(original)
        real_reader = read_file_result
        real_writer = process_mod.write_preserving_newline
        writer_raw: list[bytes] = []

        def mutating_reader(candidate: Path) -> ReadResult:
            result = real_reader(candidate)
            candidate.write_bytes(b'def concurrent_change():\n    pass\n')
            return result

        def tracked_writer(
            candidate: Path,
            source: str,
            encoding: str,
            *,
            raw: bytes,
        ) -> None:
            writer_raw.append(raw)
            real_writer(candidate, source, encoding, raw=raw)

        monkeypatch.setattr(process_mod, 'read_file_result', mutating_reader)
        monkeypatch.setattr(process_mod, 'write_preserving_newline', tracked_writer)

        report = wrap_file_report(
            path,
            config=DocwrapConfig(wrap_summary=True),
            write=True,
        )

        raw = path.read_bytes()
        assert writer_raw == [original]
        assert report.source.encode('utf-8-sig').replace(b'\n', b'\r\n') == raw
        assert b'concurrent_change' not in raw
        assert b'\r\n' in raw
        assert b'\n' not in raw.replace(b'\r\n', b'')


class TestWrapParsedDocstring:
    def test_convert_regions_does_not_mutate_parsed(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: A value.\n    """\n'
        config = DocwrapConfig(convert='numpy')
        parsed = _parse_one(source, config=DocwrapConfig())
        original_regions = parsed.regions
        _result = wrap_parsed_docstring(parsed, config, None)
        # parsed.regions must be unchanged (frozen tuple)
        assert parsed.regions is original_regions
        assert parsed.regions == original_regions

    def test_convert_sets_converted_flag(self) -> None:
        source = 'def foo(x):\n    """Summary.\n\n    Args:\n        x: A value.\n    """\n'
        config = DocwrapConfig(convert='numpy')
        parsed = _parse_one(source, config=DocwrapConfig())
        result = wrap_parsed_docstring(parsed, config, None)
        assert result.converted is True

    def test_converted_target_uses_target_render_style(self) -> None:
        source = (
            'def f(value):\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        value (dict[str, int]): This is a deliberately long typed parameter '
            'description for wrapping.\n'
            '    """\n'
        )
        config = DocwrapConfig(line_length=32, style='google', convert='sphinx')
        first = wrap_source_report(source, config=config).source
        parsed = _parse_one(first, config=DocwrapConfig())

        result = wrap_parsed_docstring(
            parsed,
            config,
            resolve_style_override(config.style),
        )

        assert parsed.detected_name == 'sphinx'
        assert result.new_literal is None

    def test_none_when_unchanged(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        parsed = _parse_one(source)
        result = wrap_parsed_docstring(parsed, DocwrapConfig(), None)
        assert result.new_literal is None
        assert result.summary_skipped is False
        assert result.summary_width is None
        assert result.converted is False

    def test_restores_wrapped_source_escape_without_doubling(self) -> None:
        spelling = BACKSLASH + 'x41'
        source = (
            'def foo():\n'
            '    """Leading words place '
            + spelling
            + ' beside enough trailing prose to force this summary to wrap."""\n'
        )
        parsed = _parse_one(source)
        config = DocwrapConfig(line_length=50, wrap_summary=True)

        result = wrap_parsed_docstring(parsed, config, None)

        assert result.new_literal is not None
        assert scan_escape_spellings(result.new_literal) == (spelling,)
        assert BACKSLASH * 2 + 'x41' not in result.new_literal

    def test_summary_skip_when_wrap_summary_false(self) -> None:
        long_summary = 'A' * 90
        source = f'def foo():\n    """{long_summary}\n\n    Body text.\n    """\n'
        config = DocwrapConfig(line_length=79, wrap_summary=False)
        parsed = _parse_one(source, config)
        result = wrap_parsed_docstring(parsed, config, None)
        assert result.summary_skipped is True
        assert result.summary_width is not None
        assert result.summary_width > 79

    def test_summary_start_second_line_zero_overhead(self) -> None:
        # 73-char summary: with first-line overhead (7 chars) it exceeds 79
        # and wraps; with second-line (0 overhead) it fits but reformats.
        # The key observable: both change the literal, but differently.
        long_summary = 'A' * 73
        source = f'def foo():\n    """{long_summary}"""\n'
        config_first = DocwrapConfig(line_length=79, summary_start='first-line', wrap_summary=True)
        config_second = DocwrapConfig(
            line_length=79, summary_start='second-line', wrap_summary=True
        )
        parsed = _parse_one(source)
        result_first = wrap_parsed_docstring(parsed, config_first, None)
        result_second = wrap_parsed_docstring(parsed, config_second, None)
        # Both produce changes (first-line wraps, second-line reformats position)
        assert result_first.new_literal is not None
        assert result_second.new_literal is not None
        # Different overhead leads to different output
        assert result_first.new_literal != result_second.new_literal

    def test_wraps_long_prose(self) -> None:
        long_line = ' '.join(['word'] * 20)  # ~99 chars of wrappable text
        source = f'def foo():\n    """Summary.\n\n    {long_line}\n    """\n'
        parsed = _parse_one(source)
        result = wrap_parsed_docstring(parsed, DocwrapConfig(line_length=60), None)
        assert result.new_literal is not None


class TestWrapSourceReportDefaultConfig:
    def test_none_config_equals_default_config(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        with_none = wrap_source_report(source, config=None)
        with_default = wrap_source_report(source, config=DocwrapConfig())
        assert with_none == with_default

    def test_omitted_config_processes_docstring(self) -> None:
        report = wrap_source_report('def foo():\n    """Short."""\n')
        assert report.total == 1


class TestWrapSourceReportDiff:
    def test_diff_populates_before_after(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        report = wrap_source_report(
            source,
            config=DocwrapConfig(wrap_summary=True),
            detail='diff',
        )
        changed = [d for d in report.docstrings if d.changed]
        assert len(changed) == 1
        assert changed[0].before is not None
        assert changed[0].after is not None
        assert changed[0].before != changed[0].after

    def test_diff_unchanged_has_none(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        report = wrap_source_report(source, config=DocwrapConfig(), detail='diff')
        assert report.docstrings[0].before is None
        assert report.docstrings[0].after is None

    def test_parenthesized_diff_uses_literal_indent(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    (\n        """{long}"""\n    )\n'

        report = wrap_source_report(
            source,
            config=DocwrapConfig(line_length=60, wrap_summary=True),
            detail='diff',
        )

        change = report.docstrings[0]
        assert change.changed
        assert change.before is not None
        assert change.after is not None
        assert change.before.startswith('        """')
        assert change.after.startswith('        """')

    def test_tab_indented_diff_uses_exact_margin(self) -> None:
        source = (
            'def f():\n'
            '\t"""Summary.\n'
            '\n'
            '\tThis tab-indented function docstring has enough prose to wrap at the '
            'configured narrow line length.\n'
            '\t"""\n'
        )

        report = wrap_source_report(
            source,
            config=DocwrapConfig(line_length=45),
            detail='diff',
        )

        change = report.docstrings[0]
        assert change.changed
        assert change.before is not None
        assert change.after is not None
        assert change.before.startswith('\t"""')
        assert change.after.startswith('\t"""')


class TestWrapSourceReportList:
    def test_list_changed_flag(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n\ndef bar():\n    """Short."""\n'
        report = wrap_source_report(
            source,
            config=DocwrapConfig(wrap_summary=True),
            detail='list',
        )
        assert len(report.docstrings) == 2
        changed = [d for d in report.docstrings if d.changed]
        assert len(changed) == 1
        assert changed[0].qualname == 'foo'

    def test_list_has_no_before_after(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        report = wrap_source_report(source, config=DocwrapConfig(), detail='list')
        for d in report.docstrings:
            assert d.before is None
            assert d.after is None

    def test_list_includes_all_docstrings(self) -> None:
        source = 'def foo():\n    """Short."""\n\ndef bar():\n    """Also short."""\n'
        report = wrap_source_report(source, config=DocwrapConfig(), detail='list')
        assert len(report.docstrings) == 2
        assert all(not d.changed for d in report.docstrings)

    def test_list_line_numbers(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        report = wrap_source_report(source, config=DocwrapConfig(), detail='list')
        assert report.docstrings[0].start_line == 2
        assert report.docstrings[0].end_line == 2

    def test_parenthesized_docstring_line_numbers(self) -> None:
        source = 'def foo():\n    (\n        """Short."""\n    )\n'
        report = wrap_source_report(source, config=DocwrapConfig(), detail='list')
        assert report.docstrings[0].start_line == 3
        assert report.docstrings[0].end_line == 3


class TestWrapSourceReportStyle:
    def test_style_detected(self) -> None:
        source = 'def foo(x):\n    """Do something.\n\n    Args:\n        x: A value.\n    """\n'
        report = wrap_source_report(source, config=DocwrapConfig())
        assert report.style == 'google'

    def test_style_mixed_yields_none(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    Args:\n'
            '        x: val.\n'
            '    """\n'
            '\n'
            'def g():\n'
            '    """Summary.\n\n'
            '    Parameters\n'
            '    ----------\n'
            '    x\n'
            '        val.\n'
            '    """\n'
        )
        config = DocwrapConfig()
        report = wrap_source_report(source, config=config, detail='diff')
        assert report.style is None

    def test_style_none_no_docstrings(self) -> None:
        report = wrap_source_report('x = 1\n', config=DocwrapConfig())
        assert report.style is None

    def test_style_none_parse_error(self) -> None:
        report = wrap_source_report('def f(:\n', config=DocwrapConfig())
        assert report.style is None

    def test_style_override(self) -> None:
        source = 'def foo(x):\n    """Do something.\n\n    Args:\n        x: A value.\n    """\n'
        report = wrap_source_report(source, config=DocwrapConfig(style='numpy'))
        assert report.style == 'numpy'

    def test_style_override_on_parse_error(self) -> None:
        source = 'def f(:\n'
        config = DocwrapConfig(style='google')
        report = wrap_source_report(source, config=config)
        assert report.style == 'google'


class TestWrapSourceReportSummary:
    def test_changed_docstring(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        report = wrap_source_report(source, config=DocwrapConfig(wrap_summary=True))
        assert report.total == 1
        assert report.wrapped == 1
        assert report.source != source
        assert report.docstrings == ()  # summary detail = no entries

    def test_no_docstrings(self) -> None:
        report = wrap_source_report('x = 1\n', config=DocwrapConfig())
        assert report.total == 0
        assert report.wrapped == 0
        assert report.source == 'x = 1\n'
        assert report.docstrings == ()

    def test_parse_error(self) -> None:
        report = wrap_source_report('def foo(\n', config=DocwrapConfig())
        assert report.parse_error is True
        assert report.source == 'def foo(\n'
        assert report.total == 0
        assert report.wrapped == 0
        assert report.docstrings == ()

    def test_unchanged_docstring(self) -> None:
        source = 'def foo():\n    """Short."""\n'
        report = wrap_source_report(source, config=DocwrapConfig())
        assert report.total == 1
        assert report.wrapped == 0
        assert report.source == source


class TestWrapSourceSkipWarning:
    def test_no_warning_without_skips(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapSkipWarning)
            wrap_source('def f():\n    """Short."""\n')

    def test_report_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = 'input source'
        expected_config = DocwrapConfig(line_length=72)
        sentinel_source = 'sentinel report source'
        report = WrapReport(
            source=sentinel_source,
            total=2,
            wrapped=0,
            skips=(
                DocstringSkip(
                    path='',
                    line=3,
                    qualname='C.first',
                    code='quote-collision',
                    message='quote delimiter collision',
                ),
                DocstringSkip(
                    path='',
                    line=8,
                    qualname='C.second',
                    code='validation-failed',
                    message='escape spellings changed during rewrite',
                ),
            ),
        )
        call_count = 0

        def fake_wrap_source_report(
            source: str,
            *,
            config: DocwrapConfig,
        ) -> WrapReport:
            nonlocal call_count
            call_count += 1
            assert source == 'input source'
            assert config is expected_config
            return report

        monkeypatch.setattr(process_mod, 'wrap_source_report', fake_wrap_source_report)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', DocwrapSkipWarning)
            frame = inspect.currentframe()
            assert frame is not None
            caller_filename = frame.f_code.co_filename
            caller_line = frame.f_lineno + 1
            result = wrap_source(
                source,
                config=expected_config,
            )

        assert call_count == 1
        assert result == sentinel_source
        assert len(captured) == 2
        assert [str(warning.message) for warning in captured] == [
            'line 3: C.first skipped (quote-collision): quote delimiter collision',
            (
                'line 8: C.second skipped (validation-failed): '
                'escape spellings changed during rewrite'
            ),
        ]
        assert all(warning.category is DocwrapSkipWarning for warning in captured)
        assert [warning.filename for warning in captured] == [caller_filename, caller_filename]
        assert [warning.lineno for warning in captured] == [caller_line, caller_line]

    def test_warns_on_skip(self) -> None:
        source = 'def f():\n    """Line one ' + BACKSLASH + '\n    continued."""\n'
        with pytest.warns(DocwrapSkipWarning, match='line-continuation'):
            result = wrap_source(source)
        assert result == source


def _generate_idempotence_cases(case_count: int) -> list[tuple[str, int, bool]]:
    rng = random.Random(_IDEMPOTENCE_SEED)  # noqa: S311 -- approved deterministic test generator
    clauses = (
        'use `value` with --line-length and grouped words',
        'preserve **bold** markup and (parenthesized) phrases',
        'keep "quoted" digits 123 and option=value stable',
        'carry a deliberately long description across original boundaries',
    )
    line_lengths = (36, 44, 52, 60, 72, 79, 88)
    markers = ('-', '*', '+', '1.', '9.', '100.')
    parser_tails = (
        'Returns: parser-shaped tail',
        'Returns ------- parser-shaped tail',
        ':param value: parser-shaped tail',
        '- parser-shaped tail',
        ':: code_line() parser-shaped tail',
    )
    cases: list[tuple[str, int, bool]] = []

    for case_index in range(case_count):
        first_clause, second_clause = rng.sample(clauses, 2)
        gap = (' ', '  ', '   ')[case_index % 3]
        clause = f'{first_clause};{gap}{second_clause}; case {case_index}.'
        line_length = rng.choice(line_lengths)
        marker = markers[case_index % len(markers)]
        prose_indent = ' ' * rng.choice((0, 2, 4, 8))
        summary = f'Generated summary {case_index} with {rng.choice(clauses)}.'
        parser_tail = parser_tails[case_index % len(parser_tails)]
        layout = case_index % 11
        wrap_summary = layout in (6, 7, 10)

        if layout == 0:
            literal = (
                f'    """{summary}\n\n'
                '    Args:\n'
                f'        value (str): {clause}\n'
                f'    {marker} {clause}\n'
                f'{prose_indent}{clause}\n'
                '    """\n'
            )
        elif layout == 1:
            literal = f'    """    {summary}\n{clause}\n    """\n'
        elif layout == 2:
            literal = f'    """\n        code_line()\n    {clause}\n    """\n'
        elif layout == 3:
            continuation_indent = ' ' * rng.choice((4, 6, 8))
            literal = (
                f'    """{summary}\n\n'
                f'    {marker} {clause}\n'
                f'{continuation_indent}{clause}\n'
                f'    trailing prose {clause}\n'
                '    """\n'
            )
        elif layout == 4:
            literal = (
                f'    """{summary}\n\n'
                '    Parameters\n'
                '    ----------\n'
                '    value : str\n'
                f'        {clause}\n'
                '    """\n'
            )
        elif layout == 5:
            literal = (
                f'    """{summary}\n\n'
                f'    :param str value: {clause}\n'
                f'    :returns: {clause}\n'
                '    """\n'
            )
        elif layout in (6, 7):
            summary_opening = (
                f'    """{summary} {parser_tail} {clause}\n',
                f'    """\n    {summary} {parser_tail} {clause}\n',
            )[layout - 6]
            literal = f'{summary_opening}\n    Mutable body {clause}\n    """\n'
        elif layout in (8, 9):
            summary_opening = (
                f'    """Physical summary {case_index}.\n',
                f'    """\n    Physical summary {case_index}.\n',
            )[layout - 8]
            literal = (
                f'{summary_opening}    Second authored summary line.\n\n'
                f'    Mutable body {clause}\n'
                '    """\n'
            )
        else:
            inline_literal = (
                f'"""{summary} {parser_tail} {clause}\n\n    Mutable body {clause}\n    """\n',
                (
                    f'"""\n    {summary} {parser_tail} {clause}\n\n'
                    f'    Mutable body {clause}\n    """\n'
                ),
            )[(case_index // 11) % 2]
            literal = inline_literal

        source = (
            (f"def generated_{case_index}(value: str = 'x') -> str:\n{literal}    return value\n"),
            f"def generated_{case_index}(value: str = 'x') -> str: {literal}",
        )[layout == 10]
        ast.parse(source)
        cases.append((source, line_length, wrap_summary))

    assert len(cases) == case_count
    assert len({source for source, _, _ in cases}) == case_count
    return cases


def _assert_generated_idempotence(
    start_index: int,
    case_count: int,
    fill_style: FillStyle,
    reflow: ReflowMode,
) -> None:
    for case_index, (source, line_length, wrap_summary) in enumerate(
        _generate_idempotence_cases(case_count)[start_index:],
        start=start_index,
    ):
        context = (
            f'seed={_IDEMPOTENCE_SEED} case={case_index} '
            f'line_length={line_length} fill_style={fill_style} '
            f'reflow={reflow} wrap_summary={wrap_summary}'
        )
        config = DocwrapConfig(
            line_length=line_length,
            fill_style=fill_style,
            reflow=reflow,
            wrap_summary=wrap_summary,
        )
        try:
            first = wrap_source(source, config=config)
            ast.parse(first)
            assert_region_projection(source, config, emitted_source=first)
        except (AssertionError, RuntimeError, SyntaxError) as exc:
            raise AssertionError(context) from exc
        second = wrap_source(first, config=config)
        assert second == first, context


def _parse_one(source: str, config: DocwrapConfig | None = None) -> ParsedDocstring:
    cfg = config or DocwrapConfig()
    docstrings = find_docstrings(source)
    results = parse_docstrings(source, docstrings, cfg, skips=[])
    assert len(results) >= 1
    return results[0]


class TestAnalyzeSource:
    def test_no_docstrings(self) -> None:
        result = analyze_source('x = 1\n', DocwrapConfig())
        assert result.docstrings == []
        assert result.parsed == []
        assert not result.parse_error

    def test_returns_parsed_docstrings(self) -> None:
        source = 'def foo():\n    """Hello world."""\n'
        result = analyze_source(source, DocwrapConfig())
        assert len(result.docstrings) == 1
        assert len(result.parsed) == 1
        assert not result.parse_error

    def test_syntax_error_returns_empty(self) -> None:
        result = analyze_source('def foo(\n', DocwrapConfig())
        assert result.parse_error
        assert result.docstrings == []
        assert result.parsed == []

    @pytest.mark.parametrize('target', ['google', 'numpy'])
    @pytest.mark.parametrize('strip_types', [False, True], ids=['keep-types', 'strip-types'])
    @pytest.mark.parametrize(
        ('fields', 'return_count'),
        [
            pytest.param(
                ':returns: First.\n:returns: Second.',
                2,
                id='two-adjacent',
            ),
            pytest.param(
                ':returns: First.\n:returns: Second.\n:returns: Third.',
                3,
                id='three-adjacent',
            ),
            pytest.param(
                ':returns: First.\n:param x: Value.\n:returns: Second.',
                2,
                id='interleaved-param',
            ),
            pytest.param(
                ':returns: First.\n:raises ValueError: Bad value.\n:returns: Second.',
                2,
                id='interleaved-raises',
            ),
        ],
    )
    def test_typeless_sphinx_return_group_skips_lossy_conversion(
        self,
        target: StyleName,
        strip_types: bool,
        fields: str,
        return_count: int,
    ) -> None:
        source = f'def f(x):\n    """Summary.\n\n{textwrap.indent(fields, "    ")}\n    """\n'
        config = DocwrapConfig(convert=target, strip_types=strip_types)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert first.source == source
        assert first.source.count(':returns:') == return_count
        assert first.converted == 0
        assert len(first.skips) == 1
        assert first.skips[0].code == 'duplicate-field'
        assert 'target cannot represent multiple typeless returns' in first.skips[0].message
        assert second.source == first.source
        assert second.source.count(':returns:') == return_count
        assert second.converted == 0
        assert second.skips == first.skips


class TestEscapeRegionCoverage:
    def test_backtick_newline_escape_preserved(self) -> None:
        body = 'Human text contains raw bytes with `' + BACKSLASH + 'n` newlines.'
        source = 'def example():\n    """' + body + '"""\n'
        result = wrap_source(source, config=DocwrapConfig())
        assert BACKSLASH + 'n' in result
        assert '` `' not in result

    def test_convert_preserves_escape(self) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x: Uses `' + BACKSLASH + 'n` in its description text.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(convert='numpy', line_length=72)
        report = wrap_source_report(source, config=cfg)
        assert BACKSLASH + 'n' in report.source
        assert scan_escape_spellings(report.source) == scan_escape_spellings(source)
        assert wrap_source_report(report.source, config=cfg).source == report.source

    def test_convert_preserves_escape_bearing_unsupported_span(self) -> None:
        unsupported_span = (
            '    Notes\n'
            '    -----\n'
            '\n'
            '    Escape `' + BACKSLASH + 'n` stays authored.\n'
            '\n'
            '    ```text\n'
            '    fenced `' + BACKSLASH + 't` text\n'
            '    ```\n'
            '\n'
            '    >>> print("' + BACKSLASH + 'x41")\n'
            '    A\n'
            '\n'
            '    Literal::\n'
            '\n'
            '        literal ' + BACKSLASH + 'x42\n'
            '\n'
        )
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    x : int\n'
            '        Input.\n'
            '\n' + unsupported_span + '    Returns\n'
            '    -------\n'
            '    int\n'
            '        Result.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert='sphinx', line_length=88)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert unsupported_span in first.source
        assert scan_escape_spellings(first.source) == scan_escape_spellings(source)
        ast.parse(first.source)
        assert_region_projection(source, config, emitted_source=first.source)
        assert second.source == first.source

    def test_convert_reorder_keeps_escape_attachment(self) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    :returns: Return text `' + BACKSLASH + 'n` stays with the return.\n'
            '    :param x: Param text `' + BACKSLASH + 't` stays with the param.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(convert='google', line_length=72)
        report = wrap_source_report(source, config=cfg)
        param_text = ' '.join(line for line in report.source.splitlines() if 'Param text' in line)
        return_text = ' '.join(
            line for line in report.source.splitlines() if 'Return text' in line
        )
        assert BACKSLASH + 't' in param_text
        assert BACKSLASH + 'n' in return_text
        assert wrap_source_report(report.source, config=cfg).source == report.source

    def test_deleted_escape_text_is_materialized_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = (
            'def f():\n'
            '    """Summary with `'
            + BACKSLASH
            + 'n`, `'
            + BACKSLASH
            + 't`, and `'
            + BACKSLASH
            + 'x41`."""\n'
        )
        config = DocwrapConfig()
        parsed = analyze_source(source, config).parsed[0]
        assert len(parsed.escape_map) == 3
        deleted_text = _CountedDeletedText([])

        def unchanged_conversion(
            parsed_docstring: ParsedDocstring,
            _config: DocwrapConfig,
            wrap_style: Style | None,
        ) -> tuple[list[Region], Style | None, bool, list[str]]:
            assert parsed_docstring is parsed
            return list(parsed_docstring.regions), wrap_style, False, deleted_text

        monkeypatch.setattr(process_mod, '_convert_parsed_regions', unchanged_conversion)

        wrap_parsed_docstring(parsed, config, None, pre_wrapped=parsed.content)

        assert deleted_text.iterations == 1

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                'def f():\n'
                '    """Summary has escape `' + BACKSLASH + 't` plus enough words to push '
                'this summary well past the limit."""\n',
                id='summary',
            ),
            pytest.param(
                'def f():\n'
                '    """Summary.\n'
                '\n'
                '    Body prose with `' + BACKSLASH + 'n` and a long tail of filler words '
                'to force this paragraph to reflow onto extra lines.\n'
                '    """\n',
                id='body-prose',
            ),
            pytest.param(
                'def f():\n'
                '    """Summary.\n'
                '\n'
                '    - item with `' + BACKSLASH + 'x00` and a long tail of filler words to '
                'force this list item to wrap onto more lines.\n'
                '    """\n',
                id='list-item',
            ),
            pytest.param(
                'def f(x):\n'
                '    """Summary.\n'
                '\n'
                '    Args:\n'
                '        x: Value described with `' + BACKSLASH + 't` and a long tail of '
                'words forcing this description to wrap onto more lines.\n'
                '    """\n',
                id='google-entry',
            ),
            pytest.param(
                'def f(x):\n'
                '    """Summary.\n'
                '\n'
                '    Parameters\n'
                '    ----------\n'
                '    x : str\n'
                '        Value described with `' + BACKSLASH + 't` and a long tail of '
                'words forcing this description to wrap onto more lines.\n'
                '    """\n',
                id='numpy-entry',
            ),
            pytest.param(
                'def f(x):\n'
                '    """Summary.\n'
                '\n'
                '    :param x: Value described with `' + BACKSLASH + 't` and a long tail '
                'of words forcing this field to wrap onto more lines.\n'
                '    """\n',
                id='sphinx-field',
            ),
            pytest.param(
                'def f():\n'
                '    """Summary.\n'
                '\n'
                '    Example:\n'
                '\n'
                '        run --pattern ' + BACKSLASH + 'd+\n'
                '    """\n',
                id='code-block',
            ),
        ],
    )
    def test_escape_survives_region(self, source: str) -> None:
        cfg = DocwrapConfig(line_length=60, wrap_summary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            result = wrap_source(source, config=cfg)
            assert scan_escape_spellings(result) == scan_escape_spellings(source)
            assert wrap_source(result, config=cfg) == result

    def test_methods_type_retained_verbatim_under_convert_strip(self) -> None:
        # `methods` is not in _CONVERTIBLE_ROLES: the converter keeps the
        # entry and its type annotation verbatim even with strip_types, so
        # its escape must survive and must NOT be treated as an allowed loss.
        source = (
            'class C:\n'
            '    """Summary.\n'
            '\n'
            '    Methods:\n'
            '        run (Pattern' + BACKSLASH + 'n): Run it.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(convert='numpy', strip_types=True, line_length=72)
        report = wrap_source_report(source, config=cfg)
        assert 'Pattern' + BACKSLASH + 'n' in report.source
        assert report.skips == ()
        assert wrap_source_report(report.source, config=cfg).source == report.source

    def test_multiline_raw_trailing_backslash_keeps_spelling_and_value(self) -> None:
        source = 'def f():\n    r"""Summary.\n\n    Root is C:' + BACKSLASH + '\n    """\n'
        original_function = ast.parse(source).body[0]
        assert isinstance(original_function, ast.FunctionDef)
        original_value = ast.get_docstring(original_function, clean=False)

        report = wrap_source_report(source)

        assert report.source == source
        assert report.wrapped == 0
        assert 'r"""' in report.source
        assert 'C:' + BACKSLASH + '\n' in report.source
        result_function = ast.parse(report.source).body[0]
        assert isinstance(result_function, ast.FunctionDef)
        assert ast.get_docstring(result_function, clean=False) == original_value
        assert wrap_source(report.source) == report.source

    def test_nul_escape_does_not_break_file(self) -> None:
        body = 'Contains a NUL escape ' + BACKSLASH + 'x00 in prose.'
        source = 'def example():\n    """' + body + '"""\n'
        result = wrap_source(source, config=DocwrapConfig())
        assert chr(0) not in result
        assert BACKSLASH + 'x00' in result
        ast.parse(result)

    @pytest.mark.parametrize(
        'fields',
        [
            pytest.param(
                '    :param x: Kept.\n    :type x: First{bs}x41\n    :type x: Second{bs}x42\n',
                id='duplicate-escape',
            ),
            pytest.param(
                '    :param x: Kept.\n    :type x: FirstType\n    :type x: SecondType\n',
                id='duplicate-plain',
            ),
            pytest.param(
                '    :param x: First.\n    :param x: Second.\n    :type x: Pattern{bs}x41\n',
                id='fanout',
            ),
            pytest.param(
                '    :param Inline{bs}x41 x: Kept.\n    :type x: Separate\n',
                id='overwrite',
            ),
        ],
    )
    def test_pure_conversion_conflict_records_duplicate_field_skip(self, fields: str) -> None:
        source = 'def f():\n    """Summary.\n\n' + fields.format(bs=chr(92)) + '    """\n'
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)

        assert report.source == source
        assert len(report.skips) == 1
        assert report.skips[0].code == 'duplicate-field'
        second = wrap_source_report(report.source, config=config)
        assert second.source == source
        assert second.skips[0].code == 'duplicate-field'

    def test_pure_conversion_keeps_empty_synthesized_returns_header(self) -> None:
        source = 'def f():\n    """Summary.\n\n    :returns:\n    """\n'
        config = DocwrapConfig(convert='google')
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary.\n\n    Returns:\n    """\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_raw_docstring_long_prose_wraps_and_preserves(self) -> None:
        body = (
            'Regex ' + BACKSLASH + 'd+ with plenty of additional prose words that '
            'force this raw docstring summary to wrap onto further lines.'
        )
        source = 'def f():\n    r"""' + body + '"""\n'
        cfg = DocwrapConfig(line_length=50, wrap_summary=True)
        result = wrap_source(source, config=cfg)
        assert result != source
        assert BACKSLASH + 'd+' in result
        assert wrap_source(result, config=cfg) == result

    def test_raw_docstring_unchanged_behavior(self) -> None:
        source = 'def f():\n    r"""Regex ' + BACKSLASH + 'd+ short."""\n'
        result = wrap_source(source, config=DocwrapConfig())
        assert result == source

    @pytest.mark.parametrize('spelling', ESCAPE_MATRIX)
    def test_round_trip_preserves_spelling(self, spelling: str) -> None:
        body = (
            'Prefix words before the escape `'
            + spelling
            + '` and enough trailing prose to force the wrapper to reflow this body.'
        )
        source = 'def f():\n    """' + body + '"""\n'
        cfg = DocwrapConfig(line_length=50, wrap_summary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            result = wrap_source(source, config=cfg)
            assert result != source  # prose wrapped
            assert scan_escape_spellings(result) == scan_escape_spellings(source)
            ast.parse(result)  # output compiles
            second = wrap_source(result, config=cfg)
        assert second == result  # idempotent

    def test_strip_types_deletes_duplicate_escape_fields(self) -> None:
        bs = chr(92)
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    :param x: Kept.\n'
            '    :type x: First' + bs + 'x41\n'
            '    :type x: Second' + bs + 'x42\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == ('def f():\n    """Summary.\n\n    :param x: Kept.\n    """\n')
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_deletes_escape_before_trailing_prose_idempotently(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    :type x: Pattern' + BACKSLASH + 'x41\n\n'
            '    Authored trailing prose remains here.\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True)

        first = wrap_source_report(source, config=config)
        second = wrap_source_report(first.source, config=config)

        assert BACKSLASH + 'x41' not in first.source
        assert 'Authored trailing prose remains here.' in first.source
        assert first.skips == ()
        assert first.source == (
            'def f():\n    """Summary.\n\n    Authored trailing prose remains here.\n    """\n'
        )
        assert second.source == first.source
        assert second.skips == ()

    def test_strip_types_deletes_escape_inside_type(self) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x (matches '
            + BACKSLASH
            + 'd): Description keeps `'
            + BACKSLASH
            + 't` here.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(strip_types=True, line_length=72)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            report = wrap_source_report(source, config=cfg)
            assert BACKSLASH + 'd' not in report.source  # deleted with the type
            assert BACKSLASH + 't' in report.source  # survives in the description
            assert report.skips == ()  # allowed loss, not a validation failure
            assert wrap_source_report(report.source, config=cfg).source == report.source

    def test_strip_types_deletes_escape_to_stable_empty_body(self) -> None:
        source = (
            'def f(x):\n    """Summary.\n\n    :type x: Pattern' + BACKSLASH + 'x41\n    """\n'
        )
        cfg = DocwrapConfig(strip_types=True)

        first = wrap_source_report(source, config=cfg)
        second = wrap_source_report(first.source, config=cfg)

        assert BACKSLASH + 'x41' not in first.source
        assert first.skips == ()
        assert first.source == 'def f(x):\n    """Summary."""\n'
        assert second.source == first.source
        assert second.skips == ()

    def test_strip_types_keeps_numpy_header_with_following_prose(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    Returns\n'
            '    -------\n'
            '    int\n\n'
            '    After prose.\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert 'Returns' in report.source
        assert 'After prose.' in report.source
        assert 'int' not in report.source
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_keeps_populated_returns_header(self) -> None:
        bs = chr(92)
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    :returns: A real description.\n'
            '    :rtype: Pattern' + bs + 'x41\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True, convert='google')
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert 'Returns:' in report.source
        assert 'A real description.' in report.source
        assert bs + 'x41' not in report.source
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_keeps_type_free_empty_synthesized_returns_header(self) -> None:
        source = 'def f():\n    """Summary.\n\n    :returns:\n    """\n'
        config = DocwrapConfig(strip_types=True, convert='google')
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary.\n\n    Returns:\n    """\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    @pytest.mark.parametrize('target_name', ['google', 'numpy', 'sphinx'])
    def test_strip_types_matched_rtype_leaves_no_empty_section(
        self, target_name: StyleName
    ) -> None:
        source = 'def f():\n    """Summary.\n\n    :returns:\n    :rtype: int\n    """\n'
        config = DocwrapConfig(strip_types=True, convert=target_name)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary."""\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_strip_types_orphan_rtype_keeps_trailing_prose(self, target_name: StyleName) -> None:
        bs = chr(92)
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    :rtype: Pattern' + bs + 'x41\n\n'
            '    Trailing prose survives.\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True, convert=target_name)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == (
            'def f():\n    """Summary.\n\n    Trailing prose survives.\n    """\n'
        )
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    def test_strip_types_orphan_rtype_leaves_no_empty_section(
        self, target_name: StyleName
    ) -> None:
        bs = chr(92)
        source = 'def f():\n    """Summary.\n\n    :rtype: Pattern' + bs + 'x41\n    """\n'
        config = DocwrapConfig(strip_types=True, convert=target_name)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary."""\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_preserves_escape(self) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        x (str): Uses `' + BACKSLASH + 't` in its description text.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(strip_types=True, line_length=72)
        report = wrap_source_report(source, config=cfg)
        assert BACKSLASH + 't' in report.source
        assert wrap_source_report(report.source, config=cfg).source == report.source

    def test_strip_types_removes_emptied_google_returns_section(self) -> None:
        source = 'def f():\n    """Summary.\n\n    Returns:\n        int:\n    """\n'
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary."""\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_removes_emptied_header_before_outside_prose(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    Returns:\n'
            '        int:\n\n'
            '    Closing prose at base indent.\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == (
            'def f():\n    """Summary.\n\n    Closing prose at base indent.\n    """\n'
        )
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    def test_strip_types_removes_emptied_section_between_sections(self) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n\n'
            '    Returns:\n'
            '        int:\n\n'
            '    Raises:\n'
            '        ValueError: When invalid.\n'
            '    """\n'
        )
        config = DocwrapConfig(strip_types=True)
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == (
            'def f():\n'
            '    """Summary.\n\n'
            '    Raises:\n'
            '        ValueError: When invalid.\n'
            '    """\n'
        )
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source

    @pytest.mark.parametrize(
        ('source_name', 'source'),
        [
            pytest.param(
                'google',
                'def f():\n    """Summary.\n\n    Returns:\n        int:\n    """\n',
                id='google',
            ),
            pytest.param(
                'numpy',
                'def f():\n    """Summary.\n\n    Returns\n    -------\n    int\n    """\n',
                id='numpy',
            ),
        ],
    )
    def test_strip_types_removes_emptied_section_when_converting_to_sphinx(
        self, source_name: StyleName, source: str
    ) -> None:
        config = DocwrapConfig(style=source_name, strip_types=True, convert='sphinx')
        report = wrap_source_report(source, config=config)

        assert report.skips == ()
        assert report.source == 'def f():\n    """Summary."""\n'
        second = wrap_source_report(report.source, config=config)
        assert second.source == report.source


class TestNbspPreservation:
    _OVERLONG_CONTINUATION = (
        'def f(a):\n'
        '    """Summary.\n'
        '\n'
        '    Args:\n'
        '        a: Short first line.\n'
        '            \u00a0\u00a0aligned continuation content which is deliberately'
        ' made far too long to fit inside the configured width.\n'
        '    """\n'
    )

    def test_ascii_indentation_still_trims(self) -> None:
        source = (
            'def f(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: First line of the description which is deliberately made far'
            ' too long to fit the width.\n'
            '            plain continuation content here.\n'
            '    """\n'
        )
        out = wrap_source(source, config=DocwrapConfig(reflow='full'))
        assert 'plain continuation content here.' in out
        assert '  plain continuation' not in out

    @pytest.mark.parametrize('reflow', ['cautious', 'full'])
    def test_nbsp_survives_rewrap(self, reflow: ReflowMode) -> None:
        cfg = DocwrapConfig(reflow=reflow)
        report = wrap_source_report(self._OVERLONG_CONTINUATION, config=cfg)
        assert report.source.count('\u00a0') == 2
        assert [skip.code for skip in report.skips] == []
        assert wrap_source(report.source, config=cfg) == report.source

    def test_nbsp_untouched_line_is_stable(self) -> None:
        source = (
            'def f(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: First line of the description.\n'
            '            \u00a0\u00a0aligned continuation content here.\n'
            '    """\n'
        )
        out = wrap_source(source)
        assert out.count('\u00a0') == 2

    def test_tab_indentation_still_trims(self) -> None:
        source = (
            'def f(a):\n'
            '    """Summary.\n'
            '\n'
            '    Args:\n'
            '        a: First line of the description which is deliberately made far'
            ' too long to fit the width.\n'
            '            \tplain continuation content here.\n'
            '    """\n'
        )
        out = wrap_source(source, config=DocwrapConfig(reflow='full'))
        assert 'plain continuation content here.' in out
        assert '\t' not in out

    @pytest.mark.parametrize(
        'whitespace',
        [pytest.param('\x0b', id='vertical-tab'), pytest.param('\x0c', id='form-feed')],
    )
    def test_textwrap_whitespace_normalization_unchanged(self, whitespace: str) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            f'    Prose with{whitespace}whitespace that is deliberately far too long for'
            ' the configured line length and wraps.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(reflow='full')
        out = wrap_source(source, config=cfg)
        assert 'with whitespace' in out
        assert whitespace not in out
        assert wrap_source(out, config=cfg) == out


class TestNoWrapSummaryProvenance:
    @staticmethod
    def _source(summary_lines: tuple[str, ...], body: str, layout: str) -> str:
        if layout == 'first-line':
            opening = f'    """{summary_lines[0]}\n'
            summary = ''.join(f'    {line}\n' for line in summary_lines[1:])
        else:
            opening = '    """\n'
            summary = ''.join(f'    {line}\n' for line in summary_lines)
        return f'def f(x=None):\n{opening}{summary}\n{textwrap.indent(body, "    ")}\n    """\n'

    @staticmethod
    def _summary_lines(source: str, config: DocwrapConfig) -> tuple[str, ...]:
        analysis = analyze_source(source, config)
        assert not analysis.parse_error
        assert len(analysis.parsed) == 1
        summary = analysis.parsed[0].regions[0]
        assert summary.kind == 'summary'
        assert summary.original_lines is not None
        return summary.original_lines

    def test_no_wrap_default_provenance_trigger_preserves_source_lines(self) -> None:
        body = 'body ' * 25
        source = (
            'def f():\n'
            '    """Summary that runs on and on across two source lines and then\n'
            '    keeps going a bit more here.\n'
            '\n'
            f'    {body}\n'
            '    """\n'
        )
        config = DocwrapConfig()
        expected_summary = (
            'Summary that runs on and on across two source lines and then',
            'keeps going a bit more here.',
        )

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert first != source
        assert self._summary_lines(first, config) == expected_summary
        summary_source_lines = first.splitlines()[1:3]
        assert max(map(len, summary_source_lines)) <= 79
        assert second == first
        assert_region_projection(source, config, emitted_source=first)

    def test_no_wrap_google_conversion_conflict_binds_returns_header(self) -> None:
        summary_lines = (
            'Lead  text.',
            'Returns:',
            'Unaffected  trailing prose.',
        )
        source = self._source(summary_lines, ':param str value: Converted.', 'first-line')
        config = DocwrapConfig(style='sphinx', convert='google')
        expected = (
            'Lead  text. Returns:',
            'Unaffected  trailing prose.',
        )

        first = wrap_source(source, config=config)

        assert self._summary_lines(first, config) == expected
        assert '    Args:\n        value (str): Converted.' in first
        assert wrap_source(first, config=config) == first
        assert_region_projection(source, config, emitted_source=first)

    def test_no_wrap_long_logical_provenance_remains_physically_compliant(self) -> None:
        summary_lines = tuple(f'Line {index}: ' + 'x' * 64 for index in range(1, 7))
        assert all(len(line) == 72 for line in summary_lines)
        assert len(' '.join(summary_lines)) > 400
        source = self._source(summary_lines, 'body ' * 25, 'first-line')
        config = DocwrapConfig()

        first = wrap_source(source, config=config)

        assert self._summary_lines(first, config) == summary_lines
        assert all(len(line) <= config.line_length for line in first.splitlines()[1:7])
        assert wrap_source(first, config=config) == first
        assert_region_projection(source, config, emitted_source=first)

    def test_no_wrap_numpy_conversion_conflict_binds_returns_header(self) -> None:
        summary_lines = (
            'Lead  text.',
            'Returns',
            '-------',
            'Unaffected  trailing prose.',
        )
        source = self._source(summary_lines, ':param str value: Converted.', 'first-line')
        config = DocwrapConfig(style='sphinx', convert='numpy')
        expected = (
            'Lead  text. Returns',
            '-------',
            'Unaffected  trailing prose.',
        )

        first = wrap_source(source, config=config)

        assert self._summary_lines(first, config) == expected
        assert '    Parameters\n    ----------\n    value : str' in first
        assert wrap_source(first, config=config) == first
        assert_region_projection(source, config, emitted_source=first)

    @pytest.mark.parametrize(
        ('summary_start', 'layout'),
        [
            pytest.param('first-line', 'first-line', id='first-line'),
            pytest.param('second-line', 'first-line', id='second-line'),
            pytest.param('preserve', 'first-line', id='preserve-first-line'),
            pytest.param('preserve', 'second-line', id='preserve-second-line'),
        ],
    )
    @pytest.mark.parametrize(
        ('trigger', 'body', 'base_config'),
        [
            pytest.param(
                'body',
                'body ' * 25,
                DocwrapConfig(),
                id='body',
            ),
            pytest.param(
                'google-entry',
                'Args:\n    x: ' + 'detail ' * 20,
                DocwrapConfig(style='google'),
                id='google-entry',
            ),
            pytest.param(
                'conversion',
                ':param str x: Short.',
                DocwrapConfig(style='sphinx', convert='google'),
                id='conversion',
            ),
            pytest.param(
                'strip-types',
                'Args:\n    x (str): Short.',
                DocwrapConfig(style='google', strip_types=True),
                id='strip-types',
            ),
        ],
    )
    def test_no_wrap_provenance_survives_processing_trigger(
        self,
        trigger: str,
        body: str,
        base_config: DocwrapConfig,
        summary_start: SummaryStart,
        layout: str,
    ) -> None:
        summary_lines = (
            'Authored summary first line stays exactly here.',
            'Second authored summary line stays exactly here.',
        )
        source = self._source(summary_lines, body, layout)
        config = DocwrapConfig(
            convert=base_config.convert,
            line_length=base_config.line_length,
            strip_types=base_config.strip_types,
            style=base_config.style,
            summary_start=summary_start,
        )

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert first != source
        assert self._summary_lines(first, config) == summary_lines
        if trigger == 'body':
            assert len([line for line in first.splitlines() if 'body' in line]) == 2
        elif trigger == 'google-entry':
            assert first.count('detail') == 20
            assert len(next(line for line in source.splitlines() if 'detail' in line)) > 79
            assert all(len(line) <= 79 for line in first.splitlines() if 'detail' in line)
        elif trigger == 'conversion':
            assert '    Args:' in first
            assert ':param str x:' not in first
        else:
            assert 'x (str):' not in first
            assert '        x: Short.' in first
        assert second == first
        assert_region_projection(source, config, emitted_source=first)

    def test_no_wrap_sphinx_conversion_conflict_binds_field_token(self) -> None:
        summary_lines = (
            'Lead  text.',
            ':param x: remains  prose',
            'Unaffected  trailing prose.',
        )
        body = 'Args:\n    value: Converted.'
        source = self._source(summary_lines, body, 'first-line')
        config = DocwrapConfig(style='google', convert='sphinx')
        expected = (
            'Lead  text. :param',
            'x: remains  prose',
            'Unaffected  trailing prose.',
        )

        first = wrap_source(source, config=config)

        assert self._summary_lines(first, config) == expected
        assert '    :param value: Converted.' in first
        assert wrap_source(first, config=config) == first
        assert_region_projection(source, config, emitted_source=first)


class TestOpeningTitlePreservation:
    @pytest.mark.parametrize(
        'title_lines',
        ['Heading\n-------', '=======\n  Heading\n======='],
        ids=['underline', 'overline'],
    )
    @pytest.mark.parametrize(
        ('separator', 'expected'),
        [
            ('', '-------\n    body'),
            ('\n', '-------\n\n    body'),
            ('\n\n', '-------\n\n\n    body'),
        ],
    )
    def test_authored_blank_count_is_exact(
        self,
        title_lines: str,
        separator: str,
        expected: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        expected_last_line = title_lines.splitlines()[-1]
        expected = expected.replace('-------', expected_last_line)
        source = f'def f():\n    """{indented_title}\n{separator}    body\n    """\n'
        assert expected in wrap_source(source)

    @pytest.mark.parametrize('wrap_summary', [False, True])
    @pytest.mark.parametrize(
        ('title_lines', 'summary_start', 'opening'),
        [
            (
                'Heading\n-------',
                'preserve',
                '"""Heading\n    -------',
            ),
            (
                'Heading\n-------',
                'first-line',
                '"""Heading\n    -------',
            ),
            (
                'Heading\n-------',
                'second-line',
                '"""\n    Heading\n    -------',
            ),
            (
                '=======\n  Heading\n=======',
                'preserve',
                '"""=======\n      Heading\n    =======',
            ),
            (
                '=======\n  Heading\n=======',
                'first-line',
                '"""=======\n      Heading\n    =======',
            ),
            (
                '=======\n  Heading\n=======',
                'second-line',
                '"""\n    =======\n      Heading\n    =======',
            ),
        ],
    )
    def test_configuration_preserves_title_and_body(
        self,
        wrap_summary: bool,
        title_lines: str,
        summary_start: SummaryStart,
        opening: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        source = (
            'def f():\n'
            f'    """{indented_title}\n'
            '    body words that remain wrappable below the title\n'
            '    """\n'
        )
        config = DocwrapConfig(
            line_length=35,
            summary_start=summary_start,
            wrap_summary=wrap_summary,
        )
        first = wrap_source_report(source, config=config)
        assert opening in first.source
        assert 'body words' in first.source
        assert first.skips == ()
        assert first.skipped_summaries == 0
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source

    def test_quote_adornment_uses_noncolliding_delimiter(self) -> None:
        source = 'def f():\n    \'\'\'Heading\n    """""""\n    body\n    \'\'\'\n'
        report = wrap_source_report(source, config=DocwrapConfig())
        assert 'Heading\n    """""""\n' in report.source
        assert report.skips == ()
        assert wrap_source(report.source) == report.source

    def test_raw_backslash_adornment_is_representable(self) -> None:
        adornment = BACKSLASH * len('Header')
        source = f'def f():\n    r"""Header\n    {adornment}\n    body\n    """\n'
        report = wrap_source_report(source, config=DocwrapConfig())
        assert f'Header\n    {adornment}\n' in report.source
        assert report.skips == ()
        assert wrap_source(report.source) == report.source

    def test_title_body_does_not_increment_summary_skip(self) -> None:
        source = (
            'def f():\n'
            '    """Heading\n'
            '    -------\n'
            '    body words that are intentionally much longer than the limit\n'
            '    """\n'
        )
        config = DocwrapConfig(
            fill_style='greedy',
            line_length=25,
            reflow='full',
            wrap_summary=False,
        )
        report = wrap_source_report(source, config=config, detail='list')
        assert report.skipped_summaries == 0
        assert report.docstrings[0].summary_skipped is False
        assert (
            '    body words that are\n    intentionally much\n    longer than the limit\n'
        ) in report.source

    def test_title_only_stays_multiline(self) -> None:
        source = 'def f():\n    """Heading\n    -------\n    """\n'
        result = wrap_source(source)
        assert result == source
        assert wrap_source(result) == result

    @pytest.mark.parametrize(
        ('style', 'body', 'retained'),
        [
            (None, 'Args:\n    x: Kept Google.', 'Kept Google.'),
            (
                None,
                'Parameters\n----------\nx\n    Kept NumPy.',
                'Kept NumPy.',
            ),
            (None, ':param x: Kept Sphinx.', 'Kept Sphinx.'),
            ('google', 'Args:\n    x: Kept Google.', 'Kept Google.'),
            (
                'numpy',
                'Parameters\n----------\nx\n    Kept NumPy.',
                'Kept NumPy.',
            ),
            ('sphinx', ':param x: Kept Sphinx.', 'Kept Sphinx.'),
        ],
        ids=[
            'auto-google',
            'auto-numpy',
            'auto-sphinx',
            'explicit-google',
            'explicit-numpy',
            'explicit-sphinx',
        ],
    )
    @pytest.mark.parametrize(
        'title_lines',
        ['Heading\n~~~~~~~', '=======\n  Heading\n======='],
        ids=['underline', 'overline'],
    )
    def test_title_precedes_each_supported_body_style(
        self,
        style: StyleName | None,
        body: str,
        retained: str,
        title_lines: str,
    ) -> None:
        indented_title = title_lines.replace('\n', '\n    ')
        indented_body = body.replace('\n', '\n    ')
        source = f'def f(x):\n    """{indented_title}\n    {indented_body}\n    """\n'
        config = DocwrapConfig(style=style, line_length=35)
        first = wrap_source_report(source, config=config)
        assert indented_title in first.source
        assert retained in first.source
        assert first.skips == ()
        second = wrap_source_report(first.source, config=config)
        assert second.source == first.source


class TestProcessIdempotent:
    def test_idempotent(self) -> None:
        source = textwrap.dedent("""\
            def foo(x, y):
                \"\"\"A very long summary that should be wrapped because it exceeds the limit.

                Args:
                    x: The x value with a description that is quite long.
                    y: The y value.

                Returns:
                    The computed result.
                \"\"\"
                return x + y
        """)
        first = wrap_source(source, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        second = wrap_source(first, config=DocwrapConfig(line_length=DEFAULT_LINE_LENGTH))
        assert first == second, 'Not idempotent!'

    @pytest.mark.parametrize('reflow', ['none', 'cautious'])
    def test_section_paragraph_indent_is_applied_once(self, reflow: ReflowMode) -> None:
        source = (
            'def f():\n    """Summary.\n\n    Notes:\n        Authored section prose.\n    """\n'
        )
        config = DocwrapConfig(reflow=reflow)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert first == source
        assert second == first

    @pytest.mark.parametrize(
        ('style_name', 'return_section'),
        [
            pytest.param(
                'google',
                '    Returns:\n        int: The result.',
                id='google',
            ),
            pytest.param(
                'numpy',
                '    Returns\n    -------\n    int\n        The result.',
                id='numpy',
            ),
        ],
    )
    def test_stripped_return_prose_has_four_content_spaces(
        self,
        style_name: StyleName,
        return_section: str,
    ) -> None:
        source = f'def f():\n    """Summary.\n\n{return_section}\n    """\n'
        config = DocwrapConfig(style=style_name, strip_types=True)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        compile(first, '<test>', 'exec')
        description_line = next(line for line in first.splitlines() if 'The result.' in line)
        assert description_line == '        The result.'
        assert second == first

    def test_tab_indented_docstring_is_value_stable(self) -> None:
        source = (
            'def f():\n'
            '\t"""Summary.\n'
            '\n'
            '\tThis tab-indented function docstring has enough prose to wrap at the '
            'configured narrow line length.\n'
            '\t"""\n'
        )
        config = DocwrapConfig(line_length=45)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_node = ast.parse(first).body[0]
        second_node = ast.parse(second).body[0]

        assert isinstance(first_node, ast.FunctionDef)
        assert isinstance(second_node, ast.FunctionDef)
        assert second == first
        assert ast.get_docstring(second_node, clean=False) == ast.get_docstring(
            first_node,
            clean=False,
        )
        assert all(line.startswith('\t') for line in first.splitlines()[2:] if line.strip())

    def test_tab_indented_inline_docstring_is_value_stable(self) -> None:
        source = (
            'class C:\n'
            '\tdef f(self): """This inline function docstring has enough summary '
            'text to wrap at the configured narrow line length."""\n'
        )
        config = DocwrapConfig(line_length=45, wrap_summary=True)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_class = ast.parse(first).body[0]
        second_class = ast.parse(second).body[0]

        assert isinstance(first_class, ast.ClassDef)
        assert isinstance(second_class, ast.ClassDef)
        first_node = first_class.body[0]
        second_node = second_class.body[0]
        assert isinstance(first_node, ast.FunctionDef)
        assert isinstance(second_node, ast.FunctionDef)
        assert second == first
        assert ast.get_docstring(second_node, clean=False) == ast.get_docstring(
            first_node,
            clean=False,
        )
        assert all(
            line.startswith(' ') and not line.startswith('\t')
            for line in first.splitlines()[2:]
            if line.strip()
        )
        assert first.splitlines()[-1] == ' """'

    def test_tab_indented_inline_multiline_docstring_is_value_stable(self) -> None:
        source = (
            'class C:\n'
            '\tdef f(self): """This inline function\n'
            '\tdocstring has enough summary\n'
            '\ttext to wrap at the configured\n'
            '\tnarrow line length.\n'
            '\t"""\n'
        )
        config = DocwrapConfig(line_length=45, wrap_summary=True)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_class = ast.parse(first).body[0]
        second_class = ast.parse(second).body[0]

        assert isinstance(first_class, ast.ClassDef)
        assert isinstance(second_class, ast.ClassDef)
        first_node = first_class.body[0]
        second_node = second_class.body[0]
        assert isinstance(first_node, ast.FunctionDef)
        assert isinstance(second_node, ast.FunctionDef)
        assert second == first
        assert ast.get_docstring(second_node, clean=False) == ast.get_docstring(
            first_node,
            clean=False,
        )
        assert all(
            line.startswith(' ') and not line.startswith('\t')
            for line in first.splitlines()[2:]
            if line.strip()
        )
        assert first.splitlines()[-1] == ' """'

    def test_tab_indented_parenthesized_docstring_is_value_stable(self) -> None:
        source = (
            'class C:\n'
            '\tdef f(self):\n'
            '\t\t("""This parenthesized function docstring has enough summary '
            'text to wrap at the configured narrow line length.""")\n'
        )
        config = DocwrapConfig(line_length=45, wrap_summary=True)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_class = ast.parse(first).body[0]
        second_class = ast.parse(second).body[0]

        assert isinstance(first_class, ast.ClassDef)
        assert isinstance(second_class, ast.ClassDef)
        first_node = first_class.body[0]
        second_node = second_class.body[0]
        assert isinstance(first_node, ast.FunctionDef)
        assert isinstance(second_node, ast.FunctionDef)
        assert second == first
        assert ast.get_docstring(second_node, clean=False) == ast.get_docstring(
            first_node,
            clean=False,
        )
        assert all(
            line.startswith('  ') and not line.startswith('\t')
            for line in first.splitlines()[3:]
            if line.strip()
        )
        assert first.splitlines()[-1] == '  """)'

    def test_typed_attribute_sphinx_conversion_is_idempotent(self) -> None:
        source = (
            'def f():\n    """Summary.\n\n    Attributes:\n        attr (float): Value.\n    """\n'
        )
        config = DocwrapConfig(convert='sphinx')

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert ':ivar attr: Value.\n    :vartype attr: float' in first
        assert second == first

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize('target_name', ['google', 'numpy'])
    @pytest.mark.parametrize('boundary', ['start', 'stop'])
    def test_unmapped_sphinx_field_span_has_projection_boundaries(
        self,
        role: str,
        target_name: StyleName,
        boundary: str,
    ) -> None:
        if boundary == 'start':
            source = (
                'def f(x):\n'
                '    """Summary.\n'
                '\n'
                '    :param x: Before.\n'
                '    Authored prose for the parameter group.\n'
                f'    :{role} registry: Opaque.\n'
                '    :returns: Done.\n'
                '    """\n'
            )
            ordered_text = [
                'Before.',
                'Authored prose for the parameter group.',
                f':{role} registry: Opaque.',
                'Done.',
            ]
        else:
            source = (
                'def f():\n'
                '    """Summary.\n'
                '\n'
                f'    :{role} registry: Opaque.\n'
                '    Associated prose.\n'
                '    :vartype orphan: int\n'
                '    """\n'
            )
            ordered_text = [
                f':{role} registry: Opaque.',
                'Associated prose.',
                ':vartype orphan: int',
            ]
        config = DocwrapConfig(convert=target_name)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert f'    :{role} registry: Opaque.' in first.splitlines()
        assert [first.index(text) for text in ordered_text] == sorted(
            first.index(text) for text in ordered_text
        )
        assert second == first
        assert_region_projection(
            source,
            DocwrapConfig(style='sphinx', convert=target_name),
        )

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize(
        ('target_name', 'before_header'),
        [
            pytest.param('google', '    Args:', id='google'),
            pytest.param('numpy', '    Parameters', id='numpy'),
        ],
    )
    def test_unmapped_sphinx_field_span_is_exact_and_idempotent(
        self,
        role: str,
        target_name: StyleName,
        before_header: str,
    ) -> None:
        source = (
            'def f(before):\n'
            '    """Summary.\n'
            '\n'
            '    :param int before: Converted before.\n'
            f'    :{role} registry: Authored  value.\n'
            '        continuation  spacing.\n'
            '    :vartype registry: dict[str, int]\n'
            '\n'
            '    Associated  prose after field.\n'
            '\n'
            '    :returns: Converted after.\n'
            '    :rtype: bool\n'
            '    """\n'
        )
        config = DocwrapConfig(convert=target_name)
        expected_span = (
            f'    :{role} registry: Authored  value.\n'
            '        continuation  spacing.\n'
            '    :vartype registry: dict[str, int]\n'
            '\n'
            '    Associated  prose after field.\n'
            '\n'
        )

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)

        assert expected_span in first
        assert first.index(before_header) < first.index(expected_span)
        assert first.index(expected_span) < first.index('    Returns')
        assert second == first

    @pytest.mark.parametrize('role', ['cvar', 'var'])
    @pytest.mark.parametrize(
        ('target_name', 'expected_returns'),
        [
            pytest.param(
                'google',
                '    Returns:\n        Done.',
                id='google',
            ),
            pytest.param(
                'numpy',
                '    Returns\n    -------\n        Done.',
                id='numpy',
            ),
        ],
    )
    def test_unmapped_sphinx_vartype_association_is_global(
        self,
        role: str,
        target_name: StyleName,
        expected_returns: str,
    ) -> None:
        source = (
            'def f(x):\n'
            '    """Summary.\n'
            '\n'
            f'    :{role} registry: Opaque.\n'
            '    :param x: Between.\n'
            '    :vartype registry: dict[str, int]\n'
            '    :returns: Done.\n'
            '    """\n'
        )
        config = DocwrapConfig(convert=target_name)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_lines = first.splitlines()

        assert f'    :{role} registry: Opaque.' in first_lines
        assert '    :vartype registry: dict[str, int]' in first_lines
        assert expected_returns in first
        assert first.index(f':{role} registry:') < first.index('Between.')
        assert first.index('Between.') < first.index(':vartype registry:')
        assert first.index(':vartype registry:') < first.index('Done.')
        assert second == first
        assert_region_projection(
            source,
            DocwrapConfig(style='sphinx', convert=target_name),
        )


class TestProtectedOneLineRegion:
    _SOURCE = (
        'def f():\n'
        '    """\n'
        '        result = compute_the_value(alpha, beta, gamma)'
        ' + adjust(delta, epsilon, zeta, eta)\n'
        '    """\n'
    )

    @pytest.mark.parametrize('wrap_summary', [False, True])
    def test_four_space_indentation_is_preserved_and_idempotent(
        self,
        wrap_summary: bool,
    ) -> None:
        cfg = DocwrapConfig(wrap_summary=wrap_summary)
        once = wrap_source(self._SOURCE, config=cfg)
        assert once == self._SOURCE
        assert wrap_source(once, config=cfg) == once


class TestReviewRemediationCorpus:
    def _assert_case_invariants(
        self,
        source: str,
        case: ReviewRemediationCase,
        config: DocwrapConfig,
    ) -> None:
        analysis = analyze_source(source, config)
        assert not analysis.parse_error
        assert len(analysis.parsed) == 1
        content_lines = analysis.parsed[0].content.split('\n')
        for invariant in case.line_invariants:
            matches = [
                line
                for line in content_lines
                if (line == invariant.fragment if invariant.exact else invariant.fragment in line)
            ]
            assert len(matches) == 1, (case.name, invariant.fragment, content_lines)
            line = matches[0]
            indent = len(line) - len(line.lstrip())
            assert indent == invariant.indent, (
                case.name,
                invariant.fragment,
                invariant.indent,
                indent,
            )
        for boundary in case.blank_boundaries:
            before = next(
                index for index, line in enumerate(content_lines) if boundary.before in line
            )
            after = next(
                index for index, line in enumerate(content_lines) if boundary.after in line
            )
            assert before < after
            assert '' in content_lines[before + 1 : after]

    @pytest.mark.parametrize('case', REVIEW_REMEDIATION_CASES, ids=lambda case: case.name)
    @pytest.mark.parametrize('fill_style', FILL_STYLES)
    @pytest.mark.parametrize('reflow', REFLOW_MODES)
    def test_every_case_is_valid_injective_and_idempotent(
        self,
        case: ReviewRemediationCase,
        fill_style: FillStyle,
        reflow: ReflowMode,
    ) -> None:
        config = DocwrapConfig(fill_style=fill_style, reflow=reflow)
        first = wrap_source(case.source, config=config)
        ast.parse(first)
        second = wrap_source(first, config=config)
        assert second == first, case.name
        assert_region_projection(case.source, config, emitted_source=first)
        self._assert_case_invariants(first, case, config)


class TestSafetySkipReportDetail:
    _DUPLICATE = (
        'def f():\n'
        '    """Summary.\n'
        '\n'
        '    :returns: First thing.\n'
        '    :returns: Second thing.\n'
        '    """\n'
    )
    _NAMED_RETURN = (
        'def f():\n'
        '    """Summary.\n'
        '\n'
        '    Returns\n'
        '    -------\n'
        '    count : int\n'
        '        The number of items.\n'
        '    """\n'
    )

    @pytest.mark.parametrize('detail', ['summary', 'list', 'diff'])
    def test_duplicate_field_skip_reports(self, detail: DetailLevel) -> None:
        report = wrap_source_report(
            self._DUPLICATE, config=DocwrapConfig(convert='google'), detail=detail
        )
        assert [s.code for s in report.skips] == ['duplicate-field']
        assert report.source == self._DUPLICATE

    @pytest.mark.parametrize('detail', ['summary', 'list', 'diff'])
    @pytest.mark.parametrize(
        'config',
        [DocwrapConfig(convert='sphinx'), DocwrapConfig(strip_types=True)],
    )
    def test_unrepresentable_return_skip_reports(
        self, config: DocwrapConfig, detail: DetailLevel
    ) -> None:
        report = wrap_source_report(self._NAMED_RETURN, config=config, detail=detail)
        assert [s.code for s in report.skips] == ['unrepresentable-return']
        assert report.source == self._NAMED_RETURN


class TestAlignedRowWrapping:
    def test_prose_around_listing_still_reflows(self) -> None:
        source = (
            '"""Summary.\n\n'
            'Ordinary prose continues\nonto another line.\n\n'
            '`scalar`   a number\n`mapping`  a rollup\n\n'
            'Following prose continues\nonto another line.\n"""\n'
        )
        once = wrap_source(source)
        assert 'Ordinary prose continues onto another line.' in once
        assert '`scalar`   a number\n`mapping`  a rollup' in once
        assert 'Following prose continues onto another line.' in once
        assert wrap_source(once) == once

    @pytest.mark.parametrize('width', [40, 79])
    @pytest.mark.parametrize('fill_style', ['smart', 'balanced', 'greedy'])
    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_reported_listing_preserves_rows_and_continuations(
        self, fill_style: FillStyle, reflow: ReflowMode, width: int
    ) -> None:
        source = (Path(__file__).parents[1] / 'data' / 'aligned_rows.txt').read_text()
        config = DocwrapConfig(line_length=width, fill_style=fill_style, reflow=reflow)
        once = wrap_source(source, config=config)
        listing = source[source.index('`records`') : source.rindex('"""')]
        assert listing in once
        assert wrap_source(once, config=config) == once


class TestSimpleTableWrapping:
    _BLANK_ROW_GROUP_SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    ==========  ================================================================\n'
        '    Character   Meaning\n'
        "    'a'         first row\n"
        '    ----------  ----------------------------------------------------------------\n'
        '\n'
        "    'r'         open for reading (default); this is quite a long descriptive cell\n"
        '    ==========  ================================================================\n'
        '    """\n'
    )
    _SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    ==========  ================================================================\n'
        '    Option      Meaning\n'
        '    ==========  ================================================================\n'
        "    'r'         open for reading (default); this is quite a long descriptive cell\n"
        "    'w'         open for writing, truncating the file first before any writes\n"
        '    ==========  ================================================================\n'
        '    """\n'
    )
    _HYPHEN_ONLY_SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    ----------  ----------------------------------------------------------------\n'
        '    Character   Meaning\n'
        '    ----------  ----------------------------------------------------------------\n'
        '\n'
        "    'r'         open for reading (default); this is quite a long descriptive cell\n"
        '    ----------  ----------------------------------------------------------------\n'
        '    """\n'
    )
    _TWO_TABLE_SOURCE = (
        'def f():\n'
        '    """Summary line.\n'
        '\n'
        '    =====  ==========================\n'
        '    Key    Value\n'
        '    =====  ==========================\n'
        '\n'
        '    This paragraph between two tables is deliberately long enough that it must '
        'reflow at the configured line length.\n'
        '\n'
        '    =====  ==========================\n'
        '    Key    Value\n'
        '    =====  ==========================\n'
        '    """\n'
    )

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_blank_after_hyphen_separator_survives_wrapping(self, reflow: ReflowMode) -> None:
        cfg = DocwrapConfig(reflow=reflow)
        once = wrap_source(self._BLANK_ROW_GROUP_SOURCE, config=cfg)
        assert once == self._BLANK_ROW_GROUP_SOURCE
        assert wrap_source(once, config=cfg) == once

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_hyphen_only_table_survives_wrapping(self, reflow: ReflowMode) -> None:
        cfg = DocwrapConfig(reflow=reflow)
        once = wrap_source(self._HYPHEN_ONLY_SOURCE, config=cfg)
        assert once == self._HYPHEN_ONLY_SOURCE
        assert wrap_source(once, config=cfg) == once

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_table_survives_wrapping(self, reflow: ReflowMode) -> None:
        cfg = DocwrapConfig(reflow=reflow)
        once = wrap_source(self._SOURCE, config=cfg)
        assert once == self._SOURCE
        assert wrap_source(once, config=cfg) == once

    @pytest.mark.parametrize('reflow', ['none', 'cautious', 'full'])
    def test_two_tables_preserve_tables_and_reflow_intervening_prose(
        self,
        reflow: ReflowMode,
    ) -> None:
        config = DocwrapConfig(line_length=50, reflow=reflow)
        table = (
            '    =====  ==========================\n'
            '    Key    Value\n'
            '    =====  ==========================\n'
        )

        once = wrap_source(self._TWO_TABLE_SOURCE, config=config)

        assert once.count(table) == 2
        assert 'This paragraph between two tables is deliberately long enough' not in once
        assert '    This paragraph between two tables is\n' in once
        assert wrap_source(once, config=config) == once


class TestSourceNewlines:
    def test_inline_tab_margin_projection_uses_emitted_literal(self) -> None:
        source = (
            'def f():\n'
            '\t("""Summary.\n'
            '\n'
            '\tA long body paragraph with enough words to wrap at width forty five '
            'and test margin behavior.\n'
            '\t""")\n'
        )
        config = DocwrapConfig(line_length=45)
        report = wrap_source_report(source, config=config)

        assert report.wrapped == 1
        assert report.skips == ()
        assert '\n A long body paragraph' in report.source
        assert_region_projection(source, config, emitted_source=report.source)

    def test_mixed_terminators_and_two_literals(self) -> None:
        body = (
            '"""Summary.\r\n\r'
            '    This is a long body paragraph that is long enough to require '
            'actual wrapping at width forty.\n    """'
        )
        prefix = '# before\rdef f():\n    '
        between = '; sentinel = 1\r\n# between\ndef g():\r    '
        suffix = '\r# after\r\n'
        source = prefix + body + between + body + suffix
        config = DocwrapConfig(line_length=40)
        report = docwrap.wrap_source_report(source, config=config)
        expected_literal = (
            '"""Summary.\n\n'
            '    This is a long body paragraph that\n'
            '    is long enough to require actual\n'
            '    wrapping at width forty.\n    """'
        )
        assert report.wrapped == 2
        assert report.skips == ()
        assert report.source == prefix + expected_literal + between + expected_literal + suffix
        assert docwrap.wrap_source(report.source, config=config) == report.source
        assert_region_projection(source, config, emitted_source=report.source)

    @pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'], ids=['lf', 'crlf', 'cr'])
    def test_parenthesized_non_ascii_column(self, newline: str) -> None:
        prefix = '# before' + newline + 'def café(): ('
        body = (
            '"""Summary.'
            + newline
            + newline
            + 'This is a long body paragraph that is long enough to require '
            'actual wrapping at width forty.' + newline + '"""'
        )
        suffix = '); marker = "é"' + newline + '# after'
        source = prefix + body + suffix
        config = DocwrapConfig(line_length=40)
        report = docwrap.wrap_source_report(source, config=config)
        assert report.wrapped == 1
        assert report.skips == ()
        assert report.source.startswith(prefix)
        assert report.source.endswith(suffix)
        assert 'This is a long body\nparagraph that is long\n' in report.source
        assert docwrap.wrap_source(report.source, config=config) == report.source

    @pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'], ids=['lf', 'crlf', 'cr'])
    @pytest.mark.parametrize('trailing', [False, True])
    def test_short_summary_is_unchanged(self, newline: str, trailing: bool) -> None:
        source = newline.join(['# before', 'def f():', '    """Summary."""', '# after'])
        if trailing:
            source += newline
        assert docwrap.wrap_source(source) == source
        report = docwrap.wrap_source_report(source)
        assert report.source == source
        assert report.wrapped == 0
        assert report.skips == ()
        result = docwrap.check(source)
        assert result.status == 'unchanged'
        assert result.skips == ()

    @pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'], ids=['lf', 'crlf', 'cr'])
    @pytest.mark.parametrize('trailing', [False, True])
    def test_conforming_multiline_is_unchanged(self, newline: str, trailing: bool) -> None:
        source = newline.join(
            ['def f():', '    """Summary.', '', '    Description.', '    """']
        ) + (newline if trailing else '')

        report = docwrap.wrap_source_report(source)

        assert report.source == source
        assert report.wrapped == 0
        assert report.skips == ()
        assert docwrap.check(source).status == 'unchanged'

    @pytest.mark.parametrize('newline', ['\r\n', '\r'], ids=['crlf', 'cr'])
    def test_unchanged_literal_beside_reflowed_literal(self, newline: str) -> None:
        unchanged = newline.join(
            ['def f():', '    """Summary.', '', '    Description.', '    """', '']
        )
        changed = newline.join(
            [
                'def g():',
                '    """Summary.',
                '',
                (
                    '    This is a long body paragraph that is long enough to '
                    'require actual wrapping at width forty.'
                ),
                '    """',
                '',
            ]
        )
        source = unchanged + changed
        config = DocwrapConfig(line_length=40)

        report = docwrap.wrap_source_report(source, config=config, detail='list')

        assert report.source.startswith(unchanged)
        assert report.wrapped == 1
        assert [change.changed for change in report.docstrings] == [False, True]
        assert docwrap.check(source, config=config).status == 'changed'

    @pytest.mark.parametrize('separator', ['\u0085', '\u2028'])
    def test_unicode_separators_are_not_physical_lines(self, separator: str) -> None:
        prefix = '# before\rdef f():\r    '
        body = (
            '"""Summary.' + separator + ' More.\r\r'
            '    This is a long body paragraph that is long enough to require '
            'actual wrapping at width forty.\r    """'
        )
        suffix = '\rdef g():\r    """Short."""\r'
        source = prefix + body + suffix
        config = DocwrapConfig(line_length=40)
        report = docwrap.wrap_source_report(source, config=config)
        assert report.total == 2
        assert report.wrapped == 1
        assert report.skips == ()
        assert report.source.startswith(prefix + '"""Summary.' + separator + ' More.')
        assert 'paragraph that\n' in report.source
        assert report.source.endswith(suffix)
        assert docwrap.wrap_source(report.source, config=config) == report.source

    @pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'], ids=['lf', 'crlf', 'cr'])
    @pytest.mark.parametrize('trailing', [False, True])
    def test_wrap_preserves_surrounding_source(self, newline: str, trailing: bool) -> None:
        body = (
            '    """Summary.\n\n'
            '    This is a long body paragraph that is long enough to require '
            'actual wrapping at width forty.\n    """'
        )
        prefix = '# before' + newline + 'def f():' + newline
        suffix = newline + '# after' + (newline if trailing else '')
        source = prefix + body.replace('\n', newline) + suffix
        config = DocwrapConfig(line_length=40)
        report = docwrap.wrap_source_report(source, config=config)
        assert report.wrapped == 1
        assert report.skips == ()
        assert report.source.startswith(prefix)
        assert report.source.endswith(suffix)
        expected_body = (
            '    """Summary.\n\n'
            '    This is a long body paragraph that\n'
            '    is long enough to require actual\n'
            '    wrapping at width forty.\n    """'
        )
        assert report.source == prefix + expected_body + suffix
        assert docwrap.wrap_source(source, config=config) == report.source
        assert docwrap.check(source, config=config).status == 'changed'
        assert docwrap.wrap_source(report.source, config=config) == report.source


class TestUnicodeSeparatorPreservation:
    @pytest.mark.parametrize('separator', ['\u2028', '\u2029'])
    def test_separator_survives_numpy_full_reflow(self, separator: str) -> None:
        source = (
            'def f():\n'
            '    """Summary.\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    alpha : int\n'
            f'        First part of the description{separator}second part after the'
            ' separator that should be preserved somehow here.\n'
            '    """\n'
        )
        cfg = DocwrapConfig(reflow='full')
        out = wrap_source(source, config=cfg)
        assert out.count(separator) == 1
        lines = out.split('\n')
        closing = next(i for i, ln in enumerate(lines) if ln == '    """')
        description_lines = lines[6:closing]
        assert all(line.startswith('        ') for line in description_lines if line.strip())
        assert wrap_source(out, config=cfg) == out


class TestWrapSource:
    def test_compact_summary_is_idempotent_with_default_config(self) -> None:
        source = 'def f():\n    """\n      Compact summary.\n    """\n'

        first = wrap_source(source)
        second = wrap_source(first)
        first_function = ast.parse(first).body[0]
        second_function = ast.parse(second).body[0]
        assert isinstance(first_function, ast.FunctionDef)
        assert isinstance(second_function, ast.FunctionDef)

        assert first == second
        assert '"""Compact summary."""' in first
        assert ast.get_docstring(first_function, clean=False) == ast.get_docstring(
            second_function,
            clean=False,
        )

    def test_idempotent(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        once = wrap_source(source)
        twice = wrap_source(once)
        assert once == twice

    def test_invalid_line_length_raises(self) -> None:
        with pytest.raises(InvalidLineLengthError):
            wrap_source('x = 1\n', config=UnvalidatedConfig(line_length=0))

    def test_returns_unchanged_on_parse_error(self) -> None:
        source = 'def foo(\n'
        result = wrap_source(source)
        assert result == source

    @pytest.mark.parametrize('indent', [1, 2, 3])
    @pytest.mark.parametrize('summary_start', SUMMARY_STARTS)
    def test_shallow_one_line_indentation_normalizes_for_summary_start(
        self,
        indent: int,
        summary_start: SummaryStart,
    ) -> None:
        source = 'def f():\n    """\n' + ' ' * (4 + indent) + 'Compact summary.\n    """\n'
        config = DocwrapConfig(summary_start=summary_start)

        first = wrap_source(source, config=config)
        second = wrap_source(first, config=config)
        first_function = ast.parse(first).body[0]
        second_function = ast.parse(second).body[0]
        assert isinstance(first_function, ast.FunctionDef)
        assert isinstance(second_function, ast.FunctionDef)

        assert first == second
        assert '"""Compact summary."""' in first
        assert ast.get_docstring(first_function, clean=False) == ast.get_docstring(
            second_function,
            clean=False,
        )

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(UnknownStyleError):
            wrap_source('x = 1\n', config=UnvalidatedConfig(style='bogus'))  # pyright: ignore[reportArgumentType]

    def test_wraps_long_docstring(self) -> None:
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        result = wrap_source(source, config=DocwrapConfig(wrap_summary=True))
        assert result != source


class TestWrapSourceReportValidation:
    def test_bool_line_length_raises(self) -> None:
        with pytest.raises(InvalidLineLengthError):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(line_length=True))

    def test_invalid_convert_revalidated_at_boundary(self) -> None:
        with pytest.raises(UnknownStyleError):
            wrap_source_report(
                'x = 1\n',
                config=UnvalidatedConfig(convert='bogus'),  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_detail_raises(self) -> None:
        with pytest.raises(ValueError, match='detail'):
            wrap_source_report('x = 1\n', config=DocwrapConfig(), detail='invalid')  # pyright: ignore[reportArgumentType]

    def test_invalid_fill_style_lists_valid_values(self) -> None:
        with pytest.raises(ValueError, match='/'.join(sorted(FILL_STYLES))):
            wrap_source_report(
                'x = 1\n',
                config=UnvalidatedConfig(fill_style='bogus'),  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_fill_style_raises(self) -> None:
        with pytest.raises(ValueError, match='fill_style must be one of'):
            wrap_source_report(
                'x = 1\n',
                config=UnvalidatedConfig(fill_style='bogus'),  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_line_length_raises(self) -> None:
        with pytest.raises(InvalidLineLengthError):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(line_length=0))

    def test_invalid_reflow_lists_valid_values(self) -> None:
        with pytest.raises(ValueError, match='/'.join(sorted(REFLOW_MODES))):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(reflow='bogus'))  # pyright: ignore[reportArgumentType]

    def test_invalid_reflow_raises(self) -> None:
        with pytest.raises(ValueError, match='reflow must be one of'):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(reflow='bogus'))  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('value', [-1, True, '4'])
    def test_invalid_smart_slack_raises(self, value: object) -> None:
        with pytest.raises(ValueError, match='smart_slack must be a non-negative integer'):
            wrap_source_report(
                'x = 1\n',
                config=UnvalidatedConfig(smart_slack=value),  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_strip_types_revalidated_at_boundary(self) -> None:
        with pytest.raises(TypeError, match='strip_types must be a bool'):
            wrap_source_report(
                'x = 1\n',
                config=UnvalidatedConfig(strip_types='yes'),  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_summary_start_lists_valid_values(self) -> None:
        with pytest.raises(ValueError, match='/'.join(SUMMARY_STARTS)):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(summary_start='bogus'))  # pyright: ignore[reportArgumentType]

    def test_invalid_summary_start_raises(self) -> None:
        with pytest.raises(ValueError, match='summary_start must be one of'):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(summary_start='bogus'))  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('value', ['79', None, 3.5, [1]])
    def test_non_int_line_length_raises(self, value: object) -> None:
        with pytest.raises(InvalidLineLengthError):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(line_length=value))  # pyright: ignore[reportArgumentType]

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(UnknownStyleError):
            wrap_source_report('x = 1\n', config=UnvalidatedConfig(style='bogus'))  # pyright: ignore[reportArgumentType]


class _CountedDeletedText(list[str]):
    def __init__(self, values: list[str]) -> None:
        super().__init__(values)
        self.iterations = 0

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        return super().__iter__()


@pytest.mark.parametrize('reflow', REFLOW_MODES)
@pytest.mark.parametrize('fill_style', FILL_STYLES)
def test_generated_idempotence(
    fill_style: FillStyle,
    reflow: ReflowMode,
) -> None:
    _assert_generated_idempotence(
        0,
        _IDEMPOTENCE_CASE_COUNT,
        fill_style,
        reflow,
    )


@pytest.mark.slow
@pytest.mark.parametrize('reflow', REFLOW_MODES)
@pytest.mark.parametrize('fill_style', FILL_STYLES)
def test_generated_idempotence_slow(
    fill_style: FillStyle,
    reflow: ReflowMode,
) -> None:
    _assert_generated_idempotence(
        _IDEMPOTENCE_CASE_COUNT,
        _SLOW_IDEMPOTENCE_CASE_COUNT,
        fill_style,
        reflow,
    )


@pytest.mark.parametrize(('source', 'line_length'), CASES)
def test_idempotent(source: str, line_length: int) -> None:
    first = wrap_source(source, config=DocwrapConfig(line_length=line_length))
    second = wrap_source(first, config=DocwrapConfig(line_length=line_length))
    assert first == second


@pytest.mark.parametrize('reflow', REFLOW_MODES)
@pytest.mark.parametrize('fill_style', FILL_STYLES)
def test_multispace_break_is_idempotent(
    fill_style: FillStyle,
    reflow: ReflowMode,
) -> None:
    body = 'x' * 65 + ' end.  Next words follow here to fill the line out nicely.'
    source = f'def f():\n    """Summary line.\n\n    {body}\n    """\n'
    config = DocwrapConfig(fill_style=fill_style, reflow=reflow)

    first = wrap_source(source, config=config)
    second = wrap_source(first, config=config)

    assert second == first


@pytest.mark.parametrize(('summary_before', 'style'), SUMMARY_BOUNDARY_CASES)
@pytest.mark.parametrize('reflow', REFLOW_MODES)
@pytest.mark.parametrize('fill_style', FILL_STYLES)
def test_summary_boundary_preserves_parser_projection(
    summary_before: str,
    style: StyleName | None,
    fill_style: FillStyle,
    reflow: ReflowMode,
) -> None:
    source = f'def f():\n    """{summary_before}"""\n'
    config = DocwrapConfig(
        fill_style=fill_style,
        reflow=reflow,
        style=style,
        wrap_summary=True,
    )

    first = wrap_source(source, config=config)
    ast.parse(first)
    assert_region_projection(source, config, emitted_source=first)
    second = wrap_source(first, config=config)
    function = ast.parse(first).body[0]
    assert isinstance(function, ast.FunctionDef)
    summary_after = ast.get_docstring(function, clean=False)
    assert summary_after is not None

    assert second == first
    assert re.findall(r'\S+', summary_after) == re.findall(r'\S+', summary_before)


@pytest.mark.parametrize(('source', 'line_length'), CASES)
def test_valid_python(source: str, line_length: int) -> None:
    result = wrap_source(source, config=DocwrapConfig(line_length=line_length))
    compile(result, '<test>', 'exec')
