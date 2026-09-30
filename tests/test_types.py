from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

from docwrap._exceptions import InvalidLineLengthError, UnknownStyleError
from docwrap._types import (
    DEFAULT_LINE_LENGTH,
    BaselineResult,
    BaselineStatus,
    DocstringChange,
    DocstringSkip,
    DocwrapConfig,
    FileResult,
    FillStyle,
    LineLengthSource,
    SkipCode,
    SummaryStart,
    WrapCheckResult,
    WrapOverrides,
    WrapReport,
)


class TestBaselineResult:
    def test_baseline_result_content(self) -> None:
        r = BaselineResult(status=BaselineStatus.CONTENT, content='hello')
        assert r.status is BaselineStatus.CONTENT
        assert r.content == 'hello'

    def test_baseline_result_encoding_default(self) -> None:
        r = BaselineResult(status=BaselineStatus.CONTENT, content='hello')
        assert r.encoding == 'utf-8'

    def test_baseline_result_newline_default(self) -> None:
        r = BaselineResult(status=BaselineStatus.MISSING)
        assert r.newline == '\n'

    def test_baseline_result_no_content(self) -> None:
        r = BaselineResult(status=BaselineStatus.MISSING)
        assert r.content is None


class TestDocstringChange:
    def test_before_after_none_by_default(self) -> None:
        dc = DocstringChange(qualname='foo', start_line=1, end_line=1, changed=True)
        assert dc.before is None
        assert dc.after is None

    def test_converted_default(self) -> None:
        dc = DocstringChange(qualname='foo', start_line=1, end_line=1, changed=False)
        assert dc.converted is False

    def test_converted_true(self) -> None:
        dc = DocstringChange(
            qualname='foo', start_line=1, end_line=1, changed=True, converted=True
        )
        assert dc.converted is True

    def test_frozen(self) -> None:
        dc = DocstringChange(qualname='foo', start_line=1, end_line=1, changed=False)
        with pytest.raises(AttributeError):
            dc.qualname = 'bar'  # pyright: ignore[reportAttributeAccessIssue]

    def test_post_init_invariant_after_requires_changed(self) -> None:
        with pytest.raises(ValueError, match='changed'):
            DocstringChange(qualname='foo', start_line=1, end_line=1, changed=False, after='x')

    def test_post_init_invariant_before_requires_changed(self) -> None:
        with pytest.raises(ValueError, match='changed'):
            DocstringChange(qualname='foo', start_line=1, end_line=1, changed=False, before='x')

    def test_summary_skipped_default(self) -> None:
        dc = DocstringChange(qualname='foo', start_line=1, end_line=1, changed=False)
        assert dc.summary_skipped is False
        assert dc.summary_width is None

    def test_summary_skipped_with_changed_false(self) -> None:
        dc = DocstringChange(
            qualname='foo',
            start_line=1,
            end_line=1,
            changed=False,
            summary_skipped=True,
            summary_width=93,
        )
        assert dc.summary_skipped is True
        assert dc.summary_width == 93

    def test_summary_skipped_with_changed_true(self) -> None:
        dc = DocstringChange(
            qualname='foo',
            start_line=1,
            end_line=1,
            changed=True,
            summary_skipped=True,
            summary_width=93,
        )
        assert dc.summary_skipped is True

    def test_summary_width_requires_summary_skipped(self) -> None:
        with pytest.raises(ValueError, match='summary_width'):
            DocstringChange(
                qualname='foo',
                start_line=1,
                end_line=1,
                changed=False,
                summary_width=93,
            )

    def test_valid_diff_entry(self) -> None:
        dc = DocstringChange(
            qualname='Cls.method',
            start_line=5,
            end_line=10,
            changed=True,
            before='old',
            after='new',
        )
        assert dc.before == 'old'
        assert dc.after == 'new'


class TestDocstringSkip:
    def test_fields(self) -> None:
        skip = DocstringSkip(
            path='pkg/mod.py',
            line=7,
            qualname='Cls.method',
            code='line-continuation',
            message='backslash-newline line continuation',
        )
        assert skip.path == 'pkg/mod.py'
        assert skip.line == 7
        assert skip.qualname == 'Cls.method'
        assert skip.code == 'line-continuation'
        assert skip.message == 'backslash-newline line continuation'

    def test_frozen(self) -> None:
        skip = DocstringSkip(path='', line=1, qualname='f', code='validation-failed', message='m')
        with pytest.raises(AttributeError):
            skip.line = 2  # pyright: ignore[reportAttributeAccessIssue]

    def test_wrap_check_result_skips_default(self) -> None:
        assert WrapCheckResult(status='unchanged').skips == ()

    def test_wrap_report_skips_default(self) -> None:
        assert WrapReport(source='', total=0, wrapped=0).skips == ()


class TestDocwrapConfig:
    def test_convert_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.convert is None

    def test_defaults(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.line_length == DEFAULT_LINE_LENGTH
        assert cfg.style is None

    def test_fill_style_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.fill_style == 'smart'

    def test_fill_style_values(self) -> None:
        values: tuple[FillStyle, ...] = ('balanced', 'greedy', 'smart')
        for value in values:
            assert DocwrapConfig(fill_style=value).fill_style == value

    def test_frozen(self) -> None:
        cfg = DocwrapConfig()
        with pytest.raises(AttributeError):
            cfg.line_length = 100  # pyright: ignore[reportAttributeAccessIssue]

    def test_kw_only(self) -> None:
        # Positional construction must fail
        with pytest.raises(TypeError):
            DocwrapConfig(DEFAULT_LINE_LENGTH, 'google')  # pyright: ignore[reportCallIssue]

    def test_line_length_defaults(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.line_length == DEFAULT_LINE_LENGTH

    def test_smart_slack_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.smart_slack == 8

    def test_strip_types_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.strip_types is False

    def test_summary_start_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.summary_start == 'preserve'

    def test_summary_start_values(self) -> None:
        vals: tuple[SummaryStart, ...] = ('preserve', 'first-line', 'second-line')
        for val in vals:
            cfg = DocwrapConfig(summary_start=val)
            assert cfg.summary_start == val

    def test_wrap_summary_default(self) -> None:
        cfg = DocwrapConfig()
        assert cfg.wrap_summary is False

    def test_wrap_summary_true(self) -> None:
        cfg = DocwrapConfig(wrap_summary=True)
        assert cfg.wrap_summary is True


class TestDocwrapConfigConstructionValidation:
    def test_bool_line_length_raises(self) -> None:
        with pytest.raises(InvalidLineLengthError):
            DocwrapConfig(line_length=True)

    def test_invalid_convert_raises(self) -> None:
        with pytest.raises(UnknownStyleError):
            DocwrapConfig(convert='bogus')  # pyright: ignore[reportArgumentType]

    def test_invalid_fill_style_raises(self) -> None:
        with pytest.raises(ValueError, match='fill_style must be one of'):
            DocwrapConfig(fill_style='bogus')  # pyright: ignore[reportArgumentType]

    def test_invalid_line_length_raises(self) -> None:
        with pytest.raises(InvalidLineLengthError):
            DocwrapConfig(line_length=0)

    def test_invalid_reflow_raises(self) -> None:
        with pytest.raises(ValueError, match='reflow must be one of'):
            DocwrapConfig(reflow='bogus')  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('value', [-1, True, '4'])
    def test_invalid_smart_slack_raises(self, value: object) -> None:
        with pytest.raises(ValueError, match='smart_slack must be a non-negative integer'):
            DocwrapConfig(smart_slack=value)  # pyright: ignore[reportArgumentType]

    def test_invalid_strip_types_raises(self) -> None:
        with pytest.raises(TypeError, match='strip_types must be a bool'):
            DocwrapConfig(strip_types='yes')  # pyright: ignore[reportArgumentType]

    def test_invalid_summary_start_raises(self) -> None:
        with pytest.raises(ValueError, match='summary_start must be one of'):
            DocwrapConfig(summary_start='bogus')  # pyright: ignore[reportArgumentType]

    def test_invalid_wrap_summary_raises(self) -> None:
        with pytest.raises(TypeError, match='wrap_summary must be a bool'):
            DocwrapConfig(wrap_summary='yes')  # pyright: ignore[reportArgumentType]

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(UnknownStyleError):
            DocwrapConfig(style='bogus')  # pyright: ignore[reportArgumentType]

    def test_valid_defaults_construct(self) -> None:
        assert DocwrapConfig().line_length == DEFAULT_LINE_LENGTH


class TestDocwrapConfigFieldSet:
    def test_exactly_the_approved_transformation_fields(self) -> None:
        expected = {
            'convert',
            'fill_style',
            'line_length',
            'reflow',
            'smart_slack',
            'strip_types',
            'style',
            'summary_start',
            'wrap_summary',
        }
        assert {f.name for f in fields(DocwrapConfig)} == expected


class TestFileResult:
    def test_defaults_mean_no_report_or_preview_details(self) -> None:
        result = FileResult(status=2, path=Path('broken.py'))

        assert result.total is None
        assert result.wrapped == 0
        assert result.skipped_summaries == 0
        assert result.docstrings == ()


class TestLineLengthSource:
    def test_values(self) -> None:
        assert LineLengthSource.CLI == '--length'
        assert LineLengthSource.DEFAULT == 'default'
        assert LineLengthSource.DOCWRAP == '[tool.docwrap]'


class TestResultRecordConstruction:
    def test_docstring_change_rejects_positional(self) -> None:
        changed = False
        with pytest.raises(TypeError):
            DocstringChange('foo', 1, 2, changed)  # pyright: ignore[reportCallIssue]

    def test_docstring_skip_rejects_positional(self) -> None:
        with pytest.raises(TypeError):
            DocstringSkip('p', 1, 'q', 'validation-failed', 'm')  # pyright: ignore[reportCallIssue]

    def test_wrap_report_rejects_positional(self) -> None:
        with pytest.raises(TypeError):
            WrapReport('', 0, 0)  # pyright: ignore[reportCallIssue]


class TestSkipCode:
    def test_unrepresentable_return_is_available(self) -> None:
        assert 'unrepresentable-return' in get_args(SkipCode)


class TestWrapReport:
    def test_converted_default(self) -> None:
        rpt = WrapReport(source='', total=0, wrapped=0)
        assert rpt.converted == 0
        assert rpt.convert_target is None

    def test_converted_set(self) -> None:
        rpt = WrapReport(source='', total=5, wrapped=2, converted=3, convert_target='numpy')
        assert rpt.converted == 3
        assert rpt.convert_target == 'numpy'

    def test_defaults(self) -> None:
        r = WrapReport(source='x', total=1, wrapped=0)
        assert r.docstrings == ()
        assert r.parse_error is False

    def test_frozen(self) -> None:
        r = WrapReport(source='', total=0, wrapped=0)
        with pytest.raises(AttributeError):
            r.source = 'x'  # pyright: ignore[reportAttributeAccessIssue]

    def test_percent_wrapped(self) -> None:
        r = WrapReport(source='', total=10, wrapped=3)
        assert r.percent_wrapped == 30.0

    def test_percent_wrapped_zero_total(self) -> None:
        r = WrapReport(source='', total=0, wrapped=0)
        assert r.percent_wrapped == 0.0

    def test_skipped_summaries_default(self) -> None:
        rpt = WrapReport(source='', total=0, wrapped=0)
        assert rpt.skipped_summaries == 0

    def test_skipped_summaries_set(self) -> None:
        rpt = WrapReport(source='', total=5, wrapped=2, skipped_summaries=3)
        assert rpt.skipped_summaries == 3

    def test_style_default_none(self) -> None:
        rpt = WrapReport(source='', total=0, wrapped=0)
        assert rpt.style is None

    def test_style_field(self) -> None:
        rpt = WrapReport(source='', total=0, wrapped=0, style='google')
        assert rpt.style == 'google'


def test_wrap_check_result_fields() -> None:
    r = WrapCheckResult(status='changed')
    assert r.status == 'changed'


def test_wrap_check_result_frozen() -> None:
    r = WrapCheckResult(status='unchanged')
    with pytest.raises(AttributeError):
        r.status = 'changed'  # pyright: ignore[reportAttributeAccessIssue]


def test_wrap_overrides_defaults_are_none() -> None:
    o = WrapOverrides()
    assert o.line_length is None
    assert o.style is None
    assert o.convert is None
    assert o.strip_types is None
    assert o.summary_start is None
    assert o.wrap_summary is None
    assert o.fill_style is None
    assert o.smart_slack is None
    assert o.show_skips is None
