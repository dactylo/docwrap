import importlib
import inspect
import re
import sys
import typing
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

import docwrap
from tests.helpers import process_transcript, run_setup

EXPECTED_EXPORTS = frozenset(
    {
        'DEFAULT_LINE_LENGTH',
        'DetailLevel',
        'DocstringChange',
        'DocstringSkip',
        'DocwrapConfig',
        'DocwrapConfigWarning',
        'DocwrapSkipWarning',
        'FillStyle',
        'InvalidLineLengthError',
        'ReflowMode',
        'SkipCode',
        'StyleName',
        'SummaryStart',
        'UnknownStyleError',
        'WrapCheckResult',
        'WrapReport',
        'WrapStatus',
        'check',
        'clear_config_cache',
        'get_styles',
        'resolve_config',
        'wrap_file_report',
        'wrap_source',
        'wrap_source_report',
    }
)


class TestDocumentationCoverage:
    def test_every_export_documented_in_library_reference(self) -> None:
        doc = (Path(__file__).parent.parent / 'docs' / 'library.md').read_text(encoding='utf-8')
        undocumented = [
            name
            for name in docwrap.__all__
            if not re.search(rf'^#{{2,4}} .*docwrap\.{re.escape(name)}\b', doc, re.MULTILINE)
        ]
        assert undocumented == []

    def test_library_reference_names_no_internal_paths(self) -> None:
        doc = (Path(__file__).parent.parent / 'docs' / 'library.md').read_text(encoding='utf-8')
        for internal in (
            'docwrap._core',
            'docwrap._display',
            'docwrap._styles',
            'docwrap._types',
            'eligible_ranges',
            'eligible_target',
        ):
            assert internal not in doc


class TestInternalImportPaths:
    @pytest.mark.parametrize('name', ['docwrap._core', 'docwrap._display', 'docwrap._styles'])
    def test_new_subpackage_paths_importable(self, name: str) -> None:
        assert importlib.import_module(name) is not None


class TestLazyFacade:
    def test_dir_lists_every_export(self) -> None:
        assert callable(docwrap.__dir__)
        assert callable(docwrap.__getattr__)
        assert set(docwrap.__all__) <= set(dir(docwrap))

    def test_exports_are_library_objects(self) -> None:
        library = importlib.import_module('docwrap._library')
        assert list(library.__all__) == list(docwrap.__all__)
        for name in docwrap.__all__:
            assert getattr(docwrap, name) is getattr(library, name)

    def test_first_access_in_fresh_interpreter_loads_library_lazily(self) -> None:
        script = (
            'import sys\n'
            'import docwrap\n'
            "assert 'docwrap._library' not in sys.modules, 'library loaded at import'\n"
            "assert 'docwrap._core' not in sys.modules, 'engine loaded at import'\n"
            "assert not hasattr(docwrap, '_cli')\n"
            'from docwrap import _color\n'
            "assert 'docwrap._library' not in sys.modules, 'library loaded by probes'\n"
            'for name in docwrap.__all__:\n'
            '    getattr(docwrap, name)\n'
            "assert 'docwrap._library' in sys.modules\n"
        )
        result = run_setup([sys.executable, '-c', script], check=False)
        assert result.returncode == 0, process_transcript(result)

    def test_library_import_failure_surfaces_on_first_access(self) -> None:
        script = (
            'import sys\n'
            "sys.modules['docwrap._library'] = None\n"
            'import docwrap\n'
            'try:\n'
            '    docwrap.wrap_source\n'
            'except ImportError:\n'
            '    sys.exit(0)\n'
            'sys.exit(1)\n'
        )
        result = run_setup([sys.executable, '-c', script], check=False)
        assert result.returncode == 0, process_transcript(result)

    def test_unexported_name_is_not_an_attribute(self) -> None:
        name = 'wrap_source_report_eligible'
        with pytest.raises(AttributeError):
            getattr(docwrap, name)


class TestPublicFacade:
    def test_all_exports_exact(self) -> None:
        assert set(docwrap.__all__) == set(EXPECTED_EXPORTS)

    def test_all_is_sorted(self) -> None:
        assert list(docwrap.__all__) == sorted(docwrap.__all__)

    def test_config_warning_category(self) -> None:
        assert issubclass(docwrap.DocwrapConfigWarning, UserWarning)

    def test_escaped_carriage_return_does_not_trigger_physical_newline_rewrite(self) -> None:
        source = (
            'def f():\r\n'
            '    """Summary.\r\n'
            '\r\n'
            '    Literal ' + chr(92) + 'r remains escaped.\r\n'
            '    """\r\n'
        )

        report = docwrap.wrap_source_report(source)

        assert report.source == source
        assert report.wrapped == 0
        assert docwrap.check(source).status == 'unchanged'

    def test_get_styles_returns_builtins(self) -> None:
        assert docwrap.get_styles() == frozenset({'google', 'numpy', 'sphinx'})

    def test_operations_callable(self) -> None:
        for name in (
            'check',
            'clear_config_cache',
            'get_styles',
            'resolve_config',
            'wrap_file_report',
            'wrap_source',
            'wrap_source_report',
        ):
            assert callable(getattr(docwrap, name))


class TestPublicRecordShapes:
    @pytest.mark.parametrize(
        'record',
        [
            docwrap.DocstringChange,
            docwrap.DocstringSkip,
            docwrap.DocwrapConfig,
            docwrap.WrapCheckResult,
            docwrap.WrapReport,
        ],
    )
    def test_frozen_slotted_keyword_only(self, record: type) -> None:
        assert record.__dataclass_params__.frozen  # pyright: ignore[reportUnknownMemberType]
        assert '__slots__' in record.__dict__
        assert all(f.kw_only for f in fields(record))


class TestPublicSignatures:
    @pytest.mark.parametrize(
        'func',
        [docwrap.check, docwrap.wrap_file_report, docwrap.wrap_source, docwrap.wrap_source_report],
    )
    def test_no_eligibility_parameters(self, func: Callable[..., object]) -> None:
        params = inspect.signature(func).parameters
        assert 'eligible_ranges' not in params
        assert 'eligible_target' not in params


class TestPublicTypeClosure:
    def test_owned_annotation_types_are_exported(self) -> None:
        found: set[type] = set()
        for annotation in _public_annotations():
            _collect_owned_types(annotation, found)
        unexported = {t.__name__ for t in found} - set(docwrap.__all__)
        assert unexported == set()


def _collect_owned_types(annotation: object, found: set[type]) -> None:
    if isinstance(annotation, type):
        if annotation.__module__.split('.')[0] == 'docwrap':
            found.add(annotation)
        return
    for arg in typing.get_args(annotation):
        _collect_owned_types(arg, found)


def _public_annotations() -> list[object]:
    annotations: list[object] = []
    for func in (
        docwrap.check,
        docwrap.clear_config_cache,
        docwrap.get_styles,
        docwrap.resolve_config,
        docwrap.wrap_file_report,
        docwrap.wrap_source,
        docwrap.wrap_source_report,
    ):
        annotations.extend(typing.get_type_hints(func).values())
    for record in (
        docwrap.DocstringChange,
        docwrap.DocstringSkip,
        docwrap.DocwrapConfig,
        docwrap.WrapCheckResult,
        docwrap.WrapReport,
    ):
        annotations.extend(typing.get_type_hints(record).values())
    percent = docwrap.WrapReport.percent_wrapped.fget
    assert percent is not None
    annotations.append(typing.get_type_hints(percent)['return'])
    return annotations
