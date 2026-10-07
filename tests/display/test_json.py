import json
from typing import assert_type

import pytest

from docwrap._display._json import (
    JsonFileRecord,
    format_json,
    json_error_record,
    json_success_record,
)
from docwrap._types import WrapReport


class TestFormatJson:
    def test_adds_version_envelope(self) -> None:
        record = assert_type(
            json_success_record(
                'a.py',
                WrapReport(source='', total=1, wrapped=0, style='google'),
                79,
                changed=False,
            ),
            JsonFileRecord,
        )

        output = format_json([record])

        assert output == (
            '{\n'
            '  "version": 1,\n'
            '  "files": [\n'
            '    {\n'
            '      "path": "a.py",\n'
            '      "status": "unchanged",\n'
            '      "warning": null,\n'
            '      "style": "google",\n'
            '      "line_length": 79,\n'
            '      "docstrings": {\n'
            '        "wrapped": 0,\n'
            '        "total": 1,\n'
            '        "skipped_summaries": 0\n'
            '      },\n'
            '      "skips": [],\n'
            '      "changes": []\n'
            '    }\n'
            '  ],\n'
            '  "total_skips": 0\n'
            '}'
        )
        assert json.loads(output) == {
            'version': 1,
            'files': [
                {
                    'path': 'a.py',
                    'status': 'unchanged',
                    'warning': None,
                    'style': 'google',
                    'line_length': 79,
                    'docstrings': {'wrapped': 0, 'total': 1, 'skipped_summaries': 0},
                    'skips': [],
                    'changes': [],
                }
            ],
            'total_skips': 0,
        }

    def test_pretty_prints_with_indent_two(self) -> None:
        output = format_json([])
        assert '\n  ' in output


class TestJsonErrorRecord:
    def test_builds_default_error_record(self) -> None:
        record = assert_type(json_error_record('a.py', 'invalid', 'bad syntax'), JsonFileRecord)
        assert record == {
            'path': 'a.py',
            'status': 'invalid',
            'warning': 'bad syntax',
            'style': None,
            'line_length': None,
            'docstrings': {'wrapped': 0, 'total': 0, 'skipped_summaries': 0},
            'skips': [],
            'changes': [],
        }


class TestJsonSuccessRecord:
    @pytest.mark.parametrize('changed', [False, True])
    def test_status_uses_original_source_comparison(self, *, changed: bool) -> None:
        report = WrapReport(source='same result', total=0, wrapped=0)
        record = json_success_record('a.py', report, 79, changed=changed)
        assert record['status'] == ('changed' if changed else 'unchanged')


class TestSkipFields:
    def test_empty_records_zero_total_skips(self) -> None:
        data = json.loads(format_json([]))
        assert data['version'] == 1
        assert data['total_skips'] == 0

    def test_total_skips_sums_records(self) -> None:
        rec = json_error_record('a.py', 'error', 'boom')
        rec['skips'] = [{'line': 1, 'qualname': 'f', 'code': 'line-continuation', 'message': 'm'}]
        data = json.loads(format_json([rec, json_error_record('b.py', 'error', 'x')]))
        assert data['total_skips'] == 1


def test_windows_json_path_keeps_native_spelling() -> None:
    path = r'C:\repo\file.py'
    record = json_error_record(path, 'error', 'unreadable')
    assert json.loads(format_json([record]))['files'][0]['path'] == path
