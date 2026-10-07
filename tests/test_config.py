import sys
import tomllib
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO, get_type_hints
from unittest.mock import patch

import pytest

import docwrap
import docwrap._config as config_mod
from docwrap._config import (
    ResolvedConfig,
    StatsConfig,
    _warn,
    clear_config_cache,
    effective_config,
    find_color_config,
    find_config,
    find_incremental_config,
    find_stats_config,
    render_config_warnings,
    resolve_config,
)
from docwrap._exceptions import DocwrapConfigWarning
from docwrap._types import DEFAULT_LINE_LENGTH, DocwrapConfig, LineLengthSource, WrapOverrides

_CONTROL_TEXT = 'esc\x1bcr\rtab\tdel\x7fc1\x85'
_VISIBLE_CONTROL_TEXT = r'esc\x1bcr\x0dtab\x09del\x7fc1\x85'


class TestAllowOnFailureConfig:
    def test_allow_on_failure_default_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        hook_cfg = find_config(tmp_path / 'test.py')
        assert hook_cfg.allow_on_failure is False

    def test_allow_on_failure_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nallow-on-failure = false\n')
        hook_cfg = find_config(tmp_path / 'test.py')
        assert hook_cfg.allow_on_failure is False

    def test_allow_on_failure_invalid_warns(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nallow-on-failure = "yes"\n')
        with pytest.warns(DocwrapConfigWarning, match='yes'):
            hook_cfg = find_config(tmp_path / 'test.py')
        assert hook_cfg.allow_on_failure is False

    def test_allow_on_failure_true(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nallow-on-failure = true\n')
        hook_cfg = find_config(tmp_path / 'test.py')
        assert hook_cfg.allow_on_failure is True


class TestColorConfig:
    def test_color_always(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "always"\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.color == 'always'

    def test_color_auto_default(self, tmp_path: Path) -> None:
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.color == 'auto'

    def test_color_never(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "never"\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.color == 'never'

    def test_find_config_includes_color(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "never"\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        resolved = find_config(f)
        assert resolved.color == 'never'

    def test_invalid_color_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "rainbow"\n')
        with pytest.warns(DocwrapConfigWarning, match='rainbow'):
            resolved = find_config(tmp_path / 'test.py')
        assert resolved.color == 'auto'


class TestConfigCache:
    def test_clear_also_drops_the_parsed_tool_table(self, tmp_path: Path) -> None:
        clear_config_cache()
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nline-length = 100\n', encoding='utf-8')
        resolve_config(tmp_path / 'mod.py')

        with patch.object(tomllib, 'load', wraps=tomllib.load) as load:
            pyproject.write_text('[tool.docwrap]\nline-length = 120\n', encoding='utf-8')
            clear_config_cache()
            assert resolve_config(tmp_path / 'mod.py').line_length == 120

        assert load.call_count == 1

    def test_clear_config_cache_picks_up_edits(self, tmp_path: Path) -> None:
        clear_config_cache()
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nline-length = 100\n', encoding='utf-8')
        assert resolve_config(tmp_path / 'mod.py').line_length == 100

        pyproject.write_text('[tool.docwrap]\nline-length = 120\n', encoding='utf-8')
        clear_config_cache()
        assert resolve_config(tmp_path / 'mod.py').line_length == 120

    def test_stats_config_reflects_edits_after_clear(self, tmp_path: Path) -> None:
        clear_config_cache()
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap.stats]\nlongest-per-category = 3\n', encoding='utf-8')
        find_stats_config(tmp_path / 'mod.py')

        pyproject.write_text('[tool.docwrap.stats]\nlongest-per-category = 7\n', encoding='utf-8')
        clear_config_cache()
        assert find_stats_config(tmp_path / 'mod.py').longest_per_category == 7


class TestConfigCaching:
    def test_different_directories_not_cached(self, tmp_path: Path) -> None:
        d1 = tmp_path / 'sub1'
        d1.mkdir()
        (d1 / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 88\n')
        d2 = tmp_path / 'sub2'
        d2.mkdir()
        (d2 / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 100\n')
        cfg1 = find_config(d1 / 'a.py').config
        cfg2 = find_config(d2 / 'b.py').config
        assert cfg1.line_length == 88
        assert cfg2.line_length == 100

    def test_same_directory_returns_cached_result(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('[tool.docwrap]\nline-length = 88\n')
        cfg1 = find_config(tmp_path / 'a.py').config
        cfg2 = find_config(tmp_path / 'b.py').config
        assert cfg1 is cfg2  # Same object, not just equal


class TestConvention:
    def test_docwrap_convention(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('[tool.docwrap]\nconvention = "numpy"\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.style == 'numpy'

    def test_invalid_convention_warns_and_defaults(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('[tool.docwrap]\nconvention = "bogus"\n')
        with pytest.warns(DocwrapConfigWarning, match='bogus'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.style is None

    def test_no_convention(self, tmp_path: Path) -> None:
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.style is None


class TestConvertConfig:
    def test_convert_default_none(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.convert is None

    def test_convert_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nconvert = "numpy"\n')
        cfg = resolve_config(tmp_path / 'example.py')
        assert cfg.convert == 'numpy'

    def test_invalid_convert_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nconvert = "bogus"\n')
        with pytest.warns(DocwrapConfigWarning, match='bogus'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.convert is None

    def test_invalid_strip_types_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstrip-types = "yes"\n')
        with pytest.warns(DocwrapConfigWarning, match='strip-types'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.strip_types is False

    def test_strip_types_default_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.strip_types is False

    def test_strip_types_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstrip-types = true\n')
        cfg = resolve_config(tmp_path / 'example.py')
        assert cfg.strip_types is True


class TestDirectAndResolvedParity:
    def test_same_transformation_fields_same_output(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 60\nwrap-summary = true\n'
        )
        long = 'A ' + 'very ' * 15 + 'long summary.'
        source = f'def foo():\n    """{long}"""\n'
        resolved = resolve_config(tmp_path / 'mod.py')
        direct = DocwrapConfig(line_length=60, wrap_summary=True)
        assert docwrap.wrap_source(source, config=resolved) == docwrap.wrap_source(
            source, config=direct
        )


class TestEffectiveConfig:
    def test_fill_style_override(self) -> None:
        cfg = DocwrapConfig(fill_style='balanced')
        overrides = WrapOverrides(fill_style='greedy')
        result = effective_config(cfg, overrides)
        assert result.fill_style == 'greedy'

    def test_non_config_override_is_ignored(self) -> None:
        cfg = DocwrapConfig()
        overrides = WrapOverrides(show_skips=True)
        result = effective_config(cfg, overrides)
        assert result == cfg

    def test_none_override_falls_through(self) -> None:
        cfg = DocwrapConfig(line_length=88, style='google')
        overrides = WrapOverrides()
        result = effective_config(cfg, overrides)
        assert result.line_length == 88
        assert result.style == 'google'

    def test_override_line_length(self) -> None:
        cfg = DocwrapConfig(line_length=79)
        overrides = WrapOverrides(line_length=100)
        result = effective_config(cfg, overrides)
        assert result.line_length == 100

    def test_reflow_override(self) -> None:
        cfg = DocwrapConfig(reflow='none')
        overrides = WrapOverrides(reflow='cautious')
        result = effective_config(cfg, overrides)
        assert result.reflow == 'cautious'

    def test_smart_slack_override(self) -> None:
        cfg = DocwrapConfig(smart_slack=4)
        overrides = WrapOverrides(smart_slack=6)
        result = effective_config(cfg, overrides)
        assert result.smart_slack == 6

    def test_strip_types_or_semantics(self) -> None:
        cfg = DocwrapConfig(strip_types=True)
        overrides = WrapOverrides(strip_types=False)
        result = effective_config(cfg, overrides)
        assert result.strip_types is True


class TestFillStyleConfig:
    def test_default_fill_style_is_smart(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.fill_style == 'smart'

    def test_fill_style_greedy(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "greedy"\n')
        cfg = resolve_config(tmp_path / 'test.py')
        assert cfg.fill_style == 'greedy'

    def test_fill_style_smart(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "smart"\n')
        cfg = resolve_config(tmp_path / 'test.py')
        assert cfg.fill_style == 'smart'

    def test_invalid_fill_style_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "ragged"\n')
        with pytest.warns(DocwrapConfigWarning, match='fill-style'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.fill_style == 'smart'


class TestFindColorConfig:
    def test_default_derives_from_resolved_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert find_color_config() == ResolvedConfig(config=DocwrapConfig()).color

    def test_reads_color(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "always"\n')
        assert find_color_config(tmp_path) == 'always'

    def test_resolves_color_with_invalid_style(self, tmp_path: Path) -> None:
        toml = '[tool.docwrap]\nstyle = "bogus"\ncolor = "never"\n'
        (tmp_path / 'pyproject.toml').write_text(toml)
        with pytest.warns(DocwrapConfigWarning, match='style'):
            assert find_color_config(tmp_path) == 'never'


class TestLineLength:
    def test_default_fallback(self, tmp_path: Path) -> None:
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.line_length == DEFAULT_LINE_LENGTH

    def test_docwrap_config(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('[tool.docwrap]\nline-length = 88\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.line_length == 88


class TestLineLengthSource:
    def test_default_source(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('')
        resolved = find_config(f)
        assert resolved.line_length_source is LineLengthSource.DEFAULT

    def test_docwrap_source(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 88\n')
        f = tmp_path / 'test.py'
        f.write_text('')
        resolved = find_config(f)
        assert resolved.line_length_source is LineLengthSource.DOCWRAP


class TestNonUtf8Pyproject:
    def test_resolve_config_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_bytes(
            b'[tool.docwrap]\n# caf\xe9 comment\nline-length = 100\n'
        )
        target = tmp_path / 'mod.py'
        target.write_text('def f():\n    """Ok."""\n')
        with pytest.warns(DocwrapConfigWarning):
            cfg = resolve_config(target)
        assert cfg == DocwrapConfig()


class TestReflowConfig:
    def test_default_reflow_is_cautious(self, tmp_path: Path) -> None:
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\n')
        py_file = tmp_path / 'x.py'
        py_file.write_text('')
        cfg = resolve_config(py_file)
        assert cfg.reflow == 'cautious'

    def test_reflow_cautious(self, tmp_path: Path) -> None:
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nreflow = "cautious"\n')
        py_file = tmp_path / 'x.py'
        py_file.write_text('')
        cfg = resolve_config(py_file)
        assert cfg.reflow == 'cautious'

    def test_reflow_full(self, tmp_path: Path) -> None:
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nreflow = "full"\n')
        py_file = tmp_path / 'x.py'
        py_file.write_text('')
        cfg = resolve_config(py_file)
        assert cfg.reflow == 'full'

    def test_reflow_invalid_warns_and_defaults(self, tmp_path: Path) -> None:
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nreflow = "aggressive"\n')
        py_file = tmp_path / 'x.py'
        py_file.write_text('')
        with pytest.warns(DocwrapConfigWarning, match='reflow'):
            cfg = resolve_config(py_file)
        assert cfg.reflow == 'cautious'


class TestRenderConfigWarnings:
    def test_control_text_is_visible_in_immediate_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with render_config_warnings():
            _warn(f'invalid value {_CONTROL_TEXT}')

        assert capsys.readouterr().err == f'warning: invalid value {_VISIBLE_CONTROL_TEXT}\n'

    def test_control_text_stays_raw_in_warning_object(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DocwrapConfigWarning)
            _warn(f'invalid value {_CONTROL_TEXT}')

        assert len(caught) == 1
        assert str(caught[0].message) == f'invalid value {_CONTROL_TEXT}'

    def test_nested_context_restores_outer_presentation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with render_config_warnings():
            with render_config_warnings():
                pass
            _warn('invalid [tool.docwrap].demo value 1, ignoring')
            captured = capsys.readouterr()
        assert 'warning: invalid [tool.docwrap].demo' in captured.err

    def test_preserves_unrelated_warning_timing(self) -> None:
        events: list[str] = []

        def record_warning(
            message: Warning | str,
            _category: type[Warning],
            _filename: str,
            _lineno: int,
            _file: TextIO | None = None,
            _line: str | None = None,
        ) -> None:
            events.append(f'warning: {message}')

        with warnings.catch_warnings(), patch.object(warnings, 'showwarning', record_warning):
            warnings.simplefilter('always', UserWarning)
            with render_config_warnings():
                warnings.warn('unrelated', UserWarning, stacklevel=2)
                events.append('after warning')

        assert events == ['warning: unrelated', 'after warning']

    def test_prints_config_warnings_immediately(self, capsys: pytest.CaptureFixture[str]) -> None:
        with render_config_warnings():
            _warn('invalid [tool.docwrap].demo value 1, ignoring')
            captured = capsys.readouterr()
        assert captured.err == 'warning: invalid [tool.docwrap].demo value 1, ignoring\n'
        assert captured.out == ''

    def test_renders_despite_ignore_filter(self, capsys: pytest.CaptureFixture[str]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with render_config_warnings():
                _warn('invalid [tool.docwrap].demo value 1, ignoring')
        assert 'warning: invalid [tool.docwrap].demo' in capsys.readouterr().err

    def test_renders_warnings_before_body_raises(self, capsys: pytest.CaptureFixture[str]) -> None:

        def warn_then_fail() -> None:
            _warn('invalid [tool.docwrap].demo value 1, ignoring')
            assert 'warning: invalid [tool.docwrap].demo' in capsys.readouterr().err
            raise OSError('boom')

        with pytest.raises(OSError, match='boom'), render_config_warnings():
            warn_then_fail()


class TestResolveConfig:
    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(str(f))
        assert isinstance(cfg, DocwrapConfig)

    def test_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.line_length == DEFAULT_LINE_LENGTH
        assert cfg.style is None

    def test_reads_convention_as_style(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nconvention = "numpy"\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.style == 'numpy'

    def test_reads_line_length(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 100\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.line_length == 100

    def test_style_key(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstyle = "google"\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.style == 'google'

    def test_style_wins_over_convention(self, tmp_path: Path) -> None:
        toml = '[tool.docwrap]\nconvention = "numpy"\nstyle = "google"\n'
        (tmp_path / 'pyproject.toml').write_text(toml)
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.style == 'google'

    def test_type_hints_resolvable(self) -> None:
        hints = get_type_hints(resolve_config)
        assert 'path' in hints


class TestResolveConfigPathForms:
    def test_existing_directory_starts_discovery_at_itself(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 60\n')
        assert resolve_config(tmp_path).line_length == 60

    def test_nonexistent_file_starts_discovery_at_parent(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 60\n')
        assert resolve_config(tmp_path / 'does_not_exist.py').line_length == 60

    def test_resolved_config_accepted_by_all_public_operations(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 60\n')
        f = tmp_path / 'mod.py'
        f.write_text('def foo():\n    """Short."""\n')
        cfg = resolve_config(f)
        assert docwrap.wrap_source('x = 1\n', config=cfg) == 'x = 1\n'
        assert docwrap.wrap_source_report('x = 1\n', config=cfg).total == 0
        assert docwrap.check('x = 1\n', config=cfg).status == 'unchanged'
        assert docwrap.wrap_file_report(f, config=cfg).total == 1

    def test_str_and_pathlike_accepted(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 60\n')
        assert resolve_config(str(tmp_path / 'mod.py')).line_length == 60
        assert resolve_config(tmp_path / 'mod.py').line_length == 60


class TestResolvedConfig:
    def test_composes_transformation_and_operational_settings(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'line-length = 100\n'
            'color = "never"\n'
            'incremental = "docstring"\n'
            'show-skips = true\n'
            'allow-on-failure = true\n'
        )
        resolved = find_config(tmp_path / 'test.py')
        assert isinstance(resolved, ResolvedConfig)
        assert resolved.config.line_length == 100
        assert resolved.color == 'never'
        assert resolved.incremental == 'docstring'
        assert resolved.show_skips is True
        assert resolved.allow_on_failure is True
        assert resolved.line_length_source is LineLengthSource.DOCWRAP

    def test_defaults_without_configuration(self, tmp_path: Path) -> None:
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.config == DocwrapConfig()
        assert resolved.allow_on_failure is False
        assert resolved.color == 'auto'
        assert resolved.incremental is None
        assert resolved.line_length_source is LineLengthSource.DEFAULT
        assert resolved.show_skips is False

    def test_resolve_config_returns_transformation_component(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nline-length = 100\n')
        assert resolve_config(tmp_path / 'test.py') == find_config(tmp_path / 'test.py').config


class TestSummaryStartConfig:
    def test_invalid_summary_start_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "middle"\n')
        with pytest.warns(DocwrapConfigWarning, match='summary-start'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.summary_start == 'preserve'

    def test_resolve_config_includes_summary_start(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "second-line"\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.summary_start == 'second-line'

    def test_summary_start_default(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.summary_start == 'preserve'

    def test_summary_start_first_line(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "first-line"\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.summary_start == 'first-line'

    def test_summary_start_preserve(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "preserve"\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.summary_start == 'preserve'

    def test_summary_start_second_line(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsummary-start = "second-line"\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.summary_start == 'second-line'


class TestWalkBoundary:
    def test_git_file_stops_the_walk(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 100\n', encoding='utf-8'
        )
        repo = tmp_path / 'repo'
        repo.mkdir()
        (repo / '.git').write_text('gitdir: /elsewhere\n', encoding='utf-8')
        nested = repo / 'src'
        nested.mkdir()
        assert resolve_config(nested / 'mod.py').line_length != 100

    def test_git_root_pyproject_is_found(self, tmp_path: Path) -> None:
        (tmp_path / '.git').mkdir()
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 100\n', encoding='utf-8'
        )
        nested = tmp_path / 'src' / 'pkg'
        nested.mkdir(parents=True)
        assert resolve_config(nested / 'mod.py').line_length == 100

    def test_no_git_anywhere_walks_to_root(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 100\n', encoding='utf-8'
        )
        nested = tmp_path / 'a' / 'b'
        nested.mkdir(parents=True)
        assert resolve_config(nested / 'mod.py').line_length == 100

    def test_pyproject_above_git_root_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 100\n', encoding='utf-8'
        )
        repo = tmp_path / 'repo'
        repo.mkdir()
        (repo / '.git').mkdir()
        nested = repo / 'src'
        nested.mkdir()
        assert resolve_config(nested / 'mod.py').line_length != 100


class TestWalkUp:
    def test_finds_parent_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('[tool.docwrap]\nline-length = 99\n')
        subdir = tmp_path / 'sub'
        subdir.mkdir()
        cfg = find_config(subdir / 'test.py').config
        assert cfg.line_length == 99


class TestWrapSummaryConfig:
    def test_invalid_wrap_summary_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = "yes"\n')
        with pytest.warns(DocwrapConfigWarning, match='wrap-summary'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.wrap_summary is False

    def test_resolve_config_includes_wrap_summary(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = true\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        cfg = resolve_config(f)
        assert cfg.wrap_summary is True

    def test_wrap_summary_default(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.wrap_summary is False

    def test_wrap_summary_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = false\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.wrap_summary is False

    def test_wrap_summary_true(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nwrap-summary = true\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.wrap_summary is True


@pytest.fixture(autouse=True)  # noqa: V103
def clear_config_cache_between_tests() -> Iterator[None]:
    clear_config_cache()
    yield
    clear_config_cache()


class TestCachedTomlSharing:
    @pytest.mark.parametrize('first_lookup', ['incremental', 'full'])
    @pytest.mark.parametrize('file_exists', [True, False])
    def test_file_and_directory_inputs_share_toml_cache(
        self,
        tmp_path: Path,
        first_lookup: str,
        *,
        file_exists: bool,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'color = "always"\n'
            'incremental = "file"\n'
            'style = "google"\n'
            '[tool.docwrap.stats]\n'
            'sections = ["impact"]\n'
        )
        incremental_path = tmp_path / 'incremental.py'
        full_path = tmp_path / 'full.py'
        if file_exists:
            incremental_path.write_text('pass\n')
            full_path.write_text('pass\n')

        with patch.object(tomllib, 'load', wraps=tomllib.load) as load:
            if first_lookup == 'incremental':
                incremental = find_incremental_config(incremental_path)
                full = find_config(full_path)
            else:
                full = find_config(full_path)
                incremental = find_incremental_config(incremental_path)
            stats = find_stats_config(tmp_path)
            color = find_color_config(tmp_path)

        assert incremental == 'file'
        assert full.config.style == 'google'
        assert stats.sections == ('impact',)
        assert color == 'always'
        assert load.call_count == 1

    def test_shares_toml_cache_with_find_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\ncolor = "always"\nstyle = "google"\n'
        )
        monkeypatch.chdir(tmp_path)
        config_mod._find_config_cached.cache_clear()
        config_mod._load_tool_table.cache_clear()

        with patch.object(tomllib, 'load', wraps=tomllib.load) as load:
            color = find_color_config(tmp_path)
            cfg = resolve_config(tmp_path / 'dummy.py')

        assert color == 'always'
        assert cfg.style == 'google'
        # The second lookup uses the parsed tool-table cache.
        assert load.call_count == 1

    def test_stats_and_wrap_config_share_toml_cache(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nstyle = "google"\n[tool.docwrap.stats]\nsections = ["impact"]\n'
        )
        config_mod._find_config_cached.cache_clear()
        config_mod._find_stats_config_cached.cache_clear()
        config_mod._load_tool_table.cache_clear()

        with patch.object(tomllib, 'load', wraps=tomllib.load) as load:
            wrap_config = resolve_config(tmp_path / 'dummy.py')
            stats_config = find_stats_config(tmp_path)

        assert wrap_config.style == 'google'
        assert stats_config.sections == ('impact',)
        assert load.call_count == 1


class TestIncrementalConfig:
    def test_find_config_includes_incremental(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "file"\n')
        f = tmp_path / 'test.py'
        f.write_text('pass\n')
        resolved = find_config(f)
        assert resolved.incremental == 'file'

    @pytest.mark.parametrize(
        'toml',
        [
            '[tool.docwrap]\n',
            '[tool.docwrap]\nincremental = "bogus"\n',
            '[tool.docwrap]\nincremental = 1\n',
        ],
    )
    def test_find_incremental_config_defaults_silently(
        self,
        tmp_path: Path,
        toml: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(toml)
        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            assert find_incremental_config(tmp_path) is None

    def test_find_incremental_config_ignores_other_invalid_fields(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'incremental = "docstring"\n'
            'style = "bogus"\n'
            'color = "rainbow"\n'
            'line-length = 0\n'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            assert find_incremental_config(tmp_path) == 'docstring'

    @pytest.mark.parametrize(
        ('toml_value', 'expected'),
        [
            ('"file"', 'file'),
            ('"definition"', 'definition'),
            ('"docstring"', 'docstring'),
            ('true', 'definition'),
            ('false', False),
        ],
    )
    def test_find_incremental_config_reads_valid_values(
        self, tmp_path: Path, toml_value: str, expected: object
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(f'[tool.docwrap]\nincremental = {toml_value}\n')
        assert find_incremental_config(tmp_path) == expected

    def test_incremental_default_none(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental is None

    def test_incremental_definition(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "definition"\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental == 'definition'

    def test_incremental_docstring(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "docstring"\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental == 'docstring'

    def test_incremental_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = false\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental is False

    def test_incremental_file(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "file"\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental == 'file'

    def test_incremental_invalid_warns(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = "bogus"\n')
        with pytest.warns(DocwrapConfigWarning, match='bogus'):
            resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental is None

    def test_incremental_true_becomes_definition(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nincremental = true\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.incremental == 'definition'


class TestMalformedConfig:
    @pytest.mark.parametrize(
        ('toml_value', 'invalid_value'),
        [('true', True), ('"eighty"', 'eighty'), ('0', 0), ('-1', -1)],
    )
    def test_invalid_line_length_warns_and_uses_default(
        self,
        tmp_path: Path,
        toml_value: str,
        invalid_value: object,
    ) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text(f'[tool.docwrap]\nline-length = {toml_value}\n')
        match = rf'invalid \[tool\.docwrap\]\.line-length value {invalid_value!r}'
        with pytest.warns(DocwrapConfigWarning, match=match):
            resolved = find_config(tmp_path / 'test.py')
        assert resolved.config.line_length == DEFAULT_LINE_LENGTH
        assert resolved.line_length_source is LineLengthSource.DEFAULT

    def test_malformed_toml_warns_and_uses_defaults(self, tmp_path: Path) -> None:
        toml = tmp_path / 'pyproject.toml'
        toml.write_text('this is not valid toml [[[')
        with pytest.warns(DocwrapConfigWarning, match='pyproject.toml'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.line_length == DEFAULT_LINE_LENGTH
        assert cfg.style is None

    def test_missing_line_length_is_silent(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            resolved = find_config(tmp_path / 'test.py')
        assert resolved.config.line_length == DEFAULT_LINE_LENGTH
        assert resolved.line_length_source is LineLengthSource.DEFAULT


class TestResolutionWarningContract:
    def test_cached_color_resolution_does_not_warn_again(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\ncolor = "rainbow"\n')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            find_color_config(tmp_path)
            find_color_config(tmp_path)
            clear_config_cache()
            find_color_config(tmp_path)
        config_warnings = [
            warning for warning in caught if issubclass(warning.category, DocwrapConfigWarning)
        ]
        assert len(config_warnings) == 2

    def test_cached_resolution_does_not_warn_again(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "bogus"\n')
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter('always')
            resolve_config(tmp_path / 'mod.py')
        assert any(issubclass(w.category, DocwrapConfigWarning) for w in first)
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter('always')
            resolve_config(tmp_path / 'mod.py')
        assert not any(issubclass(w.category, DocwrapConfigWarning) for w in second)
        clear_config_cache()
        with warnings.catch_warnings(record=True) as third:
            warnings.simplefilter('always')
            resolve_config(tmp_path / 'mod.py')
        assert any(issubclass(w.category, DocwrapConfigWarning) for w in third)

    def test_invalid_fields_warn_in_resolution_order(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\n'
            'zz-unknown = 1\n'
            'line-length = 0\n'
            'color = "bad"\n'
            'convert = "bad"\n'
            'fill-style = "bad"\n'
            'incremental = "bad"\n'
            'smart-slack = -1\n'
            'show-skips = "bad"\n'
            'strip-types = "bad"\n'
            'style = "bad"\n'
            'summary-start = "bad"\n'
            'reflow = "bad"\n'
            'wrap-summary = "bad"\n'
            'allow-on-failure = "bad"\n'
        )
        keys = (
            'zz-unknown',
            'line-length',
            'color',
            'convert',
            'fill-style',
            'incremental',
            'smart-slack',
            'show-skips',
            'strip-types',
            'style',
            'summary-start',
            'reflow',
            'wrap-summary',
            'allow-on-failure',
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DocwrapConfigWarning)
            find_config(tmp_path / 'mod.py')

        assert [
            next(key for key in keys if f'[tool.docwrap].{key}' in str(warning.message))
            for warning in caught
        ] == list(keys)

    def test_invalid_selected_style_ignores_valid_convention(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nstyle = "bogus"\nconvention = "numpy"\n'
        )
        with pytest.warns(DocwrapConfigWarning, match=r'invalid \[tool\.docwrap\]\.style'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg.style is None

    def test_library_resolution_prints_nothing_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nfill-style = "bogus"\n')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            resolve_config(tmp_path / 'mod.py')
        captured = capsys.readouterr()
        assert captured.err == ''
        assert captured.out == ''

    def test_malformed_toml_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('this is not toml [')
        with pytest.warns(DocwrapConfigWarning, match='pyproject.toml'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg == DocwrapConfig()

    def test_missing_file_and_missing_section_are_silent(self, tmp_path: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            missing = resolve_config(tmp_path / 'mod.py')
            clear_config_cache()
            (tmp_path / 'pyproject.toml').write_text('[tool.other]\nx = 1\n')
            missing_section = resolve_config(tmp_path / 'mod.py')
        assert missing == DocwrapConfig()
        assert missing_section == DocwrapConfig()

    def test_non_table_docwrap_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool]\ndocwrap = 3\n')
        with pytest.warns(DocwrapConfigWarning, match=r'\[tool\.docwrap\]'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg == DocwrapConfig()

    def test_non_table_tool_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('tool = 3\n')
        with pytest.warns(DocwrapConfigWarning, match=r'\[tool\]'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg == DocwrapConfig()

    def test_style_wins_over_convention_and_only_selected_is_validated(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nstyle = "numpy"\nconvention = "bogus"\n'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg.style == 'numpy'

    def test_unknown_convert_warns_and_falls_back_to_none(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nconvert = "bogus"\n')
        with pytest.warns(DocwrapConfigWarning, match=r'invalid \[tool\.docwrap\]\.convert'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg.convert is None

    def test_unknown_key_warns_and_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nmystery = 1\n')
        with pytest.warns(DocwrapConfigWarning, match=r'unknown \[tool\.docwrap\]\.mystery'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg == DocwrapConfig()

    def test_unknown_style_warns_and_falls_back_to_none(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstyle = "bogus"\n')
        with pytest.warns(DocwrapConfigWarning, match=r'invalid \[tool\.docwrap\]\.style'):
            cfg = resolve_config(tmp_path / 'mod.py')
        assert cfg.style is None

    @pytest.mark.skipif(sys.platform == 'win32', reason='chmod 0 is ineffective on Windows')
    def test_unreadable_file_warns_and_defaults(self, tmp_path: Path) -> None:
        pyproject = tmp_path / 'pyproject.toml'
        pyproject.write_text('[tool.docwrap]\nline-length = 60\n')
        pyproject.chmod(0)
        try:
            with pytest.warns(DocwrapConfigWarning, match='pyproject.toml'):
                cfg = resolve_config(tmp_path / 'mod.py')
        finally:
            pyproject.chmod(0o644)
        assert cfg == DocwrapConfig()


class TestShowSkipsConfig:
    @pytest.mark.parametrize('raw', ['1', '"false"', '[]'])
    def test_invalid_show_skips_warns_and_defaults(
        self,
        tmp_path: Path,
        raw: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(f'[tool.docwrap]\nshow-skips = {raw}\n')
        with pytest.warns(DocwrapConfigWarning, match='show-skips'):
            resolved = find_config(tmp_path / 'test.py')
        assert resolved.show_skips is False

    def test_show_skips_default(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.show_skips is False

    def test_show_skips_false(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nshow-skips = false\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.show_skips is False

    def test_show_skips_true(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nshow-skips = true\n')
        resolved = find_config(tmp_path / 'test.py')
        assert resolved.show_skips is True


class TestSmartSlackConfig:
    def test_default_smart_slack_is_eight(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')
        cfg = find_config(tmp_path / 'test.py').config
        assert cfg.smart_slack == 8

    @pytest.mark.parametrize('raw', ['-1', '"four"', 'true'])
    def test_invalid_smart_slack_warns_and_defaults(
        self,
        tmp_path: Path,
        raw: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(f'[tool.docwrap]\nsmart-slack = {raw}\n')
        with pytest.warns(DocwrapConfigWarning, match='smart-slack'):
            cfg = find_config(tmp_path / 'test.py').config
        assert cfg.smart_slack == 8

    def test_smart_slack_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nsmart-slack = 10\n')
        cfg = resolve_config(tmp_path / 'test.py')
        assert cfg.smart_slack == 10


class TestStatsConfig:
    _DEFAULT_SECTIONS = (
        'source',
        'docstrings',
        'distribution',
        'longest',
        'impact',
        'recommendations',
    )

    def test_defaults_without_stats_table(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\n')

        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            cfg = find_stats_config(tmp_path / 'test.py')

        assert cfg == StatsConfig(
            sections=self._DEFAULT_SECTIONS,
            longest_per_category=3,
        )

    def test_empty_sections_is_valid(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap.stats]\nsections = []\n')

        with warnings.catch_warnings():
            warnings.simplefilter('error', DocwrapConfigWarning)
            cfg = find_stats_config(tmp_path)

        assert cfg.sections == ()

    @pytest.mark.parametrize('raw', ['true', '0', '-1', '"three"'])
    def test_invalid_longest_per_category_warns_and_defaults(
        self,
        tmp_path: Path,
        raw: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            f'[tool.docwrap.stats]\nlongest-per-category = {raw}\n'
        )

        with pytest.warns(
            DocwrapConfigWarning,
            match=r'\[tool\.docwrap\.stats\]\.longest-per-category',
        ):
            cfg = find_stats_config(tmp_path)

        assert cfg.longest_per_category == 3

    @pytest.mark.parametrize(
        'raw',
        [
            '"source"',
            '["source", 1]',
            '["source", "unknown"]',
            '["source", "source"]',
        ],
    )
    def test_invalid_sections_warn_and_fall_back(
        self,
        tmp_path: Path,
        raw: str,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(f'[tool.docwrap.stats]\nsections = {raw}\n')

        with pytest.warns(DocwrapConfigWarning, match=r'\[tool\.docwrap\.stats\]\.sections'):
            cfg = find_stats_config(tmp_path)

        assert cfg.sections == self._DEFAULT_SECTIONS

    @pytest.mark.parametrize(
        ('table', 'expected_sections', 'expected_limit'),
        [
            ('sections = ["impact"]\n', ('impact',), 3),
            ('longest-per-category = 5\n', _DEFAULT_SECTIONS, 5),
        ],
    )
    def test_missing_fields_use_independent_defaults(
        self,
        tmp_path: Path,
        table: str,
        expected_sections: tuple[str, ...],
        expected_limit: int,
    ) -> None:
        (tmp_path / 'pyproject.toml').write_text(f'[tool.docwrap.stats]\n{table}')

        cfg = find_stats_config(tmp_path)

        assert cfg.sections == expected_sections
        assert cfg.longest_per_category == expected_limit

    def test_non_table_stats_warns_and_defaults(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text('[tool.docwrap]\nstats = "full"\n')

        with pytest.warns(DocwrapConfigWarning, match=r'\[tool\.docwrap\]\.stats') as caught:
            cfg = find_stats_config(tmp_path)

        assert cfg == StatsConfig(
            sections=self._DEFAULT_SECTIONS,
            longest_per_category=3,
        )
        assert len(caught) == 1

    def test_public_wrap_config_is_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap]\nline-length = 88\n[tool.docwrap.stats]\nsections = ["impact"]\n'
        )

        resolved = find_config(tmp_path / 'test.py')
        cfg = resolved.config

        assert isinstance(cfg, DocwrapConfig)
        assert cfg.line_length == 88
        assert resolved.line_length_source is LineLengthSource.DOCWRAP
        assert not hasattr(cfg, 'sections')
        assert not hasattr(cfg, 'longest_per_category')

    def test_sections_preserve_exact_order(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap.stats]\nsections = ["recommendations", "source", "impact"]\n'
        )

        cfg = find_stats_config(tmp_path)

        assert cfg.sections == ('recommendations', 'source', 'impact')

    def test_unknown_nested_key_warns_and_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap.stats]\n'
            'sections = ["impact"]\n'
            'longest-per-category = 5\n'
            'mystery = true\n'
        )

        with pytest.warns(DocwrapConfigWarning, match=r'\[tool\.docwrap\.stats\]\.mystery'):
            cfg = find_stats_config(tmp_path)

        assert cfg.sections == ('impact',)
        assert cfg.longest_per_category == 5

    def test_valid_longest_per_category(self, tmp_path: Path) -> None:
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.docwrap.stats]\nlongest-per-category = 7\n'
        )

        cfg = find_stats_config(tmp_path)

        assert cfg.longest_per_category == 7
