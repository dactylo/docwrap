"""Config discovery from pyproject.toml."""

import contextlib
import contextvars
import functools
import os
import sys
import tomllib
import warnings
from collections.abc import Generator
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from docwrap._color import path_text, visible_controls
from docwrap._exceptions import DocwrapConfigWarning
from docwrap._styles import get_styles
from docwrap._types import (
    COLOR_MODES,
    DEFAULT_LINE_LENGTH,
    DEFAULT_STATS_LONGEST_PER_CATEGORY,
    DEFAULT_STATS_SECTIONS,
    FILL_STYLES,
    INCREMENTAL_LEVELS,
    REFLOW_MODES,
    SUMMARY_STARTS,
    ColorMode,
    DocwrapConfig,
    FillStyle,
    IncrementalLevel,
    LineLengthSource,
    ReflowMode,
    StatsSection,
    StyleName,
    SummaryStart,
    WrapOverrides,
)

_KNOWN_STATIC_KEYS = frozenset({'line-length', 'stats'})
_RENDER_CONFIG_WARNINGS = contextvars.ContextVar[bool](
    'docwrap_render_config_warnings', default=False
)
_STATS_KEYS = frozenset({'longest-per-category', 'sections'})


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedConfig:
    """Internal resolution result composing public and operational settings.

    Attributes:
        config: Public transformation configuration.
        allow_on_failure: Hook failure policy from `allow-on-failure`.
        color: Terminal color mode consumed by the CLI.
        incremental: CLI/hook incremental selection value.
        line_length_source: Provenance of the resolved line length.
        show_skips: Whether preview output shows skipped-summary diagnostics.
    """

    config: DocwrapConfig
    allow_on_failure: bool = False
    color: ColorMode = 'auto'
    incremental: IncrementalLevel | Literal[False] | None = None
    line_length_source: LineLengthSource = LineLengthSource.DEFAULT
    show_skips: bool = False


@dataclass(frozen=True, slots=True)
class StatsConfig:
    """Resolved configuration for the human-readable stats report.

    Attributes:
        sections: Selected body sections in display order.
        longest_per_category: Maximum longest-line rows per wrappable category.
    """

    sections: tuple[StatsSection, ...] = DEFAULT_STATS_SECTIONS
    longest_per_category: int = DEFAULT_STATS_LONGEST_PER_CATEGORY


@dataclass(frozen=True, slots=True)
class _BoolField:
    """Spec for a boolean-valued [tool.docwrap] field."""

    key: str
    default: bool


@dataclass(frozen=True, slots=True)
class _EnumField[T: str | None]:
    """Spec for a string-enum [tool.docwrap] field."""

    key: str
    valid: frozenset[T]
    default: T


@dataclass(frozen=True, slots=True)
class _IncrementalField:
    """Spec for the incremental field (False | level-string | None)."""

    key: str
    default: IncrementalLevel | Literal[False] | None


@dataclass(frozen=True, slots=True)
class _IntField:
    """Spec for an integer-valued [tool.docwrap] field."""

    key: str
    default: int
    minimum: int


@dataclass(frozen=True, slots=True)
class _StyleAliasField:
    """Spec for the style field, which also accepts the 'convention' alias."""

    keys: tuple[str, ...]
    default: StyleName | None


class _ConfigFields(NamedTuple):
    """Typed field registry in configuration warning order."""

    color: _EnumField[ColorMode]
    convert: _EnumField[StyleName | None]
    fill_style: _EnumField[FillStyle]
    incremental: _IncrementalField
    smart_slack: _IntField
    show_skips: _BoolField
    strip_types: _BoolField
    style: _StyleAliasField
    summary_start: _EnumField[SummaryStart]
    reflow: _EnumField[ReflowMode]
    wrap_summary: _BoolField
    allow_on_failure: _BoolField


@functools.cache
def _config_fields() -> _ConfigFields:
    """Declare [tool.docwrap] field specs in configuration warning order."""
    _res = ResolvedConfig(config=DocwrapConfig())
    _cfg = _res.config
    return _ConfigFields(
        color=_EnumField(
            'color',
            frozenset(COLOR_MODES),
            default=_res.color,
        ),
        convert=_EnumField(
            'convert',
            frozenset(get_styles()),
            default=_cfg.convert,
        ),
        fill_style=_EnumField(
            'fill-style',
            frozenset(FILL_STYLES),
            default=_cfg.fill_style,
        ),
        incremental=_IncrementalField('incremental', default=_res.incremental),
        smart_slack=_IntField(
            'smart-slack',
            default=_cfg.smart_slack,
            minimum=0,
        ),
        show_skips=_BoolField('show-skips', default=_res.show_skips),
        strip_types=_BoolField('strip-types', default=_cfg.strip_types),
        style=_StyleAliasField(keys=('style', 'convention'), default=_cfg.style),
        summary_start=_EnumField(
            'summary-start',
            frozenset(SUMMARY_STARTS),
            default=_cfg.summary_start,
        ),
        reflow=_EnumField(
            'reflow',
            frozenset(REFLOW_MODES),
            default=_cfg.reflow,
        ),
        wrap_summary=_BoolField('wrap-summary', default=_cfg.wrap_summary),
        allow_on_failure=_BoolField('allow-on-failure', default=_res.allow_on_failure),
    )


def _find_pyproject(start: Path) -> Path | None:
    """Walk up from start looking for pyproject.toml, stopping at the git root."""
    current = start if start.is_dir() else start.parent
    for parent in (current, *current.parents):
        candidate = parent / 'pyproject.toml'
        if candidate.is_file():
            return candidate
        # Stop at git root
        if (parent / '.git').exists():
            break
    return None


def _known_docwrap_keys() -> frozenset[str]:
    """Return every supported [tool.docwrap] key, including aliases."""
    keys = set(_KNOWN_STATIC_KEYS)
    for spec in _config_fields():
        if isinstance(spec, _StyleAliasField):
            keys.update(spec.keys)
        else:
            keys.add(spec.key)
    return frozenset(keys)


def _warn(message: str) -> None:
    """Emit one configuration-resolution diagnostic."""
    if _RENDER_CONFIG_WARNINGS.get():
        print(f'warning: {visible_controls(message)}', file=sys.stderr)
        return
    warnings.warn(message, DocwrapConfigWarning, stacklevel=2)


@functools.cache
def _load_tool_table(directory: Path) -> dict[str, Any] | None:
    """Find and parse pyproject.toml, returning the [tool] table."""
    toml_path = _find_pyproject(directory)
    if toml_path is None:
        return None
    try:
        with toml_path.open('rb') as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as error:
        _warn(f'invalid pyproject.toml {path_text(toml_path)}: {error}, using defaults')
        return None
    tool_raw = data.get('tool')
    if tool_raw is None:
        return None
    if not isinstance(tool_raw, dict):
        _warn(f'invalid [tool] value {tool_raw!r} in {path_text(toml_path)}, using defaults')
        return None
    return cast('dict[str, Any]', tool_raw)


def _resolve_bool_field(dw: dict[str, Any], spec: _BoolField) -> bool:
    """Resolve a boolean [tool.docwrap] field."""
    raw: object = dw.get(spec.key)
    if isinstance(raw, bool):
        return raw
    if raw is not None:
        _warn(
            f'invalid [tool.docwrap].{spec.key} value {raw!r}, using {str(spec.default).lower()}',
        )
    return spec.default


def _resolve_enum_field[T: str | None](dw: dict[str, Any], spec: _EnumField[T]) -> T:
    """Resolve a string-enum [tool.docwrap] field."""
    raw: object = dw.get(spec.key)
    if isinstance(raw, str):
        for valid in spec.valid:
            if raw == valid:
                return valid
    if raw is not None:
        _warn(f'invalid [tool.docwrap].{spec.key} value {raw!r}, using {spec.default!r}')
    return spec.default


def _resolve_incremental_field(
    dw: dict[str, Any],
    spec: _IncrementalField,
    *,
    warn: bool = True,
) -> IncrementalLevel | Literal[False] | None:
    """Resolve the incremental field (False | level-string | None)."""
    raw: object = dw.get(spec.key)
    if raw is None:
        return spec.default
    if isinstance(raw, bool):
        return 'definition' if raw else False
    if isinstance(raw, str) and raw in INCREMENTAL_LEVELS:
        return raw
    if warn:
        _warn(f'invalid [tool.docwrap].{spec.key} value {raw!r}, ignoring')
    return spec.default


def _resolve_int_field(dw: dict[str, Any], spec: _IntField) -> int:
    """Resolve a bounded integer [tool.docwrap] field."""
    raw: object = dw.get(spec.key)
    if not isinstance(raw, bool) and isinstance(raw, int) and raw >= spec.minimum:
        return raw
    if raw is not None:
        _warn(f'invalid [tool.docwrap].{spec.key} value {raw!r}, using {spec.default}')
    return spec.default


def _resolve_line_length(tool: dict[str, Any]) -> tuple[int, LineLengthSource]:
    """Resolve line-length from [tool.docwrap]."""
    dw = tool.get('docwrap')
    if isinstance(dw, dict):
        docwrap = cast('dict[str, Any]', dw)
        val: object = docwrap.get('line-length')
        if isinstance(val, bool):
            pass
        elif isinstance(val, int) and val > 0:
            return val, LineLengthSource.DOCWRAP
        if 'line-length' in docwrap:
            _warn(
                f'invalid [tool.docwrap].line-length value {val!r}, using {DEFAULT_LINE_LENGTH}',
            )
    return DEFAULT_LINE_LENGTH, LineLengthSource.DEFAULT


def _resolve_stats_longest_per_category(stats: dict[str, Any]) -> int:
    """Resolve the longest-line row limit from [tool.docwrap.stats]."""
    raw: object = stats.get('longest-per-category')
    if not isinstance(raw, bool) and isinstance(raw, int) and raw >= 1:
        return raw
    if raw is not None:
        _warn(
            'invalid [tool.docwrap.stats].longest-per-category '
            f'value {raw!r}, using {DEFAULT_STATS_LONGEST_PER_CATEGORY}',
        )
    return DEFAULT_STATS_LONGEST_PER_CATEGORY


def _resolve_stats_sections(stats: dict[str, Any]) -> tuple[StatsSection, ...]:
    """Resolve the ordered body sections from [tool.docwrap.stats]."""
    raw: object = stats.get('sections')
    if raw is None:
        return DEFAULT_STATS_SECTIONS
    if isinstance(raw, list):
        items = cast('list[object]', raw)
        if all(isinstance(item, str) for item in items):
            names = cast('list[str]', items)
            if all(name in DEFAULT_STATS_SECTIONS for name in names) and len(names) == len(
                set(names)
            ):
                return cast('tuple[StatsSection, ...]', tuple(names))
    _warn(f'invalid [tool.docwrap.stats].sections value {raw!r}, using default')
    return DEFAULT_STATS_SECTIONS


def _resolve_stats_config(tool: dict[str, Any]) -> StatsConfig:
    """Resolve the closed [tool.docwrap.stats] display configuration."""
    raw_docwrap = tool.get('docwrap')
    if not isinstance(raw_docwrap, dict):
        return StatsConfig()
    docwrap = cast('dict[str, Any]', raw_docwrap)
    raw_stats: object = docwrap.get('stats')
    if raw_stats is None:
        return StatsConfig()
    if not isinstance(raw_stats, dict):
        _warn(f'invalid [tool.docwrap].stats value {raw_stats!r}, using defaults')
        return StatsConfig()
    stats = cast('dict[str, Any]', raw_stats)
    for key in sorted(stats.keys() - _STATS_KEYS):
        _warn(f'unknown [tool.docwrap.stats].{key}, ignoring')
    return StatsConfig(
        sections=_resolve_stats_sections(stats),
        longest_per_category=_resolve_stats_longest_per_category(stats),
    )


@functools.cache
def _find_stats_config_cached(directory: Path) -> StatsConfig:
    """Cache stats-display config resolution by resolved directory."""
    tool = _load_tool_table(directory)
    if tool is None:
        return StatsConfig()
    return _resolve_stats_config(tool)


def _resolve_style_alias_field(
    dw: dict[str, Any],
    spec: _StyleAliasField,
) -> StyleName | None:
    """Resolve style/convention: the first present key wins; only it is validated."""
    for key in spec.keys:
        if key not in dw:
            continue
        raw: object = dw[key]
        if isinstance(raw, str) and raw in get_styles():
            return raw
        _warn(f'invalid [tool.docwrap].{key} value {raw!r}, using {spec.default!r}')
        return spec.default
    return spec.default


@functools.cache
def _find_config_cached(directory: Path) -> ResolvedConfig:
    """Cached config resolution keyed by resolved directory."""
    tool = _load_tool_table(directory)
    if tool is None:
        return ResolvedConfig(config=DocwrapConfig())

    docwrap_raw = tool.get('docwrap')
    if isinstance(docwrap_raw, dict):
        dw = cast('dict[str, Any]', docwrap_raw)
        for key in sorted(dw.keys() - _known_docwrap_keys()):
            _warn(f'unknown [tool.docwrap].{key}, ignoring')
    else:
        dw = {}
        if docwrap_raw is not None:
            _warn(f'invalid [tool.docwrap] value {docwrap_raw!r}, using defaults')

    line_length, line_length_source = _resolve_line_length(tool)
    specs = _config_fields()
    color = _resolve_enum_field(dw, specs.color)
    convert = _resolve_enum_field(dw, specs.convert)
    fill_style = _resolve_enum_field(dw, specs.fill_style)
    incremental = _resolve_incremental_field(dw, specs.incremental)
    smart_slack = _resolve_int_field(dw, specs.smart_slack)
    show_skips = _resolve_bool_field(dw, specs.show_skips)
    strip_types = _resolve_bool_field(dw, specs.strip_types)
    style = _resolve_style_alias_field(dw, specs.style)
    summary_start = _resolve_enum_field(dw, specs.summary_start)
    reflow = _resolve_enum_field(dw, specs.reflow)
    wrap_summary = _resolve_bool_field(dw, specs.wrap_summary)
    allow = _resolve_bool_field(dw, specs.allow_on_failure)

    return ResolvedConfig(
        config=DocwrapConfig(
            convert=convert,
            fill_style=fill_style,
            line_length=line_length,
            reflow=reflow,
            smart_slack=smart_slack,
            strip_types=strip_types,
            style=style,
            summary_start=summary_start,
            wrap_summary=wrap_summary,
        ),
        allow_on_failure=allow,
        color=color,
        incremental=incremental,
        line_length_source=line_length_source,
        show_skips=show_skips,
    )


def clear_config_cache() -> None:
    """Clear cached configuration so later reads observe edited files.

    Configuration is cached per resolved directory for the life of the process,
    which suits a CLI invocation but not a long-running caller. Call this after
    changing a `pyproject.toml` in the same process. Clears every file-derived
    cache; `_config_fields` holds static field specifications and is
    intentionally left alone.
    """
    _load_tool_table.cache_clear()
    _find_config_cached.cache_clear()
    _find_stats_config_cached.cache_clear()


def effective_config(cfg: DocwrapConfig, overrides: WrapOverrides) -> DocwrapConfig:
    """Apply CLI overrides on top of a resolved DocwrapConfig.

    Args:
        cfg: Base resolved configuration.
        overrides: CLI flag overrides (None fields are not applied; override
            fields that are not `DocwrapConfig` fields, such as `show_skips`,
            are consumed by the CLI itself). `strip_types` combines override
            and configured enablement with OR, so `False` does not disable a
            configured `True`.

    Returns:
        New DocwrapConfig with overrides applied.
    """
    config_fields = {f.name for f in fields(cfg)}
    changes: dict[str, Any] = {
        f.name: getattr(overrides, f.name)
        for f in fields(overrides)
        if f.name in config_fields and getattr(overrides, f.name) is not None
    }
    if 'strip_types' in changes:
        changes['strip_types'] = changes['strip_types'] or cfg.strip_types
    return replace(cfg, **changes)


def find_color_config(start: Path | None = None) -> ColorMode:
    """Read just the color config from pyproject.toml.

    Uses the shared cached TOML parse so find_color_config and resolve_config
    never redundantly parse the same file.

    Args:
        start: Directory to search from (defaults to cwd).

    Returns:
        Resolved ColorMode value.
    """
    directory = (start or Path()).resolve()
    return _find_config_cached(directory).color


def find_config(path: Path) -> ResolvedConfig:
    """Discover the composed configuration for the given file path.

    Args:
        path: File or directory from which to discover configuration.

    Returns:
        The resolved configuration for the path.
    """
    directory = path.resolve().parent if not path.is_dir() else path.resolve()
    return _find_config_cached(directory)


def find_incremental_config(
    start: Path | None = None,
) -> IncrementalLevel | Literal[False] | None:
    """Read only the incremental config from `pyproject.toml`.

    Args:
        start: File or directory from which to search, defaulting to cwd.

    Returns:
        Resolved incremental value, or the default when absent or invalid.
    """
    selected = (start or Path()).resolve()
    directory = selected if selected.is_dir() else selected.parent
    tool = _load_tool_table(directory)
    if tool is None:
        return ResolvedConfig(config=DocwrapConfig()).incremental
    raw_docwrap = tool.get('docwrap')
    if not isinstance(raw_docwrap, dict):
        return ResolvedConfig(config=DocwrapConfig()).incremental
    docwrap = cast('dict[str, Any]', raw_docwrap)
    return _resolve_incremental_field(docwrap, _config_fields().incremental, warn=False)


def find_stats_config(path: Path) -> StatsConfig:
    """Discover stats-display configuration for a file or directory.

    Args:
        path: File or directory from which to discover `pyproject.toml`.

    Returns:
        Resolved stats-display configuration.
    """
    directory = path.resolve().parent if not path.is_dir() else path.resolve()
    return _find_stats_config_cached(directory)


@contextlib.contextmanager
def render_config_warnings() -> Generator[None]:
    """Render configuration warnings immediately in the CLI diagnostic format.

    Sets a context-local presentation mode for `_warn()` without intercepting
    unrelated warning categories or changing their filters and handlers.

    Yields:
        None.
    """
    token = _RENDER_CONFIG_WARNINGS.set(True)
    try:
        yield
    finally:
        _RENDER_CONFIG_WARNINGS.reset(token)


def resolve_config(path: str | os.PathLike[str]) -> DocwrapConfig:
    """Resolve configuration for a file path.

    Discovery starts at `path` when it is an existing directory and at `path`'s
    parent otherwise, including when `path` names a file that does not exist
    yet. Results are cached per resolved directory for the process lifetime;
    call `clear_config_cache()` after editing a `pyproject.toml`.

    Resolution never raises for file content. Invalid values, unknown keys, and
    unreadable or malformed files emit `DocwrapConfigWarning` and fall back to
    defaults; missing files and missing sections resolve to defaults silently.

    Args:
        path: File or directory from which to locate `pyproject.toml`.

    Returns:
        DocwrapConfig ready to pass to any public wrapping operation.
    """
    return find_config(Path(path)).config
