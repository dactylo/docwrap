# Library API

Use `wrap_source()` when you need rewritten Python source, `check()` when you need a yes/no result,
and `wrap_source_report()` when you need counts or safety-skip records. Use `wrap_file_report()`
for an encoded file on disk. The CLI is simpler for a one-off directory pass; see the
[CLI reference](cli.md).

The source APIs use the built-in defaults unless you pass a `DocwrapConfig`. For example:

```python
import docwrap

source = 'def greet():\n    """Return a greeting."""\n'
config = docwrap.DocwrapConfig(line_length=88)
wrapped = docwrap.wrap_source(source, config=config)
```

For file operations, `config=None` resolves `[tool.docwrap]` in a nearby `pyproject.toml`:

```toml
[tool.docwrap]
line-length = 88
```

The following entries describe the complete supported API, result types, and failure behavior.

The source APIs accept Python's LF, CRLF, and CR physical line endings, including mixed endings.
Untouched text and final line terminators are retained exactly; rewritten literal content uses LF.
Differences in physical line endings alone do not trigger a rewrite. File writing follows the
existing file encoding and newline policy described under `wrap_file_report()` below.

## `docwrap.wrap_source(source, *, config=None) -> str`

Returns source code with docstrings wrapped. Returns source unchanged on parse errors.

```python
import docwrap

source = 'def greet():\n    """Return a greeting."""\n'
source = docwrap.wrap_source(source)
```

**Parameters:**

- `source` -- Python source code as a string.
- `config` -- `DocwrapConfig` instance (`None` means `DocwrapConfig()`).

**Raises:** `InvalidLineLengthError` if line_length is invalid; `TypeError` if a boolean-typed
configuration field is not a bool; `UnknownStyleError` if style is not recognized; `ValueError` if
a configuration field value is invalid.

## `docwrap.wrap_source_report(source, *, config=None, detail='summary') -> WrapReport`

Wraps docstrings in Python source, returning transformed source and metadata.

```python
import docwrap

source = 'def greet():\n    """Return a greeting."""\n'
config = docwrap.DocwrapConfig(line_length=88)
report = docwrap.wrap_source_report(source, config=config, detail='list')
print(f'{report.wrapped}/{report.total} docstrings wrapped')
for ds in report.docstrings:
    if ds.changed:
        print(f'  {ds.qualname} (line {ds.start_line})')
```

**Parameters:**

- `source` -- Python source code as a string.
- `config` -- `DocwrapConfig` instance (`None` means `DocwrapConfig()`).
- `detail` -- Level of detail in the report (`'summary'`, `'list'`, `'diff'`).

**Raises:** `InvalidLineLengthError` if line_length is invalid; `TypeError` if a boolean-typed
configuration field is not a bool; `UnknownStyleError` if style is not recognized; `ValueError` if
detail or a configuration field value is invalid.

## `docwrap.wrap_file_report(path, *, config=None, detail='summary', write=False) -> WrapReport`

Wraps docstrings in a Python file. Reads the file respecting its encoding cookie and normalizes
physical line endings to LF before wrapping. `WrapReport.source` uses LF even when the file uses
CRLF or lone CR, including when no docstring changes. An optional write uses the original encoding
and the [selected newline style](platform.md#files-and-encodings). When `config` is `None`,
configuration is resolved from `pyproject.toml` (falling back to defaults).

```python
import docwrap

report = docwrap.wrap_file_report('src/mylib/core.py', write=True)
```

**Parameters:**

- `path` -- Path to a Python source file.
- `config` -- `DocwrapConfig` instance (`None` resolves configuration from `path` with
  `resolve_config()`).
- `detail` -- Level of detail in the report (`'summary'`, `'list'`, `'diff'`).
- `write` -- If `True`, write the wrapped result back to the file.

**Raises:** `InvalidLineLengthError` if line_length is invalid; `OSError` if the file cannot be
read or written; `SyntaxError` if the Python source has a malformed or unknown encoding cookie;
`TypeError` if a boolean-typed configuration field is not a bool; `UnicodeDecodeError` if source
bytes are invalid for the detected encoding; `UnknownStyleError` if style is not recognized;
`ValueError` if detail or a configuration field value is invalid.

## `docwrap.check(source, *, config=None) -> WrapCheckResult`

Returns whether `wrap_source()` would change the given source, without modifying it.

```python
import docwrap

source = 'def greet():\n    """Return a greeting."""\n'
result = docwrap.check(source)
if result.status == 'changed':
    print('Docstrings need wrapping')
```

**Parameters:**

- `source` -- Python source code as a string.
- `config` -- `DocwrapConfig` instance (`None` = use defaults).

**Returns:** `WrapCheckResult` with `status`: `'changed'`, `'unchanged'`, or `'invalid'`, plus any
safety-skip records in `skips`.

**Raises:** `InvalidLineLengthError` if line_length is invalid; `TypeError` if a boolean-typed
configuration field is not a bool; `UnknownStyleError` if style is not recognized; `ValueError` if
a configuration field value is invalid.

## `docwrap.WrapCheckResult`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class WrapCheckResult:
    status: WrapStatus
    skips: tuple[DocstringSkip, ...] = ()
```

## `docwrap.WrapStatus`

```python
WrapStatus = Literal['changed', 'unchanged', 'invalid']
```

## `docwrap.resolve_config(path) -> DocwrapConfig`

Resolves configuration by walking up the directory tree to find `pyproject.toml` and reading
`[tool.docwrap]`. Discovery starts at `path` when it names an existing directory. Otherwise,
discovery starts at its parent, including when the file does not exist yet. The search stops at the
git repository root.

Resolution never raises for file content. Invalid values, unknown keys, and unreadable or malformed
files emit `DocwrapConfigWarning` and fall back to defaults. Missing files and missing
`[tool.docwrap]` sections resolve to defaults silently. `style` takes precedence over `convention`,
and docwrap validates only the selected value.

Results are cached per resolved directory for the process lifetime. Long-running embedders must
call `clear_config_cache()` after editing a `pyproject.toml`.

```python
import docwrap

cfg = docwrap.resolve_config('src/mylib/core.py')
print(cfg.line_length, cfg.style)
```

**Parameters:**

- `path` -- `str | os.PathLike[str]` file or directory path used to locate `pyproject.toml`.

## `docwrap.clear_config_cache() -> None`

Clears cached configuration. `resolve_config()` caches results per resolved directory for the life
of the process, which suits a CLI invocation but means a long-running caller never observes an
edited `pyproject.toml`. Call this after changing configuration in the same process.

```python
import docwrap

cfg = docwrap.resolve_config('src/mylib/core.py')
# pyproject.toml is edited on disk
docwrap.clear_config_cache()
cfg = docwrap.resolve_config('src/mylib/core.py')  # reflects the edit
```

## `docwrap.get_styles() -> frozenset[StyleName]`

Returns the set of built-in style names.

```python
import docwrap

print(docwrap.get_styles())  # frozenset({'google', 'numpy', 'sphinx'})
```

## `docwrap.StyleName`

```python
StyleName = Literal['google', 'numpy', 'sphinx']
```

## `docwrap.DetailLevel`

```python
DetailLevel = Literal['summary', 'list', 'diff']
```

Controls how much information `wrap_source_report` and `wrap_file_report` include in the report:

- `'summary'` -- Only aggregate counts (`total`, `wrapped`). No per-docstring details.
- `'list'` -- Per-docstring `DocstringChange` entries with `changed` flag but no text diffs.
- `'diff'` -- Per-docstring entries with `before`/`after` text for changed docstrings.

## `docwrap.ReflowMode`

```python
ReflowMode = Literal['none', 'cautious', 'full']
```

Controls how aggressively docwrap joins lines:

- `'none'` -- Split-only. Lines under the limit are never joined.
- `'cautious'` -- Joins adjacent prose lines across nonterminal breaks when the next line starts
  with a letter, supported inline token, option, parenthesis, or quoted number.
- `'full'` -- Joins all lines and reflows to fill the available width.

## `docwrap.FillStyle`

```python
FillStyle = Literal['balanced', 'greedy', 'smart']
```

Controls how line breaks are selected:

- `'smart'` -- Greedy filling with bounded repair for short parenthesized or bracketed fragments.
- `'greedy'` -- Left-packed word-boundary planning, subject to the marker binding and boundary-gap
  stabilization shared by all fill styles. Its output is not guaranteed to match `textwrap.fill`.
- `'balanced'` -- Minimizes line count, then weighted squared slack for less ragged output.

With `fill_style='smart'`, `DocwrapConfig.smart_slack` controls the maximum additional columns that
may be removed from the greedy line by backing off the preceding gap and opener fragment to repair
a recognized local split. Width already unused by greedy filling is excluded from this budget.
`smart_slack=0` is equivalent to `'greedy'`.

## `docwrap.DEFAULT_LINE_LENGTH`

```python
DEFAULT_LINE_LENGTH = 79
```

## `docwrap.SummaryStart`

```python
SummaryStart = Literal['preserve', 'first-line', 'second-line']
```

Controls whether summaries preserve their original position or start on the first or second line of
the docstring. A docstring whose content fits in the one-line literal form remains one line because
docwrap checks that fit before applying this setting.

## `docwrap.WrapReport`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class WrapReport:
    source: str  # transformed source code
    total: int  # total docstrings found
    wrapped: int  # docstrings that were wrapped
    convert_target: StyleName | None = None  # target style for conversion, or None
    converted: int = 0  # docstrings converted to a different style
    docstrings: tuple[DocstringChange, ...] = ()  # per-docstring details
    parse_error: bool = False  # True if source failed to parse
    skipped_summaries: int = 0  # summaries skipped (exceed target)
    style: StyleName | None = None  # resolved file-level style
    skips: tuple[DocstringSkip, ...] = ()  # safety-skip records
```

`percent_wrapped` property returns the percentage of docstrings that were wrapped.

## `docwrap.DocstringChange`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class DocstringChange:
    qualname: str  # qualified name (e.g. "Cls.method")
    start_line: int  # starting line number
    end_line: int  # ending line number
    changed: bool  # whether the docstring was modified
    before: str | None = None  # original text (only when detail='diff' and changed)
    after: str | None = None  # wrapped text (only when detail='diff' and changed)
    converted: bool = False  # whether this docstring was converted
    summary_skipped: bool = False  # whether summary wrapping was skipped
    summary_width: int | None = None  # width of over-long summary (only when skipped)
```

`DocstringChange` records accept keyword arguments only. `before` and `after` may be supplied only
when `changed=True`; `summary_width` may be supplied only when `summary_skipped=True`. Construction
raises `ValueError` when either invariant is violated.

## `docwrap.DocwrapConfig`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class DocwrapConfig:
    convert: StyleName | None = None
    fill_style: FillStyle = 'smart'
    line_length: int = DEFAULT_LINE_LENGTH
    reflow: ReflowMode = 'cautious'
    smart_slack: int = 8
    strip_types: bool = False
    style: StyleName | None = None
    summary_start: SummaryStart = 'preserve'
    wrap_summary: bool = False
```

Construction validates every field and raises `InvalidLineLengthError`, `UnknownStyleError`,
`TypeError`, or `ValueError` for invalid values. Configuration files are forgiving: they warn and
use defaults. Code is strict.

## `docwrap.InvalidLineLengthError`

```python
class InvalidLineLengthError(ValueError): ...
```

Raised when `line_length` is not a positive integer; booleans are rejected. The `line_length`
attribute retains the original invalid value, including values of the wrong type. The message
distinguishes a wrong type from an integer below one.

## `docwrap.UnknownStyleError`

```python
class UnknownStyleError(ValueError): ...
```

Raised when a style name is not recognized. The `style` attribute retains the original invalid
value, including a non-string value.

## `docwrap.DocwrapConfigWarning`

`UserWarning` subclass emitted by `resolve_config()` when a discovered configuration file cannot
take effect: unreadable or malformed files, non-table sections, unknown keys, and invalid values
for known keys. Standard `warnings` filters apply; promote the category to an error with
`warnings.simplefilter('error', docwrap.DocwrapConfigWarning)` to treat configuration mistakes as
failures.

Warning source attribution points into docwrap's own resolution internals rather than the caller's
`resolve_config()` call site, so the reported module, filename, and line number are not stable
parts of the API. Filter on the category (`docwrap.DocwrapConfigWarning`), not on `module=` or
`filename=`.

## Safety skips

Raw docstrings treat backslashes as literal text. In normal, non-raw docstrings, docwrap preserves
every backslash escape spelling byte-for-byte. When a docstring cannot be rewritten safely, docwrap
leaves it unchanged and records a `DocstringSkip` instead of corrupting it.

## `docwrap.DocstringSkip`

`DocstringSkip` is a frozen, keyword-only dataclass with fields `path`, `line`, `qualname`, `code`,
and `message`. The `code` field is the stable machine-readable contract (`docwrap.SkipCode`). The
`message` wording may improve between releases. `SkipCode` values are:

- `line-continuation`: The docstring contains a backslash-newline line continuation.
- `implicit-concatenation`: The docstring is an implicitly concatenated literal.
- `sentinel-unavailable`: No internal placeholder character was available.
- `duplicate-field`: Conversion cannot merge duplicate or conflicting Sphinx type fields
  losslessly, or a Google/NumPy target cannot preserve multiple typeless Sphinx Returns or Yields
  entries. Type-field conflicts are recorded only without `strip_types`; stripping deletes every
  conflicting field instead. The typeless Returns or Yields skip applies even with `strip_types`
  because stripping cannot preserve entry multiplicity.
- `unrepresentable-return`: Conversion or type stripping cannot preserve an authored name on a
  Returns or Yields entry in the target grammar, so the entire docstring remains unchanged.
- `quote-collision`: No quote style can delimit the rewritten content safely.
- `validation-failed`: The rewritten literal failed self-validation and was reverted.
- `file-validation-failed`: The rewritten file failed to parse, so the whole file was reverted. For
  this file-level record, `line` is `0` and `qualname` is empty.

When an explicitly requested transformation (`convert` or `strip_types`) deletes text, it also
deletes escapes in that text. Preservation applies to the text that the transformation keeps.

`WrapReport.skips` and `WrapCheckResult.skips` carry these records. `wrap_file_report` fills
`path`; `wrap_source_report` leaves it empty. `wrap_source` emits one `DocwrapSkipWarning`, a
`UserWarning` subclass, per skip. Use `wrap_source_report` for structured access. Skips never
change exit codes, and a skip-only file counts as unchanged.

## `docwrap.SkipCode`

```python
SkipCode = Literal[
    'duplicate-field',
    'file-validation-failed',
    'implicit-concatenation',
    'line-continuation',
    'quote-collision',
    'sentinel-unavailable',
    'unrepresentable-return',
    'validation-failed',
]
```

The stable machine-readable reason carried by `DocstringSkip.code`. Branch on these values rather
than on `DocstringSkip.message`, whose wording may change between releases. See
[`docwrap.DocstringSkip`](#docwrapdocstringskip) for what each value means.

## `docwrap.DocwrapSkipWarning`

`UserWarning` subclass emitted once per safety skip by `wrap_source()`. Use `wrap_source_report()`
to receive structured `DocstringSkip` records instead.

## `docwrap.hooks`

The public host adapter contract for `docwrap hook`, versioned by `docwrap.hooks.API_VERSION`
(currently `1`). `HookInput`, `HookResult`, `HookResponse`, and `HookAdapter` are structural
`typing.Protocol` types. An adapter can implement its own classes or use the concrete `HostInput`
and `HostResponse` records.

- `HookInput`: read-only `event` (`'pre_tool'` or `'post_tool'`), `command` (`str | None`, the
  shell command of a pre-tool payload), `cwd` (`str`), `command_cwd` (`str | None`, the absolute
  directory a pre-tool command executes in, or `None` when unknown), and `edited_paths`
  (`Sequence[str]`, the paths a post-tool payload reports as written).
- `HookResult`: read-only `decision` (`'allow'` or `'deny'`), `reason` (`str | None`, a deny
  explanation or allow receipt; `None` for a silent allow), and `notes` (`Sequence[str]`,
  informational lines such as safety skips).
- `HookResponse`: read-only `stdout` (`str`), `stderr` (`str`), and `exit_code` (`int`).
- `HookAdapter`: read-only `name` (`str`), `api_version` (`int`), and `requires_event` (`bool`);
  `parse(payload, event) -> HookInput | None`, where `None` means the payload describes nothing
  docwrap acts on; `render(result) -> HookResponse`; and `render_failure(reason) -> HookResponse`,
  the host's fail-closed response.
- `HostInput` and `HostResponse`: immutable, slotted records that implement the matching input and
  response protocols.
- `object_field()`, `payload_cwd()`, `absolute_directory()`, and `notes_text()`: shared parsing and
  diagnostic helpers. `payload_cwd()` serves post-tool context only; `absolute_directory()` returns
  a non-empty absolute path string or `None`.

Adapters are discovered through the `docwrap.hook_adapters` entry-point group, each entry point
naming an adapter class, and selected by `docwrap hook --agent NAME`. See the
[hook reference](hook.md) for installation, `--event`, and the failure tiers.

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MyInput:
    event: Literal['pre_tool', 'post_tool']
    command: str | None
    cwd: str
    command_cwd: str | None
    edited_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class MyResponse:
    stdout: str
    stderr: str
    exit_code: int


class MyHostAdapter:
    name = 'myhost'
    api_version = 1
    requires_event = False

    def parse(self, payload: Mapping[str, object], event: str | None) -> MyInput | None: ...

    def render(self, result: object) -> MyResponse: ...

    def render_failure(self, reason: str) -> MyResponse: ...
```
