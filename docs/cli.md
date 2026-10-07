# CLI reference

Use the CLI to inspect or apply docstring wrapping across files and directories. Start with a
summary, inspect the proposed patch, then write it when ready:

```bash
docwrap src/
docwrap --diff src/
docwrap --write src/
```

These use the built-in 79-column limit, `smart` fill, smart slack 8, cautious reflow, and no
summary wrapping. To set a project width, add this to `pyproject.toml`:

```toml
[tool.docwrap]
line-length = 88
```

The detailed options below cover narrower selection, output formats, and deliberate style changes.
For agent-triggered commits and edits, see the [agent hook guide](hook.md).

## Synopsis

```text
docwrap [options] [PATH ...]
docwrap hook --agent NAME [options]
```

`docwrap` accepts `.py` files or directories (recursively expanded to `**/*.py`). Existing
non-`.py` files are silently ignored, while missing explicit paths are retained and reported as
errors. Every recursive scan prunes common non-source directories (`.git`, `__pycache__`, `.venv`,
`build`, `dist`, etc.) before Git filtering, including inside Git worktrees. Tracked descendants of
those directories are therefore omitted from directory scans. Among the remaining paths, Git
worktree scans include tracked files and untracked, non-ignored files while honoring standard Git
exclusions. An explicitly named Python file remains eligible even when Git ignores it or its parent
directory has a skipped name. Nested worktrees apply their own tracked state and exclusions. If Git
is unavailable or fails inside a worktree, docwrap warns with the reason and keeps every discovered
Python file. Outside a worktree, the filesystem fallback is silent. Symlinks are processed when
passed explicitly but skipped during directory recursion. Paths default to `.` when omitted. `hook`
is the agent hook entry point; `-a`/`--agent` selects the agent adapter (see [hook.md](hook.md)).

## Options

`--version` prints `docwrap <version>` from installed distribution metadata and exits 0. It exits 2
if that metadata is unavailable.

### Output modes (mutually exclusive)

| Flag        | Short | Description                                                 |
| ----------- | ----- | ----------------------------------------------------------- |
| `--preview` | `-p`  | Show each docstring that would change                       |
| `--diff`    | `-d`  | Show a unified diff                                         |
| `--write`   | `-w`  | Rewrite files in place                                      |
| `--json`    | `-j`  | Print results as JSON                                       |
| `--stats`   | `-s`  | Report docstring line lengths and help choose a line length |

Without a mode flag, `docwrap` shows a per-file summary of changed docstrings. Modes are mutually
exclusive -- only one may be specified at a time.

### Modifiers

| Flag                                   | Short | Description                                                                                        |
| -------------------------------------- | ----- | -------------------------------------------------------------------------------------------------- |
| `--length`                             | `-l`  | Maximum line length (default: 79)                                                                  |
| `--verbose`                            | `-v`  | Also list unchanged files                                                                          |
| `--color`                              |       | When to use color (default: `auto`)                                                                |
| `--style`                              |       | Use this docstring style instead of detecting each docstring's style (`google`, `numpy`, `sphinx`) |
| `--convert`                            |       | Rewrite docstrings in another style (`google`, `numpy`, `sphinx`)                                  |
| `--strip-types`                        |       | Remove inline types from structured entries                                                        |
| `--summary-start`                      |       | Where a multi-line summary starts (default: `preserve`)                                            |
| `--wrap-summary` / `--no-wrap-summary` |       | Wrap summaries longer than the line length (default: off)                                          |
| `--show-skips`                         |       | In preview, list summaries left unwrapped because they are too long (default: off)                 |
| `--hide-skips`                         |       | In preview, do not list summaries left unwrapped because they are too long                         |
| `--fill-style`                         |       | How to choose line breaks: `smart`, `greedy`, or `balanced` (default: `smart`)                     |
| `--smart-slack`                        |       | Columns smart fill may give up to avoid a stranded fragment (default: `8`)                         |
| `--reflow`                             |       | How freely to rejoin lines before wrapping: `none`, `cautious`, or `full` (default: `cautious`)    |

`--verbose` only affects the default summary and `--write` modes. It is silently ignored with
`--preview` and `--diff`.

### Stats-only modifiers

| Flag                           | Value                           | Description                        |
| ------------------------------ | ------------------------------- | ---------------------------------- |
| `--stats-sections`             | `SECTION[,SECTION...]`          | Report sections to show, in order  |
| `--stats-longest-per-category` | positive integer (default: `3`) | Longest lines to list per category |

Both options are valid only with `-s` or `--stats`.

### Incremental mode

| Flag                  | Short | Description                                                                                      |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------ |
| `--incremental`       | `-i`  | Wrap only docstrings affected by changes since the last commit                                   |
| `--incremental-level` |       | How much around each change to wrap (`file`, `definition`, `docstring`); implies `--incremental` |
| `--no-incremental`    |       | Wrap every docstring, even when configuration enables incremental mode                           |

`-i` / `--incremental`, `--incremental-level`, and `--no-incremental` are mutually exclusive.
`--incremental-level` implicitly enables incremental mode.

Incremental mode wraps only docstrings in definitions that changed relative to HEAD:

```
docwrap -i .                             # level from config or "definition"
docwrap --incremental-level file .       # file level -- any change wraps all
docwrap --incremental-level docstring .  # only docstrings with edited text
docwrap --no-incremental .               # full mode (overrides config)
```

`--stats` is unaffected by `--incremental` -- it always analyzes all docstrings.

All output modes compose with `--incremental`: `--diff`, `--write`, `--preview`, `--json`, and the
default summary table.

**Configuration:**

```
[tool.docwrap]
incremental = "definition"  # "file", "definition", "docstring", or false
```

CLI flags (`-i`, `--incremental-level`, `--no-incremental`) override the config value.

**Error handling:** Nonfatal incremental warnings below appear in human output modes. JSON leaves
stderr silent for these cases: paths outside Git are omitted from `files`, and configuration-only
Git failures fall back to full mode without a warning.

- Not in a git repo: files are skipped with a warning in human output modes.
- Git failure with explicit `-i`: exits non-zero.
- Git failure with config-only incremental: falls back to full mode with a warning in human output
  modes.
- File read or decoding failure: reports an `error` and exits 2.
- Syntax, tokenization, or invalid-source failure: reports `invalid` in JSON and exits 2.
- No diff hunks or no eligible definitions/docstrings: succeeds silently.

Incremental file diagnostics use the source data already read during eligibility analysis; output
rendering does not reopen or reanalyze a failed file.

## Output

### Default mode (summary)

Columnar table showing wrapping statistics per file (Name, Skip, Total, Wrap, Wrap%). Without `-v`,
only files with changes are shown. The `Skip` column counts summary-length skips. With `-v`,
unchanged files with docstrings or skipped summaries also appear. When 2+ files have changes, a
TOTAL row aggregates every displayed row, including unchanged rows shown by `--verbose`. Filenames
and percentages of changed files use yellow severity coloring.

### Write mode (`--write`)

Writes changed files in place and shows the tabular summary to stdout. With `-v`, unchanged files
are included in the summary.

### Diff mode (`--diff`)

Prints unified diff to stdout with coloring (bold file and hunk headers, red removed lines, green
added lines). File is not modified.

### Preview mode (`--preview`)

Per-docstring before/after snippets showing exactly what changed. Headers are bold, before text is
red, after text is green. When an over-long summary is skipped, preview hides the skipped-summary
diagnostic by default. `--show-skips` shows those diagnostics and skip-only preview blocks. The
setting does not affect JSON output, counts, exit codes, or before/after snippets for docstrings
that changed.

### Stats mode (`--stats`)

Analyzes docstring line lengths across the codebase. Shows configured length and scan count, source
code and docstring percentile tables, distribution histograms, longest wrappable lines, wrapping
impact at standard line lengths with line-change and scope columns, and recommended targets. Never
modifies files. A valid stats invocation over readable, parseable files exits 0, including one with
stats display configuration warnings. Other invalid project configuration can exit 2. Stats never
uses exit 1 for pending changes.

Unreadable files (including missing paths and decoding failures) are skipped and counted in the
footer. Files that cannot be parsed are omitted from measurements. Each such file is reported on
stderr, the report still covers the remaining files, and stats exits 2.

#### Report sections

The `sections` setting accepts these body section identifiers:

| Identifier        | Report content                                           |
| ----------------- | -------------------------------------------------------- |
| `source`          | Source-code percentile table                             |
| `docstrings`      | Wrappable and applicable non-wrappable percentile tables |
| `distribution`    | Docstring line-length distribution histograms            |
| `longest`         | Longest wrappable lines, grouped by wrappable category   |
| `impact`          | Wrapping-impact table                                    |
| `recommendations` | Recommended line-length targets                          |

The list controls both selection and exact display order. Each section may occur once. Omitted
sections stay hidden. An empty project list, `[]`, or an empty CLI value, `--stats-sections=`,
selects no body sections.

Scan and configuration context always stays above the body. Skip and error diagnostics always stay
below it. A section without applicable data remains absent. The sections that render retain their
configured relative order.

Select a compact report or place recommendations before impact and source:

```text
docwrap -s --stats-sections=source,recommendations .
docwrap -s --stats-sections=recommendations,impact,source .
```

#### Longest-line limit

`longest-per-category` is a positive integer that limits the lines shown independently for prose,
entries, and summary when summary wrapping is enabled. The default is `3`. The setting remains
configured when `longest` is omitted, but it has no visible effect until that section is selected.

#### Stats configuration

Configure the complete default report with a nested table:

```toml
[tool.docwrap.stats]
sections = [
  "source",
  "docstrings",
  "distribution",
  "longest",
  "impact",
  "recommendations",
]
longest-per-category = 3
```

CLI values replace the corresponding nested stats values. Nested stats values replace built-in
defaults. `--stats-sections` ignores whitespace around each comma-separated name.

Project configuration uses warnings and safe defaults:

- A non-table `[tool.docwrap].stats` value warns and restores both built-in defaults.
- A non-array `sections` value, or an array with a non-string, unknown, or duplicate member, warns
  and restores the complete default section list.
- A boolean, non-integer, zero, or negative `longest-per-category` value warns and uses `3`.
- An unknown nested key warns and is ignored. Valid recognized fields in the same table survive.

These warnings do not make an otherwise valid stats invocation fail. CLI validation is stricter:
invalid names, duplicate names, empty interior items, non-integer limits, and non-positive limits
exit 2 instead of falling back.

#### Recommended targets

When selected, `recommendations` prints up to three line-length targets at its configured position:

- **Best fit** -- the standard target closest to your source code's 95th percentile line length
  (p95). This answers: "what width matches my existing code style?" Source p95 is the length at or
  below which 95% of your non-docstring, non-blank, non-comment source lines fall -- a natural
  upper bound that excludes one-off long lines. When two standard targets are equidistant, the
  smaller (tighter) target is preferred.

- **Low churn** -- the standard target that minimizes the sum of the absolute physical-line-count
  change for each docstring if you adopt it. This measures changes in line count, not added and
  removed diff lines. Rewrapping five lines into five different lines contributes zero to this
  measure, so the selected width need not produce the smallest diff. Tiebreakers: fewer files
  changed, then fewer docstrings changed, then smaller target.

- **Configured** -- shown only when you have a docwrap-specific line length set (via `--length` or
  `[tool.docwrap]`). Shows the cost of your current configuration.

Best fit and Low churn may recommend the same target or different ones. They optimize different
measures: proximity to source p95 and changes in docstring line counts. Neither measures
readability or guarantees the fewest changed lines in a diff.

### JSON mode (`--json`)

Outputs a structured, color-free JSON report to stdout. Mutually exclusive with all other modes.
The `-v` flag does not filter JSON records. In incremental mode, `no-hunks` and `no-eligible` paths
are omitted from `files`. Configuration diagnostics may appear on stderr.

Envelope:

```text
{"version": 1, "files": [...], "total_skips": 0}
```

Each file record contains:

| Field         | Type            | Description                                                                                                                                        |
| ------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `path`        | string          | File path as given on the command line                                                                                                             |
| `status`      | string          | `unchanged`, `changed`, `error`, or `invalid`                                                                                                      |
| `warning`     | string or null  | Reason when `status` is `error` or `invalid`                                                                                                       |
| `style`       | string or null  | Resolved file-level style; null on errors or when configured/detected styles do not resolve to one file-level style (for example, absent or mixed) |
| `line_length` | integer or null | Effective line length (null on error)                                                                                                              |
| `docstrings`  | object          | `wrapped`, `total`, `skipped_summaries` counts                                                                                                     |
| `skips`       | array           | Safety-skip records; empty on error                                                                                                                |
| `changes`     | array           | Per-docstring diffs (only changed docstrings listed)                                                                                               |

Each entry in `changes` has `qualname`, `start_line`, `end_line`, `before`, `after`,
`summary_skipped`, and `summary_width`.

Each entry in `skips` has `line`, `qualname`, `code`, and `message`. The top-level `total_skips`
field is the sum of all per-file `skips` arrays.

#### Schema versioning

The top-level `version` field indicates a major-compatibility generation. It is bumped when the
schema changes in a breaking way: a field is removed, renamed, or its type/semantics change.
Additive optional fields do not bump the version -- consumers should ignore unknown fields and
treat absent optional fields as their documented default.

## Color

Color output is controlled by this precedence chain:

1. CLI `--color always` enables color or `--color never` disables it.
2. `NO_COLOR` environment variable (when set, disables color)
3. `[tool.docwrap].color` from configuration discovered from the current working directory
4. `auto` (checks whether the output stream is a terminal)

`--color auto` uses steps 2-4. Color configuration is shared by the invocation; wrapping
configuration is resolved separately for each input file.

## Configuration

For wrapping settings, `docwrap` reads `[tool.docwrap]` from `pyproject.toml`, discovered by
walking up from each input file and stopping at the git repository root. Configuration in a
directory above the repository is not used.

### Multi-project configuration

The default summary, preview, diff, write, and JSON modes resolve wrapping configuration for each
file, so one invocation can span projects. See the
[incremental guide](incremental.md#multi-project-configuration) for the first-file settings,
per-repository diff handling, and failure behavior. Stats mode also uses the first collected file
for displayed line length, sections, section order, and longest-per-category limit; analysis
resolves each file's transformation settings.

```toml
[tool.docwrap]
line-length = 99
# convert = "numpy"  # optional: convert docstring style
# strip-types = true # optional: remove inline type annotations
```

The commented settings change docstring content deliberately; see
[style conversion](#style-conversion) before enabling them. Output, fill, summary, and reflow
settings are listed in the options above and explained below.

The transformation settings `line-length`, `style`/`convention`, `convert`, `strip-types`,
`summary-start`, `wrap-summary`, `fill-style`, `smart-slack`, and `reflow` correspond to fields on
the public `DocwrapConfig` record. The operational settings `color`, `incremental`, `show-skips`,
`allow-on-failure`, and `stats` configure CLI or hook behavior outside that record.

### Config conventions

**Key naming.** TOML keys in `[tool.docwrap]` use kebab-case (`allow-on-failure`, `strip-types`,
`summary-start`, `wrap-summary`, `line-length`), matching the convention in `pyproject.toml` itself
(`requires-python`, `optional-dependencies`). Python field names on `DocwrapConfig` use snake_case
(`strip_types`, `summary_start`).

If `[tool.docwrap]` does not set `line-length`, the default is 79.

A docstring whose content fits in the one-line literal form remains one line. docwrap checks that
fit before applying `summary-start`, including when it is set to `second-line`.

`color` controls ANSI color output. `auto` checks whether the output stream is a terminal after
`NO_COLOR` and the current working directory's color configuration are considered. Explicit
`--color always` and `--color never` override both; `--color auto` does not.

**Precedence**: CLI flags (`--length`, `--style`, `--convert`, `--strip-types`, `--summary-start`,
`--wrap-summary`/`--no-wrap-summary`, `--fill-style`, `--smart-slack`, `--reflow`) override
`pyproject.toml` values. The built-in default is `--no-wrap-summary`; `[tool.docwrap].wrap-summary`
overrides that default when neither CLI spelling is present.

## Style conversion

`--convert` converts docstrings from their detected style to a target style. The source style is
auto-detected per docstring, so a codebase with mixed conventions can be unified in a single pass.
Ordinary detection leaves a highest-count marker tie undetected. During conversion only, the
requested target breaks such a tie when its markers share the lead. It does not override a unique
source-style winner or supply a style when no target markers exist.

When `--style` also supplies an explicit source style, that style remains the fallback for source
or undetected content. Once the converted output contains enough target evidence, docwrap uses the
target parser and renderer instead. Partially preserved output may therefore remain mixed rather
than becoming pure target syntax, but repeated runs keep its converted fields stable.

docwrap preserves each unsupported section as a complete source-styled span, including its header,
entries, and associated prose. Supported neighboring sections still convert, so the resulting
docstring may intentionally use mixed styles.

A multi-name NumPy parameter entry lists several parameters separated by commas. When converting
such an entry to Google style, docwrap emits one parameter entry per name in source order,
duplicates the shared description, and warns that the result may need review.

If conversion or type stripping cannot preserve an authored name on a Returns or Yields entry,
docwrap skips the complete docstring. It leaves the docstring unchanged and records an
`unrepresentable-return` safety skip instead of applying a partial conversion.

```text
docwrap --convert numpy -w .       # convert everything to NumPy style
docwrap --convert google --diff .  # preview conversion as a diff
```

`--convert` works with all output modes (`--diff`, `--write`, `--preview`, `--json`, and the
default summary).

`--strip-types` removes inline type annotations from structured entries (e.g. `param (str):`
becomes `param:`). It works independently or combined with `--convert`:

```text
docwrap --strip-types -w .                   # strip types, keep current style
docwrap --convert google --strip-types -w .  # convert AND strip in one pass
```

Both operations are idempotent: running twice produces identical output. In particular, detected
target fields are not converted or rendered through an explicit source style again.

**pyproject.toml:**

```toml
[tool.docwrap]
convert = "numpy"  # convert docstrings to numpy style
strip-types = true # remove inline type annotations
```

### `--fill-style smart|greedy|balanced`

Controls how lines are filled:

- `smart` (default): Greedy filling with bounded local repair for short parenthesized or bracketed
  fragments split at a line end.
- `greedy`: Left-packed word-boundary planning, subject to the marker binding and boundary-gap
  stabilization shared by all fill styles. Its output is not guaranteed to match `textwrap.fill`.
- `balanced`: Minimum-raggedness filling retained as an opt-in style.

`--smart-slack` controls how many additional columns `smart` may remove from the greedy line by
backing off the preceding gap and opener fragment to repair a recognized delimiter-fragment break.
Width that the greedy line already left unused does not count against this budget. Setting it to
`0` makes `smart` behave exactly like `greedy`.

### `--reflow none|cautious|full`

Controls how aggressively docwrap joins lines.

- `none`: Split-only. Lines under the limit are never joined. Lines that overflow are individually
  split.
- `cautious` (default): Reflow within runs. Joins adjacent prose lines when the preceding line has
  no terminal punctuation and the next starts with a letter, supported inline token, option,
  parenthesis, or quoted number. Fills each run while preserving sentence boundaries, colons, and
  bullets.
- `full`: Joins and reflows all lines within each eligible prose region. Protected content such as
  code blocks retains its layout.

Configurable in `pyproject.toml`:

```toml
[tool.docwrap]
reflow = "cautious"
```

## Entry point

The installed `docwrap` command and Python's module entry point accept the same arguments. Use the
module form when docwrap is installed in the active Python environment:

```bash
python -m docwrap --diff src/
```

## Exit codes

| Code | Meaning                                           |
| ---- | ------------------------------------------------- |
| 0    | No changes needed (or all writes succeeded)       |
| 1    | Changes pending (files not modified)              |
| 2    | Error (I/O failure, syntax error, encoding error) |

If stdout closes while docwrap is producing output, docwrap stops writing, examines no further
files, prints no traceback, and exits with the status of the work completed so far.

If stdout is already closed when docwrap starts, docwrap prints
`error: standard output is unavailable` to stderr and exits 2 before reading arguments,
configuration, or files. This applies to every command, including `--help`, `--version`, and
`hook`.

## Safety-skip warnings

Every human output mode prints one stderr line per safety-skipped docstring:

```text
path/to/file.py:12: Cls.method skipped (line-continuation): backslash-newline line continuation at index 9
```

Safety skips never change exit codes: check mode still exits `1` only when writing would change
files. `--hide-skips` only hides summary-length skip diagnostics; it never hides safety warnings.
The default summary's `Skip` column counts summary-length skips. Stats mode reports safety skips
through a separate `Skipped N docstrings for safety.` footer.

JSON output stays at version `1`. Each file object has a `skips` array of
`{line, qualname, code, message}` records, error records carry an empty `skips` array, and the
envelope has `total_skips`. A skip-only file keeps `status: "unchanged"`.
