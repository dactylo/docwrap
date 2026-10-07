# Incremental mode

Incremental mode limits docstring wrapping to definitions that changed relative to HEAD. This is
useful for large codebases where running docwrap on every file is too noisy or too slow during
iterative development.

When enabled, docwrap runs `git diff -U0 HEAD` to identify changed hunks, then uses the AST to
determine which definitions (or docstrings) overlap those hunks. Only matching docstrings are
wrapped; unchanged regions pass through untouched.

Diff collection covers the repository so Git can pair rename sources with their destinations. Only
selected files are processed. A renamed file with edited definitions keeps unchanged docstrings
untouched at `definition` or `docstring` level; `file` level still wraps the whole changed file.

Explicitly selected untracked files are fully eligible, including files ignored by Git. Directory
discovery continues to exclude ignored files. Tracked files with no changed hunks remain untouched.
The CLI and PostToolUse hook use the same eligibility rule for explicitly selected files.

## Levels

Incremental mode supports three levels that control how broadly changes propagate:

| Level        | Scope                                                             |
| ------------ | ----------------------------------------------------------------- |
| `file`       | Any change in a file makes all docstrings in that file eligible   |
| `definition` | Only definitions containing changed lines have docstrings wrapped |
| `docstring`  | Only docstrings whose own text was edited are wrapped             |

**`file`** is the coarsest: if a file has any diff hunk, every docstring in it is wrapped. This is
useful when you want incremental filtering at the file level but full wrapping within each changed
file.

**`definition`** (the default incremental level) maps each diff hunk to its exact owning function
or class definition. A change in a parent definition selects that parent's docstring but does not
automatically select docstrings in nested functions or methods. A change owned by a nested
definition selects that nested definition instead. Hunks outside any definition (module-level code)
make the module docstring eligible. Deletion-only hunks use the original definition boundaries, so
deleting a decorator or a statement does not select an adjacent, unchanged definition. When the
original source cannot be parsed, deleted lines are skipped; added and replaced lines remain
eligible under the current source.

**`docstring`** is the finest: only docstrings whose own line range overlaps a diff hunk are
wrapped. This is useful when you want to avoid reformatting docstrings you did not edit, even if
the surrounding definition changed.

## Configuration

### pyproject.toml

```toml
[tool.docwrap]
incremental = "definition" # "file", "definition", "docstring", or false
```

Setting `incremental` to a level string enables incremental mode at that level. Setting it to
`false` explicitly disables incremental mode. Omitting the key leaves full-file mode enabled in
every context.

### CLI flags

| Flag                  | Short | Description                                                                                      |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------ |
| `--incremental`       | `-i`  | Wrap only docstrings affected by changes since the last commit                                   |
| `--incremental-level` |       | How much around each change to wrap (`file`, `definition`, `docstring`); implies `--incremental` |
| `--no-incremental`    |       | Wrap every docstring, even when configuration enables incremental mode                           |

These three flags are mutually exclusive. `--incremental-level` implicitly enables incremental
mode.

### Precedence

CLI flags always override `pyproject.toml`. The resolution chain:

1. `--no-incremental` -- full mode, regardless of config.
2. `--incremental-level <level>` -- use specified level.
3. `-i` / `--incremental` -- use level from config, or `definition` if config is unset.
4. `[tool.docwrap].incremental` -- use config value.
5. Context default -- full mode.

### Multi-project configuration

The default summary, preview, diff, write, and JSON modes fully resolve wrapping configuration for
each file. These modes support files from multiple projects in one invocation.

Only these behaviors use the first collected file:

- Incremental mode gets its configured incremental level from the first collected file.
- Stats mode gets the sections, section order, longest-per-category limit, and displayed line
  length from the first collected file. Stats analysis resolves transformation settings for each
  file, including `wrap-summary`; CLI transformation flags still override every file.

Incremental mode resolves a repository for each file and collects diffs for each repository, so one
invocation may span repositories.

When an invocation spans repositories, a `git diff` failure is handled by mode. When a CLI flag
explicitly enables incremental mode, the failure is fatal for the whole invocation and no files are
processed. When incremental mode comes from configuration rather than a flag, only the affected
repository falls back to full wrapping; files in other repositories keep incremental mode.

A repository with no commits yet is not a failure. Every selected file in it is new, so each one is
fully eligible in both modes, as in the PostToolUse hook.

Each requested file receives one of five outcomes before rendering:

- `full`: process every docstring because incremental mode does not restrict the file or a
  file-level change makes the whole file eligible;
- `ranges`: process only the definitions or docstrings selected by changed ranges;
- `no-hunks`: succeed silently because the file has no diff hunks;
- `no-eligible`: succeed silently because its hunks select no definitions or docstrings;
- `failed`: report the stored file diagnostic without reading or analyzing the file again.

File read and decoding failures are errors. Syntax, tokenization, and other invalid-source failures
are invalid-file results. Both failure classes exit 2 in human and JSON modes. JSON uses `error`
for read or decoding failures and `invalid` for invalid source.

## Adoption workflow

To incrementally adopt docwrap in an existing codebase:

1. **Baseline run**: run `docwrap --stats .` to understand current line-length distribution and
   choose a target line length.

2. **Configure**: set the line length and incremental level in `pyproject.toml`:

   ```toml
   [tool.docwrap]
   line-length = 88
   incremental = "definition"
   ```

3. **Hook integration**: register the recommended PreToolUse hook to wrap staged files before
   commits. With the incremental configuration above, the hook wraps changed definitions and
   catches files edited outside the agent.

   **PostToolUse alternative:** register PostToolUse instead to wrap definitions after edits. Do
   not enable both modes; PreToolUse and PostToolUse are alternative integration paths.

4. **Gradual adoption**: as definitions are edited over time, their eligible docstrings are wrapped
   under the configured policy. Untouched docstrings can remain unchanged indefinitely, and safety
   skips still apply. Incremental mode does not ensure that every docstring eventually conforms.

5. **Full wrap**: when ready, run `docwrap --no-incremental -w .` to override the configured
   incremental setting and wrap the entire codebase. This produces a single formatting commit that
   can be added to `.git-blame-ignore-revs`.

## Line length configuration

Incremental mode uses the same line-length resolution chain as full mode:

1. CLI `--length` flag.
2. `[tool.docwrap].line-length` in `pyproject.toml`.
3. Default: 79.

## Parallel hook caveat

If multiple agent hooks may modify the same staged file, serialize them under one hook
registration. The [agent hook guide](hook.md#composing-with-other-hooks) explains how to order them
and the current Antigravity sequencer incompatibility when a formatter emits a successful
`systemMessage`. Keep the sequencer's deny handling intact.

## Partial staging

For a normal staged commit, the PreToolUse hook wraps only the staged content. It synchronizes the
working-tree copy only when its newline-normalized text equals the staged source before wrapping
and its byte snapshot is still unchanged before the write. Explicit selection by a preceding
`git add` does not bypass either check. Commit pathspecs such as those in `git commit file.py`
process working-tree content through the separate [working-tree path](hook.md#working-tree-path).
This preserves unstaged changes during normal staged commits.
