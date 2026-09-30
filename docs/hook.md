# Agent hook guide

Use an agent hook to wrap Python docstrings as an agent edits files or before it commits them.
Start with PreToolUse for commit-time wrapping. This page covers host setup, supported commands,
failure handling, and composition with other hooks. For direct use, see the
[CLI reference](cli.md).

## Synopsis

```text
docwrap hook -a NAME [-e NAME] [-i | --no-incremental] [--allow-on-failure]
```

The `hook` subcommand reads one host hook payload as JSON from stdin, translates it through the
agent adapter named by `--agent`, and either formats a supported commit or denies an unsupported
one. `--agent` selects the coding agent application, not a machine or hostname. It is required;
docwrap never infers the host from the payload. `--event` is required for a host whose payload
carries no event name (Antigravity) and rejected for every other host. The built-in agents are
`claude`, `codex`, and `agy`.

## pre-commit

To wrap Python files through [pre-commit](https://pre-commit.com/), put this in your project's
`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: docwrap
        name: docwrap
        entry: docwrap --write
        language: system
        types: [python]
```

Install the docwrap CLI from a checkout and make sure `docwrap` is on `PATH`, then run
`pre-commit install`. The writing hook updates files and pre-commit asks you to stage the result.
For a report-only hook, change `entry` to `docwrap`; it exits 1 when wrapping is pending and does
not edit files. This integration is separate from the agent hooks below.

The repository also ships a pre-commit hook manifest with the IDs `docwrap` (writes files) and
`docwrap-check` (reports pending changes without writing). Consumers can select these IDs when
pointing pre-commit at a published repository revision. The local configuration above works from a
checkout without a release revision.

## Configuration

Choose one delivery mechanism and one hook mode.

Delivery mechanisms:

- **Plugin**: install the standalone `PreToolUse` hook package from this repository.
- **Manual config**: add the JSON below to the agent's hook config file.

Do not configure the same hook through both a plugin and manual JSON. If both sources register it,
the hook can run twice.

Hook modes:

- **PreToolUse (recommended)** intercepts `git commit` before execution. Depending on the commit
  mode, it updates staged blobs in the index or wraps selected working-tree files before Git stages
  them. Normal staged commits preserve partial staging.
- **PostToolUse (alternative)** wraps docstrings in each touched `.py` file immediately after file
  edits. The agent sees wrapped docstrings mid-turn, which enforces wrapping at every step but
  forces re-reads and uses extra context.

**Do not enable both modes.** PostToolUse would modify files mid-turn, and PreToolUse would
redundantly re-process them at commit time.

### Plugin packages

Use this checkout as the marketplace source. The plugins package only hook configuration, so
install the docwrap CLI first and confirm that `docwrap` is on `PATH`.

Install the Claude Code plugin from a docwrap checkout:

```bash
claude plugin marketplace add .
claude plugin install docwrap@docwrap
```

Install the Codex plugin from a docwrap checkout:

```bash
codex plugin marketplace add .
codex plugin add docwrap@docwrap
```

Install the Antigravity plugin from a docwrap checkout:

```bash
agy plugin install ./plugins/agy
```

After installation, restart the host and review or trust the registered hook. In Codex, run
`/hooks` and confirm that `PreToolUse` is active. Do not retain a manual `docwrap hook`
registration alongside the plugin; duplicate registration runs the hook twice.

Each host has its own plugin directory, `plugins/claude`, `plugins/codex`, and `plugins/agy`,
because Claude Code and Codex both load a plugin's `hooks/hooks.json` and one file cannot pass two
`--agent` values. Every shipped manifest invokes `docwrap hook` with its host's `--agent` value
and, for Antigravity, `--event`.

The standalone plugins register only `PreToolUse`. For Claude Code and Codex, if another hook also
rewrites the same tool input, use one ordered sequencer registration instead of installing both
standalone plugins. See [composition limits](#composing-with-other-hooks) for Antigravity.

### Claude Code manual settings

Use `~/.claude/settings.json` when a plugin is not registering the hook.

**PreToolUse (recommended)** -- intercepts `git commit` commands before execution. Depending on the
commit mode, it updates staged blobs in the index or wraps selected working-tree files before Git
stages them. Claude works uninterrupted, and normal staged commits preserve partial staging.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent claude"
          }
        ]
      }
    ]
  }
}
```

**PostToolUse (alternative)**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent claude"
          }
        ]
      }
    ]
  }
}
```

After changing settings, restart Claude Code and approve the hook if prompted.

### Codex manual settings

Use `~/.codex/hooks.json` or the hooks table in Codex `config.toml` when a plugin is not
registering the hook.

**PreToolUse (recommended)** -- intercepts `git commit` commands before execution. This uses the
same commit-time behavior as the Claude Code setup.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^Bash$",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent codex"
          }
        ]
      }
    ]
  }
}
```

**PostToolUse (alternative)** -- wraps `.py` files touched by Codex file edits. Codex reports file
edits with canonical `tool_name: "apply_patch"`.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "^apply_patch$",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent codex"
          }
        ]
      }
    ]
  }
}
```

After changing Codex hook config or enabling a hook plugin, restart Codex and run `/hooks`. Review
or trust the hook, then confirm the event shows `Active 1`. `Installed 1` with `Active 0` means
Codex discovered the hook but will not run it.

### Antigravity manual settings

Manual Antigravity support is verified with installed `agy` 1.1.22. Use
`~/.gemini/config/hooks.json` when configuring hooks manually instead of through the plugin.

**PreToolUse (recommended)** -- intercepts `run_command` tool calls before execution. This uses the
same commit-time wrapping behavior as the Claude Code and Codex setups. Antigravity expects deny
responses as top-level JSON with exit code 0, so the `agy` adapter renders that response.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "run_command",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent agy --event PreToolUse"
          }
        ]
      }
    ]
  }
}
```

**PostToolUse (alternative)** -- wraps `.py` files touched by Antigravity file edits.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "write_to_file|replace_file_content",
        "hooks": [
          {
            "type": "command",
            "command": "docwrap hook --agent agy --event PostToolUse"
          }
        ]
      }
    ]
  }
}
```

Verified caveat for `agy` 1.1.22:

- Antigravity PostToolUse does not provide a reliable denial path for this hook. Use PreToolUse
  when the hook must block a commit on infrastructure failure.

### Per-repo enable/disable

The hook checks the `docwrap.hook` git config key. If the key is missing, the hook defaults to
enabled.

Values follow Git boolean semantics: true/yes/on/1 to enable, false/no/off/0 to disable.

To disable the hook for a single repository:

```text
git config docwrap.hook false
```

To disable the hook globally and selectively re-enable it in one repository:

```text
git config --global docwrap.hook false
git config docwrap.hook true  # re-enable in this repo
```

### `[tool.docwrap]` config

The hook resolves wrapping settings for each processed file by walking up to the git repository
root, so nested projects govern their own files. For PreToolUse, the repository-scoped commit
resolves `incremental` and `allow-on-failure` once from `[tool.docwrap]` in the repository root's
`pyproject.toml`. CLI flags in the hook command override configuration values.

```toml
[tool.docwrap]
incremental = "definition" # "file", "definition", "docstring", or false
allow-on-failure = false   # true to warn instead of denying on failure
fill-style = "smart"       # "smart", "greedy", or "balanced"
smart-slack = 8            # max characters smart fill may back off
reflow = "cautious"        # "none", "cautious", or "full"
```

`fill-style`, `smart-slack`, and `reflow` are transformation settings and fields on public
`DocwrapConfig`. `incremental` and `allow-on-failure` are hook operational settings outside that
record. The other operational settings are `color`, `show-skips`, and `stats`.

Setting `incremental` to a level string enables incremental mode at that level. Setting it to
`false` explicitly disables incremental mode. Setting `allow-on-failure = true` is equivalent to
`docwrap hook --agent NAME --allow-on-failure`.

The hook respects the `fill-style`, `smart-slack`, and `reflow` settings from `pyproject.toml`. It
uses the same config resolution as the CLI, so those values apply to hook-triggered wrapping. No
separate hook-specific configuration is needed.

## Agent selection

`-a NAME` is the short spelling of `--agent NAME`; `-e NAME` is the short spelling of
`--event NAME`.

`--agent NAME` selects a agent adapter from the `docwrap.hook_adapters` entry-point group. The
adapter translates the host payload into docwrap's neutral input and translates docwrap's decision
into the host's response. Classification, Git operations, incremental eligibility, wrapping, and
the allow or deny decision never depend on the adapter, and an adapter cannot change a decision.

- Claude Code (`claude`) and Codex (`codex`) payloads carry `hook_event_name`; `PreToolUse` with
  `tool_name` `Bash` supplies the command and `PostToolUse` with `Edit`, `Write`, or `apply_patch`
  supplies the edited paths.
- Antigravity (`agy`) payloads carry no event name, so `--event PreToolUse` or
  `--event PostToolUse` is required. `run_command` supplies the command; `write_to_file` and
  `replace_file_content` supply the edited path.
- Any other event or tool is nothing docwrap acts on: silent exit 0.

An unknown host, a host registered twice, an adapter that fails to import or reports a different
`docwrap.hooks.API_VERSION`, a missing or unexpected `--event`, an adapter that raises while
parsing, or an adapter whose `render` and `render_failure` both raise is the last resort: one
`docwrap:` line on stderr, nothing on stdout, exit 2. See Host adapters below for what exit 2 means
on each host.

## PreToolUse

Every pre-tool shell command receives exactly one of three outcomes:

1. **Unrelated.** No commit invocation is detected. Silent allow, no repository work.
2. **Supported commit.** The complete command belongs to the closed grammar below. Docwrap formats
   the content the commit will consume, then allows. A grammar-accepted command that is proven not
   to commit, a status-only commit or a routing `cd` whose target does not exist, allows with no
   repository work.
3. **Unsupported commit.** A commit invocation is detected and the command is outside the grammar.
   Docwrap denies before any repository discovery, add replay, index read, index write, or
   working-tree write, with a reason that names the first unsupported element and the nearest
   accepted form, for example
   `docwrap: unsupported commit command: segment 'git push'; supported form: [cd DIR &&] [git add ...;] git commit ...`.

An accepted committing shape whose execution directory is unknown and whose route is absent or
relative is an unsupported commit. That denial runs before hook enablement, configuration reads,
and `--allow-on-failure`, so neither `docwrap.hook=false` nor the infrastructure-failure option can
convert it into an allow.

Silent passthrough for a detected commit does not exist. The grammar runs first; an accepted
command's message content is never inspected. Otherwise the detector runs.

### Skip conditions

The hook exits 0 with no output when:

- JSON on stdin is invalid or not an object.
- The payload describes another event or a tool that is not the shell.
- The command is unrelated: no commit invocation is detected.
- The command is a supported status-only commit (`--dry-run`, `--short`, `--porcelain`, `--long`,
  `--null`, or `-z`) or routes through a `cd` whose target does not exist.
- The hook is disabled in the target repository via git config for a supported commit. Disabling
  never suppresses the deny outcome: an unsupported command has no trusted route, so reading the
  payload repository's setting would let a disabled outer repository exempt a commit aimed at an
  enabled one.

When processing proceeds the steps are unchanged: replay the supported `git add`, determine the
commit mode, discover the selected files, read them from the index or working tree, wrap, write
back, and sync working-tree copies when authorized. Safety skips emit informational `docwrap:`
lines on stderr.

### Command detection

Commands are parsed into a structural tree by a POSIX shell parser that never executes, expands, or
predicts. The grammar and the detector both consume that tree; no text is rewritten before parsing.

For a real commit, the grammar accepts this sequence, where `SEP` is `&&`, `;`, or a line feed:

```text
[cd LITERAL &&] [GIT_ADD SEP] GIT_COMMIT
```

- Every word is literal: single quotes, double quotes without `$`, backticks, or backslash escapes,
  and backslash escapes outside quotes all produce literal text. A quoted `;` is a word, never a
  separator. A backslash before a line feed continues the line. A carriage return is rejected.
- A routing `cd` must be followed by `&&`, and its operand must be absolute or begin with `./` or
  `../`. Repository routing is uniform: one `cd`, or the same `-C ROUTE` on every git segment,
  never both.
- Standalone `git status`, `git diff`, and `git log` segments before a real commit are denied,
  whether before or after an optional add. Hook staging and wrapping would run before those
  observations and could change their results or mutate files before a failing check.
- Existing proven no-commit forms (status-only commits and missing route targets) still allow
  read-only segments before the final commit with no pathspecs or positional arguments and only
  these options: `status` accepts `-s`, `-b`, any nonempty short-option cluster containing only `s`
  and `b` (including repeated letters), `--short`, `--porcelain`, `--porcelain=v1`, `--branch`,
  `--untracked-files=all`, and `-l`; `diff` accepts `--cached`, `--staged`, `--check`, `--stat`,
  `--name-only`, and `--name-status`; `log` accepts `--oneline`, `--stat`, `--format=LITERAL`,
  `-N`, and `-n N`. `--output` and `--no-index` are excluded because a later add could stage what
  they produce.
- At most one `git add [--] PATHSPEC...` or
  `git add (-A | --all | -u | --update) [--] [PATHSPEC...]`.
- Exactly one final `git commit`. Its first word may be `GIT_COMMITTER_DATE=VALUE` with an unquoted
  name, where `VALUE` is literal text in any quoting or exactly `"$(git log -1 --format=%aI)"`.
- The `-m` or `--message` value may be the heredoc message idiom, at most once:

```text
git commit -m "$(cat <<'EOF'
subject

body
EOF
)"
```

The outer double quotes and a quoted heredoc delimiter are part of the idiom. The delimiter name is
arbitrary: after quote removal, it must exactly match the closing line; `EOF` is only a familiar
example. An unquoted delimiter could expand shell syntax in the message body, while an unquoted
`$(cat ...)` is word-split by the shell and could add selection options to Git's argv, so both
forms are unsupported.

The commit option inventory is closed. Selection options: `-a`, `--all`, `--no-all`, `-i`,
`--include`, `--no-include`, `-o`, `--only`, `--no-only`, `--`, `--pathspec-from-file FILE`,
`--pathspec-from-file=FILE`, `--pathspec-file-nul`, and positional pathspecs; last option wins, and
`--all` with a pathspec, `--include` with `--only`, `--include` or `--only` without a pathspec, and
`--pathspec-from-file` with positional pathspecs are unsupported. Value-taking metadata options:
`-m`, `--message`, `-F`, `--file`, `--author`, `--date`, `-c`, `--reedit-message`, `-C`,
`--reuse-message`, `--fixup` (except a value beginning with `reword:`), `--squash`, `--trailer`,
`--cleanup`, `-t`, `--template`, `--unified`, `--inter-hunk-context`. Boolean metadata options:
`--amend`, `-e`, `--edit`, `--no-edit`, `--reset-author`, `-s`, `--signoff`, `--no-signoff`, `-n`,
`--no-verify`, `--verify`, `--no-post-rewrite`, `-q`, `--quiet`, `--no-quiet`, `-v`, `--verbose`,
`--no-verbose`, `--allow-empty`, `--no-allow-empty`, `--allow-empty-message`,
`--no-allow-empty-message`, `--status`, `--no-status`, `--branch`, `--no-branch`, `--ahead-behind`,
`--no-ahead-behind`, `-S`, `-SKEY`, `--gpg-sign`, `--gpg-sign=KEY`, `--no-gpg-sign`, `-u`,
`-uMODE`, `--untracked-files`, `--untracked-files=MODE`, `--no-untracked-files`. Short boolean
options may be clustered. `--no-patch` and `--no-interactive` are accepted and turn interactive
mode off. Status-only options `--dry-run`, `--short`, `--porcelain`, `--long`, `-z`, and `--null`
are accepted and proven noncommitting. Everything else, including `-p`, `--patch`, `--interactive`,
`-h`, `--help`, `--pathspec-from-file=-`, every abbreviation, and every unlisted `--no-` spelling,
is unsupported.

The detector decides whether a rejected command contains a commit invocation that the shell, a
listed evaluator, or a listed interpreter executes: a word naming Git by basename (`git` or
`git.exe` in any letter case after the last `/` or `\`) followed later in the same segment by a
`commit` word; a word carrying an expansion followed later in the same segment by a `commit` word;
every command substitution, including inside double quotes and inside `${...}` expansions; every
heredoc body, scanned once for the substitutions an unquoted delimiter expands, once as a shell
script, and once as a quote-stripped token stream for a `git` token followed by a `commit` token;
the script arguments of `sh`, `bash`, `zsh`, `dash`, or `ksh` after a `-c` cluster; the joined
arguments of `eval`; and the inline script after `-c` or `-e` of `node`, `perl`, `ruby`, or any
`python*`, scanned as the same token stream. Each distinct evaluator or heredoc script is parsed
and scanned at most once per detector call; the results are discarded when the call ends. ANSI-C
quoting (`$'...'`) is unsupported: unescaped contents retain their literal word text for detection,
including concatenated parts such as `com$'mit'`; contents with backslash escapes remain opaque
like dynamic expansions. The detector does not evaluate escapes or predict computed subcommands
such as `git $ACTION`. A parse that needed recovery (unsupported ANSI-C quoting, unbalanced quotes,
an unterminated substitution or heredoc, nesting deeper than eight substitutions) is never accepted
and is always classified by the detector. Only the bare spelling `git` can be accepted; a
path-qualified or `.exe` spelling denies with the element `executable '<spelling>'` and the hint
`use the bare executable name git; PATH resolves the installed git`.

When an accepted commit's staged-index wrap rewrites at least one index entry, the allow response
carries the reason `docwrap: wrapped N staged Python file for this commit` (or `files` for more
than one), which the host shows as hook feedback. Working-tree wrapping paths and accepted commits
that rewrite nothing stay silent.

Detected commands outside the grammar are denied. Examples:

```text
make check && git add -A && git commit -m "message"
git init && git add -A && git commit -m "message"
git add first.py && git add second.py && git commit -m "message"
git add *.py && git commit -m "message"
FOO=1 git commit -m "message"
git --no-pager commit -m "message"
git commit -m "message" && git push
git commit --dry-run --no-dry-run
sh -c 'git commit -m "message"'
```

Commands where commit text is only data to another program, such as `grep 'git commit' file`, are
unrelated and allowed. Inline interpreter scripts are scanned conservatively, so
`python -c "print('git commit')"` is denied. A wrapper script that commits without commit text in
the outer command is outside the guarantee.

### Checks before commits

Run checks and observations in a separate tool call, then submit the add/commit command only after
success. For example, first run `git diff --check`. After correcting any whitespace errors and
obtaining a successful check, submit `git add sample.py && git commit -m update` in a later call.
Separate lines within one tool call still reach the hook together and do not fix the ordering
problem.

If the check must follow staging, run `git add sample.py && git diff --cached --check` in a call
without a real commit. Submit `git commit -m update` separately after success. A later commit hook
may still format staged content, so the earlier check does not certify the final formatted bytes.

The denial names the overtaken command, such as `read-only git diff before git commit`, and gives
this separate-call remedy. It applies even with `docwrap.hook=false` or `--allow-on-failure`.
Standalone observations and add/check calls without a commit remain unrelated and pass through.
Status-only commits and missing routes retain their existing no-mutation paths. A routing `cd` and
the supported amend-date substitution are not standalone observations.

### Cross-repo commands

When an agent session starts in one repository but the commit should run in another, put one
literal route in the command. The hook recognizes these forms:

```text
git -C ../other-repo commit -m "message"
cd ../other-repo && git commit -m "message"
git -C ../other-repo add target.py && git -C ../other-repo commit -m "message"
```

`git -C <repo> commit ...` is the clearest form for agent-driven cross-repo commits because the
target repository is visible in the command text.

For the `cd` form, the literal operand must be absolute or begin with `./` or `../`, and `cd` must
be the first segment. A compound `git -C` form must use one `-C` with the same literal operand on
every Git segment. Mixing `cd` and `git -C`, using cumulative `-C` options, or spelling the route
differently between segments makes the complete command unsupported, and it is denied.

A bare `git commit` is accepted only when the adapter establishes the command's execution
directory. Claude Code reports that directory as the payload `cwd`; Antigravity reports it as the
`Cwd` argument of `run_command`. Codex reports only the session directory, which can differ from
where a shell call with its own `workdir` executes, so the Codex adapter reports the directory as
unknown. With an unknown directory, a bare commit or a relative route is denied before any
repository access. For a bare commit, the reason is
`docwrap: unsupported commit command: unknown working directory; supported form: git -C /absolute/repo commit ... or cd /absolute/repo && git commit ...`.
Retry with the absolute route; docwrap never substitutes the session, process, or repository
directory and never chooses a repository for you.

### Compound bootstrap commands

A compound bootstrap such as `git init && git add -A && git commit` contains a commit invocation
outside the grammar and is denied. Run `git init`, `git add`, and `git commit` as separate tool
calls so the final standalone commit reaches PreToolUse after the repository and index exist.

This rule affects only PreToolUse. PostToolUse continues to process files after supported edit
tools change them.

### Targets outside a Git worktree

After accepting a supported commit command, docwrap checks the command's resolved target directory.
A confirmed non-worktree target allows with exit 0 and no output. Git therefore owns the error for
a bare `git commit` outside a repository.

Repository-discovery failures remain fail-closed; only confirmed worktree absence is a silent
allow.

### Commit modes

The hook detects which git commit mode is in use and dispatches to the appropriate wrap path:

| Mode               | Flags                                | Wrap path                                           |
| ------------------ | ------------------------------------ | --------------------------------------------------- |
| Normal             | (no mode flags)                      | Index                                               |
| Auto-stage         | `-a`, `--all`                        | Working tree                                        |
| Include            | `-i`, `--include`                    | Working tree (pathspecs) + index (remaining staged) |
| Only / pathspec    | `-o`, `--only`, or bare `<pathspec>` | Working tree                                        |
| Pathspec from file | `--pathspec-from-file=<file>`        | Working tree                                        |
| Status only        | `--dry-run`, status format flags     | No repository work (allowed)                        |

For auto-stage and pathspec modes, the hook wraps files on the working tree so git stages the
wrapped content naturally. Interactive mode (`-p`, `--patch`, or `--interactive`),
`--pathspec-from-file=-`, and unlisted options are unsupported and denied. `--no-patch` and
`--no-interactive` are accepted and use the effective mode selected by the rest of the command.
Status-only options are accepted no-ops; their unlisted negations make the command unsupported.

### File discovery

Staged `.py` files are discovered via `git diff --cached --raw -z -M` (index path). The worktree
path discovers modified tracked files with
`git diff -z --no-relative --no-renames --diff-filter=d --name-only HEAD`, covering staged plus
unstaged changes against `HEAD`, or against the empty tree on an initial commit. Deleted files are
excluded. Only files with regular modes (`100644`, `100755`) are processed; symlinks and other
special modes are silently skipped. The `--raw -M` variant also detects renames for incremental
mode.

### Index path

The hook reads staged content from the git index via `git cat-file --batch` (batching all blobs in
a single subprocess), wraps docstrings in memory, creates new blobs with
`git hash-object -w --stdin`, and atomically updates the index via `git update-index --index-info`.
File modes (including executable bits) are preserved during the index update.

### Working tree path

For auto-stage, pathspec-only, and include modes, the hook reads file contents from the working
tree, wraps docstrings in memory, and writes the result back to the working tree. Git then stages
the modified content as part of the commit.

### Working tree sync

After updating the index for a normal commit, the hook copies a wrapped index result to the working
tree only when newline-normalized working-tree text equals the staged source that was wrapped. LF,
CRLF, and CR spellings therefore do not prevent synchronization. The write follows docwrap's
supported newline policy: any input containing CRLF uses CRLF output; otherwise output uses LF, so
lone-CR input is normalized to LF. Any other content difference leaves the working tree untouched,
preserving partial staging.

Preceding `git add` commands are replayed so the hook can inspect the content the commit would use,
but neither replay nor Git's path selection authorizes a working-tree write. An ordinary unfiltered
file added immediately before a commit normally satisfies the equality check and is synchronized.

Git attributes can make the staged and working-tree representations differ even without a user
edit. For example, an `ident` attribute stores `$Id$` in the index while expanding it in the
working tree. Docwrap still wraps the index entry, but the representation difference prevents a
working-tree write. When the original compound command later runs its real `git add`, Git may
replace docwrap's wrapped index entry with the clean-side form of the unchanged, unwrapped
working-tree file, leaving that file unwrapped for the commit. This safe no-op is preferred to
overwriting a representation docwrap cannot prove equivalent.

**Full-file mode** is enabled by default in PreToolUse. The hook wraps every docstring in each
selected staged or working-tree file. Pass `-i`/`--incremental` or set
`[tool.docwrap].incremental = "definition"` to wrap only changed definitions. See
[Incremental defaults](#incremental-defaults) for the resolution chain.

### Infrastructure failures

Incremental diff collection or parsing failures are a deliberate exception: selected staged or
working-tree files fall back to full wrapping, even without `--allow-on-failure`. Missing staged
objects and staged contents that cannot be decoded are skipped. Other Git infrastructure failures,
including malformed or truncated staged-content output and absent per-path results, deny the commit
by default. Add replay failures, blob-creation failures, and index-update failures also deny by
default. Pass `--allow-on-failure` to warn on stderr, return silent success, and leave
authorization to the provider.

### Syntax errors

Staged files with Python syntax errors are silently skipped. They do not block the commit.

### Successful completion (exit 0)

Every successful PreToolUse path exits 0. Unrelated commands, no-op wraps, and working-tree
rewrites are silent. When a normal staged commit rewrites at least one index entry, the host
adapter emits a `systemMessage` with the count of staged Python files wrapped. Each safety skip
also emits an informational `docwrap:` line on stderr. The neutral result's `decision='allow'`
means only that docwrap did not block the command; it does not grant provider permission.

## Security considerations

Docwrap never authorizes the intercepted command. Claude Code, Codex, and Antigravity receive no
permission-decision stdout on success, so each provider applies the user's normal permission prompt
and allowlist rules. Informational safety-skip stderr does not change that behavior.

| Outcome                                   | Claude Code                                   | Codex                                         | Antigravity                                   |
| ----------------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Unrelated, no-op, or working-tree rewrite | Exit 0, silent                                | Exit 0, silent                                | Exit 0, silent                                |
| Staged-index rewrite                      | `systemMessage` with file count, exit 0       | `systemMessage` with file count, exit 0       | `systemMessage` with file count, exit 0       |
| Safety skip                               | Info on stderr, exit 0                        | Info on stderr, exit 0                        | Info on stderr, exit 0                        |
| Unsupported commit                        | Deny JSON on stdout, reason on stderr, exit 0 | Deny JSON on stdout, reason on stderr, exit 0 | Top-level deny JSON, reason on stderr, exit 0 |
| Failure, fail-closed                      | Deny JSON on stdout, reason on stderr, exit 0 | Deny JSON on stdout, reason on stderr, exit 0 | Top-level deny JSON, reason on stderr, exit 0 |
| Failure, allow-on-failure                 | Warning on stderr, exit 0                     | Warning on stderr, exit 0                     | Warning on stderr, exit 0                     |
| Last resort (no functioning adapter)      | One line on stderr, exit 2: blocks            | One line on stderr, exit 2: blocks            | One line on stderr, exit 2: does not block    |

Hook mutations happen before the provider's permission decision and may remain if that later
decision rejects the commit. For normal commits, docwrap may already have rewritten staged blobs
and synchronized safe working-tree copies. For the one supported compound add, such as
`git add foo.py && git commit`, docwrap pre-executes that add so it can inspect the content the
commit would use; the staging change may remain if later wrapping, provider approval, or command
execution fails. Docwrap cannot observe or roll back those outcomes. Add replay is not
transactional.

Run `git config docwrap.hook false` in a repository to opt out of hook processing and its index or
working-tree mutations.

Opting out disables formatting and its mutations for supported commits in that repository. It does
not suppress the deny outcome for unsupported commands, which are denied before any repository is
consulted.

## PostToolUse

The PostToolUse handler fires after Claude Code Edit/Write tool invocations, Codex `apply_patch`
file edits, or Antigravity `write_to_file` and `replace_file_content` edits. It:

1. Extracts touched `.py` paths from the event JSON. Claude Code provides one
   `tool_input.file_path`; Codex provides an `apply_patch` command, from which docwrap extracts
   each touched path; Antigravity provides `toolCall.args.TargetFile`.
2. Reads each file while retaining its source encoding and original bytes for newline-preserving
   writes.
3. Resolves configuration from `pyproject.toml` (line length, convention).
4. Wraps every docstring in each selected file by default. Explicit incremental mode limits
   wrapping to eligible changed content.
5. Prints structured safety skips as informational `docwrap:` lines on stderr.
6. Writes the report's source back to the file if any docstrings changed, preserving the newline
   style and encoding.

Exceptions during file processing are caught and logged to stderr. Expected errors (`OSError`,
`UnicodeDecodeError`, `SyntaxError`, `UnknownStyleError`) print a one-line message; unexpected
exceptions print a full traceback. The PostToolUse handler never blocks Claude Code (always returns
0).

### Skip conditions (PostToolUse)

The hook exits 0 with no output when:

- Hook is disabled via git config.
- JSON on stdin is invalid or missing expected fields.
- `tool_name` is not Edit, Write, or apply_patch, and the payload is not a supported Antigravity
  edit tool.
- No touched path ends in `.py`.
- File does not exist on disk.

## Incremental defaults

The hook and CLI default to full-file mode. Incremental behavior is opt-in.

| Context     | Default behavior | Override                    |
| ----------- | ---------------- | --------------------------- |
| CLI         | Full mode        | `-i`, `--incremental-level` |
| PreToolUse  | Full mode        | `-i`, `--no-incremental`    |
| PostToolUse | Full mode        | `-i`, `--no-incremental`    |

The `[tool.docwrap].incremental` config value overrides the default in all contexts:

```toml
[tool.docwrap]
incremental = "definition" # "file", "definition", "docstring", or false
```

Setting `incremental = false` disables incremental mode everywhere (CLI, PreToolUse, PostToolUse).
CLI flags (`-i`, `--incremental-level`, `--no-incremental`) still override the config value.

Incremental diff collection preserves rename pairing and processes only the selected files.
PostToolUse treats an explicitly selected untracked file, including a Git-ignored file, as fully
eligible. Tracked files with no changed hunks remain untouched. See the
[incremental guide](incremental.md) for level-specific selection and CLI directory discovery.

## Composing with other hooks

Agent runtimes may run matching hooks concurrently. For Claude Code and Codex, when two hooks can
rewrite the same staged file, register one sequencer command for the event and matcher, with each
formatter as an ordered step. A Codex manual hook can invoke:

```sh
sequential-hooks --agent codex --step-timeout 30 defsort hook --agent codex -- docwrap hook --agent codex
```

For Claude Code, use `claude` in all three agent selections. The sequencer passes the same payload
to each step and applies each host response before moving to the next one. In particular, a
formatter deny is JSON on stdout with exit 0; the sequencer must honor it. Do not also register
standalone formatter plugins for that event, or the formatters may run twice.

**Antigravity limitation:** The current `agy` sequencer adapter rejects a successful formatter
`systemMessage` as an unexpected response and stops with a deny. A staged rewrite by docwrap or
defsort can produce that message. Do not compose these formatters through
`sequential-hooks --agent agy` in this state. Two standalone formatter hooks may run concurrently
and race over the same staged files, so they are not a substitute for ordered composition. Do not
weaken deny handling to conceal the incompatibility.

Docwrap cannot roll back a staged rewrite if a later hook fails. The
[security considerations](#security-considerations) describe other mutations that may remain after
a provider denies the command.

## Exit codes

| Code | Meaning                                                             |
| ---- | ------------------------------------------------------------------- |
| 0    | Every rendered response: allow, deny, warning, PostToolUse          |
| 2    | Last resort: no functioning agent adapter, one diagnostic on stderr |

A deny is carried by the host's JSON on stdout, not by the exit status. Exit 2 blocks on Claude
Code and Codex, which document exit 2 as blocking; Antigravity has no exit-code tier, so the last
resort does not block there. When docwrap runs under a wrapper, the wrapper's own agent adapter
decides what a child's nonzero exit means; that protection is the wrapper's contract, not
docwrap's.

## Error handling

- Invalid JSON on stdin causes a silent exit (exit 0, no output).
- A supported commit in a repository with the hook disabled via git config causes a silent exit.
- An unsupported commit and every fail-closed infrastructure failure produce the host's deny JSON
  on stdout and the reason line on stderr with exit 0.
- `--allow-on-failure` turns infrastructure failures after a command is accepted into a warning on
  stderr and a silent allow. It never applies to unsupported commits or to the last resort.
- PostToolUse reports errors on stderr and always returns 0.

## Host adapters

Docwrap reaches each host through an adapter that implements the public `docwrap.hooks` contract
(see the [library reference](library.md)). Adapters are discovered through the
`docwrap.hook_adapters` entry-point group; the built-in `claude`, `codex`, and `agy` adapters are
registered in that group like any other. An adapter translates: it turns the host payload into
neutral input (`event`, `command`, `cwd`, `command_cwd`, `edited_paths`) and turns docwrap's
neutral result (`decision`, `reason`, `notes`) into the host response (`stdout`, `stderr`,
`exit_code`). It cannot change a decision.

`command_cwd` is the absolute directory a pre-tool command executes in, or `None` when the host
contract does not establish it. An adapter must never derive it from the process directory, a
workspace root, or the session directory; `docwrap.hooks.absolute_directory` returns a payload
field only when it is a non-empty absolute path. An empty, relative, or non-string `command_cwd` is
an adapter failure that exits 2 without repository access. Post-tool input sets `command_cwd=None`
and uses `cwd` as event context.

Built-in adapter source, tests, registration, and native manifests live together in each
`plugins/<agent>` directory. The build collects every `adapter.toml` manifest and bundles all of
those runtime packages in the main `docwrap` wheel. Adding another bundled host therefore adds one
plugin directory; users do not install a separate Python adapter package.

For example, a new bundled adapter at `plugins/example/_docwrap/_adapter.py` registers itself in
`plugins/example/adapter.toml`:

```toml
[entry-points."docwrap.hook_adapters"]
example = "plugins.example._docwrap._adapter:ExampleAdapter"
```

To add a separately distributed third-party host, package a class that satisfies
`docwrap.hooks.HookAdapter`, set `api_version` to `docwrap.hooks.API_VERSION`, and register it:

```toml
[project.entry-points."docwrap.hook_adapters"]
myhost = "docwrap_host_myhost.adapter:MyHostAdapter"
```

Install the adapter package into the same Python environment as docwrap, then register the hook
command as `docwrap hook --agent myhost`. A host whose payload carries no event name sets
`requires_event` and receives `--event`. Docwrap refuses an adapter whose `api_version` differs
from its own and a name registered by two distributions.

Failure tiers:

1. Every deny, including infrastructure failures under fail-closed configuration, is rendered by
   the selected adapter's `render` or `render_failure`. The adapter encodes the host's real
   enforcement protocol; the core never chooses an exit code.
2. When no functioning adapter exists (unknown or duplicate name, import failure, wrong
   `api_version`, `parse` raising, or both `render` and `render_failure` raising), docwrap writes
   one diagnostic line to stderr and exits 2 with nothing on stdout. Exit 2 blocks on Claude Code
   and Codex and does not block on Antigravity. Docwrap cannot protect a chain from the failure of
   its own adapter.

## Differences from CLI

| Feature           | CLI                     | PreToolUse                  | PostToolUse                                                                                                     |
| ----------------- | ----------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Line length       | `--length` flag         | From `pyproject.toml` only  | From `pyproject.toml` only                                                                                      |
| Style override    | `--style` flag          | From `pyproject.toml` only  | From `pyproject.toml` only                                                                                      |
| Write mode        | `--write` flag required | Always writes (to index)    | Always writes                                                                                                   |
| Incremental       | Off by default          | Off by default              | Off by default                                                                                                  |
| Diff output       | `--diff` flag           | Not supported               | Not supported                                                                                                   |
| Report output     | Summary table           | Not supported               | Not supported                                                                                                   |
| Error reporting   | stderr + exit code 2    | Deny or warn (configurable) | stderr (always returns 0)                                                                                       |
| Scope             | Explicit paths/dirs     | Staged `.py` files          | One path for Claude Code/Antigravity; one or more deduplicated touched paths from a Codex `apply_patch` command |
| Directory recurse | Yes                     | No                          | No                                                                                                              |
| Encoding          | Preserved               | Preserved                   | Preserved                                                                                                       |
| Newline style     | Preserved               | Preserved                   | Preserved                                                                                                       |
| Fill style        | `--fill-style` flag     | From `pyproject.toml` only  | From `pyproject.toml` only                                                                                      |
| Smart slack       | `--smart-slack` flag    | From `pyproject.toml` only  | From `pyproject.toml` only                                                                                      |
| Reflow            | `--reflow` flag         | From `pyproject.toml` only  | From `pyproject.toml` only                                                                                      |
| Convert/strip     | Supported               | Disabled                    | Disabled                                                                                                        |

For rewritten files, newline preservation means selecting CRLF when the source contains CRLF,
otherwise LF. Mixed line endings are normalized to that style. See
[files and encodings](platform.md#files-and-encodings).

## Safety skips

Both hook paths report safety-skipped docstrings as informational stderr lines prefixed with
`docwrap:`; the operation is always allowed. For staged commits, line numbers refer to the staged
blob. Partial staging and index/worktree synchronization are unaffected.
