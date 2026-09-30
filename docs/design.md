# Concepts and safety

Docwrap makes docstring formatting a repeatable project policy. As people and coding agents write
and edit docstrings, it applies configured rules for wrapping and layout while preserving
recognized structure. It supports Google, NumPy, and Sphinx/reST conventions. Use the
[CLI reference](cli.md) to choose an output mode, the [library reference](library.md) to call it
from Python, or the [agent hook guide](hook.md) to run it around edits and commits.

## What docwrap changes

Docwrap distinguishes a summary, prose paragraphs, structured entries such as arguments and
returns, lists, and text that must stay verbatim. It wraps prose and descriptions to the configured
width with the indentation required by the detected style. Section headings, code examples,
doctests, literal blocks, tables, and aligned code-labelled listings retain their layout. The
[CLI statistics view](cli.md#stats-mode---stats) can help you choose a width from the existing
source and docstrings.

A file can contain different docstring styles. Docwrap detects each docstring independently, and
`style = "google"` (or `numpy` or `sphinx`) can supply a convention when detection is uncertain.
When style markers tie, ordinary wrapping leaves the style undetected and wraps only content that
does not require a style choice. Explicit [style conversion](cli.md#style-conversion) has different
tie handling and may keep unsupported sections in their original style.

## Choosing a wrapping policy

Automatic formatting trades control over individual line breaks for consistent rules. Reflow can
replace deliberate breaks, and a small wording change can alter several wrapped lines. A chosen
policy makes formatting repeatable; it does not guarantee better prose or smaller diffs.

For the same input and configuration, docwrap produces the same result. Running it again on that
result leaves it unchanged. These properties do not require differently formatted versions of the
same prose to produce identical output: some policies retain authored line breaks and summary
placement. Ordinary wrapping also retains detected docstring styles; conversion is an explicit
choice.

The default is a 79-column limit, `smart` fill with a slack of 8, `cautious` reflow, and unchanged
summary wrapping. You can change those settings in `[tool.docwrap]` or with CLI flags. For most
projects, start by changing only `line-length`:

```toml
[tool.docwrap]
line-length = 88
```

`smart` fill follows greedy word placement and repairs short parenthesized or bracketed fragments
that would otherwise be stranded at line ends. `greedy` favors packed lines. `balanced` minimizes
raggedness. The shared fill rules preserve meaningful spacing where possible and make repeated runs
stable. See [fill style options](cli.md#--fill-style-smartgreedybalanced) for the precise
`smart-slack` control.

`cautious` reflow joins lines that look like a continuation while preserving sentence boundaries,
colons, and bullets. `none` only splits lines that exceed the width. `full` joins and reflows all
eligible lines. Choose `none` when authored line breaks carry meaning; inspect a diff before using
`full` across a project. The [reflow options](cli.md#--reflow-nonecautiousfull) give the full
contract.

By default, the opening summary stays on its original line and is not wrapped. A multiline summary
can begin beside the opening quotes (D212) or on the following line (D213). If you enable
`wrap-summary`, docwrap accounts for the opening quotes when measuring the first form.
`summary-start` can choose a position when a multiline layout is needed. A docstring that fits
entirely on one line stays on one line before that preference applies.

## Preservation and safety

Docwrap uses Python syntax to locate docstrings, leaving ordinary strings and code alone. The
source-string API preserves untouched text and final line terminators. File writes retain the
detected encoding but select one newline style for rewritten files, as described under
[platform compatibility](platform.md). When docwrap rewrites a docstring, it keeps the literal's
raw spelling as far as the requested transformation permits:

- Raw docstrings treat backslashes as literal characters. Normal docstrings retain the spelling of
  each escape sequence; wrapping does not split one. Backslash-newline continuations and implicit
  literal concatenation cause that docstring to be skipped.
- Code blocks, doctests, literal blocks, simple tables, and aligned code-labelled listings are
  protected from prose wrapping.
- If new content cannot be quoted safely, docwrap skips that docstring. It validates rewritten
  literals and the complete Python file; a validation failure reverts the affected change.
- Explicit conversion or type stripping may remove text, including escapes in that text.
  Unsupported sections retain their original style where they can be preserved. A conversion that
  would lose an authored return or yield name skips the whole docstring.
- Running docwrap again on its output produces the same result. This idempotence applies across the
  supported fill and reflow policies.

Safety skips leave the affected docstring unchanged and produce diagnostics without changing the
exit status. See [CLI safety warnings](cli.md#safety-skip-warnings) and the
[library skip records](library.md#safety-skips) for their machine-readable reasons.

## File and Git boundaries

For file operations, docwrap reads Python's declared encoding, normalizes newlines while
processing, and writes in the original encoding and selected newline style. See
[platform compatibility](platform.md) for the observable file behavior. A file write is based on
the bytes read for that operation; it does not merge a concurrent edit.

Full-file wrapping is the default for the CLI and hooks. [Incremental mode](incremental.md) is
opt-in and can limit processing to changed files, definitions, or docstrings relative to Git HEAD.
At normal staged commits, the [PreToolUse hook](hook.md#pretooluse) formats staged content and only
synchronizes a working-tree copy when it is safe to do so. This preserves unstaged edits during a
partially staged commit.

## Known limits

- Only triple-quoted docstrings are wrapped. Single-quoted docstrings are left unchanged.
- Inline docstrings such as `def f(): """..."""` have limited indentation handling; put the
  docstring on its own line before formatting.
- When both triple-quote delimiters would collide with rewritten content, docwrap skips the
  docstring.
- Style detection ties without an explicit conversion target receive style-independent wrapping
  only.
