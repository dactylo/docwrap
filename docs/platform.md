# Platform compatibility

docwrap supports CPython 3.13 or later on Linux, macOS, native Windows with Git Bash, and Windows
with WSL. Git is required for incremental mode and agent hooks. Git Bash uses native Windows
Python, so filesystem paths and text-mode newline behavior follow Windows rules even though the
shell looks Unix-like.

docwrap resolves Git only through absolute directories in `PATH`, in their configured order.
Explicit absolute entries are trusted, including wrappers and Windows `PATHEXT` extensions. Empty
and relative entries are ignored, and the current directory is not searched implicitly. Each Git
subprocess receives the resolved absolute executable path. If no Git qualifies, directory scans
inside a worktree warn and keep every discovered Python file. A Git failure during such a scan has
the same behavior. Outside a worktree, scans use the filesystem silently. Git-dependent operations
follow their existing infrastructure-failure policy.

## Files and encodings

docwrap reads Python files as bytes, detects the encoding from a PEP 263 coding cookie or UTF-8
BOM, and decodes CRLF and lone CR line endings to LF for processing. When a file changes, it writes
bytes in the detected encoding. A source file with any CRLF sequence is written with CRLF; one
without CRLF is written with LF. Untouched files retain their original bytes. A mixed-newline file
therefore receives one selected style when rewritten.

The CLI, file library API, and working-tree hook paths use this policy. Staged hook content is
encoded back into a Git blob with the staged blob's selected newline style. Git receives the bytes
directly, without text-mode newline conversion. A normal staged commit changes the index first; the
hook updates a working-tree copy only when its newline-normalized content matched the original
staged content and its byte snapshot has not changed since that check. See
[partial staging](hook.md#working-tree-sync).

A file write uses the source bytes captured for that operation to choose the encoding and newline
style. It does not merge or reject an unrelated concurrent edit made after that read. Writes to an
existing path use ordinary filesystem permissions and can fail when access is denied. docwrap
reports CLI or library I/O failures; the [hook failure policy](hook.md#infrastructure-failures)
controls commit-time handling. If writing the new bytes fails, docwrap attempts to write the
original bytes back before reporting the error. If restoration also fails, the error says the
original bytes could not be restored. The file writer does not promise an atomic replace or
permission repair.

## Paths

Directory scans recurse through Python files while pruning common non-source directories and
skipping symlinks found during recursion. An explicitly named Python symlink can be processed. When
Git is available, directory scans honor Git exclusions; an explicitly named Python file remains
eligible even if Git ignores it. See [CLI file selection](cli.md#synopsis).

Git paths use byte-safe filesystem encoding and NUL-delimited records, so POSIX filenames with
non-UTF-8 bytes are not replaced during Git path parsing. Paths sent to Git use forward slashes.
Incremental mode collects the repository diff so Git can pair rename sources and destinations; only
selected files are processed.

Git Bash paths such as `/c/Users/example/repo` need special care with Windows Python. Host payload
paths, literal `cd` routes, and attached `--pathspec-from-file=/c/Users/example/paths` are
normalized at their supported boundaries. A standalone mount path inside a Git command, such as
`git -C /c/Users/example/repo commit`, is unsupported and a detected commit is denied with the
element `MSYS-style word in Windows git segment '<segment>'`. Use the native Windows spelling, such
as an absolute Windows path `git -C C:/Users/example/repo commit`, or a literal `cd` route such as
`cd /c/Users/example/repo && git commit`. Relative routes require a host that supplies the
command's working directory; the Codex adapter does not, so it denies them. Paths outside the
single-drive mount form are passed through unchanged.

For `--pathspec-from-file` in newline mode, a trailing CR is removed from each path as Git does.
NUL mode keeps raw path bytes.

## Console and environment

Human-readable file labels use forward slashes on every platform, including Windows under Git Bash
or PowerShell. Drive letters and relative paths retain their meaning. Special Windows namespace
paths keep their original spelling. Literal backslashes in POSIX filenames are preserved. This
display convention does not change JSON paths, redirected diffs, filesystem operations, or shell
quoting.

On every platform, docwrap configures its standard text streams for UTF-8 and retains their
existing error handlers. `NO_COLOR` disables automatic color output. Environment-prefixed shell
commands are outside the hook's closed commit grammar, except the accepted
`GIT_COMMITTER_DATE=VALUE` prefix immediately before the final `git commit` segment. Other
environment-prefixed detected commits are denied with guidance on the supported form. See the
[agent hook command grammar](hook.md#command-detection) for accepted commands and routing.
