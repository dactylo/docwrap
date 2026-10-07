# Changelog

## 0.2.0 (2026-10-06)

### Breaking changes

- `docwrap --stats` now exits 2 when any file cannot be read or parsed, and names each such file on
  stderr. If your scripts expect exit 0, fix or exclude those files.
- The hook now denies commits run through shell, interpreter, or `eval` names in any letter case or
  with an `.exe` suffix, such as `BASH -c` or `node.exe -e`, and through escaped backtick
  substitutions. Run `git commit` directly in the supported form.
- The hook no longer accepts `git status -l` before a `git commit --dry-run`. Drop `-l` or run the
  status check separately.

### Changed

- On Claude Code and Codex, PostToolUse errors and skip notes now reach the agent as context, not
  only on stderr.
- Hook denials explain their cause more clearly, including Git's error when replaying `git add`
  fails.
- When Git exclusions cannot be applied inside a worktree, directory scans warn and still process
  every discovered Python file.
- `docwrap --help` and `docwrap hook --help` are grouped and shorter, and hook help lists the
  installed agent adapters.
- On Windows, human-readable output shows paths with forward slashes.

### Fixed

- The hook no longer denies a plain `cat` or `tee` command that writes a quoted heredoc mentioning
  `git commit` to a file.
- The hook no longer stalls or denies on heredocs that contain many `<<` operators, such as C++
  code.
- Incremental mode works in a repository with no commits.
- Incremental mode no longer misses changes when Git uses an external diff tool or text conversion
  driver.
- Valid files are no longer reported as unparseable under `-W error` or pytest's
  `filterwarnings = error`.
- If writing a file fails, docwrap restores the original when it can and reports both errors when
  it cannot.

## 0.1.0 (2026-09-29)

Initial release.
