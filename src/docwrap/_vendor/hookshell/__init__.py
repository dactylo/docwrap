"""Classify agent-hook shell commands through supported submodule imports.

The public API is `classify_command` and `Classification` in `_classify`;
`CommitPlan`, `CommitMode`, `NoCommit`, `PreCommitAdd`, and `ADD_SELECTORS` in
`_grammar`; and `normalize_msys_path` in `_shell`. The package root imports
nothing and exports no names. Import each name from its defining module.
"""
