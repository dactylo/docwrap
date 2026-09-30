"""Compute diff-based incremental wrapping eligibility."""

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass

from docwrap._core._discover import definition_key, find_docstrings
from docwrap._git_utils import decode_git_quoted_path
from docwrap._types import EligibleTarget, IncrementalLevel

_DIFF_HEADER_RE = re.compile(r'^diff --git ')
_HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


# Current inclusive start/end (start == end + 1 for deletions), then removed
# source bytes without LF terminators. Keeping the bytes preserves coding cookies.
type DiffHunk = tuple[int, int, list[bytes]]


@dataclass(frozen=True, slots=True)
class _DefinitionScope:
    """A definition's line extent with child ranges for interval arithmetic."""

    key: tuple[int, int]
    own_start: int
    own_end: int
    child_ranges: tuple[tuple[int, int], ...]


def _append_removed_line(hunk: DiffHunk, line: str) -> None:
    """Retain a patch body's original source bytes."""
    if line.startswith('-'):
        _start, _end, removed = hunk
        removed.append(line[1:].encode('utf-8', errors='surrogateescape'))


def _collect_eligible_docstrings(
    source: str,
    changed_set: set[int],
) -> set[tuple[int, int]]:
    """Collect changed docstring source ranges."""
    docstrings = find_docstrings(source)
    eligible: set[tuple[int, int]] = set()
    for ds in docstrings:
        if any(line in changed_set for line in range(ds.lineno, ds.end_lineno + 1)):
            eligible.add((ds.lineno, ds.end_lineno))
    return eligible


def _decode_git_quoted_path(raw_path: str) -> str | None:
    """Decode one complete Git C-quoted path token."""
    if not raw_path.startswith('"'):
        return None
    decoded = decode_git_quoted_path(raw_path.encode('utf-8', errors='surrogateescape'))
    return decoded.decode('utf-8', errors='surrogateescape') if decoded is not None else None


def _definition_headers(tree: ast.Module) -> dict[tuple[int, int], int]:
    """Index definition ranges by their stable def/class header line."""
    headers = {definition_key(tree): 0}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            headers[definition_key(node)] = node.lineno
    return headers


def _deleted_docstrings(
    source: str,
    old_tree: ast.Module,
    current_lines: list[int | None],
    deleted_lines: set[int],
) -> set[tuple[int, int]]:
    """Select edited literals that survive deletion in their original owner."""
    # Match surviving literal lines, rather than only the owning function:
    # deleting one string may promote an untouched string to docstring.
    eligible_lines: set[int] = set()
    for node in ast.walk(old_tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        literal_lines = range(value.lineno, (value.end_lineno or value.lineno) + 1)
        if deleted_lines.isdisjoint(literal_lines):
            continue
        eligible_lines.update(
            line for index in literal_lines if (line := current_lines[index - 1]) is not None
        )
    return _collect_eligible_docstrings(source, eligible_lines)


def _find_child_definitions(
    node: ast.AST,
) -> list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Find definition nodes nested directly under node."""
    result: list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append(child)
        else:
            # Recurse into non-definition nodes (if/for/try/with/etc.)
            result.extend(_find_child_definitions(child))
    return result


def _has_own_line(scope: _DefinitionScope, line_set: set[int]) -> bool:
    """Check if any line in line_set falls in scope's own range (excluding children)."""
    for line in line_set:
        if scope.own_start <= line <= scope.own_end and not any(
            cs <= line <= ce for cs, ce in scope.child_ranges
        ):
            return True
    return False


def _iter_definition_scopes(
    tree: ast.Module,
    total_lines: int,
) -> Iterator[_DefinitionScope]:
    """Yield definition scopes for all nodes in the AST."""

    def _walk(
        node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Iterator[_DefinitionScope]:
        key = definition_key(node)
        if isinstance(node, ast.Module):
            own_start = 1
            own_end = total_lines
        else:
            own_start, own_end = key

        children = _find_child_definitions(node)
        child_ranges = tuple(definition_key(child) for child in children)

        for child in children:
            yield from _walk(child)

        yield _DefinitionScope(
            key=key,
            own_start=own_start,
            own_end=own_end,
            child_ranges=child_ranges,
        )

    yield from _walk(tree)


def _collect_eligible_definitions(
    tree: ast.Module,
    total_lines: int,
    changed_set: set[int],
) -> set[tuple[int, int]]:
    """Collect changed definition source ranges."""
    eligible: set[tuple[int, int]] = set()
    for scope in _iter_definition_scopes(tree, total_lines):
        if _has_own_line(scope, changed_set):
            eligible.add(scope.key)
    return eligible


def _deleted_eligibility(
    source: str,
    hunks: list[DiffHunk],
    tree: ast.Module,
    level: IncrementalLevel,
    encoding: str,
) -> set[tuple[int, int]]:
    """Map deleted lines' original owners to surviving current source ranges."""
    if not any(start > end and removed for start, end, removed in hunks):
        return set()

    old_lines = source.encode(encoding).split(b'\n')
    current_lines: list[int | None] = list(range(1, len(old_lines) + 1))
    deleted = [False] * len(old_lines)
    for first, end, removed in reversed(hunks):
        start = first - 1
        old_lines[start:end] = removed
        current_lines[start:end] = [None] * len(removed)
        deleted[start:end] = [start == end] * len(removed)
    try:
        old_tree = ast.parse(b'\n'.join(old_lines))
    except SyntaxError:
        # A repair can leave valid current source over an invalid baseline.
        # Keep current-line eligibility, without guessing deletion ownership.
        return set()

    deleted_lines = {index for index, removed in enumerate(deleted, 1) if removed}
    if level == 'docstring':
        return _deleted_docstrings(source, old_tree, current_lines, deleted_lines)

    by_header = {header: key for key, header in _definition_headers(tree).items()}
    old_headers = _definition_headers(old_tree)
    eligible: set[tuple[int, int]] = set()
    for old_scope in _iter_definition_scopes(old_tree, len(old_lines)):
        if not _has_own_line(old_scope, deleted_lines):
            continue
        old_header = old_headers[old_scope.key]
        header = current_lines[old_header - 1] if old_header else 0
        if header in by_header:
            eligible.add(by_header[header])
    return eligible


def _parse_diff_header_path(
    line: str,
    marker: str,
    side: str,
    *,
    allow_dev_null: bool = False,
) -> tuple[bool, str | None]:
    """Parse one old or new patch-header path."""
    marker_prefix = f'{marker} '
    if not line.startswith(marker_prefix):
        return False, None

    raw_path = line.removeprefix(marker_prefix)
    quoted = raw_path.startswith('"')
    if quoted:
        decoded = _decode_git_quoted_path(raw_path)
        if decoded is None:
            return True, None
        raw_path = decoded
    else:
        raw_path = raw_path.partition('\t')[0]

    if allow_dev_null and not quoted and raw_path == '/dev/null':
        return False, raw_path
    side_prefix = f'{side}/'
    if raw_path.startswith(side_prefix) and len(raw_path) > len(side_prefix):
        return False, raw_path.removeprefix(side_prefix)
    return False, None


def _range_from_hunk(hunk: re.Match[str]) -> DiffHunk:
    """Retain the current interval, including an empty interval for a deletion."""
    start = int(hunk.group(1))
    count = int(hunk.group(2)) if hunk.group(2) is not None else 1
    return (start + 1, start, []) if count == 0 else (start, start + count - 1, [])


def derive_eligible_target(incr_level: IncrementalLevel | None) -> EligibleTarget:
    """Derive the eligibility target from the incremental level.

    Args:
        incr_level: The resolved incremental level, or `None` when incremental
            mode is disabled.

    Returns:
        `docstring` when the level is `docstring`; otherwise `definition`.
    """
    return 'docstring' if incr_level == 'docstring' else 'definition'


def get_eligible_ranges(
    source: str,
    changed_lines: list[DiffHunk],
    *,
    level: IncrementalLevel,
    encoding: str = 'utf-8',
) -> set[tuple[int, int]] | None:
    """Determine which definitions or docstrings are eligible for wrapping.

    Args:
        source: Current Python source code.
        changed_lines: Parsed Git hunks, including original deleted lines.
        level: Incremental level (file, definition, docstring).
        encoding: Current source encoding, used to reconstruct baseline bytes.

    Returns:
        Set of (start_line, end_line) tuples identifying eligible definitions
        or docstrings. Returns None if all docstrings should be wrapped
        (file-level with changes). Returns empty set if no docstrings are
        eligible.

    Raises:
        SyntaxError: If `changed_lines` is nonempty and the Python source
            cannot be parsed.
    """
    if not changed_lines:
        return set()

    if level == 'file':
        ast.parse(source)
        return None  # All docstrings eligible

    # Build set of changed line numbers
    changed_set: set[int] = set()
    for start, end, _removed in changed_lines:
        changed_set.update(range(start, end + 1))

    tree = ast.parse(source)
    eligible = _deleted_eligibility(source, changed_lines, tree, level, encoding)
    if level == 'docstring':
        return eligible | _collect_eligible_docstrings(source, changed_set)

    total_lines = len(source.splitlines()) or 1
    return eligible | _collect_eligible_definitions(tree, total_lines, changed_set)


def parse_diff_hunks(diff_output: str) -> dict[str, list[DiffHunk]] | None:
    """Parse git diff -U0 output into per-file changed line ranges.

    Args:
        diff_output: Raw output from git diff -U0.

    Returns:
        Dict mapping file paths to ordered hunks with current intervals and
        original removed lines, or `None` when a patch header contains
        malformed Git quoting.
    """
    result: dict[str, list[DiffHunk]] = {}
    current_file = ''
    file_header_active = False
    in_hunk = False
    old_path_seen = False

    for line in diff_output.split('\n'):
        if _DIFF_HEADER_RE.match(line):
            current_file = ''
            file_header_active = True
            in_hunk = False
            old_path_seen = False
            continue

        hunk = _HUNK_RE.match(line)
        file_header_active = file_header_active and hunk is None
        in_hunk = in_hunk or hunk is not None

        if in_hunk and hunk is None:
            if current_file:
                _append_removed_line(result[current_file][-1], line)
            continue
        if file_header_active:
            marker = '+++' if old_path_seen else '---'
            side = 'b' if old_path_seen else 'a'
            malformed, path = _parse_diff_header_path(
                line,
                marker,
                side,
                allow_dev_null=not old_path_seen,
            )
            if malformed:
                return None
            if not old_path_seen:
                old_path_seen = path is not None
                continue

            if path is None:
                old_path_seen = False
                continue

            current_file = path
            result.setdefault(current_file, [])
            file_header_active = False
            continue

        if hunk and current_file:
            in_hunk = True
            result[current_file].append(_range_from_hunk(hunk))

    return result


def resolve_incremental_level(
    *,
    override: bool | None,
    explicit_level: IncrementalLevel | None = None,
    config_value: IncrementalLevel | bool | None = None,
) -> IncrementalLevel | None:
    """Resolve the effective incremental level from override flags + config.

    Precedence:
        1. override=False -> None (full mode)
        2. explicit_level set -> that level
        3. override=True -> config string if any, else 'definition'
        4. config_value False -> None
        5. config_value string -> that level
        6. else -> None

    Args:
        override: True to force incremental on, False to force off, None to
            defer to config_value.
        explicit_level: An explicit level (takes precedence over
            override=True).
        config_value: The resolved config value: an incremental level string, a
            bool, or None.

    Returns:
        The resolved IncrementalLevel, or None for full (non-incremental) mode.
    """
    if override is False:
        return None
    if explicit_level is not None:
        return explicit_level
    if override is True:
        return config_value if isinstance(config_value, str) else 'definition'
    if isinstance(config_value, str):
        return config_value
    return None
