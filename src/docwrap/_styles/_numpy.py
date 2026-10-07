"""NumPy-style docstring style."""

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from docwrap._canonical import CANONICAL_SECTIONS, RETURN_LIKE_ROLES
from docwrap._styles import EntryMatch
from docwrap._types import CanonicalRole, SectionKind, SphinxFieldRole, StyleName

_DYNAMIC_SECTIONS = frozenset({'Returns', 'Yields'})

_ENTRY_LIKE = frozenset(
    {
        'Attributes',
        'Methods',
        'Other Parameters',
        'Params',
        'Parameters',
        'Raises',
        'Warns',
    }
)

_ENTRY_RE = re.compile(
    r'^(\*{0,2}\w[\w.]*(?:\s*,\s*\*{0,2}\w[\w.]*)*)'  # name(s), comma-separated
    r'(?:\s*:\s*(.+))?$',  # optional : type
)

_MIN_DASHES = 3

_NUMPY_CANONICAL_TO_HEADER: dict[str, str] = {
    s.role: s.numpy_header for s in CANONICAL_SECTIONS if s.numpy_header is not None
}

_PROSE_LIKE = frozenset(
    {
        'Examples',
        'Notes',
        'References',
        'Returns',
        'See Also',
        'Todo',
        'Warnings',
        'Yields',
    }
)


class _NumpyStyle:
    """Parse and format NumPy-style docstrings."""

    name: StyleName = 'numpy'
    known_headers = _ENTRY_LIKE | _PROSE_LIKE
    section_body_indent = 0
    section_to_canonical: Mapping[str, CanonicalRole] = {
        'Attributes': 'attributes',
        'Examples': 'examples',
        'Methods': 'methods',
        'Notes': 'notes',
        'Other Parameters': 'other_param',
        'Params': 'param',
        'Parameters': 'param',
        'Raises': 'raises',
        'References': 'references',
        'Returns': 'returns',
        'See Also': 'see_also',
        'Todo': 'todo',
        'Warnings': 'warnings',
        'Warns': 'warns',
        'Yields': 'yields',
    }
    supports_section_headers: bool = True
    section_header_line_count = 2
    supports_compound_type_entries: bool = True
    supports_multi_name_entries: bool = True
    uses_field_list_entries: bool = False
    entry_continuation_indent = 4

    def canonicalize_role(
        self,
        role: CanonicalRole | SphinxFieldRole | Literal[''],
    ) -> CanonicalRole | SphinxFieldRole | Literal['']:
        """NumPy roles are already canonical.

        Args:
            role: Canonical, Sphinx field, or empty role to normalize.

        Returns:
            The role unchanged.
        """
        return role

    def emits_description_only_return_as_prose(
        self,
        role: CanonicalRole | SphinxFieldRole,
        match: EntryMatch,
        strip_types: bool,
    ) -> bool:
        """Type-less returns/yields become prose in NumPy style.

        Args:
            role: Canonical or Sphinx field role for the entry.
            match: Parsed entry data.
            strip_types: Whether conversion removes type annotations.

        Returns:
            Whether to emit the description as prose instead of an entry.
        """
        if role not in RETURN_LIKE_ROLES:
            return False
        has_type = bool(match.type_ann if not strip_types else False)
        return not has_type and not match.name

    def format_entry(
        self,
        match: EntryMatch,
        role: CanonicalRole | SphinxFieldRole,
        strip_types: bool = False,
    ) -> list[str]:
        """Produce NumPy-style entry blocks with indented descriptions.

        Args:
            match: Parsed entry data.
            role: Canonical role, such as `param`, `returns`, or `raises`.
            strip_types: If `True`, omit type annotations.

        Returns:
            Formatted NumPy-style entry blocks. Each string can contain
            multiple physical lines.
        """
        name = match.name
        type_ann = '' if strip_types else match.type_ann
        desc = match.description
        cont_indent = ' ' * self.entry_continuation_indent

        # Returns/Yields: type goes in header position, handle empty gracefully.
        # Named returns (both name and type) fall through to normal entry logic.
        if role in RETURN_LIKE_ROLES and not (name and type_ann):
            type_str = type_ann or name
            if strip_types:
                type_str = ''
            if type_str:
                head = type_str
            else:
                # No type -- just indented description
                return [f'{cont_indent}{desc}'] if desc else []
            if desc:
                return [f'{head}\n{cont_indent}{desc}']
            return [head]

        # Build header line
        if name and type_ann:
            head = f'{name} : {type_ann}'
        elif name:
            head = name
        else:
            head = type_ann

        # Append description on continuation line if present
        if desc:
            return [f'{head}\n{cont_indent}{desc}']
        return [head]

    def format_role(
        self,
        role: CanonicalRole | SphinxFieldRole,
    ) -> CanonicalRole | SphinxFieldRole:
        """NumPy uses canonical roles directly.

        Args:
            role: Canonical or Sphinx field role to format.

        Returns:
            The role unchanged.
        """
        return role

    def format_section_header(self, canonical_key: str) -> list[str]:
        """Produce NumPy-style section header (name + dashes).

        Args:
            canonical_key: Canonical role identifying the section.

        Returns:
            The NumPy header and underline, or an empty list for an unknown
            key.
        """
        name = _NUMPY_CANONICAL_TO_HEADER.get(canonical_key)
        if name is None:
            return []
        return [name, '-' * len(name)]

    def is_conservative_entry_candidate(self, content: str) -> bool:
        """Leave NumPy entry detection to the normal matcher.

        Args:
            content: Unused line content after the entry indentation.

        Returns:
            `False` because NumPy needs no conservative check.
        """
        del content
        return False

    def is_section_header(self, line: str, next_line: str | None = None) -> str | None:
        """Return section name if line + next_line form a NumPy header.

        Args:
            line: Candidate section name line.
            next_line: Candidate dashed underline.

        Returns:
            The section name, or `None` when the lines do not form a header.
        """
        if next_line is None or line != line.lstrip():
            return None
        stripped = line.strip()
        if stripped not in self.known_headers:
            return None
        # Underline must also be at indent 0 (aligned with header)
        if next_line != next_line.lstrip():
            return None
        next_stripped = next_line.strip()
        if (
            len(next_stripped) < _MIN_DASHES
            or next_stripped != '-' * len(next_stripped)
            or len(next_stripped) < len(stripped)
        ):
            return None
        return stripped

    def match_entry(
        self,
        line: str,
        *,
        role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
    ) -> EntryMatch | None:
        """Match a NumPy-style entry line for the current role.

        Args:
            line: Physical line to parse.
            role: Current canonical or Sphinx field role when known.

        Returns:
            Parsed entry data, or None when the line is not an entry.
        """
        del role
        if line != line.lstrip():
            return None
        m = _ENTRY_RE.match(line.strip())
        if m is None:
            return None
        return EntryMatch(
            name=m.group(1),
            type_ann=m.group(2) or '',
            description='',
            description_start=None,
        )

    def classify_section(
        self,
        name: str,
        body_peek: Sequence[str] = (),
    ) -> SectionKind:
        """Classify section as entry-like or prose-like.

        Args:
            name: Section header name.
            body_peek: Initial nonblank body lines used for dynamic sections.

        Returns:
            The section's entry-like or prose-like classification.
        """
        first = body_peek[0] if body_peek else None
        second = body_peek[1] if len(body_peek) > 1 else None
        role: CanonicalRole | Literal[''] = self.section_to_canonical.get(name) or ''
        if name in _DYNAMIC_SECTIONS and first is not None:
            if self.match_entry(first, role=role) is not None:
                return 'entry'
            # Compound-type heuristic: if the first body line is at indent 0
            # (not indented) and the next non-blank line IS indented, treat as
            # entry-like.  Covers types like "int or None", "list of str",
            # "Optional[int]" that _ENTRY_RE cannot match.
            if first == first.lstrip() and second is not None and second != second.lstrip():
                return 'entry'
        if name in _ENTRY_LIKE:
            return 'entry'
        return 'prose'

    def supports_canonical_role(self, role: CanonicalRole) -> bool:
        """Return whether NumPy can emit the canonical role.

        Args:
            role: Canonical role to query.

        Returns:
            Whether NumPy has a target representation for `role`.
        """
        return role in _NUMPY_CANONICAL_TO_HEADER


_INSTANCE = _NumpyStyle()
STYLE = _INSTANCE
