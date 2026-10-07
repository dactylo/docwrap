"""Google-style docstring style."""

import ast
import io
import logging
import re
import tokenize
from collections.abc import Mapping, Sequence
from typing import Literal

from docwrap._canonical import CANONICAL_SECTIONS, RETURN_LIKE_ROLES
from docwrap._styles import EntryMatch
from docwrap._types import CanonicalRole, SectionKind, SphinxFieldRole, StyleName

_log = logging.getLogger(__name__)

_DYNAMIC_SECTIONS = frozenset({'Returns', 'Return', 'Yields', 'Yield'})

_ENTRY_LIKE = frozenset(
    {
        'Args',
        'Arguments',
        'Attributes',
        'Except',
        'Exceptions',
        'Keyword Args',
        'Keyword Arguments',
        'Methods',
        'Parameters',
        'Raise',
        'Raises',
        'Warns',
    }
)

_GOOGLE_CANONICAL_TO_HEADER: dict[str, str] = {
    s.role: s.google_header for s in CANONICAL_SECTIONS if s.google_header is not None
}

_PROSE_LIKE = frozenset(
    {
        'Attention',
        'Caution',
        'Danger',
        'Example',
        'Examples',
        'Hint',
        'Important',
        'Note',
        'Notes',
        'References',
        'Return',
        'Returns',
        'See Also',
        'Tip',
        'Todo',
        'Warning',
        'Warnings',
        'Yield',
        'Yields',
    }
)


def _compile_entry_re(section_body_indent: int) -> re.Pattern[str]:
    """Compile the strict Google entry pattern for one body indent."""
    return re.compile(
        r'^[ ]{' + str(section_body_indent) + r'}(\*{0,2}\w[\w.]*)'
        r'(?:\s+\(([^)]*)\))?'
        r':\s*(.*)',
    )


def _find_annotation_delimiter(content: str) -> tuple[int, int] | None:
    """Find the first colon token outside annotation brackets."""
    delimiter: tuple[int, int] | None = None
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in '([{':
                depth += 1
            elif token.string in ')]}':
                if depth == 0:
                    break
                depth -= 1
            elif token.string == ':' and depth == 0:
                delimiter = token.start[1], token.end[1]
                break
    except (IndentationError, tokenize.TokenError):
        pass
    return delimiter


def _is_annotation_node(node: ast.AST, *, in_subscript: bool = False) -> bool:
    """Return whether an AST node belongs to the accepted annotation grammar."""
    if isinstance(node, ast.Name):
        result = True
    elif isinstance(node, ast.Attribute):
        result = _is_annotation_node(node.value, in_subscript=in_subscript)
    elif isinstance(node, ast.Subscript):
        result = _is_annotation_node(node.value) and _is_annotation_node(
            node.slice,
            in_subscript=True,
        )
    elif isinstance(node, ast.Tuple):
        result = all(_is_annotation_node(item, in_subscript=in_subscript) for item in node.elts)
    elif isinstance(node, ast.List):
        result = in_subscript and all(
            _is_annotation_node(item, in_subscript=True) for item in node.elts
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        result = _is_annotation_node(
            node.left,
            in_subscript=in_subscript,
        ) and _is_annotation_node(
            node.right,
            in_subscript=in_subscript,
        )
    elif isinstance(node, ast.Constant):
        result = (
            isinstance(node.value, str)
            or node.value is None
            or node.value is Ellipsis
            or in_subscript
        )
    else:
        result = False
    return result


class _GoogleStyle:
    """Parse and format Google-style docstrings."""

    name: StyleName = 'google'
    known_headers = _ENTRY_LIKE | _PROSE_LIKE
    section_body_indent = 4
    section_to_canonical: Mapping[str, CanonicalRole] = {
        'Args': 'param',
        'Arguments': 'param',
        'Attributes': 'attributes',
        'Example': 'examples',
        'Examples': 'examples',
        'Except': 'raises',
        'Exceptions': 'raises',
        'Keyword Args': 'keyword_param',
        'Keyword Arguments': 'keyword_param',
        'Methods': 'methods',
        'Note': 'notes',
        'Notes': 'notes',
        'Parameters': 'param',
        'Raise': 'raises',
        'Raises': 'raises',
        'References': 'references',
        'Return': 'returns',
        'Returns': 'returns',
        'See Also': 'see_also',
        'Todo': 'todo',
        'Warning': 'warnings',
        'Warnings': 'warnings',
        'Warns': 'warns',
        'Yield': 'yields',
        'Yields': 'yields',
    }
    supports_section_headers: bool = True
    section_header_line_count = 1
    supports_compound_type_entries: bool = False
    supports_multi_name_entries: bool = False
    uses_field_list_entries: bool = False
    entry_continuation_indent = 8

    # Design note: The default section_body_indent is deliberately four spaces.
    # Compiling the strict pattern from that value keeps entry matching aligned with
    # formatting and conservative return matching while rejecting continuation lines.
    # The [^)]* group cannot handle balanced parentheses in type annotations (e.g.,
    # tuple[()]). In practice, Python type hints use square brackets, so nested
    # parentheses are vanishingly rare. If this proves insufficient, switch to a
    # balanced-parentheses parser.
    _entry_re = _compile_entry_re(section_body_indent)

    def _match_return_annotation(self, line: str) -> EntryMatch | None:
        """Match a conservative return-like annotation entry."""
        indent = len(line) - len(line.lstrip())
        if indent != self.section_body_indent:
            return None
        content = line[indent:]
        delimiter = _find_annotation_delimiter(content)
        if delimiter is None:
            return None
        colon_start, colon_end = delimiter
        prefix = content[:colon_start].strip()
        if not prefix:
            return None
        try:
            node = ast.parse(prefix, mode='eval').body
            if not _is_annotation_node(node):
                return None
        except (RecursionError, SyntaxError):
            return None
        remainder = content[colon_end:]
        description = remainder.strip()
        description_start = None
        if description:
            description_start = indent + colon_end + len(remainder) - len(remainder.lstrip())
        return EntryMatch(
            name='',
            type_ann=prefix,
            description=description,
            description_start=description_start,
        )

    def canonicalize_role(
        self,
        role: CanonicalRole | SphinxFieldRole | Literal[''],
    ) -> CanonicalRole | SphinxFieldRole | Literal['']:
        """Google roles are already canonical.

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
        """Type-less returns/yields become prose in Google style.

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
        """Produce Google-style entry line.

        Args:
            match: Parsed entry data.
            role: Canonical role (e.g. 'param', 'returns', 'raises').
            strip_types: If True, omit type annotations.

        Returns:
            The formatted Google-style entry lines.
        """
        if role == 'param' and ',' in match.name:
            names = [name.strip() for name in match.name.split(',')]
            _log.warning(
                'Splitting multi-name parameter %r into %d separate entries; '
                'the shared description was duplicated and may need review',
                match.name,
                len(names),
            )
            result: list[str] = []
            for name in names:
                single = EntryMatch(
                    name=name,
                    type_ann=match.type_ann,
                    description=match.description,
                    role=match.role,
                )
                result.extend(self.format_entry(single, role, strip_types))
            return result

        indent = ' ' * self.section_body_indent
        name = match.name
        type_ann = '' if strip_types else match.type_ann
        desc = match.description

        # Returns/Yields: type goes in name position, not parenthesized.
        # Named returns (both name and type) fall through to normal entry logic.
        if role in RETURN_LIKE_ROLES and not (name and type_ann):
            type_str = type_ann or name
            if strip_types:
                type_str = ''
            if type_str:
                head = f'{indent}{type_str}:'
                return [f'{head} {desc}'] if desc else [head]
            # No type at all -- just the description as prose
            return [f'{indent}{desc}'] if desc else []

        # Build the entry line
        head = f'{indent}{name} ({type_ann}):' if type_ann else f'{indent}{name}:'

        if desc:
            return [f'{head} {desc}']
        return [head]

    def format_role(
        self,
        role: CanonicalRole | SphinxFieldRole,
    ) -> CanonicalRole | SphinxFieldRole:
        """Google uses canonical roles directly.

        Args:
            role: Canonical or Sphinx field role to format.

        Returns:
            The role unchanged.
        """
        return role

    def format_section_header(self, canonical_key: str) -> list[str]:
        """Produce Google-style section header for a canonical key.

        Args:
            canonical_key: Canonical role identifying the section.

        Returns:
            The Google header line, or an empty list for an unknown key.
        """
        name = _GOOGLE_CANONICAL_TO_HEADER.get(canonical_key)
        if name is None:
            return []
        return [f'{name}:']

    def is_conservative_entry_candidate(self, content: str) -> bool:
        """Protect a nonempty colon prefix that could become a Google entry.

        Args:
            content: Line content after the entry indentation.

        Returns:
            Whether a nonempty prefix precedes a colon.
        """
        prefix, separator, _ = content.partition(':')
        return bool(separator and prefix.strip())

    def is_section_header(self, line: str, next_line: str | None = None) -> str | None:  # noqa: ARG002
        """Return section name if line is a Google-style header.

        Args:
            line: Current line to inspect.
            next_line: Unused following line.

        Returns:
            The section name, or `None` when the line is not a header.
        """
        stripped = line.strip()
        if not stripped.endswith(':'):
            return None
        # Must be at indent 0 (no leading whitespace)
        if line != line.lstrip():
            return None
        name = stripped[:-1]
        if name in self.known_headers:
            return name
        return None

    def match_entry(
        self,
        line: str,
        *,
        role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
    ) -> EntryMatch | None:
        """Match a Google-style entry line for the current role.

        Args:
            line: Physical line to parse.
            role: Current canonical or Sphinx field role when known.

        Returns:
            Parsed entry data, or None when the line is not an entry.
        """
        m = self._entry_re.match(line)
        if m is not None:
            description = m.group(3).strip()
            return EntryMatch(
                name=m.group(1),
                type_ann=m.group(2) or '',
                description=description,
                description_start=m.start(3) if description else None,
            )
        if role in RETURN_LIKE_ROLES:
            return self._match_return_annotation(line)
        return None

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
        role: CanonicalRole | Literal[''] = self.section_to_canonical.get(name) or ''
        if (
            name in _DYNAMIC_SECTIONS
            and first is not None
            and self.match_entry(first, role=role) is not None
        ):
            return 'entry'
        if name in _ENTRY_LIKE:
            return 'entry'
        return 'prose'

    def supports_canonical_role(self, role: CanonicalRole) -> bool:
        """Return whether Google can emit the canonical role.

        Args:
            role: Canonical role to query.

        Returns:
            Whether Google has a target representation for `role`.
        """
        return role in _GOOGLE_CANONICAL_TO_HEADER


_INSTANCE = _GoogleStyle()
STYLE = _INSTANCE
