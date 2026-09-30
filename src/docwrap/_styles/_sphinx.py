"""Sphinx/reST-style docstring style."""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from docwrap._canonical import RETURN_LIKE_ROLES
from docwrap._styles import EntryMatch
from docwrap._types import CanonicalRole, SectionKind, SphinxFieldRole, StyleName

_log = logging.getLogger(__name__)

FIELD_ROLES = frozenset(
    {
        'cvar',
        'except',
        'ivar',
        'keyword',
        'param',
        'raise',
        'raises',
        'return',
        'returns',
        'rtype',
        'type',
        'var',
        'vartype',
        'warn',
        'warns',
        'yield',
        'yields',
    }
)

FIELD_MARKER_RE = re.compile(r'^\s*:(' + '|'.join(sorted(FIELD_ROLES)) + r')\b')

# Matches  :role [type] [name]: description
_FIELD_RE = re.compile(
    r'^:(' + '|'.join(sorted(FIELD_ROLES)) + r')'
    r'(?:\s+(\S.*?))?'
    r':\s*(.*)',
)


_ROLE_NORMALIZE: dict[str, str] = {
    'except': 'raises',
    'raise': 'raises',
    'return': 'returns',
    'warn': 'warns',
    'yield': 'yields',
}


class _SphinxStyle:
    """Sphinx/reST-style docstring style."""

    name: StyleName = 'sphinx'
    canonical_to_field: Mapping[CanonicalRole, SphinxFieldRole] = {
        'attributes': 'ivar',
        'keyword_param': 'keyword',
        'param': 'param',
        'raises': 'raises',
        'returns': 'returns',
        'warns': 'warns',
    }
    field_to_canonical: Mapping[SphinxFieldRole, CanonicalRole] = {
        'ivar': 'attributes',
        'keyword': 'keyword_param',
        'param': 'param',
        'raises': 'raises',
        'returns': 'returns',
        'warns': 'warns',
        'yields': 'yields',
    }
    known_headers: frozenset[str] = frozenset()
    section_body_indent = 4
    section_to_canonical: Mapping[str, CanonicalRole] = {}
    supports_section_headers: bool = False
    section_header_line_count = 1
    supports_compound_type_entries: bool = False
    supports_multi_name_entries: bool = False
    uses_field_list_entries: bool = True
    entry_continuation_indent = 4

    def canonicalize_role(
        self,
        role: CanonicalRole | SphinxFieldRole | Literal[''],
    ) -> CanonicalRole | SphinxFieldRole | Literal['']:
        """Map Sphinx field roles to their canonical equivalents.

        Args:
            role: Canonical, Sphinx field, or empty role to normalize.

        Returns:
            The canonical equivalent of a Sphinx role, otherwise the role
            unchanged.
        """
        if role in self.field_to_canonical:
            return self.field_to_canonical[cast('SphinxFieldRole', role)]
        return role

    def classify_section(
        self,
        name: str,  # noqa: ARG002
        body_peek: Sequence[str] = (),  # noqa: ARG002
    ) -> SectionKind:
        """Sphinx field lists are always entry-like.

        Args:
            name: Unused section name.
            body_peek: Unused initial body lines.

        Returns:
            The `entry` classification.
        """
        return 'entry'

    def emits_description_only_return_as_prose(
        self,
        role: CanonicalRole | SphinxFieldRole,  # noqa: ARG002
        match: EntryMatch,  # noqa: ARG002
        strip_types: bool,  # noqa: ARG002
    ) -> bool:
        """Sphinx uses :returns: field even without a type -- never prose.

        Args:
            role: Unused canonical or Sphinx field role.
            match: Unused parsed entry data.
            strip_types: Unused type-removal setting.

        Returns:
            `False` because Sphinx always emits a field entry.
        """
        return False

    def format_entry(
        self,
        match: EntryMatch,
        role: CanonicalRole | SphinxFieldRole,
        strip_types: bool = False,
    ) -> list[str]:
        """Produce Sphinx-style field entry.

        Args:
            match: Parsed entry data.
            role: Sphinx field role (e.g. 'param', 'returns', 'raises',
                'ivar').
            strip_types: If True, omit type annotations and suppress :rtype:.

        Returns:
            The formatted Sphinx field-list entry lines.
        """
        # Split NumPy multi-name params (e.g. "x1, x2") into separate entries.
        if ',' in match.name:
            names = [n.strip() for n in match.name.split(',')]
            _log.warning(
                'Splitting multi-name entry %r into %d separate entries',
                match.name,
                len(names),
            )
            result: list[str] = []
            for n in names:
                single = EntryMatch(
                    name=n,
                    type_ann=match.type_ann,
                    description=match.description,
                    role=match.role,
                )
                result.extend(self.format_entry(single, role, strip_types))
            return result

        name = match.name
        type_ann = '' if strip_types else match.type_ann
        desc = match.description

        # returns/yields: :returns: desc + optional :rtype: type
        if role in RETURN_LIKE_ROLES:
            result = [f':{role}: {desc}' if desc else f':{role}:']
            if type_ann and role == 'returns':
                result.append(f':rtype: {type_ann}')
            return result

        if role == 'ivar':
            head = f':ivar {name}:'
            result = [f'{head} {desc}'] if desc else [head]
            if type_ann:
                result.append(f':vartype {name}: {type_ann}')
            return result

        # param/raises: inline type  :role type name: desc
        head = f':{role} {type_ann} {name}:' if type_ann else f':{role} {name}:'
        return [f'{head} {desc}'] if desc else [head]

    def format_role(
        self,
        role: CanonicalRole | SphinxFieldRole,
    ) -> CanonicalRole | SphinxFieldRole:
        """Map canonical roles to Sphinx field roles for output.

        Args:
            role: Canonical or Sphinx field role to format.

        Returns:
            The Sphinx field equivalent of a canonical role, otherwise the role
            unchanged.
        """
        if role in self.canonical_to_field:
            return self.canonical_to_field[cast('CanonicalRole', role)]
        return role

    def format_section_header(self, canonical_key: str) -> list[str]:  # noqa: ARG002
        """Sphinx has no section headers.

        Args:
            canonical_key: Unused canonical section role.

        Returns:
            An empty list.
        """
        return []

    def is_section_header(
        self,
        line: str,  # noqa: ARG002
        next_line: str | None = None,  # noqa: ARG002
    ) -> str | None:
        """Sphinx has no section headers.

        Args:
            line: Unused current line.
            next_line: Unused following line.

        Returns:
            `None` because Sphinx uses field-list entries.
        """
        return None

    def match_entry(
        self,
        line: str,
        *,
        role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
    ) -> EntryMatch | None:
        """Match a Sphinx field-list entry.

        Args:
            line: Physical line to parse.
            role: Current canonical or Sphinx field role when known.

        Returns:
            Parsed entry data, or None when the line is not an entry.
        """
        del role
        stripped = line.strip()
        m = _FIELD_RE.match(stripped)
        if m is None:
            return None
        leading_ws = len(line) - len(line.lstrip())
        # _FIELD_RE constrains the role to FIELD_ROLES, which after
        # _ROLE_NORMALIZE collapses to the SphinxFieldRole alphabet.
        role = cast('SphinxFieldRole', _ROLE_NORMALIZE.get(m.group(1), m.group(1)))
        middle = m.group(2) or ''
        desc = m.group(3).strip()
        desc_start: int | None = None
        if desc:
            desc_start = m.start(3) + leading_ws

        # :param type name: desc  OR  :param name: desc
        # :keyword type name: desc  OR  :keyword name: desc
        # :ivar type name: desc  OR  :ivar name: desc
        # Parse from the right: name is always the last token,
        # everything before it is the type (handles multi-token types like dict[str, int])
        if role in ('param', 'keyword', 'ivar'):
            parts = middle.rsplit(None, 1)
            if len(parts) > 1:
                return EntryMatch(
                    name=parts[1],
                    type_ann=parts[0],
                    description=desc,
                    role=role,
                    description_start=desc_start,
                )
            return EntryMatch(
                name=middle.strip(),
                type_ann='',
                description=desc,
                role=role,
                description_start=desc_start,
            )

        # :type name: type_str  /  :rtype: type_str
        if role in ('type', 'vartype', 'rtype'):
            return EntryMatch(
                name=middle.strip(),
                type_ann='',
                description=desc,
                role=role,
                description_start=desc_start,
            )

        # :raises ExcType: desc  /  :returns: desc
        return EntryMatch(
            name=middle.strip(),
            type_ann='',
            description=desc,
            role=role,
            description_start=desc_start,
        )

    def supports_canonical_role(self, role: CanonicalRole) -> bool:
        """Return whether Sphinx can emit the canonical role.

        Args:
            role: Canonical role to query.

        Returns:
            Whether Sphinx has a target representation for `role`.
        """
        return role in self.canonical_to_field


_INSTANCE = _SphinxStyle()
STYLE = _INSTANCE
