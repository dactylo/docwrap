"""Style protocol and registry for docstring styles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from docwrap._types import CanonicalRole, SectionKind, SphinxFieldRole, StyleName

__all__ = [
    'EntryMatch',
    'Style',
    'get_style',
    'get_styles',
    'resolve_style_override',
]


@dataclass(frozen=True, slots=True)
class EntryMatch:
    """A matched entry line within a structured section.

    Args:
        name: Entry name, or an empty string for unnamed return-like entries.
        type_ann: Parsed type annotation, or an empty string.
        description: Parsed entry description, or an empty string.
        role: Canonical or Sphinx field role, when known.
        description_start: Character offset where the description begins, or
            `None` when no source position is available.
    """

    name: str
    type_ann: str
    description: str
    role: CanonicalRole | SphinxFieldRole | Literal[''] = ''
    description_start: int | None = None


class Style(Protocol):
    """Protocol that docstring style implementations must satisfy."""

    def canonicalize_role(
        self,
        role: CanonicalRole | SphinxFieldRole | Literal[''],
    ) -> CanonicalRole | SphinxFieldRole | Literal['']:
        """Return the canonical role for this style.

        Google/NumPy return role unchanged. Sphinx maps field roles to their
        canonical equivalents.

        Args:
            role: Canonical, Sphinx field, or empty role to normalize.

        Returns:
            The role expressed in the style's canonical vocabulary.
        """
        ...

    def classify_section(
        self,
        name: str,
        body_peek: Sequence[str] = (),
    ) -> SectionKind:
        """Classify a section as entry-like or prose-like.

        Args:
            name: The section header name.
            body_peek: Up to N upcoming non-blank, non-code-block body lines.
                Google and NumPy use them for section classification.

        Returns:
            The section's entry-like or prose-like classification.
        """
        ...

    def emits_description_only_return_as_prose(
        self,
        role: CanonicalRole | SphinxFieldRole,
        match: EntryMatch,
        strip_types: bool,
    ) -> bool:
        """Whether a type-less return/yield should become prose.

        Called on the TARGET style. Returns True for Google/NumPy (type-less
        returns become prose). Returns False for Sphinx (uses :returns: field).

        Args:
            role: Canonical or Sphinx field role for the entry.
            match: Parsed entry data.
            strip_types: Whether conversion removes type annotations.

        Returns:
            Whether to emit the description as prose instead of an entry.
        """
        ...

    @property
    def entry_continuation_indent(self) -> int:
        """Extra indentation for continuation lines of an entry.

        Returns:
            The continuation indentation in spaces.
        """
        ...

    def format_entry(
        self,
        match: EntryMatch,
        role: CanonicalRole | SphinxFieldRole,
        strip_types: bool = False,
    ) -> list[str]:
        """Produce entry line(s) in this style's format.

        Args:
            match: Parsed entry data.
            role: Canonical or Sphinx field role to emit.
            strip_types: Whether to omit type annotations.

        Returns:
            The formatted entry blocks. A block may contain multiple physical
            lines.
        """
        ...

    def format_role(
        self,
        role: CanonicalRole | SphinxFieldRole,
    ) -> CanonicalRole | SphinxFieldRole:
        """Return the output role to pass to format_entry.

        Google/NumPy return role unchanged. Sphinx maps canonical roles to
        their field role equivalents.

        Args:
            role: Canonical or Sphinx field role to format.

        Returns:
            The role used when formatting the entry.
        """
        ...

    def format_section_header(self, canonical_key: str) -> list[str]:
        """Produce header line(s) for a canonical section key.

        Args:
            canonical_key: Canonical role identifying the section.

        Returns:
            The formatted section-header lines, or an empty list when the style
            has no matching header.
        """
        ...

    def is_section_header(self, line: str, next_line: str | None = None) -> str | None:
        """Return section name if line is a header, else None.

        Args:
            line: The current line to check.
            next_line: The following line (needed for NumPy dashed underlines).

        Returns:
            The section name, or `None` when the lines do not form a header.
        """
        ...

    @property
    def known_headers(self) -> frozenset[str]:
        """All recognized section header names for this style.

        Returns:
            The recognized section header names.
        """
        ...

    def match_entry(
        self,
        line: str,
        *,
        role: CanonicalRole | SphinxFieldRole | Literal[''] = '',
    ) -> EntryMatch | None:
        """Parse an entry line for the current canonical role.

        Args:
            line: Physical line to parse.
            role: Current canonical or Sphinx field role when known.

        Returns:
            Parsed entry data, or None when the line is not an entry.
        """
        ...

    @property
    def name(self) -> StyleName:
        """Short identifier for this style (e.g. 'google').

        Returns:
            The built-in style name.
        """
        ...

    @property
    def section_body_indent(self) -> int:
        """Extra indentation of section body relative to the header.

        Returns:
            The section-body indentation in spaces.
        """
        ...

    @property
    def section_header_line_count(self) -> int:
        """Number of physical lines consumed by a section header.

        Returns:
            The number of physical header lines.
        """
        ...

    @property
    def section_to_canonical(self) -> Mapping[str, CanonicalRole]:
        """Map this style's section header names to canonical role keys.

        Returns:
            A mapping from recognized headers to canonical roles.
        """
        ...

    def supports_canonical_role(self, role: CanonicalRole) -> bool:
        """Return whether this style can emit the canonical role.

        Args:
            role: Canonical role to query.

        Returns:
            Whether the style has a target representation for `role`.
        """
        ...

    @property
    def supports_compound_type_entries(self) -> bool:
        """Whether entries may be compound type lines without a parsed name.

        Returns:
            Whether the style supports compound type-only entries.
        """
        ...

    @property
    def supports_multi_name_entries(self) -> bool:
        """Whether one emitted entry can carry multiple names and re-parse.

        Returns:
            Whether one entry may contain multiple names.
        """
        ...

    @property
    def supports_section_headers(self) -> bool:
        """Whether this style emits section headers.

        True for Google and NumPy; False for Sphinx field lists.

        Returns:
            Whether the style emits section headers.
        """
        ...

    @property
    def uses_field_list_entries(self) -> bool:
        """Whether this style uses field-list entries without section headers.

        Returns:
            Whether the style emits field-list entries.
        """
        ...


def get_style(name: StyleName) -> Style:
    """Look up a built-in style by name.

    Args:
        name: Built-in style name.

    Returns:
        The style implementation for `name`.

    Raises:
        ValueError: If `name` is not a built-in style.
    """
    if name not in _STYLES:
        msg = f'unknown style: {name!r}'
        raise ValueError(msg)
    return _STYLES[name]


def get_styles() -> frozenset[StyleName]:
    """Return the fixed built-in style names.

    Returns:
        The Google, NumPy, and Sphinx style names.
    """
    return frozenset(_STYLES)


def resolve_style_override(style_name: StyleName | None) -> 'Style | None':
    """Resolve a style name to a style implementation, or `None` if unset.

    Args:
        style_name: Built-in style name, or `None` when unset.

    Returns:
        The selected style implementation, or `None` when unset.

    Raises:
        ValueError: If `style_name` does not name a built-in style.
    """
    if style_name is not None:
        return get_style(style_name)
    return None


from ._google import (  # noqa: E402
    STYLE as _GOOGLE_INSTANCE,
)
from ._numpy import (  # noqa: E402
    STYLE as _NUMPY_INSTANCE,
)
from ._sphinx import (  # noqa: E402
    STYLE as _SPHINX_INSTANCE,
)

_STYLES: Mapping[StyleName, Style] = {
    'google': _GOOGLE_INSTANCE,
    'numpy': _NUMPY_INSTANCE,
    'sphinx': _SPHINX_INSTANCE,
}
