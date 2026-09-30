"""Canonical section registry for docstring roles."""

from dataclasses import dataclass

from docwrap._types import CanonicalRole


@dataclass(frozen=True, slots=True)
class CanonicalSection:
    """Registry entry whose tuple position determines canonical section order.

    The entry's position in `CANONICAL_SECTIONS` is the ordering authority.
    Sphinx field mappings and target support belong to the Sphinx style.

    Attributes:
        role: Canonical docstring role represented by this entry.
        google_header: Google header label, or `None` when unavailable.
        numpy_header: NumPy header label, or `None` when unavailable.
        return_like: Whether the role uses return-like entry handling.
    """

    role: CanonicalRole
    google_header: str | None = None
    numpy_header: str | None = None
    return_like: bool = False


CANONICAL_SECTIONS: tuple[CanonicalSection, ...] = (
    CanonicalSection('param', google_header='Args', numpy_header='Parameters'),
    CanonicalSection('keyword_param', google_header='Keyword Args'),
    CanonicalSection('other_param', numpy_header='Other Parameters'),
    CanonicalSection(
        'returns',
        google_header='Returns',
        numpy_header='Returns',
        return_like=True,
    ),
    CanonicalSection(
        'yields',
        google_header='Yields',
        numpy_header='Yields',
        return_like=True,
    ),
    CanonicalSection('raises', google_header='Raises', numpy_header='Raises'),
    CanonicalSection('warns', google_header='Warns', numpy_header='Warns'),
    CanonicalSection(
        'attributes',
        google_header='Attributes',
        numpy_header='Attributes',
    ),
    CanonicalSection('methods', google_header='Methods', numpy_header='Methods'),
    CanonicalSection('see_also', google_header='See Also', numpy_header='See Also'),
    CanonicalSection('notes', google_header='Notes', numpy_header='Notes'),
    CanonicalSection('references', google_header='References', numpy_header='References'),
    CanonicalSection('examples', google_header='Examples', numpy_header='Examples'),
    CanonicalSection('todo', google_header='Todo', numpy_header='Todo'),
    CanonicalSection('warnings', google_header='Warnings', numpy_header='Warnings'),
)

CANONICAL_ORDER: tuple[CanonicalRole, ...] = tuple(section.role for section in CANONICAL_SECTIONS)

CANONICAL_ORDER_SET: frozenset[CanonicalRole] = frozenset(CANONICAL_ORDER)

RETURN_LIKE_ROLES: frozenset[CanonicalRole] = frozenset(
    s.role for s in CANONICAL_SECTIONS if s.return_like
)
