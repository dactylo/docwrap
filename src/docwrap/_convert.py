"""Style conversion engine for docstring format transformation."""

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from docwrap._canonical import CANONICAL_ORDER, CANONICAL_ORDER_SET, RETURN_LIKE_ROLES
from docwrap._core._parse import Region, is_structural_underline
from docwrap._styles import EntryMatch, Style, get_style
from docwrap._types import CanonicalRole, SphinxFieldRole

if TYPE_CHECKING:
    _SphinxGroupResult = tuple[
        list[Region],
        dict[str, list[Region]],
        list[tuple[str | None, list[Region]]],
        list[Region],
    ]

_CONVERTIBLE_ROLES: frozenset[CanonicalRole | SphinxFieldRole] = frozenset(
    {
        'attributes',
        'ivar',
        'keyword_param',
        'other_param',
        'param',
        'raises',
        'returns',
        'warns',
        'yields',
    }
)

_IVAR_ROLES: frozenset[SphinxFieldRole] = frozenset({'ivar'})
_PARAM_KEYWORD_ROLES: frozenset[SphinxFieldRole] = frozenset({'keyword', 'param'})
_PRESERVED_SPHINX_FIELD_ROLES: frozenset[SphinxFieldRole] = frozenset({'cvar', 'var'})


class DuplicateFieldError(ValueError):
    """Raised when duplicate fields cannot be represented losslessly.

    This includes conflicting Sphinx type fields, multiple typeless `returns`
    or `yields` fields, and unrepresentable duplicates produced by type
    stripping.
    """


class UnrepresentableReturnError(ValueError):
    """Raised when a named return-like entry cannot round-trip."""


def _apply_sphinx_opaque_owners(
    regions: 'list[Region]',
    owners: list[int | None],
) -> 'list[Region]':
    """Mark owned regions and separate adjacent distinct owners."""
    result: list[Region] = []
    for index, (region, owner) in enumerate(zip(regions, owners, strict=True)):
        if (
            owner is not None
            and index
            and owners[index - 1] is not None
            and owners[index - 1] != owner
        ):
            result.append(Region(kind='blank', text=''))
        result.append(replace(region, preserve_verbatim=True) if owner is not None else region)
    return result


def _assign_sphinx_opaque_owners(
    regions: 'list[Region]',
    strip_types: bool,
    vartype_mergeable_by_name: dict[str, bool],
) -> list[int | None]:
    """Assign one owner identity to each authored opaque field family."""
    owners: list[int | None] = []
    active_preserved_names: set[str] = set()
    current_owner: int | None = None
    next_owner = 0
    for region in regions:
        if region.kind != 'entry' or region.entry_match is None:
            owners.append(current_owner)
            continue
        name = region.entry_match.name
        if region.role in _PRESERVED_SPHINX_FIELD_ROLES:
            if not active_preserved_names:
                current_owner = next_owner
                next_owner += 1
            active_preserved_names.add(name)
        elif (
            region.role == 'vartype'
            and not strip_types
            and not vartype_mergeable_by_name.get(name, False)
        ):
            if name not in active_preserved_names:
                active_preserved_names.clear()
                current_owner = next_owner
                next_owner += 1
        else:
            active_preserved_names.clear()
            current_owner = None
        owners.append(current_owner)
    return owners


def _collect_merge_targets(
    regions: 'list[Region]',
) -> 'tuple[Counter[str], Counter[str], set[str], set[str], int, bool]':
    """Collect merge target counts and occupied inline slots."""
    param_counts: Counter[str] = Counter()
    var_counts: Counter[str] = Counter()
    param_inline: set[str] = set()
    var_inline: set[str] = set()
    returns_count = 0
    returns_inline = False
    for region in regions:
        if region.preserve_verbatim or region.kind != 'entry' or region.entry_match is None:
            continue
        em = region.entry_match
        if em.role in _PARAM_KEYWORD_ROLES:
            param_counts[em.name] += 1
            if em.type_ann:
                param_inline.add(em.name)
        elif em.role == 'returns':
            returns_count += 1
            if em.type_ann:
                returns_inline = True
        elif em.role in _IVAR_ROLES:
            var_counts[em.name] += 1
            if em.type_ann:
                var_inline.add(em.name)
    return param_counts, var_counts, param_inline, var_inline, returns_count, returns_inline


def _collect_sphinx_groups(
    regions: 'list[Region]',
    source_style: 'Style',
) -> '_SphinxGroupResult':
    """Partition Sphinx regions into canonical groups with between-group content."""
    pre_entry: list[Region] = []
    groups: dict[str, list[Region]] = {}
    between: list[tuple[str | None, list[Region]]] = []
    unmapped: list[Region] = []
    seen_entry = False
    in_unmapped = False
    last_key: str | None = None
    pending: list[Region] = []
    canonical_set = CANONICAL_ORDER_SET

    for region in regions:
        if region.kind == 'entry' and region.entry_match is not None:
            seen_entry = True
            key = source_style.canonicalize_role(region.entry_match.role)
            if key in canonical_set:
                if pending:
                    if last_key is not None and key == last_key:
                        groups[last_key].extend(pending)
                    else:
                        between.append((last_key, pending))
                    pending = []
                groups.setdefault(key, []).append(region)
                last_key = key
                in_unmapped = False
            else:
                if pending:
                    between.append((last_key, pending))
                    pending = []
                unmapped.append(region)
                in_unmapped = True
        elif not seen_entry:
            pre_entry.append(region)
        else:
            (unmapped if in_unmapped else pending).append(region)

    # Trailing pending content after the last entry
    if pending and last_key is not None:
        groups[last_key].extend(pending)
    elif pending:
        unmapped.extend(pending)

    return pre_entry, groups, between, unmapped


def _collect_sphinx_type_maps(
    regions: 'list[Region]',
) -> 'tuple[dict[str, str], str | None, dict[str, str]]':
    """Scan regions to build type/rtype/vartype maps."""
    type_map: dict[str, str] = {}
    rtype_value: str | None = None
    vartype_map: dict[str, str] = {}

    for region in regions:
        if region.preserve_verbatim or region.kind != 'entry' or region.entry_match is None:
            continue
        em = region.entry_match
        if em.role == 'type':
            type_map[em.name] = em.description
        elif em.role == 'rtype':
            rtype_value = em.description
        elif em.role == 'vartype':
            vartype_map[em.name] = em.description

    return type_map, rtype_value, vartype_map


def _collect_sphinx_vartype_mergeability(
    regions: 'list[Region]',
) -> dict[str, bool]:
    """Collect whether each authored field name has an ivar merge target."""
    vartype_mergeable_by_name: dict[str, bool] = {}
    for region in regions:
        if region.kind != 'entry' or region.entry_match is None:
            continue
        name = region.entry_match.name
        if region.role in _PRESERVED_SPHINX_FIELD_ROLES:
            vartype_mergeable_by_name.setdefault(name, False)
        elif region.role in _IVAR_ROLES:
            vartype_mergeable_by_name[name] = True
    return vartype_mergeable_by_name


def _convert_header(
    region: 'Region',
    source_style: 'Style',
    target_style: 'Style',
) -> 'list[Region]':
    """Convert a section header region to the target style."""
    if region.preserve_verbatim:
        return [region]
    key = source_style.section_to_canonical.get(region.section_name or '', None)
    if key is None:
        return [region]
    if not target_style.supports_section_headers:
        return []
    header_lines = target_style.format_section_header(key)
    if not header_lines:
        # Preserve original header when target supports headers but lacks a
        # mapping for this particular canonical key (e.g. keyword_param in NumPy).
        return [region]
    display_name = header_lines[0].rstrip(':')
    return [
        Region(kind='section_header', text=line, section_name=display_name)
        for line in header_lines
    ]


def _delete_type_fields(
    regions: 'list[Region]',
    deleted_text: 'list[str] | None',
    deleted_entry_roles: set[str] | None = None,
) -> 'list[Region]':
    """Delete type-only fields and record matched entry-role provenance."""
    has_returns = any(
        region.kind == 'entry'
        and region.entry_match is not None
        and region.entry_match.role == 'returns'
        for region in regions
    )
    result: list[Region] = []
    just_deleted = False
    for region in regions:
        is_type_field = (
            not region.preserve_verbatim
            and region.kind == 'entry'
            and region.entry_match is not None
            and region.entry_match.role in ('type', 'rtype', 'vartype')
        )
        if is_type_field:
            if (
                deleted_entry_roles is not None
                and region.entry_match is not None
                and region.entry_match.role == 'rtype'
                and has_returns
            ):
                deleted_entry_roles.add('returns')
            if deleted_text is not None:
                deleted_text.append(region.text)
            just_deleted = True
            continue
        if just_deleted and region.kind == 'blank' and result and result[-1].kind == 'blank':
            just_deleted = False
            continue
        just_deleted = False
        result.append(region)
    return result


def _entry_names_for_roles(regions: 'list[Region]', roles: frozenset[str]) -> set[str]:
    """Collect entry names for regions matching the given roles."""
    return {
        r.entry_match.name
        for r in regions
        if (
            not r.preserve_verbatim
            and r.kind == 'entry'
            and r.entry_match is not None
            and r.role in roles
        )
    }


def _expand_sphinx_opaque_prose(
    regions: 'list[Region]',
    owners: list[int | None],
    span_starts: list[int],
) -> None:
    """Extend opaque owners backward across unsafe contiguous prose."""
    for index in span_starts:
        owner = owners[index]
        cursor = index - 1
        while (
            cursor >= 0
            and owners[cursor] is None
            and regions[cursor].kind in ('description', 'prose')
            and not is_structural_underline(regions[cursor].text)
        ):
            owners[cursor] = owner
            cursor -= 1


def _generated_entry_region(
    line: str,
    target_style: 'Style',
    role: CanonicalRole | SphinxFieldRole,
) -> 'Region':
    """Build a generated entry with its target semantic role."""
    match = target_style.match_entry(line, role=role)
    target_role = match.role if match is not None and match.role else role
    return Region(kind='entry', text=line, entry_match=None, role=target_role)


def _generated_return_prose_region(line: str) -> 'Region':
    """Build generated description-only return prose."""
    indent = len(line) - len(line.lstrip())
    return Region(kind='prose', text=line[indent:], indent=indent)


def _convert_entry(
    region: 'Region',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
    deleted_text: list[str] | None,
) -> 'list[Region]':
    """Convert a single entry region with a parsed entry_match."""
    role = source_style.canonicalize_role(region.role)
    if role not in _CONVERTIBLE_ROLES:
        return [region]

    em = region.entry_match
    if em is None:
        raise AssertionError

    # Returns/Yields normalization: source stored type in name field
    if role in RETURN_LIKE_ROLES and em.name and not em.type_ann:
        em = replace(em, name='', type_ann=em.name)

    if strip_types and em.type_ann and deleted_text is not None:
        deleted_text.append(em.type_ann)

    # Determine target role
    target_role = target_style.format_role(role)

    formatted_lines = target_style.format_entry(em, target_role, strip_types)

    # Type-less returns/yields in Google/NumPy become description-only text.
    # Emit as prose so _wrap_regions wraps them (entry kind would bypass wrapping
    # since _wrap_entry can't parse description-only text as an entry).
    if target_style.emits_description_only_return_as_prose(role, em, strip_types):
        return [
            _generated_return_prose_region(physical_line)
            for formatted_line in formatted_lines
            for physical_line in formatted_line.split('\n')
        ]

    return [_generated_entry_region(line, target_style, role) for line in formatted_lines]


def _group_sphinx_segment(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
) -> 'list[Region]':
    """Group one representable Sphinx segment without crossing opaque spans."""
    pre_entry, groups, between, unmapped = _collect_sphinx_groups(regions, source_style)

    # Emit in canonical order with section headers and blank separators
    result: list[Region] = list(pre_entry)
    prev_emitted = False
    for key in CANONICAL_ORDER:
        if key not in groups:
            continue
        if prev_emitted and not (result and result[-1].kind == 'blank'):
            result.append(Region(kind='blank', text=''))
        header_lines = target_style.format_section_header(key)
        if header_lines:
            display_name = header_lines[0].rstrip(':')
            result.extend(
                Region(kind='section_header', text=line, section_name=display_name)
                for line in header_lines
            )
        result.extend(groups[key])
        # Emit between-content that followed this group
        for prev_key, content in between:
            if prev_key == key:
                result.extend(content)
        prev_emitted = True

    # Unmapped entries at the end
    result.extend(unmapped)
    return result


def _group_sphinx_entries(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
) -> 'list[Region]':
    """Group Sphinx entries without moving target regions across opaque spans."""
    result: list[Region] = []
    segment: list[Region] = []
    for region in regions:
        if region.preserve_verbatim:
            if segment:
                result.extend(_group_sphinx_segment(segment, source_style, target_style))
                segment = []
            result.append(region)
        else:
            segment.append(region)
    if segment:
        result.extend(_group_sphinx_segment(segment, source_style, target_style))
    return result


def _merge_type_into_entry(
    region: 'Region',
    type_map: dict[str, str],
    rtype_value: str | None,
    vartype_map: dict[str, str],
) -> 'Region':
    """Merge a type annotation into a param, returns, or ivar entry."""
    em = region.entry_match
    if em is None:
        raise AssertionError
    if em.role in ('param', 'keyword') and em.name in type_map:
        return replace(region, entry_match=replace(em, type_ann=type_map[em.name]))
    if em.role == 'returns' and rtype_value is not None:
        return replace(region, entry_match=replace(em, type_ann=rtype_value))
    if em.role in _IVAR_ROLES and em.name in vartype_map:
        return replace(region, entry_match=replace(em, type_ann=vartype_map[em.name]))
    return region


def _preserve_unheaded_entries(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
) -> 'list[Region]':
    """Preserve entries whose canonical role has no target section header."""
    preserved_names: set[str] = set()
    marked: list[Region] = []
    for region in regions:
        marked_region = region
        if (
            not region.preserve_verbatim
            and region.kind == 'entry'
            and region.entry_match is not None
        ):
            key = source_style.canonicalize_role(region.entry_match.role)
            if key in CANONICAL_ORDER_SET and not target_style.format_section_header(key):
                if region.entry_match.name:
                    preserved_names.add(region.entry_match.name)
                marked_region = replace(region, preserve_verbatim=True)
        marked.append(marked_region)
    if not preserved_names:
        return marked
    return [
        replace(region, preserve_verbatim=True)
        if (
            not region.preserve_verbatim
            and region.kind == 'entry'
            and region.entry_match is not None
            and region.entry_match.role == 'type'
            and region.entry_match.name in preserved_names
        )
        else region
        for region in marked
    ]


def _record_authored_header(
    region: 'Region',
    source_style: 'Style',
    source_headers: list[str],
) -> None:
    """Record an authored source header fragment for the current section."""
    if not source_style.uses_field_list_entries:
        source_headers.append(region.text)


def _record_entry_deletion(
    region: 'Region',
    entry_deleted: bool,
    upstream_entry_deleted: bool,
    deleted_text: list[str] | None,
) -> bool:
    """Record upstream entry provenance and return whether deletion occurred."""
    if upstream_entry_deleted and not entry_deleted and deleted_text is not None:
        deleted_text.append(region.text)
    return entry_deleted or upstream_entry_deleted


def _reindent_compound(region: 'Region', source_style: 'Style', target_style: 'Style') -> 'Region':
    """Re-indent a compound-type entry (no entry_match) for the target style."""
    if region.preserve_verbatim:
        return region
    if source_style.name == target_style.name:
        return region
    delta = target_style.section_body_indent - source_style.section_body_indent
    if delta == 0:
        return region
    reindented: list[str] = []
    for line in region.text.split('\n'):
        if not line.strip():
            reindented.append(line)
        elif delta > 0:
            reindented.append(' ' * delta + line)
        else:
            strip_amount = min(-delta, len(line) - len(line.lstrip()))
            reindented.append(line[strip_amount:])
    return Region(
        kind='entry',
        text='\n'.join(reindented),
        entry_match=None,
        role=region.role,
    )


def _reject_multiple_typeless_returns(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
) -> None:
    """Reject Sphinx Returns or Yields groups that the target cannot preserve."""
    if not source_style.uses_field_list_entries or not target_style.supports_section_headers:
        return

    saw_typeless: set[str] = set()
    for region in regions:
        if region.preserve_verbatim:
            saw_typeless.clear()
            continue
        if region.kind == 'entry' and region.entry_match is not None:
            key = source_style.canonicalize_role(region.role)
            if key in ('returns', 'yields') and not region.entry_match.type_ann:
                if key in saw_typeless:
                    msg = f'target cannot represent multiple typeless {key}'
                    raise DuplicateFieldError(msg)
                saw_typeless.add(key)


def _reject_slot_conflict(
    role: str,
    name: str,
    target_count: int,
    target_has_inline: bool,
    seen: set[str],
) -> None:
    """Raise DuplicateFieldError for a non-conforming matched type field."""
    if name in seen:
        msg = f'duplicate :{role} {name}: fields cannot be converted losslessly'
        raise DuplicateFieldError(msg)
    if target_count > 1:
        msg = f':{role} {name}: matches more than one target entry'
        raise DuplicateFieldError(msg)
    if target_has_inline:
        msg = f':{role} {name}: conflicts with an inline annotation'
        raise DuplicateFieldError(msg)
    seen.add(name)


def _check_merge_conflicts(regions: 'list[Region]') -> None:
    """Reject type fields that cannot merge into exactly one empty slot."""
    (
        param_counts,
        var_counts,
        param_inline,
        var_inline,
        returns_count,
        returns_inline,
    ) = _collect_merge_targets(regions)

    seen_type: set[str] = set()
    seen_vartype: set[str] = set()
    rtype_seen = False
    for region in regions:
        if region.preserve_verbatim or region.kind != 'entry' or region.entry_match is None:
            continue
        em = region.entry_match
        if em.role == 'type' and param_counts[em.name]:
            _reject_slot_conflict(
                'type',
                em.name,
                param_counts[em.name],
                em.name in param_inline,
                seen_type,
            )
        elif em.role == 'rtype':
            if rtype_seen:
                msg = 'duplicate :rtype: fields cannot be converted losslessly'
                raise DuplicateFieldError(msg)
            if returns_count > 1:
                msg = ':rtype: matches more than one target entry'
                raise DuplicateFieldError(msg)
            if returns_inline:
                msg = ':rtype: conflicts with an inline annotation'
                raise DuplicateFieldError(msg)
            rtype_seen = True
        elif em.role == 'vartype' and var_counts[em.name]:
            _reject_slot_conflict(
                'vartype',
                em.name,
                var_counts[em.name],
                em.name in var_inline,
                seen_vartype,
            )


def _reject_unrepresentable_named_returns(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
) -> None:
    """Reject conversion that cannot preserve a return-like entry name."""
    for region in regions:
        if region.preserve_verbatim or region.kind != 'entry' or region.entry_match is None:
            continue
        role = source_style.canonicalize_role(region.role)
        match = region.entry_match
        if role not in RETURN_LIKE_ROLES or not match.name or not match.type_ann:
            continue
        if target_style.uses_field_list_entries and role == 'returns':
            operation = 'conversion to Sphinx'
        elif strip_types and source_style.supports_section_headers:
            operation = 'type stripping'
        else:
            continue
        msg = (
            f'named {role.title()} entry {match.name!r} cannot preserve its semantic name during '
            f'{operation}'
        )
        raise UnrepresentableReturnError(msg)


def _release_sphinx_opaque_blank_boundaries(
    regions: 'list[Region]',
    owners: list[int | None],
    span_starts: list[int],
) -> None:
    """Leave authored blanks outside adjacent distinct opaque owners."""
    for index in span_starts:
        owner = owners[index]
        cursor = index - 1
        while (
            cursor >= 0
            and regions[cursor].kind == 'blank'
            and owners[cursor] is not None
            and owners[cursor] != owner
        ):
            owners[cursor] = None
            cursor -= 1


def _resolve_type_region(
    region: 'Region',
    rtype_value: str | None,
    param_names: set[str],
    var_names: set[str],
    has_returns: bool,
) -> 'Region | None':
    """Resolve a :type:/:rtype:/:vartype: region after merge pass."""
    em = region.entry_match
    if em is None:
        raise AssertionError
    if em.role == 'type':
        # Orphaned :type: (no matching param) -- preserve; otherwise drop
        return region if em.name not in param_names else None
    if em.role == 'rtype':
        if not has_returns and rtype_value is not None:
            synth_em = EntryMatch(name='', type_ann=rtype_value, description='', role='returns')
            return Region(
                kind='entry',
                text=region.text,
                entry_match=synth_em,
                role='returns',
                source_line_count=region.source_line_count,
            )
        return None
    if em.role == 'vartype':
        return region if em.name not in var_names else None
    return region


def _merge_sphinx_types(
    regions: 'list[Region]',
    strip_types: bool = False,
    deleted_text: 'list[str] | None' = None,
    *,
    deleted_entry_roles: set[str] | None = None,
) -> 'list[Region]':
    """Resolve :type:/:rtype:/:vartype: fields by merging or strip-deleting."""
    if strip_types:
        return _delete_type_fields(regions, deleted_text, deleted_entry_roles)
    _check_merge_conflicts(regions)

    type_map, rtype_value, vartype_map = _collect_sphinx_type_maps(regions)
    param_names = _entry_names_for_roles(regions, _PARAM_KEYWORD_ROLES)
    var_names = _entry_names_for_roles(regions, _IVAR_ROLES)
    has_returns = any(
        r.kind == 'entry' and r.entry_match is not None and r.role == 'returns' for r in regions
    )

    result: list[Region] = []
    for region in regions:
        if region.preserve_verbatim:
            result.append(region)
            continue
        if region.kind != 'entry' or region.entry_match is None:
            result.append(region)
            continue

        role = region.entry_match.role

        # Merge type annotation into matching entry
        if role in ('param', 'keyword', 'returns') or role in _IVAR_ROLES:
            result.append(_merge_type_into_entry(region, type_map, rtype_value, vartype_map))
            continue

        # Handle type-only regions (type/rtype/vartype)
        merged = _resolve_type_region(region, rtype_value, param_names, var_names, has_returns)
        if merged is not None:
            result.append(merged)
            if merged.role == 'returns' and merged is not region:
                has_returns = True
            continue

    return result


def _resolve_type_stripped_entry(
    region: 'Region',
    converted_entry: 'list[Region]',
    source_style: 'Style',
    deleted_entry_roles: set[str],
) -> 'tuple[list[Region], bool]':
    """Drop a description-less entry left empty by an upstream stripped type field."""
    if region.entry_match is None:
        return converted_entry, False
    role_type_stripped = source_style.canonicalize_role(region.role) in deleted_entry_roles
    if role_type_stripped and not region.entry_match.description:
        # A description-less :returns: whose only content was a stripped :rtype:
        # is meaningless. Field-list targets would re-emit a bare :returns:; drop
        # it for parity with Google/NumPy targets.
        converted_entry = []
    return converted_entry, not converted_entry and role_type_stripped


def _section_has_content(regions: 'list[Region]', start: int, end: int) -> bool:
    """Return whether a region span contains retained section content."""
    return any(regions[j].kind not in ('blank', 'description') for j in range(start, end))


def _record_unemitted_header_deletion(
    regions: 'list[Region]',
    section_start: int | None,
    section_deleted: bool,
    source_headers: list[str],
    deleted_text: list[str] | None,
) -> None:
    """Record an emptied source section whose target emits no header."""
    if (
        section_start is not None
        and section_deleted
        and not _section_has_content(regions, section_start, len(regions))
        and deleted_text is not None
    ):
        deleted_text.extend(source_headers)


def _remove_emptied_sections(
    regions: 'list[Region]',
    deleted_sections: dict[int, tuple[int, tuple[str, ...]]],
    deleted_text: list[str] | None,
) -> 'list[Region]':
    """Remove header runs whose strip-deleted section retains no content."""
    if not deleted_sections:
        return regions
    result: list[Region] = []
    i = 0
    while i < len(regions):
        region = regions[i]
        deletion = deleted_sections.get(i)
        if region.kind != 'section_header' or deletion is None:
            result.append(region)
            i += 1
            continue
        run_end, source_headers = deletion
        extent_end = run_end
        while extent_end < len(regions) and regions[extent_end].kind != 'section_header':
            extent_end += 1
        if _section_has_content(regions, run_end, extent_end):
            result.extend(regions[i:run_end])
            i = run_end
            continue
        if (
            result
            and result[-1].kind == 'blank'
            and run_end < len(regions)
            and regions[run_end].kind == 'blank'
        ):
            result.pop()
        if deleted_text is not None:
            deleted_text.extend(source_headers)
        i = run_end
    return result


def _sphinx_opaque_owner_starts(owners: list[int | None]) -> list[int]:
    """Return the first region index for each opaque owner run."""
    return [
        index
        for index, owner in enumerate(owners)
        if owner is not None and (index == 0 or owners[index - 1] != owner)
    ]


def _preserve_unmapped_sphinx_fields(
    regions: 'list[Region]',
    strip_types: bool,
) -> 'list[Region]':
    """Shield target-unrepresentable Sphinx fields and unsafe boundaries."""
    mergeability = _collect_sphinx_vartype_mergeability(regions)
    owners = _assign_sphinx_opaque_owners(regions, strip_types, mergeability)
    span_starts = _sphinx_opaque_owner_starts(owners)
    _expand_sphinx_opaque_prose(regions, owners, span_starts)
    _release_sphinx_opaque_blank_boundaries(regions, owners, span_starts)
    return _apply_sphinx_opaque_owners(regions, owners)


def _preserve_unsupported_sections(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
) -> 'list[Region]':
    """Mark complete authored section spans the target cannot represent."""
    if source_style.uses_field_list_entries:
        if target_style.uses_field_list_entries:
            return regions
        return _preserve_unmapped_sphinx_fields(regions, strip_types)

    result: list[Region] = []
    index = 0
    while index < len(regions):
        region = regions[index]
        if region.kind != 'section_header' or region.section_name is None:
            result.append(region)
            index += 1
            continue

        section_name = region.section_name
        header_end = index + 1
        while (
            header_end < len(regions)
            and regions[header_end].kind == 'section_header'
            and regions[header_end].section_name == section_name
        ):
            header_end += 1

        span_end = header_end
        while span_end < len(regions) and regions[span_end].kind != 'section_header':
            span_end += 1

        canonical_role = source_style.section_to_canonical.get(section_name)
        target_supports_role = canonical_role is not None and target_style.supports_canonical_role(
            canonical_role
        )
        has_convertible_entry = any(
            item.kind == 'entry'
            and item.entry_match is not None
            and source_style.canonicalize_role(item.role) == canonical_role
            for item in regions[header_end:span_end]
        )
        unsupported = not target_supports_role or (
            target_style.uses_field_list_entries and not has_convertible_entry
        )
        span = regions[index:span_end]
        if unsupported:
            result.extend(replace(item, preserve_verbatim=True) for item in span)
        else:
            result.extend(span)
        index = span_end

    return result


def _strip_types_entry(
    region: 'Region',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
    deleted_text: list[str] | None,
) -> 'tuple[list[Region], bool] | None':
    """Handle strip-types special cases for return-like entry regions."""
    if not strip_types:
        return None

    em = region.entry_match
    if em is None:
        raise AssertionError
    role = source_style.canonicalize_role(region.role)

    if role not in RETURN_LIKE_ROLES:
        return None

    # Drop description-less returns/yields when type is the only content.
    deleted_type = em.type_ann or em.name
    if not em.description and deleted_type:
        if deleted_text is not None:
            deleted_text.append(region.text)
        return [], True

    # Returns/Yields: type stored in em.name (Google/NumPy pattern)
    # becomes prose since the type marker is removed.
    # Exception: when targeting Sphinx, fall through to _convert_entry
    # which emits :returns: desc (suppressing :rtype: via strip_types).
    if em.name and not em.type_ann and not target_style.uses_field_list_entries:
        if deleted_text is not None:
            deleted_text.append(em.name)
        formatted_lines = target_style.format_entry(
            em,
            target_style.format_role(role),
            strip_types,
        )
        return (
            [
                _generated_return_prose_region(physical_line)
                for formatted_line in formatted_lines
                for physical_line in formatted_line.split('\n')
            ],
            False,
        )

    return None


def _strip_types_preserve_lines(
    region: 'Region',
    source_style: 'Style',
    target_style: 'Style',
    deleted_text: list[str] | None,
) -> 'Region | None':
    """Strip type annotation from an entry while preserving original line structure."""
    em = region.entry_match
    if em is None or region.original_lines is None:
        return None

    role = source_style.canonicalize_role(region.role)

    # Returns/Yields handled by _strip_types_entry or _convert_entry
    if role in RETURN_LIKE_ROLES:
        return None

    # No type to strip -- pass through unchanged
    if not em.type_ann:
        return region

    # Parse the first original line to get its description text (before join)
    first = region.original_lines[0]
    first_em = target_style.match_entry(first)
    if first_em is None:
        return None

    # Use format_entry with first-line description only (not the joined one)
    first_line_em = EntryMatch(
        name=em.name,
        type_ann='',
        description=first_em.description,
        role=em.role,
    )
    target_role = target_style.format_role(cast('CanonicalRole | SphinxFieldRole', role))
    formatted = target_style.format_entry(first_line_em, target_role, strip_types=True)
    if not formatted:
        return None
    new_first = formatted[0]

    new_lines = (new_first, *region.original_lines[1:])
    new_em = replace(em, type_ann='')
    if deleted_text is not None:
        deleted_text.append(em.type_ann)
    return Region(
        kind='entry',
        text='\n'.join(new_lines),
        source_line_count=len(new_lines),
        entry_match=new_em,
        role=region.role,
        original_lines=new_lines,
    )


def _convert_matched_entry(
    region: 'Region',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
    deleted_text: list[str] | None,
) -> 'tuple[list[Region], bool]':
    """Convert an entry region with a parsed entry_match, dispatching to the appropriate path."""
    if region.preserve_verbatim:
        return [region], False
    converted = _strip_types_entry(region, source_style, target_style, strip_types, deleted_text)
    if converted is not None:
        return converted
    if (
        source_style.name == target_style.name
        and strip_types
        and region.original_lines is not None
    ):
        fast = _strip_types_preserve_lines(region, source_style, target_style, deleted_text)
        if fast is not None:
            return [fast], False
    return _convert_entry(region, source_style, target_style, strip_types, deleted_text), False


def _target_region_separates_opaque_span(
    region: 'Region',
    target_style: 'Style',
) -> bool:
    """Return whether a target region is a parser-safe opaque boundary."""
    if region.kind in ('blank', 'code_block', 'list_item', 'section_header', 'title'):
        return True
    if region.kind != 'entry':
        return False
    first_line = region.text.split('\n', maxsplit=1)[0]
    return target_style.match_entry(first_line, role=region.role) is not None


def _needs_sphinx_opaque_boundary(
    left: 'Region',
    right: 'Region',
    target_style: 'Style',
) -> bool:
    """Return whether adjacent target regions need an opaque boundary."""
    if left.preserve_verbatim == right.preserve_verbatim:
        return False
    opaque = left if left.preserve_verbatim else right
    adjacent = right if left.preserve_verbatim else left
    if (
        right.preserve_verbatim
        and left.kind == 'entry'
        and target_style.section_body_indent > right.indent
    ):
        return True
    if left.preserve_verbatim and right.kind == 'section_header':
        return opaque.kind != 'blank'
    return opaque.kind != 'blank' and not _target_region_separates_opaque_span(
        adjacent,
        target_style,
    )


def _seal_sphinx_opaque_boundaries(
    regions: 'list[Region]',
    target_style: 'Style',
) -> 'list[Region]':
    """Insert required boundaries around converted opaque Sphinx spans."""
    result: list[Region] = []
    for region in regions:
        if result and _needs_sphinx_opaque_boundary(result[-1], region, target_style):
            result.append(Region(kind='blank', text=''))
        result.append(region)
    return result


def _trim_trailing_blanks(regions: 'list[Region]') -> 'list[Region]':
    """Return transformed regions without trailing blank separators."""
    end = len(regions)
    while end and regions[end - 1].kind == 'blank' and not regions[end - 1].preserve_verbatim:
        end -= 1
    return regions[:end]


def preserve_multi_name_attribute_spans(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
) -> 'list[Region]':
    """Preserve authored multi-name attribute spans across mixed output.

    Args:
        regions: Parsed source regions.
        source_style: Style used to parse the source regions.
        target_style: Requested output style.

    Returns:
        Regions with unrepresentable or previously preserved attribute spans
        marked opaque.
    """
    numpy_style = get_style('numpy')
    result = list(regions)
    current_role: str | None = None
    index = 0
    while index < len(regions):
        region = regions[index]
        if region.kind == 'section_header':
            current_role = source_style.section_to_canonical.get(region.section_name or '')
        if (
            not region.preserve_verbatim
            and region.kind == 'entry'
            and region.entry_match is not None
            and source_style.canonicalize_role(region.role) == 'attributes'
            and ',' in region.entry_match.name
            and not target_style.supports_multi_name_entries
        ):
            result[index] = replace(region, preserve_verbatim=True)
            index += 1
            continue
        first_line = region.original_lines[0] if region.original_lines else ''
        match = numpy_style.match_entry(first_line, role='attributes')
        mixed_context = current_role == 'attributes' or not source_style.supports_section_headers
        if (
            region.kind in ('description', 'prose')
            and mixed_context
            and match is not None
            and ',' in match.name
        ):
            result[index] = replace(region, preserve_verbatim=True)
            base_indent = len(first_line) - len(first_line.lstrip())
            index += 1
            while index < len(regions):
                continuation = regions[index]
                continuation_line = (
                    continuation.original_lines[0] if continuation.original_lines else ''
                )
                continuation_indent = len(continuation_line) - len(continuation_line.lstrip())
                if not continuation_line.strip() or continuation_indent <= base_indent:
                    break
                result[index] = replace(continuation, preserve_verbatim=True)
                index += 1
            continue
        index += 1
    return result


def _prepare_conversion_regions(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool,
    deleted_text: list[str] | None,
) -> 'tuple[list[Region], set[str]]':
    """Apply whole-span preservation and source-style preprocessing."""
    deleted_entry_roles: set[str] = set()
    regions = _preserve_unsupported_sections(
        regions,
        source_style,
        target_style,
        strip_types,
    )
    regions = preserve_multi_name_attribute_spans(regions, source_style, target_style)
    _reject_unrepresentable_named_returns(
        regions,
        source_style,
        target_style,
        strip_types,
    )
    if not source_style.uses_field_list_entries:
        return regions, deleted_entry_roles
    if not strip_types:
        _check_merge_conflicts(regions)
        if not target_style.uses_field_list_entries:
            regions = _preserve_unheaded_entries(regions, source_style, target_style)
    regions = _merge_sphinx_types(
        regions,
        strip_types,
        deleted_text,
        deleted_entry_roles=deleted_entry_roles,
    )
    if strip_types and not target_style.uses_field_list_entries:
        regions = _preserve_unheaded_entries(regions, source_style, target_style)
    _reject_multiple_typeless_returns(regions, source_style, target_style)
    if target_style.uses_field_list_entries:
        return regions, deleted_entry_roles
    return (
        _group_sphinx_entries(regions, source_style, target_style),
        deleted_entry_roles,
    )


def convert_regions(
    regions: 'list[Region]',
    source_style: 'Style',
    target_style: 'Style',
    strip_types: bool = False,
    *,
    deleted_text: list[str] | None = None,
) -> 'list[Region]':
    """Convert supported regions from one docstring style to another.

    Unsupported sections and spans that must remain opaque are preserved
    verbatim, so the result can retain source-style content.

    Args:
        regions: Parsed regions from the source docstring.
        source_style: The style the regions were parsed with.
        target_style: The style to convert to.
        strip_types: If `True`, omit supported type annotations outside
            preserved verbatim spans.
        deleted_text: Optional accumulator; receives the exact text strings
            that `strip_types` discards.

    Returns:
        A new list of regions converted to the target style where supported,
        with preserved source-style spans retained.

    Raises:
        DuplicateFieldError: If field-list content cannot be represented
            losslessly in the target style, including multiple typeless returns
            or yields.
        UnrepresentableReturnError: If a named return-like entry cannot
            preserve its name during conversion or type stripping.
    """
    regions, deleted_entry_roles = _prepare_conversion_regions(
        regions,
        source_style,
        target_style,
        strip_types,
        deleted_text,
    )

    # Headers synthesized by _group_sphinx_entries are already in target-style
    # format, so resolve them via the target_style table; otherwise use source_style.
    header_source_style = target_style if source_style.uses_field_list_entries else source_style

    result: list[Region] = []
    deleted_sections: dict[int, tuple[int, tuple[str, ...]]] = {}
    current_header_run: tuple[int, int] | None = None
    current_source_headers: list[str] = []
    current_unemitted_section_start: int | None = None
    section_deleted = False
    entry_deleted = False
    for region in regions:
        deleted_before_region = entry_deleted
        entry_deleted = False
        if region.kind == 'section_header':
            # Skip NumPy dashes line -- the name line already handled conversion
            stripped = region.text.strip()
            if not region.preserve_verbatim and stripped and set(stripped) == {'-'}:
                _record_authored_header(region, source_style, current_source_headers)
                continue
            _record_unemitted_header_deletion(
                result,
                current_unemitted_section_start,
                section_deleted,
                current_source_headers,
                deleted_text,
            )
            converted_header = _convert_header(region, header_source_style, target_style)
            header_start = len(result)
            result.extend(converted_header)
            current_header_run = (
                (header_start, len(result))
                if converted_header and converted_header[0].kind == 'section_header'
                else None
            )
            current_unemitted_section_start = header_start if current_header_run is None else None
            current_source_headers = []
            _record_authored_header(region, source_style, current_source_headers)
            section_deleted = False
        elif region.kind == 'entry' and region.entry_match is not None:
            converted_entry, entry_deleted = _convert_matched_entry(
                region,
                source_style,
                target_style,
                strip_types,
                deleted_text,
            )
            converted_entry, upstream_entry_deleted = _resolve_type_stripped_entry(
                region, converted_entry, source_style, deleted_entry_roles
            )
            result.extend(converted_entry)
            entry_deleted = _record_entry_deletion(
                region, entry_deleted, upstream_entry_deleted, deleted_text
            )
            section_deleted = section_deleted or entry_deleted
            if entry_deleted and current_header_run is not None:
                header_start, header_end = current_header_run
                deleted_sections[header_start] = (header_end, tuple(current_source_headers))
        elif region.kind == 'entry' and region.entry_match is None:
            result.append(_reindent_compound(region, source_style, target_style))
        else:
            if (
                deleted_before_region
                and region.kind == 'blank'
                and result
                and result[-1].kind == 'blank'
            ):
                continue
            result.append(region)

    _record_unemitted_header_deletion(
        result,
        current_unemitted_section_start,
        section_deleted,
        current_source_headers,
        deleted_text,
    )
    result = _remove_emptied_sections(result, deleted_sections, deleted_text)
    if source_style.uses_field_list_entries and not target_style.uses_field_list_entries:
        result = _seal_sphinx_opaque_boundaries(result, target_style)
    return _trim_trailing_blanks(result)
