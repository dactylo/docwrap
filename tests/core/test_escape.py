import pytest

from docwrap._core._escape import (
    _SENTINEL_MAX,
    _SENTINEL_MIN,
    EscapeRestoreError,
    EscapeScanError,
    restore_escapes,
    scan_escape_spellings,
    shield_escapes,
)

BACKSLASH = chr(92)


class TestRestoreEscapes:
    def test_adjacent_identity_runs_restore(self) -> None:
        body = BACKSLASH + 'n' + BACKSLASH + 'x41'
        shielded = shield_escapes(body)
        assert restore_escapes(shielded.text, shielded.escape_map) == body

    def test_duplicated_identity_raises(self) -> None:
        shielded = shield_escapes(BACKSLASH + 'n')
        identity = shielded.escape_map[0][0]
        doubled = shielded.text + ' ' + identity * 2
        with pytest.raises(EscapeRestoreError):
            restore_escapes(doubled, shielded.escape_map)

    def test_missing_identity_allowed_only_when_listed(self) -> None:
        shielded = shield_escapes('keep ' + BACKSLASH + 'n drop ' + BACKSLASH + 't')
        keep_identity = shielded.escape_map[0][0]
        drop_identity = shielded.escape_map[1][0]
        text_without_second = 'keep ' + keep_identity * 2 + ' drop'
        restored = restore_escapes(
            text_without_second,
            shielded.escape_map,
            allowed_loss=frozenset({drop_identity}),
        )
        assert restored == 'keep ' + BACKSLASH + 'n drop'

    def test_missing_identity_outside_allowed_set_raises(self) -> None:
        shielded = shield_escapes('keep ' + BACKSLASH + 'n drop ' + BACKSLASH + 't')
        drop_identity = shielded.escape_map[1][0]
        with pytest.raises(EscapeRestoreError):
            restore_escapes(
                'both placeholders gone',
                shielded.escape_map,
                allowed_loss=frozenset({drop_identity}),
            )

    def test_missing_identity_raises_by_default(self) -> None:
        shielded = shield_escapes(BACKSLASH + 'n')
        with pytest.raises(EscapeRestoreError):
            restore_escapes('no placeholders here', shielded.escape_map)

    def test_reordered_identities_stay_attached(self) -> None:
        body = 'ret ' + BACKSLASH + 'n end param ' + BACKSLASH + 't end'
        shielded = shield_escapes(body)
        first, second = shielded.escape_map
        reordered = 'param ' + second[0] * 2 + ' end ret ' + first[0] * 2 + ' end'
        restored = restore_escapes(reordered, shielded.escape_map)
        assert restored == 'param ' + BACKSLASH + 't end ret ' + BACKSLASH + 'n end'

    def test_round_trip(self) -> None:
        body = (
            'mix '
            + BACKSLASH
            + 't of '
            + BACKSLASH
            + 'u0041 escapes '
            + BACKSLASH
            + BACKSLASH
            + ' and '
            + BACKSLASH
            + 'd prose'
        )
        shielded = shield_escapes(body)
        assert restore_escapes(shielded.text, shielded.escape_map) == body

    def test_run_length_mismatch_raises(self) -> None:
        shielded = shield_escapes(BACKSLASH + 'x41')
        identity = shielded.escape_map[0][0]
        with pytest.raises(EscapeRestoreError):
            restore_escapes(identity * 2, shielded.escape_map)

    def test_unrelated_private_use_char_passes_through(self) -> None:
        body = chr(0xE000) + ' with ' + BACKSLASH + 'n'
        shielded = shield_escapes(body)
        assert restore_escapes(shielded.text, shielded.escape_map) == body


class TestScanEscapeSpellings:
    def test_adjacent_escapes(self) -> None:
        body = BACKSLASH + 'n' + BACKSLASH + 't'
        assert scan_escape_spellings(body) == (BACKSLASH + 'n', BACKSLASH + 't')

    def test_doubled_backslash_then_letter(self) -> None:
        body = BACKSLASH + BACKSLASH + 'n'
        assert scan_escape_spellings(body) == (BACKSLASH + BACKSLASH,)

    def test_escaped_quotes(self) -> None:
        body = BACKSLASH + '"' + ' and ' + BACKSLASH + "'"
        assert scan_escape_spellings(body) == (BACKSLASH + '"', BACKSLASH + "'")

    def test_hex_escape(self) -> None:
        assert scan_escape_spellings(BACKSLASH + 'x41z') == (BACKSLASH + 'x41',)

    def test_line_continuation_raises(self) -> None:
        body = 'one ' + BACKSLASH + '\ntwo'
        with pytest.raises(EscapeScanError) as excinfo:
            scan_escape_spellings(body)
        assert excinfo.value.code == 'line-continuation'

    def test_long_unicode_escape(self) -> None:
        spelling = BACKSLASH + 'U0001F600'
        assert scan_escape_spellings('x ' + spelling + ' y') == (spelling,)

    def test_named_escape_with_spaces(self) -> None:
        spelling = BACKSLASH + 'N{LATIN SMALL LETTER A}'
        assert scan_escape_spellings('pre ' + spelling + ' post') == (spelling,)

    def test_no_escapes(self) -> None:
        assert scan_escape_spellings('plain prose only') == ()

    def test_nul_hex_escape(self) -> None:
        assert scan_escape_spellings('a ' + BACKSLASH + 'x00 b') == (BACKSLASH + 'x00',)

    def test_octal_escape_stops_at_non_octal(self) -> None:
        assert scan_escape_spellings(BACKSLASH + '48') == (BACKSLASH + '4',)

    def test_octal_escape_stops_at_three_digits(self) -> None:
        assert scan_escape_spellings(BACKSLASH + '0418') == (BACKSLASH + '041',)

    def test_short_unicode_escape(self) -> None:
        assert scan_escape_spellings(BACKSLASH + 'u0041') == (BACKSLASH + 'u0041',)

    def test_simple_escape(self) -> None:
        assert scan_escape_spellings('a ' + BACKSLASH + 'n b') == (BACKSLASH + 'n',)

    def test_trailing_lone_backslash(self) -> None:
        assert scan_escape_spellings('abc' + BACKSLASH) == (BACKSLASH,)

    def test_unknown_escape(self) -> None:
        assert scan_escape_spellings(BACKSLASH + 'd+') == (BACKSLASH + 'd',)


class TestShieldEscapes:
    def test_distinct_identities_per_occurrence(self) -> None:
        body = BACKSLASH + 'n' + ' and ' + BACKSLASH + 't'
        shielded = shield_escapes(body)
        first, second = shielded.escape_map
        assert first[1] == BACKSLASH + 'n'
        assert second[1] == BACKSLASH + 't'
        assert first[0] != second[0]

    def test_equal_length_placeholders(self) -> None:
        body = 'a ' + BACKSLASH + 'N{LATIN SMALL LETTER A} b ' + BACKSLASH + 'x00'
        shielded = shield_escapes(body)
        assert len(shielded.text) == len(body)

    def test_identity_skips_user_private_use_chars(self) -> None:
        body = chr(0xE000) + ' ' + BACKSLASH + 'n'
        shielded = shield_escapes(body)
        assert shielded.escape_map[0][0] == chr(0xE001)
        assert shielded.text.startswith(chr(0xE000) + ' ')

    def test_named_escape_placeholder_has_no_whitespace(self) -> None:
        body = BACKSLASH + 'N{LATIN SMALL LETTER A}'
        shielded = shield_escapes(body)
        identity = shielded.escape_map[0][0]
        assert shielded.text == identity * len(body)
        assert ' ' not in shielded.text

    def test_no_escapes_returns_body_unchanged(self) -> None:
        shielded = shield_escapes('plain prose')
        assert shielded.text == 'plain prose'
        assert shielded.escape_map == ()

    def test_sentinel_exhaustion_raises(self) -> None:
        body = (
            ''.join(chr(cp) for cp in range(_SENTINEL_MIN, _SENTINEL_MAX + 1))
            + ' '
            + BACKSLASH
            + 'n'
        )
        with pytest.raises(EscapeScanError) as excinfo:
            shield_escapes(body)
        assert excinfo.value.code == 'sentinel-unavailable'
