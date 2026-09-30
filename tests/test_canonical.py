from docwrap._canonical import CANONICAL_ORDER


class TestCanonicalOrder:
    def test_canonical_order(self) -> None:
        assert CANONICAL_ORDER[0] == 'param'
        assert CANONICAL_ORDER.index('returns') < CANONICAL_ORDER.index('raises')
