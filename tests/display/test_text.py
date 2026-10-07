from docwrap._display._text import pluralize


class TestPluralize:
    def test_plural(self) -> None:
        assert pluralize('file', 2) == 'files'

    def test_singular(self) -> None:
        assert pluralize('file', 1) == 'file'

    def test_zero(self) -> None:
        assert pluralize('file', 0) == 'files'
