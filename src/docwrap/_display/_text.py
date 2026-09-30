"""Text formatting utilities for display output."""


def pluralize(word: str, count: int) -> str:
    """Return singular or plural form of a word.

    Appends 's' for counts other than 1 (standard English pluralization).

    Args:
        word: The singular form (e.g. 'file', 'docstring').
        count: The count determining singular vs plural.

    Returns:
        The singular word when count is one, otherwise the word with `s`
        appended.
    """
    return word if count == 1 else f'{word}s'
