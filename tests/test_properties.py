import ast
import keyword
import string
from dataclasses import dataclass
from typing import cast, get_args

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from docwrap import (
    DocwrapConfig,
    FillStyle,
    ReflowMode,
    StyleName,
    SummaryStart,
    wrap_source_report,
)

_FILL_STYLES = st.sampled_from(cast('tuple[FillStyle, ...]', get_args(FillStyle)))
_NAMES = st.from_regex(r'[a-z][a-z0-9_]{0,8}', fullmatch=True).filter(
    lambda name: not keyword.iskeyword(name) and not keyword.issoftkeyword(name)
)
_REFLOW_MODES = st.sampled_from(cast('tuple[ReflowMode, ...]', get_args(ReflowMode)))
_STYLE_NAMES = st.sampled_from(cast('tuple[StyleName, ...]', get_args(StyleName)))
_SUMMARY_STARTS = st.sampled_from(cast('tuple[SummaryStart, ...]', get_args(SummaryStart)))
_WORDS = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12)


@dataclass(frozen=True, slots=True)
class _Case:
    source: str
    summary: str
    paragraphs: tuple[str, ...]
    entries: tuple[tuple[str, str], ...]
    returns: str | None


def _function_docstring(source: str) -> str:
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    docstring = ast.get_docstring(function, clean=True)
    assert docstring is not None
    return docstring


def _sentence(max_words: int = 25) -> st.SearchStrategy[str]:
    return st.lists(_WORDS, min_size=1, max_size=max_words).map(
        lambda words: ' '.join(words) + '.'
    )


@st.composite
def _case(draw: DrawFn) -> _Case:
    summary = draw(_sentence())
    paragraphs = tuple(draw(st.lists(_sentence(60), max_size=2)))
    has_args = draw(st.booleans())
    names = draw(st.lists(_NAMES, max_size=3, unique=True)) if has_args else []
    entries = tuple((name, draw(_sentence())) for name in names)
    returns = draw(st.one_of(st.none(), _sentence()))

    blocks = [summary, *paragraphs]
    if has_args:
        blocks.append(
            '\n    '.join(['Args:', *(f'    {name}: {words}' for name, words in entries)])
        )
    if returns is not None:
        blocks.append(f'Returns:\n        {returns}')
    docstring = '\n\n    '.join(blocks)
    source = f'def f({", ".join(names)}):\n    """{docstring}\n    """\n    return 1\n'
    return _Case(source, summary, paragraphs, entries, returns)


@st.composite
def _config(draw: DrawFn) -> DocwrapConfig:
    return DocwrapConfig(
        line_length=draw(st.integers(min_value=40, max_value=120)),
        fill_style=draw(_FILL_STYLES),
        reflow=draw(_REFLOW_MODES),
        summary_start=draw(_SUMMARY_STARTS),
        wrap_summary=draw(st.booleans()),
    )


def _without_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = ''
    return ast.dump(tree)


def _wrap(source: str, config: DocwrapConfig) -> str:
    return wrap_source_report(source, config=config).source


@given(case=_case(), config=_config())
@settings(deadline=None)
def test_wrapping_is_idempotent(case: _Case, config: DocwrapConfig) -> None:
    once = _wrap(case.source, config)
    assert _wrap(once, config) == once


@given(case=_case(), config=_config())
@settings(deadline=None)
def test_wrapping_changes_only_docstrings(case: _Case, config: DocwrapConfig) -> None:
    assert _without_docstrings(_wrap(case.source, config)) == _without_docstrings(case.source)


@given(case=_case(), config=_config())
@settings(deadline=None)
def test_wrapping_preserves_docstring_words(case: _Case, config: DocwrapConfig) -> None:
    assert (
        _function_docstring(_wrap(case.source, config)).split()
        == _function_docstring(case.source).split()
    )


@given(case=_case(), target=_STYLE_NAMES, line_length=st.integers(min_value=40, max_value=120))
@settings(deadline=None)
def test_conversion_is_idempotent(case: _Case, target: StyleName, line_length: int) -> None:
    config = DocwrapConfig(convert=target, line_length=line_length)
    once = _wrap(case.source, config)
    assert _wrap(once, config) == once


@given(case=_case(), target=_STYLE_NAMES, line_length=st.integers(min_value=40, max_value=120))
@settings(deadline=None)
def test_conversion_keeps_names_and_words_in_order(
    case: _Case, target: StyleName, line_length: int
) -> None:
    converted = _wrap(case.source, DocwrapConfig(convert=target, line_length=line_length))
    expected = case.summary.split()
    for paragraph in case.paragraphs:
        expected.extend(paragraph.split())
    for name, words in case.entries:
        expected.extend((name, *words.split()))
    if case.returns is not None:
        expected.extend(case.returns.split())

    actual = iter(token.strip(':') for token in _function_docstring(converted).split())
    assert all(any(token == word.strip(':') for token in actual) for word in expected)
