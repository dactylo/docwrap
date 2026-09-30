import textwrap

from docwrap._detect import detect_style


class TestDetectCodeBlockFiltering:
    def test_style_markers_inside_code_blocks_ignored(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Example:
                >>> print("hello")

            Args:
                x: The value.
        """)
        # Line 3 is a doctest (>>>); "Example:" on line 2 is a Google header,
        # NOT a code block line.  find_code_block_lines returns {3} here.
        # Without filtering, google=2 (Example: + Args:); with filtering,
        # the doctest line is skipped but both headers still count -- same
        # winner, but the filtering path is exercised correctly.
        assert detect_style(content, code_block_lines={3}) == 'google'
        # More importantly, a Sphinx field inside a code block should not count:
        content2 = textwrap.dedent("""\
            Summary.

            Example::

                :param x: inside literal block

            Args:
                y: The value.
        """)
        # Line 4 is the literal block body (indented code).  Lines 2-3 are
        # the introducer ("Example::") and blank separator -- NOT code lines.
        # find_code_block_lines returns {4}.  Without filtering, sphinx=1
        # would tie google=1; with correct filtering, only google=1 remains.
        assert detect_style(content2, code_block_lines={4}) == 'google'


class TestDetectGoogle:
    def test_args_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                x: The value.
        """)
        assert detect_style(content) == 'google'

    def test_multiple_sections(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                x: The value.

            Returns:
                The result.

            Raises:
                ValueError: If bad.
        """)
        assert detect_style(content) == 'google'

    def test_returns_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Returns:
                The result.
        """)
        assert detect_style(content) == 'google'


class TestDetectMixed:
    def test_majority_wins(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                x: The value.

            :param y: Another value.
            :param z: Third value.
        """)
        # Two sphinx markers vs one google marker
        assert detect_style(content) == 'sphinx'

    def test_preferred_style_breaks_tie_only_when_leading(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            value : int
                The value.

            :param other: Another value.
        """)

        assert detect_style(content) is None
        assert detect_style(content, preferred='sphinx') == 'sphinx'
        assert detect_style(content, preferred='google') is None

    def test_preferred_style_does_not_override_unique_winner(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            value : int
                The value.

            Returns
            -------
            int
                The result.

            :param other: Another value.
        """)

        assert detect_style(content, preferred='sphinx') == 'numpy'

    def test_tie_returns_none(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Args:
                x: The value.

            :param y: Another value.
        """)
        # One google marker, one sphinx marker -- tie, no majority
        assert detect_style(content) is None


class TestDetectNone:
    def test_empty(self) -> None:
        assert detect_style('') is None

    def test_summary_and_description(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            A longer description with no sections.
        """)
        assert detect_style(content) is None

    def test_summary_only(self) -> None:
        assert detect_style('Just a summary.') is None


class TestDetectNumpy:
    def test_indented_numpy_header_is_not_detected(self) -> None:
        content = textwrap.dedent("""\
            Summary.

                Parameters
                ----------
                Nested content.
        """)
        assert detect_style(content) != 'numpy'

    def test_parameters_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Parameters
            ----------
            x : int
                The value.
        """)
        assert detect_style(content) == 'numpy'

    def test_returns_section(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            Returns
            -------
            int
                The result.
        """)
        assert detect_style(content) == 'numpy'


class TestDetectSphinx:
    def test_param_field(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            :param x: The value.
            :returns: The result.
        """)
        assert detect_style(content) == 'sphinx'

    def test_raises_field(self) -> None:
        content = textwrap.dedent("""\
            Summary.

            :raises ValueError: If bad.
        """)
        assert detect_style(content) == 'sphinx'


class TestDetectSplitConsistency:
    def test_line_count_matches_split_newline(self) -> None:
        # Trailing \n: split('\n') gives ['a', ''], splitlines() gives ['a']
        # The code_block_lines index 0 should correctly skip line 0.
        content = 'Args:\n    x: val.'
        assert detect_style(content, code_block_lines={0}) is None


class TestDetectTitleFallback:
    def test_body_marker_wins_over_fallback_title_marker(self) -> None:
        content = 'Args:\n-----\n:param x: Kept.'
        assert (
            detect_style(
                content,
                primary_start=2,
                fallback_line_indices=range(2),
            )
            == 'sphinx'
        )

    def test_body_tie_is_not_broken_by_fallback_title_marker(self) -> None:
        content = 'Args:\n-----\nParameters\n----------\nx : int\n    Kept.\n:param y: Also kept.'
        assert (
            detect_style(
                content,
                primary_start=2,
                fallback_line_indices=range(2),
            )
            is None
        )

    def test_default_interface_keeps_full_content_behavior(self) -> None:
        assert detect_style('Args:\n    x: Kept.') == 'google'

    def test_fallback_marker_is_used_when_body_has_no_markers(self) -> None:
        content = 'Args:\n-----\nplain body'
        assert (
            detect_style(
                content,
                primary_start=2,
                fallback_line_indices=range(2),
            )
            == 'google'
        )

    def test_overlined_marker_title_is_excluded(self) -> None:
        content = '-----\nArgs:\n-----\nplain body'
        assert detect_style(content, primary_start=3) is None
