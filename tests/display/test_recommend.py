from docwrap._display._recommend import compute_recommendations
from docwrap._stats import STANDARD_LENGTHS, ImpactRow
from docwrap._types import LineLengthSource

_Impact = dict[int, ImpactRow]


class TestBestFit:
    def test_does_not_gate_on_file_churn(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=50, fs=30, la=100),  # 30/50 = 60%
            79: _ir(79, ds=10, fs=12, la=15),  # 12/50 = 24%
            80: _ir(80, ds=8, fs=9, la=12),  # 9/50 = 18%
            88: _ir(88, ds=2, fs=1, la=3),
            99: _ir(99),
            100: _ir(100),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=79,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.target == 79

    def test_ignores_churn_when_all_targets_change_many_files(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=83, fs=26, la=72),  # 26/52 = 50%
            79: _ir(79, ds=92, fs=27, la=19, lr=5),
            80: _ir(80, ds=95, fs=28, la=16, lr=8),
            88: _ir(88, ds=105, fs=29, la=2, lr=24),
            99: _ir(99, ds=108, fs=30, la=0, lr=30),
            100: _ir(100, ds=109, fs=30, la=0, lr=31),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.target == 80

    def test_ignores_non_standard_lengths(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        impact[85] = _ir(85)  # non-standard, closest to p95=85
        recs = compute_recommendations(
            impact=impact,
            source_p95=85,
            line_length_source=LineLengthSource.DOCWRAP,
            configured_length=85,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.target == 88

    def test_omitted_when_no_source(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        recs = compute_recommendations(
            impact=impact,
            source_p95=None,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        assert not any('Best fit' in r.label for r in recs)

    def test_picks_closest_to_p95(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=50, fs=20, la=100),
            79: _ir(79, ds=10, fs=5, la=15),
            80: _ir(80, ds=8, fs=4, la=12),
            88: _ir(88, ds=2, fs=1, la=3),
            99: _ir(99),
            100: _ir(100),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.target == 80

    def test_rationale_explains_closest_standard_target(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        impact[80] = _ir(80, ds=10, fs=7, la=12)
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.rationale.startswith('Closest target to p95.')

    def test_tiebreaker_prefers_smaller(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=10, fs=5, la=10),
            79: _ir(79, ds=5, fs=3, la=5),
            80: _ir(80, ds=5, fs=3, la=5),
            88: _ir(88, ds=2, fs=1, la=2),
            99: _ir(99),
            100: _ir(100),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=84,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert best.target == 80


class TestConfigured:
    def test_hidden_for_default_source(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        assert not any('Configured' in r.label for r in recs)

    def test_shown_for_cli_length(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        impact[85] = _ir(85, ds=3, fs=2, la=5)
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.CLI,
            configured_length=85,
        )
        assert any('Configured' in r.label for r in recs)

    def test_shown_for_docwrap_source(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        impact[79] = _ir(79, ds=5, fs=3, la=8)
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DOCWRAP,
            configured_length=79,
        )
        assert any('Configured' in r.label for r in recs)


class TestDedup:
    def test_same_target_combines_labels(self) -> None:
        impact: _Impact = {
            72: _ir(72),
            79: _ir(79, ds=5, fs=3, la=5),
            80: _ir(80, ds=5, fs=3, la=5),
            88: _ir(88, ds=2, fs=1, la=2),
            99: _ir(99, ds=1, fs=1, la=1),
            100: _ir(100, ds=1, fs=1, la=1),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=72,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        combined = next(r for r in recs if r.target == 72)
        assert 'Best fit' in combined.label
        assert 'Low churn' in combined.label


class TestLowChurn:
    def test_excludes_extra_lengths(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=50, fs=20, la=100),
            79: _ir(79, ds=10, fs=5, la=15),
            80: _ir(80, ds=8, fs=4, la=12),
            85: _ir(85),  # injected extra_length
            88: _ir(88, ds=2, fs=1, la=3),
            99: _ir(99, ds=1, fs=1),
            100: _ir(100, ds=1, fs=1),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DOCWRAP,
            configured_length=85,
        )
        low = next(r for r in recs if 'Low churn' in r.label)
        assert low.target != 85

    def test_line_churn_tiebreaker_prefers_fewer_files(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=5, fs=5, la=50),
            79: _ir(79, ds=4, fs=4, la=6, lr=4),
            80: _ir(80, ds=3, fs=3, la=7, lr=3),
            88: _ir(88, ds=2, fs=2, la=11),
            99: _ir(99, ds=3, fs=3, la=12),
            100: _ir(100, ds=3, fs=3, la=13),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=None,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        low = next(r for r in recs if 'Low churn' in r.label)
        # 79 and 80 both have total churn 10; 80 has fewer files (3 vs 4)
        assert low.target == 80

    def test_picks_minimum_line_churn(self) -> None:
        impact: _Impact = {
            72: _ir(72, ds=26, fs=26, la=72),
            79: _ir(79, ds=27, fs=27, la=19, lr=5),
            80: _ir(80, ds=28, fs=28, la=16, lr=8),
            88: _ir(88, ds=30, fs=30, la=40),
            99: _ir(99, ds=31, fs=31, la=50),
            100: _ir(100, ds=32, fs=32, la=60),
        }
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        low = next(r for r in recs if 'Low churn' in r.label)
        assert low.target == 79


class TestRationale:
    def test_adds_and_removes_phrasing(self) -> None:
        impact = {line: _ir(line) for line in STANDARD_LENGTHS}
        impact[80] = _ir(80, ds=3, fs=2, la=12, lr=5)
        recommendations = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(item for item in recommendations if 'Best fit' in item.label)
        assert 'Adds 12, removes 5 lines' in best.rationale
        assert '3 docstrings across 2 files' in best.rationale

    def test_adds_phrasing(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        impact[80] = _ir(80, ds=10, fs=7, la=12)
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        best = next(r for r in recs if 'Best fit' in r.label)
        assert '12' in best.rationale

    def test_no_changes_phrasing(self) -> None:
        impact: _Impact = {ln: _ir(ln) for ln in STANDARD_LENGTHS}
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        low = next(r for r in recs if 'Low churn' in r.label)
        assert not any(c.isdigit() for c in low.rationale)

    def test_reformats_phrasing(self) -> None:
        impact: _Impact = {ln: _ir(ln, ds=1, fs=1) for ln in STANDARD_LENGTHS}
        recs = compute_recommendations(
            impact=impact,
            source_p95=80,
            line_length_source=LineLengthSource.DEFAULT,
            configured_length=99,
        )
        low = next(r for r in recs if 'Low churn' in r.label)
        assert any(c.isdigit() for c in low.rationale)


def _ir(line_length: int, ds: int = 0, fs: int = 0, la: int = 0, lr: int = 0) -> ImpactRow:
    return ImpactRow(
        line_length=line_length,
        docstrings_changed=ds,
        files_changed=fs,
        lines_added=la,
        lines_removed=lr,
    )
