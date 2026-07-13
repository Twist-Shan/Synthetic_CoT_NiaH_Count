import pandas as pd
import pytest

from scripts.build_v10_stratified_report import finalize_report_numbering
from synthetic_counting_v10.report_stratified import (
    _linear_summary,
    count_bin_from_value,
)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "1-10"), (10, "1-10"), (11, "11-20"), (20, "11-20"), (21, "21-30"), (30, "21-30")],
)
def test_count_bin_from_value(count, expected):
    assert count_bin_from_value(count) == expected


def test_linear_summary_recovers_transport_slope():
    frame = pd.DataFrame(
        {
            "count_bin": ["1-10"] * 4,
            "receiver_count": [1, 2, 3, 4],
            "predicted_count": [3, 5, 7, 9],
        }
    )

    result = _linear_summary(
        frame,
        ["count_bin"],
        x_column="receiver_count",
        y_column="predicted_count",
    ).iloc[0]

    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(1.0)


def test_report_numbering_places_descriptive_sections_before_causal_sections():
    report = """
    <section id="attention"><h2>5. 描述性 attention</h2></section>
    <section id="geometry"><h2>6. 描述性 hidden state</h2></section>
    <section id="ablation"><h2>7. Attention-head ablation</h2></section>
    <section id="patching"><h2>7. 分层 activation patching：候选 heads 是否局部充分</h2></section>
    <section id="steering"><h2>8. 分层 geometry steering：可读方向是否也是可控方向</h2></section>
    <section id="transplant"><h2>9. 分层 residual transplant：完整 count state 能否搬运</h2></section>
    <section id="synthesis"><h2>11. 综合机制结论、证据强度与尚缺环节</h2></section>
    """

    result = finalize_report_numbering(report)
    expected_headings = [
        "5. 描述性 attention",
        "6. 描述性 hidden state",
        "7. Attention-head ablation",
        "8. Attention-head patching",
        "9. Hidden-state geometry steering",
        "10. Hidden-state patching",
        "11. Attention head 与 hidden state 的双向因果联系",
        "12. 综合机制结论",
    ]

    positions = [result.index(heading) for heading in expected_headings]
    assert positions == sorted(positions)
