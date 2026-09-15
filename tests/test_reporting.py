import pytest

from mambapre.reporting import normalize_classical, summarize_values


def test_normalize_classical_uses_message_perfection_not_field_recall():
    report = {
        "overall": {
            "n": 2,
            "bpos_F1": 0.5,
            "exact_boundary_F1": 0.4,
            "perfection": 0.7,
            "msg_perfection": 0.25,
        },
        "per_protocol": [],
    }
    assert normalize_classical(report)["overall"]["message_perfection"] == 0.25


def test_seed_summary_uses_t_interval():
    result = summarize_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == 3.0
    assert result["std"] == pytest.approx(1.58113883)
    assert result["ci95_low"] < 2.0
    assert result["ci95_high"] > 4.0

