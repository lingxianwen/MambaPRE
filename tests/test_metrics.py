from mambapre.metrics import BoundaryAccumulator


def test_boundary_and_exact_field_metrics():
    acc = BoundaryAccumulator()
    acc.add(pred_cuts={2, 5}, gold_cuts={2, 4}, length=8)
    report = acc.summary()
    assert report["boundary_precision"] == 0.5
    assert report["boundary_recall"] == 0.5
    assert report["boundary_f1"] == 0.5
    assert report["message_perfection"] == 0.0
    # Only the first field (0,2) is exactly recovered.
    assert report["exact_field_precision"] == 1 / 3
    assert report["exact_field_recall"] == 1 / 3

