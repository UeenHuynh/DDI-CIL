import numpy as np

from scripts.run_c4_classifier_diagnostics import choose_strength, group_direction, metric_rows


def test_group_direction_suppresses_overpredicted_group() -> None:
    logits = np.asarray([[4, 0], [4, 0], [4, 0], [4, 0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    groups = {"head": np.asarray([0]), "medium": np.asarray([], dtype=int), "tail": np.asarray([1])}
    direction = group_direction(logits, labels, groups)
    assert direction[0] < 0
    assert direction[1] > 0


def test_choose_strength_uses_validation_macro_f1_and_prefers_smaller_tie() -> None:
    logits = np.asarray([[2, 0], [2, 0], [0.5, 0], [0.5, 0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    groups = {"head": np.asarray([0]), "medium": np.asarray([], dtype=int), "tail": np.asarray([1])}
    strength, rows = choose_strength(
        logits, labels, np.asarray([-1, 1], dtype=np.float32), groups, (0.0, 0.5, 1.0)
    )
    assert strength == 0.5
    assert len(rows) == 3


def test_metric_rows_reports_group_prediction_mass() -> None:
    labels = np.asarray([0, 0, 0, 1])
    predictions = np.asarray([0, 1, 1, 1])
    groups = {"head": np.asarray([0]), "medium": np.asarray([], dtype=int), "tail": np.asarray([1])}
    rows = {row["group"]: row for row in metric_rows(labels, predictions, groups)}
    assert rows["tail"]["prediction_to_true_ratio"] == 3.0
