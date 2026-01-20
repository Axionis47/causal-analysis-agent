"""Basic evaluation scaffolding tests."""

from evals.metrics import ate_bias, precision_recall
from evals.synthetic import generate_simple_causal


def test_synthetic_dataset_shape():
    df = generate_simple_causal(n=100)
    assert df.shape == (100, 3)


def test_metrics_helpers():
    metrics = precision_recall([("a", "b")], [("a", "b"), ("b", "c")])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert ate_bias(1.0, 1.5) == 0.5
