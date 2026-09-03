import pytest

from scripts.aggregate_longmemeval_shards import aggregate_predictions


def test_longmemeval_aggregation_weights_examples_not_shards() -> None:
    predictions = [
        {
            "exact_match": True,
            "token_f1": 1.0,
            "abstained": False,
        },
        {
            "exact_match": False,
            "token_f1": 0.5,
            "abstained": False,
        },
        {
            "exact_match": False,
            "token_f1": 0.0,
            "abstained": True,
        },
    ]

    result = aggregate_predictions(predictions)

    assert result["count"] == 3
    assert result["exact_accuracy"] == pytest.approx(1 / 3)
    assert result["mean_token_f1"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["selective_accuracy"] == pytest.approx(0.5)
