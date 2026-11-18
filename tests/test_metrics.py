from training.metrics import classification_metrics

def test_classification_metrics():
    result = classification_metrics([0, 1, 2], [0, 1, 2])
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
