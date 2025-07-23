from sklearn.dummy import DummyClassifier
from src.model import dummy_model


def test_dummy_model():
    model = dummy_model()
    assert isinstance(model, DummyClassifier)
    assert model.strategy == 'most_frequent'
