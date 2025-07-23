from sklearn.dummy import DummyClassifier


def dummy_model():
    return DummyClassifier(strategy="most_frequent")
