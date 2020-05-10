import pytest

from sklearn.datasets import load_breast_cancer, load_boston


@pytest.fixture
def classification_pred():
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 1, 0]

    return y_true, y_pred


@pytest.fixture
def multi_classification_pred():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    return y_true, y_pred


@pytest.fixture
def regression_pred():
    y_true = [1.0, 0.5, 0.333, 0.124]
    y_pred = [0.96, 0.61, 0.32, 0.01]

    return y_true, y_pred


@pytest.fixture
def classififcation_task():
    X, y = load_breast_cancer(return_X_y=True)

    return X, y


@pytest.fixture
def regression_task():
    X, y = load_boston(return_X_y=True)

    return X, y
