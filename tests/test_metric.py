import pytest

from auto_ml.metric import Metric
from auto_ml.exceptions import NotSupportedMetricException


def test_accuracy_metric(classification_pred):
    metric = Metric('accuracy')

    y_true, y_pred = classification_pred

    assert metric.score(y_true, y_pred) == 0.75


def test_recall_metric(classification_pred):
    metric = Metric('recall')

    y_true, y_pred = classification_pred

    assert metric.score(y_true, y_pred) == 0.5


def test_multi_recall_metric(multi_classification_pred):
    metric = Metric('recall', {'average': 'macro'})

    y_true, y_pred = multi_classification_pred

    assert metric.score(y_true, y_pred) == 0.3333333333333333


def test_precision_metric(classification_pred):
    metric = Metric('recall')

    y_true, y_pred = classification_pred

    assert metric.score(y_true, y_pred) == 0.5


def test_multi_precision_metric(multi_classification_pred):
    metric = Metric('precision', {'average': 'macro'})

    y_true, y_pred = multi_classification_pred

    assert metric.score(y_true, y_pred) == 0.2222222222222222


def test_f1_metric(classification_pred):
    metric = Metric('recall')

    y_true, y_pred = classification_pred

    assert metric.score(y_true, y_pred) == 0.5


def test_multi_f1_metric(multi_classification_pred):
    metric = Metric('f1', {'average': 'macro'})

    y_true, y_pred = multi_classification_pred

    assert metric.score(y_true, y_pred) == 0.26666666666666666


def test_mse_metric(regression_pred):
    metric = Metric('mse')

    y_true, y_pred = regression_pred

    assert metric.score(y_true, y_pred) == 0.00671625


def test_mae_metric(regression_pred):
    metric = Metric('mae')

    y_true, y_pred = regression_pred

    assert metric.score(y_true, y_pred) == 0.06925


def test_unknown_metric_exception(classification_pred):
    metric = Metric('precision_recall')

    y_true, y_pred = classification_pred

    with pytest.raises(NotSupportedMetricException):
        score = metric.score(y_true, y_pred)
