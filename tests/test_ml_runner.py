import pytest

from auto_ml import MLRunner
from auto_ml.exceptions import UnknownMLTaskException, InvalidBootstrapSampleSizeException


def test_runner_with_invalid_task():
    with pytest.raises(UnknownMLTaskException):
        ml_runner = MLRunner('clustering')


def test_runner_with_invalid_bootstrap_sample_size(classififcation_task):
    ml_runner = MLRunner('classification')
    data, target = classififcation_task

    with pytest.raises(InvalidBootstrapSampleSizeException):
        result = ml_runner.run(data, target, use_bootstrap=True, n_samples=1000)


def test_runner_with_invalid_models_to_model_params_size(classififcation_task):
    ml_runner = MLRunner('classification')
    data, target = classififcation_task

    with pytest.raises(ValueError):
        result = ml_runner.run(
            data, target, models=['LogisticRegression', 'RandomForestClassifier'], model_params=[{"C": 0.01}],
        )


def test_runner_classification_task_with_default_params(classififcation_task):
    ml_runner = MLRunner(task='classification')

    data, target = classififcation_task

    result = ml_runner.run(data, target)

    assert result
    assert len(result) == 3


def test_runner_classification_task_with_bootstrap(classififcation_task):
    ml_runner = MLRunner(task='classification')

    data, target = classififcation_task

    result = ml_runner.run(data, target, use_bootstrap=True, n_samples=300)

    assert result
    assert len(result) == 3


def test_runner_regression_task_with_default_params(regression_task):
    ml_runner = MLRunner(task='regression')

    data, target = regression_task

    result = ml_runner.run(data, target, metric='mse')

    assert result
    assert len(result) == 3
