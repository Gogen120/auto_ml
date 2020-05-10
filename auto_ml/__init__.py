import os
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston, load_iris

from auto_ml.exceptions import UnknownMLTaskException
from auto_ml.models import ClassificationModel, RegressionModel
from auto_ml.constants import VALID_ML_TASKS


class MLRunner:
    def __init__(self, task):
        if task in VALID_ML_TASKS:
            self._task = task
        else:
            raise UnknownMLTaskException(f'Unknown ml task type provided: {task}')

    def run(
        self, data, target, test_size=0.25, use_bootstrap=False, n_samples=1000,
        metric='accuracy', metric_params=None, models=None, model_params=None,
        result_file='result.csv', use_threads=False,
    ):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

        if self._task == 'classification':
            ml_model = ClassificationModel(
                X_train, y_train, X_test, y_test, metric, metric_params,
                use_bootstrap=use_bootstrap, n_samples=n_samples,
                models=models, model_params=model_params, result_file=result_file,
            )
        elif self._task == 'regression':
            ml_model = RegressionModel(
                X_train, y_train, X_test, y_test, metric, metric_params,
                use_bootstrap=use_bootstrap, n_samples=n_samples,
                models=models, model_params=model_params, result_file=result_file,
            )

        if use_threads:
            ml_model.run_models_threads()
        else:
            ml_model.run_models()
        return ml_model.results
