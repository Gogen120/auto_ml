import csv
from datetime import datetime

from joblib import Parallel, delayed, parallel_backend

from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import LinearSVC, LinearSVR

from auto_ml.metric import Metric
from auto_ml.exceptions import InvalidBootstrapSampleSizeException
from auto_ml.utils import get_bootstrap_indices


class MLModel:
    def __init__(
        self, X_train, y_train, X_test, y_test, metric_name, metric_params,
        use_bootstrap=False, n_samples=1000, models=None, model_params=None,
        result_file='results.csv'
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._metric = Metric(metric_name, metric_params)
        self._use_bootstrap = use_bootstrap
        self._n_samples = n_samples
        self._result_file = result_file

        if use_bootstrap and self._X_train.shape[0] < n_samples:
            raise InvalidBootstrapSampleSizeException(f'Cannot use bootstrap sampling: {n_samples} > {self._X_train.shape[0]}')

        self._models = self._init_models() if models is None else models
        self._model_params = self._init_model_params() if model_params is None else model_params

        if len(self._models) != len(self._model_params):
            raise ValueError('models list and model params list must have the same length')

        self._results = {}

        self._model_mapper = {
            # Classifiers
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'RidgeClassifier': RidgeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LinearSVC': LinearSVC,

            # Regressors
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'Ridge': Ridge,
            'LinearSVR': LinearSVR,
            'GradientBoostingRegressor': GradientBoostingRegressor,
        }

    @property
    def models(self):
        return self._models

    @property
    def model_params(self):
        return self._model_params

    @property
    def results(self):
        return self._results

    def run_models(self):
        for model in self._map_models():
            self._train_model(model)

    def run_models_threads(self):
        with parallel_backend('threading', n_jobs=-1):
            Parallel()(delayed(self._train_model)(model) for model in self._map_models())

    def _train_model(self, model):
        self._fit(model)
        y_pred = self._predict(model)
        metric_score = self._metric.score(self._y_test, y_pred)
        self._write_metric_results(model, metric_score)

    def _init_models(self):
        raise NotImplementedError

    def _fit(self, model):
        if self._use_bootstrap:
            indices = get_bootstrap_indices(self._X_train, self._n_samples)
            X_train = self._X_train[indices]
            y_train = self._y_train[indices]
        else:
            X_train = self._X_train
            y_train = self._y_train

        model.fit(X_train, y_train)

    def _predict(self, model):
        return model.predict(self._X_test)

    def _write_metric_results(self, model, metric_score):
        model_name = type(model).__name__
        model_params = str(model.get_params())
        self._results[(model_name, model_params)] = metric_score
        with open(self._result_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                (
                    datetime.utcnow(), type(model).__name__, str(model.get_params()),
                    self._metric._metric_name, metric_score,
                )
            )

    def _map_models(self):
        return [
            self._model_mapper.get(model)(**model_params)
            for model, model_params in zip(self._models, self._model_params)
        ]

    def _init_model_params(self):
        return [{}, {}, {}]


class ClassificationModel(MLModel):
    def _init_models(self):
        return ['LogisticRegression', 'RandomForestClassifier', 'LinearSVC']


class RegressionModel(MLModel):
    def _init_models(self):
        return ['LinearRegression', 'RandomForestRegressor', 'LinearSVR']
