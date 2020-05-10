from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, mean_absolute_error

from auto_ml.exceptions import NotSupportedMetricException


class Metric:
    def __init__(self, metric_name, metric_params=None):
        self._metric_name = metric_name
        self._metric_params = {} if metric_params is None else metric_params

        self._metric_mapper = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
        }

    def score(self, y_true, y_pred):
        metric_function = self._metric_mapper.get(self._metric_name, None)

        if metric_function is None:
            raise NotSupportedMetricException(f'Metric {self._metric_name} is not supported')

        return metric_function(y_true, y_pred, **self._metric_params)
