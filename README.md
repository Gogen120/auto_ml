# Auto ML

Библиотека для автоматического запуска моделей машинного обучения

### Установка

Для установки библиотеки надо склонировать репозиторий к себе и выполнить следующую команду: `pip install -e .` в склонированной директории

### Примеры использования

**Задача бинарной классификации**
```python
from sklearn.datasets import load_breast_cancer

from auto_ml import MLRunner

if __name__ == '__main__':
    data, target = load_breast_cancer(return_X_y=True)
    ml_runner = MLRunner(task='classification')
    ml_runner.run(
        data, target, use_bootstrap=False, metric='f1', metric_params=None,
        models=['LogisticRegression', 'RandomForestClassifier', 'RandomForestClassifier'],
        model_params=[{'C': 0.1, 'solver': 'liblinear'}, {'n_estimators': 50, 'max_depth': 5}, {}]
    )
```

**Задача мультклассовой классификации**
```python
from sklearn.datasets import load_iris

from auto_ml import MLRunner

if __name__ == '__main__':
    data, target = load_iris(return_X_y=True)
    ml_runner = MLRunner(task='classification')
    ml_runner.run(
        data, target, use_bootstrap=False, metric='f1', metric_params={'average': 'macro'}, n_samples=300,
        models=['LogisticRegression', 'RandomForestClassifier', 'RandomForestClassifier'],
        model_params=[{'C': 0.1, 'solver': 'liblinear'}, {'n_estimators': 50, 'max_depth': 5}, {}]
    )
```

**Задача регрессии**
```python
from sklearn.datasets import load_boston

from auto_ml import MLRunner

if __name__ == '__main__':
    data, target = load_boston(return_X_y=True)
    ml_runner = MLRunner(task='regression')
    ml_runner.run(
        data, target, use_bootstrap=False, metric='mse', n_samples=1000,
        models=['LinearRegression', 'RandomForestRegressor', 'RandomForestRegressor'],
        model_params=[{}, {'n_estimators': 50, 'max_depth': 5}, {}]
    )
```

### Описание

Основной объект библиотеки `MLRunner` имеет один метод `run`, с помощью которого можно запускать различный набор моделей с различными параметрами для переданного датасета

При создании объекта `MLRunner` нужно указать один параметр - тип решаемой задачи (в данный момент поддерживаются 2 типа - `classification` и `regression`)

**Параметры метода run**:
* `data` - pandas датафрейм с фичами
* `target` - pandas датафрейм с целевой переменной
* `test_size` - определяет пропорции разбиения на тренировочный и тестовый датасет (параметр в `train_test_split`)
* `use_ootstrap` - флаг, который показывает нужно ли использовать бутстрап для создания подвыборки с возвращением для каждой модели
* `n_samples` - размер бутстрап выборки (если этот параетр больше чем изначальный размер выборки, то выбросится исключение)
* `metric` - название метрики, которую использовать для сравнения предсказаний моделей и настоящих значений (доступный метрики указаны ниже)
* `metric_params` - словарь с параметрами метрики (можно посмотреть на соответствующей странице, с описанием метрики)
* `models` - список моделей, которые надо использовать для обучения (доступные модели смотри ниже)
* `model_params` - список словарей параметров соответствующих моделей (по индексу в списках). Для того, чтобы использовать параметры по умолчанию, надо передать пустой словарь
* `result_file` - файл, в который надо писать результаты моделей (записывается время, модель, параметры модели, имя метрики, значение метрики). По-умолчанию запишется в `results.csv` в текущей директории
* `use_threads` - флаг, который показывает надо ли использовать треды для тренировки моделей. Если `False`, то запускаться будут последовательно (по-умолчанию `False`)

**Доступные метрики**
* Классификация
  * `accuracy` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
  * `precision`- [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
  * `recall` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
  * `f1` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* Регрессия
  * `mse` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)
  * `mae` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)

**Доступные модели**
* Классификация
  * `LogisticRegression` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
  * `RandomForestClassifier` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
  * `RidgeClassifier` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)
  * `GradientBoostingClassifier` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
  * `LinearSVC` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
* Регрессия
  * `LinearRegression` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
  * `RandomForestRegressor` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
  * `Ridge` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
  * `LinearSVR` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)
  * `GradientBoostingRegressor` - [документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)

### Тесты

Для запуска тестов нужно выполнить следующую команду: `pytest`

Для запуска тестов с coverage'ом: `pytest --cov-report term-missing --cov auto_ml`

### TODO

В данный момент представлен только базовый функционал. Ниже список вещей, которые можно добавить в будущем:
* Загрузка различных типов данных (сейчас на вход принимается только pandas датафрейм)
* Предобработка данных (сейчас подразумевается, что пользователь уже сделал классическую обработку датасета)
* Использование более сложных и предобученных моделей
* Более сложные и кастомные метрики
* `GridSearch` или `RandmizedSearch` для поиска лучших параметров
* Выбор стратегии разбиения датасета (сейчас используется самый простой - `train_test_split`)
* Возможность записи результатов в различные хранилища
