# Лабораторная работа №3
# Фундаментальные концепции ИИ
# Оптимизация гиперпараметра
# Семин Александр Витальевич, М8О-109СВ-24

# Задача
    #  1. С помощью [optuna]() взять пример, аналогичный третьему туториалу документации, используя sklearn и с другим датасетом, выбрать другие  алгоритмы классификации и клстеризации не из туториала  и визуализировать графики для полученного процесса
    #     1. В качестве других моделей подойдут любые алгоритмы классификации и регрессии из sklearn которые не использовались в туториале
    #  2. Использовать 2 разных семплера и прунера
    #  3. При процессе оптимизации гиперпараметров использовать общую память через postgreSQL
    #  4. В качестве отчёта выступают: исходный код, инструкция запуска реляционной БД. 

# Для запуска docker-контейнера с postgresql:
# docker run --name postgres-optuna -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15.5

# Для удаления контейнера:
# docker stop postgres-optuna
# docker rm postgres-optuna

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
)
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn.metrics import accuracy_score

def objective_classification(trial):
    """
    Задача: классификация
    Датасет: wine
    Метод: Random Forest
    """
    # загружаем датасет и разделяем его на обучающую и тестовую выборки
    wine = sklearn.datasets.load_wine()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        wine.data, wine.target, test_size=0.25, random_state=42
    )

    # определяем гиперпараметры для оптимизации
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    # создаем и обучаем модель Random Forest с заданными гиперпараметрами
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # предсказываем и вычисляем точность модели на тестовой выборке
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# подключения к базе данных PostgreSQL
storage_url = "postgresql://postgres:postgres@localhost:5432/postgres"
study_name = "wine_classification"

# создаем объект исследования Optuna с заданными параметрами
study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    direction="maximize",
    load_if_exists=False
)

# запускаем оптимизацию гиперпараметров модели
study.optimize(objective_classification, n_trials=50)

# лучшие найденные параметры и значение функции цели
print("best params:", study.best_params)
print("best value:", study.best_value)

plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_slice(study).show()
plot_parallel_coordinate(study).show()
