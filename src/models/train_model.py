"""Module for train main model"""

import pandas as pd
import click
from xgboost import XGBClassifier
import joblib
import mlflow
from src.common_funcs import mlflow_set_tracking_config


mlflow_set_tracking_config("docker_mlflow")


@click.command()
@click.argument("output_model_path", type=click.Path())
@click.option("--pair_feautures_dataset_path", default=None, type=click.Path(exists=True))
@click.option("--pair_feautures_df", default=None)
def train_model(
    output_model_path: str,
    pair_feautures_dataset_path: str = None,
    pair_feautures_df: pd.DataFrame = None,
) -> None:
    """Функция тренировки основной модели бинарной классификиции на основе датасета
    пар-кандидатов. Датасет может быть задан двумя вариантами, через путь до файла csv,
    через прямую передачу объекта pd.Dataframe. По итогу исполнения модель сохраняется
    в файле по заданному пути.

    Должен быть задан один из опциональных параметров pair_feautures_dataset_path или
    pair_feautures_df.

    Args:
        output_model_path (str): Путь для сохранения обученной модели. Модель сохраняется
            через joblib.dump, для дальнейшей загрузки используем joblib.load.
        pair_feautures_dataset_path (str, optional): Путь к файлу csv датасета пар-кандидатов
            (если задан, датасет берется из него). Defaults to None.
        pair_feautures_df (pd.DataFrame, optional): Датасет пар-кандидатов типа pd.DataFrame
            (используется, если pair_feautures_dataset_path не задан). Defaults to None.
    """
    with mlflow.start_run():

        mlflow.get_artifact_uri()
        print(mlflow.get_artifact_uri())

        if pair_feautures_dataset_path is not None:
            pair_feautures_df = pd.read_csv(pair_feautures_dataset_path)

        # Формируем X, y для дальнейшей передачи в модель
        y_true = pair_feautures_df["target_label"]  # type: ignore

        # Все колонки фичей начинаются с 'ftr_'
        columns_to_remove = [
            col for col in pair_feautures_df.columns if not col.startswith("ftr_")  # type: ignore
        ]
        # Для экономии памяти (исключения возможности дублирования) используем inplace
        pair_feautures_df.drop(columns=columns_to_remove, inplace=True)  # type: ignore
        X_features = pair_feautures_df  # pylint: disable=C0103

        # Model params
        model_params = {"random_state": 42, "n_estimators": 10, "verbosity": 0}

        # Log model params
        mlflow.log_params(model_params)

        # Define the model
        model = XGBClassifier(**model_params)

        # Fit the model
        model.fit(X_features, y_true)

        joblib.dump(model, output_model_path)

        # Mlflow log model
        mlflow.xgboost.log_model(
            model, artifact_path="output_model_path", registered_model_name="my_baseline_xboost"
        )


if __name__ == "__main__":
    train_model()  # pylint: disable=E1120
