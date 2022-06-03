"""Module for train main model"""

import json
import pandas as pd
import click
from xgboost import XGBClassifier
import joblib
import mlflow
from mlflow.models.signature import infer_signature
from src.models.predict_model import predict_model
from src.models.evaluate import evaluate
from src.common_funcs import mlflow_set_tracking_config


mlflow_set_tracking_config("general_model")


@click.command()
@click.argument("input_pair_features_train_dataset_path", type=click.Path(exists=True))
@click.argument("input_pair_features_test_dataset_path", type=click.Path(exists=True))
@click.argument("input_original_test_dataset_path", type=click.Path(exists=True))
@click.argument("input_pair_test_metrics_path", type=click.Path(exists=True))
@click.argument("output_model_path", type=click.Path())
@click.argument("output_submission_path", type=click.Path())
@click.argument("output_metrics_path", type=click.Path())
def train_model(
    input_pair_features_train_dataset_path: str,
    input_pair_features_test_dataset_path: str,
    input_original_test_dataset_path: str,
    input_pair_test_metrics_path: str,
    output_model_path: str,
    output_submission_path: str,
    output_metrics_path: str,
) -> None:
    """Функция тренировки основной модели бинарной классификиции на основе датасета
    пар-кандидатов, включающая в себя формирование csv файл предсказаний (на основе
    отложенной выборки), в формате submission, который требуется в рамках соревнования,
    а так же генерацию метрик модели.
    Дополнительно осуществляется трекинг (посредством mlflow):
     - всех параметров модели,
     - метрик модели,
     - самой модели.

    Args:
        input_pair_features_train_dataset_path (str): Путь к файлу csv датасета пар-кандидатов,
            используемого для обучения.
        input_pair_features_test_dataset_path (str): Путь к файлу csv датасета пар-кандидатов,
            полученного на основе файла отложенной выборки).
        input_original_test_dataset_path (str): Путь к csv файлу датасета отложенной выборки,
            исходного формата (обычно split_test), по факту в нем нужна только колонка ["id"]
        input_pair_test_metrics_path (str): Путь к файлу метрик, который был сгенерирован на стадии
            формирования парного датасета отложенной выборки, для более наглядного трекинга
            экспериментов и формирования дополнительных метрик конечной модели.
        output_model_path (str): Путь для сохранения обученной модели. Модель сохраняется
            через joblib.dump, для дальнейшей загрузки используем joblib.load.
        output_submission_path (str): Путь для сохранения выходного файла submission pred (.csv),
            полученного на основе отложенной выборки.
        output_metrics_path (str): Путь до выходного json файла метрик модели.
    """

    #    with mlflow.start_run(run_name="test_run") as mlflow_run:
    pair_features_train_df = pd.read_csv(input_pair_features_train_dataset_path)

    # Формируем X, y для дальнейшей передачи в модель
    y_true = pair_features_train_df["target_label"]  # type: ignore

    # Все колонки фичей начинаются с 'ftr_'
    columns_to_remove = [
        col for col in pair_features_train_df.columns if not col.startswith("ftr_")  # type: ignore
    ]
    # Для экономии памяти (исключения возможности дублирования) используем inplace
    pair_features_train_df.drop(columns=columns_to_remove, inplace=True)  # type: ignore
    X_features = pair_features_train_df  # pylint: disable=C0103

    # Model params
    model_params = {"random_state": 42, "n_estimators": 10, "verbosity": 0}

    # Define the model
    model = XGBClassifier(**model_params)

    # Fit the model
    model.fit(X_features, y_true)

    joblib.dump(model, output_model_path)

    # Подчищаем память после тренировки модели
    del pair_features_train_df, y_true
    # Оставляем только 2 строки от X_features, далее потребуется для формирования signature,
    # необходимой для трекинга модели в mfflow
    X_features = X_features.iloc[:2]  # pylint: disable=C0103
    y_pred = model.predict(X_features)

    # Mlflow tracking model and model params
    mlflow.log_params(model_params)

    signature = infer_signature(X_features, y_pred)  # inputs, outputs
    mlflow.xgboost.log_model(
        model,
        artifact_path="output_model_path",
        registered_model_name="general_model_xboost",
        signature=signature,
    )

    # Подчищаем память
    del model, X_features, y_pred

    # Predict model on test dataset and save output_submission csv
    predict_model(
        output_model_path,
        input_original_test_dataset_path,
        input_pair_features_test_dataset_path,
        output_submission_path,
    )

    metrics_final = evaluate(
        input_original_test_dataset_path,
        output_submission_path,
        output_metrics_path,
        return_metrics=True,
    )

    with open(input_pair_test_metrics_path, encoding="UTF-8") as json_file:
        metrics_pairs = json.load(json_file)

    # Mlflow tracking experiment metrics
    mlflow.log_metrics(metrics_pairs)
    mlflow.log_metrics(metrics_final)


if __name__ == "__main__":
    train_model()  # pylint: disable=E1120
