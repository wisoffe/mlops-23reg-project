"""Module for predict of main model"""

import pandas as pd
import click
import joblib
from src.common_funcs import get_submission_predict
from src.data.generate_pairs import PAIRS_DROP_ORDER_DUBLICATES


@click.command()
@click.argument("input_model_path", type=click.Path(exists=True))
@click.argument("input_original_dataset_path", type=click.Path(exists=True))
@click.argument("output_submission_path", type=click.Path())
@click.option("--pair_feautures_dataset_path", default=None, type=click.Path())
@click.option("--pair_feautures_df", default=None)
def predict_model(
    input_model_path: str,
    input_original_dataset_path: str,
    output_submission_path: str,
    pair_feautures_dataset_path: str = None,
    pair_feautures_df: pd.DataFrame = None,
) -> None:
    """Функция предсказания основной модели бинарной классификиции на основе датасета
    пар-кандидатов. Датасет пар-кандидатов может быть задан двумя вариантами, через путь
    до файла csv, через прямую передачу объекта pd.Dataframe. По итогу исполнения формируется
    csv файл предсказаний, в формате submission, который требуется в рамках соревнования

    Должен быть задан один из опциональных параметров pair_feautures_dataset_path или
    pair_feautures_df.

    Args:
        input_model_path (str): Путь до файла обученной модели (сформирован через joblib.dump)
        input_original_dataset_path (str): Путь к csv файлу датасета, исходного формата (на
            основе которого были сгенерированы пары-кандидаты, обычно это файл отложенной
            выборки split_test или оригинальный test), по факту в нем нужна только колонка ["id"]
        output_submission_path (str): Путь для сохранения выходного файла submission pred (.csv)
        pair_feautures_dataset_path (str, optional): Путь к файлу csv датасета пар-кандидатов
            (если задан, датасет берется из него). Defaults to None.
        pair_feautures_df (pd.DataFrame, optional): Датасет пар-кандидатов типа pd.DataFrame
            (используется, если pair_feautures_dataset_path не задан). Defaults to None.
    """
    if pair_feautures_dataset_path is not None:
        pair_feautures_df = pd.read_csv(pair_feautures_dataset_path)

    pairs_ids_df = pair_feautures_df[["id_1", "id_2"]]  # type: ignore

    # Все колонки фичей начинаются с 'ftr_'
    columns_to_remove = [
        col for col in pair_feautures_df.columns if not col.startswith("ftr_")  # type: ignore
    ]
    # Для экономии памяти (исключения возможности дублирования) используем inplace
    pair_feautures_df.drop(columns=columns_to_remove, inplace=True)  # type: ignore
    X_features = pair_feautures_df  # pylint: disable=C0103

    model = joblib.load(input_model_path)

    # Get predictions
    y_pred = model.predict(X_features)

    del X_features

    original_df = pd.read_csv(input_original_dataset_path, skipinitialspace=True, usecols=["id"])

    submission_pred = get_submission_predict(
        original_df, pairs_ids_df, y_pred, PAIRS_DROP_ORDER_DUBLICATES
    )

    submission_pred.to_csv(output_submission_path, index=False)


if __name__ == "__main__":
    predict_model()  # pylint: disable=E1120
