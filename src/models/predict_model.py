"""Module for predict of main model"""

import pandas as pd
import click
import joblib
from src.common_funcs import get_submission_predict
from src.data.generate_pairs import PAIRS_DROP_ORDER_DUBLICATES


def predict_model_in_memory(
    model, original_ids_df: pd.DataFrame, pair_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Функция предсказания основной модели бинарной классификиции на основе датасета
    пар-кандидатов и датасета с оригинальными id. Работает с объектами в памяти (а не с файлами).
    По итогу исполнения формируется датафрейм предсказаний, в формате submission, который
    требуется в рамках соревнования.

    Args:
        model (_type_): Обученная модель.
        original_ids_df (_type_): Датафрейм с единственной колонкой ["id"], содержащий в себе
            все id оригинального файла, на котором требуется предсказание.
        pair_features_df (_type_): Датасета пар-кандидатов, на котором требуется предсказание.

    Returns:
        pd.DataFrame: Выходной датафрейм submission, требуемого формата.
    """
    pairs_ids_df = pair_features_df[["id_1", "id_2"]]  # type: ignore

    # Все колонки фичей начинаются с 'ftr_'
    columns_to_remove = [
        col for col in pair_features_df.columns if not col.startswith("ftr_")  # type: ignore
    ]
    # Для экономии памяти (исключения возможности дублирования) используем inplace
    pair_features_df.drop(columns=columns_to_remove, inplace=True)  # type: ignore
    X_features = pair_features_df  # pylint: disable=C0103

    # Get predictions
    y_pred = model.predict(X_features)

    del X_features

    submission_pred = get_submission_predict(
        original_ids_df, pairs_ids_df, y_pred, PAIRS_DROP_ORDER_DUBLICATES
    )

    return submission_pred


def predict_model(
    input_model_path: str,
    input_original_dataset_path: str,
    input_pair_features_dataset_path: str,
    output_submission_path: str,
) -> None:
    """Функция предсказания основной модели бинарной классификиции на основе датасета
    пар-кандидатов и датасета с оригинальными id. Работает на уровне файлов.
    По итогу исполнения формируется csv файл предсказаний, в формате submission, который
    требуется в рамках соревнования.

    Args:
        input_model_path (str): Путь до файла обученной модели (сформирован через joblib.dump)
        input_original_dataset_path (str): Путь к csv файлу датасета, исходного формата (на
            основе которого были сгенерированы пары-кандидаты, обычно это файл отложенной
            выборки split_test или оригинальный test), по факту в нем нужна только колонка ["id"]
        input_pair_features_dataset_path (str): Путь к файлу csv датасета пар-кандидатов.
        output_submission_path (str): Путь для сохранения выходного файла submission pred (.csv)
    """

    model = joblib.load(input_model_path)

    original_ids_df = pd.read_csv(
        input_original_dataset_path, skipinitialspace=True, usecols=["id"]
    )

    pair_features_df = pd.read_csv(input_pair_features_dataset_path)

    submission_pred = predict_model_in_memory(model, original_ids_df, pair_features_df)

    submission_pred.to_csv(output_submission_path, index=False)


@click.command()
@click.argument("input_model_path", type=click.Path(exists=True))
@click.argument("input_original_dataset_path", type=click.Path(exists=True))
@click.argument("input_pair_features_dataset_path", type=click.Path(exists=True))
@click.argument("output_submission_path", type=click.Path())
def predict_model_cli(
    input_model_path: str,
    input_original_dataset_path: str,
    input_pair_features_dataset_path: str,
    output_submission_path: str,
) -> None:
    """Выполняет функционал predict_model(), но предназначена для запуска модуля из коммандной
    строки."""
    predict_model(
        input_model_path,
        input_original_dataset_path,
        input_pair_features_dataset_path,
        output_submission_path,
    )


if __name__ == "__main__":
    predict_model_cli()  # pylint: disable=E1120
