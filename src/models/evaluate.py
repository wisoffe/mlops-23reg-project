"""Module for evaluate (get main metrics) of main model (trains on split df)"""

from typing import Union
import json
import click
import pandas as pd
from src.common_funcs import (
    get_submission_true,
    jaccard_score,
)


def evaluate(
    input_original_dataset_path: str,
    input_submission_pred_path: str,
    output_metrics_path: str,
    return_metrics=False,
) -> Union[None, dict]:
    """Проводим финальную оценку (получение метрик) модели. Оценка возможна только в случае
    наличия у нас датасета с true метками (т.е. не подходит для реального test датасета).

    По итогу получаем и сохраняем в файл финальные метрики.

    Args:
        input_original_dataset_path (str): Путь к csv файлу датасета, исходного формата (на
            основе которого были сгенерированы пары-кандидаты, обычно это файл отложенной
            выборки split_test), по факту в нем нужны только колонки ["id", "point_of_interest"]
        input_submission_pred_path (str): Путь до входного файла submission pred (.csv)
        output_metrics_path (str): Путь до выходного json файла метрик.
    """
    submission_true = get_submission_true(
        pd.read_csv(
            input_original_dataset_path, skipinitialspace=True, usecols=["id", "point_of_interest"]
        )
    )
    submission_pred = pd.read_csv(input_submission_pred_path)

    metrics = {}
    metrics["Jaccard_final"] = jaccard_score(submission_true, submission_pred)

    with open(output_metrics_path, "w", encoding="UTF-8") as metrics_file:
        json.dump(metrics, metrics_file)

    print(metrics)

    if return_metrics:
        return metrics
    else:
        return None


@click.command()
@click.argument("input_original_dataset_path", type=click.Path(exists=True))
@click.argument("input_submission_pred_path", type=click.Path(exists=True))
@click.argument("output_metrics_path", type=click.Path())
def evaluate_cli(
    input_original_dataset_path: str,
    input_submission_pred_path: str,
    output_metrics_path: str,
) -> None:
    """Выполняет функционал evaluate(), но предназначена для запуска модуля из коммандной строки."""
    evaluate(input_original_dataset_path, input_submission_pred_path, output_metrics_path)


if __name__ == "__main__":
    evaluate_cli()  # pylint: disable=E1120
