"""Module for build features"""

from typing import Tuple
import pandas as pd
import numpy as np
import click


def run_futures_pipeline(
    dataset: pd.DataFrame, futures_pipeline: list, prefix="ftr_"
) -> pd.DataFrame:
    """Последовательно пропускает датасет через переданный набор функций, генерирующих фичи.

    Args:
        dataset (pd.DataFrame): Датасет, на основе которого генерируются фичи.
        futures_pipeline (list): Список функций и их дополнительных параметров. Формат следующий:
            futures_pipeline = [
                {"func": generate_feature1, "params": {"param1": 1, "param2": "2" ...}},
                {"func": generate_feature2, "params": {}},
            ]
            Каждая функция должна:
                - первым параметром принимать датасет,
                - вторым параметром принимать prefix,
                - возвращать датафрейм из сгенерированных фичей (даже если фича одна, на выходе
                  должен возвращаться датафрейм из 1 колонки), все колонки имеют префикс из prefix
        prefix (str, optional): Префикс, который должен добавляться ко всем колонкам
            сгенерированных фич, на основании него будет формироваться итоговый набор колонок
            для передачи в модель (т.е. только колонки с данным префиксом будут переданы в модель).
            Defaults to "ftr_".

    Returns:
        pd.DataFrame: Датасет, из сгенерированных фичей (все колонки имеют префикс из prefix)
    """
    list_of_df_feautures = []

    for step in futures_pipeline:
        list_of_df_feautures.append(step["func"](dataset, prefix=prefix, **step["params"]))

    df_feautures = pd.concat(list_of_df_feautures, axis=1, ignore_index=False)

    return df_feautures


def add_feauture_geo_distance(
    df_pairs: pd.DataFrame, normalize=False, prefix="ftr_"
) -> pd.DataFrame:
    """Добавляем фичу для парного датасета - расстояние между точками (в км., если не нормализуем)

    Args:
        df_pairs (pd.DataFrame): Парный датасет в котором присутсвуют колонки:
            latitude_1, longitude_1, latitude_2, longitude_2,
        normalize (bool, optional): На будущее заложена возможность нормализации, в текущий момент
            не реализовано. Defaults to False.
        prefix (str, optional): Префикс, перед названием колонки. Defaults to "ftr_".

    Returns:
        pd.DataFrame: Датасет фичей из колонок:
            "{prefix}geo_distance" - расстоянием между точками
    """

    # task: Скорей всего нет смысла считать точно через haversine (с учетом шарообразности земли),
    # предпочтительней считать более грубо, но быстрее

    df_feautures = pd.DataFrame()

    df_feautures[f"{prefix}geo_distance"] = df_pairs.apply(
        lambda x: hs.haversine(
            (x.latitude_1, x.longitude_1), (x.latitude_2, x.longitude_2), unit=Unit.KILOMETERS
        ),
        axis=1,
    )

    if normalize:
        pass

    return df_feautures


def add_feauture_levenshtein_distance(
    df_pairs: pd.DataFrame, normalize=False, prefix="ftr_"
) -> pd.DataFrame:
    """Добавляем фичи для парного датасета:
        - расстояние Левенштейна между двумя строками.

    В текущем варианте добавляет расстояние только между name, но нужно будет расширить,
    как минимум еще на категории (основная проблема с ними, наличие пропущенных значений)

    Args:
        df_pairs (pd.DataFrame): Парный датасет в котором присутсвуют колонки:
            name_1, name_2
        normalize (bool, optional): На будущее заложена возможность нормализации, в текущий момент
            не реализовано. Defaults to False.
        prefix (str, optional): Префикс, перед названием колонки. Defaults to "ftr_".

    Returns:
        pd.DataFrame: Датасет фичей из колонок:
            - "{prefix}name_levenshtein" - расстояние Левенштейна между именами
    """

    df_feautures = pd.DataFrame()

    df_feautures[f"{prefix}name_levenshtein"] = df_pairs.apply(
        lambda x: fuzz.token_set_ratio(x.name_1, x.name_2), axis=1
    )

    if normalize:
        pass

    return df_feautures


def build_futures_df(df_pairs: pd.DataFrame, add_target_label: bool) -> pd.DataFrame:
    ## Формирование датасета, подходящего для передачи в модель

    # Задаем пайплайн из функций по генерации фичей
    # (описание формата в docstring run_futures_pipeline)
    pairs_futures_pipeline = [
        {"func": add_feauture_geo_distance, "params": {}},
        {"func": add_feauture_levenshtein_distance, "params": {}},
    ]

    # Генерируем фичи
    df_feautures = run_futures_pipeline(df_pairs, pairs_futures_pipeline)

    if add_target_label:
        pairs_train
