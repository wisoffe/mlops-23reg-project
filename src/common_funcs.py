"""Module for common project functions
"""
import os
import pkg_resources
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import mlflow


def mlflow_set_tracking_config(
    experiment_name: str = "default", project_name="mlops23regproject"
) -> None:
    """Настраивает трекинг экспериментов через MLFlow Tracking server.
    Реализует сценарий 4. В файле .env должны быть следующие переменные:
    MLFLOW_TRACKING_URI - URL трекинг сервера MLflow (например "http://localhost:5000")

    MLFLOW_S3_ENDPOINT_URL* - URL до s3 хранилища (например "http://localhost:9000")
    AWS_ACCESS_KEY_ID* - key ID к s3 хранилищу (с правами на запись)
    AWS_SECRET_ACCESS_KEY* - secret key к s3 хранилищу
    * используется библиотекой boto3


    Args:
        experiment_name (str, optional): Название серии экспериментов в MLflow.
            Defaults to "default".
        project_name (str, optional): Название текущего проекта, обычно соответствует
            [tool.poetry].name из pyproject.toml. Defaults to "mlops23regproject".
    """
    load_dotenv(override=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    project_ver = pkg_resources.get_distribution(project_name).version
    mlflow.set_experiment(f"{experiment_name}_v{project_ver}")


def jaccard(list_a: list, list_b: list) -> float:
    """Реализация Jaccard_index для двух списков/множеств,
    фомула: https://en.wikipedia.org/wiki/Jaccard_index
    Необходима для получения скора, используемого в соревновании.

    Args:
        list_a (list): Список 1 (далее преобразуется в set)
        list_b (list): Список 2 (далее преобразуется в set)

    Returns:
        float: возвращает Jaccard_index для пары множеств
    """
    set_a = set(list_a)
    set_b = set(list_b)
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_score(df_sub_a: pd.DataFrame, df_sub_b: pd.DataFrame) -> float:
    """Реализует подсчет полного скора, используемого в соревновании, который равен среднему
    из Jaccard_index всех строк

    Args:
        df_sub_a (pd.DataFrame): Датафрейм A, формата требуемого для submission соревнования,
            с колонками id, matches, где в matches через пробел перечислены id всех дублей POI,
            включая id сомого себя
        df_sub_b (pd.DataFrame): Датафрейм B, аналогичный датафрейм для сравнения

    Returns:
        float: скор, аналогичный тому, который используется в соревновании
    """

    # assert len(df_sub_a) == len(df_sub_b)
    df_ab_matches = pd.merge(df_sub_a, df_sub_b, on="id", suffixes=["_a", "_b"])
    return (
        df_ab_matches.apply(lambda x: jaccard(x.matches_a.split(), x.matches_b.split()), axis=1)
    ).mean()


def get_submission_predict(
    df_original: pd.DataFrame,
    df_pairs: pd.DataFrame,
    labels: np.ndarray,
    pairs_drop_orders_dublicates=False,
) -> pd.DataFrame:
    """Генерирует датафрейм формата submission, который требуется в рамках соревнования.

    Генерирует на основе входных данных, которые мы обычно имеем после предсказания моделью.
    Args:
        df_original (pd.DataFrame): Оригинальный датасет, на основе которого делаем предсказания,
            по факту из него нужна только колонка id; исходный датасет необходим т.к. в парный
            датасет обычно попадают не все id, а submission должен содержать все исходные id
        df_pairs (pd.DataFrame): парный датасет, который передавался в модель, из него берем только
            колонки [id_1, id_2]
        labels (np.ndarray): массив из 1 и 0, обычно получемый на основе предсказаний модели, но не
            обязательно, в рамках вычисления метрик парного датасета передается метки, полученные на
            основе реальной колонки "point_of_interest"; порядок должен в точности соответсвовать
            порядку df_pairs
        pairs_drop_orders_dublicates (bool, optional): флаг, признак того, оставляли ли мы на этапе
            формирования пар-кандидатов только пары в одном направлении (id_1, id_2), или же в обоих
            (id_1, id_2) и (id_2, id_1). Defaults to False.

    Returns:
        pd.DataFrame: возвращаем датафрейм формата submission, требуемого в рамках
        соревнования с колонками id, matches, где в matches через пробел перечислены,
        id всех дублей POI, включая id сомого себя.
    """

    # task: понял поздно, уже после написания, но есть проблема, нужно ее решить:
    #       проблема: если модель предсказала, что (id1, id2) и (id2, id3) дублиаты,
    #       а (id1, id3) нет, то в текущей реализации в сабмит добавятся строки
    #       (id1, [id1, id2]), (id2, [id2, id1, id3], (id3, [id3, id2]), хотя на
    #       самом деле они либо все дубликаты, либо где то предсказание ошибочно.
    #       переписать в единственном варианте, либо добавить возможность возвращать
    #       оба варианта, в зависимости от перданного параметра (и проверить на
    #       реальных сабмитах, какой вариант дает больший скор);

    # формируем датасет из пар, для которых match/label == 1
    df_pairs = df_pairs[["id_1", "id_2"]]
    df_pairs["match"] = labels
    df_pairs = df_pairs[df_pairs.match == 1][["id_1", "id_2"]]

    # если мы оставляли пары только в одном направлении (id_1, id_2),
    # то возвращаем что бы они были в обоих (id_1, id_2) и (id_2, id_1)
    if pairs_drop_orders_dublicates:
        df_pairs = pd.concat(
            [df_pairs, df_pairs.rename(columns={"id_1": "id_2", "id_2": "id_1"})]
        ).drop_duplicates()  # drop_duplicates не обязателен

    # добавляем сапоставление  id  самому себе, т.к. этого требует выходной
    # формат
    pairs_one_to_one = pd.DataFrame({"id_1": df_pairs.id_1.unique()})
    pairs_one_to_one["id_2"] = pairs_one_to_one.id_1
    df_pairs = pd.concat([pairs_one_to_one, df_pairs])

    # переводим в формат id, matches, где в matches через пробел перечислены все
    # найденные дубликаты (в том числе сам id попадет в matches)
    df_pairs = (
        df_pairs.groupby("id_1")
        .id_2.agg(" ".join)
        .to_frame()
        .reset_index()
        .rename(columns={"id_1": "id", "id_2": "matches"})
    )

    # в df_pairs остались только id, для которых найдены дубликаты, мерджим со
    # всеми id из исходного датасета и добавляем в matchs id самого себя, для
    # тех id, которые не попали в df_pairs (после merge у них matches == NaN)
    df_submission = pd.merge(df_original["id"], df_pairs, on="id", how="left")
    df_submission["matches"] = df_submission.matches.fillna(df_submission.id)

    # assert len(df_submission) == len(df_original)

    return df_submission


def get_submission_true(df_original: pd.DataFrame) -> pd.DataFrame:
    """Генерирует датафрейм формата submission, который требуется в рамках соревнования.

    Генерирует на основе датасета, формата train (по факту используются только колонки
    ["id", "point_of_interest"]), в котором дубликаты имеют одно и то же значение в
    колонке "point_of_interest".

    Args:
        df_original (pd.DataFrame): Датасет, имеющий колонки ["id", "point_of_interest"]

    Returns:
        pd.DataFrame: возвращаем датафрейм формата submission, требуемого в рамках
        соревнования с колонками id, matches, где в matches через пробел перечислены,
        id всех дублей POI, включая id сомого себя.
    """
    df_original = df_original[["id", "point_of_interest"]]
    df_poi_matches = (
        df_original.groupby("point_of_interest")
        .id.agg(" ".join)
        .to_frame()
        .reset_index()
        .rename(columns={"id": "matches"})
    )
    return pd.merge(df_original, df_poi_matches, on="point_of_interest", how="left")[
        ["id", "matches"]
    ]


def get_match_label(dataset: pd.DataFrame) -> pd.Series:
    """Получаем Series из целевых переменных (label/target) для бинарной классификации,
    где 1 - являются дубликатами (match), 0 - не являются (not match)

    Args:
        dataset (pd.DataFrame): Парный датасет в котором присутсвуют колонки:
            point_of_interest_1, point_of_interest_2

    Returns:
        pd.Series: серия из 1 или 0, ниши целевые переменные (индексы соответсувуют переданному
        парному датасету)
    """
    return (dataset["point_of_interest_1"] == dataset["point_of_interest_2"]).astype(int)
