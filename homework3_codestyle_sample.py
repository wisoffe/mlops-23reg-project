"""
    Sample module, only for MLOps homework3
"""

# Imports
import random

# Imports from
from typing import Tuple
from haversine import Unit
from sklearn.model_selection import GroupKFold  # noqa: F401 # pylint: disable=unused-import
from fuzzywuzzy import fuzz
from xgboost import XGBClassifier  # noqa: F401 # pylint: disable=unused-import

# Imports as
import numpy as np
import pandas as pd
import haversine as hs


# Fix all random seeds
random.seed(42)


# Custom Func's/Class's


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


def get_pairs_metrics(
    df_original: pd.DataFrame,
    df_pairs: pd.DataFrame,
    labels_true: np.ndarray,
    pairs_drop_orders_dublicates=False,
) -> dict:
    """Вычисляем набор метрик для датасета пар-кандидатов, это необходимо в первую очередь,
    для целей оценки качества нашей методики отбора кандидатов. Т.е. наш конечный
    результат зависит высокоуровнего от 2-х подзадач, а именно: задача выбора кандидатов на
    сравнение, задача бинарной классификации выбранных пар на дублирование
    (1 - дубль, 0 - не дубль).

    Смысл в том, что каких бы самых лучших фичей мы не нагенерировали для наших пар (уже
    сформированных) и какая бы наилучшая модель у нас не была, мы не сможем получить высокий
    итоговый скор, если у нас в парах (попарном сравнении) недостаточно реальных дубликатов.

    Подробное описание возвращаемых метрик (в текущей реализации функции) приведено в Returns.

    Args:
        df_original (pd.DataFrame): Исходный датасет (формата train), необходим, т.к. в парный
            попадают не все id
        df_pairs (pd.DataFrame): датасет наших пар-кандидатов на сравнение
        labels_true (np.ndarray): массив из 1 и 0, метки, полученные на основе реальной колонки
        "point_of_interest"

    Returns:
        dict: Возвращаем словарь метрик, на текущий момент подсчитываются:
            {
                "Jaccard (max)": Максимально возможный Jaccard скор (основная метрика
                соревнования), который можно получить из текущих выбранных кандидатов. Т.е.
                Если наша модель, справится с бинарной классификацией на 100%
            }
    """
    metrics = {}
    submission_true = get_submission_true(df_original)
    submission_pairs_max_true = get_submission_predict(
        df_original, df_pairs, labels_true, pairs_drop_orders_dublicates
    )
    metrics["Jaccard (max)"] = jaccard_score(submission_true, submission_pairs_max_true)

    return metrics


def generate_pairs_df(
    df_dataset: pd.DataFrame, drop_order_dub=False, get_metrics=False, real_test=False
) -> Tuple[pd.DataFrame, dict]:
    """Отбираем кандидатов на сравнение, для задачи дальнейшей попарной бинарной классификации.
    Дополнительно возвращаем метрики качества отобранных кандидатов.

    Args:
        df_dataset (pd.DataFrame): Исходный датасет для формирования пар-кандидатов
        drop_order_dub (bool, optional): Оставляли только пары в одном направлении (id_1, id_2),
            или же в обоих (id_1, id_2) и (id_2, id_1). Defaults to False.
        get_metrics (bool, optional): Осуществлять ли подсчет метрик, если нет, в качестве метрик
            будет возвращен пустой словарь. Defaults to False.
        real_test (bool, optional): Имеем мы дело с реальным test.csv датасетом, или каким-то
            другим; в реальном тест датасете у нас отсутсвует колонка "point_of_interest",
            соответсвенно мы при выборе необходимых колонок не должны ее учитывать, а так же без
            нее мы не можем подсчитать основные метрики качества. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, dict]: [0] - Возвращаем датасет пар-кандидитов, с суффиксами
                ["_1", "_2"]
            [1] - словарь метрик качества отобранных пар, формат см. в выходе функции
                get_pairs_metrics
    """

    # Отбираем только колонки, которые планируем использовать в дальнейшем
    # task: перенести в параметры/гиперпараметры?

    column_selection_first = [
        "id",
        "name",
        "latitude",
        "longitude",
        "country",
        "city",
        "categories",
        "point_of_interest",
    ]

    # в реальном тесте отсутсвует 'point_of_interest', удаляем из наших колонок
    if real_test:
        column_selection_first.remove("point_of_interest")

    df_dataset = df_dataset[column_selection_first]

    # task: Изучить возможности sklearn для подобных целей, скорей всего это будет наилучший
    # вариант (примерные ключевые слова: sklearn neighbors by coordinate)

    # В текущей реализации формируем пары только на основе близости координат, через округление
    # координат и дальнейщего сопоставления в пары на основе комбинации округленных координат
    # task: Если использовать подобный подход, нужно обязательно производить с перекрытием
    # (придумать как)

    # По сути это гиперпараметр, но не вынесен в таковые, т.к. это временное baseline решение
    # pylint: disable-next=invalid-name)
    FIRST_COORD_ROUND = 3  # (2) сотые ~= 1 км, (3) тысячные ~= 0.1 км

    # task: Устранить SettingWithCopyWarning

    # Первоначальный вариант (в 'lat_lon_round' мы получим строковые представления округленных
    # координат):
    # df_dataset.loc["lat_lon_group"] = (
    #     df_dataset.latitude.map(lambda x: str(round(x, FIRST_COORD_ROUND)))
    #     + "_"
    #     + df_dataset.longitude.map(lambda x: str(round(x, FIRST_COORD_ROUND)))
    # )
    # Альтернативный вариант (результат аналогичный, за исключенем того, что в 'lat_lon_round'
    # мы получим номера групп):
    df_dataset["lat_lon_group"] = df_dataset.groupby(
        [
            df_dataset.latitude.round(FIRST_COORD_ROUND),
            df_dataset.longitude.round(FIRST_COORD_ROUND),
        ]
    ).ngroup()

    # ====Формирование пар-кандидитов====

    columns_to_pairs = ["lat_lon_group"]  # колонки для совоставления в пары
    df_pairs = pd.merge(df_dataset, df_dataset, on=columns_to_pairs, suffixes=["_1", "_2"])

    # Оставляем пары только в одном направлении (id_1, id_2) или в обоих (id_1, id_2) и (id_2, id_1)
    if drop_order_dub:
        df_pairs = df_pairs[df_pairs.id_1 < df_pairs.id_2]
    else:  # удаляем только полные дубликаты (id_1, id_1)
        df_pairs = df_pairs[df_pairs.id_1 != df_pairs.id_2]

    # Generate metrics for current candidates
    metrics = {}
    if get_metrics and not real_test:
        labels = np.array(get_match_label(df_pairs))
        metrics = get_pairs_metrics(df_dataset, df_pairs, labels, drop_order_dub)

    return df_pairs, metrics


def run_futures_pipeline(
    dataset: pd.DataFrame, futures_pipeline: list, prefix="ftr_"
) -> pd.DataFrame:
    """Последовательно пропускает датасет через переданный набор функций, генерирующих фичи.

    Args:
        dataset (pd.DataFrame): Датасет, на основе которого генерируются фичи
        futures_pipeline (list): Список функций и их дополнительных параметров. Формат следующий:
            futures_pipeline = [
                {"func": generate_feature1, "params": {"param1": 1, "param2": "2" ...}},
                {"func": generate_feature2, "params": {}},
            ]
            Каждая функция должна:
                - первым параметром принимать датасет,
                - вторым параметром принимать prefix,
                - возвращать датасет в котором добавлены сгенерированные фичи (если необходимо,
                  для целей экономии памяти, допустимо внутри функций удалять колонки, которые
                  далее в пайплайне точно не будут использоваться)
        prefix (str, optional): Префикс, который должен добавляться ко всем колонкам
            сгенерированных фич, на основании него будет формироваться итоговый набор колонок
            для передачи в модель (т.е. только колонки с данным префиксом будут переданы в модель).
            Defaults to "ftr_".

    Returns:
        pd.DataFrame: Датасет, со сгенерированными фичами (колонки со сгенерированными фичами имеют
        префикс из prefix)
    """
    for step in futures_pipeline:
        dataset = step["func"](dataset, prefix=prefix, **step["params"])
    future_columns = [col for col in dataset.columns if col.startswith(prefix)]
    return dataset.reset_index(drop=True), future_columns


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
        pd.DataFrame: Парный датасет + колонка "{prefix}geo_distance" с расстоянием между точками
    """

    # task: Скорей всего нет смысла считать точно через haversine (с учетом шарообразности земли),
    # предпочтительней считать более грубо, но быстрее

    df_pairs[f"{prefix}geo_distance"] = df_pairs.apply(
        lambda x: hs.haversine(
            (x.latitude_1, x.longitude_1), (x.latitude_2, x.longitude_2), unit=Unit.KILOMETERS
        ),
        axis=1,
    )

    if normalize:
        pass

    return df_pairs


def add_feauture_levenshtein_distance(
    df_pairs: pd.DataFrame, normalize=False, prefix="ftr_"
) -> pd.DataFrame:
    """Добавляем фичи для парного датасета - расстояние Левенштейна между двумя строками.

    В текущем варианте добавляет расстояние только между name, но нужно будет расширить,
    как минимум еще на категории (основная проблема с ними, наличие пропущенных значений)

    Args:
        df_pairs (pd.DataFrame): Парный датасет в котором присутсвуют колонки:
            name_1, name_2
        normalize (bool, optional): На будущее заложена возможность нормализации, в текущий момент
            не реализовано. Defaults to False.
        prefix (str, optional): Префикс, перед названием колонки. Defaults to "ftr_".

    Returns:
        pd.DataFrame: Парный датасет + :
            - колонка "{prefix}name_levenshtein" расстояние Левенштейна между именами
    """
    df_pairs[f"{prefix}name_levenshtein"] = df_pairs.apply(
        lambda x: fuzz.token_set_ratio(x.name_1, x.name_2), axis=1
    )

    if normalize:
        pass

    return df_pairs


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
