"""Module for generate pairs candidate to match"""


from typing import Tuple
import json
import pandas as pd
import numpy as np
import click
from src.common_funcs import (
    get_submission_true,
    get_submission_predict,
    jaccard_score,
    get_match_label,
)


# Hyperparameters
PAIRS_DROP_ORDER_DUBLICATES = True


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


@click.command()
@click.argument("input_dataset_path", type=click.Path(exists=True))
@click.argument("output_pairs_dataset_path", type=click.Path())
@click.argument("output_pairs_metrics_path", type=click.Path())
@click.option("--get_metrics", default=False, type=click.BOOL)
@click.option("--real_test", default=False, type=click.BOOL)
def generate_pairs_csv(
    input_dataset_path: str,
    output_pairs_dataset_path: str,
    output_pairs_metrics_path: str,
    get_metrics=False,
    real_test=False,
) -> None:
    """На остове датафрейма, отбираем/генерируем выходной файл пар-кандидатов на сравнение,
    для задачи дальнейшей попарной бинарной классификации. Кроме того, при необходимости,
    считаем и сохраняем в json формате метрики качества отобранных кандидатов.

    Args:
        input_dataset_path (str): Путь к csv файлу исходного датасета (для формирования
            пар-кандидатов).
        output_pairs_dataset_path (str): Путь до выходного csv файла пар-кандидатов.
        output_pairs_metrics_path (str): Путь до выходного json файла метрик кандидатов.
        get_metrics (bool, optional): Осуществлять ли подсчет метрик. Defaults to False.
        real_test (bool, optional): Имеем мы дело с реальным test.csv датасетом, или каким-то
            другим (развернутое описание в функции generate_pairs_df). Defaults to False.
    """

    df_input = pd.read_csv(input_dataset_path)
    df_pairs, metrics_of_pairs = generate_pairs_df(
        df_input, PAIRS_DROP_ORDER_DUBLICATES, get_metrics=get_metrics, real_test=real_test
    )

    df_pairs.to_csv(output_pairs_dataset_path, index=False)

    if get_metrics:
        with open(output_pairs_metrics_path, "w", encoding="UTF-8") as metrics_file:
            json.dump(metrics_of_pairs, metrics_file)
        print(metrics_of_pairs)


if __name__ == "__main__":
    generate_pairs_csv()  # pylint: disable=E1120
