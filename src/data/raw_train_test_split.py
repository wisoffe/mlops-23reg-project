"""Module for split raw test to train/test"""


from typing import Tuple
from sklearn.model_selection import GroupKFold
import click
import pandas as pd


def train_test_group_split_df(
    dataset: pd.DataFrame,
    n_splits=3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбиваем исходный train датасет на train/test таким образом, что бы объекты с
    одинаковым POI (Group) не раскидывались по разным выборкам. Работает с pd.DataFrame.

    Args:
        dataset (pd.DataFrame): Исходный датафрейм (обычно raw_train)
        n_splits (int, optional): Количество частей, на которые будет разбит train датасет,
            при этом, одна из частей уйдет на split_test, все остальные на split_train,
            например: 3 = 1/2 (или 33%/88%), 4 = 1/3 (или 25%/75) и т.д.. Defaults to 3.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            [0] - выходной split_train датафрейм.
            [1] - выходной split_test датафрейм.
    """

    # Подход по применению GroupKFold для подобных целей подсмотрел здесь:
    # https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline
    g_kfold = GroupKFold(n_splits=n_splits)
    for i, (_, val_idx) in enumerate(
        g_kfold.split(dataset, dataset["point_of_interest"], dataset["point_of_interest"])
    ):
        dataset.loc[val_idx, "parts"] = str(i)

    split_test = dataset[dataset.parts == "1"].drop(columns="parts")
    split_train = dataset[~(dataset.parts == "1")].drop(columns="parts")

    print(f"Our train size: {len(split_train)}, Our test size: {len(split_test)}")

    return split_train, split_test


@click.command()
@click.argument("input_raw_train_path", type=click.Path(exists=True))
@click.argument("output_split_train_path", type=click.Path())
@click.argument("output_split_test_path", type=click.Path())
@click.option("--n_splits", default=3, type=click.INT)
def train_test_group_split_csv(
    input_raw_train_path: str,
    output_split_train_path: str,
    output_split_test_path: str,
    n_splits=3,
) -> None:
    """Разбиваем исходный train датасет на train/test таким образом, что бы объекты с
    одинаковым POI (Group) не раскидывались по разным выборкам. Работает с файлами csv.

    Args:
        input_raw_train_path (str): Путь до исходного датасета .csv (обычно train.csv)
        output_split_train_path (str): Путь до выходного split_train.csv
        output_split_test_path (str): Путь до выходного split_test.csv
        n_splits (int, optional): Количество частей, на которые будет разбит train датасет,
        при этом, одна из частей уйдет на split_test, все остальные на split_train,
        например: 3 = 1/2 (или 33%/88%), 4 = 1/3 (или 25%/75) и т.д.. Defaults to 3:int.
    """

    raw_train = pd.read_csv(input_raw_train_path)

    split_train, split_test = train_test_group_split_df(raw_train, n_splits=n_splits)

    split_train.to_csv(output_split_train_path, index=False)
    split_test.to_csv(output_split_test_path, index=False)


if __name__ == "__main__":
    train_test_group_split_csv()  # pylint: disable=E1120
