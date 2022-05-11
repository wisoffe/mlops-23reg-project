"""Module for split raw test to train/test"""

import click
import pandas as pd
from sklearn.model_selection import GroupKFold


@click.command()
@click.argument("input_raw_train_path", type=click.Path(exists=True))
@click.argument("output_split_train_path", type=click.Path())
@click.argument("output_split_test_path", type=click.Path())
@click.option("--n_splits", default=3, type=click.INT)
def raw_train_test_split(
    input_raw_train_path: str,
    output_split_train_path: str,
    output_split_test_path: str,
    n_splits=3,
) -> None:
    """Разбиваем исходный train датасет на train/test таким образом, что бы объекты с
    одинаковым POI не раскидывались по разным выборкам.

    Args:
        input_raw_train_path (str): Путь до исходного (raw) train.csv
        output_split_train_path (str): Путь до выходного split_train.csv
        output_split_test_path (str): Путь до выходного split_test.csv
        n_splits (int): Количество частей, на которые будет разбит train датасет, при этом,
            одна из частей уйдет на split_test, все остальные на split_train, например:
            3 = 1/2 (или 33%/88%), 4 = 1/3 (или 25%/75) и т.д.. Defaults to 3:int.
    """

    train = pd.read_csv(input_raw_train_path)

    # Подход по применению GroupKFold для подобных целей подсмотрел здесь:
    # https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline

    g_kfold = GroupKFold(n_splits=n_splits)
    for i, (_, val_idx) in enumerate(
        g_kfold.split(train, train["point_of_interest"], train["point_of_interest"])
    ):
        train.loc[val_idx, "parts"] = str(i)

    split_test = train[train.parts == "1"].drop(columns="parts")
    split_train = train[~(train.parts == "1")].drop(columns="parts")

    split_train.to_csv(output_split_train_path, index=False)
    split_test.to_csv(output_split_test_path, index=False)

    print(f"Our train size: {len(split_train)}, Our test size: {len(split_test)}")


if __name__ == "__main__":
    raw_train_test_split()
