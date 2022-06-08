"""Unit test for ./src/model/predict_model.py"""
from click.testing import CliRunner
import pandas as pd
import great_expectations as ge
from src.models.predict_model import predict_model_cli


INPUT_MODEL_PATH = "./models/model_on_split_df.pkl"
INPUT_ORIGINAL_DATASET_PATH = "./data/interim/split_test.csv"
INPUT_PAIR_FEATURES_DATASET_PATH = "./data/processed/split_test_pair_feautures.csv"
OUTPUT_SUBMISSION_PATH = "./data/processed/submission_pred_split.csv"


# Initialize runner
runner = CliRunner()


def test_cli_command():
    """Test execute module from CLI"""
    result = runner.invoke(
        predict_model_cli,
        "{} {} {} {}".format(  # pylint: disable=consider-using-f-string
            INPUT_MODEL_PATH,
            INPUT_ORIGINAL_DATASET_PATH,
            INPUT_PAIR_FEATURES_DATASET_PATH,
            OUTPUT_SUBMISSION_PATH,
        ),
    )
    assert result.exit_code == 0


def test_output():
    """Test format output files"""
    submission_df = pd.read_csv(OUTPUT_SUBMISSION_PATH)
    df_ge = ge.from_pandas(submission_df)

    expected_columns = ["id", "matches"]
    assert (
        df_ge.expect_table_columns_to_match_ordered_list(column_list=expected_columns).success
        is True
    )
    assert df_ge.expect_column_values_to_be_unique(column="id").success is True is True
    assert df_ge.expect_column_values_to_not_be_null(column="matches").success is True is True
