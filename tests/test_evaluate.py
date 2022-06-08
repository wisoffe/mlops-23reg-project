"""Unit test for ./src/model/evaluate.py"""
from click.testing import CliRunner
from src.models.evaluate import evaluate_cli


INPUT_ORIGINAL_DATASET_PATH = "./data/interim/split_test.csv"
INPUT_SUBMISSION_PRED_PATH = "./data/processed/submission_pred_split.csv"
OUTPUT_METRICS_PATH = "./reports/metrics_of_split_test_final.json"


# Initialize runner
runner = CliRunner()


def test_cli_command():
    """Test execute module from CLI"""
    result = runner.invoke(
        evaluate_cli,
        "{} {} {}".format(  # pylint: disable=consider-using-f-string
            INPUT_ORIGINAL_DATASET_PATH,
            INPUT_SUBMISSION_PRED_PATH,
            OUTPUT_METRICS_PATH,
        ),
    )
    assert result.exit_code == 0
