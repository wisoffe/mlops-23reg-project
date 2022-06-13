"""Module for ML service on FastAPI"""

import os
import sys
import pandas as pd
import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from src.data.generate_pairs import generate_pairs_df
from src.features.build_pair_features import build_pair_feautures_df
from src.models.predict_model import predict_model_in_memory, PAIRS_DROP_ORDER_DUBLICATES

# Load the environment variables from the .env file into the application
load_dotenv(override=True)
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Initialize the FastAPI application
app = FastAPI()


# Create a class to store the deployed model & use it for prediction
class Model:  # pylint: disable=too-few-public-methods
    """General model class for inference"""

    def __init__(self, model_name: str, model_stage: str, inference_params: dict) -> None:
        """
        To initialize the model
        model_name: Name of the model in registry
        model_stage: Stage of the model
        """
        # Load the model from Registry
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
        self.inference_params = inference_params

    def predict(self, df_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """

        df_pairs, _ = generate_pairs_df(
            df_dataset,
            drop_order_dub=self.inference_params["drop_order_dub"],
            get_metrics=False,
            real_test=True,
        )

        df_pair_feautures = build_pair_feautures_df(df_pairs, add_target_label=False)

        submission_pred = predict_model_in_memory(self.model, df_dataset[["id"]], df_pair_feautures)

        return submission_pred


global_inference_params = {"drop_order_dub": PAIRS_DROP_ORDER_DUBLICATES}

# Create model
model = Model("general_model_xboost", "Staging", global_inference_params)


# Create the POST endpoint with path '/invocations'
@app.post("/invocations", response_class=FileResponse)
async def create_upload_file(file: UploadFile = File(...)):
    """Generates .csv file of the submission format by input .csv file of the original format.

    Args:
        file (UploadFile, optional): file .csv with columns:
            id,name,latitude,longitude,address,city,state,zip,country,url,phone,categories
            Defaults to File(...).

    Raises:
        HTTPException: If file not .csv

    Returns:
        FileResponse: Return File Response (file in .csv format with columns: id,matches)
    """
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):  # pylint: disable=no-else-return
        # Create a temporary file with the same name as the uploaded
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb") as input_file:
            input_file.write(file.file.read())

        data = pd.read_csv(file.filename)
        os.remove(file.filename)  # Return a JSON object containing the model predictions

        # response file path
        response_file_path = "submission.csv"
        model.predict(data).to_csv(response_file_path, index=False)
        return response_file_path

    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


# Check if the environment variables for AWS access are available.
# If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    sys.exit(1)
