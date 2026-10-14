import os
import sys
import mlflow
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# single source of truth for column names (defined in data_transformation.py)
from src.components.data_transformation import (
    DataTransformation,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    source_data_path: str = os.path.join(
        "notebook", "data", "system_telemetry_dataset.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

        mlflow.set_experiment("Laptop_Failure_Prediction")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            with mlflow.start_run(run_name="Data_Ingestion"):

               mlflow.set_tag("pipeline_stage", "data_ingestion")
            
            df = pd.read_csv(self.ingestion_config.source_data_path)
            df.columns = df.columns.str.strip()
            logging.info(f"Read the dataset as dataframe, shape = {df.shape}")

            # ---- validate columns ----
            required = NUMERICAL_COLUMNS + [TARGET_COLUMN]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Dataset is missing columns: {missing}")

            # ---- basic cleaning ----
            rows_before = len(df)
            df = df.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
            df = df.reset_index(drop=True)
            
            logging.info(f"Rows removed during cleaning: {rows_before - len(df)}")

            # ---- information for you in the log ----
            
          
            logging.info(
                "Class balance: "
                f"{df[TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict()}"
            )
            if "System_Event" in df.columns:
                logging.info(
                    "System_Event vs Failure_Status (leakage check):\n"
                    f"{pd.crosstab(df['System_Event'], df[TARGET_COLUMN])}"
                )

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # ---- stratified split keeps the same failure % in train and test ----
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df[TARGET_COLUMN],
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            mlflow.log_params({
                "test_size": 0.2,
                "random_state": 42,
                "target_column": TARGET_COLUMN,
                "feature_count": len(NUMERICAL_COLUMNS),
            })


            mlflow.log_metric("train_rows", len(train_set))
            mlflow.log_metric("test_rows", len(test_set))

            mlflow.log_metrics({
                "rows_before_cleaning": rows_before,
                "rows_after_cleaning": len(df),
                "rows_removed": rows_before - len(df),
                "train_rows": len(train_set),
                "test_rows": len(test_set),
                "failure_rate": float(df[TARGET_COLUMN].mean()),
            })

            for path in [
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            ]:
                mlflow.log_artifact(path, artifact_path="datasets")

            

            logging.info(
                f"Ingestion of the data is completed "
                f"(train={len(train_set)}, test={len(test_set)})"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))