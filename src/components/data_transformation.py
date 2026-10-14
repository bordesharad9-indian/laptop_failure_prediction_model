import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



USE_SYSTEM_EVENT = False

NUMERICAL_COLUMNS = [
    "CPU_Temp_C",
    "GPU_Temp_C",
    "CPU_Usage_Pct",
    "GPU_Usage_Pct",
    "RAM_Usage_Pct",
    "Disk_Usage_Pct",
    "Disk_IO_MBps",
    "Battery_Level_Pct",
    "Fan_Speed_RPM",
]
CATEGORICAL_COLUMNS = ["System_Event"]
TARGET_COLUMN = "Failure_Status"


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_feature_columns(self):
        """Columns the model is trained on (and the live app must send)."""
        if USE_SYSTEM_EVENT:
            return NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
        return list(NUMERICAL_COLUMNS)

    def get_data_transformer_object(self):
        """
        Builds the preprocessing object (imputation + scaling).
        Machine_ID, Timestamp and any other unlisted column are dropped
        automatically (remainder="drop").
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            transformers = [("num_pipeline", num_pipeline, NUMERICAL_COLUMNS)]

            if USE_SYSTEM_EVENT:
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                )
                transformers.append(
                    ("cat_pipelines", cat_pipeline, CATEGORICAL_COLUMNS)
                )

            logging.info(f"Numerical columns: {NUMERICAL_COLUMNS}")
            logging.info(
                f"Categorical columns used: "
                f"{CATEGORICAL_COLUMNS if USE_SYSTEM_EVENT else 'none (System_Event excluded)'}"
            )

            # sparse_threshold=0 -> always dense, so np.c_ below always works
            preprocessor = ColumnTransformer(
                transformers, remainder="drop", sparse_threshold=0
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            feature_columns = self.get_feature_columns()
            required = feature_columns + [TARGET_COLUMN]
            for name, df in (("train", train_df), ("test", test_df)):
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"{name} data is missing columns: {missing}")

            # rows without a label cannot be used
            train_df = train_df.dropna(subset=[TARGET_COLUMN])
            test_df = test_df.dropna(subset=[TARGET_COLUMN])

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # select the model columns explicitly (instead of "everything but target")
            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[TARGET_COLUMN].astype(int)

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[TARGET_COLUMN].astype(int)

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit on TRAIN only, then transform TEST
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(
                f"Transformed feature count: {input_feature_train_arr.shape[1]}"
            )

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)