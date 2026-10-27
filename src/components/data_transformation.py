import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

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

TARGET_COLUMN = "Hardware_Failure"


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, NUMERICAL_COLUMNS)]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df[NUMERICAL_COLUMNS]
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df[NUMERICAL_COLUMNS]
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
