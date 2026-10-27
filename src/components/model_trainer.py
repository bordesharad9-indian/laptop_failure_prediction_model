import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            best_score = 0
            best_model_name = ""
            best_model = None

            for model_name, model in models.items():
                grid = GridSearchCV(model, params[model_name], cv=3, scoring='f1', n_jobs=-1)
                grid.fit(X_train, y_train)

                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                f1 = f1_score(y_test, predictions)

                if f1 > best_score:
                    best_score = f1
                    best_model_name = model_name
                    best_model = model

            if best_score < 0.6:
                raise CustomException("No best model found with sufficient score", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
