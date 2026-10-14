import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import mlflow
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.components.data_transformation import NUMERICAL_COLUMNS
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
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            logging.info(
                f"Train shape={X_train.shape}, failures in train="
                f"{y_train.mean():.3f}, failures in test={y_test.mean():.3f}"
            )

            # NOTE: n_jobs is set on GridSearchCV only (avoids nested parallelism)
            models = {
                "Random Forest": RandomForestClassifier(
                    random_state=42, class_weight="balanced"
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42, 
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [64, 120, 256],
                    "max_depth": [None, 10],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 7, 10],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [64, 120, 256],
                    "subsample": [0.8, 1.0],
                },
            }

            # ---- model selection uses CROSS-VALIDATION ON TRAIN ONLY ----
            # (the test set is touched once, at the very end)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            searches = {}

            for name, model in models.items():
                fit_params = {}
                if name == "Gradient Boosting":
                    # GB has no class_weight option -> balance with sample weights
                    fit_params["sample_weight"] = compute_sample_weight(
                        "balanced", y_train
                    )

                gs = GridSearchCV(
                    model, params[name], scoring="f1", cv=cv, n_jobs=-1
                )
                gs.fit(X_train, y_train, **fit_params)
                searches[name] = gs
                logging.info(
                    f"{name}: CV F1={gs.best_score_:.4f}, best params={gs.best_params_}"
                )

            best_model_name = max(searches, key=lambda n: searches[n].best_score_)
            best_cv_f1 = searches[best_model_name].best_score_
            best_model = searches[best_model_name].best_estimator_  # refit on full train

            logging.info(f"Best model (by CV F1 on train): {best_model_name}")
            if best_cv_f1 < 0.6:
                logging.warning(
                    f"Best CV F1 is only {best_cv_f1:.3f} - the model may be weak."
                )

            # ---- single final evaluation on the untouched test set ----
            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            prec = precision_score(y_test, predicted, zero_division=0)
            rec = recall_score(y_test, predicted, zero_division=0)
            f1 = f1_score(y_test, predicted, zero_division=0)

            logging.info(
                f"TEST -> accuracy={acc:.4f}, precision={prec:.4f}, "
                f"recall={rec:.4f}, f1={f1:.4f}"
            )

            # ---- feature importances (is any single column dominating?) ----
            if hasattr(best_model, "feature_importances_") and X_train.shape[1] == len(
                NUMERICAL_COLUMNS
            ):
                importances = pd.Series(
                    best_model.feature_importances_, index=NUMERICAL_COLUMNS
                ).sort_values(ascending=False)
                logging.info(f"Feature importances:\n{importances.round(3)}")
                print("\nFeature importances:\n", importances.round(3).to_string())

            # ---- readable summary in the terminal ----
            print("\nCross-validated F1 (train):")
            for name, gs in searches.items():
                print(f"  {name:<18} {gs.best_score_:.4f}")
            print(f"\nBest Model: {best_model_name}")
            print("Confusion matrix (test) [[TN FP] [FN TP]]:")
            print(confusion_matrix(y_test, predicted))
            print(classification_report(y_test, predicted, digits=3, zero_division=0))
            os.makedirs("artifacts", exist_ok=True)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Saved trained model")
            # Track the model in MLflow
            mlflow.set_experiment("Laptop_Failure_Prediction")
            with mlflow.start_run(
    run_name="Model_Training",
    nested=True
):
                mlflow.set_tag("pipeline_stage", "model_training")
                mlflow.set_tag("best_model_name", best_model_name)

                mlflow.log_params({
        "model_name": best_model_name,
        "cv_f1": float(best_cv_f1),
        **{
            f"best_{key}": value
            for key, value in searches[best_model_name].best_params_.items()
        },
           })


            mlflow.log_metrics({
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
    })
            mlflow.log_artifact(
        self.model_trainer_config.trained_model_file_path,
        artifact_path="local_model",
    )

     

            
            logging.info("Model, parameters and metrics logged to MLflow")


           
            

    
            return f1

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("Model trainer module loaded successfully.")