import os
import pickle
import sys

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Pickle `obj` to `file_path`.
    Written to a temp file first and then swapped in, so the live app can
    never read a half-written model.pkl / preprocessor.pkl while you retrain.
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp_path = file_path + ".tmp"
        with open(tmp_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        os.replace(tmp_path, file_path)

        logging.info(f"Saved object to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{file_path} not found. Train first with: "
                "python -m src.components.data_ingestion"
            )
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, selection="cv"):
    """
    Tunes every model and returns {model_name: F1 score}.

    selection="cv"   (default) -> mean 5-fold cross-validated F1 on the TRAIN
                                  set. Use this to choose the best model, so the
                                  test set is not used for selection.
    selection="test"           -> F1 on the test set (the old behaviour; fine
                                  for reporting, not for choosing a model).

    Each model in `models` is left with its best parameters and fitted on the
    full training set.
    """
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            if params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                cv_score = gs.best_score_
            else:
                cv_score = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
                ).mean()

            model.fit(X_train, y_train)

            if selection == "test":
                score = f1_score(y_test, model.predict(X_test), zero_division=0)
            else:
                score = cv_score

            report[model_name] = score
            logging.info(f"{model_name}: {selection} F1 = {score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)