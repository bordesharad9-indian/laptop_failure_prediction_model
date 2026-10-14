import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.components.data_transformation import NUMERICAL_COLUMNS, USE_SYSTEM_EVENT
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")


def _risk_level(probability_pct: float) -> str:
    if probability_pct >= 70:
        return "High"
    if probability_pct >= 40:
        return "Medium"
    return "Low"


class PredictPipeline:
    # loaded ONCE and shared, so live monitoring (a request every 1-2 s)
    # does not re-read the pickle files on every prediction
    _model = None
    _preprocessor = None

    def __init__(self):
        pass

    def _load(self):
        try:
            if PredictPipeline._model is None or PredictPipeline._preprocessor is None:
                logging.info("Loading model and preprocessor from artifacts")
                PredictPipeline._model = load_object(file_path=MODEL_PATH)
                PredictPipeline._preprocessor = load_object(file_path=PREPROCESSOR_PATH)
            return PredictPipeline._model, PredictPipeline._preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def reload(cls):
        """Call after retraining to pick up the new model without restarting."""
        cls._model = None
        cls._preprocessor = None

    def predict(self, features: pd.DataFrame):
        """Class labels (0 = healthy, 1 = failure). Kept for backward compatibility."""
        try:
            model, preprocessor = self._load()
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_risk(self, features: pd.DataFrame) -> dict:
        """
        Returns, for the first row of `features`:
            probability : failure probability in % (0-100)
            level       : "Low" / "Medium" / "High"
            failure     : True if probability >= 50
            prediction  : 0 or 1
        """
        try:
            model, preprocessor = self._load()
            data_scaled = preprocessor.transform(features)

            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                idx = classes.index(1) if 1 in classes else -1
                probability = float(model.predict_proba(data_scaled)[0][idx]) * 100
            else:
                probability = float(model.predict(data_scaled)[0]) * 100

            probability = round(probability, 1)
            return {
                "probability": probability,
                "level": _risk_level(probability),
                "failure": probability >= 50,
                "prediction": int(probability >= 50),
            }
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    One reading of the 9 sensor values the model was trained on.

    - Any value may be None (sensor not readable on this laptop). It is sent
      as NaN and the preprocessor fills it with the training median.
    - System_Event is accepted only so older forms/code do not break; it is
      ignored unless USE_SYSTEM_EVENT = True in data_transformation.py
      (not recommended: it leaks the label and is unknown during live use).
    """

    # keys used by the live-monitoring UI  ->  dataset column names
    UI_TO_COLUMN = {
        "cpu_temp": "CPU_Temp_C",
        "gpu_temp": "GPU_Temp_C",
        "cpu_usage": "CPU_Usage_Pct",
        "gpu_usage": "GPU_Usage_Pct",
        "ram_usage": "RAM_Usage_Pct",
        "disk_usage": "Disk_Usage_Pct",
        "disk_io": "Disk_IO_MBps",
        "battery": "Battery_Level_Pct",
        "fan_speed": "Fan_Speed_RPM",
    }

    def __init__(
        self,
        CPU_Temp_C: Optional[float] = None,
        GPU_Temp_C: Optional[float] = None,
        CPU_Usage_Pct: Optional[float] = None,
        GPU_Usage_Pct: Optional[float] = None,
        RAM_Usage_Pct: Optional[float] = None,
        Disk_Usage_Pct: Optional[float] = None,
        Disk_IO_MBps: Optional[float] = None,
        Battery_Level_Pct: Optional[float] = None,
        Fan_Speed_RPM: Optional[float] = None,
        System_Event: Optional[str] = None,
    ):
        self.CPU_Temp_C = self._to_float(CPU_Temp_C)
        self.GPU_Temp_C = self._to_float(GPU_Temp_C)
        self.CPU_Usage_Pct = self._to_float(CPU_Usage_Pct)
        self.GPU_Usage_Pct = self._to_float(GPU_Usage_Pct)
        self.RAM_Usage_Pct = self._to_float(RAM_Usage_Pct)
        self.Disk_Usage_Pct = self._to_float(Disk_Usage_Pct)
        self.Disk_IO_MBps = self._to_float(Disk_IO_MBps)
        self.Battery_Level_Pct = self._to_float(Battery_Level_Pct)
        self.Fan_Speed_RPM = self._to_float(Fan_Speed_RPM)
        self.System_Event = System_Event

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_ui_dict(cls, d: dict) -> "CustomData":
        """Build from the live-monitoring UI / /metrics keys (cpu_temp, gpu_temp, ...)."""
        values = {col: d.get(key) for key, col in cls.UI_TO_COLUMN.items()}
        return cls(**values, System_Event=d.get("system_event"))

    def missing_fields(self) -> list:
        """Sensors with no value (filled with the training median)."""
        return [c for c in NUMERICAL_COLUMNS if getattr(self, c) is None]

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            row = {
                col: [np.nan if getattr(self, col) is None else getattr(self, col)]
                for col in NUMERICAL_COLUMNS
            }
            if USE_SYSTEM_EVENT:
                row["System_Event"] = [self.System_Event or "NORMAL_OPERATION"]
            return pd.DataFrame(row)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # quick manual test: python -m src.pipeline.predict_pipeline
    sample = CustomData(
        CPU_Temp_C=92, GPU_Temp_C=88, CPU_Usage_Pct=95, GPU_Usage_Pct=90,
        RAM_Usage_Pct=94, Disk_Usage_Pct=85, Disk_IO_MBps=300,
        Battery_Level_Pct=20, Fan_Speed_RPM=5200,
    )
    print(PredictPipeline().predict_risk(sample.get_data_as_data_frame()))
    print("prediction pipeline run successfully.")