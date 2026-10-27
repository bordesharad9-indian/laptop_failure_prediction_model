import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        CPU_Temp_C: float,
        GPU_Temp_C: float,
        CPU_Usage_Pct: float,
        GPU_Usage_Pct: float,
        RAM_Usage_Pct: float,
        Disk_Usage_Pct: float,
        Disk_IO_MBps: float,
        Battery_Level_Pct: float,
        Fan_Speed_RPM: float
    ):

        self.CPU_Temp_C = CPU_Temp_C
        self.GPU_Temp_C = GPU_Temp_C
        self.CPU_Usage_Pct = CPU_Usage_Pct
        self.GPU_Usage_Pct = GPU_Usage_Pct
        self.RAM_Usage_Pct = RAM_Usage_Pct
        self.Disk_Usage_Pct = Disk_Usage_Pct
        self.Disk_IO_MBps = Disk_IO_MBps
        self.Battery_Level_Pct = Battery_Level_Pct
        self.Fan_Speed_RPM = Fan_Speed_RPM

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CPU_Temp_C": [float(self.CPU_Temp_C)],
                "GPU_Temp_C": [float(self.GPU_Temp_C)],
                "CPU_Usage_Pct": [float(self.CPU_Usage_Pct)],
                "GPU_Usage_Pct": [float(self.GPU_Usage_Pct)],
                "RAM_Usage_Pct": [float(self.RAM_Usage_Pct)],
                "Disk_Usage_Pct": [float(self.Disk_Usage_Pct)],
                "Disk_IO_MBps": [float(self.Disk_IO_MBps)],
                "Battery_Level_Pct": [float(self.Battery_Level_Pct)],
                "Fan_Speed_RPM": [float(self.Fan_Speed_RPM)],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
