import os
import sys
from flask import Flask, request, render_template

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(
    __name__,
    template_folder=os.path.join(os.getcwd(), "templates")
)

app = application


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        # Read the nine laptop sensor values from the HTML form
        data = CustomData(
            CPU_Temp_C=request.form.get("cpu_temp"),
            GPU_Temp_C=request.form.get("gpu_temp"),
            CPU_Usage_Pct=request.form.get("cpu_usage"),
            GPU_Usage_Pct=request.form.get("gpu_usage"),
            RAM_Usage_Pct=request.form.get("ram_usage"),
            Disk_Usage_Pct=request.form.get("disk_usage"),
            Disk_IO_MBps=request.form.get("disk_io"),
            Battery_Level_Pct=request.form.get("battery_level"),
            Fan_Speed_RPM=request.form.get("fan_speed"),
        )

        pred_df = data.get_data_as_data_frame()
        logging.info("DataFrame created for prediction:\n%s", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Output mapping: 0 -> Healthy, 1 -> Risk of Failure
        prediction_label = (
            "Risk of Failure" if results[0] == 1 else "Healthy (No Failure Risk)"
        )

        return render_template(
            "home.html",
            results=prediction_label,
            pred_code=int(results[0]),
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)