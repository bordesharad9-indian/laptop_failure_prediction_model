
from flask import Flask, request, render_template
import os
import sys

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

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
            Battery_Level_Pct=request.form.get("battery"),
            Fan_Speed_RPM=request.form.get("fan_speed"),
        )

        # Convert input into a DataFrame
        pred_df = data.get_data_as_data_frame()
        logging.info("Laptop sensor data received")

        # Predict failure risk
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict_risk(pred_df)

        # Identify sensors with missing values
        missing_fields = data.missing_fields()

        logging.info(
            "Prediction completed. Risk level: %s",
            result["level"]
        )

        return render_template(
            "home.html",
            results=result,
            missing_fields=missing_fields
        )

    except Exception as e:
        logging.exception("Laptop failure prediction failed")

        return render_template(
            "home.html",
            error="Prediction failed. Check the application logs and model artifacts."
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
