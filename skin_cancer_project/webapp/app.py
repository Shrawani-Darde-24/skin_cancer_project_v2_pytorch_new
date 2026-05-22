"""
Enhanced Skin Cancer Classifier Web App
=========================================
Fixed Import + Path Version
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import uuid
import pickle
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    Response,
    stream_with_context
)

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src folder to Python path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT PROJECT MODULES
# ─────────────────────────────────────────────────────────────────────────────

from heatmap import generate_risk_heatmap

from alerts import (
    dispatch_alerts,
    get_browser_alert_stream
)

from monitoring import AbnormalityMonitor

from reminders import (
    ReminderScheduler,
    get_reminder_stream
)

from seasonal_risk import compute_seasonal_risk

from time_series import (
    save_visit,
    analyze_progression
)

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
DATA_FOLDER   = os.path.join(PROJECT_ROOT, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_V2 = os.path.join(
    PROJECT_ROOT,
    "models",
    "skin_cancer_model_v2.pkl"
)

MODEL_V1 = os.path.join(
    PROJECT_ROOT,
    "models",
    "skin_cancer_model.pkl"
)

predictor = None
cnn_model = None
model_ver = None

# ─────────────────────────────────────────────────────────────────────────────
# MODULE INSTANCES
# ─────────────────────────────────────────────────────────────────────────────

monitor   = AbnormalityMonitor()
scheduler = ReminderScheduler()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def get_predictor():

    global predictor
    global cnn_model
    global model_ver

    if predictor is not None:
        return predictor

    # V2 MODEL
    if os.path.exists(MODEL_V2):

        from predict_v2 import SkinCancerPredictorV2

        predictor = SkinCancerPredictorV2(MODEL_V2)

        cnn_model = predictor.cnn_model

        model_ver = "v2"

    # V1 MODEL
    elif os.path.exists(MODEL_V1):

        from predict import SkinCancerPredictor

        predictor = SkinCancerPredictor(MODEL_V1)

        model_ver = "v1"

    else:
        print("No trained model found!")

    return predictor

# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Skin Cancer AI</title>
<style>
body{
    font-family:Arial;
    background:#f5f5f5;
    padding:40px;
}
.card{
    background:white;
    padding:30px;
    border-radius:12px;
    max-width:700px;
    margin:auto;
}
button{
    background:#534AB7;
    color:white;
    border:none;
    padding:12px 20px;
    border-radius:8px;
    cursor:pointer;
}
</style>
</head>
<body>

<div class="card">
<h1>Skin Cancer AI System</h1>

<form id="uploadForm">

<input type="file" name="image" required><br><br>

<input type="text"
       name="patient_id"
       placeholder="Patient ID"><br><br>

<button type="submit">
Analyze
</button>

</form>

<div id="result"></div>

</div>

<script>

document.getElementById("uploadForm")
.addEventListener("submit", async function(e){

    e.preventDefault();

    const formData = new FormData(this);

    const response = await fetch("/predict_full", {
        method:"POST",
        body:formData
    });

    const data = await response.json();

    document.getElementById("result").innerHTML =
        "<pre>"+JSON.stringify(data,null,2)+"</pre>";

});

</script>

</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/model_info")
def model_info():

    get_predictor()

    return jsonify({
        "version": model_ver or "none"
    })

# ─────────────────────────────────────────────────────────────────────────────
# ALERT STREAM
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/alerts/stream")
def alert_stream():

    return Response(
        stream_with_context(
            get_browser_alert_stream()
        ),
        mimetype="text/event-stream"
    )

# ─────────────────────────────────────────────────────────────────────────────
# REMINDER STREAM
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/reminders/stream")
def reminder_stream():

    return Response(
        stream_with_context(
            get_reminder_stream()
        ),
        mimetype="text/event-stream"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict_full", methods=["POST"])
def predict_full():

    predictor_model = get_predictor()

    if predictor_model is None:
        return jsonify({
            "error": "No model found"
        }), 500

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    ext = os.path.splitext(file.filename)[1]

    if ext == "":
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(image_path)

    try:

        patient_id = request.form.get(
            "patient_id",
            "P001"
        )

        # ─────────────────────────────────────────
        # PREDICTION
        # ─────────────────────────────────────────

        prediction = predictor_model.predict(
            image_path
        )

        # ─────────────────────────────────────────
        # HEATMAP
        # ─────────────────────────────────────────

        try:

            heatmap_result = generate_risk_heatmap(
                image_path=image_path,
                abcd_features=prediction.get(
                    "abcd_features",
                    {}
                ),
                predicted_class=prediction.get(
                    "predicted_class",
                    ""
                )
            )

        except Exception as e:

            heatmap_result = {
                "error": str(e)
            }

        # ─────────────────────────────────────────
        # GRADCAM
        # ─────────────────────────────────────────

        gradcam_result = {
            "error": "CNN model not loaded"
        }

        if cnn_model is not None:

            try:

                from gradcam import generate_gradcam

                gradcam_result = generate_gradcam(
                    image_path,
                    cnn_model
                )

            except Exception as e:

                gradcam_result = {
                    "error": str(e)
                }

        # ─────────────────────────────────────────
        # SEASONAL RISK
        # ─────────────────────────────────────────

        seasonal = compute_seasonal_risk(
            month=datetime.now().month,
            hemisphere="northern",
            skin_type="type_2",
            localization="face"
        )

        # ─────────────────────────────────────────
        # SAVE VISIT
        # ─────────────────────────────────────────

        save_visit(
            patient_id=patient_id,
            features=prediction.get(
                "abcd_features",
                {}
            ),
            prediction=prediction
        )

        timeline = analyze_progression(
            patient_id
        )

        # ─────────────────────────────────────────
        # ALERTS
        # ─────────────────────────────────────────

        alert_result = dispatch_alerts(
            patient_id=patient_id,
            prediction=prediction,
            risk_score=heatmap_result.get(
                "risk_score",
                0.0
            ),
            seasonal=seasonal
        )

        # ─────────────────────────────────────────
        # MONITORING
        # ─────────────────────────────────────────

        monitoring_report = monitor.analyze(
            patient_id
        )

        # ─────────────────────────────────────────
        # REMINDERS
        # ─────────────────────────────────────────

        scheduled_reminders = (
            scheduler.schedule_from_prediction(
                patient_id=patient_id,
                prediction=prediction
            )
        )

        # ─────────────────────────────────────────
        # RESPONSE
        # ─────────────────────────────────────────

        return jsonify({

            "prediction": prediction,

            "heatmap": heatmap_result,

            "gradcam": gradcam_result,

            "seasonal": seasonal,

            "timeline": timeline,

            "alert": alert_result,

            "monitoring": monitoring_report,

            "reminders": {
                "scheduled": scheduled_reminders
            }

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(image_path):
            os.remove(image_path)

# ─────────────────────────────────────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\nStarting Skin Cancer AI Web App...")
    print("Open browser:")
    print("http://127.0.0.1:5000\n")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )