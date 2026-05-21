"""
Web App — Skin Cancer Classifier
==================================
Upload a lesion image + enter patient metadata → get a 7-class prediction
with ABCD feature scores and risk level.

Run:
    pip install flask
    python webapp/app.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import uuid
import base64
from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "skin_cancer_model.pkl")

# Lazy-load the model
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        if not os.path.exists(MODEL_PATH):
            return None
        from predict import SkinCancerPredictor
        predictor = SkinCancerPredictor(MODEL_PATH)
    return predictor


# ──────────────────────────────────────────────
# HTML Template (single-file app)
# ──────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Skin Cancer Classifier — ABCD Analysis</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f4f3ee; color: #1c1c1a; min-height: 100vh; }

    header { background: #2d2b55; color: #fff; padding: 20px 40px; }
    header h1 { font-size: 22px; font-weight: 600; }
    header p  { font-size: 13px; opacity: 0.7; margin-top: 4px; }

    .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }

    .card { background: #fff; border-radius: 12px; padding: 28px; box-shadow: 0 2px 8px rgba(0,0,0,.07); margin-bottom: 24px; }
    .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 18px; color: #2d2b55; }

    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    label { font-size: 13px; color: #555; display: block; margin-bottom: 5px; }
    input, select { width: 100%; padding: 9px 12px; border: 1px solid #d8d7d0; border-radius: 8px; font-size: 14px; background: #fafaf8; }

    .upload-zone { border: 2px dashed #c5c3bc; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: border-color .2s; }
    .upload-zone:hover { border-color: #534AB7; }
    .upload-zone p { color: #888; font-size: 14px; margin-top: 8px; }
    #preview { max-width: 260px; max-height: 260px; border-radius: 8px; margin-top: 12px; display: none; }

    .btn { background: #534AB7; color: #fff; border: none; padding: 12px 28px; border-radius: 8px; font-size: 15px; cursor: pointer; width: 100%; font-weight: 500; transition: background .2s; }
    .btn:hover { background: #3C3489; }
    .btn:disabled { background: #aaa; cursor: not-allowed; }

    #result-card { display: none; }
    .risk-high { color: #a32d2d; background: #fcebeb; border: 1px solid #f09595; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 500; }
    .risk-low  { color: #0f6e56; background: #e1f5ee; border: 1px solid #5dcaa5; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 500; }

    .pred-label { font-size: 26px; font-weight: 700; color: #2d2b55; }
    .pred-sub   { font-size: 14px; color: #777; margin-top: 2px; margin-bottom: 18px; }

    .prob-bar { margin-bottom: 10px; }
    .prob-bar label { font-size: 13px; display: flex; justify-content: space-between; margin-bottom: 3px; }
    .bar-track { height: 10px; background: #eee; border-radius: 5px; overflow: hidden; }
    .bar-fill  { height: 100%; border-radius: 5px; background: #534AB7; transition: width .5s; }
    .bar-fill.high { background: #a32d2d; }

    .abcd-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .abcd-item { background: #f8f7f2; border-radius: 8px; padding: 14px; text-align: center; }
    .abcd-letter { font-size: 22px; font-weight: 700; color: #534AB7; }
    .abcd-name  { font-size: 11px; color: #888; margin-top: 2px; }
    .abcd-score { font-size: 18px; font-weight: 600; color: #1c1c1a; margin-top: 6px; }

    #error-msg { color: #a32d2d; font-size: 14px; margin-top: 8px; display: none; }
    .spinner { display: none; text-align: center; padding: 20px; color: #534AB7; }

    footer { text-align: center; font-size: 12px; color: #aaa; padding: 30px; }
    footer strong { color: #888; }

    @media (max-width: 600px) {
      .form-grid { grid-template-columns: 1fr; }
      .abcd-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Skin Cancer ABCD Classifier</h1>
    <p>HAM10000 · 7-class diagnosis · Asymmetry · Border · Color · Diameter</p>
  </header>

  <div class="container">
    <!-- Upload card -->
    <div class="card">
      <h2>Upload lesion image</h2>
      <div class="upload-zone" onclick="document.getElementById('file-input').click()">
        <svg width="40" height="40" fill="none" stroke="#aaa" stroke-width="1.5" viewBox="0 0 24 24">
          <path d="M12 16V8m0 0-3 3m3-3 3 3M6 20h12a2 2 0 0 0 2-2V8l-5-5H6a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/>
        </svg>
        <p>Click to upload or drag & drop</p>
        <p style="font-size:12px;color:#bbb">JPG, PNG — max 16 MB</p>
        <img id="preview"/>
      </div>
      <input type="file" id="file-input" accept="image/*" style="display:none" onchange="previewImage(this)">
    </div>

    <!-- Patient metadata card -->
    <div class="card">
      <h2>Patient information (optional — improves accuracy)</h2>
      <div class="form-grid">
        <div>
          <label>Age</label>
          <input type="number" id="age" min="1" max="100" placeholder="e.g. 45">
        </div>
        <div>
          <label>Sex</label>
          <select id="sex">
            <option value="">— select —</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
        <div>
          <label>Lesion location</label>
          <select id="localization">
            <option value="">— select —</option>
            <option>back</option><option>face</option><option>scalp</option>
            <option>chest</option><option>abdomen</option><option>trunk</option>
            <option>lower extremity</option><option>upper extremity</option>
            <option>hand</option><option>foot</option><option>ear</option>
            <option>neck</option><option>acral</option><option>genital</option>
          </select>
        </div>
        <div>
          <label>Diagnosis method</label>
          <select id="dx_type">
            <option value="">— select —</option>
            <option value="histo">Histopathology</option>
            <option value="follow_up">Follow-up</option>
            <option value="consensus">Expert consensus</option>
            <option value="confocal">Confocal microscopy</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Analyze button -->
    <button class="btn" id="analyze-btn" onclick="analyze()">Analyze lesion</button>
    <div id="error-msg"></div>
    <div class="spinner" id="spinner">Extracting ABCD features &amp; classifying…</div>

    <!-- Results -->
    <div class="card" id="result-card" style="margin-top:24px">
      <div id="risk-badge"></div>
      <div class="pred-label" id="pred-label"></div>
      <div class="pred-sub" id="pred-sub"></div>

      <!-- ABCD scores -->
      <h2 style="margin-bottom:14px">ABCD feature scores</h2>
      <div class="abcd-grid" id="abcd-grid"></div>

      <!-- Probability bars -->
      <h2 style="margin-top:22px;margin-bottom:14px">All class probabilities</h2>
      <div id="prob-bars"></div>
    </div>

    <footer>
      <strong>Disclaimer:</strong> This tool is for educational purposes only. It is not a medical device and should not be used for clinical diagnosis. Always consult a certified dermatologist.
    </footer>
  </div>

  <script>
    function previewImage(input) {
      const file = input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.getElementById("preview");
        img.src = e.target.result;
        img.style.display = "block";
      };
      reader.readAsDataURL(file);
    }

    async function analyze() {
      const fileInput = document.getElementById("file-input");
      if (!fileInput.files.length) {
        showError("Please upload an image first.");
        return;
      }

      document.getElementById("analyze-btn").disabled = true;
      document.getElementById("spinner").style.display = "block";
      document.getElementById("result-card").style.display = "none";
      document.getElementById("error-msg").style.display = "none";

      const formData = new FormData();
      formData.append("image", fileInput.files[0]);
      formData.append("age", document.getElementById("age").value || "");
      formData.append("sex", document.getElementById("sex").value || "");
      formData.append("localization", document.getElementById("localization").value || "");
      formData.append("dx_type", document.getElementById("dx_type").value || "");

      try {
        const resp = await fetch("/predict", { method: "POST", body: formData });
        const data = await resp.json();
        if (data.error) { showError(data.error); return; }
        renderResult(data);
      } catch (e) {
        showError("Server error: " + e.message);
      } finally {
        document.getElementById("analyze-btn").disabled = false;
        document.getElementById("spinner").style.display = "none";
      }
    }

    function renderResult(data) {
      document.getElementById("result-card").style.display = "block";

      // Risk badge
      const isHigh = data.risk_level.includes("HIGH");
      document.getElementById("risk-badge").innerHTML =
        `<div class="${isHigh ? 'risk-high' : 'risk-low'}">
          ${isHigh ? "⚠ HIGH RISK — Potentially malignant" : "✓ LOW RISK — Likely benign"}
        </div>`;

      document.getElementById("pred-label").textContent = data.predicted_label;
      document.getElementById("pred-sub").textContent =
        `Code: ${data.predicted_class}  ·  Confidence: ${(data.confidence * 100).toFixed(1)}%`;

      // ABCD grid
      const abcd = data.abcd_features;
      const abcdItems = [
        { letter: "A", name: "Asymmetry", value: abcd.asymmetry_score ?? "—" },
        { letter: "B", name: "Border", value: abcd.border_irregularity ?? "—" },
        { letter: "C", name: "Color entropy", value: abcd.color_entropy ?? "—" },
        { letter: "D", name: "Diameter (mm)", value: abcd.diameter_max_mm ?? "—" },
      ];
      document.getElementById("abcd-grid").innerHTML = abcdItems.map(item =>
        `<div class="abcd-item">
          <div class="abcd-letter">${item.letter}</div>
          <div class="abcd-name">${item.name}</div>
          <div class="abcd-score">${typeof item.value === "number" ? item.value.toFixed(3) : item.value}</div>
        </div>`
      ).join("");

      // Probability bars
      const probs = Object.entries(data.all_probabilities)
        .sort((a, b) => b[1] - a[1]);
      const highClasses = ["Melanoma", "Basal cell carcinoma", "Actinic keratosis"];
      document.getElementById("prob-bars").innerHTML = probs.map(([label, prob]) => {
        const isHigh = highClasses.includes(label);
        return `<div class="prob-bar">
          <label><span>${label}</span><span>${(prob * 100).toFixed(1)}%</span></label>
          <div class="bar-track">
            <div class="bar-fill ${isHigh ? 'high' : ''}" style="width:${(prob * 100).toFixed(1)}%"></div>
          </div>
        </div>`;
      }).join("");

      document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
    }

    function showError(msg) {
      const el = document.getElementById("error-msg");
      el.textContent = "Error: " + msg;
      el.style.display = "block";
      document.getElementById("spinner").style.display = "none";
      document.getElementById("analyze-btn").disabled = false;
    }
  </script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/predict", methods=["POST"])
def predict():
    pred = get_predictor()
    if pred is None:
        return jsonify({"error": "Model not found. Train the model first using train_model.py"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    ext = os.path.splitext(file.filename)[-1].lower() or ".jpg"
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)

    try:
        kwargs = {}
        age = request.form.get("age")
        if age:
            kwargs["age"] = float(age)
        sex = request.form.get("sex")
        if sex:
            kwargs["sex"] = sex
        localization = request.form.get("localization")
        if localization:
            kwargs["localization"] = localization
        dx_type = request.form.get("dx_type")
        if dx_type:
            kwargs["dx_type"] = dx_type

        result = pred.predict(temp_path, **kwargs)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("Starting Skin Cancer Classifier Web App...")
    print("Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
