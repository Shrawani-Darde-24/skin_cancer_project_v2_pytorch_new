"""
Enhanced Skin Cancer Classifier Web App
=========================================
Features:
  1. ABCD feature extraction + 7-class prediction
  2. Grad-CAM lesion localization (exact cancerous region)
  3. Seasonal UV risk analysis
  4. Time series progression tracking across visits
"""

import os, sys, json, uuid, pickle
from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_V2  = os.path.join(os.path.dirname(__file__), "..", "models", "skin_cancer_model_v2.pkl")
MODEL_V1  = os.path.join(os.path.dirname(__file__), "..", "models", "skin_cancer_model.pkl")

predictor   = None
cnn_model   = None
model_ver   = None


def get_predictor():
    global predictor, cnn_model, model_ver
    if predictor is not None:
        return predictor

    if os.path.exists(MODEL_V2):
        from predict_v2 import SkinCancerPredictorV2
        predictor  = SkinCancerPredictorV2(MODEL_V2)
        cnn_model  = predictor.cnn_model
        model_ver  = "v2"
    elif os.path.exists(MODEL_V1):
        from predict import SkinCancerPredictor
        predictor  = SkinCancerPredictor(MODEL_V1)
        model_ver  = "v1"
    return predictor


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Skin Cancer AI — Full Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,-apple-system,sans-serif;background:#f4f3ee;color:#1c1c1a;min-height:100vh}
    header{background:#2d2b55;color:#fff;padding:18px 36px;display:flex;align-items:center;gap:16px}
    header h1{font-size:20px;font-weight:600}
    header p{font-size:12px;opacity:.65;margin-top:3px}
    .badge{background:#534AB7;color:#fff;font-size:11px;padding:3px 8px;border-radius:20px}
    .container{max-width:1100px;margin:32px auto;padding:0 20px;display:grid;grid-template-columns:1fr 1fr;gap:22px}
    .full{grid-column:1/-1}
    .card{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
    .card h2{font-size:15px;font-weight:600;color:#2d2b55;margin-bottom:16px;display:flex;align-items:center;gap:8px}
    .card h2 .icon{width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px}
    label{font-size:12px;color:#666;display:block;margin-bottom:4px}
    input,select{width:100%;padding:8px 11px;border:1px solid #d8d7d0;border-radius:7px;font-size:13px;background:#fafaf8}
    .form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
    .upload-zone{border:2px dashed #c5c3bc;border-radius:10px;padding:32px;text-align:center;cursor:pointer;transition:border-color .2s}
    .upload-zone:hover{border-color:#534AB7}
    .upload-zone p{color:#999;font-size:13px;margin-top:6px}
    #preview{max-width:220px;max-height:220px;border-radius:8px;margin-top:10px;display:none}
    .btn{background:#534AB7;color:#fff;border:none;padding:11px 24px;border-radius:8px;font-size:14px;cursor:pointer;width:100%;font-weight:500;transition:background .2s;margin-top:4px}
    .btn:hover{background:#3C3489}
    .btn:disabled{background:#aaa;cursor:not-allowed}
    .btn-sm{width:auto;padding:7px 16px;font-size:13px}
    #result-area{display:none}
    .risk-high{color:#a32d2d;background:#fcebeb;border:1px solid #f09595;padding:11px 15px;border-radius:8px;margin-bottom:14px;font-weight:500;font-size:14px}
    .risk-low{color:#0f6e56;background:#e1f5ee;border:1px solid #5dcaa5;padding:11px 15px;border-radius:8px;margin-bottom:14px;font-weight:500;font-size:14px}
    .pred-label{font-size:24px;font-weight:700;color:#2d2b55}
    .pred-sub{font-size:13px;color:#888;margin-top:2px;margin-bottom:16px}
    .abcd-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}
    .abcd-item{background:#f8f7f2;border-radius:8px;padding:12px;text-align:center}
    .abcd-letter{font-size:20px;font-weight:700;color:#534AB7}
    .abcd-name{font-size:10px;color:#999;margin-top:1px}
    .abcd-score{font-size:16px;font-weight:600;margin-top:4px}
    .prob-bar{margin-bottom:9px}
    .prob-bar label{font-size:12px;display:flex;justify-content:space-between;margin-bottom:3px}
    .bar-track{height:9px;background:#eee;border-radius:5px;overflow:hidden}
    .bar-fill{height:100%;border-radius:5px;background:#534AB7;transition:width .5s}
    .bar-fill.mal{background:#a32d2d}
    /* Localization */
    .cam-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
    .cam-grid img{width:100%;border-radius:8px;border:1px solid #e8e6e0}
    .cam-grid p{font-size:11px;color:#888;text-align:center;margin-top:4px}
    /* Seasonal */
    .season-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;margin-bottom:12px}
    .season-danger{background:#fcebeb;color:#a32d2d}
    .season-warning{background:#faeeda;color:#854f0b}
    .season-caution{background:#fff8e6;color:#7a5c00}
    .season-safe{background:#e1f5ee;color:#0f6e56}
    .rec-list{list-style:none;padding:0}
    .rec-list li{font-size:12px;color:#555;padding:5px 0;border-bottom:1px solid #f0eeea;display:flex;gap:8px}
    .rec-list li::before{content:"→";color:#534AB7;flex-shrink:0}
    /* Time series */
    .ts-alert{background:#fcebeb;border-left:3px solid #a32d2d;padding:8px 12px;margin-bottom:8px;border-radius:0 6px 6px 0;font-size:13px;color:#a32d2d}
    .visit-item{padding:10px;border:1px solid #eee;border-radius:8px;margin-bottom:8px;font-size:13px}
    .visit-date{font-weight:600;color:#2d2b55;margin-bottom:3px}
    .tab-bar{display:flex;gap:2px;margin-bottom:20px;background:#f0eeea;padding:4px;border-radius:9px}
    .tab{flex:1;padding:8px;border:none;background:none;border-radius:7px;font-size:13px;cursor:pointer;color:#666;transition:all .15s}
    .tab.active{background:#fff;color:#2d2b55;font-weight:500;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .tab-panel{display:none}.tab-panel.active{display:block}
    .spinner{display:none;text-align:center;padding:18px;color:#534AB7;font-size:14px}
    .err{color:#a32d2d;font-size:13px;margin-top:6px;display:none}
    footer{text-align:center;font-size:11px;color:#bbb;padding:28px}
    @media(max-width:700px){.container{grid-template-columns:1fr}.abcd-grid{grid-template-columns:repeat(2,1fr)}.cam-grid{grid-template-columns:1fr 1fr}}
  </style>
</head>
<body>
<header>
  <div>
    <h1>Skin Cancer AI — Full Diagnostic Suite</h1>
    <p>ABCD Analysis · Grad-CAM Localization · Seasonal Risk · Progression Tracking</p>
  </div>
  <span class="badge" id="model-badge">Loading model...</span>
</header>

<div class="container">

  <!-- LEFT: Upload + Patient Info -->
  <div>
    <div class="card">
      <h2><span class="icon" style="background:#eeedfe">🔬</span>Upload lesion image</h2>
      <div class="upload-zone" onclick="document.getElementById('file-input').click()">
        <svg width="36" height="36" fill="none" stroke="#aaa" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 16V8m0 0-3 3m3-3 3 3M6 20h12a2 2 0 0 0 2-2V8l-5-5H6a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/></svg>
        <p>Click to upload or drag & drop</p>
        <p style="font-size:11px;color:#ccc">JPG, PNG — max 16MB</p>
        <img id="preview"/>
      </div>
      <input type="file" id="file-input" accept="image/*" style="display:none" onchange="previewImg(this)">
    </div>

    <div class="card" style="margin-top:16px">
      <h2><span class="icon" style="background:#e1f5ee">👤</span>Patient information</h2>
      <div class="form-row">
        <div><label>Patient ID (for tracking)</label><input id="patient-id" placeholder="e.g. P001" value="P001"></div>
        <div><label>Age</label><input type="number" id="age" min="1" max="110" placeholder="45"></div>
      </div>
      <div class="form-row">
        <div><label>Sex</label>
          <select id="sex"><option value="">— select —</option><option value="male">Male</option><option value="female">Female</option><option value="unknown">Unknown</option></select>
        </div>
        <div><label>Skin type (Fitzpatrick)</label>
          <select id="skin-type">
            <option value="type_2">Type II — Burns easily</option>
            <option value="type_1">Type I — Always burns</option>
            <option value="type_3">Type III — Sometimes burns</option>
            <option value="type_4">Type IV — Rarely burns</option>
            <option value="type_5">Type V — Very rarely burns</option>
            <option value="type_6">Type VI — Never burns</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div><label>Lesion location</label>
          <select id="localization">
            <option value="">— select —</option>
            <option>back</option><option>face</option><option>scalp</option>
            <option>chest</option><option>abdomen</option><option>trunk</option>
            <option>lower extremity</option><option>upper extremity</option>
            <option>hand</option><option>foot</option><option>ear</option>
            <option>neck</option><option>acral</option><option>genital</option>
          </select>
        </div>
        <div><label>Hemisphere</label>
          <select id="hemisphere"><option value="northern">Northern</option><option value="southern">Southern</option></select>
        </div>
      </div>
      <div class="form-row">
        <div><label>Diagnosis method</label>
          <select id="dx-type">
            <option value="">— select —</option>
            <option value="histo">Histopathology</option>
            <option value="follow_up">Follow-up</option>
            <option value="consensus">Expert consensus</option>
            <option value="confocal">Confocal</option>
          </select>
        </div>
        <div><label>Visit date</label><input type="date" id="visit-date"></div>
      </div>

      <button class="btn" id="analyze-btn" onclick="analyze()">Analyze lesion</button>
      <div class="err" id="err-msg"></div>
      <div class="spinner" id="spinner">Extracting ABCD features, running model & Grad-CAM...</div>
    </div>
  </div>

  <!-- RIGHT: Results -->
  <div id="result-area">
    <!-- Tabs -->
    <div class="tab-bar">
      <button class="tab active" onclick="switchTab('diagnosis')">Diagnosis</button>
      <button class="tab" onclick="switchTab('localization')">Localization</button>
      <button class="tab" onclick="switchTab('seasonal')">Seasonal Risk</button>
      <button class="tab" onclick="switchTab('timeline')">Progression</button>
    </div>

    <!-- TAB 1: Diagnosis -->
    <div class="tab-panel active" id="tab-diagnosis">
      <div class="card">
        <div id="risk-badge"></div>
        <div class="pred-label" id="pred-label"></div>
        <div class="pred-sub" id="pred-sub"></div>
        <div class="abcd-grid" id="abcd-grid"></div>
        <h2 style="margin-bottom:12px">Class probabilities</h2>
        <div id="prob-bars"></div>
      </div>
    </div>

    <!-- TAB 2: Grad-CAM Localization -->
    <div class="tab-panel" id="tab-localization">
      <div class="card">
        <h2><span class="icon" style="background:#faece7">🎯</span>Lesion localization — Grad-CAM</h2>
        <p style="font-size:12px;color:#888;margin-bottom:14px">The heatmap shows where the model focused. The green box marks the high-attention region (likely cancerous area).</p>
        <div class="cam-grid" id="cam-grid">
          <div><p style="text-align:center;color:#ccc;padding:40px 0">Run analysis first</p></div>
        </div>
      </div>
    </div>

    <!-- TAB 3: Seasonal Risk -->
    <div class="tab-panel" id="tab-seasonal">
      <div class="card">
        <h2><span class="icon" style="background:#faeeda">☀️</span>Seasonal UV risk</h2>
        <div id="season-content">
          <canvas id="uv-chart" height="200"></canvas>
          <div id="season-details" style="margin-top:16px"></div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Time Series -->
    <div class="tab-panel" id="tab-timeline">
      <div class="card">
        <h2><span class="icon" style="background:#e6f1fb">📈</span>Lesion progression</h2>
        <div id="ts-alerts"></div>
        <canvas id="ts-chart" height="220"></canvas>
        <div id="visit-history" style="margin-top:16px"></div>
      </div>
    </div>
  </div>

  <div class="full" style="margin-top:0">
    <footer>⚠ For educational purposes only. Not a medical device. Always consult a certified dermatologist.</footer>
  </div>
</div>

<script>
let uvChart = null, tsChart = null;

// ── Init ────────────────────────────────────────────────────────
fetch('/model_info').then(r=>r.json()).then(d=>{
  document.getElementById('model-badge').textContent = d.version === 'v2' ? 'CNN + ABCD Model' : 'ABCD Model';
});
document.getElementById('visit-date').value = new Date().toISOString().split('T')[0];

// ── UI helpers ──────────────────────────────────────────────────
function previewImg(input) {
  const f = input.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = e => { const i = document.getElementById('preview'); i.src = e.target.result; i.style.display='block'; };
  r.readAsDataURL(f);
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const names = ['diagnosis','localization','seasonal','timeline'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'tab-'+name);
  });
}

function showErr(msg) {
  const e = document.getElementById('err-msg');
  e.textContent = 'Error: ' + msg; e.style.display = 'block';
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('analyze-btn').disabled = false;
}

// ── Analyze ─────────────────────────────────────────────────────
async function analyze() {
  const fi = document.getElementById('file-input');
  if (!fi.files.length) { showErr('Please upload an image first.'); return; }

  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('err-msg').style.display = 'none';
  document.getElementById('result-area').style.display = 'none';

  const fd = new FormData();
  fd.append('image',       fi.files[0]);
  fd.append('patient_id',  document.getElementById('patient-id').value  || 'P001');
  fd.append('age',         document.getElementById('age').value         || '');
  fd.append('sex',         document.getElementById('sex').value         || '');
  fd.append('localization',document.getElementById('localization').value || '');
  fd.append('dx_type',     document.getElementById('dx-type').value     || '');
  fd.append('skin_type',   document.getElementById('skin-type').value   || 'type_2');
  fd.append('hemisphere',  document.getElementById('hemisphere').value  || 'northern');
  fd.append('visit_date',  document.getElementById('visit-date').value  || '');

  try {
    const r = await fetch('/predict_full', {method:'POST', body:fd});
    const d = await r.json();
    if (d.error) { showErr(d.error); return; }
    renderAll(d);
    document.getElementById('result-area').style.display = 'block';
    switchTab('diagnosis');
  } catch(e) { showErr(e.message); }
  finally {
    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('spinner').style.display = 'none';
  }
}

// ── Render all panels ───────────────────────────────────────────
function renderAll(d) {
  renderDiagnosis(d.prediction);
  renderLocalization(d.gradcam);
  renderSeasonal(d.seasonal);
  renderTimeline(d.timeline);
}

function renderDiagnosis(p) {
  const isHigh = p.risk_level.includes('HIGH');
  document.getElementById('risk-badge').innerHTML =
    `<div class="${isHigh?'risk-high':'risk-low'}">${isHigh?'⚠ HIGH RISK — Potentially malignant':'✓ LOW RISK — Likely benign'}</div>`;
  document.getElementById('pred-label').textContent = p.predicted_label;
  document.getElementById('pred-sub').textContent =
    `Code: ${p.predicted_class} · Confidence: ${(p.confidence*100).toFixed(1)}%`;

  const abcd = p.abcd_features;
  const items = [
    {l:'A', n:'Asymmetry',     v: abcd.asymmetry_score},
    {l:'B', n:'Border',        v: abcd.border_irregularity},
    {l:'C', n:'Color entropy', v: abcd.color_entropy},
    {l:'D', n:'Diameter (mm)', v: abcd.diameter_max_mm},
  ];
  document.getElementById('abcd-grid').innerHTML = items.map(i =>
    `<div class="abcd-item">
      <div class="abcd-letter">${i.l}</div>
      <div class="abcd-name">${i.n}</div>
      <div class="abcd-score">${typeof i.v==='number'?i.v.toFixed(3):'—'}</div>
    </div>`).join('');

  const mal = ['Melanoma','Basal cell carcinoma','Actinic keratosis'];
  const probs = Object.entries(p.all_probabilities).sort((a,b)=>b[1]-a[1]);
  document.getElementById('prob-bars').innerHTML = probs.map(([lbl,prob])=>
    `<div class="prob-bar">
      <label><span>${lbl}</span><span>${(prob*100).toFixed(1)}%</span></label>
      <div class="bar-track"><div class="bar-fill ${mal.includes(lbl)?'mal':''}" style="width:${(prob*100).toFixed(1)}%"></div></div>
    </div>`).join('');
}

function renderLocalization(gc) {
  if (!gc || gc.error) {
    document.getElementById('cam-grid').innerHTML = '<p style="color:#ccc;padding:30px;text-align:center">Grad-CAM not available (requires CNN model)</p>';
    return;
  }
  document.getElementById('cam-grid').innerHTML =
    `<div><img src="${gc.original}"><p>Original</p></div>
     <div><img src="${gc.heatmap}"><p>Attention heatmap</p></div>
     <div><img src="${gc.overlay}"><p>Localized region</p></div>`;
}

function renderSeasonal(s) {
  if (!s) return;
  const colorMap = {danger:'season-danger',warning:'season-warning',caution:'season-caution',safe:'season-safe'};
  const cls = colorMap[s.risk_color] || 'season-safe';
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const uvVals = Object.values(s.monthly_uv_trend);

  document.getElementById('season-details').innerHTML =
    `<div class="season-badge ${cls}">☀ ${s.season} UV Risk: ${s.risk_level} (score ${s.risk_score})</div>
     <ul class="rec-list">${s.recommendations.map(r=>`<li>${r}</li>`).join('')}</ul>`;

  if (uvChart) uvChart.destroy();
  uvChart = new Chart(document.getElementById('uv-chart'), {
    type: 'bar',
    data: {
      labels: months,
      datasets: [{
        label: 'UV Index',
        data: uvVals,
        backgroundColor: uvVals.map(v =>
          v>=9?'rgba(163,45,45,0.7)': v>=6?'rgba(186,117,23,0.7)': v>=3?'rgba(133,79,11,0.5)':'rgba(15,110,86,0.5)'),
        borderRadius: 4,
      }]
    },
    options: {
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>`UV Index: ${c.raw}`}} },
      scales:{ y:{beginAtZero:true, max:12, title:{display:true,text:'UV Index'}} }
    }
  });

  // Highlight current month
  const cur = new Date().getMonth();
  uvChart.data.datasets[0].borderColor = uvVals.map((_,i)=> i===cur?'#534AB7':'transparent');
  uvChart.data.datasets[0].borderWidth = uvVals.map((_,i)=> i===cur?2:0);
  uvChart.update();
}

function renderTimeline(ts) {
  if (!ts) return;

  // Alerts
  document.getElementById('ts-alerts').innerHTML = ts.alerts && ts.alerts.length
    ? ts.alerts.map(a=>`<div class="ts-alert">${a}</div>`).join('')
    : '<p style="font-size:12px;color:#888;margin-bottom:12px">No concerning changes detected.</p>';

  // Chart
  if (tsChart) tsChart.destroy();
  const trend = ts.trends || {};
  const feat  = Object.keys(trend)[0];
  if (feat && trend[feat].values.length > 1) {
    tsChart = new Chart(document.getElementById('ts-chart'), {
      type: 'line',
      data: {
        labels: trend[feat].dates,
        datasets: Object.entries(trend).map(([k, v], i) => ({
          label: k.replace(/_/g,' '),
          data: v.values,
          borderColor: ['#534AB7','#1D9E75','#D85A30','#D4537E','#BA7517'][i%5],
          tension: 0.3, fill: false, pointRadius: 5,
        }))
      },
      options:{ plugins:{legend:{position:'bottom'}}, scales:{y:{beginAtZero:true}} }
    });
  } else {
    document.getElementById('ts-chart').style.display = 'none';
  }

  // Visit history
  const visits = ts.history || [];
  document.getElementById('visit-history').innerHTML = visits.length
    ? '<h2 style="font-size:14px;font-weight:600;color:#2d2b55;margin-bottom:10px">Visit history</h2>' +
      [...visits].reverse().map(v=>
        `<div class="visit-item">
          <div class="visit-date">${v.date}</div>
          <div style="color:#555">${v.prediction?.label || '—'} &nbsp;·&nbsp; Confidence: ${((v.prediction?.confidence||0)*100).toFixed(1)}%</div>
          <div style="font-size:11px;color:#aaa">${v.prediction?.risk_level || ''}</div>
        </div>`).join('')
    : '<p style="font-size:12px;color:#aaa">No visits recorded yet.</p>';
}
</script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/model_info")
def model_info():
    get_predictor()
    return jsonify({"version": model_ver or "none"})


@app.route("/predict_full", methods=["POST"])
def predict_full():
    from seasonal_risk import compute_seasonal_risk
    from time_series   import save_visit, analyze_progression

    pred = get_predictor()
    if pred is None:
        return jsonify({"error": "Model not found. Run train_model.py first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    ext  = os.path.splitext(file.filename)[-1].lower() or ".jpg"
    temp = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    file.save(temp)

    try:
        # ── 1. Prediction ─────────────────────────────────────────
        kwargs = {}
        for field in ["age", "sex", "localization", "dx_type"]:
            val = request.form.get(field)
            if val:
                kwargs[field] = float(val) if field == "age" else val

        prediction = pred.predict(temp, **kwargs)

        # ── 2. Grad-CAM ───────────────────────────────────────────
        gradcam_result = {"error": "CNN model not loaded"}
        if cnn_model is not None:
            try:
                from gradcam import generate_gradcam
                gradcam_result = generate_gradcam(temp, cnn_model)
            except Exception as e:
                gradcam_result = {"error": str(e)}

        # ── 3. Seasonal risk ──────────────────────────────────────
        from datetime import datetime
        visit_date_str = request.form.get("visit_date", "")
        try:
            month = datetime.strptime(visit_date_str, "%Y-%m-%d").month if visit_date_str else datetime.now().month
        except:
            month = datetime.now().month

        seasonal = compute_seasonal_risk(
            month       = month,
            hemisphere  = request.form.get("hemisphere", "northern"),
            skin_type   = request.form.get("skin_type", "type_2"),
            localization= request.form.get("localization", "face"),
        )

        # ── 4. Save visit + time series ───────────────────────────
        patient_id = request.form.get("patient_id", "P001")
        save_visit(
            patient_id   = patient_id,
            features     = prediction.get("abcd_features", {}),
            prediction   = prediction,
            visit_date   = visit_date_str or None,
        )
        timeline = analyze_progression(patient_id)

        return jsonify({
            "prediction": prediction,
            "gradcam":    gradcam_result,
            "seasonal":   seasonal,
            "timeline":   timeline,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp):
            os.remove(temp)


if __name__ == "__main__":
    print("Starting Enhanced Skin Cancer Classifier...")
    print("Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
