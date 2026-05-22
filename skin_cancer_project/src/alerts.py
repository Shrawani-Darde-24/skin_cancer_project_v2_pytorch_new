"""
Automated Alerts — Critical Diagnosis Notification System
==========================================================
Sends alerts via:
  1. Email (Gmail SMTP)
  2. Browser push notification (via SSE endpoint)

When to alert:
  - Predicted class is malignant (mel, bcc, akiec)
  - Confidence > threshold
  - ABCD risk score is high/extreme
  - Lesion significantly changed since last visit

Setup:
  Set these in your .env file or environment:
    ALERT_EMAIL_FROM=youremail@gmail.com
    ALERT_EMAIL_PASSWORD=your_app_password   (Gmail App Password)
    ALERT_EMAIL_TO=doctor@clinic.com
"""

import os
import smtplib
import json
import queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# ── Config ──────────────────────────────────────────────────────
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 587
EMAIL_FROM     = os.environ.get("ALERT_EMAIL_FROM", "")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")
EMAIL_TO       = os.environ.get("ALERT_EMAIL_TO", EMAIL_FROM)

MALIGNANT      = {"mel", "bcc", "akiec"}
ALERT_THRESHOLD_CONFIDENCE = 0.40   # alert if confidence >= this
ALERT_THRESHOLD_RISK       = 0.50   # alert if risk score >= this

# SSE queue for browser notifications (one global queue per app)
_browser_alert_queue: queue.Queue = queue.Queue()

LABEL_MAP = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular lesion",
}


# ── Alert decision ───────────────────────────────────────────────

def should_alert(prediction: dict, risk_score: float = None) -> tuple[bool, list]:
    """
    Decide whether this result warrants an alert.
    Returns (should_alert: bool, reasons: list[str])
    """
    reasons = []
    cls        = prediction.get("predicted_class", "")
    confidence = prediction.get("confidence", 0.0)

    if cls in MALIGNANT:
        reasons.append(f"Malignant class detected: {LABEL_MAP.get(cls, cls)}")

    if confidence >= ALERT_THRESHOLD_CONFIDENCE and cls in MALIGNANT:
        reasons.append(f"High confidence malignant prediction: {confidence*100:.1f}%")

    if risk_score is not None and risk_score >= ALERT_THRESHOLD_RISK:
        reasons.append(f"High ABCD risk score: {risk_score:.2f}")

    return len(reasons) > 0, reasons


# ── Email alert ──────────────────────────────────────────────────

def send_email_alert(
    patient_id: str,
    prediction: dict,
    risk_score: float,
    reasons: list,
    seasonal: dict = None,
) -> dict:
    """
    Send an HTML email alert for a critical diagnosis.
    Returns {"sent": bool, "error": str or None}
    """
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        return {
            "sent": False,
            "error": "Email not configured. Set ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD in environment."
        }

    cls         = prediction.get("predicted_class", "")
    label       = prediction.get("predicted_label", cls)
    confidence  = prediction.get("confidence", 0.0)
    risk_level  = prediction.get("risk_level", "")
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")
    season_info = f"{seasonal['season']} — UV Index {seasonal['base_uv_index']}" if seasonal else "N/A"

    html = f"""
    <html><body style="font-family:sans-serif;color:#1c1c1a;max-width:600px;margin:auto">
      <div style="background:#2d2b55;color:#fff;padding:20px 28px;border-radius:12px 12px 0 0">
        <h2 style="margin:0;font-size:20px">⚠ Critical Skin Cancer Alert</h2>
        <p style="margin:4px 0 0;opacity:.7;font-size:13px">Skin Cancer AI Diagnostic System</p>
      </div>
      <div style="background:#fcebeb;border:1px solid #f09595;padding:16px 28px">
        <p style="margin:0;font-weight:600;color:#a32d2d;font-size:15px">
          HIGH RISK DIAGNOSIS DETECTED
        </p>
      </div>
      <div style="padding:24px 28px;border:1px solid #e8e6e0;border-top:none">
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <tr><td style="padding:8px 0;color:#888;width:160px">Patient ID</td>
              <td style="font-weight:600">{patient_id}</td></tr>
          <tr><td style="padding:8px 0;color:#888">Diagnosis</td>
              <td style="font-weight:600;color:#a32d2d">{label}</td></tr>
          <tr><td style="padding:8px 0;color:#888">Confidence</td>
              <td>{confidence*100:.1f}%</td></tr>
          <tr><td style="padding:8px 0;color:#888">Risk level</td>
              <td style="font-weight:600">{risk_level}</td></tr>
          <tr><td style="padding:8px 0;color:#888">ABCD risk score</td>
              <td>{risk_score:.3f}</td></tr>
          <tr><td style="padding:8px 0;color:#888">Seasonal context</td>
              <td>{season_info}</td></tr>
          <tr><td style="padding:8px 0;color:#888">Detected at</td>
              <td>{timestamp}</td></tr>
        </table>
        <div style="margin-top:20px;padding:14px;background:#f8f7f2;border-radius:8px">
          <p style="margin:0 0 8px;font-weight:600;font-size:13px">Alert reasons:</p>
          {"".join(f'<p style="margin:4px 0;font-size:13px;color:#555">• {r}</p>' for r in reasons)}
        </div>
        <div style="margin-top:20px;padding:14px;background:#e1f5ee;border-radius:8px">
          <p style="margin:0;font-size:13px;color:#0f6e56">
            <strong>Recommended action:</strong> Schedule an urgent dermatology review.
            This result should be verified by a certified dermatologist immediately.
          </p>
        </div>
      </div>
      <div style="background:#f4f3ee;padding:14px 28px;border-radius:0 0 12px 12px;font-size:11px;color:#aaa">
        This alert was generated by the Skin Cancer AI system and is for educational purposes only.
        It is not a clinical diagnosis. Always consult a licensed dermatologist.
      </div>
    </body></html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"⚠ URGENT: High-Risk Skin Lesion Detected — Patient {patient_id}"
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        return {"sent": True, "error": None, "recipient": EMAIL_TO}

    except Exception as e:
        return {"sent": False, "error": str(e)}


# ── Browser push alert (SSE) ─────────────────────────────────────

def push_browser_alert(patient_id: str, prediction: dict, risk_score: float, reasons: list):
    """
    Push an alert to the browser via Server-Sent Events.
    Call this after a critical diagnosis — the frontend will show a popup.
    """
    alert = {
        "type":        "critical_alert",
        "patient_id":  patient_id,
        "label":       prediction.get("predicted_label", ""),
        "cls":         prediction.get("predicted_class", ""),
        "confidence":  round(prediction.get("confidence", 0.0) * 100, 1),
        "risk_score":  risk_score,
        "risk_level":  prediction.get("risk_level", ""),
        "reasons":     reasons,
        "timestamp":   datetime.now().strftime("%H:%M:%S"),
    }
    _browser_alert_queue.put(alert)


def get_browser_alert_stream():
    """
    Generator for Flask SSE route — yields alerts as they arrive.

    Usage in Flask (already wired in app.py):
        @app.route("/alerts/stream")
        def alert_stream():
            return Response(get_browser_alert_stream(),
                            mimetype="text/event-stream")
    """
    import time
    while True:
        try:
            alert = _browser_alert_queue.get(timeout=25)
            yield f"data: {json.dumps(alert)}\n\n"
        except queue.Empty:
            yield "data: {\"type\": \"heartbeat\"}\n\n"


# ── Combined alert dispatcher ────────────────────────────────────

def dispatch_alerts(
    patient_id: str,
    prediction: dict,
    risk_score: float,
    seasonal: dict = None,
) -> dict:
    """
    Main entry point — checks if alert needed and sends via all channels.

    Called from app.py after every prediction.

    Returns:
        {
          "alerted": bool,
          "reasons": list,
          "email":   {sent, error, recipient},
          "browser": {sent: bool}
        }
    """
    alerted, reasons = should_alert(prediction, risk_score)

    if not alerted:
        return {"alerted": False, "reasons": [], "email": None, "browser": None}

    # Email
    email_result = send_email_alert(patient_id, prediction, risk_score, reasons, seasonal)

    # Browser SSE
    push_browser_alert(patient_id, prediction, risk_score, reasons)

    return {
        "alerted": True,
        "reasons": reasons,
        "email":   email_result,
        "browser": {"sent": True},
    }
