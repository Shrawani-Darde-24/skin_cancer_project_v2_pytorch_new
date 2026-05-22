"""
Scheduled Reminders — Skin Checkup & Medical Review System
===========================================================
Manages and sends reminders via:
  - Email (Gmail SMTP)
  - Browser notification (SSE)

Reminder types:
  - routine_checkup:   Regular self-exam reminder (monthly)
  - followup_review:   Follow up a previous visit (custom interval)
  - annual_screening:  Annual dermatologist visit
  - urgent_review:     Triggered by high-risk result (7 days)

Usage:
    from reminders import ReminderScheduler
    scheduler = ReminderScheduler()
    scheduler.schedule_from_prediction(patient_id, prediction, patient_email)
    scheduler.check_and_send(patient_id)   # call on startup or via cron
"""

import os
import json
import smtplib
import queue
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


REMINDERS_FILE = "data/reminders.json"
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 587
EMAIL_FROM     = os.environ.get("ALERT_EMAIL_FROM", "")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")

_reminder_browser_queue: queue.Queue = queue.Queue()

REMINDER_CONFIG = {
    "routine_checkup": {
        "label":        "Monthly self-examination reminder",
        "default_days": 30,
        "color":        "#534AB7",
        "icon":         "🔍",
    },
    "followup_review": {
        "label":        "Follow-up dermatology review",
        "default_days": 90,
        "color":        "#1D9E75",
        "icon":         "🏥",
    },
    "annual_screening": {
        "label":        "Annual skin cancer screening",
        "default_days": 365,
        "color":        "#378ADD",
        "icon":         "📅",
    },
    "urgent_review": {
        "label":        "URGENT — Dermatologist review required",
        "default_days": 7,
        "color":        "#A32D2D",
        "icon":         "⚠",
    },
}


class ReminderScheduler:

    def __init__(self, reminders_file: str = REMINDERS_FILE):
        self.file = reminders_file
        os.makedirs(os.path.dirname(reminders_file), exist_ok=True)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not os.path.exists(self.file):
            return {}
        with open(self.file) as f:
            return json.load(f)

    def _save(self, db: dict):
        with open(self.file, "w") as f:
            json.dump(db, f, indent=2)

    # ── Schedule ─────────────────────────────────────────────────────────────

    def schedule(
        self,
        patient_id:    str,
        reminder_type: str,
        due_date:      str = None,
        patient_email: str = None,
        note:          str = "",
        days_from_now: int = None,
    ) -> dict:
        """
        Schedule a reminder for a patient.

        Args:
            patient_id:    Patient identifier (must match app.py patient_id field).
            reminder_type: One of routine_checkup / followup_review /
                           annual_screening / urgent_review.
            due_date:      ISO date string (YYYY-MM-DD).
                           If None, computed from days_from_now.
            patient_email: Email address to send the reminder to.
            note:          Optional clinical note attached to the reminder.
            days_from_now: Override the type's default interval in days.

        Returns:
            The scheduled reminder dict (also persisted to data/reminders.json).
        """
        config = REMINDER_CONFIG.get(reminder_type, REMINDER_CONFIG["routine_checkup"])

        if due_date is None:
            days     = days_from_now or config["default_days"]
            due_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        reminder = {
            "id":            f"{patient_id}_{reminder_type}_{due_date}",
            "patient_id":    patient_id,
            "type":          reminder_type,
            "label":         config["label"],
            "due_date":      due_date,
            "patient_email": patient_email or EMAIL_FROM,
            "note":          note,
            "sent":          False,
            "created_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        db = self._load()
        if patient_id not in db:
            db[patient_id] = []

        # Avoid duplicates by composite ID
        existing_ids = {r["id"] for r in db[patient_id]}
        if reminder["id"] not in existing_ids:
            db[patient_id].append(reminder)
            self._save(db)

        return reminder

    def schedule_from_prediction(
        self,
        patient_id:    str,
        prediction:    dict,
        patient_email: str = None,
    ) -> list:
        """
        Auto-schedule appropriate reminders based on diagnosis result.
        Called automatically from app.py after every prediction.

        Malignant result  → urgent_review (7 days) + followup_review (30 days)
        Benign result     → routine_checkup (30 days) + annual_screening (365 days)

        Returns list of scheduled reminder dicts.
        """
        cls       = prediction.get("predicted_class", "")
        scheduled = []
        malignant = {"mel", "bcc", "akiec"}

        if cls in malignant:
            scheduled.append(self.schedule(
                patient_id, "urgent_review",
                days_from_now=7,
                patient_email=patient_email,
                note=f"High-risk diagnosis: {prediction.get('predicted_label', cls)}",
            ))
            scheduled.append(self.schedule(
                patient_id, "followup_review",
                days_from_now=30,
                patient_email=patient_email,
            ))
        else:
            scheduled.append(self.schedule(
                patient_id, "routine_checkup",
                days_from_now=30,
                patient_email=patient_email,
            ))
            scheduled.append(self.schedule(
                patient_id, "annual_screening",
                days_from_now=365,
                patient_email=patient_email,
            ))

        return scheduled

    # ── Query ────────────────────────────────────────────────────────────────

    def get_upcoming(self, patient_id: str, days_ahead: int = 90) -> list:
        """Return upcoming unsent reminders for a patient within the next N days."""
        db       = self._load()
        today    = datetime.now().strftime("%Y-%m-%d")
        cutoff   = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        reminders = db.get(patient_id, [])
        return sorted(
            [r for r in reminders if r["due_date"] >= today and r["due_date"] <= cutoff],
            key=lambda r: r["due_date"],
        )

    def get_all(self, patient_id: str) -> list:
        """Return all reminders (sent and unsent) for a patient."""
        db = self._load()
        return db.get(patient_id, [])

    # ── Send ─────────────────────────────────────────────────────────────────

    def check_and_send(self, patient_id: str = None) -> list:
        """
        Check all due reminders and send notifications.

        Args:
            patient_id: Check only this patient; pass None to check all patients.

        Returns list of {reminder, result} dicts for every reminder that was sent.
        Also called automatically from app.py after scheduling.
        """
        db       = self._load()
        today    = datetime.now().strftime("%Y-%m-%d")
        sent     = []
        patients = [patient_id] if patient_id else list(db.keys())

        for pid in patients:
            for reminder in db.get(pid, []):
                if reminder["sent"]:
                    continue
                if reminder["due_date"] <= today:
                    result = self._send_reminder(reminder)
                    reminder["sent"]    = True
                    reminder["sent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    sent.append({"reminder": reminder, "result": result})

        self._save(db)
        return sent

    def _send_reminder(self, reminder: dict) -> dict:
        config  = REMINDER_CONFIG.get(reminder["type"], REMINDER_CONFIG["routine_checkup"])
        results = {}

        # Browser SSE notification
        _reminder_browser_queue.put({
            "type":          "reminder",
            "reminder_type": reminder["type"],
            "label":         reminder["label"],
            "patient_id":    reminder["patient_id"],
            "due_date":      reminder["due_date"],
            "note":          reminder.get("note", ""),
            "color":         config["color"],
            "icon":          config["icon"],
        })
        results["browser"] = {"sent": True}

        # Email
        results["email"] = self._send_email_reminder(reminder, config)

        return results

    def _send_email_reminder(self, reminder: dict, config: dict) -> dict:
        if not EMAIL_FROM or not EMAIL_PASSWORD:
            return {"sent": False, "error": "Email not configured"}

        recipient = reminder.get("patient_email") or EMAIL_FROM
        pid       = reminder["patient_id"]
        due       = reminder["due_date"]
        label     = reminder["label"]
        note      = reminder.get("note", "")
        icon      = config["icon"]
        color     = config["color"]

        html = f"""
        <html><body style="font-family:sans-serif;color:#1c1c1a;max-width:600px;margin:auto">
          <div style="background:#2d2b55;color:#fff;padding:20px 28px;border-radius:12px 12px 0 0">
            <h2 style="margin:0;font-size:18px">{icon} Skin Health Reminder</h2>
            <p style="margin:4px 0 0;opacity:.7;font-size:12px">Skin Cancer AI — Patient Care System</p>
          </div>
          <div style="padding:24px 28px;border:1px solid #e8e6e0;border-top:none">
            <div style="background:{color}18;border-left:4px solid {color};padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:20px">
              <p style="margin:0;font-weight:600;color:{color};font-size:15px">{label}</p>
              <p style="margin:6px 0 0;font-size:13px;color:#555">Due: {due}</p>
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:14px">
              <tr><td style="padding:8px 0;color:#888;width:140px">Patient ID</td>
                  <td style="font-weight:600">{pid}</td></tr>
              <tr><td style="padding:8px 0;color:#888">Reminder type</td>
                  <td>{reminder['type'].replace('_',' ').title()}</td></tr>
              <tr><td style="padding:8px 0;color:#888">Due date</td>
                  <td style="font-weight:600">{due}</td></tr>
              {"<tr><td style='padding:8px 0;color:#888'>Clinical note</td><td>" + note + "</td></tr>" if note else ""}
            </table>
            <div style="margin-top:20px;padding:14px;background:#f8f7f2;border-radius:8px">
              <p style="margin:0;font-size:13px;color:#555">
                Please schedule your appointment with a dermatologist or perform
                your self-skin examination as recommended.
              </p>
            </div>
          </div>
          <div style="background:#f4f3ee;padding:14px 28px;border-radius:0 0 12px 12px;font-size:11px;color:#aaa">
            This reminder was automatically generated by the Skin Cancer AI system.
            Always follow your dermatologist's advice.
          </div>
        </body></html>
        """

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{icon} Reminder: {label} — {due}"
            msg["From"]    = EMAIL_FROM
            msg["To"]      = recipient
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.sendmail(EMAIL_FROM, recipient, msg.as_string())

            return {"sent": True, "recipient": recipient}

        except Exception as e:
            return {"sent": False, "error": str(e)}


# ── SSE stream ────────────────────────────────────────────────────────────────

def get_reminder_stream():
    """
    SSE generator for browser reminder notifications.
    Already wired in app.py at GET /reminders/stream.
    Yields a heartbeat every 25 s when no reminders are pending.
    """
    import json as _json
    while True:
        try:
            reminder = _reminder_browser_queue.get(timeout=25)
            yield f"data: {_json.dumps(reminder)}\n\n"
        except queue.Empty:
            yield "data: {\"type\": \"heartbeat\"}\n\n"
