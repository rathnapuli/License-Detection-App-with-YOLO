import os
import uuid
import base64
import requests
import traceback
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import google.generativeai as genai
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///vpd.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ─── ENVIRONMENT KEYS ─────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
EMAILJS_SERVICE_ID  = os.environ.get("EMAILJS_SERVICE_ID",  "service_x1emgp9")
EMAILJS_TEMPLATE_ID = os.environ.get("EMAILJS_TEMPLATE_ID", "template_t9feq2h")
EMAILJS_PUBLIC_KEY  = os.environ.get("EMAILJS_PUBLIC_KEY",  "mYQVEFFqKFz4y8v4D")
ALERT_RECIPIENT     = os.environ.get("ALERT_RECIPIENT",     "rathnapuli22@gmail.com")
BLACKLIST_CSV   = os.environ.get("BLACKLIST_CSV", "blocklist.csv")
MODEL_PATH      = os.environ.get("MODEL_PATH", "best.pt")

# ─── EXTENSIONS ───────────────────────────────────────────────────────────────
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ─── MODELS ───────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── FORMS ────────────────────────────────────────────────────────────────────
class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(3, 80)])
    email    = EmailField("Email",    validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[
        DataRequired(),
        Length(min=6, message="Min 6 characters"),
    ])
    confirm  = PasswordField("Confirm Password", validators=[
        DataRequired(), EqualTo("password", message="Passwords must match")
    ])
    submit   = SubmitField("Register")

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit   = SubmitField("Login")

class PredictForm(FlaskForm):
    image  = FileField("Image", validators=[
        FileAllowed(ALLOWED_EXTENSIONS, "JPG/JPEG/PNG only")
    ])
    submit = SubmitField("Detect Plates")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_blacklist():
    p = Path(BLACKLIST_CSV)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, header=None)
        return set(df.iloc[:, 0].astype(str).str.upper().str.strip())
    except Exception:
        return set()

BLACKLIST = load_blacklist()

def download_model_if_needed():
    """Auto-download a pretrained license plate YOLOv8 model if best.pt is missing."""
    p = Path(MODEL_PATH)
    if p.exists():
        print(f"Model found at {p}")
        return
    print("best.pt not found — downloading pretrained license plate model from Hugging Face...")
    try:
        import urllib.request
        # YOLOv8n trained on Roboflow license-plate-recognition dataset (same as your notebook)
        url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
        urllib.request.urlretrieve(url, str(p))
        print(f"Downloaded model to {p} ({p.stat().st_size // 1024 // 1024} MB)")
    except Exception as e:
        print(f"Model download failed: {e}. Detection will be unavailable.")

def load_yolo():
    download_model_if_needed()
    p = Path(MODEL_PATH)
    if not p.exists():
        return None
    try:
        return YOLO(str(p))
    except Exception as e:
        print("YOLO load error:", e)
        return None

YOLO_MODEL = load_yolo()

def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def gemini_read_plate(crop_path: Path) -> str:
    if not GEMINI_API_KEY:
        return "NO_KEY"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        with open(crop_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Extract ONLY the visible registration number or text from this license plate image. "
            "The plate may contain letters (A-Z) and/or digits (0-9). "
            "There may be between 4 and 10 characters. "
            "Return only the plate text, no spaces, punctuation, or commentary."
        )
        response = model.generate_content(
            [{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            ]}],
            generation_config={"temperature": 0.0, "max_output_tokens": 200},
        )
        text = ""
        if hasattr(response, "candidates"):
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text += part.text.strip()
        if not text.strip():
            return "NO_PLATE"
        plate = "".join(c for c in text.upper() if c.isalnum())
        return plate[:10] if len(plate) > 10 else plate or "EMPTY"
    except Exception as e:
        print("Gemini OCR error:", e)
        traceback.print_exc()
        return "ERROR"

def send_alert_email(plate_text, result_img_path=None):
    """Send blacklist alert via EmailJS REST API."""
    try:
        payload = {
            "service_id":  EMAILJS_SERVICE_ID,
            "template_id": EMAILJS_TEMPLATE_ID,
            "user_id":     EMAILJS_PUBLIC_KEY,
            "template_params": {
                "to_email": ALERT_RECIPIENT,
                "plate":    plate_text,
                "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user":     "VPD System",
            },
        }
        resp = requests.post(
            "https://api.emailjs.com/api/v1.0/email/send",
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"Alert email sent for plate {plate_text}")
        else:
            print(f"EmailJS error {resp.status_code}: {resp.text}")
    except Exception as e:
        print("Email error:", e)
        traceback.print_exc()

def run_detection(image_path: str):
    """Run YOLO + Gemini pipeline. Returns (detections, result_img_filename, alerts)."""
    uid        = uuid.uuid4().hex[:8]
    img        = cv2.imread(image_path)
    detections = []
    alerts     = []

    if YOLO_MODEL is None:
        return detections, None, alerts

    results = YOLO_MODEL(image_path)
    h, w    = img.shape[:2]

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        pad_x = int(0.25 * (x2 - x1))
        pad_y = int(0.25 * (y2 - y1))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop         = img[cy1:cy2, cx1:cx2]
        crop_enh     = enhance_for_ocr(crop)
        crop_fname   = f"crop_{uid}_{i}.jpg"
        crop_path    = UPLOAD_FOLDER / crop_fname
        cv2.imwrite(str(crop_path), crop_enh)

        plate_text = gemini_read_plate(crop_path)

        # Draw on annotated image
        color = (0, 0, 255) if plate_text.upper() in BLACKLIST else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{plate_text} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if plate_text.upper() in BLACKLIST:
            alerts.append(plate_text)

        detections.append({
            "plate": plate_text,
            "conf":  round(conf, 3),
            "crop":  crop_fname,
        })

    result_fname = f"result_{uid}.jpg"
    result_path  = UPLOAD_FOLDER / result_fname
    cv2.imwrite(str(result_path), img)

    # Send email alerts
    for plate in alerts:
        send_alert_email(plate, str(result_path))

    return detections, result_fname, alerts

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash("Username already taken.", "danger")
            return render_template("register.html", form=form)
        if User.query.filter_by(email=form.email.data).first():
            flash("Email already registered.", "danger")
            return render_template("register.html", form=form)
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=generate_password_hash(form.password.data),
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("predict"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    form       = PredictForm()
    detections = None
    result_img = None
    alerts     = []

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return render_template("predict.html", form=form)
        if not allowed_file(file.filename):
            flash("Only JPG/JPEG/PNG files are allowed.", "danger")
            return render_template("predict.html", form=form)

        fname    = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        fpath    = UPLOAD_FOLDER / fname
        file.save(str(fpath))

        detections, result_img, alerts = run_detection(str(fpath))

    return render_template(
        "predict.html",
        form=form,
        detections=detections,
        result_img=result_img,
        alerts=alerts,
        BLACKLIST=BLACKLIST,
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ─── INIT ─────────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
