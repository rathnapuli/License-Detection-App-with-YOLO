# Vehicle Plate Detection System

## Project Structure
```
vpd_project/
├── app.py                # Flask backend
├── requirements.txt      # Python dependencies
├── render.yaml           # Render.com deployment config
├── blocklist.csv         # Blacklisted plates (one per line)
├── best.pt               # ⚠️ YOU MUST ADD THIS: your trained YOLOv10 model
├── static/
│   └── uploads/          # Auto-created at runtime
└── templates/
    ├── home.html
    ├── login.html
    ├── register.html
    └── predict.html
```

---

## ✅ No model file needed!

`app.py` will **automatically download** a pretrained YOLOv8 license plate detection model
from Hugging Face on first startup. Nothing to do here.

---

## Local Testing

```bash
cd vpd_project
pip install -r requirements.txt

# Set environment variables (or create a .env file)
export SECRET_KEY="any-random-string"
export GEMINI_API_KEY="your-gemini-api-key"
export GMAIL_SENDER="your@gmail.com"
export GMAIL_PASSWORD="your-app-password"   # Gmail App Password, not your login password
export GMAIL_RECIPIENT="admin@gmail.com"

python app.py
# Visit http://localhost:5000
```

---

## Deploy to Render.com (Free Plan)

### Step 1 — Push to GitHub
1. Create a new GitHub repo (private or public)
2. Add all files **including `best.pt`** and push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

### Step 2 — Create Web Service on Render
1. Go to https://render.com → **New → Web Service**
2. Connect your GitHub repo
3. Set these values:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`

### Step 3 — Set Environment Variables in Render Dashboard
Go to your service → **Environment** tab → Add:

| Key | Value |
|-----|-------|
| `SECRET_KEY` | any long random string |
| `GEMINI_API_KEY` | your Google Gemini API key |
| `GMAIL_SENDER` | your Gmail address |
| `GMAIL_PASSWORD` | your Gmail App Password |
| `GMAIL_RECIPIENT` | email to receive alerts |

### Step 4 — Add a Disk (for file uploads)
Render free plan has ephemeral storage. To persist uploads:
- Go to **Disks** in your service settings
- Mount path: `/opt/render/project/src/static/uploads`
- Size: 1 GB

### Step 5 — Deploy
Click **Deploy** — your app will be live at `https://your-service-name.onrender.com`

---

## Gmail App Password Setup
1. Enable 2FA on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Create an App Password for "Mail"
4. Use that 16-character password as `GMAIL_PASSWORD`

---

## Blacklist Management
Edit `blocklist.csv` — one plate per line (no header needed, or use "plate" as header):
```
ABC1234
XYZ9999
MH12AB1234
```

---

## Notes for Free Render Plan
- The free plan **spins down after 15 min of inactivity** — first request after sleep takes ~30s
- The `best.pt` model file must be included in the repo (if under 100MB) or stored on a cloud drive and downloaded at startup
- If `best.pt` is large (>100MB), consider storing it on Google Drive and adding a startup download script
