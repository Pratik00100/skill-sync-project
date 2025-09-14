import os
import time
import json
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt

# OCR / NLP
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Optional model (kept non-blocking)
import joblib
import pandas as pd

# ======================
# App & Config
# ======================
app = Flask(__name__)
bcrypt = Bcrypt(app)

# CORS: allow Netlify + local dev
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://skill-sync-ai-health-claim.netlify.app",
    ]}},
)

BASE_DIR = Path(__file__).resolve().parent

# ======================
# MongoDB Setup (replacing SQLite/SQLAlchemy)
# ======================
from pymongo import MongoClient
from bson import ObjectId

MONGO_URI = "mongodb+srv://<db_username>:<db_password>@cluster1.5zc9vj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(MONGO_URI)
db = client["insurance_system"]   # choose a database name
users = db["users"]
claims = db["claims"]
claim_history = db["claim_history"]

# ======================
# Optional ML model load (non-fatal)
# ======================
MODEL = None
try:
    model_path = BASE_DIR / "models" / "rf_model.pkl"
    if model_path.exists():
        MODEL = joblib.load(model_path)
        print("[ML] RandomForest model loaded.")
    else:
        print("[ML] No rf_model.pkl found. Skipping ML.")
except Exception as e:
    print(f"[ML] Failed to load model: {e}")

# ======================
# Assessment Engine bootstrap
# ======================
from assessment.assessment_engine_v2 import (
    load_catalog, load_fees, rule_assess, price_and_eob
)

ASSESS_DIR = BASE_DIR / "assessment"
CATALOG_PATH = ASSESS_DIR / "data" / "benefit_catalog.json"
FEE_PATH     = ASSESS_DIR / "data" / "fee_schedule.json"

catalog = load_catalog(CATALOG_PATH)
fees    = load_fees(FEE_PATH)

PLAN_CFG = {}

# ======================
# Helpers: OCR + NLP + Assessment
# ======================
def extract_text_from_file(file_path: str) -> str:
    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() or ""
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            text = pytesseract.image_to_string(Image.open(file_path))
        print("--- Text extracted successfully ---")
    except Exception as e:
        print(f"[OCR] error: {e}")
    return text


def nlp_parse_text(text: str) -> dict:
    if not text:
        return {"nlp_extracted_amount": None}
    patterns = [
        r"Total\s+Amount\s+Due:?\s*\$?(\d{1,6}(?:\.\d{2})?)",
        r"Amount\s+Due:?\s*\$?(\d{1,6}(?:\.\d{2})?)",
        r"Total:?\s*\$?(\d{1,6}(?:\.\d{2})?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return {"nlp_extracted_amount": float(m.group(1))}
            except Exception:
                pass
    return {"nlp_extracted_amount": None}


def run_assessment_with_rules(description: str, amount: float, raw_text: str, nlp_amount: float | None):
    billed_amount = float(amount or 0.0)
    text = f"{raw_text or ''} {description or ''}".lower()

    service_type = "inpatient" if any(k in text for k in ["admission", "ward", "inpatient"]) else "outpatient"
    is_emergency = 1 if any(k in text for k in ["emergency", "ed ", "er "]) else 0

    claim_rec = {
        "plan_type": "hospital",
        "clinical_category": description or "",
        "billed_amount": billed_amount,
        "coverage_limit": 1e9,
        "in_network": 1,
        "provider_approved": 1,
        "hospital_tier": 1,
        "country": "au",
        "is_emergency": is_emergency,
        "service_type": service_type,
        "policy_active": 1,
    }

    hard_block, reason, details = rule_assess(claim_rec, catalog, fees)
    eob = price_and_eob(claim_rec, details, fees, PLAN_CFG)
    plan_payable = float(eob.get("plan_payable", 0) or 0)

    decision = "Approved"
    risk_score = 20
    reasons = ["Within coverage; payable amount calculated."]

    return {
        "decision": decision,
        "risk_score": risk_score,
        "reasons": reasons,
        "signals": {
            "service_type": service_type,
            "is_emergency": is_emergency,
            "plan_payable": plan_payable
        },
        "eob": eob,
    }

# ======================
# API Routes
# ======================
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": time.time()})


@app.post("/api/register")
def register():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    pwd = data.get("password", "")
    if not email or not pwd:
        return jsonify({"message": "email and password required"}), 400

    if users.find_one({"email": email}):
        return jsonify({"message": "email already exists"}), 400

    hashed = bcrypt.generate_password_hash(pwd).decode("utf-8")
    role = data.get("role", "policyholder")
    users.insert_one({"email": email, "password": hashed, "role": role, "created_at": datetime.utcnow()})
    return jsonify({"message": "New user created!"}), 201


@app.post("/api/login")
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    pwd = data.get("password", "")
    user = users.find_one({"email": email})
    if user and bcrypt.check_password_hash(user["password"], pwd):
        return jsonify({"message": "Login successful!", "role": user["role"], "access_token": "session"})
    return jsonify({"message": "Login failed! Check email and password."}), 401


@app.post("/api/submit")
def submit_claim():
    full_name = request.form.get("fullName")
    email = request.form.get("email")
    phone = request.form.get("phone")
    description = request.form.get("description")
    amount = request.form.get("amount", type=float)
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No document file part"}), 400

    upload_folder = BASE_DIR / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time())}_{file.filename}"
    file_path = str(upload_folder / filename)
    file.save(file_path)

    raw_text = extract_text_from_file(file_path)
    nlp_data = nlp_parse_text(raw_text)

    assessment = run_assessment_with_rules(description, amount, raw_text, nlp_data.get("nlp_extracted_amount"))

    new_claim = {
        "claim_id_str": f"C-{int(time.time())}",
        "full_name": full_name,
        "email_address": email,
        "phone_number": phone,
        "claim_amount": amount,
        "claim_description": description,
        "file_path": file_path,
        "nlp_extracted_amount": nlp_data.get("nlp_extracted_amount"),
        "ai_prediction": assessment["decision"],
        "status": assessment["decision"],
        "risk_score": assessment["risk_score"],
        "decision_reason": "; ".join(assessment["reasons"]),
        "signals": assessment["signals"],
        "eob": assessment["eob"],
        "created_at": datetime.utcnow(),
    }
    claims.insert_one(new_claim)

    return jsonify({
        "message": "Claim submitted and analyzed successfully!",
        "claim_id": new_claim["claim_id_str"],
        "prediction": assessment["decision"],
        "risk_score": assessment["risk_score"],
        "reasons": assessment["reasons"],
        "eob": assessment["eob"],
    }), 201


@app.get("/api/claims")
def list_claims():
    result = []
    for c in claims.find().sort("created_at", -1):
        result.append({
            "id": c["claim_id_str"],
            "status": c["status"],
            "procedure": c.get("claim_description"),
            "amount": c.get("claim_amount"),
        })
    return jsonify(result)


@app.get("/api/claims/<claim_id>")
def get_claim(claim_id):
    c = claims.find_one({"claim_id_str": claim_id})
    if not c:
        return jsonify({"error": "Claim not found"}), 404
    return jsonify({
        "id": c["claim_id_str"],
        "status": c["status"],
        "procedure": c.get("claim_description"),
        "amount": c.get("claim_amount"),
        "ai_prediction": c.get("ai_prediction"),
        "nlp_extracted_amount": c.get("nlp_extracted_amount"),
        "risk_score": c.get("risk_score", 0),
        "decision_reason": c.get("decision_reason", ""),
        "signals": c.get("signals", {}),
        "eob": c.get("eob", {}),
    })


@app.post("/api/claims/<claim_id>/decision")
def set_claim_decision(claim_id):
    data = request.get_json() or {}
    decision = data.get("decision")
    note = data.get("note", "")
    mapping = {"approve": "Approved", "reject": "Rejected", "manual_review": "Manual Review"}

    c = claims.find_one({"claim_id_str": claim_id})
    if not c:
        return jsonify({"error": "not found"}), 404

    if decision in mapping:
        claims.update_one({"claim_id_str": claim_id}, {"$set": {"status": mapping[decision], "ai_prediction": mapping[decision]}})
        claim_history.insert_one({
            "claim_id": c["_id"],
            "action": decision,
            "actor_email": "insurer@test.com",
            "note": note,
            "timestamp": datetime.utcnow()
        })
        return jsonify({"message": "updated", "status": mapping[decision]})
    return jsonify({"message": "invalid decision"}), 400


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
