from pathlib import Path
import functools
import json
import sqlite3
import subprocess
import sys
from datetime import datetime

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
import joblib
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"
DB_PATH = BASE_DIR / "student_predictions.db"

FEATURE_COLUMNS = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "study_hours",
    "previous_sem_marks",
    "participation",
    "overall_score",
    "study_performance",
]

app = Flask(__name__)
app.secret_key = "student_performance_prediction_super_secret_key_2026"


def ensure_model_exists():
    if MODEL_PATH.exists() and MODEL_INFO_PATH.exists():
        return
    subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)


ensure_model_exists()
model = joblib.load(MODEL_PATH)
if FEATURES_PATH.exists():
    feature_names = joblib.load(FEATURES_PATH)
else:
    feature_names = FEATURE_COLUMNS


def normalize_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input data must be a JSON object or form dictionary.")

    payload = dict(data)
    if not payload.get("student_name", "").strip():
        raise ValueError("Student name must not be empty.")

    for key in ["attendance", "internal_marks", "assignment_score", "study_hours", "previous_sem_marks", "participation"]:
        if key not in payload:
            if key == "previous_sem_marks" and "previous_semester_marks" in payload:
                payload[key] = payload["previous_semester_marks"]
            else:
                raise ValueError(f"Missing required field: {key}")

    if "previous_semester_marks" in payload and "previous_sem_marks" not in payload:
        payload["previous_sem_marks"] = payload["previous_semester_marks"]

    payload["student_name"] = str(payload["student_name"]).strip()

    numeric_fields = {
        "attendance": (0, 100),
        "internal_marks": (0, 100),
        "assignment_score": (0, 100),
        "study_hours": (0, 24),
        "previous_sem_marks": (0, 100),
        "participation": (0, 100),
    }

    for field, (min_value, max_value) in numeric_fields.items():
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a valid number.")

        if value < min_value or value > max_value:
            raise ValueError(f"{field} must be between {min_value} and {max_value}.")

        payload[field] = value

    return payload


def build_feature_row(data):
    data = normalize_input(data)
    overall_score = (
        data["attendance"]
        + data["internal_marks"]
        + data["assignment_score"]
        + data["previous_sem_marks"]
        + data["participation"]
    ) / 5

    study_performance = (
        (data["attendance"] + data["internal_marks"] + data["assignment_score"] + data["previous_sem_marks"]) / 4
        + (data["study_hours"] * 5)
    )

    row = {
        "attendance": data["attendance"],
        "internal_marks": data["internal_marks"],
        "assignment_score": data["assignment_score"],
        "study_hours": data["study_hours"],
        "previous_sem_marks": data["previous_sem_marks"],
        "participation": data["participation"],
        "overall_score": overall_score,
        "study_performance": study_performance,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            attendance REAL NOT NULL,
            internal_marks REAL NOT NULL,
            assignment_score REAL NOT NULL,
            study_hours REAL NOT NULL,
            previous_sem_marks REAL NOT NULL,
            participation REAL NOT NULL,
            prediction TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'student'")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass

    # Seed default Teacher (Admin) and Student (User) accounts if not already present
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    teacher_exists = conn.execute("SELECT id FROM users WHERE username = 'teacher_admin'").fetchone()
    if not teacher_exists:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "teacher_admin",
                "teacher@school.edu",
                generate_password_hash("teacher123"),
                "teacher",
                now,
            ),
        )

    student_exists = conn.execute("SELECT id FROM users WHERE username = 'student_user'").fetchone()
    if not student_exists:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "student_user",
                "student@school.edu",
                generate_password_hash("student123"),
                "student",
                now,
            ),
        )

    conn.commit()
    conn.close()


init_db()


@app.context_processor
def inject_user():
    user_id = session.get("user_id")
    if user_id:
        try:
            conn = get_db()
            user = conn.execute("SELECT id, username, email, role FROM users WHERE id = ?", (user_id,)).fetchone()
            conn.close()
            if user:
                user_dict = dict(user)
                user_dict["is_teacher"] = (user_dict.get("role") == "teacher")
                user_dict["is_student"] = (user_dict.get("role") == "student")
                return {"current_user": user_dict}
        except Exception:
            pass
    return {"current_user": None}


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please sign in or create an account to access the student dashboard.", "warning")
            return redirect(url_for("login", next=request.url))
        return view(**kwargs)
    return wrapped_view


def teacher_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please log in as a Teacher (Admin) to access this feature.", "warning")
            return redirect(url_for("login", next=request.url))
        if session.get("role") != "teacher":
            flash("Permission denied: Only Teachers (Admins) have permission to download PDF reports and administrative features.", "danger")
            return redirect(request.referrer or url_for("dashboard"))
        return view(**kwargs)
    return wrapped_view


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "student").strip().lower()

        if role not in ["teacher", "student"]:
            role = "student"

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("signup.html", form_data={"username": username, "email": email, "role": role})

        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "danger")
            return render_template("signup.html", form_data={"username": username, "email": email, "role": role})

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("signup.html", form_data={"username": username, "email": email, "role": role})

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html", form_data={"username": username, "email": email, "role": role})

        conn = get_db()
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone()

        if existing_user:
            conn.close()
            flash("Username or email already registered.", "danger")
            return render_template("signup.html", form_data={"username": username, "email": email, "role": role})

        password_hash = generate_password_hash(password)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, email, password_hash, role, now),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        session["user_id"] = user_id
        session["username"] = username
        session["role"] = role
        role_label = "Teacher (Admin)" if role == "teacher" else "Student"
        flash(f"Account created successfully as {role_label}! Welcome.", "success")
        next_url = request.args.get("next")
        return redirect(next_url or url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password", "")

        if not login_id or not password:
            flash("Please enter both username/email and password.", "danger")
            return render_template("login.html", login_id=login_id)

        conn = get_db()
        user = conn.execute(
            "SELECT id, username, email, password_hash, role FROM users WHERE username = ? OR email = ?",
            (login_id, login_id.lower()),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"] if user["role"] else "student"
            role_badge = "Teacher (Admin)" if session["role"] == "teacher" else "Student"
            flash(f"Welcome back, {user['username']}! Signed in as {role_badge}.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        else:
            flash("Invalid username/email or password.", "danger")
            return render_template("login.html", login_id=login_id)

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    user_id = session.get("user_id")
    role = session.get("role", "student")

    # Students see only their own predictions; teachers see all
    if role == "teacher":
        records = conn.execute(
            "SELECT id, student_name, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, prediction, created_at FROM predictions ORDER BY id DESC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        excellent = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Excellent'").fetchone()[0]
        average = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Average'").fetchone()[0]
        at_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'At Risk'").fetchone()[0]
    else:
        records = conn.execute(
            "SELECT id, student_name, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, prediction, created_at FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,)).fetchone()[0]
        excellent = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Excellent' AND user_id = ?", (user_id,)).fetchone()[0]
        average = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Average' AND user_id = ?", (user_id,)).fetchone()[0]
        at_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'At Risk' AND user_id = ?", (user_id,)).fetchone()[0]

    conn.close()

    risk_percentage = round((at_risk / total * 100), 2) if total else 0
    stats = {
        "total": total,
        "excellent": excellent,
        "average": average,
        "at_risk": at_risk,
        "risk_percentage": risk_percentage,
    }

    # Serialize records for dynamic JS charts
    records_json = json.dumps([dict(r) for r in records])

    return render_template("dashboard.html", stats=stats, history=records, records_json=records_json)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
        else:
            payload = request.form.to_dict()

        payload = normalize_input(payload)
        prepared = build_feature_row(payload)
        prediction = str(model.predict(prepared)[0])
        prediction_id = None

        try:
            student_name = payload["student_name"]
            user_id = session.get("user_id")
            conn = get_db()
            cursor = conn.execute(
                """
                INSERT INTO predictions (student_name, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, prediction, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_name,
                    float(payload["attendance"]),
                    float(payload["internal_marks"]),
                    float(payload["assignment_score"]),
                    float(payload["study_hours"]),
                    float(payload["previous_sem_marks"]),
                    float(payload["participation"]),
                    prediction,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_id,
                ),
            )
            conn.commit()
            prediction_id = cursor.lastrowid
            conn.close()
        except Exception as db_err:
            print(f"[Warning] Failed to save prediction to database: {db_err}")

        if request.is_json:
            return jsonify({
                "prediction": prediction,
                "student_name": payload["student_name"],
                "id": prediction_id,
            })

        return render_template(
            "index.html",
            prediction=prediction,
            form_data=payload,
            prediction_id=prediction_id,
        )
    except ValueError as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 400
        return render_template("index.html", error=str(exc))
    except Exception as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        return render_template("index.html", error=f"Unexpected error: {exc}")


@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    conn = get_db()
    record = conn.execute(
        "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
    ).fetchone()
    conn.close()

    if not record:
        flash("Student record not found.", "warning")
        return redirect(url_for("dashboard"))

    # Students can only view their own reports
    if session.get("role") != "teacher" and record["user_id"] != session.get("user_id"):
        flash("You can only view your own reports.", "danger")
        return redirect(url_for("dashboard"))

    overall_score = round(
        (
            record["attendance"]
            + record["internal_marks"]
            + record["assignment_score"]
            + record["previous_sem_marks"]
            + record["participation"]
        )
        / 5,
        1,
    )
    study_perf = round(
        ((record["attendance"] + record["internal_marks"] + record["assignment_score"] + record["previous_sem_marks"]) / 4)
        + (record["study_hours"] * 5),
        1,
    )

    pred = record["prediction"]
    recommendations = []
    if pred == "Excellent":
        summary_text = "Student is demonstrating exemplary academic engagement across all evaluations."
        recommendations = [
            "Encourage participation in advanced subject seminars, honors projects, or research papers.",
            "Consider student for peer mentoring and study group leadership roles.",
            "Maintain current study routine and exam preparation habits.",
        ]
    elif pred == "Average":
        summary_text = "Student shows satisfactory comprehension with clear potential for advancement to honors tier."
        recommendations = [
            "Increase dedicated study hours by 1-2 hours daily focusing on high-weightage topics.",
            "Work on assignment timeliness and internal evaluation mock tests.",
            "Engage more actively in classroom discussions and doubt-clearing sessions.",
        ]
    else:
        summary_text = "Student is in the academic risk zone requiring prompt and proactive institutional intervention."
        recommendations = [
            "Schedule an urgent one-on-one session with academic counselor/faculty advisor.",
            "Implement a mandatory minimum attendance target of 80% with weekly tracking.",
            "Enroll in remedial workshops for internal assessments and fundamental subject concepts.",
            "Set up structured daily study schedule under mentor guidance.",
        ]

    # Specific metric alerts
    alerts = []
    if record["attendance"] < 75:
        alerts.append("Attendance is below 75% threshold — at risk of exam debarment.")
    if record["study_hours"] < 2.5:
        alerts.append("Daily study hours (below 2.5h) are insufficient for consistent exam performance.")
    if record["internal_marks"] < 50:
        alerts.append("Internal marks are below 50% — requires targeted revision.")

    report_data = {
        "record": dict(record),
        "overall_score": overall_score,
        "study_performance": study_perf,
        "summary_text": summary_text,
        "recommendations": recommendations,
        "alerts": alerts,
        "generated_at": datetime.now().strftime("%B %d, %Y - %I:%M %p"),
    }

    return render_template("report.html", report=report_data)


@app.route("/api/stats")
@login_required
def api_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    excellent = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Excellent'").fetchone()[0]
    average = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Average'").fetchone()[0]
    at_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'At Risk'").fetchone()[0]
    conn.close()

    return jsonify(
        {
            "total_predictions": total,
            "excellent": excellent,
            "average": average,
            "at_risk": at_risk,
        }
    )


@app.route("/api/history")
@login_required
def api_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, student_name, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, prediction, created_at FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/model")
@login_required
def api_model():
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
        info = json.load(file)
    return jsonify(info)

@app.route("/api/students")
@login_required
def api_students():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
