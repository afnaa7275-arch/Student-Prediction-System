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
    
    # 1. users table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            student_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    
    # Check/add role and student_id columns to users if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'student'")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE users ADD COLUMN student_id TEXT")
    except sqlite3.OperationalError:
        pass

    # 2. predictions table
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
            created_at TEXT NOT NULL,
            user_id INTEGER
        )
        """
    )

    # 3. students table (Official Student Management Records)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            user_id INTEGER,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL DEFAULT 'B.Sc Computer Science',
            department TEXT NOT NULL DEFAULT 'Computer Science',
            semester TEXT NOT NULL DEFAULT 'Semester 4',
            attendance REAL NOT NULL DEFAULT 85.0,
            internal_marks REAL NOT NULL DEFAULT 80.0,
            assignment_score REAL NOT NULL DEFAULT 82.0,
            study_hours REAL NOT NULL DEFAULT 4.0,
            previous_sem_marks REAL NOT NULL DEFAULT 78.0,
            participation REAL NOT NULL DEFAULT 85.0,
            overall_score REAL NOT NULL DEFAULT 82.0,
            predicted_performance TEXT DEFAULT 'Excellent',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # 4. student_goals table (Personal Student Targets)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS student_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            target_attendance REAL DEFAULT 90.0,
            target_internal_marks REAL DEFAULT 85.0,
            target_assignment_score REAL DEFAULT 90.0,
            target_study_hours REAL DEFAULT 5.0,
            target_overall REAL DEFAULT 88.0,
            updated_at TEXT NOT NULL
        )
        """
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Seed Default Teacher & Student Accounts
    teacher_exists = conn.execute("SELECT id FROM users WHERE username = 'teacher_admin'").fetchone()
    if not teacher_exists:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, student_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("teacher_admin", "teacher@school.edu", generate_password_hash("teacher123"), "teacher", None, now),
        )

    student_exists = conn.execute("SELECT id FROM users WHERE username = 'student_user'").fetchone()
    if not student_exists:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, student_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("student_user", "student@school.edu", generate_password_hash("student123"), "student", "STU001", now),
        )
    else:
        conn.execute("UPDATE users SET student_id = 'STU001' WHERE username = 'student_user' AND (student_id IS NULL OR student_id = '')")

    # Seed initial demo students if students table is empty
    count_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    if count_students == 0:
        sample_students = [
            ("STU001", "Alex Morgan", "student_user", "Computer Science", "Semester 4", 88.5, 84.0, 82.0, 4.5, 80.0, 85.0, 83.9, "Excellent"),
            ("STU002", "Emma Watson", None, "Computer Science", "Semester 4", 95.0, 92.5, 94.0, 6.0, 91.0, 95.0, 93.5, "Excellent"),
            ("STU003", "Lucas Vance", None, "Information Technology", "Semester 4", 78.0, 72.0, 75.0, 3.5, 70.0, 68.0, 72.6, "Average"),
            ("STU004", "Jordan Miller", None, "Computer Science", "Semester 4", 62.0, 48.0, 52.0, 1.5, 50.0, 45.0, 51.4, "At Risk"),
            ("STU005", "Sophia Chen", None, "Data Science", "Semester 4", 91.0, 88.0, 90.0, 5.0, 86.0, 89.0, 88.8, "Excellent"),
            ("STU006", "Daniel Craig", None, "Information Technology", "Semester 4", 74.0, 68.0, 70.0, 3.0, 65.0, 64.0, 68.2, "Average"),
            ("STU007", "Maya Lin", None, "Computer Science", "Semester 4", 86.0, 81.0, 84.0, 4.0, 79.0, 80.0, 82.0, "Excellent"),
            ("STU008", "Ethan Hunt", None, "Computer Science", "Semester 4", 58.0, 45.0, 50.0, 1.2, 48.0, 42.0, 48.6, "At Risk"),
            ("STU009", "Olivia Taylor", None, "Data Science", "Semester 4", 82.0, 76.0, 79.0, 3.8, 75.0, 77.0, 77.8, "Average"),
            ("STU010", "Noah Patel", None, "Information Technology", "Semester 4", 93.0, 89.0, 91.0, 5.5, 87.0, 90.0, 90.0, "Excellent"),
        ]

        for stu in sample_students:
            conn.execute(
                """
                INSERT INTO students (student_id, name, department, semester, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, overall_score, predicted_performance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stu[0], stu[1], stu[3], stu[4], stu[5], stu[6], stu[7], stu[8], stu[9], stu[10], stu[11], stu[12], now, now)
            )

            # Insert default goal for each student
            conn.execute(
                """
                INSERT INTO student_goals (student_id, target_attendance, target_internal_marks, target_assignment_score, target_study_hours, target_overall, updated_at)
                VALUES (?, 90.0, 85.0, 88.0, 5.0, 88.0, ?)
                """,
                (stu[0], now)
            )

    # Link any unlinked student user to STU001 or create a profile
    conn.commit()
    conn.close()


init_db()


@app.context_processor
def inject_user():
    user_id = session.get("user_id")
    if user_id:
        try:
            conn = get_db()
            user = conn.execute("SELECT id, username, email, role, student_id FROM users WHERE id = ?", (user_id,)).fetchone()
            conn.close()
            if user:
                user_dict = dict(user)
                user_dict["is_teacher"] = (user_dict.get("role") == "teacher")
                user_dict["is_student"] = (user_dict.get("role") == "student")
                return {"current_user": user_dict}
        except Exception:
            pass
    return {"current_user": None}


# ========================================================
# ROLE-BASED ACCESS CONTROL DECORATORS
# ========================================================

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please sign in or create an account to access the system.", "warning")
            return redirect(url_for("login", next=request.url))
        return view(**kwargs)
    return wrapped_view


def teacher_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please log in as a Teacher / Administrator to access that page.", "warning")
            return redirect(url_for("login", next=request.url))
        if session.get("role") != "teacher":
            flash("Access Denied: You do not have permission to access teacher/administrative pages.", "danger")
            return redirect(url_for("student_dashboard"))
        return view(**kwargs)
    return wrapped_view


def student_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please log in as a Student to view your personal dashboard.", "warning")
            return redirect(url_for("login", next=request.url))
        if session.get("role") != "student":
            return redirect(url_for("teacher_dashboard"))
        return view(**kwargs)
    return wrapped_view


# ========================================================
# AUTHENTICATION & ROOT ROUTING
# ========================================================

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") == "teacher":
        return redirect(url_for("teacher_dashboard"))
    return redirect(url_for("student_dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") == "teacher":
        return redirect(url_for("teacher_dashboard"))
    return redirect(url_for("student_dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password", "")

        if not login_id or not password:
            flash("Please enter both username/email and password.", "danger")
            return render_template("login.html", login_id=login_id)

        conn = get_db()
        user = conn.execute(
            "SELECT id, username, email, password_hash, role, student_id FROM users WHERE username = ? OR email = ?",
            (login_id, login_id.lower()),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"] if user["role"] else "student"
            session["student_id"] = user["student_id"]

            role_badge = "Teacher (Admin)" if session["role"] == "teacher" else "Student"
            flash(f"Welcome back, {user['username']}! Signed in as {role_badge}.", "success")

            next_url = request.args.get("next")
            if next_url and ("admin" in next_url or "prediction" in next_url) and session["role"] == "student":
                next_url = None

            if next_url:
                return redirect(next_url)

            if session["role"] == "teacher":
                return redirect(url_for("teacher_dashboard"))
            return redirect(url_for("student_dashboard"))
        else:
            flash("Invalid username/email or password.", "danger")
            return render_template("login.html", login_id=login_id)

    return render_template("login.html")


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

        student_id = None
        if role == "student":
            # Generate a new unique Student ID e.g. STU101
            max_stu = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            student_id = f"STU{100 + max_stu + 1}"

            # Create default student record
            conn.execute(
                """
                INSERT INTO students (student_id, name, department, semester, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, overall_score, predicted_performance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (student_id, username.title(), "Computer Science", "Semester 4", 82.0, 78.0, 80.0, 3.5, 76.0, 80.0, 79.2, "Average", now, now)
            )

            # Create default personal goals
            conn.execute(
                """
                INSERT INTO student_goals (student_id, target_attendance, target_internal_marks, target_assignment_score, target_study_hours, target_overall, updated_at)
                VALUES (?, 90.0, 85.0, 88.0, 5.0, 85.0, ?)
                """,
                (student_id, now)
            )

        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash, role, student_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, email, password_hash, role, student_id, now),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        session["user_id"] = user_id
        session["username"] = username
        session["role"] = role
        session["student_id"] = student_id

        role_label = "Teacher / Administrator" if role == "teacher" else "Student"
        flash(f"Account registered successfully as {role_label}! Welcome.", "success")

        if role == "teacher":
            return redirect(url_for("teacher_dashboard"))
        return redirect(url_for("student_dashboard"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been successfully signed out.", "info")
    return redirect(url_for("login"))


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", active_page="profile")


# ========================================================
# TEACHER / ADMIN ROUTES (FULL ACCESS)
# ========================================================

@app.route("/admin/dashboard")
@teacher_required
def teacher_dashboard():
    conn = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY overall_score DESC").fetchall()
    total_students = len(students)

    avg_overall = round(sum(s["overall_score"] for s in students) / total_students, 1) if total_students else 0
    avg_attendance = round(sum(s["attendance"] for s in students) / total_students, 1) if total_students else 0

    excellent_count = sum(1 for s in students if s["predicted_performance"] == "Excellent")
    average_count = sum(1 for s in students if s["predicted_performance"] == "Average")
    at_risk_count = sum(1 for s in students if s["predicted_performance"] == "At Risk")
    risk_percentage = round((at_risk_count / total_students * 100), 1) if total_students else 0

    stats = {
        "total_students": total_students,
        "avg_overall": avg_overall,
        "avg_attendance": avg_attendance,
        "excellent_count": excellent_count,
        "average_count": average_count,
        "at_risk_count": at_risk_count,
        "risk_percentage": risk_percentage,
    }

    # Attention list: students with attendance < 75 or overall < 65 or status At Risk
    attention_students = [
        dict(s) for s in students if s["attendance"] < 75 or s["overall_score"] < 65 or s["predicted_performance"] == "At Risk"
    ]

    # Chart components
    chart_data = {
        "perf_labels": ["Attendance", "Internal Marks", "Assignments", "Study Hours (x10)", "Previous Sem", "Participation"],
        "perf_values": [
            round(sum(s["attendance"] for s in students) / total_students, 1) if total_students else 0,
            round(sum(s["internal_marks"] for s in students) / total_students, 1) if total_students else 0,
            round(sum(s["assignment_score"] for s in students) / total_students, 1) if total_students else 0,
            round(sum(s["study_hours"] * 10 for s in students) / total_students, 1) if total_students else 0,
            round(sum(s["previous_sem_marks"] for s in students) / total_students, 1) if total_students else 0,
            round(sum(s["participation"] for s in students) / total_students, 1) if total_students else 0,
        ]
    }

    conn.close()
    return render_template(
        "admin_dashboard.html",
        stats=stats,
        attention_students=attention_students,
        chart_data=chart_data,
        active_page="dashboard"
    )


@app.route("/admin/students")
@teacher_required
def admin_students():
    query = request.args.get("q", "").strip()
    selected_dept = request.args.get("dept", "").strip()
    selected_status = request.args.get("status", "").strip()

    conn = get_db()
    sql = "SELECT * FROM students WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR student_id LIKE ? OR department LIKE ?)"
        term = f"%{query}%"
        params.extend([term, term, term])

    if selected_dept:
        sql += " AND department = ?"
        params.append(selected_dept)

    if selected_status:
        sql += " AND predicted_performance = ?"
        params.append(selected_status)

    sql += " ORDER BY id DESC"
    students = conn.execute(sql, params).fetchall()

    departments = [row[0] for row in conn.execute("SELECT DISTINCT department FROM students WHERE department IS NOT NULL").fetchall()]
    conn.close()

    return render_template(
        "admin_students.html",
        students=students,
        departments=departments,
        query=query,
        selected_dept=selected_dept,
        selected_status=selected_status,
        active_page="students"
    )


@app.route("/admin/students/add", methods=["GET", "POST"])
@teacher_required
def admin_add_student():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip().upper()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "Computer Science").strip()
        semester = request.form.get("semester", "Semester 4").strip()

        try:
            attendance = float(request.form.get("attendance", 0))
            internal_marks = float(request.form.get("internal_marks", 0))
            assignment_score = float(request.form.get("assignment_score", 0))
            study_hours = float(request.form.get("study_hours", 0))
            previous_sem_marks = float(request.form.get("previous_sem_marks", 0))
            participation = float(request.form.get("participation", 0))

            overall_score = round((attendance + internal_marks + assignment_score + previous_sem_marks + participation) / 5, 1)

            # Run ML prediction on the new student's data
            payload = {
                "student_name": name,
                "attendance": attendance,
                "internal_marks": internal_marks,
                "assignment_score": assignment_score,
                "study_hours": study_hours,
                "previous_sem_marks": previous_sem_marks,
                "participation": participation,
            }
            feature_row = build_feature_row(payload)
            predicted_performance = str(model.predict(feature_row)[0])

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = get_db()
            conn.execute(
                """
                INSERT INTO students (student_id, name, department, semester, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, overall_score, predicted_performance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (student_id, name, department, semester, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, overall_score, predicted_performance, now, now)
            )

            # Insert default personal goals
            conn.execute(
                """
                INSERT OR IGNORE INTO student_goals (student_id, target_attendance, target_internal_marks, target_assignment_score, target_study_hours, target_overall, updated_at)
                VALUES (?, 90.0, 85.0, 90.0, 5.0, 88.0, ?)
                """,
                (student_id, now)
            )

            conn.commit()
            conn.close()

            flash(f"Student {name} ({student_id}) successfully enrolled with predicted tier: {predicted_performance}!", "success")
            return redirect(url_for("admin_students"))
        except sqlite3.IntegrityError:
            flash(f"Error: Student ID '{student_id}' is already registered in the directory.", "danger")
        except Exception as e:
            flash(f"Failed to add student: {e}", "danger")

    return render_template("admin_add_student.html", active_page="students")


@app.route("/admin/students/edit/<int:student_pk>", methods=["GET", "POST"])
@teacher_required
def admin_edit_student(student_pk):
    conn = get_db()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (student_pk,)).fetchone()

    if not student:
        conn.close()
        flash("Student record not found.", "warning")
        return redirect(url_for("admin_students"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "Computer Science").strip()
        semester = request.form.get("semester", "Semester 4").strip()

        try:
            attendance = float(request.form.get("attendance", 0))
            internal_marks = float(request.form.get("internal_marks", 0))
            assignment_score = float(request.form.get("assignment_score", 0))
            study_hours = float(request.form.get("study_hours", 0))
            previous_sem_marks = float(request.form.get("previous_sem_marks", 0))
            participation = float(request.form.get("participation", 0))

            overall_score = round((attendance + internal_marks + assignment_score + previous_sem_marks + participation) / 5, 1)

            payload = {
                "student_name": name,
                "attendance": attendance,
                "internal_marks": internal_marks,
                "assignment_score": assignment_score,
                "study_hours": study_hours,
                "previous_sem_marks": previous_sem_marks,
                "participation": participation,
            }
            feature_row = build_feature_row(payload)
            predicted_performance = str(model.predict(feature_row)[0])

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                UPDATE students
                SET name = ?, department = ?, semester = ?, attendance = ?, internal_marks = ?, assignment_score = ?,
                    study_hours = ?, previous_sem_marks = ?, participation = ?, overall_score = ?,
                    predicted_performance = ?, updated_at = ?
                WHERE id = ?
                """,
                (name, department, semester, attendance, internal_marks, assignment_score, study_hours, previous_sem_marks, participation, overall_score, predicted_performance, now, student_pk)
            )
            conn.commit()
            conn.close()

            flash(f"Student details updated successfully for {name}!", "success")
            return redirect(url_for("admin_students"))
        except Exception as e:
            flash(f"Failed to update student: {e}", "danger")

    conn.close()
    return render_template("admin_edit_student.html", student=student, active_page="students")


@app.route("/admin/students/delete/<int:student_pk>", methods=["POST"])
@teacher_required
def admin_delete_student(student_pk):
    conn = get_db()
    student = conn.execute("SELECT student_id, name FROM students WHERE id = ?", (student_pk,)).fetchone()
    if student:
        conn.execute("DELETE FROM student_goals WHERE student_id = ?", (student["student_id"],))
        conn.execute("DELETE FROM students WHERE id = ?", (student_pk,))
        conn.commit()
        flash(f"Student record for {student['name']} deleted successfully.", "info")
    else:
        flash("Student record not found.", "warning")
    conn.close()
    return redirect(url_for("admin_students"))


@app.route("/admin/records")
@teacher_required
def admin_records():
    conn = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY id ASC").fetchall()
    conn.close()
    return render_template("admin_records.html", students=students, active_page="records")


@app.route("/admin/performance")
@teacher_required
def admin_performance():
    conn = get_db()
    all_students = conn.execute("SELECT * FROM students ORDER BY name ASC").fetchall()
    total = len(all_students)

    class_avg = {
        "attendance": round(sum(s["attendance"] for s in all_students) / total, 1) if total else 0,
        "internal_marks": round(sum(s["internal_marks"] for s in all_students) / total, 1) if total else 0,
        "assignment_score": round(sum(s["assignment_score"] for s in all_students) / total, 1) if total else 0,
        "study_hours": round(sum(s["study_hours"] for s in all_students) / total, 1) if total else 0,
        "previous_sem_marks": round(sum(s["previous_sem_marks"] for s in all_students) / total, 1) if total else 0,
        "participation": round(sum(s["participation"] for s in all_students) / total, 1) if total else 0,
        "overall_score": round(sum(s["overall_score"] for s in all_students) / total, 1) if total else 0,
    }

    student_a_id = request.args.get("student_a") or (all_students[0]["student_id"] if all_students else None)
    student_b_id = request.args.get("student_b") or (all_students[1]["student_id"] if len(all_students) > 1 else student_a_id)

    student_a = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_a_id,)).fetchone()
    student_b = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_b_id,)).fetchone()
    conn.close()

    return render_template(
        "admin_performance.html",
        all_students=all_students,
        student_a=student_a,
        student_b=student_b,
        class_avg=class_avg,
        active_page="performance"
    )


@app.route("/prediction")
@teacher_required
def prediction():
    return render_template("prediction.html", active_page="prediction")


@app.route("/predict", methods=["POST"])
@teacher_required
def predict():
    try:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
        else:
            payload = request.form.to_dict()

        payload = normalize_input(payload)
        prepared = build_feature_row(payload)
        prediction_val = str(model.predict(prepared)[0])
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
                    prediction_val,
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
                "prediction": prediction_val,
                "student_name": payload["student_name"],
                "id": prediction_id,
            })

        return render_template(
            "prediction.html",
            prediction=prediction_val,
            form_data=payload,
            prediction_id=prediction_id,
            active_page="prediction"
        )
    except ValueError as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 400
        return render_template("prediction.html", error=str(exc), active_page="prediction")
    except Exception as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        return render_template("prediction.html", error=f"Unexpected error: {exc}", active_page="prediction")


@app.route("/admin/scoreboard")
@teacher_required
def admin_scoreboard():
    conn = get_db()
    ranked_students = conn.execute("SELECT * FROM students ORDER BY overall_score DESC").fetchall()
    conn.close()
    return render_template("admin_scoreboard.html", ranked_students=ranked_students, active_page="scoreboard")


@app.route("/admin/visualizations")
@teacher_required
def admin_visualizations():
    conn = get_db()
    students = conn.execute("SELECT * FROM students").fetchall()
    total = len(students)

    scatter_data = [{"x": s["attendance"], "y": s["overall_score"]} for s in students]

    study_chart = {
        "labels": ["0-2h (Minimal)", "2-4h (Moderate)", "4-6h (Dedicated)", "6h+ (Intensive)"],
        "scores": [
            round(sum(s["overall_score"] for s in students if s["study_hours"] < 2) / max(1, sum(1 for s in students if s["study_hours"] < 2)), 1),
            round(sum(s["overall_score"] for s in students if 2 <= s["study_hours"] < 4) / max(1, sum(1 for s in students if 2 <= s["study_hours"] < 4)), 1),
            round(sum(s["overall_score"] for s in students if 4 <= s["study_hours"] < 6) / max(1, sum(1 for s in students if 4 <= s["study_hours"] < 6)), 1),
            round(sum(s["overall_score"] for s in students if s["study_hours"] >= 6) / max(1, sum(1 for s in students if s["study_hours"] >= 6)), 1),
        ]
    }

    comp_averages = {
        "attendance": round(sum(s["attendance"] for s in students) / total, 1) if total else 0,
        "internal_marks": round(sum(s["internal_marks"] for s in students) / total, 1) if total else 0,
        "assignment_score": round(sum(s["assignment_score"] for s in students) / total, 1) if total else 0,
        "previous_sem_marks": round(sum(s["previous_sem_marks"] for s in students) / total, 1) if total else 0,
        "participation": round(sum(s["participation"] for s in students) / total, 1) if total else 0,
        "overall_score": round(sum(s["overall_score"] for s in students) / total, 1) if total else 0,
    }

    tier_counts = {
        "excellent": sum(1 for s in students if s["predicted_performance"] == "Excellent"),
        "average": sum(1 for s in students if s["predicted_performance"] == "Average"),
        "at_risk": sum(1 for s in students if s["predicted_performance"] == "At Risk"),
    }

    conn.close()
    return render_template(
        "admin_visualizations.html",
        scatter_data=scatter_data,
        study_chart=study_chart,
        comp_averages=comp_averages,
        tier_counts=tier_counts,
        active_page="visualizations"
    )


@app.route("/admin/reports")
@teacher_required
def admin_reports():
    conn = get_db()
    records = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("admin_reports.html", records=records, active_page="reports")


@app.route("/admin/model-performance")
@teacher_required
def admin_model_performance():
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)
    return render_template("admin_model_performance.html", model_info=model_info, active_page="model_performance")


# ========================================================
# STUDENT ROUTES (PERSONAL PERFORMANCE VIEW ONLY)
# ========================================================

def get_current_student(conn, user_id):
    user = conn.execute("SELECT student_id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    student = None
    if user and user["student_id"]:
        student = conn.execute("SELECT * FROM students WHERE student_id = ?", (user["student_id"],)).fetchone()
    if not student and user:
        student = conn.execute("SELECT * FROM students WHERE name LIKE ?", (f"%{user['username']}%",)).fetchone()
    if not student:
        student = conn.execute("SELECT * FROM students ORDER BY id ASC LIMIT 1").fetchone()
    return student


def generate_student_insights(student, class_avg):
    insights = []
    recommendations = []

    # 1. Attendance insight
    if student["attendance"] >= 85:
        insights.append({
            "category": "positive",
            "headline": "Outstanding Attendance Record",
            "detail": f"Your attendance rate of {student['attendance']}% is above the class average ({class_avg['attendance']}%) and well exceeds university compliance standards."
        })
    elif student["attendance"] >= 75:
        insights.append({
            "category": "info",
            "headline": "Compliant Lecture Attendance",
            "detail": f"Your attendance is {student['attendance']}%, meeting the 75% examination eligibility criteria."
        })
    else:
        insights.append({
            "category": "warning",
            "headline": "Attendance Needs Priority",
            "detail": f"Current attendance of {student['attendance']}% is below the mandatory 75% threshold. Consistent lecture presence is required."
        })
        recommendations.append({
            "action": "Prioritize Regular Attendance",
            "reason": "Avoid exam debarment by maintaining a minimum 80% weekly attendance track."
        })

    # 2. Assignment vs Internal comparison
    if student["assignment_score"] < student["internal_marks"]:
        insights.append({
            "category": "warning",
            "headline": "Assignment Score Lags Behind Internals",
            "detail": f"Your continuous assignment performance ({student['assignment_score']}%) is lower than your exam internal score ({student['internal_marks']}%)."
        })
        recommendations.append({
            "action": "Allocate Dedicated Assignment Submission Hours",
            "reason": "Starting practical work 3-4 days ahead of deadlines will directly lift assignment averages."
        })
    else:
        insights.append({
            "category": "positive",
            "headline": "Strong Continuous Assignment Submissions",
            "detail": f"Your assignment score of {student['assignment_score']}% demonstrates disciplined coursework completion."
        })

    # 3. Semester progression comparison
    if student["overall_score"] >= student["previous_sem_marks"]:
        diff = round(student["overall_score"] - student["previous_sem_marks"], 1)
        insights.append({
            "category": "positive",
            "headline": f"Upward Academic Trend (+{diff}%)",
            "detail": f"Your current semester standing of {student['overall_score']}% demonstrates positive growth over last semester ({student['previous_sem_marks']}%)."
        })
    else:
        diff = round(student["previous_sem_marks"] - student["overall_score"], 1)
        insights.append({
            "category": "info",
            "headline": "Semester Performance Reflection",
            "detail": f"Current score ({student['overall_score']}%) is slightly below your prior semester mark ({student['previous_sem_marks']}%)."
        })
        recommendations.append({
            "action": "Consolidate Key Subject Notes",
            "reason": "Weekly revision sessions help stabilize mid-term assessment scores."
        })

    # 4. Study hours consistency
    if student["study_hours"] >= 4.0:
        insights.append({
            "category": "positive",
            "headline": "Consistent Self-Study Habit",
            "detail": f"Maintaining {student['study_hours']} hours/day of independent study creates strong foundation for upcoming evaluations."
        })
    elif student["study_hours"] >= 2.5:
        insights.append({
            "category": "info",
            "headline": "Moderate Self-Study Schedule",
            "detail": f"You average {student['study_hours']} hours of independent daily study."
        })
        recommendations.append({
            "action": "Increment Daily Study by 30-45 Minutes",
            "reason": "Targeting challenging topics during peak focus hours improves retention."
        })
    else:
        insights.append({
            "category": "warning",
            "headline": "Daily Study Hours Below Recommended Target",
            "detail": f"Current daily self-study ({student['study_hours']} hrs/day) is insufficient for comprehensive course coverage."
        })
        recommendations.append({
            "action": "Establish Fixed Daily Study Window",
            "reason": "Aim for at least 3 hours daily split between revision and problem solving."
        })

    if not recommendations:
        recommendations.append({
            "action": "Maintain Current Excellence",
            "reason": "Continue your disciplined study patterns and active seminar participation."
        })

    return insights, recommendations


@app.route("/student/dashboard")
@student_required
def student_dashboard():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)

    all_students = conn.execute("SELECT * FROM students").fetchall()
    total = len(all_students)
    class_avg = {
        "attendance": round(sum(s["attendance"] for s in all_students) / total, 1) if total else 0,
        "overall_score": round(sum(s["overall_score"] for s in all_students) / total, 1) if total else 0,
    }

    insights, _ = generate_student_insights(student, class_avg)
    top_insights = [
        {"type": ins["category"], "title": ins["headline"], "message": ins["detail"]}
        for ins in insights[:3]
    ]

    conn.close()
    return render_template(
        "student_dashboard.html",
        student=student,
        top_insights=top_insights,
        active_page="student_dashboard"
    )


@app.route("/student/performance")
@student_required
def student_performance():
    return redirect(url_for("student_dashboard"))


@app.route("/student/scoreboard")
@student_required
def student_scoreboard():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)

    students = conn.execute("SELECT student_id, name, department, overall_score FROM students ORDER BY overall_score DESC").fetchall()
    total_students = len(students)

    rank = 1
    leaderboard = []
    for idx, s in enumerate(students, 1):
        if s["student_id"] == student["student_id"]:
            rank = idx

        # Mask peer names for student privacy (e.g. "J*** M***")
        parts = s["name"].split()
        masked = " ".join([p[0] + "***" for p in parts]) if parts else "Student"

        leaderboard.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "masked_name": masked,
            "department": s["department"],
            "overall_score": s["overall_score"],
        })

    class_avg = {
        "overall_score": round(sum(s["overall_score"] for s in students) / total_students, 1) if total_students else 0,
        "attendance": round(sum(s["attendance"] for s in conn.execute("SELECT attendance FROM students").fetchall()) / total_students, 1) if total_students else 0,
        "internal_marks": round(sum(s["internal_marks"] for s in conn.execute("SELECT internal_marks FROM students").fetchall()) / total_students, 1) if total_students else 0,
        "assignment_score": round(sum(s["assignment_score"] for s in conn.execute("SELECT assignment_score FROM students").fetchall()) / total_students, 1) if total_students else 0,
        "previous_sem_marks": round(sum(s["previous_sem_marks"] for s in conn.execute("SELECT previous_sem_marks FROM students").fetchall()) / total_students, 1) if total_students else 0,
    }

    conn.close()
    return render_template(
        "student_scoreboard.html",
        student=student,
        rank=rank,
        total_students=total_students,
        class_avg=class_avg,
        leaderboard=leaderboard,
        active_page="student_scoreboard"
    )


@app.route("/student/visualizations")
@student_required
def student_visualizations():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)
    conn.close()
    return render_template("student_visualizations.html", student=student, active_page="student_visualizations")


@app.route("/student/insights")
@student_required
def student_insights():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)

    all_students = conn.execute("SELECT * FROM students").fetchall()
    total = len(all_students)
    class_avg = {
        "attendance": round(sum(s["attendance"] for s in all_students) / total, 1) if total else 0,
        "overall_score": round(sum(s["overall_score"] for s in all_students) / total, 1) if total else 0,
    }

    insights, recommendations = generate_student_insights(student, class_avg)
    conn.close()
    return render_template(
        "student_insights.html",
        student=student,
        insights=insights,
        recommendations=recommendations,
        active_page="student_insights"
    )


@app.route("/student/goals", methods=["GET", "POST"])
@student_required
def student_goals():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)

    if request.method == "POST":
        target_att = float(request.form.get("target_attendance", 90.0))
        target_int = float(request.form.get("target_internal_marks", 85.0))
        target_asg = float(request.form.get("target_assignment_score", 90.0))
        target_hrs = float(request.form.get("target_study_hours", 5.0))
        target_ovr = float(request.form.get("target_overall", 88.0))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            """
            INSERT INTO student_goals (student_id, target_attendance, target_internal_marks, target_assignment_score, target_study_hours, target_overall, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                target_attendance = excluded.target_attendance,
                target_internal_marks = excluded.target_internal_marks,
                target_assignment_score = excluded.target_assignment_score,
                target_study_hours = excluded.target_study_hours,
                target_overall = excluded.target_overall,
                updated_at = excluded.updated_at
            """,
            (student["student_id"], target_att, target_int, target_asg, target_hrs, target_ovr, now)
        )
        conn.commit()
        flash("Personal academic targets updated successfully! Keep striving toward your goals.", "success")

    goals = conn.execute("SELECT * FROM student_goals WHERE student_id = ?", (student["student_id"],)).fetchone()
    if not goals:
        goals = {
            "target_attendance": 90.0,
            "target_internal_marks": 85.0,
            "target_assignment_score": 90.0,
            "target_study_hours": 5.0,
            "target_overall": 88.0,
        }

    conn.close()
    return render_template("student_goals.html", student=student, goals=goals, active_page="student_goals")


@app.route("/student/report")
@student_required
def student_report():
    conn = get_db()
    user_id = session.get("user_id")
    student = get_current_student(conn, user_id)
    conn.close()

    generated_date = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    return render_template("student_report.html", student=student, generated_date=generated_date, active_page="student_report")


# ========================================================
# REPORT & API ENDPOINTS
# ========================================================

@app.route("/report/<int:prediction_id>")
@login_required
def report(prediction_id):
    conn = get_db()
    record = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()

    if not record:
        flash("Record not found.", "warning")
        return redirect(url_for("dashboard"))

    if session.get("role") != "teacher" and record["user_id"] != session.get("user_id"):
        flash("Access Denied: You can only view your own evaluation reports.", "danger")
        return redirect(url_for("student_dashboard"))

    overall_score = round(
        (record["attendance"] + record["internal_marks"] + record["assignment_score"] + record["previous_sem_marks"] + record["participation"]) / 5,
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
        summary_text = "Student demonstrates exemplary academic engagement across all assessments."
        recommendations = [
            "Encourage participation in advanced subject seminars, honors projects, or research papers.",
            "Consider student for peer mentoring and study group leadership roles.",
            "Maintain current study routine and exam preparation habits.",
        ]
    elif pred == "Average":
        summary_text = "Student shows satisfactory comprehension with clear potential for honors tier."
        recommendations = [
            "Increase dedicated study hours by 1-2 hours daily focusing on high-weightage topics.",
            "Work on assignment timeliness and internal evaluation mock tests.",
            "Engage more actively in classroom discussions and doubt-clearing sessions.",
        ]
    else:
        summary_text = "Student is in the academic risk zone requiring proactive institutional intervention."
        recommendations = [
            "Schedule a one-on-one session with academic counselor / faculty advisor.",
            "Implement a mandatory minimum attendance target of 80% with weekly tracking.",
            "Enroll in remedial workshops for internal assessments and fundamental concepts.",
            "Set up structured daily study schedule under mentor guidance.",
        ]

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
    return jsonify({"total_predictions": total, "excellent": excellent, "average": average, "at_risk": at_risk})


@app.route("/api/history")
@login_required
def api_history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
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
    rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
