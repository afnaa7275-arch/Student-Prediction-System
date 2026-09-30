# AI-Based Role-Based Student Performance Management & Prediction System

A production-style, role-based academic performance management and AI prediction application built with Python, Flask, Scikit-Learn, SQLite, and Chart.js.

---

## 👥 User Roles & Access Control

The application enforces strict **Role-Based Access Control (RBAC)** across all routes and views:

### 1. 👩‍🏫 Teacher / Admin Role (Full Access)
- **Dashboard:** Class-level metrics, total students, average performance, average attendance, at-risk students list, class distribution charts.
- **Students Management:** Add new students, edit student profiles/marks, delete records, search and filter by department or risk status.
- **Academic Records:** View master table of all student attendance, internal marks, assignments, study hours, previous semester marks, and overall scores.
- **Performance Analysis:** Comparative analysis tool comparing any two students side-by-side with multidimensional radar charts and class benchmarks.
- **AI Prediction:** Exclusive Machine Learning prediction interface using the trained model (`best_model.pkl`) to classify performance tier (`Excellent`, `Average`, or `At Risk`).
- **Scoreboard / Leaderboard:** Class ranking podium (1st, 2nd, 3rd) and full ranked table.
- **Data Visualizations:** Scatter plots (attendance vs overall score), study hours correlation, component averages, and tier breakdown.
- **Reports:** View, generate, and download printable official PDF evaluation reports.
- **Model Performance:** Inspection of trained model metrics (Accuracy: 96.67%, Precision: 96.47%, Recall: 96.67%, F1-Score: 96.50%, active features).
- **Profile / Account:** Teacher account details.

### 2. 👨‍🎓 Student Role (Personal Performance View Only)
- **Personalized Dashboard:** Welcome banner, student identity (ID, course, semester), 7 performance summary cards, academic breakdown chart, and key observations.
- **My Scoreboard:** Personal class rank, score vs class average benchmark, and privacy-preserving class leaderboard (peer names anonymized).
- **My Visualizations:** Marks distribution, semester progression trend, attendance donut, and indicator completion bars.
- **Performance Insights:** Rule-based academic insights derived directly from data (attendance vs policy, assignment vs internals, study hours) with actionable recommendations.
- **My Goals:** Track target attendance, internal marks, assignments, study hours, and overall score using live visual progress bars, with personal goal setting.
- **Download My Report:** Confidential printable/PDF academic summary report containing ONLY the student's personal records.
- **🔒 STRICT SECURITY:** Students have **ZERO ACCESS** to AI Prediction pages (`/prediction` and `/predict`) or administrative routes. URL tampering is intercepted by backend decorators and redirected to the Student Dashboard with an "Access Denied" message.

---

## 🔑 Quick Demo Login Credentials

| Role | Username / Login ID | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Teacher (Admin)** | `teacher_admin` | `teacher123` | Full Access & ML Prediction |
| **Student (User)** | `student_user` | `student123` | Personal Student View Only |

---

## 🚀 How to Run the Application in VS Code

### Option 1: Double-Click or Run `run.bat`
```cmd
run.bat
```

### Option 2: Terminal / PowerShell
```powershell
& "C:\Users\acer\AppData\Local\Programs\Python\Python311\python.exe" app.py
```
*(Or simply `python app.py` if your Python 3.11 PATH is active in terminal)*

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Automated Testing

To verify all 11 role-based access control rules, prediction security, and routing:
```powershell
& "C:\Users\acer\AppData\Local\Programs\Python\Python311\python.exe" test_roles.py
```

All 11 tests will execute and confirm:
- Unauthenticated requests redirect to `/login`
- Teachers can access all administrative pages and run predictions
- Students can access their personal dashboard and views
- Students attempting to visit `/prediction`, `/predict`, or `/admin/*` are automatically blocked with "Access Denied"

---

## 📂 Project Structure

```
student_performance_prediction_system/
├── app.py                     # Main Flask application with RBAC & routes
├── run.bat                    # One-click Windows launch script
├── requirements.txt           # Python dependencies
├── student_predictions.db     # SQLite database (users, students, predictions, student_goals)
├── train_model.py             # ML model training script
├── test_roles.py              # Automated RBAC & security test suite
├── models/
│   ├── best_model.pkl         # Trained Logistic Regression classifier (96.67% accuracy)
│   ├── features.pkl           # Pickled feature names
│   └── model_info.json        # Evaluation metrics (accuracy, precision, recall, f1)
├── static/
│   ├── style.css              # Custom responsive stylesheet (sidebar, dark/light theme, charts)
│   └── theme.js               # Dark/Light mode theme switcher
├── templates/
│   ├── base.html              # Core layout with role-aware sidebar & topbar
│   ├── login.html             # Modern login page with demo credentials
│   ├── signup.html            # Registration with role selection (Teacher / Student)
│   ├── profile.html           # User profile & account privileges
│   ├── report.html            # Printable PDF evaluation report
│   ├── result.html            # Prediction result component
│   │
│   ├── admin_dashboard.html   # Teacher overview dashboard
│   ├── admin_students.html    # Student management directory (CRUD, search, filter)
│   ├── admin_add_student.html # Form to add new student with marks
│   ├── admin_edit_student.html# Form to update student information & marks
│   ├── admin_records.html     # Master student academic records
│   ├── admin_performance.html # Side-by-side student comparison & radar chart
│   ├── prediction.html        # ML Prediction form (Teacher Exclusive)
│   ├── admin_scoreboard.html  # Class leaderboard & podium
│   ├── admin_visualizations.html # Class visualizations (scatter, trends, distributions)
│   ├── admin_reports.html     # Reports management list
│   ├── admin_model_performance.html # Model verification & evaluation metrics
│   │
│   ├── student_dashboard.html # Personalized student dashboard
│   ├── student_scoreboard.html# My Scoreboard & privacy-protected leaderboard
│   ├── student_visualizations.html # My Visualizations & analytics
│   ├── student_insights.html  # Rule-based academic insights & suggestions
│   ├── student_goals.html     # Target setting with progress bars
│   └── student_report.html    # Downloadable personal student report
```
