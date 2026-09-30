import app

def run_tests():
    client = app.app.test_client()
    print("========================================")
    print("STARTING ROLE-BASED ACCESS CONTROL TESTS")
    print("========================================")

    # TEST 1: Unauthenticated access redirects to /login
    resp = client.get("/")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"], f"Failed unauth / redirect: {resp.status_code}"
    print("[PASS] TEST 1: Unauthenticated request to / redirects to /login")

    resp = client.get("/prediction")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"], f"Failed unauth /prediction: {resp.status_code}"
    print("[PASS] TEST 1b: Unauthenticated request to /prediction redirects to /login")

    # TEST 2: Teacher login
    with client.session_transaction() as sess:
        sess.clear()

    resp = client.post("/login", data={"login_id": "teacher_admin", "password": "teacher123"}, follow_redirects=True)
    assert resp.status_code == 200, f"Teacher login failed: {resp.status_code}"
    assert b"Teacher / Admin Dashboard" in resp.data or b"Welcome back, teacher_admin" in resp.data, "Teacher dashboard not shown"
    print("[PASS] TEST 2: Teacher login succeeds and redirects to Teacher Dashboard")

    # TEST 3: Teacher access to all Teacher features
    teacher_routes = [
        ("/admin/dashboard", b"Teacher / Admin Dashboard"),
        ("/admin/students", b"Student Directory"),
        ("/admin/students/add", b"Create Student Record"),
        ("/admin/records", b"Master Academic Records"),
        ("/admin/performance", b"Comparative Analytics"),
        ("/prediction", b"AI Student Performance Predictor"),
        ("/admin/scoreboard", b"Class Academic Scoreboard"),
        ("/admin/visualizations", b"Interactive Academic Visualizations"),
        ("/admin/reports", b"Generated Evaluation Reports"),
        ("/admin/model-performance", b"ML Model Evaluation Metrics"),
    ]

    for route, expected_text in teacher_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Teacher route {route} failed with status {resp.status_code}"
        assert expected_text in resp.data, f"Teacher route {route} missing expected text {expected_text}"
        print(f"  [PASS] Teacher access verified: {route}")

    # TEST 4: Teacher prediction execution
    pred_payload = {
        "student_name": "Test Performer",
        "attendance": 92.0,
        "internal_marks": 88.0,
        "assignment_score": 90.0,
        "study_hours": 5.0,
        "previous_sem_marks": 85.0,
        "participation": 88.0
    }
    resp = client.post("/predict", json=pred_payload)
    assert resp.status_code == 200, f"Predict API failed: {resp.status_code}"
    pred_data = resp.get_json()
    assert "prediction" in pred_data, "Prediction missing in response"
    print(f"[PASS] TEST 4: Teacher prediction works. Result: {pred_data['prediction']}")

    # TEST 5: Logout Teacher
    client.get("/logout")
    print("[PASS] TEST 5: Teacher logged out")

    # TEST 6: Student login
    resp = client.post("/login", data={"login_id": "student_user", "password": "student123"}, follow_redirects=True)
    assert resp.status_code == 200, f"Student login failed: {resp.status_code}"
    assert b"Student Academic Dashboard" in resp.data or b"Welcome back, Alex Morgan" in resp.data, "Student dashboard not shown"
    print("[PASS] TEST 6: Student login succeeds and lands on Student Dashboard")

    # TEST 7: Student accesses student-allowed pages
    student_routes = [
        ("/student/dashboard", b"Student Academic Dashboard"),
        ("/student/scoreboard", b"My Class Position"),
        ("/student/visualizations", b"Visual Analytics"),
        ("/student/insights", b"Rule-Based Academic Insights"),
        ("/student/goals", b"Academic Targets"),
        ("/student/report", b"Student Academic Summary"),
    ]

    for route, expected_text in student_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Student route {route} failed with status {resp.status_code}"
        assert expected_text in resp.data, f"Student route {route} missing expected text {expected_text}"
        # Ensure Prediction is NEVER mentioned in student sidebar or page
        assert b'<span class="menu-label">AI Prediction</span>' not in resp.data, f"Prediction menu leaked in {route}"
        print(f"  [PASS] Student access verified: {route}")

    # TEST 8: CRITICAL - Student attempts to access /prediction by URL -> MUST BE DENIED!
    resp = client.get("/prediction", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Access Denied" in resp.data, "Access Denied alert missing when student accesses /prediction"
    assert b"Student Academic Dashboard" in resp.data, "Student was not redirected to Student Dashboard"
    print("[PASS] TEST 8: Student accessing /prediction is BLOCKED and redirected to Student Dashboard with Access Denied message!")

    # TEST 9: Student attempts to POST to /predict -> MUST BE DENIED!
    resp = client.post("/predict", json=pred_payload, follow_redirects=True)
    assert b"Access Denied" in resp.data, "Student was able to submit /predict!"
    print("[PASS] TEST 9: Student attempting /predict POST is BLOCKED with Access Denied!")

    # TEST 10: Student attempts to access teacher admin routes -> MUST BE DENIED!
    forbidden_teacher_routes = [
        ("/admin/dashboard", "Teacher Dashboard"),
        ("/admin/students", "Students"),
        ("/admin/students/add", "Add Student"),
        ("/admin/records", "Records"),
        ("/admin/performance", "Performance"),
        ("/admin/scoreboard", "Scoreboard"),
        ("/admin/visualizations", "Visualizations"),
        ("/admin/reports", "Reports"),
        ("/admin/model-performance", "Model Performance")
    ]

    for route, label in forbidden_teacher_routes:
        resp = client.get(route, follow_redirects=True)
        assert b"Access Denied" in resp.data, f"Student was able to access {route} without Access Denied!"
        assert b"Student Academic Dashboard" in resp.data, f"Student was not redirected to /student/dashboard on {route}"
        print(f"  [PASS] Student accessing {route} ({label}) successfully BLOCKED and redirected!")

    # TEST 11: Student updates personal goals
    goal_data = {
        "target_attendance": 92.0,
        "target_internal_marks": 86.0,
        "target_assignment_score": 90.0,
        "target_study_hours": 5.5,
        "target_overall": 90.0
    }
    resp = client.post("/student/goals", data=goal_data, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Personal academic targets updated successfully" in resp.data, "Goal update confirmation missing"
    print("[PASS] TEST 11: Student can set and update personal goals!")

    print("========================================")
    print("ALL 11 ROLE-BASED TESTS PASSED PERFECTLY!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
