import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "static" / "plots"
DATA_PATH = DATA_DIR / "student_performance.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
FEATURES_PATH = MODELS_DIR / "features.pkl"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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


def generate_dataset():
    rng = np.random.default_rng(42)
    records = []
    names = [f"Student_{i:04d}" for i in range(1, 1201)]

    for index, name in enumerate(names):
        attendance = int(rng.integers(35, 101))
        internal_marks = int(rng.integers(25, 101))
        assignment_score = int(rng.integers(30, 101))
        study_hours = round(float(rng.uniform(0.5, 12.0)), 1)
        previous_sem_marks = int(rng.integers(30, 101))
        participation = int(rng.integers(0, 101))

        composite_score = (
            0.28 * attendance
            + 0.26 * internal_marks
            + 0.20 * assignment_score
            + 5.5 * study_hours
            + 0.24 * previous_sem_marks
            + 0.26 * participation
            + rng.normal(0, 6)
        )
        composite_score = float(np.clip(composite_score, 0, 100))

        if composite_score >= 75:
            performance_category = "Excellent"
        elif composite_score >= 60:
            performance_category = "Average"
        else:
            performance_category = "At Risk"

        records.append(
            {
                "student_name": name,
                "attendance": attendance,
                "internal_marks": internal_marks,
                "assignment_score": assignment_score,
                "study_hours": study_hours,
                "previous_sem_marks": previous_sem_marks,
                "participation": participation,
                "performance_category": performance_category,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)
    return df


def preprocess_dataset(df):
    required_columns = [
        "student_name",
        "attendance",
        "internal_marks",
        "assignment_score",
        "study_hours",
        "previous_sem_marks",
        "participation",
        "performance_category",
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop_duplicates().copy()
    for column in [
        "attendance",
        "internal_marks",
        "assignment_score",
        "study_hours",
        "previous_sem_marks",
        "participation",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["student_name"] = df["student_name"].fillna("Unknown Student")
    df["performance_category"] = df["performance_category"].fillna("Average")
    df = df.fillna(df.median(numeric_only=True))

    df["overall_score"] = (
        df["attendance"]
        + df["internal_marks"]
        + df["assignment_score"]
        + df["previous_sem_marks"]
        + df["participation"]
    ) / 5

    df["study_performance"] = (
        (df["attendance"] + df["internal_marks"] + df["assignment_score"] + df["previous_sem_marks"]) / 4
        + (df["study_hours"] * 5)
    )

    return df


def create_visualizations(df):
    category_counts = df["performance_category"].value_counts().reindex(["Excellent", "Average", "At Risk"], fill_value=0)
    plt.figure(figsize=(8, 5))
    category_counts.plot(kind="bar", color=["#22c55e", "#f59e0b", "#ef4444"])
    plt.title("Performance Category Distribution")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_distribution.png")
    plt.close()

    corr = df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="performance_category", y="attendance", hue="performance_category", dodge=False, palette=["#22c55e", "#f59e0b", "#ef4444"], legend=False)
    plt.title("Attendance vs Performance")
    plt.xlabel("Performance Category")
    plt.ylabel("Attendance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attendance_vs_performance.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="performance_category", y="study_hours", hue="performance_category", dodge=False, palette=["#22c55e", "#f59e0b", "#ef4444"], legend=False)
    plt.title("Study Hours vs Performance")
    plt.xlabel("Performance Category")
    plt.ylabel("Study Hours")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "study_hours_vs_performance.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="performance_category", y="internal_marks", hue="performance_category", dodge=False, palette=["#22c55e", "#f59e0b", "#ef4444"], legend=False)
    plt.title("Internal Marks vs Performance")
    plt.xlabel("Performance Category")
    plt.ylabel("Internal Marks")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "internal_marks_vs_performance.png")
    plt.close()


def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df["performance_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        )
        trained_models[name] = model

        print(f"\n{name}")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

    comparison = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    print("\nModel Comparison")
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    classes = ["At Risk", "Average", "Excellent"]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    metrics = {
        "best_model_name": best_model_name,
        "accuracy": round(float(comparison.iloc[0]["Accuracy"]), 4),
        "precision": round(float(comparison.iloc[0]["Precision"]), 4),
        "recall": round(float(comparison.iloc[0]["Recall"]), 4),
        "f1_score": round(float(comparison.iloc[0]["F1 Score"]), 4),
        "feature_names": FEATURE_COLUMNS,
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return best_model_name, comparison


def main():
    if not DATA_PATH.exists():
        df = generate_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    df = preprocess_dataset(df)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)

    create_visualizations(df)
    best_model_name, comparison = train_and_evaluate(df)
    print(f"\nBest model selected: {best_model_name}")
    print("Model and metadata were saved successfully.")


if __name__ == "__main__":
    main()

