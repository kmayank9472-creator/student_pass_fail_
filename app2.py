import io
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Pass/Fail Predictor")
st.write("Study hours, attendance aur midterm score se student ka result predict karein.")

FEATURES = [
    "study_hours_per_week",
    "attendance_pct",
    "midterm_score"
]
TARGET = "pass"
CSV_PATH = Path(__file__).parent / "Pass-Fail Data.csv"


@st.cache_data
def load_uploaded_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


uploaded_file = st.sidebar.file_uploader(
    "Pass-Fail CSV upload karein",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = load_uploaded_csv(uploaded_file.getvalue())
    else:
        df = pd.read_csv(CSV_PATH)
except Exception:
    st.error("CSV file nahi mili. CSV upload karein ya app.py ke folder mein rakhein.")
    st.stop()

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "CSV mein required columns nahi hain: "
        + ", ".join(missing_columns)
    )
    st.caption(
        "Required columns: study_hours_per_week, attendance_pct, "
        "midterm_score, pass"
    )
    st.stop()

try:
    df = df[required_columns].apply(
        pd.to_numeric,
        errors="raise"
    ).dropna()
except (ValueError, TypeError):
    st.error("Required columns mein numeric values honi chahiye.")
    st.stop()

if not set(df[TARGET].unique()).issubset({0, 1}):
    st.error("'pass' column mein 0 = Fail aur 1 = Pass hona chahiye.")
    st.stop()

if df[TARGET].nunique() != 2 or df[TARGET].value_counts().min() < 2:
    st.error("CSV mein Pass aur Fail dono ke kam se kam 2 records chahiye.")
    st.stop()

st.subheader("Dataset")
st.write(f"Total usable records: {len(df)}")
st.dataframe(df.head(), use_container_width=True)

X = df[FEATURES]
y = df[TARGET].astype(int)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])

    model.fit(X_train, y_train)

except ValueError as error:
    st.error(f"Model train nahi ho saka: {error}")
    st.stop()

predictions = model.predict(X_test)

st.subheader("Model evaluation")
st.metric(
    "Test accuracy",
    f"{accuracy_score(y_test, predictions):.1%}"
)

st.write("Confusion matrix")
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual Fail", "Actual Pass"],
        columns=["Predicted Fail", "Predicted Pass"]
    ),
    use_container_width=True
)

st.write("Classification report")
report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    output_dict=True,
    zero_division=0
)
st.dataframe(
    pd.DataFrame(report).T.round(3),
    use_container_width=True
)

st.subheader("Naye student ka result predict karein")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input(
        "Study hours per week",
        min_value=0.0,
        max_value=168.0,
        value=6.0,
        step=0.5
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=1.0
    )

with col3:
    midterm_score = st.number_input(
        "Midterm score",
        min_value=0.0,
        max_value=100.0,
        value=64.0,
        step=1.0
    )

if st.button("Predict result"):
    student = pd.DataFrame(
        [[study_hours, attendance, midterm_score]],
        columns=FEATURES
    )

    predicted_class = int(model.predict(student)[0])
    probabilities = model.predict_proba(student)[0]
    pass_probability = probabilities[
        list(model.classes_).index(1)
    ]

    if predicted_class == 1:
        st.success(
            f"Predicted result: PASS · Pass probability: "
            f"{pass_probability:.1%}"
        )
    else:
        st.warning(
            f"Predicted result: FAIL · Pass probability: "
            f"{pass_probability:.1%}"
        )

st.sidebar.download_button(
    "CSV download karein",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="Pass-Fail Data.csv",
    mime="text/csv"
)
