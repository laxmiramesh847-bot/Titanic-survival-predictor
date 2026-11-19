import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# TITANIC SURVIVAL PREDICTOR | LAXMI RAMESH TELI
# Dataset expected: Kaggle Titanic train.csv (891 rows)
# ---------------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor | Laxmi Ramesh",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #071827 0%, #0b2538 52%, #102f43 100%);
        color: #f4f7fa;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #061421;
        border-right: 1px solid rgba(222, 181, 105, .25);
    }
    [data-testid="stSidebar"] * { color: #edf2f6; }
    h1, h2, h3 { color: #f7f3eb !important; }
    .hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid rgba(222,181,105,.38);
        border-radius: 22px;
        background: linear-gradient(110deg, rgba(10,35,53,.98), rgba(19,57,75,.88));
        box-shadow: 0 18px 55px rgba(0,0,0,.22);
        margin-bottom: 1.4rem;
    }
    .eyebrow {
        color: #dfb66d;
        letter-spacing: .18em;
        text-transform: uppercase;
        font-size: .76rem;
        font-weight: 700;
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.3rem, 5vw, 4.1rem);
        line-height: 1.08;
        margin: .45rem 0;
    }
    .hero p { color: #c7d4df; font-size: 1.02rem; max-width: 760px; }
    .byline { color: #dfb66d; font-weight: 600; margin-top: 1rem; }
    .metric-card {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        min-height: 108px;
    }
    .metric-label { color: #aebfcb; font-size: .82rem; }
    .metric-value { color: #f1c879; font-size: 1.8rem; font-weight: 700; }
    .section-card {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 18px;
        padding: 1.35rem;
        margin: .5rem 0 1rem 0;
    }
    .small-note { color: #aebfcb; font-size: .88rem; }
    div.stButton > button[kind="primary"] {
        background: #d8ad61;
        color: #102536;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 3rem;
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #edc77f;
        color: #102536;
    }
    div[data-baseweb="select"] > div, .stNumberInput input {
        background: #102b3d !important;
        color: #f4f7fa !important;
    }
    .result {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(222,181,105,.45);
        background: linear-gradient(120deg, rgba(216,173,97,.14), rgba(255,255,255,.04));
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


@st.cache_data(show_spinner=False)
def prepare_dataset(raw):
    """Apply the preprocessing described in the project presentation."""
    df = raw.copy()

    required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("The CSV is missing required columns: " + ", ".join(sorted(missing)))

    # Keep the project target and seven final model features.
    df = df[["Survived"] + FEATURES].copy()

    # Median imputation for Age and mode imputation for Embarked.
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    embarked_mode = df["Embarked"].mode(dropna=True)
    embarked_fill = embarked_mode.iloc[0] if not embarked_mode.empty else "S"
    df["Embarked"] = df["Embarked"].fillna(embarked_fill)

    # Label encoding, matching the presentation's approach.
    sex_encoder = LabelEncoder()
    embarked_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"].astype(str).str.lower())
    df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"].astype(str))

    # Fill any remaining numeric gaps defensively.
    for col in ["Pclass", "SibSp", "Parch", "Fare"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["Survived"])
    df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
    df = df.dropna(subset=["Survived"])
    df["Survived"] = df["Survived"].astype(int)

    X = df[FEATURES]
    y = df["Survived"]
    return X, y, sex_encoder, embarked_encoder, embarked_fill, raw.copy()


@st.cache_resource(show_spinner="Training the Logistic Regression model...")
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine Learning · Case Study</div>
      <h1>Would You Have Survived?</h1>
      <p>Explore the famous Titanic passenger dataset and estimate a passenger's
      survival outcome using a Logistic Regression model.</p>
      <div class="byline">A project by Laxmi Ramesh Teli · B.Tech CSE · UVCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Dataset loading ----------
with st.sidebar:
    st.markdown("## Project menu")
    st.caption("Titanic Survival Prediction")
    uploaded = st.file_uploader("Upload Titanic train.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### Model details")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Train/test split:** 80% / 20%")
    st.write("**Random state:** 42")
    st.write("**Input features:** 7")
    st.markdown("---")
    st.caption("Dataset: Kaggle Titanic training data (891 passenger records).")

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    st.info("Upload the Kaggle Titanic **train.csv** file in the left sidebar to activate the prediction interface.")
    st.markdown(
        """
        <div class="section-card">
          <h3>About this case study</h3>
          <p class="small-note">The project presentation describes a dataset of 891 passengers
          and 12 columns. The target is <b>Survived</b> (0 = No, 1 = Yes).
          The model uses Pclass, Sex, Age, SibSp, Parch, Fare and Embarked.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    X, y, sex_encoder, embarked_encoder, embarked_fill, original_df = prepare_dataset(raw_df)
    model, accuracy = train_model(X, y)
except Exception as exc:
    st.error(f"Unable to prepare the dataset or train the model: {exc}")
    st.stop()

# ---------- Summary metrics ----------
survival_rate = float(y.mean() * 100)
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Passenger records", f"{len(y):,}")
with c2:
    metric_card("Model test accuracy", f"{accuracy * 100:.2f}%")
with c3:
    metric_card("Survived in dataset", f"{survival_rate:.1f}%")
with c4:
    metric_card("Model features", "07")

st.write("")

# ---------- Prediction form ----------
left, right = st.columns([1.05, .95], gap="large")

with left:
    st.markdown("## Passenger details")
    st.markdown('<div class="small-note">Enter passenger information to generate a model prediction.</div>', unsafe_allow_html=True)

    with st.form("passenger_form"):
        a, b = st.columns(2)
        with a:
            pclass = st.selectbox("Passenger class", [1, 2, 3], index=0,
                                  format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x])
            sex = st.selectbox("Sex", ["female", "male"])
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
            sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1)
        with b:
            parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0, step=1)
            fare = st.number_input("Fare paid (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
            embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                                    format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])
        submitted = st.form_submit_button("Predict survival")

with right:
    st.markdown("## Prediction result")
    if submitted:
        # Convert user values with the encoders fitted on the uploaded dataset.
        sex_value = sex_encoder.transform([sex])[0] if sex in sex_encoder.classes_ else 0
        embarked_value = embarked_encoder.transform([embarked])[0] if embarked in embarked_encoder.classes_ else embarked_encoder.transform([str(embarked_fill)])[0]

        passenger = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex_value,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked_value,
        }], columns=FEATURES)

        prediction = int(model.predict(passenger)[0])
        probabilities = model.predict_proba(passenger)[0]
        survival_probability = float(probabilities[list(model.classes_).index(1)] * 100)

        st.markdown('<div class="result">', unsafe_allow_html=True)
        if prediction == 1:
            st.success("Prediction: Likely to survive")
            st.markdown(f"### {survival_probability:.1f}%")
            st.write("Estimated survival probability")
        else:
            st.error("Prediction: Unlikely to survive")
            st.markdown(f"### {survival_probability:.1f}%")
            st.write("Estimated survival probability")
        st.progress(int(round(survival_probability)))
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("This is a statistical prediction from the trained model, not a guarantee.")
    else:
        st.markdown(
            """
            <div class="section-card">
              <h3>Your result will appear here</h3>
              <p class="small-note">Complete the passenger details and select
              <b>Predict survival</b> to see the predicted class and probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- Dataset insights ----------
st.markdown("---")
st.markdown("## Explore the dataset")
tab1, tab2, tab3 = st.tabs(["Survival overview", "Gender analysis", "Passenger class"])

with tab1:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(["Did not survive", "Survived"], counts.values)
    ax.set_ylabel("Number of passengers")
    ax.set_title("Survival count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    gender_df = original_df.copy()
    if {"Sex", "Survived"}.issubset(gender_df.columns):
        rates = gender_df.groupby("Sex", dropna=False)["Survived"].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(rates.index.astype(str), rates.values)
        ax.set_ylabel("Survival rate (%)")
        ax.set_title("Survival rate by gender")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("The uploaded file does not contain Sex and Survived columns.")

with tab3:
    if {"Pclass", "Survived"}.issubset(original_df.columns):
        rates = original_df.groupby("Pclass")["Survived"].mean().mul(100).reindex([1, 2, 3])
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(["1st class", "2nd class", "3rd class"], rates.values)
        ax.set_ylabel("Survival rate (%)")
        ax.set_title("Survival rate by passenger class")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("The uploaded file does not contain Pclass and Survived columns.")

# ---------- Project notes ----------
st.markdown("---")
st.markdown("## Project methodology")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("### 01 · Data preparation")
    st.write("Age: median imputation. Embarked: mode imputation. Cabin, Ticket, PassengerId and Name are excluded.")
with m2:
    st.markdown("### 02 · Model building")
    st.write("Sex and Embarked are label encoded. The seven selected features are used to train Logistic Regression.")
with m3:
    st.markdown("### 03 · Evaluation")
    st.write("An 80/20 train-test split with random_state 42 is evaluated using accuracy_score.")

st.markdown(
    """
    <div class="section-card">
      <div class="eyebrow">Prepared by</div>
      <h3 style="margin-bottom:.2rem;">Laxmi Ramesh Teli</h3>
      <div class="small-note">V Semester · B.Tech Computer Science and Engineering ·
      University Visvesvaraya College of Engineering (UVCE)</div>
    </div>
    """,
    unsafe_allow_html=True,
)
