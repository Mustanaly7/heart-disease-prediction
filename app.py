"""
Streamlit demo for the Heart Disease Prediction model.
Run locally with:  streamlit run app.py
Deploy for free at: https://share.streamlit.io
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

# ---- Must match the architecture in train.py exactly ----
class HeartNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    with open("preprocessing.pkl", "rb") as f:
        prep = pickle.load(f)
    scaler = prep["scaler"]
    feature_names = prep["feature_names"]

    model = HeartNet(len(feature_names))
    model.load_state_dict(torch.load("heart_net.pt", map_location="cpu"))
    model.eval()
    return model, scaler, feature_names


model, scaler, feature_names = load_model()

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "A PyTorch neural network trained on the UCI Heart Disease (Cleveland) "
    "dataset — 83.6% test accuracy, 0.861 F1. For demonstration purposes "
    "only, not a medical device."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest pain type",
            ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 246)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"])
        restecg = st.selectbox(
            "Resting ECG result", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
        )
    with col2:
        thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}[thal],
    }
    x = np.array([[row[f] for f in feature_names]], dtype=np.float32)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        logit = model(torch.tensor(x_scaled, dtype=torch.float32))
        proba = torch.sigmoid(logit).item()

    st.divider()
    if proba >= 0.5:
        st.error(f"⚠️ Higher predicted risk of heart disease — probability {proba:.1%}")
    else:
        st.success(f"✅ Lower predicted risk of heart disease — probability {proba:.1%}")
    st.progress(proba)
    st.caption("Model output is a probability, not a diagnosis.")

st.divider()
st.markdown("[View source code & training pipeline on GitHub](#)")
