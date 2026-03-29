import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from scipy.interpolate import PchipInterpolator

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Thermal Stress Predictor",
    layout="centered"
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return joblib.load("model/thermal_model.pkl")

if not os.path.exists("model/thermal_model.pkl"):
    st.error("Model file not found. Run training first.")
    st.stop()

model = load_model()

# =====================================
# SIDEBAR INPUTS
# =====================================
st.sidebar.title("Input Parameters")

FA = st.sidebar.slider("Fly Ash (%)", 0, 50, 10)
Thickness = st.sidebar.number_input("Thickness (mm)", 100, 500, 200)
E = st.sidebar.number_input("Modulus of Elasticity (MPa)", 10000.0, 50000.0, 30000.0)

T_top = st.sidebar.number_input("Top Temperature (°C)", 20.0, 80.0, 40.0)
T_mid = st.sidebar.number_input("Mid Temperature (°C)", 20.0, 80.0, 35.0)
T_bot = st.sidebar.number_input("Bottom Temperature (°C)", 20.0, 80.0, 30.0)

# =====================================
# TITLE
# =====================================
st.title("Thermal Stress Prediction using Machine Learning")

# =====================================
# PREDICTION
# =====================================
if st.button("Predict", use_container_width=True):

    # ---------------------------------
    # PREPARE INPUT
    # ---------------------------------
    input_df = pd.DataFrame([{
        "FA": FA,
        "Thickness": Thickness,
        "E": E,
        "T_top": T_top,
        "T_mid": T_mid,
        "T_bot": T_bot
    }])

    # Feature Engineering
    input_df["dT_total"] = input_df["T_top"] - input_df["T_bot"]
    input_df["curvature"] = input_df["T_top"] - 2*input_df["T_mid"] + input_df["T_bot"]

    input_df = input_df[
        ["FA","Thickness","E","dT_total","curvature"]
    ]

    # ---------------------------------
    # PREDICT
    # ---------------------------------
    pred = model.predict(input_df)[0]
    sigma_top, sigma_mid, sigma_bot = pred

    # ---------------------------------
    # DISPLAY RESULTS
    # ---------------------------------
    st.subheader("Predicted Stresses (MPa)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top", f"{sigma_top:.3f}")
    c2.metric("Mid", f"{sigma_mid:.3f}")
    c3.metric("Bottom", f"{sigma_bot:.3f}")

    # =====================================
    # STRESS vs DEPTH
    # =====================================
    st.subheader("Stress Distribution Through Depth")

    depths = np.array([0, Thickness/2, Thickness])
    stresses = np.array([sigma_top, sigma_mid, sigma_bot])

    interp = PchipInterpolator(depths, stresses)

    depth_smooth = np.linspace(0, Thickness, 100)
    stress_smooth = interp(depth_smooth)

    fig1, ax1 = plt.subplots()

    ax1.plot(stress_smooth, depth_smooth, label="Stress Profile")
    ax1.scatter(stresses, depths)

    ax1.set_xlabel("Stress (MPa)")
    ax1.set_ylabel("Depth (mm)")
    ax1.set_title("Stress vs Depth")

    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend()

    st.pyplot(fig1)

    # =====================================
    # STRESS vs TEMPERATURE GRADIENT
    # =====================================
    st.subheader("Stress vs Temperature Gradient")

    grad_range = np.linspace(0, 30, 50)

    stress_top_list = []
    stress_mid_list = []
    stress_bot_list = []

    for g in grad_range:

        Tt = T_mid + g/2
        Tb = T_mid - g/2
        Tm = T_mid

        temp_df = pd.DataFrame([{
            "FA": FA,
            "Thickness": Thickness,
            "E": E,
            "T_top": Tt,
            "T_mid": Tm,
            "T_bot": Tb
        }])

        temp_df["dT_total"] = temp_df["T_top"] - temp_df["T_bot"]
        temp_df["curvature"] = temp_df["T_top"] - 2*temp_df["T_mid"] + temp_df["T_bot"]

        temp_df = temp_df[
            ["FA","Thickness","E","dT_total","curvature"]
        ]

        pred = model.predict(temp_df)[0]

        stress_top_list.append(pred[0])
        stress_mid_list.append(pred[1])
        stress_bot_list.append(pred[2])

    fig2, ax2 = plt.subplots()

    ax2.plot(grad_range, stress_top_list, label="Top")
    ax2.plot(grad_range, stress_mid_list, label="Mid")
    ax2.plot(grad_range, stress_bot_list, label="Bottom")

    ax2.set_xlabel("Temperature Gradient ΔT (°C)")
    ax2.set_ylabel("Stress (MPa)")
    ax2.set_title("Stress vs Temperature Gradient")

    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)