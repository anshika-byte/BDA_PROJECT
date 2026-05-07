# =========================
# IMPORTS
# =========================
import streamlit as st
import numpy as np
import pickle


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🌾 Smart Crop ", layout="wide")

# =========================
# CSS UI DESIGN
# =========================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.55);
    z-index: 0;
}

/* Center container */
.block-container {
    position: relative;
    z-index: 1;
    max-width: 1000px;
    margin: auto;
    text-align: center;
}

/* Headings */
h1, h2, h3, h4, p, label {
    color: white !important;
    font-weight: bold !important;
    text-align: center;
}

/* Button */
div.stButton > button {
    background-color: #000;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 10px;
}

/* Result box */
.result-box {
    background: #2e7d32;
    color: white;
    padding: 18px;
    border-radius: 10px;
    font-size: 26px;
    font-weight: bold;
    margin-top: 15px;
}

/* Top 3 box */
.top3 {
    background: rgba(0,0,0,0.75);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
model = pickle.load(open("xgb_final.pkl", "rb"))
scaler = pickle.load(open("scaler_final.pkl", "rb"))
labels = pickle.load(open("label_encoder_final.pkl", "rb"))

# =========================
# INPUT UI
# =========================
st.markdown("## 🌱 Enter Soil & Climate Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌿 Soil Nutrients")
    N = st.slider("Nitrogen (N)", 0, 150, 50)
    P = st.slider("Phosphorus (P)", 0, 150, 50)
    K = st.slider("Potassium (K)", 0, 150, 50)

with col2:
    st.markdown("### 🌦 Climate")
    temp = st.slider("Temperature (°C)", 10, 45, 28)
    humidity = st.slider("Humidity (%)", 10, 100, 65)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 120)

# =========================
# FEATURE ENGINEERING
# =========================
np_ratio = N / (P + 1)
nk_ratio = N / (K + 1)
pk_ratio = P / (K + 1)

features = np.array([[N, P, K, temp, humidity, rainfall,
                      np_ratio, nk_ratio, pk_ratio]])

features_scaled = scaler.transform(features)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Crop", use_container_width=True):

    probs = model.predict_proba(features_scaled)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]

    top3_crops = labels.inverse_transform(top3_idx)

    # =========================
    # RESULT BOX
    # =========================
    st.markdown(
        f"<div class='result-box'>🌾 Recommended Crop: {top3_crops[0].upper()}</div>",
        unsafe_allow_html=True
    )

    # =========================
    # TOP 3 RESULTS
    # =========================
    st.markdown("### 🌿 Top 3 Recommendations")

    for i in range(3):
        st.markdown(
            f"<div class='top3'>✔ {top3_crops[i]} → {probs[top3_idx[i]]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

  
