import streamlit as st
import pandas as pd
import joblib

# ----------------- Page Setup ----------------- #
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# ----------------- Load Model, Scaler & Data ----------------- #
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/simple_electricity_model.pkl")
    scaler = joblib.load("model/simple_scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

model, scaler = load_artifacts()
df = load_dataset()

# Retrieve latest 3 lag readings from dataset
default_lag1 = float(df["Appliances"].iloc[-1])
default_lag2 = float(df["Appliances"].iloc[-2])
default_lag3 = float(df["Appliances"].iloc[-3])

# ----------------- UI Layout ----------------- #
st.title("⚡ Electricity Consumption Prediction")
st.write(
    "Predict household appliance electricity consumption (Wh) based on "
    "environmental conditions, lighting, and recent usage trends."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌤️ Environmental & Lighting Conditions")
    t_out = st.number_input("Outside Temperature (°C)", value=10.0, step=0.1)
    rh_out = st.number_input("Outside Humidity (%)", value=75.0, min_value=0.0, max_value=100.0, step=0.5)
    lights = st.number_input("Lights Consumption (Wh)", value=0.0, min_value=0.0, step=5.0)

    st.subheader("🕒 Date & Time Context")
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    day = st.slider("Day of Month (1-31)", 1, 31, 15)
    month = st.slider("Month (1-12)", 1, 12, 5)
    day_of_week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
    )
    is_weekend = 1 if day_of_week >= 5 else 0

with col2:
    st.subheader("📈 Previous Consumption (Lags)")
    st.info("Pre-filled with the latest actual readings from the dataset.")
    lag1 = st.number_input("Consumption 10 mins ago (Wh)", value=default_lag1, step=1.0)
    lag2 = st.number_input("Consumption 20 mins ago (Wh)", value=default_lag2, step=1.0)
    lag3 = st.number_input("Consumption 30 mins ago (Wh)", value=default_lag3, step=1.0)

st.divider()

# ----------------- Prediction Trigger ----------------- #
if st.button("⚡ Predict Consumption", type="primary"):
    # Build input DataFrame exactly matching model training columns
    input_data = pd.DataFrame([{
        "T_out": t_out,
        "RH_out": rh_out,
        "lights": lights,
        "Hour": hour,
        "Day": day,
        "Month": month,
        "Day_of_Week": day_of_week,
        "Is_Weekend": is_weekend,
        "Appliances_Lag_1": lag1,
        "Appliances_Lag_2": lag2,
        "Appliances_Lag_3": lag3
    }])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### Predicted Electricity Consumption: **{prediction:.2f} Wh**")