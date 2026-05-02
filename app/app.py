import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from datetime import datetime

st.set_page_config(page_title="Uber Style Taxi AI", page_icon="🚕", layout="wide")

st.title("🚕 Taxi Fare Prediction ")

spark = SparkSession.builder.appName("TaxiApp").getOrCreate()

# ================= MODEL =================
model_choice = st.radio("Choose Model", ["Linear Regression", "Random Forest"])

model_path = "models/lr_pipeline" if model_choice == "Linear Regression" else "models/rf_pipeline"
model = PipelineModel.load(model_path)

st.divider()

# ================= INPUT UI =================
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("📍 Trip Distance (KM)", min_value=0.1, value=2.0)

with col2:
    passengers = st.number_input("👥 Passengers", min_value=1, max_value=6, value=1)

st.divider()

# ================= AUTO CALCULATIONS =================
now = datetime.now()
hour = now.hour

# افتراض سرعة ثابتة (زي نظام Uber approximation)
speed = 40  # km/h
duration = (distance / speed) * 60  # minutes

# ================= SHOW INFO =================
st.info(f"""
⏱ Auto-calculated trip info:
- Hour: {hour}
- Estimated Duration: {round(duration, 2)} minutes
- Model: {model_choice}
""")

# ================= PREDICT =================
if st.button("🚀 Predict Fare"):

    data = [(distance, duration, hour, passengers)]

    df = spark.createDataFrame(
        data,
        ["trip_distance", "trip_duration", "hour", "passenger_count"]
    )

    prediction = model.transform(df).collect()[0]["prediction"]

    st.success(f"💰 Estimated Fare: ${round(prediction, 2)}")

    st.metric("Fare", f"${round(prediction, 2)}")
    st.metric("Distance", f"{distance} KM")
    st.metric("Duration", f"{round(duration, 1)} min")