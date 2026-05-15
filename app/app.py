import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 Taxi Fare Prediction System")
st.caption("Big Data + Spark ML Project")

# ================= SPARK SESSION =================
spark = SparkSession.builder \
    .appName("TaxiStreamlitApp") \
    .getOrCreate()

# ================= MODEL =================
model_choice = st.radio(
    "Choose Model",
    ["Linear Regression", "Random Forest"],
    horizontal=True
)

model_path = (
    "models/lr_pipeline"
    if model_choice == "Linear Regression"
    else "models/rf_pipeline"
)

model = PipelineModel.load(model_path)

st.divider()

# ================= INPUT SECTION =================
st.subheader("📥 Trip Information")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input(
        "📍 Trip Distance (KM)",
        min_value=0.1,
        value=3.0
    )

    passengers = st.number_input(
        "👥 Passenger Count",
        min_value=1,
        max_value=6,
        value=1
    )

with col2:
    hour = st.slider(
        "⏰ Hour of Day",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.selectbox(
        "📅 Day of Week",
        [
            (1, "Monday"),
            (2, "Tuesday"),
            (3, "Wednesday"),
            (4, "Thursday"),
            (5, "Friday"),
            (6, "Saturday"),
            (7, "Sunday")
        ],
        format_func=lambda x: x[1]
    )[0]

# ================= ENGINEERED FEATURES =================
is_weekend = 1 if day_of_week in [6, 7] else 0

avg_speed = 40  # assumed constant

trip_duration = (distance / avg_speed) * 60

# ================= INFO BOX =================
st.info(f"""
🧠 Auto Engineered Features:

- Weekend: {"Yes" if is_weekend else "No"}
- Estimated Duration: {trip_duration:.2f} minutes
- Avg Speed: {avg_speed} km/h
""")

# ================= PREDICTION =================
if st.button("🚀 Predict Fare", use_container_width=True):

    input_data = [(
        distance,
        trip_duration,
        hour,
        day_of_week,
        is_weekend,
        avg_speed,
        passengers
    )]

    df = spark.createDataFrame(
        input_data,
        [
            "trip_distance",
            "trip_duration",
            "hour",
            "day_of_week",
            "is_weekend",
            "avg_speed",
            "passenger_count"
        ]
    )

    prediction = model.transform(df).collect()[0]["prediction"]

    st.success(f"💰 Estimated Fare: ${prediction:.2f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fare", f"${prediction:.2f}")

    with col2:
        st.metric("Distance", f"{distance} km")

    with col3:
        st.metric("Duration", f"{trip_duration:.1f} min")

# ================= FOOTER =================
st.divider()
st.caption("Built using Apache Spark ML + Streamlit")