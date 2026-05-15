from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    unix_timestamp,
    dayofweek,
    when
)

# ================= SPARK SESSION =================
spark = SparkSession.builder \
    .appName("Taxi Preprocessing") \
    .getOrCreate()

# ================= LOAD =================
df = spark.read.parquet("data/taxi.parquet")

# ================= FILL MISSING VALUES =================
df = df.fillna({
    "passenger_count": 1,
    "RatecodeID": 1,
    "store_and_fwd_flag": "N",
    "congestion_surcharge": 0,
    "Airport_fee": 0,
    "cbd_congestion_fee": 0
})

# ================= CLEANING =================
df = df.filter(col("fare_amount") > 0)
df = df.filter(col("trip_distance") > 0)

# remove invalid time logic
df = df.filter(
    col("tpep_dropoff_datetime") >
    col("tpep_pickup_datetime")
)

# ================= FEATURE ENGINEERING =================

# Trip duration in minutes
df = df.withColumn(
    "trip_duration",
    (
        unix_timestamp("tpep_dropoff_datetime") -
        unix_timestamp("tpep_pickup_datetime")
    ) / 60
)

# remove unrealistic trips
df = df.filter(col("trip_duration") > 0)
df = df.filter(col("trip_duration") < 300)

# Pickup hour
df = df.withColumn(
    "hour",
    hour("tpep_pickup_datetime")
)

# Day of week
df = df.withColumn(
    "day_of_week",
    dayofweek("tpep_pickup_datetime")
)

# Weekend flag
df = df.withColumn(
    "is_weekend",
    when(col("day_of_week").isin([1, 7]), 1).otherwise(0)
)

# Average speed
df = df.withColumn(
    "avg_speed",
    col("trip_distance") / (col("trip_duration") / 60)
)

# remove unrealistic speeds
df = df.filter(col("avg_speed") < 120)

# ================= FINAL DATA =================
df = df.select(
    "fare_amount",
    "trip_distance",
    "trip_duration",
    "hour",
    "day_of_week",
    "is_weekend",
    "avg_speed",
    "passenger_count"
)

# ================= SAVE =================
df.write.mode("overwrite").parquet("data/processed")

print("Preprocessing Done ✔")

# ================= STOP SPARK =================
spark.stop()