from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, unix_timestamp

spark = SparkSession.builder.appName("Preprocessing").getOrCreate()

# ================= LOAD =================
df = spark.read.parquet("data/taxi.parquet")

# ================= CLEANING =================
df = df.dropna()
df = df.filter(col("fare_amount") > 0)
df = df.filter(col("trip_distance") > 0)

# ================= FEATURE ENGINEERING =================
df = df.withColumn(
    "trip_duration",
    (unix_timestamp("tpep_dropoff_datetime") -
     unix_timestamp("tpep_pickup_datetime")) / 60
)

df = df.withColumn("hour", hour("tpep_pickup_datetime"))

# ================= FINAL DATA =================
df = df.select(
    "fare_amount",
    "trip_distance",
    "trip_duration",
    "hour",
    "passenger_count"
)

# ================= SAVE =================
df.write.mode("overwrite").parquet("data/processed")

print("Preprocessing Done ✔")