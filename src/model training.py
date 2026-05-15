from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

# ================= SPARK SESSION =================
spark = SparkSession.builder \
    .appName("Training") \
    .getOrCreate()

# ================= LOAD PROCESSED DATA =================
df = spark.read.parquet("data/processed")

# ================= TRAIN / TEST SPLIT =================
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Save train/test split
train_data.write.mode("overwrite").parquet("data/train")
test_data.write.mode("overwrite").parquet("data/test")

# ================= FEATURES =================
assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "trip_duration",
        "hour",
        "day_of_week",
        "is_weekend",
        "avg_speed",
        "passenger_count"
    ],
    outputCol="features"
)

# ================= MODELS =================
lr = LinearRegression(
    featuresCol="features",
    labelCol="fare_amount"
)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="fare_amount"
)

# ================= PIPELINES =================
lr_pipeline = Pipeline(stages=[assembler, lr])
rf_pipeline = Pipeline(stages=[assembler, rf])

# ================= TRAIN =================
lr_model = lr_pipeline.fit(train_data)
rf_model = rf_pipeline.fit(train_data)

# ================= SAVE MODELS =================
lr_model.write().overwrite().save("models/lr_pipeline")
rf_model.write().overwrite().save("models/rf_pipeline")

print("Training Done ✔ (Two Models Saved)")

# ================= STOP SPARK =================
spark.stop()