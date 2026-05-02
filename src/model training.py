from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

spark = SparkSession.builder.appName("Training").getOrCreate()

# ================= LOAD PROCESSED DATA =================
df = spark.read.parquet("data/processed")

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# ================= FEATURES =================
assembler = VectorAssembler(
    inputCols=["trip_distance", "trip_duration", "hour", "passenger_count"],
    outputCol="features"
)

# ================= MODELS =================
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount")

# ================= PIPELINES =================
lr_pipeline = Pipeline(stages=[assembler, lr])
rf_pipeline = Pipeline(stages=[assembler, rf])

# ================= TRAIN =================
lr_model = lr_pipeline.fit(train_data)
rf_model = rf_pipeline.fit(train_data)

# ================= SAVE =================
lr_model.write().overwrite().save("models/lr_pipeline")
rf_model.write().overwrite().save("models/rf_pipeline")

print("Training Done ✔ (Two Models Saved)")