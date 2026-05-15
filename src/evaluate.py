from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# ================= SPARK SESSION =================
spark = SparkSession.builder \
    .appName("Evaluate Models") \
    .getOrCreate()

# ================= LOAD TEST DATA =================
test_data = spark.read.parquet("data/test")

# ================= LOAD MODELS =================
lr_model = PipelineModel.load("models/lr_pipeline")
rf_model = PipelineModel.load("models/rf_pipeline")

# ================= PREDICTIONS =================
lr_pred = lr_model.transform(test_data)
rf_pred = rf_model.transform(test_data)

# ================= EVALUATION =================
evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="rmse"
)

# Calculate RMSE
lr_rmse = evaluator.evaluate(lr_pred)
rf_rmse = evaluator.evaluate(rf_pred)

# ================= RESULTS =================
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")

# Compare models
if lr_rmse < rf_rmse:
    print("Best Model: Linear Regression")
else:
    print("Best Model: Random Forest")

# ================= STOP SPARK =================
spark.stop()