from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("Evaluate").getOrCreate()

# ================= LOAD DATA =================
df = spark.read.parquet("data/processed")

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# ================= LOAD MODELS =================
lr_model = PipelineModel.load("models/lr_pipeline")
rf_model = PipelineModel.load("models/rf_pipeline")

# ================= PREDICT =================
lr_pred = lr_model.transform(test_data)
rf_pred = rf_model.transform(test_data)

# ================= EVALUATION =================
evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="rmse"
)

print("LR RMSE:", evaluator.evaluate(lr_pred))
print("RF RMSE:", evaluator.evaluate(rf_pred))