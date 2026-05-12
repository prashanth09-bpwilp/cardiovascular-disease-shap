from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession.builder.appName("CVD_Batch_Scoring").master("local[*]").getOrCreate()

# 1. Load New Unseen Data 
# (For testing in Colab, you can just reload your featured dataset, or a new test CSV)
new_data = spark.read.parquet("featured_data.parquet")

# 2. Load Saved Preprocessor and Model
preprocessor = PipelineModel.load("models/preprocessor")
rf_model = RandomForestClassificationModel.load("models/rf_model")

# 3. Predict
prep_data = preprocessor.transform(new_data)
predictions = rf_model.transform(prep_data)

# 4. Save Predictions
predictions.select("features", "target", "prediction", "probability") \
    .write.mode("overwrite") \
    .parquet("batch_predictions_results.parquet")

print("Phase 4: Batch Scoring Complete. Results saved to batch_predictions_results.parquet")
# Show top 5 results
predictions.select("target", "prediction", "probability").show(5)
