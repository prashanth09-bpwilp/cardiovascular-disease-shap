from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("CVD_Phase3_ML_Training").master("local[*]").getOrCreate()

# 1. Load Featured Data
data = spark.read.parquet("featured_data.parquet")

# Ensure target is treated as a double for PySpark ML
data = data.withColumn("target", F.col("target").cast("double"))

# 2. Pipeline Setup
categorical_cols = ["gender", "smoking", "alcohol", "physical_activity"]
numeric_cols = ["age", "blood_pressure", "cholesterol", "hdl", "glucose", "bmi", "heart_rate", "cholesterol_ratio", "bp_age_interaction"]

stages = []
# Index categorical strings
for col in categorical_cols:
    stages.append(StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep"))

# Assemble all features
assembler_inputs = numeric_cols + [f"{c}_indexed" for c in categorical_cols] + ["age_group"]
stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip"))

# Scale features
stages.append(StandardScaler(inputCol="raw_features", outputCol="features"))

# 3. Train/Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 4. Fit Preprocessor and Transform
preprocessor = Pipeline(stages=stages).fit(train_data)
train_prep = preprocessor.transform(train_data)

# 5. Train Models (Random Forest used as primary example)
lr = LogisticRegression(featuresCol="features", labelCol="target").fit(train_prep)
rf = RandomForestClassifier(featuresCol="features", labelCol="target", seed=42).fit(train_prep)

# 6. Save Preprocessor and Models Locally
preprocessor.write().overwrite().save("models/preprocessor")
rf.write().overwrite().save("models/rf_model")
lr.write().overwrite().save("models/lr_model")

print("Phase 3: Training Complete. Models saved locally in /models/")
