from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark Session for local execution
spark = SparkSession.builder.appName("CVD_Phase3_ML_Training").master("local[*]").getOrCreate()

# 1. Load Featured Data
data = spark.read.parquet("featured_data.parquet")

# 2. Pipeline Setup - using EXACT column names from your dataset
categorical_cols = [
    "Sex", 
    "Smoking Status", 
    "Diabetes Status", 
    "Physical Activity Level", 
    "Family History of CVD", 
    "Blood Pressure Category"
]

numeric_cols = [
    "Age", 
    "Weight (kg)", 
    "BMI", 
    "Abdominal Circumference (cm)", 
    "Total Cholesterol (mg/dL)", 
    "HDL (mg/dL)", 
    "Fasting Blood Sugar (mg/dL)", 
    "Waist-to-Height Ratio", 
    "Systolic BP", 
    "Diastolic BP", 
    "Estimated LDL (mg/dL)",
    "cholesterol_ratio",      # Added in Phase 2
    "bp_age_interaction"      # Added in Phase 2
]

stages = []

# Index categorical strings
for col in categorical_cols:
    stages.append(StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep"))

# Index the target variable (Converts 'Low', 'Moderate', 'High' to 0.0, 1.0, 2.0 automatically)
stages.append(StringIndexer(inputCol="CVD Risk Level", outputCol="label", handleInvalid="skip"))

# Assemble all features
assembler_inputs = numeric_cols + [f"{c}_indexed" for c in categorical_cols] + ["age_group"]
stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip"))

# Scale features
stages.append(StandardScaler(inputCol="raw_features", outputCol="features"))

# 3. Train/Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 4. Fit Preprocessor and Transform Data
preprocessor = Pipeline(stages=stages).fit(train_data)
train_prep = preprocessor.transform(train_data)
test_prep = preprocessor.transform(test_data)

# 5. Train Models 
lr = LogisticRegression(featuresCol="features", labelCol="label").fit(train_prep)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42).fit(train_prep)

# 6. Evaluate the Primary Model (Random Forest)
rf_predictions = rf.transform(test_prep)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
f1_score = evaluator.evaluate(rf_predictions, {evaluator.metricName: "f1"})
weighted_precision = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedPrecision"})
weighted_recall = evaluator.evaluate(rf_predictions, {evaluator.metricName: "weightedRecall"})

print("\n--- MODEL EVALUATION RESULTS (Random Forest) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall:    {weighted_recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}\n")

# 7. Save Preprocessor and Models Locally
preprocessor.write().overwrite().save("models/preprocessor")
rf.write().overwrite().save("models/rf_model")
lr.write().overwrite().save("models/lr_model")

print("Phase 3: Training Complete. Models saved locally in /models/")
# --- COLAB OUTPUT VISUALIZATION ---
print("\n--- PHASE 3 OUTPUT: TRANSFORMED ML FEATURES ---")
# Showing the final dense vector that the machine learning model actually trains on
train_prep.select("CVD Risk Level", "label", "features").show(15, truncate=True)
