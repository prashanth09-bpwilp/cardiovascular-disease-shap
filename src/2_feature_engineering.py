#Feature Engineering
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer

spark = SparkSession.builder.appName("CVD_Phase2_Feature_Engineering").master("local[*]").getOrCreate()

# 1. Load Cleaned Data
data = spark.read.parquet("cleaned_data.parquet")

# 2. Derived Features (Using the exact column names from your dataset)
data = data.withColumn(
    "cholesterol_ratio",
    F.col("Total Cholesterol (mg/dL)") / (F.col("HDL (mg/dL)") + 1)
)

data = data.withColumn(
    "bp_age_interaction",
    F.col("Systolic BP") * F.col("Age")
)

# 3. Age Groups (Note the capital 'A' in Age)
bucketizer = Bucketizer(
    splits=[0.0, 30.0, 50.0, 70.0, 100.0],
    inputCol="Age",
    outputCol="age_group"
)
data = bucketizer.transform(data)

# 4. Save to Featured Zone locally
data.write.mode("overwrite").parquet("featured_data.parquet")

print("Phase 2: Feature Engineering Complete. Data saved to featured_data.parquet")

# --- COLAB OUTPUT VISUALIZATION ---
print("\n--- PHASE 2 OUTPUT: ENGINEERED FEATURES ---")
data.select("Age", "age_group", "cholesterol_ratio", "bp_age_interaction").show(15, truncate=False)
