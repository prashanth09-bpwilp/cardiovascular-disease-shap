#Ingestion ETL
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer

# Local Spark session for running on your laptop or Google Colab
spark = SparkSession.builder.appName("CVD_Phase1_Ingestion").master("local[*]").getOrCreate()

# 1. Read Raw Data
data = spark.read.csv("healthcare_data.csv", header=True, inferSchema=True)

# 2. Drop Leakage and Redundant Columns (As specified in your Dissertation)
cols_to_drop = ["Blood Pressure (mmHg)", "Height (cm)", "CVD Risk Score"]
clean_data = data.drop(*cols_to_drop)

# 3. Handle Missing Values dynamically
# Find all columns that are integers or doubles
numeric_cols = [t[0] for t in clean_data.dtypes if t[1] in ['int', 'double']]

# Use the Median strategy for imputation
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols).setStrategy("median")
imputed_data = imputer.fit(clean_data).transform(clean_data)

# 4. Save Cleaned Data locally as Parquet
imputed_data.write.mode("overwrite").parquet("cleaned_data.parquet")

print("Phase 1: Ingestion ETL Complete. Data saved to cleaned_data.parquet")

# --- COLAB OUTPUT VISUALIZATION ---
print("\n--- PHASE 1 OUTPUT: CLEANED DATA ---")
print(f"Total Rows: {imputed_data.count()}")
imputed_data.select("Age", "Systolic BP", "Total Cholesterol (mg/dL)", "CVD Risk Level").show(15, truncate=False)
