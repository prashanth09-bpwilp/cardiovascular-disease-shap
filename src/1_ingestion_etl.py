#Note: Run this block of code separately first
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CVD_Risk_Pipeline") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)

#Next code block
from pyspark.ml.feature import Imputer
import pyspark.sql.functions as F

# Load dataset
data = spark.read.csv(
    "healthcare_data.csv",
    header=True,
    inferSchema=True
)

print("Original Rows:", data.count())

# Drop leakage/redundant columns
cols_to_drop = [
    "Blood Pressure (mmHg)",
    "Height (cm)",
    "CVD Risk Score"
]

data = data.drop(*cols_to_drop)

# Missing value handling
numeric_cols = [
    t[0] for t in data.dtypes
    if t[1] in ['int', 'double']
]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=numeric_cols
).setStrategy("median")

data = imputer.fit(data).transform(data)

print("ETL Complete")

data.show(5)
