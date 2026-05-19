#Feature Engineering
from pyspark.ml.feature import Bucketizer

# Cholesterol ratio
data = data.withColumn(
    "cholesterol_ratio",
    F.col("Total Cholesterol (mg/dL)") /
    (F.col("HDL (mg/dL)") + 1)
)

# BP x Age interaction
data = data.withColumn(
    "bp_age_interaction",
    F.col("Systolic BP") * F.col("Age")
)

# Age groups
bucketizer = Bucketizer(
    splits=[0.0, 30.0, 50.0, 70.0, 100.0],
    inputCol="Age",
    outputCol="age_group"
)

data = bucketizer.transform(data)

print("Feature Engineering Complete")

data.select(
    "Age",
    "age_group",
    "cholesterol_ratio",
    "bp_age_interaction"
).show(5)
