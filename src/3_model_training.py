# Convert Spark DataFrame to Pandas
pandas_df = data.toPandas()

print("Converted to Pandas")
print(pandas_df.shape)

pandas_df.head()

#MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Encode categorical columns
label_encoders = {}

for col in pandas_df.select_dtypes(include=['object']).columns:

    le = LabelEncoder()

    pandas_df[col] = le.fit_transform(
        pandas_df[col].astype(str)
    )

    label_encoders[col] = le

# Features and target
X = pandas_df.drop("CVD Risk Level", axis=1)

y = pandas_df["CVD Risk Level"]

feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest Training Complete")
