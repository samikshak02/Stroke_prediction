import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("stroke_data.csv")

# Drop ID column
df = df.drop("id", axis=1)

# Numeric and Categorical Columns
numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include="object").columns

# Fill Missing Values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Categorical to Numeric
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Files
joblib.dump(model, "stroke_model.pkl")
joblib.dump(scaler, "stroke_scaler.pkl")
joblib.dump(X.columns.tolist(), "stroke_columns.pkl")

print("\nModel, Scaler and Columns saved successfully.")