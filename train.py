import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/student_performance.csv")

print(df.head())
print(df.info())

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

numeric_features = [
    "Study_Hours",
    "Attendance",
    "Previous_Grade",
    "Sleep_Hours"
]

categorical_features = [
    "Gender",
    "Parent_Education",
    "Internet_Access",
    "Family_Support",
    "Extracurricular"
]
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("---------------------------")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importances = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features")
print(importance_df.head(10))

# Plot
plt.figure(figsize=(10,6))
plt.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()
