import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1. Load dataset

df = pd.read_csv("dataset/energydata_complete.csv")

# 2. Convert date column

df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# 3. Sort by date

df = df.sort_values("date").reset_index(drop=True)

# 4. Create time-based features

df["Hour"] = df["date"].dt.hour
df["Day"] = df["date"].dt.day
df["Month"] = df["date"].dt.month
df["Day_of_Week"] = df["date"].dt.dayofweek
df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

# 5. Create lag features

df["Appliances_Lag_1"] = df["Appliances"].shift(1)
df["Appliances_Lag_2"] = df["Appliances"].shift(2)
df["Appliances_Lag_3"] = df["Appliances"].shift(3)

# 6. Create rolling mean

df["Appliances_Rolling_Mean"] = (
    df["Appliances"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

# 7. Remove missing values

df = df.dropna()

# 8. Remove duplicates

df = df.drop_duplicates()

# 9. Remove unnecessary columns

df = df.drop(columns=["date", "rv1", "rv2"])

# 10. Separate features and target

X = df.drop(columns=["Appliances"])
y = df["Appliances"]

# 11. Time-based Train-Test Split

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 12. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 13. Create models

linear_model = LinearRegression()

decision_tree_model = DecisionTreeRegressor(
    random_state=42
)

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 14. Train Linear Regression

print("Training Linear Regression...")

linear_model.fit(
    X_train_scaled,
    y_train
)

# 15. Train Decision Tree

print("Training Decision Tree...")

decision_tree_model.fit(
    X_train,
    y_train
)

# 16. Train Random Forest

print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

# 17. Make predictions

linear_predictions = linear_model.predict(
    X_test_scaled
)

tree_predictions = decision_tree_model.predict(
    X_test
)

forest_predictions = random_forest_model.predict(
    X_test
)

# 18. Evaluation function

def calculate_metrics(actual, predictions):

    mae = mean_absolute_error(
        actual,
        predictions
    )

    mse = mean_squared_error(
        actual,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predictions
    )

    return mae, mse, rmse, r2


# 19. Calculate metrics

linear_metrics = calculate_metrics(
    y_test,
    linear_predictions
)

tree_metrics = calculate_metrics(
    y_test,
    tree_predictions
)

forest_metrics = calculate_metrics(
    y_test,
    forest_predictions
)

# 20. Model comparison

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        linear_metrics[0],
        tree_metrics[0],
        forest_metrics[0]
    ],

    "MSE": [
        linear_metrics[1],
        tree_metrics[1],
        forest_metrics[1]
    ],

    "RMSE": [
        linear_metrics[2],
        tree_metrics[2],
        forest_metrics[2]
    ],

    "R2": [
        linear_metrics[3],
        tree_metrics[3],
        forest_metrics[3]
    ]
})

# 21. Display results

print("\n")
print("=" * 65)
print("MODEL COMPARISON AFTER LAG FEATURES")
print("=" * 65)

print(results.round(2))

# 22. Find best model based on RMSE

best_index = results["RMSE"].idxmin()

best_model_name = results.loc[
    best_index,
    "Model"
]

print("\nBest model based on RMSE:")
print(best_model_name)

# 23. Select best predictions

if best_model_name == "Linear Regression":

    best_predictions = linear_predictions

elif best_model_name == "Decision Tree":

    best_predictions = tree_predictions

else:

    best_predictions = forest_predictions


# 24. Actual vs Predicted graph

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values[:300],
    label="Actual"
)

plt.plot(
    best_predictions[:300],
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Electricity Consumption"
)

plt.xlabel("Test Sample")

plt.ylabel("Appliances Consumption")

plt.legend()

plt.tight_layout()

plt.show()

# 25. Feature Importance for Linear Regression

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

# Absolute coefficient for ranking
feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

# Display top 15 features
print("\n")
print("=" * 65)
print("TOP 15 FEATURES")
print("=" * 65)

print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(15).round(4)
)

# 26. Plot top 15 features

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Absolute_Coefficient"]
)

plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")

plt.title(
    "Top 15 Features for Electricity Consumption Prediction"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# 27. Save trained model and scaler

os.makedirs("model", exist_ok=True)

joblib.dump(
    linear_model,
    "model/electricity_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nModel saved successfully!")
print("Saved: model/electricity_model.pkl")
print("Saved: model/scaler.pkl")