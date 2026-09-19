import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# -----------------------------------
# 1. Load dataset
# -----------------------------------

df = pd.read_csv("dataset/energydata_complete.csv")


# -----------------------------------
# 2. Convert date
# -----------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True
)


# -----------------------------------
# 3. Sort by date
# -----------------------------------

df = df.sort_values("date").reset_index(drop=True)


# -----------------------------------
# 4. Create time features
# -----------------------------------

df["Hour"] = df["date"].dt.hour

df["Day"] = df["date"].dt.day

df["Month"] = df["date"].dt.month

df["Day_of_Week"] = df["date"].dt.dayofweek

df["Is_Weekend"] = (
    df["Day_of_Week"] >= 5
).astype(int)


# -----------------------------------
# 5. Create lag features
# -----------------------------------

df["Appliances_Lag_1"] = (
    df["Appliances"].shift(1)
)

df["Appliances_Lag_2"] = (
    df["Appliances"].shift(2)
)

df["Appliances_Lag_3"] = (
    df["Appliances"].shift(3)
)


# -----------------------------------
# 6. Remove missing values
# -----------------------------------

df = df.dropna()


# -----------------------------------
# 7. Select simple features
# -----------------------------------

features = [
    "T_out",
    "RH_out",
    "lights",
    "Hour",
    "Day",
    "Month",
    "Day_of_Week",
    "Is_Weekend",
    "Appliances_Lag_1",
    "Appliances_Lag_2",
    "Appliances_Lag_3"
]


X = df[features]

y = df["Appliances"]


# -----------------------------------
# 8. Time-based train-test split
# -----------------------------------

split_index = int(len(df) * 0.80)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# -----------------------------------
# 9. Scale features
# -----------------------------------

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# -----------------------------------
# 10. Create model
# -----------------------------------

model = LinearRegression()


# -----------------------------------
# 11. Train model
# -----------------------------------

print("Training simplified model...")

model.fit(
    X_train_scaled,
    y_train
)


# -----------------------------------
# 12. Predictions
# -----------------------------------

predictions = model.predict(
    X_test_scaled
)


# -----------------------------------
# 13. Evaluation
# -----------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# -----------------------------------
# 14. Display results
# -----------------------------------

print("\n")
print("=" * 55)
print("SIMPLIFIED MODEL RESULTS")
print("=" * 55)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")


# -----------------------------------
# 15. Save model
# -----------------------------------

os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    model,
    "model/simple_electricity_model.pkl"
)


joblib.dump(
    scaler,
    "model/simple_scaler.pkl"
)


print("\nModel saved successfully!")

print(
    "Saved: model/simple_electricity_model.pkl"
)

print(
    "Saved: model/simple_scaler.pkl"
)