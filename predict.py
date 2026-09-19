import pandas as pd
import joblib


# -----------------------------------
# 1. Load model and scaler
# -----------------------------------

model = joblib.load(
    "model/simple_electricity_model.pkl"
)

scaler = joblib.load(
    "model/simple_scaler.pkl"
)


# -----------------------------------
# 2. Load dataset
# -----------------------------------

df = pd.read_csv(
    "dataset/energydata_complete.csv"
)

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True
)

df = df.sort_values("date").reset_index(drop=True)


# -----------------------------------
# 3. Title
# -----------------------------------

print("=" * 55)
print("ELECTRICITY CONSUMPTION PREDICTION")
print("=" * 55)


# -----------------------------------
# 4. Get current information
# -----------------------------------

print("\nEnter the current information:\n")

T_out = float(
    input("Outside Temperature (°C): ")
)

RH_out = float(
    input("Outside Humidity (%): ")
)

lights = float(
    input("Lights Consumption: ")
)

Hour = int(
    input("Hour (0-23): ")
)

Day = int(
    input("Day (1-31): ")
)

Month = int(
    input("Month (1-12): ")
)

Day_of_Week = int(
    input("Day of Week (0=Monday, 6=Sunday): ")
)


# -----------------------------------
# 5. Calculate weekend automatically
# -----------------------------------

if Day_of_Week >= 5:
    Is_Weekend = 1
else:
    Is_Weekend = 0


# -----------------------------------
# 6. Get previous 3 consumption values
# -----------------------------------

Appliances_Lag_1 = df["Appliances"].iloc[-1]

Appliances_Lag_2 = df["Appliances"].iloc[-2]

Appliances_Lag_3 = df["Appliances"].iloc[-3]


print("\nPrevious consumption values obtained automatically:")

print(
    f"10 minutes ago: {Appliances_Lag_1:.2f}"
)

print(
    f"20 minutes ago: {Appliances_Lag_2:.2f}"
)

print(
    f"30 minutes ago: {Appliances_Lag_3:.2f}"
)


# -----------------------------------
# 7. Create input DataFrame
# -----------------------------------

input_data = pd.DataFrame([{

    "T_out": T_out,

    "RH_out": RH_out,

    "lights": lights,

    "Hour": Hour,

    "Day": Day,

    "Month": Month,

    "Day_of_Week": Day_of_Week,

    "Is_Weekend": Is_Weekend,

    "Appliances_Lag_1":
        Appliances_Lag_1,

    "Appliances_Lag_2":
        Appliances_Lag_2,

    "Appliances_Lag_3":
        Appliances_Lag_3

}])


# -----------------------------------
# 8. Scale input
# -----------------------------------

input_scaled = scaler.transform(
    input_data
)


# -----------------------------------
# 9. Make prediction
# -----------------------------------

prediction = model.predict(
    input_scaled
)


# -----------------------------------
# 10. Display result
# -----------------------------------

print("\n" + "=" * 55)
print("PREDICTION RESULT")
print("=" * 55)

print(
    f"Predicted Electricity Consumption: "
    f"{prediction[0]:.2f}"
)

print("=" * 55)