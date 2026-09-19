# ⚡ Electricity Consumption Prediction

## 📌 Project Overview

This project uses Machine Learning to predict electricity consumption based on environmental conditions, lighting consumption, time-related features, and recent electricity consumption.

The project uses the UCI Individual Household Electric Power Consumption-style energy dataset containing household electricity and environmental measurements.

The prediction model uses Linear Regression and time-based features to estimate electricity consumption for a 10-minute interval.

---

## 🎯 Objectives

- Predict household electricity consumption using Machine Learning.
- Analyze the relationship between environmental conditions and electricity usage.
- Create time-based features from the date and time.
- Use previous electricity consumption as lag features.
- Compare different Machine Learning models.
- Evaluate the model using MAE, MSE, RMSE, and R².
- Create a Python program for making predictions.

---

## 🗂️ Project Structure

```text
Electricity-Consumption-Prediction
│
├── dataset
│   └── energydata_complete.csv
│
├── model
│   ├── simple_electricity_model.pkl
│   └── simple_scaler.pkl
│
├── train_model.py
├── train_simple_model.py
├── predict.py
└── README.md

Dataset Information:

Source: UCI Appliances Energy Prediction dataset.

Target variable: Appliances energy consumption in Wh.