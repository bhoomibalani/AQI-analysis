# import pandas as pd
# import pickle

# # Load dataset
# df = pd.read_csv("final_dataset.csv")

# # Create date
# df["full_date"] = pd.to_datetime({
#     "year": df["Year"],
#     "month": df["Month"],
#     "day": df["Date"]
# })

# df = df.sort_values(by="full_date")

# # ---------------- ADD WEATHER (dummy for training) ---------------- #
# df["Temperature"] = 25
# df["Humidity"] = 60
# df["WindSpeed"] = 5

# # ---------------- FEATURES ---------------- #
# X = df.drop(columns=["AQI", "full_date"])
# y = df["AQI"]

# # ---------------- TRAIN MODEL ---------------- #
# from xgboost import XGBRegressor

# model = XGBRegressor()
# model.fit(X, y)

# # ---------------- SAVE ---------------- #
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# df.to_csv("processed_data.csv", index=False)

# print("✅ Model trained and saved!")

# - AQI peaks during November, indicating severe winter pollution in NCR
# - PM10 shows the strongest correlation with AQI (~0.9)
# - PM2.5 is also a major contributor (~0.8)
# - CO has moderate influence on AQI
# - Ozone shows negative correlation with AQI



# - Performed EDA to identify seasonal AQI trends
# - Found PM10 and PM2.5 as major contributors to AQI
# - Built Random Forest model for AQI prediction
# - Improved prediction using lag features
# - Visualized feature importance for interpretability

import pandas as pd
import pickle
from xgboost import XGBRegressor

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("final_dataset.csv")

# ---------------- CREATE DATE ---------------- #
df["full_date"] = pd.to_datetime({
    "year": df["Year"],
    "month": df["Month"],
    "day": df["Date"]
})

# Sort by date (important)
df = df.sort_values(by="full_date")

# ---------------- ADD WEATHER FEATURES ---------------- #
# (Dummy values for training — real values will come from API in app)
df["Temperature"] = 25
df["Humidity"] = 60
df["WindSpeed"] = 5

# ---------------- FEATURE SELECTION ---------------- #
X = df.drop(columns=["AQI", "full_date"])
y = df["AQI"]

# ---------------- TRAIN MODEL ---------------- #
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X, y)

# ---------------- SAVE MODEL ---------------- #
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ---------------- SAVE PROCESSED DATA ---------------- #
df.to_csv("data.csv", index=False)

print("✅ Model trained successfully!")
print("✅ Files saved: model.pkl, data.csv")