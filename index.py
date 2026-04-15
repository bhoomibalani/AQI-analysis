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

import glob
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_area_wise_data():
    month_to_num = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    files = glob.glob("AQI_daily_2024_*_Delhi_*.xlsx")
    if not files:
        raise FileNotFoundError("No area-wise Delhi AQI Excel files found.")

    frames = []
    for file_path in files:
        base = os.path.basename(file_path)
        area = base.replace("AQI_daily_2024_", "")
        area = area.split("_Delhi_")[0].replace("_", " ").strip()

        wide = pd.read_excel(file_path)
        long_df = wide.melt(id_vars="Day", var_name="MonthName", value_name="AQI")
        long_df["Month"] = long_df["MonthName"].map(month_to_num)
        long_df["Year"] = 2024
        long_df["Area"] = area
        long_df["Date"] = long_df["Day"]
        long_df["full_date"] = pd.to_datetime(
            {"year": long_df["Year"], "month": long_df["Month"], "day": long_df["Date"]},
            errors="coerce",
        )
        frames.append(long_df)

    df = pd.concat(frames, ignore_index=True)
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["AQI", "full_date"])
    df["Days"] = df["full_date"].dt.weekday
    df["Holidays_Count"] = 0
    return df.sort_values("full_date")


def train_and_save():
    df = load_area_wise_data()

    feature_cols = ["Date", "Month", "Year", "Days", "Holidays_Count", "Area"]
    target_col = "AQI"

    X = df[feature_cols]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("area_ohe", OneHotEncoder(handle_unknown="ignore"), ["Area"]),
            ("num", "passthrough", ["Date", "Month", "Year", "Days", "Holidays_Count"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Keep this output compatible with app charts.
    output = df[
        ["Date", "Month", "Year", "Holidays_Count", "Days", "Area", "AQI", "full_date"]
    ].copy()
    output.to_csv("data.csv", index=False)

    print("Model trained on area-wise Delhi data.")
    print(f"Rows used: {len(df)}")
    print(f"Areas used: {df['Area'].nunique()}")
    print("Saved: model.pkl, data.csv")


if __name__ == "__main__":
    train_and_save()