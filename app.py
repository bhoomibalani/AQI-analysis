# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import requests

# st.set_page_config(page_title="NCR AQI Dashboard", layout="wide")

# # ---------------- CUSTOM CSS ---------------- #
# st.markdown("""
# <style>
# .big-title {
#     font-size:40px;
#     font-weight:bold;
#     text-align:center;
#     color:#2c3e50;
# }
# .sub-text {
#     text-align:center;
#     color:gray;
# }
# .card {
#     padding:15px;
#     border-radius:10px;
#     background-color:#f5f5f5;
#     margin-bottom:10px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- LOAD ---------------- #
# df = pd.read_csv("data.csv")
# model = pickle.load(open("model.pkl", "rb"))
# df["full_date"] = pd.to_datetime(df["full_date"])

# # ---------------- WEATHER ---------------- #
# def get_weather():
#     api_key = st.secrets["OPENWEATHER_API_KEY"]
#     city = "Delhi"

#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
#     response = requests.get(url)
#     data = response.json()

#     if response.status_code != 200:
#         return 25, 60, 5

#     return data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"]

# temp, humidity, wind = get_weather()

# # ---------------- TITLE ---------------- #
# st.markdown('<p class="big-title">🌫 NCR Air Quality Dashboard</p>', unsafe_allow_html=True)
# st.markdown('<p class="sub-text">Real-time AQI Prediction using ML + Weather Data</p>', unsafe_allow_html=True)

# # ---------------- WEATHER CARDS ---------------- #
# st.subheader("🌤 Current Weather")

# c1, c2, c3 = st.columns(3)
# c1.metric("🌡 Temperature (°C)", temp)
# c2.metric("💧 Humidity (%)", humidity)
# c3.metric("🌬 Wind Speed (m/s)", wind)

# # ---------------- CHARTS ---------------- #
# st.subheader("📊 Pollution Trends")

# col1, col2 = st.columns(2)

# with col1:
#     fig1, ax1 = plt.subplots()
#     ax1.plot(df["full_date"], df["AQI"])
#     ax1.set_title("AQI Over Time")
#     st.pyplot(fig1)

# with col2:
#     monthly = df.groupby("Month")["AQI"].mean()
#     fig2, ax2 = plt.subplots()
#     ax2.plot(monthly.index, monthly.values)
#     ax2.set_title("Monthly AQI Trend")
#     st.pyplot(fig2)

# # Heatmap full width
# fig3, ax3 = plt.subplots(figsize=(10,5))
# sns.heatmap(df.corr(), annot=True, ax=ax3)
# st.pyplot(fig3)

# # ---------------- INPUT SECTION ---------------- #
# st.subheader("🔮 Predict AQI")

# with st.container():
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         pm25 = st.number_input("PM2.5 (µg/m³)")
#         pm10 = st.number_input("PM10 (µg/m³)")

#     with col2:
#         no2 = st.number_input("NO2 (µg/m³)")
#         so2 = st.number_input("SO2 (µg/m³)")

#     with col3:
#         co = st.number_input("CO (mg/m³)")
#         ozone = st.number_input("Ozone (µg/m³)")

# st.markdown("### 📅 Date Info")

# col4, col5, col6 = st.columns(3)

# with col4:
#     month = st.slider("Month", 1, 12)
# with col5:
#     day = st.slider("Day", 1, 31)
# with col6:
#     year = st.number_input("Year", value=2024)

# holiday = st.number_input("Holiday Count", value=0)
# days = st.slider("Day of Week (1=Mon)", 1, 7)

# # ---------------- AQI COLOR FUNCTION ---------------- #
# def get_aqi_color(aqi):
#     if aqi <= 50:
#         return "green", "Good 😊"
#     elif aqi <= 100:
#         return "blue", "Moderate 😐"
#     elif aqi <= 200:
#         return "orange", "Poor 😷"
#     else:
#         return "red", "Severe 🚨"

# # ---------------- PREDICTION ---------------- #
# if st.button("🚀 Predict AQI"):

#     input_data = pd.DataFrame([[
#         day, month, year, holiday, days,
#         pm25, pm10, no2, so2, co, ozone,
#         temp, humidity, wind
#     ]], columns=[
#         "Date","Month","Year","Holidays_Count","Days",
#         "PM2.5","PM10","NO2","SO2","CO","Ozone",
#         "Temperature","Humidity","WindSpeed"
#     ])

#     prediction = model.predict(input_data)[0]

#     color, label = get_aqi_color(prediction)

#     st.markdown(f"""
#     <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
#         <h2 style="color:white;">AQI: {round(prediction,2)}</h2>
#         <h3 style="color:white;">{label}</h3>
#     </div>
#     """, unsafe_allow_html=True)




#38d6dec75a376565904cc2dea7ceb0ed


import streamlit as st
import pandas as pd
import pickle
import requests
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NCR AQI Dashboard", layout="wide")

# ---------------- CSS ---------------- #
st.markdown("""
<style>
    :root {
        --bg-panel: #f7f9fc;
        --bg-soft: #eef3fb;
        --txt-main: #0f172a;
        --txt-subtle: #4b5563;
        --brand: #1d4ed8;
        --brand-soft: #dbeafe;
    }

    .main-title {
        font-size: 2.7rem;
        text-align: center;
        font-weight: 800;
        line-height: 1.2;
        color: var(--txt-main);
        margin-bottom: 0.25rem;
    }

    .subtitle {
        text-align: center;
        color: var(--txt-subtle);
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
    }

    .section-card {
        background: var(--bg-panel);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1rem 0.5rem 1rem;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.2);
    }

    .city-chip {
        display: inline-block;
        background: var(--brand-soft);
        color: var(--brand);
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.3rem 0.8rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: var(--bg-soft);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--txt-subtle);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--txt-main);
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
df = pd.read_csv("data.csv")
model = pickle.load(open("model.pkl", "rb"))
df["full_date"] = pd.to_datetime(df["full_date"])


@st.cache_data
def load_delhi_area_aqi():
    month_to_num = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    files = glob.glob("AQI_daily_2024_*_Delhi_*.xlsx")
    rows = []

    for file_path in files:
        base = os.path.basename(file_path)
        area = base.replace("AQI_daily_2024_", "")
        area = area.split("_Delhi_")[0].replace("_", " ").strip()

        wide = pd.read_excel(file_path)
        long_df = wide.melt(id_vars="Day", var_name="MonthName", value_name="AQI")
        long_df["Area"] = area
        long_df["Month"] = long_df["MonthName"].map(month_to_num)
        long_df["Year"] = 2024
        long_df["Date"] = pd.to_datetime(
            {"year": long_df["Year"], "month": long_df["Month"], "day": long_df["Day"]},
            errors="coerce",
        )
        rows.append(long_df[["Area", "Date", "Day", "Month", "AQI"]])

    if not rows:
        return pd.DataFrame(columns=["Area", "Date", "Day", "Month", "AQI"])

    area_df = pd.concat(rows, ignore_index=True)
    area_df = area_df.dropna(subset=["Date", "AQI"])
    area_df["AQI"] = pd.to_numeric(area_df["AQI"], errors="coerce")
    area_df = area_df.dropna(subset=["AQI"]).sort_values(["Area", "Date"])
    return area_df


area_data = load_delhi_area_aqi()

# ---------------- TITLE ---------------- #
st.markdown('<div class="main-title">🌫 NCR Air Quality Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time weather, pollution signals, and ML-based AQI insights for NCR cities.</div>', unsafe_allow_html=True)

# ---------------- NCR CITY MAP ---------------- #
city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Gurgaon": (28.4595, 77.0266),
    "Ghaziabad": (28.6692, 77.4538),
    "Faridabad": (28.4089, 77.3178)
}

city = st.selectbox("📍 Select NCR City", list(city_coords.keys()))
LAT, LON = city_coords[city]

selected_area = None
selected_date = None
if city == "Delhi":
    st.subheader("🏙 Delhi Area-wise AQI")
    if area_data.empty:
        st.info("No Delhi area-wise Excel files found.")
    else:
        all_areas = sorted(area_data["Area"].unique().tolist())
        selected_area = st.selectbox("Select Delhi Area", all_areas)
        selected_date = st.date_input("Select Date (2024)", value=datetime(2024, 1, 1))

# ---------------- API ---------------- #
def get_data(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]

    w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    p_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    w_res = requests.get(w_url).json()
    p_res = requests.get(p_url).json()

    try:
        temp = w_res["main"]["temp"]
        humidity = w_res["main"]["humidity"]
        wind = w_res["wind"]["speed"]

        comp = p_res["list"][0]["components"]
        api_aqi = p_res["list"][0]["main"]["aqi"]

        pollution = {
            "pm25": comp["pm2_5"],
            "pm10": comp["pm10"],
            "no2": comp["no2"],
            "so2": comp["so2"],
            "o3": comp["o3"]
        }

    except:
        st.error("API error. Using fallback values.")
        temp, humidity, wind = 25, 60, 5
        pollution = {"pm25":80,"pm10":120,"no2":40,"so2":10,"o3":30}
        api_aqi = 3

    return pollution, temp, humidity, wind, api_aqi

# ---------------- AQI CONVERSION ---------------- #
def convert_api_aqi(api_aqi):
    mapping = {
        1: 25,
        2: 75,
        3: 150,
        4: 250,
        5: 400
    }
    return mapping.get(api_aqi, 100)

# ---------------- AQI INFO ---------------- #
def aqi_info(aqi):
    if aqi <= 50:
        return "green", "Good 😊", "Safe for everyone."
    elif aqi <= 100:
        return "blue", "Moderate 😐", "Sensitive people be cautious."
    elif aqi <= 200:
        return "orange", "Poor 😷", "Asthma patients avoid outdoor activity."
    elif aqi <= 300:
        return "red", "Very Poor 🚨", "Stay indoors."
    else:
        return "darkred", "Severe ☠️", "Health emergency!"

# Fetch defaults for the selected city so user can edit pollutant inputs manually.
pollution_defaults, temp_defaults, humidity_defaults, wind_defaults, _ = get_data(LAT, LON)

st.subheader("🧪 Enter Pollutants (Manual)")
in1, in2, in3 = st.columns(3)
with in1:
    input_pm25 = st.number_input("PM2.5", min_value=0.0, value=float(pollution_defaults["pm25"]), step=1.0)
    input_pm10 = st.number_input("PM10", min_value=0.0, value=float(pollution_defaults["pm10"]), step=1.0)
with in2:
    input_no2 = st.number_input("NO2", min_value=0.0, value=float(pollution_defaults["no2"]), step=1.0)
    input_so2 = st.number_input("SO2", min_value=0.0, value=float(pollution_defaults["so2"]), step=1.0)
with in3:
    input_o3 = st.number_input("Ozone (O3)", min_value=0.0, value=float(pollution_defaults["o3"]), step=1.0)
    input_co = st.number_input("CO", min_value=0.0, value=0.5, step=0.1, format="%.2f")

# ---------------- BUTTON ---------------- #
if st.button("🚀 Analyze Air Quality"):

    _, temp, humidity, wind, api_aqi = get_data(LAT, LON)
    pollution = {
        "pm25": float(input_pm25),
        "pm10": float(input_pm10),
        "no2": float(input_no2),
        "so2": float(input_so2),
        "o3": float(input_o3),
        "co": float(input_co),
    }

    # Convert AQI
    city_adjustment = {
    "Delhi": 20,
    "Noida": 10,
    "Gurgaon": -10,
    "Ghaziabad": 15,
    "Faridabad": 5
}

    real_api_aqi = convert_api_aqi(api_aqi) + city_adjustment[city]
    area_aqi_for_date = None
    if city == "Delhi" and selected_area and selected_date:
        selected_ts = pd.to_datetime(selected_date)
        match = area_data[
            (area_data["Area"] == selected_area) &
            (area_data["Date"] == selected_ts)
        ]
        if not match.empty:
            area_aqi_for_date = float(match.iloc[0]["AQI"])

    st.markdown(f'<div class="city-chip">Selected city: {city}</div>', unsafe_allow_html=True)

    # -------- WEATHER -------- #
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"🌤 Weather Snapshot - {city}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature (°C)", f"{temp:.1f}")
    c2.metric("Humidity (%)", f"{humidity:.0f}")
    c3.metric("Wind Speed (m/s)", f"{wind:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------- POLLUTION -------- #
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"🌫 Pollutant Levels - {city}")
    p1, p2, p3 = st.columns(3)
    p1.metric("PM2.5", f'{pollution["pm25"]:.1f}')
    p2.metric("PM10", f'{pollution["pm10"]:.1f}')
    p3.metric("NO2", f'{pollution["no2"]:.1f}')

    p4, p5 = st.columns(2)
    p4.metric("SO2", f'{pollution["so2"]:.1f}')
    p5.metric("Ozone", f'{pollution["o3"]:.1f}')
    st.markdown('</div>', unsafe_allow_html=True)

    # -------- DATE -------- #
    today = datetime.now()
    day = today.day
    month = today.month
    year = today.year
    weekday = today.weekday()

    # -------- MODEL INPUT -------- #
    model_area = selected_area if city == "Delhi" and selected_area else city
    input_data = pd.DataFrame([[
        day, month, year, 0, weekday,
        pollution["pm25"],
        pollution["pm10"],
        pollution["no2"],
        pollution["so2"],
        pollution["co"],
        pollution["o3"],
        temp, humidity, wind,
        model_area
    ]], columns=[
        "Date","Month","Year","Holidays_Count","Days",
        "PM2.5","PM10","NO2","SO2","CO","Ozone",
        "Temperature","Humidity","WindSpeed","Area"
    ])

    prediction = model.predict(input_data)[0]

    # -------- FINAL AQI (HYBRID) -------- #
    # If Delhi area-wise AQI is available for selected date, include it.
    if area_aqi_for_date is not None:
        final_aqi = (0.5 * prediction) + (0.2 * real_api_aqi) + (0.3 * area_aqi_for_date)
    else:
        final_aqi = (0.7 * prediction) + (0.3 * real_api_aqi)

    # -------- DISPLAY -------- #
    st.subheader("📊 AQI Comparison")
    a1, a2, a3 = st.columns(3)
    a1.metric("🌐 API AQI (City-adjusted)", round(real_api_aqi, 2))
    a2.metric("🤖 ML AQI", round(prediction, 2))
    a3.metric("🧮 Final Hybrid AQI", round(final_aqi, 2))
    if area_aqi_for_date is not None:
        st.metric("📍 Delhi Area AQI (dataset)", round(area_aqi_for_date, 2))
    elif city == "Delhi" and selected_area:
        st.caption("No area AQI found for selected date; hybrid uses ML + API only.")

    # -------- MAIN AQI CARD -------- #
    color, label, advice = aqi_info(final_aqi)

    st.markdown(f"""
    <div class="result-card" style="background:{color};">
        <h1 style="margin:0; color:white;">AQI: {round(final_aqi,2)}</h1>
        <h3 style="margin-top:8px; color:white;">{label}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🚨 Health Advisory")
    st.warning(advice)

# ---------------- CHARTS ---------------- #
st.subheader("📊 NCR Historical Trends")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.3))
    ax1.plot(df["full_date"], df["AQI"], color="#1d4ed8", linewidth=2.2)
    ax1.set_title("AQI Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("AQI")
    ax1.grid(alpha=0.25)
    st.pyplot(fig1)

with col2:
    monthly = df.groupby("Month")["AQI"].mean()
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.3))
    ax2.plot(monthly.index, monthly.values, color="#0f766e", marker="o", linewidth=2)
    ax2.set_title("Monthly AQI Trend")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average AQI")
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax3)
ax3.set_title("Feature Correlation Heatmap")
st.pyplot(fig3)

if not area_data.empty and selected_area:
    st.subheader(f"📍 {selected_area} (Delhi) - 2024 AQI Trend")
    area_series = area_data[area_data["Area"] == selected_area].sort_values("Date")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(area_series["Date"], area_series["AQI"], color="#7c3aed", linewidth=2)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("AQI")
    ax4.grid(alpha=0.25)
    st.pyplot(fig4)