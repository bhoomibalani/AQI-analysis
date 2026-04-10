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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NCR AQI Dashboard", layout="wide")

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.main-title {
    font-size: 45px;
    text-align: center;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
df = pd.read_csv("data.csv")
model = pickle.load(open("model.pkl", "rb"))
df["full_date"] = pd.to_datetime(df["full_date"])

# ---------------- TITLE ---------------- #
st.markdown('<div class="main-title">🌫 NCR Air Quality Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Pollution + ML Prediction</div>', unsafe_allow_html=True)

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

# ---------------- BUTTON ---------------- #
if st.button("🚀 Analyze Air Quality"):

    pollution, temp, humidity, wind, api_aqi = get_data(LAT, LON)

    # Convert AQI
    city_adjustment = {
    "Delhi": 20,
    "Noida": 10,
    "Gurgaon": -10,
    "Ghaziabad": 15,
    "Faridabad": 5
}

    real_api_aqi = convert_api_aqi(api_aqi) + city_adjustment[city]

    # -------- WEATHER -------- #
    st.subheader(f"🌤 Weather in {city}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temp (°C)", temp)
    c2.metric("Humidity (%)", humidity)
    c3.metric("Wind (m/s)", wind)

    # -------- POLLUTION -------- #
    st.subheader(f"🌫 Pollution in {city}")
    p1, p2, p3 = st.columns(3)
    p1.metric("PM2.5", pollution["pm25"])
    p2.metric("PM10", pollution["pm10"])
    p3.metric("NO2", pollution["no2"])

    p4, p5 = st.columns(2)
    p4.metric("SO2", pollution["so2"])
    p5.metric("Ozone", pollution["o3"])

    # -------- DATE -------- #
    today = datetime.now()
    day = today.day
    month = today.month
    year = today.year
    weekday = today.weekday()

    # -------- MODEL INPUT -------- #
    input_data = pd.DataFrame([[
        day, month, year, 0, weekday,
        pollution["pm25"],
        pollution["pm10"],
        pollution["no2"],
        pollution["so2"],
        0.5,
        pollution["o3"],
        temp, humidity, wind
    ]], columns=[
        "Date","Month","Year","Holidays_Count","Days",
        "PM2.5","PM10","NO2","SO2","CO","Ozone",
        "Temperature","Humidity","WindSpeed"
    ])

    prediction = model.predict(input_data)[0]

    # -------- FINAL AQI (HYBRID) -------- #
    final_aqi = (0.6 * real_api_aqi) + (0.4 * prediction)

    # -------- DISPLAY -------- #
    st.subheader("📊 AQI Comparison")
    a1, a2 = st.columns(2)
    a1.metric("🌐 API AQI (Real Scale)", real_api_aqi)
    a2.metric("🤖 ML AQI", round(prediction, 2))

    # -------- MAIN AQI CARD -------- #
    color, label, advice = aqi_info(final_aqi)

    st.markdown(f"""
    <div style="background:{color}; padding:25px; border-radius:15px; text-align:center;">
        <h1 style="color:white;">AQI: {round(final_aqi,2)}</h1>
        <h2 style="color:white;">{label}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🚨 Health Advisory")
    st.warning(advice)

# ---------------- CHARTS ---------------- #
st.subheader("📊 NCR Historical Trends")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(df["full_date"], df["AQI"])
    ax1.set_title("AQI Over Time")
    st.pyplot(fig1)

with col2:
    monthly = df.groupby("Month")["AQI"].mean()
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly.index, monthly.values)
    ax2.set_title("Monthly Trend")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, ax=ax3)
st.pyplot(fig3)