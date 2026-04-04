import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.set_page_config(page_title="NCR AQI Dashboard", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#2c3e50;
}
.sub-text {
    text-align:center;
    color:gray;
}
.card {
    padding:15px;
    border-radius:10px;
    background-color:#f5f5f5;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
df = pd.read_csv("processed_data.csv")
model = pickle.load(open("model.pkl", "rb"))
df["full_date"] = pd.to_datetime(df["full_date"])

# ---------------- WEATHER ---------------- #
def get_weather():
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    city = "Delhi"

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return 25, 60, 5

    return data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"]

temp, humidity, wind = get_weather()

# ---------------- TITLE ---------------- #
st.markdown('<p class="big-title">🌫 NCR Air Quality Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Real-time AQI Prediction using ML + Weather Data</p>', unsafe_allow_html=True)

# ---------------- WEATHER CARDS ---------------- #
st.subheader("🌤 Current Weather")

c1, c2, c3 = st.columns(3)
c1.metric("🌡 Temperature (°C)", temp)
c2.metric("💧 Humidity (%)", humidity)
c3.metric("🌬 Wind Speed (m/s)", wind)

# ---------------- CHARTS ---------------- #
st.subheader("📊 Pollution Trends")

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
    ax2.set_title("Monthly AQI Trend")
    st.pyplot(fig2)

# Heatmap full width
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, ax=ax3)
st.pyplot(fig3)

# ---------------- INPUT SECTION ---------------- #
st.subheader("🔮 Predict AQI")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)")
        pm10 = st.number_input("PM10 (µg/m³)")

    with col2:
        no2 = st.number_input("NO2 (µg/m³)")
        so2 = st.number_input("SO2 (µg/m³)")

    with col3:
        co = st.number_input("CO (mg/m³)")
        ozone = st.number_input("Ozone (µg/m³)")

st.markdown("### 📅 Date Info")

col4, col5, col6 = st.columns(3)

with col4:
    month = st.slider("Month", 1, 12)
with col5:
    day = st.slider("Day", 1, 31)
with col6:
    year = st.number_input("Year", value=2024)

holiday = st.number_input("Holiday Count", value=0)
days = st.slider("Day of Week (1=Mon)", 1, 7)

# ---------------- AQI COLOR FUNCTION ---------------- #
def get_aqi_color(aqi):
    if aqi <= 50:
        return "green", "Good 😊"
    elif aqi <= 100:
        return "blue", "Moderate 😐"
    elif aqi <= 200:
        return "orange", "Poor 😷"
    else:
        return "red", "Severe 🚨"

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict AQI"):

    input_data = pd.DataFrame([[
        day, month, year, holiday, days,
        pm25, pm10, no2, so2, co, ozone,
        temp, humidity, wind
    ]], columns=[
        "Date","Month","Year","Holidays_Count","Days",
        "PM2.5","PM10","NO2","SO2","CO","Ozone",
        "Temperature","Humidity","WindSpeed"
    ])

    prediction = model.predict(input_data)[0]

    color, label = get_aqi_color(prediction)

    st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:white;">AQI: {round(prediction,2)}</h2>
        <h3 style="color:white;">{label}</h3>
    </div>
    """, unsafe_allow_html=True)