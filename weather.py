import streamlit as st
import requests
from bs4 import BeautifulSoup

def fetch_weather(location):
    try:
        url = f"https://www.weather-forecast.com/locations/{location}/forecasts/latest"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        weather_info = soup.find("span", {"class": "phrase"})
        return weather_info.text if weather_info else "Weather data unavailable"
    except Exception as e:
        return f"Error fetching weather: {e}"

def show_weather():
    st.subheader("Check Weather")
    location = st.text_input("Enter Destination (e.g., Paris)")
    if location:
        location = location.replace(" ", "-")
        weather = fetch_weather(location)
        st.write(f"Weather for {location}: {weather}")
