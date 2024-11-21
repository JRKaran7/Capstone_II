import streamlit as st
from trivia import show_trivia
from gamification import show_rewards
from recommendations import get_recommendations
from weather import show_weather
from chatbot import start_chatbot
from group_planning import group_planning
from virtual_souvenirs import show_souvenirs

def main():
    st.title("AI-Driven Gamified Travel Advisor")
    menu = [
        "Home",
        "Trivia",
        "Rewards",
        "Recommendations",
        "Weather",
        "Chatbot",
        "Group Travel Planning",
        "Virtual Souvenirs",
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Travel Advisor!")
        st.write("Plan your trips with fun and engaging features.")
    elif choice == "Trivia":
        show_trivia()
    elif choice == "Rewards":
        show_rewards()
    elif choice == "Recommendations":
        get_recommendations()
    elif choice == "Weather":
        show_weather()
    elif choice == "Chatbot":
        start_chatbot()
    elif choice == "Group Travel Planning":
        group_planning()
    elif choice == "Virtual Souvenirs":
        show_souvenirs()

if __name__ == "__main__":
    main()
