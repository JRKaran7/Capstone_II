import streamlit as st

def get_recommendations():
    st.subheader("Personalized Travel Recommendations")
    st.write("Answer a few questions to find your ideal travel package.")

    destination = st.selectbox("Pick your travel destination", ["Paris", "Tokyo", "Sydney"])
    preference = st.radio("What kind of experience do you prefer?", ["Adventure", "Relaxation", "Culture"])

    st.write(f"Based on your choice of {destination} and preference for {preference}:")
    st.success(f"We recommend a {preference} package to {destination}!")
