import streamlit as st
from trivia import show_trivia
from gamification import show_rewards
from recommendations import get_recommendations  # Ensure this is importing the correct function
from chatbot import start_chatbot
from group_planning import group_planning
from virtual_souvenirs import show_souvenirs
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to train model
def train_model(data):
    # Encode categorical variables
    label_encoders = {}
    for column in ["Weather", "Budget Level"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Features and target variable
    X = data[["Weather", "Budget Level", "Average Trip Cost ($)", "Rating"]]
    y = data["Package ID"]

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return model, label_encoders

# Function to get recommendations based on user inputs
def get_recommendations():
    # Load dataset and model
    data_file = "travel_packages.csv"  # Replace with the path to your dataset
    data = load_data(data_file)
    model, label_encoders = train_model(data)

    st.title("Travel Package Recommendation System")

    # User inputs
    weather = st.selectbox("Select Weather Preference", label_encoders["Weather"].classes_)
    budget_level = st.selectbox("Select Budget Level", label_encoders["Budget Level"].classes_)
    avg_cost = st.slider("Preferred Average Trip Cost ($)", data["Average Trip Cost ($)"].min(), data["Average Trip Cost ($)"].max(), 800)
    rating = st.slider("Minimum Rating (1-5)", data["Rating"].min(), data["Rating"].max(), 4.5)

    # Encode user inputs
    weather_encoded = label_encoders["Weather"].transform([weather])[0]
    budget_encoded = label_encoders["Budget Level"].transform([budget_level])[0]

    # Prepare input for model
    features = pd.DataFrame({
        "Weather": [weather_encoded],
        "Budget Level": [budget_encoded],
        "Average Trip Cost ($)": [avg_cost],
        "Rating": [rating]
    })

    # Predict and recommend package
    predicted_package = model.predict(features)
    recommended_activities = data[data["Package ID"] == predicted_package[0]]["Activities Included"].values[0]

    st.subheader("Recommended Package")
    st.write(f"We recommend the package **{predicted_package[0]}**:")
    st.success(f"Activities Included: {recommended_activities}")

# Main function to run the app
def main():
    st.title("AI-Driven Gamified Travel Advisor")
    menu = [
        "Home",
        "Trivia",
        "Rewards",
        "Recommendations",
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
        get_recommendations()  # This will now call the updated recommendations function
    elif choice == "Chatbot":
        start_chatbot()
    elif choice == "Group Travel Planning":
        group_planning()
    elif choice == "Virtual Souvenirs":
        show_souvenirs()

if __name__ == "__main__":
    main()
