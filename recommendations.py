import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Train the model
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, label_encoders

# Streamlit application
def get_recommendations(model, label_encoders, data):
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

# Main function
def main():
    # Load dataset
    data_file = "Lakshadweep_Travel_Packages.csv"  # Replace with the path to your dataset
    data = load_data(data_file)

    # Train the model
    model, label_encoders = train_model(data)

    # Display recommendations
    get_recommendations(model, label_encoders, data)

# Run the application
if __name__ == "__main__":
    main()
