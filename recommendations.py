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

# Streamlit app
def main():
    st.title("Travel Package Recommendation System")

    # Load dataset
    data_file = "Lakshadweep_Travel_Packages.csv"  # Replace with the path to your dataset
    data = load_data(data_file)

    # Train the model
    model, label_encoders = train_model(data)

    # User inputs
    st.header("Enter Your Preferences")

    # Weather preference
    weather_options = label_encoders["Weather"].classes_
    weather_choice = st.selectbox("Select Weather Preference:", weather_options)
    weather_encoded = label_encoders["Weather"].transform([weather_choice])[0]

    # Budget level
    budget_options = label_encoders["Budget Level"].classes_
    budget_choice = st.selectbox("Select Budget Level:", budget_options)
    budget_encoded = label_encoders["Budget Level"].transform([budget_choice])[0]

    # Average trip cost
    avg_cost = st.slider(
        "Preferred Average Trip Cost ($):",
        min_value=float(data["Average Trip Cost ($)"].min()),
        max_value=float(data["Average Trip Cost ($)"].max()),
        value=float(data["Average Trip Cost ($)"].mean())
    )

    # Minimum rating
    rating = st.slider("Minimum Rating:", min_value=1.0, max_value=5.0, step=0.1, value=4.0)

    # Prepare input for model
    features = pd.DataFrame({
        "Weather": [weather_encoded],
        "Budget Level": [budget_encoded],
        "Average Trip Cost ($)": [avg_cost],
        "Rating": [rating]
    })

    # Predict and recommend package
    if st.button("Get Recommendation"):
        predicted_package = model.predict(features)
        recommended_activities = data[data["Package ID"] == predicted_package[0]]["Activities Included"].values[0]

        st.subheader("Recommended Package")
        st.write(f"We recommend the package **{predicted_package[0]}**:")
        st.write(f"Activities Included: {recommended_activities}")

# Run the application
if __name__ == "__main__":
    main()
