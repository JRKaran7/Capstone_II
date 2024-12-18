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

# Function to get recommendations based on user input
def get_recommendations(model, label_encoders, data):
    print("Travel Package Recommendation System")

    # User inputs
    print("Select Weather Preference:")
    for i, weather in enumerate(label_encoders["Weather"].classes_):
        print(f"{i + 1}. {weather}")
    weather_choice = int(input("Enter choice (1-{}): ".format(len(label_encoders["Weather"].classes_)))) - 1
    weather = label_encoders["Weather"].classes_[weather_choice]

    print("Select Budget Level:")
    for i, budget in enumerate(label_encoders["Budget Level"].classes_):
        print(f"{i + 1}. {budget}")
    budget_choice = int(input("Enter choice (1-{}): ".format(len(label_encoders["Budget Level"].classes_)))) - 1
    budget_level = label_encoders["Budget Level"].classes_[budget_choice]

    avg_cost = float(input(f"Preferred Average Trip Cost ($) (Min: {data['Average Trip Cost ($)'].min()}, Max: {data['Average Trip Cost ($)'].max()}): "))
    rating = float(input(f"Minimum Rating (1-5): "))

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

    print("\nRecommended Package")
    print(f"We recommend the package **{predicted_package[0]}**:")
    print(f"Activities Included: {recommended_activities}")

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
