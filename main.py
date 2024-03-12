import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('/content/drug.csv')

# Split data into features and target variable
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate model

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example of making a prediction for new data
# Replace the values below with the actual patient data
new_data = pd.DataFrame({
    'itching': [1],
    'skin_rash': [1],
    'nodal_skin_eruptions': [1],
    'continuous_sneezing': [0],
    'shivering': [0],
    'chills': [0],
    'stomach_pain': [0],
    'ulcers_on_tongue': [0],
    'vomiting': [0],
    'weight_loss': [0],
    'restlessness': [0],
    'irregular_sugar_level': [0],
    'cough': [0],
    'high_fever': [0],
    'breathlessness': [0],
    'headache': [0],
    'yellowish_skin': [0],
    'dark_urine': [0],
    'loss_of_appetite': [0],
    'abdominal_pain': [0],
    'diarrhoea': [0],
    'yellow_urine': [0],
    'runny_nose': [0],
    'chest_pain': [0],
    'fast_heart_rate': [0],
    'excessive_hunger': [0],
    'muscle_pain': [0],
    'red_spots_over_body': [0],
    'increased_appetite': [0]
})

predicted_drug = rf_classifier.predict(new_data)
print("Predicted drug for the patient:", predicted_drug[0])
