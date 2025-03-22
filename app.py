import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load dataset
file_path = "diabetes.csv"
df = pd.read_csv(file_path)

# Preprocess data
X = df.drop(columns=['Outcome'])
y = df['Outcome']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump((model, scaler), "diabetes_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(x) for x in request.form.values()]
        input_features = np.array(features).reshape(1, -1)

        # Load model
        model, scaler = joblib.load("diabetes_model.pkl")
        input_features = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_features)[0]
        result = "Diabetic" if prediction == 1 else "Normal"

        return render_template('index.html', prediction_text=f'Result: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
