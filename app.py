from flask import Flask, jsonify, render_template
import cv2
import requests
import json

app = Flask(__name__)

# Azure Custom Vision endpoint and prediction key
ENDPOINT = "https://asedetection-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/cb499bb3-445c-4295-8689-d8f584dea194/classify/iterations/Iteration3/image"
PREDICTION_KEY = "470836658fc844cb9f9416498467580d"
PROJECT_ID = "cb499bb3-445c-4295-8689-d8f584dea194"

# Function to detect gender and age using Azure Custom Vision
def detect_gender_age(image):
    headers = {
        'Prediction-Key': PREDICTION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'projectId': PROJECT_ID
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, params=params, data=image)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        result = response.json()
        # Extract gender and age predictions
        predictions = result.get('predictions', [])
        gender = next((pred['tagName'] for pred in predictions if pred['tagName'].lower() in ['male', 'female']), None)
        age = next((pred['tagName'] for pred in predictions if pred['tagName'].lower() not in ['male', 'female']), None)
        return gender, age
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Unknown', 'Unknown'

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to get gender and age predictions
@app.route('/predict')
def predict():
    # Capture frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Convert the frame to binary
    _, image = cv2.imencode('.jpg', frame)
    image_data = image.tobytes()

    # Detect gender and age
    gender, age = detect_gender_age(image_data)

    # Return JSON response
    return jsonify({"gender": gender, "age": age})

if __name__ == '__main__':
    app.run(debug=True)
