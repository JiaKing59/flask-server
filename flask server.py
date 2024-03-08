from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
import os
import datetime
import uuid
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and set it to evaluation mode
model = torch.jit.load("model4.pt", map_location=torch.device('cpu'))
model.eval()

# Load class names from CSV file
def load_class_names(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        class_names = [row['Label'] for row in reader]
    return class_names

# Specify the path to your CSV file
csv_file_path = 'labels4.csv'

# Define class names
class_names = load_class_names(csv_file_path)

# Define the upload folder for saving images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(file_path):
    # Load and preprocess the image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Set the upload folder
        app.config['UPLOAD_FOLDER'] = 'uploads'

        # Ensure the upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Generate a unique filename based on timestamp and/or random string
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        random_string = str(uuid.uuid4())[:8]
        unique_filename = f"temp_image_{timestamp}_{random_string}.jpg"

        # Save the image to the upload folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(image_path)

        # Preprocess the image
        image = preprocess_image(image_path)

        # Make prediction
        with torch.no_grad():
            prediction = model(image)

        # Obtain class probabilities
        class_probabilities = torch.nn.functional.softmax(prediction, dim=1)

        # Get the predicted class index and associated probability
        predicted_class_index = prediction.argmax().item()
        predicted_class_probability = class_probabilities[0][predicted_class_index].item()

        # Get the predicted class name
        prediction_name = class_names[predicted_class_index]

        # Define image_url
        image_url = f"{request.url_root}uploads/{unique_filename}"

        # Check if the probability is less than 50%
        if predicted_class_probability < 0.5:
            return jsonify({'prediction': 'Cannot detect flower', 'image_url': image_url})

        # Return the URL of the saved image
        return jsonify({'prediction': prediction_name, 'image_url': image_url})

    except Exception as e:
        return jsonify({'error': f'Internal server error: {e}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)