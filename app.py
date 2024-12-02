from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set memory growth or limit memory
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # Or use a hard memory limit
            # tf.config.experimental.set_virtual_device_configuration(
            #     device,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)])  # Memory limit in MiB
    except Exception as e:
        print(f"Error setting memory growth: {e}")

# Initialize Flask app
app = Flask(__name__)

# Load the model once when the app starts
MODEL_PATH = './dementia_classification_model/model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Open the image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Resize the image to match the model's input size
    x = np.array(img.resize((128,128)))

    # Convert the image to a NumPy array and normalize pixel values
    x = x.reshape(1,128,128,3)
    return x

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html', result=None)

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result="No file selected")

    try:
        # Read the image bytes
        image_bytes = file.read()

        # Preprocess the image
        image_array = preprocess_image(image_bytes)

        # Predict using the model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])  # Get class with highest probability
        confidence = predictions[0][predicted_class] * 100  # Confidence score

        # Map the prediction to a label
        LABELS = {
            0: 'Non Demented',
            1: 'Mild Dementia',
            2: 'Moderate Dementia',
            3: 'Very Mild Dementia'
        }
        label = LABELS.get(predicted_class, 'Error in Prediction')

        result = f"Prediction: {label} with {confidence:.2f}% confidence"

        return render_template('index.html', result=result)
    except Exception as e:
        print(e)
        return render_template('index.html', result="Failed to process image")

if __name__ == '__main__':
    app.run(debug=True)

