from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load your trained model
model = load_model("Blood Cell.h5")

# Class names based on training
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file to static folder with a unique name
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join('static', filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    # Send result to HTML
    return render_template("result.html", prediction=predicted_class, image_path=filename)

if __name__ == "__main__":
    app.run(debug=True)
