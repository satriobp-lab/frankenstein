from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import json
import base64

app = Flask(__name__)

# Load primary Model 
model1 = load_model('modelCNN_atas.h5', compile=False)
model2 = load_model('modelDenseNet121_bawah.h5', compile=False)
model3 = load_model('modelVGG16_samping.h5', compile=False)

# Load models on startup
# load_models()

@app.route('/')
def index():
    return "Halloooo", 200

@app.route("/getpredict", methods=["POST"])
def predict_api():
    
    file1 = request.files['atas']
    file2 = request.files['bawah']
    file3 = request.files['samping']

    # Check if any file is not selected for upload
    if any(file.filename == '' for file in [file1, file2, file3]):
        return jsonify({"msg": 'One or more files are not selected for upload'}), 400
    
    try:
        # Perform predictions using all three models

        # Process file1
        img1 = Image.open(file1)
        img1 = img1.resize((224, 224))
        img_array1 = np.array(img1)
        img_array1 = np.expand_dims(img_array1, axis=0)
        prediction_1 = model1.predict(img_array1)
        predicted_class_1 = interpret_class(prediction_1)

        # Process file2
        img2 = Image.open(file2)
        img2 = img2.resize((224, 224))
        img_array2 = np.array(img2)
        img_array2 = np.expand_dims(img_array2, axis=0)
        prediction_2 = model2.predict(img_array2)
        predicted_class_2 = interpret_class(prediction_2)

        # Process file3
        img3 = Image.open(file3)
        img3 = img3.resize((224, 224))
        img_array3 = np.array(img3)
        img_array3 = np.expand_dims(img_array3, axis=0)
        prediction_3 = model3.predict(img_array3)
        predicted_class_3 = interpret_class(prediction_3)

        # Determine the final prediction based on individual predictions
        final_prediction = determine_final_prediction(predicted_class_1, predicted_class_2, predicted_class_3)

        # You can further process the results or construct the response as needed

        response = {
            'predicted_class_1': predicted_class_1,
            'predicted_class_2': predicted_class_2,
            'predicted_class_3': predicted_class_3,
            'final_prediction': final_prediction,
            # You can add more fields or modify the response as needed
        }
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error performing predictions: {str(e)}")
        return jsonify({"msg": "Internal Server Error"}), 500

def interpret_class(prediction):
    # Define a threshold for interpretation
    threshold = 0.5

    # Check if the predicted probability is above the threshold
    if prediction[0][0] > threshold:
        return "Abnormal"
    else:
        return "Normal"

def determine_final_prediction(predicted_class_1, predicted_class_2, predicted_class_3):
    # If any prediction is abnormal, the final prediction is abnormal
    if 'Abnormal' in [predicted_class_1, predicted_class_2, predicted_class_3]:
        return 'Abnormal'
    else:
        return 'Normal'

if __name__ == '__main__':
    app.run(debug=True, port=1112)