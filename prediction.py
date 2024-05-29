from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# load model, specify which version 
model_save_dir = r"FOLDER-WHERE-MODEL-LOCATED-HERE"
version = 1.0 # edit this to the current version you have developed
model_save_path = os.path.join(model_save_dir, f'stm-v.{version}.keras')
model = load_model(model_save_path)

# load json file with indices because i fucked up and didnt save the class indices in the model training
class_indices_path = os.path.join(model_save_dir, 'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
class_indices = {v: k for k, v in class_indices.items()}  # reverse the class indices dictionary

def predict_image(image_path):
    img_width, img_height = 224, 224  # this is needed because the model was trained to analyze images of this dimension
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    predicted_class = class_indices[predicted_class_index]
    
    print(f"Predicted class: {predicted_class}")


test_image_path = r"IMAGE-TO-BE-PREDICTED-HERE" 
predict_image(test_image_path)
