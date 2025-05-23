from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from io import BytesIO
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
from typing import Dict, Any
from database import get_database
from pydantic import BaseModel
from datetime import datetime,timedelta
from aggressive import predict_aggressive_animal

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize FastAPI
app = FastAPI()

# Enable CORS
#test2
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db, collection = get_database()


class AnimalData(BaseModel):
    estimatedAnimalLocation: Dict[str, float]
    class_name: str
    timestamp: str


# Load the trained classification model
try:
    MODEL = tf.keras.models.load_model("models/008_model.keras")
    CLASS_NAMES = ["Deer", "Elephant", "Leopard", "Peacock"]
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Load YOLOv8 for object detection
try:
    yolo_model = YOLO("yolov8n.pt")  # Ensure `yolov8n.pt` is in the working directory
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {str(e)}")


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"message": "Hello, I am alive"}


# Helper function to preprocess images for classification
def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        img = load_img(BytesIO(data), target_size=(256, 256))
        img_array = img_to_array(img)
        return preprocess_input(img_array)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict the class of an uploaded image."""
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        predictions = MODEL.predict(img_batch)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        if predicted_class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_class_idx]
            return {
                "class_index": int(predicted_class_idx),
                "class_name": predicted_class,
                "confidence": float(confidence) * 100
            }
        else:
            return {
                "class_index": -1,
                "class_name": "Difficult to identify",
                "confidence": 0.0
            }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/get_animal_height")
async def get_animal_height(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Detects the largest animal in an image using YOLO, estimates its height in pixels,
    and classifies the animal using the trained model.
    """
    try:
        # Read image
        contents = await file.read()
        image = np.array(cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR))

        # Perform object detection using YOLO
        results = yolo_model(image)
        detected_objects = results[0].boxes.data.cpu().numpy()

        if len(detected_objects) == 0:
            return {"error": "No animal detected"}

        # Find the largest detected object (assumed to be the main animal)
        largest_object = max(detected_objects, key=lambda obj: (obj[2] - obj[0]) * (obj[3] - obj[1]))

        x_min, y_min, x_max, y_max, confidence, class_id = largest_object
        animal_height_pixels = y_max - y_min  # Height in pixels

        # Crop the detected animal region
        cropped_animal = image[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Resize and preprocess the cropped image for classification
        cropped_animal = cv2.resize(cropped_animal, (256, 256))  # Resize to match model input size
        cropped_animal = img_to_array(cropped_animal)
        cropped_animal = preprocess_input(cropped_animal)
        img_batch = np.expand_dims(cropped_animal, axis=0)  # Add batch dimension

        # Predict animal class
        predictions = MODEL.predict(img_batch)
        predicted_class_idx = np.argmax(predictions[0])  # Get the class index
        class_confidence = np.max(predictions[0])  # Get the confidence value

        # Validate and map index to class name
        if predicted_class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_class_idx]
        else:
            predicted_class = "Unknown"

        return {
            "class_index": int(predicted_class_idx),
            "class_name": predicted_class,
            "classification_confidence": float(class_confidence) * 100,
            "height_pixels": int(animal_height_pixels),
            "detection_confidence": float(confidence) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting height and classifying animal: {str(e)}")

@app.post("/predict_aggressive_animal")
async def predict_aggressive_animal_api(file: UploadFile = File(...)) -> Dict[str, Any]:
    """API endpoint to predict the class of an uploaded animal image based on its features"""
    try:
        # Read image bytes from the uploaded file
        image_bytes = await file.read()

        # Call the predict_aggressive_animal function to get predictions
        predicted_class, similarity_score = predict_aggressive_animal(image_bytes)

        # Return the prediction and similarity score
        return {
            "predicted_class": predicted_class,
            "similarity_score": similarity_score


        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting aggressive animal: {str(e)}")

@app.post("/save_animal_data")
async def save_animal_data(data: AnimalData):
    try:
        # Log incoming data
        print(f"Received data: {data}")

        # Convert the timestamp string to datetime object if needed
        timestamp_str = data.timestamp

        # Handle the "Z" in ISO format, replace with "+00:00"
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"

        # Parse the timestamp string
        timestamp = datetime.fromisoformat(timestamp_str)

        # Adjust the timestamp to Sri Lankan time (UTC +5:30)
        sri_lankan_time = timestamp + timedelta(hours=5, minutes=30)

        # Update the timestamp in the data
        data.timestamp = sri_lankan_time

        # Convert the Pydantic model to a dictionary
        data_dict = data.dict()

        # Log the adjusted timestamp
        print(f"Adjusted Sri Lankan Time: {sri_lankan_time}")

        # Insert the data into MongoDB
        result = collection.insert_one(data_dict)

        # Log the result or return the inserted ID
        print(f"Data inserted with ID: {result.inserted_id}")

        return {"message": "Animal data saved successfully!", "inserted_id": str(result.inserted_id)}

    except Exception as e:
        # Log the exception details
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving data: {str(e)}")


@app.post("/get_animal_height1")
async def get_animal_height1(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Detects an animal using YOLOv8 and extracts its height in pixels.
    Then classifies the animal using the InceptionV3 model with proper validations.
    """
    try:
        # Read image
        contents = await file.read()
        image = np.array(cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR))

        # Perform object detection using YOLO
        results = yolo_model(image)
        detected_objects = results[0].boxes.data.cpu().numpy()

        if len(detected_objects) == 0:
            return {"error": "No animal detected"}

        # Valid YOLO class IDs for animals
        valid_animal_classes = {14, 18, 19, 20, 21}  # Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe

        # Find the largest valid detected animal
        largest_object = None
        for obj in detected_objects:
            x_min, y_min, x_max, y_max, confidence, class_id = obj
            if int(class_id) in valid_animal_classes:
                if largest_object is None or (x_max - x_min) * (y_max - y_min) > (
                        largest_object[2] - largest_object[0]) * (largest_object[3] - largest_object[1]):
                    largest_object = obj

        # If no valid animal is detected
        if largest_object is None:
            return {"error": "No valid animal detected"}

        x_min, y_min, x_max, y_max, confidence, class_id = largest_object
        animal_height_pixels = y_max - y_min  # Height in pixels

        # Crop the detected animal
        cropped_animal = image[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Resize and preprocess for classification
        resized_animal = cv2.resize(cropped_animal, (256, 256))
        input_arr = np.expand_dims(preprocess_input(resized_animal), axis=0)

        # Predict the class
        predictions = MODEL.predict(input_arr)
        predicted_class_idx = np.argmax(predictions)
        confidence_score = np.max(predictions)

        # Validate and map index to class name
        if predicted_class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_class_idx]
        else:
            predicted_class = "Unknown"

        return {
            "class_index": int(predicted_class_idx),
            "class_name": predicted_class,
            "classification_confidence": float(confidence_score) * 100,
            "height_pixels": int(animal_height_pixels),
            "detection_confidence": float(confidence) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting height and classifying animal: {str(e)}")

@app.get("/get_animal_data")
async def get_animal_data():
    try:
        # Fetch all animal data from the collection
        animal_data = list(collection.find({}, {"_id": 0}))  # Exclude _id field

        # Convert timestamp format
        for data in animal_data:
            if "timestamp" in data and isinstance(data["timestamp"], dict) and "$date" in data["timestamp"]:
                timestamp_ms = data["timestamp"]["$date"]["$numberLong"]
                timestamp = datetime.utcfromtimestamp(int(timestamp_ms) / 1000)
                
                sri_lankan_time = timestamp + timedelta(hours=5, minutes=30)
                data["timestamp"] = sri_lankan_time.isoformat()

        return animal_data

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")



import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))  # Fallback to 3000 for local
    uvicorn.run("main:app", host="0.0.0.0", port=port)

