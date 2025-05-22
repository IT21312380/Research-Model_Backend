import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Load the precomputed class features
with open("class_features.pkl", "rb") as f:
    class_features = pickle.load(f)

# Load the VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=feature_extractor.input, outputs=feature_extractor.output)

def extract_features_from_bytes(image_bytes):
    """Extracts features from an image file given in bytes format."""
    img = image.load_img(BytesIO(image_bytes), target_size=(224, 224))  # Load image from bytes

    # Convert image to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Extract features and flatten
    features = feature_extractor.predict(img_array)
    features = features.flatten()

    return features

def predict_aggressive_animal(image_bytes):
    """Predicts the class of an image using cosine similarity."""
    image_features = extract_features_from_bytes(image_bytes)

    similarities = {}
    for class_name, class_feature in class_features.items():
        similarity = cosine_similarity(image_features.reshape(1, -1), class_feature.reshape(1, -1))
        similarities[class_name] = float(similarity[0][0])  # Convert to Python float

    predicted_class = max(similarities, key=similarities.get)
    similarity_score = similarities[predicted_class]

    return predicted_class, similarity_score
