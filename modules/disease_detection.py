import os
import json
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = os.path.join("models", "crop_disease_model.h5")
CLASS_INDICES_PATH = os.path.join("database", "class_indices.json")
DISEASE_INFO_PATH = os.path.join("database", "disease_info.json")

_model = None
_class_indices = None
_disease_info = None

def load_resources():
    global _model, _class_indices, _disease_info
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        
        with open(CLASS_INDICES_PATH, "r") as f:
            _class_indices = json.load(f)
            
        with open(DISEASE_INFO_PATH, "r") as f:
            _disease_info = json.load(f)

def is_plant_like_image(img_pil):
    """
    Analyzes image color spectrum to filter out timing diagrams, schematics,
    charts, and non-plant objects.
    """
    img_hsv = img_pil.convert("HSV")
    hsv_array = np.array(img_hsv)
    
    h = hsv_array[:, :, 0] # Hue (0 - 255)
    s = hsv_array[:, :, 1] # Saturation (0 - 255)
    
    # Saturation check for diagrams and drawings
    if np.mean(s) < 14.0:
        return False
        
    # Plant/Leaf Hue Range Check (Green, Yellow, Golden, Brown)
    plant_hue_pixels = np.sum((h >= 15) & (h <= 115) & (s > 20))
    if (plant_hue_pixels / h.size) < 0.05:
        return False
        
    return True

def predict_crop_disease(image_input):
    """
    Predicts disease across all supported crops, field plants, and fruit trees.
    """
    load_resources()
    
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input
        
    img_oriented = ImageOps.exif_transpose(img)
    img_rgb = img_oriented.convert("RGB")
    
    # Step 1: Reject diagrams, schematics, or non-plant images
    if not is_plant_like_image(img_rgb):
        return {
            "is_non_plant": True,
            "is_uncertain": True,
            "confidence": 0.0,
            "warning_title": "⚠️ Non-Plant Image Detected",
            "warning_message": "The uploaded file appears to be a diagram, schematic, chart, or non-plant object. Please upload a clear photograph of a real crop leaf or plant sample."
        }
    
    # Step 2: Preprocess and Run AI Model Prediction
    img_resized = img_rgb.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = _model.predict(img_array, verbose=0)
    probs = predictions[0]
    
    predicted_idx_int = int(np.argmax(probs))
    predicted_idx_str = str(predicted_idx_int)
    confidence = float(probs[predicted_idx_int] * 100)
    
    raw_class_name = _class_indices.get(predicted_idx_str, "Unknown")
    
    is_uncertain = confidence < 45.0
    info = _disease_info.get(raw_class_name, None)
    
    # Proper Parsing for both Standard (Plant___Disease) and Non-standard Kaggle folder names
    parts = raw_class_name.split("___")
    if len(parts) > 1:
        crop_name = parts[0].replace("_", " ")
        condition_name = parts[1].replace("_", " ")
    else:
        crop_name = "Crop Sample"
        condition_name = raw_class_name.replace("_", " ")
    
    if info:
        disease_display_name = info.get("disease_name", f"{crop_name} - {condition_name}")
        symptoms = info.get("symptoms", "Visual symptom analysis available for this condition.")
        cause = info.get("cause", "Pathogen activity identified on foliage.")
        treatment = info.get("treatment", "Apply standard recommended management practices.")
    else:
        disease_display_name = f"{crop_name} - {condition_name}"
        symptoms = f"Visual symptoms observed on foliage matching {condition_name}."
        cause = f"Condition or stress related to {condition_name}."
        treatment = "Inspect foliage, isolate affected leaves, and apply appropriate crop protection."
    
    return {
        "is_non_plant": False,
        "is_uncertain": is_uncertain,
        "raw_class_name": raw_class_name,
        "crop": crop_name,
        "condition": condition_name,
        "full_disease_name": disease_display_name,
        "confidence": confidence,
        "symptoms": symptoms,
        "cause": cause,
        "treatment": treatment
    }