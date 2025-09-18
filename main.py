from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uvicorn

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "crop_disease_model.keras"
IMG_SIZE = 128

PLANT_DISEASES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DISEASE_DATABASE = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'description': 'Fungal infection caused by Venturia inaequalis affecting apple trees.',
        'symptoms': 'Dark green or brown spots with a velvety texture on leaves and fruit.',
        'treatment': 'Use fungicides, remove fallen leaves, and prune for better air circulation.'
    },
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight',
        'description': 'A fungal infection due to Alternaria solani that affects tomato plants.',
        'symptoms': 'Brown to black spots with rings appearing first on lower leaves.',
        'treatment': 'Remove infected leaves, apply fungicides, rotate crops, and maintain spacing.'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'description': 'A disease caused by Phytophthora infestans, linked to the Irish Potato Famine.',
        'symptoms': 'Water-soaked lesions, rapid leaf browning, and white fuzzy growth in humid weather.',
        'treatment': 'Apply fungicides, remove infected plants, and avoid excessive moisture.'
    },
    'Grape___Black_rot': {
        'name': 'Grape Black Rot',
        'description': 'A fungal disease from Guignardia bidwellii affecting grape vines.',
        'symptoms': 'Reddish-brown leaf spots turning black; shriveled, darkened fruit.',
        'treatment': 'Use fungicides, remove infected plant parts, and ensure proper pruning.'
    }
}

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model successfully loaded from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/")
async def home():
    return {
        "api_name": "Plant Disease Detection API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Upload an image for disease detection",
            "/": "GET - API information"
        }
    }

def fetch_disease_info(disease_code):
    if disease_code in DISEASE_DATABASE:
        return DISEASE_DATABASE[disease_code]
    else:
        name = disease_code.replace('___', ' - ').replace('_', ' ')
        return {
            'name': name,
            'description': 'No additional data available.',
            'symptoms': 'Consult an expert for diagnosis.',
            'treatment': 'Seek guidance from agricultural specialists.'
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image uploaded")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Try again later.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        disease_code = PLANT_DISEASES[predicted_class_idx]
        disease_info = fetch_disease_info(disease_code)
        
        return {
            "prediction": {
                "disease_code": disease_code,
                "disease_name": disease_info["name"],
                "confidence": round(confidence * 100, 2)
            },
            "details": disease_info,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
