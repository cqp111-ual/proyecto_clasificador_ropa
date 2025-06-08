from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import io
import os
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

### config
load_dotenv()

# support two different models
MODELS_DIR = os.getenv("MODELS_DIR", "../models")
DEEPFASHION_MODEL = os.getenv("DEEPFASHION_MODEL", "modelo_ropa_50clases.h5")
FASHION_MNIST_MODEL = os.getenv("FASHION_MNIST_MODEL", "fashion_mnist_model_2_layer_50_epoc.h5")

model_paths = {
    "deepfashion": os.path.join(MODELS_DIR, DEEPFASHION_MODEL),
    "fashionmnist": os.path.join(MODELS_DIR, FASHION_MNIST_MODEL),
}

class_names_by_model = {
    "deepfashion": [
        "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
        "Jacket", "Jersey", "Parka", "Peacoat", "Poncho", "Sweater", "Tank", "Tee", "Top", "Turtleneck",
        "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
        "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts", "Trunks", "Caftan", "Cape", "Coat", "Coverup",
        "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress", "Onesie", "Robe", "Romper", "Shirtdress", "Sundress"
    ],
    "fashionmnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
}

# Inicializar FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cargar modelos ---
print("Cargando modelos...")
models = {}
for name, path in model_paths.items():
    print(f"- {name}: {path}")
    models[name] = tf.keras.models.load_model(path)
print("Modelos cargados.")

# --- Procesar imagen (diferente segun modelo) ---
def transform_image_deepfashion(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Asegurate que el tamaño coincida con el modelo
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:  # quitar canal alpha si existe
        img_array = img_array[:, :, :3]
    return np.expand_dims(img_array, axis=0)

def transform_image_fashionmnist(image: Image.Image) -> np.ndarray:
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    return img_array.reshape(1, 28, 28, 1)

# --- Leer imagen de entrada ---
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

# --- Endpoints ---
@app.post("/predict/fashionmnist")
async def predict_fashionmnist(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    image_array = transform_image_fashionmnist(image)

    predictions = models["fashionmnist"].predict(image_array)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "model": "fashionmnist",
        "prediction": class_names_by_model["fashionmnist"][class_index],
        "confidence": round(confidence, 4)
    }

@app.post("/predict/deepfashion")
async def predict_deepfashion(file: UploadFile = File(...)):
    image = read_imagefile(await file.read()).convert("RGB")
    image_array = transform_image_deepfashion(image)

    predictions = models["deepfashion"].predict(image_array)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "model": "deepfashion",
        "prediction": class_names_by_model["deepfashion"][class_index],
        "confidence": round(confidence, 4)
    }
