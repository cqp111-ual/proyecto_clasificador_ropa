from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo Keras
model = tf.keras.models.load_model("modelo_ropa_50clases.h5")

# Lista de clases (ajusta según tu modelo)
class_names = [
    "Anorak",
    "Blazer",
    "Blouse",
    "Bomber",
    "Button-Down",
    "Cardigan",
    "Flannel",
    "Halter",
    "Henley",
    "Hoodie",
    "Jacket",
    "Jersey",
    "Parka",
    "Peacoat",
    "Poncho",
    "Sweater",
    "Tank",
    "Tee",
    "Top",
    "Turtleneck",
    "Capris",
    "Chinos",
    "Culottes",
    "Cutoffs",
    "Gauchos",
    "Jeans",
    "Jeggings",
    "Jodhpurs",
    "Joggers",
    "Leggings",
    "Sarong",
    "Shorts",
    "Skirt",
    "Sweatpants",
    "Sweatshorts",
    "Trunks",
    "Caftan",
    "Cape",
    "Coat",
    "Coverup",
    "Dress",
    "Jumpsuit",
    "Kaftan",
    "Kimono",
    "Nightdress",
    "Onesie",
    "Robe",
    "Romper",
    "Shirtdress",
    "Sundress"
]

# Procesar imagen
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def transform(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Asegurate que el tamaño coincida con el modelo
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:  # quitar canal alpha si existe
        img_array = img_array[:, :, :3]
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read()).convert("RGB")
    image_array = transform(image)

    predictions = model.predict(image_array)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "prediction": class_names[class_index],
        "confidence": round(confidence, 4)
    }
