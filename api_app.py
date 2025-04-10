# --- api_app.py ---
# Step 2: Create FastAPI App to Serve Predictions
# -----------------------------------------------

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load model once at startup
model = load_model("results/mnist_model.h5")


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


@app.post("/predict")
async def predict(image_bytes: UploadFile = File(...)):
    try:
        image_bytes = await image_bytes.read()
        img_array = preprocess_image(image_bytes)
        prediction = model.predict(img_array)
        prediction_digit = int(np.argmax(prediction))
        return prediction_digit
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
        