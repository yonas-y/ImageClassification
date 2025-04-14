# --- api_app.py ---
# Step 2: Create FastAPI App to Serve Predictions
# -----------------------------------------------

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import csv


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the User Info API"}

# Load model once at startup
model = load_model("results/mnist_model.h5")


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


# Preprocess the input csv data!!
def preprocess_csv(csv_bytes):
    decoded = csv_bytes.decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    data = list(reader)

    flat_pixels = [float(p) for row in data for p in row]
    image_array = np.array(flat_pixels).reshape(28, 28) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array


# Convert the csv input and display the image!!
def image_from_csv(csv_bytes):
    decoded = csv_bytes.decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    data = list(reader)

    flat_pixels = [float(p) for row in data for p in row]
    image_array = np.array(flat_pixels).reshape(28, 28).astype(np.uint8)
    return Image.fromarray(image_array)


# A FastAPI end point to display the input image.
@app.post("/view-image/")
async def view_image(file: UploadFile = File(...)):
    filename = file.filename.lower()
    file_bytes = await file.read()

    try:
        if filename.endswith(".csv"):
            image = image_from_csv(file_bytes)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes)).convert("L").resize((28, 28))
        else:
            return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# A FastAPI end point to make the prediction based on the input.
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()

    try:
        file_bytes = await file.read()

        if filename.endswith(".csv"):
            image_array = preprocess_csv(file_bytes)

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image_array = preprocess_image(file_bytes)

        else:
            return JSONResponse(content={"error": "Unsupported file format. Use CSV, PNG, or JPG."}, status_code=400)

        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction))

        return {"predicted_class": predicted_class}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
