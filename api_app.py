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
