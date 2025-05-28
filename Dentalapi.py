import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load ONNX model
onnx_path = "Model/best_model.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Class names (update with your actual classes)
class_names = [
    "Caries", "Caries_Cavity", "Caries_Cavity_Crack", "Caries_Crack",
    "Cavity", "Cavity_Crack", "Crack", "Normal"
]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dim
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    ort_outs = session.run(None, ort_inputs)
    pred_idx = int(np.argmax(ort_outs[0]))
    pred_class = class_names[pred_idx]
    return {"predicted_class": str(pred_class), "class_index": int(pred_idx)}