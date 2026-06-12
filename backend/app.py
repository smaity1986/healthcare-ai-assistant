from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch

from PIL import Image

from backend.predict import predict_image

app = FastAPI(
    title="AI Chest X-Ray Screening Assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():

    return {
        "status": "running",
        "gpu": "AMD MI300X"
    }


@app.get("/gpu-info")
def gpu_info():

    props = torch.cuda.get_device_properties(0)

    return {
        "architecture": props.gcnArchName,
        "memory_gb":
            round(props.total_memory / 1024**3),
        "compute_units":
            props.multi_processor_count
    }


@app.post("/predict")
async def predict(file: UploadFile):

    image = Image.open(
        file.file
    ).convert("RGB")

    result = predict_image(image)

    if result["prediction"] == "Pneumonia":

        result["explanation"] = (
            "Possible signs of pneumonia detected. "
            "Please consult a healthcare professional."
        )

        result["risk"] = "High"

    else:

        result["explanation"] = (
            "No signs of pneumonia detected."
        )

        result["risk"] = "Low"

    return result