from fastapi import FastAPI
from fastapi import UploadFile

from PIL import Image

from predict import predict_image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):

    image = Image.open(file.file).convert("RGB")

    result = predict_image(image)

    if result["prediction"] == "Pneumonia":

        result["explanation"] = (
            "Possible signs of pneumonia detected. "
            "Consult a healthcare professional."
        )

    else:

        result["explanation"] = (
            "No pneumonia detected."
        )

    return result