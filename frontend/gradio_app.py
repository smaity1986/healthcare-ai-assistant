import requests
import gradio as gr

API_URL = "http://localhost:8000/predict"


def get_gpu_info():
    try:
        response = requests.get(
            "http://localhost:8000/gpu-info"
        )

        gpu = response.json()

        return (
            f"Architecture: {gpu['architecture']}\n"
            f"Memory: {gpu['memory_gb']} GB\n"
            f"Compute Units: {gpu['compute_units']}"
        )

    except Exception as e:
        return str(e)

gpu_info_text = get_gpu_info()


def analyze(image):

    image.save("temp_xray.jpg")

    with open("temp_xray.jpg", "rb") as f:

        response = requests.post(
            API_URL,
            files={
                "file": f
            }
        )

    result = response.json()

    return (
        result["prediction"],
        result["confidence"],
        result["risk"],
        result["explanation"]
    )


with gr.Blocks(
    title="AI Chest X-Ray Screening Assistant"
) as demo:

    gr.Markdown(
        "# 🏥 AI Chest X-Ray Screening Assistant"
    )

    gr.Markdown("## AMD Hardware Information")

    gr.Textbox(
        value=gpu_info_text,
        label="AMD GPU Information",
        interactive=False,
        lines=4
    )

    gr.Markdown(
        "Upload a Chest X-Ray image for analysis"
    )

    gr.Markdown(
        """
        ⚠️ Educational demonstration only.
        This application is not a medical diagnosis system.
        Consult qualified healthcare professionals.
        """
    )

    image = gr.Image(
        type="pil",
        label="Chest X-Ray"
    )

    btn = gr.Button(
        "Analyze"
    )

    prediction = gr.Textbox(
        label="Prediction"
    )

    confidence = gr.Number(
        label="Confidence (%)"
    )

    risk = gr.Textbox(
        label="Risk Level"
    )

    explanation = gr.Textbox(
        label="Explanation"
    )

    btn.click(
        fn=analyze,
        inputs=image,
        outputs=[
            prediction,
            confidence,
            risk,
            explanation
        ]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    debug=True
)