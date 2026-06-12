# AI Chest X-Ray Screening Assistant

## 1. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Verify AMD GPU

```bash
python gpu_check.py
```

Expected Output:

```text
CUDA Available: True
Architecture: gfx942:sramecc+:xnack-
Memory (GB): 192
Compute Units: 304
```

---

## 4. Verify Dataset

```bash
cd train

python check_dataset.py
```

Expected Output:

```text
['NORMAL', 'PNEUMONIA']
5216
```

---

## 5. Train Model

From project root:

```bash
python train/train.py
```

Expected Output:

```text
Epoch 1/5
Epoch 2/5
...
Model Saved
```

Model will be stored at:

```text
backend/models/best_model.pth
```

---

## 6. Evaluate Model

```bash
python evaluate.py
```

Expected Output:

```text
Test Accuracy: XX%
```

---

## 7. Verify Model Predictions

```bash
python inspect_predictions.py
```

Expected Output:

```text
Testing NORMAL image...
Prediction: Normal

Testing PNEUMONIA image...
Prediction: Pneumonia
```

---

## 8. Start FastAPI Backend

From project root:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Backend Endpoints:

```text
GET  /
GET  /gpu-info
POST /predict
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## 9. Verify Backend

Health Check:

```bash
curl http://localhost:8000/
```

GPU Information:

```bash
curl http://localhost:8000/gpu-info
```

---

## 10. Start Gradio Frontend

Open a second terminal:

```bash
python frontend/gradio_app.py
```

Expected Output:

```text
Running on local URL:
http://127.0.0.1:7860
```

---

## 11. Open Application

```text
http://localhost:7860
```

Upload a Chest X-Ray image and click Analyze.

---

## Project Architecture

```text
          Gradio UI
               |
               |
          FastAPI API
               |
               |
      EfficientNet-B0
          PyTorch
               |
               |
           ROCm 7
               |
               |
        AMD MI300X
```
