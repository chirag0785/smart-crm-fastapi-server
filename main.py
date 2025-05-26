from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from dotenv import load_dotenv
import requests
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face API token from environment
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

URGENCY_MODEL_API_URL = "https://api-inference.huggingface.co/models/KS-Vijay/urgency-model-aura"
DEPARTMENT_MODEL_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

CANDIDATE_LABELS = ["HR", "Sales", "Accounting", "Marketing", "IT", "Legal"]

# Input schemas
class Message(BaseModel):
    text: str

class ClassificationRequest(BaseModel):
    text: str

# Load pipeline locally if model is cached or downloaded
urgency_classifier = pipeline("text-classification", model="KS-Vijay/urgency-model-aura")

# Endpoint 1: Urgency prediction using local model
@app.post("/predict")
async def predict(message: Message):
    result = urgency_classifier(message.text)
    return {"result": result}

# Endpoint 2: Department classification using HuggingFace Inference API
@app.post("/classify")
def classify_text(request: ClassificationRequest):
    payload = {
        "inputs": request.text,
        "parameters": {
            "candidate_labels": CANDIDATE_LABELS,
            "multi_label": False,
            "hypothesis_template": "This text is about {}."
        }
    }
    response = requests.post(DEPARTMENT_MODEL_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        return {
            "input_text": request.text,
            "predicted_label": data["labels"][0],
            "all_labels": data["labels"],
            "scores": dict(zip(data["labels"], data["scores"]))
        }
    else:
        return {"error": "Failed to get classification", "details": response.text}
