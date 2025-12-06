import os
import json
import nltk
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure nltk data
nltk.download("punkt", quiet=True)

app = FastAPI(title="Meraki Tagger API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Configuration & Model Loading
# ---------------------------------------------------------
MODEL_ID = "AaranNihalani/MerakiTagger"

print("🚀 Loading model... (This may take a minute)")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    raise e

# Load thresholds locally from backend folder (or download if needed, but we packaged it)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "backend", "thresholds.json")

try:
    with open(THRESHOLDS_PATH, "r") as f:
        THRESHOLDS = json.load(f)
    print("✅ Loaded thresholds.")
except Exception as e:
    print(f"⚠️ Warning: Could not load thresholds.json: {e}")
    THRESHOLDS = {}

# ---------------------------------------------------------
# Logic
# ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str

def normalize_text(text):
    return " ".join([l.strip() for l in text.split("\n") if l.strip()])

@app.post("/api/predict")
async def predict(request: AnalyzeRequest):
    raw_text = request.text.strip()
    if not raw_text:
        return {"results": []}

    # 1. Split sentences
    clean_text = normalize_text(raw_text)
    sentences = nltk.sent_tokenize(clean_text)

    results = []
    
    # 2. Local Inference Loop
    # Note: For very large batches, you might want to chunk this.
    # Spaces free tier has 2 vCPU, so we process sequentially or in small batches.
    
    try:
        # Tokenize all sentences at once (padding to longest in batch)
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=384)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            # Sigmoid for multi-label
            probs = torch.sigmoid(logits).numpy()
            
        # 3. Process results
        # Get id2label from model config
        id2label = model.config.id2label

        for i, sentence in enumerate(sentences):
            sent_probs = probs[i]
            valid_tags = []

            # Check every class score against threshold
            for label_id, score in enumerate(sent_probs):
                label = id2label[str(label_id)] # id2label keys are usually strings in config
                
                thresh = THRESHOLDS.get(label, 0.5)
                
                if score >= thresh:
                    valid_tags.append({
                        "label": label,
                        "score": round(float(score), 3)
                    })
            
            # Sort tags by confidence
            valid_tags.sort(key=lambda x: x['score'], reverse=True)
            
            results.append({
                "sentence": sentence,
                "tags": valid_tags
            })

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return {"results": results}

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Spaces expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
