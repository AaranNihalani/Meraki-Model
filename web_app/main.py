import os
import json
import nltk
import torch
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
import pandas as pd
from sentence_transformers import SentenceTransformer, util

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
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://keljpfqgfjzvmgffdenf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "inference_results")

print("🚀 Loading model... (This may take a minute)")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    raise e

# Load thresholds and id2label locally from backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "backend", "thresholds.json")
ID2LABEL_PATH = os.path.join(BASE_DIR, "backend", "id2label.json")

try:
    with open(THRESHOLDS_PATH, "r") as f:
        THRESHOLDS = json.load(f)
    print("✅ Loaded thresholds.")
except Exception as e:
    print(f"⚠️ Warning: Could not load thresholds.json: {e}")
    THRESHOLDS = {}

try:
    with open(ID2LABEL_PATH, "r") as f:
        LOCAL_ID2LABEL = json.load(f)
    print("✅ Loaded id2label mapping.")
except Exception as e:
    print(f"⚠️ Warning: Could not load id2label.json: {e}")
    LOCAL_ID2LABEL = None

# Load Sentence Transformer
print("🚀 Loading Sentence Transformer model... (This may take a minute)")
try:
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    st_model.eval()
    print("✅ Sentence Transformer loaded.")
except Exception as e:
    print(f"❌ Error loading Sentence Transformer: {e}")
    st_model = None

# Load Codebook
CODEBOOK_PATH = os.path.join(BASE_DIR, "backend", "codebook.json")
CODEBOOK = {}
CODEBOOK_EMBEDDINGS = {}

def load_codebook():
    global CODEBOOK, CODEBOOK_EMBEDDINGS
    try:
        if os.path.exists(CODEBOOK_PATH):
            with open(CODEBOOK_PATH, "r") as f:
                CODEBOOK = json.load(f)
            print(f"✅ Loaded codebook with {len(CODEBOOK)} entries.")
            
            # Precompute embeddings
            if st_model:
                labels = list(CODEBOOK.keys())
                definitions = list(CODEBOOK.values())
                # Only if definitions are strings
                definitions = [d if isinstance(d, str) else "" for d in definitions]
                
                if definitions:
                    embeddings = st_model.encode(definitions, convert_to_tensor=True)
                    CODEBOOK_EMBEDDINGS = {label: emb for label, emb in zip(labels, embeddings)}
                    print("✅ Precomputed codebook embeddings.")
        else:
            print("⚠️ Codebook not found at path.")
            CODEBOOK = {}
            CODEBOOK_EMBEDDINGS = {}
            
    except Exception as e:
        print(f"⚠️ Warning: Could not load codebook.json: {e}")
        CODEBOOK = {}
        CODEBOOK_EMBEDDINGS = {}

load_codebook()

# ---------------------------------------------------------
# Logic
# ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str

def normalize_text(text):
    return " ".join([l.strip() for l in text.split("\n") if l.strip()])

def split_on_full_stop(text):
    parts = [p.strip() for p in text.split('.')]
    return [p for p in parts if p]

def sentence_case(s):
    s = s.strip()
    return (s[:1].upper() + s[1:]) if s else s

@app.post("/api/codebook")
async def update_codebook(file: UploadFile = File(...)):
    global CODEBOOK, CODEBOOK_EMBEDDINGS
    
    filename = file.filename.lower()
    content = await file.read()
    new_data = {}
    
    try:
        if filename.endswith(".json"):
            new_data = json.loads(content)
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except:
                # Try different encoding or delimiter
                df = pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1")
                
            if "Label" in df.columns and "Definition" in df.columns:
                new_data = dict(zip(df["Label"], df["Definition"]))
            else:
                new_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
            if "Label" in df.columns and "Definition" in df.columns:
                new_data = dict(zip(df["Label"], df["Definition"]))
            else:
                new_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        else:
             raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON, CSV, or Excel.")
             
        # Update CODEBOOK
        CODEBOOK.update(new_data)
        
        # Save to file
        with open(CODEBOOK_PATH, "w") as f:
            json.dump(CODEBOOK, f, indent=2)
            
        # Recompute embeddings
        load_codebook()
        
        return {"message": "Codebook updated successfully", "total_entries": len(CODEBOOK)}
        
    except Exception as e:
        print(f"Update Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update codebook: {str(e)}")

@app.post("/api/predict")
async def predict(request: AnalyzeRequest):
    raw_text = request.text.strip()
    if not raw_text:
        return {"results": []}

    # 1. Split sentences
    clean_text = normalize_text(raw_text)
    sentences = split_on_full_stop(clean_text)

    results = []
    
    # 2. Local Inference Loop
    # Note: For very large batches, you might want to chunk this.
    # Spaces free tier has 2 vCPU, so we process sequentially or in small batches.
    
    try:
        # Tokenize all sentences at once (padding to longest in batch)
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=384)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).numpy()

        # Compute Sentence Transformer embeddings for input sentences
        sent_embeddings = None
        if st_model:
            try:
                sent_embeddings = st_model.encode(sentences, convert_to_tensor=True)
            except Exception as e:
                print(f"ST Encode Error: {e}")

        cfg_id2label = getattr(model.config, "id2label", None)

        def resolve_label(idx):
            if LOCAL_ID2LABEL:
                return LOCAL_ID2LABEL.get(str(idx)) or LOCAL_ID2LABEL.get(idx)
            if isinstance(cfg_id2label, dict):
                return cfg_id2label.get(idx) or cfg_id2label.get(str(idx))
            if isinstance(cfg_id2label, list):
                if 0 <= idx < len(cfg_id2label):
                    return cfg_id2label[idx]
            return f"LABEL_{idx}"

        DEFAULT_THRESHOLD = 0.40
        HIGH_CONF_THRESHOLD = 0.65

        for i, sentence in enumerate(sentences):
            sent_probs = probs[i]
            sent_emb = sent_embeddings[i] if sent_embeddings is not None else None
            valid_tags = []

            for label_id, score in enumerate(sent_probs):
                label = resolve_label(label_id)
                # Use a slightly more aggressive base threshold
                thr = max(THRESHOLDS.get(label, DEFAULT_THRESHOLD), 0.35)
                
                if score >= thr:
                    model_prob = float(score)
                    final_score = model_prob
                    alignment_score = 0.0
                    definition = CODEBOOK.get(label)
                    
                    if definition and sent_emb is not None:
                         if label in CODEBOOK_EMBEDDINGS:
                             def_emb = CODEBOOK_EMBEDDINGS[label]
                             sim = util.cos_sim(sent_emb, def_emb).item()
                             alignment_score = max(0.0, sim)
                             # Increase weight of semantic alignment to 50% to punish "hallucinations"
                             final_score = (model_prob * 0.5) + (alignment_score * 0.5)
                    
                    # Filter out if alignment is terrible (e.g. model confident but meaning is totally wrong)
                    if definition and alignment_score < 0.25:
                        continue

                    if final_score >= thr:
                        valid_tags.append({
                            "label": label, 
                            "score": round(final_score, 3),
                            "explanation": {
                                "definition": definition,
                                "alignment_score": round(alignment_score, 3)
                            }
                        })

            valid_tags.sort(key=lambda x: x["score"], reverse=True)

            if not valid_tags:
                import numpy as _np
                best_id = int(_np.argmax(sent_probs))
                best_label = resolve_label(best_id)
                model_prob = float(sent_probs[best_id])
                
                # Check semantic alignment for the "best" statistical guess
                final_score = model_prob
                alignment_score = 0.0
                definition = CODEBOOK.get(best_label)
                
                if definition and sent_emb is not None and best_label in CODEBOOK_EMBEDDINGS:
                    def_emb = CODEBOOK_EMBEDDINGS[best_label]
                    sim = util.cos_sim(sent_emb, def_emb).item()
                    alignment_score = max(0.0, sim)
                    # Use balanced weight
                    final_score = (model_prob * 0.5) + (alignment_score * 0.5)
                
                # If the best guess is still trash semantically, don't return it blindly.
                # Instead, search for the best SEMANTIC match among top 5 probabilities
                if alignment_score < 0.2:
                    top_indices = _np.argsort(sent_probs)[-5:]
                    best_alt_label = None
                    best_alt_score = -1.0
                    
                    for idx in top_indices:
                        lbl = resolve_label(int(idx))
                        if lbl in CODEBOOK_EMBEDDINGS and sent_emb is not None:
                            d_emb = CODEBOOK_EMBEDDINGS[lbl]
                            s_sim = util.cos_sim(sent_emb, d_emb).item()
                            if s_sim > best_alt_score and s_sim > 0.3: # Minimum alignment
                                best_alt_score = s_sim
                                best_alt_label = lbl
                                
                    if best_alt_label:
                        best_label = best_alt_label
                        definition = CODEBOOK.get(best_label)
                        alignment_score = best_alt_score
                        # Recalculate final score for this alternative
                        m_prob = float(sent_probs[int(idx)]) # Approximate
                        final_score = (m_prob * 0.4) + (alignment_score * 0.6)

                valid_tags = [{
                    "label": best_label, 
                    "score": round(final_score, 3),
                    "explanation": {
                        "definition": definition,
                        "alignment_score": round(alignment_score, 3)
                    }
                }]

            results.append({"sentence": sentence_case(sentence), "tags": valid_tags[:2]})

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    try:
        if SUPABASE_KEY:
            import requests
            url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
            payload = []
            now_iso = datetime.now(timezone.utc).isoformat()
            for item in results:
                payload.append({
                    "sentence": item["sentence"],
                    "tags": item["tags"],
                    "raw_text": raw_text,
                    "created_at": now_iso
                })
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        print(f"Supabase write failed: {e}")

    return {"results": results}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or ""
    ext = os.path.splitext(name)[1].lower()
    data = await file.read()
    text = ""
    if ext == ".txt":
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid text file")
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            text = "\n".join(parts)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read DOCX")
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                t = t.strip()
                if t:
                    parts.append(t)
            text = "\n".join(parts)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read PDF")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted")
    return {"text": normalize_text(text)}

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Spaces expects port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
