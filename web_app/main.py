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
nltk.download("vader_lexicon", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

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
CODEBOOK_SENTIMENTS = {} # Store sentiment scores of definitions
CODEBOOK_KEYWORDS = {}   # Store unique nouns from definitions

sia = SentimentIntensityAnalyzer()

def extract_keywords(text):
    """Extract Nouns/Proper Nouns from text."""
    try:
        # Pre-process: remove punctuation except hyphens/apostrophes
        import string
        text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        
        keywords = set()
        for word, tag in tags:
            # Keep Nouns (NN*), Adjectives (JJ*) and Foreign Words (FW)
            # Adjectives are crucial for languages (e.g. "Burmese" is often JJ)
            if tag.startswith(('NN', 'JJ', 'FW')) and len(word) > 2:
                keywords.add(word.lower())
                
        return keywords
    except:
        return set()

def load_codebook():
    global CODEBOOK, CODEBOOK_EMBEDDINGS, CODEBOOK_SENTIMENTS, CODEBOOK_KEYWORDS
    try:
        if os.path.exists(CODEBOOK_PATH):
            with open(CODEBOOK_PATH, "r") as f:
                CODEBOOK = json.load(f)
            print(f"✅ Loaded codebook with {len(CODEBOOK)} entries.")
            
            # Precompute embeddings and metadata
            if st_model:
                labels = list(CODEBOOK.keys())
                definitions = list(CODEBOOK.values())
                # Only if definitions are strings
                definitions = [d if isinstance(d, str) else "" for d in definitions]
                
                if definitions:
                    embeddings = st_model.encode(definitions, convert_to_tensor=True)
                    CODEBOOK_EMBEDDINGS = {label: emb for label, emb in zip(labels, embeddings)}
                    print("✅ Precomputed codebook embeddings.")
            
                    # Precompute Sentiments and Keywords
                    CODEBOOK_SENTIMENTS = {}
                    CODEBOOK_KEYWORDS = {}
                    
                    # Compute TF-IDF to find important words dynamically
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    
                    if definitions:
                        # Use sklearn to find top keywords per definition relative to the whole codebook
                        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
                        tfidf_matrix = vectorizer.fit_transform(definitions)
                        feature_names = vectorizer.get_feature_names_out()
                        
                        for idx, (label, definition) in enumerate(CODEBOOK.items()):
                            if isinstance(definition, str):
                                # Sentiment
                                CODEBOOK_SENTIMENTS[label] = sia.polarity_scores(definition)['compound']
                                
                                # Keywords: Get top 3 words with highest TF-IDF score for this definition
                                row = tfidf_matrix[idx]
                                top_n = 3
                                # Sort indices by score descending
                                sorted_indices = row.toarray().flatten().argsort()[::-1]
                                top_indices = sorted_indices[:top_n]
                                
                                # Keep only words with non-zero score
                                keys = {feature_names[i] for i in top_indices if row[0, i] > 0}
                                CODEBOOK_KEYWORDS[label] = keys
                                
                    print("✅ Precomputed sentiments and keywords (TF-IDF).")

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

def _table_to_codebook(df: pd.DataFrame):
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Codebook file is empty.")
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Codebook must have at least 2 columns (label, definition). Found {df.shape[1]} column(s).",
        )
    norm = {str(c).strip().lower(): c for c in df.columns}
    label_col = None
    def_col = None
    for cand in ("label", "tag", "code", "name"):
        if cand in norm:
            label_col = norm[cand]
            break
    for cand in ("definition", "description", "def", "meaning"):
        if cand in norm:
            def_col = norm[cand]
            break
    if label_col is None or def_col is None:
        label_col = df.columns[0]
        def_col = df.columns[1]
    out = {}
    for _, row in df.iterrows():
        lbl = row.get(label_col)
        dfn = row.get(def_col)
        if pd.isna(lbl) or pd.isna(dfn):
            continue
        lbl = str(lbl).strip()
        dfn = str(dfn).strip()
        if not lbl or not dfn:
            continue
        out[lbl] = dfn
    if not out:
        raise HTTPException(status_code=400, detail="No valid (label, definition) rows found in codebook.")
    return out

@app.post("/api/codebook")
async def update_codebook(file: UploadFile = File(...)):
    global CODEBOOK, CODEBOOK_EMBEDDINGS
    
    filename = (file.filename or "").lower()
    content = await file.read()
    new_data = {}
    
    try:
        if filename.endswith(".json"):
            parsed = json.loads(content.decode("utf-8", errors="ignore"))
            if isinstance(parsed, dict):
                new_data = {
                    str(k).strip(): (str(v).strip() if v is not None else "")
                    for k, v in parsed.items()
                    if str(k).strip() and (str(v).strip() if v is not None else "")
                }
            else:
                raise HTTPException(status_code=400, detail="JSON codebook must be an object: {label: definition}.")
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            except:
                df = pd.read_csv(io.BytesIO(content), encoding="ISO-8859-1", sep=None, engine="python")
            new_data = _table_to_codebook(df)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
            new_data = _table_to_codebook(df)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON, CSV, or Excel.")
             
        if not new_data:
            raise HTTPException(status_code=400, detail="Parsed codebook contains no entries.")

        CODEBOOK = dict(new_data)
        
        # Save to file
        with open(CODEBOOK_PATH, "w") as f:
            json.dump(CODEBOOK, f, indent=2)
            
        # Recompute embeddings
        load_codebook()
        
        return {"message": "Codebook updated successfully", "total_entries": len(CODEBOOK)}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Update Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update codebook: {str(e)}")

@app.get("/api/codebook/download")
async def download_codebook():
    if not os.path.exists(CODEBOOK_PATH):
        raise HTTPException(status_code=404, detail="Codebook not found")
    
    # Return file as download
    from fastapi.responses import FileResponse
    return FileResponse(
        path=CODEBOOK_PATH, 
        filename="codebook.json", 
        media_type='application/json'
    )

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

            # 3. Dynamic Analysis of Input Sentence
            sent_sentiment = sia.polarity_scores(sentence)['compound']
            sent_keywords = extract_keywords(sentence)

            for label_id, score in enumerate(sent_probs):
                label = resolve_label(label_id)
                # Use a slightly more aggressive base threshold
                thr = max(THRESHOLDS.get(label, DEFAULT_THRESHOLD), 0.35)
                
                if score >= thr:
                    model_prob = float(score)
                    final_score = model_prob
                    alignment_score = 0.0
                    definition = CODEBOOK.get(label)
                    
                    penalty = 0.0

                    if definition and sent_emb is not None:
                         if label in CODEBOOK_EMBEDDINGS:
                             def_emb = CODEBOOK_EMBEDDINGS[label]
                             sim = util.cos_sim(sent_emb, def_emb).item()
                             alignment_score = max(0.0, sim)
                             
                             # --- Sentiment Check ---
                             # If definition is strongly positive (>0.3) and sentence is strongly negative (<-0.3), penalize
                             def_sent = CODEBOOK_SENTIMENTS.get(label, 0.0)
                             if def_sent > 0.3 and sent_sentiment < -0.3:
                                 penalty += 0.4 # Heavy penalty for polarity mismatch
                             elif def_sent < -0.3 and sent_sentiment > 0.3:
                                 penalty += 0.4

                             # --- Keyword Overlap Check ---
                             # If definition has unique keywords (like "Burmese", "English"), check overlap
                             def_keys = CODEBOOK_KEYWORDS.get(label, set())
                             
                             if def_keys:
                                 # Compute overlap
                                 overlap = def_keys.intersection(sent_keywords)
                                 
                                 # Soft penalty for missing keywords (0.15), not a death sentence
                                 # This handles cases where user says "I speak the local language" instead of "Burmese"
                                 if len(def_keys) >= 1 and len(overlap) == 0:
                                     penalty += 0.15

                             # Restore balanced weights: 60% Model, 40% Alignment
                             # We trust the model's training more than the semantic scorer for nuance
                             final_score = (model_prob * 0.6) + (alignment_score * 0.4) - penalty
                             final_score = max(0.0, final_score)
                    
                    # Filter out if alignment is terrible (e.g. model confident but meaning is totally wrong)
                    # Relaxed strictness: < 0.25 is garbage
                    if definition and alignment_score < 0.25:
                        continue
                    
                    # Removed strict veto > 0.35. Penalty just lowers score now.

                    if final_score >= thr:
                        valid_tags.append({
                            "label": label, 
                            "score": round(final_score, 3),
                            "explanation": {
                                "definition": definition,
                                "alignment_score": round(alignment_score, 3),
                                "sentiment_penalty": penalty
                            }
                        })

            # --- Post-Processing: Conflict Resolution ---
            # If multiple tags from same "family" exist (e.g. "Skills Learned: ..."), keep only the winner.
            valid_tags.sort(key=lambda x: x["score"], reverse=True)
            
            final_filtered_tags = []
            seen_prefixes = set()
            
            for tag in valid_tags:
                label = tag["label"]
                # Detect prefix (e.g. "Skills Learned", "Advocacy Achievement")
                if ":" in label:
                    prefix = label.split(":")[0].strip()
                    # If we already have a higher-score tag from this family, skip this one
                    if prefix in seen_prefixes:
                        continue
                    seen_prefixes.add(prefix)
                
                final_filtered_tags.append(tag)
            
            valid_tags = final_filtered_tags
            
            # --- End Post-Processing ---

            if not valid_tags:
                    # Fallback Logic: If no tags passed the strict filter
                    import numpy as _np
                    top_indices = _np.argsort(sent_probs)[-10:] # Look deeper, top 10
                    best_label = None
                    best_final_score = -1.0
                    best_alignment_score = 0.0
                    best_definition = None

                    for idx in top_indices:
                        idx = int(idx)
                        lbl = resolve_label(idx)
                        m_prob = float(sent_probs[idx])
                        
                        # Skip if model thinks it's trash (<10%) unless alignment is amazing
                        if m_prob < 0.10: continue

                        d_def = CODEBOOK.get(lbl)
                        s_align = 0.0
                        fallback_penalty = 0.0
                        
                        if d_def and sent_emb is not None and lbl in CODEBOOK_EMBEDDINGS:
                            d_emb = CODEBOOK_EMBEDDINGS[lbl]
                            s_align = max(0.0, util.cos_sim(sent_emb, d_emb).item())
                            
                            # Re-run soft checks for fallback candidates
                            def_keys = CODEBOOK_KEYWORDS.get(lbl, set())
                            if def_keys:
                                overlap = def_keys.intersection(sent_keywords)
                                if len(def_keys) >= 1 and len(overlap) == 0:
                                    fallback_penalty = 0.15
                            
                            # Sentiment check
                            def_sent = CODEBOOK_SENTIMENTS.get(lbl, 0.0)
                            if def_sent > 0.3 and sent_sentiment < -0.3:
                                fallback_penalty += 0.2 # Softer penalty
                            elif def_sent < -0.3 and sent_sentiment > 0.3:
                                fallback_penalty += 0.2

                        if s_align < 0.25: continue # Minimum alignment for fallback

                        # Balanced weight in fallback
                        f_score = (m_prob * 0.5) + (s_align * 0.5) - fallback_penalty
                        
                        if f_score > best_final_score:
                            best_final_score = f_score
                            best_label = lbl
                            best_alignment_score = s_align
                            best_definition = d_def

                    if best_label:
                        valid_tags = [{
                            "label": best_label, 
                            "score": round(best_final_score, 3),
                            "explanation": {
                                "definition": best_definition,
                                "alignment_score": round(best_alignment_score, 3)
                            }
                        }]
                    else:
                        valid_tags = [] # Truly nothing matches

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
