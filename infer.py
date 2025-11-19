import os
import json
import torch
import nltk
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openpyxl import Workbook

# ---------------------------------------------------------
# Force CPU
# ---------------------------------------------------------
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
DEVICE = torch.device("cpu")

# NLTK sentence splitter
nltk.download("punkt", quiet=True)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
MODEL_DIR = "./models/meraki_sentence_tagger"
ID2LABEL_PATH = os.path.join(MODEL_DIR, "id2label.json")
MLB_PATH = os.path.join(MODEL_DIR, "mlb.pkl")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "thresholds.json")

print("🔥 Using device:", DEVICE)
print("🔥 Loading best checkpoint:", MODEL_DIR)

# ---------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

# Load id2label mapping
id2label = json.load(open(ID2LABEL_PATH))
thresholds = None
if os.path.exists(THRESHOLDS_PATH):
    try:
        thresholds = json.load(open(THRESHOLDS_PATH))
        print("⚖️ Loaded per-label thresholds")
    except Exception:
        thresholds = None

# Load binarizer
import pickle
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict(sentence, threshold=0.20, top_k=5):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # apply per-label thresholds if available
    results = []
    for idx in range(len(probs)):
        label = id2label[str(idx)]
        thr = thresholds.get(label, threshold) if thresholds else threshold
        score = float(probs[idx])
        if score >= thr:
            results.append((label, score))
    # keep top_k by score
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    return results


# ---------------------------------------------------------
# Clean & split
# ---------------------------------------------------------
def normalize_paragraph(text):
    merged = " ".join([l.strip() for l in text.split("\n") if l.strip() != ""])
    return merged

def split_sentences(text):
    return nltk.sent_tokenize(text)


# ---------------------------------------------------------
# Tag whole paragraph
# ---------------------------------------------------------
def tag_paragraph(paragraph, threshold=0.20, top_k=5):
    text = normalize_paragraph(paragraph)
    sentences = split_sentences(text)

    rows = []

    print("\n=============== PREDICTIONS ================\n")

    for sent in sentences:
        tags = predict(sent, threshold, top_k)

        print(f"• {sent}")
        if not tags:
            print("   → (no tags)\n")
            rows.append([sent, ""])
            continue

        for label, score in tags:
            print(f"   → {label} ({score:.3f})")
        print()

        tag_str = ", ".join([f"{label} ({score:.3f})" for label, score in tags])
        rows.append([sent, tag_str])

    return rows


# ---------------------------------------------------------
# Save XLSX
# ---------------------------------------------------------
def save_to_excel(rows, out_path="tagged_sentences.xlsx"):
    df = pd.DataFrame(rows, columns=["Sentence", "Predicted Tags"])
    df.to_excel(out_path, index=False)
    print(f"\n📄 Saved results to: {out_path}")


# ---------------------------------------------------------
# PLACEHOLDER PARAGRAPH
# ---------------------------------------------------------
PARAGRAPH = """
Sowkat Ali said that I am over seventy old man. after death of my wife, I was getting emotional
imbalance.my grandson’s family deprived me from getting food and proper take care. NRC
advocated for marriage and data card which is paved the way to get separate family attestation.
Now, I get food essential with data card. I live with my wife with happily and she takes care me. I
teach religious subject to the young boys and girls, for this reason I get 1500 taka from a NGOs
but last 2 month I didn’t get this salary. I am physically well because of NRC referred to Handicap
International to get medical support. I don’t get non-food support because of SIM card. NRC also
advocates to get SIM card I hope upcoming month I will support the noon-food item. I am grateful
to NRC for their unequivocal support now I live in camp -25 with happily.
"""


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    rows = tag_paragraph(PARAGRAPH)
    save_to_excel(rows)
