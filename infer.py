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
MLB_PATH = os.path.join(MODEL_DIR, "mlb.pkl")
ID2LABEL_PATH = os.path.join(MODEL_DIR, "id2label.json")
CANONICAL_PATH = "./canonical_mapping.json"   # MUST exist

print("🔥 Using device:", DEVICE)
print("🔥 Loading best checkpoint:", MODEL_DIR)

# ---------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

# Load label mapping
id2label = json.load(open(ID2LABEL_PATH))

# Canonical dictionary
CANONICAL = json.load(open(CANONICAL_PATH))

# Load binarizer
import pickle
with open(MLB_PATH, "rb") as f:
    mlb = pickle.load(f)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict(sentence, threshold=0.20):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].cpu().numpy()

    indices = np.where(probs >= threshold)[0]

    if len(indices) == 0:
        return []

    labels = [id2label[str(i)] for i in indices]
    return labels


# ---------------------------------------------------------
# Normalize → canonical
# ---------------------------------------------------------
def canonicalize(label_list):
    if len(label_list) == 0:
        return ""

    key = ", ".join(label_list)

    if key in CANONICAL:
        return CANONICAL[key]

    return "UNKNOWN"


# ---------------------------------------------------------
# Clean & split sentences
# ---------------------------------------------------------
def normalize_paragraph(text):
    cleaned = " ".join([l.strip() for l in text.split("\n") if l.strip() != ""])
    return cleaned

def split_sentences(paragraph):
    return nltk.sent_tokenize(paragraph)


# ---------------------------------------------------------
# Tag whole paragraph
# ---------------------------------------------------------
def tag_paragraph(paragraph, threshold=0.20):
    paragraph = normalize_paragraph(paragraph)
    sentences = split_sentences(paragraph)

    rows = []

    for sent in sentences:
        raw_labels = predict(sent, threshold)
        canonical = canonicalize(raw_labels)
        rows.append([sent, canonical])

        print(f"\nSENTENCE: {sent}")
        print("RAW LABELS:", raw_labels)
        print("CANONICAL:", canonical)

    return rows


# ---------------------------------------------------------
# Save XLSX
# ---------------------------------------------------------
def save_to_excel(rows, out="tagged_sentences.xlsx"):
    df = pd.DataFrame(rows, columns=["Sentence", "Canonical Tag"])
    df.to_excel(out, index=False)
    print(f"\n📄 Saved to {out}")


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