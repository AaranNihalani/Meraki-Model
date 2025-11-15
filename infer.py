import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cpu")
print("🔥 Using device:", DEVICE)

# ------------------------------------------------------------
# Load model + tokenizer + label map
# ------------------------------------------------------------
MODEL_DIR = "./models/meraki_sentence_tagger"

print(f"\n🔥 Loading best checkpoint: {MODEL_DIR}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

id2label = json.load(open("./models/id2label.json"))
num_labels = len(id2label)

# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
def predict(sentence, threshold=0.30, top_k=5):
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**encoding.to(DEVICE)).logits.squeeze(0)

    probs = torch.sigmoid(logits).cpu().numpy()

    # Sort by highest probability
    sorted_idx = np.argsort(probs)[::-1]

    results = []
    for idx in sorted_idx:
        label = id2label[str(idx)]
        score = probs[idx]

        if score >= threshold:
            results.append((label, float(score)))

        # If nothing is above threshold, return top_k anyway
    if len(results) == 0:
        for idx in sorted_idx[:top_k]:
            label = id2label[str(idx)]
            score = float(probs[idx])
            results.append((label, score))

    return results


# ------------------------------------------------------------
# Demo sentences
# ------------------------------------------------------------
sentences = [
    "The team distributed food and water in the southern camps.",
    "Health clinics are overwhelmed by cholera cases.",
    "Many families still lack access to education for their children.",
    "I am really happy.",
    "The new infrastructure has been really useful."
]

print("\n================ PREDICTIONS ================\n")

for s in sentences:
    preds = predict(s, threshold=0.30)
    print(f"• {s}")
    if preds:
        for lbl, score in preds:
            print(f"   → {lbl}  ({score:.3f})")
    else:
        print("   → No tags")
    print()
