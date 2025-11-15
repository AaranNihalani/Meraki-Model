import os
import json
import torch
import numpy as np
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
DEVICE = torch.device("cpu")

print("🔥 Using device:", DEVICE)


# ---------------------------------------------------------
# Load best checkpoint
# ---------------------------------------------------------
def get_best_checkpoint(model_dir="./models/meraki_sentence_tagger"):
    args_path = os.path.join(model_dir, "training_args.bin")
    torch.serialization.add_safe_globals([TrainingArguments])
    args = torch.load(args_path, map_location="cpu", weights_only=False)

    if getattr(args, "best_model_checkpoint", None):
        return args.best_model_checkpoint

    ckpts = [os.path.join(model_dir, d)
             for d in os.listdir(model_dir)
             if d.startswith("checkpoint-")]

    ckpts = sorted(ckpts, key=lambda x: int(x.split("checkpoint-")[-1]))
    return ckpts[-1]


BEST = get_best_checkpoint()
print(f"\n🔥 Loading best checkpoint: {BEST}\n")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained(BEST).to(DEVICE)
id2tag = json.load(open("./models/id2tag.json"))


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict(sentence, threshold=0.30):
    enc = tokenizer(sentence, return_tensors="pt",
                    truncation=True, padding=True,
                    max_length=128).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    tags = [id2tag[str(i)] for i, p in enumerate(probs) if p > threshold]
    return tags


def tag_paragraph(text):
    sentences = nltk.sent_tokenize(text)
    rows = []

    for s in sentences:
        t = predict(s)
        print(f"• {s} → {t}")
        rows.append({"Sentence": s, "Tags": ", ".join(t)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------
example = """
The team distributed food and water in the southern camps.
Health clinics are overwhelmed by cholera cases.
Many families still lack access to education for their children.
I am really happy.
The new infrastructure has been really useful.
"""

df = tag_paragraph(example)
print("\nFinal table:\n", df)

df.to_csv("./tagged_output.csv", index=False)
print("\n✅ Saved to tagged_output.csv")
